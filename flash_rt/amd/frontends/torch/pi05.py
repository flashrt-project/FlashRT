"""FlashRT AMD -- CDNA4 Pi0.5 torch frontend.

Loads HuggingFace PyTorch safetensors checkpoints + drives the
framework-agnostic :class:`~flash_rt.amd.models.pi05.pipeline.Pi05Pipeline`.

Mirror of :class:`flash_rt.frontends.torch.pi05_rtx.Pi05TorchFrontendRtx`
with the AMD deltas: pipeline / attention backend / kernels come from
``flash_rt.amd``, the D2D copy helpers go through the HIP runtime, and
the experimental RTX-only paths (INT8, RL/CFG, batched B=2) are dropped.
The weight conversion (``convert_pi05_safetensors``), the decoder style
precompute and the FP8 quantization scheme are kept EXACTLY — they are
correctness-critical and hardware-neutral.

torch on ROCm keeps the "cuda" device string and the ``torch.cuda``
stream API, so all torch-side device handling is unchanged.

Usage::

    from flash_rt.amd.frontends.torch.pi05 import Pi05TorchFrontendAmd
    pipe = Pi05TorchFrontendAmd("/path/to/pi05_libero_pytorch", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs_dict])   # once, ~1 s
    out = pipe.infer({"image": img, "wrist_image": wrist})
    actions = out["actions"]     # (chunk_size, 7) numpy
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
from flash_rt.amd.hardware.cdna4.attn_backend import Cdna4AttnBackend
from flash_rt.amd.models.pi05.pipeline import (
    Pi05Pipeline,
    VIS_L, VIS_D, VIS_H, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H,
    DEC_L, DEC_D, DEC_H, DEC_HD,
    ACTION_DIM, NUM_STEPS_DEFAULT,
)
from flash_rt.core.utils.pi05_prompt import PI05_STATE_PROMPT_MAX_LEN, format_pi05_prompt

logger = logging.getLogger(__name__)

bf16 = torch.bfloat16
fp8_e4m3 = torch.float8_e4m3fn

CHUNK_SIZE = 10
IMG_HW = 224
MAX_PROMPT_LEN_DEFAULT = 48


# ════════════════════════════════════════════════════════════════════
#   HF safetensors → pipeline weight dict (BF16 torch tensors)
# ════════════════════════════════════════════════════════════════════


def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Interleave Q/K output dim from HF contiguous to JAX RoPE format."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .permute(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


def convert_pi05_safetensors(safetensors_path: Union[str, pathlib.Path]) -> dict:
    """Convert a HuggingFace Pi0.5 safetensors file to BF16 torch tensor dict.

    Key transformations (verified bit-exact against the openpi PyTorch
    reference forward on LIBERO data):

      - Vision attention: separate Q/K/V → merged, transposed (in, 3*out).
      - Vision patch embedding: ``(C_out, C_in, H, W)`` → ``(H, W, C_in, C_out)``.
      - Encoder RMSNorm fold: multiply Q/K/V/gate/up weights by ``(1 + norm_w)``
        in FP32 to avoid bf16 rounding near -1.0.
      - Encoder Q/K heads: interleave for fused RoPE kernel.
      - Decoder Q/K heads: interleave (no RMS fold — AdaRMSNorm is runtime).
      - Decoder AdaRMSNorm modulation: ``input_layernorm.dense`` →
        ``pre_attn_norm_mod`` (kept separate, BF16).
      - Output projection: frontend pre-scales ``decoder_action_out_proj_w/b``
        by ``-1.0 / num_steps`` (matching the flow-matching residual accumulation).
      - 10-step sinusoidal time embeddings.
    """
    from safetensors import safe_open
    from flash_rt.executors.torch_weights import _autodetect_strip_prefix

    logger.info("Loading Pi0.5 safetensors: %s", safetensors_path)
    f = safe_open(str(safetensors_path), framework="pt")
    # Auto-strip the lerobot HF policy ``model.`` wrap so the openpi
    # bare-key lookups below resolve transparently on either layout.
    _strip = _autodetect_strip_prefix(set(f.keys()))

    def g(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key).to(bf16)

    def g_raw(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key)

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) ──
    vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    pe_w = g(f"{vp}.embeddings.patch_embedding.weight")   # (1152, 3, 14, 14)
    # Target layout (14, 14, 3, 1152) flattens contiguously to (588, 1152)
    # row-major as (h, w, c, o) — matches the patch_im2col output order.
    ckpt["vision_patch_embedding_w"] = pe_w.permute(2, 3, 1, 0).contiguous()
    ckpt["vision_patch_embedding_b"] = g(f"{vp}.embeddings.patch_embedding.bias")
    ckpt["vision_position_embedding"] = g(f"{vp}.embeddings.position_embedding.weight")

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    for i in range(VIS_L):
        lp = f"{vp}.encoder.layers.{i}"
        q_w = g(f"{lp}.self_attn.q_proj.weight")
        k_w = g(f"{lp}.self_attn.k_proj.weight")
        v_w = g(f"{lp}.self_attn.v_proj.weight")
        qkv_w_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        q_b = g(f"{lp}.self_attn.q_proj.bias")
        k_b = g(f"{lp}.self_attn.k_proj.bias")
        v_b = g(f"{lp}.self_attn.v_proj.bias")
        qkv_b_list.append(torch.cat([q_b, k_b, v_b]))

        o_w_list.append(g(f"{lp}.self_attn.out_proj.weight").t())
        o_b_list.append(g(f"{lp}.self_attn.out_proj.bias"))

        up_w_list.append(g(f"{lp}.mlp.fc1.weight").t())
        up_b_list.append(g(f"{lp}.mlp.fc1.bias"))

        down_w_list.append(g(f"{lp}.mlp.fc2.weight").t())
        down_b_list.append(g(f"{lp}.mlp.fc2.bias"))

        ln1_w_list.append(g(f"{lp}.layer_norm1.weight"))
        ln1_b_list.append(g(f"{lp}.layer_norm1.bias"))
        ln2_w_list.append(g(f"{lp}.layer_norm2.weight"))
        ln2_b_list.append(g(f"{lp}.layer_norm2.bias"))

    ckpt["vision_attn_qkv_w"] = torch.stack(qkv_w_list)
    ckpt["vision_attn_qkv_b"] = torch.stack(qkv_b_list)
    ckpt["vision_attn_o_w"] = torch.stack(o_w_list)
    ckpt["vision_attn_o_b"] = torch.stack(o_b_list)
    ckpt["vision_ffn_up_w"] = torch.stack(up_w_list)
    ckpt["vision_ffn_up_b"] = torch.stack(up_b_list)
    ckpt["vision_ffn_down_w"] = torch.stack(down_w_list)
    ckpt["vision_ffn_down_b"] = torch.stack(down_b_list)
    ckpt["vision_pre_attn_norm_w"] = torch.stack(ln1_w_list)
    ckpt["vision_pre_attn_norm_b"] = torch.stack(ln1_b_list)
    ckpt["vision_pre_ffn_norm_w"] = torch.stack(ln2_w_list)
    ckpt["vision_pre_ffn_norm_b"] = torch.stack(ln2_b_list)
    ckpt["vision_final_norm_w"] = g(f"{vp}.post_layernorm.weight")
    ckpt["vision_final_norm_b"] = g(f"{vp}.post_layernorm.bias")

    # ── Multi-modal projector ──
    mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    ckpt["encoder_multi_modal_projector_w"] = g(f"{mp}.weight").t()
    ckpt["encoder_multi_modal_projector_b"] = g(f"{mp}.bias")

    # ── Encoder (18 Gemma-2B layers with RMSNorm fold) ──
    ep = "paligemma_with_expert.paligemma.model.language_model.layers"
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []

    for i in range(ENC_L):
        # CRITICAL: fuse in FP32 — bf16 rounds values near -1.0 to exactly
        # -1.0, collapsing (1 + scale) to 0 and zeroing entire channels.
        attn_scale = g_raw(f"{ep}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale  # (2048,)

        q_w = g_raw(f"{ep}.{i}.self_attn.q_proj.weight").float()
        k_w = g_raw(f"{ep}.{i}.self_attn.k_proj.weight").float()
        v_w = g_raw(f"{ep}.{i}.self_attn.v_proj.weight").float()
        q_w = _interleave_qk(q_w, 8)
        k_w = _interleave_qk(k_w, 1)
        q_w = q_w * fuse_attn.unsqueeze(0)
        k_w = k_w * fuse_attn.unsqueeze(0)
        v_w = v_w * fuse_attn.unsqueeze(0)
        qkv = torch.cat([q_w, k_w, v_w], dim=0).t().to(bf16)
        enc_qkv_list.append(qkv)

        enc_o_list.append(g(f"{ep}.{i}.self_attn.o_proj.weight").t())

        ffn_scale = g_raw(f"{ep}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale

        gate_w = g_raw(f"{ep}.{i}.mlp.gate_proj.weight").float() * fuse_ffn.unsqueeze(0)
        up_w = g_raw(f"{ep}.{i}.mlp.up_proj.weight").float() * fuse_ffn.unsqueeze(0)
        enc_gate_list.append(gate_w.t().to(bf16))
        enc_up_list.append(up_w.t().to(bf16))

        enc_down_list.append(g(f"{ep}.{i}.mlp.down_proj.weight").t())

    ckpt["encoder_attn_qkv_w"] = torch.stack(enc_qkv_list)
    ckpt["encoder_attn_o_w"] = torch.stack(enc_o_list)
    ckpt["encoder_ffn_gate_w"] = torch.stack(enc_gate_list)
    ckpt["encoder_ffn_up_w"] = torch.stack(enc_up_list)
    ckpt["encoder_ffn_down_w"] = torch.stack(enc_down_list)

    # ── Decoder (18 Gemma-300M layers) ──
    dp = "paligemma_with_expert.gemma_expert.model.layers"
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    dec_attn_mod_w_list, dec_attn_mod_b_list = [], []
    dec_ffn_mod_w_list, dec_ffn_mod_b_list = [], []

    for i in range(DEC_L):
        dec_attn_mod_w_list.append(g(f"{dp}.{i}.input_layernorm.dense.weight").t())
        dec_attn_mod_b_list.append(g(f"{dp}.{i}.input_layernorm.dense.bias"))

        q_w = g(f"{dp}.{i}.self_attn.q_proj.weight")
        k_w = g(f"{dp}.{i}.self_attn.k_proj.weight")
        v_w = g(f"{dp}.{i}.self_attn.v_proj.weight")
        q_w = _interleave_qk(q_w.float(), 8).to(q_w.dtype)
        k_w = _interleave_qk(k_w.float(), 1).to(k_w.dtype)
        dec_qkv_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        dec_o_list.append(g(f"{dp}.{i}.self_attn.o_proj.weight").t())

        dec_ffn_mod_w_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.weight").t())
        dec_ffn_mod_b_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.bias"))

        dec_gate_list.append(g(f"{dp}.{i}.mlp.gate_proj.weight").t())
        dec_up_list.append(g(f"{dp}.{i}.mlp.up_proj.weight").t())
        dec_down_list.append(g(f"{dp}.{i}.mlp.down_proj.weight").t())

    ckpt["decoder_attn_qkv_w"] = torch.stack(dec_qkv_list)
    ckpt["decoder_attn_o_w"] = torch.stack(dec_o_list)
    ckpt["decoder_ffn_gate_w"] = torch.stack(dec_gate_list)
    ckpt["decoder_ffn_up_w"] = torch.stack(dec_up_list)
    ckpt["decoder_ffn_down_w"] = torch.stack(dec_down_list)
    ckpt["decoder_pre_attn_norm_mod_w"] = torch.stack(dec_attn_mod_w_list)
    ckpt["decoder_pre_attn_norm_mod_b"] = torch.stack(dec_attn_mod_b_list)
    ckpt["decoder_pre_ffn_norm_mod_w"] = torch.stack(dec_ffn_mod_w_list)
    ckpt["decoder_pre_ffn_norm_mod_b"] = torch.stack(dec_ffn_mod_b_list)

    ckpt["decoder_final_norm_mod_w"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight").t()
    ckpt["decoder_final_norm_mod_b"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.bias")

    # ── Time MLP + sinusoidal embeddings ──
    ckpt["decoder_time_mlp_in_w"] = g("time_mlp_in.weight").t()
    ckpt["decoder_time_mlp_in_b"] = g("time_mlp_in.bias")
    ckpt["decoder_time_mlp_out_w"] = g("time_mlp_out.weight").t()
    ckpt["decoder_time_mlp_out_b"] = g("time_mlp_out.bias")

    num_steps = NUM_STEPS_DEFAULT
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    embedding_dim = DEC_D
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    time_emb_list = []
    for _ in range(num_steps):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb_list.append(
            torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(bf16)
        )
        t = t + dt
    ckpt["decoder_time_embeds"] = torch.cat(time_emb_list, dim=0)  # (10, 1024)

    # ── Action projections (pre-scaled by frontend before pipeline build) ──
    ckpt["decoder_action_in_proj_w"] = g("action_in_proj.weight").t()
    ckpt["decoder_action_in_proj_b"] = g("action_in_proj.bias")
    ckpt["decoder_action_out_proj_w"] = g("action_out_proj.weight").t()
    ckpt["decoder_action_out_proj_b"] = g("action_out_proj.bias")

    # ── Embedding matrix (for prompt tokenisation) ──
    ckpt["embedding_weight"] = g("paligemma_with_expert.paligemma.lm_head.weight")

    logger.info("Converted %d weight groups", len(ckpt))
    return ckpt


def _embed_prompt(prompt_text: str, embedding_weight: torch.Tensor,
                  max_len: int = 48, state=None) -> tuple[torch.Tensor, int]:
    """Tokenise + embed via PaliGemma embedding table (device, bf16)."""
    # PaliGemma tokenizer resolution — see
    # `flash_rt.utils.paligemma_tokenizer` for the search order and
    # the download instructions emitted on failure.
    try:
        # Preferred: openpi's PaligemmaTokenizer (exact same vocab,
        # same prompt prefix logic FlashRT was built against).
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_len)
        tokens_np, mask_np = tokenizer.tokenize(prompt_text, state=state)
        prompt_len = int(mask_np.sum())
        token_ids = torch.tensor(
            tokens_np[:prompt_len], dtype=torch.long, device="cuda")
    except (ImportError, FileNotFoundError, OSError, RuntimeError):
        # Fallback: locate the SentencePiece model directly via the
        # FlashRT helper (clear error if not found — never silent
        # segfault).
        from flash_rt.utils.paligemma_tokenizer import (
            load_paligemma_sentencepiece,
        )
        sp = load_paligemma_sentencepiece()
        if state is None:
            # 108 is PaliGemma's `\n` token, used by openpi as the
            # prompt-end separator before the action prefix.
            tokens = [sp.bos_id()] + sp.Encode(prompt_text) + [108]
        else:
            tokens = sp.Encode(format_pi05_prompt(prompt_text, state),
                               add_bos=True)
        token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
        prompt_len = len(token_ids)

    if embedding_weight.device.type != "cuda":
        embedding_weight = embedding_weight.to(device="cuda")

    embeds = F.embedding(token_ids, embedding_weight)
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds, prompt_len


# ════════════════════════════════════════════════════════════════════
#   Weight FP8 quantization + precomputed decoder styles
# ════════════════════════════════════════════════════════════════════


def _quantize_fp8_e4m3(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 E4M3 quantization."""
    amax = w_bf16.float().abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w_bf16.float() / scale).clamp(-448.0, 448.0).to(fp8_e4m3)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device="cuda")
    return w_fp8, scale_tensor


def _select_fp8_layout(fp8_layout: Optional[str]) -> str:
    """Choose the FP8 weight layout used by the AMD frontend kernels.

    ``kn``: weights are stored as [K,N] and use ``fp8_nn_dev``.
    ``nk``: weights are stored as [N,K] and use ``fp8_nt_dev`` (default —
    measured faster in-graph across every pi05 FP8 GEMM shape on CDNA4,
    -2.8 ms E2E vs kn in the single-variable A/B).
    """
    if fp8_layout is not None:
        if fp8_layout not in ("kn", "nk"):
            raise ValueError(f"fp8_layout must be 'kn' or 'nk', got {fp8_layout!r}")
        return fp8_layout
    return "nk"


def _precompute_decoder_styles(ckpt: dict, chunk_size: int,
                               num_steps: int = NUM_STEPS_DEFAULT) -> dict:
    """Pre-compute the time-MLP + per-layer style modulations in torch.

    Output dict has numpy arrays (dtype bf16 via torch→numpy view):
        time_emb:    (num_steps, chunk_size, DEC_D)
        style_attn:  (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_ffn:   (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_final: (num_steps, chunk_size, 3 * DEC_D)

    All computation runs on the GPU in bf16, then is moved to CPU and viewed
    as uint16 so it can be uploaded verbatim to HipBuffer (bf16 = 2 bytes,
    numpy doesn't natively support bf16 but the bytes round-trip).

    Time embeddings are regenerated from scratch for the given num_steps so
    that any step count works correctly (e.g. num_steps=5 gives
    t=1.0, 0.8, 0.6, 0.4, 0.2 with dt=-0.2, not a truncation of the
    10-step table stored in the checkpoint).
    """
    W = {k: v.to("cuda", bf16) if isinstance(v, torch.Tensor) else v
         for k, v in ckpt.items()}

    # Regenerate sinusoidal time embeddings for the given num_steps / dt.
    # The checkpoint stores a 10-step table; generate fresh ones so any
    # step count gets the correct (t=1, t=1-dt, …) time schedule.
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    _time_emb_rows = []
    for _ in range(num_steps):
        # period has shape (DEC_D//2,); t is a scalar tensor → sinusoid: (DEC_D//2,)
        sinusoid = t * (1.0 / period) * 2 * math.pi
        _time_emb_rows.append(
            torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).to(bf16))
        t = t + dt
    time_emb_schedule = torch.stack(_time_emb_rows, dim=0).to("cuda")  # (steps, DEC_D)
    t_in_w = W["decoder_time_mlp_in_w"]                       # (1024, 1024)
    t_in_b = W["decoder_time_mlp_in_b"]                       # (1024,)
    t_out_w = W["decoder_time_mlp_out_w"]
    t_out_b = W["decoder_time_mlp_out_b"]

    attn_mod_w = W["decoder_pre_attn_norm_mod_w"]             # (L, 1024, 3072)
    attn_mod_b = W["decoder_pre_attn_norm_mod_b"]             # (L, 3072)
    ffn_mod_w = W["decoder_pre_ffn_norm_mod_w"]
    ffn_mod_b = W["decoder_pre_ffn_norm_mod_b"]
    final_mod_w = W["decoder_final_norm_mod_w"]               # (1024, 3072)
    final_mod_b = W["decoder_final_norm_mod_b"]               # (3072,)

    time_emb_out = torch.empty(num_steps, chunk_size, DEC_D, dtype=bf16, device="cuda")
    style_attn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")
    style_ffn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")
    style_final = torch.empty(num_steps, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")

    for step in range(num_steps):
        te = time_emb_schedule[step:step + 1]                 # (1, 1024)
        tmp = te @ t_in_w + t_in_b[None, :]                   # SiLU input
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(bf16)
        tmp2 = tmp @ t_out_w + t_out_b[None, :]
        tmp2 = (tmp2.float() * torch.sigmoid(tmp2.float())).to(bf16)
        te_expanded = tmp2.expand(chunk_size, -1).contiguous()  # (chunk, 1024)
        time_emb_out[step] = te_expanded

        for i in range(DEC_L):
            style_attn[step, i] = te_expanded @ attn_mod_w[i] + attn_mod_b[i][None, :]
            style_ffn[step, i] = te_expanded @ ffn_mod_w[i] + ffn_mod_b[i][None, :]

        style_final[step] = te_expanded @ final_mod_w + final_mod_b[None, :]

    # View as uint16 (bf16 bit pattern) so numpy can round-trip bytes.
    def _to_np_u16(t: torch.Tensor) -> np.ndarray:
        return t.contiguous().view(torch.uint16).cpu().numpy()

    return {
        "time_emb": _to_np_u16(time_emb_out),
        "style_attn": _to_np_u16(style_attn),
        "style_ffn": _to_np_u16(style_ffn),
        "style_final": _to_np_u16(style_final),
    }


# ════════════════════════════════════════════════════════════════════
#   Pi05TorchFrontendAmd frontend
# ════════════════════════════════════════════════════════════════════


class Pi05TorchFrontendAmd:
    """AMD CDNA4 Pi0.5 Torch frontend.

    Mirrors the RTX frontend's public API (``set_prompt`` + ``infer`` +
    ``calibrate_with_real_data`` + ``get_latency_stats``) so the same
    eval scripts work on both hardware families.
    """

    def __init__(self,
                 checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 2,
                 chunk_size: int = CHUNK_SIZE,
                 max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
                 num_steps: int = NUM_STEPS_DEFAULT,
                 vision_pool_factor: int = 1,
                 vision_num_layers: Optional[int] = None,
                 cache_frames: int = 1,
                 use_fp8: bool = True,
                 hardware: Optional[str] = None,
                 fp8_layout: Optional[str] = None,
                 state_prompt_mode: str = "exact",
                 state_prompt_fixed_max_len: Optional[int] = None):
        # gfx950-only gate, FIRST: the MFMA tile shapes and FP8 paths in
        # the extension are CDNA4-specific — refuse before touching the
        # checkpoint unless BOTH the visible device arch (e.g.
        # "gfx950:sramecc+:xnack-") and the extension's compile-time
        # gpu_arch are gfx950. Anything else computes garbage, not a
        # fallback (a forced hardware="amd_cdna4" on gfx942 fails here).
        from flash_rt.amd import flash_rt_amd_kernels as _fvk_gate
        # Compare the base target only ("gfx950" from
        # "gfx950:sramecc+:xnack-"); a prefix test would also accept a
        # future "gfx9500".
        _dev_arch = str(_fvk_gate.device_arch())
        _build_arch = str(dict(_fvk_gate.build_info()).get("gpu_arch",
                                                           "unknown"))
        if not (_dev_arch.split(":", 1)[0] == "gfx950"
                and _build_arch.split(":", 1)[0] == "gfx950"):
            raise RuntimeError(
                "Pi05TorchFrontendAmd is gfx950-only (CDNA4 / MI350-series): "
                f"running device arch is {_dev_arch!r} and the extension "
                f"was built for gpu_arch {_build_arch!r}. Rebuild with "
                "scripts/amd/build_amd.sh on a gfx950 machine; other AMD "
                "arches are not supported by this backend.")

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        # State-in-prompt graph strategy (Pi0.5 renders robot state into the
        # prompt, so its token length drifts with the state values):
        #   "exact" (default): a separate pipeline captured per exact length,
        #       cached; pair with warm_state_prompt_buckets() to front-load the
        #       lengths you expect so the control loop avoids a mid-loop capture.
        #   "fixed": ONE pipeline + ONE captured graph at the max prompt length;
        #       every length is served by masking the padded prefix +
        #       appending decoder K/V at the valid offset (devpos),
        #       so a changing length never re-captures and no warmup is needed.
        # Env override: FLASHRT_PI05_STATE_PROMPT_MODE.
        _spm = os.environ.get("FLASHRT_PI05_STATE_PROMPT_MODE", state_prompt_mode)
        if _spm not in ("fixed", "exact"):
            raise ValueError(
                f"state_prompt_mode must be 'fixed' or 'exact', got {_spm!r}")
        self._state_prompt_mode = _spm
        # Fixed-mode padded prompt capacity. The one-graph strategy pays
        # per padded token through the whole encoder, so right-sizing
        # this to the deployment's real prompt+state length (instead of
        # the PI05_STATE_PROMPT_MAX_LEN=200 ceiling) is the main
        # fixed-mode latency lever. Same semantics as the Thor frontend;
        # env override FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN.
        _fixed_cap = PI05_STATE_PROMPT_MAX_LEN
        if _spm == "fixed":
            _fixed_cap = os.environ.get(
                "FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN",
                state_prompt_fixed_max_len)
            if _fixed_cap is None:
                _fixed_cap = PI05_STATE_PROMPT_MAX_LEN
            _fixed_cap = int(_fixed_cap)
            if _fixed_cap <= 0:
                raise ValueError(
                    "state_prompt_fixed_max_len must be a positive integer, "
                    f"got {_fixed_cap}")
        self._state_prompt_fixed_max_len = _fixed_cap
        self.num_views = int(num_views)
        # Observation contract: 2 views (base + wrist camera — the LIBERO
        # deployment) or 3 (+ right wrist camera). All input buffers and
        # the encoder sequence are sized to num_views, and the observation
        # keys accepted by _gather_view_images() only express 2 or 3.
        if self.num_views not in (2, 3):
            raise ValueError(
                "num_views must be 2 (base + wrist camera) or 3 "
                f"(+ right wrist camera); got {self.num_views}")
        self.chunk_size = int(chunk_size)
        self.max_prompt_len = int(max_prompt_len)
        self._num_steps = int(num_steps)
        self._vision_pool_factor = int(vision_pool_factor)
        if self._num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self._num_steps}")
        if self._vision_pool_factor not in (1, 2, 4):
            raise ValueError(
                "vision_pool_factor must be one of {1, 2, 4}; "
                f"got {self._vision_pool_factor}")
        # Temporal K/V caching: run full pipeline every `cache_frames` frames,
        # intermediate frames reuse the cached encoder K/V (decoder-only).
        # cache_frames=1 (default) = no caching, every frame is full.
        # cache_frames=2 = full, decode, full, decode, ...
        self._cache_frames = int(cache_frames)
        if self._cache_frames < 1:
            raise ValueError(f"cache_frames must be >= 1, got {self._cache_frames}")
        self._frame_count = 0
        self._vision_num_layers = VIS_L if vision_num_layers is None else int(vision_num_layers)
        if not 1 <= self._vision_num_layers <= VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {VIS_L}], "
                f"got {self._vision_num_layers}")
        self.use_fp8 = bool(use_fp8)
        self.hardware = hardware if hardware is not None else "amd_cdna4"
        self.fp8_layout = _select_fp8_layout(fp8_layout)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        self.current_prompt_len = 0
        self.pipeline: Optional[Pi05Pipeline] = None
        self._prompt_pipeline_cache: dict[int, Pi05Pipeline] = {}
        # Fixed-shape (state_prompt_mode="fixed") pipeline, cached separately so
        # switching to a no-state prompt and back reuses the already-calibrated,
        # already-captured graph instead of rebuilding it.
        self._fixed_pipeline: Optional[Pi05Pipeline] = None
        # BF16-only escape hatch (no FP8 support probe needed on CDNA4 —
        # MI350X has native OCP FP8 E4M3; use the env knob or use_fp8=False
        # to force the BF16 baseline).
        env_force_bf16 = os.environ.get("FVK_PI05_AMD_FORCE_BF16", "0") == "1"
        self._force_bf16 = env_force_bf16

        # ── Load norm_stats ──
        self._load_norm_stats(checkpoint_dir)

        # ── Load + convert safetensors ──
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path} — "
                "Pi05TorchFrontendAmd expects a HuggingFace-style PyTorch checkpoint")
        self._checkpoint_path = str(safetensors_path)
        raw_ckpt = convert_pi05_safetensors(safetensors_path)

        # Move all tensors to device bf16 (retain as member attrs so their
        # memory stays alive across pipeline rebuilds).
        self._ckpt_bf16 = {}
        for k, v in raw_ckpt.items():
            if isinstance(v, torch.Tensor):
                self._ckpt_bf16[k] = v.to("cuda", bf16).contiguous()
            else:
                self._ckpt_bf16[k] = v
        self.embedding_weight = self._ckpt_bf16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps.
        # Scaling is specific to the step count (ODE integration step size).
        num_steps = self._num_steps
        self._ckpt_bf16["decoder_action_out_proj_w"] = \
            self._ckpt_bf16["decoder_action_out_proj_w"] * (-1.0 / num_steps)
        self._ckpt_bf16["decoder_action_out_proj_b"] = \
            self._ckpt_bf16["decoder_action_out_proj_b"] * (-1.0 / num_steps)

        # ── Low-precision weight stores ──
        # FP8 weight quantization is gated on use_fp8 so BF16-only loading
        # works without the FP8 kernels present (the quantize itself is
        # pure torch, but the resulting pointers would route the pipeline
        # through fp8_* fvk entries).
        self._fp8_weights: dict = {}
        self._fp8_packed: dict = {}  # decoder weights in MFMA-packed layout
        self._fp8_store: list = []  # holds tensors alive
        if self.use_fp8 and not self._force_bf16:
            self._quantize_all_fp8()

        # ── Pre-compute decoder styles (time MLP + style modulation) ──
        self._precomputed_styles = _precompute_decoder_styles(
            self._ckpt_bf16, self.chunk_size, num_steps=self._num_steps)

        # ── Attention backend (torch, owns Q/K/V/O) ──
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = self._make_attn_backend(enc_seq_max)

        # ── fvk module + GemmRunner ──
        from flash_rt.amd import flash_rt_amd_kernels as fvk
        self.fvk = fvk
        # (gfx950 gate already ran at the top of __init__.)
        self.gemm = fvk.GemmRunner()

        # ── Reusable pre-allocated input buffers (match Thor style) ──
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=bf16, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        from flash_rt.amd.core.hip_buffer import _check, _hip
        self._hip = _hip
        # Shared HIP return-code checker (single int compare on success;
        # raises RuntimeError with the error code + operation on failure).
        # Every raw self._hip.* call MUST route its return through this so
        # a failed copy/sync can never surface stale buffer contents.
        self._hip_check = _check

        logger.info(
            "Pi05TorchFrontendAmd initialised (num_views=%d, chunk=%d, fp8_layout=%s)",
            self.num_views, self.chunk_size, self.fp8_layout)

    def _make_attn_backend(self, enc_seq_max: int):
        """Construct the CDNA4 attention backend.

        ``FVK_AMD_ATTN=sdpa|aiter`` selects the implementation (default
        "sdpa" — the interim torch-SDPA backend). "aiter" dispatches to
        the aiter asm flash-attention backend
        (:class:`~flash_rt.amd.hardware.cdna4.attn_backend_aiter.Cdna4AiterAttnBackend`,
        same buffers/surface); if aiter is unavailable the frontend logs
        a warning and falls back to sdpa.
        """
        kwargs = dict(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.chunk_size,
            num_encoder_layers=ENC_L)
        choice = os.environ.get("FVK_AMD_ATTN", "aiter").strip().lower()
        if choice not in ("sdpa", "aiter"):
            raise ValueError(
                f"FVK_AMD_ATTN must be 'sdpa' or 'aiter', got {choice!r}")
        if choice == "aiter":
            from flash_rt.amd.hardware.cdna4.attn_backend_aiter import (
                Cdna4AiterAttnBackend,
            )
            try:
                backend = Cdna4AiterAttnBackend(**kwargs)
                logger.info("CDNA4 attention backend: aiter (FVK_AMD_ATTN)")
                return backend
            except ImportError as ex:
                logger.warning(
                    "FVK_AMD_ATTN=aiter but the aiter backend is "
                    "unavailable (%s); falling back to sdpa", ex)
        return Cdna4AttnBackend(**kwargs)

    def _ensure_prompt_capacity(self, required_prompt_len: int) -> None:
        """Grow attention buffers before building longer prompt pipelines."""
        if required_prompt_len <= self.max_prompt_len:
            return
        self.max_prompt_len = int(required_prompt_len)
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = self._make_attn_backend(enc_seq_max)
        self._prompt_pipeline_cache.clear()
        self._fixed_pipeline = None
        self.pipeline = None
        self.current_prompt_len = 0
        self.graph_recorded = False
        self.calibrated = False
        logger.info("Grew Pi0.5 AMD prompt capacity to %d tokens",
                    self.max_prompt_len)

    def _pipeline_precision_kwargs(self) -> dict:
        if self._force_bf16:
            logger.warning(
                "FVK_PI05_AMD_FORCE_BF16=1 set: disabling FP8 paths for "
                "the Pi0.5 AMD pipeline.")
            return {
                "use_fp8": False,
                "use_fp8_decoder": False,
            }
        return {
            "use_fp8": self.use_fp8,
            "use_fp8_decoder": self.use_fp8,
        }

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir: pathlib.Path) -> None:
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, pi05_candidates,
        )
        try:
            self.norm_stats = load_norm_stats(
                pi05_candidates(checkpoint_dir), checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"norm_stats not found near checkpoint: {e}") from e

    def _quantize_all_fp8(self) -> None:
        """Pre-quantize all large GEMM weights to FP8 E4M3."""
        W = self._ckpt_bf16
        store = self._fp8_store
        fp8 = self._fp8_weights

        def quant(name: str, w: torch.Tensor):
            if self.fp8_layout == "nk":
                w = w.t().contiguous()
            else:
                w = w.contiguous()
            w_fp8, scale = _quantize_fp8_e4m3(w)
            store.append(w_fp8)
            store.append(scale)
            fp8[name] = (w_fp8.data_ptr(), scale.data_ptr())

        # Vision (27 layers × 4) + projector
        for i in range(VIS_L):
            quant(f"vision_attn_qkv_w_{i}", W["vision_attn_qkv_w"][i])
            quant(f"vision_attn_o_w_{i}", W["vision_attn_o_w"][i])
            quant(f"vision_ffn_up_w_{i}", W["vision_ffn_up_w"][i])
            quant(f"vision_ffn_down_w_{i}", W["vision_ffn_down_w"][i])
        quant("vision_projector_w", W["encoder_multi_modal_projector_w"])

        # Encoder (18 layers × 4) — fuse gate+up into (D, 2H)
        for i in range(ENC_L):
            quant(f"encoder_attn_qkv_w_{i}", W["encoder_attn_qkv_w"][i])
            quant(f"encoder_attn_o_w_{i}", W["encoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["encoder_ffn_gate_w"][i], W["encoder_ffn_up_w"][i]], dim=1
            ).contiguous()
            quant(f"encoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"encoder_ffn_down_w_{i}", W["encoder_ffn_down_w"][i])

        # Decoder (18 layers × 4)
        for i in range(DEC_L):
            quant(f"decoder_attn_qkv_w_{i}", W["decoder_attn_qkv_w"][i])
            quant(f"decoder_attn_o_w_{i}", W["decoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["decoder_ffn_gate_w"][i], W["decoder_ffn_up_w"][i]], dim=1
            ).contiguous()
            quant(f"decoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"decoder_ffn_down_w_{i}", W["decoder_ffn_down_w"][i])

        # MFMA-packed copies of the decoder weights (nk layout only):
        # repacked once here into the smallm_mfma_nt_packed per-lane
        # consumption order (see csrc/amd/gemm/smallm_mfma.h) so the
        # decoder small-M GEMMs stream weights as one linear slab per
        # workgroup. The plain fp8 copy is kept for the hipBLASLt
        # fallback and autotune.
        if self.fp8_layout == "nk":
            packed = self._fp8_packed
            for name, (w_ptr, _) in list(fp8.items()):
                if not name.startswith("decoder_"):
                    continue
                w_fp8 = next(t for t in store
                             if t.data_ptr() == w_ptr and t.dim() == 2)
                n_dim, k_dim = w_fp8.shape
                if n_dim % 16 != 0 or k_dim % 1024 != 0:
                    continue
                wp = (w_fp8.view(torch.uint8)
                      .view(n_dim // 16, 16, k_dim // 64, 2, 4, 8)
                      .permute(0, 2, 4, 1, 3, 5).contiguous())
                store.append(wp)
                packed[name] = wp.data_ptr()
            logger.info("MFMA-packed %d decoder GEMM weights", len(packed))

        logger.info("FP8 quantized %d GEMM weights (layout=%s)", len(fp8), self.fp8_layout)

    def _build_pipeline_weights(self) -> dict:
        """Produce the pointer dict that Pi05Pipeline expects."""
        W = self._ckpt_bf16

        def p(key: str) -> int:
            return W[key].data_ptr()

        def p_list(key: str) -> list[int]:
            t = W[key]
            stride = t.stride(0) * t.element_size()
            base = t.data_ptr()
            return [base + i * stride for i in range(t.shape[0])]

        weights = {
            # Vision BF16
            "vision_patch_embedding_w": p("vision_patch_embedding_w"),
            "vision_patch_embedding_b": p("vision_patch_embedding_b"),
            "vision_position_embedding": p("vision_position_embedding"),
            "vision_pre_attn_norm_w": p_list("vision_pre_attn_norm_w"),
            "vision_pre_attn_norm_b": p_list("vision_pre_attn_norm_b"),
            "vision_pre_ffn_norm_w": p_list("vision_pre_ffn_norm_w"),
            "vision_pre_ffn_norm_b": p_list("vision_pre_ffn_norm_b"),
            "vision_attn_qkv_w": p_list("vision_attn_qkv_w"),  # BF16 fallback
            "vision_attn_qkv_b": p_list("vision_attn_qkv_b"),
            "vision_attn_o_w": p_list("vision_attn_o_w"),
            "vision_attn_o_b": p_list("vision_attn_o_b"),
            "vision_ffn_up_w": p_list("vision_ffn_up_w"),
            "vision_ffn_up_b": p_list("vision_ffn_up_b"),
            "vision_ffn_down_w": p_list("vision_ffn_down_w"),
            "vision_ffn_down_b": p_list("vision_ffn_down_b"),
            "vision_final_norm_w": p("vision_final_norm_w"),
            "vision_final_norm_b": p("vision_final_norm_b"),

            # Encoder
            "encoder_multi_modal_projector_w": p("encoder_multi_modal_projector_w"),
            "encoder_multi_modal_projector_b": p("encoder_multi_modal_projector_b"),
            "encoder_attn_qkv_w": p_list("encoder_attn_qkv_w"),
            "encoder_attn_o_w": p_list("encoder_attn_o_w"),
            "encoder_ffn_gate_w": p_list("encoder_ffn_gate_w"),
            "encoder_ffn_up_w": p_list("encoder_ffn_up_w"),
            "encoder_ffn_down_w": p_list("encoder_ffn_down_w"),

            # Decoder
            "decoder_action_in_proj_w": p("decoder_action_in_proj_w"),
            "decoder_action_in_proj_b": p("decoder_action_in_proj_b"),
            "decoder_action_out_proj_w": p("decoder_action_out_proj_w"),
            "decoder_action_out_proj_b": p("decoder_action_out_proj_b"),
            "decoder_attn_qkv_w": p_list("decoder_attn_qkv_w"),
            "decoder_attn_o_w": p_list("decoder_attn_o_w"),
            "decoder_ffn_gate_w": p_list("decoder_ffn_gate_w"),
            "decoder_ffn_up_w": p_list("decoder_ffn_up_w"),
            "decoder_ffn_down_w": p_list("decoder_ffn_down_w"),

            # FP8 quantized weights
            "fp8": self._fp8_weights,
            "fp8_packed": self._fp8_packed,
            "fp8_layout": self.fp8_layout,
            "hardware": self.hardware,

            # Precomputed decoder styles (numpy bf16 as uint16 view)
            "precomputed": self._precomputed_styles,
        }
        return weights

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_prompt(self, prompt_text: str, state=None) -> None:
        """Tokenise prompt + (re)build the pipeline for the exact prompt length."""
        if state is not None:
            max_len = (self._state_prompt_fixed_max_len
                       if self._state_prompt_mode == "fixed"
                       else PI05_STATE_PROMPT_MAX_LEN)
        else:
            max_len = MAX_PROMPT_LEN_DEFAULT
        embeds, prompt_len = _embed_prompt(
            prompt_text, self.embedding_weight, max_len=max_len, state=state)

        if self._state_prompt_mode == "fixed" and state is not None:
            self._set_prompt_fixed(prompt_len)
        else:
            self._set_prompt_per_length(state, prompt_len)

        # The attention backend is shared across pipelines, so sync its
        # fixed-shape mode to the now-active pipeline BEFORE running it. Without
        # this, a frontend that ran a fixed state prompt and then a no-state
        # prompt (which falls back to a per-length pipeline) would keep the
        # backend in fixed mode and reuse stale seqused/devpos buffers.
        self.attn_backend.set_fixed_shape(
            bool(getattr(self.pipeline, "_fixed_shape", False)))

        # Upload language embeds into pipeline's encoder_x slot. In fixed mode
        # set_language_embeds pads to max + updates the seqused/devpos buffers.
        embeds_np = embeds.contiguous().view(torch.uint16).cpu().numpy()
        self.pipeline.set_language_embeds(embeds_np)
        self._frame_count = 0
        logger.info("Set prompt: '%s' (%d tokens, state=%s, mode=%s)",
                    prompt_text, prompt_len, state is not None,
                    self._state_prompt_mode)

    def _set_prompt_fixed(self, prompt_len: int) -> None:
        """Fixed-shape mode: build ONE max-length pipeline + one graph; later
        prompt lengths only update embeds + seqused/devpos (no re-capture).

        The fixed pipeline is cached in ``self._fixed_pipeline`` so that
        switching to a no-state prompt (which activates a per-length pipeline)
        and back REUSES the already-calibrated, already-captured graph instead
        of rebuilding it — a rebuild would re-run FP8 calibration/autotune on a
        backend the per-length pipeline has since touched and would also
        perturb numerics via autotune variance.
        """
        cap = self._state_prompt_fixed_max_len
        if prompt_len > cap:
            raise ValueError(
                f"fixed-mode prompt+state is {prompt_len} tokens but the "
                f"padded graph capacity is {cap}; raise "
                "state_prompt_fixed_max_len (or the "
                "FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN env) — the graph "
                "is captured once at this capacity and cannot grow.")
        self._ensure_prompt_capacity(cap)
        if self._fixed_pipeline is None:
            logger.info("Building fixed-shape Pi05Pipeline (max_prompt_len=%d)...",
                        cap)
            pipeline_weights = self._build_pipeline_weights()
            self._fixed_pipeline = Pi05Pipeline(
                gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                weights=pipeline_weights,
                num_views=self.num_views,
                max_prompt_len=cap,
                chunk_size=self.chunk_size,
                num_steps=self._num_steps,
                vision_pool_factor=self._vision_pool_factor,
                vision_num_layers=self._vision_num_layers,
                fixed_shape=True,
                **self._pipeline_precision_kwargs())
        # (Re)activate the cached fixed pipeline, restoring calibration/capture
        # state from the instance (mirrors the per-length cache reuse path) so
        # predict() does not re-calibrate or re-capture on switch-back.
        if self.pipeline is not self._fixed_pipeline:
            self.pipeline = self._fixed_pipeline
            self.graph_recorded = (
                getattr(self._fixed_pipeline, "_graph", None) is not None)
            self.calibrated = (
                self.graph_recorded
                or bool(getattr(self._fixed_pipeline, "fp8_calibrated", False)))
        self.current_prompt_len = prompt_len

    def _set_prompt_per_length(self, state, prompt_len: int) -> None:
        """Legacy 'exact' mode: a separate pipeline captured per exact length
        (cached so a recurring length is not re-built)."""
        required_capacity = (PI05_STATE_PROMPT_MAX_LEN if state is not None
                             else prompt_len)
        self._ensure_prompt_capacity(required_capacity)

        if self.pipeline is None or prompt_len != self.current_prompt_len:
            cached = self._prompt_pipeline_cache.get(prompt_len)
            self.current_prompt_len = prompt_len
            if cached is not None:
                self.pipeline = cached
                self.graph_recorded = getattr(cached, "_graph", None) is not None
                self.calibrated = (
                    self.graph_recorded
                    or bool(getattr(cached, "fp8_calibrated", False)))
                logger.info("Reusing cached Pi05Pipeline for prompt_len=%d",
                            prompt_len)
            else:
                logger.info("Building Pi05Pipeline for prompt_len=%d...",
                            prompt_len)
                self.graph_recorded = False
                self.calibrated = False

                pipeline_weights = self._build_pipeline_weights()
                self.pipeline = Pi05Pipeline(
                    gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                    weights=pipeline_weights,
                    num_views=self.num_views,
                    max_prompt_len=prompt_len,
                    chunk_size=self.chunk_size,
                    num_steps=self._num_steps,
                    vision_pool_factor=self._vision_pool_factor,
                    vision_num_layers=self._vision_num_layers,
                    **self._pipeline_precision_kwargs())
                self._prompt_pipeline_cache[prompt_len] = self.pipeline

    def warm_state_prompt_buckets(self, prompt_text: str, states,
                                  sample_observation: dict) -> list[int]:
        """Pre-build runtime buckets for Pi0.5 state-in-prompt lengths.

        The prompt text is kept in the OpenPI format. This method only
        front-loads graph capture/autotune for the token lengths reached
        by the supplied representative states.
        """
        if isinstance(states, np.ndarray) and states.ndim == 1:
            state_list = [states]
        else:
            state_list = list(states)
        if not state_list:
            raise ValueError("states must contain at least one representative state")

        warmed: set[int] = set()
        for state in state_list:
            self.set_prompt(prompt_text, state=state)
            prompt_len = int(self.current_prompt_len)
            if prompt_len in warmed and getattr(self.pipeline, "_graph", None) is not None:
                continue
            if not self.calibrated:
                self.calibrate_with_real_data([sample_observation])
            warmed.add(prompt_len)

        logger.info("Warmed Pi0.5 state prompt buckets: %s", sorted(warmed))
        return sorted(warmed)

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration entry point.

        N=1 → single-frame path, bit-equal to legacy.
        N>=2 → per-sample amax, reduced via ``np.percentile(..., axis=0)``.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before calibrate")
        if self.calibrated:
            logger.warning(
                "calibrate() called a second time; returning without re-running.")
            return

        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        if not getattr(self.pipeline, "use_fp8", False):
            # BF16 pipeline: no FP8 scales to collect — the single-frame
            # path just warms buffers and captures the graph.
            if n > 1:
                logger.info(
                    "BF16 pipeline has no activation scales to calibrate; "
                    "using the first sample to warm buffers and capture the graph.")
            self._calibrate_single_frame(obs_list[0])
            return

        if n == 1:
            self._calibrate_single_frame(obs_list[0])
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    def _calibrate_single_frame(self, sample) -> None:
        logger.info("Preparing Pi0.5 runtime with a single real sample...")

        # Create a dedicated torch stream for both the calibration pass and
        # graph capture so the backend's torch attention ops + our fvk
        # kernels land on the same stream.
        self._graph_torch_stream = torch.cuda.Stream()

        with torch.cuda.stream(self._graph_torch_stream):
            images = self._stack_images(sample)
            noise = torch.randn(
                self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")

            stream_int = self._graph_torch_stream.cuda_stream
            self._copy_tensor_to_pipeline_buf_stream(
                images, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                noise, self.pipeline.input_noise_buf, stream_int)

            self.pipeline.run_pipeline(stream=stream_int)

            self._hip_check(self._hip.hipStreamSynchronize(
                ctypes.c_void_p(stream_int)), "hipStreamSynchronize")

            # FP8 calibration (no-op for BF16 pipelines).
            self.pipeline.calibrate_fp8()
            self.pipeline.autotune_gemms()
            self.pipeline.record_infer_graph(external_stream_int=stream_int)

        self.calibrated = True
        self.graph_recorded = True
        self._precision_spec = self._snapshot_precision_spec(
            method="single_frame", n=1, percentile=None)
        self._warn_if_scale_ceiling_exceeded()
        logger.info("Calibration + graph capture complete")

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        from flash_rt.core.calibration import (
            accumulate_amax,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Preparing Pi0.5 runtime across %d real samples (percentile=%.2f)...",
            n, percentile)
        self._graph_torch_stream = torch.cuda.Stream()
        self.pipeline.fp8_calibrated = False

        per_sample: list[np.ndarray] = []
        names: Optional[list[str]] = None

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            for i, obs in enumerate(obs_list):
                images = self._stack_images(obs)
                noise = torch.randn(
                    self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
                self._copy_tensor_to_pipeline_buf_stream(
                    images, self.pipeline.input_images_buf, stream_int)
                self._copy_tensor_to_pipeline_buf_stream(
                    noise, self.pipeline.input_noise_buf, stream_int)
                self._zero_pipeline_scales()
                self.pipeline.run_pipeline(stream=stream_int)
                self._hip_check(self._hip.hipStreamSynchronize(
                    ctypes.c_void_p(stream_int)), "hipStreamSynchronize")

                if names is None:
                    names = list(self.pipeline.fp8_act_scales.keys())
                sample_vec = np.array(
                    [float(self.pipeline.fp8_act_scales[k].download_new(
                        (1,), np.float32)[0]) for k in names],
                    dtype=np.float32)
                per_sample.append(sample_vec)

                if verbose and (i + 1) % max(1, n // 10) == 0:
                    logger.info("  calibration sample %d/%d", i + 1, n)

            final_amax = accumulate_amax(per_sample, percentile=percentile)
            if verbose:
                logger.info(format_summary(
                    summarize_amax_dispersion(per_sample, final_amax)))

            for idx, name in enumerate(names or []):
                self.pipeline.fp8_act_scales[name].upload(
                    np.array([final_amax[idx]], dtype=np.float32))

            self.pipeline.fp8_calibrated = True
            self.pipeline.autotune_gemms()
            self.pipeline.record_infer_graph(external_stream_int=stream_int)

        self.calibrated = True
        self.graph_recorded = True
        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)
        self._warn_if_scale_ceiling_exceeded(label=f"pi05_amd_N{n}")
        logger.info(
            "Pi0.5 multi-frame calibration + graph capture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def _zero_pipeline_scales(self) -> None:
        for buf in self.pipeline.fp8_act_scales.values():
            buf.zero_()

    def _warn_if_scale_ceiling_exceeded(self, label: str = "pi05_amd") -> None:
        """Diagnostic warning if any FP8 scale exceeds the sanity ceiling."""
        from flash_rt.core.calibration import check_scale_ceiling
        scales = {
            name: float(buf.download_new((1,), np.float32)[0])
            for name, buf in self.pipeline.fp8_act_scales.items()
        }
        check_scale_ceiling(scales, label=label)

    def _snapshot_precision_spec(self, *, method: str, n: int,
                                  percentile: Optional[float]):
        from flash_rt.core.precision_spec import (
            ModelPrecisionSpec,
            PrecisionSpec,
        )

        spec = ModelPrecisionSpec(source="calibration")
        for name, buf in self.pipeline.fp8_act_scales.items():
            scale_val = float(buf.download_new((1,), np.float32)[0])
            entry = PrecisionSpec(
                dtype="fp8_e4m3",
                granularity="per_tensor",
                scheme="symmetric",
                scale_source="calibration",
                scale=np.array([scale_val], dtype=np.float32),
                calibration_method=method,
                calibration_samples=n,
                calibration_percentile=percentile,
            )
            entry.validate()
            if name.startswith("vision_"):
                spec.activation_specs[name] = entry
            elif name.startswith("encoder_"):
                spec.encoder_layer_specs[name] = entry
            elif name.startswith("decoder_") or name.startswith("action_"):
                spec.decoder_layer_specs[name] = entry
            else:
                spec.activation_specs[name] = entry
        return spec

    @property
    def precision_spec(self):
        """:class:`ModelPrecisionSpec` captured at calibration time."""
        return getattr(self, "_precision_spec", None)

    def infer(self, observation: dict, debug: bool = False,
              noise: Optional[np.ndarray] = None) -> dict:
        """Run inference on a single observation.

        All GPU work happens on ``self._graph_torch_stream`` — the same
        stream the graph was captured on — so replay + pre/post D2D copies
        are serialized correctly.

        ``noise``: optional ``(chunk_size, 32)`` array used as the denoise
        starting point instead of a fresh random draw — the same contract
        as openpi's ``policy.infer(example, noise=...)``. Serving wants the
        default fresh draw (action diversity); parity/repro judging MUST
        pin the noise, since the flow integration is conditioned on it and
        a random draw adds ~1e-3-level cos-vs-reference lottery that is
        indistinguishable from a real numerics regression.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before infer")

        t0 = time.perf_counter()

        # Temporal K/V caching: every cache_frames-th frame runs the full
        # pipeline (vision + encoder + decoder); intermediate frames skip
        # vision and encoder and replay only the decoder with fresh noise,
        # reusing the encoder K/V cache from the last full forward.
        self._frame_count += 1
        use_full = (self._cache_frames <= 1 or
                    self._frame_count % self._cache_frames == 1)

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            if noise is not None:
                self._noise_buf.copy_(torch.as_tensor(
                    np.asarray(noise)).reshape(self._noise_buf.shape))
            else:
                self._noise_buf.normal_()
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self.pipeline.input_noise_buf, stream_int)

            if use_full:
                self._fill_img_buf(observation)
                self._copy_tensor_to_pipeline_buf_stream(
                    self._img_buf, self.pipeline.input_images_buf, stream_int)
                out_ptr = self.pipeline.forward()
            else:
                # Decode-only: skip vision+encoder, reuse cached K/V
                out_ptr = self.pipeline.forward_decode_only()

            # D2D download → staging torch tensor. Both the copy and the
            # sync below are checked: if the graph replay, this D2D copy
            # or the sync fails, we must raise rather than read stale
            # _noise_out contents back as actions.
            self._hip_check(self._hip.hipMemcpyAsync(
                ctypes.c_void_p(self._noise_out.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out.numel() * 2, 3, stream_int),
                "hipMemcpyAsync D2D")

        self._hip_check(self._hip.hipStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream)),
            "hipStreamSynchronize")

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = self._noise_out.float().cpu().numpy()  # (chunk, 32)
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]

        if debug:
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("Latency: %.1f ms", latency_ms)

        return {"actions": robot_actions}

    def get_latency_stats(self) -> dict:
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "std_ms": float(np.std(lat)),
            "min_ms": float(np.min(lat)),
            "max_ms": float(np.max(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "hz": float(1000 / np.mean(lat)),
        }

    # -----------------------------------------------------------------
    # Not-yet-ported RTX extensions (explicit failure, no silent fallback)
    # -----------------------------------------------------------------
    # The single-stream production surface (set_prompt / calibrate /
    # infer, both state_prompt_mode graphs, FP8 + BF16) is aligned with
    # Pi05TorchFrontendRtx. The RL-conditioned and CFG-batched modes
    # below are RTX-only so far; they raise instead of degrading.

    _NOT_PORTED = (
        "is not ported to the AMD backend yet — supported surface: "
        "set_prompt / warm_state_prompt_buckets / calibrate / "
        "calibrate_with_real_data / infer / precision_spec / "
        "get_latency_stats (state_prompt_mode 'exact' and 'fixed'). "
        "See docs/deployment_amd.md for the support matrix.")

    def set_rl_mode(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Pi05TorchFrontendAmd.set_rl_mode {self._NOT_PORTED}")

    def set_batched_mode(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Pi05TorchFrontendAmd.set_batched_mode {self._NOT_PORTED}")

    def set_prompt_batch(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Pi05TorchFrontendAmd.set_prompt_batch {self._NOT_PORTED}")

    def calibrate_batch(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Pi05TorchFrontendAmd.calibrate_batch {self._NOT_PORTED}")

    def infer_batch(self, *args, **kwargs) -> list:
        raise NotImplementedError(
            f"Pi05TorchFrontendAmd.infer_batch {self._NOT_PORTED}")

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    # Named per-view observation keys, in buffer order (same naming as the
    # RTX frontend). Alternatively the observation may carry an "images"
    # list with exactly num_views entries in this order.
    _VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")

    def _gather_view_images(self, observation: dict) -> list:
        """Return exactly ``num_views`` validated (224, 224, 3) images.

        Strict by design: a missing or misshaped view raises ValueError
        instead of silently leaving stale ``_img_buf`` rows (which the
        captured graph would then consume as a real camera view).
        """
        expected_keys = self._VIEW_KEYS[:self.num_views]
        if "images" in observation:
            img_list = list(observation["images"])
            if len(img_list) != self.num_views:
                raise ValueError(
                    f"observation['images'] has {len(img_list)} entries; "
                    f"this frontend was built with num_views={self.num_views}")
            labels = [f"images[{i}]" for i in range(len(img_list))]
        else:
            missing = [k for k in expected_keys if k not in observation]
            if missing:
                raise ValueError(
                    f"observation is missing image key(s) {missing} — "
                    f"num_views={self.num_views} requires "
                    f"{list(expected_keys)} (or an 'images' list)")
            if self.num_views < 3 and "wrist_image_right" in observation:
                raise ValueError(
                    "observation provides 'wrist_image_right' but this "
                    f"frontend was built with num_views={self.num_views}; "
                    "construct it with num_views=3 to use a third view")
            img_list = [observation[k] for k in expected_keys]
            labels = list(expected_keys)
        for label, im in zip(labels, img_list):
            if not isinstance(im, np.ndarray) or im.dtype != np.uint8:
                raise ValueError(
                    f"observation image {label!r} must be a uint8 numpy "
                    f"array, got {type(im).__name__}"
                    f"{'/' + str(im.dtype) if isinstance(im, np.ndarray) else ''}")
            if im.shape != (IMG_HW, IMG_HW, 3):
                raise ValueError(
                    f"observation image {label!r} has shape {im.shape}; "
                    f"expected ({IMG_HW}, {IMG_HW}, 3)")
        return img_list

    def _stack_images(self, observation: dict) -> torch.Tensor:
        """Stack and normalize observation images into a new bf16 tensor."""
        img_list = self._gather_view_images(observation)
        tensors = []
        for im in img_list:
            tensors.append(
                torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).to("cuda", bf16))
        return torch.stack(tensors)

    def _fill_img_buf(self, observation: dict) -> None:
        """Fill ``self._img_buf`` in place without allocating new tensors."""
        img_list = self._gather_view_images(observation)
        for v, im in enumerate(img_list):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(bf16))

    def _copy_tensor_to_pipeline_buf(self, src: torch.Tensor, dst_buf) -> None:
        """D2D hipMemcpyAsync from a torch tensor into a HipBuffer slot.

        Uses the current torch stream so downstream ops see the copy.
        """
        stream_int = torch.cuda.current_stream().cuda_stream
        self._copy_tensor_to_pipeline_buf_stream(src, dst_buf, stream_int)

    def _copy_tensor_to_pipeline_buf_stream(
            self, src: torch.Tensor, dst_buf, stream_int: int) -> None:
        """D2D hipMemcpyAsync on a specific stream."""
        nbytes = src.numel() * src.element_size()
        assert nbytes == dst_buf.nbytes, \
            f"size mismatch: src {nbytes} vs dst {dst_buf.nbytes}"
        self._hip_check(self._hip.hipMemcpyAsync(
            dst_buf.ptr, ctypes.c_void_p(src.data_ptr()), nbytes, 3,
            stream_int), "hipMemcpyAsync D2D")
