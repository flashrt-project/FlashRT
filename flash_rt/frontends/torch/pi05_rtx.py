"""FlashRT -- RTX Pi0.5 torch frontend.

Loads HuggingFace PyTorch safetensors checkpoints + drives the
framework-agnostic :class:`~flash_rt.models.pi05.pipeline_rtx.Pi05Pipeline`.

This is the "reference" RTX frontend. The RTX JAX frontend
(:mod:`flash_rt.frontends.jax.pi05_rtx`) mirrors this API but loads
from Orbax and uses JAX for weight quantization.

Usage::

    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtxRtx
    pipe = Pi05TorchFrontendRtxRtx("/path/to/pi05_libero_pytorch", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs_dict])   # once, ~1 s
    out = pipe.infer({"image": img, "wrist_image": wrist})
    actions = out["actions"]     # (chunk_size, 7) numpy
"""

from __future__ import annotations

import ctypes
import json
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
from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
from flash_rt.models.pi05.pipeline_rtx import (
    Pi05Pipeline,
    VIS_L, VIS_D, VIS_H, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H,
    DEC_L, DEC_D, DEC_H, DEC_HD,
    ACTION_DIM, NUM_STEPS_DEFAULT,
)
from flash_rt.models.pi05.pipeline_rtx_cfg import Pi05CFGPipeline
from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline
from flash_rt.models.pi05.pipeline_rtx_cfg_batched import Pi05CFGBatchedPipeline
from flash_rt.hardware.rtx.attn_backend_batched_pi05 import (
    PI05_BATCH_SIZE,
    RtxFlashAttnBatchedBackendPi05,
)

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
                  max_len: int = 48) -> tuple[torch.Tensor, int]:
    """Tokenise + embed via PaliGemma embedding table (CUDA, bf16)."""
    # PaliGemma tokenizer resolution — see
    # `flash_rt.utils.paligemma_tokenizer` for the search order and
    # the download instructions emitted on failure.
    try:
        # Preferred: openpi's PaligemmaTokenizer (exact same vocab,
        # same prompt prefix logic FlashRT was built against).
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_len)
        tokens_np, mask_np = tokenizer.tokenize(prompt_text)
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
        # 108 is PaliGemma's `\n` token, used by openpi as the
        # prompt-end separator before the action prefix.
        tokens = [sp.bos_id()] + sp.Encode(prompt_text) + [108]
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


def _precompute_decoder_styles(ckpt: dict, chunk_size: int,
                               num_steps: int = NUM_STEPS_DEFAULT) -> dict:
    """Pre-compute the time-MLP + per-layer style modulations in torch.

    Output dict has numpy arrays (dtype bf16 via torch→numpy view):
        time_emb:    (num_steps, chunk_size, DEC_D)
        style_attn:  (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_ffn:   (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_final: (num_steps, chunk_size, 3 * DEC_D)

    All computation runs on CUDA in bf16, then is moved to CPU and viewed
    as uint16 so it can be uploaded verbatim to CudaBuffer (bf16 = 2 bytes,
    numpy doesn't natively support bf16 but the bytes round-trip).
    """
    W = {k: v.to("cuda", bf16) if isinstance(v, torch.Tensor) else v
         for k, v in ckpt.items()}

    time_emb_schedule = W["decoder_time_embeds"]              # (num_steps, 1024)
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
#   Pi05TorchFrontendRtx frontend
# ════════════════════════════════════════════════════════════════════


class Pi05TorchFrontendRtx:
    """RTX consumer GPU Pi0.5 Torch frontend.

    Mirrors the :class:`ThorPipelineTorch` public API (``set_prompt`` +
    ``infer`` + ``calibrate_with_real_data`` + ``get_latency_stats``) so the
    same eval scripts work on both hardware families.
    """

    def __init__(self,
                 checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 2,
                 chunk_size: int = CHUNK_SIZE,
                 max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
                 use_fp8: bool = True):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.max_prompt_len = int(max_prompt_len)
        self.use_fp8 = bool(use_fp8)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        self.current_prompt_len = 0
        self.pipeline: Optional[Pi05Pipeline] = None
        # RL inference configuration. ``None`` = default behaviour (single
        # forward, no advantage-conditioned prompt injection). When set
        # by :meth:`set_rl_mode`, the next :meth:`set_prompt` call builds
        # a Pi05CFGPipeline and runs classifier-free guidance.
        self._rl_config: Optional[dict] = None
        self._rl_current_prompt_text: Optional[str] = None

        # ── Load norm_stats ──
        self._load_norm_stats(checkpoint_dir)

        # ── Load + convert safetensors ──
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path} — "
                "Pi05TorchFrontendRtx expects a HuggingFace-style PyTorch checkpoint")
        self._checkpoint_path = str(safetensors_path)
        raw_ckpt = convert_pi05_safetensors(safetensors_path)

        # Move all tensors to CUDA bf16 (retain as member attrs so their
        # memory stays alive across pipeline rebuilds).
        self._ckpt_bf16 = {}
        for k, v in raw_ckpt.items():
            if isinstance(v, torch.Tensor):
                self._ckpt_bf16[k] = v.to("cuda", bf16).contiguous()
            else:
                self._ckpt_bf16[k] = v
        self.embedding_weight = self._ckpt_bf16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps
        num_steps = NUM_STEPS_DEFAULT
        self._ckpt_bf16["decoder_action_out_proj_w"] = \
            self._ckpt_bf16["decoder_action_out_proj_w"] * (-1.0 / num_steps)
        self._ckpt_bf16["decoder_action_out_proj_b"] = \
            self._ckpt_bf16["decoder_action_out_proj_b"] * (-1.0 / num_steps)

        # ── FP8 quantize large GEMM weights ──
        self._fp8_weights: dict = {}
        self._fp8_store: list = []  # holds tensors alive
        if self.use_fp8:
            self._quantize_all_fp8()

        # ── Pre-compute decoder styles (time MLP + style modulation) ──
        self._precomputed_styles = _precompute_decoder_styles(
            self._ckpt_bf16, self.chunk_size, num_steps=num_steps)

        # ── Attention backend (torch, owns Q/K/V/O) ──
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.chunk_size,
            num_encoder_layers=ENC_L)

        # ── fvk module + GemmRunner ──
        from flash_rt import flash_rt_kernels as fvk
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        # ── Reusable pre-allocated input buffers (match Thor style) ──
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=bf16, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        from flash_rt.core.cuda_buffer import _cudart
        self._cudart = _cudart

        logger.info("Pi05TorchFrontendRtx initialised (num_views=%d, chunk=%d)",
                    self.num_views, self.chunk_size)

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir: pathlib.Path) -> None:
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir.parent / "pi05_libero" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            pathlib.Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/"
                         "assets/physical-intelligence/libero/norm_stats.json"),
            *lerobot_candidates(checkpoint_dir),
        ]
        try:
            self.norm_stats = load_norm_stats(
                candidates, checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"norm_stats not found near checkpoint: {e}") from e

    def _quantize_all_fp8(self) -> None:
        """Pre-quantize all large GEMM weights to FP8 E4M3."""
        W = self._ckpt_bf16
        store = self._fp8_store
        fp8 = self._fp8_weights

        def quant(name: str, w: torch.Tensor):
            w_fp8, scale = _quantize_fp8_e4m3(w.contiguous())
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

        logger.info("FP8 quantized %d GEMM weights", len(fp8))

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

            # Precomputed decoder styles (numpy bf16 as uint16 view)
            "precomputed": self._precomputed_styles,
        }
        return weights

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_rl_mode(
        self,
        *,
        cfg_enable: bool = True,
        cfg_beta: float = 1.5,
        advantage_positive: bool = True,
    ) -> None:
        """Enable / configure advantage-conditioned RL inference (opt-in).

        Once enabled, subsequent :meth:`set_prompt` calls will build a
        :class:`Pi05CFGPipeline` instead of the standard
        :class:`Pi05Pipeline`. The conditioned prompt has the
        ``"Advantage: positive"`` (or ``"negative"``) tag appended; the
        unconditioned prompt is the original task text. Each denoising
        step runs the action expert twice and combines the two velocity
        predictions with strength ``cfg_beta``.

        Calling this with ``cfg_enable=False`` clears any RL configuration
        so the next :meth:`set_prompt` reverts to the standard pipeline
        (this rebuilds the pipeline so the change takes effect).

        Args:
            cfg_enable: If ``True``, activate CFG inference. If
                ``False``, clear any previous RL configuration.
            cfg_beta: CFG guidance strength. Must be ``>= 1.0``. Common
                deployment range is ``[1.5, 2.5]``. Ignored when
                ``cfg_enable`` is ``False``.
            advantage_positive: Whether the conditioned prompt uses the
                positive advantage tag (the standard "select for high
                advantage" use case). Set ``False`` only for debugging.
        """
        if not cfg_enable:
            self._rl_config = None
            # If a CFG pipeline was previously built, drop it so the
            # next set_prompt rebuilds the standard pipeline.
            if isinstance(self.pipeline, Pi05CFGPipeline):
                self.pipeline = None
                self.current_prompt_len = 0
                self.graph_recorded = False
                self.calibrated = False
            return
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        new_config = {
            "cfg_beta": float(cfg_beta),
            "advantage_positive": bool(advantage_positive),
        }
        if self._rl_config != new_config:
            self._rl_config = new_config
            # Force pipeline rebuild on next set_prompt so the new mode
            # / beta takes effect.
            self.pipeline = None
            self.current_prompt_len = 0
            self.graph_recorded = False
            self.calibrated = False
        logger.info(
            "RL mode enabled: cfg_beta=%.2f, advantage_positive=%s",
            new_config["cfg_beta"], new_config["advantage_positive"])

    def set_prompt(self, prompt_text: str) -> None:
        """Tokenise prompt + (re)build the pipeline for the exact prompt length.

        When RL mode is enabled (see :meth:`set_rl_mode`), this also
        builds the unconditioned prompt embeddings and uploads both into
        the CFG-aware pipeline.
        """
        if self._rl_config is not None:
            self._set_prompt_rl(prompt_text)
            return

        embeds, prompt_len = _embed_prompt(
            prompt_text, self.embedding_weight, max_len=MAX_PROMPT_LEN_DEFAULT)

        if self.pipeline is None or prompt_len != self.current_prompt_len:
            logger.info("Building Pi05Pipeline for prompt_len=%d...", prompt_len)
            # Rebuild the pipeline with the exact prompt length to avoid
            # wasted compute on padding tokens.
            self.current_prompt_len = prompt_len
            self.graph_recorded = False
            self.calibrated = False

            pipeline_weights = self._build_pipeline_weights()
            self.pipeline = Pi05Pipeline(
                gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                weights=pipeline_weights,
                num_views=self.num_views,
                max_prompt_len=prompt_len,
                chunk_size=self.chunk_size,
                use_fp8=self.use_fp8, use_fp8_decoder=self.use_fp8)

        # Upload language embeds into pipeline's encoder_x slot
        embeds_np = embeds.contiguous().view(torch.uint16).cpu().numpy()
        self.pipeline.set_language_embeds(embeds_np)
        logger.info("Set prompt: '%s' (%d tokens)", prompt_text, prompt_len)

    def _set_prompt_rl(self, prompt_text: str) -> None:
        """RL-mode set_prompt: build conditioned + unconditioned embeddings.

        When batched mode is also active (Phase 3b), the pipeline type
        is :class:`Pi05CFGBatchedPipeline` which runs cond + uncond as
        the two slots of a B=2 fused forward. Otherwise the serial
        :class:`Pi05CFGPipeline` runs them sequentially (Phase 1+2).
        """
        from flash_rt.core.rl import build_acp_tagged_task

        cfg = self._rl_config
        if cfg is None:
            raise RuntimeError("_set_prompt_rl called without RL config")

        cond_text = build_acp_tagged_task(
            prompt_text, is_positive=cfg["advantage_positive"])
        uncond_text = prompt_text

        cond_embeds, cond_len = _embed_prompt(
            cond_text, self.embedding_weight, max_len=MAX_PROMPT_LEN_DEFAULT)
        uncond_embeds, uncond_len = _embed_prompt(
            uncond_text, self.embedding_weight, max_len=MAX_PROMPT_LEN_DEFAULT)
        target_len = max(cond_len, uncond_len)

        use_batched_cfg = getattr(self, "_batched_active", False)

        if use_batched_cfg:
            expected_cls = Pi05CFGBatchedPipeline
            cls_name = "Pi05CFGBatchedPipeline"
        else:
            expected_cls = Pi05CFGPipeline
            cls_name = "Pi05CFGPipeline"

        rebuild = (
            self.pipeline is None
            or not isinstance(self.pipeline, expected_cls)
            or target_len != self.current_prompt_len
            or self.pipeline.cfg_beta != cfg["cfg_beta"])

        if rebuild:
            logger.info(
                "Building %s for prompt_len=%d (cfg_beta=%.2f)...",
                cls_name, target_len, cfg["cfg_beta"])
            self.current_prompt_len = target_len
            self.graph_recorded = False
            self.calibrated = False

            pipeline_weights = self._build_pipeline_weights()
            if use_batched_cfg:
                # Need the batched attention backend (already set up by
                # set_batched_mode).
                if not isinstance(self.attn_backend,
                                  RtxFlashAttnBatchedBackendPi05):
                    raise RuntimeError(
                        "batched CFG requires set_batched_mode(enable=True) "
                        "to have been called first to install the batched "
                        "attention backend")
                self.pipeline = Pi05CFGBatchedPipeline(
                    gemm=self.gemm, fvk=self.fvk,
                    attn_backend=self.attn_backend,
                    weights=pipeline_weights,
                    num_views=self.num_views,
                    max_prompt_len=target_len,
                    chunk_size=self.chunk_size,
                    use_fp8=self.use_fp8, use_fp8_decoder=self.use_fp8,
                    cfg_beta=cfg["cfg_beta"])
            else:
                self.pipeline = Pi05CFGPipeline(
                    gemm=self.gemm, fvk=self.fvk,
                    attn_backend=self.attn_backend,
                    weights=pipeline_weights,
                    num_views=self.num_views,
                    max_prompt_len=target_len,
                    chunk_size=self.chunk_size,
                    use_fp8=self.use_fp8, use_fp8_decoder=self.use_fp8,
                    cfg_beta=cfg["cfg_beta"])

        cond_np = cond_embeds.contiguous().view(torch.uint16).cpu().numpy()
        uncond_np = uncond_embeds.contiguous().view(torch.uint16).cpu().numpy()

        if use_batched_cfg:
            # Pad both to target_len here (the batched set_language_embeds_batch
            # inherited by Pi05CFGBatchedPipeline expects equal prompt lengths).
            def _pad(arr, to_len):
                if arr.shape[0] == to_len:
                    return np.ascontiguousarray(arr)
                pad = np.zeros((to_len - arr.shape[0], arr.shape[1]),
                               dtype=arr.dtype)
                return np.ascontiguousarray(np.concatenate([arr, pad], axis=0))
            cond_np = _pad(cond_np, target_len)
            uncond_np = _pad(uncond_np, target_len)
            # Also seed parent's B=1 lang slot for the FP8 calibration pass
            # (same pattern set_prompt_batch uses).
            self.pipeline.set_language_embeds(cond_np)

        self.pipeline.set_language_embeds_pair(cond_np, uncond_np)
        self._rl_current_prompt_text = prompt_text
        logger.info(
            "Set RL prompt: '%s' (cond_len=%d, uncond_len=%d, padded=%d, batched=%s)",
            prompt_text, cond_len, uncond_len, target_len, use_batched_cfg)

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration entry point (see Pi0TorchFrontendRtx.calibrate).

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

        if n == 1:
            self._calibrate_single_frame(obs_list[0])
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    def _calibrate_single_frame(self, sample) -> None:
        logger.info("Calibrating FP8 with a single real sample...")

        # Create a dedicated torch stream for both the calibration pass and
        # graph capture so flash_attn_func + our fvk kernels land on the
        # same stream.
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

            # Batched pipelines carry their own calibrate_fp8 that drives
            # a parent-B=1 forward internally — calling run_pipeline here
            # would fire the batched path with only the parent's B=1
            # slots populated. Skip the preemptive run for batched
            # subclasses and let calibrate_fp8 do the work.
            if not isinstance(self.pipeline, Pi05BatchedPipeline):
                self.pipeline.run_pipeline(stream=stream_int)

            self._cudart.cudaStreamSynchronize(
                ctypes.c_void_p(stream_int))

            # Flip calibrated flag (scales populated by run_pipeline above,
            # or by calibrate_fp8's own B=1 forward for batched pipelines)
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
            "Calibrating FP8 across %d real samples (percentile=%.2f)...",
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
                self._cudart.cudaStreamSynchronize(
                    ctypes.c_void_p(stream_int))

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
        self._warn_if_scale_ceiling_exceeded(label=f"pi05_rtx_N{n}")
        logger.info(
            "Pi0.5 multi-frame calibration + graph capture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def _zero_pipeline_scales(self) -> None:
        zero = np.zeros(1, dtype=np.float32)
        for buf in self.pipeline.fp8_act_scales.values():
            buf.upload(zero)

    def _warn_if_scale_ceiling_exceeded(self, label: str = "pi05_rtx") -> None:
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

    def infer(self, observation: dict, debug: bool = False) -> dict:
        """Run inference on a single observation.

        All GPU work happens on ``self._graph_torch_stream`` — the same
        stream the graph was captured on — so replay + pre/post D2D copies
        are serialized correctly.

        When the active pipeline is :class:`Pi05CFGBatchedPipeline`
        (RL mode + batched mode both on), this routes through a B=2
        forward that fuses CFG's conditioned and unconditioned branches
        into a single captured graph. The single ``observation`` is
        replicated across both batch slots (cond and uncond use the
        same image / state); the two prompts differ and were already
        uploaded by :meth:`_set_prompt_rl`.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before infer")

        if isinstance(self.pipeline, Pi05CFGBatchedPipeline):
            return self._infer_cfg_batched(observation, debug=debug)

        t0 = time.perf_counter()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            self._fill_img_buf(observation)
            self._noise_buf.normal_()

            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self.pipeline.input_noise_buf, stream_int)

            # Graph replay (on the same captured stream)
            out_ptr = self.pipeline.forward()

            # D2D download → staging torch tensor
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = self._noise_out.float().cpu().numpy()  # (chunk, 32)
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]

        if debug:
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("Latency: %.1f ms", latency_ms)

        return {"actions": robot_actions}

    def _infer_cfg_batched(self, observation: dict,
                           debug: bool = False) -> dict:
        """Batched CFG inference: single obs replicated across cond + uncond slots."""
        t0 = time.perf_counter()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            # Replicate the single observation into both batch slots.
            stacked = self._stack_images(observation)
            for b in range(PI05_BATCH_SIZE):
                self._img_buf_b2[b].copy_(stacked)
            # Each denoising step starts from independent noise in each
            # slot; cond slot is the one CFG reads / updates. Sampling
            # once and copying into both slots ensures the uncond slot
            # starts at the same noise the cond does, which matches
            # the paper-faithful CFG contract.
            self._noise_buf.normal_()
            for b in range(PI05_BATCH_SIZE):
                self._noise_buf_b2[b].copy_(self._noise_buf)

            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf_b2, self.pipeline.input_images_buf_b2, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf_b2, self.pipeline.input_noise_buf_b2, stream_int)

            # Graph replay returns the cond slot's noise pointer.
            out_ptr = self.pipeline.forward()

            # D2D download of just the cond slot (chunk * ACTION_DIM bf16)
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = self._noise_out.float().cpu().numpy()
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]

        if debug:
            logger.info(
                "CFG batched raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("CFG batched latency: %.1f ms", latency_ms)

        return {"actions": robot_actions}

    # -----------------------------------------------------------------
    # Batched (B=2) inference path — additive, default API unchanged
    # -----------------------------------------------------------------

    def set_batched_mode(self, *, enable: bool = True) -> None:
        """Enable / disable the B=2 batched inference path (opt-in).

        Once enabled, the next :meth:`set_prompt_batch` call builds a
        :class:`Pi05BatchedPipeline` (with a
        :class:`RtxFlashAttnBatchedBackendPi05` attention backend) and
        :meth:`infer_batch` becomes available. The single-sample
        :meth:`infer` API path remains untouched.

        Disabling rebuilds the standard single-sample pipeline on the
        next :meth:`set_prompt`.
        """
        if not enable:
            if isinstance(self.pipeline, Pi05BatchedPipeline):
                self.pipeline = None
                self.current_prompt_len = 0
                self.graph_recorded = False
                self.calibrated = False
                self._batched_active = False
            return
        # Switch to a batched-capable attention backend if not already.
        if not isinstance(self.attn_backend, RtxFlashAttnBatchedBackendPi05):
            enc_seq_max = self.num_views * 256 + self.max_prompt_len
            self.attn_backend = RtxFlashAttnBatchedBackendPi05(
                num_views=self.num_views,
                encoder_seq_max=enc_seq_max,
                chunk_size=self.chunk_size,
                num_encoder_layers=ENC_L)
        self._batched_active = True
        # Force pipeline rebuild so set_prompt_batch picks the batched class.
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            self.pipeline = None
            self.current_prompt_len = 0
            self.graph_recorded = False
            self.calibrated = False
        # Pre-allocate batched input/output staging tensors.
        self._img_buf_b2 = torch.empty(
            PI05_BATCH_SIZE, self.num_views, IMG_HW, IMG_HW, 3,
            dtype=bf16, device="cuda")
        self._noise_buf_b2 = torch.empty(
            PI05_BATCH_SIZE, self.chunk_size, ACTION_DIM,
            dtype=bf16, device="cuda")
        self._noise_out_b2 = torch.empty(
            PI05_BATCH_SIZE, self.chunk_size, ACTION_DIM,
            dtype=bf16, device="cuda")
        logger.info(
            "Pi05TorchFrontendRtx: batched mode enabled (B=%d)",
            PI05_BATCH_SIZE)

    def set_prompt_batch(self, prompts: list) -> None:
        """Set per-sample prompts for the batched pipeline.

        Args:
            prompts: list of length B (currently 2). Each entry is a
                task description string. Prompts are individually
                tokenised, then padded to a common length so the
                encoder sees a fixed-shape buffer.
        """
        if not getattr(self, "_batched_active", False):
            raise RuntimeError(
                "set_batched_mode(enable=True) must be called before "
                "set_prompt_batch")
        if len(prompts) != PI05_BATCH_SIZE:
            raise ValueError(
                f"set_prompt_batch expects {PI05_BATCH_SIZE} prompts, "
                f"got {len(prompts)}")
        embeds_list = []
        prompt_lens = []
        for p in prompts:
            e, plen = _embed_prompt(p, self.embedding_weight,
                                    max_len=MAX_PROMPT_LEN_DEFAULT)
            embeds_list.append(e)
            prompt_lens.append(plen)
        target_len = max(prompt_lens)

        # Pad each embed to target_len (BF16 zeros are valid pad tokens).
        padded_np_list = []
        for e, plen in zip(embeds_list, prompt_lens):
            arr = e.contiguous().view(torch.uint16).cpu().numpy()
            if plen < target_len:
                pad = np.zeros(
                    (target_len - plen, arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            padded_np_list.append(np.ascontiguousarray(arr))

        rebuild = (
            self.pipeline is None
            or not isinstance(self.pipeline, Pi05BatchedPipeline)
            or target_len != self.current_prompt_len)

        if rebuild:
            logger.info(
                "Building Pi05BatchedPipeline (B=%d) for prompt_len=%d...",
                PI05_BATCH_SIZE, target_len)
            self.current_prompt_len = target_len
            self.graph_recorded = False
            self.calibrated = False
            pipeline_weights = self._build_pipeline_weights()
            self.pipeline = Pi05BatchedPipeline(
                gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                weights=pipeline_weights,
                num_views=self.num_views,
                max_prompt_len=target_len,
                chunk_size=self.chunk_size,
                use_fp8=self.use_fp8, use_fp8_decoder=self.use_fp8)

        # Also seed the parent's B=1 lang slot from sample 0; the parent's
        # B=1 pipeline path is what calibrate_fp8 uses for FP8 scale collection.
        self.pipeline.set_language_embeds(padded_np_list[0])
        self.pipeline.set_language_embeds_batch(padded_np_list)
        logger.info(
            "Set batch prompt (B=%d, padded_len=%d): %s",
            PI05_BATCH_SIZE, target_len,
            [p[:30] + ("…" if len(p) > 30 else "") for p in prompts])

    def calibrate_batch(self, sample_observations) -> None:
        """Calibrate FP8 scales for the batched pipeline.

        Uses the parent B=1 calibration pass (per-tensor scales are
        sample-invariant) on the first observation; the batched B=2
        forward then reuses those scales.
        """
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            raise RuntimeError(
                "calibrate_batch requires set_prompt_batch to have built a "
                "Pi05BatchedPipeline first")
        if isinstance(sample_observations, dict):
            sample_observations = [sample_observations]
        sample = sample_observations[0]

        # Mirror calibrate(): write inputs into the parent B=1 buffers,
        # call parent's calibrate_fp8 + autotune + record graph.
        self._graph_torch_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            images = self._stack_images(sample)
            noise = torch.randn(self.chunk_size, ACTION_DIM,
                                dtype=bf16, device="cuda")
            self._copy_tensor_to_pipeline_buf_stream(
                images, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                noise, self.pipeline.input_noise_buf, stream_int)
            self.pipeline.calibrate_fp8()
            self.pipeline.autotune_gemms()
            self.pipeline.record_infer_graph(external_stream_int=stream_int)
        self.calibrated = True
        self.graph_recorded = True

    def infer_batch(self, observations: list) -> list:
        """Run B=2 inference on two independent observations.

        Args:
            observations: list of length B (currently 2) of obs dicts
                matching :meth:`infer`'s contract (``image``,
                ``wrist_image`` if ``num_views >= 2``, ``state``).

        Returns:
            List of length B; each entry is ``{"actions": (action_horizon, action_dim)}``.
        """
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            raise RuntimeError("set_batched_mode + set_prompt_batch required")
        if len(observations) != PI05_BATCH_SIZE:
            raise ValueError(
                f"infer_batch expects {PI05_BATCH_SIZE} observations, "
                f"got {len(observations)}")
        t0 = time.perf_counter()

        # Stage per-sample inputs into the B=2 staging tensors, then D2D.
        for b, obs in enumerate(observations):
            self._img_buf_b2[b].copy_(self._stack_images(obs))
        self._noise_buf_b2.normal_()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf_b2, self.pipeline.input_images_buf_b2, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf_b2, self.pipeline.input_noise_buf_b2, stream_int)

            out_ptr = self.pipeline.forward()

            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out_b2.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out_b2.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        results = []
        for b in range(PI05_BATCH_SIZE):
            raw = self._noise_out_b2[b].float().cpu().numpy()
            unnorm = unnormalize_actions(raw, self.norm_stats)
            results.append({"actions": unnorm[:, :LIBERO_ACTION_DIM]})
        return results

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
    # Internals
    # -----------------------------------------------------------------

    def _stack_images(self, observation: dict) -> torch.Tensor:
        """Stack and normalize observation images into a new bf16 tensor."""
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        tensors = []
        for im in img_list[:self.num_views]:
            tensors.append(
                torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).to("cuda", bf16))
        return torch.stack(tensors)

    def _fill_img_buf(self, observation: dict) -> None:
        """Fill ``self._img_buf`` in place without allocating new tensors."""
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(bf16))

    def _copy_tensor_to_pipeline_buf(self, src: torch.Tensor, dst_buf) -> None:
        """D2D cudaMemcpyAsync from a torch tensor into a CudaBuffer slot.

        Uses the current torch stream so downstream ops see the copy.
        """
        stream_int = torch.cuda.current_stream().cuda_stream
        self._copy_tensor_to_pipeline_buf_stream(src, dst_buf, stream_int)

    def _copy_tensor_to_pipeline_buf_stream(
            self, src: torch.Tensor, dst_buf, stream_int: int) -> None:
        """D2D cudaMemcpyAsync on a specific stream."""
        nbytes = src.numel() * src.element_size()
        assert nbytes == dst_buf.nbytes, \
            f"size mismatch: src {nbytes} vs dst {dst_buf.nbytes}"
        self._cudart.cudaMemcpyAsync(
            dst_buf.ptr, ctypes.c_void_p(src.data_ptr()), nbytes, 3, stream_int)
