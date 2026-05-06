"""FlashRT -- RTX Pi0 torch frontend.

Mirrors :mod:`flash_rt.frontends.torch.pi05_rtx` but targets Pi0 (not
Pi0.5): standard RMSNorm decoder, ``state_proj`` + ``action_time_mlp``
pre-block, ``S_dec = Sa + 1`` sequence, and a state-masked cross-attention
via the attention backend's ``run("decoder", ..., state_nk=...)`` kwarg.

Checkpoint layout -- expects the PyTorch safetensors produced by
``openpi_src/examples/convert_jax_model_to_pytorch.py`` (a converted
Orbax checkpoint) plus ``assets/physical-intelligence/<task>/norm_stats.json``.

Usage::

    from flash_rt.frontends.torch.pi0_rtx import Pi0TorchFrontendRtx
    pipe = Pi0TorchFrontendRtx("/path/to/pi0_libero_pytorch", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs])
    out = pipe.infer({"image": img, "wrist_image": wrist, "state": state})
    actions = out["actions"]
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
from flash_rt.models.pi0.pipeline_rtx import (
    Pi0Pipeline,
    VIS_L, VIS_D, VIS_H, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H,
    DEC_L, DEC_D, DEC_H, DEC_HD,
    ACTION_DIM, CHUNK_SIZE_DEFAULT, NUM_STEPS_DEFAULT,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8_e4m3 = torch.float8_e4m3fn

IMG_HW = 224
MAX_PROMPT_LEN_DEFAULT = 48


# ════════════════════════════════════════════════════════════════════
#   HF safetensors → pipeline weight dict
# ════════════════════════════════════════════════════════════════════


def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .permute(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


def convert_pi0_safetensors(safetensors_path: Union[str, pathlib.Path]) -> dict:
    """Convert a Pi0 PyTorch safetensors file to the BF16 weight dict.

    Key transformations (matching pi05 for the shared vision/encoder
    backbones, Pi0-specific for the decoder):

      * Vision: patch embed permute, Q/K/V concat-transpose.
      * Encoder: Q/K head interleave for fused-RoPE; fuse
        ``(1 + input_layernorm.weight)`` into QKV, ``(1 + post_attn)``
        into gate+up — FP32 to avoid bf16 rounding near -1.
      * Decoder: same RMSNorm fold pattern as the encoder. Pi0 uses a
        standard (non-AdaRMS) RMSNorm so the expert layer weights are
        folded into QKV/gate+up exactly like the encoder. Final norm
        is folded inline ``(1 + w)``.
      * Action / state MLPs: raw transposes for GEMM-friendly layout,
        with ``action_out_proj`` pre-scaled by ``-1 / num_steps`` to
        match the flow-matching residual accumulation.
      * ``action_time_mlp_in.weight`` is split on the input axis: the
        action half (``[:, :Da]``) is kept as a runtime GEMM weight;
        the time half (``[:, Da:]``) is consumed by
        :func:`_precompute_time_proj_all` and discarded.
    """
    from safetensors import safe_open
    from flash_rt.executors.torch_weights import _autodetect_strip_prefix

    logger.info("Loading Pi0 safetensors: %s", safetensors_path)
    f = safe_open(str(safetensors_path), framework="pt")
    # Auto-strip the lerobot HF policy ``model.`` wrap so the openpi
    # bare-key lookups below resolve on either layout.
    _strip = _autodetect_strip_prefix(set(f.keys()))

    def g(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key).to(fp16)

    def g_raw(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key)

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) — identical to Pi0.5 ──
    vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    pe_w = g(f"{vp}.embeddings.patch_embedding.weight")
    ckpt["vision_patch_embedding_w"] = pe_w.permute(2, 3, 1, 0).contiguous()
    ckpt["vision_patch_embedding_b"] = g(
        f"{vp}.embeddings.patch_embedding.bias")
    ckpt["vision_position_embedding"] = g(
        f"{vp}.embeddings.position_embedding.weight")

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

    # ── Encoder (18 Gemma-2B layers) — identical fold to Pi0.5 ──
    ep = "paligemma_with_expert.paligemma.model.language_model.layers"
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []
    for i in range(ENC_L):
        attn_scale = g_raw(f"{ep}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale
        q_w = g_raw(f"{ep}.{i}.self_attn.q_proj.weight").float()
        k_w = g_raw(f"{ep}.{i}.self_attn.k_proj.weight").float()
        v_w = g_raw(f"{ep}.{i}.self_attn.v_proj.weight").float()
        q_w = _interleave_qk(q_w, 8)
        k_w = _interleave_qk(k_w, 1)
        q_w = q_w * fuse_attn.unsqueeze(0)
        k_w = k_w * fuse_attn.unsqueeze(0)
        v_w = v_w * fuse_attn.unsqueeze(0)
        enc_qkv_list.append(torch.cat([q_w, k_w, v_w], dim=0).t().to(fp16))
        enc_o_list.append(g(f"{ep}.{i}.self_attn.o_proj.weight").t())

        ffn_scale = g_raw(f"{ep}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale
        gate_w = (g_raw(f"{ep}.{i}.mlp.gate_proj.weight").float()
                  * fuse_ffn.unsqueeze(0))
        up_w = (g_raw(f"{ep}.{i}.mlp.up_proj.weight").float()
                * fuse_ffn.unsqueeze(0))
        enc_gate_list.append(gate_w.t().to(fp16))
        enc_up_list.append(up_w.t().to(fp16))
        enc_down_list.append(g(f"{ep}.{i}.mlp.down_proj.weight").t())

    ckpt["encoder_attn_qkv_w"] = torch.stack(enc_qkv_list)
    ckpt["encoder_attn_o_w"] = torch.stack(enc_o_list)
    ckpt["encoder_ffn_gate_w"] = torch.stack(enc_gate_list)
    ckpt["encoder_ffn_up_w"] = torch.stack(enc_up_list)
    ckpt["encoder_ffn_down_w"] = torch.stack(enc_down_list)

    # ── Decoder (18 Gemma-300M layers) — Pi0 standard RMSNorm fold ──
    dp = "paligemma_with_expert.gemma_expert.model.layers"
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    for i in range(DEC_L):
        attn_scale = g_raw(f"{dp}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale
        q_w = g_raw(f"{dp}.{i}.self_attn.q_proj.weight").float()
        k_w = g_raw(f"{dp}.{i}.self_attn.k_proj.weight").float()
        v_w = g_raw(f"{dp}.{i}.self_attn.v_proj.weight").float()
        q_w = _interleave_qk(q_w, 8)
        k_w = _interleave_qk(k_w, 1)
        q_w = q_w * fuse_attn.unsqueeze(0)
        k_w = k_w * fuse_attn.unsqueeze(0)
        v_w = v_w * fuse_attn.unsqueeze(0)
        dec_qkv_list.append(torch.cat([q_w, k_w, v_w], dim=0).t().to(fp16))
        dec_o_list.append(g(f"{dp}.{i}.self_attn.o_proj.weight").t())

        ffn_scale = g_raw(f"{dp}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale
        gate_w = (g_raw(f"{dp}.{i}.mlp.gate_proj.weight").float()
                  * fuse_ffn.unsqueeze(0))
        up_w = (g_raw(f"{dp}.{i}.mlp.up_proj.weight").float()
                * fuse_ffn.unsqueeze(0))
        dec_gate_list.append(gate_w.t().to(fp16))
        dec_up_list.append(up_w.t().to(fp16))
        dec_down_list.append(g(f"{dp}.{i}.mlp.down_proj.weight").t())

    ckpt["decoder_attn_qkv_w"] = torch.stack(dec_qkv_list)
    ckpt["decoder_attn_o_w"] = torch.stack(dec_o_list)
    ckpt["decoder_ffn_gate_w"] = torch.stack(dec_gate_list)
    ckpt["decoder_ffn_up_w"] = torch.stack(dec_up_list)
    ckpt["decoder_ffn_down_w"] = torch.stack(dec_down_list)

    # Final norm — fold (1 + w) inline.
    final_norm_raw = g_raw(
        "paligemma_with_expert.gemma_expert.model.norm.weight").float()
    ckpt["decoder_final_norm_w"] = (1.0 + final_norm_raw).to(fp16).contiguous()

    # ── Pi0 action / state projections ──
    # state_proj.weight is (Da, 32); transpose to (32, Da) for GEMM.
    ckpt["state_proj_w"] = g("state_proj.weight").t().contiguous()
    ckpt["state_proj_b"] = g("state_proj.bias").contiguous()

    ckpt["decoder_action_in_proj_w"] = g("action_in_proj.weight").t().contiguous()
    ckpt["decoder_action_in_proj_b"] = g("action_in_proj.bias").contiguous()

    # action_time_mlp_in.weight is (Da, 2*Da). Action half goes to runtime
    # GEMM (transposed for our NN kernel); time half is absorbed into
    # the pre-computed per-step ``time_proj_all`` buffer.
    atmlp_in_full = g_raw("action_time_mlp_in.weight").to(fp16)  # (Da, 2*Da)
    ckpt["action_time_mlp_in_wa_w"] = (
        atmlp_in_full[:, :DEC_D].t().contiguous())  # (Da, Da) GEMM-B
    ckpt["_action_time_mlp_in_wt_raw"] = (
        atmlp_in_full[:, DEC_D:].contiguous())      # (Da, Da) for precompute
    ckpt["_action_time_mlp_in_b"] = g("action_time_mlp_in.bias").contiguous()

    ckpt["action_time_mlp_out_w"] = g("action_time_mlp_out.weight").t().contiguous()
    ckpt["action_time_mlp_out_b"] = g("action_time_mlp_out.bias").contiguous()

    # action_out_proj — pre-scaled by -1/num_steps.
    dt_scale = -1.0 / NUM_STEPS_DEFAULT
    ckpt["decoder_action_out_proj_w"] = (
        g("action_out_proj.weight").t().contiguous() * dt_scale).contiguous()
    ckpt["decoder_action_out_proj_b"] = (
        g("action_out_proj.bias").contiguous() * dt_scale).contiguous()

    # ── Embedding matrix (prompt tokenisation) ──
    ckpt["embedding_weight"] = g(
        "paligemma_with_expert.paligemma.lm_head.weight")

    logger.info("Converted %d Pi0 weight groups", len(ckpt))
    return ckpt


def _precompute_time_proj_all(ckpt: dict, chunk_size: int,
                              num_steps: int = NUM_STEPS_DEFAULT) -> torch.Tensor:
    """Pre-compute ``silu(time_emb[s] @ W_t.T + b) → broadcast to Sa rows``.

    Output: ``(num_steps * chunk_size, DEC_D)`` bf16 on CUDA. Frontend
    stores the pointer; the pipeline reads step slice
    ``[s * Sa * DEC_D * 2 : (s+1) * Sa * DEC_D * 2]`` bytes per step.

    Note: unlike Pi0.5's style pre-compute, Pi0 does NOT apply silu here
    — the ``action_time_mlp`` silu lives inside the per-step assembly
    (``fused_add_silu_bf16``). What we pre-compute is just the time
    half of the linear map + bias, identical to ``pi0_thor``.
    """
    # Keep the projection step in fp32 to match the reference's dtype
    # (PI0Pytorch's embed_suffix casts time_emb to timestep.dtype — in
    # the eval harness, timestep is fp32, so the linear map stays full
    # precision). bf16 casting here is only applied to the final buffer
    # that the decoder kernel consumes.
    W_t = ckpt["_action_time_mlp_in_wt_raw"].to("cuda", torch.float32)
    b = ckpt["_action_time_mlp_in_b"].to("cuda", torch.float32)

    fraction = torch.linspace(0.0, 1.0, DEC_D // 2,
                              dtype=torch.float64, device="cuda")
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling = 1.0 / period * 2 * math.pi

    out = torch.empty(num_steps, chunk_size, DEC_D, dtype=fp16, device="cuda")
    for step in range(num_steps):
        t_val = 1.0 - step / num_steps
        sin_input = scaling * t_val  # fp64
        time_emb_f32 = torch.cat(
            [torch.sin(sin_input), torch.cos(sin_input)],
            dim=-1).to(torch.float32)  # (Da,) fp32
        tp = (time_emb_f32.unsqueeze(0) @ W_t.t()
              + b.unsqueeze(0)).to(fp16)  # (1, Da) → bf16
        out[step] = tp.expand(chunk_size, -1)

    return out.reshape(num_steps * chunk_size, DEC_D).contiguous()


# ════════════════════════════════════════════════════════════════════
#   Weight FP8 quantization
# ════════════════════════════════════════════════════════════════════


def _quantize_fp8_e4m3(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w_bf16.float().abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w_bf16.float() / scale).clamp(-448.0, 448.0).to(fp8_e4m3)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device="cuda")
    return w_fp8, scale_tensor


# ════════════════════════════════════════════════════════════════════
#   Pi0TorchFrontendRtx
# ════════════════════════════════════════════════════════════════════


class Pi0TorchFrontendRtx:
    """RTX Pi0 Torch frontend. Mirrors the Pi0.5 frontend public API
    (``set_prompt`` / ``infer`` / ``calibrate_with_real_data`` /
    ``get_latency_stats``) so eval scripts can swap between Pi0 and Pi0.5.
    ``observation`` must include a ``"state"`` numpy array.
    """

    def __init__(self,
                 checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 2,
                 chunk_size: int = CHUNK_SIZE_DEFAULT,
                 max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
                 use_fp8: bool = True,
                 use_fp8_decoder: bool = True):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)            # Sa
        self.S_dec = self.chunk_size + 1
        self.max_prompt_len = int(max_prompt_len)
        self._use_fp8 = bool(use_fp8)
        self._use_fp8_decoder = bool(use_fp8_decoder)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        self.current_prompt_len = 0
        self.pipeline: Optional[Pi0Pipeline] = None

        self._load_norm_stats(checkpoint_dir)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path} — "
                "Pi0TorchFrontendRtx expects a PyTorch safetensors Pi0 "
                "checkpoint (generated by "
                "openpi_src/examples/convert_jax_model_to_pytorch.py)")
        self._checkpoint_path = str(safetensors_path)
        raw_ckpt = convert_pi0_safetensors(safetensors_path)

        self._ckpt_fp16: dict = {}
        for k, v in raw_ckpt.items():
            if isinstance(v, torch.Tensor):
                self._ckpt_fp16[k] = v.to("cuda", fp16).contiguous()
            else:
                self._ckpt_fp16[k] = v
        self.embedding_weight = self._ckpt_fp16["embedding_weight"]

        # Pre-compute time_proj_all (keeps underlying tensor alive).
        self._time_proj_all = _precompute_time_proj_all(
            self._ckpt_fp16, self.chunk_size, num_steps=NUM_STEPS_DEFAULT)

        # FP8 quantize large GEMMs.
        self._fp8_weights: dict = {}
        self._fp8_store: list = []
        self._quantize_all_fp8()

        # Attention backend — pass S_dec as the "decoder chunk" so its
        # dec_Q / dec_O_masked / enc_K extend KV capacity include the
        # extra state-token row.
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.S_dec,
            num_encoder_layers=ENC_L,
            dtype=fp16)

        from flash_rt import flash_rt_kernels as fvk
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        self._state_buf_host = torch.empty(
            1, ACTION_DIM, dtype=fp16, device="cuda")
        self._cudart = ctypes.CDLL("libcudart.so")

        logger.info("Pi0TorchFrontendRtx initialised (num_views=%d, Sa=%d)",
                    self.num_views, self.chunk_size)

    def _load_norm_stats(self, checkpoint_dir: pathlib.Path) -> None:
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero"
            / "norm_stats.json",
            checkpoint_dir.parent / "pi0_base" / "assets"
            / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            *lerobot_candidates(checkpoint_dir),
        ]
        try:
            self.norm_stats = load_norm_stats(
                candidates, checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"norm_stats not found near Pi0 checkpoint {checkpoint_dir}: "
                f"{e}") from e

    def _quantize_all_fp8(self) -> None:
        W = self._ckpt_fp16
        store = self._fp8_store
        fp8 = self._fp8_weights

        def quant(name: str, w: torch.Tensor) -> None:
            w_fp8, scale = _quantize_fp8_e4m3(w.contiguous())
            store.append(w_fp8)
            store.append(scale)
            fp8[name] = (w_fp8.data_ptr(), scale.data_ptr())

        for i in range(VIS_L):
            quant(f"vision_attn_qkv_w_{i}", W["vision_attn_qkv_w"][i])
            quant(f"vision_attn_o_w_{i}", W["vision_attn_o_w"][i])
            quant(f"vision_ffn_up_w_{i}", W["vision_ffn_up_w"][i])
            quant(f"vision_ffn_down_w_{i}", W["vision_ffn_down_w"][i])
        quant("vision_projector_w", W["encoder_multi_modal_projector_w"])

        for i in range(ENC_L):
            quant(f"encoder_attn_qkv_w_{i}", W["encoder_attn_qkv_w"][i])
            quant(f"encoder_attn_o_w_{i}", W["encoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["encoder_ffn_gate_w"][i], W["encoder_ffn_up_w"][i]],
                dim=1).contiguous()
            quant(f"encoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"encoder_ffn_down_w_{i}", W["encoder_ffn_down_w"][i])

        for i in range(DEC_L):
            quant(f"decoder_attn_qkv_w_{i}", W["decoder_attn_qkv_w"][i])
            quant(f"decoder_attn_o_w_{i}", W["decoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["decoder_ffn_gate_w"][i], W["decoder_ffn_up_w"][i]],
                dim=1).contiguous()
            quant(f"decoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"decoder_ffn_down_w_{i}", W["decoder_ffn_down_w"][i])

        logger.info("FP8 quantized %d Pi0 GEMM weights", len(fp8))

    def _build_pipeline_weights(self) -> dict:
        W = self._ckpt_fp16

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
            "vision_attn_qkv_w": p_list("vision_attn_qkv_w"),
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
            "encoder_multi_modal_projector_w": p(
                "encoder_multi_modal_projector_w"),
            "encoder_multi_modal_projector_b": p(
                "encoder_multi_modal_projector_b"),
            "encoder_attn_qkv_w": p_list("encoder_attn_qkv_w"),
            "encoder_attn_o_w": p_list("encoder_attn_o_w"),
            "encoder_ffn_gate_w": p_list("encoder_ffn_gate_w"),
            "encoder_ffn_up_w": p_list("encoder_ffn_up_w"),
            "encoder_ffn_down_w": p_list("encoder_ffn_down_w"),
            # Decoder BF16 fallback weights
            "decoder_attn_qkv_w": p_list("decoder_attn_qkv_w"),
            "decoder_attn_o_w": p_list("decoder_attn_o_w"),
            "decoder_ffn_gate_w": p_list("decoder_ffn_gate_w"),
            "decoder_ffn_up_w": p_list("decoder_ffn_up_w"),
            "decoder_ffn_down_w": p_list("decoder_ffn_down_w"),
            # Pi0 specifics
            "state_proj_w": p("state_proj_w"),
            "state_proj_b": p("state_proj_b"),
            "decoder_action_in_proj_w": p("decoder_action_in_proj_w"),
            "decoder_action_in_proj_b": p("decoder_action_in_proj_b"),
            "action_time_mlp_in_wa_w": p("action_time_mlp_in_wa_w"),
            "action_time_mlp_out_w": p("action_time_mlp_out_w"),
            "action_time_mlp_out_b": p("action_time_mlp_out_b"),
            "time_proj_all": self._time_proj_all.data_ptr(),
            "decoder_final_norm_w": p("decoder_final_norm_w"),
            "decoder_action_out_proj_w": p("decoder_action_out_proj_w"),
            "decoder_action_out_proj_b": p("decoder_action_out_proj_b"),
            # FP8 quantized weights
            "fp8": self._fp8_weights,
        }
        return weights

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_prompt(self, prompt_text) -> None:
        embeds, prompt_len = _embed_prompt(
            prompt_text, self.embedding_weight, max_len=self.max_prompt_len)

        if self.pipeline is None or prompt_len != self.current_prompt_len:
            logger.info("Building Pi0Pipeline for prompt_len=%d...", prompt_len)
            self.current_prompt_len = prompt_len
            self.graph_recorded = False
            self.calibrated = False

            pipeline_weights = self._build_pipeline_weights()
            # Pi0 on RTX runs the FP16 pipeline with FP8 on by default
            # to match pi0_thor's FP16 math path. The alternative
            # non-FP8 ``else`` fallback paths in the pipeline exist for
            # pi0.5 compatibility and are not fully wired for FP16
            # buffers; avoid hitting them by keeping FP8 enabled.
            use_fp8 = bool(self._use_fp8)
            use_fp8_decoder = bool(self._use_fp8_decoder)
            self.pipeline = Pi0Pipeline(
                gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                weights=pipeline_weights,
                num_views=self.num_views,
                max_prompt_len=prompt_len,
                chunk_size=self.chunk_size,
                use_fp8=use_fp8, use_fp8_decoder=use_fp8_decoder)

        embeds_np = embeds.contiguous().view(torch.uint16).cpu().numpy()
        self.pipeline.set_language_embeds(embeds_np)
        prompt_str = (prompt_text if isinstance(prompt_text, str)
                      else f"<{prompt_len} tokens>")
        logger.info("Set prompt: '%s' (%d tokens)", prompt_str, prompt_len)

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration entry point.

        Args:
            observations: single dict, list of dicts, or any Iterable[dict].
                N=1 falls back to the legacy single-frame path (bit-equal).
                N>=2 runs per-sample forwards, collects per-tensor amax,
                and reduces via ``numpy.percentile(..., percentile, axis=0)``
                before writing the final scales back to the device.
            percentile: percentile to apply in multi-sample mode.
                100.0 == traditional max; 99.9 (default) clips outliers.
                Ignored for N=1.
            max_samples: optional cap on how many samples to consume from
                the iterable.
            verbose: if True, log a per-point dispersion summary after
                reduction.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before calibrate")
        if self.calibrated:
            logger.warning(
                "calibrate() called a second time; returning without re-running.")
            return

        # Materialise + limit
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
        self._graph_torch_stream = torch.cuda.Stream()

        with torch.cuda.stream(self._graph_torch_stream):
            images = self._stack_images(sample)
            state = sample.get("state")
            noise = torch.randn(
                self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")

            stream_int = self._graph_torch_stream.cuda_stream
            self._copy_tensor_to_pipeline_buf_stream(
                images, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                noise, self.pipeline.input_noise_buf, stream_int)
            self._fill_state_buf(state)
            self._copy_tensor_to_pipeline_buf_stream(
                self._state_buf_host, self.pipeline.input_state_buf,
                stream_int)
            self.pipeline.run_pipeline(stream=stream_int)

            self._cudart.cudaStreamSynchronize(
                ctypes.c_void_p(stream_int))

            # Debug toggle: skip flipping fp8_calibrated so inference
            # re-measures per-step scales dynamically. Used only to
            # isolate calibration quality from implementation bugs.
            if os.environ.get("PI0_FORCE_DYNAMIC_FP8") != "1":
                self.pipeline.calibrate_fp8()
            self.pipeline.autotune_gemms()
            self.pipeline.record_infer_graph(external_stream_int=stream_int)

        self.calibrated = True
        self.graph_recorded = True
        self._precision_spec = self._snapshot_precision_spec(
            method="single_frame", n=1, percentile=None)
        self._warn_if_scale_ceiling_exceeded()
        logger.info("Pi0 calibration + graph capture complete")

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
                state = obs.get("state")
                noise = torch.randn(
                    self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
                self._copy_tensor_to_pipeline_buf_stream(
                    images, self.pipeline.input_images_buf, stream_int)
                self._copy_tensor_to_pipeline_buf_stream(
                    noise, self.pipeline.input_noise_buf, stream_int)
                self._fill_state_buf(state)
                self._copy_tensor_to_pipeline_buf_stream(
                    self._state_buf_host, self.pipeline.input_state_buf,
                    stream_int)
                # Reset scales so each sample records its own amax (not a
                # cumulative max across samples).
                self._zero_pipeline_scales()
                self.pipeline.run_pipeline(stream=stream_int)
                self._cudart.cudaStreamSynchronize(
                    ctypes.c_void_p(stream_int))

                if names is None:
                    names = list(self.pipeline.fp8_act_scales.keys())
                sample_vec = np.array(
                    [float(self.pipeline.fp8_act_scales[n_].download_new(
                        (1,), np.float32)[0]) for n_ in names],
                    dtype=np.float32)
                per_sample.append(sample_vec)

                if verbose and (i + 1) % max(1, n // 10) == 0:
                    logger.info("  calibration sample %d/%d", i + 1, n)

            final_amax = accumulate_amax(per_sample, percentile=percentile)
            if verbose:
                logger.info(format_summary(
                    summarize_amax_dispersion(per_sample, final_amax)))

            # Write reduced scales back into the pipeline's scale buffers.
            for idx, name in enumerate(names or []):
                self.pipeline.fp8_act_scales[name].upload(
                    np.array([final_amax[idx]], dtype=np.float32))

            if os.environ.get("PI0_FORCE_DYNAMIC_FP8") != "1":
                self.pipeline.fp8_calibrated = True
            self.pipeline.autotune_gemms()
            self.pipeline.record_infer_graph(external_stream_int=stream_int)

        self.calibrated = True
        self.graph_recorded = True
        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)
        self._warn_if_scale_ceiling_exceeded(label=f"pi0_rtx_N{n}")
        logger.info(
            "Pi0 multi-frame calibration + graph capture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def _zero_pipeline_scales(self) -> None:
        zero = np.zeros(1, dtype=np.float32)
        for buf in self.pipeline.fp8_act_scales.values():
            buf.upload(zero)

    def _warn_if_scale_ceiling_exceeded(self, label: str = "pi0_rtx") -> None:
        """Diagnostic warning if any FP8 scale exceeds the sanity ceiling."""
        from flash_rt.core.calibration import check_scale_ceiling
        scales = {
            name: float(buf.download_new((1,), np.float32)[0])
            for name, buf in self.pipeline.fp8_act_scales.items()
        }
        check_scale_ceiling(scales, label=label)

    def _snapshot_precision_spec(self, *, method: str, n: int,
                                  percentile: Optional[float]):
        """Build a :class:`ModelPrecisionSpec` from current pipeline scales."""
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
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before infer")

        t0 = time.perf_counter()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            self._fill_img_buf(observation)
            self._noise_buf.normal_()
            self._fill_state_buf(observation.get("state"))

            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self.pipeline.input_noise_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._state_buf_host, self.pipeline.input_state_buf,
                stream_int)

            out_ptr = self.pipeline.forward()

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
    # Internals
    # -----------------------------------------------------------------

    def _stack_images(self, observation: dict) -> torch.Tensor:
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        tensors = []
        for im in img_list[:self.num_views]:
            tensors.append(
                torch.from_numpy(
                    im.astype(np.float32) / 127.5 - 1.0).to("cuda", fp16))
        return torch.stack(tensors)

    def _fill_img_buf(self, observation: dict) -> None:
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(fp16))

    def _fill_state_buf(self, state) -> None:
        """Normalise and upload state vector into ``_state_buf_host``.

        Accepts either a torch tensor or numpy array. Shorter-than-32
        inputs are zero-padded to ``ACTION_DIM``. Host tensor is
        persistent so we skip reallocation per-call.
        """
        self._state_buf_host.zero_()
        if state is None:
            return
        if isinstance(state, torch.Tensor):
            s = state.to(dtype=fp16, device="cuda").reshape(-1)
        else:
            s_np = np.asarray(state, dtype=np.float32).reshape(-1)
            s = torch.from_numpy(s_np).to("cuda", fp16)
        n = min(s.numel(), ACTION_DIM)
        self._state_buf_host[0, :n].copy_(s[:n])

    def _copy_tensor_to_pipeline_buf_stream(
            self, src: torch.Tensor, dst_buf, stream_int: int) -> None:
        nbytes = src.numel() * src.element_size()
        assert nbytes == dst_buf.nbytes, \
            f"size mismatch: src {nbytes} vs dst {dst_buf.nbytes}"
        self._cudart.cudaMemcpyAsync(
            dst_buf.ptr, ctypes.c_void_p(src.data_ptr()), nbytes, 3,
            stream_int)


# ════════════════════════════════════════════════════════════════════
#   Prompt embedding helper
# ════════════════════════════════════════════════════════════════════


def _embed_prompt(prompt_text,
                  embedding_weight: torch.Tensor,
                  max_len: int = 48) -> tuple[torch.Tensor, int]:
    """Tokenise + embed via PaliGemma embedding table. Accepts either a
    plain string or a raw token-id list/ndarray (for reference-trace
    replay against PI0Pytorch)."""
    if isinstance(prompt_text, (np.ndarray, list)):
        token_ids = torch.tensor(
            np.asarray(prompt_text, dtype=np.int64), dtype=torch.long,
            device="cuda")
        prompt_len = int(token_ids.numel())
    else:
        # PaliGemma tokenizer resolution — see
        # `flash_rt.utils.paligemma_tokenizer` for the search order
        # and the download instructions emitted on failure.
        try:
            from openpi.models.tokenizer import PaligemmaTokenizer
            tokenizer = PaligemmaTokenizer(max_len=max_len)
            tokens_np, mask_np = tokenizer.tokenize(prompt_text)
            prompt_len = int(mask_np.sum())
            token_ids = torch.tensor(
                tokens_np[:prompt_len], dtype=torch.long, device="cuda")
        except (ImportError, FileNotFoundError, OSError, RuntimeError):
            from flash_rt.utils.paligemma_tokenizer import (
                load_paligemma_sentencepiece,
            )
            sp = load_paligemma_sentencepiece()
            tokens = [sp.bos_id()] + sp.Encode(prompt_text) + [108]
            token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
            prompt_len = len(token_ids)

    if embedding_weight.device.type != "cuda":
        embedding_weight = embedding_weight.to(device="cuda")

    embeds = F.embedding(token_ids, embedding_weight)
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds, prompt_len
