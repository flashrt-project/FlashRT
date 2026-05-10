"""FlashRT -- RTX Pi0.5 JAX frontend.

Loads Pi0.5 Orbax checkpoints (the JAX-native format used by openpi) and
drives the same framework-agnostic ``Pi05Pipeline`` as the torch frontend.

Stage 1 design: this is a **thin shim** over :class:`Pi05TorchFrontendRtx`.
The JAX-specific work is the weight loader (Orbax -> bf16 torch tensors
with the same dict schema as ``convert_pi05_safetensors``). Once those
weights are in the same format, every other line of the frontend -- FP8
quantize, decoder style precompute, FP8 calibration, CUDA Graph capture,
infer -- is shared with the torch path.

This still imports torch; removing that last dependency is on the
roadmap. The current goal is to prove that JAX checkpoints produce the
same outputs as torch checkpoints through the rtx pipeline.

Usage::

    from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
    pipe = Pi05JaxFrontendRtx("/path/to/pi05_libero", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs])
    out = pipe.infer({"image": img, "wrist_image": wrist})
    actions = out["actions"]   # (chunk_size, 7) numpy
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Optional, Union

import ml_dtypes
import numpy as np
import torch

from flash_rt.models.pi05.pipeline_rtx import (
    ACTION_DIM,
    DEC_L,
    ENC_L,
    NUM_STEPS_DEFAULT,
    VIS_L,
)
from flash_rt.frontends.torch.pi05_rtx import (
    CHUNK_SIZE,
    MAX_PROMPT_LEN_DEFAULT,
    Pi05TorchFrontendRtx,
    _interleave_qk,
)

logger = logging.getLogger(__name__)

bf16 = torch.bfloat16


# ════════════════════════════════════════════════════════════════════
#   Orbax → bf16 torch dict (rtx schema)
# ════════════════════════════════════════════════════════════════════
#
# This routine produces a dict with the **exact same key names + shapes**
# as ``convert_pi05_safetensors`` so the rest of the rtx torch frontend
# works unchanged. The only difference is the source of the weights.
#
# JAX Orbax stores weights with PaliGemma's flax key names. We:
#  1. Load the Orbax checkpoint via the existing ``_load_orbax`` helper
#     (returns a flat numpy dict, fp32).
#  2. Bit-truncate fp32 → bf16 → fp32. JAX weights are stored as fp32 but
#     production loads them as bf16; truncating up-front guarantees the
#     FP8 quantization scales we compute later match the torch path
#     bit-for-bit.
#  3. Reshape JAX einsum layouts ((num_heads, in_dim, head_dim) etc) into
#     the row-major (in_dim, out_dim) layout the rtx pipeline expects.
#  4. Apply the same encoder RMSNorm fold ``w *= (1 + scale)`` in fp32 to
#     avoid bf16 rounding near -1.0 (the same trap that costs ~10% LIBERO
#     accuracy if missed).
#  5. Apply Q/K head-dim interleave for the fused RoPE kernel.


def _to_bf16_cuda(arr: np.ndarray) -> torch.Tensor:
    """Numpy → contiguous BF16 cuda tensor.

    The numpy array can be fp32, fp16, or ml_dtypes.bfloat16. We go via a
    contiguous fp32 staging step (when needed) so the final ``.to(bf16)``
    cast is well-defined regardless of input dtype.
    """
    if arr.dtype == ml_dtypes.bfloat16:
        # ml_dtypes.bfloat16 → uint16 → torch.uint16 → torch.bfloat16 view
        u16 = np.ascontiguousarray(arr).view(np.uint16)
        t = torch.from_numpy(u16).view(bf16)
        return t.to("cuda", non_blocking=False).contiguous()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(
        device="cuda", dtype=bf16
    ).contiguous()


def convert_pi05_orbax(
    checkpoint_dir: Union[str, pathlib.Path]
) -> dict:
    """Convert a Pi0.5 Orbax JAX checkpoint to the rtx pipeline weight dict.

    Output schema is **identical** to ``convert_pi05_safetensors`` (torch
    bf16 cuda tensors with rtx key names) so the same downstream FP8
    quantize + style precompute + pipeline build code applies to both
    frontends. See
    ``flash_rt.frontends.torch.pi05_rtx.convert_pi05_safetensors`` for
    the full schema.
    """
    from flash_rt.core.weights.loader import _load_orbax

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    logger.info("Loading Pi0.5 Orbax checkpoint: %s", checkpoint_dir)
    raw = _load_orbax(str(checkpoint_dir))

    # Bit-truncate fp32 → bf16 → fp32. Production loads everything as
    # bf16; truncating now guarantees byte-identical FP8 scales vs the
    # torch frontend (which loads from safetensors that were already
    # saved in bf16).
    raw = {
        k: v.astype(ml_dtypes.bfloat16).astype(np.float32)
        if v.dtype == np.float32 else v
        for k, v in raw.items()
    }

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) ──
    #
    # Patch embedding: JAX stores ``(14, 14, 3, 1152)`` (HWCO) which is
    # exactly what the rtx pipeline expects after its
    # ``permute(2, 3, 1, 0)`` step in the safetensors path. So we keep
    # the JAX layout as-is.
    pe_w = raw["PaliGemma.img.embedding.kernel"]   # (14, 14, 3, 1152)
    ckpt["vision_patch_embedding_w"] = _to_bf16_cuda(pe_w)
    ckpt["vision_patch_embedding_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.embedding.bias"])
    # JAX position embedding has a leading batch axis (1, 256, 1152)
    pos_emb = raw["PaliGemma.img.pos_embedding"].squeeze(0)  # (256, 1152)
    ckpt["vision_position_embedding"] = _to_bf16_cuda(pos_emb)

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    enc_blk = "PaliGemma.img.Transformer.encoderblock"
    for i in range(VIS_L):
        # LayerNorms (stacked)
        ln1_w_list.append(raw[f"{enc_blk}.LayerNorm_0.scale"][i])
        ln1_b_list.append(raw[f"{enc_blk}.LayerNorm_0.bias"][i])
        ln2_w_list.append(raw[f"{enc_blk}.LayerNorm_1.scale"][i])
        ln2_b_list.append(raw[f"{enc_blk}.LayerNorm_1.bias"][i])

        # Attention Q/K/V einsum: JAX (1152, 16, 72) → row-major (1152, 1152)
        # then concat into (1152, 3456). Note: rtx schema expects
        # in_dim-first, i.e. (in=1152, 3*out=3456) — same as the
        # safetensors path's ``torch.cat([q,k,v], dim=0).t()``.
        q_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.kernel"][i]
        # JAX einsum kernel for ``BTD,DNH->BTNH`` is (D, N, H) = (1152, 16, 72).
        # Reshape to (D, N*H) = (1152, 1152), no transpose.
        q_2d = q_w.reshape(1152, -1)
        k_2d = k_w.reshape(1152, -1)
        v_2d = v_w.reshape(1152, -1)
        qkv_w_list.append(np.concatenate([q_2d, k_2d, v_2d], axis=1))
        # Biases: JAX (16, 72) → flat (1152,)
        q_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.bias"][i].reshape(-1)
        k_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.bias"][i].reshape(-1)
        v_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.bias"][i].reshape(-1)
        qkv_b_list.append(np.concatenate([q_b, k_b, v_b]))

        # O projection: JAX (16, 72, 1152) — einsum ``BTNH,NHD->BTD``
        # In rtx schema we want (in=1152, out=1152). The N*H dim is the
        # input axis, so reshape to (1152, 1152) (no transpose).
        o_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.kernel"][i]
        o_w_list.append(o_w.reshape(-1, 1152))
        o_b_list.append(raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.bias"][i])

        # FFN up: JAX (1152, 4304) — already in (in, out) layout.
        up_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.kernel"][i])
        up_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.bias"][i])

        # FFN down: JAX (4304, 1152) — already (in, out)
        down_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.kernel"][i])
        down_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.bias"][i])

    ckpt["vision_attn_qkv_w"] = _to_bf16_cuda(np.stack(qkv_w_list))
    ckpt["vision_attn_qkv_b"] = _to_bf16_cuda(np.stack(qkv_b_list))
    ckpt["vision_attn_o_w"] = _to_bf16_cuda(np.stack(o_w_list))
    ckpt["vision_attn_o_b"] = _to_bf16_cuda(np.stack(o_b_list))
    ckpt["vision_ffn_up_w"] = _to_bf16_cuda(np.stack(up_w_list))
    ckpt["vision_ffn_up_b"] = _to_bf16_cuda(np.stack(up_b_list))
    ckpt["vision_ffn_down_w"] = _to_bf16_cuda(np.stack(down_w_list))
    ckpt["vision_ffn_down_b"] = _to_bf16_cuda(np.stack(down_b_list))
    ckpt["vision_pre_attn_norm_w"] = _to_bf16_cuda(np.stack(ln1_w_list))
    ckpt["vision_pre_attn_norm_b"] = _to_bf16_cuda(np.stack(ln1_b_list))
    ckpt["vision_pre_ffn_norm_w"] = _to_bf16_cuda(np.stack(ln2_w_list))
    ckpt["vision_pre_ffn_norm_b"] = _to_bf16_cuda(np.stack(ln2_b_list))
    ckpt["vision_final_norm_w"] = _to_bf16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.scale"])
    ckpt["vision_final_norm_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.bias"])

    # ── Multi-modal projector ──
    # JAX kernel (1152, 2048) — already (in, out)
    ckpt["encoder_multi_modal_projector_w"] = _to_bf16_cuda(
        raw["PaliGemma.img.head.kernel"])
    ckpt["encoder_multi_modal_projector_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.head.bias"])

    # ── Encoder (18 Gemma-2B layers with RMSNorm fold) ──
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []

    for i in range(ENC_L):
        # CRITICAL: fuse in fp32 — bf16 rounds values near -1.0 to exactly
        # -1.0, collapsing (1 + scale) to 0 and zeroing entire channels.
        attn_scale = raw[
            "PaliGemma.llm.layers.pre_attention_norm.scale"][i].astype(np.float32)
        fuse_attn = 1.0 + attn_scale  # (2048,)

        # Q einsum: JAX (8, 2048, 256) for "BTD,NDH->BTNH"
        #   → (N*H, D) = (2048, 2048) row-major (out, in)
        #   → interleave heads for RoPE
        #   → fold the LN scale into the in_dim
        #   → transpose to (in, out) = (2048, 2048) for rtx schema
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 2048)
        q_2d = _interleave_qk_np(q_2d, 8)
        q_2d = q_2d * fuse_attn[None, :]

        # KV einsum: JAX (2, 1, 2048, 256) for "BTD,NDH->BTNH" with N=1
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 2048)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 2048)
        k_2d = _interleave_qk_np(k_2d, 1)
        k_2d = k_2d * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        # Concat → (2560, 2048) → transpose → (2048, 2560) (in, out)
        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T
        enc_qkv_list.append(qkv)

        # O: JAX (8, 256, 2048) for "BTNH,NHD->BTD".
        # The einsum has D as the *output* axis and N*H as the *input* axis,
        # so reshape (N*H, D) is already in the (in, out) layout the rtx
        # pipeline expects — NO transpose. (The torch frontend's `.t()` is
        # because HF stores weights in (out, in) PyTorch convention.)
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i].astype(np.float32)
        enc_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        # Gate / Up: JAX (2, 2048, 16384) — both already (in, out)
        ffn_scale = raw[
            "PaliGemma.llm.layers.pre_ffw_norm.scale"][i].astype(np.float32)
        fuse_ffn = 1.0 + ffn_scale

        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i].astype(np.float32)
        gate_w = gu_w[0] * fuse_ffn[:, None]   # (2048, 16384)
        up_w = gu_w[1] * fuse_ffn[:, None]
        enc_gate_list.append(gate_w)
        enc_up_list.append(up_w)

        # Down: JAX (16384, 2048) — already (in, out), no fold
        enc_down_list.append(
            raw["PaliGemma.llm.layers.mlp.linear"][i].astype(np.float32))

    ckpt["encoder_attn_qkv_w"] = _to_bf16_cuda(np.stack(enc_qkv_list))
    ckpt["encoder_attn_o_w"] = _to_bf16_cuda(np.stack(enc_o_list))
    ckpt["encoder_ffn_gate_w"] = _to_bf16_cuda(np.stack(enc_gate_list))
    ckpt["encoder_ffn_up_w"] = _to_bf16_cuda(np.stack(enc_up_list))
    ckpt["encoder_ffn_down_w"] = _to_bf16_cuda(np.stack(enc_down_list))

    # ── Decoder (18 Gemma-300M expert layers) ──
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    dec_attn_mod_w_list, dec_attn_mod_b_list = [], []
    dec_ffn_mod_w_list, dec_ffn_mod_b_list = [], []

    for i in range(DEC_L):
        # AdaRMSNorm modulation: JAX (1024, 3072) — already (in, out)
        dec_attn_mod_w_list.append(
            raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][i])
        dec_attn_mod_b_list.append(
            raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][i])
        dec_ffn_mod_w_list.append(
            raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][i])
        dec_ffn_mod_b_list.append(
            raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][i])

        # Q einsum: JAX (8, 1024, 256) → (2048, 1024) (out, in) → interleave
        # → transpose → (1024, 2048) (in, out)
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum_1.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 1024)
        q_2d = _interleave_qk_np(q_2d, 8)

        # KV: JAX (2, 1, 1024, 256)
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum_1.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = _interleave_qk_np(k_2d, 1)

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T  # (1024, 2560)
        dec_qkv_list.append(qkv)

        # O: JAX (8, 256, 1024) for "BTNH,NHD->BTD".
        # Same logic as encoder: reshape to (N*H, D) = (2048, 1024) is
        # already (in, out). The torch frontend's .t() gets (1024, 2048)
        # which is wrong vs the schema; the rtx torch pipeline expects
        # (out=1024, in=2048) here per its decoder_attn_o_w[18, 2048, 1024]
        # buffer shape — that *is* (in=2048, out=1024) interpreting the
        # last two dims as the row-major matrix. Match the torch path.
        #
        # Look at the rtx pipeline shape declarations:
        #   decoder_attn_o_w (18, 2048, 1024) — used as A @ W where A is
        #   (10, 2048) and output is (10, 1024). For row-major NN GEMM
        #   that needs W shape (in=2048, out=1024). So torch's .t() is
        #   correct: HF stores (out=1024, in=2048), .t() → (in=2048, out=1024).
        #
        # JAX einsum BTNH,NHD->BTD has D as output, N*H as input. So the
        # raw (N*H, D) = (2048, 1024) is already (in, out). NO transpose
        # needed.
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][i].astype(np.float32)
        dec_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        # Gate / Up: JAX (2, 1024, 4096) — already (in, out), no fold
        gu_w = raw["PaliGemma.llm.layers.mlp_1.gating_einsum"][i].astype(np.float32)
        dec_gate_list.append(gu_w[0])
        dec_up_list.append(gu_w[1])

        # Down: JAX (4096, 1024) — already (in, out)
        dec_down_list.append(
            raw["PaliGemma.llm.layers.mlp_1.linear"][i].astype(np.float32))

    ckpt["decoder_attn_qkv_w"] = _to_bf16_cuda(np.stack(dec_qkv_list))
    ckpt["decoder_attn_o_w"] = _to_bf16_cuda(np.stack(dec_o_list))
    ckpt["decoder_ffn_gate_w"] = _to_bf16_cuda(np.stack(dec_gate_list))
    ckpt["decoder_ffn_up_w"] = _to_bf16_cuda(np.stack(dec_up_list))
    ckpt["decoder_ffn_down_w"] = _to_bf16_cuda(np.stack(dec_down_list))
    ckpt["decoder_pre_attn_norm_mod_w"] = _to_bf16_cuda(np.stack(dec_attn_mod_w_list))
    ckpt["decoder_pre_attn_norm_mod_b"] = _to_bf16_cuda(np.stack(dec_attn_mod_b_list))
    ckpt["decoder_pre_ffn_norm_mod_w"] = _to_bf16_cuda(np.stack(dec_ffn_mod_w_list))
    ckpt["decoder_pre_ffn_norm_mod_b"] = _to_bf16_cuda(np.stack(dec_ffn_mod_b_list))

    ckpt["decoder_final_norm_mod_w"] = _to_bf16_cuda(
        raw["PaliGemma.llm.final_norm_1.Dense_0.kernel"])
    ckpt["decoder_final_norm_mod_b"] = _to_bf16_cuda(
        raw["PaliGemma.llm.final_norm_1.Dense_0.bias"])

    # ── Time MLP ──
    # JAX kernel (1024, 1024) is used as ``x @ kernel`` (in_dim, out_dim).
    # The rtx pipeline does NN GEMM ``x @ W`` so it also wants
    # (in, out) = (1024, 1024) — i.e. JAX layout directly, NO transpose.
    # The torch path does .t() because HF stores (out, in).
    ckpt["decoder_time_mlp_in_w"] = _to_bf16_cuda(raw["time_mlp_in.kernel"])
    ckpt["decoder_time_mlp_in_b"] = _to_bf16_cuda(raw["time_mlp_in.bias"])
    ckpt["decoder_time_mlp_out_w"] = _to_bf16_cuda(raw["time_mlp_out.kernel"])
    ckpt["decoder_time_mlp_out_b"] = _to_bf16_cuda(raw["time_mlp_out.bias"])

    # ── Sinusoidal time embeddings (10-step flow-matching schedule) ──
    # Identical to the safetensors path — schedule is determined by
    # num_steps + min/max_period only, not the checkpoint.
    num_steps = NUM_STEPS_DEFAULT
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    embedding_dim = 1024
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    time_emb_list = []
    for _ in range(num_steps):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb_list.append(
            torch.cat(
                [torch.sin(sinusoid_input), torch.cos(sinusoid_input)],
                dim=-1
            ).to(bf16)
        )
        t = t + dt
    ckpt["decoder_time_embeds"] = torch.cat(time_emb_list, dim=0).to("cuda")

    # ── Action projections ──
    # JAX action_in_proj.kernel: (32, 1024). The rtx pipeline expects
    # (in=32, out=1024) which is exactly this layout — torch's safetensors
    # path stores ``action_in_proj.weight.t()`` and the source HF weight
    # is (1024, 32), so safetensors does .t() to get (32, 1024). JAX has
    # no transpose to do.
    ckpt["decoder_action_in_proj_w"] = _to_bf16_cuda(raw["action_in_proj.kernel"])
    ckpt["decoder_action_in_proj_b"] = _to_bf16_cuda(raw["action_in_proj.bias"])
    # action_out_proj.kernel: JAX (1024, 32). Same logic — we want
    # (in=1024, out=32) which is the JAX layout directly.
    ckpt["decoder_action_out_proj_w"] = _to_bf16_cuda(raw["action_out_proj.kernel"])
    ckpt["decoder_action_out_proj_b"] = _to_bf16_cuda(raw["action_out_proj.bias"])

    # ── Embedding matrix (for prompt tokenisation) ──
    # JAX stores the input embedding under PaliGemma.llm.embedder, the lm_head
    # is tied to it. For prompt embedding we use the input embedder.
    ckpt["embedding_weight"] = _to_bf16_cuda(
        raw["PaliGemma.llm.embedder.input_embedding"])

    logger.info("Converted %d weight groups from Orbax", len(ckpt))
    return ckpt


def _interleave_qk_np(w: np.ndarray, num_heads: int) -> np.ndarray:
    """Numpy version of the QK head-dim interleave."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .transpose(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


# ════════════════════════════════════════════════════════════════════
#   Pi05JaxFrontendRtx — JAX Orbax frontend (thin shim over Pi05TorchFrontendRtx)
# ════════════════════════════════════════════════════════════════════


class Pi05JaxFrontendRtx(Pi05TorchFrontendRtx):
    """RTX consumer GPU Pi0.5 frontend backed by a JAX Orbax checkpoint.

    This class is a **thin override** of :class:`Pi05TorchFrontendRtx`: only the
    weight loader changes (Orbax instead of safetensors). Everything else
    — FP8 quantize, decoder style precompute, FP8 calibration, CUDA Graph
    capture, the ``infer`` hot path — is shared with the torch path. The
    pipeline math is therefore byte-identical between the two frontends.

    Future revision will move both frontends onto a pure-C BF16 FMHA
    backend so the JAX path can drop the torch dependency entirely.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        chunk_size: int = CHUNK_SIZE,
        max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
        use_fp8: bool = True,
    ):
        # Don't chain to Pi05TorchFrontendRtx.__init__ — it expects a safetensors
        # file. We replicate the body and swap the loader.
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.max_prompt_len = int(max_prompt_len)
        self.use_fp8 = bool(use_fp8)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        self.current_prompt_len = 0
        self.pipeline = None

        # RL CFG state — kept in sync with the torch frontend so the JAX
        # path goes through the same set_prompt / infer hot path. Both
        # default to None (= standard non-CFG inference).
        self._rl_config: Optional[dict] = None
        self._rl_current_prompt_text: Optional[str] = None

        # ── norm_stats (same locations as torch frontend) ──
        self._load_norm_stats(checkpoint_dir)

        # ── Load + convert Orbax ──
        params_dir = checkpoint_dir
        if (checkpoint_dir / "params").is_dir():
            # The "real" checkpoint root may be the parent — but the loader
            # autodetects, so just pass through.
            pass
        self._checkpoint_path = str(checkpoint_dir)
        raw_ckpt = convert_pi05_orbax(checkpoint_dir)

        self._ckpt_bf16 = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in raw_ckpt.items()
        }
        self.embedding_weight = self._ckpt_bf16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps
        # (matches the torch frontend's pre-scaling step that bakes the
        # flow-matching residual coefficient into the weights).
        num_steps = NUM_STEPS_DEFAULT
        self._ckpt_bf16["decoder_action_out_proj_w"] = (
            self._ckpt_bf16["decoder_action_out_proj_w"] * (-1.0 / num_steps)
        )
        self._ckpt_bf16["decoder_action_out_proj_b"] = (
            self._ckpt_bf16["decoder_action_out_proj_b"] * (-1.0 / num_steps)
        )

        # ── FP8 quantize large GEMM weights (shared method) ──
        self._fp8_weights: dict = {}
        self._fp8_store: list = []
        if self.use_fp8:
            self._quantize_all_fp8()

        # ── Pre-compute decoder styles (shared helper) ──
        from flash_rt.frontends.torch.pi05_rtx import _precompute_decoder_styles
        self._precomputed_styles = _precompute_decoder_styles(
            self._ckpt_bf16, self.chunk_size, num_steps=num_steps
        )

        # ── Attention backend, fvk, GemmRunner, reusable buffers ──
        from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
        from flash_rt import flash_rt_kernels as fvk

        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.chunk_size,
            num_encoder_layers=ENC_L,
        )
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        IMG_HW = 224  # local — matches Pi05TorchFrontendRtx's constant
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=bf16, device="cuda"
        )
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda"
        )
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda"
        )
        from flash_rt.core.cuda_buffer import _cudart
        self._cudart = _cudart

        logger.info(
            "Pi05JaxFrontendRtx initialised (num_views=%d, chunk=%d)",
            self.num_views, self.chunk_size,
        )
