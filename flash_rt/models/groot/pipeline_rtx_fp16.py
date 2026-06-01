"""FlashRT — RTX (consumer discrete GPU) GROOT N1.6 full-FP16 inference pipeline.

Full-FP16 baseline variant of ``pipeline_rtx.py``: every GEMM runs in FP16
with no FP8 quantization and no activation calibration. It exists as an
A/B reference against the default FP8 path — equal architecture, equal
buffers; the only difference is that each FP8 ``quantize + fp8_descale``
pair is replaced by a single ``fp16_nn`` GEMM on FP16 weights.

Framework-agnostic pipeline for GROOT N1.6 on consumer RTX GPUs (Blackwell
SM120 / Ada SM89). Mirrors the rtx Pi0.5 design (pipeline_pi05.py): the
pipeline owns kernel composition + scratch buffers, frontends own weight
loading + framework choice.

Architecture::

    frontend (torch safetensors, future JAX, ...)
        │
        ├── loads FP16 weights into framework-native tensors,
        │   exposes raw device pointer ints
        │
        ├── instantiates RtxFlashAttnBackendGroot (owns SigLIP / Qwen3 /
        │   DiT self / DiT cross Q/K/V slots as fp16 torch tensors;
        │   framework-neutral, the name reflects hardware family not
        │   the frontend framework)
        │
        └── constructs GrootSigLIP2(...), GrootQwen3(...), GrootDiT(...)
            and drives them in sequence (vision → backbone → action head)

GROOT N1.6 architecture::

    [observation images, num_views=2 typically]
            │
            v  SigLIP2 (27 layers, 256 patches/view, FP16 GEMMs + FlashAttn)
            v  post-LayerNorm
            v  pixel_unshuffle 2x2  → 64 tokens/view
            v  mlp1: LN → Linear(4608→2048) → GELU → Linear(2048→2048)
            v
    [vision_features:  S_img × 2048   fp16]
            │
            +-- text embeddings (from prompt tokens)
            v
    [input_embeds:  Se × 2048  fp16]
            │
            v  Qwen3-1.7B encoder (16 layers, GQA 16Q/8KV, head_dim=128,
            v                      q_norm/k_norm, RoPE θ=1e6, FP16 GEMMs)
            v  vlln (post-Qwen3 LayerNorm)
            v
    [backbone_features:  Se × 2048  fp16]
            │
            +-- mask split: image tokens vs text tokens
            v
    [kv_text:  Se × 2048,  kv_img:  Se × 2048]   (zeroed in the other half)
            │
            v  precompute cross-attn K/V projections (16 cross blocks ×
            v                                          K + V GEMMs, once)
            v
            v  4-step flow-matching DiT loop:
            v    state_encode → state_feat (1, 1536)
            v    action_encode → action_feat (T, 1536)
            v    concat → hidden (Sa = 1+T, 1536)
            v    32 alternating self/cross AdaLayerNorm DiT blocks
            v    final AdaLayerNorm + proj_out_2 (1024)
            v    action_decode (1024 → action_dim)
            v    actions += dt * velocity
            v
    [actions:  T × action_dim  fp32]

Per-attention head_dim:
  SigLIP2: 72        (flash_attn rounds to 96 internally)
  Qwen3:   128        (direct)
  DiT:     48         (rounds to 64)

The single torch dependency in v1 is inside the attention backend (see
``attn_backend_groot.py``). A future revision will replace it with a
pure-C BF16/FP16 FMHA library so both frontends become fully
framework-agnostic.
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.cuda_graph import CUDAGraph

logger = logging.getLogger(__name__)


# ── GROOT N1.6 fixed model dimensions (verified vs HF checkpoint) ──

VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_PATCH_FLAT = 14 * 14 * 3   # 588
VIS_SPV_RAW = 256              # patches per view before pixel unshuffle
VIS_SPV = 64                   # tokens per view after 2x2 pixel unshuffle
VIS_MLP1_IN = 4608             # 1152 * 4 (post-pixel-unshuffle channel dim)

QWEN3_L = 16                   # select_layer=16, checkpoint truncated
QWEN3_D = 2048
QWEN3_H = 6144
QWEN3_NHQ = 16
QWEN3_NHKV = 8
QWEN3_HD = 128
QWEN3_QKV_DIM = QWEN3_NHQ * QWEN3_HD + 2 * QWEN3_NHKV * QWEN3_HD   # 4096

DIT_L = 32
DIT_D = 1536
DIT_H = 6144                   # GELU FFN (NOT GeGLU; would be 12288)
DIT_NH = 32
DIT_HD = 48                    # 1536 / 32

ACTION_DIM = 128               # padded max
STATE_DIM = 128                # padded max
ACTION_HORIZON_MAX = 50
NUM_FLOW_STEPS = 4
DIT_OUTPUT_DIM = 1024

# Sizing dtypes — CudaBuffer cares about bytes, not numerics
FP16_NP = np.float16    # FP16 IS the actual numeric dtype for GROOT
FP32 = np.float32


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


# ══════════════════════════════════════════════════════════════════
#  Phase A — SigLIP2 vision encoder
# ══════════════════════════════════════════════════════════════════

class GrootSigLIP2FP16:
    """SigLIP2 27-layer vision encoder + post-LN + pixel unshuffle + mlp1 (FP16).

    Input: ``input_images`` CudaBuffer (num_views, 224, 224, 3) fp16
           — frontend writes per-call.
    Output: ``vision_features`` CudaBuffer (num_views * 64, 2048) fp16
            — consumed by GrootQwen3 as image token embeddings.

    Internally identical to the Pi0.5 SigLIP path (same 27-layer template,
    same GEMM pattern) — the differences from rtx Pi0.5 are:
      * fp16 throughout (Pi0.5 uses bf16)
      * post-LayerNorm + 2x2 pixel_unshuffle + mlp1 follow the encoder
        (Pi0.5 just does final_norm + multi_modal_projector)

    Weights dict (all int device pointers unless noted):
      vision_patch_embedding_w  — Linear (1152, 588)  ★ NOT Conv2d
      vision_patch_embedding_b  — (1152,)
      vision_position_embedding — (256, 1152)
      vision_pre_attn_norm_w[L], vision_pre_attn_norm_b[L]
      vision_pre_ffn_norm_w[L],  vision_pre_ffn_norm_b[L]
      vision_attn_qkv_b[L], vision_attn_o_b[L]
      vision_ffn_up_b[L],   vision_ffn_down_b[L]
      vision_post_norm_w, vision_post_norm_b      (post-LN before unshuffle)
      mlp1_ln_w, mlp1_ln_b                        (LayerNorm on (4608,))
      mlp1_fc1_w (4608, 2048), mlp1_fc1_b (2048,)
      mlp1_fc2_w (2048, 2048), mlp1_fc2_b (2048,)
      FP16 GEMM weights (per-layer int pointer lists, row-major):
        vision_attn_qkv_w[L], vision_attn_o_w[L]
        vision_ffn_up_w[L],   vision_ffn_down_w[L]
    """

    def __init__(self, gemm, fvk, attn_backend, weights, num_views: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.num_views = int(num_views)
        self.S_raw = self.num_views * VIS_SPV_RAW       # 512 for 2 views
        self.S_img = self.num_views * VIS_SPV           # 128 for 2 views

        self._cudart = ctypes.CDLL("libcudart.so")
        self._attn_ptrs = attn_backend.get_ptrs()

        self.bufs = self._allocate_buffers()

    def _allocate_buffers(self) -> dict:
        nv = self.num_views
        S = self.S_raw
        B = {}
        # Frontend writes the input image here
        B["input_images"] = CudaBuffer.device_empty(nv * 224 * 224 * 3, FP16_NP)
        # SigLIP residual stream
        B["sig_x"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        B["sig_x_norm"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        # Patch im2col output: (S, 588) fp16
        B["sig_patches"] = CudaBuffer.device_empty(S * VIS_PATCH_FLAT, FP16_NP)
        # QKV merged
        B["sig_qkv"] = CudaBuffer.device_empty(S * 3 * VIS_D, FP16_NP)
        # FFN hidden
        B["sig_hidden"] = CudaBuffer.device_empty(S * VIS_H, FP16_NP)
        # Post-LN output (before pixel unshuffle)
        B["sig_postln"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        # mlp1 staging buffers
        B["mlp1_in"] = CudaBuffer.device_empty(self.S_img * VIS_MLP1_IN, FP16_NP)
        B["mlp1_ln"] = CudaBuffer.device_empty(self.S_img * VIS_MLP1_IN, FP16_NP)
        B["mlp1_fc1"] = CudaBuffer.device_empty(self.S_img * QWEN3_D, FP16_NP)
        # Final output: (S_img, 2048) fp16
        B["vision_features"] = CudaBuffer.device_empty(
            self.S_img * QWEN3_D, FP16_NP)
        return B

    def forward(self, stream: int = 0) -> None:
        """Run SigLIP2 → post-LN → pixel_unshuffle → mlp1.

        Frontend must have written ``input_images`` first. After this
        returns, ``vision_features`` holds the (S_img, 2048) image tokens
        ready to be merged with text embeddings for Qwen3.
        """
        self._patch_embed(stream)
        self._siglip_layers(stream)
        self._post_ln_unshuffle_mlp1(stream)

    # ── Patch embed: im2col → Linear → bias + pos ──
    def _patch_embed(self, stream: int) -> None:
        fvk = self.fvk
        B = self.bufs
        W = self.weights
        S = self.S_raw

        # im2col: (nv, 224, 224, 3) → (S, 588) fp16
        fvk.patch_im2col(
            _p(B["input_images"]),
            _p(B["sig_patches"]),
            self.num_views, stream)

        # GEMM: (S, 588) @ (588, 1152) → (S, 1152)
        # Note: GROOT's patch_embedding is Linear (NOT Conv2d like SigLIP1).
        # The frontend stores the weight as (588, 1152) row-major to match
        # rtx pipeline_pi05's vision_patch_embedding_w schema.
        self.gemm.fp16_nn(
            _p(B["sig_patches"]),
            W["vision_patch_embedding_w"],
            _p(B["sig_x"]),
            S, VIS_D, VIS_PATCH_FLAT, stream)

        # Bias + position embedding (FP16-only kernel — works here since we're FP16)
        fvk.patch_embed_bias_pos(
            _p(B["sig_x"]),
            W["vision_patch_embedding_b"],
            W["vision_position_embedding"],
            S, VIS_D, VIS_SPV_RAW, stream)

    # ── 27 SigLIP layers (FP16 GEMMs) ──
    def _siglip_layers(self, stream: int) -> None:
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        S = self.S_raw
        attn_ptrs = self._attn_ptrs

        for i in range(VIS_L):
            self._siglip_one_layer(i, S, attn_ptrs, stream)

    def _siglip_one_layer(self, i: int, S: int, attn_ptrs: dict,
                          stream: int) -> None:
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights

        # Attention LayerNorm → x_norm
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_pre_attn_norm_w"][i], W["vision_pre_attn_norm_b"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, 1e-6, stream)

        # QKV GEMM (FP16): (S, D) @ (D, 3D) → (S, 3D)
        gemm.fp16_nn(
            _p(B["sig_x_norm"]), W["vision_attn_qkv_w"][i],
            _p(B["sig_qkv"]),
            S, 3 * VIS_D, VIS_D, stream)

        # Add QKV bias
        fvk.add_bias_fp16(
            _p(B["sig_qkv"]), W["vision_attn_qkv_b"][i],
            S, 3 * VIS_D, stream)

        # Split QKV → attn backend's vis_Q/K/V (per-view 4D layout)
        # Q at offset 0, K at offset D, V at offset 2D, all in flat (S, 3D)
        qkv_ptr = _p(B["sig_qkv"])
        # Each split copies (S, 1152) into (num_views, 256, 16, 72) — same memory.
        # The fvk.gpu_strided_copy_fp16 reads contiguous (S, 3D) src and writes
        # to (S, D) dst with src_stride=3D and col_offset for the slice.
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_Q"],
            S, VIS_D, 3 * VIS_D, 0, stream)
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_K"],
            S, VIS_D, 3 * VIS_D, VIS_D, stream)
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_V"],
            S, VIS_D, 3 * VIS_D, 2 * VIS_D, stream)

        # FlashAttention (per-view batched, B=num_views, S=256, NH=16, HD=72).
        # Dispatch attention through the AttentionBackend protocol
        # (``run("siglip", layer, q_seq)``). Same flash_attn_func call
        # under the hood as the legacy ``vision_attn()`` method.
        attn_out_ptr = self.attn.run(
            "siglip", i, q_seq=VIS_SPV_RAW, stream=stream)

        # Attn output projection (FP16): (S, D) → (S, D) into x_norm
        gemm.fp16_nn(
            attn_out_ptr, W["vision_attn_o_w"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, VIS_D, stream)

        # x += x_norm + attn_o_bias (bias_residual_fp16: x = residual + x + bias)
        fvk.bias_residual_fp16(
            _p(B["sig_x"]), _p(B["sig_x_norm"]),
            W["vision_attn_o_b"][i],
            S, VIS_D, stream)

        # FFN LayerNorm
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_pre_ffn_norm_w"][i], W["vision_pre_ffn_norm_b"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, 1e-6, stream)

        # FFN up + GELU (FP16): (S, D) → (S, H)
        gemm.fp16_nn(
            _p(B["sig_x_norm"]), W["vision_ffn_up_w"][i],
            _p(B["sig_hidden"]),
            S, VIS_H, VIS_D, stream)
        fvk.add_bias_fp16(
            _p(B["sig_hidden"]), W["vision_ffn_up_b"][i],
            S, VIS_H, stream)
        fvk.gelu_inplace_fp16(_p(B["sig_hidden"]), S * VIS_H, stream)

        # FFN down (FP16): (S, H) → (S, D) into x_norm
        gemm.fp16_nn(
            _p(B["sig_hidden"]), W["vision_ffn_down_w"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, VIS_H, stream)
        # x += x_norm + down_bias
        fvk.bias_residual_fp16(
            _p(B["sig_x"]), _p(B["sig_x_norm"]),
            W["vision_ffn_down_b"][i],
            S, VIS_D, stream)

    # ── Post-LayerNorm + pixel_unshuffle (frontend-side) + mlp1 ──
    def _post_ln_unshuffle_mlp1(self, stream: int) -> None:
        """Post-SigLIP LayerNorm + 2x2 pixel_unshuffle + mlp1.

        The pixel_unshuffle has to happen as a strided memory rearrangement.
        We do it inside the captured graph by calling fvk.gpu_strided_copy_fp16
        with the right strides — but pixel_unshuffle is a 4D rearrangement
        that doesn't map to a single strided copy. Instead, the frontend
        precomputes the index map and we use a custom kernel — OR we keep
        the post-LN outside the graph and do the unshuffle in torch.

        Thor's design captures SigLIP graph through the post-LN ONLY, then
        does pixel_unshuffle + mlp1 OUTSIDE the graph (the input embedding
        building stage). We do the same here — this method actually only
        runs the post-LN inside the graph; pixel_unshuffle + mlp1 are
        executed by ``run_post_ln_unshuffle_mlp1_torch`` below from the
        frontend during the input-embedding build phase.
        """
        fvk = self.fvk
        B = self.bufs
        W = self.weights
        S = self.S_raw
        # Post-LayerNorm only (the unshuffle + mlp1 are torch-side)
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_post_norm_w"], W["vision_post_norm_b"],
            _p(B["sig_postln"]),
            S, VIS_D, 1e-6, stream)


# ══════════════════════════════════════════════════════════════════
#  Phase B — Qwen3-1.7B encoder backbone
# ══════════════════════════════════════════════════════════════════

class GrootQwen3FP16:
    """Qwen3-1.7B 16-layer GQA encoder (FP16).

    Architecture (per layer):
      - input_layernorm (RMSNorm) → x_norm
      - QKV GEMM (FP16): (Se, 2048) @ (2048, 4096) → (Se, 4096) flat
      - split Q (Se, 16, 128), K (Se, 8, 128), V (Se, 8, 128)
      - q_norm / k_norm: per-head RMSNorm (16-h Q, 8-h K)
      - RoPE (theta=1e6) on Q and K
      - GQA flash_attn → (Se, 16, 128)
      - O projection (fp16): (Se, 2048) → (Se, 2048) + residual
      - post_attention_layernorm (RMSNorm) → x_norm
      - FFN gate+up (FP16): (Se, 2048) @ (2048, 12288) → split gate/up
      - SiLU(gate) * up → down (FP16) → (Se, 2048) + residual
      - final RMSNorm

    Then vlln (LayerNorm) is applied externally to produce backbone_features.

    Weights dict (per-layer lists indexed by layer):
      qwen3_ln_attn_w[L]      — input_layernorm scale (2048,)
      qwen3_ln_ffn_w[L]       — post_attention_layernorm scale (2048,)
      qwen3_q_norm_w[L]       — (128,)  per-head Q RMSNorm
      qwen3_k_norm_w[L]       — (128,)
      qwen3_o_w_fp16[L]       — (D, D) fp16
      qwen3_final_norm_w      — (2048,)
      vlln_w / vlln_b         — (2048,) — applied externally
      FP16 GEMM weights (per-layer int pointer lists, row-major):
        qwen3_qkv_w[L]        — (D, QKV_DIM)
        qwen3_gate_up_w[L]    — (D, 2H)
        qwen3_down_w[L]       — (H, D)
    """

    def __init__(self, gemm, fvk, attn_backend, weights, encoder_seq_max: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.Se_max = int(encoder_seq_max)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._cudart = ctypes.CDLL("libcudart.so")

        # Precomputed RoPE (cos/sin tables for the rotate_half kernel)
        self._rope_cos, self._rope_sin = self._build_rope()

        # Set Se = full max sequence by default; frontend overrides per-prompt
        self.Se = self.Se_max

        self.bufs = self._allocate_buffers()

    def set_seq_len(self, Se: int) -> None:
        """Reset Se for a new prompt. Must be called before forward()."""
        assert Se <= self.Se_max
        self.Se = int(Se)

    def _build_rope(self) -> tuple:
        """Build cos/sin tables for Qwen3 rotate_half RoPE (theta=1e6).

        rope_rotate_half_fp16 expects cos/sin tables of shape (max_seq, HD)
        where each row is [cos_full] or [sin_full] — i.e., the half-frequencies
        repeated to fill HD. This matches HF Qwen3's rotate_half convention.
        """
        theta = 1e6
        max_seq = max(self.Se_max, 1024)
        HD = QWEN3_HD
        freqs = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        positions = np.arange(max_seq, dtype=np.float64)
        angles = positions[:, None] * freqs[None, :]   # (max_seq, HD/2)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(FP16_NP)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(FP16_NP)
        return CudaBuffer.from_numpy(cos), CudaBuffer.from_numpy(sin)

    def _allocate_buffers(self) -> dict:
        Se = self.Se_max
        D = QWEN3_D
        H = QWEN3_H
        QKV = QWEN3_QKV_DIM
        B = {}
        B["x"] = CudaBuffer.device_empty(Se * D, FP16_NP)        # input_embeds
        B["x_norm"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["qkv"] = CudaBuffer.device_empty(Se * QKV, FP16_NP)
        # b_attn = (Se, NHQ * HD) flat — receives flash_attn output reshaped
        B["attn_out_flat"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["o_out"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["gate_up"] = CudaBuffer.device_empty(Se * 2 * H, FP16_NP)
        B["gate"] = CudaBuffer.device_empty(Se * H, FP16_NP)
        B["up"] = CudaBuffer.device_empty(Se * H, FP16_NP)
        B["down"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        # vlln output (post-Qwen3 LayerNorm) — final backbone features
        B["backbone_features"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        return B

    def forward(self, stream: int = 0) -> None:
        """Run Qwen3 16 layers + vlln on ``bufs['x']``.

        Output: ``bufs['backbone_features']`` (Se, 2048) fp16.
        Frontend must have written ``x`` (input embeddings) before calling.
        """
        Se = self.Se
        D = QWEN3_D
        H = QWEN3_H
        QKV = QWEN3_QKV_DIM
        NHQ, NHKV, HD = QWEN3_NHQ, QWEN3_NHKV, QWEN3_HD

        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        attn_ptrs = self._attn_ptrs

        cos_ptr = self._rope_cos.ptr.value
        sin_ptr = self._rope_sin.ptr.value

        for i in range(QWEN3_L):
            # ── input_layernorm (RMSNorm) → x_norm ──
            fvk.rms_norm_fp16(
                _p(B["x"]), W["qwen3_ln_attn_w"][i],
                _p(B["x_norm"]),
                Se, D, 1e-6, stream)

            # ── QKV GEMM (FP16) ──
            gemm.fp16_nn(
                _p(B["x_norm"]), W["qwen3_qkv_w"][i],
                _p(B["qkv"]),
                Se, QKV, D, stream)

            # ── Split QKV into attn backend's qwen3_Q/K/V (4D fp16 slots) ──
            # qkv layout: [Q (NHQ*HD) | K (NHKV*HD) | V (NHKV*HD)] per row
            qkv_ptr = _p(B["qkv"])
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_Q"],
                Se, NHQ * HD, QKV, 0, stream)
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_K"],
                Se, NHKV * HD, QKV, NHQ * HD, stream)
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_V"],
                Se, NHKV * HD, QKV, NHQ * HD + NHKV * HD, stream)

            # ── q_norm / k_norm (per-head RMSNorm) ──
            # rms_norm_fp16 with seq_len = Se*NHQ and dim = HD treats each
            # (token, head) row independently. Same shared q_norm_w (HD,)
            # is broadcast across all heads.
            fvk.rms_norm_fp16(
                attn_ptrs["qwen3_Q"], W["qwen3_q_norm_w"][i],
                attn_ptrs["qwen3_Q"],
                Se * NHQ, HD, 1e-6, stream)
            fvk.rms_norm_fp16(
                attn_ptrs["qwen3_K"], W["qwen3_k_norm_w"][i],
                attn_ptrs["qwen3_K"],
                Se * NHKV, HD, 1e-6, stream)

            # ── RoPE rotate_half on Q and K ──
            fvk.rope_rotate_half_fp16(
                attn_ptrs["qwen3_Q"], cos_ptr, sin_ptr,
                Se, NHQ, HD, stream)
            fvk.rope_rotate_half_fp16(
                attn_ptrs["qwen3_K"], cos_ptr, sin_ptr,
                Se, NHKV, HD, stream)

            # ── GQA self-attention via flash_attn ──
            # Dispatched through AttentionBackend protocol's
            # ``run("qwen3", i, q_seq=Se)``.
            attn_out_ptr = self.attn.run(
                "qwen3", i, q_seq=Se, stream=stream)

            # ── O projection (fp16) ──
            # attn output is (Se, NHQ, HD) row-major; flat = (Se, D)
            gemm.fp16_nn(
                attn_out_ptr, W["qwen3_o_w_fp16"][i],
                _p(B["o_out"]),
                Se, D, D, stream)
            # x += o_out (residual)
            fvk.residual_add_fp16(
                _p(B["x"]), _p(B["o_out"]),
                Se * D, stream)

            # ── post_attention_layernorm (RMSNorm) ──
            fvk.rms_norm_fp16(
                _p(B["x"]), W["qwen3_ln_ffn_w"][i],
                _p(B["x_norm"]),
                Se, D, 1e-6, stream)

            # ── FFN gate+up (FP16) ──
            gemm.fp16_nn(
                _p(B["x_norm"]), W["qwen3_gate_up_w"][i],
                _p(B["gate_up"]),
                Se, 2 * H, D, stream)

            # Split gate / up via strided copy
            gu_ptr = _p(B["gate_up"])
            fvk.gpu_strided_copy_fp16(
                gu_ptr, _p(B["gate"]),
                Se, H, 2 * H, 0, stream)
            fvk.gpu_strided_copy_fp16(
                gu_ptr, _p(B["up"]),
                Se, H, 2 * H, H, stream)

            # SiLU(gate) * up → gate (compose: silu in-place then elementwise mul)
            fvk.silu_inplace_fp16(_p(B["gate"]), Se * H, stream)
            fvk.mul_fp16(
                _p(B["gate"]), _p(B["up"]), _p(B["gate"]),
                Se * H, stream)

            # FFN down (FP16): (Se, H) → (Se, D) into b_down
            gemm.fp16_nn(
                _p(B["gate"]), W["qwen3_down_w"][i],
                _p(B["down"]),
                Se, D, H, stream)
            fvk.residual_add_fp16(
                _p(B["x"]), _p(B["down"]),
                Se * D, stream)

        # ── Final RMSNorm ──
        fvk.rms_norm_fp16(
            _p(B["x"]), W["qwen3_final_norm_w"],
            _p(B["x_norm"]),
            Se, D, 1e-6, stream)

        # ── vlln (LayerNorm with affine, applied to the Qwen3 output) ──
        fvk.layer_norm_fp16(
            _p(B["x_norm"]),
            W["vlln_w"], W["vlln_b"],
            _p(B["backbone_features"]),
            Se, D, 1e-5, stream)


# ══════════════════════════════════════════════════════════════════
#  Phase C — AlternateVLDiT action head
# ══════════════════════════════════════════════════════════════════

class GrootDiTFP16:
    """32-layer AlternateVLDiT action head + 4-step flow matching (FP16).

    Per layer:
      - AdaLayerNorm: layer_norm_no_affine then ada_layer_norm
        x_norm = LN(x) * (1 + scale[step,layer]) + shift[step,layer]
      - Self-attn (odd layers) OR cross-attn (even layers)
      - O projection (fp16) + residual
      - LayerNorm (no params) → FFN
      - GELU FFN: ff_up (FP16 + bias + GELU) → ff_down (FP16 + bias) + residual

    Final:
      - Output AdaLN: shift first, scale second
      - proj_out_2 (fp16, 1024)
      - action_decoder (per-embodiment): Linear → ReLU → Linear → velocity
      - actions += dt * velocity (gpu_euler_step)

    Weights dict (per-layer lists for the 32 DiT blocks):
      dit_q_w_fp16[L]   — (D, D) for cross-attn Q
      dit_q_b[L]
      dit_o_w_fp16[L]   — (D, D)
      dit_o_b[L]
      dit_ff_up_b[L]    — (H,) for ff_up bias
      dit_ff_down_b[L]  — (D,) for ff_down bias
      FP16 GEMM weights (per-layer int pointer lists, row-major):
        dit_qkv_w[L]      — (D, 3D) — only for self-attn (odd) layers
        dit_ff_up_w[L]    — (D, H)
        dit_ff_down_w[L]  — (H, D)
      dit_qkv_b_self[L]  — (3D,) — only valid at odd indices, dummy elsewhere

    Per-step / per-layer pre-computed conditioning:
      ada_scales — (num_steps, L, D) fp16
      ada_shifts — (num_steps, L, D) fp16
      out_scales — (num_steps, D)    fp16
      out_shifts — (num_steps, D)    fp16
      action_time_embeds — (num_steps, action_horizon, D) fp16  (sinusoidal)

    Embodiment MLPs (per the chosen embodiment id, set by frontend):
      state_enc_w1, state_enc_b1, state_enc_w2, state_enc_b2
      action_enc_w1, action_enc_b1, action_enc_w2, action_enc_b2
      action_enc_w3, action_enc_b3
      action_dec_w1, action_dec_b1, action_dec_w2, action_dec_b2
      pos_emb       — (action_horizon_max, D)
      proj_out_2_w  — (D, output_dim)
      proj_out_2_b  — (output_dim,)
    """

    def __init__(self, gemm, fvk, attn_backend, weights,
                 action_horizon: int, encoder_seq: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.T = int(action_horizon)
        self.Sa = 1 + self.T              # 1 state + T actions
        self.Se = int(encoder_seq)        # backbone seq len (for cross-attn KV)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._cudart = ctypes.CDLL("libcudart.so")
        # Argtypes for cudaMemcpy2DAsync — used to build the interleaved
        # (T, 2D) action-encoder concat in _run_step. The naive two-block
        # byte-copy pattern (a_emb flat | time_emb flat) does NOT produce
        # a row-major (T, 2D) layout where row i = [a_emb[i], time_emb[i]];
        # it produces [a_emb[2i], a_emb[2i+1]] / [time_emb[...]] which is a
        # different input to action_encoder.W2 entirely. cudaMemcpy2DAsync
        # with a destination pitch of 2*D*2 bytes lays out each source row
        # into the correct 2D-wide destination slot.
        self._cudart.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._cudart.cudaMemcpy2DAsync.restype = ctypes.c_int

        self.bufs = self._allocate_buffers()

    def _allocate_buffers(self) -> dict:
        D = DIT_D
        H = DIT_H
        T = self.T
        Sa = self.Sa
        Se = self.Se
        NH = DIT_NH
        HD = DIT_HD
        B = {}

        # Inputs (frontend writes per-call)
        # b_actions: fp32 noise tensor (T, action_dim) — euler-stepped in place
        B["actions"] = CudaBuffer.device_zeros(T * ACTION_DIM, FP32)
        # state_feat (1, D) — frontend pre-computes via state_encoder MLP
        B["state_feat"] = CudaBuffer.device_empty(D, FP16_NP)
        # KV inputs from Qwen3 backbone (split into text/img halves by mask)
        B["kv_text"] = CudaBuffer.device_empty(Se * QWEN3_D, FP16_NP)
        B["kv_img"] = CudaBuffer.device_empty(Se * QWEN3_D, FP16_NP)

        # NB: cross-attn K/V slots are NOT allocated here — they live on
        # the attention backend (so flash_attn can read them as torch
        # tensor views with stable allocator slots). The pipeline writes
        # the projected K/V into the backend's slots via the pointers
        # exposed by attn_backend.get_ptrs()["dit_cross_K"][idx] etc.

        # Action encoder scratch
        B["actions_fp16"] = CudaBuffer.device_empty(T * ACTION_DIM, FP16_NP)
        B["a_emb"] = CudaBuffer.device_empty(T * D, FP16_NP)
        B["concat"] = CudaBuffer.device_empty(T * 2 * D, FP16_NP)
        B["enc_h"] = CudaBuffer.device_empty(T * D, FP16_NP)

        # DiT main residual stream
        B["hidden"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["h_norm"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["qkv"] = CudaBuffer.device_empty(Sa * 3 * D, FP16_NP)
        B["o_out"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["ff_h"] = CudaBuffer.device_empty(Sa * H, FP16_NP)
        B["ff_out"] = CudaBuffer.device_empty(Sa * D, FP16_NP)

        # Final output
        B["model_out"] = CudaBuffer.device_empty(Sa * DIT_OUTPUT_DIM, FP16_NP)
        B["dec_h"] = CudaBuffer.device_empty(Sa * 1024, FP16_NP)
        B["velocity"] = CudaBuffer.device_empty(Sa * ACTION_DIM, FP16_NP)
        return B

    # ── Cross-attention KV precompute (called before each diffusion loop) ──
    def precompute_cross_kv(self, stream: int = 0) -> None:
        """Project backbone KV through 16 cross-attn K + V GEMMs (constant
        across the 4 diffusion steps).

        Writes the projected (Se, D) fp16 features into the **backend-owned**
        ``dit_cross_K[idx]`` / ``dit_cross_V[idx]`` slots so flash_attn can
        read them as torch tensors during graph replay.
        """
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        Se = self.Se
        D = DIT_D
        attn_ptrs = self._attn_ptrs

        for block_idx in range(DIT_L // 2):
            l = block_idx * 2     # cross blocks at l = 0, 2, 4, ..., 30
            kv_src_buf = B["kv_text"] if (l % 4 == 0) else B["kv_img"]
            kv_src_ptr = _p(kv_src_buf)
            # K projection: (Se, D_kv=2048) @ (2048, D=1536) → (Se, D)
            cross_k_ptr = attn_ptrs["dit_cross_K"][block_idx]
            gemm.fp16_nn(
                kv_src_ptr, W["dit_k_w_fp16"][l],
                cross_k_ptr,
                Se, D, QWEN3_D, stream)
            fvk.add_bias_fp16(
                cross_k_ptr, W["dit_k_b"][l],
                Se, D, stream)
            # V projection
            cross_v_ptr = attn_ptrs["dit_cross_V"][block_idx]
            gemm.fp16_nn(
                kv_src_ptr, W["dit_v_w_fp16"][l],
                cross_v_ptr,
                Se, D, QWEN3_D, stream)
            fvk.add_bias_fp16(
                cross_v_ptr, W["dit_v_b"][l],
                Se, D, stream)

    # ── 4-step flow matching loop ──
    def run_steps(self, stream: int = 0) -> None:
        """Run all 4 flow-matching steps in sequence.

        Inputs already in buffers:
          - actions  (T, action_dim) fp32 — initial noise (frontend writes)
          - state_feat (1, D) fp16 — frontend pre-computes
          - cross_K/V_{idx} — frontend called precompute_cross_kv()

        Output:
          - actions (T, action_dim) fp32 — final velocity-stepped actions
        """
        for step in range(NUM_FLOW_STEPS):
            self._run_step(step, stream)

    def _run_step(self, step: int, stream: int) -> None:
        D = DIT_D
        H = DIT_H
        T = self.T
        Sa = self.Sa
        NH, HD = DIT_NH, DIT_HD
        Se = self.Se
        dt = 1.0 / NUM_FLOW_STEPS

        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        attn_ptrs = self._attn_ptrs

        # Per-step pre-computed bufs (frontend uploaded these as packed
        # arrays with sizes (steps, L, D) etc — slice via byte offsets)
        # ada_scales[step, l] is at byte offset (step * L + l) * D * 2
        ada_scales_base = W["ada_scales"]    # int ptr to (steps, L, D) fp16
        ada_shifts_base = W["ada_shifts"]
        out_scales_base = W["out_scales"]    # int ptr to (steps, D) fp16
        out_shifts_base = W["out_shifts"]
        ate_base = W["action_time_embeds"]   # int ptr to (steps, T, D) fp16

        # ── Action encode ──
        # cast actions fp32 → fp16 into actions_fp16
        fvk.gpu_cast_fp32_to_fp16(
            _p(B["actions"]), _p(B["actions_fp16"]),
            T * ACTION_DIM, stream)

        # action_enc.W1: (T, action_dim) @ (action_dim, D) → (T, D) into a_emb
        gemm.fp16_nn(
            _p(B["actions_fp16"]), W["action_enc_w1"],
            _p(B["a_emb"]),
            T, D, ACTION_DIM, stream)
        fvk.add_bias_fp16(
            _p(B["a_emb"]), W["action_enc_b1"],
            T, D, stream)

        # Concat [a_emb (T,D), time_emb (T,D)] → (T, 2D) into concat.
        # Must be a ROW-WISE concat (row i = [a_emb[i], time_emb[i]]),
        # matching torch.cat([a_emb, time_emb], dim=-1) in the Isaac-GR00T
        # reference. The old two-gpu_copy pattern packed a_emb and
        # time_emb as two flat blocks, which the subsequent fp16_nn
        # (treating concat as (T, 2D) row-major) read as
        # [a_emb[2i], a_emb[2i+1]] / [time_emb[...]], not
        # [a_emb[i], time_emb[i]]. Silent on untrained embodiments but
        # breaks any trained one. cudaMemcpy2DAsync with dst pitch =
        # 2*D*2 bytes writes each source row into its correct 2D slot.
        ate_ptr_step = ate_base + step * T * D * 2   # bytes
        row_bytes = D * 2
        dst_pitch = 2 * D * 2
        stream_h = ctypes.c_void_p(stream)
        concat_ptr = _p(B["concat"])
        self._cudart.cudaMemcpy2DAsync(
            concat_ptr, dst_pitch,
            _p(B["a_emb"]), row_bytes,
            row_bytes, T,
            3, stream_h)     # 3 = cudaMemcpyDeviceToDevice
        self._cudart.cudaMemcpy2DAsync(
            concat_ptr + row_bytes, dst_pitch,
            ate_ptr_step, row_bytes,
            row_bytes, T,
            3, stream_h)

        # action_enc.W2: (T, 2D) @ (2D, D) → enc_h
        gemm.fp16_nn(
            _p(B["concat"]), W["action_enc_w2"],
            _p(B["enc_h"]),
            T, D, 2 * D, stream)
        fvk.add_bias_fp16(
            _p(B["enc_h"]), W["action_enc_b2"],
            T, D, stream)
        fvk.silu_inplace_fp16(_p(B["enc_h"]), T * D, stream)

        # action_enc.W3: (T, D) @ (D, D) → a_emb (overwrite)
        gemm.fp16_nn(
            _p(B["enc_h"]), W["action_enc_w3"],
            _p(B["a_emb"]),
            T, D, D, stream)
        fvk.add_bias_fp16(
            _p(B["a_emb"]), W["action_enc_b3"],
            T, D, stream)
        # Add position embedding (T, D)
        fvk.residual_add_fp16(
            _p(B["a_emb"]), W["pos_emb"],
            T * D, stream)

        # ── Build hidden = [state_feat (1,D), a_emb (T,D)] → (Sa, D) ──
        fvk.gpu_copy(
            _p(B["hidden"]), _p(B["state_feat"]),
            D * 2, stream)
        fvk.gpu_copy(
            _p(B["hidden"]) + D * 2,
            _p(B["a_emb"]),
            T * D * 2, stream)

        # ── 32 alternating self/cross blocks ──
        for l in range(DIT_L):
            is_self = (l % 2 == 1)
            ada_scale_ptr = ada_scales_base + (step * DIT_L + l) * D * 2
            ada_shift_ptr = ada_shifts_base + (step * DIT_L + l) * D * 2

            # ── AdaLayerNorm: x_norm = LN_no_affine(x) * (1+scale) + shift ──
            fvk.ada_layer_norm_fp16(
                _p(B["hidden"]),
                ada_scale_ptr, ada_shift_ptr,
                _p(B["h_norm"]),
                Sa, D, 1e-5, stream)

            if is_self:
                # ── Self-attention ──
                # QKV merged GEMM (FP16): (Sa, D) @ (D, 3D) → (Sa, 3D)
                gemm.fp16_nn(
                    _p(B["h_norm"]), W["dit_qkv_w"][l],
                    _p(B["qkv"]),
                    Sa, 3 * D, D, stream)
                fvk.add_bias_fp16(
                    _p(B["qkv"]),
                    W["dit_qkv_b_self"][l],
                    Sa, 3 * D, stream)

                # Split into attn backend's dit_self_Q/K/V (4D fp16)
                qkv_ptr = _p(B["qkv"])
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_Q"],
                    Sa, D, 3 * D, 0, stream)
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_K"],
                    Sa, D, 3 * D, D, stream)
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_V"],
                    Sa, D, 3 * D, 2 * D, stream)

                # Dispatch via AttentionBackend protocol
                # ``run("dit_self", l, q_seq=Sa)``. l is the full DiT
                # layer index (1, 3, 5, ..., 31 for self-attn blocks);
                # the backend uses the shared self-attn slot.
                attn_out_ptr = self.attn.run(
                    "dit_self", l, q_seq=Sa, stream=stream)
            else:
                # ── Cross-attention ──
                # Q projection: h_norm (Sa, D) @ q_w (D, D) → dit_cross_Q
                gemm.fp16_nn(
                    _p(B["h_norm"]), W["dit_q_w_fp16"][l],
                    attn_ptrs["dit_cross_Q"],
                    Sa, D, D, stream)
                fvk.add_bias_fp16(
                    attn_ptrs["dit_cross_Q"], W["dit_q_b"][l],
                    Sa, D, stream)
                # K/V already projected into the backend's cross_K[idx]/cross_V[idx]
                # by precompute_cross_kv() called before the diffusion loop.
                # Dispatch via AttentionBackend protocol
                # ``run("dit_cross", l, q_seq=Sa, kv_seq=Se)``. Pipeline
                # passes the full DiT layer index ``l`` (even layers
                # 0, 2, ..., 30); the backend computes
                # ``block_idx = l // 2`` internally and routes to the
                # right per-block K/V tensor.
                attn_out_ptr = self.attn.run(
                    "dit_cross", l,
                    q_seq=Sa, kv_seq=Se, stream=stream)

            # ── O projection (fp16) + residual ──
            # attn_out is (Sa, NH, HD) flat = (Sa, D)
            gemm.fp16_nn(
                attn_out_ptr, W["dit_o_w_fp16"][l],
                _p(B["o_out"]),
                Sa, D, D, stream)
            fvk.add_bias_fp16(
                _p(B["o_out"]), W["dit_o_b"][l],
                Sa, D, stream)
            fvk.residual_add_fp16(
                _p(B["hidden"]), _p(B["o_out"]),
                Sa * D, stream)

            # ── FFN: layer_norm_no_affine → fp16 GEMM+gelu+bias → fp16 GEMM+bias ──
            fvk.layer_norm_no_affine_fp16(
                _p(B["hidden"]), _p(B["h_norm"]),
                Sa, D, 1e-5, stream)

            # FFN-up (FP16): GEMM + bias + GELU
            gemm.fp16_nn(
                _p(B["h_norm"]), W["dit_ff_up_w"][l],
                _p(B["ff_h"]),
                Sa, H, D, stream)
            fvk.add_bias_fp16(
                _p(B["ff_h"]), W["dit_ff_up_b"][l],
                Sa, H, stream)
            fvk.gelu_inplace_fp16(_p(B["ff_h"]), Sa * H, stream)

            # FFN-down (FP16): GEMM + bias
            gemm.fp16_nn(
                _p(B["ff_h"]), W["dit_ff_down_w"][l],
                _p(B["ff_out"]),
                Sa, D, H, stream)
            fvk.add_bias_fp16(
                _p(B["ff_out"]), W["dit_ff_down_b"][l],
                Sa, D, stream)
            fvk.residual_add_fp16(
                _p(B["hidden"]), _p(B["ff_out"]),
                Sa * D, stream)

        # ── Final AdaLayerNorm (output conditioning) ──
        out_scale_ptr = out_scales_base + step * D * 2
        out_shift_ptr = out_shifts_base + step * D * 2
        fvk.ada_layer_norm_fp16(
            _p(B["hidden"]), out_scale_ptr, out_shift_ptr,
            _p(B["h_norm"]),
            Sa, D, 1e-6, stream)

        # ── proj_out_2 (fp16): (Sa, D) → (Sa, output_dim=1024) ──
        gemm.fp16_nn(
            _p(B["h_norm"]), W["proj_out_2_w"],
            _p(B["model_out"]),
            Sa, DIT_OUTPUT_DIM, D, stream)
        fvk.add_bias_fp16(
            _p(B["model_out"]), W["proj_out_2_b"],
            Sa, DIT_OUTPUT_DIM, stream)

        # ── action_decoder: 2-layer MLP ──
        gemm.fp16_nn(
            _p(B["model_out"]), W["action_dec_w1"],
            _p(B["dec_h"]),
            Sa, 1024, DIT_OUTPUT_DIM, stream)
        fvk.add_bias_fp16(
            _p(B["dec_h"]), W["action_dec_b1"],
            Sa, 1024, stream)
        fvk.relu_inplace_fp16(_p(B["dec_h"]), Sa * 1024, stream)
        gemm.fp16_nn(
            _p(B["dec_h"]), W["action_dec_w2"],
            _p(B["velocity"]),
            Sa, ACTION_DIM, 1024, stream)
        fvk.add_bias_fp16(
            _p(B["velocity"]), W["action_dec_b2"],
            Sa, ACTION_DIM, stream)

        # ── actions += dt * velocity (Euler step) ──
        # velocity is (Sa, ACTION_DIM) — only the last T tokens are
        # the action predictions; gpu_euler_step takes vel_offset to
        # skip the leading state row.
        vel_offset = (Sa - self.T) * ACTION_DIM
        fvk.gpu_euler_step(
            _p(B["actions"]), _p(B["velocity"]),
            self.T, ACTION_DIM, dt, vel_offset, stream)
