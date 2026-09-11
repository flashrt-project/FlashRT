"""FlashRT AMD — CDNA4 (MI350X) Pi0.5 inference pipeline.

Mirror of :mod:`flash_rt.models.pi05.pipeline_rtx` for the AMD/ROCm
backend. Framework-agnostic pipeline for Pi0.5 on CDNA4 datacenter GPUs:
the pipeline owns kernel composition, frontends own weights/framework
choice.

Controlled deltas vs the RTX original (everything else is structurally
identical — dimension constants, buffer sizes, hoisted-LayerNorm scheme,
kernel call order and fvk entry names, the two-graph capture logic,
warmup + sync sequence):

  * :class:`flash_rt.core.cuda_buffer.CudaBuffer` →
    :class:`flash_rt.amd.core.hip_buffer.HipBuffer`;
    :class:`flash_rt.core.cuda_graph.CUDAGraph` →
    :class:`flash_rt.amd.core.hip_graph.HipGraph`.
  * INT8 branches and INT8 scratch/calibration machinery dropped
    (the ``use_int8_*`` knobs were experimental RTX/Orin paths). BF16 +
    FP8 branches stay, controlled by ``use_fp8`` / ``use_fp8_decoder``
    exactly as upstream; the FP8 scratch + ``_fp8_gemm`` helpers are kept
    so M4 can flip FP8 on without touching this file.
  * The exec-contract replay routing (``FLASHRT_PI05_USE_EXEC``),
    ``export_runtime`` / ``export_model_runtime`` and the subgraph
    capture hooks are dropped — they bind to the CUDA-only
    ``runtime_export`` / ``subgraphs`` machinery and are out of scope for
    the AMD bring-up.

Architecture::

    frontend (torch safetensors, ...)
        │
        ├── loads + FP8-quantizes weights into framework-native tensors
        │   then exposes raw device pointer ints
        │
        ├── instantiates Cdna4AttnBackend (owns Q/K/V/O bf16 tensors)
        │
        └── constructs Pi05Pipeline(gemm, fvk, attn_backend, weights_ptrs, dims)
                │
                ├── allocates internal working buffers as HipBuffer (no torch)
                ├── runs fvk kernels via pointers for all GEMM / norm / fused ops
                ├── delegates attention to attn_backend (pluggable)
                └── captures a HIP graph via flash_rt.amd.core.hip_graph

All activations are BF16. FP8 E4M3 quantization on weights + activations
for the large GEMMs (vision attn/FFN, encoder attn/FFN, decoder
attn/FFN); see ``_quantize_weights_fp8`` in the frontend.
"""

from __future__ import annotations

import ctypes
import logging
import os

import numpy as np
import ml_dtypes

from flash_rt.amd.core.hip_buffer import HipBuffer
from flash_rt.amd.core.hip_graph import HipGraph

logger = logging.getLogger(__name__)


def _fp8_nt_autotune_enabled(hardware: str | None, fp8_layout: str) -> bool:
    """Whether to run BLAS top-N autotune for FP8 NT GEMMs.

    Mirrors the RTX policy knob (``FLASHRT_FP8_NT_AUTOTUNE``); the RTX
    Ada/SM89 skip heuristic does not apply to any AMD hardware tag, so
    the ``auto`` default resolves to enabled here.
    """
    policy = os.environ.get("FLASHRT_FP8_NT_AUTOTUNE", "auto").strip().lower()
    if policy in ("1", "true", "on", "force"):
        return True
    if policy in ("0", "false", "off", "safe", "skip"):
        return False
    if policy != "auto":
        logger.warning(
            "Unknown FLASHRT_FP8_NT_AUTOTUNE=%r; using auto policy", policy)
    return True


# Fixed Pi0.5 model dimensions
VIS_L = 27           # SigLIP-L layers
VIS_D = 1152         # SigLIP hidden dim
VIS_H = 4304         # SigLIP FFN hidden
VIS_NH = 16          # SigLIP num attn heads
VIS_HD = 72          # SigLIP head dim
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3  # 588

ENC_L = 18           # Gemma-2B encoder layers
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8           # query heads (GQA)
ENC_NKV = 1          # kv heads (GQA)
ENC_HD = 256

DEC_L = 18           # Gemma-300M decoder layers
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
NUM_STEPS_DEFAULT = 10

# BF16 tagging for numpy arrays. There are two cases:
#   BF16 (np.float16): SIZING-ONLY. HipBuffer allocations only use this
#     to compute nbytes = count * 2, which matches BF16. Safe because
#     the buffer is never filled with numeric values from numpy directly.
#   BF16_NP (ml_dtypes.bfloat16): REAL BF16 with the correct bit layout.
#     Must be used for any numpy staging array that carries meaningful
#     numeric values (ones vector, RoPE cos/sin, ...). ``np.float16`` has
#     the same byte width but a different exponent/mantissa layout —
#     using it for BF16 data yields silent ~500x underflow when the
#     BF16 kernel re-interprets the bits.
BF16 = np.float16            # sizing placeholder only
BF16_NP = ml_dtypes.bfloat16  # real BF16 numpy dtype for numeric staging
FP8 = np.uint8
FP32 = np.float32


class Pi05Pipeline:
    """Pi0.5 inference pipeline for AMD CDNA4 (MI350X) GPUs.

    The pipeline composes ``flash_rt_amd_kernels`` kernels + ``GemmRunner``
    BLAS calls + an injected attention backend
    (:class:`~flash_rt.amd.hardware.cdna4.attn_backend.Cdna4AttnBackend`)
    into the full SigLIP → Gemma encoder → Gemma decoder flow.

    Args:
        gemm:         ``fvk.GemmRunner()`` — BF16/FP8 GEMM driver.
        fvk:          The ``flash_rt_amd_kernels`` module (raw pointer kernels).
        attn_backend: Attention backend (owns Q/K/V/O tensors, runs attention).
        weights:      Dict of weight pointers — see class docstring for schema.
        num_views:    Number of observation camera views (1/2/3).
        max_prompt_len: Max tokenised prompt length (language embeds buffer size).
        chunk_size:   Diffusion action chunk length (default 10).
        use_fp8:      Enable FP8 E4M3 quantization for large GEMMs.
        use_fp8_decoder: Enable FP8 on decoder branch (else BF16).
        num_steps:    Diffusion denoise steps (default 10).

    Expected weights dict keys:
        Vision BF16:
            vision_patch_embedding_w (588,1152), vision_patch_embedding_b (1152,),
            vision_position_embedding (256,1152),
            vision_pre_attn_norm_w[L], vision_pre_attn_norm_b[L],
            vision_pre_ffn_norm_w[L], vision_pre_ffn_norm_b[L],
            vision_attn_qkv_b[L], vision_attn_o_b[L],
            vision_ffn_up_b[L], vision_ffn_down_b[L],
            vision_final_norm_w, vision_final_norm_b,
            encoder_multi_modal_projector_b,
        Vision FP8 (each entry is (fp8_ptr, scale_ptr) tuple):
            fp8.vision_attn_qkv_w_{0..26}, fp8.vision_attn_o_w_{0..26},
            fp8.vision_ffn_up_w_{0..26},   fp8.vision_ffn_down_w_{0..26},
            fp8.vision_projector_w,
        Encoder FP8:
            fp8.encoder_attn_qkv_w_{0..17}, fp8.encoder_attn_o_w_{0..17},
            fp8.encoder_ffn_gate_up_w_{0..17}  (merged gate+up: (D,2H)),
            fp8.encoder_ffn_down_w_{0..17},
        Decoder BF16:
            decoder_time_mlp_in_w/b, decoder_time_mlp_out_w/b,
            decoder_time_embeds (10,1024),
            decoder_pre_attn_norm_mod_w/b[L], decoder_pre_ffn_norm_mod_w/b[L],
            decoder_final_norm_mod_w/b,
            decoder_action_in_proj_w/b,
            decoder_action_out_proj_w/b   (frontend MUST pre-scale by -1/num_steps),
        Decoder FP8:
            fp8.decoder_attn_qkv_w_{0..17}, fp8.decoder_attn_o_w_{0..17},
            fp8.decoder_ffn_gate_up_w_{0..17}, fp8.decoder_ffn_down_w_{0..17},
        Language (rebound per-prompt by frontend):
            language_embeds_ptr   — pointer to bf16 (max_prompt_len, 2048) buffer
    """

    def __init__(self, gemm, fvk, attn_backend, weights, *,
                 num_views: int, max_prompt_len: int,
                 chunk_size: int = NUM_STEPS_DEFAULT,
                 use_fp8: bool = True, use_fp8_decoder: bool = True,
                 vision_pool_factor: int = 1,
                 vision_num_layers: int = VIS_L,
                 num_steps: int = NUM_STEPS_DEFAULT,
                 fixed_shape: bool = False):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights

        # Fixed-shape state-prompt mode: one captured graph at the MAX prompt
        # length serves every length via seqused masking + devpos K/V append.
        # The attention backend masks padded keys; this pipeline appends the
        # decoder K/V right after the valid prefix.
        # NOTE: the backend is SHARED across pipelines, so the frontend re-syncs
        # backend.set_fixed_shape(active_pipeline._fixed_shape) on every prompt
        # — this build-time call must not be relied on for run-time mode.
        self._fixed_shape = bool(fixed_shape)
        self.attn.set_fixed_shape(self._fixed_shape)

        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.num_steps = int(num_steps)
        self.use_fp8 = bool(use_fp8)
        self.use_fp8_decoder = bool(use_fp8_decoder)
        # Decoder small-M GEMM backend: "mfma" (default) routes the
        # decoder GEMMs whose weights carry an MFMA-packed copy to
        # smallm_mfma_nt_packed; "hipblaslt" keeps the library path.
        # Read once here (env flips after capture never re-evaluate).
        self.dec_gemm_backend = os.environ.get(
            "FVK_AMD_DEC_GEMM", "mfma").strip().lower()
        self.vision_pool_factor = int(vision_pool_factor)
        self.vision_num_layers = int(vision_num_layers)
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.vision_pool_factor not in (1, 2, 4):
            raise ValueError(
                "vision_pool_factor must be one of {1, 2, 4}; "
                f"got {self.vision_pool_factor}")
        if not 1 <= self.vision_num_layers <= VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {VIS_L}], "
                f"got {self.vision_num_layers}")
        self.fp8_layout = weights.get("fp8_layout", "kn")
        if self.fp8_layout not in ("kn", "nk"):
            raise ValueError(f"unsupported FP8 layout: {self.fp8_layout!r}")
        self.hardware = weights.get("hardware")
        self._autotune_fp8_nt = _fp8_nt_autotune_enabled(
            self.hardware, self.fp8_layout)
        if self.fp8_layout == "nk" and not self._autotune_fp8_nt:
            logger.info(
                "Skipping FP8 NT top-N autotune for hardware=%s; keeping "
                "heuristic top-1 algorithm. Set "
                "FLASHRT_FP8_NT_AUTOTUNE=force to override.",
                self.hardware or "unknown")

        # Derived sizes
        # vision_seq: full SigLIP token count (pre-pooling) — used for SigLIP buffers
        # vision_seq_enc: token count fed to the Gemma encoder (post-pooling)
        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        pf = self.vision_pool_factor
        self.vision_seq_enc = self.vision_seq // (pf * pf)
        self.encoder_seq_len = self.vision_seq_enc + self.max_prompt_len
        self.total_kv = self.encoder_seq_len + self.chunk_size

        # Attention pointers (owned by attn_backend)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._enc_kv_layer_stride = self._attn_ptrs["enc_k_layer_stride_bytes"]

        # Allocate internal buffers (all HipBuffer, all BF16 unless noted)
        self.bufs = self._allocate_buffers()

        # RoPE table (max positions = encoder_seq_len + chunk_size)
        self._build_rope_table()

        # valid_encoder_len placeholder — updated per forward
        _valid_len = np.array([self.vision_seq + 1], dtype=np.int32)
        self.bufs["valid_encoder_len"] = HipBuffer.from_numpy(_valid_len)

        # Pre-allocated RMS norm "ones" weight vectors (REAL BF16 bit pattern)
        _ones_2048 = np.ones(ENC_D, dtype=BF16_NP)
        _ones_1024 = np.ones(DEC_D, dtype=BF16_NP)
        self._rms_ones_enc = HipBuffer.from_numpy(_ones_2048)
        self._rms_ones_dec = HipBuffer.from_numpy(_ones_1024)

        # FP8 activation scratch buffers + per-layer static scales
        self.fp8_act_scales = {}  # name -> HipBuffer(1, fp32)
        self.fp8_calibrated = False
        # Fused decoder attn->FP8 epilogue: armed once post-calibration by
        # _arm_decoder_attn_fp8out() (see there for the env gate).
        self._dec_attn_fp8out_armed = False
        # Set once autotune_gemms() has benchmarked this pipeline's (fixed) GEMM
        # shapes, so repeat calls (frontend + record_infer_graph) are no-ops.
        self._gemms_autotuned = False
        self._fp8_current_decoder_step = -1
        self._allocate_fp8_scratch()

        # Pre-computed decoder style params — frontend pre-computes these in
        # its native framework and passes raw bf16 bytes; see frontend's
        # ``_precompute_decoder_styles`` helper.
        self._upload_precomputed_styles()

        # HIP graph state (set by record_infer_graph)
        self._graph = None
        self._decoder_only_graph = None  # for temporal K/V caching
        self._graph_stream = None  # ctypes.c_void_p
        from flash_rt.amd.core.hip_buffer import _check, _hip
        self._hip = _hip
        # Shared HIP return-code checker: every raw self._hip.* call must
        # route its return through this (raises RuntimeError on failure).
        self._hip_check = _check

        # Pre-expand vision position embedding across num_views (see
        # ``vision_encoder``'s patch embed path). We replicate the
        # (256, VIS_D) pos_emb ``num_views`` times into a dedicated
        # pipeline buffer so we can feed it to the BF16 ``bias_residual``
        # kernel.
        self._build_pos_embed_expanded()

    # ══════════════════════════════════════════════════════════════════
    #   Buffer allocation
    # ══════════════════════════════════════════════════════════════════

    def _allocate_buffers(self) -> dict:
        """Allocate all pipeline working buffers as HipBuffer."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        ds = self.chunk_size
        B = {}

        # ── Vision (SigLIP) ──
        # observation_images_normalized is the input slot; frontend writes here.
        B["observation_images_normalized"] = HipBuffer.device_empty(
            nv * 224 * 224 * 3, BF16)
        # Patch-embedded + residual stream (full pre-pool token count)
        B["vision_x"] = HipBuffer.device_empty(vs * VIS_D, BF16)
        B["vision_x_norm"] = HipBuffer.device_empty(vs * VIS_D, BF16)
        # Pooled vision tokens fed to the Gemma encoder (== vision_x when pool_factor=1)
        vs_enc = self.vision_seq_enc
        if self.vision_pool_factor > 1:
            B["vision_x_pooled"] = HipBuffer.device_empty(vs_enc * VIS_D, BF16)
        else:
            B["vision_x_pooled"] = B["vision_x"]  # no-op alias
        B["vision_QKV"] = HipBuffer.device_empty(vs * 3 * VIS_D, BF16)
        B["vision_hidden"] = HipBuffer.device_empty(vs * VIS_H, BF16)
        # Position embedding expanded (nv * 256, 1152) — built once, reused
        B["vision_pos_embed_expanded"] = HipBuffer.device_empty(vs * VIS_D, BF16)

        # ── Encoder (Gemma-2B) ──
        B["encoder_rope_weights"] = HipBuffer.device_empty(es * 2 * ENC_HD // 2, BF16)
        # (Sized conservatively: encoder_seq_len * 256 bf16 elements.)
        B["encoder_x"] = HipBuffer.device_empty(es * ENC_D, BF16)
        B["encoder_x_norm"] = HipBuffer.device_empty(es * ENC_D, BF16)
        B["encoder_QKV"] = HipBuffer.device_empty(es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, BF16)
        B["encoder_hidden"] = HipBuffer.device_empty(es * ENC_H, BF16)
        # Merged gate+up output (legacy BF16 path) or gate-only buffer for SiLU-gated.
        B["encoder_gate_merged"] = HipBuffer.device_empty(es * 2 * ENC_H, BF16)
        # Gate buffer for SiLU-gated fusion (encoder): (es, ENC_H) BF16
        B["encoder_gate_buf"] = HipBuffer.device_empty(es * ENC_H, BF16)

        # ── Decoder (Gemma-300M) ──
        B["decoder_rope_weights"] = HipBuffer.device_empty(ds * 256, BF16)
        B["decoder_x"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        B["decoder_action_buf"] = HipBuffer.device_empty(ds * ACTION_DIM, BF16)
        # Pre-computed per-step, per-layer style params (BF16)
        B["decoder_time_emb"] = HipBuffer.device_empty(
            self.num_steps * ds * DEC_D, BF16)
        B["decoder_style_attn"] = HipBuffer.device_empty(
            self.num_steps * DEC_L * ds * 3 * DEC_D, BF16)
        B["decoder_style_ffn"] = HipBuffer.device_empty(
            self.num_steps * DEC_L * ds * 3 * DEC_D, BF16)
        B["decoder_style_final"] = HipBuffer.device_empty(
            self.num_steps * ds * 3 * DEC_D, BF16)
        B["decoder_QKV"] = HipBuffer.device_empty(
            ds * (DEC_NH + 2 * DEC_NKV) * DEC_HD, BF16)
        B["decoder_hidden"] = HipBuffer.device_empty(ds * DEC_H, BF16)
        B["decoder_gate_merged"] = HipBuffer.device_empty(ds * 2 * DEC_H, BF16)
        # Gate buffer for SiLU-gated fusion (decoder): (ds, DEC_H) BF16
        B["decoder_gate_buf"] = HipBuffer.device_empty(ds * DEC_H, BF16)
        B["diffusion_noise"] = HipBuffer.device_empty(ds * ACTION_DIM, BF16)
        # Optional RTC-prefix conditioning slot. It is only read by the
        # explicit rtc-prefix action graph, not by the default full graph.
        B["rtc_prev_action_chunk"] = HipBuffer.device_empty(
            ds * ACTION_DIM, BF16)
        # Optional full RTC-guidance controls. These are producer hot inputs
        # for a future VJP-guided action graph; default/prefix graphs ignore
        # them and therefore keep byte-identical behavior.
        B["rtc_prefix_weights"] = HipBuffer.device_empty(ds, FP32)
        B["rtc_guidance_weight"] = HipBuffer.device_empty(1, FP32)
        # Decoder scratch for ada_rms_norm output + gate
        B["x_normed_buf"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        B["gate_buf"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        # Scratch for vision patch im2col output (BF16 (nv*256, 588))
        B["vision_patches"] = HipBuffer.device_empty(vs * VIS_PATCH_FLAT, BF16)

        return B

    def _allocate_fp8_scratch(self) -> None:
        """Allocate reusable FP8 activation scratch + scale buffers."""
        if not self.use_fp8:
            return
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        ds = self.chunk_size

        # Vision FP8 scratch — sized for D=1152 and H=4304
        B["vis_act_fp8"] = HipBuffer.device_zeros(vs * VIS_D, FP8)
        B["vis_act_fp8_large"] = HipBuffer.device_zeros(vs * VIS_H, FP8)
        B["vis_act_scale"] = HipBuffer.device_zeros(1, FP32)

        # Encoder FP8 scratch — D=2048, H_merged=32768 (2*16384)
        B["enc_act_fp8"] = HipBuffer.device_zeros(es * ENC_D, FP8)
        B["enc_act_fp8_large"] = HipBuffer.device_zeros(es * 2 * ENC_H, FP8)
        B["enc_act_scale"] = HipBuffer.device_zeros(1, FP32)

        # Decoder FP8 scratch — D=1024, H_merged=8192
        B["dec_act_fp8"] = HipBuffer.device_zeros(ds * DEC_D, FP8)
        B["dec_act_fp8_large"] = HipBuffer.device_zeros(ds * 2 * DEC_H, FP8)
        B["dec_act_scale"] = HipBuffer.device_zeros(1, FP32)

    # ══════════════════════════════════════════════════════════════════
    #   RoPE table
    # ══════════════════════════════════════════════════════════════════

    def _build_rope_table(self) -> None:
        """Build the BF16 interleaved cos/sin RoPE table (max_pos, 256).

        Uploaded to ``bufs['encoder_rope_weights']`` for the prefix range
        [0 .. encoder_seq_len). The decoder RoPE slice [encoder_seq_len ..
        encoder_seq_len+chunk_size) is set per-prompt by :meth:`forward`.
        """
        max_pos = self.encoder_seq_len + self.chunk_size
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]  # (max_pos, 128)
        cos = np.cos(phase).astype(BF16_NP)
        sin = np.sin(phase).astype(BF16_NP)
        # Interleaved [cos0, sin0, cos1, sin1, ...] → (max_pos, 256)
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        self._rope_table_np = interleaved  # keep on host for per-prompt decoder slice

        # Initial encoder RoPE slice = first encoder_seq_len rows
        enc_rope_slice = interleaved[:self.encoder_seq_len]
        # Need to allocate a correctly-sized HipBuffer, overwriting the placeholder
        self.bufs["encoder_rope_weights"] = HipBuffer.from_numpy(
            np.ascontiguousarray(enc_rope_slice))

        # Decoder RoPE: placeholder, will be overwritten per-prompt.
        # Decoder suffix positions start after the valid prefix tokens.
        # This mirrors openpi's ``prefix_offsets + cumsum(suffix_mask) - 1``:
        # the first action token is at position ``vision_seq_enc + prompt_len``.
        # Use vision_seq_enc (not vision_seq) so the rope table isn't overrun
        # when vision_pool_factor > 1.
        dec_rope_slice = interleaved[
            self.vision_seq_enc : self.vision_seq_enc + self.chunk_size]
        self.bufs["decoder_rope_weights"] = HipBuffer.from_numpy(
            np.ascontiguousarray(dec_rope_slice))

    def _set_decoder_rope_for_prompt(self, prompt_len: int) -> None:
        """Update ``decoder_rope_weights`` for a new prompt length."""
        start = self.vision_seq_enc + prompt_len
        end = start + self.chunk_size
        self.bufs["decoder_rope_weights"].upload(
            np.ascontiguousarray(self._rope_table_np[start:end]))

    def _build_pos_embed_expanded(self) -> None:
        """Replicate (256, VIS_D) pos_emb ``num_views`` times into a pipeline buffer.

        The source pos_emb comes from the frontend as a raw device pointer
        and is BF16 ``(256, VIS_D)``. We D2D-copy it num_views times into
        a pipeline-owned contiguous (num_views*256, VIS_D) BF16 buffer so
        that ``bias_residual`` can read it as a flat row-major tensor.
        """
        pos_src_ptr = self.weights["vision_position_embedding"]
        per_view_nbytes = VIS_SEQ_PER_VIEW * VIS_D * 2  # bf16 = 2 bytes
        dst_buf = self.bufs["vision_pos_embed_expanded"]
        assert dst_buf.nbytes == self.num_views * per_view_nbytes
        for v in range(self.num_views):
            self._hip_check(self._hip.hipMemcpy(
                ctypes.c_void_p(dst_buf.ptr.value + v * per_view_nbytes),
                ctypes.c_void_p(pos_src_ptr),
                per_view_nbytes, 3), "hipMemcpy D2D")
        self._hip_check(self._hip.hipDeviceSynchronize(),
                        "hipDeviceSynchronize")

    # ══════════════════════════════════════════════════════════════════
    #   Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _p(buf) -> int:
        """Extract int pointer from a HipBuffer."""
        return buf.ptr.value

    def _weight_fp8(self, name: str) -> tuple[int, int]:
        """Look up an FP8-quantized weight: returns (data_ptr, scale_ptr)."""
        return self.weights["fp8"][name]

    def _fp8_matmul(self, act_fp8_ptr: int, weight_fp8_ptr: int,
                    out_bf16_ptr: int, M: int, N: int, K: int,
                    act_scale_ptr: int, weight_scale_ptr: int,
                    stream: int) -> None:
        """Dispatch the FP8 GEMM for the selected weight layout."""
        if self.fp8_layout == "nk":
            self.gemm.fp8_nt_dev(
                act_fp8_ptr, weight_fp8_ptr, out_bf16_ptr,
                M, N, K, act_scale_ptr, weight_scale_ptr, stream=stream)
        else:
            self.gemm.fp8_nn_dev(
                act_fp8_ptr, weight_fp8_ptr, out_bf16_ptr,
                M, N, K, act_scale_ptr, weight_scale_ptr, stream=stream)

    def _autotune_fp8_matmul(self, act_fp8_ptr: int, weight_fp8_ptr: int,
                             out_bf16_ptr: int, M: int, N: int, K: int,
                             act_scale_ptr: int, weight_scale_ptr: int) -> None:
        """Autotune the FP8 GEMM for the selected weight layout."""
        # Heuristic pool depth for the timed selection. Deeper pools
        # find faster encoder algos (gate_up 40.6 -> 34.1us at 64+) but
        # widen run-to-run variance: with 128 candidates a mis-timed
        # trial occasionally wins and ships a slow/odd-numerics pick
        # (observed: one 128-pool process at +10 ms E2E). Default stays
        # the long-validated 16; opt into a deeper pool via
        # FLASHRT_FP8_ALGO_POOL when setup-time re-runs can vet it.
        pool = int(os.environ.get("FLASHRT_FP8_ALGO_POOL", "16"))
        if self.fp8_layout == "nk":
            if not getattr(self, "_autotune_fp8_nt", True):
                return
            self.gemm.autotune_fp8_nt_dev(
                act_fp8_ptr, weight_fp8_ptr, out_bf16_ptr,
                M, N, K, act_scale_ptr, weight_scale_ptr, pool)
        else:
            self.gemm.autotune_fp8_nn_dev(
                act_fp8_ptr, weight_fp8_ptr, out_bf16_ptr,
                M, N, K, act_scale_ptr, weight_scale_ptr, pool)

    def _upload_precomputed_styles(self) -> None:
        """Upload frontend-precomputed decoder style/time buffers.

        Expects ``self.weights['precomputed']`` dict with keys (numpy
        arrays, dtype = BF16-equivalent, 2 bytes per element):
            time_emb      — (num_steps, chunk_size, DEC_D)
            style_attn    — (num_steps, DEC_L, chunk_size, 3*DEC_D)
            style_ffn     — (num_steps, DEC_L, chunk_size, 3*DEC_D)
            style_final   — (num_steps, chunk_size, 3*DEC_D)
        """
        pre = self.weights["precomputed"]
        self.bufs["decoder_time_emb"].upload(
            np.ascontiguousarray(pre["time_emb"]))
        self.bufs["decoder_style_attn"].upload(
            np.ascontiguousarray(pre["style_attn"]))
        self.bufs["decoder_style_ffn"].upload(
            np.ascontiguousarray(pre["style_ffn"]))
        self.bufs["decoder_style_final"].upload(
            np.ascontiguousarray(pre["style_final"]))

    def _style_slice_ptr(self, buf_name: str, step: int,
                         layer: int | None = None) -> int:
        """Compute the device pointer of a per-step (per-layer) style slice.

        Layout (bf16, 2 bytes per element):
            style_attn/ffn: (num_steps, DEC_L, chunk_size, 3*DEC_D)
                            slice ``[step, layer]`` has nbytes = chunk * 3*D * 2
            style_final:   (num_steps, chunk_size, 3*DEC_D)
                            slice ``[step]`` has nbytes = chunk * 3*D * 2
            time_emb:      (num_steps, chunk_size, DEC_D)
                            slice ``[step]`` has nbytes = chunk * D * 2
        """
        base = self.bufs[buf_name].ptr.value
        ds = self.chunk_size
        if buf_name == "decoder_time_emb":
            return base + step * ds * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * ds * 3 * DEC_D * 2
        # style_attn / style_ffn
        per_layer = ds * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer

    def _enc_kv_layer_ptrs(self, layer: int, offset_tokens: int = 0) -> tuple[int, int]:
        """Compute (K_ptr, V_ptr) for encoder/decoder cross-attn layer cache.

        The attention backend owns ``enc_K`` / ``enc_V`` as
        ``(num_enc_layers, total_kv_max, 1, HD) bf16``. We compute the
        per-layer slice plus an optional token offset (used by the decoder
        which writes into ``[enc_seq : enc_seq+dec_seq]``).
        """
        k_base = self._attn_ptrs["enc_K"]
        v_base = self._attn_ptrs["enc_V"]
        layer_stride = self._enc_kv_layer_stride  # bytes
        token_offset_bytes = offset_tokens * ENC_NKV * ENC_HD * 2  # bf16
        return (k_base + layer * layer_stride + token_offset_bytes,
                v_base + layer * layer_stride + token_offset_bytes)

    # ══════════════════════════════════════════════════════════════════
    #   FP8 GEMM helpers
    # ══════════════════════════════════════════════════════════════════

    def _fp8_scale_buf(self, name: str) -> HipBuffer:
        """Get (lazy-allocate) the per-layer FP8 activation scale buffer."""
        buf = self.fp8_act_scales.get(name)
        if buf is None:
            buf = HipBuffer.device_zeros(1, FP32)
            self.fp8_act_scales[name] = buf
        return buf

    def _fp8_decoder_step_scale_name(self, weight_name: str) -> str:
        step = getattr(self, "_fp8_current_decoder_step", -1)
        if step >= 0 and weight_name.startswith("decoder_"):
            return f"{weight_name}__step_{step}"
        return weight_name

    def _fp8_static_scale_ptr(self, weight_name: str) -> int:
        step_name = self._fp8_decoder_step_scale_name(weight_name)
        buf = self.fp8_act_scales.get(step_name)
        if buf is None:
            buf = self.fp8_act_scales[weight_name]
        return buf.ptr.value

    def _pick_fp8_scratch(self, weight_name: str, act_n: int) -> tuple[int, int]:
        """Return (act_fp8_ptr, act_scale_ptr) scratch for the right domain.

        Chooses between vision/encoder/decoder FP8 scratch buffers based on
        the weight name prefix, and between the small and large variant
        based on ``act_n`` (number of elements to quantize).
        """
        B = self.bufs
        if weight_name.startswith("vision_") or weight_name == "vision_projector_w":
            small = B["vis_act_fp8"]
            large = B["vis_act_fp8_large"]
            scratch_scale = B["vis_act_scale"]
        elif weight_name.startswith("encoder_"):
            small = B["enc_act_fp8"]
            large = B["enc_act_fp8_large"]
            scratch_scale = B["enc_act_scale"]
        else:
            small = B["dec_act_fp8"]
            large = B["dec_act_fp8_large"]
            scratch_scale = B["dec_act_scale"]
        buf = small if act_n <= (small.nbytes // 1) else large
        return buf.ptr.value, scratch_scale.ptr.value

    def _fp8_gemm(self, act_bf16_ptr: int, act_n: int, weight_name: str,
                  out_bf16_ptr: int, M: int, N: int, K: int, stream: int) -> None:
        """FP8 GEMM path: dynamic-quantize activation → FP8 matmul → BF16 out.

        After calibration, uses the per-layer static scale buffer (single
        kernel path). During calibration, writes the dynamic scale into the
        layer's static scale buffer so subsequent inference can reuse it.
        """
        fvk = self.fvk
        w_fp8_ptr, w_scale_ptr = self._weight_fp8(weight_name)
        act_fp8_ptr, scratch_scale_ptr = self._pick_fp8_scratch(weight_name, act_n)

        if self.fp8_calibrated and weight_name in self.fp8_act_scales:
            static_scale_ptr = self._fp8_static_scale_ptr(weight_name)
            fvk.quantize_fp8_static(
                act_bf16_ptr, act_fp8_ptr, static_scale_ptr, act_n, stream=stream)
            self._fp8_matmul(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, static_scale_ptr, w_scale_ptr, stream)
        else:
            # Dynamic calibration path: use the per-call scratch scale for
            # this GEMM, while accumulating a max into the persistent static
            # scale. Decoder weights are visited once per denoise step; direct
            # writes would leave only the final step's scale.
            layer_scale = self._fp8_scale_buf(weight_name)
            step_scale = self._fp8_scale_buf(
                self._fp8_decoder_step_scale_name(weight_name))
            fvk.quantize_fp8_device(
                act_bf16_ptr, act_fp8_ptr, scratch_scale_ptr, act_n, stream=stream)
            fvk.fp8_accumulate_scale_max(
                scratch_scale_ptr, layer_scale.ptr.value, stream=stream)
            if step_scale is not layer_scale:
                fvk.fp8_accumulate_scale_max(
                    scratch_scale_ptr, step_scale.ptr.value, stream=stream)
            self._fp8_matmul(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, scratch_scale_ptr, w_scale_ptr, stream)

    def _fp8_gemm_fused(self, act_fp8_ptr: int, weight_name: str,
                        out_bf16_ptr: int, M: int, N: int, K: int,
                        act_scale_ptr: int, stream: int) -> None:
        """FP8 GEMM with pre-quantized activation (from fused norm→FP8 kernel)."""
        w_fp8_ptr, w_scale_ptr = self._weight_fp8(weight_name)
        # Measured routing (in-situ profiling, gfx950): the MFMA
        # packed kernel wins in-situ on the K<=2048 and wide-N decoder
        # shapes (qkv/o/gate_up ~4.7-5.0us vs hipblaslt ~6.1); the
        # 64-workgroup K=4096 down-proj stays on hipblaslt (7.17 vs 6.2).
        if (self.dec_gemm_backend == "mfma"
                and M <= 16 and N % 16 == 0 and K % 1024 == 0
                and (K <= 2048 or N >= 8192)):
            packed = self.weights.get("fp8_packed", {})
            wp_ptr = packed.get(weight_name)
            if wp_ptr is not None:
                self.fvk.smallm_mfma_nt_packed(
                    act_fp8_ptr, wp_ptr, out_bf16_ptr,
                    M, N, K, act_scale_ptr, w_scale_ptr, stream=stream)
                return
        self._fp8_matmul(
            act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
            M, N, K, act_scale_ptr, w_scale_ptr, stream)

    def _bias_add_bf16(self, x_ptr: int, bias_ptr: int,
                       seq: int, dim: int, stream: int) -> None:
        """Broadcast-add bias to an (seq, dim) bf16 buffer in place.

        TODO(v2): add a dedicated ``add_bias_bf16`` kernel path here. For
        now we reuse ``bias_residual`` with a dedicated persistent zero
        buffer as the ``x`` operand, yielding ``x += bias``. The wasted
        bandwidth is ~0.2 ms total for vision (27 layers × 2 bias adds).
        """
        if not hasattr(self, "_bias_zero_buf"):
            # Size once for the largest shape we'll ever bias-add.
            nbytes = max(
                self.vision_seq * 3 * VIS_D,  # vision QKV
                self.vision_seq * VIS_H,      # vision FFN up
            )
            self._bias_zero_buf = HipBuffer.device_zeros(nbytes, BF16)
        self.fvk.bias_residual(
            x_ptr, self._bias_zero_buf.ptr.value, bias_ptr,
            seq, dim, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase A: Vision (SigLIP)
    # ══════════════════════════════════════════════════════════════════

    def vision_encoder(self, stream: int = 0) -> None:
        """Run the full 27-layer SigLIP vision encoder.

        Input:  ``bufs['observation_images_normalized']`` (num_views, 224, 224, 3) bf16
        Output: ``bufs['vision_x']`` (num_views * 256, 1152) bf16
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        nv = self.num_views

        # A1: Patch embed — im2col + GEMM + bias + pos embed.
        # We use the BF16 ``bias_residual`` kernel with a pre-replicated
        # pos_emb buffer.
        fvk.patch_im2col(
            B["observation_images_normalized"].ptr.value,
            B["vision_patches"].ptr.value,
            nv, stream)
        gemm.bf16_nn(
            B["vision_patches"].ptr.value,
            W["vision_patch_embedding_w"],
            B["vision_x"].ptr.value,
            seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        fvk.bias_residual(
            B["vision_x"].ptr.value,
            B["vision_pos_embed_expanded"].ptr.value,
            W["vision_patch_embedding_b"],
            seq, VIS_D, stream=stream)

        # A2-A6: SigLIP transformer layers (vision_num_layers ≤ VIS_L=27)
        # Reducing layers is a quality/speed trade-off (untrained skip).
        use_fp8 = self.use_fp8 and "vision_attn_qkv_w_0" in self.weights.get("fp8", {})

        # Layer-0 pre-attn LayerNorm runs here (the rest are run at the
        # prior layer's post-FFN position, so each iteration arrives with
        # vision_x_norm already containing LN of vision_x).
        fvk.layer_norm(
            B["vision_x"].ptr.value,
            W["vision_pre_attn_norm_w"][0], W["vision_pre_attn_norm_b"][0],
            B["vision_x_norm"].ptr.value,
            seq, VIS_D, 1e-5, stream=stream)

        last = self.vision_num_layers - 1
        for i in range(self.vision_num_layers):
            self._vision_layer(i, seq, use_fp8, stream, is_last=(i == last))

        # A7 (optional): Spatial average pooling — reduce (nv*256, D) to (nv*64, D)
        # Runs only when vision_pool_factor > 1. vision_x_pooled is a separate
        # buffer; vision_x stays intact so this is non-destructive.
        if self.vision_pool_factor > 1:
            H = 16  # VIS_SEQ_PER_VIEW = 256 = 16 × 16 patch grid
            fvk.avg_pool_vision_tokens(
                B["vision_x"].ptr.value,
                B["vision_x_pooled"].ptr.value,
                nv, H, H, VIS_D, self.vision_pool_factor, stream)

    def _vision_layer(self, i: int, seq: int, use_fp8: bool, stream: int,
                       is_last: bool = False) -> None:
        """One SigLIP transformer layer (pre-norm, LayerNorm, GELU FFN).

        Pre-attn LayerNorm is hoisted: layer 0's runs in
        :meth:`vision_encoder` before the loop; layers 1..N-1's runs
        at the previous layer's post-FFN-down position. Each iteration
        enters with ``vision_x_norm`` already holding
        ``LayerNorm(vision_x)`` for *this* layer's pre-attn site.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs

        # QKV GEMM (FP8 / BF16) → vision_QKV
        if use_fp8:
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, seq * VIS_D,
                f"vision_attn_qkv_w_{i}",
                B["vision_QKV"].ptr.value,
                seq, 3 * VIS_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                B["vision_x_norm"].ptr.value, W["vision_attn_qkv_w"][i],
                B["vision_QKV"].ptr.value,
                seq, 3 * VIS_D, VIS_D, stream=stream)
        # add_bias_bf16 is a proper in-place add (no zero-buffer bandwidth waste)
        fvk.add_bias_bf16(
            B["vision_QKV"].ptr.value, W["vision_attn_qkv_b"][i],
            seq, 3 * VIS_D, stream=stream)

        # Split QKV into attn_backend's Q/K/V buffers
        fvk.qkv_split(
            B["vision_QKV"].ptr.value,
            attn_ptrs["vis_Q"], attn_ptrs["vis_K"], attn_ptrs["vis_V"],
            seq, VIS_D, VIS_D, VIS_D, stream=stream)

        # Self-attention (per-view batched)
        vis_o_ptr = self.attn.run(
            "siglip", i, q_seq=VIS_SEQ_PER_VIEW, stream=stream)

        # Attn output projection → x_norm
        if use_fp8:
            self._fp8_gemm(
                vis_o_ptr, seq * VIS_D,
                f"vision_attn_o_w_{i}",
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                vis_o_ptr, W["vision_attn_o_w"][i],
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_D, stream=stream)
        # x += x_norm + o_bias; x_norm = LayerNorm(x).
        # The un-fused upstream pair keeps numerics matching the RTX
        # default BF16/FP8 path bit-for-bit.
        fvk.bias_residual(
            B["vision_x"].ptr.value, B["vision_x_norm"].ptr.value,
            W["vision_attn_o_b"][i], seq, VIS_D, stream=stream)
        fvk.layer_norm(
            B["vision_x"].ptr.value,
            W["vision_pre_ffn_norm_w"][i], W["vision_pre_ffn_norm_b"][i],
            B["vision_x_norm"].ptr.value,
            seq, VIS_D, 1e-5, stream=stream)

        # FFN up → hidden, + bias, + GELU
        if use_fp8:
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, seq * VIS_D,
                f"vision_ffn_up_w_{i}",
                B["vision_hidden"].ptr.value,
                seq, VIS_H, VIS_D, stream)
        else:
            gemm.bf16_nn(
                B["vision_x_norm"].ptr.value, W["vision_ffn_up_w"][i],
                B["vision_hidden"].ptr.value,
                seq, VIS_H, VIS_D, stream=stream)
        # bias + GELU on the FFN-hidden buffer (un-fused upstream pair).
        self._bias_add_bf16(
            B["vision_hidden"].ptr.value, W["vision_ffn_up_b"][i],
            seq, VIS_H, stream)
        fvk.gelu_inplace(
            B["vision_hidden"].ptr.value, seq * VIS_H, stream=stream)

        # FFN down → x_norm, then x += x_norm + down_bias
        if use_fp8:
            self._fp8_gemm(
                B["vision_hidden"].ptr.value, seq * VIS_H,
                f"vision_ffn_down_w_{i}",
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_H, stream)
        else:
            gemm.bf16_nn(
                B["vision_hidden"].ptr.value, W["vision_ffn_down_w"][i],
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, VIS_H, stream=stream)
        if is_last:
            # Last layer: just bias_residual; vision_x is the final residual,
            # consumed by avg_pool_vision_tokens / multi-modal projector.
            # No follow-up LN to run.
            fvk.bias_residual(
                B["vision_x"].ptr.value, B["vision_x_norm"].ptr.value,
                W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
        else:
            # Layers 0..N-2: post-FFN bias_residual + NEXT layer's pre-attn
            # LayerNorm (un-fused upstream pair). vision_x_norm ends up
            # holding LN(vision_x post-residual) ready for layer i+1's
            # attn QKV.
            fvk.bias_residual(
                B["vision_x"].ptr.value, B["vision_x_norm"].ptr.value,
                W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
            fvk.layer_norm(
                B["vision_x"].ptr.value,
                W["vision_pre_attn_norm_w"][i + 1],
                W["vision_pre_attn_norm_b"][i + 1],
                B["vision_x_norm"].ptr.value,
                seq, VIS_D, 1e-5, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase B: Gemma-2B encoder
    # ══════════════════════════════════════════════════════════════════

    def transformer_encoder(self, stream: int = 0) -> None:
        """Project vision output + language tokens through the Gemma-2B encoder.

        The language embeddings have already been copied into
        ``bufs['encoder_x'][nv*256 : nv*256+prompt_len]`` by :meth:`forward`.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.encoder_seq_len
        # vs_enc: token count fed to the encoder (post-pool); equals vision_seq when no pooling
        vs_enc = self.vision_seq_enc
        use_fp8 = self.use_fp8

        # B0: LayerNorm(vision output) → project 1152→2048 + bias → encoder_x[:vs_enc]
        # When vision_pool_factor > 1, read from vision_x_pooled (reduced token count);
        # otherwise vision_x_pooled aliases vision_x so behaviour is unchanged.
        fvk.layer_norm(
            B["vision_x_pooled"].ptr.value,
            W["vision_final_norm_w"], W["vision_final_norm_b"],
            B["vision_x_norm"].ptr.value,
            vs_enc, VIS_D, 1e-5, stream=stream)

        if use_fp8 and "vision_projector_w" in self.weights.get("fp8", {}):
            self._fp8_gemm(
                B["vision_x_norm"].ptr.value, vs_enc * VIS_D,
                "vision_projector_w",
                B["encoder_x"].ptr.value,
                vs_enc, ENC_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                B["vision_x_norm"].ptr.value, W["encoder_multi_modal_projector_w"],
                B["encoder_x"].ptr.value,
                vs_enc, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_bf16(
            B["encoder_x"].ptr.value, W["encoder_multi_modal_projector_b"],
            vs_enc, ENC_D, stream=stream)

        # Language embeds have been written by frontend into encoder_x[vs_enc:vs_enc+lang_len]

        # B1-B5: 18 encoder layers. Fuse previous B5 residual into this B1's RMS→FP8
        fused = use_fp8 and self.fp8_calibrated
        for i in range(ENC_L):
            self._encoder_layer(i, seq, fuse_b1=(i > 0 and fused), stream=stream)

    def _encoder_layer(self, i: int, seq: int, fuse_b1: bool, stream: int) -> None:
        """One Gemma-2B encoder layer."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs
        fused = self.use_fp8 and self.fp8_calibrated

        # B1: RMSNorm → QKV GEMM
        if fused:
            qkv_name = f"encoder_attn_qkv_w_{i}"
            act_scale_ptr = self.fp8_act_scales[qkv_name].ptr.value
            if fuse_b1:
                # Fused: previous layer B5 residual + this layer's RMSNorm → FP8
                fvk.residual_add_rms_norm_fp8(
                    B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                    self._rms_ones_enc.ptr.value,
                    B["enc_act_fp8"].ptr.value,
                    seq, ENC_D, 1e-6, act_scale_ptr, stream=stream)
            else:
                fvk.rms_norm_fp8(
                    B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                    B["enc_act_fp8"].ptr.value,
                    seq, ENC_D, 1e-6, act_scale_ptr, stream=stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8"].ptr.value, qkv_name,
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
                act_scale_ptr, stream)
        elif self.use_fp8:
            # Pre-calibration FP8 path: separate RMSNorm + dynamic quant inside _fp8_gemm
            fvk.rms_norm(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["encoder_x_norm"].ptr.value, seq * ENC_D,
                f"encoder_attn_qkv_w_{i}",
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream)
        else:
            fvk.rms_norm(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            gemm.bf16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_attn_qkv_w"][i],
                B["encoder_QKV"].ptr.value,
                seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)

        # Split QKV + apply RoPE. K/V go into attn_backend's layer cache slice.
        k_ptr, v_ptr = self._enc_kv_layer_ptrs(i, offset_tokens=0)
        fvk.qkv_split_rope(
            B["encoder_QKV"].ptr.value,
            B["encoder_rope_weights"].ptr.value,
            attn_ptrs["enc_Q"],  # (seq, 8, 256) — split from QKV into attn buf
            k_ptr, v_ptr,
            seq, ENC_NH * ENC_HD, ENC_NKV * ENC_HD, ENC_NKV * ENC_HD,
            ENC_HD, stream=stream)

        if i == ENC_L - 1:
            # Last layer: no post-attn projection/FFN needed — encoder output is
            # the K/V cache which the decoder reads. Skip to next phase.
            return

        # B2: Attention (GQA) — returns output pointer (no copy).
        enc_o_ptr = self.attn.run("encoder", i, q_seq=seq, stream=stream)

        # B3: Attn output projection → x_norm
        if self.use_fp8:
            self._fp8_gemm(
                enc_o_ptr, seq * ENC_D,
                f"encoder_attn_o_w_{i}",
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_D, stream)
        else:
            gemm.bf16_nn(
                enc_o_ptr, W["encoder_attn_o_w"][i],
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_D, stream=stream)

        # B4: (residual_add + RMSNorm) → FFN gate_up GEMM
        if fused:
            # Fused: residual + RMS → FP8 in one kernel, then FP8 GEMM
            gu_name = f"encoder_ffn_gate_up_w_{i}"
            act_scale_gu = self.fp8_act_scales[gu_name].ptr.value
            fvk.residual_add_rms_norm_fp8(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                self._rms_ones_enc.ptr.value,
                B["enc_act_fp8"].ptr.value,
                seq, ENC_D, 1e-6, act_scale_gu, stream=stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8"].ptr.value, gu_name,
                B["encoder_gate_merged"].ptr.value,
                seq, 2 * ENC_H, ENC_D, act_scale_gu, stream)
        elif self.use_fp8:
            # Pre-calibration: separate residual + rms + fp8_gemm
            fvk.residual_add(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)
            fvk.rms_norm(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm(
                B["encoder_x_norm"].ptr.value, seq * ENC_D,
                f"encoder_ffn_gate_up_w_{i}",
                B["encoder_gate_merged"].ptr.value,
                seq, 2 * ENC_H, ENC_D, stream)
        else:
            fvk.residual_add(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)
            fvk.rms_norm(
                B["encoder_x"].ptr.value, self._rms_ones_enc.ptr.value,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, 1e-6, stream=stream)
            # Non-FP8 path: gate+up are separate 2 GEMMs into gate and hidden
            gemm.bf16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_ffn_gate_w"][i],
                B["encoder_gate_merged"].ptr.value,
                seq, ENC_H, ENC_D, stream=stream)
            gemm.bf16_nn(
                B["encoder_x_norm"].ptr.value, W["encoder_ffn_up_w"][i],
                B["encoder_hidden"].ptr.value,
                seq, ENC_H, ENC_D, stream=stream)

        # SiLU(gate) * up → hidden, then FFN down GEMM.
        if fused:
            down_name = f"encoder_ffn_down_w_{i}"
            act_scale_down = self.fp8_act_scales[down_name].ptr.value
            fvk.gate_geglu_merged_fp8(
                B["encoder_gate_merged"].ptr.value,
                B["enc_act_fp8_large"].ptr.value,
                seq, ENC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                B["enc_act_fp8_large"].ptr.value, down_name,
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, act_scale_down, stream)
        elif self.use_fp8:
            fvk.gate_geglu_merged(
                B["encoder_gate_merged"].ptr.value,
                B["encoder_hidden"].ptr.value,
                seq, ENC_H, stream=stream)
            self._fp8_gemm(
                B["encoder_hidden"].ptr.value, seq * ENC_H,
                f"encoder_ffn_down_w_{i}",
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, stream)
        else:
            fvk.gate_geglu(
                B["encoder_gate_merged"].ptr.value,
                B["encoder_hidden"].ptr.value,
                B["encoder_hidden"].ptr.value,
                seq * ENC_H, stream=stream)
            gemm.bf16_nn(
                B["encoder_hidden"].ptr.value, W["encoder_ffn_down_w"][i],
                B["encoder_x_norm"].ptr.value,
                seq, ENC_D, ENC_H, stream=stream)

        # B5: Residual (skipped in fused mode — next layer's B1 handles it)
        if not fused:
            fvk.residual_add(
                B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value,
                seq * ENC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase C: Gemma-300M decoder (flow matching)
    # ══════════════════════════════════════════════════════════════════

    def _copy_rtc_prefix(self, prefix_len: int, stream: int = 0) -> None:
        if prefix_len <= 0:
            return
        if prefix_len > self.chunk_size:
            raise ValueError(
                f"rtc prefix_len {prefix_len} exceeds chunk_size "
                f"{self.chunk_size}")
        nbytes = int(prefix_len) * ACTION_DIM * 2
        self.fvk.gpu_copy(
            self.bufs["diffusion_noise"].ptr.value,
            self.bufs["rtc_prev_action_chunk"].ptr.value,
            nbytes,
            stream,
        )

    def transformer_decoder(self, stream: int = 0,
                            rtc_prefix_len: int = 0) -> None:
        """Run 10-step diffusion denoise on ``bufs['diffusion_noise']``."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        enc_seq = self.encoder_seq_len
        ds = self.chunk_size
        fused = self.use_fp8_decoder and self.fp8_calibrated

        prev_decoder_step = getattr(self, "_fp8_current_decoder_step", -1)
        try:
            for step in range(self.num_steps):
                self._fp8_current_decoder_step = step
                if self._dec_attn_fp8out_armed:
                    # Host-side scale-row selector: the fp8out epilogue must
                    # use the SAME per-(step, layer) o-proj static scale ptr
                    # the quantize path uses; frozen into the graph at
                    # capture like every other per-launch pointer.
                    self.attn.set_decoder_fp8out_step(step)
                self._copy_rtc_prefix(rtc_prefix_len, stream)
                # C0: Action input projection: noise (ds, 32) → decoder_x (ds, 1024)
                gemm.bf16_nn(
                    B["diffusion_noise"].ptr.value,
                    W["decoder_action_in_proj_w"],
                    B["decoder_x"].ptr.value,
                    ds, DEC_D, ACTION_DIM, stream=stream)
                self._bias_add_bf16(
                    B["decoder_x"].ptr.value, W["decoder_action_in_proj_b"],
                    ds, DEC_D, stream)

                # 18 decoder layers
                for i in range(DEC_L):
                    skip_c1 = fused and i > 0
                    self._decoder_layer(i, step, enc_seq, ds, skip_c1, stream)

                # C8: Final AdaRMSNorm + output projection
                fvk.ada_rms_norm_style(
                    B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                    self._style_slice_ptr("decoder_style_final", step),
                    B["x_normed_buf"].ptr.value, B["gate_buf"].ptr.value,
                    ds, DEC_D, 1e-6, stream=stream)
                gemm.bf16_nn(
                    B["x_normed_buf"].ptr.value,
                    W["decoder_action_out_proj_w"],
                    B["decoder_action_buf"].ptr.value,
                    ds, ACTION_DIM, DEC_D, stream=stream)
                self._bias_add_bf16(
                    B["decoder_action_buf"].ptr.value,
                    W["decoder_action_out_proj_b"],
                    ds, ACTION_DIM, stream)
                # noise += action_buf (weights pre-scaled by -1/num_steps by frontend)
                fvk.residual_add(
                    B["diffusion_noise"].ptr.value,
                    B["decoder_action_buf"].ptr.value,
                    ds * ACTION_DIM, stream=stream)
                self._copy_rtc_prefix(rtc_prefix_len, stream)
        finally:
            self._fp8_current_decoder_step = prev_decoder_step

    def _decoder_layer(self, i: int, step: int, enc_seq: int, ds: int,
                       skip_c1: bool, stream: int) -> None:
        """One Gemma-300M decoder layer."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        attn_ptrs = self._attn_ptrs
        fused = self.use_fp8_decoder and self.fp8_calibrated

        # C1: AdaRMSNorm with style modulation → FP8 (fused) or BF16
        qkv_name = f"decoder_attn_qkv_w_{i}"
        if fused:
            act_scale_qkv = self._fp8_static_scale_ptr(qkv_name)
            if not skip_c1:
                fvk.ada_rms_norm_style_fp8(
                    B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                    self._style_slice_ptr("decoder_style_attn", step, i),
                    B["dec_act_fp8"].ptr.value, B["gate_buf"].ptr.value,
                    ds, DEC_D, 1e-6, act_scale_qkv, stream=stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8"].ptr.value, qkv_name,
                B["decoder_QKV"].ptr.value,
                ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D,
                act_scale_qkv, stream)
        else:
            fvk.ada_rms_norm_style(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_attn", step, i),
                B["x_normed_buf"].ptr.value, B["gate_buf"].ptr.value,
                ds, DEC_D, 1e-6, stream=stream)
            if self.use_fp8_decoder:
                self._fp8_gemm(
                    B["x_normed_buf"].ptr.value, ds * DEC_D,
                    qkv_name,
                    B["decoder_QKV"].ptr.value,
                    ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream)
            else:
                gemm.bf16_nn(
                    B["x_normed_buf"].ptr.value, W["decoder_attn_qkv_w"][i],
                    B["decoder_QKV"].ptr.value,
                    ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)

        # C2: QKV split + RoPE. Decoder K/V write into enc cache after prefix.
        if self._fixed_shape:
            # Fixed graph: append the action K/V right after the VALID prefix
            # (runtime row offset = devpos), so cross-attn over [0:valid+chunk]
            # is one contiguous (seqused) call. K/V pointers are cache base.
            k_base, v_base = self._enc_kv_layer_ptrs(i, offset_tokens=0)
            fvk.qkv_split_rope_devpos(
                B["decoder_QKV"].ptr.value,
                B["decoder_rope_weights"].ptr.value,
                attn_ptrs["dec_Q"],
                k_base, v_base, self.attn.dec_devpos.data_ptr(),
                ds, DEC_NH * DEC_HD, DEC_NKV * DEC_HD, DEC_NKV * DEC_HD,
                DEC_HD, stream=stream)
        else:
            k_ptr, v_ptr = self._enc_kv_layer_ptrs(i, offset_tokens=enc_seq)
            fvk.qkv_split_rope(
                B["decoder_QKV"].ptr.value,
                B["decoder_rope_weights"].ptr.value,
                attn_ptrs["dec_Q"],
                k_ptr, v_ptr,
                ds, DEC_NH * DEC_HD, DEC_NKV * DEC_HD, DEC_NKV * DEC_HD,
                DEC_HD, stream=stream)

        # C3: Cross-attention (decoder Q over enc+dec K/V cache) — returns ptr.
        dec_o_ptr = self.attn.run(
            "decoder", i,
            q_seq=ds,
            kv_seq=enc_seq + ds,
            stream=stream,
        )

        # C4: Attn output projection.
        # When the fused decoder attn->FP8 epilogue is armed (see
        # _arm_decoder_attn_fp8out), the attention kernel already emitted
        # the o-proj activation's fp8 bytes with the SAME static scale
        # pointer _fp8_gemm would use — consume them directly and skip the
        # standalone quantize launch (~180/inference). Bit-equal numerics.
        o_name = f"decoder_attn_o_w_{i}"
        fp8out = None
        if fused and self._dec_attn_fp8out_armed:
            get_fp8out = getattr(self.attn, "get_decoder_fp8out_ptr", None)
            fp8out = get_fp8out() if get_fp8out is not None else None
        if fp8out is not None:
            fp8_ptr, act_scale_ptr = fp8out
            # The epilogue and the quantize path must share one scale
            # pointer, or the fp8 bytes would differ from today's path.
            assert act_scale_ptr == self._fp8_static_scale_ptr(o_name), (
                f"decoder fp8out scale ptr mismatch at layer {i} step "
                f"{self._fp8_current_decoder_step}")
            self._fp8_gemm_fused(
                fp8_ptr, o_name,
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_NH * DEC_HD, act_scale_ptr, stream)
        elif self.use_fp8_decoder:
            self._fp8_gemm(
                dec_o_ptr, ds * DEC_NH * DEC_HD,
                o_name,
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_NH * DEC_HD, stream)
        else:
            gemm.bf16_nn(
                dec_o_ptr, W["decoder_attn_o_w"][i],
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_NH * DEC_HD, stream=stream)

        # C4→C5: gate*residual + AdaRMSNorm + FFN gate_up
        gu_name = f"decoder_ffn_gate_up_w_{i}"    # FP8 merged name
        if fused:
            act_scale_gu = self._fp8_static_scale_ptr(gu_name)
            fvk.gate_residual_ada_norm_fp8(
                B["decoder_x"].ptr.value, B["x_normed_buf"].ptr.value,
                B["gate_buf"].ptr.value,
                self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_ffn", step, i),
                B["dec_act_fp8"].ptr.value, B["gate_buf"].ptr.value,
                ds, DEC_D, 1e-6, act_scale_gu, stream=stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8"].ptr.value, gu_name,
                B["decoder_gate_merged"].ptr.value,
                ds, 2 * DEC_H, DEC_D, act_scale_gu, stream)
        else:
            fvk.gate_mul_residual(
                B["decoder_x"].ptr.value, B["x_normed_buf"].ptr.value,
                B["gate_buf"].ptr.value, ds * DEC_D, stream=stream)
            fvk.ada_rms_norm_style(
                B["decoder_x"].ptr.value, self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_ffn", step, i),
                B["x_normed_buf"].ptr.value, B["gate_buf"].ptr.value,
                ds, DEC_D, 1e-6, stream=stream)
            if self.use_fp8_decoder:
                self._fp8_gemm(
                    B["x_normed_buf"].ptr.value, ds * DEC_D,
                    gu_name,
                    B["decoder_gate_merged"].ptr.value,
                    ds, 2 * DEC_H, DEC_D, stream)
            else:
                gemm.bf16_nn(
                    B["x_normed_buf"].ptr.value, W["decoder_ffn_gate_w"][i],
                    B["decoder_gate_merged"].ptr.value,
                    ds, DEC_H, DEC_D, stream=stream)
                gemm.bf16_nn(
                    B["x_normed_buf"].ptr.value, W["decoder_ffn_up_w"][i],
                    B["decoder_hidden"].ptr.value,
                    ds, DEC_H, DEC_D, stream=stream)

        # C6: SiLU(gate) * up → FFN down
        down_name = f"decoder_ffn_down_w_{i}"
        if fused:
            act_scale_down = self._fp8_static_scale_ptr(down_name)
            fvk.gate_geglu_merged_fp8(
                B["decoder_gate_merged"].ptr.value,
                B["dec_act_fp8_large"].ptr.value,
                ds, DEC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                B["dec_act_fp8_large"].ptr.value, down_name,
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_H, act_scale_down, stream)
        elif self.use_fp8_decoder:
            fvk.gate_geglu_merged(
                B["decoder_gate_merged"].ptr.value,
                B["decoder_hidden"].ptr.value,
                ds, DEC_H, stream=stream)
            self._fp8_gemm(
                B["decoder_hidden"].ptr.value, ds * DEC_H,
                down_name,
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_H, stream)
        else:
            fvk.gate_geglu(
                B["decoder_gate_merged"].ptr.value,
                B["decoder_hidden"].ptr.value,
                B["decoder_hidden"].ptr.value,
                ds * DEC_H, stream=stream)
            gemm.bf16_nn(
                B["decoder_hidden"].ptr.value, W["decoder_ffn_down_w"][i],
                B["x_normed_buf"].ptr.value,
                ds, DEC_D, DEC_H, stream=stream)

        # C7→C1_next: gate*residual + next layer's AdaRMSNorm → FP8 (fused)
        if fused and i < DEC_L - 1:
            next_qkv = f"decoder_attn_qkv_w_{i + 1}"
            act_scale_next = self._fp8_static_scale_ptr(next_qkv)
            fvk.gate_residual_ada_norm_fp8(
                B["decoder_x"].ptr.value, B["x_normed_buf"].ptr.value,
                B["gate_buf"].ptr.value,
                self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_attn", step, i + 1),
                B["dec_act_fp8"].ptr.value, B["gate_buf"].ptr.value,
                ds, DEC_D, 1e-6, act_scale_next, stream=stream)
        else:
            fvk.gate_mul_residual(
                B["decoder_x"].ptr.value, B["x_normed_buf"].ptr.value,
                B["gate_buf"].ptr.value, ds * DEC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Full pipeline + calibration + graph
    # ══════════════════════════════════════════════════════════════════

    def run_pipeline(self, stream: int = 0) -> None:
        """Run vision → encoder → decoder end-to-end on ``stream``.

        Re-copies the stored language embeddings into ``encoder_x`` at the
        prompt slot because the encoder layers overwrite that region in
        place (residual stream). Without this, the second call on the
        same pipeline would feed the encoder its own previous output
        instead of fresh language tokens.
        """
        self._copy_lang_embeds_to_encoder_x(stream=stream)
        # Every pre-calibration FP8 forward IS a calibration pass (dynamic
        # quant writes the amax scales as a side effect), so the
        # deterministic-attention window tracks exactly that predicate:
        # nondeterministic library attention (aiter) would otherwise turn
        # into permanent run-to-run scale jitter. Post-calibration forwards
        # (warmup, capture) run the fast library path with frozen scales.
        self.attn.set_calibrating(self.use_fp8 and not self.fp8_calibrated)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)

    def calibrate_fp8(self) -> None:
        """Run one forward pass with dynamic quantization to collect FP8 scales.

        Dynamic quant writes scales directly into each layer's persistent
        scale buffer (via ``_fp8_gemm``), so flipping ``fp8_calibrated =
        True`` afterwards enables the static-scale fused kernels with zero
        further work.
        """
        if not self.use_fp8 or self.fp8_calibrated:
            return

        if len(self.fp8_act_scales) > 0:
            self.fp8_calibrated = True
            logger.info("FP8 reuse-from-forward: %d scales already populated",
                        len(self.fp8_act_scales))
            return

        # Dynamic calibration pass (writes scales into fp8_act_scales as a
        # side effect of _fp8_gemm's non-calibrated code path).
        self.fp8_calibrated = False
        self.run_pipeline(stream=0)
        self._hip_check(self._hip.hipDeviceSynchronize(),
                        "hipDeviceSynchronize")
        self.fp8_calibrated = True
        logger.info("FP8 calibrated: %d activation scales collected",
                    len(self.fp8_act_scales))

    def _arm_decoder_attn_fp8out(self) -> None:
        """Arm the fused decoder attention -> FP8 epilogue (capture prep).

        Post-calibration, the decoder attn-output -> o-proj site quantizes
        the attention output with the layer's (per-step) STATIC scale
        before the FP8 GEMM. The custom decoder attention kernel can emit
        those exact fp8 bytes as an in-kernel epilogue instead
        (attention_decoder_gqa_fp8out), removing ~180 standalone
        quantize_fp8_static launches per inference with bit-equal numerics
        (same scale pointers, same conversion math).

        Called once from :meth:`record_infer_graph` after
        ``calibrate_fp8()`` — the point where the static scales are final.
        Arms the backend with the per-(step, layer) o-proj static-scale
        device pointers; :meth:`transformer_decoder` selects the step row
        (host-side) so each captured attention launch freezes the same
        scale pointer the o-proj quantize would use.

        Env gate: ``FVK_AMD_ATTN_FP8OUT=0`` disables arming entirely, in
        which case pipeline behavior is IDENTICAL to the pre-fusion path
        (split attention + standalone quantize). Default is ON
        post-calibration.
        """
        if self._dec_attn_fp8out_armed:
            return
        if not (self.use_fp8 and self.use_fp8_decoder and self.fp8_calibrated):
            return
        if os.environ.get("FVK_AMD_ATTN_FP8OUT", "1").strip().lower() in (
                "0", "off", "false"):
            logger.info("Decoder attn FP8-out epilogue disabled "
                        "(FVK_AMD_ATTN_FP8OUT=0)")
            return
        supported = getattr(self.attn, "decoder_fp8out_supported", None)
        if supported is None or not supported():
            return
        scale_ptrs = []
        for step in range(self.num_steps):
            row = []
            for i in range(DEC_L):
                name = f"decoder_attn_o_w_{i}"
                buf = (self.fp8_act_scales.get(f"{name}__step_{step}")
                       or self.fp8_act_scales.get(name))
                if buf is None:
                    logger.warning(
                        "Decoder attn FP8-out: scale for %s missing — "
                        "leaving epilogue disarmed", name)
                    return
                row.append(buf.ptr.value)
            scale_ptrs.append(row)
        self.attn.set_decoder_fp8out(scale_ptrs)
        self._dec_attn_fp8out_armed = True
        logger.info(
            "Decoder attn FP8-out epilogue armed: %d steps x %d layers "
            "(standalone o-proj quantize launches eliminated)",
            self.num_steps, DEC_L)

    def autotune_gemms(self) -> None:
        """Benchmark GEMM algorithms for each shape and cache the best.

        Call after ``calibrate_fp8()`` and before ``record_infer_graph()``.
        Idempotent: a pipeline's GEMM shapes are fixed, so once tuned, repeat
        calls return immediately (the frontend tunes before capture and
        record_infer_graph tunes again — without this guard that ran twice).
        """
        if self._gemms_autotuned:
            return
        B = self.bufs
        W = self.weights
        gemm = self.gemm
        vs = self.vision_seq
        seq = self.encoder_seq_len
        ds = self.chunk_size

        logger.info("Autotuning GEMM algorithms...")

        # Vision patch embedding (BF16)
        gemm.autotune_bf16_nn(
            B["vision_patches"].ptr.value,
            W["vision_patch_embedding_w"],
            B["vision_x"].ptr.value,
            vs, VIS_D, VIS_PATCH_FLAT)

        # Vision BF16 attention + FFN shapes (when FP8 is disabled).
        # Autotuning picks the best algorithm for each shape; all 27
        # layers share these 5 shapes so one autotune run covers the
        # entire SigLIP stack. Output buffers must be large enough for
        # each shape's output.
        if not self.use_fp8:
            for M_val, N_val, K_val, act_key, out_key, weight_ptr in [
                (vs, 3 * VIS_D, VIS_D, "vision_x_norm", "vision_QKV",
                 W["vision_attn_qkv_w"][0]),
                (vs, VIS_D,     VIS_D, "vision_x_norm", "vision_x_norm",
                 W["vision_attn_o_w"][0]),
                (vs, VIS_H,     VIS_D, "vision_x_norm", "vision_hidden",
                 W["vision_ffn_up_w"][0]),
                (vs, VIS_D,     VIS_H, "vision_hidden",  "vision_x_norm",
                 W["vision_ffn_down_w"][0]),
                (self.vision_seq_enc, ENC_D, VIS_D, "vision_x_norm", "encoder_x",
                 W["encoder_multi_modal_projector_w"]),
            ]:
                gemm.autotune_bf16_nn(
                    B[act_key].ptr.value, weight_ptr,
                    B[out_key].ptr.value,
                    M_val, N_val, K_val)

        # Vision FP8 shapes
        if self.use_fp8 and self.fp8_calibrated and "vision_attn_qkv_w_0" in self.weights.get("fp8", {}):
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("vision_attn_qkv_w_0", vs, 3 * VIS_D, VIS_D, "vision_QKV"),
                ("vision_attn_o_w_0",   vs, VIS_D,     VIS_D, "vision_x_norm"),
                ("vision_ffn_up_w_0",   vs, VIS_H,     VIS_D, "vision_hidden"),
                ("vision_ffn_down_w_0", vs, VIS_D,     VIS_H, "vision_x_norm"),
                ("vision_projector_w",  vs, ENC_D,     VIS_D, "encoder_x"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf_ptr, _ = self._pick_fp8_scratch(name_prefix, M_val * K_val)
                self._autotune_fp8_matmul(
                    act_buf_ptr, w_fp8_ptr, B[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)

        # Encoder FP8 shapes
        if self.use_fp8 and self.fp8_calibrated:
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("encoder_attn_qkv_w_0",    seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, "encoder_QKV"),
                ("encoder_attn_o_w_0",      seq, ENC_D,      ENC_D, "encoder_x_norm"),
                ("encoder_ffn_gate_up_w_0", seq, 2 * ENC_H,  ENC_D, "encoder_gate_merged"),
                ("encoder_ffn_down_w_0",    seq, ENC_D,      ENC_H, "encoder_x_norm"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf_ptr, _ = self._pick_fp8_scratch(name_prefix, M_val * K_val)
                self._autotune_fp8_matmul(
                    act_buf_ptr, w_fp8_ptr, B[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)
        # Plain encoder GEMMs fall back to BF16 when FP8 does not own the
        # encoder matmuls.
        elif not self.use_fp8:
            for M_val, N_val, K_val, act_key, out_key, weight_ptr in [
                (seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
                 "encoder_x_norm", "encoder_QKV",
                 W["encoder_attn_qkv_w"][0]),
                (seq, ENC_D, ENC_D,
                 "encoder_x_norm", "encoder_x_norm",
                 W["encoder_attn_o_w"][0]),
                (seq, ENC_H, ENC_D,
                 "encoder_x_norm", "encoder_gate_merged",
                 W["encoder_ffn_gate_w"][0]),
                (seq, ENC_H, ENC_D,
                 "encoder_x_norm", "encoder_hidden",
                 W["encoder_ffn_up_w"][0]),
                (seq, ENC_D, ENC_H,
                 "encoder_hidden", "encoder_x_norm",
                 W["encoder_ffn_down_w"][0]),
            ]:
                gemm.autotune_bf16_nn(
                    B[act_key].ptr.value, weight_ptr,
                    B[out_key].ptr.value,
                    M_val, N_val, K_val)

        # Decoder FP8 shapes
        if self.use_fp8 and self.use_fp8_decoder and self.fp8_calibrated:
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("decoder_attn_qkv_w_0",    ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, "decoder_QKV"),
                ("decoder_attn_o_w_0",      ds, DEC_D,     DEC_NH * DEC_HD, "x_normed_buf"),
                ("decoder_ffn_gate_up_w_0", ds, 2 * DEC_H, DEC_D, "decoder_gate_merged"),
                ("decoder_ffn_down_w_0",    ds, DEC_D,     DEC_H, "x_normed_buf"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf_ptr, _ = self._pick_fp8_scratch(name_prefix, M_val * K_val)
                self._autotune_fp8_matmul(
                    act_buf_ptr, w_fp8_ptr, B[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)
        # Plain decoder GEMMs fall back to BF16 when FP8 does not own the
        # decoder matmuls.
        elif not self.use_fp8_decoder:
            # BF16 decoder shapes dominate baseline runtime. Tune them
            # explicitly before graph capture instead of relying on the
            # heuristic top-1 selected at first use.
            decoder_shapes = [
                (ds, DEC_D, ACTION_DIM,
                 "diffusion_noise", "decoder_x",
                 W["decoder_action_in_proj_w"]),
                (ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D,
                 "x_normed_buf", "decoder_QKV",
                 W["decoder_attn_qkv_w"][0]),
                (ds, DEC_D, DEC_NH * DEC_HD,
                 "decoder_QKV", "x_normed_buf",
                 W["decoder_attn_o_w"][0]),
                (ds, DEC_H, DEC_D,
                 "x_normed_buf", "decoder_gate_merged",
                 W["decoder_ffn_gate_w"][0]),
                (ds, DEC_H, DEC_D,
                 "x_normed_buf", "decoder_hidden",
                 W["decoder_ffn_up_w"][0]),
                (ds, DEC_D, DEC_H,
                 "decoder_hidden", "x_normed_buf",
                 W["decoder_ffn_down_w"][0]),
                (ds, ACTION_DIM, DEC_D,
                 "x_normed_buf", "decoder_action_buf",
                 W["decoder_action_out_proj_w"]),
            ]
            for M_val, N_val, K_val, act_key, out_key, weight_ptr in decoder_shapes:
                gemm.autotune_bf16_nn(
                    B[act_key].ptr.value, weight_ptr,
                    B[out_key].ptr.value,
                    M_val, N_val, K_val)

        self._hip_check(self._hip.hipDeviceSynchronize(),
                        "hipDeviceSynchronize")
        self._gemms_autotuned = True
        logger.info("Autotune complete")

    def record_infer_graph(self, external_stream_int: int | None = None) -> None:
        """Capture the full pipeline as a HIP Graph.

        Because the attention backend may run framework-specific kernels
        (torch sdpa on the interim CDNA4 backend), HIP graph capture must
        use a stream that the backend's framework treats as "current" —
        otherwise the backend's kernels will launch on the framework's
        default stream, creating a cross-stream dependency that
        ``hipStreamBeginCapture`` refuses to record.

        Frontends that use a framework-aware attention backend should pass
        ``external_stream_int`` (the raw ``hipStream_t`` int handle of a
        framework-owned stream) and wrap the call in their framework's
        "current stream" context. For example, the torch frontend does::

            torch_stream = torch.cuda.Stream()
            with torch.cuda.stream(torch_stream):
                pipeline.record_infer_graph(
                    external_stream_int=torch_stream.cuda_stream)

        If ``external_stream_int`` is ``None``, a raw HIP stream is created
        and used directly (this works only with pure-C attention backends).
        """
        if self.use_fp8 and not self.fp8_calibrated:
            self.calibrate_fp8()
        self.autotune_gemms()
        # Scales are final now — arm the fused decoder attn->FP8 epilogue
        # so warmup and capture both take the fused route (see helper for
        # the FVK_AMD_ATTN_FP8OUT env gate).
        self._arm_decoder_attn_fp8out()

        self._graph = HipGraph()
        if external_stream_int is None:
            stream = self._graph.create_stream()
            stream_int = stream.value or 0
            stream_handle = stream
        else:
            stream_int = int(external_stream_int)
            stream_handle = ctypes.c_void_p(stream_int)
        self._graph_stream = stream_handle

        # Warmup on the capture stream to stabilize allocations
        for _ in range(3):
            self.run_pipeline(stream=stream_int)
        self._hip_check(self._hip.hipStreamSynchronize(stream_handle),
                        "hipStreamSynchronize")

        # Capture full pipeline
        self._graph.begin_capture(stream_handle)
        self.run_pipeline(stream=stream_int)
        self._graph.end_capture(stream_handle)
        self._hip_check(self._hip.hipStreamSynchronize(stream_handle),
                        "hipStreamSynchronize")
        logger.info("HIP Graph captured for Pi05Pipeline")

        # Also capture a decoder-only graph for temporal K/V caching.
        # This graph skips vision_encoder + transformer_encoder and runs only
        # transformer_decoder, reusing the K/V cache from the last full forward.
        # The language embeds and encoder K/V buffers are left intact.
        self._decoder_only_graph = HipGraph()
        for _ in range(3):
            self.transformer_decoder(stream=stream_int)
        self._hip_check(self._hip.hipStreamSynchronize(stream_handle),
                        "hipStreamSynchronize")
        self._decoder_only_graph.begin_capture(stream_handle)
        self.transformer_decoder(stream=stream_int)
        self._decoder_only_graph.end_capture(stream_handle)
        self._hip_check(self._hip.hipStreamSynchronize(stream_handle),
                        "hipStreamSynchronize")
        logger.info("HIP Graph captured for Pi05Pipeline (decoder-only)")

    # ══════════════════════════════════════════════════════════════════
    #   Public API
    # ══════════════════════════════════════════════════════════════════

    # ── Input buffer handles (frontend writes to these) ──

    @property
    def input_images_buf(self) -> HipBuffer:
        """Pipeline input: observation images (num_views, 224, 224, 3) bf16."""
        return self.bufs["observation_images_normalized"]

    @property
    def input_noise_buf(self) -> HipBuffer:
        """Pipeline input/output: diffusion noise (chunk_size, 32) bf16.

        Frontend writes initial noise here before :meth:`forward`; reads
        final actions from here after.
        """
        return self.bufs["diffusion_noise"]

    @property
    def input_rtc_prev_action_chunk_buf(self) -> HipBuffer:
        """Optional RTC-prefix input: previous raw action chunk (chunk, 32)."""
        return self.bufs["rtc_prev_action_chunk"]

    @property
    def input_rtc_prefix_weights_buf(self) -> HipBuffer:
        """Optional RTC VJP input: per-action prefix weights (chunk,)."""
        return self.bufs["rtc_prefix_weights"]

    @property
    def input_rtc_guidance_weight_buf(self) -> HipBuffer:
        """Optional RTC VJP input: scalar maximum guidance weight."""
        return self.bufs["rtc_guidance_weight"]

    @property
    def input_encoder_x_buf(self) -> HipBuffer:
        """Pipeline input: encoder_x, with language embeds at [vs:vs+len]."""
        return self.bufs["encoder_x"]

    def set_language_embeds(self, lang_embeds_np) -> None:
        """Store language embeddings for this prompt.

        The embeds are kept as a persistent device-side copy and are
        re-copied into ``encoder_x[vs:vs+prompt_len]`` at the start of
        every :meth:`forward` call, because the encoder layers overwrite
        that slot in-place during each pass (transformer residual stream).

        ``lang_embeds_np`` is a numpy array of shape ``(prompt_len, ENC_D)``
        with 2-byte (bf16) elements.
        """
        prompt_len = lang_embeds_np.shape[0]
        assert prompt_len <= self.max_prompt_len, \
            f"prompt_len {prompt_len} exceeds max_prompt_len {self.max_prompt_len}"
        assert lang_embeds_np.shape[1] == ENC_D

        if self._fixed_shape and prompt_len < self.max_prompt_len:
            # Pad to max with zeros so the in-graph embeds copy is a fixed
            # MAX-row copy; the padded prompt rows are masked (seqused) in
            # attention, so they never affect the valid output.
            padded = np.zeros((self.max_prompt_len, ENC_D),
                              dtype=lang_embeds_np.dtype)
            padded[:prompt_len] = lang_embeds_np
            lang_embeds_np = padded

        # Store a persistent device copy of the (prompt_len, ENC_D) embeds.
        # HIP Graph capture bakes the source pointer used by
        # _copy_lang_embeds_to_encoder_x(), so same-shape prompt updates must
        # upload into the existing buffer instead of replacing it.
        arr = np.ascontiguousarray(lang_embeds_np)
        if (hasattr(self, "_lang_embeds_buf")
                and self._lang_embeds_buf.nbytes == arr.nbytes):
            self._lang_embeds_buf.upload(arr)
        else:
            self._lang_embeds_buf = HipBuffer.from_numpy(arr)
        self._current_prompt_len = prompt_len

        # Update decoder RoPE slice for this prompt length
        self._set_decoder_rope_for_prompt(prompt_len)

        # Fixed-shape: update the valid prefix length read by the attention
        # key masks + the devpos K/V append offset. Runtime device buffers;
        # no recapture.
        if self._fixed_shape:
            self.attn.set_fixed_valid_len(self.vision_seq_enc + prompt_len)

        # Initial copy into encoder_x (so the FIRST calibration pass sees
        # the correct lang embeds without needing a prior forward() call).
        self._copy_lang_embeds_to_encoder_x()

    def _copy_lang_embeds_to_encoder_x(self, stream: int = 0) -> None:
        """D2D copy stored lang embeds into encoder_x[vs:vs+prompt_len]."""
        if not hasattr(self, "_lang_embeds_buf"):
            return  # set_language_embeds not called yet (first build)
        start_byte = self.vision_seq_enc * ENC_D * 2  # language embeds follow pooled vision tokens
        dst_ptr = self.bufs["encoder_x"].ptr.value + start_byte
        self._hip_check(self._hip.hipMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf.ptr,
            self._lang_embeds_buf.nbytes, 3, stream), "hipMemcpyAsync D2D")

    def forward(self) -> int:
        """Replay the captured graph (or fall back to ``run_pipeline``).

        The frontend must have already written the inputs:
            - ``input_images_buf``       (observation images)
            - ``input_noise_buf``        (initial diffusion noise)
            - language embeds via :meth:`set_language_embeds` (once per prompt)

        After this returns, ``input_noise_buf`` contains the final actions.
        Returns the device pointer of that buffer for the frontend to read.
        """
        # replay() checks the hipGraphLaunch return; the sync below is
        # checked too so a failed replay can never let the caller read a
        # stale diffusion_noise buffer as real actions.
        if self._graph is not None:
            self._graph.replay(self._graph_stream)
            self._hip_check(self._hip.hipStreamSynchronize(self._graph_stream),
                            "hipStreamSynchronize")
        else:
            self.run_pipeline(stream=0)
            self._hip_check(self._hip.hipDeviceSynchronize(),
                            "hipDeviceSynchronize")
        return self.bufs["diffusion_noise"].ptr.value

    def forward_decode_only(self) -> int:
        """Replay the decoder-only graph for temporal K/V caching.

        Skips vision_encoder + transformer_encoder and runs only
        transformer_decoder, reusing the encoder K/V cache from the last
        full :meth:`forward` call. The frontend must write fresh noise into
        ``input_noise_buf`` before calling this.

        Returns the device pointer of ``diffusion_noise`` (final actions).
        """
        if hasattr(self, "_decoder_only_graph") and self._decoder_only_graph is not None:
            self._decoder_only_graph.replay(self._graph_stream)
            self._hip_check(self._hip.hipStreamSynchronize(self._graph_stream),
                            "hipStreamSynchronize")
        else:
            self.transformer_decoder(stream=0)
            self._hip_check(self._hip.hipDeviceSynchronize(),
                            "hipDeviceSynchronize")
        return self.bufs["diffusion_noise"].ptr.value
