"""FlashRT — Pi0.5 RTX inference pipeline with hardcoded B=2 batched forward.

Subclass of :class:`flash_rt.models.pi05.pipeline_rtx.Pi05Pipeline` that
runs vision + Gemma-2B encoder + Gemma-300M decoder for two independent
samples in a single forward pass. Sample-batched activation buffers and
attention buffers live alongside the parent's B=1 buffers; the parent's
methods are not modified.

Hardcoded B=2 for v0.1.0 — chosen specifically as the foundation for
:class:`Pi05CFGBatchedPipeline`, which fuses CFG's conditioned and
unconditioned forwards into a single batched pass.

Calibration: the parent's :meth:`Pi05Pipeline.calibrate_fp8` runs the
B=1 pipeline once and writes per-tensor activation scales into
``fp8_act_scales``. Those scales transfer to the B=2 path because
per-tensor FP8 scales depend on max activation magnitude across the
``M*N`` GEMM inputs, which is sample-invariant for the Pi0.5 model
under typical observation distributions. The batched run therefore
reuses the parent's calibration without an extra pass.
"""

from __future__ import annotations

import ctypes
import logging

from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.hardware.rtx.attn_backend_batched_pi05 import (
    PI05_BATCH_SIZE,
    RtxFlashAttnBatchedBackendPi05,
)

from .pipeline_rtx import (
    ACTION_DIM,
    BF16,
    DEC_D,
    DEC_H,
    DEC_HD,
    DEC_L,
    DEC_NH,
    DEC_NKV,
    ENC_D,
    ENC_H,
    ENC_HD,
    ENC_L,
    ENC_NH,
    ENC_NKV,
    FP8,
    Pi05Pipeline,
    VIS_D,
    VIS_H,
    VIS_L,
    VIS_PATCH_FLAT,
)

logger = logging.getLogger(__name__)


class Pi05BatchedPipeline(Pi05Pipeline):
    """Pi0.5 RTX pipeline running B=2 samples in a single forward pass.

    The constructor requires ``attn_backend`` to be a
    :class:`RtxFlashAttnBatchedBackendPi05` so the batched attention
    path has its B-folded Q/K/V/output buffers available.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.attn, RtxFlashAttnBatchedBackendPi05):
            raise TypeError(
                "Pi05BatchedPipeline requires attn_backend to be a "
                "RtxFlashAttnBatchedBackendPi05; got "
                f"{type(self.attn).__name__}")
        self.B = self.attn.batch_size
        if self.B != PI05_BATCH_SIZE:
            raise ValueError(
                f"Pi05BatchedPipeline expects B={PI05_BATCH_SIZE}, "
                f"backend reports {self.B}")
        self._attn_ptrs_b2 = self.attn.get_ptrs_b2()
        self._allocate_b2_buffers()

        # Per-sample language-embed CudaBuffers (set by
        # set_language_embeds_batch).
        self._lang_embeds_buf_b2: list[CudaBuffer | None] = [None] * self.B
        self._current_prompt_len_b2: int | None = None

    # ══════════════════════════════════════════════════════════════════
    #   B-fold buffer allocation (additive — parent buffers untouched)
    # ══════════════════════════════════════════════════════════════════

    def _allocate_b2_buffers(self) -> None:
        """Allocate B=2-folded versions of every per-sample working buffer.

        Naming convention: every parent buffer key ``X`` gets a
        ``X_b2`` sibling here. Per-layer FP8 weights and per-tensor
        scales (``self.fp8_act_scales``) are NOT duplicated — they
        depend only on the model, not on batch.
        """
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        ds = self.chunk_size
        B = self.B

        # ── Inputs ──
        self.bufs["observation_images_normalized_b2"] = CudaBuffer.device_empty(
            B * nv * 224 * 224 * 3, BF16)
        self.bufs["diffusion_noise_b2"] = CudaBuffer.device_empty(
            B * ds * ACTION_DIM, BF16)

        # ── Vision ──
        self.bufs["vision_x_b2"] = CudaBuffer.device_empty(B * vs * VIS_D, BF16)
        self.bufs["vision_x_norm_b2"] = CudaBuffer.device_empty(B * vs * VIS_D, BF16)
        self.bufs["vision_QKV_b2"] = CudaBuffer.device_empty(B * vs * 3 * VIS_D, BF16)
        self.bufs["vision_hidden_b2"] = CudaBuffer.device_empty(B * vs * VIS_H, BF16)
        self.bufs["vision_patches_b2"] = CudaBuffer.device_empty(
            B * vs * VIS_PATCH_FLAT, BF16)
        # Pos embed expanded across (B*nv) views — replicate parent's
        # vision_pos_embed_expanded (single-sample, vs * VIS_D) twice in M.
        self.bufs["vision_pos_embed_expanded_b2"] = CudaBuffer.device_empty(
            B * vs * VIS_D, BF16)
        self._fill_pos_embed_b2()

        # ── Encoder ──
        self.bufs["encoder_x_b2"] = CudaBuffer.device_empty(B * es * ENC_D, BF16)
        self.bufs["encoder_x_norm_b2"] = CudaBuffer.device_empty(B * es * ENC_D, BF16)
        self.bufs["encoder_QKV_b2"] = CudaBuffer.device_empty(
            B * es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, BF16)
        self.bufs["encoder_hidden_b2"] = CudaBuffer.device_empty(B * es * ENC_H, BF16)
        self.bufs["encoder_gate_merged_b2"] = CudaBuffer.device_empty(
            B * es * 2 * ENC_H, BF16)

        # ── Decoder ──
        self.bufs["decoder_x_b2"] = CudaBuffer.device_empty(B * ds * DEC_D, BF16)
        self.bufs["decoder_action_buf_b2"] = CudaBuffer.device_empty(
            B * ds * ACTION_DIM, BF16)
        self.bufs["decoder_QKV_b2"] = CudaBuffer.device_empty(
            B * ds * (DEC_NH + 2 * DEC_NKV) * DEC_HD, BF16)
        self.bufs["decoder_hidden_b2"] = CudaBuffer.device_empty(B * ds * DEC_H, BF16)
        self.bufs["decoder_gate_merged_b2"] = CudaBuffer.device_empty(
            B * ds * 2 * DEC_H, BF16)
        self.bufs["x_normed_buf_b2"] = CudaBuffer.device_empty(B * ds * DEC_D, BF16)
        self.bufs["gate_buf_b2"] = CudaBuffer.device_empty(B * ds * DEC_D, BF16)

        # ── FP8 activation scratch (B-folded) ──
        # Each scratch buffer is allocated with one extra row of slack
        # so quantize_fp8_static and downstream FP8 GEMM cannot trip a
        # by-one overflow at the buffer boundary on the batched path.
        if self.use_fp8:
            self.bufs["vis_act_fp8_b2"] = CudaBuffer.device_zeros(
                (B + 1) * vs * VIS_D, FP8)
            self.bufs["vis_act_fp8_large_b2"] = CudaBuffer.device_zeros(
                (B + 1) * vs * VIS_H, FP8)
            self.bufs["enc_act_fp8_b2"] = CudaBuffer.device_zeros(
                (B + 1) * es * ENC_D, FP8)
            self.bufs["enc_act_fp8_large_b2"] = CudaBuffer.device_zeros(
                (B + 1) * es * 2 * ENC_H, FP8)
        if self.use_fp8_decoder:
            self.bufs["dec_act_fp8_b2"] = CudaBuffer.device_zeros(
                (B + 1) * ds * DEC_D, FP8)
            self.bufs["dec_act_fp8_large_b2"] = CudaBuffer.device_zeros(
                (B + 1) * ds * 2 * DEC_H, FP8)

        # Override parent's lazy ``_bias_zero_buf``. Parent allocates it
        # on first call to ``_bias_add_bf16`` sized for **B=1** vision
        # shapes (max ~2.2 MB). The batched pipeline calls
        # ``_bias_add_bf16(m=B*seq, dim)`` with payloads up to
        # ``B * encoder_seq_len * (ENC_NH+2*ENC_NKV) * ENC_HD * 2`` bytes
        # — multiple times the parent's buffer. Without this override,
        # the kernel reads ``m * dim * 2`` bytes from the zero buffer:
        # slot 0's range is within bounds (matches what parent expected
        # for B=1) and produces correct output, but slot 1's range —
        # starting at ``seq*dim*2`` — falls into **out-of-bounds device
        # memory** belonging to neighbouring allocations. Slot 1's bias
        # add then folds in non-zero residual garbage, corrupting every
        # subsequent layer.
        # Symptom: slot 0 ≈ B=1 (cos > 0.999), slot 1 unrelated to its
        # input (cos vs B=1-with-slot-1's-prompt = -0.24). Latent in
        # symmetric inputs (both slots OOB-read the same garbage so the
        # corrupted streams stay equal); only asymmetric prompts (e.g.
        # CFG cond/uncond) expose it.
        max_count = max(
            B * nv * vs * 3 * VIS_D,            # vision QKV bias
            B * nv * vs * VIS_H,                # vision FFN up bias
            B * es * (ENC_NH + 2 * ENC_NKV) * ENC_HD,  # encoder QKV
            B * es * ENC_D,                      # encoder generic
            B * ds * (DEC_NH + 2 * DEC_NKV) * DEC_HD,  # decoder QKV
            B * ds * ACTION_DIM,                 # decoder action_buf
            B * ds * DEC_D,                      # decoder generic
        )
        self._bias_zero_buf = CudaBuffer.device_zeros(max_count, BF16)

        # B-tiled decoder style buffers.
        #
        # The decoder's ``ada_rms_norm_style_kernel`` (and the fused
        # ``ada_rms_norm_style_fp8`` / ``gate_residual_ada_norm_fp8``
        # variants) index the style tensor as
        # ``style + row * 3 * dim`` for rows ``0..seq_len-1``. The
        # parent's per-step style slice has ``ds`` rows (one per chunk
        # token); the batched call passes ``seq_len = m = B*ds`` rows.
        # Slot 0 (rows 0..ds-1) reads within bounds of the per-step
        # slice; slot 1 (rows ds..2*ds-1) reads ``ds*3*dim`` bytes past
        # the slice end, into whatever is in the next style step's
        # memory or unrelated allocations. The result is slot 1's
        # AdaRMSNorm modulation gets fed nonsense scale/shift/gate
        # values, breaking every decoder layer for slot 1.
        # Symptom: slot 1 vs B=1-with-slot-1's-prompt cos ≈ -0.2 even
        # under symmetric mirrored noise; latent under symmetric
        # prompts because both slots OOB-read the same bytes from the
        # next step's slice and stay in sync.
        # Fix: replicate each per-step (and per-layer) slice ``B`` times
        # along the row dim into ``*_b2`` style buffers — so rows
        # ``[b*ds:(b+1)*ds]`` of the b2 slice equal the parent's
        # ``ds``-row slice. The kernel's ``row * 3 * dim`` indexing
        # then walks correctly through both slots' replicated data.
        # Override :meth:`_style_slice_ptr` to return b2 slice pointers
        # so all decoder calls automatically pick up the fix without
        # touching call sites.
        import numpy as np
        pre = self.weights["precomputed"]
        # style_attn / style_ffn: (num_steps, DEC_L, ds, 3*DEC_D)
        sa = np.ascontiguousarray(np.tile(pre["style_attn"], (1, 1, B, 1)))
        sf = np.ascontiguousarray(np.tile(pre["style_ffn"], (1, 1, B, 1)))
        # style_final: (num_steps, ds, 3*DEC_D)
        sfin = np.ascontiguousarray(np.tile(pre["style_final"], (1, B, 1)))
        self.bufs["decoder_style_attn_b2"] = CudaBuffer.from_numpy(sa)
        self.bufs["decoder_style_ffn_b2"] = CudaBuffer.from_numpy(sf)
        self.bufs["decoder_style_final_b2"] = CudaBuffer.from_numpy(sfin)

    def _style_slice_ptr(self, buf_name: str, step: int,
                          layer: int | None = None) -> int:
        """Override parent's slice indexing to point into B-tiled b2 buffers.

        Each per-step (and per-layer for attn/ffn) slice in the b2
        buffer has ``B * ds`` rows × ``3 * DEC_D`` columns; the kernel's
        per-row stride is ``3 * DEC_D`` elements (× 2 bytes), unchanged
        from the parent layout. We just route the pointer to the
        replicated buffer so the kernel's ``seq_len = m = B*ds``
        per-row reads stay within the slice bounds for both slots.
        """
        # Standard (non-decoder-style) buffers fall through to parent.
        if buf_name not in (
                "decoder_style_attn", "decoder_style_ffn",
                "decoder_style_final"):
            return super()._style_slice_ptr(buf_name, step, layer)
        b2_key = buf_name + "_b2"
        base = self.bufs[b2_key].ptr.value
        # Per-row size is 3 * DEC_D * 2 bytes; per-slice has B*ds rows.
        per_slice = self.B * self.chunk_size * 3 * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * per_slice
        # style_attn / style_ffn: (num_steps, DEC_L, B*ds, 3*DEC_D)
        per_layer = per_slice
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer

    def _fill_pos_embed_b2(self) -> None:
        """Replicate the parent's per-sample pos_embed across B in the M dim."""
        nbytes = self.bufs["vision_pos_embed_expanded"].nbytes
        dst_base = self.bufs["vision_pos_embed_expanded_b2"].ptr.value
        for b in range(self.B):
            # ctypes.c_void_p() wrap is mandatory: passing a raw Python
            # int directly to ``cudaMemcpy`` lets ctypes' default int
            # marshalling truncate to 32 bits on 64-bit systems, which
            # corrupts the destination pointer when the offset pushes
            # ``dst`` past the low-32-bit boundary and silently writes
            # to wild addresses (manifests as glibc heap corruption).
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_base + b * nbytes),
                self.bufs["vision_pos_embed_expanded"].ptr,
                nbytes, 3)

    # ══════════════════════════════════════════════════════════════════
    #   Public API: per-sample language-embed upload
    # ══════════════════════════════════════════════════════════════════

    def set_language_embeds_batch(self, embeds_np_list) -> None:
        """Store per-sample language embeddings.

        Args:
            embeds_np_list: list of length B; each entry a numpy array
                of shape ``(prompt_len, ENC_D)`` with 2-byte BF16
                elements. All entries must share the same prompt_len
                so the encoder can run on a fixed-shape buffer.
        """
        if len(embeds_np_list) != self.B:
            raise ValueError(
                f"set_language_embeds_batch expects {self.B} entries, "
                f"got {len(embeds_np_list)}")
        prompt_len = embeds_np_list[0].shape[0]
        for i, e in enumerate(embeds_np_list):
            if e.shape[1] != ENC_D:
                raise ValueError(
                    f"sample {i}: last dim must be {ENC_D}, got {e.shape[1]}")
            if e.shape[0] != prompt_len:
                raise ValueError(
                    f"sample {i}: prompt_len mismatch (expected {prompt_len}, "
                    f"got {e.shape[0]}). Pad to a common length before calling.")
        if prompt_len > self.max_prompt_len:
            raise ValueError(
                f"prompt_len {prompt_len} exceeds max_prompt_len "
                f"{self.max_prompt_len}")
        import numpy as np
        for b, e in enumerate(embeds_np_list):
            arr = np.ascontiguousarray(e)
            self._lang_embeds_buf_b2[b] = CudaBuffer.from_numpy(arr)
        self._current_prompt_len_b2 = prompt_len
        # Reuse the parent's decoder-RoPE setup for this prompt length.
        self._set_decoder_rope_for_prompt(prompt_len)
        self._copy_lang_embeds_to_encoder_x_b2()

    def _copy_lang_embeds_to_encoder_x_b2(self, stream: int = 0) -> None:
        """D2D copy each sample's lang embeds into ``encoder_x_b2[b, vs:vs+L]``."""
        if self._current_prompt_len_b2 is None:
            return
        sample_stride_bytes = self.encoder_seq_len * ENC_D * 2  # bf16
        slot_off_bytes = self.vision_seq * ENC_D * 2
        for b in range(self.B):
            src = self._lang_embeds_buf_b2[b]
            if src is None:
                continue
            dst = (self.bufs["encoder_x_b2"].ptr.value
                   + b * sample_stride_bytes
                   + slot_off_bytes)
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(dst), src.ptr, src.nbytes, 3, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Helper: B=2 FP8 scratch picker + dynamic-quant GEMM
    # ══════════════════════════════════════════════════════════════════

    def _pick_fp8_scratch_b2(self, weight_name: str,
                             act_n: int) -> tuple[int, int]:
        """B=2 sibling of :meth:`Pi05Pipeline._pick_fp8_scratch`.

        Returns ``(act_fp8_ptr, scratch_scale_ptr)`` from the b2
        scratch buffers. Used by :meth:`_fp8_gemm_b2`; the parent's
        ``_pick_fp8_scratch`` (which returns the parent's B=1
        scratch) is left untouched and still serves the parent's body.
        """
        Bb = self.bufs
        if (weight_name.startswith("vision_")
                or weight_name == "vision_projector_w"):
            small = Bb["vis_act_fp8_b2"]
            large = Bb["vis_act_fp8_large_b2"]
            scratch_scale = Bb["vis_act_scale"]
        elif weight_name.startswith("encoder_"):
            small = Bb["enc_act_fp8_b2"]
            large = Bb["enc_act_fp8_large_b2"]
            scratch_scale = Bb["enc_act_scale"]
        else:
            small = Bb["dec_act_fp8_b2"]
            large = Bb["dec_act_fp8_large_b2"]
            scratch_scale = Bb["dec_act_scale"]
        buf = small if act_n <= small.nbytes else large
        return buf.ptr.value, scratch_scale.ptr.value

    def _fp8_gemm_b2(self, act_bf16_ptr: int, act_n: int, weight_name: str,
                     out_bf16_ptr: int, M: int, N: int, K: int,
                     stream: int) -> None:
        """B=2 sibling of :meth:`Pi05Pipeline._fp8_gemm`.

        Identical math, but routes the dynamic-quantization scratch
        write through the b2 scratch buffers so M=B*seq writes do not
        overflow the parent's B=1 scratch slots.
        """
        fvk = self.fvk
        w_fp8_ptr, w_scale_ptr = self._weight_fp8(weight_name)
        act_fp8_ptr, _scratch_scale_ptr = self._pick_fp8_scratch_b2(
            weight_name, act_n)
        if self.fp8_calibrated and weight_name in self.fp8_act_scales:
            static_scale_ptr = self.fp8_act_scales[weight_name].ptr.value
            fvk.quantize_fp8_static(
                act_bf16_ptr, act_fp8_ptr, static_scale_ptr, act_n,
                stream=stream)
            self._fp8_matmul(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, static_scale_ptr, w_scale_ptr, stream)
        else:
            layer_scale = self._fp8_scale_buf(weight_name)
            fvk.quantize_fp8_device(
                act_bf16_ptr, act_fp8_ptr, layer_scale.ptr.value, act_n,
                stream=stream)
            self._fp8_matmul(
                act_fp8_ptr, w_fp8_ptr, out_bf16_ptr,
                M, N, K, layer_scale.ptr.value, w_scale_ptr, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Helper: per-layer batched KV cache pointers
    # ══════════════════════════════════════════════════════════════════

    def _enc_kv_layer_ptrs_b2(self, layer: int,
                              offset_tokens: int = 0) -> tuple[int, int]:
        """K/V buffer pointers for the given encoder layer in the B=2 cache.

        Returns the pointer of slot ``[layer, sample=0, offset_tokens]``;
        the per-sample stride is :attr:`_attn_ptrs_b2["enc_k_sample_stride_bytes"]`.
        Per-token stride is ``1 * 256 * 2`` (bf16).
        """
        ap = self._attn_ptrs_b2
        layer_stride = ap["enc_k_layer_stride_bytes"]
        # Per-token stride within (sample, total_kv, 1, 256)
        row_stride = 1 * 256 * 2  # bf16 elements
        offset_bytes = offset_tokens * row_stride
        k_ptr = ap["enc_K"] + layer * layer_stride + offset_bytes
        v_ptr = ap["enc_V"] + layer * layer_stride + offset_bytes
        return k_ptr, v_ptr

    # ══════════════════════════════════════════════════════════════════
    #   Phase A: SigLIP vision encoder (batched)
    # ══════════════════════════════════════════════════════════════════

    def vision_encoder_batched(self, stream: int = 0) -> None:
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        m = self.B * self.vision_seq  # M-folded sequence length
        nv_b = self.B * self.num_views

        # A1: Patch embed — im2col + GEMM + bias + pos embed
        fvk.patch_im2col(
            Bb["observation_images_normalized_b2"].ptr.value,
            Bb["vision_patches_b2"].ptr.value,
            nv_b, stream)
        gemm.bf16_nn(
            Bb["vision_patches_b2"].ptr.value,
            W["vision_patch_embedding_w"],
            Bb["vision_x_b2"].ptr.value,
            m, VIS_D, VIS_PATCH_FLAT, stream=stream)
        fvk.bias_residual(
            Bb["vision_x_b2"].ptr.value,
            Bb["vision_pos_embed_expanded_b2"].ptr.value,
            W["vision_patch_embedding_b"],
            m, VIS_D, stream=stream)

        use_fp8 = self.use_fp8 and "vision_attn_qkv_w_0" in self.weights.get("fp8", {})
        for i in range(VIS_L):
            self._vision_layer_batched(i, m, use_fp8, stream)

    def _vision_layer_batched(self, i: int, m: int, use_fp8: bool,
                              stream: int) -> None:
        """One SigLIP layer running on (B*nv*256) folded rows + batched attn."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        ap = self._attn_ptrs_b2

        # Attention LayerNorm → x_norm
        fvk.layer_norm(
            Bb["vision_x_b2"].ptr.value,
            W["vision_pre_attn_norm_w"][i], W["vision_pre_attn_norm_b"][i],
            Bb["vision_x_norm_b2"].ptr.value,
            m, VIS_D, 1e-5, stream=stream)

        # QKV GEMM
        if use_fp8:
            self._fp8_gemm_b2(
                Bb["vision_x_norm_b2"].ptr.value, m * VIS_D,
                f"vision_attn_qkv_w_{i}",
                Bb["vision_QKV_b2"].ptr.value,
                m, 3 * VIS_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                Bb["vision_x_norm_b2"].ptr.value, W["vision_attn_qkv_w"][i],
                Bb["vision_QKV_b2"].ptr.value,
                m, 3 * VIS_D, VIS_D, stream=stream)
        self._bias_add_bf16(
            Bb["vision_QKV_b2"].ptr.value, W["vision_attn_qkv_b"][i],
            m, 3 * VIS_D, stream)

        # Split QKV into batched attn buffers
        fvk.qkv_split(
            Bb["vision_QKV_b2"].ptr.value,
            ap["vis_Q"], ap["vis_K"], ap["vis_V"],
            m, VIS_D, VIS_D, VIS_D, stream=stream)

        # Self-attention (B*nv views in one call)
        vis_o_ptr = self.attn.run_batched(
            "siglip", i, q_seq=VIS_L, stream=stream)  # q_seq unused for siglip

        # Attn output projection → x_norm
        if use_fp8:
            self._fp8_gemm_b2(
                vis_o_ptr, m * VIS_D,
                f"vision_attn_o_w_{i}",
                Bb["vision_x_norm_b2"].ptr.value,
                m, VIS_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                vis_o_ptr, W["vision_attn_o_w"][i],
                Bb["vision_x_norm_b2"].ptr.value,
                m, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual(
            Bb["vision_x_b2"].ptr.value, Bb["vision_x_norm_b2"].ptr.value,
            W["vision_attn_o_b"][i], m, VIS_D, stream=stream)

        # FFN LayerNorm → x_norm
        fvk.layer_norm(
            Bb["vision_x_b2"].ptr.value,
            W["vision_pre_ffn_norm_w"][i], W["vision_pre_ffn_norm_b"][i],
            Bb["vision_x_norm_b2"].ptr.value,
            m, VIS_D, 1e-5, stream=stream)

        # FFN up + bias + GELU
        if use_fp8:
            self._fp8_gemm_b2(
                Bb["vision_x_norm_b2"].ptr.value, m * VIS_D,
                f"vision_ffn_up_w_{i}",
                Bb["vision_hidden_b2"].ptr.value,
                m, VIS_H, VIS_D, stream)
        else:
            gemm.bf16_nn(
                Bb["vision_x_norm_b2"].ptr.value, W["vision_ffn_up_w"][i],
                Bb["vision_hidden_b2"].ptr.value,
                m, VIS_H, VIS_D, stream=stream)
        self._bias_add_bf16(
            Bb["vision_hidden_b2"].ptr.value, W["vision_ffn_up_b"][i],
            m, VIS_H, stream)
        fvk.gelu_inplace(Bb["vision_hidden_b2"].ptr.value, m * VIS_H, stream=stream)

        # FFN down → residual
        if use_fp8:
            self._fp8_gemm_b2(
                Bb["vision_hidden_b2"].ptr.value, m * VIS_H,
                f"vision_ffn_down_w_{i}",
                Bb["vision_x_norm_b2"].ptr.value,
                m, VIS_D, VIS_H, stream)
        else:
            gemm.bf16_nn(
                Bb["vision_hidden_b2"].ptr.value, W["vision_ffn_down_w"][i],
                Bb["vision_x_norm_b2"].ptr.value,
                m, VIS_D, VIS_H, stream=stream)
        fvk.bias_residual(
            Bb["vision_x_b2"].ptr.value, Bb["vision_x_norm_b2"].ptr.value,
            W["vision_ffn_down_b"][i], m, VIS_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase B: Gemma-2B encoder (batched)
    # ══════════════════════════════════════════════════════════════════

    def transformer_encoder_batched(self, stream: int = 0) -> None:
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        seq = self.encoder_seq_len
        vs = self.vision_seq
        m_vis = self.B * vs
        m_seq = self.B * seq
        use_fp8 = self.use_fp8

        # B0: LayerNorm(vision output) → project 1152→2048 + bias → encoder_x[:vs] (per sample)
        fvk.layer_norm(
            Bb["vision_x_b2"].ptr.value,
            W["vision_final_norm_w"], W["vision_final_norm_b"],
            Bb["vision_x_norm_b2"].ptr.value,
            m_vis, VIS_D, 1e-5, stream=stream)

        # The projection writes into vision_x_norm_b2 (intermediate); we then
        # need to scatter the (B*vs, ENC_D) result into the per-sample
        # encoder_x_b2[b, :vs, :] slots, leaving each sample's lang slot
        # (already populated by set_language_embeds_batch) intact. Use
        # per-sample copies because encoder_x_b2 has stride (es*ENC_D)
        # per sample but vision_x_norm_b2 has stride (vs*ENC_D).
        # First do the GEMM into a per-sample-contiguous scratch we can
        # split: use a temp slot from encoder_QKV_b2 (large enough,
        # unused at this point in the pipeline).
        scratch_proj = Bb["encoder_QKV_b2"].ptr.value  # holds B*vs*ENC_D bf16
        if use_fp8 and "vision_projector_w" in self.weights.get("fp8", {}):
            self._fp8_gemm_b2(
                Bb["vision_x_norm_b2"].ptr.value, m_vis * VIS_D,
                "vision_projector_w",
                scratch_proj,
                m_vis, ENC_D, VIS_D, stream)
        else:
            gemm.bf16_nn(
                Bb["vision_x_norm_b2"].ptr.value,
                W["encoder_multi_modal_projector_w"],
                scratch_proj,
                m_vis, ENC_D, VIS_D, stream=stream)
        self._bias_add_bf16(
            scratch_proj, W["encoder_multi_modal_projector_b"],
            m_vis, ENC_D, stream)

        # Scatter (B, vs, ENC_D) → encoder_x_b2[b, 0:vs, :] for each b
        per_sample_enc_bytes = seq * ENC_D * 2
        per_sample_proj_bytes = vs * ENC_D * 2
        for b in range(self.B):
            src = scratch_proj + b * per_sample_proj_bytes
            dst = Bb["encoder_x_b2"].ptr.value + b * per_sample_enc_bytes
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(dst), ctypes.c_void_p(src),
                per_sample_proj_bytes, 3, stream)

        # B1-B5: 18 encoder layers
        fused = use_fp8 and self.fp8_calibrated
        for i in range(ENC_L):
            self._encoder_layer_batched(i, m_seq, seq,
                                         fuse_b1=(i > 0 and fused), stream=stream)

    def _encoder_layer_batched(self, i: int, m: int, seq: int,
                                fuse_b1: bool, stream: int) -> None:
        """One Gemma-2B encoder layer with B=2 sample batching.

        ``m = B * seq`` is the M-folded total row count for GEMMs and
        per-row ops; ``seq`` (per-sample) is what the batched attention
        call expects.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        ap = self._attn_ptrs_b2
        fused = self.use_fp8 and self.fp8_calibrated

        # B1: RMSNorm → QKV GEMM
        if fused:
            qkv_name = f"encoder_attn_qkv_w_{i}"
            act_scale_ptr = self.fp8_act_scales[qkv_name].ptr.value
            if fuse_b1:
                fvk.residual_add_rms_norm_fp8(
                    Bb["encoder_x_b2"].ptr.value, Bb["encoder_x_norm_b2"].ptr.value,
                    self._rms_ones_enc.ptr.value,
                    Bb["enc_act_fp8_b2"].ptr.value,
                    m, ENC_D, 1e-6, act_scale_ptr, stream=stream)
            else:
                fvk.rms_norm_fp8(
                    Bb["encoder_x_b2"].ptr.value, self._rms_ones_enc.ptr.value,
                    Bb["enc_act_fp8_b2"].ptr.value,
                    m, ENC_D, 1e-6, act_scale_ptr, stream=stream)
            self._fp8_gemm_fused(
                Bb["enc_act_fp8_b2"].ptr.value, qkv_name,
                Bb["encoder_QKV_b2"].ptr.value,
                m, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
                act_scale_ptr, stream)
        elif self.use_fp8:
            fvk.rms_norm(
                Bb["encoder_x_b2"].ptr.value, self._rms_ones_enc.ptr.value,
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm_b2(
                Bb["encoder_x_norm_b2"].ptr.value, m * ENC_D,
                f"encoder_attn_qkv_w_{i}",
                Bb["encoder_QKV_b2"].ptr.value,
                m, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream)
        else:
            fvk.rms_norm(
                Bb["encoder_x_b2"].ptr.value, self._rms_ones_enc.ptr.value,
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, 1e-6, stream=stream)
            gemm.bf16_nn(
                Bb["encoder_x_norm_b2"].ptr.value, W["encoder_attn_qkv_w"][i],
                Bb["encoder_QKV_b2"].ptr.value,
                m, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)

        # Split QKV + RoPE — KV write into batched per-sample cache slot.
        # qkv_split_rope expects M-folded input/output rows. The K/V dst
        # pointers point at sample-0 slot; sample-1 slot is at +sample_stride.
        # Since the input QKV is M-folded as [s0_seq | s1_seq | ...], we
        # need the K/V destination to interleave per sample. But the cache
        # layout is (L, B, total_kv, 1, 256) — sample-1 starts at
        # +total_kv*1*256*2 bytes from sample-0. The qkv_split_rope kernel
        # writes contiguously, so we need to call it per sample to land
        # rows in the right slot.
        per_sample_qkv_bytes = seq * (ENC_NH + 2 * ENC_NKV) * ENC_HD * 2
        # Q dst stride MUST come from the attention backend's actual
        # buffer layout (es_max-based), not from the per-call ``seq``.
        # enc_Q_b2 is sized for
        # encoder_seq_max along the time dim; offsetting slot 1 by
        # ``seq * NH * HD * 2`` (when seq < es_max) lands the writes
        # inside slot 0's tail and leaves slot 1 uninitialised.
        sample_stride_q = self._attn_ptrs_b2["enc_q_sample_stride_bytes"]
        sample_stride_kv = self._attn_ptrs_b2["enc_k_sample_stride_bytes"]
        for b in range(self.B):
            qkv_src = Bb["encoder_QKV_b2"].ptr.value + b * per_sample_qkv_bytes
            q_dst = ap["enc_Q"] + b * sample_stride_q
            k_ptr_b, v_ptr_b = self._enc_kv_layer_ptrs_b2(i, offset_tokens=0)
            k_ptr_b += b * sample_stride_kv
            v_ptr_b += b * sample_stride_kv
            fvk.qkv_split_rope(
                qkv_src,
                Bb["encoder_rope_weights"].ptr.value,
                q_dst,
                k_ptr_b, v_ptr_b,
                seq, ENC_NH * ENC_HD, ENC_NKV * ENC_HD, ENC_NKV * ENC_HD,
                ENC_HD, stream=stream)

        if i == ENC_L - 1:
            return

        # B2: Batched attention (B=2 in the leading dim)
        enc_o_ptr = self.attn.run_batched(
            "encoder", i, q_seq=seq, stream=stream)

        # B3: Attn output projection
        if self.use_fp8:
            self._fp8_gemm_b2(
                enc_o_ptr, m * ENC_D,
                f"encoder_attn_o_w_{i}",
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, ENC_D, stream)
        else:
            gemm.bf16_nn(
                enc_o_ptr, W["encoder_attn_o_w"][i],
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, ENC_D, stream=stream)

        # B4: RMSNorm → FFN gate+up
        if fused:
            gu_name = f"encoder_ffn_gate_up_w_{i}"
            act_scale_gu = self.fp8_act_scales[gu_name].ptr.value
            fvk.residual_add_rms_norm_fp8(
                Bb["encoder_x_b2"].ptr.value, Bb["encoder_x_norm_b2"].ptr.value,
                self._rms_ones_enc.ptr.value,
                Bb["enc_act_fp8_b2"].ptr.value,
                m, ENC_D, 1e-6, act_scale_gu, stream=stream)
            self._fp8_gemm_fused(
                Bb["enc_act_fp8_b2"].ptr.value, gu_name,
                Bb["encoder_gate_merged_b2"].ptr.value,
                m, 2 * ENC_H, ENC_D, act_scale_gu, stream)
        elif self.use_fp8:
            fvk.residual_add(
                Bb["encoder_x_b2"].ptr.value, Bb["encoder_x_norm_b2"].ptr.value,
                m * ENC_D, stream=stream)
            fvk.rms_norm(
                Bb["encoder_x_b2"].ptr.value, self._rms_ones_enc.ptr.value,
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, 1e-6, stream=stream)
            self._fp8_gemm_b2(
                Bb["encoder_x_norm_b2"].ptr.value, m * ENC_D,
                f"encoder_ffn_gate_up_w_{i}",
                Bb["encoder_gate_merged_b2"].ptr.value,
                m, 2 * ENC_H, ENC_D, stream)
        else:
            fvk.residual_add(
                Bb["encoder_x_b2"].ptr.value, Bb["encoder_x_norm_b2"].ptr.value,
                m * ENC_D, stream=stream)
            fvk.rms_norm(
                Bb["encoder_x_b2"].ptr.value, self._rms_ones_enc.ptr.value,
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, 1e-6, stream=stream)
            gemm.bf16_nn(
                Bb["encoder_x_norm_b2"].ptr.value, W["encoder_ffn_gate_w"][i],
                Bb["encoder_gate_merged_b2"].ptr.value,
                m, ENC_H, ENC_D, stream=stream)
            gemm.bf16_nn(
                Bb["encoder_x_norm_b2"].ptr.value, W["encoder_ffn_up_w"][i],
                Bb["encoder_hidden_b2"].ptr.value,
                m, ENC_H, ENC_D, stream=stream)

        # SiLU(gate) * up → hidden
        if fused:
            down_name = f"encoder_ffn_down_w_{i}"
            act_scale_down = self.fp8_act_scales[down_name].ptr.value
            fvk.gate_geglu_merged_fp8(
                Bb["encoder_gate_merged_b2"].ptr.value,
                Bb["enc_act_fp8_large_b2"].ptr.value,
                m, ENC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                Bb["enc_act_fp8_large_b2"].ptr.value, down_name,
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, ENC_H, act_scale_down, stream)
        elif self.use_fp8:
            fvk.gate_geglu_merged(
                Bb["encoder_gate_merged_b2"].ptr.value,
                Bb["encoder_hidden_b2"].ptr.value,
                m, ENC_H, stream=stream)
            self._fp8_gemm_b2(
                Bb["encoder_hidden_b2"].ptr.value, m * ENC_H,
                f"encoder_ffn_down_w_{i}",
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, ENC_H, stream)
        else:
            fvk.gate_geglu(
                Bb["encoder_gate_merged_b2"].ptr.value,
                Bb["encoder_hidden_b2"].ptr.value,
                Bb["encoder_hidden_b2"].ptr.value,
                m * ENC_H, stream=stream)
            gemm.bf16_nn(
                Bb["encoder_hidden_b2"].ptr.value, W["encoder_ffn_down_w"][i],
                Bb["encoder_x_norm_b2"].ptr.value,
                m, ENC_D, ENC_H, stream=stream)

        # B5: Residual (skipped in fused mode)
        if not fused:
            fvk.residual_add(
                Bb["encoder_x_b2"].ptr.value, Bb["encoder_x_norm_b2"].ptr.value,
                m * ENC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Phase C: Gemma-300M decoder (batched)
    # ══════════════════════════════════════════════════════════════════

    def transformer_decoder_batched(self, stream: int = 0) -> None:
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        enc_seq = self.encoder_seq_len
        ds = self.chunk_size
        m = self.B * ds
        fused = self.use_fp8_decoder and self.fp8_calibrated

        for step in range(self.num_steps):
            # C0: Action input projection
            gemm.bf16_nn(
                Bb["diffusion_noise_b2"].ptr.value,
                W["decoder_action_in_proj_w"],
                Bb["decoder_x_b2"].ptr.value,
                m, DEC_D, ACTION_DIM, stream=stream)
            self._bias_add_bf16(
                Bb["decoder_x_b2"].ptr.value, W["decoder_action_in_proj_b"],
                m, DEC_D, stream)

            for i in range(DEC_L):
                skip_c1 = fused and i > 0
                self._decoder_layer_batched(i, step, enc_seq, ds, m,
                                             skip_c1, stream)

            # C8: Final AdaRMSNorm + output projection
            fvk.ada_rms_norm_style(
                Bb["decoder_x_b2"].ptr.value, self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_final", step),
                Bb["x_normed_buf_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                m, DEC_D, 1e-6, stream=stream)
            gemm.bf16_nn(
                Bb["x_normed_buf_b2"].ptr.value,
                W["decoder_action_out_proj_w"],
                Bb["decoder_action_buf_b2"].ptr.value,
                m, ACTION_DIM, DEC_D, stream=stream)
            self._bias_add_bf16(
                Bb["decoder_action_buf_b2"].ptr.value,
                W["decoder_action_out_proj_b"],
                m, ACTION_DIM, stream)
            fvk.residual_add(
                Bb["diffusion_noise_b2"].ptr.value,
                Bb["decoder_action_buf_b2"].ptr.value,
                m * ACTION_DIM, stream=stream)

    def _decoder_layer_batched(self, i: int, step: int, enc_seq: int,
                                ds: int, m: int, skip_c1: bool,
                                stream: int) -> None:
        """One Gemma-300M decoder layer with B=2 sample batching.

        ``ds`` is per-sample chunk length; ``m = B*ds`` is the M-fold
        for GEMMs and per-row ops.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        ap = self._attn_ptrs_b2
        fused = self.use_fp8_decoder and self.fp8_calibrated

        # C1: AdaRMSNorm with style → FP8 (fused) or BF16
        qkv_name = f"decoder_attn_qkv_w_{i}"
        if fused:
            act_scale_qkv = self.fp8_act_scales[qkv_name].ptr.value
            if not skip_c1:
                fvk.ada_rms_norm_style_fp8(
                    Bb["decoder_x_b2"].ptr.value, self._rms_ones_dec.ptr.value,
                    self._style_slice_ptr("decoder_style_attn", step, i),
                    Bb["dec_act_fp8_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                    m, DEC_D, 1e-6, act_scale_qkv, stream=stream)
            self._fp8_gemm_fused(
                Bb["dec_act_fp8_b2"].ptr.value, qkv_name,
                Bb["decoder_QKV_b2"].ptr.value,
                m, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D,
                act_scale_qkv, stream)
        else:
            fvk.ada_rms_norm_style(
                Bb["decoder_x_b2"].ptr.value, self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_attn", step, i),
                Bb["x_normed_buf_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                m, DEC_D, 1e-6, stream=stream)
            if self.use_fp8_decoder:
                self._fp8_gemm_b2(
                    Bb["x_normed_buf_b2"].ptr.value, m * DEC_D,
                    qkv_name,
                    Bb["decoder_QKV_b2"].ptr.value,
                    m, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream)
            else:
                gemm.bf16_nn(
                    Bb["x_normed_buf_b2"].ptr.value, W["decoder_attn_qkv_w"][i],
                    Bb["decoder_QKV_b2"].ptr.value,
                    m, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)

        # C2: QKV split + RoPE — per-sample loop into batched KV cache
        per_sample_qkv_bytes = ds * (DEC_NH + 2 * DEC_NKV) * DEC_HD * 2
        per_sample_q_bytes = ds * DEC_NH * DEC_HD * 2
        sample_stride_kv = ap["enc_k_sample_stride_bytes"]
        for b in range(self.B):
            qkv_src = Bb["decoder_QKV_b2"].ptr.value + b * per_sample_qkv_bytes
            q_dst = ap["dec_Q"] + b * per_sample_q_bytes
            k_ptr_b, v_ptr_b = self._enc_kv_layer_ptrs_b2(i, offset_tokens=enc_seq)
            k_ptr_b += b * sample_stride_kv
            v_ptr_b += b * sample_stride_kv
            fvk.qkv_split_rope(
                qkv_src,
                Bb["decoder_rope_weights"].ptr.value,
                q_dst,
                k_ptr_b, v_ptr_b,
                ds, DEC_NH * DEC_HD, DEC_NKV * DEC_HD, DEC_NKV * DEC_HD,
                DEC_HD, stream=stream)

        # C3: Batched cross-attention (per-sample contexts)
        dec_o_ptr = self.attn.run_batched(
            "decoder", i, q_seq=ds, kv_seq=enc_seq + ds, stream=stream)

        # C4: Attn output projection
        if self.use_fp8_decoder:
            self._fp8_gemm_b2(
                dec_o_ptr, m * DEC_NH * DEC_HD,
                f"decoder_attn_o_w_{i}",
                Bb["x_normed_buf_b2"].ptr.value,
                m, DEC_D, DEC_NH * DEC_HD, stream)
        else:
            gemm.bf16_nn(
                dec_o_ptr, W["decoder_attn_o_w"][i],
                Bb["x_normed_buf_b2"].ptr.value,
                m, DEC_D, DEC_NH * DEC_HD, stream=stream)

        # C4→C5: gate*residual + AdaRMSNorm + FFN gate_up
        gu_name = f"decoder_ffn_gate_up_w_{i}"
        if fused:
            act_scale_gu = self.fp8_act_scales[gu_name].ptr.value
            fvk.gate_residual_ada_norm_fp8(
                Bb["decoder_x_b2"].ptr.value, Bb["x_normed_buf_b2"].ptr.value,
                Bb["gate_buf_b2"].ptr.value,
                self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_ffn", step, i),
                Bb["dec_act_fp8_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                m, DEC_D, 1e-6, act_scale_gu, stream=stream)
            self._fp8_gemm_fused(
                Bb["dec_act_fp8_b2"].ptr.value, gu_name,
                Bb["decoder_gate_merged_b2"].ptr.value,
                m, 2 * DEC_H, DEC_D, act_scale_gu, stream)
        else:
            fvk.gate_mul_residual(
                Bb["decoder_x_b2"].ptr.value, Bb["x_normed_buf_b2"].ptr.value,
                Bb["gate_buf_b2"].ptr.value, m * DEC_D, stream=stream)
            fvk.ada_rms_norm_style(
                Bb["decoder_x_b2"].ptr.value, self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_ffn", step, i),
                Bb["x_normed_buf_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                m, DEC_D, 1e-6, stream=stream)
            if self.use_fp8_decoder:
                self._fp8_gemm_b2(
                    Bb["x_normed_buf_b2"].ptr.value, m * DEC_D,
                    gu_name,
                    Bb["decoder_gate_merged_b2"].ptr.value,
                    m, 2 * DEC_H, DEC_D, stream)
            else:
                gemm.bf16_nn(
                    Bb["x_normed_buf_b2"].ptr.value, W["decoder_ffn_gate_w"][i],
                    Bb["decoder_gate_merged_b2"].ptr.value,
                    m, DEC_H, DEC_D, stream=stream)
                gemm.bf16_nn(
                    Bb["x_normed_buf_b2"].ptr.value, W["decoder_ffn_up_w"][i],
                    Bb["decoder_hidden_b2"].ptr.value,
                    m, DEC_H, DEC_D, stream=stream)

        # C6: SiLU(gate) * up → FFN down
        down_name = f"decoder_ffn_down_w_{i}"
        if fused:
            act_scale_down = self.fp8_act_scales[down_name].ptr.value
            fvk.gate_geglu_merged_fp8(
                Bb["decoder_gate_merged_b2"].ptr.value,
                Bb["dec_act_fp8_large_b2"].ptr.value,
                m, DEC_H, act_scale_down, stream=stream)
            self._fp8_gemm_fused(
                Bb["dec_act_fp8_large_b2"].ptr.value, down_name,
                Bb["x_normed_buf_b2"].ptr.value,
                m, DEC_D, DEC_H, act_scale_down, stream)
        elif self.use_fp8_decoder:
            fvk.gate_geglu_merged(
                Bb["decoder_gate_merged_b2"].ptr.value,
                Bb["decoder_hidden_b2"].ptr.value,
                m, DEC_H, stream=stream)
            self._fp8_gemm_b2(
                Bb["decoder_hidden_b2"].ptr.value, m * DEC_H,
                down_name,
                Bb["x_normed_buf_b2"].ptr.value,
                m, DEC_D, DEC_H, stream)
        else:
            fvk.gate_geglu(
                Bb["decoder_gate_merged_b2"].ptr.value,
                Bb["decoder_hidden_b2"].ptr.value,
                Bb["decoder_hidden_b2"].ptr.value,
                m * DEC_H, stream=stream)
            gemm.bf16_nn(
                Bb["decoder_hidden_b2"].ptr.value, W["decoder_ffn_down_w"][i],
                Bb["x_normed_buf_b2"].ptr.value,
                m, DEC_D, DEC_H, stream=stream)

        # C7→C1_next: gate*residual + next layer's AdaRMSNorm → FP8 (fused)
        if fused and i < DEC_L - 1:
            next_qkv = f"decoder_attn_qkv_w_{i + 1}"
            act_scale_next = self.fp8_act_scales[next_qkv].ptr.value
            fvk.gate_residual_ada_norm_fp8(
                Bb["decoder_x_b2"].ptr.value, Bb["x_normed_buf_b2"].ptr.value,
                Bb["gate_buf_b2"].ptr.value,
                self._rms_ones_dec.ptr.value,
                self._style_slice_ptr("decoder_style_attn", step, i + 1),
                Bb["dec_act_fp8_b2"].ptr.value, Bb["gate_buf_b2"].ptr.value,
                m, DEC_D, 1e-6, act_scale_next, stream=stream)
        else:
            fvk.gate_mul_residual(
                Bb["decoder_x_b2"].ptr.value, Bb["x_normed_buf_b2"].ptr.value,
                Bb["gate_buf_b2"].ptr.value, m * DEC_D, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #   Top-level: full pipeline + calibration + graph capture
    # ══════════════════════════════════════════════════════════════════

    def run_pipeline(self, stream: int = 0) -> None:
        """Run the B=2 pipeline end-to-end.

        Override of the parent's :meth:`Pi05Pipeline.run_pipeline`. The
        captured CUDA Graph (recorded by the inherited
        :meth:`record_infer_graph`) replays this batched flow.
        """
        self._copy_lang_embeds_to_encoder_x_b2(stream=stream)
        self.vision_encoder_batched(stream)
        self.transformer_encoder_batched(stream)
        self.transformer_decoder_batched(stream)

    def autotune_gemms(self) -> None:
        """Autotune GEMM tactics for both B=1 and B=2 M values.

        The parent's :meth:`Pi05Pipeline.autotune_gemms` only sees the
        B=1 ``M = vs / seq / ds`` shapes; running with B=2 doubles each
        M and cuBLASLt then has to pick a tactic on the fly, which can
        fail (CUBLAS_STATUS_INTERNAL_ERROR) on shapes the autotune
        cache hasn't seen. We run the parent's tune first (so the B=1
        calibration / parent-path forwards remain bit-equal) and then
        an additional pass at the B=2 M values using the b2 buffers.
        """
        # First, the parent's B=1 tune (covers calibration-time GEMMs).
        super().autotune_gemms()

        # Then run an additional autotune at the B=2 ``M = B*seq`` shapes.
        # When this code first landed we hit
        # ``CUBLAS_STATUS_INTERNAL_ERROR`` selecting a B=2 FP8 tactic on
        # RTX 5090; that turned out to depend on input-buffer state at
        # autotune time, not the shape itself, and was masked by simply
        # skipping the pass. After fixing Bugs 5+6 (slot-1 OOB in
        # ``_bias_zero_buf`` and decoder style buffers), the B=2 buffers
        # carry valid post-warmup activations during autotune and the
        # tactic picker no longer trips. Reaching M=B*seq tactics
        # tightens the production-path cosine vs serial CFG by ~1.5%
        # (β=1.5: 0.98 → 0.995) — worth the autotune time.

        # Then, the same set of GEMM shapes at M=B*seq for the batched path.
        Bb = self.bufs
        W = self.weights
        gemm = self.gemm
        nv = self.num_views
        vs = self.vision_seq
        seq = self.encoder_seq_len
        ds = self.chunk_size
        m_vis = self.B * vs
        m_seq = self.B * seq
        m_dec = self.B * ds

        logger.info("Autotuning GEMM algorithms at B=%d M values...", self.B)

        # Vision patch embedding (BF16) at B*vs
        gemm.autotune_bf16_nn(
            Bb["vision_patches_b2"].ptr.value,
            W["vision_patch_embedding_w"],
            Bb["vision_x_b2"].ptr.value,
            m_vis, VIS_D, VIS_PATCH_FLAT)

        # Vision FP8 at B*vs
        if self.use_fp8 and self.fp8_calibrated and "vision_attn_qkv_w_0" in self.weights.get("fp8", {}):
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("vision_attn_qkv_w_0", m_vis, 3 * VIS_D, VIS_D, "vision_QKV_b2"),
                ("vision_attn_o_w_0",   m_vis, VIS_D,     VIS_D, "vision_x_norm_b2"),
                ("vision_ffn_up_w_0",   m_vis, VIS_H,     VIS_D, "vision_hidden_b2"),
                ("vision_ffn_down_w_0", m_vis, VIS_D,     VIS_H, "vision_x_norm_b2"),
                ("vision_projector_w",  m_vis, ENC_D,     VIS_D, "encoder_x_b2"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf = (Bb["vis_act_fp8_large_b2"] if K_val == VIS_H
                           else Bb["vis_act_fp8_b2"])
                self._autotune_fp8_matmul(
                    act_buf.ptr.value, w_fp8_ptr, Bb[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)

        # Encoder FP8 at B*seq
        if self.use_fp8 and self.fp8_calibrated:
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("encoder_attn_qkv_w_0",    m_seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, "encoder_QKV_b2"),
                ("encoder_attn_o_w_0",      m_seq, ENC_D,      ENC_D, "encoder_x_norm_b2"),
                ("encoder_ffn_gate_up_w_0", m_seq, 2 * ENC_H,  ENC_D, "encoder_gate_merged_b2"),
                ("encoder_ffn_down_w_0",    m_seq, ENC_D,      ENC_H, "encoder_x_norm_b2"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf = (Bb["enc_act_fp8_large_b2"] if K_val == ENC_H
                           else Bb["enc_act_fp8_b2"])
                self._autotune_fp8_matmul(
                    act_buf.ptr.value, w_fp8_ptr, Bb[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)

        # Decoder FP8 at B*ds
        if self.use_fp8 and self.use_fp8_decoder and self.fp8_calibrated:
            for name_prefix, M_val, N_val, K_val, out_key in [
                ("decoder_attn_qkv_w_0",    m_dec, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, "decoder_QKV_b2"),
                ("decoder_attn_o_w_0",      m_dec, DEC_D,     DEC_NH * DEC_HD, "x_normed_buf_b2"),
                ("decoder_ffn_gate_up_w_0", m_dec, 2 * DEC_H, DEC_D, "decoder_gate_merged_b2"),
                ("decoder_ffn_down_w_0",    m_dec, DEC_D,     DEC_H, "x_normed_buf_b2"),
            ]:
                w_fp8_ptr, w_scale_ptr = self._weight_fp8(name_prefix)
                act_scale_ptr = self.fp8_act_scales[name_prefix].ptr.value
                act_buf = (Bb["dec_act_fp8_large_b2"] if K_val == DEC_H
                           else Bb["dec_act_fp8_b2"])
                self._autotune_fp8_matmul(
                    act_buf.ptr.value, w_fp8_ptr, Bb[out_key].ptr.value,
                    M_val, N_val, K_val, act_scale_ptr, w_scale_ptr)

        self._cudart.cudaDeviceSynchronize()
        logger.info("B=%d autotune complete", self.B)

    def calibrate_fp8(self) -> None:
        """Calibrate FP8 activation scales using the parent's B=1 pipeline.

        Per-tensor FP8 scales are sample-invariant for Pi0.5 under typical
        observation distributions, so the parent's single-sample
        calibration pass produces scales that apply directly to the B=2
        forward. We run the parent's :meth:`Pi05Pipeline.run_pipeline`
        rather than this subclass's batched override so the calibration
        path uses the parent's B=1 buffers (already pre-populated by
        :meth:`Pi05Pipeline.set_language_embeds`).
        """
        if not self.use_fp8 or self.fp8_calibrated:
            return
        if len(self.fp8_act_scales) > 0:
            self.fp8_calibrated = True
            return
        self.fp8_calibrated = False
        Pi05Pipeline.run_pipeline(self, stream=0)
        self._cudart.cudaDeviceSynchronize()
        self.fp8_calibrated = True
        logger.info("Pi05BatchedPipeline FP8 calibrated via B=1 path: "
                    "%d activation scales", len(self.fp8_act_scales))

    # ══════════════════════════════════════════════════════════════════
    #   Public API: input/output buffer handles for the batched path
    # ══════════════════════════════════════════════════════════════════

    def forward(self) -> int:
        """Replay the captured B=2 graph and return the batched-noise ptr.

        Override of :meth:`Pi05Pipeline.forward` so the returned pointer
        points at the batched diffusion-noise output buffer rather than
        the parent's B=1 slot (which the batched pipeline never writes
        to). The inherited graph replay is otherwise identical.
        """
        if self._graph is not None:
            self._graph.replay(self._graph_stream)
            self._cudart.cudaStreamSynchronize(self._graph_stream)
        else:
            self.run_pipeline(stream=0)
            self._cudart.cudaDeviceSynchronize()
        return self.bufs["diffusion_noise_b2"].ptr.value

    @property
    def input_images_buf_b2(self) -> CudaBuffer:
        """Pipeline input: per-sample observation images (B*nv, 224, 224, 3)."""
        return self.bufs["observation_images_normalized_b2"]

    @property
    def input_noise_buf_b2(self) -> CudaBuffer:
        """Pipeline input/output: per-sample diffusion noise (B*chunk, 32)."""
        return self.bufs["diffusion_noise_b2"]
