"""FlashRT — Pi0.5 RTX pipeline with classifier-free guidance (CFG).

Subclass of :class:`flash_rt.models.pi05.pipeline_rtx.Pi05Pipeline` that
adds per-step CFG inference for advantage-conditioned policies trained
with the RECAP recipe (arXiv:2511.14759, Appendix E).

Per-step CFG runs the action expert twice per denoising step — once
with the conditioned prompt (``"task\\nAdvantage: positive"``) and once
with the unconditioned prompt (``"task"``) — and combines the two
velocity predictions on every step:

    v_guided = v_uncond + beta * (v_cond - v_uncond)

The combined velocity is then accumulated into the diffusion noise
buffer just as the standard pipeline does. The vision encoder is shared
across the two prompts (image features are identical); the Gemma-2B
encoder runs twice (cond / uncond text tokens differ); the Gemma-300M
decoder runs ``2 * num_steps`` times.

This module deliberately does **not** modify the parent pipeline's
behaviour. Standard inference paths (``run_pipeline``,
``transformer_decoder``) on :class:`Pi05Pipeline` are bit-equal to the
pre-CFG implementation; CFG opt-in is at the frontend level by choosing
to instantiate this subclass instead.

Prompt-length handling: the conditioned and unconditioned prompts have
slightly different token counts (the advantage tag adds ~5 tokens). To
share a single CUDA Graph capture we pad both to the same length at
:meth:`set_language_embeds_pair` time (the encoder seq_len is fixed at
graph capture).
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer

# Raw-byte dtype for the encoder K/V cache snapshots — the snapshot is
# a verbatim copy of attn.enc_K bytes, no element-wise math runs on it.
_BYTE = np.uint8

from .pipeline_rtx import (
    ACTION_DIM,
    BF16,
    DEC_D,
    DEC_L,
    ENC_D,
    Pi05Pipeline,
)

logger = logging.getLogger(__name__)


class Pi05CFGPipeline(Pi05Pipeline):
    """Pi05 inference pipeline with classifier-free guidance.

    Args:
        cfg_beta: Guidance strength. Must be ``>= 1.0``. ``1.0`` is
            equivalent to the standard pipeline (one forward per step,
            unconditioned skipped); the frontend should prefer the base
            :class:`Pi05Pipeline` in that case to avoid the extra
            buffers and capture cost. Common deployment range is
            ``[1.5, 2.5]`` per the π*0.6 paper.
        Other arguments are forwarded to :class:`Pi05Pipeline`.
    """

    def __init__(self, *args, cfg_beta: float = 1.5, **kwargs):
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        super().__init__(*args, **kwargs)
        self.cfg_beta = float(cfg_beta)
        self._allocate_cfg_buffers()

        # Set during set_language_embeds_pair; both are padded to the same
        # length so they can share the captured graph's encoder shape.
        self._lang_embeds_buf_cond: Optional[CudaBuffer] = None
        self._lang_embeds_buf_uncond: Optional[CudaBuffer] = None
        self._cfg_prompt_len: Optional[int] = None

        # Calibration mode flag: when set, run_pipeline runs the
        # conditioned prompt only (single forward) so FP8 scale
        # collection sees the standard activation magnitudes.
        self._calibration_mode = False

    # ══════════════════════════════════════════════════════════════════
    #   CFG-specific buffer allocation
    # ══════════════════════════════════════════════════════════════════

    def _allocate_cfg_buffers(self) -> None:
        es = self.encoder_seq_len
        ds = self.chunk_size

        # Cached encoder outputs for the two prompts.
        self.bufs["enc_out_cond"] = CudaBuffer.device_empty(es * ENC_D, BF16)
        self.bufs["enc_out_uncond"] = CudaBuffer.device_empty(es * ENC_D, BF16)

        # Per-step velocity snapshots (decoder action_buf size).
        self.bufs["v_cond"] = CudaBuffer.device_empty(ds * ACTION_DIM, BF16)
        self.bufs["v_uncond"] = CudaBuffer.device_empty(ds * ACTION_DIM, BF16)

        # Saved diffusion noise for the start of each step (so the
        # uncond forward sees the same input as the cond forward).
        self.bufs["noise_step_input"] = CudaBuffer.device_empty(
            ds * ACTION_DIM, BF16)

        # Scratch for CFG combine: holds (v_cond - v_uncond) and then
        # v_uncond + beta * (v_cond - v_uncond) before being accumulated
        # into diffusion_noise.
        self.bufs["cfg_combine_scratch"] = CudaBuffer.device_empty(
            ds * ACTION_DIM, BF16)

        # Encoder K/V cache snapshots — one pair per CFG branch.
        #
        # Why: the parent's encoder writes its layer-wise K/V into the
        # shared ``attn.enc_K`` / ``attn.enc_V`` cache, and the decoder's
        # cross-attention reads from that same cache. The per-layer
        # ``encoder`` last-layer comment is explicit:
        # "encoder output is the K/V cache which the decoder reads"
        # (see pipeline_rtx.py at the end of `_encoder_layer`).
        # Running ``transformer_encoder`` twice (cond, uncond) overwrites
        # the cache: by the time the per-step decoder loop runs, only
        # the *uncond* K/V is left in place, regardless of which branch
        # we are computing. ``encoder_x`` (the residual stream that
        # ``_save/_load_encoder_x_from`` shuffle around) is *not*
        # consumed by the decoder, so swapping it has no effect on
        # cross-attention.
        # Result before this fix: v_cond and v_uncond both used uncond
        # KV, the CFG combine became a near-no-op (cos vs cond-only at
        # beta=1.0 measured 0.95 instead of the mathematically expected
        # ~1.0).
        # Fix: snapshot the full enc_K / enc_V tensor after each encoder
        # pass, then memcpy the appropriate snapshot back into the
        # shared cache at the start of each per-step decoder forward.
        # The decoder writes its own K/V at offset ``[enc_seq:enc_seq+ds]``
        # per layer; we restore the full layer slab and let the decoder
        # overwrite the action-token region as it always does.
        kv_total_bytes = (self._enc_kv_layer_stride
                          * self.attn._num_encoder_layers)
        self.bufs["enc_K_cache_cond"] = CudaBuffer.device_empty(
            kv_total_bytes, _BYTE)  # 1-byte dtype: we treat as raw bytes
        self.bufs["enc_V_cache_cond"] = CudaBuffer.device_empty(
            kv_total_bytes, _BYTE)
        self.bufs["enc_K_cache_uncond"] = CudaBuffer.device_empty(
            kv_total_bytes, _BYTE)
        self.bufs["enc_V_cache_uncond"] = CudaBuffer.device_empty(
            kv_total_bytes, _BYTE)
        self._enc_kv_total_bytes = kv_total_bytes

    # ══════════════════════════════════════════════════════════════════
    #   Public API: prompt embedding pair
    # ══════════════════════════════════════════════════════════════════

    def set_language_embeds_pair(self, cond_embeds_np, uncond_embeds_np) -> None:
        """Store the conditioned and unconditioned prompt embeddings.

        Both arrays must have shape ``(prompt_len, ENC_D)`` with 2-byte
        BF16 elements. They are padded to a common length here so the
        encoder can run on a fixed-shape buffer (CUDA Graph requires
        fixed shapes).

        Args:
            cond_embeds_np: Embeddings for the conditioned prompt
                (``"task\\nAdvantage: positive"``).
            uncond_embeds_np: Embeddings for the unconditioned prompt
                (``"task"``).
        """
        if cond_embeds_np.shape[1] != ENC_D:
            raise ValueError(
                f"cond_embeds last dim must be {ENC_D}, "
                f"got {cond_embeds_np.shape[1]}")
        if uncond_embeds_np.shape[1] != ENC_D:
            raise ValueError(
                f"uncond_embeds last dim must be {ENC_D}, "
                f"got {uncond_embeds_np.shape[1]}")

        cond_len = cond_embeds_np.shape[0]
        uncond_len = uncond_embeds_np.shape[0]
        target_len = max(cond_len, uncond_len)

        if target_len > self.max_prompt_len:
            raise ValueError(
                f"padded prompt length {target_len} exceeds "
                f"max_prompt_len {self.max_prompt_len}")

        cond_padded = self._pad_prompt(cond_embeds_np, target_len)
        uncond_padded = self._pad_prompt(uncond_embeds_np, target_len)

        self._lang_embeds_buf_cond = CudaBuffer.from_numpy(cond_padded)
        self._lang_embeds_buf_uncond = CudaBuffer.from_numpy(uncond_padded)
        self._cfg_prompt_len = target_len

        # Reuse the parent's RoPE setup for the padded length so the
        # decoder cross-attention sees the right number of KV tokens.
        self._set_decoder_rope_for_prompt(target_len)

        # Initial copy into encoder_x using the conditioned prompt
        # (matches calibration / first-pass behaviour).
        self._copy_lang_cond_to_encoder_x()

    @staticmethod
    def _pad_prompt(embeds_np, target_len: int):
        """Right-pad a (L, ENC_D) BF16-bytes array to (target_len, ENC_D)."""
        cur_len = embeds_np.shape[0]
        if cur_len == target_len:
            return np.ascontiguousarray(embeds_np)
        if cur_len > target_len:
            raise ValueError(
                f"current length {cur_len} exceeds target {target_len}")
        # The numpy array carries raw BF16 bytes (uint16-shaped float16);
        # padding with zeros is correct (BF16 zero is all-zero bits).
        pad_shape = (target_len - cur_len, embeds_np.shape[1])
        pad = np.zeros(pad_shape, dtype=embeds_np.dtype)
        return np.ascontiguousarray(np.concatenate([embeds_np, pad], axis=0))

    # ══════════════════════════════════════════════════════════════════
    #   Internal: encoder context swap
    # ══════════════════════════════════════════════════════════════════

    def _copy_lang_cond_to_encoder_x(self, stream: int = 0) -> None:
        """D2D copy of the conditioned prompt into encoder_x[vs:vs+L]."""
        if self._lang_embeds_buf_cond is None:
            return
        start_byte = self.vision_seq * ENC_D * 2
        dst_ptr = self.bufs["encoder_x"].ptr.value + start_byte
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf_cond.ptr,
            self._lang_embeds_buf_cond.nbytes, 3, stream)  # D2D

    def _copy_lang_uncond_to_encoder_x(self, stream: int = 0) -> None:
        """D2D copy of the unconditioned prompt into encoder_x[vs:vs+L]."""
        if self._lang_embeds_buf_uncond is None:
            return
        start_byte = self.vision_seq * ENC_D * 2
        dst_ptr = self.bufs["encoder_x"].ptr.value + start_byte
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf_uncond.ptr,
            self._lang_embeds_buf_uncond.nbytes, 3, stream)  # D2D

    def _save_encoder_x_to(self, dst_buf: CudaBuffer, stream: int) -> None:
        """D2D copy the full encoder_x residual stream into ``dst_buf``."""
        nbytes = self.encoder_seq_len * ENC_D * 2
        self._cudart.cudaMemcpyAsync(
            dst_buf.ptr,
            self.bufs["encoder_x"].ptr,
            nbytes, 3, stream)  # D2D

    def _load_encoder_x_from(self, src_buf: CudaBuffer, stream: int) -> None:
        """D2D copy a saved encoder context back into encoder_x."""
        nbytes = self.encoder_seq_len * ENC_D * 2
        self._cudart.cudaMemcpyAsync(
            self.bufs["encoder_x"].ptr,
            src_buf.ptr,
            nbytes, 3, stream)  # D2D

    # ── Encoder K/V cache snapshot/restore (CFG semantic correctness) ──
    #
    # See ``_allocate_cfg_buffers`` for the why. The encoder K/V cache
    # owned by the attention backend is what the decoder cross-attention
    # actually reads; running the encoder twice (cond, uncond) overwrites
    # it, so we capture it after each pass and restore the right one
    # before each per-step decoder forward.

    def _save_enc_kv_to(self, k_dst: CudaBuffer, v_dst: CudaBuffer,
                         stream: int) -> None:
        nbytes = self._enc_kv_total_bytes
        self._cudart.cudaMemcpyAsync(
            k_dst.ptr,
            ctypes.c_void_p(self._attn_ptrs["enc_K"]),
            nbytes, 3, stream)
        self._cudart.cudaMemcpyAsync(
            v_dst.ptr,
            ctypes.c_void_p(self._attn_ptrs["enc_V"]),
            nbytes, 3, stream)

    def _load_enc_kv_from(self, k_src: CudaBuffer, v_src: CudaBuffer,
                           stream: int) -> None:
        nbytes = self._enc_kv_total_bytes
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(self._attn_ptrs["enc_K"]),
            k_src.ptr,
            nbytes, 3, stream)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(self._attn_ptrs["enc_V"]),
            v_src.ptr,
            nbytes, 3, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Internal: per-step decoder body (extracted from parent without
    #   the noise residual_add, so the caller can do CFG combine first).
    # ══════════════════════════════════════════════════════════════════

    def run_one_decoder_step(self, step: int, stream: int) -> None:
        """Execute one denoising step and leave the velocity in
        ``bufs['decoder_action_buf']``.

        Functionally identical to the body of :meth:`Pi05Pipeline.transformer_decoder`'s
        per-step loop, **except** that the final ``residual_add`` of
        ``decoder_action_buf`` into ``diffusion_noise`` is omitted —
        the CFG combine takes that role instead.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        enc_seq = self.encoder_seq_len
        ds = self.chunk_size
        fused = self.use_fp8_decoder and self.fp8_calibrated

        # C0: noise (ds, 32) → decoder_x (ds, DEC_D)
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

        # C8: Final AdaRMSNorm + output projection → action_buf
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

    # ══════════════════════════════════════════════════════════════════
    #   Internal: CFG combine
    # ══════════════════════════════════════════════════════════════════

    def _cfg_combine_into_noise(self, stream: int) -> None:
        """Compute ``noise += v_uncond + beta * (v_cond - v_uncond)`` in-place.

        v0.2.0 implementation: a single fused CUDA kernel
        (``fvk.cfg_combine_into_residual``) that reads ``v_cond`` and
        ``v_uncond``, applies the affine combine in fp32 internally,
        and accumulates into ``diffusion_noise`` — all in one launch on
        the supplied stream. Graph-capturable.

        Replaces the v0.1.0 torch-eager combine that staged through
        cudaMemcpy + torch ops on the default stream; that path could
        not be captured into a CUDA Graph and added ~1 ms of memcpy
        overhead per denoising step.
        """
        n = self.chunk_size * ACTION_DIM
        self.fvk.cfg_combine_into_residual(
            self.bufs["diffusion_noise"].ptr.value,
            self.bufs["v_cond"].ptr.value,
            self.bufs["v_uncond"].ptr.value,
            self.cfg_beta, n, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Public: CFG-aware run_pipeline
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    #   Per-step velocity / noise trace (oracle / debug only)
    # ══════════════════════════════════════════════════════════════════
    #
    # When ``_trace_enabled`` is True, the per-step decoder loop copies
    # ``v_cond``, ``v_uncond``, and the running diffusion noise into
    # dedicated trace buffers sized for the full ``num_steps`` schedule.
    # This is intended only for offline correctness checks against the
    # FP32 reference (oracle C2/C5 — see ``docs/precision_spec.md``)
    # and the debug helpers in ``tests/``; production inference goes
    # through the captured CUDA Graph, which is recorded with tracing
    # disabled and therefore pays zero runtime cost.

    def enable_velocity_trace(self) -> None:
        """Allocate per-step trace buffers and turn on velocity logging.

        Must be called *before* ``record_infer_graph`` if the caller
        wants the trace path to populate alongside the graph capture
        warmup; for the typical oracle workflow we just call
        ``run_pipeline_eager_with_trace`` directly, bypassing the graph.
        """
        if getattr(self, "_trace_enabled", False):
            return
        ds = self.chunk_size
        n = self.num_steps
        per_step = ds * ACTION_DIM
        self.bufs["trace_v_cond"] = CudaBuffer.device_empty(n * per_step, BF16)
        self.bufs["trace_v_uncond"] = CudaBuffer.device_empty(n * per_step, BF16)
        # noise_per_step[k] = noise BEFORE step k; one extra for final.
        self.bufs["trace_noise"] = CudaBuffer.device_empty(
            (n + 1) * per_step, BF16)
        self._trace_enabled = True

    def _trace_step(self, step: int, stream: int) -> None:
        """Copy current v_cond / v_uncond / pre-step noise into trace slots."""
        if not getattr(self, "_trace_enabled", False):
            return
        ds = self.chunk_size
        nbytes = ds * ACTION_DIM * 2
        slot_off = step * nbytes
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_v_cond"].ptr.value + slot_off),
            self.bufs["v_cond"].ptr, nbytes, 3, stream)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_v_uncond"].ptr.value + slot_off),
            self.bufs["v_uncond"].ptr, nbytes, 3, stream)

    def _trace_noise(self, step: int, stream: int) -> None:
        """Copy current diffusion noise into trace slot ``step``."""
        if not getattr(self, "_trace_enabled", False):
            return
        ds = self.chunk_size
        nbytes = ds * ACTION_DIM * 2
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_noise"].ptr.value + step * nbytes),
            self.bufs["diffusion_noise"].ptr, nbytes, 3, stream)

    def read_velocity_trace(self) -> dict:
        """Download trace buffers to host. Returns numpy float32 arrays.

        Dictionary keys: ``v_cond_per_step`` (num_steps, ds, ACTION_DIM),
        ``v_uncond_per_step`` (num_steps, ds, ACTION_DIM),
        ``noise_per_step`` (num_steps + 1, ds, ACTION_DIM).
        """
        if not getattr(self, "_trace_enabled", False):
            raise RuntimeError(
                "velocity trace not enabled; call enable_velocity_trace() "
                "and use run_pipeline_eager_with_trace()")
        import numpy as np
        ds = self.chunk_size
        n = self.num_steps
        ad = ACTION_DIM
        self._cudart.cudaDeviceSynchronize()

        def _read(buf: CudaBuffer, shape) -> np.ndarray:
            count = int(np.prod(shape))
            arr = np.zeros(count * 2, dtype=np.uint8)
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(arr.ctypes.data),
                buf.ptr, count * 2, 2)
            u16 = arr.view(np.uint16)
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32).reshape(shape)
        return {
            "v_cond_per_step": _read(
                self.bufs["trace_v_cond"], (n, ds, ad)),
            "v_uncond_per_step": _read(
                self.bufs["trace_v_uncond"], (n, ds, ad)),
            "noise_per_step": _read(
                self.bufs["trace_noise"], (n + 1, ds, ad)),
        }

    def run_pipeline_eager_with_trace(self, stream: int = 0) -> None:
        """Eager pipeline forward that populates the velocity trace.

        Bypasses any captured CUDA Graph so the per-step trace
        memcpys (gated on ``_trace_enabled``) execute. Use only from
        offline tests / fixture generators; production inference must
        continue to call ``forward()`` for the captured graph replay.
        """
        if not getattr(self, "_trace_enabled", False):
            raise RuntimeError(
                "call enable_velocity_trace() before "
                "run_pipeline_eager_with_trace()")
        self.run_pipeline(stream=stream)
        self._cudart.cudaDeviceSynchronize()

    def run_pipeline(self, stream: int = 0) -> None:
        """Run the CFG-augmented pipeline end-to-end on ``stream``.

        Phases:
            A. Vision encoder (shared between cond / uncond).
            B. Encoder ×2: once with the conditioned prompt, once with
               the unconditioned prompt. Save both encoder outputs.
            C. Decoder loop: for each of ``num_steps`` denoising steps,
               run the action expert with the cond context, then with
               the uncond context, then combine the two velocities and
               accumulate into the diffusion noise.

        In :attr:`_calibration_mode` (set during :meth:`calibrate_fp8`),
        Phase A runs once with the conditioned prompt and the standard
        single-forward decoder is used so the FP8 calibration pass sees
        the same activation magnitudes as production inference would.
        """
        if self._calibration_mode:
            self._copy_lang_cond_to_encoder_x(stream=stream)
            self.vision_encoder(stream)
            self.transformer_encoder(stream)
            self.transformer_decoder(stream)
            return

        if self._lang_embeds_buf_cond is None or self._lang_embeds_buf_uncond is None:
            raise RuntimeError(
                "set_language_embeds_pair must be called before run_pipeline "
                "on a Pi05CFGPipeline")

        # ── Phase A: vision (shared) ──
        # The vision encoder writes to vision_x and consumes the input
        # images; it does not touch the language slot of encoder_x.
        # We seed encoder_x with the cond prompt before vision so the
        # encoder slot is ready for Phase B's first sub-pass.
        self._copy_lang_cond_to_encoder_x(stream=stream)
        self.vision_encoder(stream)

        # ── Phase B: encoder twice (cond, then uncond) ──
        # NOTE: ``encoder_x`` is the residual stream and is *not* read
        # by the decoder; the decoder cross-attention reads the
        # attention backend's ``enc_K`` / ``enc_V`` cache instead. We
        # therefore snapshot the K/V cache after each encoder pass so
        # the per-step decoder loop can restore the appropriate branch
        # before each forward (without this snapshot, the second
        # encoder pass overwrites the cache and both decoder branches
        # end up running with uncond K/V).
        self.transformer_encoder(stream)
        self._save_encoder_x_to(self.bufs["enc_out_cond"], stream)
        self._save_enc_kv_to(self.bufs["enc_K_cache_cond"],
                              self.bufs["enc_V_cache_cond"], stream)

        # Reset encoder_x: re-run vision + swap to uncond prompt.
        # The vision tokens in encoder_x were modified by the encoder's
        # self-attention residual stream, so we must rerun vision_encoder
        # to restore the vision slot to its post-vision state.
        self.vision_encoder(stream)
        self._copy_lang_uncond_to_encoder_x(stream=stream)
        self.transformer_encoder(stream)
        self._save_encoder_x_to(self.bufs["enc_out_uncond"], stream)
        self._save_enc_kv_to(self.bufs["enc_K_cache_uncond"],
                              self.bufs["enc_V_cache_uncond"], stream)

        # ── Phase C: per-step CFG decoder ──
        # We maintain the diffusion_noise tensor as the residual stream;
        # at the start of each step the same noise feeds both the cond
        # and uncond forwards.
        for step in range(self.num_steps):
            # Trace the noise BEFORE step k (so trace_noise[k] is the input
            # to step k; trace_noise[num_steps] is the final action chunk).
            self._trace_noise(step, stream)
            # Snapshot noise for the uncond forward to see the same input.
            self._save_noise_for_step(stream)

            # cond forward: restore cond K/V cache (the residual-stream
            # load below is kept for parity with the original code path
            # and as a safety net if any future hook reads encoder_x;
            # the cross-attention itself is gated entirely by enc_K/V).
            self._load_enc_kv_from(self.bufs["enc_K_cache_cond"],
                                    self.bufs["enc_V_cache_cond"], stream)
            self._load_encoder_x_from(self.bufs["enc_out_cond"], stream)
            self.run_one_decoder_step(step, stream)
            self._copy_action_buf_to(self.bufs["v_cond"], stream)

            # Restore noise to the start-of-step value before the uncond
            # forward (the cond forward leaves noise unchanged today —
            # we omit residual_add inside run_one_decoder_step — but the
            # snapshot/restore is cheap and future-proofs against any
            # in-place mutation in the decoder body).
            self._restore_noise_for_step(stream)

            # uncond forward: restore uncond K/V cache (see cond branch
            # comment).
            self._load_enc_kv_from(self.bufs["enc_K_cache_uncond"],
                                    self.bufs["enc_V_cache_uncond"], stream)
            self._load_encoder_x_from(self.bufs["enc_out_uncond"], stream)
            self.run_one_decoder_step(step, stream)
            self._copy_action_buf_to(self.bufs["v_uncond"], stream)

            # Trace v_cond / v_uncond BEFORE combine consumes them.
            self._trace_step(step, stream)
            # CFG combine: noise += v_uncond + beta * (v_cond - v_uncond)
            self._cfg_combine_into_noise(stream)
        # Trace final noise (= action chunk) at index num_steps.
        self._trace_noise(self.num_steps, stream)

    # ── Helpers: noise snapshot / action_buf save ──

    def _save_noise_for_step(self, stream: int) -> None:
        ds = self.chunk_size
        nbytes = ds * ACTION_DIM * 2
        self._cudart.cudaMemcpyAsync(
            self.bufs["noise_step_input"].ptr,
            self.bufs["diffusion_noise"].ptr,
            nbytes, 3, stream)

    def _restore_noise_for_step(self, stream: int) -> None:
        ds = self.chunk_size
        nbytes = ds * ACTION_DIM * 2
        self._cudart.cudaMemcpyAsync(
            self.bufs["diffusion_noise"].ptr,
            self.bufs["noise_step_input"].ptr,
            nbytes, 3, stream)

    def _copy_action_buf_to(self, dst_buf: CudaBuffer, stream: int) -> None:
        ds = self.chunk_size
        nbytes = ds * ACTION_DIM * 2
        self._cudart.cudaMemcpyAsync(
            dst_buf.ptr,
            self.bufs["decoder_action_buf"].ptr,
            nbytes, 3, stream)

    # ══════════════════════════════════════════════════════════════════
    #   FP8 calibration: single forward (no CFG) so scales match
    #   production activation magnitudes.
    # ══════════════════════════════════════════════════════════════════

    def calibrate_fp8(self) -> None:
        """Calibrate FP8 activation scales using the conditioned prompt only.

        CFG is symmetric in the encoder/decoder weight scales; calibrating
        on either prompt gives equivalent results because the activation
        magnitudes are dominated by the (shared) image and base task
        tokens, not by the few advantage tag tokens. We pick the cond
        prompt for the calibration pass.
        """
        if self._lang_embeds_buf_cond is None:
            raise RuntimeError(
                "set_language_embeds_pair must be called before calibrate_fp8")
        was = self._calibration_mode
        self._calibration_mode = True
        try:
            super().calibrate_fp8()
        finally:
            self._calibration_mode = was

    # CUDA Graph capture for the CFG path: inherited from
    # :meth:`Pi05Pipeline.record_infer_graph`. The captured graph runs
    # ``self.run_pipeline`` (CFG flow: vision once + encoder ×2 +
    # per-step (decoder ×2 + cfg_combine kernel)). All ops in
    # :meth:`run_pipeline` are kernel launches via the fvk binding; the
    # combine kernel landed in v0.2.0 replaced the torch-eager combine
    # that previously made the flow uncapturable.
