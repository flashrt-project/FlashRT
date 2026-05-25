"""FlashRT — Pi0.5 RTX CFG inference fused into a single B=2 batched forward.

Single-inheritance subclass of
:class:`flash_rt.models.pi05.pipeline_rtx_batched.Pi05BatchedPipeline`
that reuses the sample-batched vision / encoder / decoder infrastructure
to run classifier-free guidance's conditioned + unconditioned branches
as the two slots of a single B=2 forward. Each denoising step fuses
the combine with a single ``cfg_combine_into_residual`` kernel call
(the same kernel Phase 2 wired for the serial CFG pipeline).

This stacks cleanly on top of two earlier pieces:

  * Phase 1 (serial CFG) — :class:`Pi05CFGPipeline` — prompt-pair
    builder, per-step combine via the cfg_combine kernel, graph capture
  * Phase 3a (generic batched) — :class:`Pi05BatchedPipeline` —
    sample-batched buffers, attention, FP8 scratch, KV cache

Phase 3b's only job is to instruct the batched pipeline to treat slot 0
as the conditioned pass and slot 1 as the unconditioned pass, then
apply the CFG combine on the two per-step velocity outputs.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer

from .pipeline_rtx import ACTION_DIM, BF16, DEC_D, DEC_L, ENC_D, Pi05Pipeline
from .pipeline_rtx_batched import Pi05BatchedPipeline

logger = logging.getLogger(__name__)


class Pi05CFGBatchedPipeline(Pi05BatchedPipeline):
    """B=2-fused classifier-free guidance pipeline for pi0.5 RTX.

    Slot 0 runs the conditioned prompt (``"task\\nAdvantage: positive"``);
    slot 1 runs the unconditioned prompt (``"task"``). Each denoising
    step runs the batched decoder once (producing velocity for both
    slots in parallel) and combines them into the conditioned slot's
    noise buffer via the ``cfg_combine_into_residual`` kernel:

        noise_cond += v_uncond + beta * (v_cond - v_uncond)

    Since the conditioned slot is the one the user actually reads back,
    the uncond slot's noise does not need to be updated — we only care
    about the combined output in slot 0. (The batched decoder still
    produces both slots' velocity this step, but the cond slot is what
    flows forward for the next step and into the final action decode.)

    Args:
        cfg_beta: CFG guidance strength. Must be ``>= 1.0``. Passed in
            at construction time; to change at runtime the frontend
            must rebuild the pipeline (same contract as
            :class:`Pi05CFGPipeline`).
        Other args forwarded to :class:`Pi05BatchedPipeline`.
    """

    # Slot assignment inside the batched pair. Fixed for v0.1.0 so the
    # frontend, the CFG combine kernel call, and the action readback
    # all agree.
    COND_SLOT = 0
    UNCOND_SLOT = 1

    # The π0.6 RECAP paper recommends β ∈ [1.5, 2.5] as the "moderate"
    # CFG range (Appendix E); the production batched path now meets
    # the ≥0.99 cosine release floor against the FP32 reference across
    # the entire range after the enc_Q stride fix. Measured cosine vs
    # FP32 R_fp32 reference:
    #   β=1.0 → 0.9996  (paper default; CFG collapses to v_cond)
    #   β=1.5 → 0.9989
    #   β=2.0 → 0.9977
    #   β=2.5 → 0.9958
    # No path-selection guidance is required from the constructor —
    # callers can use any β in the paper's recommended range.

    def __init__(self, *args, cfg_beta: float = 1.5, **kwargs):
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        super().__init__(*args, **kwargs)
        self.cfg_beta = float(cfg_beta)

    # ══════════════════════════════════════════════════════════════════
    #   FP8 calibration: B=2 joint pass over (cond, uncond) contexts
    # ══════════════════════════════════════════════════════════════════

    def calibrate_fp8(self) -> None:
        """Calibrate FP8 activation scales using a B=2 joint forward.

        The parent ``Pi05BatchedPipeline.calibrate_fp8`` runs a B=1
        cond-only pass and reuses those scales for the batched runtime.
        That works for the generic batched path (where slot 1 in
        production also typically holds the same kind of obs as slot 0)
        but is suboptimal for CFG: at runtime slot 1 carries the
        unconditioned prompt, whose lang-token activations live in a
        slightly different per-tensor magnitude range than the cond
        prompt's. Calibrating on cond alone leaves the static scales
        clipping uncond's tails — which is the dominant source of the
        residual ~2% cosine gap vs serial CFG at ``beta=1.5``.

        Override: drive the **batched** ``run_pipeline`` for one warmup
        forward with cond in slot 0 and uncond in slot 1. The
        dynamic-quant ``_fp8_gemm_b2`` path then writes a per-layer
        scale based on the max-abs across all ``B*seq`` rows — covering
        both contexts simultaneously, which is exactly the magnitude
        envelope the static-scale fused kernels see at production time.

        We stage the same image+noise into both B=2 input slots from
        the parent's B=1 staging buffers (the frontend pre-populated
        those for its own setup); CFG runs both branches against the
        same image and the same noise per-step, so seeding both slots
        identically here is faithful to the production input pattern.
        """
        if not self.use_fp8 or self.fp8_calibrated:
            return
        if len(self.fp8_act_scales) > 0:
            self.fp8_calibrated = True
            return
        if (self._lang_embeds_buf_b2[self.COND_SLOT] is None
                or self._lang_embeds_buf_b2[self.UNCOND_SLOT] is None):
            raise RuntimeError(
                "Pi05CFGBatchedPipeline.calibrate_fp8: prompt pair must "
                "be set via set_language_embeds_pair() before calibration")

        # Two-pass calibration: run the parent's B=1 forward once with
        # the cond prompt, capture scales; then again with the uncond
        # prompt, take the elementwise max. The parent's
        # ``Pi05Pipeline.run_pipeline`` reads its B=1 ``encoder_x`` lang
        # slot which the frontend has already populated with cond bytes,
        # so the first pass is a drop-in invocation of the standard
        # calibration path. For the second pass we swap in the uncond
        # prompt bytes via D2D copy from the B=2 lang buffer.
        ENC_D_BYTES = ENC_D * 2  # bf16

        def _swap_parent_lang_to(slot: int) -> None:
            src = self._lang_embeds_buf_b2[slot]
            if src is None:
                return
            dst_lang = (self.bufs["encoder_x"].ptr.value
                        + self.vision_seq * ENC_D_BYTES)
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_lang), src.ptr, src.nbytes, 3)
            self._cudart.cudaDeviceSynchronize()

        # Pass 1: cond — produces fp8_act_scales for cond-context magnitudes.
        # The frontend has already seeded encoder_x with cond bytes via
        # ``Pi05Pipeline.set_language_embeds(cond_np)``; do the explicit
        # swap anyway for robustness against future refactors.
        _swap_parent_lang_to(self.COND_SLOT)
        self.fp8_calibrated = False
        Pi05Pipeline.run_pipeline(self, stream=0)
        self._cudart.cudaDeviceSynchronize()
        cond_scales: dict[str, float] = {}
        for name, buf in self.fp8_act_scales.items():
            arr = np.zeros(1, dtype=np.float32)
            self._cudart.cudaMemcpy(arr.ctypes.data, buf.ptr, 4, 2)
            cond_scales[name] = float(arr[0])

        # Pass 2: uncond — overwrites fp8_act_scales with uncond
        # magnitudes. We then merge by taking the max of the two passes
        # per layer so the static scale covers both prompt distributions.
        _swap_parent_lang_to(self.UNCOND_SLOT)
        Pi05Pipeline.run_pipeline(self, stream=0)
        self._cudart.cudaDeviceSynchronize()
        merged_count = 0
        for name, buf in self.fp8_act_scales.items():
            arr = np.zeros(1, dtype=np.float32)
            self._cudart.cudaMemcpy(arr.ctypes.data, buf.ptr, 4, 2)
            uncond_val = float(arr[0])
            cond_val = cond_scales.get(name, 0.0)
            if cond_val > uncond_val:
                arr[0] = cond_val
                self._cudart.cudaMemcpy(buf.ptr, arr.ctypes.data, 4, 1)
                merged_count += 1
        self._cudart.cudaDeviceSynchronize()

        # Restore parent's B=1 lang slot to cond so any downstream code
        # path (e.g. Pi05Pipeline-style calibration data dumps) sees the
        # same bytes the frontend originally set.
        _swap_parent_lang_to(self.COND_SLOT)
        self.fp8_calibrated = True
        logger.info(
            "Pi05CFGBatchedPipeline FP8 calibrated via 2x B=1 (cond, uncond) "
            "passes: %d activation scales (%d cond-dominated)",
            len(self.fp8_act_scales), merged_count)

    # ══════════════════════════════════════════════════════════════════
    #   Public API: prompt pair -> batched slot 0/1
    # ══════════════════════════════════════════════════════════════════

    def set_language_embeds_pair(self, cond_embeds_np, uncond_embeds_np) -> None:
        """Upload the conditioned + unconditioned prompts into slot 0 / slot 1.

        Args:
            cond_embeds_np: ``(prompt_len, ENC_D)`` bf16-bytes numpy for
                the conditioned prompt (advantage tag appended).
            uncond_embeds_np: same shape for the unconditioned prompt
                (no advantage tag). Both arrays must already be padded
                to a common length — the caller (frontend) is the
                natural place to pad, since it knows the tokenised
                lengths.
        """
        pair = [None, None]
        pair[self.COND_SLOT] = cond_embeds_np
        pair[self.UNCOND_SLOT] = uncond_embeds_np
        # Pi05BatchedPipeline.set_language_embeds_batch expects a list
        # of length B; set_prompt on the frontend has already validated
        # that both arrays share a prompt length.
        self.set_language_embeds_batch(pair)

    # ══════════════════════════════════════════════════════════════════
    #   Decoder override: run batched step, then apply CFG combine
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    #   Per-step velocity / noise trace (oracle / debug only)
    # ══════════════════════════════════════════════════════════════════
    #
    # Mirrors :class:`Pi05CFGPipeline`'s trace machinery. Production
    # inference (``forward()`` → captured CUDA Graph) is unaffected
    # when tracing is off; with tracing on, callers must use
    # ``run_pipeline_eager_with_trace()`` so the eager Python control
    # flow's ``if self._trace_enabled`` branches actually fire.

    def enable_velocity_trace(self) -> None:
        if getattr(self, "_trace_enabled", False):
            return
        ds = self.chunk_size
        n = self.num_steps
        per_slot = ds * ACTION_DIM
        self.bufs["trace_v_cond_b2"] = CudaBuffer.device_empty(
            n * per_slot, BF16)
        self.bufs["trace_v_uncond_b2"] = CudaBuffer.device_empty(
            n * per_slot, BF16)
        # noise_per_step: per-slot, only cond slot is meaningful (mirror
        # makes uncond slot identical to cond at every step boundary).
        self.bufs["trace_noise_b2"] = CudaBuffer.device_empty(
            (n + 1) * per_slot, BF16)
        self._trace_enabled = True

    def _trace_step_batched(self, step: int, v_cond_ptr: int,
                             v_uncond_ptr: int, stream: int) -> None:
        if not getattr(self, "_trace_enabled", False):
            return
        per_slot_bytes = self.chunk_size * ACTION_DIM * 2
        slot_off = step * per_slot_bytes
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_v_cond_b2"].ptr.value + slot_off),
            ctypes.c_void_p(v_cond_ptr),
            per_slot_bytes, 3, stream)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_v_uncond_b2"].ptr.value + slot_off),
            ctypes.c_void_p(v_uncond_ptr),
            per_slot_bytes, 3, stream)

    def _trace_noise_batched(self, step: int, stream: int) -> None:
        if not getattr(self, "_trace_enabled", False):
            return
        per_slot_bytes = self.chunk_size * ACTION_DIM * 2
        cond_noise_ptr = (self.bufs["diffusion_noise_b2"].ptr.value
                          + self.COND_SLOT * per_slot_bytes)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(
                self.bufs["trace_noise_b2"].ptr.value + step * per_slot_bytes),
            ctypes.c_void_p(cond_noise_ptr),
            per_slot_bytes, 3, stream)

    def read_velocity_trace(self) -> dict:
        if not getattr(self, "_trace_enabled", False):
            raise RuntimeError(
                "velocity trace not enabled; call enable_velocity_trace() "
                "and use run_pipeline_eager_with_trace()")
        import numpy as np
        ds = self.chunk_size
        n = self.num_steps
        ad = ACTION_DIM
        self._cudart.cudaDeviceSynchronize()

        def _read(buf, shape):
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
                self.bufs["trace_v_cond_b2"], (n, ds, ad)),
            "v_uncond_per_step": _read(
                self.bufs["trace_v_uncond_b2"], (n, ds, ad)),
            "noise_per_step": _read(
                self.bufs["trace_noise_b2"], (n + 1, ds, ad)),
        }

    def run_pipeline_eager_with_trace(self, stream: int = 0) -> None:
        if not getattr(self, "_trace_enabled", False):
            raise RuntimeError(
                "call enable_velocity_trace() before "
                "run_pipeline_eager_with_trace()")
        self.run_pipeline(stream=stream)
        self._cudart.cudaDeviceSynchronize()

    def transformer_decoder_batched(self, stream: int = 0) -> None:
        """Run the diffusion denoise loop with per-step CFG combine.

        The per-step body matches
        :meth:`Pi05BatchedPipeline.transformer_decoder_batched` exactly
        until the point where the batched action delta is accumulated
        into the diffusion noise. Instead of the plain ``residual_add``
        there, we pull out the per-slot velocities, call the fused
        ``cfg_combine_into_residual`` kernel on the conditioned slot's
        noise, and keep the unconditioned slot in sync so the next
        step's decoder forwards both contexts from the correct noise.
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        Bb = self.bufs
        enc_seq = self.encoder_seq_len
        ds = self.chunk_size
        m = self.B * ds
        fused = self.use_fp8_decoder and self.fp8_calibrated

        for step in range(self.num_steps):
            # Trace pre-step noise (oracle / debug only).
            self._trace_noise_batched(step, stream)
            # C0: Action input projection (same as parent)
            gemm.bf16_nn(
                Bb["diffusion_noise_b2"].ptr.value,
                W["decoder_action_in_proj_w"],
                Bb["decoder_x_b2"].ptr.value,
                m, DEC_D, ACTION_DIM, stream=stream)
            self._bias_add_bf16(
                Bb["decoder_x_b2"].ptr.value, W["decoder_action_in_proj_b"],
                m, DEC_D, stream)

            # 18 decoder layers (inherited body)
            for i in range(DEC_L):
                skip_c1 = fused and i > 0
                self._decoder_layer_batched(i, step, enc_seq, ds, m,
                                             skip_c1, stream)

            # C8: final AdaRMSNorm + output projection → decoder_action_buf
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

            # CFG combine:
            #   noise[cond]   += v_uncond + beta * (v_cond - v_uncond)
            #   noise[uncond] := noise[cond]   (both branches must read
            #                                    the SAME guided noise
            #                                    at step k+1, per
            #                                    arXiv:2511.14759 App. E
            #                                    and Pi05CFGPipeline's
            #                                    snapshot/restore flow —
            #                                    serial CFG feeds the
            #                                    same N_k into both
            #                                    forwards each step)
            # The batched decoder_action_buf_b2 holds both slots' velocity
            # contiguous in the M dim: slot 0 first (ds * ACTION_DIM
            # elements), then slot 1.
            per_slot_n = ds * ACTION_DIM
            elt = 2  # bf16 = 2 bytes
            per_slot_bytes = per_slot_n * elt
            v_cond_ptr = (Bb["decoder_action_buf_b2"].ptr.value
                          + self.COND_SLOT * per_slot_bytes)
            v_uncond_ptr = (Bb["decoder_action_buf_b2"].ptr.value
                            + self.UNCOND_SLOT * per_slot_bytes)
            noise_cond_ptr = (Bb["diffusion_noise_b2"].ptr.value
                              + self.COND_SLOT * per_slot_bytes)
            noise_uncond_ptr = (Bb["diffusion_noise_b2"].ptr.value
                                + self.UNCOND_SLOT * per_slot_bytes)
            # Trace v_cond / v_uncond BEFORE combine consumes them
            # (oracle / debug only).
            self._trace_step_batched(step, v_cond_ptr, v_uncond_ptr, stream)
            # Guided update on the cond slot (uses fused kernel).
            fvk.cfg_combine_into_residual(
                noise_cond_ptr, v_cond_ptr, v_uncond_ptr,
                self.cfg_beta, per_slot_n, stream)
            # Mirror the guided noise into the uncond slot so both slots
            # enter step k+1 with identical input — matches the
            # save/restore contract of Pi05CFGPipeline. Without this the
            # uncond trajectory drifts (plain-Euler vs guided) and the
            # cosine vs serial CFG degrades from 1.0 to ~0.97 over 10
            # denoising steps.
            # cudaMemcpyDeviceToDevice = 3; graph-capturable.
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(noise_uncond_ptr),
                ctypes.c_void_p(noise_cond_ptr),
                per_slot_bytes, 3, stream)
        # Trace final noise (= action chunk) at index num_steps.
        self._trace_noise_batched(self.num_steps, stream)

    # ══════════════════════════════════════════════════════════════════
    #   Output routing: action readback comes from the cond slot
    # ══════════════════════════════════════════════════════════════════

    def forward(self) -> int:
        """Replay the captured B=2 CFG graph; return the cond slot pointer.

        The batched pipeline runs both slots' decoder per step, but the
        CFG user only wants the combined cond-slot noise as actions.
        Point the caller at slot 0 of ``diffusion_noise_b2``.
        """
        if self._graph is not None:
            self._graph.replay(self._graph_stream)
            self._cudart.cudaStreamSynchronize(self._graph_stream)
        else:
            self.run_pipeline(stream=0)
            self._cudart.cudaDeviceSynchronize()
        per_slot_bytes = self.chunk_size * ACTION_DIM * 2
        return (self.bufs["diffusion_noise_b2"].ptr.value
                + self.COND_SLOT * per_slot_bytes)
