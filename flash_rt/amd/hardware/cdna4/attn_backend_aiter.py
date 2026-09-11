"""FlashRT AMD — CDNA4 attention backend v2: aiter asm flash-attention.

Drop-in replacement for :class:`Cdna4AttnBackend` that dispatches the
per-site attention math to ``aiter`` (AMD's asm flash-attention library,
the vLLM-ROCm production attention path) instead of torch SDPA.

Construction signature, buffer ownership/shapes/dtypes, ``get_ptrs()``,
``run()`` dispatch and the fixed-shape surface are IDENTICAL to the sdpa
backend (this class subclasses it and overrides only the three per-site
attention calls), so the frontend can swap backends via a single
parameter without touching the pipeline.

WHY AITER
---------
rocprof on the FP8 graph shows sdpa's attn_fwd at ~44us/call x ~225
calls = ~11.1ms/inference = 32.6% of GPU time — the #1 kernel bucket.
Isolated-call survey (bench_amd_attention.py) on the exact pi05 shapes:

    vision   aiter 25.0us  vs sdpa 34.7us
    encoder  aiter 72.9us  vs sdpa 122.7us
    decoder  aiter 68.0us  vs sdpa  62.4us   (sdpa wins tiny-M)

CALL CONTRACT
-------------
``aiter.mha_fwd`` takes (B, S, H, D) tensors and supports GQA natively
(K/V may carry a single KV head against 8 Q heads — no expand, no
materialized repeat). Our owned buffers already match that layout as
zero-copy views:

    vision   Q/K/V (nv, 256, 16, 72)         — used directly (B=nv)
    encoder  Q (seq, 8, 256).unsqueeze(0), K/V (seq, 1, 256).unsqueeze(0)
    decoder  Q (chunk, 8, 256).unsqueeze(0), K/V (total_kv, 1, 256).unsqueeze(0)

All slices are along leading dims of contiguous buffers, so every view
handed to aiter is contiguous. ``softmax_scale = 1/sqrt(head_dim)`` is
passed explicitly (matches the RTX FA2 default and the sdpa backend).
``window_size_left = window_size_right = -1`` disables local windowing
(FA2-family "no window" sentinel); ``is_causal=False`` — all Pi0.5
sites are bidirectional.

POINTER STABILITY / CAPTURE SAFETY
----------------------------------
``mha_fwd(..., out=...)`` writes the result into a caller-provided
tensor, so each site passes a view of the SAME pre-allocated O buffer
the sdpa backend owns — the returned pointer is stable across graph
replays and the sdpa backend's final ``copy_`` disappears entirely.

``return_softmax_lse=False`` / ``return_dropout_randval=False`` are
requested so no result tensors need to be returned, but aiter may still
allocate the LSE (and any workspace) internally on every call. Those
per-call allocations go through torch's caching allocator, so they rely
on the SAME warmup-then-capture contract the interim sdpa path already
uses: the pipeline runs 3 warmup iterations on the capture stream before
``hipStreamBeginCapture`` so the allocator reaches steady state and
in-capture allocations are served from cached blocks. The dedicated
capture gate in the dev tests (test_amd_attn_aiter.py) verifies this
holds — it is the make-or-break check for this backend.

Like the sdpa backend, aiter launches on torch's CURRENT stream; the
frontend wraps calibration + capture + replay in
``torch.cuda.stream(...)`` so all work lands on the capture stream.

FIXED-SHAPE MODE
----------------
Masking by slicing is NOT an option in-graph (the valid length changes
without recapture), so each site needs a pointer-read mechanism:

- decoder: the hand-written split-KV kernel takes the FA2-style
  ``seqused`` DEVICE pointer natively (``dec_seqused``, maintained by
  :meth:`set_fixed_valid_len`), so the custom route — including the
  fused FP8-out epilogue — serves fixed mode directly at full speed.
- encoder: aiter's dense ``mha_fwd`` has no runtime mask input
  (varlen/``cu_seqlens`` would need device-read offsets), so this site
  delegates to the sdpa base class, whose boolean key mask is read by
  pointer at replay. This is the remaining fixed-mode overhead.
- vision: no mask in either mode; always aiter.

Exact-shape mode (one graph per prompt length) runs fully on
aiter + the custom decoder kernel.

CALIBRATION DETERMINISM
-----------------------
aiter's asm attention is nondeterministic ACROSS PROCESSES (graph
replay within a process stays bit-stable). FP8 amax scales collected
while it runs therefore jitter run-to-run — observed as an exact-mode
cos-vs-reference band of ~0.997-0.999 versus ~0.9994+ on fully
deterministic arms. During the calibration window (``set_calibrating``,
driven by the pipeline's pre-calibration forwards) the ENCODER site
routes to the deterministic MFMA flash kernel; vision stays on aiter
(cleared as a jitter source by the deterministic fixed-mode arms) and
the custom decoder kernel is deterministic already. Escape hatch:
``FVK_AMD_CALIB_DET_ATTN=off``.
"""

from __future__ import annotations

import math

from flash_rt.amd.hardware.cdna4.attn_backend import Cdna4AttnBackend


def _resolve_mha_fwd():
    """Import aiter and resolve its ``mha_fwd`` entry point.

    Raises ImportError with a clear message when aiter is unavailable so
    the frontend can fall back to the sdpa backend.
    """
    try:
        import aiter  # noqa: F401
    except ImportError as ex:
        raise ImportError(
            "Cdna4AiterAttnBackend requires the 'aiter' package "
            "(AMD asm flash-attention); import failed — install aiter "
            "or use the sdpa backend (FVK_AMD_ATTN=sdpa)."
        ) from ex
    fn = getattr(aiter, "mha_fwd", None)
    if fn is None:
        # Older/newer layouts keep mha_fwd under aiter.ops.mha.
        try:
            from aiter.ops.mha import mha_fwd as fn  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "aiter imported but exposes no 'mha_fwd' (checked "
                "aiter.mha_fwd and aiter.ops.mha.mha_fwd) — aiter "
                "version mismatch; use the sdpa backend "
                "(FVK_AMD_ATTN=sdpa)."
            ) from ex
    return fn


class Cdna4AiterAttnBackend(Cdna4AttnBackend):
    """Pi0.5 attention backend for AMD CDNA4 dispatching to aiter mha_fwd.

    Same construction signature and consumed surface as
    :class:`Cdna4AttnBackend`; only the three per-site attention calls
    are overridden. See the module docstring for the call contract,
    capture-safety notes and the fixed-shape delegation policy.
    """

    def __init__(self, num_views: int, encoder_seq_max: int, chunk_size: int,
                 num_encoder_layers: int = 18, dtype=None):
        super().__init__(num_views, encoder_seq_max, chunk_size,
                         num_encoder_layers=num_encoder_layers, dtype=dtype)
        # Resolve at construction so an unusable environment fails fast
        # (the frontend catches ImportError and falls back to sdpa).
        self._mha_fwd = _resolve_mha_fwd()
        # Explicit softmax scales (mha_fwd has no "default scale" arg on
        # the positional signature; match FA2's 1/sqrt(head_dim)).
        self._vis_scale = 1.0 / math.sqrt(self.vis_Q.shape[-1])   # D=72
        self._enc_scale = 1.0 / math.sqrt(self.enc_Q.shape[-1])   # D=256

        # Optional hand-written decoder attention (csrc/amd/attention/
        # decoder_flash.hip). FVK_AMD_DEC_ATTN=custom routes the decoder
        # site to it; default "lib" keeps the aiter/sdpa library path.
        # Workspace is pre-allocated once (pointer-stable, capture-safe);
        # size per the kernel header: MAX_SPLIT * Hq * Sq * (D + 2) floats.
        import os
        self._dec_custom = (
            os.environ.get("FVK_AMD_DEC_ATTN", "custom").strip().lower()
            == "custom")
        if self._dec_custom:
            from flash_rt.amd import flash_rt_amd_kernels as _fvk
            if not hasattr(_fvk, "attention_decoder_gqa"):
                raise ImportError(
                    "FVK_AMD_DEC_ATTN=custom but attention_decoder_gqa "
                    "is not in flash_rt_amd_kernels")
            self._fvk = _fvk
            hq, d = self.dec_Q.shape[-2], self.dec_Q.shape[-1]
            ws_floats = 32 * hq * chunk_size * (d + 2)
            self._dec_attn_ws = self._torch.empty(
                ws_floats, dtype=self._torch.float32, device="cuda")

        # Fixed-mode encoder attention: the MFMA flash kernel with the
        # seqused device pointer (default when available and D == 256);
        # FVK_AMD_FIXED_ENC_ATTN=sdpa keeps the masked-sdpa base path.
        from flash_rt.amd import flash_rt_amd_kernels as _fvk_ef
        self._fvk_ef = _fvk_ef
        self._enc_flash_ok = (hasattr(_fvk_ef, "encoder_attention_flash")
                              and self.enc_Q.shape[-1] == 256)
        self._fixed_enc_flash = (
            os.environ.get("FVK_AMD_FIXED_ENC_ATTN", "flash").strip().lower()
            == "flash" and self._enc_flash_ok)

        # Calibration-window determinism: aiter's asm attention is
        # nondeterministic ACROSS PROCESSES (replay within a process is
        # bit-stable), so encoder activations — and therefore the FP8
        # amax scales collected during calibration — jitter from run to
        # run. While set_calibrating(True) is active, the encoder site
        # routes to the deterministic MFMA flash kernel instead, pinning
        # the scales; capture/replay then run the fast aiter path with
        # frozen scales. FVK_AMD_CALIB_DET_ATTN=off restores the old
        # behaviour (calibrate on aiter). The vision site stays on aiter
        # in both windows: the deterministic fixed-mode arms measured a
        # tight cos band with aiter vision, clearing it as a jitter
        # source; the custom decoder kernel is already deterministic.
        self._calib_det = (
            os.environ.get("FVK_AMD_CALIB_DET_ATTN", "flash").strip().lower()
            != "off" and self._enc_flash_ok)

        # Fused FP8-out epilogue state (see set_decoder_fp8out). Disarmed
        # until the pipeline arms it post-calibration.
        self._dec_fp8out_scales = None   # list[list[int]] [step][layer] or None
        self._dec_fp8out_step = 0        # host-side step selector (capture-time)
        self._dec_fp8out_buf = None      # (chunk, Hq*D) fp8-byte tensor, reused
        self._dec_fp8out_last = None     # (fp8_ptr, scale_ptr) of last call

    # ── Fused FP8-out epilogue (decoder custom route only) ──
    #
    # The pipeline's decoder attn-output -> o-proj site quantizes the
    # attention output with a per-weight STATIC scale before the FP8
    # GEMM (~180 standalone quantize launches / inference). The custom
    # decoder kernel can emit those exact fp8 bytes as an in-kernel
    # epilogue instead. Protocol note: decoder_attn/run keep their
    # signatures — arming is a separate one-shot call, and the fp8
    # result is picked up out-of-band via get_decoder_fp8out_ptr().

    def decoder_fp8out_supported(self) -> bool:
        """Whether the decoder site can produce fused fp8-out bytes."""
        return (self._dec_custom
                and getattr(self, "_fvk", None) is not None
                and hasattr(self._fvk, "attention_decoder_gqa_fp8out"))

    def set_decoder_fp8out(self, layer_scale_ptrs) -> None:
        """Arm (or disarm) the fused FP8-out decoder epilogue.

        ``layer_scale_ptrs``: ``None`` disarms; a ``list[int]`` of
        per-layer device float pointers (the o-proj static activation
        scales) arms every step with the same per-layer scales; a
        ``list[list[int]]`` indexed ``[step][layer]`` arms with the
        pipeline's per-(step, layer) static scales — required for
        bit-equality when the calibration recorded per-step scales
        (select the step via :meth:`set_decoder_fp8out_step`).

        Called ONCE by the pipeline after calibration (capture prep).
        Allocates the single reused (chunk, Hq*D) fp8-byte output
        tensor on first arm — pointer-stable across replays.
        """
        if layer_scale_ptrs is None:
            self._dec_fp8out_scales = None
            self._dec_fp8out_last = None
            return
        if not self.decoder_fp8out_supported():
            raise RuntimeError(
                "set_decoder_fp8out: custom decoder route with "
                "attention_decoder_gqa_fp8out is not available "
                "(FVK_AMD_DEC_ATTN != custom or symbol missing)")
        first = layer_scale_ptrs[0]
        if isinstance(first, (list, tuple)):
            mat = [list(row) for row in layer_scale_ptrs]
        else:
            mat = [list(layer_scale_ptrs)]
        self._dec_fp8out_scales = mat
        self._dec_fp8out_step = 0
        self._dec_fp8out_last = None
        if self._dec_fp8out_buf is None:
            chunk, hq, d = self.dec_Q.shape
            self._dec_fp8out_buf = self._torch.empty(
                chunk, hq * d, dtype=self._torch.uint8, device="cuda")

    def set_decoder_fp8out_step(self, step: int) -> None:
        """Select the denoise step whose scale row decoder_attn uses.

        Host-side only (the pipeline calls it at the top of each step
        while recording — the chosen pointers are frozen into the
        graph at capture, exactly like the o-proj quantize's scale
        pointer today). No-op when disarmed.
        """
        self._dec_fp8out_step = int(step)

    def get_decoder_fp8out_ptr(self):
        """(fp8_ptr, scale_ptr) of the LAST decoder_attn call, or None.

        None when disarmed, or when the last decoder_attn did not run
        the fused fp8-out kernel (fixed-shape delegation / lib route)
        — the caller must then fall back to its own quantize path.
        """
        return self._dec_fp8out_last

    # ── aiter dispatch ──

    def _aiter_attn(self, q, k, v, out, scale):
        """mha_fwd with the pipeline's fixed contract: (B, S, H, D)
        layout, no dropout, bidirectional (no causal mask, no window),
        native GQA (Hkv may be 1), result written in-place into ``out``
        (a view of the pre-allocated, pointer-stable O buffer)."""
        self._mha_fwd(
            q, k, v,
            0.0,          # dropout_p
            scale,        # softmax_scale (explicit 1/sqrt(head_dim))
            False,        # is_causal
            -1,           # window_size_left  (no local window)
            -1,           # window_size_right (no local window)
            False,        # return_softmax_lse
            False,        # return_dropout_randval
            out=out,
        )

    # ── Attention calls (override sdpa math; same pointers returned) ──

    def vision_attn(self, stream: int = 0) -> int:
        # Buffers are (nv, 256, 16, 72) = (B, S, H, D) already — no
        # views needed on either side. Vision has no mask in fixed-shape
        # mode either, so this site always runs on aiter.
        self._aiter_attn(self.vis_Q, self.vis_K, self.vis_V,
                         self._vis_O, self._vis_scale)
        return self._vis_O.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        if self._fixed_shape:
            # aiter's dense mha_fwd has no runtime-mask input. The MFMA
            # flash kernel takes the FA2-style seqused device pointer
            # (enc_seqused) and only computes the runtime-valid rows, so
            # it serves fixed mode; masked sdpa remains the env escape
            # (FVK_AMD_FIXED_ENC_ATTN=sdpa) and the non-256-headdim
            # fallback. See module docstring.
            if self._fixed_enc_flash:
                self._fvk_ef.encoder_attention_flash(
                    self.enc_Q.data_ptr(),
                    self.enc_K[layer_idx].data_ptr(),
                    self.enc_V[layer_idx].data_ptr(),
                    self._enc_O.data_ptr(),
                    seq, self.enc_Q.shape[-2], self.enc_Q.shape[-1],
                    self._enc_scale, stream,
                    31, self.enc_seqused.data_ptr())
                return self._enc_O.data_ptr()
            return super().encoder_attn(layer_idx, seq, stream=stream)
        if self._calibrating and self._calib_det:
            # FP8-calibration window: deterministic MFMA flash kernel so
            # the collected amax scales are run-to-run stable (seqused=0
            # -> the full seq is valid; math identical to the fixed-mode
            # route above). Never taken inside the captured graph.
            self._fvk_ef.encoder_attention_flash(
                self.enc_Q.data_ptr(),
                self.enc_K[layer_idx].data_ptr(),
                self.enc_V[layer_idx].data_ptr(),
                self._enc_O.data_ptr(),
                seq, self.enc_Q.shape[-2], self.enc_Q.shape[-1],
                self._enc_scale, stream, 31, 0)
            return self._enc_O.data_ptr()
        # (1, seq, 8, 256) vs (1, seq, 1, 256): native GQA, no expand.
        q = self.enc_Q[:seq].unsqueeze(0)
        k = self.enc_K[layer_idx, :seq].unsqueeze(0)
        v = self.enc_V[layer_idx, :seq].unsqueeze(0)
        self._aiter_attn(q, k, v, self._enc_O[:seq].unsqueeze(0),
                         self._enc_scale)
        return self._enc_O.data_ptr()

    def decoder_attn(self, layer_idx: int, enc_seq: int, dec_seq: int,
                     stream: int = 0) -> int:
        # Reset per call: get_decoder_fp8out_ptr() must reflect THIS
        # call only (fixed-shape delegation / lib route produce none).
        self._dec_fp8out_last = None
        if self._fixed_shape and not self._dec_custom:
            # Only the aiter lib route lacks runtime masking — delegate
            # to the sdpa base. The hand-written split-KV kernel takes
            # the FA2-style seqused device pointer directly (validated
            # by the seqused-mode gates in its kernel suite), so the
            # custom route below serves fixed mode natively: enc_seq is
            # the padded capacity, dec_seqused[0] wins at replay.
            return super().decoder_attn(layer_idx, enc_seq, dec_seq,
                                        stream=stream)
        total_kv = enc_seq + dec_seq
        seqused_ptr = self.dec_seqused.data_ptr() if self._fixed_shape else 0
        if self._dec_custom:
            # Hand-written split-KV kernel; launches on the pipeline
            # stream directly (raw-pointer ABI, no torch dispatch).
            scales = self._dec_fp8out_scales
            if scales is not None:
                # Armed: fused FP8-out epilogue. Same attention + bf16
                # O as the plain call, plus the o-proj activation's fp8
                # bytes (bit-equal to quantize_fp8_static with the same
                # scale pointer).
                row = scales[min(self._dec_fp8out_step, len(scales) - 1)]
                scale_ptr = row[layer_idx]
                fp8_ptr = self._dec_fp8out_buf.data_ptr()
                self._fvk.attention_decoder_gqa_fp8out(
                    self.dec_Q.data_ptr(),
                    self.enc_K[layer_idx].data_ptr(),
                    self.enc_V[layer_idx].data_ptr(),
                    self._dec_O.data_ptr(),
                    fp8_ptr,
                    scale_ptr,
                    self._dec_attn_ws.data_ptr(),
                    dec_seq, total_kv,
                    self.dec_Q.shape[-2], self.dec_Q.shape[-1],
                    seqused_ptr,         # 0 = exact; device int in fixed mode
                    self._enc_scale,
                    stream)
                self._dec_fp8out_last = (fp8_ptr, scale_ptr)
                return self._dec_O.data_ptr()
            self._fvk.attention_decoder_gqa(
                self.dec_Q.data_ptr(),
                self.enc_K[layer_idx].data_ptr(),
                self.enc_V[layer_idx].data_ptr(),
                self._dec_O.data_ptr(),
                self._dec_attn_ws.data_ptr(),
                dec_seq, total_kv,
                self.dec_Q.shape[-2], self.dec_Q.shape[-1],
                seqused_ptr,             # 0 = exact; device int in fixed mode
                self._enc_scale,
                stream)
            return self._dec_O.data_ptr()
        # Chunk queries cross-attend the shared [encoder + appended chunk]
        # K/V range bidirectionally — same semantics as the sdpa path.
        q = self.dec_Q[:dec_seq].unsqueeze(0)                 # (1, chunk, 8, 256)
        k = self.enc_K[layer_idx, :total_kv].unsqueeze(0)     # (1, kv, 1, 256)
        v = self.enc_V[layer_idx, :total_kv].unsqueeze(0)
        self._aiter_attn(q, k, v, self._dec_O[:dec_seq].unsqueeze(0),
                         self._enc_scale)
        return self._dec_O.data_ptr()
