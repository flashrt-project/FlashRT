"""FlashRT — Thor JAX Pi0.5 frontend with NVFP4 encoder-FFN subset.

STRICTLY ADDITIVE. Subclasses :class:`Pi05JaxFrontendThor` and overlays an
optional NVFP4 path on a configurable subset of encoder FFN layers. With the
FP4 flag off, behaves bit-identically to the base FP8 JAX frontend.

Mirrors :class:`flash_rt.frontends.torch.pi05_thor_fp4.Pi05TorchFrontendThorFP4`
in its public surface and calibration discipline, but keeps the runtime
torch-free: all FP4 weights / activation scratch / AWQ inv_s buffers live in
:class:`CudaBuffer`, and this module pulls *zero* torch symbols at import or
inference time.

The only bridge to :func:`encoder_forward_with_fp4_subset` (which was written
for the torch pi05_thor_fp4 path and calls ``.data_ptr()`` on weight shards)
is a tiny ``_PtrShim`` wrapper — it exposes ``.data_ptr()`` over a
CudaBuffer pointer so ``shared_primitives_fp4`` does not need to be touched.

Usage::

    pipe = Pi05JaxFrontendThorFP4(
        "/path/to/checkpoint",
        num_views=2,
        use_fp4_encoder_ffn=True,    # default False → bit-identical to base
        fp4_layers=(0, 1, ..., 17),  # 18 layers = full production preset
        use_awq=True,
        use_p1_split_gu=True,
    )
    pipe.set_prompt("pick up the red cup")
    actions = pipe.infer(obs)["actions"]
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor
from flash_rt.executors.fp4_utils_cb import (
    FP4ActScratchCB,
    FP4BufferCB,
    _PtrShim,
    pick_variant,
    quant_act_nvfp4_cb,
    quant_weight_nvfp4_from_cb,
    quant_weight_nvfp4_inplace_from_cb,
)

try:
    import flash_rt.flash_rt_fp4 as fvk_fp4
    _HAS_FP4 = fvk_fp4.has_nvfp4()
except Exception:  # pragma: no cover
    fvk_fp4 = None
    _HAS_FP4 = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Frontend class
# ──────────────────────────────────────────────────────────────────────
class Pi05JaxFrontendThorFP4(Pi05JaxFrontendThor):
    """Pi0.5 Thor JAX frontend with optional NVFP4 encoder-FFN layers."""

    def __init__(
        self,
        checkpoint_dir,
        *,
        engine_path=None,
        fmha_path=None,
        use_cuda_graph: bool = True,
        num_views: int = 2,
        autotune: int = 3,
        weight_cache: bool = True,
        # FP4 kwargs (parallel to Pi05TorchFrontendThorFP4)
        use_fp4_encoder_ffn: bool = False,
        fp4_layers: Iterable[int] = (7, 8, 9),
        use_awq: bool = False,
        awq_alpha: float = 0.5,
        awq_calib_iters: int = 8,
        use_p1_split_gu: bool = False,
        use_fp8: bool = True,
        **kwargs,
    ):
        # Set FP4 state BEFORE super().__init__ because the base class
        # calls self._upload_weights(engine_w, ...) and immediately frees
        # engine_w. Our override below reads self.use_fp4_encoder_ffn to
        # decide whether to stash fp16 gate_up/down slices for Chunk 2's
        # NVFP4 quantization.
        self.use_fp4_encoder_ffn = bool(use_fp4_encoder_ffn)
        self._fp4_layers = (
            frozenset(int(x) for x in fp4_layers)
            if self.use_fp4_encoder_ffn else frozenset()
        )
        self.use_awq = bool(use_awq) and self.use_fp4_encoder_ffn
        self.awq_alpha = float(awq_alpha)
        self.awq_calib_iters = int(awq_calib_iters)
        self.use_p1_split_gu = bool(use_p1_split_gu) and self.use_fp4_encoder_ffn

        # FP4-specific state; populated by _prepare_fp4_encoder_jax.
        self._fp4_weights: dict = {}      # l -> {'gate_up': {...}, 'down': {...}, 'gate'?, 'up'?}
        self._fp4_scratch: dict = {}      # shared scratch for encoder_forward_with_fp4_subset
        self._awq_inv_s_gu: dict = {}     # l -> CudaBuffer fp16 [D]   (only use_awq)
        self._awq_inv_s_dn: dict = {}     # l -> CudaBuffer fp16 [H]
        self._awq_calibrated: bool = False

        # Stash for fp16 gate_up / down slices, populated by
        # _upload_weights (cache-miss path) or by the FP8-cache-hit
        # fallback inside _prepare_fp4_encoder_jax.
        self._stashed_fp4_engine_w: dict = {}

        # Base class does the full FP8 load + graph setup. This calls
        # our overridden _upload_weights which will populate the stash.
        super().__init__(
            checkpoint_dir,
            engine_path=engine_path,
            fmha_path=fmha_path,
            use_cuda_graph=use_cuda_graph,
            num_views=num_views,
            autotune=autotune,
            weight_cache=weight_cache,
            use_fp8=use_fp8,
            **kwargs,
        )

        if self._fp4_layers:
            if not _HAS_FP4:
                raise RuntimeError(
                    "use_fp4_encoder_ffn=True but flash_rt_fp4 is not "
                    "available (SM100+ required). Build with NVFP4 support "
                    "or set use_fp4_encoder_ffn=False.")
            self._prepare_fp4_encoder_jax()
            logger.info(
                "Pi05 JAX FP4 enabled on encoder layers: %s (AWQ=%s, P1=%s)",
                sorted(self._fp4_layers), self.use_awq, self.use_p1_split_gu)

    # ------------------------------------------------------------------
    # _upload_weights override — stash fp16 FFN weights before super frees
    # engine_w. Only active when FP4 is enabled; otherwise no-op overhead.
    # ------------------------------------------------------------------
    def _upload_weights(self, engine_w, quantize_fp8, compute_time_embeddings):
        if self.use_fp4_encoder_ffn and self._fp4_layers:
            for l in self._fp4_layers:
                gu_key = f"encoder.layer.{l}.gate_up.weight"
                dn_key = f"encoder.layer.{l}.down.weight"
                # Copies so super's `del engine_w; gc.collect()` doesn't
                # strand views back on freed memory.
                self._stashed_fp4_engine_w[int(l)] = {
                    'gate_up': np.ascontiguousarray(engine_w[gu_key]).copy(),
                    'down':    np.ascontiguousarray(engine_w[dn_key]).copy(),
                }
        return super()._upload_weights(engine_w, quantize_fp8,
                                        compute_time_embeddings)

    # ------------------------------------------------------------------
    # Weight preparation (once per construction; FP4 weight cache hit or
    # Orbax-engine-weights → NVFP4 derivation).
    # ------------------------------------------------------------------
    def _prepare_fp4_encoder_jax(self) -> None:
        """Derive NVFP4 encoder weights from the FP8 frontend's in-memory
        engine-weight representation.

        Pipeline:
            1. Pull already-fused fp16 ``encoder.layer.{l}.gate_up.weight``
               [2H, D] and ``encoder.layer.{l}.down.weight`` [D, H] from
               the Orbax-derived weight cache. (FusedGateUp with
               (1+pre_ffw_norm.scale) is already applied by
               core.weights.transformer lines 497-528.)
            2. If AWQ is enabled, apply per-input-channel pre-scale and
               record inv_s in ``_awq_inv_s_{gu,dn}[l]`` as CudaBuffers.
            3. Quantize each shard via ``quant_weight_nvfp4_from_cb``.
            4. If P1 split-GU is enabled, additionally store gate / up
               shards separately (each [H, D]).
            5. Build the shared ``_fp4_scratch`` dict consumed by
               ``encoder_forward_with_fp4_subset``.

        [Chunk 2 — construction-time NVFP4 derivation, per-layer + P1.
         Chunk 3 — builds _fp4_scratch and wires the graph.]
        """
        from flash_rt.core.cuda_buffer import CudaBuffer

        De = int(self.De)
        He = int(self.He)

        # ── 1. Source fp16 weights ──────────────────────────────────
        # Cache-miss path stashed by _upload_weights; cache-hit path
        # (FP8 cache restored) has no stash → reload Orbax once.
        if self._stashed_fp4_engine_w:
            src = self._stashed_fp4_engine_w
            self._stashed_fp4_engine_w = {}  # release refs
        else:
            logger.info(
                "FP4 prep: FP8 weight cache hit so engine_w was not stashed — "
                "reloading Orbax for FP4 layer weights (one-time).")
            from flash_rt.core.weights.loader import load_weights, detect_format
            from flash_rt.core.weights.transformer import transform_jax_weights
            fmt = detect_format(self._checkpoint_path)
            raw = load_weights(self._checkpoint_path, format=fmt)
            engine_w = transform_jax_weights(raw)
            del raw
            src = {
                int(l): {
                    'gate_up': np.ascontiguousarray(
                        engine_w[f"encoder.layer.{int(l)}.gate_up.weight"]).copy(),
                    'down':    np.ascontiguousarray(
                        engine_w[f"encoder.layer.{int(l)}.down.weight"]).copy(),
                }
                for l in self._fp4_layers
            }
            del engine_w
            import gc
            gc.collect()

        # ── 2. Per-layer quantize (+ optional AWQ + P1) ──────────────
        for l in sorted(self._fp4_layers):
            gu_fp16 = src[l]['gate_up'].astype(np.float16, copy=False)
            dn_fp16 = src[l]['down'].astype(np.float16, copy=False)

            assert gu_fp16.shape == (2 * He, De), (
                f"L{l} gate_up shape {gu_fp16.shape} != {(2*He, De)}")
            assert dn_fp16.shape == (De, He), (
                f"L{l} down shape {dn_fp16.shape} != {(De, He)}")
            assert gu_fp16.flags['C_CONTIGUOUS']
            assert dn_fp16.flags['C_CONTIGUOUS']

            # AWQ per-input-channel pre-scale (weight-only amax at
            # construction time; refined in Chunk 4 with activation amax).
            if self.use_awq:
                gu_scaled, inv_s_gu = self._awq_scale_weight_np(gu_fp16)
                dn_scaled, inv_s_dn = self._awq_scale_weight_np(dn_fp16)
                self._awq_inv_s_gu[l] = CudaBuffer.from_numpy(inv_s_gu)  # fp16 [D]
                self._awq_inv_s_dn[l] = CudaBuffer.from_numpy(inv_s_dn)  # fp16 [H]
            else:
                gu_scaled = gu_fp16
                dn_scaled = dn_fp16

            # Upload fp16 weights to CudaBuffer, quantize, drop fp16 copy.
            gu_cb = CudaBuffer.from_numpy(gu_scaled)
            dn_cb = CudaBuffer.from_numpy(dn_scaled)
            gu_q = quant_weight_nvfp4_from_cb(gu_cb, 2 * He, De)
            dn_q = quant_weight_nvfp4_from_cb(dn_cb, De, He)

            entry = {
                'gate_up': self._wrap_fp4_shards(gu_q),
                'down':    self._wrap_fp4_shards(dn_q),
            }

            if self.use_p1_split_gu:
                # gu_scaled shape [2H, D] = [gate || up] along N.
                g_fp16 = np.ascontiguousarray(gu_scaled[:He, :])
                u_fp16 = np.ascontiguousarray(gu_scaled[He:, :])
                g_cb = CudaBuffer.from_numpy(g_fp16)
                u_cb = CudaBuffer.from_numpy(u_fp16)
                g_q = quant_weight_nvfp4_from_cb(g_cb, He, De)
                u_q = quant_weight_nvfp4_from_cb(u_cb, He, De)
                entry['gate'] = self._wrap_fp4_shards(g_q)
                entry['up']   = self._wrap_fp4_shards(u_q)
                # Keep gu cache too — the full merged path may still be
                # used by non-P1 calibration replays.
                del g_cb, u_cb

            self._fp4_weights[l] = entry
            del gu_cb, dn_cb

        # Variant selection — cache for Chunk 3 graph capture.
        self._fp4_variant_gu = pick_variant(2 * He, De)
        self._fp4_variant_dn = pick_variant(De, He)

        logger.info(
            "FP4 weights quantized for %d layer(s); variant_gu=%d variant_dn=%d",
            len(self._fp4_weights), self._fp4_variant_gu, self._fp4_variant_dn)

    @staticmethod
    def _wrap_fp4_shards(q: dict) -> dict:
        """Wrap a ``{packed_cb, sfb_cb, ...}`` dict returned by
        ``quant_weight_nvfp4_from_cb`` into the shape expected by
        ``shared_primitives_fp4.encoder_forward_with_fp4_subset``.

        The primitive calls ``.data_ptr()`` on the ``'packed'`` and
        ``'sfb'`` fields — so those must be objects exposing a
        ``.data_ptr()`` method (torch tensor *or* our ``_PtrShim``).
        The ``_cb`` fields are retained to keep the CudaBuffers alive
        (``_PtrShim`` only caches the int, not the CudaBuffer).
        """
        return {
            'packed':    _PtrShim(q['packed_cb']),
            'sfb':       _PtrShim(q['sfb_cb']),
            'packed_cb': q['packed_cb'],
            'sfb_cb':    q['sfb_cb'],
            'N':         int(q['N']),
            'K':         int(q['K']),
        }

    def _awq_scale_weight_np(self,
                             W: np.ndarray,
                             activation_amax: np.ndarray | None = None,
                             ) -> tuple[np.ndarray, np.ndarray]:
        """AWQ per-input-channel (K axis) pre-scale in numpy.

        Parallel of ``Pi05TorchFrontendThorFP4._awq_scale_weight`` but kept
        torch-free. If ``activation_amax`` is ``None`` falls back to
        weight-only amax (construction-time init before real data is seen).

        Returns ``(W', inv_s)`` both fp16 numpy with
        ``W' = W * s[None, :]`` and ``s = (a/a.mean())^alpha``.
        """
        if activation_amax is not None:
            a = np.clip(activation_amax.astype(np.float32), 1e-6, None)
        else:
            a = np.clip(np.abs(W).astype(np.float32).max(axis=0), 1e-6, None)
        # f32 math throughout; avoid implicit f64 (calibration.md §2.3).
        s = np.clip(np.power(a / a.mean(), np.float32(self.awq_alpha)),
                    np.float32(0.25), np.float32(4.0)).astype(np.float32)
        inv_s = (np.float32(1.0) / s).astype(np.float16)   # [K]
        # Broadcast along N axis: W is [N, K], s is [K].
        W_scaled = (W.astype(np.float32) * s[None, :]).astype(np.float16)
        return np.ascontiguousarray(W_scaled), np.ascontiguousarray(inv_s)

    # ------------------------------------------------------------------
    # FP4 runtime scratch — allocated once on first graph capture, reused
    # on every subsequent recapture so pointer addresses stay stable
    # (in-place AWQ refit relies on this).
    # ------------------------------------------------------------------
    def _build_fp4_scratch(self) -> None:
        from flash_rt.core.cuda_buffer import CudaBuffer

        Se_max = int(self.Se_max)
        De = int(self.De)
        He = int(self.He)

        self._fp4_scratch_gu_act   = FP4ActScratchCB(Se_max, De)
        self._fp4_scratch_down_act = FP4ActScratchCB(Se_max, He)

        # fp16 intermediates consumed by shared_primitives_fp4 as raw pointers.
        self._fp4_scratch_x_normed = CudaBuffer.device_empty(Se_max * De,      np.float16)
        self._fp4_scratch_gate_out = CudaBuffer.device_empty(Se_max * 2 * He,  np.float16)
        self._fp4_scratch_hid_fp16 = CudaBuffer.device_empty(Se_max * He,      np.float16)
        self._fp4_scratch_fg_fp16  = CudaBuffer.device_empty(Se_max * De,      np.float16)

        sc = {
            'gu_act':     self._fp4_scratch_gu_act,
            'down_act':   self._fp4_scratch_down_act,
            'x_normed':   int(self._fp4_scratch_x_normed.ptr.value),
            'gate_out':   int(self._fp4_scratch_gate_out.ptr.value),
            'hid_fp16':   int(self._fp4_scratch_hid_fp16.ptr.value),
            'fg_fp16':    int(self._fp4_scratch_fg_fp16.ptr.value),
            'variant_gu': int(self._fp4_variant_gu),
            'variant_dn': int(self._fp4_variant_dn),
        }

        if self.use_p1_split_gu:
            self._fp4_scratch_p1_gate = FP4BufferCB(Se_max, He)
            self._fp4_scratch_p1_up   = FP4BufferCB(Se_max, He)
            sc['p1_gate_p4']  = self._fp4_scratch_p1_gate.packed_ptr
            sc['p1_gate_sfa'] = self._fp4_scratch_p1_gate.sfa_ptr
            sc['p1_up_p4']    = self._fp4_scratch_p1_up.packed_ptr
            sc['p1_up_sfa']   = self._fp4_scratch_p1_up.sfa_ptr

        if self.use_awq:
            sc['awq_inv_s_gu'] = {
                int(l): int(self._awq_inv_s_gu[l].ptr.value)
                for l in self._fp4_layers
            }
            sc['awq_inv_s_dn'] = {
                int(l): int(self._awq_inv_s_dn[l].ptr.value)
                for l in self._fp4_layers
            }

        self._fp4_scratch = sc

    # ------------------------------------------------------------------
    # Graph capture override
    # ------------------------------------------------------------------
    def _capture_enc_ae_graph(self) -> None:
        """Capture Encoder+AE as CUDA graph with the FP4 subset path.

        When ``self._fp4_layers`` is empty, delegates to the base class so
        behaviour is bit-identical to the pure-FP8 frontend. Otherwise the
        encoder call is routed through
        ``shared_primitives_fp4.encoder_forward_with_fp4_subset`` with the
        pre-built ``_fp4_weights`` / ``_fp4_scratch`` / ``_awq_inv_s_*``
        state.
        """
        if not self._fp4_layers:
            return super()._capture_enc_ae_graph()

        from flash_rt.core.cuda_graph import CUDAGraph
        from flash_rt.hardware.thor.shared_primitives_fp4 import (
            encoder_forward_with_fp4_subset,
        )
        from flash_rt.models.pi05.pipeline_thor import decoder_forward

        stream = self._stream
        _cudart = self._cudart
        stream_int = stream.value or 0

        enc_bufs, enc_weights, enc_dims = self._build_enc_dicts(stream_int)
        ae_bufs,  ae_weights,  ae_dims  = self._build_ae_dicts(stream_int)

        if not self._fp4_scratch:
            self._build_fp4_scratch()
        fp4_scratch = self._fp4_scratch

        def _run(st):
            self.Kc.zero_(self._stream); self.Vc.zero_(self._stream)
            encoder_forward_with_fp4_subset(
                self._gemm, self._fvk, fvk_fp4,
                enc_bufs, enc_weights, enc_dims, st,
                attn=self._attn,
                fp4_layers=self._fp4_layers,
                fp4_weights=self._fp4_weights,
                fp4_scratch=fp4_scratch,
                use_p1_split_gu=self.use_p1_split_gu,
            )
            decoder_forward(self._ctx, self._fvk,
                            ae_bufs, ae_weights, ae_dims, st,
                            attn=self._attn)

        for _ in range(3):
            _run(stream_int)
        _cudart.cudaStreamSynchronize(stream)

        self.enc_ae_graph = CUDAGraph()
        self.enc_ae_graph.begin_capture(stream)
        _run(stream_int)
        self.enc_ae_graph.end_capture(stream)
        _cudart.cudaStreamSynchronize(stream)
        logger.info(
            "Enc+AE CUDA Graph captured FP4 (%d layer(s), Se=%d, P1=%s, AWQ=%s)",
            len(self._fp4_layers), self.Se, self.use_p1_split_gu, self.use_awq)

    # ------------------------------------------------------------------
    # AWQ amax collection + in-place refit (run post-calibration)
    # ------------------------------------------------------------------
    def _collect_awq_activation_amax_jax(self) -> tuple[dict, dict]:
        """Run one encoder replay (FP8 path, no FP4) and snapshot per-
        channel activation amax at every FP4 layer's Gate+Up input
        (post-rms, shape [D]) and Down input (post-silu*mul, shape [H]).

        Reads back the intermediate fp16 activations from CudaBuffer to
        numpy for percentile/amax math.

        Returns two dicts keyed by layer index ``l``:
            act_gu[l] : np.ndarray  fp32 [D]
            act_dn[l] : np.ndarray  fp32 [H]

        Parallel of :meth:`Pi05TorchFrontendThorFP4._collect_awq_activation_amax`
        but without torch.
        """
        import math
        from flash_rt.core.cuda_buffer import CudaBuffer

        Se = int(self.Se); De = int(self.De); He = int(self.He)
        NHe = int(self.NHe); HDe = int(self.HDe); Le = int(self.Le)
        total_keys = int(self.total_keys)
        Q_dim = NHe * HDe; K_dim = HDe
        attn_scale = 1.0 / math.sqrt(float(HDe))

        # Snapshot buffers (CudaBuffer) reused across samples; allocate once.
        # Use MANAGED memory so download() goes through memmove — device memcpy
        # has a ~16 MB single-shot limit on Thor and Se*He*fp16 can exceed it.
        if not hasattr(self, "_awq_x_snap"):
            self._awq_x_snap = {l: CudaBuffer.empty(Se * De, np.float16)
                                for l in self._fp4_layers}
            self._awq_h_snap = {l: CudaBuffer.empty(Se * He, np.float16)
                                for l in self._fp4_layers}

        x_snap = self._awq_x_snap
        h_snap = self._awq_h_snap

        # All pointers as ints (shared_primitives-style).
        enc_bufs, enc_weights, enc_dims = self._build_enc_dicts(0)
        x        = int(self.enc_x.ptr.value)
        x_fp8    = enc_bufs['x_fp8']
        qkv_buf  = enc_bufs['qkv']
        logits   = enc_bufs['logits']
        attn_out = enc_bufs['attn_out']
        o_fp8    = enc_bufs['o_fp8']
        gate_buf = enc_bufs['gate']
        hid_fp8  = enc_bufs['hid_fp8']
        fg       = enc_bufs['fg']
        rope     = enc_weights['rope']
        Kc       = enc_weights['Kc']
        Vc       = enc_weights['Vc']
        act_scales = enc_weights['act_scales']
        alpha_host = enc_weights['alpha_host']
        qkv_w      = enc_weights['qkv_w']
        o_w        = enc_weights['o_w']
        gu_w       = enc_weights['gate_w']
        dn_w       = enc_weights['down_w']

        fvk = self._fvk
        self.Kc.zero_(self._stream); self.Vc.zero_(self._stream)

        for l in range(Le):
            last = (l == Le - 1)
            as_qkv = act_scales + (l * 4 + 0) * 4
            as_o   = act_scales + (l * 4 + 1) * 4
            as_gu  = act_scales + (l * 4 + 2) * 4
            as_d   = act_scales + (l * 4 + 3) * 4

            fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, De, as_qkv, 0)
            fvk.cutlass_fp8_sq(x_fp8, qkv_w[l], qkv_buf,
                               Se, 2560, De, alpha_host[l * 4 + 0], 0.0, 0)
            kv_elem_off = l * total_keys * HDe
            fvk.qkv_split_rope_kvcache_fp16(
                qkv_buf, rope, attn_out, Kc, Vc,
                Se, Q_dim, K_dim, HDe, 2560,
                kv_elem_off, HDe, 0)
            if last:
                continue

            if self._attn is not None:
                self._attn.run("encoder", l, q_seq=Se, stream=0)
            else:
                fvk.attention_qkv_fp16(
                    self._ctx, attn_out,
                    Kc + kv_elem_off * 2, Vc + kv_elem_off * 2,
                    logits, attn_out,
                    Se, Se, NHe, HDe, attn_scale, 0)

            fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, Se * De, 0)
            fvk.cutlass_fp8_sq(o_fp8, o_w[l], fg,
                               Se, De, De, alpha_host[l * 4 + 1], 0.0, 0)

            if l in self._fp4_layers:
                # Snapshot fp16 Gate+Up input (post-rms of x+fg, weight-less).
                fvk_fp4.residual_add_rms_norm_noweight_fp16(
                    x, fg, int(x_snap[l].ptr.value), Se, De, 0)

            # Production residual update — always FP8 path during calibrate.
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                x, fg, x_fp8, Se, De, as_gu, 0)

            fvk.cutlass_fp8_t1(x_fp8, gu_w[l], gate_buf,
                               Se, He * 2, De, alpha_host[l * 4 + 2], 0.0, 0)

            if l in self._fp4_layers:
                # Snapshot fp16 Down input (post-silu*mul).
                fvk.gate_geglu_merged_fp16(
                    gate_buf, int(h_snap[l].ptr.value), Se, He, 0)

            fvk.gate_geglu_merged_fp8_fp16(gate_buf, hid_fp8, Se, He, as_d, 0)
            fvk.cutlass_fp8_wide(hid_fp8, dn_w[l], fg,
                                 Se, De, He, alpha_host[l * 4 + 3], 0.0, 0)
            as_next = act_scales + ((l + 1) * 4 + 0) * 4
            fvk.residual_add_rms_norm_fp8_noweight_fp16(
                x, fg, x_fp8, Se, De, as_next, 0)

        self._cudart.cudaStreamSynchronize(self._stream)

        # Download snapshots → numpy per-channel amax [D] / [H] (fp32).
        act_gu: dict = {}
        act_dn: dict = {}
        for l in self._fp4_layers:
            x_np = x_snap[l].download_new((Se, De), np.float16).astype(np.float32)
            h_np = h_snap[l].download_new((Se, He), np.float16).astype(np.float32)
            act_gu[l] = np.abs(x_np).max(axis=0)   # [De]
            act_dn[l] = np.abs(h_np).max(axis=0)   # [He]
        return act_gu, act_dn

    def _requant_fp4_weights_with_awq_jax(self,
                                          act_gu: dict,
                                          act_dn: dict) -> None:
        """Re-quantize FP4 weights using activation-aware AWQ scales.

        Updates packed / sfb bytes IN-PLACE (pointer addresses unchanged)
        so the captured CUDA Graph replays without recapture. Also
        overwrites ``_awq_inv_s_gu[l]`` / ``_awq_inv_s_dn[l]`` in place.

        Parallel of :meth:`Pi05TorchFrontendThorFP4._requant_fp4_weights_with_awq`.
        """
        from flash_rt.core.cuda_buffer import CudaBuffer

        De = int(self.De); He = int(self.He)

        # Reload fp16 source weights from Orbax — same as Chunk 2's
        # stash-or-reload logic, but we need them again since the stash
        # is cleared after construction.
        if not hasattr(self, "_awq_refit_src") or self._awq_refit_src is None:
            from flash_rt.core.weights.loader import load_weights, detect_format
            from flash_rt.core.weights.transformer import transform_jax_weights
            fmt = detect_format(self._checkpoint_path)
            raw = load_weights(self._checkpoint_path, format=fmt)
            engine_w = transform_jax_weights(raw); del raw
            self._awq_refit_src = {
                int(l): {
                    'gate_up': np.ascontiguousarray(
                        engine_w[f"encoder.layer.{int(l)}.gate_up.weight"]
                    ).astype(np.float16, copy=False),
                    'down': np.ascontiguousarray(
                        engine_w[f"encoder.layer.{int(l)}.down.weight"]
                    ).astype(np.float16, copy=False),
                }
                for l in self._fp4_layers
            }
            del engine_w
            import gc; gc.collect()

        src = self._awq_refit_src

        for l in sorted(self._fp4_layers):
            gu_fp16 = src[l]['gate_up']
            dn_fp16 = src[l]['down']

            gu_scaled, inv_s_gu = self._awq_scale_weight_np(
                gu_fp16, activation_amax=act_gu[l])
            dn_scaled, inv_s_dn = self._awq_scale_weight_np(
                dn_fp16, activation_amax=act_dn[l])

            # In-place inv_s overwrite — pointer addresses preserved so the
            # captured graph sees new values on next replay.
            self._awq_inv_s_gu[l].upload(inv_s_gu)
            self._awq_inv_s_dn[l].upload(inv_s_dn)

            # In-place NVFP4 re-quantize into existing packed/sfb buffers.
            gu_cb = CudaBuffer.from_numpy(gu_scaled)
            dn_cb = CudaBuffer.from_numpy(dn_scaled)
            quant_weight_nvfp4_inplace_from_cb(
                gu_cb, {'packed_cb': self._fp4_weights[l]['gate_up']['packed_cb'],
                        'sfb_cb':    self._fp4_weights[l]['gate_up']['sfb_cb'],
                        'N': 2 * He, 'K': De})
            quant_weight_nvfp4_inplace_from_cb(
                dn_cb, {'packed_cb': self._fp4_weights[l]['down']['packed_cb'],
                        'sfb_cb':    self._fp4_weights[l]['down']['sfb_cb'],
                        'N': De, 'K': He})

            if self.use_p1_split_gu and 'gate' in self._fp4_weights[l]:
                g_scaled = np.ascontiguousarray(gu_scaled[:He, :])
                u_scaled = np.ascontiguousarray(gu_scaled[He:, :])
                g_cb = CudaBuffer.from_numpy(g_scaled)
                u_cb = CudaBuffer.from_numpy(u_scaled)
                quant_weight_nvfp4_inplace_from_cb(
                    g_cb, {'packed_cb': self._fp4_weights[l]['gate']['packed_cb'],
                           'sfb_cb':    self._fp4_weights[l]['gate']['sfb_cb'],
                           'N': He, 'K': De})
                quant_weight_nvfp4_inplace_from_cb(
                    u_cb, {'packed_cb': self._fp4_weights[l]['up']['packed_cb'],
                           'sfb_cb':    self._fp4_weights[l]['up']['sfb_cb'],
                           'N': He, 'K': De})

    # ------------------------------------------------------------------
    # Multi-frame calibration — two-phase just like the torch side
    # ------------------------------------------------------------------
    def _calibrate_multi_frame(self, obs_list, *,
                                percentile: float,
                                verbose: bool) -> None:
        """Pi0.5 JAX FP4 multi-sample (N>=2) calibration.

        Phase 1 — FP8 scale percentile reduction + graph recapture
            super()._calibrate_multi_frame(...) on the base FP8 JAX class.
            The final recapture goes through this class's
            ``_capture_enc_ae_graph`` override and picks up the current
            (still stale from construction) AWQ + NVFP4 buffers.

        Phase 2 — per-sample AWQ amax collection + in-place refit
            Only runs when ``use_awq`` is True. Writes inv_s + NVFP4
            packed/sfb bytes IN PLACE so no second graph capture is
            needed.

        If ``self._fp4_layers`` is empty, delegates straight to the base
        class.

        Parallel of :meth:`Pi05TorchFrontendThorFP4._calibrate_multi_frame`.
        """
        from flash_rt.core.cuda_buffer import CudaBuffer
        from flash_rt.core.calibration import accumulate_amax, check_scale_ceiling
        from flash_rt.hardware.thor.shared_primitives import encoder_forward_calibrate
        from flash_rt.models.pi05.pipeline_thor import decoder_forward_calibrate

        if not self._fp4_layers:
            # Pure-FP8 JAX multi-frame is out of Task A scope — surface the
            # same message the shim would.
            from flash_rt.core.calibration_api import implicit_calibrate
            return implicit_calibrate(self, obs_list, percentile=percentile,
                                       verbose=verbose)

        n = len(obs_list)
        logger.info(
            "Pi0.5 JAX FP4 multi-frame calibrate: N=%d, percentile=%.2f",
            n, percentile)

        CB = self._CudaBuffer
        stream = self._stream
        stream_int = stream.value or 0
        Se = int(self.Se); De = int(self.De); He = int(self.He)
        NHe = int(self.NHe); HDe = int(self.HDe); Le = int(self.Le); La = int(self.La)
        Sa = int(self.Sa); Da = int(self.Da); Ha = int(self.Ha)
        total_keys = int(self.total_keys)
        nv = self.num_views

        # Scratch buffers reused across samples.
        _norm_scratch = CB.device_empty(Se * De, np.float16)
        _x_scratch    = CB.device_empty(Se * De, np.float16)
        _enc_calib_buf = CB.zeros(Le * 4, np.float32)
        _d_scale = CB.zeros(1, np.float32)
        _fp8_scratch = CB.device_zeros(Se * max(De, He), np.uint8)
        _ones_buf = CB.from_numpy(np.ones(De, dtype=np.float16))
        _ae_calib_buf = CB.zeros(La * 4, np.float32)
        _ae_d_scale = CB.zeros(1, np.float32)
        _ae_hidden_scratch = CB.device_empty(Sa * Ha, np.float16)
        _ae_fp8_scratch = CB.device_zeros(Sa * max(Da, Ha), np.uint8)

        def _upload_and_siglip(obs):
            if 'images' in obs:
                imgs = obs['images']
            else:
                imgs = [obs['image']]
                if nv >= 2: imgs.append(obs.get('wrist_image', obs['image']))
                if nv >= 3: imgs.append(obs.get('wrist_image_right', imgs[-1]))
            def _to_fp16(im):
                if getattr(im, 'dtype', None) == np.float16:
                    return im
                return (np.asarray(im).astype(np.float32) / 127.5 - 1.0
                       ).astype(np.float16)
            images_np = np.stack([_to_fp16(im) for im in imgs[:nv]])
            self.img_buf.upload(images_np)
            self.siglip_graph.replay(stream)
            self._cudart.cudaStreamSynchronize(stream)

        # ── Phase 1: per-sample FP8 scale collection ──
        per_sample_enc: list = []
        per_sample_ae: list = []
        _orig_randn = np.random.randn
        for i, obs in enumerate(obs_list):
            _upload_and_siglip(obs)

            enc_scales_buf = CB.zeros(Le * 4, np.float32)
            enc_bufs = {
                'x': self.enc_x.ptr.value, 'x_fp8': self.enc_buf[0].ptr.value,
                'qkv': self.enc_buf[1].ptr.value,
                'logits': self.enc_buf[2].ptr.value,
                'attn_out': self.enc_buf[3].ptr.value,
                'o_fp8': self.enc_buf[5].ptr.value,
                'gate': self.enc_buf[6].ptr.value,
                'hidden': self.enc_buf[7].ptr.value,
                'hid_fp8': self.enc_buf[9].ptr.value,
                'fg': self.enc_buf[10].ptr.value,
                'ctx': self._ctx,
                'norm_scratch': _norm_scratch.ptr.value,
                'x_scratch': _x_scratch.ptr.value,
                'calib_buf': _enc_calib_buf.ptr.value,
                'd_scale': _d_scale.ptr.value,
                'fp8_scratch': _fp8_scratch.ptr.value,
                'ones': _ones_buf.ptr.value,
            }
            enc_weights = {
                'qkv_w': [self.ew[0].ptr.value + j * De * 2560 for j in range(Le)],
                'o_w':   [self.ew[1].ptr.value + j * De * De  for j in range(Le)],
                'gate_w':[self.ew[2].ptr.value + j * De * He * 2 for j in range(Le)],
                'down_w':[self.ew[4].ptr.value + j * He * De  for j in range(Le)],
                'rope': self.enc_rope.ptr.value,
                'Kc': self.Kc.ptr.value, 'Vc': self.Vc.ptr.value,
                'w_scales': self.enc_w_dev.ptr.value,
            }
            enc_dims = {'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
                        'L': Le, 'total_keys': total_keys}
            self.Kc.zero_(stream); self.Vc.zero_(stream)
            encoder_forward_calibrate(self._gemm, self._fvk,
                                       enc_bufs, enc_weights, enc_dims,
                                       enc_scales_buf.ptr.value,
                                       stream=stream_int)
            self._cudart.cudaStreamSynchronize(stream)
            per_sample_enc.append(
                enc_scales_buf.download_new((Le * 4,), np.float32))

            ae_scales_buf = CB.zeros(La * 4, np.float32)
            noise_np = _orig_randn(Sa, 32).astype(np.float16)
            self.g_noise.upload(noise_np)
            ae_bufs, ae_weights, ae_dims = self._build_ae_dicts(stream_int)
            ae_bufs['calib_buf'] = _ae_calib_buf.ptr.value
            ae_bufs['d_scale']   = _ae_d_scale.ptr.value
            ae_bufs['hidden_scratch'] = _ae_hidden_scratch.ptr.value
            ae_bufs['fp8_scratch'] = _ae_fp8_scratch.ptr.value
            decoder_forward_calibrate(self._ctx, self._fvk,
                                       ae_bufs, ae_weights, ae_dims,
                                       ae_scales_buf.ptr.value,
                                       stream=stream_int)
            self._cudart.cudaStreamSynchronize(stream)
            per_sample_ae.append(
                ae_scales_buf.download_new((La * 4,), np.float32))

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  FP8-scale sample %d/%d", i + 1, n)

        # Percentile reduce FP8 scales element-wise.
        enc_stack = np.stack(per_sample_enc, axis=0)
        ae_stack  = np.stack(per_sample_ae,  axis=0)
        enc_final = np.percentile(enc_stack, percentile, axis=0).astype(np.float32)
        ae_final  = np.percentile(ae_stack,  percentile, axis=0).astype(np.float32)

        self.enc_calib_scales = CB.from_numpy_managed(enc_final)
        self.ae_calib_scales  = CB.from_numpy_managed(ae_final)
        enc_ws_np = self.enc_w_dev.download_new((Le * 4,), np.float32)
        self.enc_alpha_host = [
            float(np.float32(enc_final[i]) * np.float32(enc_ws_np[i]))
            for i in range(Le * 4)
        ]

        # Recapture graph with fresh FP8 scales — FP4 buffers still hold the
        # Chunk 2 weight-only AWQ bytes; Phase 2 will refit them in place.
        self._capture_enc_ae_graph()

        # ── Phase 2: per-sample AWQ amax + in-place refit ──
        if not self.use_awq:
            logger.info(
                "JAX FP4 multi-frame: AWQ disabled — Phase 2 skipped.")
            self._real_data_calibrated = True
            return

        per_sample_gu: dict = {l: [] for l in self._fp4_layers}
        per_sample_dn: dict = {l: [] for l in self._fp4_layers}
        for i, obs in enumerate(obs_list):
            _upload_and_siglip(obs)
            act_gu, act_dn = self._collect_awq_activation_amax_jax()
            for l in self._fp4_layers:
                per_sample_gu[l].append(act_gu[l].astype(np.float32))
                per_sample_dn[l].append(act_dn[l].astype(np.float32))
            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  AWQ-amax sample %d/%d", i + 1, n)

        final_gu = {l: accumulate_amax(per_sample_gu[l], percentile=percentile)
                    for l in self._fp4_layers}
        final_dn = {l: accumulate_amax(per_sample_dn[l], percentile=percentile)
                    for l in self._fp4_layers}

        check_scale_ceiling(
            {f"L{l}_dn_max": float(final_dn[l].max()) for l in self._fp4_layers},
            label=f"pi05_jax_thor_fp4_awq_N{n}")

        self._requant_fp4_weights_with_awq_jax(final_gu, final_dn)
        self._awq_calibrated = True
        self._real_data_calibrated = True
        logger.info(
            "JAX FP4 multi-frame calibrate complete (N=%d, percentile=%.2f)",
            n, percentile)

    # ------------------------------------------------------------------
    # Public calibrate surface — route N>=2 to the native multi-frame
    # path above when FP4 is active; delegate to the base shim otherwise.
    # ------------------------------------------------------------------
    def calibrate(self, observations, *, percentile: float = 99.9,
                  max_samples=None, verbose: bool = False) -> None:
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
        if n == 1 or not self._fp4_layers:
            # Fall back to the inherited N=1 implicit path (through the
            # calibration_api shim, which just calls infer(obs_list[0])).
            return super().calibrate(obs_list, percentile=percentile,
                                      max_samples=None, verbose=verbose)
        return self._calibrate_multi_frame(
            obs_list, percentile=percentile, verbose=verbose)

    # ------------------------------------------------------------------
    # Public API surface otherwise inherited:
    #   set_prompt, infer, calibrate_with_real_data,
    #   precision_spec, _recalibrate_with_real_data.
    # ------------------------------------------------------------------
