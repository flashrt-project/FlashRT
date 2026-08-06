"""FlashRT — Thor SM110 attention backend for standalone Chameleon-7B.

One site:

* **chameleon** — 32-layer Chameleon-7B LLM self-attention (NH=32, HD=128,
  causal). Q/O aliased into the frontend's xn buffer. Dispatch order:
  FA4 (FlashAttention-4, CuTe-DSL, optional fast path) → CUTLASS causal
  FMHA (``libfmha_fp16_causal.so``) → cuBLAS decomposed causal MHA
  (``attention_mha_causal_fp16``).

Prefill (``run``) uses the top-left causal alignment (SQ == SK); single-query
incremental decode (``run_decode``) uses the bottom-right alignment via
``fmha_fp16_causal_br`` (or FA4 / plain cuBLAS MHA, both bottom-right
equivalent at SQ == 1).
"""

from __future__ import annotations

import ctypes
import logging
import pathlib

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec

logger = logging.getLogger(__name__)

# ── CUTLASS causal FMHA (SM100/110) dynamic loading ──
_fmha_causal_fn = None
_fmha_causal_br_fn = None  # bottom-right aligned variant (decode, SQ < SK)


def _load_fmha_causal_library() -> bool:
    """Load libfmha_fp16_causal.so and resolve the fmha_fp16_causal symbol."""
    global _fmha_causal_fn, _fmha_causal_br_fn
    if _fmha_causal_fn is not None:
        return True
    search_paths = [
        pathlib.Path(__file__).parent.parent.parent / "libfmha_fp16_causal.so",
        pathlib.Path(__file__).parent.parent.parent.parent / "build" / "libfmha_fp16_causal.so",
    ]
    argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    for p in search_paths:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                fn = lib.fmha_fp16_causal
                fn.restype = ctypes.c_int
                fn.argtypes = argtypes
                _fmha_causal_fn = fn
                # Bottom-right aligned causal (decode). Older .so builds may
                # lack it; run_decode then falls through to the cuBLAS tier.
                try:
                    fn_br = lib.fmha_fp16_causal_br
                    fn_br.restype = ctypes.c_int
                    fn_br.argtypes = argtypes
                    _fmha_causal_br_fn = fn_br
                except AttributeError:
                    logger.warning(
                        "libfmha_fp16_causal.so lacks fmha_fp16_causal_br "
                        "(rebuild for CUTLASS decode path)")
                logger.info("CUTLASS causal FMHA loaded from %s", p)
                return True
            except OSError as e:
                logger.warning("Failed to load causal FMHA from %s: %s", p, e)
    logger.warning("CUTLASS causal FMHA not found — will fall back to cuBLAS MHA")
    return False


class ThorChameleonAttnBackend(AttentionBackendBase):
    """Standalone Chameleon-7B attention backend on Thor (SM110).

    Single ``chameleon`` site: 32-layer causal MHA, Q/O aliased, per-layer
    KV cache with ``layer_stride`` bytes between consecutive layers.
    """

    def __init__(self, spec: AttentionSpec, ctx, *, chameleon_slots: dict) -> None:
        super().__init__(spec)

        expected_sites = {"chameleon"}
        got = set(spec.sites.keys())
        if got != expected_sites:
            raise ValueError(
                f"ThorChameleonAttnBackend expects sites {expected_sites}, "
                f"got {got}")

        self._ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx
        self._slots = {"chameleon": dict(chameleon_slots)}
        self._require_keys("chameleon",
                           ("Q_O", "Kc", "Vc", "logits", "layer_stride", "scale"))

        self._per_layer_kv = {}
        s = self._slots["chameleon"]
        nL = spec.site("chameleon").num_layers
        stride = int(s["layer_stride"])
        Kc = int(s["Kc"])
        Vc = int(s["Vc"])
        self._per_layer_kv["chameleon"] = [
            (Kc + l * stride, Vc + l * stride) for l in range(nL)
        ]

        self._fvk = None
        self._has_causal_fmha = _load_fmha_causal_library()

        # FA4 (FlashAttention-4, CuTe-DSL) fast path state. Populated by the
        # frontend with torch-tensor references to the live Q (xn) and KV
        # cache buffers, so run() can slice views (metadata-only, capture
        # safe) and dispatch to FA4 instead of the CUTLASS FMHA kernel.
        self._fa4_enabled = False
        self._fa4_q_tensor = None
        self._fa4_kv_cache = None

    def set_fa4_attn(self, q_tensor, kv_cache) -> None:
        """Enable the FA4 (FlashAttention-4, CuTe-DSL) fast path.

        q_tensor must alias the chameleon Q_O slot buffer (the frontend's xn
        buffer, which Q GEMMs write into and O is read from). kv_cache must
        alias the per-layer K/V buffer with layers at [li, 0, :Se] (K) and
        [li, 1, :Se] (V). Both are torch tensors so run() can build
        capture-safe views.
        """
        from flash_rt.hardware.thor import fa4_backend
        if not fa4_backend.is_available():
            logger.warning("FA4 unavailable (%s); keeping CUTLASS FMHA",
                           fa4_backend.status())
            return
        if q_tensor is None or kv_cache is None:
            raise ValueError("FA4 requires the Q (xn) and KV cache tensor refs")
        self._fa4_q_tensor = q_tensor
        self._fa4_kv_cache = kv_cache
        self._fa4_enabled = True
        logger.info("FA4 causal FMHA enabled for chameleon site")

    def disable_fa4_attn(self) -> None:
        """Revert chameleon site to the CUTLASS FMHA path."""
        self._fa4_enabled = False

    def _require_keys(self, site, keys):
        slot = self._slots[site]
        for k in keys:
            if k not in slot:
                raise ValueError(f"{site}_slots missing required key {k!r}")
            if k in ("Q_O", "Kc", "Vc", "logits"):
                if int(slot[k]) == 0:
                    raise ValueError(
                        f"{site}_slots[{k!r}] is a null device pointer")

    def _fvk_mod(self):
        if self._fvk is None:
            import flash_rt.flash_rt_kernels as fvk
            self._fvk = fvk
        return self._fvk

    def get_slot_ptrs(self, site, layer_idx):
        """Return {Q, K, V, O} device pointer ints for (site, layer_idx)."""
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")
        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r}")
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        q_o = int(self._slots[site]["Q_O"])
        return {"Q": q_o, "K": K_ptr, "V": V_ptr, "O": q_o}

    def kv_row_ptrs(self, site, layer_idx, pos):
        """Return (K_row, V_row) device pointer ints for token position pos.

        Rows are fp16 [NH*HD] within the layer's cache segment, so decode
        can GEMM the new token's K/V directly into the cache.
        """
        if site != "chameleon":
            raise KeyError(f"unknown site {site!r}")
        spec = self._spec.site(site)
        if not (0 <= layer_idx < spec.num_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r}")
        if not (0 <= pos < spec.max_kv_seq):
            raise IndexError(f"pos {pos} out of range for site {site!r}")
        row_bytes = spec.num_kv_heads * spec.head_dim * 2  # fp16
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        return K_ptr + pos * row_bytes, V_ptr + pos * row_bytes

    def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0,
            state_nk=None, cross_attn=False):
        """Dispatch causal attention for (site, layer_idx)."""
        if site != "chameleon":
            raise KeyError(f"unknown site {site!r}")
        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r}")
        if kv_seq is None:
            kv_seq = q_seq
        if q_seq == 1 and kv_seq > q_seq:
            raise ValueError(
                "run() got a decode shape (q_seq=1 < kv_seq); the top-left "
                "causal mask misaligns here — call run_decode() instead")

        s = self._slots[site]
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        NH = site_spec.num_q_heads
        HD = site_spec.head_dim

        # ── FA4 (FlashAttention-4, CuTe-DSL) fast path ──
        if self._fa4_enabled and self._fa4_q_tensor is not None:
            try:
                from flash_rt.hardware.thor import fa4_backend
                fa4 = fa4_backend.fa4_func()
                if fa4 is not None:
                    import torch
                    qv = self._fa4_q_tensor[:q_seq].view(1, q_seq, NH, HD)
                    kv_cache = self._fa4_kv_cache
                    k_view = kv_cache[layer_idx, 0, :kv_seq].view(1, kv_seq, NH, HD)
                    v_view = kv_cache[layer_idx, 1, :kv_seq].view(1, kv_seq, NH, HD)
                    with torch.no_grad():
                        o = fa4(qv.contiguous(), k_view.contiguous(),
                                v_view.contiguous(), causal=True, pack_gqa=True)
                    if isinstance(o, tuple):
                        o = o[0]
                    o_flat = o.reshape(q_seq, NH * HD)
                    # Q_O slot aliases xn (self._fa4_q_tensor): overwrite
                    # Q with O in place via a metadata-only view.
                    self._fa4_q_tensor[:q_seq].view(q_seq, NH * HD).copy_(o_flat)
                    return int(s["Q_O"])
            except Exception as e:  # pragma: no cover - defensive fallback
                logger.warning("FA4 failed at layer %d (%s); falling back to CUTLASS",
                               layer_idx, e)
                self._fa4_enabled = False

        # ── CUTLASS Causal FMHA (SM100/110) — preferred ──
        if self._has_causal_fmha and _fmha_causal_fn is not None:
            ret = _fmha_causal_fn(
                ctypes.c_void_p(int(s["Q_O"])),
                ctypes.c_void_p(K_ptr),
                ctypes.c_void_p(V_ptr),
                ctypes.c_void_p(int(s["Q_O"])),
                1, q_seq, kv_seq, NH, NH, HD,
                ctypes.c_void_p(stream),
            )
            if ret != 0:
                logger.warning(
                    "CUTLASS causal FMHA returned %d, falling back to cuBLAS",
                    ret)
                fvk.attention_mha_causal_fp16(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq, NH, HD,
                    float(s["scale"]), stream,
                )
        else:
            # ── Fallback: cuBLAS decomposed causal MHA ──
            fvk.attention_mha_causal_fp16(
                self._ctx_cpp,
                int(s["Q_O"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["Q_O"]),
                q_seq, kv_seq, NH, HD,
                float(s["scale"]), stream,
            )
        return int(s["Q_O"])

    def run_decode(self, site, layer_idx, kv_len, *, stream=0):
        """Dispatch single-query (decode) attention for (site, layer_idx).

        Q is the single row in the Q_O slot; K/V are the first kv_len rows
        of the layer's cache segment. The causal mask must be bottom-right
        aligned (query at position kv_len-1 attends all kv_len keys).
        """
        if site != "chameleon":
            raise KeyError(f"unknown site {site!r}")
        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r}")
        if not (1 <= kv_len <= site_spec.max_kv_seq):
            raise IndexError(f"kv_len {kv_len} out of range for site {site!r}")

        s = self._slots[site]
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        NH = site_spec.num_q_heads
        HD = site_spec.head_dim

        # ── FA4 fast path: causal is bottom-right (offset_k = sk - sq) ──
        if self._fa4_enabled and self._fa4_q_tensor is not None:
            try:
                from flash_rt.hardware.thor import fa4_backend
                fa4 = fa4_backend.fa4_func()
                if fa4 is not None:
                    import torch
                    qv = self._fa4_q_tensor[:1].view(1, 1, NH, HD)
                    kv_cache = self._fa4_kv_cache
                    k_view = kv_cache[layer_idx, 0, :kv_len].view(1, kv_len, NH, HD)
                    v_view = kv_cache[layer_idx, 1, :kv_len].view(1, kv_len, NH, HD)
                    with torch.no_grad():
                        o = fa4(qv.contiguous(), k_view.contiguous(),
                                v_view.contiguous(), causal=True, pack_gqa=True)
                    if isinstance(o, tuple):
                        o = o[0]
                    self._fa4_q_tensor[:1].view(1, NH * HD).copy_(
                        o.reshape(1, NH * HD))
                    return int(s["Q_O"])
            except Exception as e:  # pragma: no cover - defensive fallback
                logger.warning("FA4 decode failed at layer %d (%s); falling back",
                               layer_idx, e)
                self._fa4_enabled = False

        # ── CUTLASS bottom-right causal FMHA ──
        if self._has_causal_fmha and _fmha_causal_br_fn is not None:
            ret = _fmha_causal_br_fn(
                ctypes.c_void_p(int(s["Q_O"])),
                ctypes.c_void_p(K_ptr),
                ctypes.c_void_p(V_ptr),
                ctypes.c_void_p(int(s["Q_O"])),
                1, 1, kv_len, NH, NH, HD,
                ctypes.c_void_p(stream),
            )
            if ret != 0:
                logger.warning(
                    "CUTLASS causal FMHA (br) returned %d at layer %d, "
                    "falling back to cuBLAS", ret, layer_idx)
                fvk.attention_mha_fp16(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    1, kv_len, NH, HD,
                    float(s["scale"]), stream,
                )
        else:
            # ── cuBLAS non-causal MHA: with q_seq==1 the bottom-right
            #    causal mask is the identity, so this is equivalent. ──
            fvk.attention_mha_fp16(
                self._ctx_cpp,
                int(s["Q_O"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["Q_O"]),
                1, kv_len, NH, HD,
                float(s["scale"]), stream,
            )
        return int(s["Q_O"])


def make_chameleon_attention_spec(*, seq_max: int) -> AttentionSpec:
    """Build the standalone Chameleon-7B Thor AttentionSpec (1 site)."""
    spec = AttentionSpec()
    spec.add_site(
        "chameleon",
        num_layers=32, num_q_heads=32, num_kv_heads=32, head_dim=128,
        max_q_seq=int(seq_max), max_kv_seq=int(seq_max),
        causal=True,
    )
    return spec


__all__ = ["ThorChameleonAttnBackend", "make_chameleon_attention_spec"]
