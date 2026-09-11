"""LTX-2.5 attention backend swap.

Implements the ltx-core ``AttentionCallable`` protocol
(``fn(q, k, v, heads) -> out`` with flat ``[B, T, H*D]`` tensors) on top of
FlashRT's raw-pointer sage2 kernels:

    q: quant_per_warp_int8_bf16_d128   -> int8 [B,L,H,128] + f32 [B,H,ceil(L/32)]
    k: quant_per_block_int8_bf16_d128  -> int8 [B,L,H,128] + f32 [B,H,ceil(L/64)]
    v: transpose-pad-permute + fp8     -> fp8 [B,128,H,pad64(L)] + f32 [B,H,128]
    sage2_qk_int8_sv_f8_bf16_nhd_d128  -> bf16 [B,L,H,128]

All buffers are preallocated per (B, Lq, Lk, H) and reused, so repeated calls
issue an identical kernel sequence on identical pointers — safe to record into
a CUDA graph.

Site routing: the kernels are d128-only, so audio sites (head_dim 64) and any
other non-d128 shape fall back to torch SDPA, which also matches the measured
selection on 5090 (short-sequence d64 attention is faster on SDPA than on any
quantized path).

Backend choices (``make_ltx25_attention``):
    "sage2-fvk"  raw FlashRT kernels (default; graph-capture path)
    "sage2"      upstream ``sageattention`` package (needs it installed)
    "sage3"      ``sageattn3`` FP4 Blackwell package (needs it installed)
    "sdpa"       torch SDPA passthrough (baseline / fallback)
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    fvk = importlib.import_module("flash_rt.flash_rt_kernels")
except ModuleNotFoundError as exc:
    # Only the extension's own absence is optional. An undefined symbol, a
    # CUDA ABI mismatch, or a transitive dependency failing to load all
    # surface as plain ImportError, and swallowing those would report a
    # broken build as "kernels unavailable" and silently run the fallback.
    #
    # The import has to go through importlib for that distinction to exist
    # at all: ``from flash_rt import flash_rt_kernels`` reaches the
    # package's PEP 562 ``__getattr__``, which answers an absent extension
    # with build instructions raised as ImportError -- so the two cases
    # arrive as the same type and neither can be told from the other.
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    fvk = None

_REQUIRED_SYMS = (
    "quant_per_warp_int8_bf16_d128",
    "quant_per_block_int8_bf16_d128",
    "concat3_v_transpose_pad_permute_bf16_d128",
    "v_tpp_bf16_quant_fp8_d128",
    "sage2_qk_int8_sv_f8_bf16_nhd_d128",
)


def fvk_sage2_available() -> bool:
    return fvk is not None and all(hasattr(fvk, s) for s in _REQUIRED_SYMS)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
          heads: int) -> torch.Tensor:
    b, _, hd = q.shape
    d = hd // heads
    q4, k4, v4 = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q4, k4, v4)
    return o.transpose(1, 2).reshape(b, -1, hd)


class FvkSage2Attention:
    """sage2 qk-int8 / pv-fp8 attention on FlashRT raw kernels."""

    label = "fvk-sage2-qk8-pvfp8"

    def __init__(self, stream_fn=None) -> None:
        # stream_fn returns the raw cudaStream_t the pipeline runs on; during
        # graph capture it must be the capture stream. Default: torch current.
        self._stream_fn = stream_fn or (
            lambda: torch.cuda.current_stream().cuda_stream)
        self._bufs: dict[tuple, tuple[torch.Tensor, ...]] = {}

    def _buffers(self, device: torch.device, b: int, lq: int, lk: int,
                 h: int) -> tuple[torch.Tensor, ...]:
        key = (device, b, lq, lk, h)
        bufs = self._bufs.get(key)
        if bufs is None:
            pad_lk = ((lk + 63) // 64) * 64
            q8 = torch.empty(b, lq, h, 128, dtype=torch.int8, device=device)
            k8 = torch.empty(b, lk, h, 128, dtype=torch.int8, device=device)
            vt = torch.empty(b, 128, h, pad_lk, dtype=torch.bfloat16,
                             device=device)
            v8 = torch.empty(b, 128, h, pad_lk, dtype=torch.float8_e4m3fn,
                             device=device)
            qs = torch.empty(b, h, (lq + 31) // 32, dtype=torch.float32,
                             device=device)
            ks = torch.empty(b, h, (lk + 63) // 64, dtype=torch.float32,
                             device=device)
            vs = torch.empty(b, h, 128, dtype=torch.float32, device=device)
            out = torch.empty(b, lq, h, 128, dtype=torch.bfloat16,
                              device=device)
            bufs = (q8, k8, vt, v8, qs, ks, vs, out)
            self._bufs[key] = bufs
        return bufs

    @torch.compiler.disable
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 heads: int) -> torch.Tensor:
        b, lq, hd = q.shape
        d = hd // heads
        if d != 128 or q.dtype != torch.bfloat16 or not fvk_sage2_available():
            return _sdpa(q, k, v, heads)
        lk = k.shape[1]

        qn = q.view(b, lq, heads, d)
        kn = k.view(b, lk, heads, d)
        vn = v.view(b, lk, heads, d)
        qn = qn if qn.is_contiguous() else qn.contiguous()
        kn = kn if kn.is_contiguous() else kn.contiguous()
        vn = vn if vn.is_contiguous() else vn.contiguous()

        q8, k8, vt, v8, qs, ks, vs, out = self._buffers(
            q.device, b, lq, lk, heads)
        stream = self._stream_fn()

        fvk.quant_per_warp_int8_bf16_d128(
            int(qn.data_ptr()), int(q8.data_ptr()), int(qs.data_ptr()),
            b, lq, heads, stream)
        fvk.quant_per_block_int8_bf16_d128(
            int(kn.data_ptr()), int(k8.data_ptr()), int(ks.data_ptr()),
            b, lk, heads, stream)
        fvk.concat3_v_transpose_pad_permute_bf16_d128(
            int(vn.data_ptr()), int(vn.data_ptr()), int(vn.data_ptr()),
            int(vt.data_ptr()),
            b, lk, 0, 0, heads,
            int(vn.stride(0)), int(vn.stride(1)),
            int(vn.stride(0)), int(vn.stride(1)),
            int(vn.stride(0)), int(vn.stride(1)), stream)
        fvk.v_tpp_bf16_quant_fp8_d128(
            int(vt.data_ptr()), int(v8.data_ptr()), int(vs.data_ptr()),
            b, lk, heads, stream)
        rc = fvk.sage2_qk_int8_sv_f8_bf16_nhd_d128(
            int(q8.data_ptr()), int(k8.data_ptr()), int(v8.data_ptr()),
            int(out.data_ptr()), int(qs.data_ptr()), int(ks.data_ptr()),
            int(vs.data_ptr()), b, lq, lk, heads,
            float(d ** -0.5), stream)
        if rc != 0:
            raise RuntimeError(f"[ltx25.sage2] raw attention rc={rc}")
        return out.view(b, lq, hd)


class SagePkgAttention:
    """Upstream ``sageattention`` package (sm120 default dispatch)."""

    label = "sageattention-pkg"

    def __init__(self) -> None:
        import sageattention
        self._fn = sageattention.sageattn

    def __call__(self, q, k, v, heads):
        b, _, hd = q.shape
        d = hd // heads
        if d not in (64, 128) or d == 64:
            return _sdpa(q, k, v, heads)
        q4, k4, v4 = (t.view(b, -1, heads, d).transpose(1, 2)
                      for t in (q, k, v))
        o = self._fn(q4, k4, v4, tensor_layout="HND", is_causal=False)
        return o.transpose(1, 2).reshape(b, -1, hd)


class Sage3Attention:
    """``sageattn3`` FP4 Blackwell package. Fastest, lowest per-call cos;
    exposed as an opt-in for speed-first runs."""

    label = "sageattn3-fp4"

    def __init__(self) -> None:
        from sageattn3 import sageattn3_blackwell
        self._fn = sageattn3_blackwell

    def __call__(self, q, k, v, heads):
        b, _, hd = q.shape
        d = hd // heads
        if d != 128:
            return _sdpa(q, k, v, heads)
        q4, k4, v4 = (t.view(b, -1, heads, d).transpose(1, 2)
                      for t in (q, k, v))
        o = self._fn(q4, k4, v4, is_causal=False)
        return o.transpose(1, 2).reshape(b, -1, hd)


class SdpaAttention:
    label = "sdpa"

    def __call__(self, q, k, v, heads):
        return _sdpa(q, k, v, heads)


def make_ltx25_attention(kind: Optional[str], stream_fn=None):
    """Resolve an attention backend name to an ltx-core AttentionCallable.

    ``None`` or ``"auto"`` picks the raw fvk sage2 path when the kernels are
    present, else the sageattention package, else SDPA.
    """
    kind = (kind or "auto").lower()
    if kind == "auto":
        if fvk_sage2_available():
            kind = "sage2-fvk"
        else:
            try:
                importlib.import_module("sageattention")
                kind = "sage2"
            except ModuleNotFoundError as exc:
                # The package not being installed is a reason to fall back.
                # The package being installed and failing to load is not:
                # that is a broken environment, and answering it with SDPA
                # would hide the breakage behind a quiet slowdown.
                if exc.name != "sageattention":
                    raise
                kind = "sdpa"

    if kind == "sage2-fvk":
        if not fvk_sage2_available():
            raise RuntimeError(
                "ltx25 attention 'sage2-fvk' requires flash_rt_kernels with "
                "the sage2 raw entry points")
        return FvkSage2Attention(stream_fn=stream_fn)
    if kind == "sage2":
        return SagePkgAttention()
    if kind == "sage3":
        return Sage3Attention()
    if kind == "sdpa":
        return SdpaAttention()
    raise ValueError(
        f"Unknown ltx25 attention backend: {kind!r} "
        "(expected auto|sage2-fvk|sage2|sage3|sdpa)")
