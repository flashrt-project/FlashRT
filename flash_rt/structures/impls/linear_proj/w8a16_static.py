"""Weight-only INT8 implementation of the ``linear_proj`` structure.

The decode-band twin of the FP8 projection impl: weights are quantized
per output channel to INT8 at bind time, activations stay BF16, so the
seam needs no calibration data at all. This is the projection-shaped
slice of the recipe already shipped for ``decoder_ffn`` —
``w8a16_static`` there covers the gated MLP, this file covers the
attention Q/K/V/O family and any other single projection the discovery
qualifies.

The ``flashrt/weight-only-ffn`` package's linear entry point qualifies
its auto dispatch narrowly, and this impl mirrors that table exactly
rather than stretching it (``torch_binding.cpp``: ``check_variant`` and
``w8_auto_linear_supported``):

- M in [1, 4] — the decode band; and
- K <= 1024 always qualifies; K <= 4096 needs N >= 1024; larger K needs
  N >= 1024 for M <= 2 and N >= 2048 for M in {3, 4}.

Calls outside the band are dispatched to the retained host module by
declared plan, counted in the ledger — prefill runs the host GEMM,
decode runs the kernel, and the qualification record states which band
the kernel serves.

The linear entry point carries no bias operand. A projection with a
bias gets it added in BF16 after the GEMM — one [M<=4, N] elementwise
add inside the decode band, where the weight read dominates end to end.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/weight-only-ffn",
    "version": ">=1",
}

SUPPORT = {
    "K": {"min": 512, "max": 16384},
    "N": {"min": 128, "max": 262144},
    "M": {"min": 1, "max": 4},
    "m_classes": ("micro",),
}


def _qualified(m: int, n: int, k: int) -> bool:
    """The kernel's own auto-dispatch qualification, mirrored.

    Copied from ``w8_auto_linear_supported`` plus the ``variant=0``
    M-bound in the package's ``torch_binding.cpp`` — the kernel raises
    outside this table, so the band dispatch must agree with it, not
    rediscover it as runtime errors.
    """
    if not SUPPORT["M"]["min"] <= m <= SUPPORT["M"]["max"]:
        return False
    if k <= 1024:
        return True
    if k <= 4096:
        return n >= 1024
    return n >= (1024 if m <= 2 else 2048)


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w = weights["w"]
    if w.dim() != 2:
        raise ValueError(f"w must be [N, K], got {tuple(w.shape)}")
    n, k = w.shape
    for name, dim in (("K", k), ("N", n)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]")
    b = weights.get("b")
    if b is not None and tuple(b.shape) != (n,):
        raise ValueError(
            f"bias shape {tuple(b.shape)} does not match N={n}")
    return n, k


class BoundLinearProjW8A16:
    """Projection callable: x[M, K] in, y[M, N] out (BF16)."""

    def __init__(self, linear_fn, w_q, w_scale, bias, n, k):
        self._linear = linear_fn
        self._w_q = w_q
        self._w_scale = w_scale
        self._bias = bias
        self._n = n
        self._k = k

    def project(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        m = flat.shape[0]
        if not _qualified(m, self._n, self._k):
            raise ValueError(
                f"M={m} outside the W8A16 auto-dispatch qualification "
                f"for N={self._n}, K={self._k} (decode band M in "
                f"[1, {SUPPORT['M']['max']}])")
        y = self._linear(flat.to(torch.bfloat16).contiguous(),
                         self._w_q, self._w_scale)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self._n).to(x.dtype)

    __call__ = project


class LinearProjW8A16(GuardedSeam, torch.nn.Module):
    """Drop-in projection module with declared M-dispatch.

    The weight-only kernel covers the decode band; calls with larger M
    (prefill) are dispatched to the retained host module. This is part
    of the declared plan — per-M dispatch on the real workload — not a
    fallback: both paths are first-class, and the ledger counts the
    dispatch so neither path's share of the calls is ever unknown.

    ``original`` is retained whole and attribute lookups fall through to
    it, so host code that introspects ``weight``/``in_features`` keeps
    working.
    """

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, bound: BoundLinearProjW8A16,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        # the same tensors, reachable through *module* attributes: an
        # exporter attributes a tensor by its access path, and a tensor
        # reached only through a plain object gets lifted as an
        # anonymous immutable constant — unnameable in a
        # weights-external package. Identity is unchanged.
        self.register_buffer("_frt_w_q", bound._w_q)
        self.register_buffer("_frt_w_scale", bound._w_scale)
        if bound._bias is not None:
            self.register_buffer("_frt_bias", bound._bias)
        else:
            self._frt_bias = None
        if original is not None:
            self.host_linear = original
        guard = self._frt_arm(dtypes=CAST_OK,
                              device=bound._w_q.device,
                              k=int(bound._k))
        guard.notes["dispatched_by_band"] = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_linear":
                raise
            return getattr(super().__getattr__("host_linear"), name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        m = x.numel() // x.shape[-1]
        if not _qualified(m, self._bound._n, self._bound._k):
            host = self._frt_host()
            if host is not None:
                guard = self._frt_guard
                if guard is not None and not torch.compiler.is_compiling():
                    guard.notes["dispatched_by_band"] += 1
                return host(x)
            return self._bound.project(x)   # states the refusal
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        y = self._bound._linear(flat.to(torch.bfloat16).contiguous(),
                                self._frt_w_q, self._frt_w_scale)
        if self._frt_bias is not None:
            y = y + self._frt_bias
        return y.reshape(*shape[:-1], self._bound._n).to(x.dtype)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    original: torch.nn.Module | None = None,
) -> LinearProjW8A16:
    """Bind one projection with weight-only INT8.

    ``weights['w']`` is checkpoint-layout ``[N, K]``, exactly what the
    kernel consumes — no transpose. No calibration data is required:
    quantization is per-output-channel on weights only, and the optional
    ``weights['b']`` is kept in BF16.
    """
    n, k = _check(weights)
    if not _qualified(1, n, k):
        raise ValueError(
            f"refused: N={n}, K={k} has no qualified fast path even at "
            f"M=1; the W8A16 projection cannot serve this seam at any M")
    kern = _kernel()
    w = weights["w"].to("cuda", torch.bfloat16).contiguous()
    w_q, w_scale = kern.quantize_w8_weight_bf16(w)
    bias = weights.get("b")
    if bias is not None:
        bias = bias.detach().to("cuda", torch.bfloat16)
    bound = BoundLinearProjW8A16(kern.w8a16_linear_bf16, w_q, w_scale,
                                 bias, n, k)
    # bind-time smoke: one M=1 launch through the real entry point before
    # the seam is handed out. A stale build or missing symbol must
    # surface here as a clean bind refusal, not later inside the host's
    # forward — identical output cannot catch it there, because the
    # fallback path is numerically exact.
    probe = bound.project(torch.zeros(1, k, device=w_q.device,
                                      dtype=torch.bfloat16))
    if probe.shape != (1, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: w8a16 bind smoke produced shape "
            f"{tuple(probe.shape)}, finite={bool(torch.isfinite(probe).all())}")
    return LinearProjW8A16(bound, original=original)
