"""FP8-static implementation of the ``linear_proj`` structure.

A projection has more than one executable form, and which one applies is
a property of the host's own weights rather than a tuning knob:

- ``bias``: ``bf16_fp8_linear_bias_bf16`` — fused input quantization,
  FP8 weights, BF16 bias and output. Three kernels per call.
- ``no_bias``: the quantize as its own kernel followed by
  ``fp8_gemm_bf16``. Two kernels. Hosts whose projections carry no bias
  (the whole Gemma/Llama/Qwen attention family) would otherwise get a
  zero bias built for them and pay a kernel to add it.
- ``fp8_in``: ``fp8_gemm_bf16`` straight, for a seam whose producer
  already emits FP8. One kernel, no quantization at all.

Each form is qualified by its own work band, because a band expresses
what that form is amortizing and the forms do not carry the same cost.
Measured against the host's own BF16 Linear at real shapes on RTX 5090
(work = M*N*K):

    form      band            evidence
    bias      [2e8, inf)      1.05e8 ties the host (6.68 vs 6.66 us);
                              3.0e9 wins 2.5x (26.0 vs 66.6) — the
                              fused quantize needs a large GEMM to
                              disappear behind
    no_bias   [2e7, 1e9]      2.6e7 wins 0.67us, 1.3e7 loses 0.25us
                              (one quantize launch to amortize); above
                              the band it *loses* to the bias form even
                              with no bias to add — 3.0e9: 26.21 vs
                              26.04 — because the fused entry bundles
                              its quantize instead of launching one
    fp8_in    [0, inf)        nothing to amortize; the producer paid

Hence a large projection with no bias still takes the bias form and gets
a zero bias built for it: that is the measured faster path at that size,
not an oversight. The rule is which form the measurements put this shape
in, not which parts the host happens to have.

Outside every band the projection is refused and the refusal names the
form, so "refused" never reads as "this projection cannot be bound" —
only as "not in that form, at this size".
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import torch

from ...guard import CAST_OK, FP8_ONLY, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-ffn",
    "version": ">=1",
}
QUANT_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-gemm-epilogues",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

# per-form work bands (M*N*K), each measured against the host's own
# BF16 Linear at real shapes — see the module docstring
_BAND = {"bias": (2.0e8, float("inf")),
         "no_bias": (2.0e7, 1.0e9),
         "fp8_in": (0.0, float("inf"))}

SUPPORT = {
    "work_band": _BAND,
    "K": {"min": 512, "max": 16384},
    "N": {"min": 128, "max": 16384},
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


@lru_cache(maxsize=1)
def _quant_kernel():
    """The standalone quantize the no-bias form needs, or None.

    Its absence is not an error: the form simply does not qualify and
    the projection falls back to the bias form's floor.
    """
    from flash_rt.structures.impls import hub_kernel

    try:
        return hub_kernel(QUANT_DEP["repo"], QUANT_DEP["version"])
    except Exception:                                   # noqa: BLE001
        return None


def _amax_scale(t: torch.Tensor) -> torch.Tensor:
    return (t.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


class FusedLinearProj(GuardedSeam, torch.nn.Module):
    """Drop-in replacement for one nn.Linear projection.

    ``original`` is retained whole and attribute lookups fall through to
    it, so host code that introspects ``weight``/``bias``/``in_features``
    keeps working — and a call outside the calibrated form runs it.
    """

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, w_fp8, bias, input_scale, weight_scale,
                 original: torch.nn.Module | None = None,
                 form: str = "bias"):
        super().__init__()
        self._w_fp8 = w_fp8
        self._bias = bias
        self._input_scale = input_scale
        self._weight_scale = weight_scale
        self._bufs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.form = form
        # resolve the op at bind time: calling the hub loader inside
        # forward makes dynamo trace through kernels.get_kernel's
        # version resolution (network + inspect.Signature) — 26 graph
        # breaks that fragment the surrounding compiled region
        kf = _kernel()
        self._gemm = kf.fp8_gemm_bf16
        self._fn = kf.bf16_fp8_linear_bias_bf16
        if form == "no_bias":
            ke = _quant_kernel()
            self._quantize = ke.channel_scale_quantize_fp8_static_bf16
            # the standalone quantize is a per-channel one; a flat
            # per-tensor scale is the identity channel vector. Held like
            # the other tensors here, as a plain attribute.
            self._chan = torch.ones(w_fp8.shape[1], device=w_fp8.device,
                                    dtype=torch.bfloat16)
        if original is not None:
            self.host_linear = original
        self._frt_arm(
            dtypes=FP8_ONLY if form == "fp8_in" else CAST_OK,
            device=w_fp8.device, k=int(w_fp8.shape[1]))

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
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        m = flat.shape[0]
        if torch.compiler.is_compiling():
            # traced regions run the ops functionally: a persistent
            # out-buffer is module state mutated by the op, which
            # functionalization cannot rewrite (the whole-graph export
            # hit exactly that), and inside a compiled graph the
            # allocation is planned away regardless
            x_fp8 = out = None
        else:
            bufs = self._bufs.get(m)
            if bufs is None:
                bufs = (torch.empty(m, self._w_fp8.shape[1],
                                    device=x.device, dtype=_FP8),
                        torch.empty(m, self._w_fp8.shape[0],
                                    device=x.device,
                                    dtype=torch.bfloat16))
                self._bufs[m] = bufs
            x_fp8, out = bufs
        if self.form == "fp8_in":
            y = self._gemm(flat, self._w_fp8, self._input_scale,
                           self._weight_scale, out=out)
            return y.reshape(*shape[:-1], y.shape[-1])
        flat = flat.to(torch.bfloat16).contiguous()
        if self.form == "no_bias":
            self._quantize(flat, self._chan, self._input_scale, out=x_fp8)
            y = self._gemm(x_fp8, self._w_fp8, self._input_scale,
                           self._weight_scale, out=out)
        else:
            y = self._fn(flat, self._w_fp8, self._bias, self._input_scale,
                         self._weight_scale, input_fp8=x_fp8, out=out)
        return y.reshape(*shape[:-1], y.shape[-1]).to(x.dtype)


def _form_for(bias: torch.Tensor | None, in_dtype: str,
              work: float) -> str:
    """Which form this projection takes, from what it is *and* its size.

    Two inputs, not one. What the host has decides which forms are
    available: a projection fed FP8 has no input to quantize, one with a
    real bias must add it. Size decides which of the available forms is
    actually faster — the no-bias form trades a bundled quantize for a
    separate launch, which is a win only while the GEMM is short enough
    for a launch to matter. Above its band a projection with no bias is
    better off in the bias form with a zero bias, and that is measured,
    not conceded.
    """
    if in_dtype == "fp8_static":
        return "fp8_in"
    has_bias = bias is not None and bool(bias.any())
    lo, hi = _BAND["no_bias"]
    if not has_bias and _quant_kernel() is not None and work <= hi:
        return "no_bias"
    return "bias"


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    input_scale: float,
    row_profile: Sequence[int],
    original: torch.nn.Module | None = None,
    in_dtype: str = "bf16",
) -> FusedLinearProj:
    """Bind one projection: ``weights['w']`` is checkpoint-layout [N, K].

    ``input_scale`` is the calibrated per-tensor FP8 scale at this
    projection's input. ``row_profile`` is the row counts that input
    arrived with across calibration — a shape observation, not a
    statistic, and the median of it is what the work-based form
    qualification is measured against.
    """
    if not row_profile:
        raise ValueError("row_profile must be non-empty")
    w = weights["w"]
    n, k = w.shape
    for name, dim in (("K", k), ("N", n)):
        lo, hi = SUPPORT[name]["min"], SUPPORT[name]["max"]
        if not lo <= dim <= hi:
            raise ValueError(f"{name}={dim} outside support envelope")
    raw_bias = weights.get("b")
    ms = sorted(int(m) for m in row_profile)
    m_med = ms[len(ms) // 2]
    work = float(m_med) * n * k
    form = _form_for(raw_bias, in_dtype, work)
    lo, _ = _BAND[form]
    if work < lo:
        raise ValueError(
            f"projection work {m_med}x{n}x{k} below the {form} form's "
            f"band ({lo:.0e}) — host keeps its Linear. Bands are per "
            "form: the same projection may qualify in another one")
    if not w.is_cuda:
        raise ValueError("fp8_static requires CUDA-resident weights")

    device = w.device
    w_scale = _amax_scale(w)
    w_fp8 = (w.float() / w_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    scale = torch.tensor(float(input_scale), device=device)
    bias = raw_bias
    if bias is None:
        bias = torch.zeros(n, device=device, dtype=torch.bfloat16)
    else:
        bias = bias.detach().to(torch.bfloat16)
    bound = FusedLinearProj(w_fp8, bias, scale.view(1),
                            w_scale.view(1), original=original, form=form)
    for m in set(ms):  # pre-allocate per calibrated M: keeps the hot
        bound._bufs[m] = (  # path allocation-free (graph/compile safe)
            torch.empty(m, k, device=device, dtype=_FP8),
            torch.empty(m, n, device=device, dtype=torch.bfloat16))
    return bound
