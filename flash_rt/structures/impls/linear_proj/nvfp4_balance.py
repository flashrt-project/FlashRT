"""NVFP4 (W4A4) ``linear_proj`` with per-input-channel balance.

The W4 recipe as a formal impl variant: the weight is folded with the
activation-only channel balance fitted on calibrated per-channel amax
(``fit_input_channel_balance`` — the production formula), packed to
NVFP4 (E2M1 data plus per-16-element-block scale factors), and the
activation is quantized dynamically per call after the inverse fold.
The fold is exact (``x' @ W'.T == x @ W.T``) and only then is either
side quantized, so the balance costs no arithmetic identity — it moves
quantization error out of the hot channels.

Activations carry dynamic per-block scales computed per call, so there
is no static activation scale to drift across a denoise schedule; the
per-channel amax calibration feeds the *balance*, not a scale.

Unlike :mod:`.nvfp4_dynamic` (the adoption path for checkpoints that
are already packed and have no host form to return to), this variant
retains the host module whole: a call outside the calibrated form runs
it, and detach restores it bit-exact.
"""

from __future__ import annotations

from typing import Mapping

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam
from .nvfp4_dynamic import KERNEL_DEP, _check, _kernel  # noqa: F401

_VARIANT = 2  # the qualified GEMM dispatch across the served shapes


class LinearProjNvfp4Balance(GuardedSeam, torch.nn.Module):
    """Balanced-fold projection: dynamic FP4 quantize + FP4 GEMM."""

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, w_packed, w_sfb, inv_s, bias, n, k,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self.register_buffer("_w_packed", w_packed)
        self.register_buffer("_w_sfb", w_sfb)
        self.register_buffer("_inv_s", inv_s)
        if bias is not None:
            self.register_buffer("_bias", bias)
        else:
            self._bias = None
        self._n = n
        kern = _kernel()
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        # capability probe: the fused-bias epilogue entry, where the
        # installed package variant ships it — absence is the ordinary
        # two-launch path, never a refusal
        self._gemm_bias = getattr(kern, "nvfp4_gemm_bias_bf16", None)
        if original is not None:
            self.host_linear = original
        self._frt_arm(dtypes=CAST_OK, device=w_packed.device, k=int(k))

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
        flat = (x.reshape(-1, shape[-1]).to(torch.float16)
                * self._inv_s).contiguous()
        a_packed, a_sfa = self._kern.quantize_fp4_sfa_fp16(flat)
        if self._bias is not None and self._gemm_bias is not None:
            y = self._gemm_bias(a_packed, self._w_packed, a_sfa,
                                self._w_sfb, self._bias)
        else:
            y = self._gemm(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           variant=_VARIANT)
            if self._bias is not None:
                y = y + self._bias
        return y.reshape(*shape[:-1], self._n).type_as(x)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    channel_amax,
    original: torch.nn.Module | None = None,
    alpha: float = 0.5,
    clamp=(0.25, 4.0),
) -> LinearProjNvfp4Balance:
    """Bind one projection from a dense ``[N, K]`` weight.

    ``channel_amax`` is the calibrated per-input-channel amax vector at
    this projection's input (``[K]``) — it parameterises the balance
    fold, never a scale. ``alpha``/``clamp`` are the decision's recipe
    payload.
    """
    from flash_rt.core.quantization import fit_input_channel_balance

    n, k = _check(weights)
    kern = _kernel()
    w = weights["w"].detach()
    amax = torch.as_tensor(channel_amax, device=w.device,
                           dtype=torch.float32)
    w_bal, inv_s = fit_input_channel_balance(
        w.float(), amax, alpha=alpha,
        clamp=(float(clamp[0]), float(clamp[1])),
        out_dtype=torch.float32)
    w_packed, w_sfb = kern.quantize_fp4_sfa_fp16(
        w_bal.to("cuda", torch.float16).contiguous(), is_sfb=True)
    bias = weights.get("b")
    if bias is not None:
        bias = bias.detach().to("cuda", torch.bfloat16)
    bound = LinearProjNvfp4Balance(
        w_packed, w_sfb, inv_s.to("cuda", torch.float16), bias, n, k,
        original=original)
    probe = bound(torch.zeros(1, k, device=w_packed.device,
                              dtype=torch.bfloat16))
    if probe.shape != (1, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: nvfp4_balance bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound
