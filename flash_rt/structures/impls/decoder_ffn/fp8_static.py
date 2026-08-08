"""FP8-static implementation of the ``decoder_ffn`` structure.

Composes the fused FP8 gate/up -> activation -> down block from the
``flashrt/flashrt-fp8-swiglu-ffn`` Hub kernel behind the structure
boundary. Two bind entrypoints share the packing and calibration code:
``bind`` covers the full structure (norm and AdaLN modulation run in
torch ahead of the fused block); ``bind_mlp_seam`` covers the
normed-input -> ffn-output slice for hosts whose replaceable module
boundary is the MLP. Activation scales are static per-tensor,
calibrated from caller-provided representative inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch

from ...guard import CAST_OK, FP8_ONLY, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-swiglu-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_ENTRYPOINTS = {"gelu": "bf16_fp8_geglu_mlp_bf16",
                "silu": "bf16_fp8_swiglu_mlp_bf16"}
# fp8 entry: the upstream producer already emitted fp8 with the shared
# activation scale, so the kernel's own input quantization is dead work.
# Same math, one less kernel per call.
_ENTRYPOINTS_FP8 = {"gelu": "fp8_geglu_mlp_bf16",
                    "silu": "fp8_swiglu_mlp_bf16"}

SUPPORT = {
    "D": {"min": 512, "max": 16384},
    "F": {"min": 1024, "max": 16384},
    "m_classes": ("micro", "small", "medium"),
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _activation(variant: Mapping[str, str]) -> tuple[str, Callable]:
    name = variant.get("activation", "gelu")
    if name not in _ENTRYPOINTS:
        raise ValueError(f"unsupported activation: {name!r}")
    if name == "gelu":
        return name, lambda t: torch.nn.functional.gelu(t, approximate="tanh")
    return name, torch.nn.functional.silu


def _amax_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


def _quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)


def _normalize(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    mode: str,
    cond_scale: torch.Tensor | None,
    cond_shift: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    if mode == "offset":
        h = h * (1.0 + w_norm.float())
    elif mode == "direct":
        h = h * w_norm.float()
    else:
        raise ValueError(f"unknown norm_weight_mode: {mode!r}")
    if cond_scale is not None:
        h = h * (1.0 + cond_scale.float())
    if cond_shift is not None:
        h = h + cond_shift.float()
    return h.to(torch.bfloat16)


def _check_and_pack(weights: Mapping[str, torch.Tensor]):
    """Validate dims against the support envelope; pack FP8 weights."""
    w_gate, w_up, w_down = weights["w_gate"], weights["w_up"], weights["w_down"]
    dim_d, dim_f = w_gate.shape
    if w_up.shape != (dim_d, dim_f) or w_down.shape != (dim_f, dim_d):
        raise ValueError(
            f"inconsistent weight dims: gate {tuple(w_gate.shape)}, "
            f"up {tuple(w_up.shape)}, down {tuple(w_down.shape)}"
        )
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]"
            )
    if not (w_gate.is_cuda and w_up.is_cuda and w_down.is_cuda):
        raise ValueError("fp8_static requires CUDA-resident weights")
    gate_up = torch.cat([w_gate.t(), w_up.t()], dim=0).contiguous()
    down = w_down.t().contiguous()
    return gate_up, down, _amax_scale(gate_up), _amax_scale(down)


def _calibrate_scales(
    normed_samples: Sequence[torch.Tensor],
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    act: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static per-tensor input/hidden scales from normed activations."""
    if not normed_samples:
        raise ValueError("calibration samples must be non-empty")
    device = w_gate.device
    input_amax = torch.zeros((), device=device)
    hidden_amax = torch.zeros((), device=device)
    with torch.no_grad():
        for h in normed_samples:
            flat = h.reshape(-1, h.shape[-1]).float().to(device)
            hidden = act(flat @ w_gate.float()) * (flat @ w_up.float())
            input_amax = torch.maximum(input_amax, flat.abs().max())
            hidden_amax = torch.maximum(hidden_amax, hidden.abs().max())
    return ((input_amax / _FP8_MAX).clamp(min=1e-8),
            (hidden_amax / _FP8_MAX).clamp(min=1e-8))


@dataclass(frozen=True)
class BoundDecoderFfnFp8:
    """Bound callable for the full structure boundary."""

    fused_mlp: Callable[..., torch.Tensor]
    w_norm: torch.Tensor
    gate_up_fp8: torch.Tensor
    down_fp8: torch.Tensor
    input_scale: torch.Tensor
    gate_up_scale: torch.Tensor
    hidden_scale: torch.Tensor
    down_scale: torch.Tensor
    norm_weight_mode: str
    eps: float
    in_dtype: str = "bf16"

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        """The normed-input -> ffn-output slice (no norm, no residual).

        On the BF16 entry the kernel quantizes the input itself; on the
        FP8 entry the producer already did, and the input passes
        straight through."""
        shape = normed.shape
        if getattr(self, "in_dtype", "bf16") == "fp8_static":
            out = self.fused_mlp(
                normed.reshape(-1, shape[-1]),
                self.gate_up_fp8, self.down_fp8,
                self.input_scale.view(1), self.gate_up_scale.view(1),
                self.hidden_scale.view(1), self.down_scale.view(1))
            return out.reshape(*shape[:-1], out.shape[-1])
        out = self.fused_mlp(
            normed.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous(),
            self.gate_up_fp8,
            self.down_fp8,
            self.input_scale.view(1),
            self.gate_up_scale.view(1),
            self.hidden_scale.view(1),
            self.down_scale.view(1),
        )
        return out.reshape(shape).to(normed.dtype)

    def __call__(
        self,
        x: torch.Tensor,
        *,
        cond_scale: torch.Tensor | None = None,
        cond_shift: torch.Tensor | None = None,
        cond_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = _normalize(x, self.w_norm, self.norm_weight_mode,
                       cond_scale, cond_shift, self.eps)
        out = self.ffn(h)
        if cond_gate is not None:
            out = out * cond_gate
        return x + out.to(x.dtype)


class FusedGeGluMlp(GuardedSeam, torch.nn.Module):
    """MLP-seam module for hosts whose replaceable boundary is the MLP.

    The host keeps its own norm, AdaLN gate, and residual. ``original``
    is retained whole (host MLP naming varies across model families), and
    attribute lookups fall through to it so hosts that introspect the
    projection attributes of the module they call keep working. Retaining
    it is also what makes the seam reversible per call: an input outside
    the calibrated form runs the host MLP instead of this kernel.
    """

    _frt_host_attr = "host_mlp"
    _frt_can_fallback = True

    def __init__(self, bound: BoundDecoderFfnFp8,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        if original is not None:
            self.host_mlp = original
        self._frt_arm(
            dtypes=(FP8_ONLY if bound.in_dtype == "fp8_static" else CAST_OK),
            device=bound.gate_up_fp8.device,
            k=int(bound.gate_up_fp8.shape[1]))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_mlp":
                raise
            return getattr(super().__getattr__("host_mlp"), name)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden)
        if admitted is not PROCEED:
            return admitted
        return self._bound.ffn(hidden)


def _build(weights, variant, input_scale, hidden_scale, eps):
    name, _ = _activation(variant)
    gate_up, down, gate_up_scale, down_scale = _check_and_pack(weights)
    in_dtype = variant.get("in_dtype", "bf16")
    table = (_ENTRYPOINTS_FP8 if in_dtype == "fp8_static"
             else _ENTRYPOINTS)
    return BoundDecoderFfnFp8(
        fused_mlp=getattr(_kernel(), table[name]),
        w_norm=weights["w_norm"],
        gate_up_fp8=_quantize(gate_up, gate_up_scale),
        down_fp8=_quantize(down, down_scale),
        input_scale=input_scale,
        gate_up_scale=gate_up_scale,
        hidden_scale=hidden_scale,
        down_scale=down_scale,
        in_dtype=in_dtype,
        norm_weight_mode=variant.get("norm_weight_mode", "offset"),
        eps=eps,
    )


@torch.no_grad()
def bind(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    calibration_inputs: Sequence[Mapping[str, torch.Tensor]],
    eps: float = 1e-6,
) -> BoundDecoderFfnFp8:
    """Bind the full structure: calibration inputs are boundary inputs.

    ``calibration_inputs`` must be drawn from the real input distribution
    of the target binding; static FP8 scales are only as trustworthy as
    the data they were measured on.
    """
    if not calibration_inputs:
        raise ValueError("calibration_inputs must be non-empty")
    _, act = _activation(variant)
    mode = variant.get("norm_weight_mode", "offset")
    normed = [
        _normalize(sample["x"], weights["w_norm"], mode,
                   sample.get("cond_scale"), sample.get("cond_shift"), eps)
        for sample in calibration_inputs
    ]
    input_scale, hidden_scale = _calibrate_scales(
        normed, weights["w_gate"], weights["w_up"], act)
    return _build(weights, variant, input_scale, hidden_scale, eps)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    input_scale: float,
    hidden_scale: float,
    original: torch.nn.Module | None = None,
    eps: float = 1e-6,
) -> FusedGeGluMlp:
    """Bind the MLP-seam slice from two already-calibrated scales.

    The scales arrive measured, not derived: ``input_scale`` is the amax at
    this MLP's input and ``hidden_scale`` the amax at its down
    projection's input — which is exactly the gated activation this kernel
    quantises. Recomputing the second one here would mean keeping the
    seam's inputs alive to run gate/up over them again, and the amax it
    would arrive at is the one the host already produced.

    Both are per-tensor FP8 scales (amax/448), reduced across calibration
    samples by the caller through ``flash_rt.core.calibration``.
    """
    dev = weights["w_gate"].device
    bound = _build(weights, variant,
                   torch.tensor(float(input_scale), device=dev),
                   torch.tensor(float(hidden_scale), device=dev), eps)
    return FusedGeGluMlp(bound, original=original)
