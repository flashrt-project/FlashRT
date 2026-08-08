"""FP8-static implementation of the ``vision_ffn`` structure.

Composes the fused FP8 fc1 -> GELU -> fc2 block (biases included) from
the ``flashrt/flashrt-fp8-ffn`` Hub kernel. ``bind`` covers the full
structure boundary; ``bind_mlp_seam`` covers the normed-input ->
ffn-output slice for hosts whose replaceable module boundary is the MLP.
Weights use the checkpoint-native (out, in) layout directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Sequence

import torch

from ...guard import CAST_OK, FP8_ONLY, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

SUPPORT = {
    "D": {"min": 512, "max": 16384},
    "F": {"min": 1024, "max": 16384},
    "m_classes": ("small", "medium", "large"),
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _amax_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.float().abs().max() / _FP8_MAX).clamp(min=1e-8)


def _quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)


@dataclass(frozen=True)
class BoundVisionFfnFp8:
    """Bound callable for the full structure boundary."""

    fused_mlp: Callable[..., torch.Tensor]
    w_norm: torch.Tensor | None
    b_norm: torch.Tensor | None
    fc1_fp8: torch.Tensor
    b_fc1: torch.Tensor
    fc2_fp8: torch.Tensor
    b_fc2: torch.Tensor
    input_scale: torch.Tensor
    fc1_scale: torch.Tensor
    hidden_scale: torch.Tensor
    fc2_scale: torch.Tensor
    eps: float
    in_dtype: str = "bf16"

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        """The normed-input -> ffn-output slice (no norm, no residual).

        On the BF16 entry the kernel quantizes the input itself; on the
        FP8 entry an upstream producer already did, with the shared
        activation scale, so the input passes straight through."""
        shape = normed.shape
        if getattr(self, "in_dtype", "bf16") == "fp8_static":
            out = self.fused_mlp(
                normed.reshape(-1, shape[-1]),
                self.fc1_fp8, self.b_fc1, self.fc2_fp8, self.b_fc2,
                self.input_scale.view(1), self.fc1_scale.view(1),
                self.hidden_scale.view(1), self.fc2_scale.view(1))
            return out.reshape(*shape[:-1], out.shape[-1])
        out = self.fused_mlp(
            normed.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous(),
            self.fc1_fp8,
            self.b_fc1,
            self.fc2_fp8,
            self.b_fc2,
            self.input_scale.view(1),
            self.fc1_scale.view(1),
            self.hidden_scale.view(1),
            self.fc2_scale.view(1),
        )
        return out.reshape(shape).to(normed.dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.layer_norm(
            x.float(), (x.shape[-1],),
            (self.w_norm.float() if self.w_norm is not None else None),
            (self.b_norm.float() if self.b_norm is not None else None),
            self.eps).to(x.dtype)
        return x + self.ffn(h).to(x.dtype)


class FusedGeluMlp(GuardedSeam, torch.nn.Module):
    """MLP-seam module: the host keeps its own norm and residual.

    ``original`` is retained whole (host MLP naming varies across model
    families), and attribute lookups fall through to it so hosts that
    introspect the module they call keep working. It is also the per-call
    way back: an input outside the calibrated form runs the host MLP.
    """

    _frt_host_attr = "host_mlp"
    _frt_can_fallback = True

    def __init__(self, bound: BoundVisionFfnFp8,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        if original is not None:
            self.host_mlp = original
        self._frt_arm(
            dtypes=(FP8_ONLY if bound.in_dtype == "fp8_static" else CAST_OK),
            device=bound.fc1_fp8.device,
            k=int(bound.fc1_fp8.shape[1]))

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


def _calibrate(normed_samples, w_fc1, b_fc1):
    if not normed_samples:
        raise ValueError("calibration samples must be non-empty")
    device = w_fc1.device
    input_amax = torch.zeros((), device=device)
    hidden_amax = torch.zeros((), device=device)
    for h in normed_samples:
        flat = h.reshape(-1, h.shape[-1]).float().to(device)
        hidden = torch.nn.functional.gelu(
            flat @ w_fc1.float().t() + b_fc1.float(), approximate="tanh")
        input_amax = torch.maximum(input_amax, flat.abs().max())
        hidden_amax = torch.maximum(hidden_amax, hidden.abs().max())
    return ((input_amax / _FP8_MAX).clamp(min=1e-8),
            (hidden_amax / _FP8_MAX).clamp(min=1e-8))


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w_fc1, w_fc2 = weights["w_fc1"], weights["w_fc2"]
    dim_f, dim_d = w_fc1.shape
    if w_fc2.shape != (dim_d, dim_f):
        raise ValueError(
            f"inconsistent weight dims: fc1 {tuple(w_fc1.shape)}, "
            f"fc2 {tuple(w_fc2.shape)}"
        )
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]"
            )
    if not (w_fc1.is_cuda and w_fc2.is_cuda):
        raise ValueError("fp8_static requires CUDA-resident weights")
    return dim_d, dim_f


def _build(weights, input_scale, hidden_scale, eps, variant=None):
    variant = variant or {}
    _check(weights)
    fc1_scale = _amax_scale(weights["w_fc1"])
    fc2_scale = _amax_scale(weights["w_fc2"])
    to_bf16 = lambda t: t.to(torch.bfloat16)
    # capability probe: the v2 entries carry the down bias in the GEMM
    # epilogue (one launch and one full output write fewer). Prefer
    # them when the installed package ships them; absence is a
    # fallback, never a refusal.
    kern = _kernel()
    if variant.get("in_dtype") == "fp8_static":
        fused = (getattr(kern, "fp8_gelu_mlp_v2_bf16", None)
                 or kern.fp8_gelu_mlp_bf16)
    else:
        fused = (getattr(kern, "bf16_fp8_gelu_mlp_v2_bf16", None)
                 or kern.bf16_fp8_gelu_mlp_bf16)
    return BoundVisionFfnFp8(
        fused_mlp=fused,
        in_dtype=variant.get("in_dtype", "bf16"),
        w_norm=weights["w_norm"],
        b_norm=weights["b_norm"],
        fc1_fp8=_quantize(weights["w_fc1"], fc1_scale),
        b_fc1=to_bf16(weights["b_fc1"]),
        fc2_fp8=_quantize(weights["w_fc2"], fc2_scale),
        b_fc2=to_bf16(weights["b_fc2"]),
        input_scale=input_scale,
        fc1_scale=fc1_scale,
        hidden_scale=hidden_scale,
        fc2_scale=fc2_scale,
        eps=eps,
    )


@torch.no_grad()
def bind(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    calibration_inputs: Sequence[Mapping[str, torch.Tensor]],
    eps: float = 1e-6,
) -> BoundVisionFfnFp8:
    """Bind the full structure: calibration inputs are boundary inputs."""
    if variant.get("activation", "gelu") != "gelu":
        raise ValueError("vision_ffn fp8_static supports gelu only")
    if not calibration_inputs:
        raise ValueError("calibration_inputs must be non-empty")
    normed = [
        torch.nn.functional.layer_norm(
            s["x"].float(), (s["x"].shape[-1],),
            (weights["w_norm"].float()
             if weights["w_norm"] is not None else None),
            (weights["b_norm"].float()
             if weights["b_norm"] is not None else None),
            eps)
        for s in calibration_inputs
    ]
    input_scale, hidden_scale = _calibrate(
        normed, weights["w_fc1"], weights["b_fc1"])
    return _build(weights, input_scale, hidden_scale, eps)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    input_scale: float,
    hidden_scale: float,
    original: torch.nn.Module | None = None,
    eps: float = 1e-6,
) -> FusedGeluMlp:
    """Bind the MLP-seam slice from two already-calibrated scales.

    ``input_scale`` is the amax at this MLP's input, ``hidden_scale`` the
    amax at its second projection's input — which is the post-activation
    hidden this kernel quantises. Measured where they are, not recomputed
    from kept inputs.
    """
    dev = weights["w_fc1"].device
    bound = _build(weights,
                   torch.tensor(float(input_scale), device=dev),
                   torch.tensor(float(hidden_scale), device=dev), eps)
    return FusedGeluMlp(bound, original=original)
