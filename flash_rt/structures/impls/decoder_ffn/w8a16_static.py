"""Weight-only INT8 implementation of the ``decoder_ffn`` structure.

Composes the fused W8A16 gate/up -> activation -> down block from the
``flashrt/weight-only-ffn`` Hub kernel behind the structure boundary.
Activations stay BF16, so binding needs no calibration data: packing is
a pure weight transform, and qualification still runs the parity gate on
real host activations like every other implementation.

The kernel's optimized dispatch covers the decode band (M in [1, 8]);
larger M is outside the support envelope and is refused at call time
rather than routed to a slow path.
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

_ENTRYPOINTS = {"gelu": "w8a16_geglu_ffn_bf16", "silu": "w8a16_swiglu_ffn_bf16"}

SUPPORT = {
    "D": {"min": 512, "max": 16384, "multiple_of": 64},
    "F": {"min": 1024, "max": 16384, "multiple_of": 64},
    "M": {"min": 1, "max": 8},
    "m_classes": ("micro",),
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _entrypoint(variant: Mapping[str, str]):
    name = variant.get("activation", "gelu")
    if name not in _ENTRYPOINTS:
        raise ValueError(f"unsupported activation: {name!r}")
    return getattr(_kernel(), _ENTRYPOINTS[name])


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w_gate, w_up, w_down = (weights["w_gate"], weights["w_up"],
                            weights["w_down"])
    dim_f, dim_d = w_gate.shape
    if w_up.shape != (dim_f, dim_d) or w_down.shape != (dim_d, dim_f):
        raise ValueError(
            f"inconsistent weight dims: gate {tuple(w_gate.shape)}, "
            f"up {tuple(w_up.shape)}, down {tuple(w_down.shape)}")
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]")
        if dim % bounds["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a multiple of "
                f"{bounds['multiple_of']}")
    return dim_d, dim_f


class BoundDecoderFfnW8A16:
    """MLP-seam callable: normed activations in, FFN output out (BF16)."""

    def __init__(self, ffn_fn, gate_up_q, gate_up_scale, down_q, down_scale,
                 dim_d):
        self._ffn = ffn_fn
        self._gate_up_q = gate_up_q
        self._gate_up_scale = gate_up_scale
        self._down_q = down_q
        self._down_scale = down_scale
        self._dim_d = dim_d

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        shape = normed.shape
        x = normed.reshape(-1, shape[-1])
        m = x.shape[0]
        m_max = SUPPORT["M"]["max"]
        if m > m_max:
            raise ValueError(
                f"M={m} outside the weight-only decode envelope "
                f"[1, {m_max}]")
        variant = 0 if m <= 4 else 3
        out = self._ffn(x.to(torch.bfloat16).contiguous(),
                        self._gate_up_q, self._gate_up_scale,
                        self._down_q, self._down_scale, variant=variant)
        return out.reshape(shape).to(normed.dtype)

    __call__ = ffn


class FusedGluMlpW8A16(GuardedSeam, torch.nn.Module):
    """MLP-seam module with declared M-dispatch.

    The weight-only kernel covers the decode band (M in [1, 8]); calls
    with larger M are dispatched to the retained host module. This is
    part of the declared plan — per-M dispatch on the real workload —
    not a fallback: both paths are first-class and the qualification
    record states which band the kernel serves. The ledger keeps the two
    apart under separate names for exactly that reason, and still counts
    the dispatch: "by design" is a reason for a path to exist, not a
    reason for its share of the calls to be unknown.

    ``original`` is retained whole (host MLP naming varies across model
    families), and attribute lookups fall through to it so hosts that
    introspect the module they call keep working.
    """

    _frt_host_attr = "host_mlp"
    _frt_can_fallback = True

    def __init__(self, bound: BoundDecoderFfnW8A16,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        if original is not None:
            self.host_mlp = original
        guard = self._frt_arm(dtypes=CAST_OK,
                              device=bound._gate_up_q.device,
                              k=int(bound._dim_d))
        guard.notes["dispatched_by_band"] = 0

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
        m = hidden.numel() // hidden.shape[-1]
        if m > SUPPORT["M"]["max"]:
            host = self._frt_host()
            if host is not None:
                guard = self._frt_guard
                if guard is not None and not torch.compiler.is_compiling():
                    guard.notes["dispatched_by_band"] += 1
                return host(hidden)
        return self._bound.ffn(hidden)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    original: torch.nn.Module | None = None,
):
    """Bind the MLP-seam slice of ``decoder_ffn`` with weight-only INT8.

    ``weights`` uses checkpoint-native ``[out, in]`` projection layout
    (``w_gate``/``w_up``: ``[F, D]``, ``w_down``: ``[D, F]``). No
    calibration data is required: quantization is per-output-channel on
    weights only.
    """
    dim_d, _ = _check(weights)
    k = _kernel()
    ffn_fn = _entrypoint(variant)
    gate_up = torch.cat(
        [weights["w_gate"].to("cuda", torch.bfloat16),
         weights["w_up"].to("cuda", torch.bfloat16)], dim=0).contiguous()
    down = weights["w_down"].to("cuda", torch.bfloat16).contiguous()
    gate_up_q, gate_up_scale = k.quantize_w8_weight_bf16(gate_up)
    down_q, down_scale = k.quantize_w8_weight_bf16(down)
    bound = BoundDecoderFfnW8A16(
        ffn_fn, gate_up_q, gate_up_scale, down_q, down_scale, dim_d)
    return FusedGluMlpW8A16(bound, original=original)
