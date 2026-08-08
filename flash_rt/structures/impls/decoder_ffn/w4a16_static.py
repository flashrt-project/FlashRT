"""Weight-only NVFP4 implementation of the ``decoder_ffn`` structure.

Composes the fused W4A16 gate/up -> activation -> down block from the
``flashrt/weight-only-ffn`` Hub kernel behind the structure boundary —
the 4-bit twin of ``w8a16_static``, with the same decode-band envelope
and half the weight bytes. Weights are packed to NVFP4 (E2M1 data plus
per-16-element-block scale factors) at bind time; activations stay
BF16, so binding needs no calibration data, and qualification still
runs the parity gate on real host activations like every other
implementation.

The kernel's auto dispatch is qualified more narrowly than the INT8
twin's, and this impl mirrors that table exactly rather than stretching
it: M in [1, 3], with a per-M minimum on total weight elements (the
kernel refuses below it — ``weight-only-ffn`` ``torch_binding.cpp``,
the W4 branch). Calls outside the band are dispatched to the retained
host module by declared plan, counted in the ledger.
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

_ENTRYPOINTS = {"gelu": "w4a16_geglu_ffn_bf16", "silu": "w4a16_swiglu_ffn_bf16"}

SUPPORT = {
    "D": {"min": 512, "max": 16384, "multiple_of": 64},
    "F": {"min": 1024, "max": 16384, "multiple_of": 64},
    "M": {"min": 1, "max": 3},
    "m_classes": ("micro",),
}

#: the kernel's own auto-dispatch qualification: per M, the minimum
#: total weight elements (gate+up+down) it accepts. Copied from the W4
#: branch of the package's torch_binding.cpp — the kernel raises below
#: these, so the band dispatch must agree with them, not rediscover
#: them as runtime errors.
_AUTO_FLOOR = {1: 12 << 20, 2: 32 << 20, 3: 64 << 20}


def _in_band(m: int, weight_elements: int) -> bool:
    floor = _AUTO_FLOOR.get(m)
    return floor is not None and weight_elements >= floor


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


class BoundDecoderFfnW4A16:
    """MLP-seam callable: normed activations in, FFN output out (BF16)."""

    def __init__(self, ffn_fn, gate_up_q, gate_up_sfb, down_q, down_sfb,
                 dim_d, weight_elements):
        self._ffn = ffn_fn
        self._gate_up_q = gate_up_q
        self._gate_up_sfb = gate_up_sfb
        self._down_q = down_q
        self._down_sfb = down_sfb
        self._dim_d = dim_d
        self._weight_elements = weight_elements

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        shape = normed.shape
        x = normed.reshape(-1, shape[-1])
        m = x.shape[0]
        if not _in_band(m, self._weight_elements):
            raise ValueError(
                f"M={m} outside the W4A16 auto-dispatch qualification "
                f"(M in [1, 3], weight elements >= "
                f"{_AUTO_FLOOR.get(min(m, 3), 0)} at this M; "
                f"have {self._weight_elements})")
        out = self._ffn(x.to(torch.bfloat16).contiguous(),
                        self._gate_up_q, self._gate_up_sfb,
                        self._down_q, self._down_sfb, variant=0)
        return out.reshape(shape).to(normed.dtype)

    __call__ = ffn


class FusedGluMlpW4A16(GuardedSeam, torch.nn.Module):
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

    def __init__(self, bound: BoundDecoderFfnW4A16,
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
        if not _in_band(m, self._bound._weight_elements):
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
    """Bind the MLP-seam slice of ``decoder_ffn`` with weight-only NVFP4.

    ``weights`` uses checkpoint-native ``[out, in]`` projection layout
    (``w_gate``/``w_up``: ``[F, D]``, ``w_down``: ``[D, F]``). No
    calibration data is required: quantization is per-16-element-block
    on weights only (``quantize_w4_weight_bf16`` returns the packed
    E2M1 data and the SFB scale tensor the FFN entry points consume).
    """
    dim_d, dim_f = _check(weights)
    weight_elements = 3 * dim_d * dim_f
    if weight_elements < _AUTO_FLOOR[1]:
        raise ValueError(
            f"refused: {weight_elements} weight elements is below the "
            f"kernel's auto-dispatch floor ({_AUTO_FLOOR[1]}) even at "
            f"M=1; the W4A16 path cannot serve this seam at any M")
    k = _kernel()
    ffn_fn = _entrypoint(variant)
    gate_up = torch.cat(
        [weights["w_gate"].to("cuda", torch.bfloat16),
         weights["w_up"].to("cuda", torch.bfloat16)], dim=0).contiguous()
    down = weights["w_down"].to("cuda", torch.bfloat16).contiguous()
    gate_up_q, gate_up_sfb = k.quantize_w4_weight_bf16(gate_up)
    down_q, down_sfb = k.quantize_w4_weight_bf16(down)
    bound = BoundDecoderFfnW4A16(
        ffn_fn, gate_up_q, gate_up_sfb, down_q, down_sfb, dim_d,
        weight_elements)
    # bind-time smoke: one M=1 launch through the real entry point before
    # the seam is handed out. A stale build or missing symbol must
    # surface here as a clean bind refusal, not later inside the host's
    # forward — identical output cannot catch it there, because the
    # fallback path is numerically exact.
    probe = bound.ffn(torch.zeros(1, dim_d, device=gate_up_q.device,
                                  dtype=torch.bfloat16))
    if probe.shape != (1, dim_d) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: w4a16 bind smoke produced shape "
            f"{tuple(probe.shape)}, finite={bool(torch.isfinite(probe).all())}")
    return FusedGluMlpW4A16(bound, original=original)
