"""norm_fused — a plain norm run by a fused kernel at compute dtype.

Vision towers commonly keep their LayerNorms in FP32 while the rest of
the block runs in BF16, so every norm pays a dtype round trip plus an
unfused mean/variance pass. This replacement runs the norm in one fused
BF16 kernel and hands the result back in the host's dtype.

Qualification is the host's own dtype: the win comes from collapsing an
FP32 norm into a fused BF16 one, so a norm the host already runs in
BF16 is left alone (there is nothing to collapse), and the parity gate
adjudicates the numerical difference the dtype change introduces.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-residual-norm-quant",
    "version": ">=1",
}


class FusedNorm(GuardedSeam, torch.nn.Module):
    """Drop-in for an affine LayerNorm, computed by a fused kernel."""

    _frt_host_attr = "host_norm"
    _frt_can_fallback = True

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        self.host_norm = original
        ks = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._fn = ks.layer_norm_bf16
        self.register_buffer("w", original.weight.detach().to(
            torch.bfloat16))
        self.register_buffer("b", original.bias.detach().to(
            torch.bfloat16))
        self.eps = float(getattr(original, "eps", 1e-6))
        self._frt_arm(dtypes=CAST_OK, device=self.w.device,
                      k=int(self.w.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        # the kernel's contract is 2D [rows, width]; hosts hand the norm
        # whatever leading shape their block carries
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        y = self._fn(flat, self.w, self.b, self.eps)
        return y.reshape(shape).to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_norm"), name)


def bind_norm_fused(original: torch.nn.Module,
                    host_dtypes=None) -> FusedNorm:
    """Bind a fused norm, refusing where there is nothing to collapse.

    ``host_dtypes`` is the set of input dtypes this norm was observed with
    during calibration — one observation, not a statistic. A host already
    running the norm at a compute dtype has nothing for this structure to
    collapse, and the refusal names the dtype so it reads as "not in this
    form" rather than "not supported".
    """
    if getattr(original, "weight", None) is None or \
            getattr(original, "bias", None) is None:
        raise ValueError("norm_fused: needs an affine norm (weight+bias)")
    if host_dtypes and torch.float32 not in set(host_dtypes):
        raise ValueError(
            "norm_fused: host already runs this norm at compute "
            f"dtype ({sorted(str(d) for d in host_dtypes)}) — nothing "
            "to collapse")
    bound = FusedNorm(original)
    # bind-time smoke through the real entry point, at a 3D host shape:
    # a stale build, a missing symbol, or a kernel whose rank contract
    # moved must surface here as a clean bind refusal, not mid-forward
    probe_in = torch.zeros(1, 2, bound.w.shape[0], device=bound.w.device)
    probe = bound(probe_in)
    if probe.shape != probe_in.shape or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: norm_fused bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound
