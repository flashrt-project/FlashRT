"""BF16 lowering for processor-preflattened full-patch Conv3D modules."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-gemm-epilogues",
    "version": ">=1",
}


def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


class FlatPatchProjection(GuardedSeam, torch.nn.Module):
    """Drop-in replacement for an exact full-patch Conv3D wrapper.

    The host owns checkpoint-layout Conv3D weights ``[N,C,T,P,P]``. Binding
    flattens them to the Hub API's ``[K,N]`` GEMM layout exactly once. The
    retained host module remains the fallback and state-dict owner.
    """

    _frt_host_attr = "host_patch"
    _frt_can_fallback = True

    def __init__(
        self,
        weight_kn: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        row_capacity: int,
        host_dtypes: Sequence[torch.dtype],
        original: torch.nn.Module,
        kernel=None,
    ) -> None:
        super().__init__()
        self._weight_kn = weight_kn
        self._bias = bias
        self._row_capacity = int(row_capacity)
        self.host_patch = original
        self._ops = _kernel() if kernel is None else kernel
        entry = (
            "bf16_linear_bias_bf16"
            if bias is not None
            else "bf16_linear_bf16"
        )
        try:
            self._fn = getattr(self._ops, entry)
        except AttributeError as exc:
            raise ValueError(
                f"patch_projection Hub artifact lacks {entry}"
            ) from exc
        self._out = torch.empty(
            self._row_capacity,
            weight_kn.shape[1],
            device=weight_kn.device,
            dtype=torch.bfloat16,
        )
        self._frt_arm(
            dtypes=tuple(host_dtypes) or CAST_OK,
            device=weight_kn.device,
            k=int(weight_kn.shape[0]),
            row_capacity=self._row_capacity,
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_patch":
                raise
            return getattr(super().__getattr__("host_patch"), name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        flat = x.reshape(-1, self._weight_kn.shape[0])
        rows = flat.shape[0]
        out = self._out[:rows]
        flat = flat.to(torch.bfloat16).contiguous()
        if self._bias is None:
            return self._fn(flat, self._weight_kn, out=out)
        return self._fn(flat, self._weight_kn, self._bias, out=out)


@torch.no_grad()
def bind_flat_patch_projection(
    weights: Mapping[str, torch.Tensor],
    *,
    row_profile: Sequence[int],
    host_dtypes: Sequence[torch.dtype],
    original: torch.nn.Module,
) -> FlatPatchProjection:
    """Bind a full-patch projection from checkpoint weights ``w[N,K]``."""
    if not row_profile:
        raise ValueError("patch_projection: no real patch rows were observed")
    w = weights["w"]
    if w.dim() != 2 or w.dtype is not torch.bfloat16 or not w.is_cuda:
        raise ValueError(
            "patch_projection requires CUDA BF16 checkpoint weights [N,K]"
        )
    b = weights.get("b")
    if b is not None:
        if b.shape != (w.shape[0],):
            raise ValueError("patch_projection bias width does not match N")
        b = b.detach().to(torch.bfloat16).contiguous()
    weight_kn = w.detach().t().contiguous()
    capacity = max(int(row) for row in row_profile)
    bound = FlatPatchProjection(
        weight_kn,
        b,
        row_capacity=capacity,
        host_dtypes=host_dtypes,
        original=original,
    )
    # A fallback-capable seam must prove the formal artifact launches at bind
    # time; otherwise a stale package would look numerically perfect by
    # silently running the retained host.
    sample_dtype = next(iter(host_dtypes), torch.bfloat16)
    sample = torch.zeros(
        capacity,
        w.shape[1],
        device=w.device,
        dtype=sample_dtype,
    )
    bound(sample)
    if bound._frt_guard is not None:
        bound._frt_guard.calls = 0
    return bound
