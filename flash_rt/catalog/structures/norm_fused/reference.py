"""Ground-truth reference for the ``norm_fused`` structure."""

from __future__ import annotations

import torch


def norm_fused_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-6,
    variant: dict[str, str] | None = None,
) -> torch.Tensor:
    """Plain-torch reference: the affine layer norm itself.

    Which dtype the kernel computes in is an execution decision; at the
    declared boundary this is the host's own norm, which is why binding
    refuses when the host is already at compute dtype.
    """
    del variant
    xf = x.to(torch.float32)
    normed = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(
        xf.var(-1, keepdim=True, unbiased=False) + eps)
    return (normed * w.to(torch.float32) + b.to(torch.float32)).to(x.dtype)
