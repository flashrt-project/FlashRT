"""Ground-truth reference for the ``adaln_producer`` structure."""

from __future__ import annotations

import torch


def adaln_producer_ref(
    x: torch.Tensor,
    cond: torch.Tensor,
    style_w: torch.Tensor,
    style_b: torch.Tensor | None = None,
    norm_w: torch.Tensor | None = None,
    *,
    variant: dict[str, str] | None = None,
    eps: float = 1e-6,
):
    """Plain-torch reference: norm(x) * (1 + scale) + shift, and gate.

    The style projection is computed here rather than looked up — the
    step table is an execution decision, and at the declared boundary
    the structure is the conditioning projection plus the modulated
    norm. ``style_w`` uses the declared [C, 3*D] slot layout, its three
    chunks being scale, shift and gate.
    """
    variant = variant or {}
    style = cond.to(torch.float32) @ style_w.to(torch.float32)
    if style_b is not None:
        style = style + style_b.to(torch.float32)
    scale, shift, gate = style.chunk(3, dim=-1)
    xf = x.to(torch.float32)
    if variant.get("norm", "rms") == "rms":
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    else:
        normed = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(
            xf.var(-1, keepdim=True, unbiased=False) + eps)
    if norm_w is not None:
        normed = normed * norm_w.to(torch.float32)
    y = normed * (1.0 + scale) + shift
    return y.to(x.dtype), gate.to(x.dtype)
