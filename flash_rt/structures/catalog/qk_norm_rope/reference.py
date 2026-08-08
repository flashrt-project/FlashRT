"""Ground-truth reference for the ``qk_norm_rope`` structure."""

from __future__ import annotations

import torch


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_mode: str,
    normalization_scope: str,
) -> torch.Tensor:
    xf = x.to(torch.float32)
    if normalization_scope == "per_head":
        reduction = (-1,)
        expected = x.shape[-1]
        weight_shape = (1, 1, 1, x.shape[-1])
    elif normalization_scope == "projection":
        reduction = (-2, -1)
        expected = x.shape[-2] * x.shape[-1]
        weight_shape = (1, 1, x.shape[-2], x.shape[-1])
    else:
        raise ValueError(
            f"unknown normalization scope: {normalization_scope!r}")
    if weight.numel() != expected:
        raise ValueError(
            f"{normalization_scope} norm needs {expected} weight elements, "
            f"got {weight.numel()}")
    normed = xf * torch.rsqrt(
        xf.square().mean(dim=reduction, keepdim=True) + eps)
    wf = weight.to(torch.float32)
    if weight_mode == "offset":
        wf = wf + 1.0
    return normed * wf.reshape(weight_shape)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    xr = x[..., :rotary_dim]
    tail = x[..., rotary_dim:]
    cosf = cos.to(torch.float32).unsqueeze(-2)
    sinf = sin.to(torch.float32).unsqueeze(-2)

    if layout == "half":
        half = rotary_dim // 2
        rotated = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
        cosf = torch.cat((cosf, cosf), dim=-1)
        sinf = torch.cat((sinf, sinf), dim=-1)
    elif layout == "interleaved":
        rotated = torch.stack(
            (-xr[..., 1::2], xr[..., 0::2]),
            dim=-1,
        ).flatten(-2)
        cosf = cosf.repeat_interleave(2, dim=-1)
        sinf = sinf.repeat_interleave(2, dim=-1)
    else:
        raise ValueError(f"unknown rope layout: {layout!r}")

    out = xr * cosf + rotated * sinf
    return torch.cat((out, tail), dim=-1) if tail.shape[-1] else out


def qk_norm_rope_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float = 1e-6,
    rotary_dim: int | None = None,
    variant: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-head Q/K RMSNorm and a pre-expanded RoPE table.

    Q, K and V use ``(B, T, H, D)`` layout. The position geometry is
    deliberately outside this boundary: 1D, MRoPE and decomposed 3D hosts
    all hand over the same ``cos``/``sin`` table after expanding it to
    ``(B, T, R/2)`` (or the broadcastable ``(T, R/2)`` form).
    """
    variant = variant or {}
    layout = variant.get("rope_layout", "half")
    weight_mode = variant.get("norm_weight_mode", "direct")
    normalization_scope = variant.get("normalization_scope", "per_head")
    rotary_dim = q.shape[-1] if rotary_dim is None else int(rotary_dim)

    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q, k and v must share batch and token dimensions")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share head_dim")
    if rotary_dim <= 0 or rotary_dim > q.shape[-1] or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    expected_freq = rotary_dim // 2
    if cos.shape != sin.shape or cos.shape[-1] != expected_freq:
        raise ValueError("cos and sin must end in rotary_dim / 2")

    qn = _rms_norm(
        q, q_norm_weight, eps, weight_mode, normalization_scope)
    kn = _rms_norm(
        k, k_norm_weight, eps, weight_mode, normalization_scope)
    q_out = _apply_rope(qn, cos, sin, rotary_dim, layout).to(q.dtype)
    k_out = _apply_rope(kn, cos, sin, rotary_dim, layout).to(k.dtype)
    return q_out, k_out, v
