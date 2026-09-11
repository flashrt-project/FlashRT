"""Ground-truth reference for the ``attention_core`` structure."""

from __future__ import annotations

import torch


def attention_core_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    variant: dict[str, str] | None = None,
) -> torch.Tensor:
    """Plain-torch reference: masked softmax attention, host layout in,
    (B, Sq, H, D) out.

    Packing and kernel choice are execution decisions; at the declared
    boundary this is ordinary attention against the host's own mask,
    which is what the packing must reproduce.
    """
    del variant
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    heads, kv_heads = q.shape[1], k.shape[1]
    if kv_heads != heads:
        k = k.expand(-1, heads, -1, -1) if kv_heads == 1 else \
            k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.expand(-1, heads, -1, -1) if kv_heads == 1 else \
            v.repeat_interleave(heads // kv_heads, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask.float()
    out = torch.softmax(scores, dim=-1) @ v.float()
    return out.transpose(1, 2).to(q.dtype)
