"""Plain PyTorch reference for the modulated-norm to QKV chain."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def modnorm_qkv_chain_ref(
    x: torch.Tensor,
    cond: torch.Tensor,
    w_cond: torch.Tensor,
    b_cond: torch.Tensor,
    w_q: torch.Tensor,
    b_q: torch.Tensor,
    w_k: torch.Tensor | None = None,
    b_k: torch.Tensor | None = None,
    w_v: torch.Tensor | None = None,
    b_v: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    fanout: str = "qkv",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm/modulation followed by the declared Q-only or QKV fanout."""
    scale, shift = F.linear(F.silu(cond), w_cond, b_cond).chunk(2, dim=-1)
    normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    normed = normed * (1 + scale[:, None]) + shift[:, None]
    query = F.linear(normed, w_q, b_q)
    if fanout == "q_only":
        return query
    if fanout != "qkv":
        raise ValueError(f"unsupported fanout: {fanout!r}")
    if w_k is None or w_v is None:
        raise ValueError("qkv fanout requires K and V weights")
    return (
        query,
        F.linear(normed, w_k, b_k),
        F.linear(normed, w_v, b_v),
    )
