"""Ground-truth reference for the ``qkv_pack`` structure."""

from __future__ import annotations

from typing import Sequence

import torch


def qkv_pack_ref(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor] | None = None,
    *,
    variant: dict[str, str] | None = None,
) -> list[torch.Tensor]:
    """Plain-torch reference: the sibling projections, computed apart.

    Packing is an execution decision — at the declared boundary the
    structure is exactly the group of independent projections, in
    sibling order. ``weights`` use the declared [K, N_i] slot layout.
    The module bind form is judged at the same boundary: its attention
    and output projection belong to the host, and only the packed
    projections are this structure's contract.
    """
    del variant
    outs = []
    for i, w in enumerate(weights):
        y = x.to(torch.float32) @ w.to(torch.float32)
        if biases is not None and biases[i] is not None:
            y = y + biases[i].to(torch.float32)
        outs.append(y.to(x.dtype))
    return outs
