"""Ground-truth reference for the ``decoder_block`` structure."""

from __future__ import annotations

from typing import Callable

import torch


def decoder_block_ref(
    x: torch.Tensor,
    cond: torch.Tensor | None,
    *,
    norm_in: Callable,
    attn: Callable,
    norm_out: Callable,
    ffn: Callable,
    variant: dict[str, str] | None = None,
) -> torch.Tensor:
    """Plain-torch pre-norm block: the dataflow, not the kernels.

    This structure's content is the order of operations between the
    sublayers, so the reference takes the sublayers as callables and
    declares only how their results combine. Each norm returns
    ``(normed, gate)``; a ``None`` gate is the ungated residual.
    """
    del variant

    def residual(r, y, gate):
        return r + y if gate is None else r + y * gate

    r = x
    h, gate = norm_in(x, cond)
    h = residual(r, attn(h), gate)

    r = h
    h, gate = norm_out(h, cond)
    return residual(r, ffn(h), gate)
