"""Ground-truth reference for the ``cadence_static`` structure."""

from __future__ import annotations

from typing import Callable

import torch


def cadence_static_ref(
    x: torch.Tensor,
    host_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    variant: dict[str, str] | None = None,
) -> torch.Tensor:
    """Plain-torch reference: simply run the host's own computation.

    Holding the result in a buffer is an execution decision. At the
    declared boundary this structure is the identity on the host's
    module, which is exactly why binding must first prove the output
    does not move within the hot loop.
    """
    del variant
    return host_fn(x)
