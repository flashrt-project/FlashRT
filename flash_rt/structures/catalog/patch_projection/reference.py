"""Reference for a processor-preflattened full-patch projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_projection_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project flattened patches; ``w`` uses checkpoint layout ``[N,K]``."""
    return F.linear(x, w, b)
