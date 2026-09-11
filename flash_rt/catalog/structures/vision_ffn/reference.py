"""Reference implementation for the ``vision_ffn`` structure.

Ground truth for the qualification gates: the plainest possible PyTorch,
never executed on a serving hot path. LayerNorm statistics follow the
boundary dtype's promotion rules via float32, matching the vision-tower
convention of the model families this structure binds to.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vision_ffn_ref(
    x: torch.Tensor,
    w_norm: torch.Tensor | None,
    b_norm: torch.Tensor | None,
    w_fc1: torch.Tensor,
    b_fc1: torch.Tensor,
    w_fc2: torch.Tensor,
    b_fc2: torch.Tensor,
    *,
    activation: str = "gelu",
    eps: float = 1e-6,
) -> torch.Tensor:
    """x -> LayerNorm -> fc1 -> GELU(tanh) -> fc2 -> + x."""
    if activation != "gelu":
        raise ValueError(f"unsupported activation: {activation!r}")
    h = F.layer_norm(x.float(), (x.shape[-1],),
                     (w_norm.float() if w_norm is not None else None),
                     (b_norm.float() if b_norm is not None else None),
                     eps).to(x.dtype)
    hidden = F.gelu(h @ w_fc1.t() + b_fc1, approximate="tanh")
    return x + (hidden @ w_fc2.t() + b_fc2)
