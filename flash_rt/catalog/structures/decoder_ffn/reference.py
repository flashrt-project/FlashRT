"""Reference implementation for the ``decoder_ffn`` structure.

Ground truth for the qualification gates: the plainest possible PyTorch,
never executed on a serving hot path. Norm statistics are computed in
float32 and cast back to the boundary dtype, matching the numerical
convention of the model families this structure binds to.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ACTIVATIONS = {
    "gelu": lambda t: F.gelu(t, approximate="tanh"),
    "silu": F.silu,
}


def decoder_ffn_ref(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    activation: str = "gelu",
    norm_weight_mode: str = "offset",
    cond_scale: torch.Tensor | None = None,
    cond_shift: torch.Tensor | None = None,
    cond_gate: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """x -> RMSNorm -> gate/up GEMM -> act(gate) * up -> down GEMM -> + x.

    ``norm_weight_mode``: "offset" multiplies by ``1 + w_norm`` (Gemma
    convention), "direct" multiplies by ``w_norm`` (Qwen convention).
    ``cond_scale``/``cond_shift``/``cond_gate`` apply AdaLN-style
    modulation under the ``ada_ln`` conditioning variant: the norm output
    becomes ``normed * (1 + scale) + shift`` and the residual becomes
    ``x + ffn_out * gate``. A binding whose AdaLN branch has no learned
    norm weight maps ``w_norm`` to zeros under "offset" mode.
    """
    act = _ACTIVATIONS[activation]

    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    if norm_weight_mode == "offset":
        h = h * (1.0 + w_norm.float())
    elif norm_weight_mode == "direct":
        h = h * w_norm.float()
    else:
        raise ValueError(f"unknown norm_weight_mode: {norm_weight_mode!r}")
    if cond_scale is not None:
        h = h * (1.0 + cond_scale.float())
    if cond_shift is not None:
        h = h + cond_shift.float()
    h = h.to(x.dtype)

    hidden = act(h @ w_gate) * (h @ w_up)
    out = hidden @ w_down
    if cond_gate is not None:
        out = out * cond_gate
    return x + out
