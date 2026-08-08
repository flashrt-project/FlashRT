"""Ground-truth reference for the ``linear_proj`` structure."""

from __future__ import annotations

import torch


def linear_proj_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    *,
    variant: dict[str, str] | None = None,
) -> torch.Tensor:
    """Plain-torch reference: y = x @ w (+ b) (+ epilogue).

    ``w`` uses the declared [K, N] slot layout. Epilogue variants:
    ``none`` returns the projection; ``residual_add`` adds the residual
    input; ``gelu_quant_fp8`` applies tanh-GELU (the fp8 quantization of
    the epilogue output is an implementation detail of fused impls — at
    the declared boundary the reference stays in the compute dtype).
    """
    variant = variant or {}
    y = x.to(torch.float32) @ w.to(torch.float32)
    if b is not None and variant.get("bias", "add") != "none":
        y = y + b.to(torch.float32)
    epilogue = variant.get("epilogue", "none")
    if epilogue == "gelu_quant_fp8":
        y = torch.nn.functional.gelu(y, approximate="tanh")
    elif epilogue == "residual_add":
        if residual is None:
            raise ValueError("residual_add epilogue requires residual input")
        y = y + residual.to(torch.float32)
    return y.to(x.dtype)
