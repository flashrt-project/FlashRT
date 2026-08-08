"""Shared quantisation algorithms.

One implementation per algorithm, called by every consumer — native
frontends, structure impls, and framework adapters. The first resident
is the activation-aware per-input-channel balance used by the
production NVFP4 paths: it moves quantisation difficulty from
activations into weights (or back) without changing the mathematics,
and the FP4 quantiser then sees a better-conditioned tensor.

This module is pure tensor math. It owns no bytes: scale-factor
layouts, packing, kernel choice and the fusion of the activation-side
multiply live with the executor that consumes the result.
"""

from __future__ import annotations

import torch

__all__ = ["fit_input_channel_balance"]


def fit_input_channel_balance(
    weight: torch.Tensor,
    activation_amax: torch.Tensor | None = None,
    *,
    alpha: float = 0.5,
    clamp: tuple[float, float] = (0.25, 4.0),
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-input-channel balance ahead of a low-bit weight quantiser.

    With per-channel activation amax ``a`` over the input (K) axis::

        s[k] = clamp((a[k] / mean(a)) ** alpha, *clamp)
        W'   = W * s        (broadcast along the output axis)
        x'   = x * (1 / s)

    so ``x' @ W'.T == x @ W.T`` exactly, and only then are ``W'`` and
    ``x'`` quantised. Without ``activation_amax`` the balance falls back
    to the weight's own per-channel amax.

    ``weight`` is ``[N, K]`` (checkpoint-native ``[out, in]``);
    ``activation_amax`` is ``[K]``. Returns ``(scaled_weight,
    inv_scale)`` in ``out_dtype`` (the weight's dtype by default);
    ``inv_scale`` is what the activation must be multiplied by before
    the quantised GEMM — fused into the producing kernel by the
    executor, never as a separate eager pass on a hot path.

    The default ``alpha``/``clamp``/``eps`` and the operation order are
    the production recipe validated on Pi0.5 (Thor FP4) and reused by
    the Motus video FFN — activation-only balance. The
    activation/weight SmoothQuant ratio is a different recipe and is
    deliberately not this function; recipes are selected by the caller,
    never by environment variables.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be [N, K], got {tuple(weight.shape)}")
    dim_k = weight.shape[1]
    if activation_amax is not None:
        if activation_amax.numel() != dim_k:
            raise ValueError(
                f"activation_amax has {activation_amax.numel()} channels, "
                f"weight K is {dim_k}")
        a = activation_amax.reshape(-1).float().clamp(min=eps)
    else:
        a = weight.abs().amax(dim=0).float().clamp(min=eps)
    lo, hi = clamp
    s = (a / a.mean()).pow(alpha).clamp(min=lo, max=hi)
    dt = out_dtype if out_dtype is not None else weight.dtype
    inv_scale = (1.0 / s).to(dt).contiguous()
    scaled_weight = (weight.float() * s.unsqueeze(0)).to(dt).contiguous()
    return scaled_weight, inv_scale
