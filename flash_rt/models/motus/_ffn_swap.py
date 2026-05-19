"""G3d — Replace Wan/Action/Und FFN nn.Sequential with direct fvk chain.

Each Wan / Action / Und FFN is a 3-layer nn.Sequential:

    nn.Linear(dim, ffn_dim, bias=True)       # up_proj
    nn.GELU(approximate='tanh')              # GELU(tanh)
    nn.Linear(ffn_dim, dim, bias=True)       # down_proj

After G3b the two Linears already use fvk.bf16_nn (with a torch
bias add). After G3c the attention path uses vendored FA2. G3d
goes one step further on the FFN itself: replace the *entire*
Sequential.forward with a single function that issues FIVE fvk ops
back-to-back, removing:

    * nn.Sequential's per-child Python-side dispatch + hooks.
    * The G3b _make_linear_forward closure indirection (we pull
      weight ptrs out of the Linears at install time and bind them
      directly into the FFN closure).
    * torch.add_ overhead for bias (replaced with fvk.add_bias_bf16).

Per-FFN op chain after G3d:

    1. fvk.bf16_nn(x, w_up_t, up_out, M, ffn_dim, dim)        # GEMM
    2. fvk.add_bias_bf16(up_out, b_up, M, ffn_dim)            # bias
    3. fvk.gelu_inplace(up_out, M*ffn_dim)                    # GELU(tanh)
    4. fvk.bf16_nn(up_out, w_down_t, down_out, M, dim, ffn_dim) # GEMM
    5. fvk.add_bias_bf16(down_out, b_down, M, dim)            # bias

Same number of launches as G3b+SDPA-GELU but every launch is a
direct fvk pybind crossing — no torch nn.Module dispatch, no
intermediate tensor metadata churn.

GELU activation match: fvk.gelu_inplace implements
``x / (1 + exp(-1.5957691216 * x * (1 + 0.044715 * x^2)))``
which is mathematically identical to nn.GELU(approximate='tanh')
``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``
via the identity 0.5*(1+tanh(z)) = 1/(1+exp(-2z)) with
2*sqrt(2/pi) ~ 1.5957691216. Cos drift expected at ULP level only.

Scope filter: same hot-path prefixes as _linear_swap.py. Only
nn.Sequential modules whose children match
[nn.Linear, nn.GELU, nn.Linear] are eligible. VLM and VAE FFNs
(Qwen3-VL has its own FFN structure, VAE uses Conv-based blocks)
are skipped automatically by the structural check.
"""

from __future__ import annotations

import logging

import torch

from flash_rt.models.motus._stream import cs
import torch.nn as nn

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)


# Same hot-path prefixes as _linear_swap.py. Imported lazily to avoid
# a forward dependency cycle.
def _hot_path_prefixes():
    from flash_rt.models.motus._linear_swap import (
        _HOT_PATH_PREFIXES, _SKIP_PREFIXES,
    )
    return _HOT_PATH_PREFIXES, _SKIP_PREFIXES


# ──────────────────────────────────────────────────────────────────
# FFN forward factory
# ──────────────────────────────────────────────────────────────────

def _make_ffn_forward(
    up_lin: nn.Linear,
    gelu_mod: nn.GELU,
    down_lin: nn.Linear,
    gemm: fvk.GemmRunner,
    label: str = 'ffn',
):
    """Return a closure that bypasses the upstream nn.Sequential.

    NOTE: G3b already swapped each Linear's storage to its transposed
    [K, N] layout in-place, so ``up_lin.weight`` is shape [dim, ffn_dim]
    and ``down_lin.weight`` is shape [ffn_dim, dim] — exactly the NN
    layout fvk.bf16_nn wants.
    """
    # Up projection: dim -> ffn_dim
    up_w = up_lin.weight                   # [dim, ffn_dim] post-G3b transpose
    up_b = up_lin.bias                     # [ffn_dim] or None
    K_up = int(up_w.shape[0])              # dim
    N_up = int(up_w.shape[1])              # ffn_dim
    up_w_ptr = int(up_w.data_ptr())
    up_b_ptr = int(up_b.data_ptr()) if up_b is not None else 0

    # Down projection: ffn_dim -> dim
    down_w = down_lin.weight               # [ffn_dim, dim] post-G3b transpose
    down_b = down_lin.bias                 # [dim] or None
    K_down = int(down_w.shape[0])          # ffn_dim
    N_down = int(down_w.shape[1])          # dim
    down_w_ptr = int(down_w.data_ptr())
    down_b_ptr = int(down_b.data_ptr()) if down_b is not None else 0

    # Sanity: chained dims must match.
    assert N_up == K_down, (
        f"FFN dim mismatch at {label}: up.N={N_up} vs down.K={K_down}")

    # Sanity: bias dtype/device.
    if up_b is not None:
        assert up_b.is_cuda and up_b.dtype == torch.bfloat16, \
            f"{label}.up_proj bias wrong dtype/device"
    if down_b is not None:
        assert down_b.is_cuda and down_b.dtype == torch.bfloat16, \
            f"{label}.down_proj bias wrong dtype/device"

    # Pin lifetimes via closure.
    _refs = (up_w, up_b, down_w, down_b)

    def forward(x: torch.Tensor) -> torch.Tensor:
        _ = _refs
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K_up)
        M = flat.shape[0]
        device = flat.device

        # 1) up_proj GEMM:  up_out[M, ffn_dim] = flat[M, dim] @ w_up_t[dim, ffn_dim]
        up_out = torch.empty(M, N_up, dtype=torch.bfloat16, device=device)
        gemm.bf16_nn(
            int(flat.data_ptr()), up_w_ptr, int(up_out.data_ptr()),
            M, N_up, K_up, cs(),
        )

        # 2-3) G7.11: fused (bias + GELU(tanh)) in-place on up_out.
        # Replaces add_bias_bf16 + gelu_inplace pair (2 launches -> 1).
        if up_b_ptr:
            fvk.bias_gelu_inplace_bf16(
                int(up_out.data_ptr()), up_b_ptr, M, N_up, cs())
        else:
            fvk.gelu_inplace(int(up_out.data_ptr()), M * N_up, cs())

        # 4) down_proj GEMM:  down_out[M, dim] = up_out[M, ffn_dim] @ w_down_t[ffn_dim, dim]
        down_out = torch.empty(M, N_down, dtype=torch.bfloat16, device=device)
        gemm.bf16_nn(
            int(up_out.data_ptr()), down_w_ptr, int(down_out.data_ptr()),
            M, N_down, K_down, cs(),
        )

        # 5) bias add (in-place on down_out)
        if down_b_ptr:
            fvk.add_bias_bf16(
                int(down_out.data_ptr()), down_b_ptr, M, N_down, cs())

        if in_dtype != torch.bfloat16:
            down_out = down_out.to(in_dtype)
        return down_out.view(*in_shape[:-1], N_down)

    return forward


# ──────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────

def install_fvk_ffns(model, gemm: fvk.GemmRunner | None = None) -> dict:
    """Walk ``model`` and replace nn.Sequential FFNs (Linear+GELU+Linear)
    under hot-path scopes with a single fvk-direct forward.

    Must run AFTER install_fvk_linears (G3b) so Linear weights are
    already in [K, N] layout. The G3b per-Linear monkey-patches are
    *bypassed* by this swap (we don't go through the Linear's forward
    anymore — we read the weight ptrs directly), which removes one
    layer of Python indirection per FFN call.

    Returns counts dict.
    """
    if gemm is None:
        gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()
        model._g3b_gemm = gemm  # share with G3b

    HOT, SKIP = _hot_path_prefixes()
    counts = {'ffn_replaced': 0, 'ffn_skipped_struct': 0,
              'ffn_skipped_scope': 0}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue

        in_hot = any(name.startswith(p) for p in HOT)
        in_skip = any(name.startswith(p) for p in SKIP)
        if (not in_hot) or in_skip:
            continue

        # Structural match: exactly [Linear, GELU, Linear].
        children = list(module)
        if len(children) != 3:
            continue
        if not isinstance(children[0], nn.Linear):
            continue
        if not isinstance(children[1], nn.GELU):
            continue
        if not isinstance(children[2], nn.Linear):
            continue

        # FFN-end suffix sanity (skip e.g. action_encoder which is
        # [Linear, SiLU, Linear, SiLU, Linear] — already filtered by
        # len above, but be explicit). Time embeddings are
        # [Linear, SiLU, Linear] — same length, but their first
        # element is a SiLU not GELU. Already filtered by isinstance.
        # If shapes look wrong, skip.
        try:
            module.forward = _make_ffn_forward(
                children[0], children[1], children[2], gemm, label=name)
            counts['ffn_replaced'] += 1
        except AssertionError as e:
            logger.warning(f"[g3d] skip {name}: {e}")
            counts['ffn_skipped_struct'] += 1

    logger.info(
        f"[g3d] FFN swap: replaced={counts['ffn_replaced']}, "
        f"struct-skipped={counts['ffn_skipped_struct']}")
    return counts
