"""G3a — fvk BF16 norm substitution for Motus.

Walks the loaded Motus ``nn.Module`` tree, replaces the ``forward``
method of every ``WanLayerNorm`` and ``WanRMSNorm`` instance with a
fvk-backed equivalent. Pure data path; no source changes.

Why monkey-patch instead of subclass replacement?
    * The model has hundreds of WanLayerNorm / WanRMSNorm instances
      scattered across video / action / und experts (norm1/norm2/norm3
      in 30 DiT layers × 3 experts; q_norm/k_norm in self-attn; plus
      adapter and timestep paths). Replacing each via subclass would
      need to walk the parent and reassign attributes — invasive.
    * Patching ``instance.forward`` only touches the call path. The
      param tensor (``self.weight`` for RMSNorm; nothing for LayerNorm
      no-affine) stays bound to the original Module so the state_dict
      is still loadable.

Numerical contract:
    * Both upstream ``WanLayerNorm.forward`` and ``WanRMSNorm.forward``
      do ``x.float() -> compute -> .type_as(x)`` (compute in fp32, return
      in input dtype).
    * The fvk bf16 kernels also do internal fp32 reduction. Output is
      bf16. So per-tensor output should match within fp16/bf16 ULP
      (~1-2 LSBs), which compounds to cos drop of 1.0 -> ~0.999 over
      30 layers × 10 steps.

Targeted G3a cos floor: ≥ 0.9990, std < 0.0005 (per
MOTUS_EXECUTION_PLAN.md §G3.5).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from flash_rt.models.motus._stream import cs

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)


# Kept compatible with the upstream wan/modules/model.py classes;
# imported lazily to avoid loading wan at module import time.
def _get_wan_norm_classes() -> Tuple[type, type]:
    from wan.modules.model import WanRMSNorm, WanLayerNorm  # type: ignore
    return WanRMSNorm, WanLayerNorm


# ──────────────────────────────────────────────────────────────────
# fvk-backed forward implementations
# ──────────────────────────────────────────────────────────────────

# Set to True via env to enable per-call NaN tracing (very slow).
import os as _os
_TRACE = _os.environ.get('FLASH_RT_MOTUS_NORM_TRACE', '0') == '1'


def _trace_in_out(label: str, x_in, out, weight=None, bias=None):
    """Cheap NaN/Inf gate; only emits when actual problem."""
    out_bad = torch.isnan(out).any() or torch.isinf(out).any()
    in_bad = torch.isnan(x_in).any() or torch.isinf(x_in).any()
    if out_bad or in_bad:
        msg = (f"[NORM_NAN] {label}: in_dtype={x_in.dtype} "
               f"in_nan={torch.isnan(x_in).any().item()} "
               f"in_inf={torch.isinf(x_in).any().item()} "
               f"out_nan={torch.isnan(out).any().item()} "
               f"out_inf={torch.isinf(out).any().item()} "
               f"in_shape={tuple(x_in.shape)}")
        if weight is not None:
            msg += (f" w_nan={torch.isnan(weight).any().item()}"
                    f" w_max={weight.abs().max().item():.4f}")
        logger.error(msg)


def _make_rms_norm_forward(
    weight: torch.Tensor, eps: float, dim: int, label: str = "rms",
):
    """Return a forward(x) that applies fvk.rms_norm in BF16.

    IMPORTANT: ``weight`` MUST already be a contiguous tensor that lives
    for the lifetime of the closure. Calling ``.contiguous().data_ptr()``
    inside this factory is a footgun — the temporary tensor is freed
    once we return, leaving a dangling pointer (cos -> NaN). We hold
    a strong reference to the contiguous weight in the closure cell.
    """
    # Materialize contiguous storage and pin the reference.
    w_pinned = weight if weight.is_contiguous() else weight.contiguous()
    w_ptr = int(w_pinned.data_ptr())

    def forward(x: torch.Tensor) -> torch.Tensor:
        _ = w_pinned  # keep weight alive in closure
        # Dtype guard: fvk.rms_norm interprets bytes as bf16 — wrong dtype
        # silently corrupts (NaN). Cast back to bf16 if upstream gave fp32.
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        flat = x_c.reshape(-1, dim)
        seq_len = flat.shape[0]
        out = torch.empty_like(flat)
        fvk.rms_norm(
            int(flat.data_ptr()), w_ptr, int(out.data_ptr()),
            int(seq_len), int(dim), float(eps), cs(),
        )
        if _TRACE:
            _trace_in_out(label, flat, out, weight=w_pinned)
        return out.view_as(x_c)

    return forward


def _make_layer_norm_no_affine_forward(
    eps: float, dim: int, label: str = "ln_noaff",
):
    """Return a forward(x) that applies fvk.layer_norm_no_affine_bf16."""
    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        flat = x_c.reshape(-1, dim)
        seq_len = flat.shape[0]
        out = torch.empty_like(flat)
        fvk.layer_norm_no_affine_bf16(
            int(flat.data_ptr()), int(out.data_ptr()),
            int(seq_len), int(dim), float(eps), cs(),
        )
        if _TRACE:
            _trace_in_out(label, flat, out)
        return out.view_as(x_c)

    return forward


def _make_layer_norm_affine_forward(
    weight: torch.Tensor, bias: torch.Tensor, eps: float, dim: int,
    label: str = "ln_aff",
):
    """Return a forward(x) that applies fvk.layer_norm (BF16 with affine).

    Used for Wan ``norm3`` (cross-attn input norm) which is constructed
    as ``WanLayerNorm(dim, eps, elementwise_affine=True)`` — 30 such
    instances total (one per DiT block).
    """
    w_pinned = weight if weight.is_contiguous() else weight.contiguous()
    b_pinned = bias if bias.is_contiguous() else bias.contiguous()
    w_ptr = int(w_pinned.data_ptr())
    b_ptr = int(b_pinned.data_ptr())

    def forward(x: torch.Tensor) -> torch.Tensor:
        _ = (w_pinned, b_pinned)
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        flat = x_c.reshape(-1, dim)
        seq_len = flat.shape[0]
        out = torch.empty_like(flat)
        fvk.layer_norm(
            int(flat.data_ptr()), w_ptr, b_ptr, int(out.data_ptr()),
            int(seq_len), int(dim), float(eps), cs(),
        )
        if _TRACE:
            _trace_in_out(label, flat, out, weight=w_pinned, bias=b_pinned)
        return out.view_as(x_c)

    return forward


# ──────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────

def install_fvk_norms(model) -> dict:
    """Walk ``model`` and replace WanLayerNorm / WanRMSNorm forwards
    with fvk-backed implementations.

    Returns a stats dict with counts (for the G3a smoke test).

    Constraints:
        * model must already be on cuda + bf16 (frontend's `.to(device)`)
        * weights are read once at install time; if you ``model.to(...)``
          again afterwards, re-run install_fvk_norms
        * the upstream WanLayerNorm uses ``elementwise_affine=False``
          (verified in audit) so we use layer_norm_no_affine_bf16
        * the upstream WanLayerNorm does NOT support a per-instance
          weight; if a future ckpt has affine=True we'd need to fall
          back — guarded with assert below
    """
    WanRMSNorm, WanLayerNorm = _get_wan_norm_classes()
    counts = {"rms": 0, "layer_no_affine": 0, "layer_affine": 0, "skipped": 0}

    for name, module in model.named_modules():
        if isinstance(module, WanRMSNorm):
            assert hasattr(module, "weight"), \
                f"WanRMSNorm {name} missing .weight"
            assert module.weight.is_cuda and \
                module.weight.dtype == torch.bfloat16, \
                f"WanRMSNorm {name}.weight on wrong device/dtype: " \
                f"{module.weight.device}/{module.weight.dtype}"
            module.forward = _make_rms_norm_forward(
                module.weight, float(module.eps), int(module.dim),
                label=name)
            counts["rms"] += 1

        elif isinstance(module, WanLayerNorm):
            dim = int(module.normalized_shape[0])
            eps = float(module.eps)
            if module.weight is None and module.bias is None:
                # Most Wan norm1/norm2 (elementwise_affine=False).
                module.forward = _make_layer_norm_no_affine_forward(
                    eps, dim, label=name)
                counts["layer_no_affine"] += 1
            elif module.weight is not None and module.bias is not None:
                # Wan norm3 (cross-attn input norm, elementwise_affine=True).
                # 30 instances, one per DiT block.
                assert module.weight.is_cuda and \
                    module.weight.dtype == torch.bfloat16, \
                    f"WanLayerNorm {name}.weight on wrong device/dtype"
                assert module.bias.is_cuda and \
                    module.bias.dtype == torch.bfloat16, \
                    f"WanLayerNorm {name}.bias on wrong device/dtype"
                module.forward = _make_layer_norm_affine_forward(
                    module.weight, module.bias, eps, dim, label=name)
                counts["layer_affine"] += 1
            else:
                # Mixed (weight without bias or vice-versa) — unexpected;
                # leave alone instead of silently miscomputing.
                logger.warning(
                    f"[g3a] {name} WanLayerNorm has partial affine "
                    f"(weight={module.weight is not None}, "
                    f"bias={module.bias is not None}); skipping")
                counts["skipped"] += 1

    logger.info(
        f"[g3a] norm swap installed: rms_norm={counts['rms']}, "
        f"layer_norm_no_affine={counts['layer_no_affine']}, "
        f"layer_norm_affine={counts['layer_affine']}, "
        f"skipped={counts['skipped']}")
    return counts
