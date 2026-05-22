"""Cast/bind LingBot-VLA loaded weights to a target dtype + device.

The WeightLoader leaves tensors in their on-disk fp32 layout (CPU
mmap from safetensors). is the "bridge" step: walk every attribute the
spec wrote and move it to ``(device, dtype)`` for kernel handoff. The
upstream LingBot baseline inference runs in bf16+cuda
(``policy.eval().cuda().to(torch.bfloat16)``), so that is the default.

Why a separate step (not a Transform on the spec):
  * The spec stays "lossless round-trip" — no quant decisions baked into
    declarative form. Quant decisions live in calibration.
  * The same loaded target can be cast to different dtypes (bf16 for
    inference, fp32 for a precision-debug build) without rebuilding the
    spec.
  * Frontend lifecycle is clearer: ``load_weights → bind_to_device →
    set_prompt``. Each step has one job.

Usage:

    from flash_rt.executors.torch_weights import SafetensorsSource
    from flash_rt.executors.weight_loader import WeightLoader
    from flash_rt.frontends.torch._lingbot_thor_spec import build_spec
    from flash_rt.models.lingbot.buffer_binder import bind_target_to_device

    source = SafetensorsSource("lingbot-vla-4b/model.safetensors",
                               device="cpu", strip_prefix="")
    target = SomeFrontend()  # any object with attribute slots
    WeightLoader(source, target=target, spec=build_spec()).run()

    bind_target_to_device(target, dtype=torch.bfloat16, device="cuda")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


@dataclass
class BindStats:
    """Diagnostic counters returned by ``bind_target_to_device``."""

    num_singletons: int = 0
    num_layered: int = 0
    num_tensors_total: int = 0
    bytes_before: int = 0          # sum of nbytes BEFORE the cast/move
    bytes_after: int = 0           # sum of nbytes AFTER the cast/move
    skipped_attrs: list[str] = None  # attrs that were not tensors/lists of tensors

    def __post_init__(self):
        if self.skipped_attrs is None:
            self.skipped_attrs = []


def _is_loaded_tensor(obj) -> bool:
    return isinstance(obj, torch.Tensor)


def _is_loaded_list(obj) -> bool:
    """A list whose entries are all tensors (i.e. ``TensorList`` sink output)."""
    return (isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(x, torch.Tensor) for x in obj))


def _attrs_to_bind(target) -> Iterable[str]:
    """Yield public attribute names; skip dunders + callables + descriptors."""
    for name in sorted(vars(target)):
        if name.startswith("_"):
            continue
        yield name


def bind_target_to_device(
    target,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    non_blocking: bool = True,
) -> BindStats:
    """Cast every loaded tensor on ``target`` to ``(device, dtype)`` in place.

    ``target`` should be the object that received the WeightLoader output
    (i.e. has attributes set by the spec — both ``Attr`` singletons
    and ``TensorList`` lists). Non-tensor attributes are left untouched
    and counted in ``BindStats.skipped_attrs``.

    Args:
        target: object loaded by ``WeightLoader``.
        dtype: target dtype. Default ``bfloat16`` (matches LingBot baseline).
        device: target device. Default ``"cuda"``.
        non_blocking: passed through to ``Tensor.to``.

    Returns:
        ``BindStats`` with counts of moved tensors and pre/post byte sizes
        for memory-budget verification.

    Notes:
        Each tensor moves to GPU then casts; the original CPU tensor is
        garbage-collected by the caller releasing the reference (we
        overwrite the attribute slot with the new device tensor). The
        SafetensorsSource itself may still hold mmap views but those are
        not in resident RAM.

        Layered attributes (``TensorList``) are rebuilt as fresh lists
        with the moved tensors; the original list object is replaced so
        the previous CPU tensors lose all references and can be GC'd.
    """
    device = torch.device(device) if isinstance(device, str) else device
    stats = BindStats()

    for name in list(_attrs_to_bind(target)):
        obj = getattr(target, name)

        if _is_loaded_tensor(obj):
            stats.bytes_before += obj.element_size() * obj.numel()
            new_t = obj.to(device=device, dtype=dtype, non_blocking=non_blocking)
            setattr(target, name, new_t)
            stats.num_singletons += 1
            stats.num_tensors_total += 1
            stats.bytes_after += new_t.element_size() * new_t.numel()

        elif _is_loaded_list(obj):
            new_list: list[torch.Tensor] = []
            for t in obj:
                stats.bytes_before += t.element_size() * t.numel()
                new_t = t.to(device=device, dtype=dtype, non_blocking=non_blocking)
                new_list.append(new_t)
                stats.num_tensors_total += 1
                stats.bytes_after += new_t.element_size() * new_t.numel()
            setattr(target, name, new_list)
            stats.num_layered += len(new_list)

        else:
            stats.skipped_attrs.append(name)

    if device.type == "cuda":
        # Ensure the H2D copies have actually committed before the caller
        # touches the tensors. Without this, .data_ptr() reads on a fresh
        # graph capture can race the async copy.
        torch.cuda.synchronize(device)

    logger.info(
        "[lingbot] bound %d tensors to %s/%s (%.2f GB -> %.2f GB)",
        stats.num_tensors_total, device, dtype,
        stats.bytes_before / (1 << 30), stats.bytes_after / (1 << 30),
    )
    return stats


__all__ = ["bind_target_to_device", "BindStats"]
