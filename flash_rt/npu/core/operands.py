"""Operand checks for the Ascend kernels, in front of every raw pointer.

Every native entry point on this backend is reached through ``ctypes`` with
``data_ptr()``, so nothing between the caller and the kernel knows what the
pointer points at. A host tensor's address handed to a kernel is not a wrong
answer, it is a device fault or a silent read of whatever the device has at that
address, and the frames a camera produces are on the host by default.

So the checks live here rather than being restated per call site, and they
inspect only tensor attributes — which is why they can be tested on a machine
with no Ascend device at all.
"""

from __future__ import annotations

import torch


def require(tensor, name: str, *, device, dtype=None, shape=None,
            row_pitch: int | None = None, contiguous: bool = False) -> torch.Tensor:
    """Refuse an operand before its address reaches a kernel.

    ``device`` is mandatory and is compared exactly, index included: a kernel
    launched on one die cannot read a buffer allocated on another, and
    ``torch.device("npu")`` and ``torch.device("npu:0")`` are not the same
    object to compare against.

    ``shape`` may carry ``None`` in any axis to leave it free. ``row_pitch``
    checks the last axis's stride, for the operands a kernel reads as a slice of
    a wider buffer; ``contiguous`` is the stricter form for the ones it does not.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    want = torch.device(device)
    if tensor.device != want:
        raise ValueError(
            f"{name} must be on {want}, got {tensor.device}. The Ascend kernels "
            "are handed raw addresses, so a host tensor here is a device fault "
            "rather than a wrong answer — copy it to the device first.")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None:
        actual = tuple(tensor.shape)
        if len(actual) != len(shape) or any(
                want_axis is not None and want_axis != have
                for want_axis, have in zip(shape, actual)):
            pretty = "(" + ", ".join("*" if a is None else str(a) for a in shape) + ")"
            raise ValueError(f"{name} must be {pretty}, got {actual}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tensor.stride()}")
    if row_pitch is not None:
        if tensor.dim() < 2:
            raise ValueError(f"{name} needs at least two axes to have a row pitch")
        if tensor.stride(-1) != 1 or tensor.stride(-2) != row_pitch:
            raise ValueError(
                f"{name} must have unit element stride and a row pitch of "
                f"{row_pitch}, got strides {tensor.stride()}")
    return tensor


def require_npu(tensor, name: str) -> torch.device:
    """The one check that cannot be expressed as "same as that other operand".

    Every other operand of a launch is compared against the first one, so a
    launch whose operands are consistently on the host passes all of those and
    still faults. This is what refuses it, and it returns the device the rest are
    then held to.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.device.type != "npu":
        raise ValueError(
            f"{name} must be on an Ascend device, got {tensor.device}. The native "
            "kernels are handed raw addresses, so a host tensor here is a device "
            "fault rather than a wrong answer.")
    return tensor.device


def same_device(*tensors) -> torch.device:
    """The one device every operand of a launch has to be on.

    Raises rather than picking one: mixing devices inside a launch is the same
    fault as passing a host tensor, and it is worth naming which operand broke
    the agreement.
    """
    if not tensors:
        raise ValueError("a launch needs at least one operand")
    first = tensors[0][1].device
    for name, tensor in tensors:
        if tensor.device != first:
            raise ValueError(
                f"every operand of one launch must be on the same device; "
                f"{tensors[0][0]} is on {first} and {name} is on {tensor.device}")
    return first
