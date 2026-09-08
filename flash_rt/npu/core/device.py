"""Ascend NPU device helpers (torch_npu).

The CUDA/AMD backends manage device memory through ctypes over
libcudart/libamdhip64 and pass raw pointers to C++ kernels. The NPU
backend has no such kernel ABI: torch_npu tensors are the buffers and
every op goes through torch_npu/aclnn. These helpers are the small
shared seam (init, streams, synchronize) the rest of ``flash_rt/npu``
builds on.
"""

from __future__ import annotations


def import_torch_npu():
    """Import torch_npu (registers ``torch.npu``); raise ImportError if absent."""
    import torch_npu  # noqa: F401
    return torch_npu


def is_available() -> bool:
    """True when torch_npu is installed and an Ascend NPU is usable."""
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


def ensure_npu() -> None:
    """Fail loudly when no Ascend NPU runtime is available.

    Raised before any checkpoint/weight work so a misconfigured box fails
    with a readable message instead of a mid-compute AttributeError.
    """
    if not is_available():
        raise RuntimeError(
            "FlashRT NPU backend requires torch_npu with a usable Ascend "
            "device (torch.npu.is_available()==False). Install a torch_npu "
            "matching the CANN toolkit and check the device with npu-smi.")


def device_count() -> int:
    import torch
    import torch_npu  # noqa: F401
    return int(torch.npu.device_count())


def device_name(index: int = 0) -> str:
    import torch
    import torch_npu  # noqa: F401
    return str(torch.npu.get_device_name(index))


def synchronize() -> None:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.synchronize()


def current_stream():
    import torch
    import torch_npu  # noqa: F401
    return torch.npu.current_stream()


def set_stream(stream) -> None:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_stream(stream)


def memory_reserved() -> int:
    import torch
    import torch_npu  # noqa: F401
    return int(torch.npu.memory_reserved())
