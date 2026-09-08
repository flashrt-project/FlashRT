"""The Ascend NPU dispatch gates.

The NPU backend is routed before the CUDA/ROCm checks in ``detect_arch``:
on a CANN box ``torch.cuda.is_available()`` is False while torch_npu
drives the part. These tests pin that routing and the NPU frontend
registry so a regression back to "raise on any non-CUDA box" is caught
on the box that can run it, and skipped (not failed) on CI without an
Ascend device.
"""

import pytest


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _npu_available(), reason="no usable Ascend NPU on this machine")


def test_detect_arch_returns_npu_on_ascend_box():
    from flash_rt.hardware import detect_arch
    assert detect_arch() == "npu"


def test_pi05_npu_registered_in_pipeline_map():
    from flash_rt.hardware import resolve_pipeline_class
    cls = resolve_pipeline_class("pi05", "torch", "npu")
    assert cls.__name__ == "Pi05TorchFrontendNpu"


def test_npu_package_imports():
    from flash_rt.npu.core import device
    assert device.is_available()


def test_npu_frontend_requires_valid_checkpoint():
    from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu
    # The NPU frontend is implemented; a missing checkpoint must fail
    # loudly with a clear error, not a NotImplementedError stub.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        Pi05TorchFrontendNpu("/nonexistent/pi05_checkpoint")
