"""Reject unsupported deployment configurations before loading a checkpoint."""
from types import SimpleNamespace

import pytest
import torch

from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu


def test_fixed_prompt_mode_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="fixed padded prompts"):
        Pi05TorchFrontendNpu("missing-checkpoint", state_prompt_mode="fixed")


def test_compiled_architecture_must_match_current_device(monkeypatch):
    from flash_rt.npu.core import device, native_kernels

    def query_soc():
        return b"Ascend910B4"

    library = SimpleNamespace(flashrt_npu_soc_version=query_soc)
    monkeypatch.setattr(device, "ensure_npu", lambda: None)
    monkeypatch.setattr(device, "device_name", lambda index: "Ascend310P3")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    monkeypatch.setattr(native_kernels, "DecoderRope", lambda: SimpleNamespace(library=library))
    monkeypatch.setattr(native_kernels, "GatedAdaRms", lambda: object())
    with pytest.raises(RuntimeError, match="targets Ascend910B4.*Ascend310P3"):
        Pi05TorchFrontendNpu("missing-checkpoint")
