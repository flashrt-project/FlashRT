"""Reject unsupported deployment configurations before loading a checkpoint."""
from types import SimpleNamespace

import pytest
import torch

from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu


def test_fixed_prompt_mode_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="fixed padded prompts"):
        Pi05TorchFrontendNpu("missing-checkpoint", state_prompt_mode="fixed")


def test_compiled_architecture_must_match_current_device(monkeypatch):
    """The check moved out of this frontend and into the loader that every
    shared object goes through, so it now covers all four rather than one."""
    from flash_rt.npu.core import abi

    class _Symbol:
        def __init__(self, value):
            self._value = value
            self.restype = None
            self.argtypes = None

        def __call__(self):
            return self._value

    library = SimpleNamespace(
        flashrt_npu_abi_version=_Symbol(abi.ABI_VERSION),
        flashrt_npu_soc_version=_Symbol(abi.SOC_VERSION.encode()))
    monkeypatch.setattr(abi, "_running_soc", lambda: "Ascend310P3")
    with pytest.raises(RuntimeError, match="targets Ascend910B4.*Ascend310P3"):
        abi.verify(library, "dispatch")
