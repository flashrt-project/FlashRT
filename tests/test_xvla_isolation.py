"""Preflight and device policy work without weights or X-VLA extensions."""
from types import SimpleNamespace

import pytest
import torch

from flash_rt.frontends.torch import xvla_rtx as module


def test_missing_kernels_fail_before_checkpoint_access(monkeypatch):
    from flash_rt import _extensions
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    seen = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: seen.append(d) or (12, 0))
    monkeypatch.setattr(_extensions, "require", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module.XVLATorchFrontendRtx, "_load_weights",
                        lambda *a: pytest.fail("checkpoint must not load before kernel checks"))
    with pytest.raises(RuntimeError, match="FLASHRT_BUILD_XVLA=ON"):
        module.XVLATorchFrontendRtx("nonexistent-checkpoint", device="cuda:1")
    assert seen == [torch.device("cuda:1")]


def test_present_stubs_are_not_mistaken_for_a_supported_profile(monkeypatch):
    from flash_rt import _extensions
    core = SimpleNamespace(**{name: object() for name in module._REQUIRED_CORE})
    fa2 = SimpleNamespace(**{name: object() for name in module._REQUIRED_FA2})
    fa2.has_xvla_tile64 = lambda: False
    monkeypatch.setattr(_extensions, "require", lambda name="flash_rt_kernels", **kw:
                        core if name == "flash_rt_kernels" else fa2)
    with pytest.raises(RuntimeError, match="FA2 X-VLA BF16 tile64 profile"):
        module._require_kernels()


@pytest.mark.parametrize("requested", ["cpu", "cuda:0", "cuda:2"])
def test_noncurrent_device_rejected(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    with pytest.raises(ValueError, match="current CUDA device"):
        module.XVLATorchFrontendRtx._resolve_device(requested)


def test_unspecified_cuda_device_resolves_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert module.XVLATorchFrontendRtx._resolve_device("cuda") == torch.device("cuda:2")


def test_changed_current_device_rejected_before_prompt_or_capture(monkeypatch):
    frontend = object.__new__(module.XVLATorchFrontendRtx)
    frontend.device = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(RuntimeError, match="device changed"):
        frontend.set_prompt("pick")
    with pytest.raises(RuntimeError, match="device changed"):
        frontend.capture()
    with pytest.raises(RuntimeError, match="device changed"):
        frontend.infer({})
    with pytest.raises(RuntimeError, match="device changed"):
        frontend.predict_actions(None, None)


def test_pipeline_restores_scope_even_on_failure():
    from flash_rt.models.xvla.pipeline_rtx import XVLARTXPipeline
    state = {"active": False}
    class Scope:
        def __enter__(self): state["active"] = True
        def __exit__(self, *args): state["active"] = False
    pipeline = object.__new__(XVLARTXPipeline)
    pipeline.use_pdl = True
    pipeline.fvk = SimpleNamespace(xvla_pdl_scope=lambda enabled: Scope())
    def fail():
        assert state["active"]
        raise ValueError("launch failed")
    pipeline._run = fail
    with pytest.raises(ValueError, match="launch failed"):
        pipeline.run()
    assert not state["active"]


def test_required_core_surface_is_still_bound():
    from pathlib import Path
    bindings = (Path(__file__).resolve().parents[1] / "csrc/bindings.cpp").read_text()
    for name in module._REQUIRED_CORE:
        assert '"' + name + '"' in bindings, name
