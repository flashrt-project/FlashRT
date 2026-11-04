"""Native-provider boundaries must remain importable without ROCm or Triton."""
import builtins
import importlib
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")


def test_provider_import_does_not_require_triton_or_hip_extension(monkeypatch):
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == "triton" or name.startswith("triton.") or name.endswith("flash_rt_amd_kernels"):
            raise AssertionError(f"provider imported device implementation eagerly: {name}")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    for name in ("attention","normalization"):
        module = importlib.import_module("flash_rt.amd.hardware.rdna."+name)
        importlib.reload(module)


def test_stale_native_extension_has_actionable_error(monkeypatch):
    import flash_rt.amd
    from flash_rt.amd.hardware.rdna._native import kernels
    monkeypatch.setitem(flash_rt.amd.__dict__,"flash_rt_amd_kernels",SimpleNamespace())
    with pytest.raises(RuntimeError,match="rebuild csrc/amd"):
        kernels()
