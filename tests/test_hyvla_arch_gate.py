"""HyVLA hardware-gate (fail-fast) tests — torch.cuda is mocked, no GPU needed."""

import os

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_thor as hy_mod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_thor frontend not importable: {exc}", allow_module_level=True)


class _Probe:
    """Run _require_arch against mocked CUDA state."""

    def __init__(self, available, capability):
        self._cls = hy_mod.HyVLATorchFrontendThor
        self._available = available
        self._capability = capability

    def run(self):
        obj = object.__new__(self._cls)
        return self._cls._require_arch(obj)


def test_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe(False, None).run()


def test_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(RuntimeError, match="requires Jetson Thor SM110"):
        _Probe(True, (8, 9)).run()


def test_accepts_sm110(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    _Probe(True, (11, 0)).run()  # must not raise


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe(False, None).run()
