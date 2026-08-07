"""HyVLA Orin hardware-gate (fail-fast) tests — torch.cuda is mocked, no GPU needed."""

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_orin as orin_mod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_orin frontend not importable: {exc}", allow_module_level=True)


class _Probe:
    """Run _require_arch against mocked CUDA state."""

    _cls = orin_mod.HyVLATorchFrontendOrin

    def run(self):
        obj = object.__new__(self._cls)
        return self._cls._require_arch(obj)


def test_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe().run()


def test_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    with pytest.raises(RuntimeError, match="requires Jetson Orin SM87"):
        _Probe().run()


def test_accepts_sm87(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 7))
    _Probe().run()  # must not raise


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe().run()


def test_fp4_rejected_before_any_cuda_work():
    # SM87 has no FP4 tensor cores; the constructor must fail fast,
    # before checkpoint loading or CUDA allocation.
    with pytest.raises(RuntimeError, match="does not support FP4"):
        orin_mod.HyVLATorchFrontendOrin("/nonexistent/fake-ckpt", use_fp4=True)


def test_missing_prompt_raises_runtime_error():
    fe = orin_mod.HyVLATorchFrontendOrin.__new__(orin_mod.HyVLATorchFrontendOrin)
    # Minimal state to reach the prompt contract check only.
    fe._prompt = None
    fe._lang_tokens = None
    with pytest.raises(RuntimeError, match="set_prompt"):
        orin_mod.HyVLATorchFrontendOrin.predict_actions(fe, images=None)
