"""HyVLA RTX (SM120) hardware-gate and input-contract tests (no GPU needed).

Mirrors ``tests/test_orin_hyvla05_arch_gate.py`` for the RTX consumer
Blackwell target. ``torch.cuda`` is mocked, so this module runs anywhere.
SM121 is deliberately rejected: this backend ships an sm_120a build.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_rtx as rtx_mod
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)


class _Probe:
    """Run _require_arch against mocked CUDA state."""

    _cls = rtx_mod.HyVLATorchFrontendRtx

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
    with pytest.raises(RuntimeError, match="RTX Blackwell SM120"):
        _Probe().run()


def test_accepts_sm120(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    _Probe().run()  # must not raise


def test_rejects_sm121(monkeypatch):
    # SM121 / GB10 is out of scope for the sm_120a build; the gate must fail
    # fast with a clear message rather than defer to a kernel-launch failure.
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    with pytest.raises(RuntimeError, match="SM121"):
        _Probe().run()


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe().run()


@pytest.mark.parametrize("value", ["0", "false", "False", "yes"])
def test_other_env_values_do_not_skip_probe(monkeypatch, value):
    # Only the exact value "1" skips the probe (matches the Thor/Orin gate).
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe().run()


def test_tier_kwargs_are_declared_for_load_model_routing():
    # load_model feature-detects named tiers via inspect.signature; the RTX
    # frontend must declare the standard names or the optimized tier is not
    # reachable through the public API.
    import inspect

    params = inspect.signature(
        rtx_mod.HyVLATorchFrontendRtx.__init__).parameters
    for name in ("use_fp8", "use_fp4", "use_fp4_expert", "use_fused",
                 "use_fp8_block128"):
        assert name in params, f"{name} missing from the RTX frontend signature"
    # use_fp4_expert is a conservative opt-in: the expert denoise tower stays
    # on FP8 unless a caller explicitly enables NVFP4 for it.
    assert params["use_fp4_expert"].default is False


def _bare_frontend():
    fe = rtx_mod.HyVLATorchFrontendRtx.__new__(
        rtx_mod.HyVLATorchFrontendRtx)
    fe.max_state_dim = 8
    fe.chunk = 4
    fe.max_action_dim = 3
    fe._prompt = None
    fe._lang_tokens = object()
    fe._vit_merge = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("invalid inputs reached ViT"))
    return fe


def test_invalid_state_rejected_before_gpu_or_cache_work():
    with pytest.raises(ValueError, match="state has 9 dims"):
        _bare_frontend().predict_actions(None, state=[0] * 9)


def test_invalid_noise_rejected_before_gpu_or_cache_work():
    with pytest.raises(ValueError, match="noise must have 12 elements"):
        _bare_frontend().predict_actions(None, noise=[0] * 11)
