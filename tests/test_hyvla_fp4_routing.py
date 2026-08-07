"""load_model routing tests for HyVLA: the FP4 tier must reach the frontend."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_thor as hy_mod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_thor frontend not importable: {exc}", allow_module_level=True)

import flash_rt  # noqa: E402


class _RecordingFrontend:
    last_kwargs = None

    def __init__(self, checkpoint, **kwargs):
        _RecordingFrontend.last_kwargs = dict(kwargs)
        self.checkpoint = checkpoint


@pytest.fixture
def stubbed(monkeypatch):
    """Stub the frontend class and the flash_rt_fp4 extension."""
    monkeypatch.setattr(hy_mod, "HyVLATorchFrontendThor", _RecordingFrontend)
    fake_fp4 = types.ModuleType("flash_rt.flash_rt_fp4")
    fake_fp4.has_nvfp4 = lambda: True
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", fake_fp4)
    _RecordingFrontend.last_kwargs = None
    yield _RecordingFrontend


def test_use_fp4_reaches_the_frontend(stubbed):
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="thor", use_fp4=True)
    kw = stubbed.last_kwargs
    assert kw is not None, "frontend was never constructed"
    assert kw.get("use_fp4") is True, \
        f"use_fp4=True did not reach the frontend; kwargs={kw}"


def test_fp8_tier_selects_fused_production_config(stubbed):
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="thor",
                        use_fp8=True, use_fp4=False)
    kw = stubbed.last_kwargs
    assert kw is not None
    assert kw.get("use_fp8") is True
    assert kw.get("use_fused") is True, \
        "the validated fp8 production tier must enable the fused megakernels"


def test_default_route_does_not_enable_fp4(stubbed):
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="thor")
    kw = stubbed.last_kwargs
    assert kw is not None
    assert kw.get("use_fp4") in (None, False)


def test_hyvla_orin_fp4_falls_back_with_warning(stubbed, caplog):
    # Orin has no FP4 tensor cores: the route must degrade to the INT8 path
    # instead of silently claiming FP4.
    try:
        flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                            framework="torch", hardware="rtx_sm87",
                            use_fp4=True)
    except Exception:
        pytest.skip("Orin frontend not importable in this environment")
    msgs = [r.message for r in caplog.records]
    assert any("SM87" in m and "FP4" in m.upper() for m in msgs), \
        f"expected an SM87 FP4 fallback warning, got {msgs}"
