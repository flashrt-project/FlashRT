"""load_model routing tests for HyVLA on RTX SM120: the named tiers must reach
the frontend (mirrors tests/test_hyvla_fp4_routing.py)."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_rtx as rtx_mod
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_rtx frontend not importable: {exc}", allow_module_level=True)

import flash_rt  # noqa: E402


class _RecordingFrontend:
    """Stub mirroring HyVLATorchFrontendRtx's constructor signature.

    load_model feature-detects accepted kwargs via inspect.signature(pipe_cls),
    so the named tier parameters must be declared exactly like the real
    frontend or load_model will not forward them.
    """
    last_kwargs = None

    def __init__(self, checkpoint_dir, *, hardware="rtx_sm120",
                 use_fp8=False, use_fp4=False, use_fp4_expert=False,
                 use_fused=False, use_fp8_block128=None, **kwargs):
        _RecordingFrontend.last_kwargs = {
            "hardware": hardware, "use_fp8": use_fp8, "use_fp4": use_fp4,
            "use_fp4_expert": use_fp4_expert, "use_fused": use_fused,
            "use_fp8_block128": use_fp8_block128, **kwargs,
        }
        self.checkpoint_dir = checkpoint_dir


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(rtx_mod, "HyVLATorchFrontendRtx", _RecordingFrontend)
    _RecordingFrontend.last_kwargs = None
    yield _RecordingFrontend


def test_default_route_is_nvfp4_vit_prefill_with_fp8_expert(stubbed):
    # load_model's default on rtx_sm120 = the validated production tier:
    # NVFP4 ViT + prefill, expert tower on FP8 (action cosine 0.99977).
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="rtx_sm120")
    kw = stubbed.last_kwargs
    assert kw is not None, "frontend was never constructed"
    assert kw.get("use_fp8") is True, \
        f"load_model's default use_fp8 must reach the RTX frontend; kwargs={kw}"
    assert kw.get("use_fp4") is True, \
        f"the default tier must enable NVFP4 on ViT + prefill; kwargs={kw}"
    assert kw.get("use_fp4_expert") is False, \
        f"the default tier must keep the expert tower on FP8; kwargs={kw}"


def test_explicit_use_fp4_keeps_expert_on_fp8(stubbed):
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="rtx_sm120", use_fp4=True)
    kw = stubbed.last_kwargs
    assert kw is not None
    assert kw.get("use_fp4") is True, \
        f"use_fp4=True did not reach the RTX frontend; kwargs={kw}"
    # use_fp4_expert defaults to False, so the expert tower stays on FP8;
    # enabling NVFP4 for the expert is an explicit opt-in.
    assert kw.get("use_fp4_expert") is False, \
        f"the expert tower must stay on FP8 by default; kwargs={kw}"


def test_bf16_route_disables_all_quant(stubbed):
    flash_rt.load_model("/nonexistent/fake-ckpt", config="hyvla",
                        framework="torch", hardware="rtx_sm120", use_fp8=False)
    kw = stubbed.last_kwargs
    assert kw is not None
    assert kw.get("use_fp8") in (None, False)
    assert kw.get("use_fp4") in (None, False)
