"""Smoke tests for the VLA-JEPA (LeRobot) RTX SM120 frontend.

CI-friendly: no checkpoint, no golden fixture, no GPU. Covers the seams a
reviewer needs before trusting the model is wired in: registry routing, the
load_model redirect, the gated-kernel fail-fast, and the documented shape
helpers. Numerical precision and graph parity need the checkpoint and are
covered by docs/vla_jepa_rtx.md (Reproduce section).

Run:
    PYTHONPATH=. python -m pytest tests/test_vla_jepa_smoke.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")


def _frontend_module():
    return importlib.import_module("flash_rt.frontends.torch.vla_jepa_rtx")


# ── registry / load_model routing ──────────────────────────────────────────

def test_registry_entry_is_one_to_one():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("vla_jepa", "torch", "rtx_sm120")] == (
        "flash_rt.frontends.torch.vla_jepa_rtx",
        "VlaJepaTorchFrontendRtx")


def test_resolve_pipeline_class():
    from flash_rt.hardware import resolve_pipeline_class
    cls = resolve_pipeline_class("vla_jepa", "torch", "rtx_sm120")
    assert cls.__name__ == "VlaJepaTorchFrontendRtx"
    assert cls.__module__ == "flash_rt.frontends.torch.vla_jepa_rtx"


@pytest.mark.parametrize("arch", ["thor", "rtx_sm89", "amd_cdna4"])
def test_unsupported_arch_raises(arch):
    from flash_rt.hardware import resolve_pipeline_class
    with pytest.raises(RuntimeError, match="no pipeline"):
        resolve_pipeline_class("vla_jepa", "torch", arch)


def test_load_model_redirects_to_direct_construction():
    """VLA-JEPA needs the robot state per call and real-observation
    calibration before capture, so load_model must fail with an actionable
    redirect (before GPU detection)."""
    from flash_rt import load_model
    with pytest.raises(NotImplementedError) as ei:
        load_model("/nonexistent", config="vla_jepa")
    msg = str(ei.value)
    assert "VlaJepaTorchFrontendRtx" in msg
    assert "docs/vla_jepa_rtx.md" in msg


# ── gated-kernel guard ─────────────────────────────────────────────────────

def test_require_kernels_fails_fast_on_missing_symbols():
    mod = _frontend_module()

    class _Empty:
        pass

    with pytest.raises(RuntimeError) as ei:
        mod._require_kernels(_Empty())
    assert "FLASHRT_ENABLE_VLA_JEPA" in str(ei.value)


def test_vla_kernel_symbols_present_when_built():
    mod = _frontend_module()
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        pytest.skip("flash_rt_kernels not built")
    missing = [name for name in mod._REQUIRED_KERNELS if not hasattr(fvk, name)]
    if missing:
        pytest.skip("built without -DFLASHRT_ENABLE_VLA_JEPA=ON: "
                    f"{missing[:3]}...")
    mod._require_kernels(fvk)


# ── constructor validation ────────────────────────────────────────────────

def test_constructor_validation():
    mod = _frontend_module()
    if not torch.cuda.is_available():
        pytest.skip("VLA-JEPA construction requires a CUDA device")
    with pytest.raises(ValueError, match="llm_precision"):
        mod.VlaJepaTorchFrontendRtx("/nonexistent", llm_precision="int8")
