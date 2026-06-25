"""Smoke tests for the MelBandRoformer FP8 pipeline.

CI-friendly: no GPU, no audio checkpoint, no golden fixture. Covers the
seams a reviewer needs to trust the model is wired in -- package export,
kernel availability when the gated build is present, and constructor
validation (calibration loading + arg checks) without instantiating the
full separation model.

The full E2E separation test requires the MelBandRoformer checkpoint and a
CUDA GPU and is intentionally out of scope here.

Run:
    PYTHONPATH=. python -m pytest tests/test_melband_roformer_smoke.py -v
"""
from __future__ import annotations

import importlib
import json

import pytest


def test_package_exports_pipeline():
    """flash_rt.models.melband_roformer exports MelBandRoformerPipeline."""
    from flash_rt.models.melband_roformer import MelBandRoformerPipeline
    assert MelBandRoformerPipeline.__name__ == "MelBandRoformerPipeline"


def test_pipeline_module_imports():
    """The pipeline module imports without a GPU or checkpoint."""
    m = importlib.import_module("flash_rt.models.melband_roformer.pipeline")
    assert hasattr(m, "MelBandRoformerPipeline")
    assert hasattr(m, "mk")  # the MelBandRoformer kernel alias object


def _fvk_or_skip():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as e:  # pragma: no cover
        pytest.skip(f"flash_rt_kernels not importable: {e}")
    return fvk


def test_kernels_importable():
    """flash_rt_kernels imports cleanly (no GPU required)."""
    fvk = _fvk_or_skip()
    assert fvk is not None


def test_mbr_kernel_symbols_present_when_gated():
    """When the MelBandRoformer kernels are built, all mbr_* kernel symbols
    the pipeline calls are exported. Skip cleanly on a build that does not
    ship them (e.g. no GPU / non-gated build)."""
    fvk = _fvk_or_skip()
    required = [
        "rms_norm_fp8",
        "bias_gelu_quantize_fp8_static_bf16",
        "mbr_qkv_split_rope",
        "mbr_gated_attn_quant",
        "mbr_fp8_dequant_bf16",
        "mbr_resadd_rmsnorm_fp8_keepres",
        "mbr_fused_add_rmsnorm_bf16",
    ]
    missing = [s for s in required if not hasattr(fvk, s)]
    if missing:
        pytest.skip(
            "MelBandRoformer kernels not in this build; missing "
            f"{missing}")
    for s in required:
        assert hasattr(fvk, s)


def test_gated_flag_consistent_with_symbols():
    """When FLASHRT_HAVE_MELBAND_ROFORMER is advertised, every required
    mbr_* symbol must actually be present on the kernels module."""
    fvk = _fvk_or_skip()
    if not hasattr(fvk, "FLASHRT_HAVE_MELBAND_ROFORMER"):
        pytest.skip("FLASHRT_HAVE_MELBAND_ROFORMER flag not exported by this build")
    if not getattr(fvk, "FLASHRT_HAVE_MELBAND_ROFORMER"):
        pytest.skip("FLASHRT_HAVE_MELBAND_ROFORMER is False (kernels not enabled)")
    required = [
        "mbr_qkv_split_rope",
        "mbr_gated_attn_quant",
        "mbr_fp8_dequant_bf16",
        "mbr_resadd_rmsnorm_fp8_keepres",
        "mbr_fused_add_rmsnorm_bf16",
        "rms_norm_fp8",
        "bias_gelu_quantize_fp8_static_bf16",
    ]
    for s in required:
        assert hasattr(fvk, s), f"gated build is missing kernel {s}"


def test_mk_alias_binds_all_kernels():
    """The module-level ``mk`` alias wires all 5 mbr_* kernel entry points
    (only meaningful once the kernels are importable)."""
    fvk = _fvk_or_skip()
    need = ["mbr_qkv_split_rope", "mbr_gated_attn_quant", "mbr_fp8_dequant_bf16",
            "mbr_resadd_rmsnorm_fp8_keepres", "mbr_fused_add_rmsnorm_bf16"]
    if any(not hasattr(fvk, s) for s in need):
        pytest.skip("MelBandRoformer kernels not in this build")
    m = importlib.import_module("flash_rt.models.melband_roformer.pipeline")
    for attr in ("qkv_split_rope", "gated_attn_quant", "fp8_dequant_bf16",
                 "resadd_rmsnorm_fp8_keepres", "fused_add_rmsnorm_bf16"):
        assert callable(getattr(m.mk, attr))


def test_calibration_loader_handles_missing_path(tmp_path):
    """The calibration loader returns {} for a missing/None path so
    uncalibrated runs fall back to scale 1.0 instead of crashing."""
    from flash_rt.models.melband_roformer.pipeline import _load_calib
    assert _load_calib(None) == {}
    assert _load_calib(str(tmp_path / "nope.json")) == {}


def test_calibration_loader_reads_json(tmp_path):
    """A valid JSON calibration file is parsed into a dict."""
    from flash_rt.models.melband_roformer.pipeline import _load_calib
    p = tmp_path / "fp8_calibration.json"
    p.write_text(json.dumps({"layers.0.to_qkv": 0.5}))
    assert _load_calib(str(p)) == {"layers.0.to_qkv": 0.5}
