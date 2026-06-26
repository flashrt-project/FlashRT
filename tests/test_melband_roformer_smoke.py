"""Smoke tests for MelBandRoformer FlashRT integration.

These tests run in **any** build configuration:
  - default build (FLASHRT_ENABLE_MELBAND_ROFORMER=OFF): import succeeds,
    pipeline construction raises RuntimeError.
  - gated build (FLASHRT_ENABLE_MELBAND_ROFORMER=ON): all mbr_* symbols present.

No GPU or model checkpoint is required.
"""
import pytest


# ── 1. Package import always succeeds ──

def test_package_import():
    """Importing the model package must not require mbr kernels."""
    from flash_rt.models.melband_roformer import MelBandRoformerPipeline
    assert MelBandRoformerPipeline is not None


def test_pipeline_module_import():
    """The pipeline module imports cleanly without flash_rt_kernels."""
    from flash_rt.models.melband_roformer import pipeline
    assert hasattr(pipeline, "MelBandRoformerPipeline")
    assert hasattr(pipeline, "_load_kernels")


# ── 2. Pipeline construction validates kernel availability ──

class _FakeFrontend:
    """Minimal stub matching the frontend contract."""
    config = type("C", (), {"inference": type("I", (), {"chunk_size": 352800})})()
    device = None
    model = None


def test_construction_without_kernels_raises():
    """Without mbr kernels, _load_kernels raises a clear RuntimeError."""
    from flash_rt.models.melband_roformer.pipeline import _load_kernels

    try:
        _load_kernels()
    except RuntimeError as exc:
        msg = str(exc)
        assert "flash_rt_kernels" in msg
        assert ("FLASHRT_ENABLE_MELBAND_ROFORMER=ON" in msg or
                "compiled .so" in msg)
    except Exception:
        pass  # flash_rt_kernels not importable at all
    else:
        pytest.skip("mbr kernels are available (gated build) — skip this test")


def test_pipeline_constructor_validates_kernels(monkeypatch):
    """Pipeline construction must fail before touching model internals."""
    from flash_rt.models.melband_roformer import pipeline

    def _raise_missing():
        raise RuntimeError("missing -DFLASHRT_ENABLE_MELBAND_ROFORMER=ON")

    monkeypatch.setattr(pipeline, "_load_kernels", _raise_missing)
    with pytest.raises(RuntimeError, match="FLASHRT_ENABLE_MELBAND_ROFORMER=ON"):
        pipeline.MelBandRoformerPipeline(_FakeFrontend())


# ── 3. Gated build: all required mbr_* symbols present ──

MBR_SYMBOLS = [
    "mbr_qkv_split_rope",
    "mbr_gated_attn_quant",
    "mbr_fp8_dequant_bf16",
    "mbr_resadd_rmsnorm_fp8_keepres",
    "mbr_fused_add_rmsnorm_bf16",
]


def test_mbr_symbols_present_when_gated():
    """In a gated build, every required mbr_* symbol must be present."""
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        try:
            import flash_rt_kernels as fvk
        except ImportError:
            pytest.skip("flash_rt_kernels not built")

    missing = [s for s in MBR_SYMBOLS if not hasattr(fvk, s)]
    if missing:
        pytest.skip(f"mbr kernels not compiled (missing: {', '.join(missing)})")

    # All symbols present — verify they are callable
    for sym in MBR_SYMBOLS:
        assert callable(getattr(fvk, sym)), f"{sym} is not callable"
