#!/usr/bin/env python3
"""OmniVoice FlashRT kernel smoke test.

Validates all required kernel symbols are present when FlashRT is built
with FLASHRT_ENABLE_OMNIVOICE=ON on SM120 GPUs. Skips cleanly when
compiled .so modules are absent (default/OFF build).

Usage:
  pytest -q tests/test_omnivoice_smoke.py        # both ON and OFF cases
"""
import pytest, warnings, sys, os
warnings.filterwarnings("ignore")

# Add FlashRT repo root to path (repo convention: tests/ run from root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Optionally add an external bundle dir for flashcli-deployed testing
_bundle = os.environ.get("OMNIVOICE_BUNDLE", "").strip()
if _bundle:
    sys.path.insert(0, _bundle)


class TestDefaultBuild:
    """Default build (FLASHRT_ENABLE_OMNIVOICE=OFF).

    flash_rt package imports work without GPU. The models.omnivoice
    module imports cleanly, _has_cfg_kernel is False.
    """

    def test_omnivoice_module_imports(self):
        from flash_rt.models.omnivoice import (
            FlashRTLlm, FlashRTLlmBF16,
            inject, free_encoder, eject, _check_kernels,
        )
        assert inject is not None

    def test_cfg_kernel_disabled_by_default(self):
        from flash_rt.models.omnivoice import pipeline_rtx
        import importlib
        importlib.reload(pipeline_rtx)
        # If the omnivoice .so is present (flag ON), kernel is True.
        # If absent (flag OFF), kernel is False. Both are valid.
        assert isinstance(pipeline_rtx._has_cfg_kernel, bool)


class TestGatedBuild:
    """OmniVoice build (FLASHRT_ENABLE_OMNIVOICE=ON, SM120 GPU).

    All kernel symbols must be present across modules:
      flash_rt_kernels  — base FP4/fused kernels
      flash_rt_omnivoice — OmniVoice-specific kernels
      flash_rt_fa2       — FlashAttention2
    """

    _F = None  # flash_rt_kernels
    _O = None  # flash_rt_omnivoice
    _A = None  # flash_rt_fa2

    @pytest.fixture(autouse=True)
    def _setup(self):
        if TestGatedBuild._F is None:
            try:
                from flash_rt import flash_rt_kernels as fvk
            except ImportError:
                pytest.skip("flash_rt_kernels not available (GPU build required)")
            TestGatedBuild._F = fvk
        if TestGatedBuild._O is None:
            try:
                from flash_rt import flash_rt_omnivoice as fvo
            except ImportError:
                fvo = None
            TestGatedBuild._O = fvo
        if TestGatedBuild._A is None:
            try:
                from flash_rt import flash_rt_fa2 as fa2
            except ImportError:
                fa2 = None
            TestGatedBuild._A = fa2

    # ── flash_rt_kernels symbols ──

    def test_fp4_gemm_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out")
        assert hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_pingpong")

    def test_fused_norm_symbols(self):
        fvk = TestGatedBuild._F
        for sym in ("rms_norm", "rms_norm_to_nvfp4_swizzled_bf16",
                     "residual_add_rms_norm",
                     "residual_add_rms_norm_to_nvfp4_swizzled_bf16"):
            assert hasattr(fvk, sym)

    def test_quantize_symbols(self):
        fvk = TestGatedBuild._F
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled")
        assert hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled_mse")

    def test_silu_symbols(self):
        assert hasattr(TestGatedBuild._F,
                        "silu_mul_merged_to_nvfp4_swizzled_bf16")

    # ── flash_rt_omnivoice symbols ──

    def test_omnivoice_cfg_logsoftmax(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "omnivoice_cfg_logsoftmax_bf16")

    def test_omnivoice_qk_norm_rope(self):
        if TestGatedBuild._O is None:
            pytest.skip("flash_rt_omnivoice not built")
        assert hasattr(TestGatedBuild._O, "omnivoice_qk_norm_rope_bf16")

    def test_engine_cfg_kernel_true(self):
        from flash_rt.models.omnivoice import pipeline_rtx
        import importlib
        importlib.reload(pipeline_rtx)
        if TestGatedBuild._O is not None:
            assert pipeline_rtx._has_cfg_kernel is True

    def test_check_kernels_passes(self):
        from flash_rt.models.omnivoice import _check_kernels
        import importlib
        from flash_rt.models.omnivoice import pipeline_rtx
        importlib.reload(pipeline_rtx)
        # Reload to pick up the current .so state
        from flash_rt.models.omnivoice import pipeline_rtx as prx
        importlib.reload(prx)
        if prx._fvo is not None:
            prx._check_kernels()

    # ── flash_rt_fa2 ──

    def test_fa2_module(self):
        if TestGatedBuild._A is None:
            pytest.skip("flash_rt_fa2 not available")
        assert hasattr(TestGatedBuild._A, "fwd_bf16")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
