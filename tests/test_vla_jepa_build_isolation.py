"""Configured source and runtime symbol checks for VLA-JEPA's K128 variants."""
import importlib
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=("fp4_w4a16_gemm_sm120_bf16out_k128", "fp4_w4a16_gemm_residual_sm120_bf16out_k128",
         "fp4_w4a16_gemm_sm120_bf16out_bias_k128", "fp4_w4a16_gemm_bias_gelu_fp4out_sm120_k128")


def test_k128_sources_follow_the_model_build_gate():
    directory=Path(os.environ.get("FLASHRT_BUILD_DIR",ROOT/"build"))
    cache=directory/"CMakeCache.txt"
    if not cache.exists():pytest.skip("requires a configured CUDA build")
    enabled="FLASHRT_ENABLE_VLA_JEPA:BOOL=ON" in cache.read_text()
    commands=subprocess.check_output(["ninja","-C",str(directory),"-t","commands","flash_rt_kernels"],text=True)
    assert ("cutlass_nvfp4_gemm_bias_gelu_fp4out_k128_sm120.cu" in commands)==enabled
    for line in commands.splitlines():
        if " -c " in line and "cutlass_nvfp4_w4a16_gemm_sm120.cu " in line:
            assert ("FLASHRT_HAVE_VLA_JEPA=1" in line)==enabled
            break
    else:pytest.fail("shared W4A16 translation unit is missing")


def test_k128_symbols_follow_the_built_model_gate():
    mode=os.environ.get("FLASHRT_TEST_VLA_JEPA_EXPORTS")
    if mode not in ("ON","OFF"):pytest.skip("select the built module's ON/OFF export contract")
    kernels=importlib.import_module("flash_rt.flash_rt_kernels")
    for symbol in (*SYMBOLS,"VlaJepaKernelPolicy"):
        assert hasattr(kernels,symbol)==(mode=="ON"),symbol
    # These pre-existing shared interfaces must survive either setting.
    for symbol in ("fp4_w4a16_gemm_sm120_bf16out","nvfp4_gemm_bias_res_k128_bf16out"):
        assert callable(getattr(kernels,symbol))
