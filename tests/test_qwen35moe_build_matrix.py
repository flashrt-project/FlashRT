"""The qwen3_5_moe tiers must not reach a build that turned them off.

Checked by reading the gates -- CMake's tier blocks and the bindings' guards --
because the property is about configurations nobody builds. A build proves its
own configuration works; only the gates can say what the other four contain.

The configure matrix these correspond to is printed by
``scripts/qwen35moe_build_matrix.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "qwen35moe_build_matrix",
    Path(__file__).resolve().parent.parent
    / "scripts" / "qwen35moe_build_matrix.py",
)
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


def test_no_tier_source_is_compiled_by_a_default_build():
    assert matrix.ungated_model_sources() == []


def test_no_tier_symbol_is_exported_by_a_default_build():
    assert matrix.ungated_model_symbols() == []


def test_every_tier_contributes_sources_and_symbols():
    sources, symbols = matrix.tier_sources(), matrix.guarded_symbols()
    for option, gate in matrix.TIERS.items():
        assert sources[option], f"{option} adds no sources"
        assert symbols[gate], f"{gate} guards no bindings"


def test_the_grouped_moe_gemm_has_its_own_gate():
    """It used to be a second source in the object library every Thor build
    compiles, which meant an unrelated Thor build paid for it."""
    symbols = matrix.guarded_symbols()
    gated = symbols["FLASHRT_HAVE_QWEN35MOE_GROUPED_SM100"]
    assert "moe_grouped_gemm_nvfp4_sm100_bf16out" in gated
    assert "moe_grouped_gemm_nvfp4_sm100_scratch_bytes" in gated

    cmake = (Path(__file__).resolve().parent.parent
             / "CMakeLists.txt").read_text(encoding="utf-8")
    guard = 'if(GPU_ARCH STREQUAL "110" AND FLASHRT_ENABLE_QWEN35MOE_W4A16)'
    assert guard in cmake


def test_the_grouped_quantisers_left_the_shared_quantiser():
    """They wrote the grouped GEMM's own scale-factor layout, so they belong
    to the tier that calls them, not to the file every NVFP4 build compiles."""
    symbols = matrix.guarded_symbols()["FLASHRT_HAVE_QWEN35MOE_W4A16"]
    for name in ("moe_grouped_quant_nvfp4_bf16",
                 "moe_grouped_silu_quant_nvfp4_bf16",
                 "moe_grouped_silu_quant_nvfp4_warp_bf16"):
        assert name in symbols

    root = Path(__file__).resolve().parent.parent
    quantize = (root / "csrc" / "kernels" / "quantize.cu").read_text(
        encoding="utf-8")
    assert "moe_grouped_quant_nvfp4_kernel" not in quantize


def test_thor_fa2_is_opt_in():
    """Every other Thor model uses its own attention path and would only be
    paying the compile time and the binary."""
    cmake = (Path(__file__).resolve().parent.parent
             / "CMakeLists.txt").read_text(encoding="utf-8")
    assert ('option(FLASHRT_ENABLE_THOR_FA2\n'
            '       "Build the vendored FA2 attention kernels on Jetson AGX '
            'Thor (sm_110)" OFF)') in cmake
    assert '(GPU_ARCH STREQUAL "110" AND FLASHRT_ENABLE_THOR_FA2) OR' in cmake


def test_the_block_scaled_tier_refuses_a_target_without_the_mma():
    """CUTLASS compiles those units anywhere and substitutes an invalid
    control path for the MMA, so a successful build would fail at run time."""
    cmake = (Path(__file__).resolve().parent.parent
             / "CMakeLists.txt").read_text(encoding="utf-8")
    block = cmake[cmake.index("if(FLASHRT_ENABLE_QWEN35MOE_W4A4)"):]
    block = block[:block.index("message(STATUS \"qwen3_5_moe block-scaled")]
    assert "FATAL_ERROR" in block
    assert "NOT ENABLE_NVFP4" in block


@pytest.mark.parametrize(
    "name,flags,expect", matrix.CONFIGURE_MATRIX)
def test_the_configure_matrix_is_documented(name, flags, expect):
    """The five configurations a reviewer reproduces, and the doc that lists
    the two supported ones. Cheap, but it is what keeps the build commands in
    the guide from drifting from the flags the gates actually read."""
    doc = (Path(__file__).resolve().parent.parent
           / "docs" / "qwen36_moe_usage.md").read_text(encoding="utf-8")
    for flag in flags.split():
        if flag.startswith("-DFLASHRT_"):
            assert flag.split("=")[0].removeprefix("-D") in doc, (
                f"{name}: {flag} is not mentioned in the model guide")
