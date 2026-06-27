"""Build-inventory baseline for the VLA-deployment kernel-build split.

This test records the *current* compile surface of the FlashRT pybind modules
so the source-gating units (slim builds) can be reviewed against a known
baseline. It is a build-structure guardrail, not a CUDA behavior test: it reads
the configured CMake build dir and asserts the directly-compiled translation
unit counts per target.

It skips cleanly when there is no configured build dir (e.g. CI without a CUDA
toolchain), so it never blocks unrelated work.

As each gating unit lands, the expected baseline below is updated in the same
commit so the test keeps describing the real configured surface.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FLASHRT_BUILD_DIR", REPO_ROOT / "build"))

# Baseline captured on the local SM89 configure (GPU_ARCH=89,
# FLASHRT_BUILD_QWEN3_VL=ON, FLASHRT_ENABLE_MOTUS=ON, QWEN35MOE/MELBAND OFF)
# before any slim-build gating. Units 1-3 gate model/arch-specific TUs behind
# FLASHRT_SLIM_BUILD; with the compat default (OFF) the default-configure count
# here is unchanged at 55. A slim SM89 build (FLASHRT_SLIM_BUILD=ON) drops 22 TUs
# to 33: -5 Motus VAE FP8 (Unit 1), -10 Qwen3.6/linear-attn (Unit 2), -7
# SM120/NVFP4-named (Unit 3). The slim count is build-dir specific and not
# asserted here (CI configures the default build only).
BASELINE_KERNELS_TU = 55

# Category breakdown of flash_rt_kernels (mirrors AGENTS.md "Current Build
# Layout"). These are the groups Units 1-3 gate.
BASELINE_KERNELS_CATEGORIES = {
    "generic_shared": 15,
    "qwen36_linear_attention": 12,
    "sm120_nvfp4_named": 7,
    "motus_video_fp8_history": 7,
    "dit_video": 2,
    "qwen3_family": 2,
    "other": 10,
}


def _load_inventory():
    path = REPO_ROOT / "scripts" / "build_inventory.py"
    spec = importlib.util.spec_from_file_location("build_inventory", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load_inventory()


def _require_build_dir():
    if not BUILD_DIR.is_dir():
        pytest.skip(f"no configured build dir at {BUILD_DIR}")


def _kernels_entry():
    _require_build_dir()
    report = inv.collect(BUILD_DIR)
    entry = report["targets"]["flash_rt_kernels"]
    if not entry.get("configured"):
        pytest.skip("flash_rt_kernels not configured in this build dir")
    return entry


def test_inventory_script_imports_and_collects():
    """The inventory tooling itself must be importable and runnable."""
    _require_build_dir()
    report = inv.collect(BUILD_DIR)
    assert "targets" in report
    assert set(inv.TARGETS) <= set(report["targets"])


def test_kernels_baseline_tu_count():
    """flash_rt_kernels compile surface matches the recorded baseline.

    A change here means a unit added/removed a TU from the default build. If
    that is intended, update BASELINE_KERNELS_TU in the same commit.
    """
    entry = _kernels_entry()
    assert entry["count"] == BASELINE_KERNELS_TU, (
        f"flash_rt_kernels now compiles {entry['count']} TUs, baseline is "
        f"{BASELINE_KERNELS_TU}. If this is an intended gating change, update "
        f"the baseline in the same commit."
    )


def test_kernels_category_breakdown():
    """Per-group counts match AGENTS.md's Current Build Layout."""
    entry = _kernels_entry()
    assert entry["categories"] == BASELINE_KERNELS_CATEGORIES


def test_neutral_helpers_in_generic_core():
    """The neutral helpers from #112 must stay in the generic core, never
    gated behind a model-specific option."""
    entry = _kernels_entry()
    generic = set(entry["category_sources"]["generic_shared"])
    assert "csrc/kernels/bf16_matmul_bf16.cu" in generic
    assert "csrc/kernels/embedding_lookup_bf16.cu" in generic
