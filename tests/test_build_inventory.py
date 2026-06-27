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
# FLASHRT_BUILD_QWEN3_VL=ON, FLASHRT_ENABLE_MOTUS=ON, QWEN35MOE/MELBAND OFF).
# Units 1-3 gate model/arch-specific TUs behind FLASHRT_SLIM_BUILD. This test is
# build-mode aware: it reads FLASHRT_SLIM_BUILD from the configured build dir's
# CMakeCache.txt and asserts the matching surface.
#
#   compat default (FLASHRT_SLIM_BUILD=OFF): 55 direct TUs, unchanged.
#   slim SM89   (FLASHRT_SLIM_BUILD=ON):     33 direct TUs (-22):
#       -5 Motus VAE FP8 (Unit 1), -10 Qwen3.6/linear-attn (Unit 2),
#       -7 SM120/NVFP4-named (Unit 3).
# The direct-source breakdowns below are SM89-specific; other arches add/remove
# arch-owned sources (for example SM120 adds nvfp4_sf_reshape_sm120.cu), so the
# direct-count/category asserts are limited to SM89.
COMPAT_KERNELS_TU = 55
SLIM_SM89_KERNELS_TU = 33

# Per-group breakdown of flash_rt_kernels (mirrors AGENTS.md "Current Build
# Layout"). categorize() drops empty groups, so slim omits sm120_nvfp4_named.
COMPAT_KERNELS_CATEGORIES = {
    "generic_shared": 15,
    "qwen36_linear_attention": 12,
    "sm120_nvfp4_named": 7,
    "motus_video_fp8_history": 7,
    "dit_video": 2,
    "qwen3_family": 2,
    "other": 10,
}
SLIM_SM89_KERNELS_CATEGORIES = {
    "generic_shared": 15,
    "qwen36_linear_attention": 2,
    "motus_video_fp8_history": 2,
    "dit_video": 2,
    "qwen3_family": 2,
    "other": 10,
}


def _cache_bool(name: str) -> bool:
    """Read a BOOL from the configured build dir's CMakeCache.txt."""
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return False
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(f"{name}:"):
            return line.rsplit("=", 1)[-1].strip().upper() in ("ON", "TRUE", "1", "YES")
    return False


def _cache_value(name: str) -> str | None:
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(f"{name}:"):
            return line.rsplit("=", 1)[-1].strip()
    return None


def _is_slim() -> bool:
    return _cache_bool("FLASHRT_SLIM_BUILD")


def _is_sm89() -> bool:
    return (_cache_value("GPU_ARCH") or "").strip() == "89"


def _expected_tu() -> int:
    return SLIM_SM89_KERNELS_TU if _is_slim() else COMPAT_KERNELS_TU


def _expected_categories() -> dict:
    return SLIM_SM89_KERNELS_CATEGORIES if _is_slim() else COMPAT_KERNELS_CATEGORIES


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
    """flash_rt_kernels direct-source surface matches the recorded baseline.

    Mode-aware: compat default expects 55, slim SM89 expects 33. A change means
    a unit added/removed a TU from that mode's build; if intended, update the
    matching constant in the same commit.
    """
    if not _is_sm89():
        pytest.skip("direct TU baseline is recorded for SM89 only")
    entry = _kernels_entry()
    expected = _expected_tu()
    assert entry["count"] == expected, (
        f"flash_rt_kernels now compiles {entry['count']} direct TUs, "
        f"{'slim' if _is_slim() else 'compat'} baseline is {expected}. If this "
        f"is an intended gating change, update the baseline in the same commit."
    )


def test_kernels_category_breakdown():
    """Per-group counts match AGENTS.md's Current Build Layout (mode-aware)."""
    if not _is_sm89():
        pytest.skip("category baseline is recorded for SM89 only")
    entry = _kernels_entry()
    assert entry["categories"] == _expected_categories()


def test_neutral_helpers_in_generic_core():
    """The neutral helpers from #112 must stay in the generic core, never
    gated behind a model-specific option."""
    entry = _kernels_entry()
    generic = set(entry["category_sources"]["generic_shared"])
    assert "csrc/kernels/bf16_matmul_bf16.cu" in generic
    assert "csrc/kernels/embedding_lookup_bf16.cu" in generic


def test_object_libraries_counted_in_total():
    """Object-library TUs linked via $<TARGET_OBJECTS:...> are not in a
    module's direct sources; the inventory must account for them separately so
    the reported compile surface is not understated.

    flash_rt_fa2 compiles 1 direct TU but links the fa2_vendor_obj object
    library; its total must exceed its direct count when attribution is
    available (Makefile link.txt / Ninja build.ninja present).
    """
    _require_build_dir()
    report = inv.collect(BUILD_DIR)
    fa2 = report["targets"]["flash_rt_fa2"]
    if not fa2.get("configured"):
        pytest.skip("flash_rt_fa2 not configured in this build dir")
    if "object_tu_count" not in fa2:
        pytest.skip("no linker manifest in this build dir; attribution skipped")
    assert fa2["object_tu_count"] >= 1
    assert fa2["total_tu_count"] == fa2["count"] + fa2["object_tu_count"]
    assert fa2["total_tu_count"] > fa2["count"]
