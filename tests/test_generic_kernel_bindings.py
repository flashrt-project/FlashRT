"""Binding-name regression net for the generic-kernel-helper cleanup (#112)
and the slim-build source gating (Units 1-3).

This guards the ownership cleanup that moves model-neutral helpers out of
Qwen3.6-named files into neutral files, and stays correct under both build
modes. The contract is:

- Neutral binding names (``bf16_matmul_bf16``, ``embedding_lookup_bf16``) MUST
  exist in EVERY build mode: they live in always-compiled neutral TUs, so the
  slim gate never removes them. Existing call sites and external users rely on
  them.
- Legacy binding names (``bf16_matmul_qwen36_bf16``,
  ``qwen36_embedding_lookup_bf16``) live inside Qwen3.6 TUs gated behind
  ``FLASHRT_HAVE_QWEN36_KERNELS``. In the compat default build they MUST exist;
  in a slim build (FLASHRT_SLIM_BUILD=ON) they are intentionally gated out, so
  the test must NOT require them there.

Because these assertions run against the imported ``.so`` (not a build dir), we
detect the build mode from the module itself: a sentinel Qwen3.6 symbol that is
gated under the same ``FLASHRT_HAVE_QWEN36_KERNELS`` macro as the legacy
bindings. If the sentinel is absent, this is a slim build and the legacy names
are expected to be absent too.

These are CPU/import-friendly: importing the compiled ``.so`` does not require
a CUDA device or any model checkpoint, only that the extension built.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FLASHRT_BUILD_DIR", REPO_ROOT / "build"))

# A binding gated under FLASHRT_HAVE_QWEN36_KERNELS alongside the legacy helper
# names. Present in the compat build, absent in slim. Used only as a consistency
# check after the build mode is read from CMakeCache.txt.
QWEN36_SENTINEL = "causal_conv1d_qwen36_bf16"


def _import_kernels():
    try:
        return importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")


def _import_vl_kernels():
    try:
        return importlib.import_module("flash_rt.flash_rt_qwen3_vl_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_qwen3_vl_kernels not importable: {exc}")


def _qwen36_kernels_built(m) -> bool:
    """True if the module was built with the Qwen3.6 kernels (compat build)."""
    return hasattr(m, QWEN36_SENTINEL)


def _cache_bool(name: str) -> bool:
    """Read a BOOL from the configured build dir's CMakeCache.txt.

    Do not infer the mode from exported symbols: that can hide compat-build
    regressions where an entire binding group disappears.
    """
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        pytest.skip(f"no CMakeCache.txt at {cache}; cannot determine build mode")
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(f"{name}:"):
            return line.rsplit("=", 1)[-1].strip().upper() in (
                "ON",
                "TRUE",
                "1",
                "YES",
            )
    pytest.skip(f"{name} not found in {cache}; cannot determine build mode")


def _is_slim_build() -> bool:
    return _cache_bool("FLASHRT_SLIM_BUILD")


def test_legacy_matmul_bindings_exist():
    """Compat build: legacy name present. Slim build: gated out by design."""
    m = _import_kernels()
    if _is_slim_build():
        assert not _qwen36_kernels_built(m), (
            "slim build still exposes Qwen3.6 sentinel binding; the "
            "FLASHRT_HAVE_QWEN36_KERNELS gate did not drop it"
        )
        assert not hasattr(m, "bf16_matmul_qwen36_bf16"), (
            "slim build still exposes legacy bf16_matmul_qwen36_bf16; the "
            "Qwen3.6 TU gate did not drop it"
        )
    else:
        assert _qwen36_kernels_built(m), (
            "compat build is missing Qwen3.6 sentinel binding; the "
            "FLASHRT_HAVE_QWEN36_KERNELS gate may not have been enabled"
        )
        assert hasattr(m, "bf16_matmul_qwen36_bf16")


def test_legacy_embedding_binding_exists():
    """Compat build: legacy name present. Slim build: gated out by design."""
    m = _import_kernels()
    if _is_slim_build():
        assert not _qwen36_kernels_built(m), (
            "slim build still exposes Qwen3.6 sentinel binding; the "
            "FLASHRT_HAVE_QWEN36_KERNELS gate did not drop it"
        )
        assert not hasattr(m, "qwen36_embedding_lookup_bf16"), (
            "slim build still exposes legacy qwen36_embedding_lookup_bf16; the "
            "Qwen3.6 TU gate did not drop it"
        )
    else:
        assert _qwen36_kernels_built(m), (
            "compat build is missing Qwen3.6 sentinel binding; the "
            "FLASHRT_HAVE_QWEN36_KERNELS gate may not have been enabled"
        )
        assert hasattr(m, "qwen36_embedding_lookup_bf16")


def test_legacy_cublaslt_binding_exists_on_vl_module():
    m = _import_vl_kernels()
    if not hasattr(m, "fp8_block128_gemm_blockscaled_sm89_bf16out"):
        pytest.skip(
            "bf16_matmul_cublaslt_bf16 is part of the SM89 Qwen3-VL module; "
            "SM120 builds use the SM120 path instead"
        )
    assert hasattr(m, "bf16_matmul_cublaslt_bf16")


def test_neutral_matmul_binding_exists():
    """Neutral helper must exist in every build mode (never gated)."""
    m = _import_kernels()
    assert hasattr(m, "bf16_matmul_bf16")


def test_neutral_embedding_binding_exists():
    """Neutral helper must exist in every build mode (never gated)."""
    m = _import_kernels()
    assert hasattr(m, "embedding_lookup_bf16")
