"""Binding-name regression net for the generic-kernel-helper cleanup (#112).

This guards the ownership cleanup that moves model-neutral helpers out of
Qwen3.6-named files into neutral files. The contract for that refactor is:

- Legacy binding names MUST keep existing (we do not remove old Python
  bindings in the cleanup PR), so existing call sites and external users do
  not break.
- As each neutral helper is introduced, its neutral binding name MUST also
  exist on the same module, sharing one implementation with the legacy name.

Unit 0 only asserts the legacy baseline (neutral names do not exist yet). The
neutral-name assertions below are activated in the unit that introduces each
neutral binding (Unit 1 for matmul, Unit 2 for embedding) by flipping the
corresponding ``NEUTRAL_*_AVAILABLE`` switch.

These are CPU/import-friendly: importing the compiled ``.so`` does not require
a CUDA device or any model checkpoint, only that the extension built.
"""

from __future__ import annotations

import importlib

import pytest

# Cleanup progress switches. Each is flipped to True in the same commit that
# introduces the corresponding neutral binding, which is also where the
# matching neutral-name assertion starts being enforced.
NEUTRAL_MATMUL_AVAILABLE = False  # Unit 1: bf16_matmul_bf16
NEUTRAL_EMBEDDING_AVAILABLE = False  # Unit 2: embedding_lookup_bf16


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


def test_legacy_matmul_bindings_exist():
    m = _import_kernels()
    assert hasattr(m, "bf16_matmul_qwen36_bf16")


def test_legacy_embedding_binding_exists():
    m = _import_kernels()
    assert hasattr(m, "qwen36_embedding_lookup_bf16")


def test_legacy_cublaslt_binding_exists_on_vl_module():
    m = _import_vl_kernels()
    assert hasattr(m, "bf16_matmul_cublaslt_bf16")


@pytest.mark.skipif(
    not NEUTRAL_MATMUL_AVAILABLE,
    reason="neutral bf16_matmul_bf16 binding introduced in Unit 1",
)
def test_neutral_matmul_binding_exists():
    m = _import_kernels()
    assert hasattr(m, "bf16_matmul_bf16")


@pytest.mark.skipif(
    not NEUTRAL_EMBEDDING_AVAILABLE,
    reason="neutral embedding_lookup_bf16 binding introduced in Unit 2",
)
def test_neutral_embedding_binding_exists():
    m = _import_kernels()
    assert hasattr(m, "embedding_lookup_bf16")
