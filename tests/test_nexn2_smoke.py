"""Smoke tests for the Nex-N2-mini (qwen3_5_moe) frontend.

CI-friendly: no 35B checkpoint, no golden fixture. Covers the seams a
reviewer needs to trust the model is wired in -- registry routing, static
dims, constructor validation, and (when the gated kernels are built) the
availability of the SM120 kernel symbols the kernelized path calls.

The full cos-vs-golden / token-exact E2E lives in nexn2_dev/tests/ (needs
the checkpoint + refs/nexn2_golden_long2.pt); see docs/nexn2_usage.md.

Run:
    PYTHONPATH=. python -m pytest tests/test_nexn2_smoke.py -v
"""
from __future__ import annotations

import importlib

import pytest


def test_registry_resolves_nexn2():
    """flash_rt.load_model can discover the frontend via the pipeline map."""
    from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class
    # raw map entry (no import side effects)
    assert _PIPELINE_MAP[("nexn2", "torch", "rtx_sm120")] == (
        "flash_rt.frontends.torch.nexn2_rtx", "Nexn2TorchFrontendRtx")
    # resolver imports and returns the class object
    cls = resolve_pipeline_class("nexn2", "torch", "rtx_sm120")
    assert cls.__name__ == "Nexn2TorchFrontendRtx"
    assert cls.__module__ == "flash_rt.frontends.torch.nexn2_rtx"


def test_load_model_redirects_to_direct_construction():
    """load_model marks nexn2 as not-in-the-VLA-path with a clear redirect
    (it is a text LLM), instead of a cryptic kwargs TypeError."""
    import flash_rt
    with pytest.raises(NotImplementedError) as ei:
        flash_rt.load_model("/nonexistent", config="nexn2")
    assert "Nexn2TorchFrontendRtx" in str(ei.value)


def test_frontend_imports():
    """The frontend module + class import without a GPU or checkpoint."""
    m = importlib.import_module("flash_rt.frontends.torch.nexn2_rtx")
    assert hasattr(m, "Nexn2TorchFrontendRtx")


def test_constructor_rejects_bad_quant():
    """quant is validated before any weight load (no checkpoint touched)."""
    from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx
    with pytest.raises(ValueError):
        Nexn2TorchFrontendRtx("/nonexistent", quant="int4")


def test_static_dims_consistent():
    """The forward module's compile-time dims match the qwen3_5_moe config."""
    fwd = importlib.import_module(
        "flash_rt.frontends.torch._nexn2_rtx_forward")
    assert fwd.HID == 2048
    assert (fwd.NQ, fwd.NKV, fwd.HD) == (16, 2, 256)        # full-attn GQA
    assert fwd.NQ % fwd.NKV == 0                            # GQA group
    assert (fwd.NK, fwd.NV, fwd.HK, fwd.HV) == (16, 32, 128, 128)  # GDN
    assert fwd.TOPK == 8
    # conv channels = 2*KD + VD (q/k 16x128 each + v 32x128)
    assert fwd.CONV == 2 * fwd.KD + fwd.VD


def test_attn_backend_shape_metadata():
    """The full-attn backend advertises the GQA / head_dim the kernel needs."""
    from flash_rt.hardware.rtx.attn_backend_nexn2 import (
        make_nexn2_attention_spec,
    )
    spec = make_nexn2_attention_spec(max_seq=2048)
    full = spec["sites"][0]
    assert (full["num_q_heads"], full["num_kv_heads"], full["head_dim"]) \
        == (16, 2, 256)
    assert spec["linear_attn"]["layer_count"] == 30


# ── gated-kernel availability (skipped unless built + GPU present) ──

def _fvk_or_skip():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as e:                          # pragma: no cover
        pytest.skip(f"flash_rt_kernels not importable: {e}")
    return fvk


def test_qwen35moe_kernel_symbols_present():
    """When built with -DFLASHRT_ENABLE_QWEN35MOE=ON the kernelized forward's
    kernels are exported. Skip cleanly on a non-gated build (the symbols are
    intentionally absent there)."""
    fvk = _fvk_or_skip()
    required = [
        "w16a16_gemm_sm120_bf16",          # dense GEMMs (router/lm_head/...)
        "moe_blocktile_mma_sm120_bf16",    # MoE routed experts
        "moe_weighted_sum_sm120_bf16",     # fused MoE unpermute
        "moe_router_topk_sm120_bf16",      # decode router top-k
        "qwen36_partial_rope_qk_bf16",     # partial RoPE
        "causal_conv1d_qwen36_bf16",       # GDN conv1d (prefill)
        "gdn_recurrent_seq_sm120_bf16",    # GDN sequential scan
    ]
    missing = [s for s in required if not hasattr(fvk, s)]
    if missing:
        pytest.skip(
            "qwen3_5_moe kernels not in this build (need "
            f"-DFLASHRT_ENABLE_QWEN35MOE=ON); missing {missing}")
    for s in required:
        assert hasattr(fvk, s)


def test_fa2_causal_available():
    """Prefill full-attn depends on the vendored FA2 causal kernel."""
    try:
        from flash_rt import flash_rt_fa2 as fa2
    except Exception as e:                          # pragma: no cover
        pytest.skip(f"flash_rt_fa2 not importable: {e}")
    assert hasattr(fa2, "fwd_bf16_causal")
