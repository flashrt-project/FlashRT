"""Smoke tests for the MolmoAct2 (allenai) RTX SM120 frontend.

CI-friendly: no checkpoint, no golden fixture, no GPU. Covers the seams a
reviewer needs to trust the model is wired in -- registry routing, the
load_model redirect, the gated-kernel guard, the static dimensions of the
three towers, and the pure prompt/state helpers.

The cosine-vs-reference and CUDA-Graph parity harness needs the checkpoint
and is documented in docs/molmoact2_usage.md (Reproduce section).

Run:
    PYTHONPATH=. python -m pytest tests/test_molmoact2_smoke.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

# MolmoAct2's frontend imports both extension modules at module scope: the
# model kernels behind -DFLASHRT_ENABLE_MOLMOACT2=ON and the RTX FA2 module.
# Skip the import-dependent tests (not the module) on builds that omit either.
_REQUIRED_KERNELS = (
    "molmo_mul_bf16", "molmo_qk_norm_rope_bf16", "molmo_pool_attn_bf16",
    "molmo_resize_patchify_bf16", "molmo_state_discretize_f32",
    "molmo_action_mk_layers", "molmo_action_denorm_bf16",
    "molmo_vit_attn_bf16", "molmo_vit_attn_bf16_fp8out",
    "molmo_smallm_gemm_bias_res_bf16out",
)


def _extensions_or_skip():
    try:
        import flash_rt.flash_rt_fa2  # noqa: F401
        import flash_rt.flash_rt_kernels as fvk
    except ImportError as exc:                     # pragma: no cover
        pytest.skip(f"FlashRT CUDA extensions not built: {exc}")
    return fvk


def _frontend_module():
    fvk = _extensions_or_skip()
    return importlib.import_module("flash_rt.frontends.torch.molmoact2_rtx"), fvk


# ── registry / load_model routing ──────────────────────────────────────────

def test_registry_entry_is_one_to_one():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("molmoact2", "torch", "rtx_sm120")] == (
        "flash_rt.frontends.torch.molmoact2_rtx",
        "MolmoAct2TorchFrontendRtx")


def test_resolve_pipeline_class():
    _extensions_or_skip()
    from flash_rt.hardware import resolve_pipeline_class
    cls = resolve_pipeline_class("molmoact2", "torch", "rtx_sm120")
    assert cls.__name__ == "MolmoAct2TorchFrontendRtx"
    assert cls.__module__ == "flash_rt.frontends.torch.molmoact2_rtx"


@pytest.mark.parametrize("arch", ["thor", "rtx_sm89"])
def test_unsupported_arch_raises(arch):
    from flash_rt.hardware import resolve_pipeline_class
    with pytest.raises(RuntimeError, match="no pipeline"):
        resolve_pipeline_class("molmoact2", "torch", arch)


def test_load_model_redirects_to_direct_construction():
    """MolmoAct2 needs the robot state per call and a real-data calibration
    before capture, so load_model must fail with an actionable redirect."""
    from flash_rt import load_model
    with pytest.raises(NotImplementedError) as ei:
        load_model("/nonexistent", config="molmoact2")
    msg = str(ei.value)
    assert "MolmoAct2TorchFrontendRtx" in msg
    assert "docs/molmoact2_usage.md" in msg


# ── gated-kernel guard ─────────────────────────────────────────────────────

def test_require_kernels_fails_fast_on_missing_symbols():
    mod, _ = _frontend_module()

    class _Empty:
        pass

    with pytest.raises(RuntimeError) as ei:
        mod._require_kernels(_Empty())
    assert "FLASHRT_ENABLE_MOLMOACT2" in str(ei.value)


def test_molmo_kernel_symbols_present_when_built():
    mod, fvk = _frontend_module()
    missing = [name for name in _REQUIRED_KERNELS if not hasattr(fvk, name)]
    if missing:
        pytest.skip("built without -DFLASHRT_ENABLE_MOLMOACT2=ON: "
                    f"{missing[:3]}...")
    mod._require_kernels(fvk)      # every name the frontend declares


def test_constructor_validation():
    mod, fvk = _frontend_module()
    if not hasattr(fvk, _REQUIRED_KERNELS[0]):
        pytest.skip("needs -DFLASHRT_ENABLE_MOLMOACT2=ON")
    with pytest.raises(ValueError, match="precision"):
        mod.MolmoAct2TorchFrontendRtx("/nonexistent", precision="int8")


# ── static dimensions ──────────────────────────────────────────────────────

def test_static_dims_match_the_checkpoint():
    p = importlib.import_module("flash_rt.models.molmoact2.pipeline_rtx")
    assert (p.NUM_VIEWS, p.IMG_H, p.IMG_W) == (3, 360, 640)
    assert (p.GRID, p.POS_TOKENS, p.POOL_GRID, p.POOL_TOKENS) == (27, 729, 14, 196)
    assert p.TOTAL_PATCHES == 2187 and p.TOTAL_POOLED == 588
    # Vision tower: 25 blocks, 16 x 72 = 1152, two feature layers concatenated.
    assert (p.VIS_LAYERS, p.VIS_NH, p.VIS_HD, p.VIS_D) == (25, 16, 72, 1152)
    assert p.VIS_FEAT_D == 2304 and p.VIS_FFN_PAD == 4352
    # Text tower: Qwen2.5-style GQA 32/8 heads of 128.
    assert (p.TXT_LAYERS, p.TXT_NH, p.TXT_NKV, p.TXT_HD) == (36, 32, 8, 128)
    assert (p.TXT_D, p.TXT_FFN) == (2560, 9728)
    assert p.TXT_QKV_DIM == 6144 and p.TXT_KV_DIM == 1024
    # Action expert: 36 layers, 8 x 96, 10-step flow, 30-step chunk.
    assert (p.ACT_LAYERS, p.ACT_NH, p.ACT_HD, p.ACT_D, p.ACT_H) == \
        (36, 8, 96, 768, 3072)
    assert (p.ACT_STEPS, p.ACT_HORIZON, p.ACT_MAX_DIM) == (10, 30, 32)


# ── pure helpers ────────────────────────────────────────────────────────────

def test_question_normalization():
    mod, _ = _frontend_module()
    assert mod._normalize_question_text("What  is   the robot doing?") == \
        "is the robot doing"
    assert mod._normalize_question_text("  Please  Pick up the block.  ") == \
        "please pick up the block"
    assert mod._normalize_question_text("Go home? Stop.") == "go home; stop"


def test_robot_prompt_contains_the_static_tags():
    mod, _ = _frontend_module()
    text = mod._build_robot_text(
        task="pick up the block", discrete_state_string="<state_1><state_2>",
        setup_type="yam_dual", control_mode="joint", num_images=3)
    assert text.startswith("Image 1<|image|>Image 2<|image|>Image 3<|image|>")
    assert "The task is to pick up the block." in text
    assert mod.SETUP_START_TOKEN in text and mod.SETUP_END_TOKEN in text
    assert mod.CONTROL_START_TOKEN in text and mod.CONTROL_END_TOKEN in text
    assert "<state_1><state_2>" in text
    assert text.endswith(mod.ACTION_OUTPUT_TOKEN)


def test_state_discretization_is_bounded():
    mod, _ = _frontend_module()
    import numpy as np
    tokens = 64
    # Normalized inputs are clipped to [-1, 1] before binning: a wildly
    # out-of-range state can never produce an embedding index outside
    # [0, num_tokens - 1].
    assert mod._discrete_state_string(
        np.array([-2.0, 0.0, 0.5, 2.0], np.float32), tokens) == \
        (f"{mod.STATE_START_TOKEN}<state_0><state_32><state_47>"
         f"<state_63>{mod.STATE_END_TOKEN}")
    assert mod._discrete_state_string(
        np.array([-1.0, 1.0], np.float32), tokens) == \
        f"{mod.STATE_START_TOKEN}<state_0><state_63>{mod.STATE_END_TOKEN}"


# ── numerical unit tests for the new elementwise producers ─────────────────

def _cuda_kernels_or_skip():
    fvk = _extensions_or_skip()
    if not hasattr(fvk, "molmo_mul_bf16"):
        pytest.skip("needs -DFLASHRT_ENABLE_MOLMOACT2=ON")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return fvk


def test_mul_bf16_matches_torch():
    fvk = _cuda_kernels_or_skip()
    n = 4096
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    rc = fvk.molmo_mul_bf16(a.data_ptr(), b.data_ptr(), out.data_ptr(), n)
    assert rc == 0
    torch.cuda.synchronize()
    # Both paths round one float product to bf16.
    torch.testing.assert_close(out, a * b, rtol=0, atol=0)


def test_pool_mean_matches_masked_mean():
    fvk = _cuda_kernels_or_skip()
    P, I, D, rows = 4, 4, 64, 10
    torch.manual_seed(0)
    feats = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
    idx = torch.tensor([[0, 1, 2, 3],
                        [1, 2, 3, 4],
                        [-1, -1, 5, 6],
                        [7, -1, 8, 9]], dtype=torch.int32, device="cuda")
    out = torch.empty(P, D, device="cuda", dtype=torch.bfloat16)
    rc = fvk.molmo_pool_mean_bf16(feats.data_ptr(), idx.data_ptr(),
                                  out.data_ptr(), P, I, D)
    assert rc == 0
    torch.cuda.synchronize()
    valid = idx >= 0
    gathered = feats[idx.clamp_min(0).long()].float()          # (P, I, D)
    ref = (gathered * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
    torch.testing.assert_close(out, ref.bfloat16(), rtol=0, atol=1e-2)


def test_parse_layer_set():
    mod, _ = _frontend_module()
    n = mod.VIS_LAYERS
    assert mod.MolmoAct2TorchFrontendRtx._parse_layer_set("0") == frozenset()
    assert mod.MolmoAct2TorchFrontendRtx._parse_layer_set("all") == \
        frozenset(range(n))
    assert mod.MolmoAct2TorchFrontendRtx._parse_layer_set("0-2,4") == \
        frozenset({0, 1, 2, 4})
    # Out-of-range indices are dropped rather than forwarded to a kernel.
    assert mod.MolmoAct2TorchFrontendRtx._parse_layer_set("0,999") == \
        frozenset({0})
