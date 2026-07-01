from __future__ import annotations

import importlib


def test_qwen3_vl_rtx_bf16_frontend_imports():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    assert hasattr(m, 'Qwen3VlTorchFrontendRtxBF16')
    assert hasattr(m, '_require_qwen3_vl_rtx_bf16_kernels')


def test_qwen3_vl_rtx_bf16_kernel_lists_are_bf16_only():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    names = set(m._QWEN3_VL_RTX_BF16_CORE_FNS) | set(m._QWEN3_VL_RTX_BF16_VISION_FNS)
    assert 'bf16_matmul_bf16' in names
    assert 'qwen3_vl_bf16_gemv_m1' in names
    assert 'qwen3_q_norm_rope_qstage_bf16' in names
    assert not any('fp8' in name or 'nvfp4' in name for name in names)
