"""Installed Pi0.5 Thor extension ownership and adapter contracts."""
import importlib

import pytest


@pytest.fixture
def extension():
    return pytest.importorskip("flash_rt.flash_rt_pi05_thor")


def test_model_extension_exports_only_owned_entry_points(extension):
    exports = {
        name for name in dir(extension)
        if not name.startswith("_") and callable(getattr(extension, name))
    }
    assert exports
    assert all(name.startswith("pi05_") for name in exports)
    assert not hasattr(extension, "attention_qkv_fp16_lt")


def test_model_adapters_preserve_shared_helper_names(extension):
    adapters = importlib.import_module("flash_rt.models.pi05.thor_kernels")
    from flash_rt import flash_rt_kernels, flash_rt_fp4

    assert adapters.kernels is not flash_rt_kernels
    assert adapters.fp4 is not flash_rt_fp4
    assert adapters.kernels.quantize_fp8_static_fp16 is extension.pi05_quantize_fp8_static_fp16
    assert adapters.kernels.cutlass_fp8_sq is extension.pi05_cutlass_fp8_sq
    assert adapters.kernels.qkv_split_rope_kvcache_fp16_vec is extension.pi05_qkv_split_rope_kvcache_fp16_vec
    assert adapters.fp4.set_pdl is extension.pi05_set_pdl
    assert adapters.fp4.pi05_siglip_gemm_bias_gelu_fp4out is extension.pi05_siglip_gemm_bias_gelu_fp4out
    assert adapters.fp4.pi05_siglip_gemm_bias_res_fp16 is extension.pi05_siglip_gemm_bias_res_fp16
