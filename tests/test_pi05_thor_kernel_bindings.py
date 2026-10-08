"""Installed Pi0.5 Thor extension ownership and adapter contracts."""
import importlib
from types import SimpleNamespace

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


@pytest.mark.parametrize("decoder", [False, True])
def test_legacy_constructor_uses_shared_fp4_variant_symbol(monkeypatch, decoder):
    """Execute the legacy constructor without loading weights or using CUDA."""
    module = pytest.importorskip("flash_rt.frontends.torch.pi05_thor_fp4")
    from flash_rt import flash_rt_fp4

    assert hasattr(flash_rt_fp4, "cutlass_fp4_gemm_num_variants")
    assert not hasattr(flash_rt_fp4, "pi05_cutlass_fp4_gemm_num_variants")
    calls = []

    def variant_count():
        calls.append("shared")
        return 11

    def base_init(self, *args, **kwargs):
        self.Le = 18

    cls = module.Pi05TorchFrontendThorFP4
    monkeypatch.setattr(module.Pi05TorchFrontendThor, "__init__", base_init)
    monkeypatch.setattr(module, "_HAS_FP4", True)
    # Deliberately expose only the shared ABI, never the model-owned alias.
    monkeypatch.setattr(module, "fvk_fp4", SimpleNamespace(
        cutlass_fp4_gemm_num_variants=variant_count))
    monkeypatch.setattr(cls, "_prepare_fp4_encoder", lambda self: None)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _: "NVIDIA Thor")
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda _: (11, 0))

    if decoder:
        # Stop at the range guard, before decoder weight loading or allocation.
        with pytest.raises(ValueError, match="decoder_qkv_variant must be in"):
            cls("checkpoint", use_fp4_decoder=True, decoder_qkv_variant=11)
    else:
        frontend = cls("checkpoint", use_fp4_encoder_ffn=True)
        assert frontend._fp4_layers == frozenset((7, 8, 9))
        assert not frontend.use_fp4_decoder
    assert calls == ["shared"]
