"""Contract surface of the pre-quantized checkpoint door.

The scheme door quantizes at runtime; this door adopts checkpoints that
are already quantized in someone else's packed layout. These tests pin
what is checkable without a GPU or the compressed-tensors package: the
detection predicate, the loud refusals, and the impl's envelope.
"""

import pytest
import torch

from flash_rt.structures.prequantized import (AdoptionReport,
                                              _is_ct_nvfp4_linear,
                                              adopt_prequantized)


def test_detection_wants_the_run_compressed_form():
    # the compressed load path leaves a plain Linear carrying the packed
    # parameters; an ordinary Linear must not be detected
    plain = torch.nn.Linear(8, 8)
    assert not _is_ct_nvfp4_linear(plain)
    packed = torch.nn.Linear(8, 8)
    packed.weight_packed = torch.zeros(8, 4, dtype=torch.uint8)
    assert _is_ct_nvfp4_linear(packed)
    # a non-Linear with the attribute is not a projection
    holder = torch.nn.Module()
    holder.weight_packed = packed.weight_packed
    assert not _is_ct_nvfp4_linear(holder)


def test_unknown_format_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown pre-quantized format"):
        adopt_prequantized(torch.nn.Linear(8, 8), "gguf_q4")


def test_report_summary_shape():
    rep = AdoptionReport(fmt="ct_nvfp4")
    rep.replaced = ["a", "b"]
    rep.conversion_rel_l2 = {"a": 0.10, "b": 0.13}
    s = rep.summary()
    assert s["replaced_projections"] == 2
    assert s["conversion_rel_l2_max"] == 0.13
    assert rep.worst_conversion == 0.13


def test_nvfp4_impl_contract_surface():
    from flash_rt.structures.impls.linear_proj import nvfp4_dynamic

    assert nvfp4_dynamic.KERNEL_DEP["repo"] == "flashrt/fp4-gemm"
    assert nvfp4_dynamic._check({"w": torch.zeros(1024, 4096)}) == (
        1024, 4096)
    # the envelope mirrors the kernel's own checks and nothing more:
    # the 27B host's 17408-wide FFN is a qualified shape, not a wall
    assert nvfp4_dynamic._check({"w": torch.zeros(1024, 17408)}) == (
        1024, 17408)
    # NVFP4 block scale factors are per 16 elements; K must divide
    with pytest.raises(ValueError, match="multiple of"):
        nvfp4_dynamic._check({"w": torch.zeros(1024, 4104)})
    with pytest.raises(ValueError, match="bias shape"):
        nvfp4_dynamic._check({"w": torch.zeros(1024, 4096),
                              "b": torch.zeros(960)})


def test_nvfp4_activation_quantizer_prefers_direct_bf16_entry():
    from flash_rt.structures.impls.linear_proj import nvfp4_dynamic

    calls = []

    class Kern:
        @staticmethod
        def quantize_fp4_sfa_bf16(x):
            calls.append(("bf16", x.dtype, x.is_contiguous()))
            return "bf16-packed"

        @staticmethod
        def quantize_fp4_sfa_fp16(x):
            calls.append(("fp16", x.dtype, x.is_contiguous()))
            return "fp16-packed"

    got = nvfp4_dynamic._quantize_activation(
        Kern(), torch.zeros((1, 32), dtype=torch.bfloat16))
    assert got == "bf16-packed"
    assert calls == [("bf16", torch.bfloat16, True)]


def test_nvfp4_activation_quantizer_retains_legacy_fallback():
    from flash_rt.structures.impls.linear_proj import nvfp4_dynamic

    calls = []

    class LegacyKern:
        @staticmethod
        def quantize_fp4_sfa_fp16(x):
            calls.append((x.dtype, x.is_contiguous()))
            return "fp16-packed"

    got = nvfp4_dynamic._quantize_activation(
        LegacyKern(), torch.zeros((1, 32), dtype=torch.bfloat16))
    assert got == "fp16-packed"
    assert calls == [(torch.float16, True)]


def test_adoption_is_exported_from_the_package_door():
    from flash_rt import structures

    assert "adopt_prequantized" in structures.__all__
