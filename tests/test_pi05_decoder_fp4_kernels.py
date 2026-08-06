"""Numerical contracts for Pi0.5 decoder-specific NVFP4 preprocess kernels."""

import torch

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk
from flash_rt.executors.fp4_utils import quant_weight_nvfp4


def test_pi05_decoder_fp4_preprocess_matches_explicit_path():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA Thor"
    assert tuple(torch.cuda.get_device_capability(0)) == (11, 0)
    assert fvk_fp4.has_nvfp4()

    torch.manual_seed(20260725)
    rows, dim, out_dim = 10, 1024, 1024
    x = torch.randn(rows, dim, dtype=torch.float16, device='cuda') * 0.2
    style = torch.randn(
        rows, 3 * dim, dtype=torch.float16, device='cuda') * 0.1
    packed = torch.empty(rows, dim // 2, dtype=torch.uint8, device='cuda')
    packed_ref = torch.empty_like(packed)
    packed_native = torch.empty_like(packed)
    sfa = torch.zeros(
        fvk_fp4.sfa_size_bytes(rows, dim, False),
        dtype=torch.uint8, device='cuda')
    sfa_ref = torch.zeros_like(sfa)
    sfa_native = torch.zeros_like(sfa)
    gate = torch.empty_like(x)
    gate_ref = torch.empty_like(x)
    gate_native = torch.empty_like(x)
    normed = torch.empty_like(x)

    fvk_fp4.pi05_adarms_fp4_sfa_fp16(
        x.data_ptr(), style.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
        gate.data_ptr(), rows, dim, 0)
    fvk.adarms_fp16(
        x.data_ptr(), style.data_ptr(), normed.data_ptr(), gate_ref.data_ptr(),
        rows, dim, 0)
    assert fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        normed.data_ptr(), packed_ref.data_ptr(), sfa_ref.data_ptr(),
        rows, dim, False, 0) == 0
    fvk_fp4.pi05_adarms_fp4_sfa_native_fp16(
        x.data_ptr(), style.data_ptr(), packed_native.data_ptr(),
        sfa_native.data_ptr(), gate_native.data_ptr(), rows, dim, 0)
    torch.cuda.synchronize()

    assert torch.equal(packed, packed_ref)
    assert torch.equal(sfa, sfa_ref)
    assert torch.equal(gate, gate_ref)
    assert int((packed_native != packed_ref).sum().item()) <= (
        packed_native.numel() // 100)
    assert int((sfa_native != sfa_ref).sum().item()) <= (
        sfa_native.numel() // 1000)
    assert torch.equal(gate_native, gate_ref)

    residual = torch.randn_like(x) * 0.15
    residual_ref = residual.clone()
    residual_native = residual.clone()
    delta = torch.randn_like(x) * 0.1
    prev_gate = torch.randn_like(x) * 0.1
    packed.zero_()
    packed_ref.zero_()
    sfa.zero_()
    sfa_ref.zero_()
    sfa_native.zero_()

    fvk_fp4.pi05_gate_res_adarms_fp4_sfa_fp16(
        delta.data_ptr(), prev_gate.data_ptr(), residual.data_ptr(),
        style.data_ptr(), packed.data_ptr(), sfa.data_ptr(), gate.data_ptr(),
        rows, dim, 0)
    fvk.gate_res_fp16(
        delta.data_ptr(), prev_gate.data_ptr(), residual_ref.data_ptr(),
        rows * dim, 0)
    fvk.adarms_fp16(
        residual_ref.data_ptr(), style.data_ptr(), normed.data_ptr(),
        gate_ref.data_ptr(), rows, dim, 0)
    assert fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        normed.data_ptr(), packed_ref.data_ptr(), sfa_ref.data_ptr(),
        rows, dim, False, 0) == 0
    fvk_fp4.pi05_gate_res_adarms_fp4_sfa_native_fp16(
        delta.data_ptr(), prev_gate.data_ptr(), residual_native.data_ptr(),
        style.data_ptr(), packed_native.data_ptr(), sfa_native.data_ptr(),
        gate_native.data_ptr(), rows, dim, 0)

    weight = torch.randn(
        out_dim, dim, dtype=torch.float16, device='cuda') * 0.03
    weight_fp4 = quant_weight_nvfp4(weight)
    output = torch.empty(
        rows, out_dim, dtype=torch.float16, device='cuda')
    output_ref = torch.empty_like(output)
    output_native = torch.empty_like(output)
    assert fvk_fp4.cutlass_fp4_gemm_variant(
        7, packed.data_ptr(), sfa.data_ptr(),
        weight_fp4['packed'].data_ptr(), weight_fp4['sfb'].data_ptr(),
        output.data_ptr(), rows, out_dim, dim, 1.0, 0.0, 0) == 0
    assert fvk_fp4.cutlass_fp4_gemm_variant(
        7, packed_ref.data_ptr(), sfa_ref.data_ptr(),
        weight_fp4['packed'].data_ptr(), weight_fp4['sfb'].data_ptr(),
        output_ref.data_ptr(), rows, out_dim, dim, 1.0, 0.0, 0) == 0
    assert fvk_fp4.cutlass_fp4_gemm_variant(
        7, packed_native.data_ptr(), sfa_native.data_ptr(),
        weight_fp4['packed'].data_ptr(), weight_fp4['sfb'].data_ptr(),
        output_native.data_ptr(), rows, out_dim, dim, 1.0, 0.0, 0) == 0
    torch.cuda.synchronize()

    assert torch.equal(residual, residual_ref)
    assert torch.equal(gate, gate_ref)
    assert torch.equal(residual_native, residual_ref)
    assert torch.equal(gate_native, gate_ref)
    assert int((packed_native != packed_ref).sum().item()) <= (
        packed_native.numel() // 100)
    assert int((sfa_native != sfa_ref).sum().item()) <= (
        sfa_native.numel() // 1000)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), output_ref.float().flatten(), dim=0).item()
    assert cosine >= 0.9999, cosine
    native_cosine = torch.nn.functional.cosine_similarity(
        output_native.float().flatten(), output_ref.float().flatten(),
        dim=0).item()
    assert native_cosine >= 0.9999, native_cosine


if __name__ == "__main__":
    test_pi05_decoder_fp4_preprocess_matches_explicit_path()
    print("Pi0.5 decoder FP4 preprocess kernel test passed")
