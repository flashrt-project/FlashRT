"""Fused dynamic-FP8 quantize kernels: bitwise equality vs unfused paths.

The Chameleon Thor pipeline replaced three two-kernel sequences with fused
kernels that fold the amax measurement into the producer's write pass:

- rms_norm_quantize_dynamic_fp8_fp16            == rms_norm_fp16 + quantize_fp8_device_fp16
- gate_geglu_quantize_dynamic_fp8_fp16          == gate_geglu_fp16 + quantize_fp8_device_fp16
- residual_add_rms_norm_quantize_dynamic_fp8_fp16
    == residual_add_fp16 + rms_norm_quantize_dynamic_fp8_fp16

Per CONTRIBUTING.md ("Validate fused replacements against unfused reference
paths"), these tests assert the fused kernels produce **bit-identical**
outputs (fp16 intermediate, fp8 quantized output, and scale) to the unfused
reference. No model checkpoint required.
"""

from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk

fp16 = torch.float16
fp8 = torch.uint8  # storage dtype of __nv_fp8_e4m3 buffers in Python


def _alloc(shape, dtype=fp16):
    return torch.zeros(shape, dtype=dtype, device="cuda")


def _same_scale(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.cpu(), b.cpu()))


def test_rms_norm_quantize_dynamic_fp8_matches_unfused():
    S, D = 64, 1024
    x = (torch.randn(S, D, dtype=fp16, device="cuda") * 3.0)
    w = (torch.randn(D, dtype=fp16, device="cuda") + 1.0)

    # Fused
    xn_f, fp8_f, scale_f = _alloc((S, D)), _alloc((S, D), fp8), _alloc((1,), torch.float32)
    fvk.rms_norm_quantize_dynamic_fp8_fp16(
        x.data_ptr(), w.data_ptr(), xn_f.data_ptr(), fp8_f.data_ptr(),
        scale_f.data_ptr(), S, D, 1e-5, 0)

    # Unfused: rms_norm_fp16 + quantize_fp8_device_fp16
    xn_u, fp8_u, scale_u = _alloc((S, D)), _alloc((S, D), fp8), _alloc((1,), torch.float32)
    fvk.rms_norm_fp16(x.data_ptr(), w.data_ptr(), xn_u.data_ptr(), S, D, 1e-5, 0)
    fvk.quantize_fp8_device_fp16(
        xn_u.data_ptr(), fp8_u.data_ptr(), scale_u.data_ptr(), S * D, 0)
    torch.cuda.synchronize()

    assert torch.equal(xn_f, xn_u), "fused xn differs from unfused rms_norm"
    assert torch.equal(fp8_f, fp8_u), "fused fp8 output differs from unfused quantize"
    assert _same_scale(scale_f, scale_u), "fused scale differs from unfused amax path"


def test_gate_geglu_quantize_dynamic_fp8_matches_unfused():
    n = 64 * 4096  # SwiGLU intermediate (Se * Dff)
    gate = (torch.randn(n, dtype=fp16, device="cuda") * 0.5)
    up = (torch.randn(n, dtype=fp16, device="cuda") * 0.5)

    h_f, fp8_f, scale_f = _alloc((n,)), _alloc((n,), fp8), _alloc((1,), torch.float32)
    fvk.gate_geglu_quantize_dynamic_fp8_fp16(
        gate.data_ptr(), up.data_ptr(), h_f.data_ptr(), fp8_f.data_ptr(),
        scale_f.data_ptr(), n, 0)

    h_u, fp8_u, scale_u = _alloc((n,)), _alloc((n,), fp8), _alloc((1,), torch.float32)
    fvk.gate_geglu_fp16(gate.data_ptr(), up.data_ptr(), h_u.data_ptr(), n, 0)
    fvk.quantize_fp8_device_fp16(
        h_u.data_ptr(), fp8_u.data_ptr(), scale_u.data_ptr(), n, 0)
    torch.cuda.synchronize()

    assert torch.equal(h_f, h_u), "fused SwiGLU output differs from gate_geglu_fp16"
    assert torch.equal(fp8_f, fp8_u), "fused fp8 output differs from unfused quantize"
    assert _same_scale(scale_f, scale_u), "fused scale differs from unfused amax path"


def test_residual_add_rms_norm_quantize_dynamic_fp8_matches_unfused():
    S, D = 64, 1024
    x = torch.randn(S, D, dtype=fp16, device="cuda") * 3.0
    o = torch.randn(S, D, dtype=fp16, device="cuda") * 0.5
    w = torch.randn(D, dtype=fp16, device="cuda") + 1.0

    # Fused
    x_f = x.clone()
    xn_f, fp8_f, scale_f = _alloc((S, D)), _alloc((S, D), fp8), _alloc((1,), torch.float32)
    fvk.residual_add_rms_norm_quantize_dynamic_fp8_fp16(
        x_f.data_ptr(), o.data_ptr(), w.data_ptr(), xn_f.data_ptr(),
        fp8_f.data_ptr(), scale_f.data_ptr(), S, D, 1e-5, 0)

    # Unfused: residual_add_fp16 + rms_norm_quantize_dynamic_fp8_fp16
    x_u = x.clone()
    xn_u, fp8_u, scale_u = _alloc((S, D)), _alloc((S, D), fp8), _alloc((1,), torch.float32)
    fvk.residual_add_fp16(x_u.data_ptr(), o.data_ptr(), S * D, 0)
    fvk.rms_norm_quantize_dynamic_fp8_fp16(
        x_u.data_ptr(), w.data_ptr(), xn_u.data_ptr(), fp8_u.data_ptr(),
        scale_u.data_ptr(), S, D, 1e-5, 0)
    torch.cuda.synchronize()

    assert torch.equal(x_f, x_u), "fused residual differs from residual_add_fp16"
    assert torch.equal(xn_f, xn_u), "fused xn differs from unfused norm"
    assert torch.equal(fp8_f, fp8_u), "fused fp8 output differs from unfused quantize"
    assert _same_scale(scale_f, scale_u), "fused scale differs from unfused amax path"
