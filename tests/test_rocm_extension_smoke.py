import pytest
import torch

from flash_rt import flash_rt_rocm_kernels as rocm


_FP8_HIPBLASLT_ALGO_AVAILABLE = None


def _requires_fp8_hipblaslt_algo():
    global _FP8_HIPBLASLT_ALGO_AVAILABLE
    if _FP8_HIPBLASLT_ALGO_AVAILABLE is None:
        a = torch.randn(16, 32, device="cuda", dtype=torch.float32) * 0.5
        b = torch.randn(32, 16, device="cuda", dtype=torch.float32) * 0.5
        a_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        b_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        a8 = (a / a_scale).to(torch.float8_e4m3fnuz)
        b8 = (b / b_scale).to(torch.float8_e4m3fnuz)
        try:
            rocm.hipblaslt_matmul_fp8_e4m3fnuz_bf16(a8, b8, a_scale, b_scale)
            rocm.hip_sync()
        except RuntimeError as exc:
            if "hipBLASLt did not return a usable" not in str(exc):
                raise
            _FP8_HIPBLASLT_ALGO_AVAILABLE = False
        else:
            _FP8_HIPBLASLT_ALGO_AVAILABLE = True

    if not _FP8_HIPBLASLT_ALGO_AVAILABLE:
        pytest.skip("hipBLASLt FP8 GEMM algorithm is unavailable on this ROCm runtime")


def _assert_fp8_codes_adjacent(actual, expected, *, max_mismatch_ratio=0.03):
    equal_value = actual.float() == expected.float()
    code_delta = (
        actual.view(torch.uint8).to(torch.int16)
        - expected.view(torch.uint8).to(torch.int16)
    ).abs()
    code_delta = torch.where(equal_value, torch.zeros_like(code_delta), code_delta)
    assert int(code_delta.max().item()) <= 1
    assert float((code_delta != 0).float().mean().item()) < max_mismatch_ratio


def test_rocm_extension_smoke():
    assert rocm.has_rocm()
    assert rocm.device_count() >= 1
    assert rocm.hipblaslt_available()
    assert isinstance(rocm.hipblaslt_algo_cache_size(), int)
    assert isinstance(rocm.hipblaslt_algo_cache_keys(), list)
    assert isinstance(rocm.hipblaslt_linear_plan_cache_size(), int)
    assert isinstance(rocm.hipblaslt_linear_plan_cache_keys(), list)

    hipblaslt = rocm.hipblaslt_probe()
    assert hipblaslt["available"], hipblaslt
    assert hipblaslt["status_name"] == "HIPBLAS_STATUS_SUCCESS"
    assert hipblaslt["version"] > 0

    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)
    out = rocm.vector_add_f32(a, b)
    rocm.hip_sync()

    torch.testing.assert_close(out, a + b)


def test_rocm_rms_norm_float32():
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float32)
    w = torch.randn(2048, device="cuda", dtype=torch.float32) * 0.01

    out = rocm.rms_norm(x, w, 1e-6)
    rocm.hip_sync()

    ref = x * torch.rsqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6)
    ref = ref * (1.0 + w.float())
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_rocm_layer_norm_bfloat16_raw_pointer():
    rows = 5
    hidden = 1152
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    out = torch.empty_like(x)

    rocm.layer_norm_bf16_ptr(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.hip_sync()

    ref = torch.nn.functional.layer_norm(
        x.float(), (hidden,), weight.float(), bias.float(), 1e-5
    ).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.02)


def test_rocm_add_bias_bfloat16_raw_pointer():
    rows = 7
    hidden = 257
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    before = x.clone()

    rocm.add_bias_bf16_ptr(x.data_ptr(), bias.data_ptr(), rows, hidden)
    rocm.hip_sync()

    ref = (before.float() + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(x.float(), ref.float(), rtol=0, atol=0.0078125)


def test_rocm_bias_residual_bfloat16_raw_pointer():
    rows = 7
    hidden = 257
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    before = residual.clone()

    rocm.bias_residual_bf16_ptr(
        residual.data_ptr(), x.data_ptr(), bias.data_ptr(), rows, hidden
    )
    rocm.hip_sync()

    ref = (before.float() + x.float() + bias.float()).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), ref.float(), rtol=0, atol=0.0078125)


def test_rocm_residual_add_bfloat16_raw_pointer():
    residual = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    x = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    before = residual.clone()

    rocm.residual_add_bf16_ptr(residual.data_ptr(), x.data_ptr(), residual.numel())
    rocm.hip_sync()

    ref = (before.float() + x.float()).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), ref.float(), rtol=0, atol=0.0078125)


def test_rocm_gate_mul_residual_bfloat16_raw_pointer():
    residual = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    x = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    before = residual.clone()

    rocm.gate_mul_residual_bf16_ptr(
        residual.data_ptr(), x.data_ptr(), gate.data_ptr(), residual.numel()
    )
    rocm.hip_sync()

    ref = (before.float() + x.float() * gate.float()).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), ref.float(), rtol=0, atol=0.0078125)


def test_rocm_residual_add_rms_norm_bfloat16_raw_pointer():
    rows = 5
    hidden = 2048
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = (torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    out = torch.empty_like(residual)
    before = residual.clone()

    rocm.residual_add_rms_norm_bf16_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float()).to(torch.bfloat16)
    ref = updated.float() * torch.rsqrt(
        torch.mean(torch.square(updated.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * (1.0 + weight.float())).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.02)


def test_rocm_residual_add_rms_norm_fp8_e4m3fnuz_raw_pointer():
    rows = 5
    hidden = 2048
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = (torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(residual, dtype=torch.float8_e4m3fnuz)
    before = residual.clone()

    rocm.residual_add_rms_norm_fp8_e4m3fnuz_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        scale.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float()).to(torch.bfloat16)
    ref = updated.float() * torch.rsqrt(
        torch.mean(torch.square(updated.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * (1.0 + weight.float())).to(torch.bfloat16)
    ref = (ref.float() / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0)
    _assert_fp8_codes_adjacent(out, ref)


def test_rocm_ada_rms_norm_style_bfloat16_raw_pointer():
    rows = 5
    hidden = 1024
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    style = (torch.randn(rows, 3 * hidden, device="cuda", dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    out = torch.empty_like(x)
    gate = torch.empty_like(x)

    rocm.ada_rms_norm_style_bf16_ptr(
        x.data_ptr(),
        weight.data_ptr(),
        style.data_ptr(),
        out.data_ptr(),
        gate.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    scale, shift, gate_ref = style.float().chunk(3, dim=-1)
    ref = x.float() * torch.rsqrt(
        torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * (1.0 + scale) + shift).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.02)
    torch.testing.assert_close(gate.float(), gate_ref.float(), rtol=0, atol=0)


def test_rocm_ada_rms_norm_style_fp8_e4m3fnuz_raw_pointer():
    rows = 5
    hidden = 1024
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    style = torch.randn(rows, 3 * hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    scale_out = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    gate = torch.empty_like(x)

    rocm.ada_rms_norm_style_fp8_e4m3fnuz_ptr(
        x.data_ptr(),
        weight.data_ptr(),
        style.data_ptr(),
        out.data_ptr(),
        gate.data_ptr(),
        scale_out.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    scale, shift, gate_ref = style.float().chunk(3, dim=-1)
    ref = x.float() * torch.rsqrt(
        torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * (1.0 + scale) + shift).to(torch.bfloat16)
    ref = (ref.float() / scale_out).to(torch.float8_e4m3fnuz)
    _assert_fp8_codes_adjacent(out, ref)
    torch.testing.assert_close(gate.float(), gate_ref.float(), rtol=0, atol=0)


def test_rocm_gate_residual_ada_norm_bfloat16_raw_pointer():
    rows = 5
    hidden = 1024
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    gate_in = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    style = torch.randn(rows, 3 * hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    out = torch.empty_like(residual)
    gate_out = gate_in.clone()
    before = residual.clone()

    rocm.gate_residual_ada_norm_bf16_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        gate_out.data_ptr(),
        weight.data_ptr(),
        style.data_ptr(),
        out.data_ptr(),
        gate_out.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float() * gate_in.float()).to(torch.bfloat16)
    scale, shift, gate_ref = style.float().chunk(3, dim=-1)
    ref = updated.float() * torch.rsqrt(
        torch.mean(torch.square(updated.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * weight.float() * (1.0 + scale) + shift).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0.02)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.02)
    torch.testing.assert_close(gate_out.float(), gate_ref.float(), rtol=0, atol=0)


def test_rocm_gate_residual_ada_norm_fp8_e4m3fnuz_raw_pointer():
    rows = 5
    hidden = 1024
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    gate_in = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    weight = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    style = (torch.randn(rows, 3 * hidden, device="cuda", dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    scale_out = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(residual, dtype=torch.float8_e4m3fnuz)
    gate_out = gate_in.clone()
    before = residual.clone()

    rocm.gate_residual_ada_norm_fp8_e4m3fnuz_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        gate_out.data_ptr(),
        weight.data_ptr(),
        style.data_ptr(),
        out.data_ptr(),
        gate_out.data_ptr(),
        scale_out.data_ptr(),
        rows,
        hidden,
        1e-6,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float() * gate_in.float()).to(torch.bfloat16)
    scale, shift, gate_ref = style.float().chunk(3, dim=-1)
    ref = updated.float() * torch.rsqrt(
        torch.mean(torch.square(updated.float()), dim=-1, keepdim=True) + 1e-6
    )
    ref = (ref * (1.0 + scale) + shift).to(torch.bfloat16)
    ref = (ref.float() / scale_out).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0.02)
    _assert_fp8_codes_adjacent(out, ref)
    torch.testing.assert_close(gate_out.float(), gate_ref.float(), rtol=0, atol=0)


def test_rocm_bias_residual_layer_norm_bfloat16_raw_pointer():
    rows = 5
    hidden = 1152
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias_pre = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    norm_bias = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    out = torch.empty_like(residual)
    before = residual.clone()

    rocm.bias_residual_layer_norm_bf16_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        bias_pre.data_ptr(),
        norm_weight.data_ptr(),
        norm_bias.data_ptr(),
        out.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float() + bias_pre.float()).to(torch.bfloat16)
    ref = torch.nn.functional.layer_norm(
        updated.float(), (hidden,), norm_weight.float(), norm_bias.float(), 1e-5
    ).to(torch.bfloat16)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0.0078125)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.02)


def test_rocm_bias_residual_layer_norm_fp8_e4m3fnuz_raw_pointer():
    rows = 5
    hidden = 1152
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias_pre = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    norm_bias = torch.randn(hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(residual, dtype=torch.float8_e4m3fnuz)
    before = residual.clone()

    rocm.bias_residual_layer_norm_fp8_e4m3fnuz_ptr(
        residual.data_ptr(),
        x.data_ptr(),
        bias_pre.data_ptr(),
        norm_weight.data_ptr(),
        norm_bias.data_ptr(),
        out.data_ptr(),
        scale.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.hip_sync()

    updated = (before.float() + x.float() + bias_pre.float()).to(torch.bfloat16)
    ref = torch.nn.functional.layer_norm(
        updated.float(), (hidden,), norm_weight.float(), norm_bias.float(), 1e-5
    )
    ref = (ref / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(residual.float(), updated.float(), rtol=0, atol=0.0078125)
    _assert_fp8_codes_adjacent(out, ref)


def test_rocm_qkv_split_bfloat16_raw_pointer():
    rows = 9
    q_dim = 16
    k_dim = 8
    v_dim = 12
    qkv = torch.arange(
        rows * (q_dim + k_dim + v_dim),
        device="cuda",
        dtype=torch.float32,
    ).reshape(rows, q_dim + k_dim + v_dim).to(torch.bfloat16)
    q = torch.empty((rows, q_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((rows, k_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.empty((rows, v_dim), device="cuda", dtype=torch.bfloat16)

    rocm.qkv_split_bf16_ptr(
        qkv.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        rows,
        q_dim,
        k_dim,
        v_dim,
    )
    rocm.hip_sync()

    torch.testing.assert_close(q, qkv[:, :q_dim], rtol=0, atol=0)
    torch.testing.assert_close(k, qkv[:, q_dim : q_dim + k_dim], rtol=0, atol=0)
    torch.testing.assert_close(v, qkv[:, q_dim + k_dim :], rtol=0, atol=0)


def test_rocm_hipblaslt_matmul_bfloat16():
    a = torch.randn(64, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    torch.cuda.synchronize()
    out = rocm.hipblaslt_matmul_bf16(a, b)
    rocm.hip_sync()

    ref = torch.matmul(a, b)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_rocm_hipblaslt_matmul_fp8_e4m3fnuz_bfloat16():
    _requires_fp8_hipblaslt_algo()

    a = torch.randn(64, 128, device="cuda", dtype=torch.float32) * 0.5
    b = torch.randn(128, 96, device="cuda", dtype=torch.float32) * 0.5
    a_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    b_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    a8 = (a / a_scale).to(torch.float8_e4m3fnuz)
    b8 = (b / b_scale).to(torch.float8_e4m3fnuz)

    torch.cuda.synchronize()
    out = rocm.hipblaslt_matmul_fp8_e4m3fnuz_bf16(a8, b8, a_scale, b_scale)
    rocm.hip_sync()

    ref = torch.matmul(a8.float() * a_scale, b8.float() * b_scale).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=6e-2, atol=3e-1)


def test_rocm_hipblaslt_linear_bfloat16_with_bias():
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out = rocm.hipblaslt_linear_bf16(x, weight, bias)
    rocm.hip_sync()

    ref = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_rocm_hipblaslt_linear_bfloat16_out_with_bias():
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty((2, 17, 96), device="cuda", dtype=torch.bfloat16)

    rocm.hipblaslt_linear_bf16_out(x, weight, out, bias)
    rocm.hip_sync()

    ref = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_rocm_hipblaslt_linear_bfloat16_raw_pointers_with_bias():
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty((34, 96), device="cuda", dtype=torch.bfloat16)

    rocm.hipblaslt_linear_bf16_ptr(
        x.contiguous().reshape(-1, 128).data_ptr(),
        weight.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        out.data_ptr(),
        34,
        96,
        128,
    )
    rocm.hip_sync()

    ref = torch.nn.functional.linear(x, weight, bias).reshape(-1, 96)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_rocm_hipblaslt_linear_bfloat16_no_bias():
    x = torch.randn(33, 64, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    weight = torch.randn(48, 64, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out = rocm.hipblaslt_linear_bf16(x, weight)
    rocm.hip_sync()

    ref = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_rocm_hipblaslt_linear_fp8_e4m3fnuz_bfloat16_with_bias():
    _requires_fp8_hipblaslt_algo()

    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32) * 0.5
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32) * 0.5
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    x8 = (x / x_scale).to(torch.float8_e4m3fnuz)
    weight8 = (weight / weight_scale).to(torch.float8_e4m3fnuz)

    out = rocm.hipblaslt_linear_fp8_e4m3fnuz_bf16(
        x8, weight8, x_scale, weight_scale, bias
    )
    rocm.hip_sync()

    ref = torch.nn.functional.linear(
        x8.float() * x_scale,
        weight8.float() * weight_scale,
        bias.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=6e-2, atol=3e-1)


def test_rocm_hipblaslt_linear_fp8_e4m3fnuz_bfloat16_out_with_bias():
    _requires_fp8_hipblaslt_algo()

    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.float32) * 0.5
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32) * 0.5
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    x8 = (x / x_scale).to(torch.float8_e4m3fnuz)
    weight8 = (weight / weight_scale).to(torch.float8_e4m3fnuz)
    out = torch.empty((2, 17, 96), device="cuda", dtype=torch.bfloat16)

    rocm.hipblaslt_linear_fp8_e4m3fnuz_bf16_out(
        x8,
        weight8,
        x_scale,
        weight_scale,
        out,
        bias,
    )
    rocm.hip_sync()

    ref = torch.nn.functional.linear(
        x8.float() * x_scale,
        weight8.float() * weight_scale,
        bias.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=6e-2, atol=3e-1)


def test_rocm_hipblaslt_linear_fp8_e4m3fnuz_bfloat16_raw_pointers():
    _requires_fp8_hipblaslt_algo()

    x = torch.randn(17, 128, device="cuda", dtype=torch.float32) * 0.5
    weight = torch.randn(96, 128, device="cuda", dtype=torch.float32) * 0.5
    bias = torch.randn(96, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    x8 = (x / x_scale).to(torch.float8_e4m3fnuz).contiguous()
    weight8 = (weight / weight_scale).to(torch.float8_e4m3fnuz).contiguous()
    out = torch.empty((17, 96), device="cuda", dtype=torch.bfloat16)

    rocm.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
        x8.data_ptr(),
        weight8.data_ptr(),
        x_scale.data_ptr(),
        weight_scale.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        17,
        96,
        128,
    )
    rocm.hip_sync()

    ref = torch.nn.functional.linear(
        x8.float() * x_scale,
        weight8.float() * weight_scale,
        bias.float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=6e-2, atol=3e-1)


def test_rocm_quantize_to_fp8_e4m3fnuz_bfloat16():
    x = torch.randn(257, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)

    out = rocm.quantize_to_fp8_e4m3fnuz(x, scale)
    rocm.hip_sync()

    ref = (x / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


def test_rocm_quantize_to_fp8_e4m3fnuz_float32():
    x = torch.randn(257, device="cuda", dtype=torch.float32)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)

    out = rocm.quantize_to_fp8_e4m3fnuz(x, scale)
    rocm.hip_sync()

    ref = (x / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


def test_rocm_quantize_to_fp8_e4m3fnuz_out_reuses_buffer():
    x = torch.randn(257, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)

    rocm.quantize_to_fp8_e4m3fnuz_out(x, scale, out)
    rocm.hip_sync()

    ref = (x / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


def test_rocm_quantize_bf16_to_fp8_e4m3fnuz_raw_pointer():
    x = torch.randn(257, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)

    rocm.quantize_bf16_to_fp8_e4m3fnuz_ptr(
        x.data_ptr(), scale.data_ptr(), out.data_ptr(), x.numel()
    )
    rocm.hip_sync()

    ref = (x / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


def test_rocm_dynamic_quantize_to_fp8_e4m3fnuz_bfloat16():
    x = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out, scale = rocm.dynamic_quantize_to_fp8_e4m3fnuz(x)
    rocm.hip_sync()

    ref_scale = torch.clamp(x.float().abs().max() / 240.0, min=1.0e-8).reshape(1)
    ref = (x / ref_scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-8)
    _assert_fp8_codes_adjacent(out, ref)


def test_rocm_dynamic_quantize_bf16_to_fp8_e4m3fnuz_raw_pointer():
    x = torch.randn(1025, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scale = torch.empty((1,), device="cuda", dtype=torch.float32)
    partial = torch.empty((5,), device="cuda", dtype=torch.float32)

    rocm.dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr(
        x.data_ptr(),
        out.data_ptr(),
        scale.data_ptr(),
        partial.data_ptr(),
        partial.numel(),
        x.numel(),
    )
    rocm.hip_sync()

    ref_scale = torch.clamp(x.float().abs().max() / 240.0, min=1.0e-8).reshape(1)
    ref = (x / ref_scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-8)
    _assert_fp8_codes_adjacent(out, ref)


def test_rocm_dynamic_quantize_to_fp8_e4m3fnuz_float32():
    x = torch.randn(1025, device="cuda", dtype=torch.float32)

    out, scale = rocm.dynamic_quantize_to_fp8_e4m3fnuz(x)
    rocm.hip_sync()

    ref_scale = torch.clamp(x.abs().max() / 240.0, min=1.0e-8).reshape(1)
    ref = (x / ref_scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


def test_rocm_gelu_tanh_mul_bfloat16():
    gate = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out = rocm.gelu_tanh_mul_bf16(gate, up)
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=0, atol=0)


def test_rocm_gelu_tanh_mul_bfloat16_out():
    gate = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty_like(gate)

    rocm.gelu_tanh_mul_bf16_out(gate, up, out)
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=0, atol=0)


def test_rocm_gelu_tanh_mul_bfloat16_raw_pointer():
    gate = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty_like(gate)

    rocm.gelu_tanh_mul_bf16_ptr(
        gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.numel()
    )
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    torch.testing.assert_close(out, ref.to(torch.bfloat16), rtol=0, atol=0)


def test_rocm_gelu_tanh_mul_quantize_fp8_e4m3fnuz_out():
    gate = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(gate, dtype=torch.float8_e4m3fnuz)

    rocm.gelu_tanh_mul_quantize_fp8_e4m3fnuz_out(gate, up, scale, out)
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    ref = (ref / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(
        out.float() * scale,
        ref.float() * scale,
        rtol=0,
        atol=0.03,
    )


def test_rocm_gelu_tanh_mul_quantize_fp8_e4m3fnuz_raw_pointer():
    gate = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    up = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(gate, dtype=torch.float8_e4m3fnuz)

    rocm.gelu_tanh_mul_quantize_fp8_e4m3fnuz_ptr(
        gate.data_ptr(), up.data_ptr(), scale.data_ptr(), out.data_ptr(), gate.numel()
    )
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    ref = (ref / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(
        out.float() * scale,
        ref.float() * scale,
        rtol=0,
        atol=0.03,
    )


def test_rocm_gelu_tanh_quantize_fp8_e4m3fnuz_raw_pointer():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)

    rocm.gelu_tanh_quantize_fp8_e4m3fnuz_ptr(
        x.data_ptr(), scale.data_ptr(), out.data_ptr(), x.numel()
    )
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(torch.bfloat16)
    ref = (ref.float() / scale).to(torch.float8_e4m3fnuz)
    torch.testing.assert_close(
        out.float() * scale,
        ref.float() * scale,
        rtol=0,
        atol=0.03,
    )


def test_rocm_layer_norm_fp8_matches_bf16_boundary():
    rows = 13
    hidden = 1152
    x = (torch.randn(rows, hidden, device="cuda", dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    weight = (1.0 + torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    bias = (torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    out_bf16 = torch.empty_like(x)
    ref_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    out_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)

    rocm.layer_norm_bf16_ptr(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out_bf16.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.quantize_bf16_to_fp8_e4m3fnuz_ptr(
        out_bf16.data_ptr(),
        scale.data_ptr(),
        ref_fp8.data_ptr(),
        out_bf16.numel(),
    )
    rocm.layer_norm_fp8_e4m3fnuz_ptr(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out_fp8.data_ptr(),
        scale.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.hip_sync()

    _assert_fp8_codes_adjacent(out_fp8, ref_fp8)


def test_rocm_bias_residual_layer_norm_fp8_matches_bf16_boundary():
    rows = 11
    hidden = 1152
    residual = (torch.randn(rows, hidden, device="cuda", dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    x = (torch.randn(rows, hidden, device="cuda", dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    bias = (torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    weight = (1.0 + torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    norm_bias = (torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)

    residual_bf16 = residual.clone()
    residual_fp8 = residual.clone()
    out_bf16 = torch.empty_like(residual)
    ref_fp8 = torch.empty_like(residual, dtype=torch.float8_e4m3fnuz)
    out_fp8 = torch.empty_like(residual, dtype=torch.float8_e4m3fnuz)

    rocm.bias_residual_layer_norm_bf16_ptr(
        residual_bf16.data_ptr(),
        x.data_ptr(),
        bias.data_ptr(),
        weight.data_ptr(),
        norm_bias.data_ptr(),
        out_bf16.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.quantize_bf16_to_fp8_e4m3fnuz_ptr(
        out_bf16.data_ptr(),
        scale.data_ptr(),
        ref_fp8.data_ptr(),
        out_bf16.numel(),
    )
    rocm.bias_residual_layer_norm_fp8_e4m3fnuz_ptr(
        residual_fp8.data_ptr(),
        x.data_ptr(),
        bias.data_ptr(),
        weight.data_ptr(),
        norm_bias.data_ptr(),
        out_fp8.data_ptr(),
        scale.data_ptr(),
        rows,
        hidden,
        1e-5,
    )
    rocm.hip_sync()

    torch.testing.assert_close(residual_fp8, residual_bf16, rtol=0, atol=0)
    _assert_fp8_codes_adjacent(out_fp8, ref_fp8)


def test_rocm_gelu_tanh_merged_bfloat16_raw_pointer():
    rows = 7
    hidden = 128
    gate = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    up = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    gate_up = torch.cat((gate, up), dim=-1).contiguous()
    out = torch.empty_like(gate)

    rocm.gelu_tanh_merged_bf16_ptr(
        gate_up.data_ptr(), out.data_ptr(), rows, hidden
    )
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    torch.testing.assert_close(out.float(), ref.to(torch.bfloat16).float(), rtol=0, atol=0)


def test_rocm_gelu_tanh_merged_quantize_fp8_e4m3fnuz_raw_pointer():
    rows = 7
    hidden = 128
    gate = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    up = torch.randn(rows, hidden, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    gate_up = torch.cat((gate, up), dim=-1).contiguous()
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    out = torch.empty_like(gate, dtype=torch.float8_e4m3fnuz)

    rocm.gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
        gate_up.data_ptr(), scale.data_ptr(), out.data_ptr(), rows, hidden
    )
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()
    ref = ref.to(torch.bfloat16)
    ref = (ref.float() / scale).to(torch.float8_e4m3fnuz)
    _assert_fp8_codes_adjacent(out, ref)


def test_rocm_gelu_tanh_bfloat16():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out = rocm.gelu_tanh_bf16(x)
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(x.float(), approximate="tanh").to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0.0078125)


def test_rocm_gelu_tanh_bfloat16_raw_pointer_inplace():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    before = x.clone()

    rocm.gelu_tanh_bf16_ptr(x.data_ptr(), x.data_ptr(), x.numel())
    rocm.hip_sync()

    ref = torch.nn.functional.gelu(before.float(), approximate="tanh").to(
        torch.bfloat16
    )
    torch.testing.assert_close(x, ref, rtol=0, atol=0.0078125)


def test_rocm_silu_bfloat16():
    x = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)

    out = rocm.silu_bf16(x)
    rocm.hip_sync()

    ref = torch.nn.functional.silu(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0.0078125)


def test_rocm_rms_norm_bfloat16():
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(2048, device="cuda", dtype=torch.float32) * 0.01).to(torch.bfloat16)

    out = rocm.rms_norm(x, w, 1e-6)
    rocm.hip_sync()

    ref = x * torch.rsqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6)
    ref = (ref * (1.0 + w.float())).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0.0078125)


def test_rocm_rms_norm_bfloat16_float32_weight():
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(2048, device="cuda", dtype=torch.float32) * 0.01

    out = rocm.rms_norm(x, w, 1e-6)
    rocm.hip_sync()

    ref = x * torch.rsqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6)
    ref = (ref * (1.0 + w.float())).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0.0078125)


def test_rocm_rms_norm_bfloat16_raw_pointer():
    x = torch.randn(6, 128, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(128, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    out = torch.empty_like(x)

    rocm.rms_norm_bf16_ptr(x.data_ptr(), w.data_ptr(), out.data_ptr(), 6, 128, 1e-6)
    rocm.hip_sync()

    ref = x * torch.rsqrt(torch.mean(torch.square(x.float()), dim=-1, keepdim=True) + 1e-6)
    ref = (ref * (1.0 + w.float())).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0.0078125)


def test_rocm_qkv_split_rope_bfloat16_raw_pointer():
    seq, q_dim, k_dim, v_dim, head_dim = 4, 8, 4, 4, 4
    qkv = (
        torch.arange(seq * (q_dim + k_dim + v_dim), device="cuda", dtype=torch.float32)
        .reshape(seq, q_dim + k_dim + v_dim)
        .mul_(0.01)
        .to(torch.bfloat16)
        .contiguous()
    )
    rope = torch.zeros(seq, head_dim, device="cuda", dtype=torch.bfloat16)
    rope[:, 0::2] = 1.0
    q = torch.empty(seq, q_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(seq, k_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(seq, v_dim, device="cuda", dtype=torch.bfloat16)

    rocm.qkv_split_rope_bf16_ptr(
        qkv.data_ptr(), rope.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
        seq, q_dim, k_dim, v_dim, head_dim,
    )
    rocm.hip_sync()

    torch.testing.assert_close(q, qkv[:, :q_dim], rtol=0, atol=0)
    torch.testing.assert_close(k, qkv[:, q_dim : q_dim + k_dim], rtol=0, atol=0)
    torch.testing.assert_close(v, qkv[:, q_dim + k_dim :], rtol=0, atol=0)
