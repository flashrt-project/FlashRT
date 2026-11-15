"""Weight-only quantized few-row projections against the BF16 lowering of the dequantized weights."""

from pathlib import Path
import platform

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.fused import FusedPlan, quantize_columns
from flash_rt.hardware.apple_native.plan import Weight

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or not torch.backends.mps.is_available()
    or not (
        Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib"
    ).is_file(),
    reason="Built native plan library and Apple host GPU required",
)


def dequantize(weight, bits, *, bf16=True):
    """Dequantized [K, N] weight; bf16=False keeps the FP32 values the exact-row FMA kernels use."""
    codes, scales, biases = quantize_columns(weight, bits)
    if bits == 8:
        w = torch.from_numpy(codes.astype(np.float32)) * torch.from_numpy(scales)[:, None]
    else:
        q = torch.from_numpy(codes)
        q = torch.stack((q & 15, q >> 4), -1).reshape(q.shape[0], -1).float()
        s = torch.from_numpy(scales.view(np.int16)).view(torch.bfloat16).float().repeat_interleave(64, 1)
        b = torch.from_numpy(biases.view(np.int16)).view(torch.bfloat16).float().repeat_interleave(64, 1)
        w = torch.addcmul(b, s, q)
    w = w.T.contiguous()
    return w.bfloat16().float() if bf16 else w


def bf16(x):
    return torch.as_tensor(x, dtype=torch.float32).bfloat16().float()


def run(plan, record, inputs):
    plan.stage = "decode"
    tensors = {k: plan.allocate(v.shape) for k, v in inputs.items()}
    outputs = record(plan, tensors)
    plan.finish()
    plan.set_resident(True)
    results = []
    for mode in ("direct", "icb"):
        for k, v in inputs.items():
            plan.write(tensors[k], v)
        plan.run(mode)
        results.append([plan.read(t) for t in outputs])
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    return results[0]


def within_rounding(actual, expected, scale):
    # FP32 accumulation order versus the FP64 reference may flip a BF16 rounding of an intermediate
    # of magnitude `scale` (the rounded GEMM output), which survives cancellation in the final value.
    ulp = (np.abs(expected) + np.abs(scale)) * 2.0 ** -7 + 1e-6
    diff = np.abs(actual - expected)
    assert (diff <= 2 * ulp).all(), float((diff / ulp).max())
    assert (diff > 0).mean() < 0.05


# rows=10 selects the exact-row FMA kernels (FP32 dequantized weights); other row counts the
# TensorOps tiles (weights dequantized to BF16).
@pytest.mark.parametrize("rows", [10, 12])
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("k", [1024, 4096])
def test_quantized_projection_residual(rows, bits, k):
    torch.manual_seed(bits + k + rows)
    d = 1024
    weight = (torch.randn(k, d) / k ** 0.5).bfloat16()
    inputs = dict(x=torch.randn(rows, k).bfloat16(), res=torch.randn(rows, d).bfloat16())

    def record(p, t):
        out = p.allocate((rows, d))
        p.linear(out, t["x"], Weight(weight, "w"))
        p.ops.residual(t["res"], out, t["res"])
        return [t["res"]]

    plan = FusedPlan(decode_weights={8: "int8", 4: "int4"}[bits])
    try:
        (actual,) = run(plan, record, inputs)
        assert plan.weight_bytes < weight.numel() * (bits / 8 + 0.1)
    finally:
        plan.close()
    w = dequantize(weight, bits, bf16=rows != 10).double()
    product = bf16((inputs["x"].double() @ w).float())
    expected = bf16(inputs["res"].float() + product)
    within_rounding(actual, expected.numpy(), product.numpy())


@pytest.mark.parametrize("rows", [10, 12])
@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_gate_up_gelu_mul(rows, bits):
    torch.manual_seed(31 + bits + rows)
    k, h = 1024, 4096
    weight = (torch.randn(k, 2 * h) / k ** 0.5).bfloat16()
    inputs = dict(x=torch.randn(rows, k).bfloat16(), hidden=torch.zeros(rows, h).bfloat16())

    def record(p, t):
        merged = p.allocate((rows, 2 * h))
        p.linear(merged, t["x"], Weight(weight, "w"))
        p.ops.gelu_mul_merged(t["hidden"], merged)
        return [t["hidden"]]

    plan = FusedPlan(decode_weights={8: "int8", 4: "int4"}[bits])
    try:
        (actual,) = run(plan, record, inputs)
        if rows == 10:
            w = dequantize(weight, bits, bf16=False)
        else:
            # The TensorOps path quantizes after packing gate/up column blocks; unpack again.
            packed = dequantize(plan._packed_gelu_mul(Weight(weight, "w"), 32).tensor, bits)
            blocks = packed.reshape(k, -1, 2, 32)
            w = torch.cat((blocks[:, :, 0].reshape(k, h), blocks[:, :, 1].reshape(k, h)), 1)
    finally:
        plan.close()
    y = bf16((inputs["x"].double() @ w.double()).float())
    g, u = y[:, :h], y[:, h:]
    expected = bf16(0.5 * g * (1 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g ** 3))) * u)
    within_rounding(actual, expected.numpy(), (g.abs() * u.abs()).numpy())


def test_int4_quantization_roundtrip_error_is_bounded():
    torch.manual_seed(2)
    weight = torch.randn(256, 96).bfloat16()
    error = (dequantize(weight, 4).float() - weight.float()).abs().T.reshape(96, -1, 64)
    span = weight.float().T.reshape(96, -1, 64)
    step = ((span.amax(-1) - span.amin(-1)) / 15)[..., None]
    # Half a code step, plus BF16 rounding of the scale (up to 15 steps away) and of the value.
    bound = step * (0.5 + 16 * 2.0 ** -8) + span.abs() * 2.0 ** -8 + 1e-3
    assert (error <= bound).all()
