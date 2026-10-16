#!/usr/bin/env python3
"""fp32-weight linear projections for small shapes.

Two tiny projections with 32-bit float weights and a fused bias:

    linear_f32_in_bf16_out   out[t, n] = bf16( sum_k x[t, k] * W[n, k] + b[n] )
    linear_bf16_in_f32_out   out[t, n] = sum_k fp32(x[t, k]) * W[n, k] + b[n]

Both accumulate in fp32 with no TF32. They are separate kernels rather than one
with a cast on each end because the second splits the k loop across a warp and
reduces in registers, a different summation order from the first's sequential
loop. `W` is an nn.Linear weight stored (N, K).

Usage:
  PYTHONPATH=. pytest -q tests/test_linear_f32.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16
# the shapes these serve: a handful of rows against a few hundred columns
SHAPES = [(10, 1024, 32), (10, 32, 1024), (1, 512, 64), (10, 4096, 32)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _inputs(T, N, K, seed=0):
    torch.manual_seed(seed)
    xf = torch.randn(T, K, device="cuda")
    xb = xf.to(BF16)
    W = torch.randn(N, K, device="cuda") * 0.1
    b = torch.randn(N, device="cuda") * 0.1
    return xf, xb, W, b


class TestLinearF32InBf16Out:
    @pytest.mark.parametrize("T,N,K", SHAPES)
    def test_matches_a_fp64_reference(self, T, N, K):
        fvk = _fvk()
        xf, _, W, b = _inputs(T, N, K)
        out = torch.zeros(T, N, device="cuda", dtype=BF16)
        fvk.linear_f32_in_bf16_out(xf.data_ptr(), W.data_ptr(), b.data_ptr(),
                                   out.data_ptr(), T, N, K,
                                   torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref = (xf.double() @ W.double().t() + b.double()).to(BF16)
        frac = (out != ref).float().mean().item()
        assert frac <= 0.01, f"{frac:.4f} of elements differ from an fp64 reference"

    def test_the_bias_is_fused(self):
        fvk = _fvk()
        T, N, K = 10, 256, 32
        xf, _, W, b = _inputs(T, N, K)
        a = torch.zeros(T, N, device="cuda", dtype=BF16)
        c = torch.zeros_like(a)
        for bias, out in ((b, a), (torch.zeros_like(b), c)):
            fvk.linear_f32_in_bf16_out(xf.data_ptr(), W.data_ptr(), bias.data_ptr(),
                                       out.data_ptr(), T, N, K,
                                       torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(a, c)


class TestLinearBf16InF32Out:
    @pytest.mark.parametrize("T,N,K", SHAPES)
    def test_matches_a_fp64_reference(self, T, N, K):
        fvk = _fvk()
        _, xb, W, b = _inputs(T, N, K)
        out = torch.zeros(T, N, device="cuda", dtype=torch.float32)
        fvk.linear_bf16_in_f32_out(xb.data_ptr(), W.data_ptr(), b.data_ptr(),
                                   out.data_ptr(), T, N, K,
                                   torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref = xb.double() @ W.double().t() + b.double()
        rel = ((out.double() - ref).abs() / (ref.abs() + 1e-6)).mean().item()
        assert rel < 1e-5, f"mean relative error {rel}"

    def test_the_output_is_fp32_and_not_rounded(self):
        """A bf16-rounded result would still pass a loose relative check."""
        fvk = _fvk()
        T, N, K = 10, 64, 128
        _, xb, W, b = _inputs(T, N, K, seed=5)
        out = torch.zeros(T, N, device="cuda", dtype=torch.float32)
        fvk.linear_bf16_in_f32_out(xb.data_ptr(), W.data_ptr(), b.data_ptr(),
                                   out.data_ptr(), T, N, K,
                                   torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        as_bf16 = out.to(BF16).to(torch.float32)
        assert not torch.equal(out, as_bf16), "output looks bf16-rounded"


class TestGuards:
    def test_null_pointers_fail_on_the_host(self):
        """A null pointer must raise, not launch into an illegal access."""
        fvk = _fvk()
        x = torch.zeros(4, 8, device="cuda")
        W = torch.zeros(16, 8, device="cuda")
        b = torch.zeros(16, device="cuda")
        out = torch.zeros(4, 16, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.linear_f32_in_bf16_out(x.data_ptr(), 0, b.data_ptr(), out.data_ptr(),
                                       4, 16, 8, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()   # the context is still usable

    def test_empty_shapes_are_a_no_op(self):
        fvk = _fvk()
        x = torch.zeros(0, 8, device="cuda")
        W = torch.zeros(16, 8, device="cuda")
        b = torch.zeros(16, device="cuda")
        out = torch.zeros(0, 16, device="cuda", dtype=BF16)
        fvk.linear_f32_in_bf16_out(x.data_ptr(), W.data_ptr(), b.data_ptr(),
                                   out.data_ptr(), 0, 16, 8,
                                   torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
