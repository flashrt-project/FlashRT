#!/usr/bin/env python3
"""Gated GeGLU activation: bf16 out, and straight into a static FP8 input.

    bf16 : out = bf16( bf16(gelu_tanh(gate)) * up )
    FP8  : out = fp8_e4m3( clamp( that / act_scale, +-448 ) )

Both round the activation to bf16 before the product and round the product
again. activation.cu's gate_geglu computes the same function with one rounding,
so the tests pin the roundings rather than a tolerance.

Usage:
  PYTHONPATH=. pytest -q tests/test_gelu_mul.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16
SIZES = [16384, 1024, 8, 64 * 4304]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _inputs(n, seed=0):
    torch.manual_seed(seed)
    return ((torch.randn(n, device="cuda") * 2).to(BF16),
            (torch.randn(n, device="cuda") * 0.5).to(BF16))


def _product(gate, up):
    act = torch.nn.functional.gelu(gate.float(), approximate="tanh").to(BF16)
    return (act.float() * up.float()).to(BF16)


class TestGeluMulBf16:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_the_two_rounding_form(self, n):
        fvk = _fvk()
        gate, up = _inputs(n, seed=1)
        out = torch.empty(n, device="cuda", dtype=BF16)
        fvk.gelu_mul_bf16(gate.data_ptr(), up.data_ptr(), out.data_ptr(), n,
                          torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(out, _product(gate, up))

    def test_the_intermediate_rounding_is_present(self):
        """A one-rounding form must fail this, which is what gate_geglu is."""
        fvk = _fvk()
        n = 8192
        gate, up = _inputs(n, seed=2)
        out = torch.empty(n, device="cuda", dtype=BF16)
        fvk.gelu_mul_bf16(gate.data_ptr(), up.data_ptr(), out.data_ptr(), n,
                          torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        single = (torch.nn.functional.gelu(gate.float(), approximate="tanh")
                  * up.float()).to(BF16)
        assert not torch.equal(single, _product(gate, up)), "the forms must differ"
        assert not torch.equal(out, single), "the intermediate rounding is missing"


class TestGeluMulQuantFp8:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_the_two_pass_form(self, n):
        fvk = _fvk()
        gate, up = _inputs(n, seed=3)
        sc = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        out = torch.zeros(n, device="cuda", dtype=torch.float8_e4m3fn)
        fvk.gelu_mul_quant_fp8_static_bf16(gate.data_ptr(), up.data_ptr(), sc.data_ptr(),
                                           out.data_ptr(), n,
                                           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        act = _product(gate, up)
        ref = torch.zeros_like(out)
        fvk.quantize_fp8_static(act.data_ptr(), ref.data_ptr(), sc.data_ptr(), n,
                                torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        nd = int((out.view(torch.uint8) != ref.view(torch.uint8)).sum().item())
        assert nd == 0, f"{nd} of {n} bytes differ from the two-pass form"

    def test_the_scale_is_applied(self):
        fvk = _fvk()
        n = 4096
        gate, up = _inputs(n, seed=4)
        outs = []
        for s in (0.02, 0.5):
            sc = torch.tensor([s], device="cuda", dtype=torch.float32)
            o = torch.zeros(n, device="cuda", dtype=torch.float8_e4m3fn)
            fvk.gelu_mul_quant_fp8_static_bf16(gate.data_ptr(), up.data_ptr(),
                                               sc.data_ptr(), o.data_ptr(), n,
                                               torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            outs.append(o)
        assert not torch.equal(outs[0].view(torch.uint8), outs[1].view(torch.uint8))


class TestGuards:
    def test_null_pointers_fail_on_the_host(self):
        fvk = _fvk()
        g = torch.zeros(64, device="cuda", dtype=BF16)
        o = torch.zeros(64, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.gelu_mul_bf16(0, g.data_ptr(), o.data_ptr(), 64,
                              torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

    def test_the_fp8_form_needs_a_multiple_of_eight(self):
        fvk = _fvk()
        g = torch.zeros(100, device="cuda", dtype=BF16)
        o = torch.zeros(100, device="cuda", dtype=torch.float8_e4m3fn)
        sc = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="multiple of 8"):
            fvk.gelu_mul_quant_fp8_static_bf16(g.data_ptr(), g.data_ptr(), sc.data_ptr(),
                                               o.data_ptr(), 100,
                                               torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
