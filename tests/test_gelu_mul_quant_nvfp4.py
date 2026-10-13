#!/usr/bin/env python3
"""Gated GeGLU activation written straight into a clipped static NVFP4 input.

    out = quant_nvfp4( bf16( bf16(gelu_tanh(gate)) * up ) )

per-16-element e2m1 values with a UE4M3 block scale in the CUTLASS sm120
swizzled layout, and a per-group clip from `cap`.

The contract is that this equals the *unfused* chain it replaces: form the gated
product with HF's two roundings into a bf16 activation, then run the clipped
NVFP4 quantizer over it. That is what the test compares between, using FlashRT's
own quantizer, so a wrong activation function or a missing rounding shows up.

Usage:
  PYTHONPATH=. pytest -q tests/test_gelu_mul_quant_nvfp4.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16

SHAPES = [(568, 16384), (2048, 4304), (16, 256), (300, 512), (128, 16)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _gated_product(gate, up):
    """bf16(bf16(gelu_tanh(gate)) * up), the two roundings HF performs."""
    act = torch.nn.functional.gelu(gate.float(), approximate="tanh").to(BF16)
    return (act.float() * up.float()).to(BF16)


def _measure(fvk, rows, cols, cap_mode, seed=0):
    torch.manual_seed(seed)
    gate = (torch.randn(rows, cols, device="cuda") * 2).to(BF16)
    up = (torch.randn(rows, cols, device="cuda") * 0.5).to(BF16)
    nb = cols // 16
    if cap_mode == "null":
        cap = None
    elif cap_mode == "ones":
        cap = torch.ones(nb, device="cuda", dtype=torch.float32)
    elif cap_mode == "tight":
        cap = (torch.rand(nb, device="cuda") * 0.2 + 0.05).to(torch.float32)
    elif cap_mode == "loose":
        cap = torch.full((nb,), 1e4, device="cuda", dtype=torch.float32)
    else:
        cap = (torch.rand(nb, device="cuda") * 3 + 0.5).to(torch.float32)
    n_sf = int(fvk.gelu_mul_quant_nvfp4_sf_bytes(rows, cols))
    packed = torch.zeros(rows, cols // 2, device="cuda", dtype=torch.uint8)
    sf = torch.zeros(n_sf, device="cuda", dtype=torch.uint8)
    fvk.gelu_mul_quant_nvfp4_swizzled(
        gate.data_ptr(), up.data_ptr(), 0 if cap is None else cap.data_ptr(),
        packed.data_ptr(), sf.data_ptr(), rows, cols,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return gate, up, cap, packed, sf


class TestGeluMulQuantNvfp4:
    @pytest.mark.parametrize("rows,cols", SHAPES)
    def test_matches_the_unfused_chain(self, rows, cols):
        fvk = _fvk()
        gate, up, cap, packed, sf = _measure(fvk, rows, cols, "clipped")
        act = _gated_product(gate, up)
        ref_packed = torch.zeros_like(packed)
        ref_sf = torch.zeros_like(sf)
        fvk.quantize_bf16_to_nvfp4_swizzled_clipped(
            act.data_ptr(), cap.data_ptr(), ref_packed.data_ptr(), ref_sf.data_ptr(),
            rows, cols, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        dp = int((packed != ref_packed).sum().item())
        ds = int((sf != ref_sf).sum().item())
        assert dp == 0, f"{dp} of {packed.numel()} packed bytes differ"
        assert ds == 0, f"{ds} of {sf.numel()} scale bytes differ"

    def test_the_activation_rounding_is_present(self):
        """A one-rounding product differs from the two-rounding chain."""
        fvk = _fvk()
        rows, cols = 512, 1024
        gate, up, cap, packed, sf = _measure(fvk, rows, cols, "clipped", seed=3)
        act2 = _gated_product(gate, up)
        act1 = (torch.nn.functional.gelu(gate.float(), approximate="tanh")
                * up.float()).to(BF16)
        assert not torch.equal(act1, act2), "the two forms must differ for this to test anything"
        single = torch.zeros_like(packed)
        sfs = torch.zeros_like(sf)
        fvk.quantize_bf16_to_nvfp4_swizzled_clipped(
            act1.data_ptr(), cap.data_ptr(), single.data_ptr(), sfs.data_ptr(),
            rows, cols, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(packed, single), "the intermediate bf16 round is missing"

    def test_the_clip_is_applied(self):
        fvk = _fvk()
        rows, cols = 256, 512
        _, _, _, a, _ = _measure(fvk, rows, cols, "null", seed=4)
        _, _, _, b, _ = _measure(fvk, rows, cols, "ones", seed=4)
        assert not torch.equal(a, b), "a clip of 1.0 changed nothing"

    def test_a_null_cap_is_no_clip(self):
        """A null cap must differ from a tight cap and match a very loose one."""
        fvk = _fvk()
        rows, cols = 256, 512
        nb = cols // 16
        _, _, _, loose, _ = _measure(fvk, rows, cols, "loose", seed=5)
        gate, up, cap, tight, _ = _measure(fvk, rows, cols, "tight", seed=5)
        packed = torch.zeros_like(tight)
        sf = torch.zeros_like(_measure(fvk, rows, cols, "null", seed=5)[4])
        fvk.gelu_mul_quant_nvfp4_swizzled(
            gate.data_ptr(), up.data_ptr(), 0, packed.data_ptr(), sf.data_ptr(),
            rows, cols, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(packed, tight)
        assert torch.equal(packed, loose)

    def test_guards(self):
        fvk = _fvk()
        gate = torch.zeros(4, 32, device="cuda", dtype=BF16)
        packed = torch.zeros(4, 16, device="cuda", dtype=torch.uint8)
        sf = torch.zeros(int(fvk.gelu_mul_quant_nvfp4_sf_bytes(4, 32)),
                         device="cuda", dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="multiple of 16"):
            fvk.gelu_mul_quant_nvfp4_swizzled(
                gate.data_ptr(), gate.data_ptr(), 0, packed.data_ptr(), sf.data_ptr(),
                4, 24, torch.cuda.current_stream().cuda_stream)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.gelu_mul_quant_nvfp4_swizzled(
                0, gate.data_ptr(), 0, packed.data_ptr(), sf.data_ptr(),
                4, 32, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
