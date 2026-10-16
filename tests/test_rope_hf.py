#!/usr/bin/env python3
"""HuggingFace rotate_half RoPE, in place, with HF's intermediate roundings.

    x'[d]      = bf16( bf16(x[d]      * cos[d])      + bf16(-x[d+half] * sin[d]) )
    x'[d+half] = bf16( bf16(x[d+half] * cos[d+half]) + bf16( x[d]      * sin[d+half]) )

Two things distinguish this from the RoPE kernels already in the shared set, and
both change the result, so both are tested:

  * each product is rounded to bf16 *before* the pair is summed, where
    rope_neox_qk_bf16 rounds once after the sum;
  * the cos/sin tables are full-width [seq, dim] and each half is read
    separately, where rope_neox_qk_bf16 reads a half-width table shared by both.

Usage:
  PYTHONPATH=. pytest -q tests/test_rope_hf.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16

SHAPES = [(568, 8, 256, 0), (128, 4, 256, 0), (1, 1, 256, 0), (300, 8, 256, 64), (7, 2, 64, 0)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _bf(t):
    return t.to(BF16)


def _reference(x, row_stride, heads, dim, cos, sin, seq):
    """HF's formulation, roundings included, evaluated on the visible head rows."""
    out = x.clone().reshape(-1)
    xf = x.reshape(-1)
    half = dim // 2
    for s in range(seq):
        for h in range(heads):
            base = s * row_stride + h * dim
            xr = xf[base:base + dim].float()
            c1, c2 = cos[s, :half].float(), cos[s, half:].float()
            s1, s2 = sin[s, :half].float(), sin[s, half:].float()
            out[base:base + half] = _bf(_bf(xr[:half] * c1) + _bf(-xr[half:] * s1))
            out[base + half:base + dim] = _bf(_bf(xr[half:] * c2) + _bf(xr[:half] * s2))
    return out.view_as(x)


def _run(fvk, seq, heads, dim, pad, seed=0):
    torch.manual_seed(seed)
    row_stride = heads * dim + pad
    x = (torch.randn(seq, row_stride, device="cuda") * 2).to(BF16)
    cos = torch.randn(seq, dim, device="cuda").to(BF16)
    sin = torch.randn(seq, dim, device="cuda").to(BF16)
    got = x.clone()
    fvk.rope_hf_bf16(got.data_ptr(), row_stride, heads, dim,
                     cos.data_ptr(), sin.data_ptr(), seq,
                     torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return got, _reference(x, row_stride, heads, dim, cos, sin, seq)


class TestRopeHf:
    @pytest.mark.parametrize("seq,heads,dim,pad", SHAPES)
    def test_matches_the_hf_formulation(self, seq, heads, dim, pad):
        fvk = _fvk()
        got, ref = _run(fvk, seq, heads, dim, pad)
        assert torch.equal(got, ref), \
            f"{int((got != ref).sum().item())} of {ref.numel()} elements differ"


class TestDisciplines:
    """Each of the two distinguishing properties, pinned on its own."""

    def test_rounds_each_product_before_the_sum(self):
        """A one-rounding implementation is what rope_neox_qk_bf16 does."""
        fvk = _fvk()
        seq, heads, dim = 64, 4, 256
        got, ref = _run(fvk, seq, heads, dim, 0, seed=1)
        half = dim // 2
        torch.manual_seed(1)
        row_stride = heads * dim
        x = (torch.randn(seq, row_stride, device="cuda") * 2).to(BF16)
        cos = torch.randn(seq, dim, device="cuda").to(BF16)
        sin = torch.randn(seq, dim, device="cuda").to(BF16)
        single = x.clone().reshape(-1)
        xf = x.reshape(-1)
        for s in range(seq):
            for h in range(heads):
                base = s * row_stride + h * dim
                xr = xf[base:base + dim].float()
                c1, c2 = cos[s, :half].float(), cos[s, half:].float()
                s1, s2 = sin[s, :half].float(), sin[s, half:].float()
                single[base:base + half] = _bf(xr[:half] * c1 - xr[half:] * s1)
                single[base + half:base + dim] = _bf(xr[half:] * c2 + xr[:half] * s2)
        single = single.view_as(x)
        assert not torch.equal(single, ref), "the reference must pin the two roundings"
        assert torch.equal(got, ref)

    def test_reads_both_halves_of_the_table(self):
        """A half-width-table implementation ignores cos/sin past dim/2."""
        fvk = _fvk()
        seq, heads, dim = 32, 2, 256
        half = dim // 2
        torch.manual_seed(2)
        row_stride = heads * dim
        x = (torch.randn(seq, row_stride, device="cuda") * 2).to(BF16)
        cos = torch.randn(seq, dim, device="cuda").to(BF16)
        sin = torch.randn(seq, dim, device="cuda").to(BF16)
        cos2 = cos.clone()
        cos2[:, half:] = torch.randn(seq, half, device="cuda").to(BF16)
        outs = []
        for c in (cos, cos2):
            o = x.clone()
            fvk.rope_hf_bf16(o.data_ptr(), row_stride, heads, dim,
                             c.data_ptr(), sin.data_ptr(), seq,
                             torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            outs.append(o)
        assert not torch.equal(outs[0], outs[1]), "the second table half is not read"


class TestProperties:
    def test_rows_are_strided_and_untouched_beyond_the_heads(self):
        fvk = _fvk()
        seq, heads, dim, pad = 40, 4, 128, 32
        got, ref = _run(fvk, seq, heads, dim, pad, seed=3)
        assert torch.equal(got, ref)
        # the padding past heads*dim must be left alone
        for s in range(seq):
            assert torch.equal(got[s, heads * dim:], ref[s, heads * dim:])

    def test_guards(self):
        fvk = _fvk()
        x = torch.zeros(8, 256, device="cuda", dtype=BF16)
        cos = torch.zeros(8, 128, device="cuda", dtype=BF16)
        sin = torch.zeros(8, 128, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.rope_hf_bf16(0, 256, 1, 256, cos.data_ptr(), sin.data_ptr(), 8,
                             torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="dim must be even"):
            fvk.rope_hf_bf16(x.data_ptr(), 256, 1, 255, cos.data_ptr(), sin.data_ptr(), 8,
                             torch.cuda.current_stream().cuda_stream)
        with pytest.raises(RuntimeError, match="row stride"):
            fvk.rope_hf_bf16(x.data_ptr(), 3, 1, 256, cos.data_ptr(), sin.data_ptr(), 8,
                             torch.cuda.current_stream().cuda_stream)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
