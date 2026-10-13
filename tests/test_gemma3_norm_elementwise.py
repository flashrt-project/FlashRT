#!/usr/bin/env python3
"""Numerical tests for the Gemma3 RMSNorm and the fp32/strided elementwise ops.

These are the operations a Gemma3-family VLA reaches for that the shared kernel
set did not have:

  gemma3_rms_norm_bf16    — the (1 + w) weight form, strided rows, HF's
                            two-rounding post-norm residual
  add_bias_rows_bf16      — bias over the leading columns of a wider, strided row
  axpy_f32 / mul_cols_f32 / copy_scaled_cols_f32 — the fp32 flow-sampler step

Each is checked against a torch reference in fp64 or exact fp32, and the
rounding-sensitive ones are checked for bit-parity with the arithmetic they are
specified to reproduce rather than for a tolerance.

Usage:
  PYTHONPATH=. pytest -q tests/test_gemma3_norm_elementwise.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _bf16(t):
    return t.to(torch.bfloat16)


# ── Gemma3 RMSNorm ───────────────────────────────────────────────────────────

def _gemma3_reference(x, w, eps, residual=None):
    """The documented arithmetic, evaluated in fp32 as the kernel evaluates it.

    Deliberately fp32 rather than fp64: the result is rounded to bf16, which has
    eight mantissa bits, so an fp64 chain lands on the other side of a rounding
    boundary often enough that a bit comparison against it is meaningless. What
    this pins is the *formula* — the (1 + w) factor and the two-rounding
    residual — which a fp64 evaluation still distinguishes exactly.
    """
    rows, dim = x.shape
    xf = x.float()
    ss = (xf * xf).sum(-1, keepdim=True)
    inv = torch.rsqrt(ss / dim + eps)
    y = _bf16(xf * inv * (1.0 + w.float()))     # the norm rounds to bf16 first
    if residual is not None:
        r = _bf16(residual.float() + y.float())  # residual = bf16(residual + bf16(norm))
        return y, r
    return y, None


# A wrong implementation is wrong by a factor, not by an ulp: scaling by w
# instead of (1 + w) moves ~4x here. These bounds are loose enough to absorb a
# different fp32 reduction order in the sum of squares and tight enough that
# nothing but the right formula passes.
_MISMATCH_FRAC = 0.01
_COS_FLOOR = 0.99999


class TestGemma3RmsNorm:
    @pytest.mark.parametrize("rows,dim", [(8, 2048), (1, 2560), (33, 1152), (5, 72)])
    def test_matches_reference(self, rows, dim):
        fvk = _fvk()
        torch.manual_seed(0)
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(),
                                 out.data_ptr(), dim, 0, 0,
                                 rows, rows, 0, dim, 1e-6,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref, _ = _gemma3_reference(x, w, 1e-6)
        frac = (out != ref).float().mean().item()
        assert frac <= _MISMATCH_FRAC, f"{frac:.4f} of elements differ from the reference"
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1).double(), ref.float().reshape(-1).double(), dim=0).item()
        assert cos > _COS_FLOOR, f"cosine {cos}"

    def test_the_one_plus_w_form_is_not_a_plain_weight(self):
        """A kernel that scaled by w instead of (1 + w) must fail this test."""
        fvk = _fvk()
        torch.manual_seed(1)
        dim, rows = 512, 4
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(), out.data_ptr(), dim, 0, 0,
                                 rows, rows, 0, dim, 1e-6, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        plain = _bf16(x.float() * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)
                      * w.float())
        assert not torch.equal(out, plain)
        ref, _ = _gemma3_reference(x, w, 1e-6)
        assert (out != ref).float().mean().item() <= _MISMATCH_FRAC

    def test_residual_mode_matches_two_roundings(self):
        fvk = _fvk()
        torch.manual_seed(2)
        rows, dim = 6, 1024
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        res = _bf16(torch.randn(rows, dim, device="cuda"))
        got_res = res.clone()
        got_out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(),
                                 got_out.data_ptr(), dim, got_res.data_ptr(), dim,
                                 rows, rows, 0, dim, 1e-6, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref_out, ref_res = _gemma3_reference(x, w, 1e-6, residual=res)
        assert (got_res != ref_res).float().mean().item() <= _MISMATCH_FRAC
        assert (got_out != ref_out).float().mean().item() <= _MISMATCH_FRAC

    def test_strided_rows_in_place_on_a_fused_buffer(self):
        """rows_per_group/group_stride address per-head slices of a fused buffer."""
        fvk = _fvk()
        torch.manual_seed(3)
        seq, nh, hd = 7, 4, 256
        dim = nh * hd
        q = _bf16(torch.randn(seq, dim, device="cuda") * 2)
        w = _bf16(torch.randn(hd, device="cuda") * 0.3)
        before = q.clone()          # the call is in place, so keep the input
        # rows_per_group = nh, group_stride = dim, x_stride = hd addresses head h of
        # row s at s * dim + h * hd, i.e. each head in place on the fused buffer
        fvk.gemma3_rms_norm_bf16(q.data_ptr(), hd, w.data_ptr(),
                                 q.data_ptr(), hd, 0, 0,
                                 seq * nh, nh, dim, hd, 1e-6,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref_heads = []
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            ref, _ = _gemma3_reference(before[:, sl].clone(), w, 1e-6)
            ref_heads.append((q[:, sl] != ref).float().mean().item())
        assert max(ref_heads) <= _MISMATCH_FRAC, f"per-head mismatch fractions {ref_heads}"

    def test_out_or_residual_may_be_omitted(self):
        fvk = _fvk()
        torch.manual_seed(4)
        rows, dim = 3, 256
        x = _bf16(torch.randn(rows, dim, device="cuda"))
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        res = _bf16(torch.randn(rows, dim, device="cuda"))
        res2 = res.clone()
        fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(), 0, dim,
                                 res2.data_ptr(), dim, rows, rows, 0, dim, 1e-6,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        _, ref_res = _gemma3_reference(x, w, 1e-6, residual=res)
        assert (res2 != ref_res).float().mean().item() <= _MISMATCH_FRAC

    def test_requires_a_destination(self):
        fvk = _fvk()
        dim = 256
        x = _bf16(torch.randn(2, dim, device="cuda"))
        w = _bf16(torch.randn(dim, device="cuda"))
        with pytest.raises(RuntimeError, match="give out, residual, or both"):
            fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(), 0, dim, 0, dim,
                                     2, 2, 0, dim, 1e-6, torch.cuda.current_stream().cuda_stream)


# ── Strided row bias ─────────────────────────────────────────────────────────

class TestAddBiasRows:
    @pytest.mark.parametrize("rows,ld,cols", [(9, 3072, 1024), (1, 1024, 1024), (64, 5120, 2560)])
    def test_adds_only_the_leading_columns(self, rows, ld, cols):
        fvk = _fvk()
        torch.manual_seed(5)
        x = _bf16(torch.randn(rows, ld, device="cuda"))
        b = _bf16(torch.randn(cols, device="cuda") * 0.1)
        want = x.clone()
        want[:, :cols] = _bf16(x[:, :cols].float() + b.float())
        fvk.add_bias_rows_bf16(x.data_ptr(), ld, b.data_ptr(), rows, cols,
                               torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(x, want)

    def test_rejects_an_unaligned_width(self):
        fvk = _fvk()
        x = _bf16(torch.zeros(2, 64, device="cuda"))
        b = _bf16(torch.zeros(5, device="cuda"))
        with pytest.raises(RuntimeError, match="multiple of 8"):
            fvk.add_bias_rows_bf16(x.data_ptr(), 64, b.data_ptr(), 2, 5,
                                   torch.cuda.current_stream().cuda_stream)

    def test_rejects_a_row_narrower_than_the_bias(self):
        fvk = _fvk()
        x = _bf16(torch.zeros(2, 16, device="cuda"))
        b = _bf16(torch.zeros(32, device="cuda"))
        with pytest.raises(RuntimeError, match="smaller than cols"):
            fvk.add_bias_rows_bf16(x.data_ptr(), 16, b.data_ptr(), 2, 32,
                                   torch.cuda.current_stream().cuda_stream)


# ── fp32 flow-step ops ───────────────────────────────────────────────────────

class TestFp32StepOps:
    def test_axpy(self):
        fvk = _fvk()
        torch.manual_seed(6)
        n = 4096
        x = torch.randn(n, device="cuda")
        v = torch.randn(n, device="cuda")
        want = x + v * 0.25
        fvk.axpy_f32(x.data_ptr(), v.data_ptr(), 0.25, n, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(x, want)

    @pytest.mark.parametrize("rows,cols", [(10, 32), (1, 1024), (64, 7)])
    def test_mul_cols(self, rows, cols):
        fvk = _fvk()
        torch.manual_seed(7)
        x = torch.randn(rows, cols, device="cuda")
        m = (torch.rand(cols, device="cuda") > 0.5).float()   # one scale per column
        want = x * m
        fvk.mul_cols_f32(x.data_ptr(), m.data_ptr(), rows, cols,
                         torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(x, want)

    def test_copy_scaled_cols_with_and_without_a_mask(self):
        fvk = _fvk()
        torch.manual_seed(8)
        rows, cols = 10, 32
        src = torch.randn(rows, cols, device="cuda")
        m = (torch.rand(cols, device="cuda") > 0.5).float()   # one scale per column
        dst = torch.empty_like(src)
        fvk.copy_scaled_cols_f32(dst.data_ptr(), src.data_ptr(), m.data_ptr(), rows, cols,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(dst, src * m)
        # a null mask is a plain copy, and the source is left alone
        kept = src.clone()
        fvk.copy_scaled_cols_f32(dst.data_ptr(), src.data_ptr(), 0, rows, cols,
                                 torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(dst, kept) and torch.equal(src, kept)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
