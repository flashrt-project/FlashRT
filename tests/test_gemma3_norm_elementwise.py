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

BF16 = torch.bfloat16


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


# ── Gemma3 RMSNorm writing a static FP8 GEMM input ───────────────────────────

class TestGemma3RmsNormFp8:
    """The fp8 form must be exactly the bf16 form plus the quantize pass.

    The contract is byte identity with `quantize_fp8_static` applied to the bf16
    output of the same norm, not a tolerance: a consumer calibrated against that
    pass buckets a one-ULP drift into a different integer, so the fused kernel
    has to land on the same side of every rounding boundary.
    """

    @pytest.mark.parametrize("rows,dim", [(568, 2048), (40, 256), (1, 4096), (7, 1152)])
    def test_byte_identical_to_a_standalone_quantize(self, rows, dim):
        fvk = _fvk()
        torch.manual_seed(9)
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        fused = torch.zeros(rows, dim, device="cuda", dtype=torch.float8_e4m3fn)
        fvk.gemma3_rms_norm_fp8_bf16(x.data_ptr(), w.data_ptr(), fused.data_ptr(),
                                     scale.data_ptr(), rows, dim, 1e-6,
                                     torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        # the two passes the fused form replaces, over the same input
        normed = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.gemma3_rms_norm_bf16(x.data_ptr(), dim, w.data_ptr(), normed.data_ptr(), dim,
                                 0, dim, rows, rows, 0, dim, 1e-6,
                                 torch.cuda.current_stream().cuda_stream)
        ref = torch.zeros_like(fused)
        fvk.quantize_fp8_static(normed.reshape(-1).data_ptr(), ref.reshape(-1).data_ptr(),
                                scale.data_ptr(), rows * dim,
                                torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        nd = int((fused.view(torch.uint8) != ref.view(torch.uint8)).sum().item())
        assert nd == 0, f"{nd} of {fused.numel()} bytes differ from the two-pass form"

    def test_the_scale_is_actually_applied(self):
        """A kernel that skipped the division would still pass a cosine check."""
        fvk = _fvk()
        torch.manual_seed(10)
        rows, dim = 16, 512
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        a = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        b = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        oa = torch.zeros(rows, dim, device="cuda", dtype=torch.float8_e4m3fn)
        ob = torch.zeros_like(oa)
        for scale, out in ((a, oa), (b, ob)):
            fvk.gemma3_rms_norm_fp8_bf16(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                                         scale.data_ptr(), rows, dim, 1e-6,
                                         torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(oa.view(torch.uint8), ob.view(torch.uint8))

    def test_rejects_an_unaligned_width(self):
        fvk = _fvk()
        x = _bf16(torch.zeros(2, 100, device="cuda"))
        w = _bf16(torch.zeros(100, device="cuda"))
        out = torch.zeros(2, 100, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="multiple of 8"):
            fvk.gemma3_rms_norm_fp8_bf16(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                                         scale.data_ptr(), 2, 100, 1e-6,
                                         torch.cuda.current_stream().cuda_stream)


# ── Adaptive RMSNorm with a shared modulation ────────────────────────────────

class TestAdaRmsNorm:
    """y = bf16( bf16(x) * rsqrt(mean(x^2) + eps) * (1 + scale) + shift ).

    One (scale, shift) pair for every row, no weight, no gate output -- the
    contract ada_rms_norm_style does not cover.
    """

    @pytest.mark.parametrize("rows,dim", [(10, 1024), (3, 256), (1, 4096), (50, 512)])
    def test_matches_the_documented_form(self, rows, dim):
        fvk = _fvk()
        torch.manual_seed(11)
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        mod = _bf16(torch.randn(2 * dim, device="cuda") * 0.5)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.ada_rms_norm(x.data_ptr(), mod.data_ptr(), out.data_ptr(), rows, dim, 1e-6,
                         torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
        ref = _bf16(xf * inv * (1.0 + mod[:dim].float()) + mod[dim:].float())
        assert (out != ref).float().mean().item() <= _MISMATCH_FRAC

    def test_the_one_plus_scale_form_is_not_a_plain_scale(self):
        """A kernel using scale instead of (1 + scale) must fail this."""
        fvk = _fvk()
        torch.manual_seed(12)
        rows, dim = 4, 512
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        mod = _bf16(torch.randn(2 * dim, device="cuda") * 0.5)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.ada_rms_norm(x.data_ptr(), mod.data_ptr(), out.data_ptr(), rows, dim, 1e-6,
                         torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
        plain = _bf16(xf * inv * mod[:dim].float() + mod[dim:].float())
        assert (out != plain).float().mean().item() > 0.5

    def test_the_shift_half_is_read(self):
        """A kernel reading only the scale half would ignore mod[dim:]."""
        fvk = _fvk()
        torch.manual_seed(13)
        rows, dim = 4, 256
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        mod_a = _bf16(torch.randn(2 * dim, device="cuda") * 0.5)
        mod_b = mod_a.clone()
        mod_b[dim:] = _bf16(torch.randn(dim, device="cuda") * 0.5)
        oa = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        ob = torch.empty_like(oa)
        for m, o in ((mod_a, oa), (mod_b, ob)):
            fvk.ada_rms_norm(x.data_ptr(), m.data_ptr(), o.data_ptr(), rows, dim, 1e-6,
                             torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(oa, ob)


# ── The vectorized-order Gemma3 RMSNorm ──────────────────────────────────────

class TestGemma3RmsNormV8:
    """Same formula as gemma3_rms_norm_bf16, a different summation order.

    Both orders exist in the caller this serves, and which one a call site needs
    depends on the call site: a fused norm+RoPE kernel that reduces warp-first
    cannot be reproduced by the strided-loop form, even though the two agree on
    almost every element.
    """

    @pytest.mark.parametrize("rows,dim", [(40, 2560), (568, 256), (4, 4096), (7, 512)])
    def test_matches_the_documented_form(self, rows, dim):
        fvk = _fvk()
        torch.manual_seed(21)
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        out = torch.empty(rows, dim, device="cuda", dtype=torch.bfloat16)
        fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), dim, w.data_ptr(), out.data_ptr(), dim,
                                    0, dim, rows, rows, 0, dim, 1e-6,
                                    torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref, _ = _gemma3_reference(x, w, 1e-6)
        assert (out != ref).float().mean().item() <= _MISMATCH_FRAC

    def test_residual_mode_matches_two_roundings(self):
        fvk = _fvk()
        torch.manual_seed(22)
        rows, dim = 16, 1024
        x = _bf16(torch.randn(rows, dim, device="cuda") * 2)
        w = _bf16(torch.randn(dim, device="cuda") * 0.3)
        res = _bf16(torch.randn(rows, dim, device="cuda"))
        got_res, got_out = res.clone(), torch.empty_like(x)
        fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), dim, w.data_ptr(), got_out.data_ptr(), dim,
                                    got_res.data_ptr(), dim, rows, rows, 0, dim, 1e-6,
                                    torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        ref_out, ref_res = _gemma3_reference(x, w, 1e-6, residual=res)
        assert (got_res != ref_res).float().mean().item() <= _MISMATCH_FRAC
        assert (got_out != ref_out).float().mean().item() <= _MISMATCH_FRAC

    def test_per_head_group_addressing(self):
        """rows_per_group/group_stride address head h of row s at s*gs + h*xs."""
        fvk = _fvk()
        torch.manual_seed(23)
        seq, nh, hd = 5, 4, 256
        dim = nh * hd
        x = _bf16(torch.randn(seq, dim, device="cuda") * 2)
        w = _bf16(torch.randn(hd, device="cuda") * 0.3)
        before = x.clone()
        fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), hd, w.data_ptr(), x.data_ptr(), hd,
                                    0, hd, seq * nh, nh, dim, hd, 1e-6,
                                    torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        worst = 0.0
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            ref, _ = _gemma3_reference(before[:, sl].clone(), w, 1e-6)
            worst = max(worst, (x[:, sl] != ref).float().mean().item())
        assert worst <= _MISMATCH_FRAC, f"worst per-head mismatch {worst}"

    def test_guards(self):
        fvk = _fvk()
        x = _bf16(torch.zeros(4, 64, device="cuda"))
        w = _bf16(torch.zeros(64, device="cuda"))
        with pytest.raises(RuntimeError, match="no destination"):
            fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), 64, w.data_ptr(), 0, 64, 0, 64,
                                        4, 4, 0, 64, 1e-6,
                                        torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="multiple of 8"):
            fvk.gemma3_rms_norm_v8_bf16(x.data_ptr(), 64, w.data_ptr(), x.data_ptr(), 64,
                                        0, 64, 4, 4, 0, 60, 1e-6,
                                        torch.cuda.current_stream().cuda_stream)


# ── bias + GELU + FP8, with the activation rounded to bf16 first ─────────────

class TestBiasGeluQuantFp8Rounded:
    """The rounding is the whole difference from the sibling entry.

    The sibling quantizes the fp32 gelu directly; this one rounds it to bf16
    first, which is what running bias_gelu_bf16 and then quantize_fp8_static back
    to back produces. A caller calibrated against that pair needs this one, so
    the test pins both the agreement with the two-pass form and the
    disagreement with the one-pass sibling.
    """

    def _run(self, fvk, rows, cols, rounded, seed=0):
        torch.manual_seed(seed)
        h = (torch.randn(rows, cols, device="cuda") * 1.5).to(BF16)
        b = (torch.randn(cols, device="cuda") * 0.1).to(BF16)
        sc = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        out = torch.zeros(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
        fn = (fvk.bias_gelu_quantize_fp8_static_bf16_rounded if rounded
              else fvk.bias_gelu_quantize_fp8_static_bf16)
        fn(h.data_ptr(), b.data_ptr(), out.data_ptr(), sc.data_ptr(), rows, cols,
           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return h, b, sc, out

    @pytest.mark.parametrize("rows,cols", [(2048, 4304), (2048, 1152), (16, 256)])
    def test_agrees_with_the_two_pass_form(self, rows, cols):
        fvk = _fvk()
        h, b, sc, got = self._run(fvk, rows, cols, rounded=True)
        # the two passes it replaces, run in order: bias+gelu writes a bf16
        # activation in place, then quantize_fp8_static reads it back
        fvk.bias_gelu_bf16(h.data_ptr(), b.data_ptr(), rows, cols,
                           torch.cuda.current_stream().cuda_stream)
        ref = torch.zeros_like(got)
        fvk.quantize_fp8_static(h.data_ptr(), ref.data_ptr(), sc.data_ptr(),
                                rows * cols, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        nd = int((got.view(torch.uint8) != ref.view(torch.uint8)).sum().item())
        assert nd == 0, f"{nd} of {got.numel()} bytes differ from the two-pass form"

    def test_differs_from_the_unrounded_sibling(self):
        """A copy of the sibling would pass a tolerance check; it must not pass this."""
        fvk = _fvk()
        _, _, _, a = self._run(fvk, 2048, 1152, rounded=True, seed=7)
        _, _, _, c = self._run(fvk, 2048, 1152, rounded=False, seed=7)
        frac = (a.view(torch.uint8) != c.view(torch.uint8)).float().mean().item()
        assert frac > 0.005, f"the two entries differ on only {frac:.4f} of bytes"

    def test_a_null_bias_is_a_plain_activation(self):
        fvk = _fvk()
        rows, cols = 64, 512
        torch.manual_seed(8)
        h = (torch.randn(rows, cols, device="cuda") * 1.5).to(BF16)
        sc = torch.tensor([0.02], device="cuda", dtype=torch.float32)
        out = torch.zeros(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
        fvk.bias_gelu_quantize_fp8_static_bf16_rounded(
            h.data_ptr(), 0, out.data_ptr(), sc.data_ptr(), rows, cols,
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        act = h.clone()
        # a zero bias rather than null: bias_gelu_bf16 takes a real pointer, and
        # adding 0.0f in fp32 is exact, so this is the same activation
        zb = torch.zeros(cols, device="cuda", dtype=BF16)
        fvk.bias_gelu_bf16(act.data_ptr(), zb.data_ptr(), rows, cols,
                           torch.cuda.current_stream().cuda_stream)
        ref = torch.zeros_like(out)
        fvk.quantize_fp8_static(act.data_ptr(), ref.data_ptr(), sc.data_ptr(),
                                rows * cols, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
