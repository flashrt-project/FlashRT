#!/usr/bin/env python3
"""bf16 NCHW patch im2col with a parametrised patch size.

    in  (nv, C, H, W) bf16 planar  ->  out (nv * gh * gw, C * P * P) bf16

Column order is (c, kh, kw), which is what conv.weight.reshape(out_ch, C*P*P)
expects, and patches are numbered row-major over the (gh, gw) grid.

Distinct from the fp16 NHWC patch_im2col in the same file on dtype, layout,
patch-size parametrisation and feature order, so the tests pin the layout
against torch's own unfold rather than only against a sibling.

Usage:
  PYTHONPATH=. pytest -q tests/test_patch_im2col_nchw.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16
SHAPES = [(2, 3, 448, 448, 14), (1, 3, 224, 224, 16), (2, 3, 64, 96, 8), (1, 1, 32, 32, 32)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _reference(img, C, P):
    """(nv, C, H, W) -> (nv*gh*gw, C*P*P) with columns ordered (c, kh, kw)."""
    nv = img.shape[0]
    unf = img.unfold(2, P, P).unfold(3, P, P)               # (nv, C, gh, gw, P, P)
    return unf.permute(0, 2, 3, 1, 4, 5).reshape(-1, C * P * P)


def _run(fvk, nv, C, H, W, P, seed=0):
    torch.manual_seed(seed)
    img = torch.randn(nv, C, H, W, device="cuda").to(BF16)
    out = torch.zeros(nv * (H // P) * (W // P), C * P * P, device="cuda", dtype=BF16)
    fvk.patch_im2col_nchw_bf16(img.data_ptr(), out.data_ptr(), nv, C, H, W, P,
                               torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return img, out


class TestPatchIm2colNchw:
    @pytest.mark.parametrize("nv,C,H,W,P", SHAPES)
    def test_matches_unfold(self, nv, C, H, W, P):
        fvk = _fvk()
        img, out = _run(fvk, nv, C, H, W, P)
        ref = _reference(img, C, P)
        assert torch.equal(out, ref)

    def test_the_column_order_is_c_kh_kw_not_kh_kw_c(self):
        """The other order is what a NHWC im2col produces; it must not pass."""
        fvk = _fvk()
        nv, C, H, W, P = 1, 3, 64, 64, 8
        img, out = _run(fvk, nv, C, H, W, P, seed=1)
        unf = img.unfold(2, P, P).unfold(3, P, P)
        wrong = unf.permute(0, 2, 3, 4, 5, 1).reshape(-1, C * P * P)   # (kh,kw,c)
        assert not torch.equal(out, wrong), "the feature order is not pinned"
        assert torch.equal(out, _reference(img, C, P))

    def test_patches_are_row_major_over_the_grid(self):
        """Patch 1 of a 2-wide grid must be one step right, not one step down."""
        fvk = _fvk()
        nv, C, H, W, P = 1, 1, 32, 64, 32          # gh=1, gw=2
        img, out = _run(fvk, nv, C, H, W, P, seed=2)
        assert out.shape[0] == 2
        assert torch.equal(out[0], img[0, 0, :, :P].reshape(-1))
        assert torch.equal(out[1], img[0, 0, :, P:].reshape(-1))

    def test_guards(self):
        fvk = _fvk()
        img = torch.zeros(1, 3, 33, 33, device="cuda", dtype=BF16)
        out = torch.zeros(4, 3 * 8 * 8, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="multiples of P"):
            fvk.patch_im2col_nchw_bf16(img.data_ptr(), out.data_ptr(), 1, 3, 33, 33, 8,
                                       torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.patch_im2col_nchw_bf16(0, out.data_ptr(), 1, 3, 32, 32, 8,
                                       torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
