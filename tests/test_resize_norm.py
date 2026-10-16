#!/usr/bin/env python3
"""uint8 image resize + rescale-to-bf16 preprocessing.

The kernel reproduces OpenCV's INTER_LINEAR bilinear resize on uint8 and then
maps to bf16 through (x - 127.5) / 127.5, in (nv, H, W, 3) uint8 HWC to
(nv, 3, OH, OW) bf16 CHW.

The strong test compares against cv2 itself, which is what the arithmetic is
defined against and what a caller's reference pipeline uses. cv2 is optional, so
that test skips when it is absent; the remaining tests check properties that do
not need it.

Usage:
  PYTHONPATH=. pytest -q tests/test_resize_norm.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16

SHAPES = [
    (2, 448, 448, 448, 448),   # identity, the served shape
    (2, 480, 640, 448, 448),   # downscale, non-square source
    (1, 224, 224, 448, 448),   # upscale
    (2, 448, 448, 224, 224),   # downscale by 2
    (1, 100, 137, 64, 64),     # odd, non-multiple sizes
    (1, 33, 33, 17, 31),       # both axes non-integral
]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _to_bf16(v_uint8):
    """The kernel's tail: fp32 (x - 127.5) / 127.5, rounded to bf16."""
    return ((v_uint8.to(torch.float32) - 127.5) / 127.5).to(BF16)


class TestAgainstCv2:
    @pytest.mark.parametrize("nv,H,W,OH,OW", SHAPES)
    def test_matches_cv2_inter_linear(self, nv, H, W, OH, OW):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        fvk = _fvk()
        torch.manual_seed(0)
        src = torch.randint(0, 256, (nv, H, W, 3), device="cuda", dtype=torch.uint8)
        got = torch.zeros(nv, 3, OH, OW, device="cuda", dtype=BF16)
        fvk.resize_norm_u8_hwc_to_bf16_chw(src.data_ptr(), got.data_ptr(),
                                           nv, H, W, OH, OW,
                                           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        host = src.cpu().numpy()
        want = np.stack([
            cv2.resize(host[i], (OW, OH), interpolation=cv2.INTER_LINEAR)
            for i in range(nv)
        ])                                        # (nv, OH, OW, 3) uint8
        ref = _to_bf16(torch.from_numpy(want).permute(0, 3, 1, 2).contiguous().cuda())
        nd = int((got != ref).sum().item())
        assert nd == 0, f"{nd} of {ref.numel()} elements differ from cv2 INTER_LINEAR"


class TestProperties:
    def test_a_constant_image_is_a_constant_and_is_normalized(self):
        fvk = _fvk()
        for value in (0, 127, 128, 255):
            src = torch.full((1, 40, 40, 3), value, device="cuda", dtype=torch.uint8)
            out = torch.zeros(1, 3, 20, 20, device="cuda", dtype=BF16)
            fvk.resize_norm_u8_hwc_to_bf16_chw(src.data_ptr(), out.data_ptr(),
                                               1, 40, 40, 20, 20,
                                               torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            want = _to_bf16(torch.tensor(float(value))).item()
            assert int((out != want).sum().item()) == 0, f"value {value} -> {out.flatten()[0]}"

    def test_the_output_is_chw_not_hwc(self):
        """A kernel emitting HWC would still pass a constant-image check."""
        fvk = _fvk()
        torch.manual_seed(1)
        src = torch.zeros(1, 16, 16, 3, device="cuda", dtype=torch.uint8)
        src[..., 0] = 255          # red plane only
        src[..., 1] = 0
        src[..., 2] = 128
        out = torch.zeros(1, 3, 8, 8, device="cuda", dtype=BF16)
        fvk.resize_norm_u8_hwc_to_bf16_chw(src.data_ptr(), out.data_ptr(), 1, 16, 16, 8, 8,
                                           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert float(out[0, 0].float().mean()) > 0.9      # channel 0 is the 255 plane
        assert float(out[0, 1].float().mean()) < -0.9     # channel 1 is the 0 plane
        assert abs(float(out[0, 2].float().mean())) < 0.05  # channel 2 is 128

    def test_the_views_are_independent(self):
        fvk = _fvk()
        torch.manual_seed(2)
        src = torch.randint(0, 256, (2, 32, 32, 3), device="cuda", dtype=torch.uint8)
        out = torch.zeros(2, 3, 16, 16, device="cuda", dtype=BF16)
        fvk.resize_norm_u8_hwc_to_bf16_chw(src.data_ptr(), out.data_ptr(), 2, 32, 32, 16, 16,
                                           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert not torch.equal(out[0], out[1])


class TestGuards:
    def test_null_pointers_fail_on_the_host(self):
        fvk = _fvk()
        src = torch.zeros(1, 8, 8, 3, device="cuda", dtype=torch.uint8)
        out = torch.zeros(1, 3, 4, 4, device="cuda", dtype=BF16)
        with pytest.raises(RuntimeError, match="null pointer"):
            fvk.resize_norm_u8_hwc_to_bf16_chw(0, out.data_ptr(), 1, 8, 8, 4, 4,
                                               torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()   # the context is still usable

    def test_empty_shapes_are_a_no_op(self):
        fvk = _fvk()
        src = torch.zeros(0, 8, 8, 3, device="cuda", dtype=torch.uint8)
        out = torch.zeros(0, 3, 4, 4, device="cuda", dtype=BF16)
        fvk.resize_norm_u8_hwc_to_bf16_chw(src.data_ptr(), out.data_ptr(), 0, 8, 8, 4, 4,
                                           torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
