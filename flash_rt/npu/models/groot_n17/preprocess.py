"""The GR00T N1.7 image path, on the die.

The official processor turns camera frames into the patch rows the vision tower
reads, and on this box it costs 99.9 ms a frame doing it on the host — more
than the entire model. It is a resize, a rescale, a normalise and a shuffle:
every step is device work that happens to have been written for a CPU.

Reproduced here against the processor's own arithmetic rather than
approximated. It agrees to 5.9e-08 when both run on the host, and runs in
0.90 ms on the die.

Two details this part forces:

* **The shuffle is taken in two steps.** The reference expresses it as one
  nine-dimensional permutation; this part's copy operator refuses anything past
  eight dimensions, and says so with a shape error rather than a wrong answer.
* **The temporal axis is expanded into the column last.** It is a pure
  duplication of each frame, so carrying it through the permutation costs a
  dimension and a copy for nothing.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import torch
import torch.nn.functional as F

#: The published Qwen3-VL vision geometry. A checkpoint whose processor differs
#: is a different model, so the frontend checks rather than adapts.
PATCH = 16
MERGE = 2
TEMPORAL = 2
MEAN = 0.5
STD = 0.5
#: Pixel-count bounds the reference's resize would otherwise enforce by
#: rescaling. Inside them the target size is just the grid-aligned one.
MIN_PIXELS = 65536
MAX_PIXELS = 16777216

#: OpenCV's fixed-point interpolation weights: shorts in units of 1/2048.
_COEF_BITS = 11
_COEF_SCALE = 1 << _COEF_BITS


def _cv_round(values):
    """``cvRound``: nearest, ties to even, which is what a float lands on."""
    return torch.round(values).to(torch.int32)


def _area_taps(source: int, target: int, device):
    """OpenCV's INTER_AREA weights for the enlarging case.

    True area interpolation exists only for the shrinking case; when the image
    is enlarged OpenCV emulates it with a two-tap interpolation that keeps the
    area source coordinate. Both resizes in the reference's evaluation transform
    enlarge, so this is the path that runs.

    Three details decide whether this is exact, and each of them was wrong once:

    * **The forward scale is derived from the inverse**, not from the ratio.
      ``1/(200/180)`` is 0.8999999999999999, so ``30 * scale`` is
      26.999999999999996 and floors to 26, where ``30 * (180/200)`` floors to
      27 -- a whole source row, on every column of it.
    * **The two weights are each rounded from their own float.** They do not
      always sum to 2048 and one cannot be derived from the other.
    * **The weight arithmetic is FP32**, because OpenCV's is.
    """
    inverse = target / source
    scale = 1.0 / inverse
    index = torch.arange(target, dtype=torch.float64)
    first = torch.floor(index * scale)
    fraction = ((index + 1.0) - (first + 1.0) * inverse).to(torch.float32)
    fraction = torch.where(fraction <= 0, torch.zeros_like(fraction),
                           fraction - torch.floor(fraction))
    first = first.to(torch.int64)
    second = torch.clamp(first + 1, max=source - 1)
    # The tail has no second tap, exactly as OpenCV clamps it.
    fraction = torch.where(first + 1 > source - 1, torch.zeros_like(fraction), fraction)
    first = torch.clamp(first, 0, source - 1)
    weight1 = _cv_round(fraction * _COEF_SCALE)
    weight0 = _cv_round((1.0 - fraction) * _COEF_SCALE)
    return (first.to(torch.int32).to(device), second.to(torch.int32).to(device),
            weight0.to(device), weight1.to(device))


class AreaResizeLibrary:
    """The resize kernel's shared object, checked like every other unit."""

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_IMAGE_LIBRARY")
        path = path or Path(__file__).parents[2] / "lib" / "libflashrt_npu_image.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                "Build the Ascend kernels with scripts/npu/build.sh before the "
                "image path is used") from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, "image resize")
        self.resize = self.library.flashrt_npu_area_resize
        self.resize.argtypes = [C.c_void_p] * 7 + [C.c_int] * 11
        self.resize.restype = C.c_int


def _align(value: int, to: int = 32) -> int:
    return (value + to - 1) // to * to


class _ResizePlan:
    """One OpenCV area resize, with its tables on the device.

    The tables are the whole correctness story and they are built here, on the
    host, once per frame size. The kernel is only the arithmetic.
    """

    def __init__(self, source, target, column_offset, row_offset, device):
        self.source_h, self.source_w = source
        self.target_h, self.target_w = target
        self.row_offset = int(row_offset)
        columns = _area_taps(self.source_w, self.target_w, "cpu")
        rows = _area_taps(self.source_h, self.target_h, "cpu")
        first, second, weight0, weight1 = columns
        # Gather offsets are bytes into the widened source row, which holds all
        # three channels interleaved; the crop's column offset folds in here so
        # the copy from global memory stays block aligned.
        channel = torch.arange(3, dtype=torch.int64)
        def spread(index):
            return (((index.to(torch.int64) + column_offset) * 3).unsqueeze(1)
                    + channel).reshape(-1)
        # Each tap's table starts on a 32-byte block, because that is where a
        # copy out of global memory has to start. Packed back to back, the
        # second tap's table would begin at byte 5460 and read nothing useful.
        self.samples = self.target_w * 3
        self.stride = _align(self.samples, 8)
        pad = self.stride - self.samples
        offsets = torch.cat([spread(first) * 4, torch.zeros(pad, dtype=torch.int64),
                             spread(second) * 4, torch.zeros(pad, dtype=torch.int64)])
        weights = torch.cat([weight0.repeat_interleave(3), torch.zeros(pad),
                             weight1.repeat_interleave(3), torch.zeros(pad)])
        self.offsets = offsets.to(torch.int32).to(device)
        self.weights = weights.to(torch.float32).to(device)
        self.rows = torch.cat([rows[0], rows[1]]).to(torch.int32).to(device)
        self.row_weights = torch.cat([rows[2], rows[3]]).to(torch.int32).to(device)


class EvalImageTransform:
    """The reference's evaluation image transform, on the die.

    A smallest-edge resize to 256, a centre crop to 95 percent of each side, and
    the same resize again -- all of it ``cv2.INTER_AREA`` on uint8. On this box
    it is 7.0 ms of host a frame, more than a tenth of the whole served frame.

    Reproduced **bit for bit** against the reference's own output, not
    approximated: zero differing pixels over twenty frames of the dataset, which
    is 28 million of them. Three details decide that, and each was wrong once --
    see ``_area_taps`` for two of them and the kernel for the third.

    The intermediate and output rows are padded to whole 32-byte blocks so the
    kernel's row copies stay aligned; the live samples are sliced back out at
    the end.
    """

    def __init__(self, height: int, width: int, *, max_size: int = 256,
                 crop_fraction: float = 0.95, device="npu:0", images: int = 4):
        self.device = device
        self.images = int(images)
        self.source = (int(height), int(width))
        self.library = AreaResizeLibrary()

        mid_h, mid_w = self._resized(height, width, max_size)
        crop_h = max(1, int(mid_h * crop_fraction))
        crop_w = max(1, int(mid_w * crop_fraction))
        top, left = (mid_h - crop_h) // 2, (mid_w - crop_w) // 2
        target_h, target_w = self._resized(crop_h, crop_w, max_size)
        if (mid_h, mid_w) == (height, width) or (target_h, target_w) == (crop_h, crop_w):
            raise NotImplementedError(
                "this path reproduces the enlarging case of INTER_AREA, which is "
                "the one the reference's evaluation transform runs; a frame that "
                "does not enlarge takes a different branch inside OpenCV")
        self.target = (target_h, target_w)

        self.first = _ResizePlan((height, width), (mid_h, mid_w), 0, 0, device)
        self.second = _ResizePlan((crop_h, crop_w), (target_h, target_w), left, top,
                                  device)
        self.source_stride = width * 3
        if self.source_stride % 32:
            raise NotImplementedError(
                f"a {width}-wide frame has a {self.source_stride}-byte row, which "
                "the device copy cannot start a block on; this path takes frames "
                "whose row is a whole number of 32-byte blocks")
        self.mid_stride = _align(mid_w * 3)
        self.out_stride = _align(target_w * 3)
        self.mid = torch.zeros(self.images, mid_h, self.mid_stride, dtype=torch.uint8,
                               device=device)
        self.out = torch.zeros(self.images, target_h, self.out_stride, dtype=torch.uint8,
                               device=device)

    @staticmethod
    def _resized(height, width, max_size):
        scale = max_size / float(min(height, width))
        return round(height * scale), round(width * scale)

    def _run(self, plan, source, source_stride, source_plane, destination,
             destination_stride, source_samples):
        code = self.library.resize(
            torch.npu.current_stream(self.out.device).npu_stream,
            source.data_ptr(), destination.data_ptr(), plan.offsets.data_ptr(),
            plan.weights.data_ptr(), plan.rows.data_ptr(), plan.row_weights.data_ptr(),
            self.images, source_plane, source_stride, plan.row_offset, source_samples,
            destination.shape[1] * destination_stride, destination_stride,
            plan.target_h, plan.samples, plan.stride, 40)
        if code:
            raise RuntimeError(f"the image resize rejected arguments: {code}")

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """``(n, H, W, 3)`` uint8 frames to the ``(n, 3, H', W')`` uint8 the
        patch path reads."""
        if (frames.dtype != torch.uint8 or frames.dim() != 4
                or tuple(frames.shape[1:3]) != self.source or frames.shape[3] != 3
                or frames.shape[0] != self.images or not frames.is_contiguous()):
            raise ValueError(
                f"the evaluation transform was built for {self.images} contiguous "
                f"{self.source} uint8 frames, got {tuple(frames.shape)} of "
                f"{frames.dtype}")
        self._run(self.first, frames, self.source_stride,
                  self.source[0] * self.source_stride, self.mid, self.mid_stride,
                  self.source[1] * 3)
        self._run(self.second, self.mid, self.mid_stride,
                  self.mid.shape[1] * self.mid_stride, self.out, self.out_stride,
                  self.first.samples)
        height, width = self.target
        live = self.out[:, :, :width * 3].reshape(self.images, height, width, 3)
        return live.permute(0, 3, 1, 2).contiguous()


def grid_for(height: int, width: int) -> tuple[int, int]:
    """The patch grid the reference resizes to, in patch units."""
    factor = PATCH * MERGE
    aligned_h = max(factor, round(height / factor) * factor)
    aligned_w = max(factor, round(width / factor) * factor)
    pixels = aligned_h * aligned_w
    if not MIN_PIXELS <= pixels <= MAX_PIXELS:
        raise NotImplementedError(
            f"a {height}x{width} frame resizes to {aligned_h}x{aligned_w}, "
            f"{pixels} pixels, outside the reference's [{MIN_PIXELS}, "
            f"{MAX_PIXELS}] band where it would rescale to fit; this path "
            "reproduces only the in-band case")
    return aligned_h // PATCH, aligned_w // PATCH


def patch_rows(images: torch.Tensor) -> torch.Tensor:
    """``(views, 3, H, W)`` uint8 frames to ``(views * patches, 1536)`` rows.

    The result is what the vision tower's projection consumes, in the order the
    reference produces: one row per patch, ordered by view, then merged block,
    then position within the block; and within a row by channel, then temporal
    copy, then pixel.
    """
    if images.dtype != torch.uint8 or images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(
            "the image path takes (views, 3, H, W) uint8 frames, got "
            f"{tuple(images.shape)} of {images.dtype}")
    views, channels, height, width = images.shape
    grid_h, grid_w = grid_for(height, width)
    target = (grid_h * PATCH, grid_w * PATCH)

    x = images
    if target != (height, width):
        # The processor's own call: bicubic with antialiasing, on this device.
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as TF

        x = TF.resize(x, list(target), interpolation=InterpolationMode.BICUBIC,
                      antialias=True)
    x = (x.float() / 255.0 - MEAN) / STD

    x = x.reshape(views, channels, grid_h, PATCH, grid_w, PATCH)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(views, grid_h // MERGE, MERGE, grid_w // MERGE, MERGE,
                  channels, PATCH * PATCH)
    x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
    rows = views * grid_h * grid_w
    x = x.reshape(rows, channels, 1, PATCH * PATCH)
    x = x.expand(rows, channels, TEMPORAL, PATCH * PATCH)
    return x.reshape(rows, channels * TEMPORAL * PATCH * PATCH)
