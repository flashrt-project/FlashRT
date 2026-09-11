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
