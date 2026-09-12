"""The action head's fused add-and-normalise.

A draw on time against the vendor's, and it is here for its arithmetic: the
residual sum is rounded to BF16 once, because that value is what the next block
carries forward, and the normalisation runs in FP32 from it rather than
rounding a second time in between.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import torch

from flash_rt.npu.core import operands


class DitNormLibrary:
    """Vector-only translation unit, and its own shared object.

    Each Ascend unit here is loaded separately and honours its own environment
    override, so each carries the ABI pair and each loader checks it.
    """

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_DIT_NORM_LIBRARY")
        path = path or Path(__file__).parents[2] / "lib" / "libflashrt_npu_dit_norm.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                "Build the Ascend kernels with scripts/npu/build.sh before NPU "
                "graph construction") from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, "DiT add-and-normalise")
        self.add_layer_norm = self.library.flashrt_npu_dit_add_layer_norm
        self.add_layer_norm.argtypes = [C.c_void_p] * 8 + [C.c_int] * 3 + [C.c_float]
        self.add_layer_norm.restype = C.c_int


_library = None


def library() -> DitNormLibrary:
    global _library
    if _library is None:
        _library = DitNormLibrary()
    return _library


MAX_NORM_COLS = 2048


def serves(x: torch.Tensor) -> bool:
    """Whether the native norm can take this tensor.

    The broadcast of the mean and the reciprocal square root rides a 64-element
    repeat, so the row has to be a whole number of them, and the row is held in
    UB in FP32. A tensor that is not on an Ascend device is not served either:
    routing a host tensor here would hand its address to a kernel.
    """
    return (x.device.type == "npu" and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.shape[-1] % 64 == 0
            and x.shape[-1] <= MAX_NORM_COLS)


def add_layer_norm(residual: torch.Tensor, branch: torch.Tensor, gamma: torch.Tensor,
                   beta: torch.Tensor, eps: float, branch_bias=None, out=None):
    """``residual + branch``, and the affine LayerNorm of the sum.

    Returns ``(normalised, sum)``, the same pair and the same arithmetic as the
    vendor's fused form: the sum is rounded to BF16 before it is normalised,
    because the sum is what the next block carries forward.

    ``branch_bias`` is the bias of the projection that produced ``branch``, added
    here instead of by that projection. It is the same number added to the same
    sum in FP32, and it saves a launch: a biased matmul casts its bias on every
    call, as its own kernel, for numbers that never change.

    ``out`` is a preallocated destination for the normalised row, wider than the
    row itself. The extra columns are the caller's -- typically a constant one, so
    that the *next* projection can carry its bias as one more input channel -- and
    this only ever writes the first ``cols`` of each row.
    """
    if not serves(residual):
        raise ValueError(
            f"the native norm takes a contiguous BF16 row on an Ascend device "
            f"that is a whole number of 64 elements and at most {MAX_NORM_COLS} "
            f"wide, got {tuple(residual.shape)} of {residual.dtype} on "
            f"{residual.device}")
    shape = residual.shape
    rows, cols = residual.numel() // shape[-1], shape[-1]
    device = operands.require_npu(residual, "residual")
    # Every address this launch is handed, checked before any of them is taken.
    # These are raw pointers: a host tensor, a mismatched width or a strided row
    # is a device fault or a silent read of the wrong memory, not a wrong number.
    operands.require(residual, "residual", device=device, dtype=torch.bfloat16,
                     contiguous=True)
    operands.require(branch, "branch", device=device, dtype=torch.bfloat16,
                     shape=shape, contiguous=True)
    for name, affine in (("gamma", gamma), ("beta", beta)):
        operands.require(affine, name, device=device, dtype=torch.bfloat16,
                         shape=(cols,), contiguous=True)
    if branch_bias is not None:
        operands.require(branch_bias, "branch_bias", device=device,
                         dtype=torch.bfloat16, shape=(cols,), contiguous=True)
    if out is None:
        norm, pitch = torch.empty_like(residual), cols
    else:
        pitch = out.shape[-1]
        if pitch < cols or (pitch - cols) % 16:
            raise ValueError(
                f"the normalised row is written at a pitch, which must be at "
                f"least {cols} and a multiple of 16 past it, got {pitch}")
        operands.require(out, "out", device=device, dtype=torch.bfloat16,
                         shape=shape[:-1] + (pitch,), contiguous=True)
        norm = out
    total = torch.empty_like(residual)
    code = library().add_layer_norm(
        torch.npu.current_stream(residual.device).npu_stream,
        residual.data_ptr(), branch.data_ptr(), gamma.data_ptr(), beta.data_ptr(),
        norm.data_ptr(), total.data_ptr(),
        0 if branch_bias is None else branch_bias.data_ptr(),
        rows, cols, pitch, float(eps))
    if code:
        raise RuntimeError(f"native add-layer-norm rejected arguments: {code}")
    return norm, total
