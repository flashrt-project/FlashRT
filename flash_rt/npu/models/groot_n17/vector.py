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


class DitVectorLibrary:
    """Vector-only translation unit, and its own shared object.

    Each Ascend unit here is loaded separately and honours its own environment
    override, so each carries the ABI pair and each loader checks it.
    """

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_DIT_VECTOR_LIBRARY")
        path = path or Path(__file__).parents[2] / "lib" / "libflashrt_npu_dit_vector.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                "Build the Ascend kernels with scripts/npu/build.sh before NPU "
                "graph construction") from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, "DiT elementwise")
        self.add_layer_norm = self.library.flashrt_npu_dit_add_layer_norm
        self.add_layer_norm.argtypes = [C.c_void_p] * 7 + [C.c_int] * 2 + [C.c_float]
        self.add_layer_norm.restype = C.c_int


_library = None


def library() -> DitVectorLibrary:
    global _library
    if _library is None:
        _library = DitVectorLibrary()
    return _library


MAX_NORM_COLS = 2048


def serves(x: torch.Tensor) -> bool:
    """Whether the native norm can take this tensor.

    The broadcast of the mean and the reciprocal square root rides a 64-element
    repeat, so the row has to be a whole number of them, and the row is held in
    UB in FP32.
    """
    return (x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.shape[-1] % 64 == 0 and x.shape[-1] <= MAX_NORM_COLS)


def add_layer_norm(residual: torch.Tensor, branch: torch.Tensor, gamma: torch.Tensor,
                   beta: torch.Tensor, eps: float):
    """``residual + branch``, and the affine LayerNorm of the sum.

    Returns ``(normalised, sum)``, the same pair and the same arithmetic as the
    vendor's fused form: the sum is rounded to BF16 before it is normalised,
    because the sum is what the next block carries forward.
    """
    shape = residual.shape
    rows, cols = residual.numel() // shape[-1], shape[-1]
    norm = torch.empty_like(residual)
    total = torch.empty_like(residual)
    code = library().add_layer_norm(
        torch.npu.current_stream(residual.device).npu_stream,
        residual.data_ptr(), branch.data_ptr(), gamma.data_ptr(), beta.data_ptr(),
        norm.data_ptr(), total.data_ptr(), rows, cols, float(eps))
    if code:
        raise RuntimeError(f"native add-layer-norm rejected arguments: {code}")
    return norm, total
