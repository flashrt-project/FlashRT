"""Native multi-head attention for the GR00T N1.7 DiT.

The vendor prompt flash-attention operator charges 42 to 49 us per call at
every one of this model's three DiT geometries, and the cost barely moves
between 13 keys and 448 -- it is the operator's fixed cost, not the work. The
41x41 case moves 378 KB and does 10 MFLOP, so its physical floor is a launch
plus about a microsecond. This binds a kernel written for that gap.

Two things the kernel asks of its caller:

* **V comes in one of two layouts.** A raw ``Mmad`` B operand takes its source
  in ``(N, K)`` form, and the second GEMM's B operand is V with K on the key
  axis, so ``(heads * head_dim, keys)`` needs no conversion -- which is free
  wherever V is a frame constant, as it is in every cross-attention layer of
  this model. Where it is not, pass V exactly as the projection wrote it,
  ``(keys, heads * head_dim)`` with a row pitch, and the kernel transposes it on
  the way from L1 to L0B: producing the transposed form for a self-attention
  layer cost a slice and a permute, 17 us a layer for 147 KB, because at this
  size both are fixed cost.

* **Every operand is padded to the fractal with real zeros.** Nd2Nz fills only
  the rows it is given, so a tail the kernel pretends is zero is actually
  uninitialised L1 that the Mmad then reads. ``buffers()`` hands out the padded
  buffers and the caller writes the live rows into the views it returns; the
  padding is written once, at allocation, and never again.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import torch

from flash_rt.npu.core import operands


class DitAttentionLibrary:
    """The kernel drives ``Mmad`` on the cube and the softmax on the vector
    cores inside one launch, so it is a mixed translation unit and cannot share
    a library with the cube-only units or with the Pi0.5 decode attention.
    """

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_DIT_ATTENTION_LIBRARY")
        path = path or Path(__file__).parents[2] / "lib" / "libflashrt_npu_dit_attn.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                "Build the Ascend kernels with scripts/npu/build.sh before NPU "
                "graph construction") from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, "DiT attention")
        self.launch = self.library.flashrt_npu_dit_attn
        self.launch.argtypes = [C.c_void_p] * 8 + [C.c_int] * 7 + [C.c_float]
        self.launch.restype = C.c_int


def _blocks(value: int, block: int = 16) -> int:
    return (value + block - 1) // block * block


class DitAttention:
    """One geometry's attention, with its scratch and its launch width fixed.

    Scratch lives here rather than in the kernel because a captured graph may
    not allocate: the score and probability planes, and the FP32 context the
    cube writes before the vector cores round it, are sized once at setup.

    Sites with the same geometry share an instance. The planes are written and
    consumed inside one launch, so two sites cannot be in flight at once and
    reuse is ordering rather than aliasing.
    """

    def __init__(self, heads: int, queries: int, keys: int, head_dim: int,
                 *, device="npu:0", scale: float | None = None):
        if head_dim % 16:
            raise ValueError(
                f"the native DiT attention needs a 16-aligned head width, got {head_dim}")
        self.heads = int(heads)
        self.queries = int(queries)
        self.keys = int(keys)
        self.head_dim = int(head_dim)
        self.scale = float(scale) if scale is not None else head_dim ** -0.5
        self.width = self.heads * self.head_dim
        self.rows, self.columns = _blocks(self.queries), _blocks(self.keys)
        rows, columns = self.rows, self.columns
        # Twenty cube cores; an even share beats a wider launch with one core
        # doing twice the heads of the rest.
        waves = (self.heads + 19) // 20
        self.cores = max(1, self.heads // waves)
        self.library = DitAttentionLibrary()
        self.scores = torch.zeros(self.heads * rows * columns, dtype=torch.float32,
                                  device=device)
        self.probs = torch.zeros(self.heads * rows * columns, dtype=torch.bfloat16,
                                 device=device)
        self.context = torch.zeros(self.heads * rows * self.head_dim,
                                   dtype=torch.float32, device=device)
        self.out = torch.zeros(rows, self.width, dtype=torch.bfloat16, device=device)

    def buffers(self, device=None):
        """Padded query, key and transposed-value buffers for one site.

        Returns the three buffers together with the live views to write into.
        The buffers are zero everywhere; a caller that only ever writes the
        views keeps the fractal padding zero for the life of the graph.
        """
        device = device or self.out.device
        query = torch.zeros(self.rows, self.width, dtype=torch.bfloat16, device=device)
        key = torch.zeros(self.columns, self.width, dtype=torch.bfloat16, device=device)
        value_t = torch.zeros(self.width, self.columns, dtype=torch.bfloat16,
                              device=device)
        return (query, key, value_t), (query[:self.queries], key[:self.keys],
                                       value_t[:, :self.keys])

    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 stride: int | None = None,
                 value_stride: int = 0) -> torch.Tensor:
        """``stride`` is the query and key row pitch, for when they are column
        slices of a wider buffer that one GEMM produced. ``value_stride`` says
        the same of the value and, by being non-zero, that the value is in the
        layout the projection wrote rather than transposed."""
        stride = self.width if stride is None else int(stride)
        value_stride = int(value_stride)
        # Raw addresses again: every operand is checked before any of them is
        # taken, and the checks name the operand because "pad with buffers()" is
        # only useful advice if the caller knows which one was wrong.
        device = self.out.device
        for name, tensor, rows in (("query", query, self.rows),
                                   ("key", key, self.columns)):
            operands.require(tensor, name, device=device, dtype=torch.bfloat16,
                             shape=(rows, self.width), row_pitch=stride)
        if value_stride:
            operands.require(value, "value", device=device, dtype=torch.bfloat16,
                             shape=(self.columns, self.width),
                             row_pitch=value_stride)
        else:
            operands.require(value, "value (transposed)", device=device,
                             dtype=torch.bfloat16,
                             shape=(self.width, self.columns), contiguous=True)
        code = self.library.launch(
            torch.npu.current_stream(self.out.device).npu_stream,
            query.data_ptr(), key.data_ptr(), value.data_ptr(), self.out.data_ptr(),
            self.scores.data_ptr(), self.probs.data_ptr(), self.context.data_ptr(),
            self.heads, self.queries, self.keys, self.head_dim, self.cores, stride,
            value_stride, self.scale)
        if code:
            raise RuntimeError(f"native DiT attention rejected arguments: {code}")
        return self.out
