"""Setup-owned INT8 decoder projections and their producer-side kernels.

Every decoder projection runs at ten action rows, where the cost is the weight
stream and nothing else, so halving the weight width is the whole optimisation.
The projections only pay if the frame does not grow a kernel to feed or drain
them, so the quantise rides in the AdaRMS and the GELU that already write the
activation, and the FP16 the cube emits is consumed by the rotary and the
residual as it stands.

Quantisation parameters are frozen at setup: the weights are per-output-channel
INT8, the activations use one scale per (layer, denoise step, projection), and
the dequant vector folds the two together so replay never measures anything.
"""
import ctypes as C
import os
from dataclasses import dataclass
from pathlib import Path

from flash_rt.npu.core.native_kernels import _NativeLibrary

# NT * KT is one half of L0B. A K-split only accumulates correctly at that size,
# and a single K step is free to use less.
_L0B_HALF = 32768
# Fixpipe reads the accumulator in 16-row fractals.
_L0C_ROWS = 16
# Two cube cores writing through fixpipe into the same aligned 512-byte region
# of GM corrupt each other's bytes. It is silent, invisible on an idle device,
# and only a real frame's memory traffic exposes it. Tiles are therefore handed
# out in groups whose combined row write covers one whole region, so a region
# belongs to exactly one core.
_WRITE_GRAIN = 512
# Grouping is ruinous where the output row is narrow: at 1024 columns a row is
# only four regions, so grouping drops the kernel to four cores. Those
# projections write fractal NZ instead, where a tile's block is 16 x NT x 2
# bytes, contiguous and aligned whatever the row width, so every tile can be
# handed out on its own. Which sites want it is measured, not derived, and the
# callers opt in one at a time.


class DecoderGemmLibrary:
    """The cube-only INT8 GEMM lives in its own translation unit.

    A mixed cube/vector unit wakes the vector cores on every launch, which the
    vector kernels in this family cannot afford, so the two are compiled apart.
    """

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_DECODER_LIBRARY")
        path = path or Path(__file__).parents[1] / "lib" / "libflashrt_npu_decoder.so"
        self.library = C.CDLL(str(path))
        self.launch = self.library.flashrt_npu_decoder_gemm
        self.launch.argtypes = [C.c_void_p] * 5 + [C.c_int] * 9
        self.launch.restype = C.c_int


class DecoderQuantKernels(_NativeLibrary):
    """AdaRMS with an INT8 norm, gated GELU with an INT8 output, FP16 rotary."""

    def __init__(self):
        super().__init__()
        self.ada = self.library.flashrt_npu_gated_ada_quant
        self.ada.argtypes = [C.c_void_p] * 9 + [C.c_int] * 5
        self.ada.restype = C.c_int
        self.geglu = self.library.flashrt_npu_geglu_quant
        self.geglu.argtypes = [C.c_void_p] * 4 + [C.c_int] * 2
        self.geglu.restype = C.c_int
        self.rope = self.library.flashrt_npu_decoder_rope_fp16
        self.rope.argtypes = [C.c_void_p] * 7 + [C.c_int] * 2
        self.rope.restype = C.c_int
        self.rope_vt = self.library.flashrt_npu_decoder_rope_vt
        self.rope_vt.argtypes = [C.c_void_p] * 7 + [C.c_int] * 2
        self.rope_vt.restype = C.c_int
        self.vt_prefix = self.library.flashrt_npu_vt_prefix
        self.vt_prefix.argtypes = [C.c_void_p] * 3 + [C.c_int] * 2
        self.vt_prefix.restype = C.c_int

    def gated_ada(self, residual, branch, gate, gamma, shift, inverse=None,
                  branch_nz=False):
        """Returns (norm, updated); norm is INT8 when an inverse scale is given."""
        import torch
        if (residual.dtype != torch.float32 or residual.ndim != 2
                or residual.shape[1] != 1024 or not residual.is_contiguous()
                or residual.device.type != "npu"):
            raise ValueError("gated AdaRMS requires a contiguous FP32 NPU (rows,1024) residual")
        quantize = inverse is not None
        checks = [(gamma, torch.bfloat16, (1024,)), (shift, torch.float32, (1024,))]
        if branch is not None:
            if branch.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("the AdaRMS branch must be BF16 or FP16")
            # A fractal-NZ branch is one flat run of 16-row blocks, so it is
            # 16 * D elements whatever the row count, and the kernel gathers a
            # row out of it with one strided copy.
            shape = (16 * 1024,) if branch_nz else residual.shape
            checks += [(branch, branch.dtype, shape), (gate, torch.bfloat16, (1024,))]
        if quantize:
            checks.append((inverse, torch.float32, inverse.shape))
            if inverse.numel() != 1:
                raise ValueError("the activation scale is a single frozen value")
        for tensor, dtype, shape in checks:
            if (tensor is None or tensor.dtype != dtype or tensor.shape != shape
                    or tensor.device != residual.device or not tensor.is_contiguous()):
                raise ValueError("invalid gated AdaRMS input, style or scale")
        norm = torch.empty(residual.shape, device=residual.device,
                           dtype=torch.int8 if quantize else torch.bfloat16)
        updated = torch.empty_like(residual) if branch is not None else residual
        code = self.ada(torch.npu.current_stream(residual.device).npu_stream,
            residual.data_ptr(), branch.data_ptr() if branch is not None else None,
            gate.data_ptr() if gate is not None else None, gamma.data_ptr(),
            shift.data_ptr(), norm.data_ptr(), updated.data_ptr(),
            inverse.data_ptr() if quantize else None, residual.shape[0],
            int(branch is not None),
            int(branch is not None and branch.dtype == torch.float16), int(quantize),
            int(branch_nz))
        if code:
            raise RuntimeError(f"native gated AdaRMS quantization rejected arguments: {code}")
        return norm, updated

    def gated_gelu(self, gate_up, inverse, out):
        """Tanh GELU of the left half times the right half, frozen INT8 out."""
        import torch
        columns = out.shape[1]
        if (gate_up.dtype != torch.float16 or gate_up.ndim != 2
                or gate_up.shape != (out.shape[0], 2 * columns)
                or not gate_up.is_contiguous() or gate_up.device.type != "npu"):
            raise ValueError("gated GELU expects a contiguous FP16 NPU (rows,2H) slab")
        if (out.dtype != torch.int8 or not out.is_contiguous() or columns % 32
                or out.device != gate_up.device):
            raise ValueError("gated GELU writes a contiguous INT8 (rows,H) matrix")
        if (inverse.dtype != torch.float32 or inverse.numel() != 1
                or inverse.device != gate_up.device or not inverse.is_contiguous()):
            raise ValueError("gated GELU requires one frozen FP32 activation scale")
        code = self.geglu(torch.npu.current_stream(out.device).npu_stream,
                          gate_up.data_ptr(), inverse.data_ptr(), out.data_ptr(),
                          out.shape[0], columns)
        if code:
            raise RuntimeError(f"native gated GELU quantization rejected arguments: {code}")
        return out

    def decoder_rope(self, qkv, cos, sin, query, keys, values, prefix):
        """Same rotation as the shipped rotary kernel, over an FP16 QKV slab."""
        import torch
        rows = query.shape[0]
        if (qkv.dtype != torch.float16 or qkv.ndim != 2 or qkv.shape != (rows, 2560)
                or not qkv.is_contiguous() or qkv.device.type != "npu"):
            raise ValueError("decoder rotary expects a contiguous FP16 NPU (rows,2560) slab")
        if prefix < 0 or rows <= 0:
            raise ValueError("invalid KV prefix or action row count")
        for tensor, dtype, columns in ((cos, torch.float32, 256), (sin, torch.float32, 256),
                                       (keys, torch.bfloat16, 256), (values, torch.bfloat16, 256),
                                       (query, torch.bfloat16, 2048)):
            if (tensor.dtype != dtype or tensor.ndim != 2 or tensor.shape[1] != columns
                    or tensor.device != qkv.device or not tensor.is_contiguous()):
                raise ValueError("invalid rotary table, query or KV buffer")
        for buffer in (keys, values):
            if buffer.shape[0] < prefix + rows:
                raise ValueError("the KV buffers must cover the appended rows")
        code = self.rope(torch.npu.current_stream(qkv.device).npu_stream,
            qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), query.data_ptr(),
            keys.data_ptr(), values.data_ptr(), prefix, rows)
        if code:
            raise RuntimeError(f"native FP16 decoder rotary rejected arguments: {code}")
        return query

    def decoder_rope_transposed(self, qkv, cos, sin, query, keys, values, position):
        """``decoder_rope`` for the cache that keeps V transposed in fractal NZ.

        The rotation and its FP16 input are identical; what differs is where
        the two cache halves land. Keys take rows ``[0, rows)`` because the
        action suffix leads that cache, and values are transposed into fractal
        block zero. ``position`` is still the real rotary position of the first
        action row, so the numbers are the same ones.
        """
        import torch
        rows = query.shape[0]
        if (qkv.dtype != torch.float16 or qkv.ndim != 2 or qkv.shape != (rows, 2560)
                or not qkv.is_contiguous() or qkv.device.type != "npu"):
            raise ValueError("decoder rotary expects a contiguous FP16 NPU (rows,2560) slab")
        if position < 0 or rows <= 0 or rows > 16:
            raise ValueError("invalid rotary position or action row count")
        for tensor, dtype, columns in ((cos, torch.float32, 256), (sin, torch.float32, 256),
                                       (keys, torch.bfloat16, 256), (query, torch.bfloat16, 2048)):
            if (tensor.dtype != dtype or tensor.ndim != 2 or tensor.shape[1] != columns
                    or tensor.device != qkv.device or not tensor.is_contiguous()):
                raise ValueError("invalid rotary table, query or key buffer")
        if (values.dtype != torch.bfloat16 or values.ndim != 1
                or values.numel() < 16 * 256 or not values.is_contiguous()):
            raise ValueError("the transposed value cache must be a flat contiguous BF16 buffer")
        if keys.shape[0] < rows:
            raise ValueError("the key buffer must cover the appended rows")
        code = self.rope_vt(torch.npu.current_stream(qkv.device).npu_stream,
            qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), query.data_ptr(),
            keys.data_ptr(), values.data_ptr(), position, rows)
        if code:
            raise RuntimeError(f"native transposed decoder rotary rejected arguments: {code}")
        return query

    def transpose_prefix(self, source, values, rows: int, column: int):
        """One layer of encoder prefix values into the NZ transpose, starting
        at cache column ``column``. Replaces the straight copy the row-major
        cache took, so the frame does not grow a launch for the layout.
        """
        import torch
        if (source.dtype != torch.bfloat16 or source.ndim != 2 or source.shape[1] != 256
                or not source.is_contiguous() or source.device.type != "npu"):
            raise ValueError("the prefix transpose expects a contiguous BF16 (rows,256) source")
        if (values.dtype != torch.bfloat16 or values.ndim != 1 or not values.is_contiguous()
                or values.device != source.device):
            raise ValueError("the transposed value cache must be a flat contiguous BF16 buffer")
        if rows <= 0 or rows > source.shape[0] or column < 0 or column % 16:
            raise ValueError("invalid prefix row count or 16-aligned cache column")
        if values.numel() < (column + rows) * 256:
            raise ValueError("the transposed value cache must cover the prefix")
        code = self.vt_prefix(torch.npu.current_stream(source.device).npu_stream,
                              source.data_ptr(), values.data_ptr(), rows, column)
        if code:
            raise RuntimeError(f"native prefix transpose rejected arguments: {code}")
        return values


def column_tile(k: int) -> tuple:
    """Column and depth tile for a projection with reduction width ``k``.

    Thirty-two columns is the measured optimum at every decoder shape, and the
    depth then follows: a K split only accumulates correctly when the L0B slot
    is exactly half the buffer, and a narrower tile faults the accumulator.
    """
    if k <= 0 or k % 32:
        raise ValueError("the reduction width must be a positive multiple of 32")
    columns = 32
    depth = min(k, _L0B_HALF // columns)
    while depth > 0 and k % depth:
        depth //= 2
    if depth < 32:
        raise ValueError(f"no valid depth tile for a reduction width of {k}")
    return columns, depth


@dataclass(frozen=True)
class DecoderInt8Projection:
    """One projection: INT8 weight, per-(step) dequant vectors, an FP16 result.

    ``dequant`` holds, per denoise step, the product of the frozen activation
    scale and the per-output-channel weight scale, in the FP32 bit pattern the
    fixpipe quantiser reads. ``inverse`` is what the producing kernel multiplies
    by before rounding to INT8.
    """

    weight: object
    dequant: object
    inverse: object
    out: object
    library: object
    columns: int
    depth: int
    nz: bool
    rows: int

    @classmethod
    def bind(cls, weight, activation_amax, rows, nz=False, out=None):
        """Freeze a stored ``(N, K)`` BF16 weight against per-step activations."""
        import numpy as np
        import torch
        if weight.ndim != 2 or weight.device.type != "npu":
            raise ValueError("a decoder projection is a two-dimensional NPU weight")
        n, k = weight.shape
        if n % 16 or k % 32:
            raise ValueError("decoder projections need N divisible by 16 and K by 32")
        amax = np.asarray(activation_amax, dtype=np.float64).reshape(-1)
        if amax.size == 0 or not np.isfinite(amax).all() or (amax <= 0).any():
            raise ValueError("every denoise step needs a positive finite activation maximum")
        if int(rows) <= 0:
            raise ValueError("row count must be positive")
        scale = (weight.float().abs().amax(dim=1) / 127.0).clamp_min(1e-8)
        packed = (weight.float() / scale[:, None]).round().clamp(-127, 127)
        packed = packed.to(torch.int8).contiguous()
        activation = amax / 127.0
        columns_scale = scale.cpu().numpy().astype(np.float64)
        product = np.float32(columns_scale[None, :] * activation[:, None])
        dequant = torch.from_numpy(product.view(np.uint32).astype(np.int64)).to(weight.device)
        inverse = torch.tensor(1.0 / activation, dtype=torch.float32, device=weight.device)
        columns, depth = column_tile(k)
        if out is None:
            out = cls.result(n, int(rows), nz, weight.device)
        return cls(packed, dequant.contiguous(), inverse.contiguous(), out,
                   DecoderGemmLibrary(), columns, depth, nz, int(rows))

    @staticmethod
    def result(columns: int, rows: int, nz: bool, device):
        """The buffer a projection writes into.

        A fractal-NZ result is one flat run of 16-row blocks rather than a
        matrix: element (r, c) sits at (c // 16) * 256 + r * 16 + (c % 16),
        which is independent of the column tile that produced it.

        Every layer shares one of these per site. A projection's result is read
        by the next kernel and dead by the time the following layer reaches the
        same site, so the write-after-read between adjacent layers is real
        ordering rather than an accident, and eighteen separate buffers only
        remove it.
        """
        import torch
        shape = (_L0C_ROWS * columns,) if nz else (rows, columns)
        return torch.zeros(*shape, dtype=torch.float16, device=device)

    def __call__(self, activation, step: int):
        """INT8 ``activation`` times the frozen weight, dequantised to FP16."""
        import torch
        columns, k = self.weight.shape
        rows = self.rows
        if (activation.dtype != torch.int8 or activation.ndim != 2
                or activation.shape != (rows, k)
                or not activation.is_contiguous() or activation.device != self.out.device):
            raise ValueError("the decoder GEMM consumes a contiguous INT8 activation")
        if not 0 <= step < self.dequant.shape[0]:
            raise ValueError("denoise step is outside the calibrated range")
        code = self.library.launch(torch.npu.current_stream(self.out.device).npu_stream,
            activation.data_ptr(), self.weight.data_ptr(), self.dequant[step].data_ptr(),
            self.out.data_ptr(), rows, columns, k,
            self.columns, self.depth, _L0C_ROWS, 20, _WRITE_GRAIN, int(self.nz))
        if code:
            raise RuntimeError(f"native decoder GEMM rejected arguments: {code}")
        return self.out


@dataclass(frozen=True)
class DecoderInt8Pack:
    """Every frozen decoder projection plus the buffers the step reuses.

    The attention output is the one activation with no kernel of ours in front
    of it, so it takes the shipped row quantiser; the other three are quantised
    by the kernel that was already writing them.
    """

    layers: tuple
    kernels: object
    row_quant: object
    attention_scales: object
    attention_int8: object
    hidden: object
    query: object
    steps: int

    @classmethod
    def build(cls, weights, prefix, layers, amax, rows, steps):
        """``amax`` maps a projection name to a ``(layers, steps)`` array."""
        import numpy as np
        import torch
        from flash_rt.npu.core.native_kernels import RowQuantizer
        # Fractal NZ is worth it where the row is too narrow to hand whole
        # 512-byte regions to twenty cores: at 1024 columns a row is four
        # regions and grouping drops to four cores, while at 2560 and 8192 the
        # row-major write with grouping is free or nearly so.
        names = {"qkv": False, "self_attn.o_proj": True,
                 "gu": False, "mlp.down_proj": True}
        for name in names:
            block = np.asarray(amax[name], dtype=np.float64)
            if block.shape != (layers, steps):
                raise ValueError(f"'{name}' needs one activation maximum per layer and step")
        first = weights[f"{prefix}.0.qkv.weight"]
        shared = {name: DecoderInt8Projection.result(
            weights[f"{prefix}.0.{name}.weight"].shape[0], rows, nz, first.device)
            for name, nz in names.items()}
        bound = []
        for i in range(layers):
            bound.append({name: DecoderInt8Projection.bind(
                weights[f"{prefix}.{i}.{name}.weight"], np.asarray(amax[name])[i],
                rows, nz, shared[name]) for name, nz in names.items()})
        device = bound[0]["qkv"].weight.device
        attention = np.maximum(np.asarray(amax["self_attn.o_proj"], dtype=np.float64), 1e-12)
        scales = torch.tensor(np.repeat((127.0 / attention)[:, :, None], rows, axis=2),
                              dtype=torch.float32, device=device).contiguous()
        hidden_width = bound[0]["mlp.down_proj"].weight.shape[1]
        attention_width = bound[0]["self_attn.o_proj"].weight.shape[1]
        return cls(tuple(bound), DecoderQuantKernels(), RowQuantizer(), scales,
                   torch.zeros(rows, attention_width, dtype=torch.int8, device=device),
                   torch.zeros(rows, hidden_width, dtype=torch.int8, device=device),
                   torch.zeros(rows, 2048, dtype=torch.bfloat16, device=device), steps)

    def quantize_attention(self, attention, layer: int, step: int):
        """Freeze the attention output for the output projection."""
        import torch
        rows, columns = self.attention_int8.shape
        if (attention.dtype != torch.bfloat16 or attention.shape != (rows, columns)
                or not attention.is_contiguous()):
            raise ValueError("the attention output must be a contiguous BF16 matrix")
        code = self.row_quant.launch(
            torch.npu.current_stream(attention.device).npu_stream, attention.data_ptr(),
            self.attention_scales[layer, step].data_ptr(), self.attention_int8.data_ptr(),
            rows, columns)
        if code:
            raise RuntimeError(f"native row quantization rejected arguments: {code}")
        return self.attention_int8
