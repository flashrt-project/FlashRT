"""Setup/capture adapters for the standalone checked NPU pointer ABI."""
import ctypes as C
import os
from pathlib import Path


class _NativeLibrary:
    def __init__(self):
        library = os.environ.get("FLASHRT_NPU_LIBRARY")
        path = Path(library) if library else Path(__file__).parents[1] / "lib" / "libflashrt_npu.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError("Build the Ascend kernels with scripts/npu/build.sh before NPU graph construction") from exc


class RowQuantizer(_NativeLibrary):
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_quantize_rows
        self.launch.argtypes = [C.c_void_p] * 4 + [C.c_int] * 2
        self.launch.restype = C.c_int

    def __call__(self, x, inverse_scales):
        # Invoked during eager calibration validation and graph construction,
        # never from the raw AscendCL replay path.
        import torch
        if (x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous()
                or x.shape[1] % 32 or x.device.type != "npu"):
            raise ValueError("row quantization requires a contiguous BF16 NPU matrix with K divisible by 32")
        if (inverse_scales.dtype != torch.float32 or not inverse_scales.is_contiguous()
                or inverse_scales.shape != (x.shape[0],) or inverse_scales.device != x.device):
            raise ValueError("row scales must be a contiguous FP32 vector on the input device")
        out = torch.empty_like(x, dtype=torch.int8)
        code = self.launch(torch.npu.current_stream(x.device).npu_stream,
                           x.data_ptr(), inverse_scales.data_ptr(), out.data_ptr(),
                           x.shape[0], x.shape[1])
        if code:
            raise RuntimeError(f"native row quantization rejected arguments: {code}")
        return out


class DecoderRope(_NativeLibrary):
    """Fused merged-QKV split, FP32 rotary math and in-place KV append."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_decoder_rope
        self.launch.argtypes = [C.c_void_p] * 7 + [C.c_int] * 2
        self.launch.restype = C.c_int

    def __call__(self, qkv, cos, sin, keys, values, prefix):
        import torch
        if (qkv.dtype != torch.bfloat16 or qkv.ndim != 2 or qkv.shape[1] != 2560
                or not qkv.is_contiguous() or qkv.device.type != "npu"):
            raise ValueError("decoder rotary input must be a contiguous BF16 NPU (rows,2560) matrix")
        rows = qkv.shape[0]
        if prefix < 0 or rows <= 0:
            raise ValueError("invalid KV prefix or action row count")
        for tensor, dtype in ((cos, torch.float32), (sin, torch.float32),
                               (keys, torch.bfloat16), (values, torch.bfloat16)):
            if (tensor.dtype != dtype or not tensor.is_contiguous() or tensor.ndim != 2
                    or tensor.shape[1] != 256 or tensor.shape[0] < prefix + rows
                    or tensor.device != qkv.device):
                raise ValueError("invalid rotary table or KV buffer")
        query = torch.empty((rows, 2048), dtype=qkv.dtype, device=qkv.device)
        code = self.launch(torch.npu.current_stream(qkv.device).npu_stream,
            qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(), query.data_ptr(),
            keys.data_ptr(), values.data_ptr(), prefix, rows)
        if code:
            raise RuntimeError(f"native decoder rotary rejected arguments: {code}")
        return query


class GatedAdaRms(_NativeLibrary):
    """FP32 residual, BF16-rounded gated update, and shifted AdaRMS output."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_gated_ada
        self.launch.argtypes = [C.c_void_p] * 8 + [C.c_int] * 2
        self.launch.restype = C.c_int

    def __call__(self, residual, branch, gate, gamma, shift):
        import torch
        if (residual.dtype != torch.float32 or residual.ndim != 2
                or residual.shape[1] != 1024 or not residual.is_contiguous()
                or residual.device.type != "npu"):
            raise ValueError("gated AdaRMS requires a contiguous FP32 NPU (rows,1024) residual")
        tensors = [(gamma, torch.bfloat16, (1024,)), (shift, torch.float32, (1024,))]
        if branch is not None:
            tensors += [(branch, torch.bfloat16, residual.shape), (gate, torch.bfloat16, (1024,))]
        for tensor, dtype, shape in tensors:
            if (tensor is None or tensor.dtype != dtype or tensor.shape != shape
                    or tensor.device != residual.device or not tensor.is_contiguous()):
                raise ValueError("invalid gated AdaRMS input or style")
        norm = torch.empty_like(residual, dtype=torch.bfloat16)
        updated = torch.empty_like(residual) if branch is not None else residual
        code = self.launch(torch.npu.current_stream(residual.device).npu_stream,
            residual.data_ptr(), branch.data_ptr() if branch is not None else None,
            gate.data_ptr() if gate is not None else None, gamma.data_ptr(), shift.data_ptr(),
            norm.data_ptr(), updated.data_ptr(), residual.shape[0], int(branch is not None))
        if code:
            raise RuntimeError(f"native gated AdaRMS rejected arguments: {code}")
        return norm, updated


class GeluMulQuant(_NativeLibrary):
    """Tanh GELU, BF16-rounded product, then frozen row quantization."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_gelu_mul_quant
        self.launch.argtypes = [C.c_void_p] * 5 + [C.c_int] * 2
        self.launch.restype = C.c_int

    def __call__(self, gate, up, inverse_scales):
        import torch
        if (gate.ndim != 2 or gate.dtype != torch.bfloat16 or gate.device.type != "npu"
                or not gate.is_contiguous() or gate.shape[1] % 32):
            raise ValueError("GELU quantization requires a contiguous BF16 NPU matrix")
        if (up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device
                or not up.is_contiguous()):
            raise ValueError("gate and up matrices must have identical layout")
        if (inverse_scales.shape != (gate.shape[0],) or inverse_scales.dtype != torch.float32
                or inverse_scales.device != gate.device or not inverse_scales.is_contiguous()):
            raise ValueError("GELU quantization requires contiguous FP32 row scales")
        out = torch.empty_like(gate, dtype=torch.int8)
        code = self.launch(torch.npu.current_stream(gate.device).npu_stream,
            gate.data_ptr(), up.data_ptr(), inverse_scales.data_ptr(), out.data_ptr(),
            gate.shape[0], gate.shape[1])
        if code:
            raise RuntimeError(f"native GELU quantization rejected arguments: {code}")
        return out
