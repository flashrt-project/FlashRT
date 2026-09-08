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


class RmsRowQuant(_NativeLibrary):
    """Optional BF16 residual addition, RMS normalization and static INT8."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_rms_row_quant
        self.launch.argtypes = [C.c_void_p] * 7 + [C.c_int] * 2
        self.launch.restype = C.c_int

    def __call__(self, x, other, gamma, inverse_scales):
        import torch
        if (x.ndim != 2 or x.shape[1] != 2048 or x.dtype != torch.bfloat16
                or x.device.type != "npu" or not x.is_contiguous()):
            raise ValueError("RMS quantization requires a contiguous BF16 NPU (rows,2048) matrix")
        tensors = [(gamma, torch.bfloat16, (2048,)),
                   (inverse_scales, torch.float32, (x.shape[0],))]
        if other is not None:
            tensors.append((other, x.dtype, x.shape))
        for tensor, dtype, shape in tensors:
            if (tensor.dtype != dtype or tensor.shape != shape or tensor.device != x.device
                    or not tensor.is_contiguous()):
                raise ValueError("invalid RMS quantization operand or row scales")
        quantized = torch.empty_like(x, dtype=torch.int8)
        residual = torch.empty_like(x) if other is not None else x
        code = self.launch(torch.npu.current_stream(x.device).npu_stream,
            x.data_ptr(), other.data_ptr() if other is not None else None,
            gamma.data_ptr(), inverse_scales.data_ptr(), quantized.data_ptr(),
            residual.data_ptr(), x.shape[0], int(other is not None))
        if code:
            raise RuntimeError(f"native RMS quantization rejected arguments: {code}")
        return quantized, residual


class EncoderRope(_NativeLibrary):
    """FP32 Q/K rotation; tables must repeat their first 128 columns."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_encoder_rope
        self.launch.argtypes = [C.c_void_p] * 7 + [C.c_int]
        self.launch.restype = C.c_int

    def __call__(self, q, k, cos, sin):
        import torch
        if (q.ndim != 2 or q.shape[1] != 2048 or q.dtype != torch.bfloat16
                or q.device.type != "npu" or not q.is_contiguous()):
            raise ValueError("encoder RoPE requires a contiguous BF16 NPU (rows,2048) query")
        if (k.shape != (q.shape[0], 256) or k.dtype != q.dtype or k.device != q.device
                or not k.is_contiguous()):
            raise ValueError("encoder RoPE requires a matching contiguous (rows,256) key")
        if (cos.ndim != 2 or cos.shape[1] != 256 or cos.shape[0] < q.shape[0]
                or sin.shape != cos.shape):
            raise ValueError("encoder RoPE tables must cover every query row")
        for table in (cos, sin):
            if (table.dtype != torch.float32 or table.device != q.device
                    or not table.is_contiguous()):
                raise ValueError("encoder RoPE requires contiguous FP32 device tables")
        qo, ko = torch.empty_like(q), torch.empty_like(k)
        code = self.launch(torch.npu.current_stream(q.device).npu_stream,
            q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            qo.data_ptr(), ko.data_ptr(), q.shape[0])
        if code:
            raise RuntimeError(f"native encoder RoPE rejected arguments: {code}")
        return qo, ko


class EulerUpdate(_NativeLibrary):
    """Promote BF16 velocity, then multiply and subtract separately in FP32."""
    def __init__(self):
        super().__init__()
        self.launch = self.library.flashrt_npu_euler_update
        self.launch.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_float]
        self.launch.restype = C.c_int

    def __call__(self, x, velocity, dt):
        import torch
        if (x.ndim != 2 or x.shape[1] != 32 or x.dtype != torch.float32
                or x.device.type != "npu" or not x.is_contiguous()):
            raise ValueError("Euler update requires contiguous FP32 NPU (rows,32) actions")
        if (velocity.shape != x.shape or velocity.dtype != torch.bfloat16
                or velocity.device != x.device or not velocity.is_contiguous()):
            raise ValueError("Euler update requires matching BF16 velocity")
        if not 0.0 < dt <= 1.0:
            raise ValueError("Euler step must be finite and in (0,1]")
        out = torch.empty_like(x)
        code = self.launch(torch.npu.current_stream(x.device).npu_stream,
            x.data_ptr(), velocity.data_ptr(), out.data_ptr(), x.numel(), dt)
        if code:
            raise RuntimeError(f"native Euler update rejected arguments: {code}")
        return out
