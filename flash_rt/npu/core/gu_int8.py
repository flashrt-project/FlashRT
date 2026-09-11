"""Setup-owned packed INT8 gate/up weights and runner-owned mixed-kernel storage."""
import ctypes as C
from dataclasses import dataclass
import os
from pathlib import Path


class GuLibrary:
    def __init__(self):
        path = os.environ.get('FLASHRT_NPU_CUBE_LIBRARY')
        path = path or Path(__file__).parents[1] / 'lib' / 'libflashrt_npu_cube.so'
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                'Build the Ascend kernels with scripts/npu/build.sh before NPU '
                'graph construction') from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, 'gate/up cube')
        self.launch = self.library.flashrt_npu_gu_int8
        self.launch.argtypes = [C.c_void_p] * 10 + [C.c_int] * 3
        self.launch.restype = C.c_int
        self.tiling = self.library.flashrt_npu_gu_tiling
        self.tiling.argtypes = [C.c_void_p] + [C.c_int] * 9
        self.tiling.restype = C.c_int


@dataclass(frozen=True)
class GuInt8Weights:
    packed: object
    scales: object
    library: object
    hidden: int
    columns: int

    @classmethod
    def create(cls, gate, up, gate_scale, up_scale):
        import torch
        if (gate.ndim != 2 or gate.dtype != torch.int8 or gate.device.type != 'npu'
                or up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device):
            raise ValueError('gate/up weights must be matching INT8 NPU matrices in (K,H) layout')
        k, h = gate.shape
        if k <= 0 or k > 65536 or k % 32 or h <= 0 or h > 65536 or h % 512 or 2 * h * k > 2147483647:
            raise ValueError('gate/up requires K divisible by 32 and H divisible by 512, both at most 65536')
        for scale in (gate_scale, up_scale):
            if scale.shape != (h,) or scale.dtype != torch.float32 or scale.device != gate.device:
                raise ValueError('gate/up scales must be FP32 output-channel vectors')
        packed = torch.cat((gate.t(), up.t())).reshape(2, h // 512, 512, k)
        packed = packed.transpose(0, 1).contiguous().reshape(2 * h, k)
        scales = torch.cat((gate_scale, up_scale)).reshape(2, h // 512, 512)
        scales = scales.transpose(0, 1).contiguous().reshape(2 * h)
        return cls(packed, scales, GuLibrary(), h, k)

    def prepare(self, rows, acts, inverse, workspace):
        import torch
        if rows <= 0 or rows > 4096:
            raise ValueError('gate/up supports 1 through 4096 rows')
        for scale in (acts, inverse):
            if (scale.shape != (rows,) or scale.dtype != torch.float32
                    or scale.device != self.packed.device or not scale.is_contiguous()):
                raise ValueError('gate/up row scales must be contiguous FP32 vectors on the weight device')
        if (workspace.dtype != torch.int32 or workspace.device != self.packed.device
                or not workspace.is_contiguous() or workspace.numel() < 20 * 2 * 128 * 1024):
            raise ValueError('gate/up requires an exclusive contiguous 20 MiB INT32 workspace')
        hb = C.create_string_buffer(4096)
        size = self.library.tiling(hb, 4096, rows, 1024, self.columns, 128, 1024, 128, 128, 256)
        if size <= 0:
            raise RuntimeError(f'gate/up tiling failed: {size}')
        tiling = torch.tensor(list(hb.raw[:size]), dtype=torch.uint8, device=self.packed.device)
        index = torch.tensor([(i // 512 * 1024 + i % 512) * 4 for i in range(4096)],
                             dtype=torch.int32, device=self.packed.device)
        output = torch.empty((rows, self.hidden), dtype=torch.int8, device=self.packed.device)
        return PreparedGuInt8(self, acts, inverse, workspace, tiling, index, output, rows)


@dataclass(frozen=True)
class PreparedGuInt8:
    weights: GuInt8Weights
    acts: object
    inverse: object
    workspace: object
    tiling: object
    index: object
    output: object
    rows: int

    def __call__(self, q):
        import torch
        w = self.weights
        if (q.shape != (self.rows, w.columns) or q.dtype != torch.int8
                or q.device != w.packed.device or not q.is_contiguous()):
            raise ValueError('gate/up input differs from the prepared INT8 shape or device')
        rc = w.library.launch(torch.npu.current_stream(q.device).npu_stream,
            q.data_ptr(), w.packed.data_ptr(), w.scales.data_ptr(), self.acts.data_ptr(),
            self.inverse.data_ptr(), self.index.data_ptr(), self.output.data_ptr(),
            self.workspace.data_ptr(), self.tiling.data_ptr(), self.rows, w.hidden, w.columns)
        if rc:
            raise RuntimeError(f'gate/up launch failed: {rc}')
        return self.output
