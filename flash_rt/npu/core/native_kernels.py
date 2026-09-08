"""Setup/capture adapters for the standalone checked NPU pointer ABI."""
import ctypes as C
import os
from pathlib import Path


class RowQuantizer:
    def __init__(self):
        library = os.environ.get("FLASHRT_NPU_LIBRARY")
        path = Path(library) if library else Path(__file__).parents[1] / "lib" / "libflashrt_npu.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError("Build the Ascend kernels with scripts/npu/build.sh before INT8 calibration") from exc
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
