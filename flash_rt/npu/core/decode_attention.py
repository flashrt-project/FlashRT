"""Setup-owned decode attention for the action expert, on the raw cube path."""
import ctypes as C
import os
from pathlib import Path


class DecodeAttentionLibrary:
    """The attention kernel drives ``Mmad`` on the cube and runs the softmax on
    the vector cores inside one launch, so it is a mixed translation unit and
    cannot share a library with the cube-only decoder GEMM.
    """

    def __init__(self):
        path = os.environ.get("FLASHRT_NPU_ATTENTION_LIBRARY")
        path = path or Path(__file__).parents[1] / "lib" / "libflashrt_npu_attn.so"
        try:
            self.library = C.CDLL(str(path))
        except OSError as exc:
            raise ImportError(
                "Build the Ascend kernels with scripts/npu/build.sh before NPU "
                "graph construction") from exc
        from flash_rt.npu.core import abi
        abi.verify(self.library, "decode attention")
        self.launch = self.library.flashrt_npu_decode_attn
        self.launch.argtypes = [C.c_void_p] * 8 + [C.c_int] * 5 + [C.c_float]
        self.launch.restype = C.c_int
