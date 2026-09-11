"""FlashRT AMD — HipBuffer: hipMalloc/managed wrapper for engine-facing GPU buffers.

Twin of flash_rt/core/cuda_buffer.py over libamdhip64. Memcpy-kind and
attach-flag enums are numerically identical to CUDA's (verified at
bring-up): H2D=1, D2H=2, D2D=3, hipMemAttachGlobal=1.
"""

import ctypes
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    _hip = ctypes.CDLL("libamdhip64.so")
except OSError as exc:  # no ROCm runtime on this machine
    # Surface the conventional "optional backend unavailable" signal so
    # callers (flash_rt.api's hardware gate, test suites) can guard with
    # a plain ImportError instead of a platform-specific OSError.
    raise ImportError(
        "the AMD backend requires the ROCm runtime (libamdhip64.so), "
        f"which could not be loaded: {exc}") from exc


def _configure_hip_signatures() -> None:
    """Declare ctypes signatures — host pointers ≥2GiB truncate under the
    default c_int argtype (see cuda_buffer.py download() docstring)."""
    ptr_p = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "hipMallocManaged": (
            [ptr_p, ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
        "hipMalloc": ([ptr_p, ctypes.c_size_t], ctypes.c_int),
        "hipFree": ([ctypes.c_void_p], ctypes.c_int),
        "hipMemcpy": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int],
            ctypes.c_int,
        ),
        "hipMemcpyAsync": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int, ctypes.c_void_p],
            ctypes.c_int,
        ),
        "hipMemset": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
        "hipMemsetAsync": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
             ctypes.c_void_p],
            ctypes.c_int,
        ),
        "hipDeviceSynchronize": ([], ctypes.c_int),
        "hipStreamSynchronize": ([ctypes.c_void_p], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(_hip, name)
        fn.argtypes = argtypes
        fn.restype = restype


_configure_hip_signatures()


def _check(ret, msg=""):
    if ret != 0:
        raise RuntimeError(f"HIP error {ret}: {msg}")


class HipBuffer:
    """GPU buffer — managed or device memory."""

    def __init__(self, nbytes: int, managed: bool = True):
        self._ptr = ctypes.c_void_p()
        self._managed = managed
        if managed:
            _check(_hip.hipMallocManaged(ctypes.byref(self._ptr), nbytes, 1),
                   "hipMallocManaged")
        else:
            _check(_hip.hipMalloc(ctypes.byref(self._ptr), nbytes), "hipMalloc")
        self._nbytes = nbytes

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'HipBuffer':
        """Create device buffer, upload H2D (device memory for replay bandwidth)."""
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=False)
        _check(_hip.hipMemcpy(
            buf._ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes, 1), "H2D")
        return buf

    @classmethod
    def from_numpy_managed(cls, arr: np.ndarray) -> 'HipBuffer':
        """Create managed buffer, upload via memmove. Use for buffers needing D2H readback."""
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=True)
        ctypes.memmove(buf._ptr, arr.ctypes.data, arr.nbytes)
        return buf

    @classmethod
    def zeros(cls, count: int, dtype, managed: bool = True) -> 'HipBuffer':
        nbytes = count * np.dtype(dtype).itemsize
        buf = cls(nbytes, managed=managed)
        _check(_hip.hipMemset(buf._ptr, 0, nbytes), "hipMemset")
        return buf

    @classmethod
    def empty(cls, count: int, dtype, managed: bool = True) -> 'HipBuffer':
        return cls(count * np.dtype(dtype).itemsize, managed=managed)

    @classmethod
    def device_zeros(cls, count: int, dtype) -> 'HipBuffer':
        return cls.zeros(count, dtype, managed=False)

    @classmethod
    def device_empty(cls, count: int, dtype) -> 'HipBuffer':
        return cls.empty(count, dtype, managed=False)

    def upload(self, arr: np.ndarray):
        """Upload numpy → buffer."""
        assert arr.nbytes <= self._nbytes
        arr = np.ascontiguousarray(arr)
        if self._managed:
            ctypes.memmove(self._ptr, arr.ctypes.data, arr.nbytes)
        else:
            _check(_hip.hipMemcpy(
                self._ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes, 1), "H2D")

    def download(self, arr: np.ndarray):
        """Download buffer → numpy."""
        assert arr.nbytes <= self._nbytes
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        if self._managed:
            ctypes.memmove(arr.ctypes.data, self._ptr, arr.nbytes)
        else:
            _check(_hip.hipMemcpy(
                ctypes.c_void_p(arr.ctypes.data),
                self._ptr, arr.nbytes, 2), "D2H")

    def download_new(self, shape, dtype) -> np.ndarray:
        arr = np.empty(shape, dtype=dtype)
        self.download(arr)
        return arr

    def zero_(self, stream=None):
        if stream is not None:
            _check(_hip.hipMemsetAsync(self._ptr, 0, self._nbytes, stream),
                   "hipMemsetAsync")
        else:
            _check(_hip.hipMemset(self._ptr, 0, self._nbytes), "hipMemset")

    def __del__(self):
        try:
            if _hip is not None and hasattr(self, '_ptr') and self._ptr.value:
                ret = _hip.hipFree(self._ptr)
                self._ptr = ctypes.c_void_p()
                if ret != 0:
                    # Raising in __del__ is unraisable; a failed free cannot
                    # corrupt results, so log instead of _check here.
                    logger.warning("hipFree failed with HIP error %d", ret)
        except Exception:
            pass

    def __repr__(self):
        t = "managed" if self._managed else "device"
        return f"HipBuffer({self._nbytes}B, {t}, ptr=0x{self._ptr.value:x})"


def sync():
    _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
