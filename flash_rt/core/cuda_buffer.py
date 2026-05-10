"""FlashRT — CudaBuffer: cudaMalloc/managed wrapper for engine-facing GPU buffers."""

import ctypes
import logging
import numpy as np

logger = logging.getLogger(__name__)

_cudart = ctypes.CDLL("libcudart.so")


def _configure_cudart_signatures() -> None:
    """Declare ctypes signatures for CUDA runtime calls used here."""
    ptr_p = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "cudaMallocManaged": (
            [ptr_p, ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
        "cudaMalloc": ([ptr_p, ctypes.c_size_t], ctypes.c_int),
        "cudaFree": ([ctypes.c_void_p], ctypes.c_int),
        "cudaMemcpy": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int],
            ctypes.c_int,
        ),
        "cudaMemcpyAsync": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
             ctypes.c_int, ctypes.c_void_p],
            ctypes.c_int,
        ),
        "cudaMemset": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
        "cudaMemsetAsync": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
             ctypes.c_void_p],
            ctypes.c_int,
        ),
        "cudaDeviceSynchronize": ([], ctypes.c_int),
        "cudaStreamSynchronize": ([ctypes.c_void_p], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(_cudart, name)
        fn.argtypes = argtypes
        fn.restype = restype


_configure_cudart_signatures()


def _check(ret, msg=""):
    if ret != 0:
        raise RuntimeError(f"CUDA error {ret}: {msg}")


class CudaBuffer:
    """GPU buffer — managed or device memory."""

    def __init__(self, nbytes: int, managed: bool = True):
        self._ptr = ctypes.c_void_p()
        self._managed = managed
        if managed:
            _check(_cudart.cudaMallocManaged(ctypes.byref(self._ptr), nbytes, 1),
                   "cudaMallocManaged")
        else:
            _check(_cudart.cudaMalloc(ctypes.byref(self._ptr), nbytes), "cudaMalloc")
        self._nbytes = nbytes

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'CudaBuffer':
        """Create device buffer, upload via chunked H2D (Thor-safe ≤4MB/chunk).

        Uses device memory instead of managed for faster graph replay bandwidth.
        Thor platform constraint: single cudaMemcpy H2D ≤16MB, uses 4MB chunks.
        """
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=False)
        chunk = 4 * 1024 * 1024  # 4MB chunks, Thor-safe
        for off in range(0, arr.nbytes, chunk):
            n = min(chunk, arr.nbytes - off)
            _check(_cudart.cudaMemcpy(
                ctypes.c_void_p(buf._ptr.value + off),
                ctypes.c_void_p(arr.ctypes.data + off), n, 1), "H2D chunk")
        return buf

    @classmethod
    def from_numpy_managed(cls, arr: np.ndarray) -> 'CudaBuffer':
        """Create managed buffer, upload via memmove. Use for buffers that need D2H readback."""
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=True)
        ctypes.memmove(buf._ptr, arr.ctypes.data, arr.nbytes)
        return buf

    @classmethod
    def zeros(cls, count: int, dtype, managed: bool = True) -> 'CudaBuffer':
        nbytes = count * np.dtype(dtype).itemsize
        buf = cls(nbytes, managed=managed)
        _cudart.cudaMemset(buf._ptr, 0, nbytes)
        return buf

    @classmethod
    def empty(cls, count: int, dtype, managed: bool = True) -> 'CudaBuffer':
        return cls(count * np.dtype(dtype).itemsize, managed=managed)

    @classmethod
    def device_zeros(cls, count: int, dtype) -> 'CudaBuffer':
        return cls.zeros(count, dtype, managed=False)

    @classmethod
    def device_empty(cls, count: int, dtype) -> 'CudaBuffer':
        return cls.empty(count, dtype, managed=False)

    def upload(self, arr: np.ndarray):
        """Upload numpy → buffer."""
        assert arr.nbytes <= self._nbytes
        arr = np.ascontiguousarray(arr)
        if self._managed:
            ctypes.memmove(self._ptr, arr.ctypes.data, arr.nbytes)
        else:
            chunk = 4 * 1024 * 1024
            offset = 0
            while offset < arr.nbytes:
                n = min(chunk, arr.nbytes - offset)
                _check(_cudart.cudaMemcpy(ctypes.c_void_p(self._ptr.value + offset),
                    ctypes.c_void_p(arr.ctypes.data + offset), n, 1), "H2D chunk")
                offset += n

    def download(self, arr: np.ndarray):
        """Download buffer → numpy.

        ``arr.ctypes.data`` is a Python int holding the host pointer.
        Wrap it in ``ctypes.c_void_p`` before handing it to
        ``_cudart.cudaMemcpy`` — without an explicit argtype the
        ctypes default is ``c_int`` (32-bit signed), which truncates
        host addresses ≥ 2 GiB and sign-extends to 0xffffffff…,
        causing a segfault on the cudaMemcpy DtoH for any buffer
        large enough to push the heap past the 2 GiB boundary
        (observed first on the 11 MiB style-buffer download in the
        Stage 2 batched-CFG JAX port).
        """
        assert arr.nbytes <= self._nbytes
        _cudart.cudaDeviceSynchronize()
        if self._managed:
            ctypes.memmove(arr.ctypes.data, self._ptr, arr.nbytes)
        else:
            _check(_cudart.cudaMemcpy(
                ctypes.c_void_p(arr.ctypes.data),
                self._ptr, arr.nbytes, 2), "D2H")

    def download_new(self, shape, dtype) -> np.ndarray:
        arr = np.empty(shape, dtype=dtype)
        self.download(arr)
        return arr

    def copy_from_jax(self, jax_array):
        """Copy JAX array → this buffer via single D2D cudaMemcpy."""
        import jax
        jax.block_until_ready(jax_array)
        src = jax_array.unsafe_buffer_pointer()
        _check(_cudart.cudaMemcpy(self._ptr, ctypes.c_void_p(src), jax_array.nbytes, 3),
               "cudaMemcpy D2D")

    def zero_(self, stream=None):
        if stream is not None:
            _cudart.cudaMemsetAsync(self._ptr, 0, self._nbytes, stream)
        else:
            _cudart.cudaMemset(self._ptr, 0, self._nbytes)

    def __del__(self):
        try:
            if _cudart is not None and hasattr(self, '_ptr') and self._ptr.value:
                _cudart.cudaFree(self._ptr)
                self._ptr = ctypes.c_void_p()
        except Exception:
            pass

    def __repr__(self):
        t = "managed" if self._managed else "device"
        return f"CudaBuffer({self._nbytes}B, {t}, ptr=0x{self._ptr.value:x})"


def sync():
    _cudart.cudaDeviceSynchronize()
