"""FlashRT ROCm/HIP device buffers.

This mirrors :mod:`flash_rt.core.cuda_buffer` for the AMD pipeline. The goal is
to give ROCm pipelines stable, FlashRT-owned device addresses instead of relying
on framework tensor allocation in the hot path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _load_hip_runtime():
    candidates = []
    found = ctypes.util.find_library("amdhip64")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6",
        ]
    )
    last_exc = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_exc = exc
    raise OSError(f"Could not load HIP runtime from {candidates}") from last_exc


_hip = _load_hip_runtime()


def _configure_hip_signatures() -> None:
    ptr_p = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "hipMalloc": ([ptr_p, ctypes.c_size_t], ctypes.c_int),
        "hipMallocManaged": ([ptr_p, ctypes.c_size_t, ctypes.c_uint], ctypes.c_int),
        "hipFree": ([ctypes.c_void_p], ctypes.c_int),
        "hipMemcpy": (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            ctypes.c_int,
        ),
        "hipMemcpyAsync": (
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ],
            ctypes.c_int,
        ),
        "hipMemset": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
        "hipMemsetAsync": (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p],
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


def _check(status: int, msg: str = "") -> None:
    if status != 0:
        raise RuntimeError(f"HIP error {status}: {msg}")


class HipBuffer:
    """HIP device or managed allocation with numpy upload/download helpers."""

    def __init__(self, nbytes: int, managed: bool = False):
        self._ptr = ctypes.c_void_p()
        self._managed = bool(managed)
        self._nbytes = int(nbytes)
        if self._nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
        if self._managed:
            _check(
                _hip.hipMallocManaged(ctypes.byref(self._ptr), self._nbytes, 1),
                "hipMallocManaged",
            )
        else:
            _check(_hip.hipMalloc(ctypes.byref(self._ptr), self._nbytes), "hipMalloc")

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def managed(self) -> bool:
        return self._managed

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "HipBuffer":
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=False)
        buf.upload(arr)
        return buf

    @classmethod
    def from_numpy_managed(cls, arr: np.ndarray) -> "HipBuffer":
        arr = np.ascontiguousarray(arr)
        buf = cls(arr.nbytes, managed=True)
        ctypes.memmove(buf._ptr, arr.ctypes.data, arr.nbytes)
        return buf

    @classmethod
    def zeros(cls, count: int, dtype, managed: bool = False) -> "HipBuffer":
        buf = cls(int(count) * np.dtype(dtype).itemsize, managed=managed)
        _check(_hip.hipMemset(buf._ptr, 0, buf._nbytes), "hipMemset")
        return buf

    @classmethod
    def empty(cls, count: int, dtype, managed: bool = False) -> "HipBuffer":
        return cls(int(count) * np.dtype(dtype).itemsize, managed=managed)

    @classmethod
    def device_zeros(cls, count: int, dtype) -> "HipBuffer":
        return cls.zeros(count, dtype, managed=False)

    @classmethod
    def device_empty(cls, count: int, dtype) -> "HipBuffer":
        return cls.empty(count, dtype, managed=False)

    def upload(self, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if arr.nbytes > self._nbytes:
            raise ValueError(f"array has {arr.nbytes} bytes, buffer has {self._nbytes}")
        if self._managed:
            ctypes.memmove(self._ptr, arr.ctypes.data, arr.nbytes)
            return
        _check(
            _hip.hipMemcpy(
                self._ptr,
                ctypes.c_void_p(arr.ctypes.data),
                arr.nbytes,
                1,  # hipMemcpyHostToDevice
            ),
            "hipMemcpy H2D",
        )

    def download(self, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if arr.nbytes > self._nbytes:
            raise ValueError(f"array has {arr.nbytes} bytes, buffer has {self._nbytes}")
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        if self._managed:
            ctypes.memmove(arr.ctypes.data, self._ptr, arr.nbytes)
            return
        _check(
            _hip.hipMemcpy(
                ctypes.c_void_p(arr.ctypes.data),
                self._ptr,
                arr.nbytes,
                2,  # hipMemcpyDeviceToHost
            ),
            "hipMemcpy D2H",
        )

    def download_new(self, shape, dtype) -> np.ndarray:
        arr = np.empty(shape, dtype=dtype)
        self.download(arr)
        return arr

    def zero_(self, stream=None) -> None:
        if stream is None:
            _check(_hip.hipMemset(self._ptr, 0, self._nbytes), "hipMemset")
        else:
            _check(
                _hip.hipMemsetAsync(self._ptr, 0, self._nbytes, stream),
                "hipMemsetAsync",
            )

    def __del__(self):
        try:
            if _hip is not None and hasattr(self, "_ptr") and self._ptr.value:
                _hip.hipFree(self._ptr)
                self._ptr = ctypes.c_void_p()
        except Exception:
            pass

    def __repr__(self) -> str:
        kind = "managed" if self._managed else "device"
        ptr = self._ptr.value or 0
        return f"HipBuffer({self._nbytes}B, {kind}, ptr=0x{ptr:x})"


def sync() -> None:
    _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


__all__ = ["HipBuffer", "sync", "_hip", "_check"]
