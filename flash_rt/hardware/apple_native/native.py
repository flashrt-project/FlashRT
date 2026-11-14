"""Torch-free binding of the Apple plan C ABI.

``NativePlan`` exposes buffers, host mappings, registered-kernel dispatches and
multiple sealed programs over shared buffers. It holds no model knowledge: a
model runtime owns the buffer layout and records its programs once, then
updates buffer contents and runs programs. ``plan.Plan`` remains the BF16
tensor-descriptor builder used by Pi0.5.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
import weakref
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

READ, WRITE, READ_WRITE = 0, 1, 2


class _Timing(C.Structure):
    # frt_apple_plan_timing; plan._PlanTiming describes the same struct but its module imports torch.
    _fields_ = [(name, C.c_double) for name in (
        "encode_us", "gpu_us", "wall_us", "commit_us", "wait_us", "commit_to_gpu_us",
        "gpu_to_return_us", "kernel_us", "submit_host_s", "gpu_start_s", "gpu_end_s", "return_host_s")]

    def as_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name, _ in self._fields_}


def default_library() -> Path:
    """``FLASHRT_APPLE_PLAN_LIBRARY``, else the library built by ``scripts/apple/build_native.sh``."""
    env = os.environ.get("FLASHRT_APPLE_PLAN_LIBRARY")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / ".cache/apple-native/libflashrt_apple_plan.dylib"


class NativePlan:
    """One Metal device context: buffers, kernels and sealed programs."""

    _SIGNATURES = {
        "create": ([], C.c_void_p),
        "destroy": ([C.c_void_p], None),
        "error": ([], C.c_char_p),
        "buffer": ([C.c_void_p, C.c_size_t], C.c_int64),
        "write": ([C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t], C.c_int),
        "read": ([C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.c_size_t], C.c_int),
        "data": ([C.c_void_p, C.c_uint32, C.c_size_t, C.c_size_t], C.c_void_p),
        "dispatch": ([C.c_void_p, C.c_char_p, C.POINTER(C.c_uint32), C.POINTER(C.c_size_t),
                      C.POINTER(C.c_uint32), C.c_size_t, C.POINTER(C.c_size_t), C.c_size_t, C.c_char_p],
                     C.c_int),
        "program": ([C.c_void_p], C.c_int64),
        "finish": ([C.c_void_p], C.c_int),
        "residency": ([C.c_void_p, C.c_uint32], C.c_int),
        "run_program": ([C.c_void_p, C.c_uint32, C.c_uint32, C.POINTER(_Timing)], C.c_int),
        "profile_program": ([C.c_void_p, C.c_uint32, C.POINTER(_Timing), C.c_size_t], C.c_int),
    }

    def __init__(self, library=None):
        if sys.platform != "darwin":
            raise RuntimeError("the Apple native plan needs macOS on Apple Silicon")
        path = Path(library) if library else default_library()
        if not path.exists():
            raise RuntimeError(f"Apple plan library not found: {path}. Build it with scripts/apple/build_native.sh")
        self.lib = C.CDLL(str(path))
        for name, (args, result) in self._SIGNATURES.items():
            try:
                fn = getattr(self.lib, "frt_apple_plan_" + name)
            except AttributeError as exc:
                raise RuntimeError(f"{path} lacks frt_apple_plan_{name}; rebuild the Apple native library") from exc
            fn.argtypes, fn.restype = args, result
        self.handle = self.lib.frt_apple_plan_create()
        if not self.handle:
            self._check(-1)
        self._finalizer = weakref.finalize(self, self.lib.frt_apple_plan_destroy, self.handle)
        self.labels: Dict[int, List[str]] = {0: []}   # operation labels per program
        self._recording = 0

    def _check(self, status: int) -> None:
        if status:
            raise RuntimeError(self.lib.frt_apple_plan_error().decode())

    def close(self) -> None:
        self._finalizer()
        self.handle = None

    # ---------------------------------------------------------------- buffers
    def buffer(self, nbytes: int) -> int:
        bid = self.lib.frt_apple_plan_buffer(self.handle, int(nbytes))
        if bid < 0:
            self._check(-1)
        return int(bid)

    def view(self, buffer: int, dtype, count: int, offset: int = 0) -> np.ndarray:
        """Borrowed host mapping of ``count`` elements; valid while the plan lives. Callers serialize
        access with ``run``."""
        dtype = np.dtype(dtype)
        nbytes = dtype.itemsize * int(count)
        address = self.lib.frt_apple_plan_data(self.handle, buffer, int(offset), nbytes)
        if not address:
            self._check(-1)
        raw = np.ctypeslib.as_array((C.c_uint8 * nbytes).from_address(address))
        return raw.view(dtype)

    def write(self, buffer: int, offset: int, data) -> None:
        array = np.ascontiguousarray(data)
        self._check(self.lib.frt_apple_plan_write(self.handle, buffer, int(offset),
                                                  array.ctypes.data_as(C.c_void_p), array.nbytes))

    def read(self, buffer: int, offset: int, dtype, count: int) -> np.ndarray:
        out = np.empty(int(count), dtype=dtype)
        self._check(self.lib.frt_apple_plan_read(self.handle, buffer, int(offset),
                                                 out.ctypes.data_as(C.c_void_p), out.nbytes))
        return out

    # --------------------------------------------------------------- programs
    def program(self) -> int:
        pid = self.lib.frt_apple_plan_program(self.handle)
        if pid < 0:
            self._check(-1)
        self._recording = int(pid)
        self.labels[self._recording] = []
        return int(pid)

    def dispatch(self, kernel: str, bindings: Sequence[Tuple[int, int, int]], groups: Iterable[int],
                 threads: int, label: str = "") -> None:
        """bindings: (buffer, byte offset, access) in Metal argument order."""
        n = len(bindings)
        ids = (C.c_uint32 * n)(*[b[0] for b in bindings])
        offsets = (C.c_size_t * n)(*[b[1] for b in bindings])
        access = (C.c_uint32 * n)(*[b[2] for b in bindings])
        g = tuple(int(x) for x in groups)
        g = g + (1,) * (3 - len(g))
        self._check(self.lib.frt_apple_plan_dispatch(self.handle, kernel.encode(), ids, offsets, access, n,
                                                     (C.c_size_t * 3)(*g), int(threads),
                                                     (label or kernel).encode()))
        self.labels[self._recording].append(label or kernel)

    def finish(self, resident: bool = True) -> None:
        self._check(self.lib.frt_apple_plan_finish(self.handle))
        if resident:
            self._check(self.lib.frt_apple_plan_residency(self.handle, 1))

    def profile(self, program: int) -> List[Tuple[str, Dict[str, float]]]:
        """Diagnostic: run every operation of ``program`` (>= 1) as its own synchronous command buffer, in
        order (same effects as one run of the program); returns (label, timing) per operation."""
        labels = self.labels[int(program)]
        t = (_Timing * len(labels))()
        self._check(self.lib.frt_apple_plan_profile_program(self.handle, int(program), t, len(labels)))
        return [(label, t[i].as_dict()) for i, label in enumerate(labels)]

    def run(self, program: int = 0, execution: str = "icb") -> Dict[str, float]:
        if execution not in ("icb", "direct"):
            raise ValueError("execution must be 'icb' or 'direct'")
        t = _Timing()
        self._check(self.lib.frt_apple_plan_run_program(self.handle, int(program), int(execution == "icb"),
                                                        C.byref(t)))
        return t.as_dict()
