"""Checked AscendCL pointer/stream runtime, independent of PyTorch.

Like the AMD runtime seam, this module owns memory movement and replay only.
Graph construction and tensor/weight ownership belong to the caller.
"""
import ctypes as C
import ctypes.util
import threading

import numpy as np


class AclRuntime:
    def __init__(self):
        try:
            self.lib = C.CDLL(ctypes.util.find_library("ascendcl") or "libascendcl.so")
        except OSError as exc:
            raise ImportError("AscendCL runtime libascendcl.so is required") from exc
        p, pp = C.c_void_p, C.POINTER(C.c_void_p)
        signatures = {
            "aclrtMallocHost": [pp, C.c_size_t], "aclrtFreeHost": [p],
            "aclrtMemcpyAsync": [p, C.c_size_t, p, C.c_size_t, C.c_int, p],
            "aclrtSynchronizeStream": [p],
            "aclrtGetCurrentContext": [pp], "aclrtSetCurrentContext": [p],
            "aclmdlRICaptureGetInfo": [p, C.POINTER(C.c_int), pp],
            "aclmdlRIExecuteAsync": [p, p],
            "aclrtCreateEventWithFlag": [pp, C.c_uint],
            "aclrtDestroyEvent": [p], "aclrtRecordEvent": [p, p],
            "aclrtEventElapsedTime": [C.POINTER(C.c_float), p, p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"AscendCL {name} failed with status {code}")

    def capture_handle(self, stream):
        handle, status = C.c_void_p(), C.c_int()
        self.call("aclmdlRICaptureGetInfo", C.c_void_p(stream),
                  C.byref(status), C.byref(handle))
        if status.value != 1 or not handle.value:
            raise RuntimeError("AscendCL stream has no active captured model")
        return handle


class PinnedBuffer:
    """Pinned host storage for fixed-address asynchronous copies."""
    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.ptr = C.c_void_p()
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        if self.nbytes <= 0:
            raise ValueError("buffer size must be positive")
        runtime.call("aclrtMallocHost", C.byref(self.ptr), self.nbytes)
        self._storage = (C.c_ubyte * self.nbytes).from_address(self.ptr.value)
        self._array = np.frombuffer(self._storage, dtype=self.dtype).reshape(shape)

    @property
    def array(self):
        if not self.ptr.value:
            raise RuntimeError("pinned buffer is closed")
        return self._array

    def close(self):
        if self.ptr.value:
            self.runtime.call("aclrtFreeHost", self.ptr)
            self.ptr = C.c_void_p()
            self._array = self._storage = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # explicit close reports errors; interpreter teardown cannot


class NativeReplay:
    """Borrow a CANN graph with strong ownership of its builder and buffers.

    Device pointers are resolved at setup. Execute has no tensor dispatch,
    device allocation, or synchronization between upload, graph and download.
    A single completion wait is necessary for a synchronous host-result API.
    """
    def __init__(self, runtime, handle, stream, owner, inputs, outputs):
        self.runtime, self.handle = runtime, handle
        self.stream = C.c_void_p(stream)
        self.owner = owner
        self.inputs, self.outputs = inputs, outputs
        self.context = C.c_void_p()
        runtime.call("aclrtGetCurrentContext", C.byref(self.context))
        self.begin, self.end = C.c_void_p(), C.c_void_p()
        runtime.call("aclrtCreateEventWithFlag", C.byref(self.begin), 8)
        try:
            runtime.call("aclrtCreateEventWithFlag", C.byref(self.end), 8)
        except Exception:
            runtime.call("aclrtDestroyEvent", self.begin)
            self.begin = C.c_void_p()
            raise
        self.lock = threading.Lock()
        self.last_replay_ms = 0.0
        self._closed = False
        self._pending = False

    def execute(self):
        self.enqueue()
        self.wait()

    def enqueue(self):
        """Submit fixed-address work without a CPU completion barrier."""
        if self._closed or self._pending:
            raise RuntimeError("replay is closed or already pending")
        self._timed = False
        self._pending = True
        try:
            self._enqueue()
        except Exception:
            self.wait()
            raise

    def _enqueue(self):
        rt = self.runtime
        rt.call("aclrtSetCurrentContext", self.context)
        for host, device_ptr in self.inputs:
            rt.call("aclrtMemcpyAsync", C.c_void_p(device_ptr), host.nbytes,
                    host.ptr, host.nbytes, 1, self.stream)
        rt.call("aclrtRecordEvent", self.begin, self.stream)
        rt.call("aclmdlRIExecuteAsync", self.handle, self.stream)
        rt.call("aclrtRecordEvent", self.end, self.stream)
        self._timed = True
        for device_ptr, host in self.outputs:
            rt.call("aclrtMemcpyAsync", host.ptr, host.nbytes,
                    C.c_void_p(device_ptr), host.nbytes, 2, self.stream)
    def wait(self):
        """Complete host output access outside the asynchronous replay path."""
        if not self._pending:
            return
        rt = self.runtime
        rt.call("aclrtSetCurrentContext", self.context)
        rt.call("aclrtSynchronizeStream", self.stream)
        self._pending = False
        if not self._timed:
            return
        elapsed = C.c_float()
        rt.call("aclrtEventElapsedTime", C.byref(elapsed), self.begin, self.end)
        self.last_replay_ms = float(elapsed.value)

    def close(self):
        if getattr(self, "_pending", False):
            self.wait()
        for name in ("begin", "end"):
            event = getattr(self, name, None)
            if event and event.value:
                self.runtime.call("aclrtDestroyEvent", event)
                setattr(self, name, C.c_void_p())
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
