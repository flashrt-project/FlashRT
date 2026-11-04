"""Opt-in Instruments signposts and synchronized diagnostic stage timings.

Without synchronization, ranges describe CPU submission, not GPU duration.
Use torch.mps.profiler for GPU operation signposts on the same timeline.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import ctypes as C
import os
from pathlib import Path
import time


class Signposts:
    def __init__(self, library=None):
        default = Path(__file__).resolve().parents[3] / ".cache/apple-native/libflashrt_apple_signpost.dylib"
        path = Path(library or os.environ.get("FLASHRT_APPLE_SIGNPOST_LIBRARY", default))
        if not path.is_file():
            raise FileNotFoundError(f"Signpost bridge missing: {path}; run bash scripts/apple/build_native.sh")
        self.lib = C.CDLL(str(path.resolve()))
        self.lib.frt_apple_signpost_begin.argtypes = [C.c_char_p]
        self.lib.frt_apple_signpost_begin.restype = C.c_uint64
        self.lib.frt_apple_signpost_end.argtypes = [C.c_uint64, C.c_char_p]
        self.lib.frt_apple_signpost_end.restype = None

    @contextmanager
    def interval(self, name):
        label = name.encode("utf-8")
        token = self.lib.frt_apple_signpost_begin(label)
        try:
            yield
        finally:
            self.lib.frt_apple_signpost_end(token, label)


class StageProfiler:
    def __init__(self, *, signposts=None, synchronize=None):
        self.signposts = signposts
        self.synchronize = synchronize
        self.samples_ms = {}

    @contextmanager
    def range(self, name):
        # Import lazily: the native library/standalone benchmark needs no torch.
        import torch
        if self.synchronize:
            self.synchronize()
        interval = self.signposts.interval(name) if self.signposts else nullcontext()
        with interval, torch.profiler.record_function(name):
            start = time.perf_counter()
            try:
                yield
            finally:
                if self.synchronize:
                    self.synchronize()
                    self.samples_ms.setdefault(name, []).append((time.perf_counter()-start)*1000)


@contextmanager
def profile_pipeline(pipeline, *, signposts=True, mps_ops=True, synchronize=False, library=None):
    """Tracing is separate from benchmark runs. Nested profiling is rejected.

    synchronize=True is diagnostic only: it adds stage-boundary waits.
    PyTorch's per-operation wait_until_completed is always left disabled.
    """
    import torch
    if pipeline._stage_profiler is not None:
        raise RuntimeError("Apple pipeline profiling is already active")
    profiler = StageProfiler(signposts=Signposts(library) if signposts else None,
                             synchronize=torch.mps.synchronize if synchronize else None)
    trace = torch.mps.profiler.profile(mode="interval,event", wait_until_completed=False) if mps_ops else nullcontext()
    pipeline._stage_profiler = profiler
    try:
        with trace:
            try:
                yield profiler
            finally:
                # Drain torch work before stopping GPU operation signposts.
                if mps_ops:
                    torch.mps.synchronize()
    finally:
        pipeline._stage_profiler = None
