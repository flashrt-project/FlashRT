"""FlashRT AMD — Framework-agnostic HIP Graph capture/replay.

Twin of flash_rt/core/cuda_graph.py over libamdhip64. Same usage:

    graph = HipGraph()
    stream = graph.create_stream()
    # warmup ... then:
    graph.begin_capture(stream)
    my_kernel(args..., stream)
    graph.end_capture(stream)
    graph.replay(stream)

HIP-vs-CUDA deltas handled here:
  - the 3-arg instantiate is hipGraphInstantiateWithFlags
    (plain hipGraphInstantiate is the 5-arg errorNode/logBuffer form)
  - capture-mode enum verified on hardware: hipStreamCaptureModeRelaxed == 2
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

try:
    _hip = ctypes.CDLL("libamdhip64.so")
except OSError as exc:  # no ROCm runtime on this machine
    # Same contract as hip_buffer: report an unavailable optional
    # backend as ImportError, not a platform-specific OSError.
    raise ImportError(
        "the AMD backend requires the ROCm runtime (libamdhip64.so), "
        f"which could not be loaded: {exc}") from exc


def _configure_signatures() -> None:
    """Declare ctypes signatures — pointer args must never fall back to
    the 32-bit c_int default (see cuda_buffer.py download() incident)."""
    p = ctypes.c_void_p
    pp = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "hipStreamCreate": ([pp], ctypes.c_int),
        "hipStreamBeginCapture": ([p, ctypes.c_uint], ctypes.c_int),
        "hipStreamEndCapture": ([p, pp], ctypes.c_int),
        "hipGraphInstantiateWithFlags": ([pp, p, ctypes.c_ulonglong], ctypes.c_int),
        "hipGraphLaunch": ([p, p], ctypes.c_int),
        "hipStreamSynchronize": ([p], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(_hip, name)
        fn.argtypes = argtypes
        fn.restype = restype


_configure_signatures()


def _check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"HIP error {status}: {msg}")


class HipGraph:
    """Framework-agnostic HIP Graph using raw HIP Runtime API."""

    def __init__(self):
        self._graph = ctypes.c_void_p()
        self._graph_exec = ctypes.c_void_p()
        self._captured = False

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        _check(_hip.hipStreamCreate(ctypes.byref(stream)), "hipStreamCreate")
        return stream

    def begin_capture(self, stream: ctypes.c_void_p):
        """Begin HIP Graph capture on the given stream.

        hipStreamCaptureModeRelaxed=2: only capture ops on THIS stream,
        same rationale as the CUDA path (don't block framework streams).
        """
        _check(_hip.hipStreamBeginCapture(stream, 2), "hipStreamBeginCapture")

    def end_capture(self, stream: ctypes.c_void_p):
        """End capture and instantiate the graph for replay."""
        _check(_hip.hipStreamEndCapture(stream, ctypes.byref(self._graph)),
               "hipStreamEndCapture")
        _check(_hip.hipGraphInstantiateWithFlags(
            ctypes.byref(self._graph_exec), self._graph, 0),
               "hipGraphInstantiateWithFlags")
        self._captured = True

    def replay(self, stream: ctypes.c_void_p):
        """Replay the captured graph (single CPU call → full GPU replay)."""
        if not self._captured:
            raise RuntimeError("No graph captured")
        _check(_hip.hipGraphLaunch(self._graph_exec, stream), "hipGraphLaunch")

    def sync(self, stream: ctypes.c_void_p):
        _check(_hip.hipStreamSynchronize(stream), "hipStreamSynchronize")

    @property
    def captured(self) -> bool:
        return self._captured
