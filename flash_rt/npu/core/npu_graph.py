"""Ascend NPU graph capture wrapper.

twin of ``flash_rt/core/cuda_graph.py`` / ``flash_rt/amd/core/hip_graph.py``
over torch_npu's CUDA-graph-compatible capture surface:

    g = NpuGraph()
    with g:                      # torch.npu.graph(self._graph)
        <fixed-address torch_npu ops>
    g.replay()

``torch.npu.NPUGraph`` mirrors ``torch.cuda.CUDAGraph`` (capture_begin /
capture_end / replay / pool); ``torch.npu.graph`` is the context manager.
Capture is on fixed buffers only — the same contract as the CUDA graph
path — so graphs are rebuilt when prompt length / batch shape changes.
"""

from __future__ import annotations


class NpuGraph:
    """Capture a fixed-address sequence of torch_npu ops and replay it."""

    def __init__(self, pool=None, stream=None):
        from flash_rt.npu.core import device
        device.ensure_npu()
        import torch
        import torch_npu  # noqa: F401
        self._torch = torch
        self._graph = torch.npu.NPUGraph()
        self._pool = pool
        self._stream = stream
        self._ctx = None

    # -- context-manager form ------------------------------------------
    def __enter__(self):
        import torch_npu  # noqa: F401
        self._ctx = self._torch.npu.graph(
            self._graph, pool=self._pool, stream=self._stream)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        self._ctx = None

    # -- begin/end form (mirrors the repo's CUDAGraph capture_begin/end) --
    def capture_begin(self):
        self._graph.capture_begin(
            pool=self._pool, stream=self._stream)

    def capture_end(self):
        self._graph.capture_end()

    def replay(self):
        """Replay the captured graph on the current NPU stream."""
        self._graph.replay()

    def reset(self):
        """Free the captured graph (safe to call after error paths)."""
        self._graph.reset()

    @property
    def pool(self):
        return self._graph.pool

    def __repr__(self):
        return (f"NpuGraph(pool={self._pool is not None}, "
                f"stream={self._stream is not None})")
