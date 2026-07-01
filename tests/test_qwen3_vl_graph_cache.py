"""Regression tests for Qwen3-VL CUDA Graph cache management.

These tests avoid checkpoints and real CUDA Graph capture. They patch the small
``torch.cuda`` surface used by ``Qwen3VlTorchFrontendRtx._ensure_decode_graph``
so the test can drive graph-cache bookkeeping in CI.
"""
from __future__ import annotations

import collections
import contextlib
import sys
import types


class _TensorStub:
    def __init__(self, name):
        self.name = name
        self.copied_from = []

    def clone(self):
        return _TensorStub(self.name + "_clone")

    def copy_(self, other):
        self.copied_from.append(other)
        return self


def test_qwen3_vl_decode_graph_cache_is_lru_bounded(monkeypatch):
    """Distinct decode positions must not grow the graph cache forever."""
    import torch

    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

    class _FakeStream:
        cuda_stream = 0

        def wait_stream(self, _other):
            pass

        def synchronize(self):
            pass

    class _FakeGraph:
        def replay(self):
            pass

    class _FakeLLM:
        def __init__(self):
            self._graph_stream = _FakeStream()
            self._static_token_id = object()
            self.calls = []

        def _rope_cos_sin(self, rope_pos):
            return object(), object()

        def forward_own_decode_nvfp4(self, token, cos, sin, cache_pos):
            self.calls.append((token, cos, sin, cache_pos))

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "stream",
                        lambda _stream: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "graph",
                        lambda _graph, stream=None: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph",
                        lambda: _FakeGraph(), raising=False)

    fe = Qwen3VlTorchFrontendRtx.__new__(Qwen3VlTorchFrontendRtx)
    fe.llm = _FakeLLM()
    fe.max_decode_graphs = 2
    fe._decode_graphs = collections.OrderedDict()

    for cache_pos in range(5):
        fe._ensure_decode_graph(cache_pos, rope_pos=100 + cache_pos)

    assert len(fe._decode_graphs) == fe.max_decode_graphs
    assert list(fe._decode_graphs) == [(3, 103), (4, 104)]


def test_qwen3_vl_prefill_graph_cache_eviction_drops_static_buffers():
    """Prefill graph eviction must also drop that graph's staged inputs."""
    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

    fe = Qwen3VlTorchFrontendRtx.__new__(Qwen3VlTorchFrontendRtx)
    fe.max_prefill_graphs = 2
    fe._prefill_graphs = collections.OrderedDict()
    fe._pg_buffers = collections.OrderedDict()

    for i in range(3):
        fe._prompt = {k: _TensorStub(f"{k}_{i}") for k in fe._PG_KEYS}
        key = fe._stage_prefill_inputs(P=10 + i, S=20 + i, span=(i, i + 10))
        fe._prefill_graphs[key] = object()
        fe._trim_lru_graph_cache(
            fe._prefill_graphs, fe.max_prefill_graphs,
            lambda old_key: fe._pg_buffers.pop(old_key, None))

    assert len(fe._prefill_graphs) == fe.max_prefill_graphs
    assert len(fe._pg_buffers) == fe.max_prefill_graphs
    assert list(fe._prefill_graphs) == list(fe._pg_buffers)
    assert list(fe._prefill_graphs) == [(11, 21, 1, 11), (12, 22, 2, 12)]


def test_qwen3_vl_prefill_graph_restages_after_clear(monkeypatch):
    """clear_graphs() should not force callers to rerun set_prompt()."""
    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

    fe = Qwen3VlTorchFrontendRtx.__new__(Qwen3VlTorchFrontendRtx)
    fe.max_prefill_graphs = 2
    fe._prefill_graphs = collections.OrderedDict()
    fe._pg_buffers = collections.OrderedDict()
    fe._decode_graphs = collections.OrderedDict()
    fe._prompt = {k: _TensorStub(k) for k in fe._PG_KEYS}
    fe._prompt.update({"S": 20, "pg_key": (10, 20, 3, 13)})

    stale_graph = object()
    fe._stage_prefill_inputs(P=10, S=20, span=(3, 13))
    fe._prefill_graphs[(10, 20, 3, 13)] = stale_graph
    fe.clear_graphs()
    fe._prefill_graphs[(10, 20, 3, 13)] = stale_graph

    class _FakeGraph:
        def replay(self):
            pass

    graph = _FakeGraph()
    monkeypatch.setattr(
        Qwen3VlTorchFrontendRtx, "_capture_prefill_graph",
        lambda self, st, P, S, a, b: graph)

    class _FakeLLM:
        _cfg = {"hidden_size": 1, "vocab_size": 1}
        _weights = type(
            "Weights", (), {"ptrs": {"lm_head_w": 0}})()

    class _FakeTensor:
        def __getitem__(self, _key):
            return self

        def contiguous(self):
            return self

        def data_ptr(self):
            return 0

    class _FakeStream:
        cuda_stream = 0

    fake_fvk = types.SimpleNamespace(
        bf16_matmul_bf16=lambda *args: None,
        bf16_matmul_qwen36_bf16=lambda *args: None)
    import flash_rt
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_kernels", fake_fvk)
    monkeypatch.setattr(flash_rt, "flash_rt_kernels", fake_fvk, raising=False)
    import torch
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda: None, raising=False)

    fe.llm = _FakeLLM()
    fe._pg_last_hidden = _FakeTensor()
    fe._pg_logits = _FakeTensor()

    assert fe.prefill_graph() is fe._pg_logits
    assert list(fe._pg_buffers) == [(10, 20, 3, 13)]
    assert list(fe._prefill_graphs) == [(10, 20, 3, 13)]
    assert fe._prefill_graphs[(10, 20, 3, 13)] is graph


def test_qwen3_vl_graph_cache_stats_and_clear_graphs():
    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

    fe = Qwen3VlTorchFrontendRtx.__new__(Qwen3VlTorchFrontendRtx)
    fe.max_prefill_graphs = 7
    fe.max_decode_graphs = 5
    fe._prefill_graphs = collections.OrderedDict([(("p0",), object())])
    fe._pg_buffers = collections.OrderedDict([(("p0",), object())])
    fe._decode_graphs = collections.OrderedDict([
        ((3, 103), object()), ((4, 104), object())])

    stats = fe.graph_cache_stats()
    assert stats["prefill"]["max_graphs"] == 7
    assert stats["prefill"]["graph_count"] == 1
    assert stats["prefill"]["buffer_count"] == 1
    assert stats["prefill"]["graph_keys"] == [("p0",)]
    assert stats["decode"]["max_graphs"] == 5
    assert stats["decode"]["graph_count"] == 2
    assert stats["decode"]["graph_keys"] == [(3, 103), (4, 104)]

    fe.clear_graphs()

    assert fe._prefill_graphs == collections.OrderedDict()
    assert fe._pg_buffers == collections.OrderedDict()
    assert fe._decode_graphs == collections.OrderedDict()


def test_qwen3_vl_sm89_text_decode_graph_cache_is_lru_bounded(monkeypatch):
    import torch

    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
        Qwen3VlFp8Sm89TextFrontend,
    )

    class _FakeStream:
        cuda_stream = 0

        def wait_stream(self, _other):
            pass

        def synchronize(self):
            pass

    class _FakeGraph:
        def replay(self):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "stream",
                        lambda _stream: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "graph",
                        lambda _graph, stream=None: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph",
                        lambda: _FakeGraph(), raising=False)
    monkeypatch.setattr(torch, "inference_mode",
                        lambda: contextlib.nullcontext())

    fe = Qwen3VlFp8Sm89TextFrontend.__new__(
        Qwen3VlFp8Sm89TextFrontend)
    fe.max_decode_graphs = 2
    fe._decode_graphs = collections.OrderedDict()
    fe._graph_stream = _FakeStream()
    fe._static_token_id = object()
    fe._rope_cos_sin = lambda pos: (object(), object())
    fe.forward_own_decode_fp8 = lambda token, cos, sin, pos: None

    for cur_pos in range(5):
        fe._ensure_decode_graph(cur_pos)

    assert len(fe._decode_graphs) == fe.max_decode_graphs
    assert list(fe._decode_graphs) == [3, 4]
    stats = fe.graph_cache_stats()
    assert stats["decode"]["graph_count"] == 2
    fe.clear_graphs()
    assert fe._decode_graphs == collections.OrderedDict()


def test_qwen3_vl_sm89_multimodal_graph_caches_are_lru_bounded(monkeypatch):
    import torch

    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )

    class _FakeStream:
        cuda_stream = 0

        def wait_stream(self, _other):
            pass

        def synchronize(self):
            pass

    class _FakeGraph:
        def replay(self):
            pass

    class _FakeLLM:
        def __init__(self):
            self._graph_stream = _FakeStream()
            self._static_token_id = object()
            self._logits_buf = object()
            self.clear_count = 0

        def _rope_cos_sin(self, rope_pos):
            return object(), object()

        def forward_own_decode_fp8(self, token, cos, sin, cache_pos):
            pass

        def clear_graphs(self):
            self.clear_count += 1

        def graph_cache_stats(self):
            return {"decode": {"graph_count": 0}}

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "stream",
                        lambda _stream: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "graph",
                        lambda _graph, stream=None: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph",
                        lambda: _FakeGraph(), raising=False)
    monkeypatch.setattr(torch, "inference_mode",
                        lambda: contextlib.nullcontext())

    fe = Qwen3VlFp8Sm89Frontend.__new__(Qwen3VlFp8Sm89Frontend)
    fe.max_prefill_graphs = 2
    fe.max_decode_graphs = 2
    fe._prefill_graphs = collections.OrderedDict()
    fe._pg_buffers = collections.OrderedDict()
    fe._decode_graphs = collections.OrderedDict()
    fe.llm = _FakeLLM()

    for i in range(3):
        fe._prompt = {k: _TensorStub(f"{k}_{i}") for k in fe._PG_KEYS}
        key = fe._stage_prefill_inputs(P=10 + i, S=20 + i, span=(i, i + 10))
        fe._prefill_graphs[key] = object()
        fe._trim_lru_graph_cache(
            fe._prefill_graphs, fe.max_prefill_graphs,
            lambda old_key: fe._pg_buffers.pop(old_key, None))

    assert len(fe._prefill_graphs) == fe.max_prefill_graphs
    assert len(fe._pg_buffers) == fe.max_prefill_graphs
    assert list(fe._prefill_graphs) == list(fe._pg_buffers)
    assert list(fe._prefill_graphs) == [(11, 21, 1, 11), (12, 22, 2, 12)]

    for i in range(5):
        fe._ensure_decode_graph(cache_pos=i, rope_pos=100 + i)

    assert len(fe._decode_graphs) == fe.max_decode_graphs
    assert list(fe._decode_graphs) == [(3, 103), (4, 104)]
    stats = fe.graph_cache_stats()
    assert stats["prefill"]["graph_count"] == 2
    assert stats["prefill"]["buffer_count"] == 2
    assert stats["decode"]["graph_count"] == 2
    assert stats["text"]["decode"]["graph_count"] == 0

    fe.clear_graphs()

    assert fe._prefill_graphs == collections.OrderedDict()
    assert fe._pg_buffers == collections.OrderedDict()
    assert fe._decode_graphs == collections.OrderedDict()
    assert fe.llm.clear_count == 1


def test_qwen3_vl_rtx_bf16_graph_caches_are_lru_bounded(monkeypatch):
    import torch

    from flash_rt.frontends.torch.qwen3_vl_rtx_bf16 import (
        Qwen3VlTorchFrontendRtxBF16,
    )

    class _FakeStream:
        cuda_stream = 0

        def wait_stream(self, _other):
            pass

        def synchronize(self):
            pass

    class _FakeGraph:
        def replay(self):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)
    monkeypatch.setattr(torch.cuda, "stream",
                        lambda _stream: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "graph",
                        lambda _graph, stream=None: contextlib.nullcontext(),
                        raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph",
                        lambda: _FakeGraph(), raising=False)
    monkeypatch.setattr(torch, "inference_mode",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        Qwen3VlTorchFrontendRtxBF16, "_decode_token_tensor",
        lambda self, token, *, cache_pos, rope_pos: None)

    fe = Qwen3VlTorchFrontendRtxBF16.__new__(
        Qwen3VlTorchFrontendRtxBF16)
    fe.max_prefill_graphs = 2
    fe.max_decode_graphs = 2
    fe._prefill_graphs = collections.OrderedDict()
    fe._pg_buffers = collections.OrderedDict()
    fe._decode_graphs = collections.OrderedDict()
    fe._graph_stream = _FakeStream()
    fe._static_token_id = object()

    for i in range(3):
        fe._prompt = {k: _TensorStub(f"{k}_{i}") for k in fe._PG_KEYS}
        key = fe._stage_prefill_inputs(P=10 + i, S=20 + i, span=(i, i + 10))
        fe._prefill_graphs[key] = object()
        fe._trim_lru_graph_cache(
            fe._prefill_graphs, fe.max_prefill_graphs,
            lambda old_key: fe._pg_buffers.pop(old_key, None))

    assert len(fe._prefill_graphs) == fe.max_prefill_graphs
    assert len(fe._pg_buffers) == fe.max_prefill_graphs
    assert list(fe._prefill_graphs) == list(fe._pg_buffers)
    assert list(fe._prefill_graphs) == [(11, 21, 1, 11), (12, 22, 2, 12)]

    for i in range(5):
        fe._ensure_decode_graph(cache_pos=i, rope_pos=100 + i)

    assert len(fe._decode_graphs) == fe.max_decode_graphs
    assert list(fe._decode_graphs) == [(3, 103), (4, 104)]
    stats = fe.graph_cache_stats()
    assert stats["prefill"]["graph_count"] == 2
    assert stats["prefill"]["buffer_count"] == 2
    assert stats["decode"]["graph_count"] == 2

    fe.clear_graphs()

    assert fe._prefill_graphs == collections.OrderedDict()
    assert fe._pg_buffers == collections.OrderedDict()
    assert fe._decode_graphs == collections.OrderedDict()


def test_qwen3_vl_rtx_bf16_prefill_graph_restages_after_clear(monkeypatch):
    from flash_rt.frontends.torch.qwen3_vl_rtx_bf16 import (
        Qwen3VlTorchFrontendRtxBF16,
    )

    fe = Qwen3VlTorchFrontendRtxBF16.__new__(
        Qwen3VlTorchFrontendRtxBF16)
    fe.max_prefill_graphs = 2
    fe.max_decode_graphs = 2
    fe._prefill_graphs = collections.OrderedDict()
    fe._pg_buffers = collections.OrderedDict()
    fe._decode_graphs = collections.OrderedDict()
    fe._prompt = {k: _TensorStub(k) for k in fe._PG_KEYS}
    fe._prompt.update({"S": 20, "pg_key": (10, 20, 3, 13)})
    fe._logits = object()
    fe._cur_pos = 0

    stale_graph = object()
    fe._stage_prefill_inputs(P=10, S=20, span=(3, 13))
    fe._prefill_graphs[(10, 20, 3, 13)] = stale_graph
    fe.clear_graphs()
    fe._prefill_graphs[(10, 20, 3, 13)] = stale_graph

    class _FakeGraph:
        def replay(self):
            pass

    graph = _FakeGraph()
    monkeypatch.setattr(
        Qwen3VlTorchFrontendRtxBF16, "_capture_prefill_graph",
        lambda self, st, key: graph)

    assert fe.prefill_graph() is fe._logits
    assert list(fe._pg_buffers) == [(10, 20, 3, 13)]
    assert list(fe._prefill_graphs) == [(10, 20, 3, 13)]
    assert fe._prefill_graphs[(10, 20, 3, 13)] is graph
    assert fe._cur_pos == 20


def test_qwen3_vl_rtx_bf16_graph_cache_init_surface(monkeypatch):
    import importlib.machinery
    import inspect

    from flash_rt.frontends.torch.qwen3_vl_rtx_bf16 import (
        Qwen3VlTorchFrontendRtxBF16,
    )

    sig = inspect.signature(Qwen3VlTorchFrontendRtxBF16)
    assert "max_prefill_graphs" in sig.parameters
    assert "max_decode_graphs" in sig.parameters

    monkeypatch.setenv("FLASHRT_QWEN3_VL_PREFILL_GRAPH_CACHE_MAX", "17")
    monkeypatch.setenv("FLASHRT_QWEN3_VL_DECODE_GRAPH_CACHE_MAX", "19")

    fe = Qwen3VlTorchFrontendRtxBF16.__new__(
        Qwen3VlTorchFrontendRtxBF16)

    class _StopInit(Exception):
        pass

    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    fake_torch.bfloat16 = object()
    fake_torch.long = object()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability(_device):
            raise _StopInit

    fake_torch.cuda = _FakeCuda
    fake_torch.device = lambda device: types.SimpleNamespace(
        type="cuda", value=device)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_weights = types.ModuleType(
        "flash_rt.frontends.torch._qwen3_vl_bf16_weights")
    fake_weights.assert_extraction_invariants_qwen3_vl_bf16 = lambda *_: None
    fake_weights.extract_weights_qwen3_vl_bf16 = lambda *_args, **_kwargs: None
    fake_vision = types.ModuleType(
        "flash_rt.frontends.torch._qwen3_vl_vision_rtx")
    fake_vision.Qwen3VlVisionRtx = object
    fake_attn = types.ModuleType(
        "flash_rt.hardware.rtx.attn_backend_qwen3")
    fake_attn.RtxFlashAttnBackendQwen3 = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "flash_rt.frontends.torch._qwen3_vl_bf16_weights",
        fake_weights)
    monkeypatch.setitem(
        sys.modules, "flash_rt.frontends.torch._qwen3_vl_vision_rtx",
        fake_vision)
    monkeypatch.setitem(
        sys.modules, "flash_rt.hardware.rtx.attn_backend_qwen3", fake_attn)
    try:
        Qwen3VlTorchFrontendRtxBF16.__init__(fe, "checkpoint")
    except _StopInit:
        pass

    assert fe.max_prefill_graphs == 17
    assert fe.max_decode_graphs == 19
    assert fe._prefill_graphs == collections.OrderedDict()
    assert fe._pg_buffers == collections.OrderedDict()
    assert fe._decode_graphs == collections.OrderedDict()
