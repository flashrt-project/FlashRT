"""Generation and capture-state regressions without a checkpoint or GPU."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from flash_rt.frontends.torch.spark_x25_rtx import SparkX25TorchFrontendRtx
from flash_rt.models.spark_x25.pipeline_rtx import SparkX25Runtime


@pytest.fixture
def runtime(monkeypatch, request):
    mirror = getattr(request, 'param', True)
    rt = object.__new__(SparkX25Runtime)
    rt.device = 'cpu'
    rt.max_seq = 32
    rt.cfg = SimpleNamespace(kv_dim=2, num_key_value_heads=1)
    rt.lin_w, rt.ring_w = 3, 2
    rt.kv_offset = [(0, 12, None), (16 if mirror else None, None, 0)]
    rt.k_cache = torch.arange(80 if mirror else 16).float()
    rt.v_cache = rt.k_cache.clone() + 100
    rt.k8_cache = torch.arange(64).to(torch.uint8)
    rt.v8_cache = rt.k8_cache.clone() + 20
    rt.k8_scale = torch.arange(32).float() + 1
    rt.v8_scale = rt.k8_scale.clone() + 3
    rt.pos_i = torch.tensor([7], dtype=torch.int32)
    rt.full_klen = torch.tensor([8], dtype=torch.int32)
    rt.slide_klen = torch.tensor([2], dtype=torch.int32)
    rt.next_token = torch.tensor([4])
    rt.tokens_out = torch.arange(32, dtype=torch.int64)
    rt._capture_stream = None
    rt._loop_graph = None
    rt._loop_steps = 0
    rt._in_loop = False
    stream = SimpleNamespace(wait_stream=lambda other: None)
    active = []

    class Graph:
        def __init__(self):
            self.calls = []
        def replay(self):
            for first in self.calls:
                execute(first)

    @contextmanager
    def capture(graph, **kwargs):
        active.append(graph)
        try:
            yield
        finally:
            active.pop()

    def execute(first):
        if not first:
            rt.pos_i.add_(1)
            rt.tokens_out[int(rt.pos_i) - 1] = rt.next_token[0]
        pos, token = int(rt.pos_i), int(rt.next_token)
        rt.full_klen.fill_(pos + 1)
        rt.slide_klen.fill_(min(pos + 1, rt.ring_w))
        for lin, ring, lin8 in rt.kv_offset:
            if ring is not None:
                offsets = [lin + pos % rt.lin_w * 2,
                           lin + (pos % rt.lin_w + rt.lin_w) * 2,
                           ring + pos % rt.ring_w * 2]
                for cache in (rt.k_cache, rt.v_cache):
                    for offset in offsets:
                        cache[offset:offset + 2].fill_(token)
            else:
                if lin is not None:
                    for cache in (rt.k_cache, rt.v_cache):
                        cache[lin + pos * 2:lin + (pos + 1) * 2].fill_(token)
                for cache in (rt.k8_cache, rt.v8_cache):
                    cache[pos * 2:(pos + 1) * 2].fill_(token)
                for cache in (rt.k8_scale, rt.v8_scale):
                    cache[pos:pos + 1].fill_(token)
        rt.next_token.add_(1)

    def iteration(first):
        if active:
            active[-1].calls.append(first)
        # Also execute during fake capture to exercise restoration of every
        # recorded write, not just the two warmup iterations.
        execute(first)

    rt._decode_iteration = iteration
    monkeypatch.setattr(torch.cuda, 'Stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda s: nullcontext())
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    return rt


@pytest.mark.parametrize('runtime', [True, False], indirect=True)
@pytest.mark.parametrize('steps', [1, 2, 7])
def test_capture_restores_all_persistent_state(runtime, steps):
    before = {key: value.clone() for key, value in vars(runtime).items()
              if isinstance(value, torch.Tensor)}
    runtime.capture_decode_loop(steps)
    for key, value in before.items():
        torch.testing.assert_close(getattr(runtime, key), value, rtol=0, atol=0)
    assert not runtime._in_loop


def test_failed_capture_restores_state(runtime):
    original = runtime._decode_iteration
    before = {key: value.clone() for key, value in vars(runtime).items()
              if isinstance(value, torch.Tensor)}
    def fail(first):
        original(first)
        raise RuntimeError('warmup failed')
    runtime._decode_iteration = fail
    with pytest.raises(RuntimeError, match='warmup failed'):
        runtime.capture_decode_loop(3)
    for key, value in before.items():
        torch.testing.assert_close(getattr(runtime, key), value, rtol=0, atol=0)
    assert not runtime._in_loop


@pytest.fixture
def frontend(runtime):
    fe = object.__new__(SparkX25TorchFrontendRtx)
    fe.runtime, fe.device, fe.max_seq = runtime, 'cpu', runtime.max_seq
    fe._prompt_len = 0
    runtime.act_rows = 3
    runtime.prefill_cap = runtime.max_seq
    runtime._seed_rings = lambda pos: None
    # Run the actual chunk traversal; give each row a distinct prediction.
    def chunk(ids, pos, rows, *, last):
        runtime.logits = torch.nn.functional.one_hot(ids + 1, 64).float()
        return runtime.logits if last else None
    runtime._prefill_chunk = chunk
    # forward only needs the stream handle; fake capture does not launch CUDA.
    return fe


@pytest.mark.parametrize('length', [1, 2, 3, 4, 5, 6, 8])
def test_prefill_returns_final_prompt_prediction(frontend, monkeypatch, length):
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(cuda_stream=0))
    logits = frontend.set_prompt(torch.arange(length))
    assert logits.shape == (1, 64)
    assert int(logits.argmax()) == length


@pytest.mark.parametrize('count', [0, 1, 2, 7])
@pytest.mark.parametrize('graph_steps', [None, 1, 3, 20])
def test_generate_keeps_first_token_and_exact_budget(frontend, monkeypatch, count, graph_steps):
    stream = SimpleNamespace(cuda_stream=0, wait_stream=lambda s: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: stream)
    ids = torch.arange(5)
    for _ in range(2):
        output = frontend.generate(ids, max_new_tokens=count, graph_steps=graph_steps)
        torch.testing.assert_close(output, torch.arange(5 + count))
    if count <= 1:
        assert frontend.runtime._loop_graph is None


@pytest.mark.parametrize('count,steps', [(-1, None), (True, None), (2, 0), (2, -1), (2, True)])
def test_invalid_generation_budget(frontend, count, steps):
    with pytest.raises(ValueError):
        frontend.generate(torch.arange(3), max_new_tokens=count, graph_steps=steps)


def test_empty_prompt_and_context_overflow(frontend):
    with pytest.raises(ValueError, match='at least one token'):
        frontend.set_prompt(torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError, match='exceeds max_seq'):
        frontend.generate(torch.arange(30), max_new_tokens=3)
