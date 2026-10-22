"""CPU regressions for sliding KV lifetime; no checkpoint or CUDA allocation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
from flash_rt.frontends.torch._flint_rtx_runtime import FlintRuntime


@pytest.mark.parametrize("logical_end,physical_end,fork", [
    (20000, 20000, 1000),   # the review's early fork after wrapping the ring
    (20000, 20000, 4127),   # required history starts one slot before the retained range
    (20000, 20000, 4128),   # equality at the retained boundary is safe
    (20000, 20000, 19900),  # a recent prefix remains reusable
    (16384, 16400, 513),   # discarded lookahead also overwrites old KV
    (16384, 16400, 528),
    (1000, 1000, 63),      # retain the minimum-prefix policy
    (1000, 1000, 64),
])
def test_stream_reuses_only_resident_history(monkeypatch, logical_end, physical_end, fork):
    ring, window = 16384, 513
    # Position tags form an independent oracle for which historical KV survived.
    slots = [-1] * ring
    for p in range(physical_end):
        slots[p % ring] = p
    resident = all(slots[p % ring] == p for p in range(max(0, fork - window + 1), fork))
    expected = fork if resident and fork >= 64 else 0
    starts = []
    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    fe = FlintTorchFrontendRtx.__new__(FlintTorchFrontendRtx)
    fe.cfg = SimpleNamespace(vocab_size=10, sliding_window=window)
    fe.rt = SimpleNamespace(max_seq=32768, ring=ring, swa_kv_start=max(0, physical_end - ring),
                            tokens=torch.full((32768,), 3, dtype=torch.int32),
                            stream=SimpleNamespace(synchronize=lambda: None),
                            prefill=lambda ids, start: starts.append(start))
    fe._host_tokens = torch.empty(32768, dtype=torch.int32)
    fe._kv_ids = [1] * logical_end
    fe.min_reuse, fe.check_block, fe.eog = 64, 16, set()
    assert list(fe.stream([1] * fork + [2] * 32, 1, reuse_prefix=True)) == [[3]]
    assert starts == [expected]
    assert fe.last_reused == expected


@pytest.mark.parametrize("graphs,expected", [(False, 1), (True, 16)])
def test_prefill_tracks_physical_writes_including_graph_padding(graphs, expected):
    rt = FlintRuntime.__new__(FlintRuntime)
    rt.ring, rt.swa_kv_start = 16384, 0
    rt.pf_capacity, rt.max_seq, rt.pf_graph_on = 8192, 32768, graphs
    writes = []
    rt.launch_prefill_chunk = lambda m, pos0: writes.append((pos0, m))
    rt._pf_graph = lambda b, pos0: SimpleNamespace(replay=lambda: writes.append((pos0, b)))
    rt.fl = SimpleNamespace(advance_pos=lambda *args: None)
    rt.pos = SimpleNamespace(data_ptr=lambda: 0)
    rt.stream = SimpleNamespace(cuda_stream=0)
    rt.run_prefill_chunk(65, 16320)
    assert writes == [(16320, 80 if graphs else 65)]
    assert rt.swa_kv_start == expected
    # Rewinding a logical sequence cannot bring overwritten slots back.
    rt.run_prefill_chunk(16, 10000)
    assert rt.swa_kv_start == expected


@pytest.mark.parametrize("eager", [False, True])
def test_decode_tracks_queued_lookahead_and_keeps_bound_on_rewind(monkeypatch, eager):
    monkeypatch.setenv("FLINT_EAGER", "1" if eager else "0")
    rt = FlintRuntime.__new__(FlintRuntime)
    rt.ring, rt.max_seq, rt.host_pos = 16384, 32768, 20000
    rt.swa_kv_start = 20000 - rt.ring
    writes = []
    step = lambda **kwargs: writes.append(rt.host_pos)
    rt._graph_for = lambda pos: SimpleNamespace(replay=step)
    rt.torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda stream: nullcontext()))
    rt.stream = None
    rt.attn_buckets = [(32768, 8, 8, None)]
    rt.launch_step_fused = step
    rt.replay(16)
    assert writes == list(range(20000, 20016))
    assert rt.swa_kv_start == 20016 - rt.ring
    rt.host_pos = 10000
    rt.replay(1)
    assert rt.swa_kv_start == 20016 - rt.ring


def test_runtime_refuses_partial_prefill_with_evicted_history():
    rt = FlintRuntime.__new__(FlintRuntime)
    rt.torch = None
    rt.cfg = SimpleNamespace(sliding_window=513)
    rt.swa_kv_start = 3616
    with pytest.raises(ValueError, match="sliding.*prefill"):
        rt.prefill([1] * 1100, start=1000)


def test_full_reset_starts_a_new_cache_lifetime():
    rt = FlintRuntime.__new__(FlintRuntime)
    rt.torch = SimpleNamespace(tensor=torch.tensor, int32=torch.int32,
                               cuda=SimpleNamespace(synchronize=lambda dev: None))
    rt.max_seq, rt.dev = 32768, None
    rt.tokens = torch.zeros(32768, dtype=torch.int32)
    rt.pos, rt.n_prompt_dev = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    rt.swa_kv_start = 3616
    rt.reset([1, 2])
    assert rt.swa_kv_start == 0
    assert rt.host_pos == 0
