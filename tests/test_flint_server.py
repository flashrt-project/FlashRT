"""Serving dependencies are optional; kernel-only environments skip cleanly."""
import asyncio
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')


def test_disconnected_stream_keeps_runtime_locked_until_worker_finishes():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'flint_openai_server.py'
    spec = importlib.util.spec_from_file_location('flint_example_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release = threading.Event()
    starts = []

    def stream(*args, **kwargs):
        starts.append(1)
        yield [1]
        assert release.wait(timeout=5)

    async def scenario():
        engine = module.FlintEngine.__new__(module.FlintEngine)
        engine.lock = asyncio.Lock()
        engine.fe = SimpleNamespace(eog=set(), stream=stream, last_reused=0)
        engine.tok = SimpleNamespace(decode=lambda ids: 'x' * len(ids))
        first = engine.generate([1], 1, False, False)
        assert (await anext(first))[0] == 'text'
        close = asyncio.create_task(first.aclose())
        second = engine.generate([1], 1, False, False)
        waiting = asyncio.create_task(anext(second))
        try:
            await asyncio.sleep(0.05)
            assert engine.lock.locked()
            assert not close.done()
            assert not waiting.done()
            assert len(starts) == 1
        finally:
            release.set()
        await asyncio.wait_for(close, 5)
        assert (await asyncio.wait_for(waiting, 5))[0] == 'text'
        await second.aclose()

    asyncio.run(scenario())


def test_apple_sampling_requests_are_validated():
    pytest.importorskip('httpx')   # fastapi.testclient
    from fastapi.testclient import TestClient
    path = Path(__file__).resolve().parents[1] / 'examples' / 'flint_openai_server.py'
    spec = importlib.util.spec_from_file_location('flint_example_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def stream(ids, max_tokens, stop_ids=None, reuse_prefix=False, **sampling):
        calls.append(sampling)
        yield [5]

    engine = module.FlintEngine.__new__(module.FlintEngine)
    engine.sampling, engine.model_name, engine.lock = True, 'flint', asyncio.Lock()
    engine.prompt_ids = lambda messages, tools: [1, 2, 3]
    engine.fe = SimpleNamespace(eog=set(), stream=stream, last_reused=0, check_block=8,
                                rt=SimpleNamespace(max_seq=4096))
    engine.tok = SimpleNamespace(decode=lambda ids: 'x' * len(ids))
    client = TestClient(module.build_app(engine))
    base = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4}
    for extra in [{'temperature': -1}, {'temperature': 0.7, 'top_k': 0}, {'temperature': 0.7, 'top_k': -1},
                  {'temperature': 0.7, 'top_k': 33}, {'temperature': 0.7, 'top_p': 0},
                  {'temperature': 0.7, 'top_p': 1.5}, {'temperature': 0.7, 'top_k': 'many'}]:
        assert client.post('/v1/chat/completions', json={**base, **extra}).status_code == 400, extra
    assert not calls
    ok = client.post('/v1/chat/completions', json={**base, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 5, 'seed': 11})
    assert ok.status_code == 200
    assert calls == [{'temperature': 0.7, 'top_p': 0.9, 'top_k': 5, 'seed': 11}]
    engine.sampling = False   # the RTX frontend serves greedy decoding only
    assert client.post('/v1/chat/completions', json={**base, 'temperature': 0.7}).status_code == 400
