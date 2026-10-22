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
