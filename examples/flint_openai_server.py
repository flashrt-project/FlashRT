#!/usr/bin/env python3
"""FlashRT — Flint (official NVFP4 GGUF) OpenAI-compatible HTTP server.

Serves /v1/chat/completions from the native Flint runtime on SM120 (RTX 5090 class):
W4A16 kernels, CUDA-graph decode, chunked native prefill.

Surface (v1):
  * /v1/chat/completions  (non-stream and stream:true SSE deltas)
  * /v1/models, /health
  * Tools are rendered into the prompt by the model's chat template; tool calls come back as text.
  * Greedy decoding only (temperature 0); the runtime picks tokens with an on-device argmax.
  * ``ignore_eos: true`` (extension) generates exactly ``max_tokens`` tokens, for fixed-length benchmarks.
  * ``cache_prompt`` (extension, default true): reuse the KV cache of the longest common prefix with the
    previous request's prompt and output (one cached sequence), so a follow-up turn only prefills what is
    new. ``timings.cache_n`` reports the reused tokens.
  * Responses carry a ``timings`` extension: prompt tokens, time to first token and decode rate measured
    inside the server.

Limits (v1):
  * Batch size 1: requests are serialised behind one lock.

Usage::

    pip install fastapi uvicorn
    python examples/flint_openai_server.py --gguf /path/to/flint-nvfp4.gguf --port 8000 --max-seq 131072
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('flint_openai_server')


def _sse(obj: Any) -> str:
    return f'data: {json.dumps(obj, ensure_ascii=False)}\n\n'


class FlintEngine:
    """Owns the frontend; one generation at a time."""

    def __init__(self, gguf: str, max_seq: int, model_name: str):
        from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
        t0 = time.perf_counter()
        self.fe = FlintTorchFrontendRtx(gguf, max_seq=max_seq)
        log.info('loaded in %.1f s', time.perf_counter() - t0)
        self.tok = self.fe.tokenizer
        self.model_name = model_name
        self.lock = asyncio.Lock()

    def warmup(self) -> None:
        ids = self.tok.encode(self.tok.apply_chat_template([{'role': 'user', 'content': 'Hello'}],
                                                           add_generation_prompt=True))
        for _ in self.fe.stream(ids, 32):
            pass

    def prompt_ids(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[int]:
        msgs = [{**m, 'content': ''} if m.get('content') is None else m for m in messages]
        return self.tok.encode(self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True))

    async def generate(self, ids: List[int], max_tokens: int, ignore_eos: bool, cache_prompt: bool):
        """Async iterator of ('text', str) events, then ('done', finish_reason, usage, timings)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        finished = loop.create_future()
        eog = self.fe.eog

        def worker():
            t0 = time.perf_counter()
            t_first = None
            out: List[int] = []
            emitted = ''
            finish = 'length'
            try:
                for block in self.fe.stream(ids, max_tokens, stop_ids=() if ignore_eos else None,
                                            reuse_prefix=cache_prompt):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    out.extend(block)
                    if not ignore_eos and out and out[-1] in eog:
                        finish = 'stop'
                    text_ids = out[:-1] if finish == 'stop' else out
                    text = self.tok.decode(text_ids)
                    if text.endswith('�'):        # incomplete UTF-8 sequence: wait for more tokens
                        continue
                    if len(text) > len(emitted):
                        loop.call_soon_threadsafe(queue.put_nowait, ('text', text[len(emitted):]))
                        emitted = text
                t1 = time.perf_counter()
                n = len(out)
                timings = {'prompt_n': len(ids), 'cache_n': self.fe.last_reused, 'ttft_ms': (t_first - t0) * 1e3,
                           'predicted_n': n,
                           'predicted_ms': (t1 - t_first) * 1e3,
                           'predicted_per_second': (n - 1) / (t1 - t_first) if n > 1 and t1 > t_first else None}
                usage = {'prompt_tokens': len(ids), 'completion_tokens': n, 'total_tokens': len(ids) + n}
                loop.call_soon_threadsafe(queue.put_nowait, ('done', finish, usage, timings))
            except Exception as exc:  # surface errors to the request
                loop.call_soon_threadsafe(queue.put_nowait, ('error', repr(exc)))
            finally:
                loop.call_soon_threadsafe(finished.set_result, None)

        async with self.lock:
            threading.Thread(target=worker, daemon=True).start()
            try:
                while True:
                    ev = await queue.get()
                    yield ev
                    if ev[0] in ('done', 'error'):
                        return
            finally:
                # A disconnected client must not let another request overwrite
                # the worker's live buffers. Preserve cancellation after drain.
                while not finished.done():
                    try:
                        await asyncio.shield(finished)
                    except asyncio.CancelledError:
                        continue


def build_app(engine: FlintEngine):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title='FlashRT Flint NVFP4 OpenAI-compatible server')

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': engine.model_name, 'object': 'model', 'owned_by': 'flashrt'}]}

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/v1/chat/completions')
    async def chat_completions(req: Dict[str, Any]):
        messages = req.get('messages')
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, 'messages must be a non-empty list')
        tools = req.get('tools')
        if tools is not None and not isinstance(tools, list):
            raise HTTPException(400, 'tools must be a list')
        try:
            max_tokens = int(req.get('max_tokens', req.get('max_completion_tokens', 1024)))
            temperature = float(req.get('temperature', 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, 'max_tokens / temperature must be numbers') from exc
        if max_tokens < 1:
            raise HTTPException(400, 'max_tokens must be >= 1')
        if temperature != 0.0:
            raise HTTPException(400, 'only greedy decoding (temperature 0) is supported')
        ignore_eos = bool(req.get('ignore_eos', False))
        cache_prompt = bool(req.get('cache_prompt', True))
        stream = bool(req.get('stream', False))
        ids = engine.prompt_ids(messages, tools)
        if len(ids) + max_tokens + 2 * engine.fe.check_block > engine.fe.rt.max_seq:
            raise HTTPException(400, f'prompt ({len(ids)} tokens) + max_tokens exceeds the context '
                                     f'({engine.fe.rt.max_seq})')
        cid = f'chatcmpl-{uuid.uuid4().hex[:24]}'
        created = int(time.time())
        base = {'id': cid, 'created': created, 'model': engine.model_name}

        if not stream:
            parts: List[str] = []
            async for ev in engine.generate(ids, max_tokens, ignore_eos, cache_prompt):
                if ev[0] == 'text':
                    parts.append(ev[1])
                elif ev[0] == 'error':
                    raise HTTPException(500, ev[1])
                else:
                    _, finish, usage, timings = ev
            return {**base, 'object': 'chat.completion',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(parts)},
                                 'finish_reason': finish}],
                    'usage': usage, 'timings': timings}

        async def events():
            first = True
            async for ev in engine.generate(ids, max_tokens, ignore_eos, cache_prompt):
                if ev[0] == 'text':
                    delta = {'content': ev[1]}
                    if first:
                        delta['role'] = 'assistant'
                        first = False
                    yield _sse({**base, 'object': 'chat.completion.chunk',
                                'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})
                elif ev[0] == 'error':
                    yield _sse({'error': {'message': ev[1]}})
                    return
                else:
                    _, finish, usage, timings = ev
                    yield _sse({**base, 'object': 'chat.completion.chunk',
                                'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                                'usage': usage, 'timings': timings})
            yield 'data: [DONE]\n\n'

        return StreamingResponse(events(), media_type='text/event-stream')

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--gguf', required=True, help='official Flint NVFP4 GGUF')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--max-seq', type=int, default=131072, help='context capacity (KV cache) in tokens')
    ap.add_argument('--model-name', default='flint-nvfp4')
    args = ap.parse_args()

    import uvicorn
    engine = FlintEngine(args.gguf, args.max_seq, args.model_name)
    engine.warmup()
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
