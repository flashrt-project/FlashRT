#!/usr/bin/env python3
"""FlashRT — Qwen3-8B-NVFP4 OpenAI-compatible HTTP server.

Provides /v1/chat/completions backed by the FlashRT NVFP4 path on
RTX 5090. Clients targeting the OpenAI API can swap their base URL to
this server without code changes.

Surface (v1):
  * /v1/chat/completions  (non-stream + stream:true with token-by-token
                           SSE deltas)
  * /v1/models           (returns a single canonical model id)
  * /health
  * Tools / function calling via Qwen3 chat-template native support.
    The model emits <tool_call>{...}</tool_call> blocks; our streamer
    emits OpenAI-shape tool_calls deltas as the JSON closes.
  * Sampling: temperature / top_p / top_k / seed / stop / max_tokens.
    Greedy when temperature==0 (default), else multinomial after
    top_k+top_p truncation.

Limits (v1):
  * Batch size 1 — concurrent requests are serialised behind a single
    asyncio lock. Multi-tenant serving belongs in a higher layer.
  * Single graph-warmed shape ladder is captured at startup; first
    request at a new (prompt_len) shape pays a small one-time
    capture cost.

Usage::

    pip install fastapi uvicorn

    python examples/qwen3_openai_server.py \\
        --checkpoint /path/to/Qwen3-8B-Instruct-NVFP4 \\
        --port 8000 \\
        --warmup 32:128,128:256,256:256

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"qwen3-8b-nvfp4",
             "messages":[{"role":"user","content":"Hi"}],
             "max_tokens":64,
             "stream":true}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('qwen3_openai_server')


# Qwen3-Instruct tool-call format: model emits
#   <tool_call>{"name": "fn_name", "arguments": {...}}</tool_call>
# anywhere in the assistant turn. We parse incrementally during stream.
_TOOL_CALL_OPEN = '<tool_call>'
_TOOL_CALL_CLOSE = '</tool_call>'


# ────────────────────────────────────────────────────────────────────
# Sampling
# ────────────────────────────────────────────────────────────────────
def _sample_token(
    logits,                  # (vocab,) bf16/fp32
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    rng=None,
) -> int:
    """Greedy if temperature == 0 (or top_k == 1), else top-k+top-p multinomial.

    All operations on-device bf16/fp32. Returns Python int token id.
    """
    import torch
    if temperature <= 0.0 or top_k == 1:
        return int(logits.argmax(dim=-1).item())

    L = logits.float() / max(temperature, 1e-6)
    if top_k and 0 < top_k < L.numel():
        topv, topi = torch.topk(L, top_k)
        mask = torch.full_like(L, float('-inf'))
        mask.scatter_(0, topi, topv)
        L = mask

    if 0.0 < top_p < 1.0:
        sorted_v, sorted_idx = torch.sort(L, descending=True)
        sorted_p = torch.softmax(sorted_v, dim=-1)
        cum = sorted_p.cumsum(dim=-1)
        cutoff_mask = cum > top_p
        # Keep the first cutoff index — shift right by 1.
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False
        sorted_v[cutoff_mask] = float('-inf')
        L = torch.full_like(L, float('-inf'))
        L.scatter_(0, sorted_idx, sorted_v)

    probs = torch.softmax(L, dim=-1)
    if rng is not None:
        return int(torch.multinomial(probs, 1, generator=rng).item())
    return int(torch.multinomial(probs, 1).item())


# ────────────────────────────────────────────────────────────────────
# Stop-string + tool-call streaming parser
# ────────────────────────────────────────────────────────────────────
class StreamParser:
    """Incrementally split assistant tokens into:
      * "content" (free text deltas)
      * "tool_calls" (parsed JSON objects emitted as OAI tool_call deltas)
    Plus stop-string detection for early termination.
    """

    def __init__(self, tokenizer, stop_strings: Optional[List[str]] = None):
        self.tok = tokenizer
        self._buffer = ''         # un-flushed text (may contain partial tags)
        self._content_pos = 0     # index up to which we have flushed content
        self._in_tool = False
        self._tool_buffer = ''
        self._stop_strings = stop_strings or []
        self._tool_calls_emitted: List[dict] = []
        # OAI tool_call indexer.
        self._tool_call_idx = 0

    def feed(
        self, new_token_ids: List[int], *, final: bool = False,
    ) -> Tuple[str, List[dict], bool]:
        """Decode the running token list and return (delta_text,
        new_tool_calls, stop_hit).

        delta_text: clean content delta (excluding tool-call wrappers).
        new_tool_calls: list of {index, id, type, function: {name, arguments}}
            objects newly closed in this feed.
        stop_hit: True iff any stop string was found.

        Args:
          new_token_ids: tokens to append to the running stream (may be empty
            on the final flush).
          final: True iff no more tokens will arrive (EOS / max_tokens / stop
            string already hit upstream). When set, the entire buffer is
            flushed — no partial-tag hold-back, no max-stop-string-len
            hold-back.
        """
        # Append decoded fragment.
        if new_token_ids:
            try:
                fragment = self.tok.decode(new_token_ids, skip_special_tokens=False)
            except Exception:
                fragment = ''
            self._buffer += fragment

        delta_text = ''
        new_tool_calls: List[dict] = []
        stop_hit = False

        # Stop-string detection — scan the FULL buffer (not just the
        # flushable head) for any user-supplied stop. If a stop is in
        # the buffer, truncate the buffer there and mark stop_hit. The
        # stop string itself is dropped from the output (OpenAI semantics).
        if self._stop_strings and not self._in_tool:
            best_idx = -1
            for ss in self._stop_strings:
                idx = self._buffer.find(ss)
                if idx >= 0 and (best_idx < 0 or idx < best_idx):
                    best_idx = idx
            if best_idx >= 0:
                self._buffer = self._buffer[:best_idx]
                stop_hit = True

        # The buffer may need a tail hold for two reasons:
        #   (a) the tail of `_buffer` could be a partial `<tool_call>`
        #       opening tag whose final chars haven't streamed yet;
        #   (b) the tail could complete a stop string on the next feed.
        # Hold-back size = max(len(open_tag), max(stop_string_lens)) - 1.
        # On `final=True` (or once stop_hit fired) the hold-back is 0.
        max_stop_len = (
            max((len(s) for s in self._stop_strings), default=0)
            if self._stop_strings else 0
        )
        hold = (
            0 if (final or stop_hit)
            else max(len(_TOOL_CALL_OPEN), max_stop_len) - 1
        )

        while True:
            if self._in_tool:
                close_idx = self._buffer.find(_TOOL_CALL_CLOSE)
                if close_idx < 0:
                    self._tool_buffer += self._buffer
                    self._buffer = ''
                    break
                self._tool_buffer += self._buffer[:close_idx]
                self._buffer = self._buffer[close_idx + len(_TOOL_CALL_CLOSE):]
                self._in_tool = False
                # Try to parse the tool-call JSON.
                tc = self._parse_tool_call(self._tool_buffer.strip())
                self._tool_buffer = ''
                if tc is not None:
                    new_tool_calls.append(tc)
                    self._tool_calls_emitted.append(tc)
                continue

            open_idx = self._buffer.find(_TOOL_CALL_OPEN)
            if open_idx < 0:
                # No open tag in buffer — flush all but the hold-back tail.
                safe = max(0, len(self._buffer) - hold)
                if safe > 0:
                    delta_text += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break
            # Flush text before the open tag.
            delta_text += self._buffer[:open_idx]
            self._buffer = self._buffer[open_idx + len(_TOOL_CALL_OPEN):]
            self._in_tool = True
            # loop continues into in_tool branch

        return delta_text, new_tool_calls, stop_hit

    def _parse_tool_call(self, raw: str) -> Optional[dict]:
        """Parse the JSON inside a <tool_call>...</tool_call> block.

        Qwen3 emits compact JSON like {"name":"f","arguments":{...}}.
        Some fine-tunes wrap it in code fences — handle both.
        """
        s = raw.strip()
        if s.startswith('```'):
            # strip code fence
            s = re.sub(r'^```[^\n]*\n', '', s)
            if s.endswith('```'):
                s = s[:-3]
            s = s.strip()
        try:
            obj = json.loads(s)
        except Exception:
            return None
        name = obj.get('name')
        args = obj.get('arguments', obj.get('parameters', {}))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        idx = self._tool_call_idx
        self._tool_call_idx += 1
        return {
            'index': idx,
            'id': f'call_{uuid.uuid4().hex[:24]}',
            'type': 'function',
            'function': {'name': name, 'arguments': args},
        }


# ────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────
class Qwen3Engine:
    """Async wrapper around the Qwen3TorchFrontendRtx with streaming."""

    def __init__(self, *, checkpoint: str, device: str, model_name: str,
                  max_seq: int, max_q_seq: int):
        import torch
        from flash_rt.frontends.torch.qwen3_rtx import (
            Qwen3TorchFrontendRtx,
        )

        log.info('loading NVFP4 ckpt from %s ...', checkpoint)
        t0 = time.perf_counter()
        self.fe = Qwen3TorchFrontendRtx(
            checkpoint, device=device,
            max_seq=max_seq, max_q_seq=max_q_seq,
        )
        log.info('loaded in %.1f s', time.perf_counter() - t0)
        self.model_name = model_name
        self.lock = asyncio.Lock()
        self._torch = torch

    def warmup(self, shapes: List[Tuple[int, int]]) -> None:
        """Pre-capture decode + prefill graphs over each (prompt_len,
        max_tokens) shape so first real requests at those sizes hit
        warm graphs.
        """
        if not shapes:
            return
        torch = self._torch
        # Pre-capture all prefill bucket graphs once. Cheap (one
        # capture per bucket, ~5-15 ms each) and enables the
        # prefill_with_graph fast path for any request whose prompt
        # length fits the bucket ladder.
        t0 = time.perf_counter()
        self.fe.warmup_prefill_graphs()
        torch.cuda.synchronize()
        log.info('  warm prefill graphs (%d buckets) in %.1f s',
                 len(self.fe.prefill_buckets), time.perf_counter() - t0)

        log.info('warmup: %d (prompt, max_tok) shape(s)', len(shapes))
        for prompt_len, max_tok in shapes:
            t0 = time.perf_counter()
            dummy_text = 'a ' * (max(1, prompt_len) - 1)
            input_ids = self.fe._tokenizer(
                dummy_text, return_tensors='pt').input_ids.to('cuda')
            if input_ids.shape[1] >= prompt_len:
                input_ids = input_ids[:, :prompt_len]
            else:
                pad = torch.full(
                    (1, prompt_len - input_ids.shape[1]),
                    self.fe._tokenizer.pad_token_id or 0,
                    device='cuda', dtype=torch.long,
                )
                input_ids = torch.cat([input_ids, pad], dim=1)
            self.fe.reset_state()
            torch.cuda.synchronize()
            with torch.inference_mode():
                # Use the captured prefill graph if the prompt fits a
                # bucket; falls back to eager forward_prefill_nvfp4
                # internally otherwise. Either way leaves the KV cache
                # populated for the decode warmup that follows.
                self.fe.prefill_with_graph(input_ids)
                self.fe.warmup_decode_graphs(
                    prompt_len, prompt_len + max_tok,
                )
            torch.cuda.synchronize()
            log.info('  warm (P=%d, max_tok=%d) in %.1f s',
                     prompt_len, max_tok, time.perf_counter() - t0)

    def _render(self, messages: List[Dict[str, Any]],
                  tools: Optional[List[Dict[str, Any]]]):
        """Apply the chat template (with optional tools) to messages.

        OpenAI lets `assistant.content` be `null` when `tool_calls` is
        set, but the Qwen3 chat template iterates `content` directly
        and crashes on `None`. Normalize by mapping `None` → '' before
        rendering — semantically equivalent (no text content).
        """
        normalized = []
        for m in messages:
            if m.get('content') is None:
                m = {**m, 'content': ''}
            normalized.append(m)
        return self.fe._tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )

    async def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        stop: Optional[List[str]],
    ):
        """Async generator yielding (kind, payload) events:
          ('content', str)               — content delta
          ('tool_calls', list[dict])     — parsed tool_call deltas
          ('finish', reason: str, usage: dict)
        """
        torch = self._torch
        async with self.lock:
            prompt = self._render(messages, tools)
            input_ids = self.fe._tokenizer(
                prompt, return_tensors='pt').input_ids.to('cuda')
            P = int(input_ids.shape[1])

            rng = None
            if seed is not None:
                rng = torch.Generator(device='cuda')
                rng.manual_seed(int(seed))

            parser = StreamParser(self.fe._tokenizer, stop_strings=stop)
            eos = self.fe._tokenizer.eos_token_id

            t0 = time.perf_counter()
            self.fe.reset_state()
            with torch.inference_mode():
                # prefill_with_graph picks the smallest bucket >= P
                # and replays the captured graph; falls back to
                # eager forward_prefill_nvfp4 internally if P exceeds
                # the largest bucket. _logits_buf[:1] holds the next-
                # token logits either way.
                self.fe.prefill_with_graph(input_ids)
                ttft = time.perf_counter() - t0

                # Make sure decode graphs over [P, P+max_tokens) are
                # warm. Must stay inside inference_mode — torch 2.9+
                # rejects graph capture that touches inference tensors
                # (the prefill above marks _logits_buf / KV cache /
                # hidden buffers as inference tensors) from outside
                # an inference_mode context.
                self.fe.warmup_decode_graphs(P, P + max_tokens)

            new_tokens: List[int] = []
            cur_pos = P
            finish_reason = 'length'

            for step in range(max_tokens):
                # Sample from the current logits buffer.
                tok = _sample_token(
                    self.fe._logits_buf[0],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    rng=rng,
                )
                new_tokens.append(tok)

                # EOS check (engine-side, before emitting).
                if eos is not None and tok == eos:
                    delta, tcs, _ = parser.feed([], final=True)
                    if delta:
                        yield ('content', delta)
                    if tcs:
                        yield ('tool_calls', tcs)
                    finish_reason = (
                        'tool_calls' if parser._tool_calls_emitted
                        and not parser._buffer.strip() else 'stop'
                    )
                    break

                # Stream parse the new token.
                delta, tcs, stop_hit = parser.feed([tok])
                if delta:
                    yield ('content', delta)
                if tcs:
                    yield ('tool_calls', tcs)
                if stop_hit:
                    finish_reason = 'stop'
                    break

                # Advance KV cache via the warm decode graph.
                with torch.inference_mode():
                    self.fe.decode_step_with_graph(
                        torch.tensor([[tok]], device='cuda', dtype=torch.long),
                        cur_pos,
                    )
                cur_pos += 1

                # Yield to event loop so the SSE chunks can flush.
                if step % 8 == 0:
                    await asyncio.sleep(0)

            else:
                # Loop exhausted max_tokens.
                # Final flush of any buffered text.
                delta, tcs, _ = parser.feed([], final=True)
                if delta:
                    yield ('content', delta)
                if tcs:
                    yield ('tool_calls', tcs)

            wall = time.perf_counter() - t0
            usage = {
                'prompt_tokens': P,
                'completion_tokens': len(new_tokens),
                'total_tokens': P + len(new_tokens),
                'ttft_ms': round(ttft * 1000, 1),
                'wall_s': round(wall, 3),
                'tok_per_s': round(len(new_tokens) / wall, 1) if wall else 0,
            }
            yield ('finish', finish_reason, usage)


# ────────────────────────────────────────────────────────────────────
# HTTP layer
# ────────────────────────────────────────────────────────────────────
def build_app(engine: 'Qwen3Engine'):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title='FlashRT Qwen3-8B NVFP4 OpenAI-compatible server')

    @app.get('/v1/models')
    async def list_models():
        return {
            'object': 'list',
            'data': [{
                'id': engine.model_name,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'flash-vla',
            }],
        }

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'model': engine.model_name}

    @app.post('/v1/chat/completions')
    async def chat_completions(req: Dict[str, Any]):
        messages = req.get('messages')
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, 'messages is required (non-empty list)')
        for m in messages:
            role = m.get('role')
            if role not in ('system', 'user', 'assistant', 'tool'):
                raise HTTPException(400, f'unsupported role: {role!r}')
        tools = req.get('tools')          # OAI tools spec
        max_tokens = int(req.get('max_tokens') or 256)
        stream = bool(req.get('stream', False))
        temperature = float(req.get('temperature', 0.0))
        top_p = float(req.get('top_p', 1.0))
        top_k = int(req.get('top_k', 0))
        seed = req.get('seed')
        stop = req.get('stop')
        if isinstance(stop, str):
            stop = [stop]
        elif stop is None:
            stop = []
        elif not isinstance(stop, list):
            raise HTTPException(400, 'stop must be string or list')

        completion_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
        created = int(time.time())

        if not stream:
            content = ''
            tool_calls: List[dict] = []
            finish = 'stop'
            usage: dict = {}
            async for ev in engine.stream_generate(
                messages, tools, max_tokens, temperature, top_p, top_k,
                seed, stop,
            ):
                if ev[0] == 'content':
                    content += ev[1]
                elif ev[0] == 'tool_calls':
                    tool_calls.extend(ev[1])
                elif ev[0] == 'finish':
                    _, finish, usage = ev
            msg: dict = {'role': 'assistant', 'content': content or None}
            if tool_calls:
                msg['tool_calls'] = tool_calls
            log.info(
                'non-stream done: %s -> %s tok in %ss (%s tok/s)',
                usage.get('prompt_tokens'), usage.get('completion_tokens'),
                usage.get('wall_s'), usage.get('tok_per_s'),
            )
            return {
                'id': completion_id,
                'object': 'chat.completion',
                'created': created,
                'model': engine.model_name,
                'choices': [{
                    'index': 0,
                    'message': msg,
                    'finish_reason': finish,
                }],
                'usage': usage,
            }

        # ── Streaming SSE ──
        async def gen():
            # Emit role first.
            first = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': engine.model_name,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': ''},
                    'finish_reason': None,
                }],
            }
            yield f'data: {json.dumps(first)}\n\n'

            tc_seen = False
            async for ev in engine.stream_generate(
                messages, tools, max_tokens, temperature, top_p, top_k,
                seed, stop,
            ):
                if ev[0] == 'content':
                    chunk = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': engine.model_name,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': ev[1]},
                            'finish_reason': None,
                        }],
                    }
                    yield f'data: {json.dumps(chunk)}\n\n'
                elif ev[0] == 'tool_calls':
                    tc_seen = True
                    for tc in ev[1]:
                        chunk = {
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': engine.model_name,
                            'choices': [{
                                'index': 0,
                                'delta': {'tool_calls': [tc]},
                                'finish_reason': None,
                            }],
                        }
                        yield f'data: {json.dumps(chunk)}\n\n'
                elif ev[0] == 'finish':
                    _, finish, usage = ev
                    last = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': engine.model_name,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': (
                                'tool_calls' if tc_seen
                                and finish in ('stop', 'length')
                                else finish
                            ),
                        }],
                        'usage': usage,
                    }
                    yield f'data: {json.dumps(last)}\n\n'
                    yield 'data: [DONE]\n\n'
                    log.info(
                        'stream done: %s -> %s tok in %ss (%s tok/s)',
                        usage.get('prompt_tokens'),
                        usage.get('completion_tokens'),
                        usage.get('wall_s'), usage.get('tok_per_s'),
                    )
                    return

        return StreamingResponse(gen(), media_type='text/event-stream')

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                    help='Path to NVFP4 ckpt dir.')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--max-seq', type=int, default=2048)
    p.add_argument('--max-q-seq', type=int, default=128,
                    help='Max prompt prefill length (in tokens).')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model-name', default='qwen3-8b-nvfp4')
    p.add_argument('--warmup', default='32:128,128:256',
                    help='Comma-separated "P:max_tok" shapes to warm.')
    args = p.parse_args()

    warm: List[Tuple[int, int]] = []
    for spec in args.warmup.split(','):
        spec = spec.strip()
        if not spec:
            continue
        try:
            pl, mt = spec.split(':')
            warm.append((int(pl), int(mt)))
        except ValueError:
            sys.exit(f'invalid --warmup spec: {spec!r}')

    try:
        import uvicorn
    except ImportError:
        sys.exit('uvicorn is required: pip install fastapi uvicorn')

    engine = Qwen3Engine(
        checkpoint=args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        max_seq=args.max_seq,
        max_q_seq=args.max_q_seq,
    )
    if warm:
        engine.warmup(warm)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
