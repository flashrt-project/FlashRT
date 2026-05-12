#!/usr/bin/env python3
"""
FlashRT — Qwen3.6-27B NVFP4 OpenAI-compatible HTTP server.

Serves the /v1/chat/completions endpoint backed by the FlashRT NVFP4
inference path. Clients targeting the OpenAI API can swap their base
URL to this server without code changes.

Usage:
    pip install fastapi uvicorn

    # Required env: paired FP8 ckpt dir for the MTP head.
    export FLASHRT_QWEN36_MTP_CKPT_DIR=/path/to/qwen36_fp8_ckpt

    python examples/qwen36_openai_server.py \\
        --checkpoint /path/to/qwen36_nvfp4 \\
        --port 8000 \\
        --K 6 \\
        --warmup 8:512

    # The --warmup flag pre-captures CUDA Graphs for the listed
    # (prompt_len:max_tokens) shapes at startup so the FIRST real
    # request hits the warm 90-130 tok/s speed range. Without
    # warmup, that first request pays a ~5-25 s graph-capture
    # penalty (standard CUDA Graph cold-start cost — same trade-off
    # as SGLang / vLLM compile mode). Each warmed shape covers
    # cur_pos in [prompt_len, prompt_len + max_tokens]; the default
    # 8:512 covers the common short-chat range in one pass.
    # Pick shape(s) that span your traffic distribution.

    # Test (non-streaming):
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
              "model": "qwen3.6-27b-nvfp4",
              "messages": [{"role": "user", "content": "Hello!"}],
              "max_tokens": 128,
              "stream": false
            }'

    # OpenAI Python client:
    #   from openai import OpenAI
    #   client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
    #   resp = client.chat.completions.create(
    #       model="qwen3.6-27b-nvfp4",
    #       messages=[{"role": "user", "content": "Hi"}],
    #       max_tokens=128,
    #   )
    #
    # Function calling uses the Qwen chat-template native tool format.
    # Pass OpenAI-shaped "tools"; the server parses model-emitted
    # <tool_call>{...}</tool_call> blocks into OpenAI "tool_calls".

Limits in v1 (see docs/qwen36_usage.md):
    * Batch size 1 (concurrent requests are serialized; do not run
      multiple workers against one GPU).
    * Greedy decode only — temperature / top_p / top_k / n / seed
      / stop / logit_bias are accepted but ignored.
    * stream=True returns one chunk with the full response (true
      token-by-token streaming requires a frontend modification).
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
from typing import Any, Dict, List, Optional

# ────────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('qwen36_openai_server')


# Qwen tool-call format:
#   <tool_call>{"name": "fn_name", "arguments": {...}}</tool_call>
_TOOL_CALL_OPEN = '<tool_call>'
_TOOL_CALL_CLOSE = '</tool_call>'


class ToolCallParser:
    """Split Qwen tool-call blocks from a complete assistant response."""

    def __init__(self):
        self._tool_call_idx = 0

    def parse(self, text: str) -> tuple[str, List[dict]]:
        content_parts: List[str] = []
        tool_calls: List[dict] = []
        pos = 0
        while True:
            open_idx = text.find(_TOOL_CALL_OPEN, pos)
            if open_idx < 0:
                content_parts.append(text[pos:])
                break
            content_parts.append(text[pos:open_idx])
            raw_start = open_idx + len(_TOOL_CALL_OPEN)
            close_idx = text.find(_TOOL_CALL_CLOSE, raw_start)
            if close_idx < 0:
                content_parts.append(text[open_idx:])
                break
            tc = self._parse_tool_call(text[raw_start:close_idx].strip())
            if tc is not None:
                tool_calls.append(tc)
            pos = close_idx + len(_TOOL_CALL_CLOSE)
        return ''.join(content_parts), tool_calls

    def _parse_tool_call(self, raw: str) -> Optional[dict]:
        s = raw.strip()
        if s.startswith('```'):
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
# Frontend wrapper
# ────────────────────────────────────────────────────────────────────
class Qwen36Engine:
    """Thin wrapper around Qwen36TorchFrontendRtx with chat-template
    rendering and a single-request lock (batch=1 only)."""

    def __init__(self, checkpoint: str, *, K: int, max_seq: int,
                 device: str, model_name: str):
        import torch
        from flash_rt.frontends.torch.qwen36_rtx import (
            Qwen36TorchFrontendRtx,
        )

        log.info('loading NVFP4 ckpt from %s ...', checkpoint)
        t0 = time.perf_counter()
        self.fe = Qwen36TorchFrontendRtx(
            checkpoint, quant='nvfp4',
            device=device, max_seq=max_seq,
        )
        log.info('loaded in %.1f s', time.perf_counter() - t0)
        self.K = int(K)
        self.model_name = model_name
        self.lock = asyncio.Lock()
        self._torch = torch

        if self.fe._weights.ptrs.get('mtp') is None:
            log.warning(
                'MTP head not loaded (FLASHRT_QWEN36_MTP_CKPT_DIR '
                'unset?) — speculative decode disabled. The server '
                'will fall back to single-token decode (~36 tok/s).')
            self.spec_enabled = False
        else:
            self.spec_enabled = True
            log.info('MTP head loaded; spec K=%d enabled', self.K)

    def warmup(self, shapes: List[tuple]) -> None:
        """Pre-capture CUDA Graphs for typical (prompt_len, max_tokens)
        shapes by running dummy generations. Without this, the FIRST
        request at each new (prompt_len, max_tokens) shape pays a
        ~5-25 s graph-capture penalty (the headline 90-130 tok/s
        warm number applies only to requests AFTER the cur_pos range
        is captured). Standard practice for graph-based inference
        engines (SGLang, vLLM compile mode, etc.).

        Args:
          shapes: list of (prompt_len, max_tokens) tuples to pre-warm.
            Defaults to a single (64, 256) shape if empty.
        """
        if not shapes:
            return
        torch = self._torch
        log.info('warmup: pre-capturing graphs for %d shape(s) ...',
                 len(shapes))
        # Dummy text — only the token count matters for graph capture,
        # not the content. Pad with 'a's.
        for prompt_len, max_tok in shapes:
            t0 = time.perf_counter()
            dummy_text = 'a ' * (prompt_len - 1)  # ~1 token each
            input_ids = self.fe._tokenizer(
                dummy_text, return_tensors='pt').input_ids.cuda()
            # Trim/pad to exact prompt_len.
            if input_ids.shape[1] >= prompt_len:
                input_ids = input_ids[:, :prompt_len]
            else:
                pad = torch.full(
                    (1, prompt_len - input_ids.shape[1]),
                    self.fe._tokenizer.pad_token_id or 0,
                    device='cuda', dtype=torch.long)
                input_ids = torch.cat([input_ids, pad], dim=1)
            torch.cuda.synchronize()
            if self.spec_enabled:
                _ = self.fe.generate_own_speculative_KN_nvfp4(
                    input_ids, max_new_tokens=max_tok, K=self.K)
            else:
                _ = self._single_token_decode(input_ids, max_tok)
            torch.cuda.synchronize()
            log.info(
                '  warmup shape=(prompt=%d, max_tok=%d) in %.1f s',
                prompt_len, max_tok, time.perf_counter() - t0)
        log.info('warmup done — first real request will be at the warm '
                 '(~90-130 tok/s) speed range')

    def _render_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Apply Qwen's chat template to a list of messages."""
        normalized = []
        for m in messages:
            if m.get('content') is None:
                m = {**m, 'content': ''}
            normalized.append(m)
        return self.fe._tokenizer.apply_chat_template(
            normalized,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Run one chat-completion. Returns a dict with the new
        text and basic timing/stat fields."""
        torch = self._torch
        async with self.lock:
            prompt = self._render_chat(messages, tools)
            input_ids = self.fe._tokenizer(
                prompt, return_tensors='pt').input_ids.cuda()
            prompt_len = int(input_ids.shape[1])

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if self.spec_enabled:
                out = self.fe.generate_own_speculative_KN_nvfp4(
                    input_ids, max_new_tokens=max_tokens, K=self.K,
                )
            else:
                out = self._single_token_decode(input_ids, max_tokens)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0

            new_tokens = out[0, prompt_len:].tolist()
            raw_text = self.fe._tokenizer.decode(
                new_tokens, skip_special_tokens=False)
            for special in (
                self.fe._tokenizer.eos_token,
                self.fe._tokenizer.pad_token,
            ):
                if special:
                    raw_text = raw_text.replace(special, '')
            text, tool_calls = ToolCallParser().parse(raw_text)
            text = text.strip()

            return {
                'text': text,
                'tool_calls': tool_calls,
                'prompt_tokens': prompt_len,
                'completion_tokens': len(new_tokens),
                'wall_s': wall_s,
                'tok_per_s': len(new_tokens) / wall_s if wall_s else 0,
            }

    def _single_token_decode(self, input_ids, max_tokens):
        """Fallback when MTP is not loaded. Slower path (~36 tok/s)."""
        torch = self._torch
        fe = self.fe
        fe.reset_state()
        if not hasattr(fe, '_rope_cos_table'):
            fe._build_rope_table()

        prompt_len = int(input_ids.shape[1])
        generated = list(input_ids[0].tolist())
        cur_pos = 0
        with torch.no_grad():
            for p in range(prompt_len):
                fe._static_token_id.copy_(input_ids[:, p:p + 1])
                cos, sin = fe._rope_cos_sin(cur_pos)
                fe.forward_own_decode_nvfp4(
                    fe._static_token_id, cos, sin, cur_pos)
                cur_pos += 1
            for _ in range(max_tokens):
                tok = fe._logits_buf.argmax(
                    dim=-1, keepdim=True).view(1, 1)
                generated.append(int(tok.item()))
                fe._static_token_id.copy_(tok)
                cos, sin = fe._rope_cos_sin(cur_pos)
                fe.forward_own_decode_nvfp4(
                    fe._static_token_id, cos, sin, cur_pos)
                cur_pos += 1
        return torch.tensor([generated], device='cuda')


# ────────────────────────────────────────────────────────────────────
# OpenAI-compatible HTTP layer (FastAPI)
# ────────────────────────────────────────────────────────────────────
def build_app(engine: Qwen36Engine):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title='FlashRT Qwen3.6 NVFP4 OpenAI-compatible server')

    @app.get('/v1/models')
    async def list_models():
        return {
            'object': 'list',
            'data': [{
                'id': engine.model_name,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'flash-rt',
            }],
        }

    @app.post('/v1/chat/completions')
    async def chat_completions(req: Dict[str, Any]):
        messages = req.get('messages')
        if not messages or not isinstance(messages, list):
            raise HTTPException(400, 'messages is required')
        tools = req.get('tools')
        max_tokens = int(req.get('max_tokens') or 256)
        stream = bool(req.get('stream', False))
        # Validate roles — Qwen template accepts OpenAI chat roles.
        for m in messages:
            role = m.get('role')
            if role not in ('system', 'user', 'assistant', 'tool'):
                raise HTTPException(
                    400, f'unsupported role: {role!r}')
            content = m.get('content')
            if content is None and role == 'assistant':
                continue
            if not isinstance(content, str):
                raise HTTPException(
                    400, 'message.content must be a string')

        result = await engine.generate(messages, tools, max_tokens)
        completion_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
        created = int(time.time())

        log.info(
            'chat.completions: %d -> %d tokens in %.2fs (%.1f tok/s)',
            result['prompt_tokens'],
            result['completion_tokens'],
            result['wall_s'],
            result['tok_per_s'],
        )

        usage = {
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': result['completion_tokens'],
            'total_tokens': (result['prompt_tokens']
                             + result['completion_tokens']),
        }

        if stream:
            # We don't have token-by-token streaming yet (v1 limit);
            # emit the full message in one delta then [DONE]. Clients
            # that target streaming will see one big chunk.
            async def gen():
                first = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': engine.model_name,
                    'choices': [{
                        'index': 0,
                        'delta': {
                            'role': 'assistant',
                        },
                        'finish_reason': None,
                    }],
                }
                if result['text']:
                    first['choices'][0]['delta']['content'] = result['text']
                yield f'data: {json.dumps(first)}\n\n'
                for tc in result['tool_calls']:
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
                last = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': engine.model_name,
                    'choices': [{
                        'index': 0,
                        'delta': {},
                        'finish_reason': (
                            'tool_calls' if result['tool_calls']
                            else 'stop'
                        ),
                    }],
                    'usage': usage,
                }
                yield f'data: {json.dumps(last)}\n\n'
                yield 'data: [DONE]\n\n'

            return StreamingResponse(gen(), media_type='text/event-stream')

        content = (
            result['text'] if result['text'] or not result['tool_calls']
            else None
        )
        message = {
            'role': 'assistant',
            'content': content,
        }
        if result['tool_calls']:
            message['tool_calls'] = result['tool_calls']

        return {
            'id': completion_id,
            'object': 'chat.completion',
            'created': created,
            'model': engine.model_name,
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': (
                    'tool_calls' if result['tool_calls'] else 'stop'
                ),
            }],
            'usage': usage,
        }

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'model': engine.model_name,
                'spec_enabled': engine.spec_enabled,
                'K': engine.K}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to NVFP4 main ckpt (compressed-tensors).')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--K', type=int, default=6,
                   help='MTP draft chain length per spec cycle. '
                   'Default 6 (peak for short generations on RTX 5090).')
    p.add_argument('--max-seq', type=int, default=2048,
                   help='KV cache + scratch dim. Increase for long ctx.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model-name', default='qwen3.6-27b-nvfp4',
                   help='Identifier returned by /v1/models and echoed '
                   'in completion responses.')
    p.add_argument(
        '--warmup', default='8:512',
        help='Comma-separated list of "prompt_len:max_tokens" shapes '
        'to pre-capture CUDA Graphs for at startup. Without this, the '
        'first request at each new shape pays a ~5-25 s graph-capture '
        'penalty before reaching the headline 90-130 tok/s warm speed. '
        'Default 8:512 pre-captures cur_pos in [8, 520], covering the '
        'common short-chat range so requests with prompt_len < 32 '
        '(which the prior 32:128,128:256 default missed) start warm. '
        'Set to empty string to skip warmup.')
    args = p.parse_args()

    warmup_shapes = []
    if args.warmup.strip():
        for spec in args.warmup.split(','):
            spec = spec.strip()
            if not spec:
                continue
            try:
                pl, mt = spec.split(':')
                warmup_shapes.append((int(pl), int(mt)))
            except ValueError:
                sys.exit(f'invalid --warmup spec: {spec!r} '
                         '(expected "prompt_len:max_tokens")')

    if 'FLASHRT_QWEN36_MTP_CKPT_DIR' not in os.environ:
        log.warning(
            'FLASHRT_QWEN36_MTP_CKPT_DIR is not set — speculative '
            'decode will be disabled and tok/s will fall to ~36. See '
            'docs/qwen36_usage.md for the FP8 ckpt requirement.')

    try:
        import uvicorn
    except ImportError:
        sys.exit('uvicorn is required: pip install uvicorn fastapi')

    engine = Qwen36Engine(
        checkpoint=args.checkpoint,
        K=args.K,
        max_seq=args.max_seq,
        device=args.device,
        model_name=args.model_name,
    )
    if warmup_shapes:
        engine.warmup(warmup_shapes)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level='warning')


if __name__ == '__main__':
    main()
