#!/usr/bin/env python3
"""Minimal OpenAI-compatible vision chat server for Qwen3-VL on FlashRT.

Exposes ``POST /v1/chat/completions`` (non-streaming) for image+text
chat, backed by the FlashRT Qwen3-VL RTX path. ``--arch sm120`` expects the
FlashRT NVFP4 checkpoint; ``--arch sm89`` expects the official FP8
checkpoint. Multimodal messages use the OpenAI ``image_url`` content format
(``http(s)://`` or ``data:`` base64 URLs). Single-GPU, batch 1; concurrent
requests are serialised.

    pip install fastapi uvicorn
    python examples/qwen3_vl_openai_server.py \\
        --checkpoint /path/to/Qwen3-VL-8B-FlashRT-NVFP4 --port 8000

    curl http://localhost:8000/v1/chat/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model":"qwen3-vl","messages":[{"role":"user","content":[
              {"type":"image_url","image_url":{"url":"https://.../x.png"}},
              {"type":"text","text":"What is in this image?"}]}]}'
"""
from __future__ import annotations

import argparse
import base64
import io
import pathlib
import sys
import time
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_image(url: str):
    from PIL import Image
    if url.startswith('data:'):
        raw = base64.b64decode(url.split(',', 1)[1])
    elif url.startswith(('http://', 'https://')):
        with urllib.request.urlopen(url) as r:
            raw = r.read()
    else:
        with open(url, 'rb') as f:
            raw = f.read()
    return Image.open(io.BytesIO(raw)).convert('RGB')


def _to_frontend_messages(messages: list) -> list:
    """Translate OpenAI chat messages (with ``image_url`` parts) into the
    processor's content format (``{'type': 'image', 'image': PIL}``)."""
    out = []
    for m in messages:
        content = m.get('content')
        if isinstance(content, str):
            out.append({'role': m['role'], 'content': content})
            continue
        parts = []
        for part in content or []:
            if part.get('type') == 'text':
                parts.append({'type': 'text', 'text': part['text']})
            elif part.get('type') == 'image_url':
                parts.append({'type': 'image',
                              'image': _load_image(part['image_url']['url'])})
        out.append({'role': m['role'], 'content': parts})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--arch', default='sm120', choices=('sm120', 'sm89'),
                   help='RTX route: sm120 NVFP4 or sm89 official FP8')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--max-pixels', type=int, default=None,
                   help='cap image/video resolution (pixels) to bound TTFT; '
                        'default keeps full resolution')
    p.add_argument('--max-prefill-seq', type=int, default=None,
                   help='SM89 official-FP8 prefill buffer length; default '
                        'uses max_seq to match the SM120 VL path. Only valid '
                        'with --arch sm89.')
    p.add_argument('--fuse-gate-up', action='store_true',
                   help='use SM89 official-FP8 fused gate/up weight when '
                        'available. Only valid with --arch sm89.')
    p.add_argument('--fp8-lm-head', action='store_true',
                   help='use the SM89 official-FP8 explicit experimental '
                        'FP8 lm_head mode. Only valid with --arch sm89.')
    p.add_argument('--vision-bf16-first-blocks', type=int, default=3,
                   help='SM89 vision path: keep the first N ViT blocks in '
                        'BF16 before switching linears to FP8. '
                        'Only valid with --arch sm89.')
    p.add_argument('--model-name', default='qwen3-vl')
    args = p.parse_args()

    import asyncio

    import uvicorn
    from fastapi import FastAPI, HTTPException

    if args.arch == 'sm89':
        from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
            Qwen3VlFp8Sm89Frontend,
        )

        fe = Qwen3VlFp8Sm89Frontend(
            args.checkpoint, device=args.device, max_seq=args.max_seq,
            max_pixels=args.max_pixels, max_prefill_seq=args.max_prefill_seq,
            fuse_gate_up=args.fuse_gate_up,
            use_fp8_lm_head=args.fp8_lm_head,
            vision_bf16_first_blocks=args.vision_bf16_first_blocks)
    else:
        if args.max_prefill_seq is not None:
            p.error('--max-prefill-seq is only valid with --arch sm89')
        if args.fuse_gate_up:
            p.error('--fuse-gate-up is only valid with --arch sm89')
        if args.fp8_lm_head:
            p.error('--fp8-lm-head is only valid with --arch sm89')
        if args.vision_bf16_first_blocks != 3:
            p.error('--vision-bf16-first-blocks is only valid with --arch sm89')
        from flash_rt.frontends.torch.qwen3_vl_rtx import (
            Qwen3VlTorchFrontendRtx,
        )

        fe = Qwen3VlTorchFrontendRtx(
            args.checkpoint, device=args.device, max_seq=args.max_seq,
            max_pixels=args.max_pixels)
    lock = asyncio.Lock()
    app = FastAPI(title='FlashRT Qwen3-VL OpenAI-compatible server')

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/v1/models')
    def models():
        return {'object': 'list',
                'data': [{'id': args.model_name, 'object': 'model'}]}

    @app.post('/v1/chat/completions')
    async def chat(body: dict):
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "'messages' must be a non-empty list")
        max_new = int(body.get('max_tokens') or args.max_new_tokens)
        try:
            fe_messages = _to_frontend_messages(messages)
        except Exception as e:  # noqa: BLE001 - surface a clean 400
            raise HTTPException(400, f'bad message content: {e}')
        try:
            async with lock:
                text = await asyncio.to_thread(
                    fe.generate, fe_messages, max_new_tokens=max_new)
        except ValueError as e:  # e.g. text-only prompt to the VL frontend
            raise HTTPException(400, str(e))
        return {
            'id': f'chatcmpl-{int(time.time()*1000)}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': args.model_name,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': text},
                'finish_reason': 'stop',
            }],
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
