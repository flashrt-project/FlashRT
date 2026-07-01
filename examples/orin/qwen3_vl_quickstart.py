#!/usr/bin/env python3
"""Qwen3-VL quickstart for Jetson Orin.

Example:
  python examples/orin/qwen3_vl_quickstart.py \
    --checkpoint /root/models/Qwen3-VL-2B-Instruct \
    --image FlashRT.png \
    --prompt "Describe this image in one sentence." \
    --max-new-tokens 32
"""
from __future__ import annotations

import argparse
import statistics
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/root/models/Qwen3-VL-2B-Instruct')
    p.add_argument('--image', required=True)
    p.add_argument('--prompt', default='Describe this image in one sentence.')
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-seq', type=int, default=2048)
    p.add_argument('--max-pixels', type=int, default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', default='bf16',
                   help='runtime precision path; currently implemented: bf16')
    p.add_argument('--reps', type=int, default=1,
                   help=('repeat generation in one process; useful for '
                         'graph warm replay timing'))
    p.add_argument('--benchmark', type=int, default=0,
                   help=('if >0, separately time graph prefill and warm '
                         'graph decode'))
    p.add_argument('--no-graph', action='store_true',
                   help='disable CUDA Graph replay for prefill/decode')
    return p.parse_args()


def main() -> None:
    from PIL import Image
    import torch

    from flash_rt.frontends.torch.qwen3_vl_rtx_bf16 import (
        Qwen3VlTorchFrontendRtxBF16,
    )

    args = parse_args()
    if args.precision != 'bf16':
        raise ValueError(
            f'unsupported Qwen3-VL Orin precision {args.precision!r}; '
            'currently implemented: bf16')
    with Image.open(args.image) as im:
        image = im.convert('RGB')
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': args.prompt},
        ],
    }]

    model = Qwen3VlTorchFrontendRtxBF16(
        args.checkpoint, device=args.device, max_seq=args.max_seq,
        max_pixels=args.max_pixels)
    text = ''
    lat_ms = []
    for _ in range(max(1, args.reps)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = model.generate(
            messages, max_new_tokens=args.max_new_tokens,
            use_graph=not args.no_graph)
        torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    print(text)
    print('\nlatency_ms: ' + ', '.join(f'{v:.1f}' for v in lat_ms))
    if len(lat_ms) > 1:
        print(f'warm_latency_ms: {lat_ms[-1]:.1f}')

    if args.benchmark > 0:
        if args.no_graph:
            raise ValueError('--benchmark requires CUDA Graph replay')
        model.set_prompt(messages)
        p = model._prompt
        assert p is not None
        torch.cuda.synchronize()

        # Capture once, then measure steady graph replay.
        model.prefill_graph()
        torch.cuda.synchronize()
        ttft = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            model.prefill_graph()
            torch.cuda.synchronize()
            ttft.append((time.perf_counter() - t0) * 1000.0)
        prefill_p50 = statistics.median(ttft)

        n_dec = max(1, args.max_new_tokens)
        model.warmup_decode_graphs(n_dec)
        torch.cuda.synchronize()
        base_slot = int(p['S'])
        base_rope = int(p['mrope_max']) + 1
        t0 = time.perf_counter()
        for j in range(n_dec):
            model.decode_step_with_graph(
                0, cache_pos=base_slot + j, rope_pos=base_rope + j)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print('\n--- benchmark ---')
        print(f'vision patches               : {int(p["patches"])}')
        print(f'prompt tokens (incl. vision) : {int(p["S"])}')
        print(f'prefill graph P50            : {prefill_p50:.1f} ms')
        print(f'decode throughput (warm graph): {n_dec / dt:.1f} tok/s '
              f'({n_dec} tok in {dt * 1000.0:.1f} ms)')


if __name__ == '__main__':
    main()
