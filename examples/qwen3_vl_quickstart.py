#!/usr/bin/env python3
"""Qwen3-VL FlashRT quickstart.

Runs the Qwen3-VL multimodal RTX path on a single image + text prompt and
prints the generated description. ``--arch sm120`` expects the self-contained
NVFP4 checkpoint produced by ``tools/quantize_qwen3_vl_nvfp4.py``; ``--arch
sm89`` expects the official Qwen3-VL FP8 checkpoint.

Examples:
    # One-shot description
    python examples/qwen3_vl_quickstart.py \\
        --checkpoint /path/to/Qwen3-VL-8B-FlashRT-NVFP4 \\
        --image FlashRT.png \\
        --prompt "Describe this image in one sentence."

    # Latency benchmark (prefill TTFT + decode tok/s)
    python examples/qwen3_vl_quickstart.py \\
        --checkpoint /path/to/Qwen3-VL-8B-FlashRT-NVFP4 \\
        --image FlashRT.png --benchmark 20
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_messages(image_path: str, prompt: str) -> list:
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    return [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt},
    ]}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True,
                   help='Qwen3-VL checkpoint directory for the selected arch')
    p.add_argument('--arch', default='sm120', choices=('sm120', 'sm89'),
                   help='RTX route: sm120 NVFP4 or sm89 official FP8')
    p.add_argument('--image', required=True, help='input image path')
    p.add_argument('--prompt', default='Describe this image in one sentence.')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=4096)
    p.add_argument('--max-pixels', type=int, default=None,
                   help='cap image resolution (pixels); the patch count '
                        'drives TTFT, so e.g. 1000000 roughly halves it. '
                        'Default: checkpoint full resolution.')
    p.add_argument('--max-prefill-seq', type=int, default=None,
                   help='SM89 official-FP8 prefill buffer length; default '
                        'uses min(max_seq, 128). Only valid with --arch sm89.')
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
    p.add_argument('--benchmark', type=int, default=0,
                   help='if >0, run that many timed iterations')
    args = p.parse_args()

    import torch

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
    messages = _build_messages(args.image, args.prompt)

    text = fe.generate(messages, max_new_tokens=args.max_new_tokens)
    print('--- generated ---')
    print(text)

    if args.benchmark > 0:
        # TTFT (prefill) timing.
        fe.set_prompt(messages)
        torch.cuda.synchronize()
        ttft = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            fe.prefill_graph()
            torch.cuda.synchronize()
            ttft.append((time.perf_counter() - t0) * 1e3)
        ttft.sort()

        # Decode throughput with warm CUDA Graphs.
        s = fe._prompt['S']
        n_dec = args.max_new_tokens
        fe.warmup_decode_graphs(n_dec)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for j in range(n_dec):
            if args.arch == 'sm89':
                fe.decode_step_with_graph(0, s + j)
            else:
                fe._decode_step_graph(
                    0, s + j, fe._prompt['mrope_max'] + 1 + j)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print('--- benchmark ---')
        print(f'prompt tokens (incl. vision) : {s}')
        print(f'TTFT prefill P50             : {ttft[len(ttft)//2]:.1f} ms')
        print(f'decode throughput (warm graph): {n_dec / dt:.1f} tok/s '
              f'({n_dec} tok in {dt*1e3:.0f} ms)')


if __name__ == '__main__':
    main()
