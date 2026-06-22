#!/usr/bin/env python3
"""Qwen3-VL FlashRT quickstart.

Runs the Qwen3-VL multimodal path (NVFP4 language stack + BF16 ViT tower)
on a single image + text prompt and prints the generated description. The
checkpoint is the self-contained directory produced by
``tools/quantize_qwen3_vl_nvfp4.py``.

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
                   help='FlashRT Qwen3-VL NVFP4 checkpoint directory')
    p.add_argument('--image', required=True, help='input image path')
    p.add_argument('--prompt', default='Describe this image in one sentence.')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seq', type=int, default=4096)
    p.add_argument('--max-pixels', type=int, default=None,
                   help='cap image resolution (pixels); the patch count '
                        'drives TTFT, so e.g. 1000000 roughly halves it. '
                        'Default: checkpoint full resolution.')
    p.add_argument('--benchmark', type=int, default=0,
                   help='if >0, run that many timed iterations')
    args = p.parse_args()

    import torch

    from flash_rt.frontends.torch.qwen3_vl_rtx import Qwen3VlTorchFrontendRtx

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
            fe.prefill()
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
            fe._decode_step_graph(0, s + j, fe._prompt['mrope_max'] + 1 + j)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print('--- benchmark ---')
        print(f'prompt tokens (incl. vision) : {s}')
        print(f'TTFT prefill P50             : {ttft[len(ttft)//2]:.1f} ms')
        print(f'decode throughput (warm graph): {n_dec / dt:.1f} tok/s '
              f'({n_dec} tok in {dt*1e3:.0f} ms)')


if __name__ == '__main__':
    main()
