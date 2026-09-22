#!/usr/bin/env python3
"""
FlashRT — Spark-X2.5-4B quickstart.

Usage:
    python examples/spark_x25_quickstart.py \
        --checkpoint /models/Spark-X2.5-4B \
        --prompt "用一句话解释什么是量子纠缠。"

    # longer context, and a longer generation
    python examples/spark_x25_quickstart.py \
        --checkpoint /models/Spark-X2.5-4B \
        --max-seq 131072 --max-new-tokens 256

    # time it instead of reading it
    python examples/spark_x25_quickstart.py \
        --checkpoint /models/Spark-X2.5-4B --benchmark 128

The decode loop is captured into one CUDA Graph and replayed, so the number
`--benchmark` prints is steady-state TPOT (time per output token), not a cold
first step. See docs/spark_x25_usage.md and docs/spark_x25_rtx.md.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Spark-X2.5-4B checkpoint directory")
    ap.add_argument("--prompt", default="用一句话解释什么是量子纠缠。")
    ap.add_argument("--max-seq", type=int, default=8192,
                    help="KV capacity; also chooses the KV residency mode")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                    help="skip the answer and report decode tok/s over N steps")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    from flash_rt.frontends.torch.spark_x25_rtx import SparkX25TorchFrontendRtx

    fe = SparkX25TorchFrontendRtx(args.checkpoint, max_seq=args.max_seq,
                                  device=args.device)
    mode = "E4M3 only" if fe.kv8_only else "bf16 + E4M3 mirror"
    print(f"Spark-X2.5-4B: {fe.config.num_hidden_layers} layers, "
          f"max_seq {args.max_seq}, KV {mode}")

    if args.benchmark:
        ids = fe.tokenizer(args.prompt, return_tensors="pt")["input_ids"][0]
        with torch.no_grad():
            logits = fe.set_prompt(ids)
            fe.runtime.next_token.fill_(int(logits[-1].argmax()))
            fe.runtime.decode_loop(int(ids.numel()), args.benchmark)
            torch.cuda.synchronize()
            g = fe.runtime._loop_graph
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            per_tok = (time.perf_counter() - t0) / args.benchmark * 1000
        print(f"prompt {int(ids.numel())} tok -> decode {per_tok:.3f} ms/tok "
              f"= {1000.0/per_tok:.1f} tok/s")
        return 0

    t0 = time.perf_counter()
    text = fe.generate_text(args.prompt, max_new_tokens=args.max_new_tokens)
    print(f"\n{text}\n")
    print(f"({args.max_new_tokens} tokens in {time.perf_counter()-t0:.2f} s wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
