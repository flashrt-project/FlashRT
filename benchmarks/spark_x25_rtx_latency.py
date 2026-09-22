"""Spark-X2.5-4B decode latency vs context length, RTX SM120.

Sweeps the one axis that moves for a full-attention decoder: how much KV the
step has to stream. TTFT is one whole-prompt prefill; decode is a captured
CUDA-Graph replay, wall-clock, divided by the steps it replayed -- the same
statistic an OpenAI server reports as TPOT.

The prompt is repeated text of the requested length. That is deliberate: this
measures rate, not quality, so it must not depend on a corpus, and a decoder's
cost is set by the number of keys, not by what they say.

    python benchmarks/spark_x25_rtx_latency.py --checkpoint /models/Spark-X2.5-4B \
        --lengths 128,512,2048,32768,65536,131072,262144 --steps 128

Each length constructs its own frontend, because ``max_seq`` sets both the KV
residency mode and the split count baked into the captured graph. That makes
one length ~40 s of load+prefill at 262k; the table prints as it goes.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from flash_rt.frontends.torch.spark_x25_rtx import SparkX25TorchFrontendRtx

FILLER = ("The quick brown fox jumps over the lazy dog. "
          "人工智能推理系统的性能取决于内存带宽与算子的实现质量。")


def parse_lengths(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def build_prompt(tokenizer, target: int) -> torch.Tensor:
    """Repeated text trimmed to about ``target`` tokens.

    The filler is tokenized once and its ids are tiled. Re-tokenizing the whole
    growing string every round is O(n^2) in prompt length -- at 1M tokens that
    is tens of minutes of tokenizer time before a single kernel runs, and it
    grows quadratically, so the 512k/1M points were dominated by it.
    """
    base = tokenizer(FILLER, return_tensors="pt")["input_ids"][0]
    if int(base.numel()) == 0:
        raise RuntimeError("filler tokenized to nothing")
    reps = (target + int(base.numel()) - 1) // int(base.numel())
    return base.repeat(reps)[:target].to(torch.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lengths", default="128,512,2048,32768,65536,131072")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prefill-chunk", type=int, default=None)
    ap.add_argument("--json", default=None, help="also write the table here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    lengths = parse_lengths(args.lengths)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    print(f"{'ctx':>8} {'tok':>8} {'kv mode':>9} {'split':>6} {'TTFT ms':>10} "
          f"{'decode ms':>10} {'tok/s':>8}")
    rows = []
    for n in lengths:
        ids = build_prompt(tok, n).cuda()
        S = int(ids.numel())
        try:
            fe = SparkX25TorchFrontendRtx(
                args.checkpoint, max_seq=S + args.steps + 8,
                prefill_cap=S + args.steps + 8, prefill_chunk=args.prefill_chunk)
        except torch.OutOfMemoryError as exc:
            print(f"{n:>8} {S:>8} {'-':>9} {'-':>6} {'-':>10} {'-':>10} "
                  f"OOM(construct): {str(exc)[:32]}")
            del ids
            torch.cuda.empty_cache()
            continue
        mode = "E4M3 only" if fe.kv8_only else "bf16+E4M3"
        try:
            with torch.no_grad():
                t0 = time.perf_counter()
                logits = fe.set_prompt(ids)
                torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t0) * 1000
                fe.runtime.next_token.fill_(int(logits[-1].argmax()))
                fe.runtime.decode_loop(S, args.steps)
                torch.cuda.synchronize()
                g = fe.runtime._loop_graph
                t0 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                per_tok = (time.perf_counter() - t0) / args.steps * 1000
        except torch.OutOfMemoryError as exc:
            print(f"{n:>8} {S:>8} {'-':>9} {'-':>6} {'-':>10} {'-':>10} "
                  f"OOM: {str(exc)[:32]}")
            del fe, ids
            torch.cuda.empty_cache()
            continue
        row = {"target": n, "tokens": S, "kv_mode": mode,
               "attn_splits": fe.attn_splits,
               "attn_splits_slide": fe.attn_splits_slide,
               "ttft_ms": ttft_ms, "decode_ms_per_tok": per_tok,
               "decode_tok_s": 1000.0 / per_tok}
        rows.append(row)
        print(f"{n:>8} {S:>8} {mode:>9} {fe.attn_splits:>6} {ttft_ms:>10.1f} "
              f"{per_tok:>10.3f} {1000.0/per_tok:>8.1f}", flush=True)
        del fe, ids, logits
        torch.cuda.empty_cache()

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
