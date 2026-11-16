#!/usr/bin/env python3
"""Reference timing of mlx-lm on the same checkpoint and token fixtures as ``qwen36_moe_benchmark.py``.

Needs ``mlx`` and ``mlx-lm`` (the reference engine; FlashRT itself does not use them). Prefill: wall time of
``generate_step`` up to its first token (prompt chunks of 2048 tokens, the library default). Decode:
``--n-gen`` tokens with no stop token; tokens/s = (n_gen - 1) / (time after the first token). ``greedy`` is
temperature 0; ``default`` is temperature 1, top-p 0.95, top-k 20. One warm-up run per prompt and mode, then
``--reps`` timed runs, each printed as one JSON line, followed by the median.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-dir", required=True, help="Qwen3.6-35B-A3B MLX affine int4 checkpoint directory")
    ap.add_argument("--tokens", nargs="+", required=True, help="raw little-endian int32 prompt files")
    ap.add_argument("--n-gen", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sampling", nargs="+", choices=("greedy", "default"), default=["greedy", "default"])
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, _ = load(args.mlx_dir)
    samplers = {"greedy": make_sampler(temp=0.0), "default": make_sampler(temp=1.0, top_p=0.95, top_k=20)}
    for path in args.tokens:
        data = Path(path).read_bytes()
        ids = np.frombuffer(data, dtype="<i4")
        n = len(ids)
        for mode in args.sampling:
            prefill, rates = [], []
            for rep in range(args.reps + 1):   # repetition 0 is the warm-up
                mx.clear_cache()
                mx.random.seed(1000 + rep)
                produced, first_at = [], None
                t0 = time.perf_counter()
                for token, _ in generate_step(mx.array(ids), model, max_tokens=args.n_gen, sampler=samplers[mode]):
                    produced.append(int(token))
                    if first_at is None:
                        first_at = time.perf_counter()
                end = time.perf_counter()
                row = {"engine": "mlx-lm", "context": n, "sampling": mode, "rep": rep,
                       "prefill_s": round(first_at - t0, 4), "tok_s": round((len(produced) - 1) / (end - first_at), 2)}
                if mode == "greedy":
                    row["sha256"] = hashlib.sha256(np.asarray(produced, dtype="<i4").tobytes()).hexdigest()
                print(json.dumps(row), flush=True)
                if rep:
                    prefill.append(first_at - t0)
                    rates.append((len(produced) - 1) / (end - first_at))
            print(json.dumps({"engine": "mlx-lm", "context": n, "sampling": mode, "file": str(path),
                              "median_prefill_s": round(statistics.median(prefill), 4),
                              "median_tok_s": round(statistics.median(rates), 2),
                              "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2)}), flush=True)


if __name__ == "__main__":
    main()
