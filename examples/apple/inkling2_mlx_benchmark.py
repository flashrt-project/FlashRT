#!/usr/bin/env python3
"""Reference driver for ``inkling2_benchmark.py``: the same runs on mlx-lm's BatchGenerator (the continuous-batching
engine behind ``mlx_lm.server``), with an mlx-lm that supports ``inkling2_loop``.

C concurrent sequences of ``--prompt`` corpus tokens (the same slices as ``inkling2_benchmark.py``), exactly ``--gen``
new tokens each, greedy. Per run: prefill seconds (until every sequence has its first token) and decode seconds /
tokens per second after that, aggregate and per sequence. Run 0 is a warm-up; the median of the others is printed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="raw little-endian int32 token file")
    ap.add_argument("--conc", default="1,4,16,64")
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    model, _ = load(args.model)
    corpus = np.fromfile(args.corpus, dtype="<i4")
    sampler = make_sampler(temp=0.0)
    P, G = args.prompt, args.gen
    for C in [int(c) for c in args.conc.split(",")]:
        rows = []
        for rep in range(args.reps + 1):
            off = 7919 * rep
            prompts = [corpus[off + i * P: off + (i + 1) * P].tolist() for i in range(C)]
            gen = BatchGenerator(model, stop_tokens=None, sampler=sampler, completion_batch_size=max(C, 32),
                                 prefill_batch_size=min(C, 8))
            t0 = time.perf_counter()
            gen.insert(prompts, [G] * C)
            first, count, t_all, c_all = {}, 0, None, 0
            while responses := gen.next_generated():
                now = time.perf_counter()
                for r in responses:
                    first.setdefault(r.uid, now)
                    count += 1
                if t_all is None and len(first) == C:
                    t_all, c_all = now, count
            t1 = time.perf_counter()
            gen.close()
            decode, n_tok = t1 - t_all, count - c_all
            row = dict(prefill_s=t_all - t0, decode_s=decode, agg_tps=n_tok / decode, per_seq_tps=n_tok / decode / C)
            print(json.dumps({"conc": C, "rep": rep, **{k: round(v, 3) for k, v in row.items()}}), flush=True)
            if rep:
                rows.append(row)
            del gen
            mx.clear_cache()
        median = {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
        print(json.dumps({"conc": C, "median": median}), flush=True)


if __name__ == "__main__":
    main()
