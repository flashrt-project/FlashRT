#!/usr/bin/env python3
"""Reference driver: the same prefill and decode protocol as ``flint_benchmark.py`` on Recursive's rec
engine (MLX). Run it with the Python interpreter of a rec installation (``rec_mlx`` and ``mlx``
importable); it is used only for comparison and is not part of FlashRT.

Prefill: wall time of the engine's session prefill, including evaluation of the last position's logits.
Decode: ``generate_tokens`` with ``--n-gen`` output tokens and the stop tokens replaced by the reserved
id ``--never-token``, so every run produces exactly ``--n-gen`` tokens; the rate is the engine's own
``decode_tokens_per_second`` metric (prefill excluded). The first run per prompt pays the prefill and
is not counted; later runs reuse the engine's prompt cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

SAMPLING = {
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=0),
    "default": dict(temperature=1.0, top_p=0.95, top_k=20),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="rec's Flint model directory (contains flint-int4.json)")
    ap.add_argument("--tokens", nargs="+", required=True, help="raw little-endian int32 prompt files")
    ap.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    ap.add_argument("--prefill-reps", type=int, default=3, help="timed prefills per prompt after one warmup")
    ap.add_argument("--n-gen", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2, help="timed decode runs per prompt and sampling mode")
    ap.add_argument("--sampling", nargs="+", choices=tuple(SAMPLING), default=list(SAMPLING))
    ap.add_argument("--never-token", type=int, default=150143, help="stop-token replacement (reserved id)")
    ap.add_argument("--context-limit", type=int, default=262144)
    args = ap.parse_args()

    os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
    import mlx.core as mx
    from rec_mlx import quantized
    from rec_mlx.engine import Engine, Sampling

    original = quantized.QATModel.__init__

    def without_stop(self, *a, **k):
        original(self, *a, **k)
        self.eos_tokens = [args.never_token]
    quantized.QATModel.__init__ = without_stop
    t0 = time.perf_counter()
    eng = Engine(args.model_dir, args.model_dir / "flint-int4.json", context_limit=args.context_limit)
    eng.eos = {args.never_token}
    print(json.dumps({"event": "loaded", "seconds": round(time.perf_counter() - t0, 2)}), flush=True)

    for path in args.tokens:
        data = Path(path).read_bytes()
        ids = np.frombuffer(data, dtype="<i4").tolist()
        digest, n = hashlib.sha256(data).hexdigest(), len(ids)
        if args.phase != "decode":
            times = []
            for rep in range(args.prefill_reps + 1):   # repetition 0 is the warmup
                session = eng.runtime.make_session(max_new_tokens=8)
                mx.synchronize()
                t0 = time.perf_counter()
                logits = session.prefill(ids)
                mx.eval(logits)
                mx.synchronize()
                wall = time.perf_counter() - t0
                del session, logits
                print(json.dumps({"phase": "prefill", "tokens": n, "sha256": digest, "rep": rep,
                                  "wall_s": round(wall, 4)}), flush=True)
                if rep or not args.prefill_reps:
                    times.append(wall)
            print(json.dumps({"phase": "prefill", "tokens": n, "file": str(path),
                              "median_wall_s": round(statistics.median(times), 4)}), flush=True)
        if args.phase == "prefill":
            continue
        eng._prompt_cache = None
        eng._prefix_preparation = None
        for mode in args.sampling:
            rates = []
            for rep in range(args.reps + 1):   # run 0 pays the prefill and is not counted
                metrics = None
                sampling = Sampling(max_tokens=args.n_gen, seed=1000 + rep, **SAMPLING[mode])
                for event in eng.generate_tokens(ids, sampling):
                    if "metrics" in event:
                        metrics = event["metrics"]
                if metrics["generated_tokens"] != args.n_gen:
                    raise RuntimeError(f"generated {metrics['generated_tokens']} tokens, expected {args.n_gen}")
                rate = metrics["decode_tokens_per_second"]
                print(json.dumps({"phase": "decode", "context": n, "sampling": mode, "rep": rep,
                                  "tok_s": round(rate, 2), "counted": rep > 0}), flush=True)
                if rep:
                    rates.append(rate)
            print(json.dumps({"phase": "decode", "context": n, "sampling": mode, "file": str(path),
                              "median_tok_s": round(statistics.median(rates), 2)}), flush=True)


if __name__ == "__main__":
    main()
