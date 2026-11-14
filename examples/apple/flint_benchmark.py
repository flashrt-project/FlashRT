#!/usr/bin/env python3
"""Batch-one Flint prefill and decode wall-clock benchmark on Apple Silicon (MLX affine int4 checkpoint).

Prefill: wall time from the token upload to the first generated token (every chunk, the final norm and
LM head of the last position, token selection). Decode: ``--n-gen`` tokens after the prompt with no stop
token; tokens/s = (n_gen - 1) / (time after the first token). ``greedy`` is the on-device argmax,
``default`` is temperature 1, top-p 0.95, top-k 20. Every repetition is printed as one JSON line,
followed by the median per prompt (and sampling mode).

Token fixtures are raw little-endian int32 files (``--make-tokens``), so the reference driver
(``flint_rec_benchmark.py``) consumes exactly the same ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
SAMPLING = {
    "greedy": dict(program="greedy"),
    "default": dict(program="sampled", top_k=20, top_p=0.95, temperature=1.0),
}


def stdlib_sources():
    """The running interpreter's standard-library sources: ``*.py``, then ``*/*.py``, each sorted."""
    import os
    root = Path(os.__file__).parent
    return sorted(root.glob("*.py"), key=str) + sorted(root.glob("*/*.py"), key=str)


def make_tokens(args):
    from flash_rt.models.flint.tokenizer import FlintTokenizer
    tok = FlintTokenizer.from_directory(args.mlx_dir)
    files = [Path(p) for p in args.corpus] if args.corpus else stdlib_sources()
    total = max(args.lengths)
    ids = []
    for path in files:          # each file tokenized on its own, ids concatenated
        ids.extend(tok.encode(path.read_text(encoding="utf-8", errors="ignore")))
        if len(ids) >= total:
            break
    if len(ids) < total:
        sys.exit(f"corpus has {len(ids)} tokens, fewer than the longest prompt ({total})")
    out = Path(args.make_tokens)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ids[:total], dtype="<i4")
    for n in args.lengths:
        ids[:n].tofile(out / f"prompt-{n}.i32")
    print(json.dumps({"event": "tokens", "dir": str(out), "files": files.index(path) + 1,
                      "lengths": list(args.lengths), "sha256": hashlib.sha256(ids.tobytes()).hexdigest()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-dir", required=True, help="Flint MLX affine int4 checkpoint directory")
    ap.add_argument("--tokens", nargs="+", help="raw little-endian int32 prompt files")
    ap.add_argument("--make-tokens", metavar="DIR", help="write prompt-<n>.i32 fixtures to DIR, then exit")
    ap.add_argument("--corpus", nargs="+", help="UTF-8 text files for --make-tokens "
                    "(default: this interpreter's standard-library sources)")
    ap.add_argument("--lengths", type=lambda s: tuple(int(x) for x in s.split(",")), default=LENGTHS,
                    help="fixture lengths for --make-tokens (comma separated)")
    ap.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    ap.add_argument("--prefill-reps", type=int, default=3, help="timed prefills per prompt after one warmup")
    ap.add_argument("--n-gen", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2, help="timed decode runs per prompt and sampling mode")
    ap.add_argument("--sampling", nargs="+", choices=tuple(SAMPLING), default=list(SAMPLING))
    ap.add_argument("--max-seq", type=int, help="KV capacity (default: longest prompt + n-gen + 64)")
    args = ap.parse_args()
    if args.make_tokens:
        return make_tokens(args)
    if not args.tokens:
        ap.error("--tokens is required for measurement")
    if args.n_gen < 2 or args.reps < 1 or args.prefill_reps < 0:
        ap.error("n-gen must be at least 2, reps positive and prefill-reps non-negative")
    prompts = []
    for path in args.tokens:
        data = Path(path).read_bytes()
        ids = np.frombuffer(data, dtype="<i4").tolist()
        if not ids:
            ap.error(f"{path} is empty")
        prompts.append((path, ids, hashlib.sha256(data).hexdigest()))

    from flash_rt.frontends.torch._flint_mps_runtime import FlintRuntime
    longest = max(len(ids) for _, ids, _ in prompts)
    rt = FlintRuntime(args.mlx_dir, max_seq=args.max_seq or longest + args.n_gen + 64)
    print(json.dumps({"event": "loaded", "seconds": round(rt.load_seconds, 2), "max_seq": rt.max_seq}), flush=True)
    for path, ids, digest in prompts:
        n = len(ids)
        times = []
        for rep in range(args.prefill_reps + 1):   # repetition 0 is the warmup
            t0 = time.perf_counter()
            first = rt.prefill(ids)
            wall = time.perf_counter() - t0
            gpu = sum(t["gpu_us"] for t in rt.last_prefill_timings) / 1e6
            print(json.dumps({"phase": "prefill", "tokens": n, "sha256": digest, "rep": rep,
                              "wall_s": round(wall, 4), "gpu_s": round(gpu, 4)}), flush=True)
            if rep or not args.prefill_reps:
                times.append(wall)
            if args.phase == "decode":
                break
        if args.phase != "decode":
            print(json.dumps({"phase": "prefill", "tokens": n, "file": str(path),
                              "median_wall_s": round(statistics.median(times), 4)}), flush=True)
        if args.phase == "prefill":
            continue
        state = rt.sliding_state()   # every decode run starts from this prompt state
        for mode in args.sampling:
            rates = []
            for rep in range(args.reps):
                cfg = dict(SAMPLING[mode])
                if cfg["program"] == "sampled":
                    cfg["seed"] = 1000 + rep
                rt.restore_sliding_state(state)
                token, pos, made = first, n, 1   # every run continues the prompt from its greedy first token
                t0 = time.perf_counter()
                while made < args.n_gen:
                    block = rt.decode(token, pos, min(8, args.n_gen - made), **cfg)
                    made += len(block)
                    pos += len(block)
                    token = block[-1]
                rate = (args.n_gen - 1) / (time.perf_counter() - t0)
                rates.append(rate)
                print(json.dumps({"phase": "decode", "context": n, "sampling": mode, "rep": rep,
                                  "tok_s": round(rate, 2)}), flush=True)
            print(json.dumps({"phase": "decode", "context": n, "sampling": mode, "file": str(path),
                              "median_tok_s": round(statistics.median(rates), 2)}), flush=True)


if __name__ == "__main__":
    main()
