#!/usr/bin/env python3
"""Inkling 2 Mini decode throughput at a given concurrency on Apple Silicon (thinkingmachines/Inkling-2-Mini).

The shape of the model card's speculative-decoding benchmark: C concurrent sequences, each a ``--prompt``-token
prompt followed by exactly ``--gen`` generated tokens (no stop token), C in ``--conc``. Prompts are consecutive
slices of a token corpus (a raw little-endian int32 file, ``--make-corpus``), offset per repetition, so the
reference driver (``inkling2_mlx_benchmark.py``) consumes the same ids.

Per run: prefill seconds (every prompt to its first token, sequence by sequence: one row program per chunk) and
decode seconds / tokens per second after every sequence has its first token, aggregate and per sequence. Plain
decoding: the resident decode step (C = 1, blocks of eight tokens) or the batched row program (one row per
sequence per step). ``--mtp``: DSpark drafting with greedy verification (verify widths by C, see
``Inkling2Runtime.mtp_widths``); ``--verify`` picks the verification of one sequence (rows: the row programs, the
default; fast: the resident step with its own summation order, adaptive rows and lookup drafts; exact: the resident
step with the decode step's arithmetic). ``--q8``: attention projections and LM head from Q8_0 copies (not the
checkpoint's numerics; C = 1, with --mtp only --verify fast). Run 0 is a warm-up;
the median of the others is printed per C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import sysconfig
import time

import numpy as np


def make_corpus(args):
    from flash_rt.models.flint.tokenizer import FlintTokenizer   # tokenizer.json reader
    tok = FlintTokenizer.from_directory(args.model)
    ids = []
    for path in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")):
        ids.extend(tok.encode(path.read_text(errors="ignore")))
        if len(ids) >= args.corpus_tokens:
            break
    ids = np.asarray(ids[:args.corpus_tokens], dtype="<i4")
    ids.tofile(args.make_corpus)
    print(json.dumps({"event": "corpus", "file": args.make_corpus, "tokens": int(ids.size),
                      "sha256": hashlib.sha256(ids.tobytes()).hexdigest()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Inkling-2-Mini checkpoint directory")
    ap.add_argument("--corpus", help="raw little-endian int32 token file")
    ap.add_argument("--make-corpus", metavar="FILE", help="tokenize the standard-library sources into FILE, then exit")
    ap.add_argument("--corpus-tokens", type=int, default=1_000_000)
    ap.add_argument("--conc", default="1,4,16,64")
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mtp", action="store_true", help="DSpark speculative decoding")
    ap.add_argument("--verify", choices=("rows", "fast", "exact"), default="rows",
                    help="verification of one sequence with --mtp (C = 1)")
    ap.add_argument("--fixed-rows", action="store_true",
                    help="fast verification of one sequence: always 8 rows (default: 2-16 chosen per step)")
    ap.add_argument("--no-lookup", action="store_true",
                    help="fast verification of one sequence: no lookup drafts (at most the drafter's 8 rows)")
    ap.add_argument("--dflash2", default=None, help="DFlash 2 drafter directory (with --mtp, C = 1)")
    ap.add_argument("--q8", action="store_true", help="Q8_0 attention projections and LM head")
    args = ap.parse_args()
    if args.q8 and args.mtp and args.verify != "fast":
        ap.error("--q8 with --mtp needs --verify fast (the row programs and the exact step use the BF16 weights)")
    if args.dflash2 and not args.mtp:
        ap.error("--dflash2 is a drafter: it needs --mtp")
    if args.make_corpus:
        return make_corpus(args)
    if not args.corpus:
        ap.error("--corpus is required for measurement")
    from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime
    corpus = np.fromfile(args.corpus, dtype="<i4")
    concs = [int(c) for c in args.conc.split(",")]
    if args.q8 and concs != [1]:
        ap.error("--q8 serves one sequence (--conc 1): the batched row programs use the BF16 weights")
    P, G = args.prompt, args.gen
    needed = 7919 * args.reps + max(concs) * P
    if corpus.size < needed:
        sys.exit(f"corpus has {corpus.size} tokens, the runs need {needed}")
    chunks = [c for c in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if c <= P]
    rt = Inkling2Runtime(args.model, max_seq=P + G + 16, sequences=max(concs), row_capacity=max(chunks[0], max(concs)),
                         row_shapes=[(1, 1)] if args.mtp else [(c, 1) for c in chunks] + [(c, c) for c in concs if c > 1],
                         mtp_batches=concs if args.mtp else (), mtp_prefill=chunks if args.mtp else (), q8=args.q8,
                         dflash2=args.dflash2 if args.mtp else None)
    rt.mtp_verify = {"fast": "fast", "exact": "resident", "rows": "rows"}[args.verify]
    rt.mtp_adaptive = not args.fixed_rows
    rt.mtp_lookup = not args.no_lookup
    results = {}
    for C in concs:
        rows = []
        for rep in range(args.reps + 1):
            off = 7919 * rep
            prompts = [corpus[off + i * P: off + (i + 1) * P].astype(np.int64).tolist() for i in range(C)]
            if args.mtp:
                stats = {}
                t0 = time.perf_counter()
                rt.mtp_generate(prompts, G, stats)
                decode = stats["decode_s"]
                prefill = time.perf_counter() - t0 - decode
                extra = {"tokens_per_step": round(stats["tokens_per_step"], 3),
                         "rows_per_step": round(stats["rows_per_step"], 2),
                         "lookup_steps": stats["lookup_steps"], "lookup_only_steps": stats["lookup_only_steps"],
                         "step_ms": round(1e3 * decode / stats["steps"], 2),
                         "verify_ms": round(stats["verify_ms_per_step"], 2), "draft_ms": round(stats["draft_ms_per_step"], 2)}
            else:
                t0 = time.perf_counter()
                first = [rt.prefill(p, seq=s) for s, p in enumerate(prompts)]
                t1 = time.perf_counter()
                if C == 1:
                    tok, pos, n = first[0], P, 1
                    while n < G:
                        out = rt.decode(tok, pos, min(8, G - n), program="greedy")
                        tok, pos, n = out[-1], pos + len(out), n + len(out)
                else:
                    toks = np.array(first, np.uint32)
                    for step in range(G - 1):
                        info = np.array([[s, P + step, P + step, P + step] for s in range(C)], np.uint32)
                        toks = rt.run_rows(toks.tolist(), info, C)
                decode, prefill, extra = time.perf_counter() - t1, t1 - t0, {}
            n_tok = (G - 1) * C
            row = dict(prefill_s=prefill, decode_s=decode, agg_tps=n_tok / decode, per_seq_tps=n_tok / decode / C)
            verify = (args.verify if C == 1 else "rows") if args.mtp else None   # several sequences: row programs
            adaptive = verify == "fast" and not args.fixed_rows
            lookup = adaptive and C == 1 and not args.no_lookup
            print(json.dumps({"conc": C, "rep": rep, "mtp": args.mtp, "verify": verify, "adaptive": adaptive,
                              "lookup": lookup,
                              "q8": args.q8, **{k: round(v, 3) for k, v in row.items()},
                              **extra}), flush=True)
            if rep:
                rows.append(row)
        results[C] = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        print(json.dumps({"conc": C, "median": {k: round(v, 3) for k, v in results[C].items()}}), flush=True)


if __name__ == "__main__":
    main()
