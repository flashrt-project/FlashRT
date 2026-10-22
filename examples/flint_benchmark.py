#!/usr/bin/env python3
"""Batch-one Flint forward benchmark with separate prefill/decode wall and event timings.

Supply the same raw little-endian int32 token files to each revision. These
files are user-provided fixtures, not downloaded by this command. Capture is
warmed before timing; token upload/reset is included in the prefill interval.
Decode includes the LM head and greedy pick, without HTTP or text rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from flash_rt.models.flint.pipeline_rtx import FlintRuntime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--tokens', nargs='+', help='raw little-endian int32 prompt files')
    ap.add_argument('--make-tokens', help='create reproducible token fixtures in this directory, then exit')
    ap.add_argument('--corpus', help='UTF-8 text for fixture creation; see the pinned public corpus command in Flint usage')
    ap.add_argument('--n-gen', type=int, default=256)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--reps', type=int, default=2)
    args = ap.parse_args()
    if args.make_tokens:
        if not args.corpus:
            ap.error('--make-tokens requires --corpus')
        from flash_rt.models.flint.gguf import GGUFFile
        from flash_rt.models.flint.tokenizer import FlintTokenizer
        text = 'Summarize this project\'s build and runtime contracts.\n\n' + Path(args.corpus).read_text(encoding='utf-8')
        with GGUFFile(args.gguf) as gguf:
            ids = np.asarray(FlintTokenizer(gguf).encode(text), dtype='<i4')
        if not len(ids):
            ap.error('corpus must tokenize to a non-empty sequence')
        output = Path(args.make_tokens)
        output.mkdir(parents=True, exist_ok=True)
        lengths = (58, 1024, 4096, 8192, 16384, 32768, 65536, 102400)
        repeated = np.tile(ids, (max(lengths) + len(ids) - 1) // len(ids))
        for n in lengths:
            repeated[:n].tofile(output / f'prompt-{n}.i32')
        return
    if not args.tokens:
        ap.error('--tokens is required for measurement')
    if args.n_gen < 1 or args.reps < 1 or args.warmup < 1:
        ap.error('n-gen, reps and warmup must be positive')
    prompts = [np.fromfile(path, dtype='<i4').tolist() for path in args.tokens]
    if any(not p for p in prompts):
        ap.error('token files must contain non-empty prompts')
    rt = FlintRuntime(args.gguf, max_seq=max(map(len, prompts)) + args.n_gen + 64)
    rt.capture()
    rows = []
    for ids in prompts:
        if any(t < 0 or t >= rt.cfg.vocab_size for t in ids):
            ap.error('prompt token outside checkpoint vocabulary')
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        samples, fingerprints = [], []
        for rep in range(args.warmup + args.reps):
            rt.stream.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(rt.stream):
                events[0].record(rt.stream)
                rt.prefill(ids)
                events[1].record(rt.stream)
            rt.stream.synchronize()
            t1 = time.perf_counter()
            with torch.cuda.stream(rt.stream):
                rt.replay(args.n_gen)
                events[2].record(rt.stream)
            rt.stream.synchronize()
            t2 = time.perf_counter()
            if rep >= args.warmup:
                samples.append({'prefill_ms': events[0].elapsed_time(events[1]),
                                'decode_ms_per_token': events[1].elapsed_time(events[2]) / args.n_gen,
                                'prefill_wall_ms': (t1 - t0) * 1000,
                                'decode_wall_ms_per_token': (t2 - t1) * 1000 / args.n_gen})
                tokens = rt.tokens[len(ids):len(ids) + min(16, args.n_gen)].cpu().numpy().astype('<i4').tobytes()
                fingerprints.append(hashlib.sha256(tokens).hexdigest())
        rows.append({'prompt_tokens': len(ids), 'prompt_sha256': hashlib.sha256(np.array(ids, dtype='<i4').tobytes()).hexdigest(),
                     'samples': samples, 'first_16_tokens_sha256': fingerprints,
                     'median_prefill_ms': statistics.median(s['prefill_ms'] for s in samples),
                     'median_prefill_wall_ms': statistics.median(s['prefill_wall_ms'] for s in samples),
                     'median_decode_wall_ms_per_token': statistics.median(s['decode_wall_ms_per_token'] for s in samples),
                     'median_decode_ms_per_token': statistics.median(s['decode_ms_per_token'] for s in samples)})
    import flash_rt.flash_rt_flint as kernels
    print(json.dumps({'gpu': torch.cuda.get_device_name(rt.dev), 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'module_sha256': hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
                      'batch': 1, 'n_gen': args.n_gen, 'warmup': args.warmup, 'reps': args.reps, 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
