#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini on RTX SM120: one sequence, greedy, plain decode and DSpark speculative decoding interleaved.

The default prompt is the 512-token prompt of the reference benchmark this port is compared against (KERN's
Inkling 2 Mini runtime, ``pegainfer-project/kern-inkling-2mini``): a review task rendered with the checkpoint's
chat template (``reasoning_effort="none"``), its first 64 and last 448 token ids. ``--prompt-ids`` takes a JSON list
of token ids instead.

Reported per mode (median over ``--reps``): prefill (prompt -> first token; with DSpark it also writes the drafter
context) and decode tokens/s = (new tokens - 1) / (time from the first token to the last). DSpark's tokens are
checked against plain greedy decoding. Runs are separated by ``--cooldown`` seconds; each timed request follows
``--warm`` seconds of decode steps, so that it starts with the GPU at its clocks rather than idle.

    python examples/inkling2_rtx_benchmark.py --checkpoint <path-to-Inkling-2-Mini> --new 512 --reps 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

BODY = ("You are reviewing a service backed by a bounded worker queue. The service receives "
        "events with a user key and sequence number, applies updates in timestamp order, and "
        "retries transient failures. Discuss backpressure, idempotency, fairness, failure "
        "handling, metrics, and the tests that cover race conditions and shutdown. ")
TASK = ("Write a detailed implementation plan in English, including Python pseudocode and "
        "edge cases. Provide at least 16 numbered points with concrete examples.")


def reference_prompt(fe) -> list:
    text = fe.apply_chat_template([{"role": "user", "content": BODY * 9 + TASK}], reasoning_effort="none")
    ids = fe.tokenizer.encode(text)
    if len(ids) < 512:
        raise RuntimeError(f"reference prompt has {len(ids)} tokens, expected at least 512")
    return ids[:64] + ids[-448:]


def gpu_temperature():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device()), 0)
    except Exception:  # pragma: no cover - optional
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="Inkling-2-Mini checkpoint directory")
    ap.add_argument("--prompt-ids", help="JSON list of prompt token ids (default: the reference 512-token prompt)")
    ap.add_argument("--new", type=int, default=512, help="generated tokens per request")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds between runs")
    ap.add_argument("--warm", type=float, default=0.2, help="seconds of decode steps before each timed request")
    ap.add_argument("--width", type=int, default=4, help="DSpark verified rows per step")
    ap.add_argument("--prefill", choices=("fast", "exact"), default="fast")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--out", help="write the runs and the summary to this JSON file")
    args = ap.parse_args()

    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    t0 = time.perf_counter()
    fe = Inkling2TorchFrontendRtx(args.checkpoint, max_seq=args.max_seq, speculative=True, draft_width=args.width,
                                  prefill=args.prefill)
    ids = json.loads(Path(args.prompt_ids).read_text()) if args.prompt_ids else reference_prompt(fe)
    fe.warmup(len(ids))
    print(f"load + capture {time.perf_counter() - t0:.1f} s, prompt {len(ids)} tokens, "
          f"GPU {gpu_temperature()} C", flush=True)

    def warm():
        rt, sync = fe.rt, torch.cuda.synchronize
        sync()
        t = time.perf_counter()
        while time.perf_counter() - t < args.warm:
            for _ in range(8):
                rt.run("decode")
            sync()

    def plain():
        sync = torch.cuda.synchronize
        warm()
        sync()
        a = time.perf_counter()
        it = fe.stream(ids, max_new_tokens=args.new, stop_ids=(), speculative=False)
        out = list(next(it))
        sync()
        b = time.perf_counter()
        for block in it:
            out.extend(block)
        sync()
        c = time.perf_counter()
        return dict(prefill_s=b - a, decode_s=c - b, tok_s=(len(out) - 1) / (c - b)), out

    def dspark():
        dr, rt = fe.drafter, fe.rt
        sync = torch.cuda.synchronize
        warm()
        rt.reset()
        sync()
        a = time.perf_counter()
        first = dr.prefill(ids, exact=args.prefill == "exact")
        sync()
        b = time.perf_counter()
        out = []
        for block in dr.steps(first, args.new, check_every=fe.check_rounds):
            out.extend(block)
        sync()
        c = time.perf_counter()
        steps = dr.rounds
        return dict(prefill_s=b - a, decode_s=c - b, tok_s=(len(out) - 1) / (c - b), steps=steps,
                    tokens_per_step=(int(dr.log_count.item()) - 1) / steps), out

    _, ref = plain()
    runs = {"plain": [], "dspark": []}
    for rep in range(args.reps):
        for mode, fn in (("plain", plain), ("dspark", dspark)):
            time.sleep(args.cooldown)
            r, out = fn()
            if mode == "dspark":
                r["identical_to_greedy"] = out == ref
            r["gpu_c"] = gpu_temperature()
            runs[mode].append(r)
            print(mode, rep, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}),
                  flush=True)
    summary = {m: {"decode_tok_s_median": statistics.median(r["tok_s"] for r in rs),
                   "prefill_ms_median": 1e3 * statistics.median(r["prefill_s"] for r in rs)} for m, rs in runs.items()}
    summary["dspark"]["tokens_per_step_median"] = statistics.median(r["tokens_per_step"] for r in runs["dspark"])
    summary["dspark"]["identical_to_greedy"] = all(r["identical_to_greedy"] for r in runs["dspark"])
    summary["torch_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(dict(summary=summary, runs=runs), indent=1) + "\n")


if __name__ == "__main__":
    main()
