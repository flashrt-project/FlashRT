#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini on RTX SM120: one sequence, greedy, plain decode and speculative decoding interleaved.

The drafter is the checkpoint's DSpark, or with ``--dflash2 <directory>`` the DFlash 2 drafter
(incoai/Inkling-2-Mini-DFlash2; adaptive widths, ``--fixed-width``: ``--width`` rows every step). ``--fp8``: the
runtime's E4M3 copies of the attention / output projections and the LM head (both modes).

Single-prompt runs (``--reps`` each, median): prefill (prompt -> first token; speculative runs also write the
drafter context) and decode tokens/s = (new tokens - 1) / (time from the first token to the last), without stop
tokens; the speculative tokens are checked against plain greedy decoding. Prompts:

* ``--prompt review`` (default): a review task rendered with the checkpoint's chat template
  (``reasoning_effort="none"``), its first 64 and last ``--isl`` - 64 token ids;
* ``--prompt code``: a request to explain Python source, the source being this interpreter's standard library
  (files in sorted order), cut the same way to ``--isl`` tokens: long contexts of real text;
* ``--prompt-ids``: a JSON list of token ids.

``--suite code`` / ``--suite general`` instead runs 16 built-in requests once per mode, each up to ``--new``
tokens or the checkpoint's end-of-generation token, and reports the aggregate decode rate and the drafter's
accepted tokens per step (they depend on the text).

Runs are separated by ``--cooldown`` seconds; each timed request follows ``--warm`` seconds of decode steps, so
that it starts with the GPU at its clocks rather than idle.

    python examples/inkling2_rtx_benchmark.py --checkpoint <path-to-Inkling-2-Mini> --new 512 --reps 3
    python examples/inkling2_rtx_benchmark.py --checkpoint <path> --prompt code --isl 32768 --max-seq 40960
    python examples/inkling2_rtx_benchmark.py --checkpoint <path> --suite code --cooldown 5
    python examples/inkling2_rtx_benchmark.py --checkpoint <path> --dflash2 <path-to-Inkling-2-Mini-DFlash2> [--fp8]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sysconfig
import time
from pathlib import Path

import torch

BODY = ("You are reviewing a service backed by a bounded worker queue. The service receives "
        "events with a user key and sequence number, applies updates in timestamp order, and "
        "retries transient failures. Discuss backpressure, idempotency, fairness, failure "
        "handling, metrics, and the tests that cover race conditions and shutdown. ")
TASK = ("Write a detailed implementation plan in English, including Python pseudocode and "
        "edge cases. Provide at least 16 numbered points with concrete examples.")

SUITES = {
    "code": [
        "Write a Python function that returns the longest palindromic substring of a string, with tests.",
        "Implement an LRU cache class in Python with get and put in O(1), and explain the design.",
        "Write a C function that reverses a singly linked list in place.",
        "Write a Python generator that yields the prime numbers in increasing order.",
        "Implement binary search over a sorted list in Python, returning the leftmost match.",
        "Write a JavaScript function that debounces another function.",
        "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat.",
        "Implement Dijkstra's shortest-path algorithm in Python with a heap.",
        "Write a SQL query that returns the three highest-paid employees of every department.",
        "Write a Rust function that counts the words of a string and returns a HashMap.",
        "Implement a thread-safe bounded queue in Python with a condition variable.",
        "Write a Python function that merges overlapping intervals.",
        "Write a C++ class template for a fixed-size ring buffer.",
        "Write a bash script that finds the ten largest files under a directory.",
        "Implement quicksort in Go.",
        "Write a Python decorator that retries a function with exponential backoff.",
    ],
    "general": [
        "Explain how a heat pump can move more heat than the electricity it uses.",
        "Summarize the causes of the French Revolution in five paragraphs.",
        "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Explain.",
        "Write a polite email declining a meeting invitation and proposing two other times.",
        "Translate into French: The library opens at nine, but the reading room only at ten.",
        "Compare the advantages of renting and buying a home.",
        "Describe the water cycle for a ten-year-old.",
        "Give a weekly training plan for a beginner preparing for a 10 km run.",
        "What is the difference between weather and climate?",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain public-key cryptography with an everyday analogy.",
        "List the main stages of cell division and what happens in each.",
        "How should a small team run a weekly planning meeting?",
        "Explain inflation and how central banks try to control it.",
        "Write a haiku sequence about autumn in a city.",
        "What are good practices for keeping houseplants healthy in winter?",
    ],
}


def cut(ids: list, n: int) -> list:
    if len(ids) < n:
        raise RuntimeError(f"prompt has {len(ids)} tokens, expected at least {n}")
    return ids[:64] + ids[len(ids) - (n - 64):]


def review_prompt(fe, n: int) -> list:
    body = BODY * max(9, n // 40)
    text = fe.apply_chat_template([{"role": "user", "content": body + TASK}], reasoning_effort="none")
    return cut(fe.tokenizer.encode(text), n)


def code_prompt(fe, n: int) -> list:
    src, size = [], 0
    for f in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")):
        src.append(f.read_text(errors="replace"))
        size += len(src[-1])
        if size > 6 * n:            # well over n tokens
            break
    text = fe.apply_chat_template([{"role": "user", "content": "Explain what the following Python source does, "
                                    "function by function.\n\n```python\n" + "\n".join(src) + "\n```"}],
                                  reasoning_effort="none")
    return cut(fe.tokenizer.encode(text), n)


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
    ap.add_argument("--prompt", choices=("review", "code"), default="review")
    ap.add_argument("--isl", type=int, default=512, help="prompt tokens (--prompt review / code)")
    ap.add_argument("--prompt-ids", help="JSON list of prompt token ids instead")
    ap.add_argument("--suite", choices=sorted(SUITES), help="run the built-in requests instead of one prompt")
    ap.add_argument("--new", type=int, default=512, help="generated tokens per request (at most, with --suite)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds before each timed request")
    ap.add_argument("--warm", type=float, default=0.2, help="seconds of decode steps before each timed request")
    ap.add_argument("--width", type=int, default=4, help="verified rows per step (DSpark, DFlash 2 --fixed-width)")
    ap.add_argument("--dflash2", help="DFlash 2 drafter directory (instead of DSpark)")
    ap.add_argument("--fixed-width", action="store_true", help="DFlash 2: --width rows every step")
    ap.add_argument("--fp8", action="store_true", help="E4M3 attention / output projections and LM head")
    ap.add_argument("--prefill", choices=("fast", "exact"), default="fast")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--out", help="write the runs and the summary to this JSON file")
    args = ap.parse_args()

    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    t0 = time.perf_counter()
    fe = Inkling2TorchFrontendRtx(args.checkpoint, max_seq=args.max_seq, speculative=True, draft_width=args.width,
                                  prefill=args.prefill, fp8=args.fp8, dflash2=args.dflash2,
                                  adaptive=not args.fixed_width)
    rt, dr, sync = fe.rt, fe.drafter, torch.cuda.synchronize
    spec = "dflash2" if args.dflash2 else "dspark"
    if args.suite:
        prompts = [fe.tokenizer.encode(fe.apply_chat_template([{"role": "user", "content": q}],
                                                              reasoning_effort="none")) for q in SUITES[args.suite]]
        stop = fe.eog
    elif args.prompt_ids:
        prompts, stop = [json.loads(Path(args.prompt_ids).read_text())], ()
    else:
        prompts, stop = [(review_prompt if args.prompt == "review" else code_prompt)(fe, args.isl)], ()
    longest = max(len(p) for p in prompts)
    warm_steps = int(args.warm * 1000) + 8          # far more than the warm-up's steps take (5.6 ms each)
    need = longest + args.new + max(fe._margin(True), fe._margin(False))
    if need > args.max_seq or warm_steps > args.max_seq:
        raise SystemExit(f"--max-seq {args.max_seq} is too small (prompt + new tokens + margin = {need})")
    exact = args.prefill == "exact"
    for ids in {len(p): p for p in prompts}.values():
        fe.warmup(len(ids))
    print(f"load + capture {time.perf_counter() - t0:.1f} s, {len(prompts)} prompt(s) of up to {longest} tokens, "
          f"GPU {gpu_temperature()} C", flush=True)

    def warm():
        rt.reset()          # decode steps from position 0 (each request resets again)
        sync()
        t, done = time.perf_counter(), 0
        while time.perf_counter() - t < args.warm and done + 8 <= warm_steps:
            for _ in range(8):
                rt.run("decode")
            done += 8
            sync()

    def ended(block):
        hit = next((i for i, t in enumerate(block) if t in stop), None)
        return (block, False) if hit is None else (block[:hit + 1], True)

    def plain(ids):
        warm()
        rt.reset()
        sync()
        a = time.perf_counter()
        out = [rt.prefill(ids, exact=exact)]
        b = time.perf_counter()
        position, done = len(ids), out[0] in stop
        while not done and len(out) < args.new:
            steps = min(fe.check_block, args.new - len(out))
            for _ in range(steps):
                rt.run("decode")
            block, done = ended(rt.history[position + 1: position + 1 + steps].tolist())
            out += block
            position += steps
        sync()
        c = time.perf_counter()
        return dict(prefill_s=b - a, decode_s=c - b, tokens=len(out)), out

    def speculative(ids):
        warm()
        rt.reset()
        sync()
        a = time.perf_counter()
        first = dr.prefill(ids, exact=exact)
        sync()
        b = time.perf_counter()
        out = []
        for block in dr.steps(first, args.new, check_every=fe.check_rounds):
            block, done = ended(block)
            out += block
            if done:
                break
        sync()
        c = time.perf_counter()
        return dict(prefill_s=b - a, decode_s=c - b, tokens=len(out), steps=dr.rounds,
                    accepted=int(dr.log_count.item()) - 1), out

    runs = {"plain": [], spec: []}
    if args.suite:
        for i, ids in enumerate(prompts):
            time.sleep(args.cooldown)
            rp, ref = plain(ids)
            time.sleep(args.cooldown)
            rd, out = speculative(ids)
            rd["identical_to_greedy"] = out == ref
            for mode, r in (("plain", rp), (spec, rd)):
                r["prompt_tokens"] = len(ids)
                runs[mode].append(r)
            print(i, len(ids), json.dumps({"plain_tok_s": round((rp["tokens"] - 1) / rp["decode_s"], 1),
                                           f"{spec}_tok_s": round((rd["tokens"] - 1) / rd["decode_s"], 1),
                                           "tokens": rp["tokens"],
                                           "tokens_per_step": round(rd["accepted"] / rd["steps"], 2),
                                           "identical": rd["identical_to_greedy"]}), flush=True)
        summary = {m: {"decode_tok_s": sum(r["tokens"] - 1 for r in rs) / sum(r["decode_s"] for r in rs),
                       "prefill_ms_median": 1e3 * statistics.median(r["prefill_s"] for r in rs),
                       "tokens": sum(r["tokens"] for r in rs)} for m, rs in runs.items()}
        rs = runs[spec]
        summary[spec]["tokens_per_step"] = sum(r["accepted"] for r in rs) / sum(r["steps"] for r in rs)
        summary[spec]["tokens_per_step_range"] = [round(min(r["accepted"] / r["steps"] for r in rs), 2),
                                                  round(max(r["accepted"] / r["steps"] for r in rs), 2)]
        summary[spec]["identical_to_greedy"] = all(r["identical_to_greedy"] for r in rs)
    else:
        ids = prompts[0]
        _, ref = plain(ids)
        for rep in range(args.reps):
            for mode, fn in (("plain", plain), (spec, speculative)):
                time.sleep(args.cooldown)
                r, out = fn(ids)
                r["tok_s"] = (r["tokens"] - 1) / r["decode_s"]
                if mode == spec:
                    r["identical_to_greedy"] = out == ref
                    r["tokens_per_step"] = r["accepted"] / r["steps"]
                r["gpu_c"] = gpu_temperature()
                runs[mode].append(r)
                print(mode, rep, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}),
                      flush=True)
        summary = {m: {"decode_tok_s_median": statistics.median(r["tok_s"] for r in rs),
                       "prefill_ms_median": 1e3 * statistics.median(r["prefill_s"] for r in rs)}
                   for m, rs in runs.items()}
        summary[spec]["tokens_per_step_median"] = statistics.median(r["tokens_per_step"] for r in runs[spec])
        summary[spec]["identical_to_greedy"] = all(r["identical_to_greedy"] for r in runs[spec])
        summary["prompt_tokens"] = len(ids)
    summary["torch_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(dict(summary=summary, runs=runs), indent=1) + "\n")


if __name__ == "__main__":
    main()
