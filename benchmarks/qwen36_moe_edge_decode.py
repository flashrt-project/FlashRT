#!/usr/bin/env python3
"""First-light latency probe for the Qwen3.6-MoE (qwen3_5_moe) edge path.

Reports the numbers quoted in ``docs/qwen36_moe_usage.md``: weight load time,
resident and peak allocation, prefill latency, and decode throughput on the
eager and the captured-graph paths. The two decode paths are compared token for
token, because a throughput number for a path that emits different text is not
a throughput number for the same work.

Usage:

    PYTHONPATH=. python benchmarks/qwen36_moe_edge_decode.py \\
        --checkpoint /path/to/Qwen3.6-35B-A3B \\
        --prompt-tokens 64 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

GIB = 2 ** 30


def _sync(device: str) -> None:
    torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="path to the BF16 checkpoint directory")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-reps", type=int, default=5)
    parser.add_argument("--decode-reps", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from flash_rt.frontends.torch.qwen36_moe import Qwen36MoeTextFrontend

    # Select and initialise the device before touching the memory stats: they
    # are per-device counters and are not addressable until then.
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(args.device)
    t0 = time.perf_counter()
    frontend = Qwen36MoeTextFrontend(
        args.checkpoint, device=args.device, max_seq=args.max_seq)
    _sync(args.device)
    load_s = time.perf_counter() - t0

    print(f"runtime weight load                  {load_s:8.2f} s")
    print(f"resident allocated after load        "
          f"{torch.cuda.memory_allocated(args.device) / GIB:8.2f} GiB")
    print(f"peak allocated during load           "
          f"{torch.cuda.max_memory_allocated(args.device) / GIB:8.2f} GiB")

    base = frontend.tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ")
    ids = (base * (args.prompt_tokens // len(base) + 2))[:args.prompt_tokens]

    # Prefill: the first call carries warmup and lazy weight packing, so it is
    # reported separately rather than averaged into the steady-state figure.
    frontend.set_prompt_ids(ids)
    _sync(args.device)
    t0 = time.perf_counter()
    frontend.generate(max_new_tokens=1)
    _sync(args.device)
    first_ms = (time.perf_counter() - t0) * 1e3

    warm = []
    for _ in range(args.prefill_reps):
        frontend.set_prompt_ids(ids)
        _sync(args.device)
        t0 = time.perf_counter()
        frontend.generate(max_new_tokens=1)
        _sync(args.device)
        warm.append((time.perf_counter() - t0) * 1e3)

    print(f"first prefill, including warmup      {first_ms:8.2f} ms")
    print(f"subsequent prefill                   "
          f"{min(warm):8.2f}-{max(warm):.2f} ms")

    def run(fn) -> tuple[list[float], list[int]]:
        # Median and range over every repetition, not a best-of: a single best
        # sample hides both contention and variance, and the baseline this is
        # compared against reports the same shape.
        rates, toks = [], None
        for _ in range(args.decode_reps):
            frontend.set_prompt_ids(ids)
            _sync(args.device)
            t0 = time.perf_counter()
            out = fn()
            _sync(args.device)
            rates.append(args.max_new_tokens / (time.perf_counter() - t0))
            toks = list(out)
        return sorted(rates), toks

    def report(label: str, rates: list[float]) -> None:
        med = statistics.median(rates)
        print(f"{label:<44}{med:8.2f} tok/s   "
              f"(range {rates[0]:.2f}-{rates[-1]:.2f} over {len(rates)} runs)")

    state = frontend._decode_state
    from flash_rt.frontends.torch import _nexn2_rtx_decode as dec

    def eager():
        t = torch.tensor(ids, dtype=torch.long, device=args.device)
        with torch.no_grad():
            return dec.generate_greedy(
                state, t, args.max_new_tokens, frontend._fvk, args.device)

    eager_rate, eager_toks = run(eager)
    graph_rate, graph_toks = run(
        lambda: frontend.generate(max_new_tokens=args.max_new_tokens))

    report(f"{args.prompt_tokens}/{args.max_new_tokens} eager decode",
           eager_rate)
    report(f"{args.prompt_tokens}/{args.max_new_tokens} warm graph decode",
           graph_rate)
    free, total = torch.cuda.mem_get_info(args.device)
    print(f"{'device free memory at exit':<44}{free / GIB:8.2f} GiB "
          f"of {total / GIB:.2f} -- a shared device invalidates the timings")
    same = eager_toks == graph_toks
    print(f"eager and graph emit the same tokens {str(same):>8}"
          f"   ({sum(a == b for a, b in zip(eager_toks, graph_toks))}"
          f"/{len(graph_toks)})")
    if not same:
        raise SystemExit("eager and captured decode disagree; "
                         "the throughput numbers are not comparable")


if __name__ == "__main__":
    main()
