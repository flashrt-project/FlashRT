#!/usr/bin/env python3
"""Validate a startup expert set on prompts it was not derived from.

Filling each layer's cache at startup with its most frequently selected
experts looks strongly positive when the frequencies come from the same trace
being scored -- but that is an oracle, not a predictor. A deployment builds the
set offline, from other traffic, and then meets an unseen prompt.

This runs several unrelated prompts through one model load and reports
leave-one-out results: for each prompt, the startup set is built from the
*other* prompts' traces only. The gap between that and the oracle is what the
heuristic actually costs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from qwen36_moe_edge.route_trace import (
    DEFAULT_BLOCK_BYTES,
    cold_prefill_blocks,
    global_frequency,
    simulate_warm_lru,
)


def collect_traces(
        checkpoint: str,
        prompts: list[str],
        *,
        prompt_tokens: int,
        new_tokens: int,
        max_seq: int,
        device: str) -> list[list[list[list[int]]]]:
    """Trace every prompt's router selections in one model load."""
    from flash_rt.frontends.torch._nexn2_rtx_decode import (
        Nexn2DecodeState,
        generate_greedy,
    )
    from flash_rt.frontends.torch.qwen36_moe import (
        Qwen36MoeTextFrontend,
    )

    frontend = Qwen36MoeTextFrontend(
        checkpoint, device=device, max_seq=max_seq, quant_scope="experts")
    state = Nexn2DecodeState(frontend._weights, max_seq, device)
    state.batched_prefill = False

    traces = []
    for index, prompt in enumerate(prompts):
        input_ids = frontend.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).input_ids[:, :prompt_tokens].to(device)
        if input_ids.shape[1] != prompt_tokens:
            raise ValueError(
                f"prompt {index} is shorter than {prompt_tokens} tokens")
        state.router_trace = {
            layer: [] for layer in range(state.num_layers)}
        with torch.no_grad():
            generate_greedy(
                state, input_ids, new_tokens, frontend._fvk, device)
        traces.append([
            [list(experts) for experts in state.router_trace[layer]]
            for layer in range(state.num_layers)
        ])
        print(f"traced prompt {index + 1}/{len(prompts)}", flush=True)
    return traces


def _merge(frequencies: list[list[Counter[int]]]) -> list[Counter[int]]:
    """Sum per-layer selection counts across several traces."""
    layers = len(frequencies[0])
    merged = [Counter() for _ in range(layers)]
    for frequency in frequencies:
        for layer in range(layers):
            merged[layer].update(frequency[layer])
    return merged


def leave_one_out(
        traces: list[list[list[list[int]]]],
        *,
        prompt_tokens: int,
        quota: int,
        block_bytes: int) -> list[dict[str, float]]:
    """Score each prompt against a set built from the other prompts only."""
    frequencies = [global_frequency(trace) for trace in traces]
    results = []
    for index, trace in enumerate(traces):
        others = [f for position, f in enumerate(frequencies)
                  if position != index]
        held_out = _merge(others) if others else None
        oracle = frequencies[index]

        cold = simulate_warm_lru(
            trace, prompt_tokens=prompt_tokens, quota=quota)
        warm = simulate_warm_lru(
            trace, prompt_tokens=prompt_tokens, quota=quota,
            preload=held_out)
        best = simulate_warm_lru(
            trace, prompt_tokens=prompt_tokens, quota=quota, preload=oracle)

        def resident(frequency):
            if frequency is None:
                return [set() for _ in trace]
            return [
                {expert for expert, _ in frequency[layer].most_common(quota)}
                for layer in range(len(trace))
            ]

        results.append({
            "prompt": index,
            "cold_misses_per_token": cold["decode_misses_per_token"],
            "held_out_misses_per_token": warm["decode_misses_per_token"],
            "oracle_misses_per_token": best["decode_misses_per_token"],
            "cold_prefill_gib": cold_prefill_blocks(
                trace, prompt_tokens=prompt_tokens,
                resident=resident(None)) * block_bytes / 2 ** 30,
            "held_out_prefill_gib": cold_prefill_blocks(
                trace, prompt_tokens=prompt_tokens,
                resident=resident(held_out)) * block_bytes / 2 ** 30,
            "oracle_prefill_gib": cold_prefill_blocks(
                trace, prompt_tokens=prompt_tokens,
                resident=resident(oracle)) * block_bytes / 2 ** 30,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-file", type=Path, required=True,
                        help="one prompt per line; blank lines ignored")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--quotas", default="43,57,64")
    parser.add_argument("--block-bytes", type=int, default=DEFAULT_BLOCK_BYTES)
    parser.add_argument(
        "--save-traces", type=Path,
        help="write the per-prompt traces, so a warm set built from some of "
             "them can be replayed against another on real hardware")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    prompts = [
        line.strip()
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) < 3:
        parser.error("leave-one-out needs at least three prompts")

    traces = collect_traces(
        args.checkpoint, prompts,
        prompt_tokens=args.prompt_tokens,
        new_tokens=args.new_tokens,
        max_seq=args.max_seq,
        device=args.device,
    )

    if args.save_traces is not None:
        args.save_traces.parent.mkdir(parents=True, exist_ok=True)
        with args.save_traces.open("w", encoding="utf-8") as f:
            json.dump({"prompt_tokens": args.prompt_tokens,
                       "traces": traces}, f)
            f.write("\n")
        print(f"wrote {len(traces)} traces to {args.save_traces}")

    quotas = tuple(int(value) for value in args.quotas.split(","))
    report = {"prompt_count": len(prompts), "quotas": list(quotas),
              "prompt_tokens": args.prompt_tokens, "by_quota": {}}
    for quota in quotas:
        rows = leave_one_out(
            traces, prompt_tokens=args.prompt_tokens, quota=quota,
            block_bytes=args.block_bytes)
        report["by_quota"][str(quota)] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\n{'quota':>6} {'metric':<22} {'cold':>9} {'held-out':>9} "
          f"{'oracle':>9} {'held-out win':>13}")
    for quota in quotas:
        rows = report["by_quota"][str(quota)]
        for label, keys in (
            ("decode miss/token", (
                "cold_misses_per_token", "held_out_misses_per_token",
                "oracle_misses_per_token")),
            ("cold prefill GiB", (
                "cold_prefill_gib", "held_out_prefill_gib",
                "oracle_prefill_gib")),
        ):
            cold, held, oracle = (
                statistics.mean(row[key] for row in rows) for key in keys)
            win = (1.0 - held / cold) * 100.0 if cold else 0.0
            print(f"{quota:>6} {label:<22} {cold:>9.2f} {held:>9.2f} "
                  f"{oracle:>9.2f} {win:>12.1f}%")


if __name__ == "__main__":
    main()
