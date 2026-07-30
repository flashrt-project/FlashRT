#!/usr/bin/env python3
"""Collect router selections and simulate bounded per-layer expert caches.

Two cache policies are simulated from the same trace:

``simulate_lru``
    One LRU per layer holding every access. Prefill touches far more experts
    than the cache can hold, so by the time decode starts the LRU contains
    whatever the end of the prompt happened to use.

``simulate_two_tier``
    A per-layer warm set chosen from the prompt and never evicted, plus a
    small LRU ring for everything else. Prefill can no longer displace the
    warm set, so the decode hit rate follows warm-set coverage instead of
    prompt churn.

The reported miss count per token, multiplied by the expert block size, is the
per-token read volume a streaming runtime has to sustain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path

import torch

from flash_rt.frontends.torch._nexn2_rtx_decode import (
    Nexn2DecodeState,
    generate_greedy,
)
from flash_rt.frontends.torch.qwen36_moe_rtx import (
    Qwen36MoeTextFrontendRtx,
)


# INT4 group-16 routed-expert block, matching quantize_experts._layout.
DEFAULT_BLOCK_BYTES = 1769472


def simulate_lru(
        trace: list[list[list[int]]],
        *,
        prompt_tokens: int,
        quota: int) -> dict[str, float]:
    """Single per-layer LRU shared by prefill and decode."""
    prompt_accesses = prompt_misses = 0
    decode_accesses = decode_misses = 0
    for layer_trace in trace:
        cache: OrderedDict[int, None] = OrderedDict()
        for step, experts in enumerate(layer_trace):
            prompt = step < prompt_tokens
            for expert in experts:
                if prompt:
                    prompt_accesses += 1
                else:
                    decode_accesses += 1
                if expert in cache:
                    cache.move_to_end(expert)
                    continue
                if prompt:
                    prompt_misses += 1
                else:
                    decode_misses += 1
                if len(cache) >= quota:
                    cache.popitem(last=False)
                cache[expert] = None
    decode_steps = len(trace[0]) - prompt_tokens
    return {
        "prompt_hit_rate": 1.0 - prompt_misses / prompt_accesses,
        "decode_hit_rate": 1.0 - decode_misses / decode_accesses,
        "decode_misses_per_token": decode_misses / decode_steps,
    }


def simulate_two_tier(
        trace: list[list[list[int]]],
        *,
        prompt_tokens: int,
        pinned: int,
        stream: int,
        warm_from: str = "prompt") -> dict[str, float]:
    """Pinned per-layer warm set plus a per-layer LRU ring.

    ``warm_from="prompt"`` selects the warm set from prompt-phase selection
    counts, which is what a runtime can actually do. ``warm_from="decode"``
    selects it from the decode phase instead; that is not implementable, but
    it bounds how much a better warm-set heuristic could win.
    """
    if warm_from not in ("prompt", "decode"):
        raise ValueError(f"unsupported warm_from: {warm_from!r}")
    decode_accesses = decode_misses = warm_hits = 0
    for layer_trace in trace:
        source = (
            layer_trace[:prompt_tokens] if warm_from == "prompt"
            else layer_trace[prompt_tokens:]
        )
        counts: Counter[int] = Counter()
        for experts in source:
            counts.update(experts)
        warm = {expert for expert, _ in counts.most_common(pinned)}
        ring: OrderedDict[int, None] = OrderedDict()
        for experts in layer_trace[prompt_tokens:]:
            for expert in experts:
                decode_accesses += 1
                if expert in warm:
                    warm_hits += 1
                    continue
                if expert in ring:
                    ring.move_to_end(expert)
                    continue
                decode_misses += 1
                if stream:
                    if len(ring) >= stream:
                        ring.popitem(last=False)
                    ring[expert] = None
    decode_steps = len(trace[0]) - prompt_tokens
    return {
        "decode_hit_rate": 1.0 - decode_misses / decode_accesses,
        "warm_hit_rate": warm_hits / decode_accesses,
        "decode_misses_per_token": decode_misses / decode_steps,
    }


def read_volume(
        misses_per_token: float,
        *,
        block_bytes: int,
        bandwidths: tuple[float, ...]) -> dict[str, float]:
    """Per-token read volume and the tok/s each bandwidth would allow."""
    per_token = misses_per_token * block_bytes
    result = {"mb_per_token": per_token / 1e6}
    for bandwidth in bandwidths:
        result[f"tok_s_at_{bandwidth:g}gbps"] = (
            bandwidth * 1e9 / per_token if per_token else float("inf"))
    return result


_POLICIES = ("single_lru", "two_tier", "two_tier_oracle_warm")


def summarize(
        trace: list[list[list[int]]],
        *,
        prompt_tokens: int,
        quotas: tuple[int, ...],
        stream_fraction: float,
        block_bytes: int,
        bandwidths: tuple[float, ...]) -> dict[str, dict]:
    """Compare both policies across per-layer quotas."""
    summary: dict[str, dict] = {}
    for quota in quotas:
        stream = max(1, int(round(quota * stream_fraction)))
        pinned = max(0, quota - stream)
        entry = {
            "quota": quota,
            "pinned": pinned,
            "stream": stream,
            "single_lru": simulate_lru(
                trace, prompt_tokens=prompt_tokens, quota=quota),
            "two_tier": simulate_two_tier(
                trace, prompt_tokens=prompt_tokens,
                pinned=pinned, stream=stream),
            "two_tier_oracle_warm": simulate_two_tier(
                trace, prompt_tokens=prompt_tokens,
                pinned=pinned, stream=stream, warm_from="decode"),
        }
        for policy in _POLICIES:
            entry[policy].update(read_volume(
                entry[policy]["decode_misses_per_token"],
                block_bytes=block_bytes,
                bandwidths=bandwidths,
            ))
        summary[str(quota)] = entry
    return summary


def format_summary(
        summary: dict[str, dict],
        *,
        quotas: tuple[int, ...],
        bandwidths: tuple[float, ...]) -> str:
    header = f"{'quota':>6} {'pin/str':>8} {'policy':<22}"
    header += f" {'hit':>7} {'miss/tok':>9} {'MB/tok':>8}"
    for bandwidth in bandwidths:
        header += f" {f'{bandwidth:g}GB/s':>9}"
    lines = [header]
    for quota in quotas:
        entry = summary[str(quota)]
        split = f"{entry['pinned']}/{entry['stream']}"
        for policy in _POLICIES:
            values = entry[policy]
            line = f"{quota:>6} {split:>8} {policy:<22}"
            line += f" {values['decode_hit_rate']:>7.4f}"
            line += f" {values['decode_misses_per_token']:>9.2f}"
            line += f" {values['mb_per_token']:>8.1f}"
            for bandwidth in bandwidths:
                line += f" {values[f'tok_s_at_{bandwidth:g}gbps']:>9.2f}"
            lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--quotas", default="8,16,24,27,32,43,64")
    parser.add_argument(
        "--stream-fraction", type=float, default=0.25,
        help="share of each layer's quota held as an evictable LRU ring")
    parser.add_argument("--block-bytes", type=int, default=DEFAULT_BLOCK_BYTES)
    parser.add_argument(
        "--bandwidths", default="1.0,1.5,2.0",
        help="storage read bandwidths in GB/s to project tok/s for")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frontend = Qwen36MoeTextFrontendRtx(
        args.checkpoint,
        device=args.device,
        max_seq=args.max_seq,
        quant_scope="experts",
    )
    input_ids = frontend.tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :args.prompt_tokens].to(args.device)
    if input_ids.shape[1] != args.prompt_tokens:
        parser.error("the supplied prompt is shorter than --prompt-tokens")

    state = Nexn2DecodeState(
        frontend._weights, args.max_seq, args.device)
    state.batched_prefill = False
    state.router_trace = {
        layer: [] for layer in range(state.num_layers)}
    with torch.no_grad():
        generated = generate_greedy(
            state,
            input_ids,
            args.new_tokens,
            frontend._fvk,
            args.device,
        )

    trace = [
        [list(experts) for experts in state.router_trace[layer]]
        for layer in range(state.num_layers)
    ]
    quotas = tuple(int(value) for value in args.quotas.split(","))
    bandwidths = tuple(
        float(value) for value in args.bandwidths.split(","))
    summary = summarize(
        trace,
        prompt_tokens=args.prompt_tokens,
        quotas=quotas,
        stream_fraction=args.stream_fraction,
        block_bytes=args.block_bytes,
        bandwidths=bandwidths,
    )
    result = {
        "prompt_tokens": args.prompt_tokens,
        "block_bytes": args.block_bytes,
        "stream_fraction": args.stream_fraction,
        "generated_tokens": generated,
        "trace": trace,
        "cache": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")

    print(format_summary(
        summary, quotas=quotas, bandwidths=bandwidths))


if __name__ == "__main__":
    main()
