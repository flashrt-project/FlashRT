#!/usr/bin/env python3
"""Collect SM120 router selections and simulate bounded per-layer LRUs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from flash_rt.frontends.torch._nexn2_rtx_decode import (
    Nexn2DecodeState,
    generate_greedy,
)
from flash_rt.frontends.torch.qwen36_moe_rtx import (
    Qwen36MoeTextFrontendRtx,
)


def simulate_lru(
        trace: list[list[list[int]]],
        *,
        prompt_tokens: int,
        quota: int) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--quotas", default="8,16,24,32,48,64")
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
    result = {
        "prompt_tokens": args.prompt_tokens,
        "generated_tokens": generated,
        "trace": trace,
        "lru": {},
    }
    for quota in (int(value) for value in args.quotas.split(",")):
        result["lru"][str(quota)] = simulate_lru(
            trace, prompt_tokens=args.prompt_tokens, quota=quota)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")

    for quota, values in result["lru"].items():
        print(
            f"quota={quota}: decode_hit={values['decode_hit_rate']:.4f}, "
            f"misses/token={values['decode_misses_per_token']:.2f}"
        )


if __name__ == "__main__":
    main()
