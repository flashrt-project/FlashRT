#!/usr/bin/env python3
"""Score routed-expert quantization against real routed activations.

``probe.py --mode quality`` samples experts with ``torch.randn`` activations
and quantizes the activations too. Neither matches how the edge runtime will
use these weights:

- The activations an expert actually sees are the post-norm hidden states of
  tokens the router sent to *that* expert. Gaussian noise has none of their
  structure, and a scale calibrated against it hides errors that real inputs
  expose.
- At M=1 the activation is 4 KiB against a 1.7 MiB weight block, so quantizing
  it buys no bandwidth. The expert path is weight-only, W4A16 or W8A16.

This tool captures the real activations from a forward pass, replays each
sampled expert in BF16 for the reference, and scores the weight-only
reconstructions against it. It can also write a small bundle so a device can
check its own dequantization and kernel against the same references without
loading the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen36_moe_edge.quantize_experts import (
    HIDDEN,
    INTERMEDIATE,
    NUM_LAYERS,
    CheckpointReader,
    _int4_weight,
    _int8_weight,
    _rht16,
    dequantize_int4,
)


SCHEMES = ("w8a16", "w4a16", "w4a16_rht16")


def expert_forward(
        activation: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor) -> torch.Tensor:
    """One routed expert: gate_up, SwiGLU, down."""
    projected = activation @ gate_up.T
    hidden = (
        F.silu(projected[:, :INTERMEDIATE])
        * projected[:, INTERMEDIATE:]
    )
    return hidden @ down.T


def _reconstruct(
        weight: torch.Tensor,
        *,
        scheme: str,
        group_size: int) -> torch.Tensor:
    """Quantize a weight and dequantize it, as the runtime's kernel will."""
    columns = weight.shape[1]
    if scheme == "w8a16":
        quantized, scale = _int8_weight(weight)
        return quantized.float() * scale.float()[:, None]
    source = _rht16(weight) if scheme == "w4a16_rht16" else weight
    packed, scale = _int4_weight(source, group_size)
    return dequantize_int4(packed, scale, columns, group_size)


def score_expert(
        activation: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        scheme: str,
        group_size: int) -> dict[str, float]:
    """Cosine and relative L2 of a weight-only scheme against BF16."""
    reference = expert_forward(activation, gate_up, down)

    gate_up_q = _reconstruct(
        gate_up, scheme=scheme, group_size=group_size)
    down_q = _reconstruct(down, scheme=scheme, group_size=group_size)
    if scheme == "w4a16_rht16":
        # The transform is orthonormal, so rotating both sides of each dot
        # product leaves it unchanged. The runtime rotates activations the
        # same way before calling the kernel.
        projected = _rht16(activation) @ gate_up_q.T
        hidden = (
            F.silu(projected[:, :INTERMEDIATE])
            * projected[:, INTERMEDIATE:]
        )
        output = _rht16(hidden) @ down_q.T
    else:
        output = expert_forward(activation, gate_up_q, down_q)

    difference = (output - reference).flatten()
    return {
        "cosine": F.cosine_similarity(
            reference.flatten(), output.flatten(), dim=0).item(),
        "relative_l2": (
            difference.norm() / reference.flatten().norm().clamp_min(1e-12)
        ).item(),
    }


def collect_activations(
        checkpoint: str,
        *,
        prompt: str,
        prompt_tokens: int,
        new_tokens: int,
        max_seq: int,
        device: str) -> tuple[list[list[list[int]]], list[list[torch.Tensor]]]:
    """Run one eager forward, returning per-layer selections and MoE inputs."""
    from flash_rt.frontends.torch._nexn2_rtx_decode import (
        Nexn2DecodeState,
        generate_greedy,
    )
    from flash_rt.frontends.torch.qwen36_moe_rtx import (
        Qwen36MoeTextFrontendRtx,
    )

    frontend = Qwen36MoeTextFrontendRtx(
        checkpoint, device=device, max_seq=max_seq, quant_scope="experts")
    input_ids = frontend.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).input_ids[:, :prompt_tokens].to(device)
    if input_ids.shape[1] != prompt_tokens:
        raise ValueError("the supplied prompt is shorter than prompt_tokens")

    state = Nexn2DecodeState(frontend._weights, max_seq, device)
    state.batched_prefill = False
    state.router_trace = {layer: [] for layer in range(state.num_layers)}
    state.moe_input_trace = {layer: [] for layer in range(state.num_layers)}
    with torch.no_grad():
        generate_greedy(
            state, input_ids, new_tokens, frontend._fvk, device)

    selections = [
        [list(experts) for experts in state.router_trace[layer]]
        for layer in range(state.num_layers)
    ]
    activations = [
        list(state.moe_input_trace[layer])
        for layer in range(state.num_layers)
    ]
    return selections, activations


def _sampled_pairs(
        selections: list[list[list[int]]],
        *,
        layers: tuple[int, ...],
        experts_per_layer: int) -> list[tuple[int, int, int]]:
    """Pick (layer, expert, step) triples the router actually produced."""
    pairs = []
    for layer in layers:
        seen: dict[int, int] = {}
        for step, experts in enumerate(selections[layer]):
            for expert in experts:
                seen.setdefault(expert, step)
        for expert, step in list(seen.items())[:experts_per_layer]:
            pairs.append((layer, expert, step))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument(
        "--layers", default="0,1,3,19,20,39",
        help="layers to sample; the default spans both attention kinds and "
             "the first, middle and last MoE blocks")
    parser.add_argument("--experts-per-layer", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    layers = tuple(int(value) for value in args.layers.split(","))
    for layer in layers:
        if not 0 <= layer < NUM_LAYERS:
            parser.error(f"layer {layer} is outside 0..{NUM_LAYERS - 1}")

    selections, activations = collect_activations(
        str(args.checkpoint),
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        new_tokens=args.new_tokens,
        max_seq=args.max_seq,
        device=args.device,
    )
    pairs = _sampled_pairs(
        selections, layers=layers, experts_per_layer=args.experts_per_layer)
    print(f"scoring {len(pairs)} routed (layer, expert) pairs", flush=True)

    reader = CheckpointReader(args.checkpoint)
    scores: dict[str, list[float]] = {
        f"{scheme}.{metric}": []
        for scheme in SCHEMES for metric in ("cosine", "relative_l2")
    }
    records = []
    golden: dict[str, torch.Tensor] = {}
    for layer, expert, step in pairs:
        activation = activations[layer][step].to(
            device=args.device, dtype=torch.float32)
        gate_up = reader.expert(layer, "gate_up_proj", expert).to(
            device=args.device, dtype=torch.float32)
        down = reader.expert(layer, "down_proj", expert).to(
            device=args.device, dtype=torch.float32)

        record = {"layer": layer, "expert": expert, "step": step}
        for scheme in SCHEMES:
            values = score_expert(
                activation, gate_up, down,
                scheme=scheme, group_size=args.group_size)
            record[scheme] = values
            for metric, value in values.items():
                scores[f"{scheme}.{metric}"].append(value)
        records.append(record)

        if args.golden is not None:
            key = f"layer{layer:02d}.expert{expert:03d}"
            golden[f"{key}.activation"] = (
                activation.to(torch.bfloat16).cpu())
            golden[f"{key}.reference"] = expert_forward(
                activation, gate_up, down).to(torch.bfloat16).cpu()

    summary = {
        name: {
            "min": min(values),
            "mean": statistics.mean(values),
            "max": max(values),
        }
        for name, values in scores.items()
    }
    result = {
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "group_size": args.group_size,
        "layers": list(layers),
        "pair_count": len(pairs),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    if args.golden is not None:
        from safetensors.torch import save_file

        args.golden.parent.mkdir(parents=True, exist_ok=True)
        save_file(golden, str(args.golden), metadata={
            "hidden_size": str(HIDDEN),
            "intermediate_size": str(INTERMEDIATE),
            "pairs": ",".join(
                f"{layer}:{expert}" for layer, expert, _ in pairs),
        })
        print(f"wrote {len(golden) // 2} reference pairs to {args.golden}")

    print(f"\n{'scheme':<14} {'cos min':>10} {'cos mean':>10} "
          f"{'relL2 max':>10} {'relL2 mean':>11}")
    for scheme in SCHEMES:
        cosine = summary[f"{scheme}.cosine"]
        l2 = summary[f"{scheme}.relative_l2"]
        print(f"{scheme:<14} {cosine['min']:>10.6f} {cosine['mean']:>10.6f} "
              f"{l2['max']:>10.5f} {l2['mean']:>11.5f}")


if __name__ == "__main__":
    main()
