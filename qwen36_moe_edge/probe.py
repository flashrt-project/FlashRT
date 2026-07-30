#!/usr/bin/env python3
"""Memory and quantization probes for the Qwen3.6-MoE edge path."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from qwen36_moe_edge.quantize_experts import (
    HIDDEN,
    INTERMEDIATE,
    NUM_EXPERTS,
    NUM_LAYERS,
    CheckpointReader,
    _hadamard16,
    _int4_weight,
    _int8_weight,
    _layout,
    dequantize_int4,
)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _category(name: str) -> str:
    if ".mlp.experts." in name:
        return "routed_experts"
    if name in (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
    ):
        return "embed_lm_head"
    if ".linear_attn." in name and not name.endswith("norm.weight"):
        return "gdn_weights"
    dense_markers = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".linear_attn.out_proj.weight",
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
        ".mlp.shared_expert.down_proj.weight",
    )
    if any(marker in name for marker in dense_markers):
        return "other_dense"
    return "norm_router_misc"


def _quantized_bytes(
        shape: tuple[int, ...], *, bits: int, group_size: int
) -> int:
    elements = _numel(shape)
    if len(shape) < 2:
        return elements * 2
    rows = _numel(shape[:-1])
    columns = shape[-1]
    if bits == 8:
        return elements + rows * 2
    return (elements + 1) // 2 + rows * (
        (columns + group_size - 1) // group_size)


def memory_probe(
        checkpoint: Path,
        group_size: int,
        budget_gib: float,
        runtime_reserve_gib: float) -> None:
    with (checkpoint / "model.safetensors.index.json").open(
            encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    readers = {
        shard: safe_open(
            checkpoint / shard, framework="pt", device="cpu")
        for shard in set(weight_map.values())
    }
    categories: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    for name, shard in weight_map.items():
        if ".visual." in name or name.startswith("mtp."):
            continue
        shape = tuple(readers[shard].get_slice(name).get_shape())
        categories.setdefault(_category(name), []).append(
            (_numel(shape), shape))

    print("category              bf16 GiB   int8 GiB   int4 GiB")
    for category, tensors in sorted(categories.items()):
        bf16 = sum(elements * 2 for elements, _ in tensors)
        int8 = sum(
            _quantized_bytes(shape, bits=8, group_size=group_size)
            for _, shape in tensors
        )
        int4 = sum(
            _quantized_bytes(shape, bits=4, group_size=group_size)
            for _, shape in tensors
        )
        print(
            f"{category:22s} {bf16 / 2**30:8.3f} "
            f"{int8 / 2**30:10.3f} {int4 / 2**30:10.3f}"
        )
    for quant_format in ("int8", "int4"):
        layout = _layout(quant_format, group_size)
        print(
            f"{quant_format} expert block: "
            f"{sum(layout.values()) / 2**20:.4f} MiB"
        )

    int8_resident = sum(
        _quantized_bytes(shape, bits=8, group_size=group_size)
        for category, tensors in categories.items()
        if category != "routed_experts"
        for _, shape in tensors
    )
    int4_resident = sum(
        (
            elements * 2
            if category == "gdn_weights"
            else _quantized_bytes(
                shape, bits=4, group_size=group_size)
        )
        for category, tensors in categories.items()
        if category != "routed_experts"
        for elements, shape in tensors
    )
    budget_bytes = int(budget_gib * 2**30)
    reserve_bytes = int(runtime_reserve_gib * 2**30)
    print(
        f"\n{budget_gib:.2f} GiB budget, "
        f"{runtime_reserve_gib:.2f} GiB runtime reserve"
    )
    for quant_format, resident in (
        ("int8", int8_resident),
        ("int4-mixed", int4_resident),
    ):
        block_format = "int8" if quant_format == "int8" else "int4"
        block_bytes = sum(_layout(block_format, group_size).values())
        available = max(0, budget_bytes - reserve_bytes - resident)
        quota = available // block_bytes // NUM_LAYERS
        cache_bytes = quota * NUM_LAYERS * block_bytes
        projected = resident + cache_bytes + reserve_bytes
        print(
            f"{quant_format:10s}: resident={resident / 2**30:.3f} GiB, "
            f"quota={quota} experts/layer, "
            f"cache={cache_bytes / 2**30:.3f} GiB, "
            f"projected={projected / 2**30:.3f} GiB"
        )


def quality_probe(
        checkpoint: Path,
        *,
        layers: int,
        group_size: int,
        device: str,
) -> None:
    reader = CheckpointReader(checkpoint)
    generator = torch.Generator(device=device).manual_seed(2026)
    scores = {
        "w8a16": [],
        "w8a8": [],
        "int4_w4a4": [],
        "int4_rht16_w4a4": [],
    }
    selected_layers = torch.linspace(
        0, NUM_LAYERS - 1, steps=layers).round().int().tolist()
    for layer in selected_layers:
        expert = (layer * 37 + 11) % NUM_EXPERTS
        gate_up = reader.expert(
            layer, "gate_up_proj", expert
        ).to(device=device, dtype=torch.float32)
        down = reader.expert(
            layer, "down_proj", expert
        ).to(device=device, dtype=torch.float32)
        activation = torch.randn(
            16, HIDDEN, generator=generator, device=device)
        projected = activation @ gate_up.T
        reference = (
            F.silu(projected[:, :INTERMEDIATE])
            * projected[:, INTERMEDIATE:]
        ) @ down.T

        gu8, gu8_scale = _int8_weight(gate_up)
        dn8, dn8_scale = _int8_weight(down)
        gu8 = gu8.float() * gu8_scale.float()[:, None]
        dn8 = dn8.float() * dn8_scale.float()[:, None]
        for mode in ("w8a16", "w8a8"):
            current = activation
            if mode == "w8a8":
                scale = (
                    current.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                    / 127.0
                )
                current = (
                    current / scale
                ).round().clamp(-127, 127) * scale
            projected = current @ gu8.T
            current = (
                F.silu(projected[:, :INTERMEDIATE])
                * projected[:, INTERMEDIATE:]
            )
            if mode == "w8a8":
                scale = (
                    current.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                    / 127.0
                )
                current = (
                    current / scale
                ).round().clamp(-127, 127) * scale
            output = current @ dn8.T
            scores[mode].append(F.cosine_similarity(
                reference.flatten(), output.flatten(), dim=0).item())

        for mode, use_rht, current_group in (
            ("int4_w4a4", False, group_size),
            ("int4_rht16_w4a4", True, 16),
        ):
            transform = _hadamard16(gate_up.device)
            gu_source = gate_up
            current = activation
            if use_rht:
                gu_source = (
                    gate_up.reshape(-1, 16) @ transform
                ).reshape_as(gate_up)
                current = (
                    activation.reshape(-1, 16) @ transform
                ).reshape_as(activation)
            gu4, gu4_scale, gu4_alpha = _int4_weight(
                gu_source, current_group)
            gu4 = dequantize_int4(
                gu4, gu4_scale, HIDDEN, current_group, gu4_alpha)
            current4, current4_scale, current4_alpha = _int4_weight(
                current, current_group)
            current = dequantize_int4(
                current4, current4_scale, HIDDEN, current_group,
                current4_alpha)
            projected = current @ gu4.T
            current = (
                F.silu(projected[:, :INTERMEDIATE])
                * projected[:, INTERMEDIATE:]
            )
            dn_source = down
            if use_rht:
                dn_source = (
                    down.reshape(-1, 16) @ transform
                ).reshape_as(down)
                current = (
                    current.reshape(-1, 16) @ transform
                ).reshape_as(current)
            dn4, dn4_scale, dn4_alpha = _int4_weight(
                dn_source, current_group)
            dn4 = dequantize_int4(
                dn4, dn4_scale, INTERMEDIATE, current_group, dn4_alpha)
            current4, current4_scale, current4_alpha = _int4_weight(
                current, current_group)
            current = dequantize_int4(
                current4, current4_scale, INTERMEDIATE, current_group,
                current4_alpha)
            output = current @ dn4.T
            scores[mode].append(F.cosine_similarity(
                reference.flatten(), output.flatten(), dim=0).item())

    for mode, values in scores.items():
        print(
            f"{mode}: min={min(values):.6f} "
            f"mean={statistics.mean(values):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("memory", "quality"), required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--sample-layers", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-gib", type=float, default=7.0)
    parser.add_argument("--runtime-reserve-gib", type=float, default=1.5)
    args = parser.parse_args()
    if args.mode == "memory":
        memory_probe(
            args.checkpoint,
            args.group_size,
            args.budget_gib,
            args.runtime_reserve_gib,
        )
    else:
        quality_probe(
            args.checkpoint,
            layers=args.sample_layers,
            group_size=args.group_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
