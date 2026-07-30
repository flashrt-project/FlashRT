#!/usr/bin/env python3
"""Write Qwen3.6 routed experts as fixed-size INT8 or INT4 blocks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open


NUM_LAYERS = 40
NUM_EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 512

# Expert blocks are padded so that every block's offset and length are a
# multiple of the logical block size. A device whose memory holds only a
# fraction of the experts cannot afford to stream them through the page
# cache -- the cache competes with the resident weights for the same
# physical memory -- so the reader has to use O_DIRECT, which requires
# aligned offsets and lengths.
BLOCK_ALIGNMENT = 4096


class CheckpointReader:
    def __init__(self, checkpoint: Path):
        index_path = checkpoint / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.readers = {
            shard: safe_open(
                checkpoint / shard, framework="pt", device="cpu")
            for shard in set(self.weight_map.values())
        }

    def expert(self, layer: int, name: str, expert: int) -> torch.Tensor:
        key = (
            f"model.language_model.layers.{layer}."
            f"mlp.experts.{name}"
        )
        shard = self.weight_map[key]
        return self.readers[shard].get_slice(key)[expert]


def _int8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        weight.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0
    ).to(torch.float16)
    quantized = (
        weight.float() / scale.float()[:, None]
    ).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _hadamard16(device: torch.device) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    for _ in range(4):
        matrix = torch.cat((
            torch.cat((matrix, matrix), dim=1),
            torch.cat((matrix, -matrix), dim=1),
        ), dim=0)
    return matrix / 4.0


def _rht16(weight: torch.Tensor) -> torch.Tensor:
    rows, columns = weight.shape
    return (
        weight.float().reshape(rows, columns // 16, 16)
        @ _hadamard16(weight.device)
    ).reshape(rows, columns)


def _int4_weight(
        weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(
            f"K={columns} is not divisible by group_size={group_size}")
    grouped = weight.float().reshape(rows, columns // group_size, group_size)
    scale = (
        grouped.abs().amax(dim=2).clamp_min(1e-8) / 7.0
    ).to(torch.float8_e4m3fn)
    scale_float = scale.float().clamp_min(2.0**-9)
    values = (
        grouped / scale_float.unsqueeze(-1)
    ).round().clamp(-7, 7).to(torch.int8).reshape(rows, columns)
    magnitude = values.abs().to(torch.uint8)
    code = magnitude | ((values < 0).to(torch.uint8) << 3)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.contiguous(), scale.view(torch.uint8).contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().cpu().numpy().tobytes()


def quantize_expert(
        gate_up: torch.Tensor,
        down: torch.Tensor,
        *,
        quant_format: str,
        group_size: int,
        device: str,
) -> bytes:
    if quant_format not in ("int8", "int4", "int4-rht"):
        raise ValueError(f"unsupported quantization format: {quant_format}")
    if quant_format == "int4-rht" and group_size != 16:
        raise ValueError("int4-rht requires group_size=16")
    gate_up = gate_up.to(device=device, dtype=torch.float32)
    down = down.to(device=device, dtype=torch.float32)
    if quant_format == "int8":
        gu_weight, gu_scale = _int8_weight(gate_up)
        dn_weight, dn_scale = _int8_weight(down)
    else:
        if quant_format == "int4-rht":
            gate_up = _rht16(gate_up)
            down = _rht16(down)
        gu_weight, gu_scale = _int4_weight(gate_up, group_size)
        dn_weight, dn_scale = _int4_weight(down, group_size)
    return b"".join((
        _tensor_bytes(gu_weight),
        _tensor_bytes(gu_scale),
        _tensor_bytes(dn_weight),
        _tensor_bytes(dn_scale),
        bytes(_layout(quant_format, group_size)["padding"]),
    ))


def _parse_layers(value: str) -> range:
    start, stop = (int(part) for part in value.split(":", 1))
    if start < 0 or stop > NUM_LAYERS or start >= stop:
        raise argparse.ArgumentTypeError(
            f"layers must satisfy 0 <= start < stop <= {NUM_LAYERS}")
    return range(start, stop)


def _layout(quant_format: str, group_size: int) -> dict[str, int]:
    if quant_format == "int8":
        gu_weight = 2 * INTERMEDIATE * HIDDEN
        gu_scale = 2 * INTERMEDIATE * 2
        dn_weight = HIDDEN * INTERMEDIATE
        dn_scale = HIDDEN * 2
    else:
        gu_weight = 2 * INTERMEDIATE * HIDDEN // 2
        gu_scale = 2 * INTERMEDIATE * (HIDDEN // group_size)
        dn_weight = HIDDEN * INTERMEDIATE // 2
        dn_scale = HIDDEN * (INTERMEDIATE // group_size)
    layout = {
        "gate_up_weight": gu_weight,
        "gate_up_scale": gu_scale,
        "down_weight": dn_weight,
        "down_scale": dn_scale,
    }
    # Trailing pad keeps every expert's offset and length aligned. The INT4
    # group-16 payload happens to be a multiple already; the INT8 payload is
    # 3,151,872 B, which is 769.5 blocks.
    layout["padding"] = -sum(layout.values()) % BLOCK_ALIGNMENT
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("int8", "int4", "int4-rht"), required=True)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--layers", type=_parse_layers, default=range(40))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.format != "int8" and args.group_size not in (16, 32, 64, 128):
        parser.error("--group-size must be one of 16, 32, 64, or 128")
    if args.format == "int4-rht" and args.group_size != 16:
        parser.error("--format int4-rht requires --group-size 16")

    args.output.mkdir(parents=True, exist_ok=True)
    layout = _layout(args.format, args.group_size)
    block_bytes = sum(layout.values())
    manifest = {
        "format": f"flashrt-qwen36-moe-{args.format}-experts-v1",
        "group_size": args.group_size if args.format != "int8" else None,
        "rht": args.format == "int4-rht",
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "block_layout": list(layout),
        "block_sizes": layout,
        "block_bytes": block_bytes,
        "block_alignment": BLOCK_ALIGNMENT,
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    reader = CheckpointReader(args.checkpoint)
    for layer in args.layers:
        output_path = args.output / f"experts_layer_{layer:02d}.bin"
        expected_bytes = NUM_EXPERTS * block_bytes
        if output_path.is_file() and output_path.stat().st_size == expected_bytes:
            print(f"layer {layer}: already complete")
            continue
        temporary_path = output_path.with_suffix(".bin.tmp")
        started = time.perf_counter()
        with temporary_path.open("wb") as f:
            for expert in range(NUM_EXPERTS):
                gate_up = reader.expert(
                    layer, "gate_up_proj", expert)
                down = reader.expert(layer, "down_proj", expert)
                block = quantize_expert(
                    gate_up,
                    down,
                    quant_format=args.format,
                    group_size=args.group_size,
                    device=args.device,
                )
                if len(block) != block_bytes:
                    raise RuntimeError(
                        f"expert block is {len(block)} bytes; "
                        f"expected {block_bytes}")
                f.write(block)
        os.replace(temporary_path, output_path)
        elapsed = time.perf_counter() - started
        gib = expected_bytes / 2**30
        print(
            f"layer {layer}: {gib:.3f} GiB in {elapsed:.2f}s "
            f"({gib / elapsed:.2f} GiB/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
