#!/usr/bin/env python3
"""Export a FlashRT quant manifest from a ModelOpt/ONNX QDQ graph.

The script intentionally requires a mapping file from ONNX tensor/node names
to FlashRT internal GEMM site names.  That keeps graph-specific fusion rules
(Q/K/V -> qkv, gate/up -> gate_up) outside the runtime.

Mapping schema:

{
  "model": "pi05",
  "sites": {
    "encoder_ffn_down_w_16": {
      "weight_scale": "onnx_initializer_name",
      "activation_scale": "onnx_initializer_name"
    },
    "encoder_attn_qkv_w_0": {
      "onnx_nodes": [
        "encoder.layers.0.attn.q/MatMul",
        "encoder.layers.0.attn.k/MatMul",
        "encoder.layers.0.attn.v/MatMul"
      ]
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_onnx(onnx_path: Path):
    try:
        import onnx
        from onnx import numpy_helper
    except Exception as exc:
        raise SystemExit(
            "onnx is required. Install the isolated env from "
            "scripts/setup_onnx_modelopt_env.sh") from exc

    model = onnx.load(str(onnx_path))
    return model, numpy_helper


def _load_initializer_scalars(model, numpy_helper) -> dict[str, float | list[float]]:
    out: dict[str, float | list[float]] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).astype("float32").reshape(-1)
        if arr.size == 1:
            out[init.name] = float(arr[0])
        else:
            out[init.name] = [float(x) for x in arr.tolist()]
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _node_indexes(model) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes_by_name = {}
    nodes_by_output = {}
    for node in model.graph.node:
        if node.name:
            nodes_by_name[node.name] = node
        for output in node.output:
            nodes_by_output[output] = node
    return nodes_by_name, nodes_by_output


def _find_qdq_scale_name(
    tensor_name: str,
    nodes_by_output: dict[str, Any],
    depth: int = 0,
) -> str | None:
    if depth > 8:
        return None
    node = nodes_by_output.get(tensor_name)
    if node is None:
        return None
    if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
        return node.input[1]
    if node.op_type in {"Identity", "Reshape", "Squeeze", "Unsqueeze", "Transpose"}:
        if node.input:
            return _find_qdq_scale_name(node.input[0], nodes_by_output, depth + 1)
    return None


def _infer_scales_from_nodes(
    rec: dict[str, Any],
    nodes_by_name: dict[str, Any],
    nodes_by_output: dict[str, Any],
    input_index: int,
) -> list[str]:
    names = []
    for node_name in _as_list(rec.get("onnx_node")) + _as_list(rec.get("onnx_nodes")):
        node = nodes_by_name.get(str(node_name))
        if node is None:
            raise KeyError(f"ONNX node not found in mapping: {node_name}")
        if input_index >= len(node.input):
            raise ValueError(
                f"ONNX node {node_name!r} has no input index {input_index}")
        scale_name = _find_qdq_scale_name(node.input[input_index], nodes_by_output)
        if scale_name is None:
            raise ValueError(
                f"could not infer Q/DQ scale for node {node_name!r} "
                f"input {input_index}")
        names.append(scale_name)
    return names


def _resolve_scale_names(
    rec: dict[str, Any],
    singular_key: str,
    plural_key: str,
    nodes_by_name: dict[str, Any],
    nodes_by_output: dict[str, Any],
    input_index: int,
) -> str | list[str] | None:
    explicit = _as_list(rec.get(singular_key)) + _as_list(rec.get(plural_key))
    inferred = []
    if not explicit and (rec.get("onnx_node") or rec.get("onnx_nodes")):
        inferred = _infer_scales_from_nodes(
            rec, nodes_by_name, nodes_by_output, input_index)
    names = [str(x) for x in explicit + inferred]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return names


def _tensor_spec(
    scale_names: str | list[str] | None,
    scales: dict[str, Any],
) -> dict[str, Any] | None:
    if not scale_names:
        return None
    names = [str(x) for x in _as_list(scale_names)]
    missing = [name for name in names if name not in scales]
    if missing:
        raise KeyError(
            "scale initializer(s) not found in ONNX graph: "
            + ", ".join(missing))
    values = [scales[name] for name in names]
    if all(isinstance(value, float) for value in values):
        # Fused FlashRT sites (QKV, Gate+Up) need one per-tensor scale.
        # Using max(scale_i) is equivalent to quantizing the concatenated
        # tensor when each source scale came from amax / fp8_max.
        scale = max(float(value) for value in values)
    elif len(values) == 1:
        scale = values[0]
    else:
        scale = []
        for value in values:
            if isinstance(value, float):
                scale.append(float(value))
            else:
                scale.extend(float(x) for x in value)
    granularity = "per_tensor" if isinstance(scale, float) else "per_channel"
    return {
        "dtype": "fp8_e4m3fn",
        "granularity": granularity,
        "scale": scale,
        "scale_name": scale_names,
    }


def export_manifest(
    onnx_path: Path,
    mapping_path: Path,
    out_path: Path,
    include_source_paths: bool = False,
) -> None:
    model, numpy_helper = _load_onnx(onnx_path)
    scales = _load_initializer_scalars(model, numpy_helper)
    nodes_by_name, nodes_by_output = _node_indexes(model)
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    sites = {}
    for site_name, rec in mapping.get("sites", {}).items():
        weight_input = int(rec.get("weight_input", 1))
        activation_input = int(rec.get("activation_input", 0))
        weight_scale = _resolve_scale_names(
            rec, "weight_scale", "weight_scales",
            nodes_by_name, nodes_by_output, weight_input)
        activation_scale = _resolve_scale_names(
            rec, "activation_scale", "activation_scales",
            nodes_by_name, nodes_by_output, activation_input)
        weight = _tensor_spec(weight_scale, scales)
        activation = _tensor_spec(activation_scale, scales)
        sites[site_name] = {
            "precision": rec.get("precision", "fp8"),
            "weight": weight,
            "activation": activation,
            "onnx_nodes": rec.get("onnx_nodes", []),
            "notes": rec.get("notes"),
        }

    manifest = {
        "format": "flashrt_quant_manifest_v1",
        "model": mapping.get("model", "pi05"),
        "source": "onnx_modelopt",
        "metadata": {
            "producer": "onnx_modelopt_export_manifest.py",
            "onnx_file": onnx_path.name,
            "mapping_file": mapping_path.name,
        },
        "sites": sites,
    }
    if include_source_paths:
        manifest["metadata"]["onnx_path"] = str(onnx_path)
        manifest["metadata"]["mapping_path"] = str(mapping_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path} ({len(sites)} site(s))")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--include-source-paths",
        action="store_true",
        help="Record input ONNX and mapping paths in manifest metadata.",
    )
    args = parser.parse_args()
    export_manifest(
        args.onnx,
        args.mapping,
        args.out,
        include_source_paths=args.include_source_paths,
    )


if __name__ == "__main__":
    main()
