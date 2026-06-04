#!/usr/bin/env python3
"""Generate a Pi0.5 ONNX-to-FlashRT quant mapping skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rt.core.quant.pi05_sites import iter_pi05_fp8_sites  # noqa: E402


def _load_onnx_nodes(path: Path) -> list[Any]:
    try:
        import onnx
    except Exception as exc:
        raise SystemExit(
            "onnx is required. Install the isolated env from "
            "scripts/setup_onnx_modelopt_env.sh") from exc
    model = onnx.load(str(path))
    return list(model.graph.node)


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _node_haystack(node: Any) -> str:
    pieces = [node.name, node.op_type]
    pieces.extend(str(x) for x in node.input)
    pieces.extend(str(x) for x in node.output)
    return _norm(" ".join(pieces))


def _find_node_for_alias(alias: str, nodes: list[Any]) -> str | None:
    alias_norm = _norm(alias)
    candidates = []
    for node in nodes:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        haystack = _node_haystack(node)
        if alias_norm in haystack:
            candidates.append(node)
    if not candidates:
        # Some exporters strip parent prefixes but keep the leaf projection
        # path. Fall back to the final two module components.
        tail = ".".join(alias.split(".")[-2:])
        tail_norm = _norm(tail)
        candidates = [
            node for node in nodes
            if node.op_type in {"MatMul", "Gemm"} and tail_norm in _node_haystack(node)
        ]
    if not candidates:
        return None
    candidates.sort(key=lambda node: (len(node.name), node.name))
    return candidates[0].name


def generate_mapping(
    onnx_path: Path | None,
    *,
    include_inactive: bool = False,
    vision_layers: int = 27,
    encoder_layers: int = 18,
    decoder_layers: int = 18,
) -> dict[str, Any]:
    nodes = _load_onnx_nodes(onnx_path) if onnx_path is not None else []
    sites = {}
    matched = 0
    missing = []

    for spec in iter_pi05_fp8_sites(
        vision_layers=vision_layers,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    ):
        if not include_inactive and not spec.runtime_active:
            continue

        rec: dict[str, Any] = {
            "precision": "fp8",
            "activation_input": 0,
            "weight_input": 1,
        }

        if onnx_path is None:
            rec["onnx_nodes"] = []
            rec["notes"] = "Fill with ModelOpt ONNX MatMul/Gemm node name(s)."
            rec["node_aliases"] = list(spec.aliases)
        else:
            node_names = []
            for alias in spec.aliases:
                node_name = _find_node_for_alias(alias, nodes)
                if node_name is not None:
                    node_names.append(node_name)
            # Keep order but remove duplicates.
            node_names = list(dict.fromkeys(node_names))
            rec["onnx_nodes"] = node_names
            if len(node_names) == len(spec.aliases):
                matched += 1
            else:
                missing.append(spec.name)
                rec["notes"] = (
                    "Auto-mapping incomplete; inspect node_aliases and fill "
                    "onnx_nodes manually.")
                rec["node_aliases"] = list(spec.aliases)

        sites[spec.name] = rec

    metadata = {
        "producer": "pi05_onnx_modelopt_mapping.py",
        "runtime_active_sites": len(sites),
        "matched_sites": matched if onnx_path is not None else None,
        "missing_sites": missing if onnx_path is not None else None,
    }
    if onnx_path is not None:
        metadata["onnx_file"] = onnx_path.name

    return {
        "model": "pi05",
        "metadata": metadata,
        "sites": sites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--vision-layers", type=int, default=27)
    parser.add_argument("--encoder-layers", type=int, default=18)
    parser.add_argument("--decoder-layers", type=int, default=18)
    args = parser.parse_args()

    mapping = generate_mapping(
        args.onnx,
        include_inactive=args.include_inactive,
        vision_layers=args.vision_layers,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
        f.write("\n")

    metadata = mapping["metadata"]
    missing = metadata.get("missing_sites") or []
    print(
        f"wrote {args.out} sites={metadata['runtime_active_sites']} "
        f"matched={metadata['matched_sites']} missing={len(missing)}")
    if missing:
        print("missing: " + ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else ""))


if __name__ == "__main__":
    main()
