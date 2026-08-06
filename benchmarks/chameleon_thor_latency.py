#!/usr/bin/env python3
"""Chameleon-7B (Thor sm_110) latency benchmark.

Measures standalone Chameleon-7B prefill latency on real images with clean
stage separation:

- ``--reuse-input-ids``: build real-image input ids once, then time only
  embed + backbone + lm_head (transformer-prefill-only, HF-comparable).
- Default: full ``prefill()`` E2E including VQGAN tokenization.
- ``--use-trt-vqgan``: explicit TensorRT VQGAN opt-in (recommended when
  compatible engines exist; the generic default stays eager).
- FA4 attention: enable with ``FLASHRT_CHAMELEON_FA4_ATTN=1`` (needs the
  ``thor-fa4`` pip extra; prints whether it is active).

Latency is wall-clock P50 (per CONTRIBUTING.md: quickstart --benchmark
style; CUDA-graph replayed latency is what the pipeline measures inside the
graph). Every result row records device, VQGAN backend, FA4 state, Se,
fp8/fp16 and graph settings for reproducible reporting.

Usage:

    python benchmarks/chameleon_thor_latency.py \
        --checkpoint /path/to/Chameleon_7B_mGPT \
        --image-dir /path/to/images \
        --iters 20 --warmup 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import torch


def _stats(xs: List[float]) -> Dict[str, float]:
    a = sorted(xs)
    n = len(a)
    return {"mean": sum(a) / n, "p50": a[n // 2], "min": a[0], "max": a[-1]}


def _load_images(image_dir: pathlib.Path, max_images: int):
    from PIL import Image

    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not paths:
        raise FileNotFoundError(f"No real images under {image_dir}")
    paths = paths[:max_images]
    return [Image.open(p).convert("RGB") for p in paths], [str(p) for p in paths]


def _pad_ids(input_ids: List[int], pad_id: int = 1):
    real_len = len(input_ids)
    padded = list(input_ids)
    rem = len(padded) % 16
    if rem:
        padded.extend([pad_id] * (16 - rem))
    return padded, real_len


def _prefill_once(fe, prompt, images, cached_ids, use_graph: bool) -> Dict[str, float]:
    """One timed prefill; returns stage latencies in ms."""
    times: Dict[str, float] = {}
    t0 = time.perf_counter()
    ids = fe.encode_prompt(prompt, images) if cached_ids is None else cached_ids
    torch.cuda.synchronize()
    times["encode_ms"] = (time.perf_counter() - t0) * 1000.0

    padded, real_len = _pad_ids(ids)
    fe._real_len = real_len
    fe.Se = len(padded)
    fe._last_input_ids = padded
    if fe._use_autotune:
        fe._autotune_gemms(fe.Se)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    fe._embed_ids(padded)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    if use_graph:
        fe._capture_graph(fe.Se)
        fe._infer_graph.replay()
    else:
        fe._run_backbone(fe.Se)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    fe._project_last()
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    times["prepare_ms"] = (t1 - t0) * 1000.0 - times["encode_ms"]
    times["embed_ms"] = (t2 - t1) * 1000.0
    times["backbone_ms"] = (t3 - t2) * 1000.0
    times["lm_head_ms"] = (t4 - t3) * 1000.0
    times["transformer_ms"] = times["embed_ms"] + times["backbone_ms"] + times["lm_head_ms"]
    times["total_ms"] = (t4 - t0) * 1000.0
    return times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-dir", required=True,
                    help="Directory of real input images")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--max-images", type=int, default=1)
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--use-trt-vqgan", action="store_true",
                    help="Use TensorRT VQGAN if compatible engines exist "
                         "(recommended when engines are available; default is eager VQGAN)")
    ap.add_argument("--reuse-input-ids", action="store_true",
                    help="time transformer prefill only (VQGAN excluded)")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--use-fp16", action="store_true",
                    help="FP16 reference path instead of dynamic FP8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output", default=None, help="JSON output path")
    args = ap.parse_args()

    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor
    from flash_rt.hardware.thor import fa4_backend

    images, image_paths = _load_images(pathlib.Path(args.image_dir), args.max_images)
    use_graph = not args.no_graph
    fp8 = not args.use_fp16

    fe = ChameleonTorchFrontendThor(
        args.checkpoint, use_fp8=fp8, use_cuda_graph=use_graph,
        target_size=args.target_size, use_trt_vqgan=args.use_trt_vqgan)

    cached_ids = None
    if args.reuse_input_ids:
        cached_ids = fe.encode_prompt(args.prompt, images)
        torch.cuda.synchronize()

    for _ in range(args.warmup):
        _prefill_once(fe, args.prompt, images, cached_ids, use_graph)

    rows: Dict[str, List[float]] = {}
    for _ in range(args.iters):
        t = _prefill_once(fe, args.prompt, images, cached_ids, use_graph)
        for k, v in t.items():
            rows.setdefault(k, []).append(v)

    result = {
        "model": "chameleon-7b",
        "device": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "checkpoint": args.checkpoint,
        "image_paths": image_paths,
        "prompt": args.prompt,
        "target_size": args.target_size,
        "fp8": fp8,
        "cuda_graph": use_graph,
        "vqgan_backend": fe.vqgan_backend,
        "fa4_attn": fe.fa4_attn_active,
        "fa4_status": fa4_backend.status(),
        "reuse_input_ids": bool(args.reuse_input_ids),
        "Se": int(fe.Se),
        "latency_ms": {k: _stats(v) for k, v in rows.items()},
    }
    for k, v in result["latency_ms"].items():
        print(f"[chameleon] {k:14s} p50={v['p50']:8.1f} ms  mean={v['mean']:8.1f}")
    print(f"[chameleon] device={result['device']} vqgan={fe.vqgan_backend} "
          f"fa4={fe.fa4_attn_active} fp8={fp8} graph={use_graph} Se={fe.Se}")

    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"[chameleon] wrote {args.output}")


if __name__ == "__main__":
    main()
