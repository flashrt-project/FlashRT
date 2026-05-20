#!/usr/bin/env python3
"""Opt-in Pi0.5 RTX full-FP16 CUDA Graph benchmark."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

import flash_rt


def _stats(xs: list[float]) -> str:
    xs = sorted(float(x) for x in xs)
    return (
        f"n={len(xs)} "
        f"p50={xs[int(0.50 * (len(xs) - 1))]:.3f} "
        f"p90={xs[int(0.90 * (len(xs) - 1))]:.3f} "
        f"p95={xs[int(0.95 * (len(xs) - 1))]:.3f} "
        f"mean={statistics.mean(xs):.3f} "
        f"min={xs[0]:.3f} max={xs[-1]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--hardware",
        default="auto",
        choices=("auto", "rtx_sm120", "rtx_sm89"),
        help="Hardware route. Use auto on the target GPU; override for debugging.",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the red block and place it in the tray",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    images = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(args.num_views)
    ]

    print("=== Pi0.5 RTX full-FP16 opt-in benchmark ===", flush=True)
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    print("checkpoint", args.checkpoint, flush=True)
    print("num_views", args.num_views, "steps", args.steps, "hardware", args.hardware, flush=True)

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="pi05",
        hardware=args.hardware,
        num_views=args.num_views,
        num_steps=args.steps,
        cache_frames=1,
        use_fp8=False,
        use_fp16=True,
    )
    torch.cuda.synchronize()
    print(f"load_model_s={time.perf_counter() - t0:.3f}", flush=True)
    print("pipe", type(model._pipe).__name__, flush=True)

    t0 = time.perf_counter()
    out = model.predict(images, prompt=args.prompt)
    torch.cuda.synchronize()
    if not np.isfinite(out).all():
        raise RuntimeError("FP16 output contains NaN or Inf")
    print(
        f"first_predict_build_s={time.perf_counter() - t0:.3f} "
        f"actions_shape={out.shape}",
        flush=True,
    )

    for _ in range(args.warmup):
        out = model.predict(images)
        if not np.isfinite(out).all():
            raise RuntimeError("FP16 output contains NaN or Inf during warmup")
    torch.cuda.synchronize()
    model._pipe.latency_records.clear()

    wall = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = model.predict(images)
        torch.cuda.synchronize()
        if not np.isfinite(out).all():
            raise RuntimeError("FP16 output contains NaN or Inf during benchmark")
        wall.append((time.perf_counter() - t0) * 1000.0)

    print("RESULT wall_ms", _stats(wall), flush=True)
    print("RESULT internal_ms", _stats(list(model._pipe.latency_records)), flush=True)


if __name__ == "__main__":
    main()
