#!/usr/bin/env python3
"""FlashRT — Pi0.5 Jetson AGX Orin benchmark.

Measures steady-state inference latency for Pi0.5 on Orin (SM87) across
configurable num_views / vision_pool_factor / vision_num_layers / num_steps
/ cache_frames.

Usage (lossless cache_frames=1, 2-cam, 8.0 Hz target):
    FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
        --checkpoint /path/to/pi05_droid_pytorch \
        --num-views 2 --pool 1 --layers 27 --steps 10 \
        --cache-frames 1

Usage (production cache_frames=2, 2-cam, 12.2 Hz target, cos 0.991):
    FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
        --checkpoint /path/to/pi05_droid_pytorch \
        --num-views 2 --pool 1 --layers 27 --steps 10 \
        --cache-frames 2

Quick presets:
    --preset lossless    → 2cam pool=1 steps=10 cache=1  (~124ms / 8.0 Hz, bit-equal)
    --preset production  → 2cam pool=1 steps=10 cache=2  (~127/39ms / 12.2 Hz, cos 0.991)
    --preset balanced    → 2cam pool=1 steps=5  cache=1  (~107ms / 9.3 Hz)
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np


PRESETS = {
    "lossless":   dict(num_views=2, pool=1, layers=27, steps=10, cache=1),
    "production": dict(num_views=2, pool=1, layers=27, steps=10, cache=2),
    "balanced":   dict(num_views=2, pool=1, layers=27, steps=5,  cache=1),
}


def parse_args():
    p = argparse.ArgumentParser(description="FlashRT Pi0.5 Orin benchmark")
    p.add_argument("--checkpoint", "-c", required=True,
                   help="Path to pi05_droid_pytorch checkpoint")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Quick config preset (overrides individual flags)")
    p.add_argument("--num-views", type=int, default=2, choices=[1, 2])
    p.add_argument("--pool", type=int, default=1, choices=[1, 2, 4],
                   help="vision_pool_factor (1=no pool, 2=2x2, 4=4x4)")
    p.add_argument("--layers", type=int, default=27,
                   help="SigLIP layers to run (1-27, default=27 for lossless)")
    p.add_argument("--steps", type=int, default=10,
                   help="Diffusion ODE steps (default=10 for best quality)")
    p.add_argument("--cache-frames", type=int, default=1, choices=[1, 2],
                   help="1=bit-equal lossless, 2=temporal K/V reuse (cos 0.991)")
    p.add_argument("--prompt", default="pick up the black envelope on the table")
    p.add_argument("--warmup", type=int, default=8,
                   help="Warmup iterations before measurement")
    p.add_argument("--reps", type=int, default=15,
                   help="Measurement iterations")
    p.add_argument("--int8", action="store_true", default=True,
                   help="Enable INT8 fast path (sets FVK_PI05_RTX_FORCE_INT8=1, default ON)")
    p.add_argument("--no-int8", dest="int8", action="store_false",
                   help="Run BF16 reference path instead of INT8 fast path")
    return p.parse_args()


def main():
    args = parse_args()

    if args.preset:
        cfg = PRESETS[args.preset]
        args.num_views    = cfg["num_views"]
        args.pool         = cfg["pool"]
        args.layers       = cfg["layers"]
        args.steps        = cfg["steps"]
        args.cache_frames = cfg["cache"]
        print(f"Preset '{args.preset}': "
              f"num_views={args.num_views} pool={args.pool} "
              f"layers={args.layers} steps={args.steps} "
              f"cache_frames={args.cache_frames}")

    if args.int8 and not os.environ.get("FVK_PI05_RTX_FORCE_INT8"):
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
        print("Auto-set FVK_PI05_RTX_FORCE_INT8=1")

    import logging
    logging.basicConfig(level=logging.WARNING)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)

    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    print(f"\nLoading checkpoint: {args.checkpoint}")
    pipe = Pi05TorchFrontendRtx(
        args.checkpoint,
        num_views=args.num_views,
        num_steps=args.steps,
        vision_pool_factor=args.pool,
        vision_num_layers=args.layers,
        cache_frames=args.cache_frames,
    )

    pipe.set_prompt(args.prompt)

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": img}

    print("Calibrating…")
    pipe.calibrate_with_real_data([obs])

    print(f"Warmup ({args.warmup} iters)…")
    for _ in range(args.warmup):
        pipe.infer(obs)

    print(f"Measuring ({args.reps} iters)…")
    lat = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = pipe.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000)

    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95)]

    print()
    print("=" * 55)
    print(f"  Config : num_views={args.num_views}  pool={args.pool}"
          f"  layers={args.layers}  steps={args.steps}"
          f"  cache_frames={args.cache_frames}")
    print(f"  Actions: {out['actions'].shape}")
    print(f"  p50    : {p50:.1f} ms   → {1000/p50:.2f} Hz")
    print(f"  p95    : {p95:.1f} ms")
    print(f"  min    : {lat[0]:.1f} ms   → {1000/lat[0]:.2f} Hz")
    print("=" * 55)


if __name__ == "__main__":
    main()
