#!/usr/bin/env python3
"""
FlashRT — Quickstart.

Usage:
    # PyTorch:
    python examples/quickstart.py \
        --checkpoint /path/to/pi05_checkpoint

    # JAX:
    python examples/quickstart.py \
        --checkpoint /path/to/orbax_checkpoint \
        --framework jax

    # Benchmark:
    python examples/quickstart.py \
        --checkpoint /path/to/pi05_checkpoint \
        --benchmark 20

    # Thorough autotune (JAX, try harder):
    python examples/quickstart.py \
        --checkpoint /path/to/orbax_checkpoint \
        --framework jax --autotune 5

    # Skip autotune (fastest startup):
    python examples/quickstart.py \
        --checkpoint /path/to/pi05_checkpoint \
        --autotune 0
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import flash_rt


def main():
    parser = argparse.ArgumentParser(description="FlashRT quickstart")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--framework', default='torch', choices=['torch', 'jax'])
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--prompt', default='pick up the red block and place it in the tray')
    parser.add_argument('--config', default='pi05',
                        help="Model config: pi05, pi0, groot, pi0fast")
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=500,
                        help="Warmup iters before timed run. RTX 5090 needs "
                             "~500 to reach boost P-state; Thor can use 20.")
    parser.add_argument('--autotune', type=int, default=3,
                        help="Autotune trials: 0=off, 3=default, 5=thorough")
    parser.add_argument('--recalibrate', action='store_true',
                        help="Force fresh FP8 calibration (ignore cache)")
    parser.add_argument('--decode_cuda_graph', action='store_true',
                        help="Pi0-FAST: capture decode loop as CUDA Graph (max throughput)")
    parser.add_argument('--decode_graph_steps', type=int, default=80,
                        help="Pi0-FAST: action tokens to capture in decode graph")
    parser.add_argument('--max_steps', type=int, default=50,
                        help="Pi0-FAST: max decode steps for benchmark")
    parser.add_argument('--hardware', default='auto',
                        choices=['auto', 'thor', 'rtx_sm120', 'rtx_sm89', 'rtx_sm87'],
                        help="Backend selection; default auto-detects SM level")
    parser.add_argument('--embodiment_tag', default=None,
                        help="GROOT only. Trained slots in GR00T-N1.6-3B base: "
                             "gr1, robocasa_panda_omron, behavior_r1_pro. "
                             "Any other tag emits noise.")
    parser.add_argument('--action_horizon', type=int, default=None,
                        help="GROOT only. Number of action steps to generate "
                             "(default 50; pass 16 for LIBERO)")
    parser.add_argument('--use_fp4', action='store_true',
                        help="Pi0.5 torch only. Enable NVFP4 quantization "
                             "with the production preset: full 18 encoder "
                             "FFN layers + AWQ + P1 split-GU "
                             "(LIBERO Spatial 491/500 = 98.2%%, matches FP8 "
                             "baseline). Requires Thor SM110 / SM100+.")
    args = parser.parse_args()

    # ══════════════════════════════════════════
    #  3 lines
    # ══════════════════════════════════════════
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework=args.framework,
        num_views=args.num_views,
        autotune=args.autotune,
        recalibrate=args.recalibrate,
        config=args.config,
        decode_cuda_graph=args.decode_cuda_graph,
        decode_graph_steps=args.decode_graph_steps,
        max_decode_steps=args.max_steps,
        hardware=args.hardware,
        embodiment_tag=args.embodiment_tag,
        action_horizon=args.action_horizon,
        use_fp4=args.use_fp4,
    )

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Supply ``num_views`` images. The public API accepts up to 3 views
    # (image / wrist_image / wrist_image_right); sending the wrong count
    # silently drops frames and biases latency measurements.
    imgs = [img] * max(1, int(args.num_views))
    actions = model.predict(
        images=imgs,
        prompt=args.prompt,
    )
    # ══════════════════════════════════════════

    print(f"\nactions: shape={actions.shape}, sample={actions[0,:5].round(4)}")
    print(f"  range: [{actions.min():.4f}, {actions.max():.4f}]")
    ok = not np.isnan(actions).any() and len(actions.shape) == 2
    print(f"  sanity: {'PASS' if ok else 'FAIL'}")

    # Second call — reuses prompt (no recalibration, no graph recapture)
    actions2 = model.predict(images=imgs)
    print(f"  reuse prompt: shape={actions2.shape} OK")

    if args.benchmark > 0:
        # RTX 5090 takes ~500 replay iterations to climb from idle P8
        # (195 MHz) to boost P1 (~2870 MHz). Small warmup biases P50 by
        # 2-3 ms. Jetson Thor settles faster, 50 is plenty there.
        warmup = args.warmup
        for _ in range(warmup):
            model.predict(images=imgs)
        times = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            model.predict(images=imgs)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        p50 = times[len(times) // 2]
        print(f"\nBenchmark ({args.benchmark} iter, warmup={warmup}):")
        print(f"  P50: {p50:.1f} ms ({1000/p50:.0f} Hz)")
        print(f"  min: {min(times):.1f}, mean: {np.mean(times):.1f}, max: {max(times):.1f} ms")


if __name__ == '__main__':
    main()
