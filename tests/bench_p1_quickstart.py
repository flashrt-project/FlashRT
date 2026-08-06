"""Serial A/B bench of current AWQ vs P1+AWQ, quickstart-style.

Runs each config N times, interleaved, with Thor cool between. Each
iteration: cold build → warmup 50 → bench 100 → report P50. Collects
all trial P50s across interleaved runs to check if the 41ms LIBERO
numbers were Thor regime state vs real parity.

Usage:
    PYTHONPATH=<repo_root> \
      python3 tests/bench_p1_quickstart.py
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch


def run_once(build_fn, label, warmup=50, bench=100):
    t0 = time.perf_counter()
    pipe = build_fn()
    build_s = time.perf_counter() - t0
    pipe.set_prompt("pick up the red block and place it in the tray")

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": img}

    for _ in range(warmup):
        pipe.infer(obs)
    times = []
    for _ in range(bench):
        t0 = time.perf_counter()
        pipe.infer(obs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times)//2]
    p25 = times[int(len(times)*0.25)]
    p75 = times[int(len(times)*0.75)]
    p90 = times[int(len(times)*0.9)]
    print(f"[{label}] build={build_s:.1f}s  "
          f"P25={p25:.2f}  P50={p50:.2f}  P75={p75:.2f}  P90={p90:.2f}  ms")
    del pipe
    torch.cuda.empty_cache()
    return p50


def cool(s):
    print(f"  cool {s}s..."); sys.stdout.flush(); time.sleep(s)


def main():
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
    CKPT = os.environ.get("FLASH_RT_PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted")
    LAYERS = tuple(range(17))

    def build_awq():
        return Pi05TorchFrontendThorFP4(
            CKPT, num_views=2, autotune=3,
            use_fp4_encoder_ffn=True, fp4_layers=LAYERS,
            use_awq=True, awq_alpha=0.5, use_p1_split_gu=False)

    def build_p1():
        return Pi05TorchFrontendThorFP4(
            CKPT, num_views=2, autotune=3,
            use_fp4_encoder_ffn=True, fp4_layers=LAYERS,
            use_awq=True, awq_alpha=0.5, use_p1_split_gu=True)

    N_TRIALS = 5
    awq_p50s, p1_p50s = [], []

    for trial in range(N_TRIALS):
        print(f"\n=== Trial {trial+1}/{N_TRIALS} ===")
        awq_p50s.append(run_once(build_awq, f"AWQ/t{trial+1}"))
        cool(60)
        p1_p50s.append(run_once(build_p1, f"P1 /t{trial+1}"))
        if trial < N_TRIALS - 1:
            cool(60)

    awq_sorted = sorted(awq_p50s); p1_sorted = sorted(p1_p50s)
    awq_med = awq_sorted[len(awq_sorted)//2]
    p1_med  = p1_sorted[len(p1_sorted)//2]

    print()
    print("=" * 60)
    print(f"AWQ (current) P50s: {[f'{x:.2f}' for x in awq_p50s]}")
    print(f"              median = {awq_med:.2f} ms  "
          f"(min {min(awq_p50s):.2f}, max {max(awq_p50s):.2f})")
    print(f"P1+AWQ        P50s: {[f'{x:.2f}' for x in p1_p50s]}")
    print(f"              median = {p1_med:.2f} ms  "
          f"(min {min(p1_p50s):.2f}, max {max(p1_p50s):.2f})")
    print(f"Median Δ (P1 - AWQ): {p1_med - awq_med:+.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
