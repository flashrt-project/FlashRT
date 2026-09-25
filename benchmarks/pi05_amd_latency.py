"""Pi0.5 AMD latency benchmark on real LIBERO observation frames.

The measured boundary is the public frontend call: a real uint8 observation
and pinned denoising noise enter ``infer`` and a host action array returns.
Prompt setup, graph capture, and FP8 calibration happen before timing.

On multi-NUMA GPU hosts, bind this process to CPUs local to the selected GPU.
For example, the measured RunPod MI300X maps GPU 0 to node 1::

    taskset -c 48-63 python benchmarks/pi05_amd_latency.py \
      --ckpt /workspace/checkpoints/pi05_libero_pytorch \
      --data /workspace/data/libero_spatial_sample/episode0_real_samples.npz \
      --json-out /workspace/pi05-mi300x-fp8.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": statistics.mean(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True,
                    help="real LIBERO .npz with prompt/state/camera frames")
    ap.add_argument("--bf16", action="store_true",
                    help="measure the unquantized BF16 tier")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--noise-seed", type=int, default=1)
    ap.add_argument("--json-out")
    ap.add_argument("--actions-out",
                    help="save the final pinned-noise action array as .npy")
    args = ap.parse_args()

    import flash_rt

    data = np.load(args.data)
    prompt = str(data["prompt"][0])
    state = data["state"][0]
    observations = [
        {"image": data["image"][i], "wrist_image": data["wrist_image"][i]}
        for i in range(len(data["state"]))
    ]
    noise = np.random.RandomState(args.noise_seed).randn(10, 32).astype(np.float32)

    model = flash_rt.load_model(
        args.ckpt,
        config="pi05",
        framework="torch",
        hardware="auto",
        num_views=2,
        use_fp8=not args.bf16,
    )
    frontend = model.pipeline
    frontend.set_prompt(prompt, state=state)
    # This also prepares buffers and captures the graph for BF16; that tier
    # skips activation-statistics collection inside calibrate().
    frontend.calibrate(observations, percentile=99.9)

    for i in range(args.warmup):
        frontend.infer(observations[i % len(observations)], noise=noise)
    torch.cuda.synchronize()

    samples = []
    result = None
    for i in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = frontend.infer(observations[i % len(observations)], noise=noise)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    if args.bf16:
        precision = "BF16"
    elif frontend.pipeline.use_fp8_decoder:
        precision = "FP8 FNUZ"
    elif frontend.pipeline.cdna3_bf16_smallm:
        precision = "FP8 vision/encoder + packed BF16 decoder"
    else:
        precision = "FP8 vision/encoder + BF16 decoder"
    report = {
        "model": "pi0.5",
        "precision": precision,
        "device": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "data": str(Path(args.data).resolve()),
        "frames": len(observations),
        "prompt_tokens": frontend.current_prompt_len,
        "noise_seed": args.noise_seed,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "warmup": args.warmup,
        "iterations": args.iters,
        "boundary": "real uint8 observation + pinned noise -> host action array",
        "action_shape": list(result["actions"].shape),
        "latency": _summary(samples),
    }
    latency = report["latency"]
    print(
        f"{report['precision']}: min {latency['min_ms']:.2f} ms  "
        f"p50 {latency['p50_ms']:.2f} ms  p95 {latency['p95_ms']:.2f} ms  "
        f"max {latency['max_ms']:.2f} ms"
    )
    print(f"CPU affinity: {report['cpu_affinity']}")

    if args.actions_out:
        action_path = Path(args.actions_out)
        action_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(action_path, result["actions"])
        print(f"wrote {action_path}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
