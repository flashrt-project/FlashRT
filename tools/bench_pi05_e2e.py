"""Synchronized Pi0.5 observation-to-action benchmark (no partial graph timing)."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True,
                        help="NPZ with n, img_i, wrist_i and state_i arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("default", "skinny", "nvfp4"), default="default")
    parser.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-state-prompt", action="store_true",
                        help="Measure the fixed-text-only input contract explicitly")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt import flash_rt_kernels

    with np.load(args.observations, allow_pickle=False) as data:
        observations = [{"image": data[f"img_{i}"].copy(),
                         "wrist_image": data[f"wrist_{i}"].copy(),
                         "state": data[f"state_{i}"].copy()}
                        for i in range(int(data["n"]))]
    if not observations:
        parser.error("observation fixture is empty")
    options = {"num_views": 2}
    if args.profile != "default":
        options["decoder_kernel"] = "skinny"
    if args.profile == "nvfp4":
        options["prefix_precision"] = "nvfp4"
    torch.manual_seed(args.seed)
    model = Pi05TorchFrontendRtx(args.checkpoint, **options)
    def set_observation_prompt(observation):
        if args.no_state_prompt:
            return
        state = np.asarray(observation["state"], dtype=np.float32)
        stats = model.norm_stats["state"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)[:state.size]
        q99 = np.asarray(stats["q99"], dtype=np.float32)[:state.size]
        normalized = (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        model.set_prompt(args.prompt, state=normalized)

    if args.no_state_prompt:
        model.set_prompt(args.prompt)
        model.calibrate(observations)
    else:
        # Pre-capture every fixture length before steady-state timing.
        for observation in observations:
            set_observation_prompt(observation)
            if not model.calibrated:
                model.calibrate(observations)
    for i in range(args.warmup):
        observation = observations[i % len(observations)]
        set_observation_prompt(observation)
        model.infer(observation)
    torch.cuda.synchronize()

    # Seeding is outside timing. Default inference still generates its own
    # noise; no debug export or injected input changes the measured path.
    torch.manual_seed(args.seed + 1)
    samples, outputs = [], []
    for i in range(args.iterations):
        observation = observations[i % len(observations)]
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        set_observation_prompt(observation)
        result = model.infer(observation)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        outputs.append(result["actions"].copy())
    actions = np.stack(outputs)
    if not np.isfinite(actions).all():
        raise RuntimeError("non-finite final actions")
    report = {
        "metric": "synchronized observation-to-final-action wall-clock E2E",
        "profile": args.profile, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "views": 2, "batch": 1,
        "denoise_steps": model._num_steps, "chunk_size": model.chunk_size,
        "state_in_prompt": not args.no_state_prompt,
        "warmup": args.warmup, "iterations": args.iterations, "seed": args.seed,
        "fixture_sha256": hashlib.sha256(Path(args.observations).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(Path(flash_rt_kernels.__file__).read_bytes()).hexdigest(),
        "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)), "min_ms": float(np.min(samples)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "finite": True,
        "boundary": "CPU observation -> per-frame state normalization/tokenization/embedding (unless explicitly disabled) -> image normalization/H2D -> vision -> prefix -> all denoise steps -> action unnormalization/D2H; one-time capture/calibration excluded",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output.with_suffix(".npz"), actions=actions, e2e_ms=samples)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
