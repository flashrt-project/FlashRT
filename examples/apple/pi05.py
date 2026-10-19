"""Run a real Pi0.5 checkpoint on Apple MPS and save reproducible smoke/latency results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--input", type=Path, default=Path(__file__).parents[1] / "thor/assets/pi05_libero.npz")
    parser.add_argument("--output", type=Path, required=True, help="Directory for report.json and actions.npz")
    parser.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--cpu-reference", action="store_true", help="Compare one sample to the shared CPU BF16 pipeline using FP32 GEMM accumulation")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.iters < 1 or args.samples < 1:
        parser.error("steps, iters and samples must be positive; warmup must be nonnegative")
    if args.tokenizer:
        if not args.tokenizer.is_file():
            parser.error(f"Tokenizer not found: {args.tokenizer}")
        os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = str(args.tokenizer.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import flash_rt

    torch.manual_seed(args.seed)
    print(f"Loading {args.checkpoint} on MPS BF16 ({args.steps} denoising steps)", flush=True)
    started = time.perf_counter()
    model = flash_rt.load_model(
        args.checkpoint, hardware="apple_mps", config="pi05", framework="torch",
        num_views=2, num_steps=args.steps, action_dim=7, use_fp8=False, precision="bf16")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    frontend = model.pipeline
    print(f"Loaded in {load_seconds:.2f}s; MPS tensors {torch.mps.current_allocated_memory() / 2**30:.2f} GiB", flush=True)
    with np.load(args.input, allow_pickle=False) as data:
        observations = [dict(image=data[f"img_{index}"].copy(),
                             wrist_image=data[f"wrist_{index}"].copy(),
                             state=data[f"state_{index}"].copy())
                        for index in range(min(args.samples, int(data["n"])))]
    if not observations:
        raise ValueError("Input fixture is empty")
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal((frontend.chunk_size, 32)).astype(np.float32)

    def set_prompt(observation):
        state = observation["state"]
        stats = frontend.norm_stats["state"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)[:state.size]
        q99 = np.asarray(stats["q99"], dtype=np.float32)[:state.size]
        normalized = 2 * (state - q01) / (q99 - q01 + 1e-6) - 1
        model.set_prompt(args.prompt, state=normalized)

    def run(observation):
        result = model.infer(observation, noise=noise, debug=True)
        if result["actions"].shape != (frontend.chunk_size, 7):
            raise ValueError(f"Unexpected action shape: {result['actions'].shape}")
        if not np.isfinite(result["raw_actions"]).all() or not np.isfinite(result["actions"]).all():
            raise FloatingPointError("Non-finite model actions")
        return result

    set_prompt(observations[0])
    first = run(observations[0])
    print(f"First inference: {first['latency_ms']:.2f} ms, actions {first['actions'].shape}", flush=True)
    for _ in range(args.warmup):
        run(observations[0])
    latency = []
    repeated = []
    for index in range(args.iters):
        result = run(observations[0])
        latency.append(result["latency_ms"])
        repeated.append(result["raw_actions"].copy())
        print(f"Timed {index + 1}/{args.iters}: {latency[-1]:.2f} ms", flush=True)
    actions, raw_actions, prompt_lengths, end_to_end = [], [], [], []
    for index, observation in enumerate(observations):
        torch.mps.synchronize()
        started = time.perf_counter()
        set_prompt(observation)
        result = run(observation)
        end_to_end.append((time.perf_counter() - started) * 1000)
        actions.append(result["actions"].copy())
        raw_actions.append(result["raw_actions"].copy())
        prompt_lengths.append(frontend._prompt_len)
        print(f"Sample {index + 1}/{len(observations)}: {end_to_end[-1]:.2f} ms including state/prompt update", flush=True)
    repeat_max_abs = float(np.max(np.abs(np.stack(repeated) - repeated[0])))
    input_delta = float(np.max(np.abs(np.stack(raw_actions) - raw_actions[0])))
    if len(observations) > 1 and input_delta == 0:
        raise AssertionError("Different observations produced identical fixed-noise outputs")
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend="apple_mps", dtype="bfloat16", python=platform.python_version(),
        torch=torch.__version__, macos=platform.mac_ver()[0],
        chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        checkpoint=str(args.checkpoint.resolve()), input=str(args.input.resolve()),
        prompt=args.prompt, num_views=2, num_steps=args.steps, action_horizon=frontend.chunk_size,
        action_dim=7, shape=list(np.stack(actions).shape), seed=args.seed,
        prompt_lengths=prompt_lengths, load_seconds=load_seconds,
        first_inference_ms=first["latency_ms"], warmup=args.warmup, iterations=args.iters,
        latency_ms=dict(p50=float(np.median(latency)), p95=float(np.percentile(latency, 95)),
                        mean=float(np.mean(latency)), samples=latency),
        state_prompt_end_to_end_ms=end_to_end,
        repeat_max_abs=repeat_max_abs, different_observations_max_abs=input_delta,
        all_finite=True, mps_tensor_bytes=torch.mps.current_allocated_memory(),
        mps_driver_bytes=torch.mps.driver_allocated_memory(),
        qualification="Pipeline smoke test; not an independent OpenPI accuracy or LIBERO task-success evaluation")
    arrays = dict(actions=np.stack(actions), raw_actions=np.stack(raw_actions), noise=noise,
                  repeated_raw_actions=np.stack(repeated))
    manifest_path = args.checkpoint / "download-manifest.json"
    if manifest_path.is_file():
        report["checkpoint_source"] = json.loads(manifest_path.read_text())

    def save_results():
        np.savez(args.output / "actions.npz", **arrays)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Preserve the completed GPU run even if the optional CPU comparison fails.
    save_results()
    if args.cpu_reference:
        from flash_rt.hardware.torch_reference import TorchAttentionBackend, TorchGemmBackend, TorchTensorOps
        from flash_rt.models.pi05.torch_pipeline import Pi05TorchPipeline

        class CpuBf16Gemm(TorchGemmBackend):
            def linear(self, out, x, weight, bias=None):
                # On arm64 the native CPU BF16 GEMM can be a scalar fallback.
                # Use Accelerate's FP32 GEMM on exact BF16 operand values,
                # rounding back to BF16 before the separate bias operation.
                value = (x.float() @ weight.float()).to(x.dtype)
                if bias is not None:
                    value = value + bias
                out.copy_(value)

        set_prompt(observations[0])
        mps_raw = run(observations[0])["raw_actions"].copy()
        images = frontend._image_buf.cpu()
        prompt = frontend._prompt_buf[:frontend._prompt_len].cpu()
        print("Comparing one sample against the shared CPU BF16 pipeline...", flush=True)
        started = time.perf_counter()
        # Keep one copy of model weights: move them to CPU in place after all
        # GPU measurements. Both frontend and pipeline share this dictionary.
        for key in frontend.weights:
            frontend.weights[key] = frontend.weights[key].cpu()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        cpu = Pi05TorchPipeline(
            frontend.weights, TorchTensorOps(), CpuBf16Gemm(), TorchAttentionBackend(),
            num_views=2, max_prompt_len=frontend.max_prompt_len, chunk_size=frontend.chunk_size,
            num_steps=args.steps, device="cpu")
        cpu_raw = cpu.forward_with_inputs(
            images, prompt, frontend._prompt_len, torch.from_numpy(noise).bfloat16())[0].float().numpy()
        a, b = mps_raw.astype(np.float64).ravel(), cpu_raw.astype(np.float64).ravel()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        report["shared_cpu_reference"] = dict(
            cosine=float(np.dot(a, b) / denominator) if denominator else None,
            mean_abs=float(np.mean(np.abs(a - b))), max_abs=float(np.max(np.abs(a - b))),
            raw_action_dim=32, robot_action_dim=frontend.action_dim,
            seconds_including_transfer=time.perf_counter() - started,
            gemm_accumulation="float32, rounded to BF16 before bias",
            note="Same shared model, BF16 weights and activations; compares device/provider numerics only")
        # Report the robot channels separately: unchanged padded noise can
        # otherwise inflate cosine similarity over the full 32-D model output.
        from flash_rt.core.utils.actions import unnormalize_actions
        for prefix, left, right in (
            ("robot_raw", mps_raw[:, :frontend.action_dim], cpu_raw[:, :frontend.action_dim]),
            ("robot_action", unnormalize_actions(mps_raw, frontend.norm_stats)[:, :frontend.action_dim],
             unnormalize_actions(cpu_raw, frontend.norm_stats)[:, :frontend.action_dim]),
        ):
            left, right = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
            norm = np.linalg.norm(left) * np.linalg.norm(right)
            report["shared_cpu_reference"].update({
                f"{prefix}_cosine": float(left @ right / norm) if norm else None,
                f"{prefix}_mean_abs": float(np.abs(left - right).mean()),
                f"{prefix}_max_abs": float(np.abs(left - right).max()),
            })
        arrays.update(cpu_raw_actions=cpu_raw, mps_reference_raw_actions=mps_raw)
        print(json.dumps(report["shared_cpu_reference"]), flush=True)
    save_results()
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
