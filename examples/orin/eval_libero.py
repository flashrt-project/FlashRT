#!/usr/bin/env python3
"""FlashRT Orin — LIBERO closed-loop benchmark for Pi0.5 cache configs.

This follows the Thor LIBERO evaluation style: task-language prompt only,
no state in prompt, OffScreenRenderEnv, LIBERO initial states, and subprocess
isolation so each cache config/task owns one model instance.

Usage:
    MUJOCO_GL=egl python examples/orin/eval_libero.py \
        --checkpoint /root/models/pi05_libero_finetuned_v044 \
        --task_suite libero_spatial --quick
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

# MuJoCo rendering setup must happen before GL imports.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _patch_egl_cleanup() -> None:
    """Avoid EGL destructor issues on Jetson unified memory."""
    try:
        import robosuite.renderers.context.egl_context as egl

        egl.EGLGLContext.free = lambda self: None
        egl.EGLGLContext.__del__ = lambda self: None
    except Exception:
        pass
    try:
        import robosuite.utils.binding_utils as binding_utils

        binding_utils.MjRenderContext.__del__ = lambda self: None
    except Exception:
        pass


_patch_egl_cleanup()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_ENV_RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
CONFIGS = {
    "bf16_baseline": {"cache": 1, "adaptive": False},
    "bf16_cache2": {"cache": 2, "adaptive": False},
    "bf16_adaptive_cache2": {"cache": 2, "adaptive": True},
    "bf16_cache3": {"cache": 3, "adaptive": False},
    "bf16_adaptive_cache3": {"cache": 3, "adaptive": True},
    "bf16_cache4": {"cache": 4, "adaptive": False},
    "bf16_adaptive_cache4": {"cache": 4, "adaptive": True},
}


def resize_with_pad(img: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    result = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    result[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return result


def prepare_obs(obs: dict) -> dict:
    """Convert LIBERO obs to FlashRT Pi0.5 image inputs."""

    def _image(key: str) -> np.ndarray:
        img = obs[key]
        if img.ndim == 4:
            img = img[0]
        img = np.ascontiguousarray(img[::-1, ::-1])
        return resize_with_pad(img, 224, 224)

    image = _image("agentview_image")
    wrist = _image("robot0_eye_in_hand_image")
    return {"images": [image, wrist], "image": image, "wrist_image": wrist}


def pixel_delta_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def reset_episode_model_state(model, prompt: str) -> None:
    """Reset per-episode frontend/cache state without changing eval protocol."""
    for name in ("reset", "reset_cache", "reset_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            fn()
            return
    # VLAModel exposes set_prompt(), which delegates to the Pi0.5 frontend
    # and resets its temporal cache frame counter. This keeps trials isolated
    # even when consecutive trials use the same task prompt.
    fn = getattr(model, "set_prompt", None)
    if callable(fn):
        fn(prompt)


def run_episode(
    model,
    env,
    task_description: str,
    max_steps: int,
    *,
    replan_steps: int,
    use_adaptive: bool,
    threshold: float,
    obs: dict,
) -> tuple[bool, int, int, int, int, list[float], list[dict]]:
    """Run one closed-loop episode.

    Returns:
        success, executed_steps, forced_full_count, inference_count,
        used_full_count, inference_latencies_ms, inference_trace
    """
    action_plan = collections.deque()
    last_full_image = None
    last_full_wrist = None
    forced_count = 0
    infer_count = 0
    used_full_count = 0
    latencies_ms: list[float] = []
    trace: list[dict] = []
    reset_episode_model_state(model, task_description)

    for t in range(max_steps + 10):
        if t < 10:
            obs, _, _, _ = env.step(DUMMY_ACTION)
            continue

        if not action_plan:
            model_obs = prepare_obs(obs)
            force_full = False
            delta = None
            if use_adaptive and last_full_image is not None:
                delta = max(
                    pixel_delta_mae(model_obs["image"], last_full_image),
                    pixel_delta_mae(model_obs["wrist_image"], last_full_wrist),
                )
                force_full = delta >= threshold
                forced_count += int(force_full)

            if infer_count == 0:
                t0 = time.perf_counter()
                actions = model.predict(
                    images=model_obs["images"],
                    prompt=task_description,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)
                used_full = True
            else:
                t0 = time.perf_counter()
                result = model.infer(model_obs, force_full=force_full)
                if "used_full_pipeline" not in result:
                    raise RuntimeError(
                        "infer() result is missing used_full_pipeline: "
                        f"{list(result.keys())}"
                )
                actions = result["actions"]
                used_full = bool(result["used_full_pipeline"])
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(latency_ms)

            trace.append(
                {
                    "step": int(t - 9),
                    "infer_index": int(infer_count),
                    "delta": None if delta is None else float(delta),
                    "force_full": bool(force_full),
                    "used_full": bool(used_full),
                    "latency_ms": float(latency_ms),
                }
            )

            if used_full:
                used_full_count += 1
                last_full_image = model_obs["image"].copy()
                last_full_wrist = model_obs["wrist_image"].copy()

            action_plan.extend(actions[:replan_steps])
            infer_count += 1

        action = action_plan.popleft()
        if hasattr(action, "tolist"):
            action = action.tolist()
        obs, _, done, info = env.step(action)
        success = bool(done)
        if isinstance(info, dict):
            success = (
                success or
                bool(info.get("success", False)) or
                bool(info.get("is_success", False))
            )
        if success:
            return (
                True,
                t - 9,
                forced_count,
                infer_count,
                used_full_count,
                latencies_ms,
                trace,
            )

    return (
        False,
        max_steps,
        forced_count,
        infer_count,
        used_full_count,
        latencies_ms,
        trace,
    )


def summarize_latency(values: list[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "hz_p50": 0.0,
        }
    n = len(vals)
    p95_idx = min(n - 1, max(0, round((n - 1) * 0.95)))
    p50 = float(np.median(vals))
    return {
        "mean_ms": float(np.mean(vals)),
        "p50_ms": p50,
        "p95_ms": float(vals[p95_idx]),
        "min_ms": float(vals[0]),
        "max_ms": float(vals[-1]),
        "hz_p50": 1000.0 / p50 if p50 > 0 else 0.0,
    }


def eval_single_config_task(args: argparse.Namespace, config_name: str, task_id: int) -> dict:
    import flash_rt
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = CONFIGS[config_name]
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    max_steps = MAX_STEPS[args.task_suite]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    logger.info(
        "Config=%s cache=%d adaptive=%s task=%d: %s",
        config_name,
        cfg["cache"],
        cfg["adaptive"],
        task_id,
        task.language,
    )
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        num_views=2,
        cache_frames=cfg["cache"],
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)

    successes = 0
    episode_results = []
    forced_total = 0
    infer_total = 0
    used_full_total = 0
    all_latencies_ms: list[float] = []
    all_trace: list[dict] = []
    t_start = time.perf_counter()

    for trial in range(args.num_trials):
        env.reset()
        obs = env.set_init_state(initial_states[trial % len(initial_states)])
        success, steps, forced, infers, used_full, latencies_ms, trace = run_episode(
            model,
            env,
            task.language,
            max_steps,
            replan_steps=args.replan_steps,
            use_adaptive=cfg["adaptive"],
            threshold=args.threshold,
            obs=obs,
        )
        for event in trace:
            event.update(
                {
                    "config": config_name,
                    "task_id": int(task_id),
                    "trial": int(trial),
                    "success": bool(success),
                    "threshold": float(args.threshold),
                }
            )
        successes += int(success)
        forced_total += forced
        infer_total += infers
        used_full_total += used_full
        all_latencies_ms.extend(latencies_ms)
        all_trace.extend(trace)
        episode_results.append(
            {
                "trial": trial,
                "success": bool(success),
                "steps": int(steps),
                "forced_full": int(forced),
                "inferences": int(infers),
                "used_full": int(used_full),
                "latency": summarize_latency(latencies_ms),
            }
        )
        logger.info(
            "  %s task=%d trial=%d %s steps=%d forced_full=%d/%d "
            "used_full=%d/%d lat_p50=%.1fms",
            config_name,
            task_id,
            trial,
            "SUCCESS" if success else "FAIL",
            steps,
            forced,
            infers,
            used_full,
            infers,
            summarize_latency(latencies_ms)["p50_ms"],
        )

    env.close()
    elapsed = time.perf_counter() - t_start
    latency = summarize_latency(all_latencies_ms)
    return {
        "config": config_name,
        "task_id": task_id,
        "task_description": task.language,
        "cache_frames": cfg["cache"],
        "adaptive": cfg["adaptive"],
        "threshold": args.threshold if cfg["adaptive"] else None,
        "successes": successes,
        "num_trials": args.num_trials,
        "success_rate": successes / args.num_trials,
        "forced_full": forced_total,
        "inferences": infer_total,
        "used_full": used_full_total,
        "mean_forced_full_rate": forced_total / infer_total if infer_total else 0.0,
        "used_full_rate": used_full_total / infer_total if infer_total else 0.0,
        "latency": latency,
        "elapsed_sec": elapsed,
        "episodes": episode_results,
        "trace": all_trace if args.trace_jsonl else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashRT Orin LIBERO closed-loop cache benchmark"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task_suite", default="libero_spatial", choices=list(MAX_STEPS))
    parser.add_argument("--task-idx", type=int, nargs="+", default=None)
    parser.add_argument(
        "--configs",
        default="bf16_baseline,bf16_cache2,bf16_cache3,bf16_adaptive_cache3",
    )
    parser.add_argument("--num_trials", "--episodes", type=int, default=3)
    parser.add_argument("--replan_steps", "--replan-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick: 3 tasks x 3 trials")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="Optional path for per-inference delta/refresh trace JSONL.",
    )
    parser.add_argument("--_task_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_config", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_worker_result(result: dict) -> None:
    out_path = os.environ.get("_FLASHRT_ORIN_LIBERO_RESULT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)


def run_parent(args: argparse.Namespace) -> None:
    from libero.libero import benchmark

    requested = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in requested if name not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. Valid: {sorted(CONFIGS)}")

    if args.quick:
        task_ids = [0, 1, 2]
        args.num_trials = 3
    elif args.task_idx is not None:
        task_ids = args.task_idx
    else:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        task_ids = list(range(task_suite.n_tasks))

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"orin_libero_{args.task_suite}_cache_{stamp}.json"

    print("=" * 72)
    print("FlashRT Orin — LIBERO closed-loop cache benchmark")
    print(f"  Suite:      {args.task_suite}")
    print(f"  Tasks:      {task_ids}")
    print(f"  Configs:    {requested}")
    print(f"  Trials:     {args.num_trials}")
    print(f"  Replan:     {args.replan_steps}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 72)

    results = []
    for config_name in requested:
        for task_id in task_ids:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                out_path = tmp.name

            env = os.environ.copy()
            env["_FLASHRT_ORIN_LIBERO_RESULT"] = out_path
            cmd = [
                sys.executable,
                __file__,
                "--checkpoint",
                args.checkpoint,
                "--task_suite",
                args.task_suite,
                "--num_trials",
                str(args.num_trials),
                "--replan_steps",
                str(args.replan_steps),
                "--threshold",
                str(args.threshold),
                "--seed",
                str(args.seed),
                "--trace-jsonl",
                str(args.trace_jsonl or ""),
                "--_config",
                config_name,
                "--_task_id",
                str(task_id),
            ]
            logger.info("Launching config=%s task=%d", config_name, task_id)
            ret = subprocess.run(cmd, env=env, timeout=7200)
            if ret.returncode != 0:
                logger.error(
                    "config=%s task=%d failed with exit code %d",
                    config_name,
                    task_id,
                    ret.returncode,
                )
                continue
            try:
                with open(out_path) as f:
                    results.append(json.load(f))
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    totals = {}
    for result in results:
        totals.setdefault(result["config"], [0, 0, 0, 0, 0, []])
        totals[result["config"]][0] += result["successes"]
        totals[result["config"]][1] += result["num_trials"]
        totals[result["config"]][2] += result["forced_full"]
        totals[result["config"]][3] += result["inferences"]
        totals[result["config"]][4] += result.get("used_full", 0)
        for episode in result.get("episodes", []):
            # Retain only aggregate task-level latencies in the parent JSON;
            # per-inference samples can be added later if needed.
            pass
        totals[result["config"]][5].append(result["latency"])
        forced = ""
        if result["adaptive"]:
            forced = (
                f" forced_full={result['forced_full']}/"
                f"{result['inferences']}"
            )
        print(
            f"  {result['config']:16s} task={result['task_id']:2d} "
            f"{result['successes']:2d}/{result['num_trials']:2d} "
            f"= {result['success_rate']:.1%}{forced}  "
            f"lat_p50={result['latency']['p50_ms']:.1f}ms "
            f"({result['latency']['hz_p50']:.1f}Hz)  "
            f"{result['task_description']}"
        )

    print("-" * 72)
    summary = {}
    for config_name, (succ, trials, forced, infers, used_full, lat_parts) in totals.items():
        rate = succ / trials if trials else 0.0
        forced_rate = forced / infers if infers else 0.0
        used_full_rate = used_full / infers if infers else 0.0
        # Aggregate task-level latency summaries by mean. This is sufficient
        # for comparing closed-loop realized speed without storing every sample
        # from child processes in the parent JSON.
        lat_mean = float(np.mean([x["mean_ms"] for x in lat_parts])) if lat_parts else 0.0
        lat_p50 = float(np.mean([x["p50_ms"] for x in lat_parts])) if lat_parts else 0.0
        lat_p95 = float(np.mean([x["p95_ms"] for x in lat_parts])) if lat_parts else 0.0
        hz_p50 = 1000.0 / lat_p50 if lat_p50 > 0 else 0.0
        summary[config_name] = {
            "successes": succ,
            "num_trials": trials,
            "success_rate": rate,
            "forced_full": forced,
            "inferences": infers,
            "forced_full_rate": forced_rate,
            "used_full": used_full,
            "used_full_rate": used_full_rate,
            "latency_mean_ms": lat_mean,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "hz_p50": hz_p50,
        }
        extra = f" forced_full={forced}/{infers} ({forced_rate:.1%})" if forced else ""
        print(
            f"  {config_name:16s} overall {succ}/{trials} = {rate:.1%}{extra} "
            f"used_full={used_full}/{infers} ({used_full_rate:.1%}) "
            f"lat_p50={lat_p50:.1f}ms ({hz_p50:.1f}Hz)"
        )

    output = {
        "task_suite": args.task_suite,
        "checkpoint": args.checkpoint,
        "configs": requested,
        "task_ids": task_ids,
        "num_trials": args.num_trials,
        "replan_steps": args.replan_steps,
        "threshold": args.threshold,
        "seed": args.seed,
        "summary": summary,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    if args.trace_jsonl:
        with open(args.trace_jsonl, "w") as f:
            for result in results:
                for event in result.get("trace", []):
                    f.write(json.dumps(event) + "\n")
        logger.info("Per-inference trace saved to %s", args.trace_jsonl)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)


def main() -> None:
    args = parse_args()
    if args._config is not None or args._task_id is not None:
        if args._config is None or args._task_id is None:
            raise SystemExit("--_config and --_task_id must be provided together")
        result = eval_single_config_task(args, args._config, args._task_id)
        _write_worker_result(result)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
