"""Exercise the native Pi0.5 candidate through predict(), including host shape setup.

LIBERO uses two 256px cameras at 20 Hz. Replanning every five actions gives
the host 250 ms. Run soak and rollout as separate processes to isolate memory.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from flash_rt import load_model
from flash_rt.core.quant.calibrator import _checkpoint_hash
from flash_rt.models.pi05.native_plan_mps import NativePi05Plan

PROMPT = "pick up the black bowl and place it on the plate"
PERIOD = 0.25


def save(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(path)


def stats(values):
    return dict(
        zip(
            ("p50", "p95", "p99", "max"),
            np.percentile(values, [50, 95, 99, 100]).tolist(),
        )
    )


class Host:
    def __init__(self, args):
        self.model = load_model(
            args.checkpoint,
            hardware="apple_mps",
            framework="torch",
            config="pi05",
            num_views=2,
            action_dim=7,
            num_steps=10,
        )
        self.front = self.model.pipeline
        self.pipe = self.front.pipeline
        self.library = args.library
        self.amax = {}
        self.records = []

    def normalized(self, state):
        s = self.front.norm_stats["state"]
        lo, hi = (np.asarray(s[k])[: state.size] for k in ("q01", "q99"))
        return 2 * (state - lo) / (hi - lo + 1e-6) - 1

    def select(self, options, *, dynamic_prompt=True):
        # The experiment exposes NativePi05Plan directly. Keep one resident
        # plan and use the public predict path for image/noise/action handling.
        self.pipe.disable_native_pipeline()
        self.pipe._native_pipeline = NativePi05Plan(
            self.pipe,
            self.front._prompt_len,
            library=self.library,
            int8_options=options,
            dynamic_prompt=dynamic_prompt,
        )

    def calibrate(self, samples):
        for index, (images, state) in enumerate(samples[:4]):
            state = self.normalized(state)
            self.model.set_prompt(PROMPT, state=state)
            plan = self.pipe._native_pipeline
            if plan is None or plan.prompt_len != self.front._prompt_len:
                # Observe only real tokens; padded query activations are not
                # part of the calibration distribution.
                self.select({"stages": ("prefix",), "observe": True}, dynamic_prompt=False)
            noise = (
                np.random.default_rng(20261102 + index)
                .standard_normal((10, 32))
                .astype(np.float32)
            )
            self.front.infer({"images": images}, noise=noise)
            for k, v in self.pipe._native_pipeline.plan.activation_maxima().items():
                self.amax[k] = np.maximum(self.amax.get(k, v), v)
        self.pipe.disable_native_pipeline()
        if len(self.amax) != 70:
            raise RuntimeError(f"Expected 70 prefix sites, observed {len(self.amax)}")

    def predict(self, variant, task, images, raw_state, seed):
        torch.manual_seed(seed)
        start = time.perf_counter()
        state = self.normalized(raw_state)
        self.model.set_prompt(task, state=state)
        length = self.front._prompt_len
        plan = self.pipe._native_pipeline
        rebuild = plan is None or not plan.accepts_prompt_length(length)
        build_ms = 0.0
        if rebuild:
            t = time.perf_counter()
            options = (
                None
                if variant == "bf16"
                else {
                    "stages": ("vision", "prefix", "decode"),
                    "smooth_amax": self.amax,
                }
            )
            self.select(options)
            build_ms = (time.perf_counter() - t) * 1000
        actions = self.model.predict(images, prompt=task, state=state)
        ms = (time.perf_counter() - start) * 1000
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError("Invalid robot action chunk")
        self.records.append(
            {"ms": ms, "build_ms": build_ms, "rebuild": rebuild, "length": length,
             "shape_update_us": self.pipe._native_pipeline.last_shape_update_us}
        )
        return actions


def run_soak(host, samples, count, result, path):
    import psutil

    process = psutil.Process()
    # Cycle natural lengths throughout the soak, including state/image/noise
    # changes and periodic exact replay checks after intervening lengths.
    for i in range(10):
        host.predict("int8", PROMPT, *samples[4 + i % 4], i)
    reference = [
        host.predict("int8", PROMPT, *samples[4 + i], 6000 + i) for i in range(4)
    ]
    host.records.clear()
    memory = []
    started = time.perf_counter()
    for i in range(count):
        tick = time.perf_counter()
        probe = i % 200
        seed = 6000 + probe if probe < 4 else 1000 + i
        actions = host.predict("int8", PROMPT, *samples[4 + i % 4], seed)
        if probe < 4 and not np.array_equal(actions, reference[probe]):
            raise AssertionError("Repeated input/noise changed actions during soak")
        if i % 40 == 0 or i == count - 1:
            memory.append(
                {
                    "call": i,
                    "rss": int(process.memory_info().rss),
                    "torch_mps_driver": int(torch.mps.driver_allocated_memory()),
                }
            )
        time.sleep(max(0.0, PERIOD - (time.perf_counter() - tick)))
        if (i + 1) % 240 == 0:
            result.update(
                completed_calls=i + 1, memory=memory, steady=host.records.copy()
            )
            save(path, result)
            print("soak", i + 1, flush=True)
    steady = host.records.copy()
    result.update(
        duration_s=time.perf_counter() - started,
        memory=memory,
        steady=steady,
        wall_ms=stats([r["ms"] for r in steady]),
        misses=sum(r["ms"] > PERIOD * 1000 for r in steady),
    )
    save(path, result)
    host.records.clear()
    mixed_reference = []
    for i in range(24):
        actions = host.predict("int8", PROMPT, *samples[4 + i % 4], 3000 + i % 4)
        if i < 4:
            mixed_reference.append(actions.copy())
        elif not np.array_equal(actions, mixed_reference[i % 4]):
            raise AssertionError("Length switching changed fixed-input/fixed-noise actions")
    result["mixed"] = host.records.copy()
    result["mixed_wall_ms"] = stats([r["ms"] for r in host.records])
    result["mixed_misses"] = sum(r["ms"] > PERIOD * 1000 for r in host.records)
    result["mixed_repeat_checks"] = 20


def run_lengths(host, samples, result, path):
    """Compare one reused plan with fresh compact plans at both precisions."""
    inputs = []
    for index in range(4, 8):
        images, state = samples[index]
        host.model.set_prompt(PROMPT, state=host.normalized(state))
        images = (np.stack(images).astype(np.float32) / 127.5 - 1)
        prompt = host.front._prompt_cpu[:host.front._prompt_len].clone()
        noise = torch.from_numpy(np.random.default_rng(9000 + index).standard_normal(
            (10, 32)).astype(np.float32)).bfloat16()
        inputs.append((images, prompt, noise))
    # Boundary stress inputs are synthetic embeddings, separate from rollout
    # task-success evidence. Include the old M=522 benchmark and tile edges.
    images, prompt, noise = inputs[0]
    for length in (1, 10, 127, 128, 129, host.pipe.max_prompt_len - 1, host.pipe.max_prompt_len):
        embeds = prompt.repeat((length + len(prompt) - 1) // len(prompt), 1)[:length].clone()
        inputs.append((images, embeds, noise))
    result["length_validation"] = []
    for variant in ("bf16", "int8"):
        options = None if variant == "bf16" else {
            "stages": ("vision", "prefix", "decode"), "smooth_amax": host.amax,
        }
        references, compact_times = [], []
        for values in inputs:
            native = NativePi05Plan(host.pipe, len(values[1]), library=host.library,
                                   int8_options=options, dynamic_prompt=False)
            try:
                for _ in range(3):
                    reference, _ = native.forward(*values)
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    native.forward(*values)
                    times.append((time.perf_counter() - start) * 1000)
                references.append(reference)
                compact_times.append(stats(times))
            finally:
                native.close()
        native = NativePi05Plan(host.pipe, len(inputs[0][1]), library=host.library,
                               int8_options=options)
        try:
            identity = (native.plan.handle, native.plan.dispatch_count,
                        native.plan.weight_bytes, len(native.plan.uniforms))
            # Walk forward and backward with changed images, lengths and noise.
            for index in list(range(len(inputs))) + list(reversed(range(len(inputs)))):
                values = inputs[index]
                for warmup in range(3):
                    started = time.perf_counter()
                    actual, _ = native.forward(*values)
                    if warmup == 0:
                        first_ms = (time.perf_counter() - started) * 1000
                        shape_update_us = native.last_shape_update_us
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    repeated, _ = native.forward(*values)
                    times.append((time.perf_counter() - start) * 1000)
                    np.testing.assert_array_equal(actual, repeated)
                ref = references[index].ravel().astype(np.float64)
                got = actual.ravel().astype(np.float64)
                cosine = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
                nrmse = float(np.linalg.norm(ref - got) / np.linalg.norm(ref))
                exact = np.array_equal(actual, references[index])
                if not exact:
                    raise AssertionError(f"Compact dynamic output differs from the fixed plan: {variant}, {index}")
                dispatches = native.plan.resolved_dispatches()
                prefix_rows = sorted({d["params"][0] for d in dispatches
                                      if d["stage"] == "prefix" and
                                      ((d["kernel"].startswith(("gemm", "int8_gemm"))
                                        and d["params"][2] != 1152)
                                       or d["kernel"] == "tr_rms")})
                if prefix_rows != [512 + len(values[1])]:
                    raise AssertionError(f"Prefix still computes padding: {prefix_rows}")
                if not np.isfinite(got).all() or cosine < .999 or nrmse > .03:
                    raise AssertionError(f"Dynamic prompt precision failed: {variant}, {index}, {cosine}, {nrmse}")
                if identity != (native.plan.handle, native.plan.dispatch_count,
                                native.plan.weight_bytes, len(native.plan.uniforms)):
                    raise AssertionError("Length change rebuilt native resources")
                result["length_validation"].append({
                    "variant": variant, "sample": index, "length": len(values[1]),
                    "cosine": cosine, "nrmse": nrmse,
                    "exact": exact, "prefix_rows": prefix_rows,
                    "first_ms": first_ms, "shape_update_us": shape_update_us,
                    "max_abs": float(np.max(np.abs(ref - got))),
                    "compact_ms": compact_times[index], "dynamic_ms": stats(times),
                })
                save(path, result)
                print("length", variant, index, len(values[1]), cosine, nrmse, stats(times), flush=True)
        finally:
            native.close()


def run_rollouts(host, args, result, path):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        with torch.serialization.safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype("float64")),
            ]
        ):
            states = suite.get_task_init_states(task_id)
        for trial in args.trials:
            initial_hash = None
            variants = ("bf16", "int8") if trial % 2 else ("int8", "bf16")
            for variant in variants:
                host.pipe.disable_native_pipeline()
                host.records.clear()
                seed = 1000 * task_id + trial
                np.random.seed(seed)
                env = OffScreenRenderEnv(
                    bddl_file_name=str(
                        Path(get_libero_path("bddl_files"))
                        / task.problem_folder
                        / task.bddl_file
                    ),
                    camera_heights=256,
                    camera_widths=256,
                    control_freq=20,
                )
                try:
                    env.seed(seed)
                    env.reset()
                    env.set_init_state(states[trial])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    digest = hashlib.sha256()
                    for key in (
                        "agentview_image",
                        "robot0_eye_in_hand_image",
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ):
                        digest.update(np.ascontiguousarray(obs[key]).tobytes())
                    obs_hash = digest.hexdigest()
                    if initial_hash is not None and obs_hash != initial_hash:
                        raise RuntimeError("Paired initial observations differ")
                    initial_hash = obs_hash
                    success = False
                    steps = 0
                    while steps < 220 and not success:
                        images = [
                            np.asarray(
                                Image.fromarray(
                                    np.ascontiguousarray(obs[k][::-1, ::-1])
                                ).resize((224, 224), Image.Resampling.BILINEAR)
                            )
                            for k in ("agentview_image", "robot0_eye_in_hand_image")
                        ]
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"].copy()),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        chunk = host.predict(
                            variant, task.language, images, state, seed * 1000 + steps
                        )
                        for action in chunk[:5]:
                            obs, _, done, _ = env.step(action)
                            steps += 1
                            success = bool(env.check_success())
                            if done and not success:
                                raise RuntimeError(
                                    "Unexpected termination before task success"
                                )
                            if success:
                                break
                    row = {
                        "task_id": task_id,
                        "task": task.language,
                        "init": trial,
                        "variant": variant,
                        "initial_hash": obs_hash,
                        "steps": steps,
                        "success": success,
                        "calls": host.records.copy(),
                    }
                    result["episodes"].append(row)
                    print(variant, task_id, trial, steps, success, flush=True)
                    save(path, result)
                finally:
                    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("soak", "rollout", "lengths"))
    for key in ("checkpoint", "tokenizer", "library", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parents[1] / "thor/assets/pi05_libero.npz",
    )
    p.add_argument("--calls", type=int, default=1200)
    p.add_argument("--tasks", nargs="+", type=int, default=[6, 8, 9])
    p.add_argument("--trials", nargs="+", type=int, default=[17, 18])
    args = p.parse_args()
    if args.calls < 1:
        p.error("calls must be positive")
    if any(t < 0 or t >= 10 for t in args.tasks):
        p.error("libero_spatial task IDs must be in [0,10)")
    if any(t < 0 or t >= 50 for t in args.trials):
        p.error("LIBERO initial-state IDs must be in [0,50)")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.trials)) != len(
        args.trials
    ):
        p.error("task and initial-state IDs must be unique")
    if any(
        os.environ.get(k, "0") not in ("", "0")
        for k in (
            "MTL_DEBUG_LAYER",
            "MTL_SHADER_VALIDATION",
            "MTL_CAPTURE_ENABLED",
            "PYTORCH_MPS_TRACE_SIGNPOSTS",
        )
    ):
        p.error("Disable GPU validation and tracing for acceptance timing")
    os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = str(args.tokenizer.resolve())
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    result = {
        "status": "started",
        "mode": args.mode,
        "period_s": PERIOD,
        "episodes": [],
        "checkpoint": _checkpoint_hash(str(args.checkpoint)),
        "fixture": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "tokenizer": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "library": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "macos": platform.mac_ver()[0],
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "driver_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_calls": args.calls,
        "tasks": args.tasks,
        "initial_states": args.trials,
        "calibration_indices": [0, 1, 2, 3],
        "timing_scope": (
            "prepared CPU inputs to raw CPU actions; 3 warmups, 10 timed calls per case"
            if args.mode == "lengths" else
            "state normalization, prompt encoding, initial plan setup, public predict pixels/noise/native replay/action unnormalization"
        ),
    }
    host = None
    try:
        with np.load(args.input, allow_pickle=False) as data:
            samples = [
                ([data[f"img_{i}"], data[f"wrist_{i}"]], data[f"state_{i}"])
                for i in range(8)
            ]
        with torch.inference_mode():
            host = Host(args)
            host.calibrate(samples)
            result["channel_amax"] = {
                k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in host.amax.items()
            }
            save(path, result)
            if args.mode == "soak":
                run_soak(host, samples, args.calls, result, path)
            elif args.mode == "lengths":
                run_lengths(host, samples, result, path)
            else:
                run_rollouts(host, args, result, path)
            result["status"] = "measured"
    except BaseException as exc:
        result.update(status="error", error=repr(exc))
        raise
    finally:
        if host is not None:
            host.pipe.disable_native_pipeline()
        save(path, result)


if __name__ == "__main__":
    main()
