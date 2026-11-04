"""Run one task of the two-camera LIBERO-10 evaluation, or capture calibration."""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["calibrate", "torch", "trt", "flashrt"])
parser.add_argument("task", nargs="?", type=int)
parser.add_argument("--output", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--calibration-dir", required=True)
parser.add_argument("--source-dir", required=True)
parser.add_argument("--engine-dir")
parser.add_argument("--seed", type=int, default=20261026)
parser.add_argument("--render-backend", choices=["egl", "osmesa"], default="egl")
args = parser.parse_args()
if args.stage != "calibrate" and (args.task is None or not 0 <= args.task < 10):
    parser.error("Scoring needs a task index from 0 to 9")
if args.stage == "trt" and not args.engine_dir:
    parser.error("TensorRT needs --engine-dir with two-camera engines")

import os
import sys

sys.path.insert(0, args.source_dir)
os.environ["MUJOCO_GL"] = args.render_backend
os.environ["PYOPENGL_PLATFORM"] = args.render_backend

from pathlib import Path
import gc, hashlib, importlib, importlib.metadata, json, shutil, sys, time, traceback
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.utils import get_libero_path
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
from groot_compare.backends.flashrt import FlashRTBackend, capture_calibration
from groot_compare.backends.jal import TorchBackend, TensorRTBackend
from groot_compare.closed_loop import (
    make_sim_policy,
    make_fixed_state_env,
    run_episode,
    case_seed,
    array_fingerprint,
)
from groot_compare.common import shared_noise

ROOT = Path(args.output)
ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = args.checkpoint
CALIBRATION = Path(args.calibration_dir)
SEED = args.seed
MAX_STEPS = 720
EPISODES = 50
stage = args.stage
torch.set_num_threads(2)
suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
assert suite.get_num_tasks() == 10


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "distribution metadata unavailable"))


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def states_for(task):
    p = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    states = torch.load(p, map_location="cpu", weights_only=False)
    assert len(states) == EPISODES
    return p, states


def check_space():
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Disk reserve below 15 GiB")


def raw_render_delta(observation, reference):
    if set(observation) != set(reference):
        raise ValueError("Observation groups differ")
    for group in ["state", "language"]:
        if set(observation[group]) != set(reference[group]):
            raise ValueError("Observation keys differ")
        for key in observation[group]:
            if group == "state":
                if not np.array_equal(observation[group][key], reference[group][key]):
                    raise ValueError("Robot state differs")
            elif observation[group][key] != reference[group][key]:
                raise ValueError("Instruction differs")
    if set(observation["video"]) != set(reference["video"]):
        raise ValueError("Camera keys differ")
    total = 0
    maximum = 0
    pixel_count = 0
    cameras = {}
    for key in observation["video"]:
        a = np.asarray(observation["video"][key])
        b = np.asarray(reference["video"][key])
        if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape:
            raise ValueError("Camera format differs")
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        n = int(np.count_nonzero(np.any(delta, axis=-1)))
        m = int(delta.max())
        pixel_count += int(a.size // 3)
        total += n
        maximum = max(maximum, m)
        cameras[key] = {"different_pixels": n, "max_uint8_difference": m}
    if maximum > 1 or total / pixel_count > 1e-4:
        raise ValueError(
            f"Render difference exceeds one-LSB/sparse-render bound: pixels={total}/{pixel_count},max={maximum}"
        )
    return {
        "different_pixels_total": total,
        "pixel_count_total": pixel_count,
        "different_pixel_fraction": total / pixel_count,
        "max_uint8_difference": maximum,
        "cameras": cameras,
    }


if stage == "calibrate":
    caps = CALIBRATION
    caps.mkdir(parents=True, exist_ok=False)
    records = []
    for task_id in range(10):
        check_space()
        task = suite.get_task(task_id)
        p, states = states_for(task)
        raw_path = p.with_name(p.name.replace(".pruned_init", ".init"))
        raw_states = torch.load(raw_path, map_location="cpu", weights_only=False)
        test_bytes = {np.asarray(s).tobytes() for s in states}
        index = next(
            i
            for i, s in enumerate(raw_states)
            if np.asarray(s).tobytes() not in test_bytes
        )
        initial = np.asarray(raw_states[index]).copy()
        bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        env = LiberoEnv(task_bddl_file=str(bddl), task_description=task.language)
        try:
            env._env.seed(case_seed(SEED, "libero_10", task.name, 100000 + index))
            env._env.reset()
            raw = env._env.set_init_state(initial)
            for _ in range(5):
                raw, _, _, _ = env._env.step(np.zeros(7))
            if env._env.check_success():
                raise RuntimeError("Calibration state already successful")
            flat = env._process_observation(raw)
            obs = {
                "video": {
                    k: np.ascontiguousarray(flat["video." + k][None, None])
                    for k in ["image", "wrist_image"]
                },
                "state": {
                    k: np.asarray(flat["state." + k], dtype=np.float32)[None, None]
                    for k in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                },
                "language": {
                    "annotation.human.action.task_description": [[task.language]]
                },
            }
        finally:
            env.close()
        print(f"CALIBRATING task={task_id} reserved_raw_state={index}", flush=True)
        capture_calibration(
            CHECKPOINT, obs, str(caps / f"task{task_id}.pt"), cpu_threads=2
        )
        records.append(
            {
                "task_id": task_id,
                "prompt": task.language,
                "raw_state_index": index,
                "state": array_fingerprint(initial),
                "disjoint_from_all_50_test_states": True,
                "test_state_file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
        write(caps / "records.json", {"completed": False, "records": records})
    write(caps / "records.json", {"completed": True, "records": records})
    print("CALIBRATION_COMPLETE", flush=True)
else:
    assert stage in ["torch", "trt", "flashrt"]
    task_id = args.task
    assert 0 <= task_id < 10
    out = ROOT / stage / f"task{task_id}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "backend": stage,
        "task_id": task_id,
        "completed": False,
        "target": 50,
        "seed": SEED,
        "cameras": ["image", "wrist_image"],
        "denoise_steps": 4,
        "execute_actions": 8,
        "settling_steps": 5,
        "max_steps": MAX_STEPS,
        "render_backend": args.render_backend,
        "runtime": {
            k: package_version(k)
            for k in ["torch", "numpy", "transformers", "robosuite", "mujoco"]
        },
        "episodes": [],
    }
    if (out / "results.json").exists():
        previous = json.loads((out / "results.json").read_text())
        assert (
            not previous["completed"]
            and previous["backend"] == stage
            and previous["task_id"] == task_id
        )
        for k in [
            "seed",
            "cameras",
            "denoise_steps",
            "execute_actions",
            "settling_steps",
            "max_steps",
        ]:
            assert previous[k] == report[k]
        indices = [e["state_index"] for e in previous["episodes"]]
        assert len(indices) == len(set(indices)) and all(0 <= i < 50 for i in indices)
        assert all((out / f"actions-state{i}.npz").exists() for i in indices)
        report = previous
        report.setdefault("resume_history", []).append(
            {"previous_error": report.pop("error", None), "time": time.time()}
        )
    existing_states = {e["state_index"] for e in report["episodes"]}
    write(out / "results.json", report)
    backend = None
    try:
        check_space()
        task = suite.get_task(task_id)
        p, states = states_for(task)
        report["prompt"] = task.language
        report["test_state_file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        cap = CALIBRATION / f"task{task_id}.pt"
        calibration = torch.load(cap, map_location="cpu", weights_only=False)
        if stage == "torch":
            backend = TorchBackend(
                checkpoint=CHECKPOINT, source_dir=args.source_dir, cpu_threads=2
            )
        elif stage == "trt":
            backend = TensorRTBackend(
                checkpoint=CHECKPOINT,
                engine_dir=args.engine_dir,
                source_dir=args.source_dir,
                cpu_threads=2,
            )
        else:
            backend = FlashRTBackend(
                checkpoint=CHECKPOINT, calibration_capture=str(cap), cpu_threads=2
            )
        report["backend_metadata"] = backend.metadata
        report["warmup_calls"] = 20
        noise = shared_noise("unscored-warmup", SEED)
        for _ in range(20):
            backend.get_action(calibration["inputs"], noise)
        backend.synchronize()
        policy, inner = make_sim_policy(
            backend, "libero_10", task.name, task.language, SEED
        )
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=8)
        assert contract.action_horizon == 16 and contract.video_delta_indices == (0,)
        bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        reference = {}
        if stage != "torch":
            ref = json.loads(
                (ROOT / "torch" / f"task{task_id}" / "results.json").read_text()
            )
            assert ref["completed"] and len(ref["episodes"]) == 50
            reference = {x["state_index"]: x for x in ref["episodes"]}
        # Keep numerical/state pairing strict; explicitly bound the observed EGL rounding difference.
        original_fingerprints = backend.input_fingerprints
        current_case = {"state_index": None}
        pairing_receipts = {}

        def checked_fingerprints(observation):
            actual = original_fingerprints(observation)
            idx = current_case["state_index"]
            torch.save(observation, out / f"initial-observation-state{idx}.pt")
            receipt = {"mode": "exact" if reference else "reference"}
            if reference and actual != reference[idx]["initial_processed_inputs"]:
                expected = reference[idx]["initial_processed_inputs"]
                different = [k for k in actual if actual[k] != expected[k]]
                write(
                    out / f"input-mismatch-state{idx}.json",
                    {
                        "state_index": idx,
                        "actual": actual,
                        "expected": expected,
                        "different_fields": different,
                    },
                )
                if different != ["pixel_values"]:
                    raise RuntimeError(f"Unpaired non-image input at state {idx}")
                raw_path = (
                    ROOT
                    / "torch"
                    / f"task{task_id}"
                    / f"initial-observation-state{idx}.pt"
                )
                reference_backend = "torch"
                if not raw_path.exists():
                    raw_path = (
                        ROOT
                        / "trt"
                        / f"task{task_id}"
                        / f"initial-observation-state{idx}.pt"
                    )
                    reference_backend = "trt"
                    trt_report = json.loads(
                        (raw_path.parent / "results.json").read_text()
                    )
                    trt_case = next(
                        e for e in trt_report["episodes"] if e["state_index"] == idx
                    )
                    assert (
                        trt_case["initial_processed_inputs"] == expected
                    ), "Fallback raw reference must match Torch preprocessing exactly"
                raw_reference = torch.load(
                    raw_path, map_location="cpu", weights_only=False
                )
                delta = raw_render_delta(observation, raw_reference)
                if delta["different_pixels_total"] == 0:
                    raise RuntimeError(
                        "Identical raw pixels produced different preprocessing; not a rendering difference"
                    )
                receipt = {
                    "mode": "render_rounding_one_lsb_sparse",
                    "raw_reference_backend": reference_backend,
                    **delta,
                    "actual_pixel_sha256": actual["pixel_values"]["sha256"],
                    "reference_pixel_sha256": expected["pixel_values"]["sha256"],
                }
                write(out / f"input-pairing-state{idx}.json", receipt)
            pairing_receipts[idx] = receipt
            return actual

        backend.input_fingerprints = checked_fingerprints
        started = time.time()
        report.setdefault("started_unix", started)
        for state_index in range(50):
            if state_index in existing_states:
                continue
            check_space()
            current_case["state_index"] = state_index
            env, base = make_fixed_state_env(
                task, np.asarray(states[state_index]).copy(), contract, bddl, MAX_STEPS
            )
            t = time.perf_counter()
            try:
                result = run_episode(
                    env,
                    policy,
                    state_index,
                    case_seed(SEED, "libero_10", task.name, state_index),
                    MAX_STEPS,
                )
                settled = base.settled_state_fingerprint
            finally:
                env.close()
            result.update(
                state_index=state_index,
                elapsed_seconds=time.perf_counter() - t,
                initial_state=array_fingerprint(states[state_index]),
                settled_state=settled,
                initial_processed_inputs=inner.initial_processed_inputs,
                decisions=inner.trace,
            )
            result["input_pairing"] = pairing_receipts[state_index]
            if reference:
                expected = reference[state_index]
                for key in ["initial_state", "settled_state"]:
                    assert (
                        result[key] == expected[key]
                    ), f"Unpaired {key} at state {state_index}"
                if (
                    result["initial_processed_inputs"]
                    != expected["initial_processed_inputs"]
                ):
                    assert (
                        result["input_pairing"]["mode"]
                        == "render_rounding_one_lsb_sparse"
                    )
                common = min(len(result["decisions"]), len(expected["decisions"]))
                assert all(
                    result["decisions"][i]["noise_sha256"]
                    == expected["decisions"][i]["noise_sha256"]
                    for i in range(common)
                )
            np.savez_compressed(
                out / f"actions-state{state_index}.npz",
                physical_actions=np.stack(inner.chunks),
            )
            report["episodes"].append(result)
            report["successes"] = sum(e["success"] for e in report["episodes"])
            report["updated_unix"] = time.time()
            write(out / "results.json", report)
            print(
                json.dumps(
                    {
                        "backend": stage,
                        "task": task_id,
                        "episode": state_index,
                        "success": result["success"],
                        "steps": result["environment_steps"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "completed": len(report["episodes"]),
                        "successes": report["successes"],
                    }
                ),
                flush=True,
            )
        report["completed"] = True
        report["finished_unix"] = time.time()
        write(out / "results.json", report)
    except BaseException as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        write(out / "results.json", report)
        raise
    finally:
        if backend is not None:
            backend.close()
        gc.collect()
        torch.cuda.empty_cache()
    print("TASK_COMPLETE", stage, task_id, flush=True)
