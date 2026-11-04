"""Paired, batch-one LIBERO rollouts using the pinned official environment wrappers.

Requires simulator dependencies in the selected backend runtime. This module does
not install them. It measures task success; its elapsed times are not latency
benchmarks. The default task subset is the exact set of evaluation prompts in the
shared manifest, with separate task-specific FlashRT calibration captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .common import file_sha256, shared_noise, validate_manifest, write_json
from .dataset import _content_digest, _read_modality_configs
from .metrics import ACTION_KEYS, action_array
from .run import _backend, prompt_key

ISAAC_REVISION = "9c7e746b2cd37a810070a98ef41d290a07e806c2"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXECUTION_HORIZON = 8
SETTLING_STEPS = 5


def task_camera_keys(checkpoint, allow_camera_ablation=False):
    """Task validation preserves the checkpoint's two-camera observation by default."""
    video = _read_modality_configs(checkpoint, "libero_sim")["video"]
    keys = tuple(video["modality_keys"])
    if keys not in (("image",), ("image", "wrist_image")) or video["delta_indices"] != [0]:
        raise ValueError("Expected LIBERO image or image/wrist_image cameras, frame [0]")
    if keys == ("image",) and not allow_camera_ablation:
        raise ValueError("Task validation requires image and wrist_image. Use the two-camera "
                         "checkpoint, or explicitly select --allow-camera-ablation for a "
                         "one-camera diagnostic.")
    return keys


def array_fingerprint(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError("Initial simulator state must be a finite numeric array")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def case_seed(seed, suite_name, task_name, state_index):
    key = f"{seed}:{suite_name}:{task_name}:initial-state-{state_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def noise_record_id(suite_name, task_name, state_index, decision_index):
    return f"closed-loop:{suite_name}:{task_name}:state-{state_index}:decision-{decision_index}"


def _batch_sim_observation(observation):
    """Add only the vector-env batch axis; physical transforms stay upstream."""
    result = {}
    for key, value in observation.items():
        if key.startswith("video."):
            array = np.asarray(value)
            if array.dtype != np.uint8 or array.ndim != 4:
                raise ValueError(f"Unexpected official video observation: {key}")
            result[key] = np.ascontiguousarray(array[None])
        elif key.startswith("state."):
            # SyncVectorEnv's float32 Box buffer performs this same dtype cast.
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError(f"Unexpected official state observation: {key}")
            result[key] = np.ascontiguousarray(array[None])
        elif isinstance(value, str):
            result[key] = [value]
        else:
            raise ValueError(f"Unsupported official simulation observation: {key}")
    return result


def _unbatch_sim_actions(actions):
    expected = {f"action.{key}" for key in ACTION_KEYS}
    if set(actions) != expected:
        raise ValueError("Simulator actions must have all seven official physical-action keys")
    result = {}
    for key, value in actions.items():
        array = np.asarray(value)
        if array.shape != (1, 16, 1) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Expected float32 actions [1,16,1]: {key}")
        result[key] = array[0]
    return result


def select_tasks(suite, manifest, task_ids=None, full_suite=False):
    evaluation = {r["prompt"] for r in manifest["records"] if r["split"] == "evaluation"}
    calibration = {r["prompt"] for r in manifest["records"] if r["split"] == "calibration"}
    all_tasks = [(i, suite.get_task(i)) for i in range(suite.get_num_tasks())]
    if full_suite:
        selected = all_tasks
    elif task_ids is not None:
        if len(set(task_ids)) != len(task_ids) or any(i < 0 or i >= len(all_tasks) for i in task_ids):
            raise ValueError("Task IDs must be unique valid LIBERO task indices")
        selected = [all_tasks[i] for i in task_ids]
    else:
        selected = [(i, task) for i, task in all_tasks if task.language in evaluation]
        missing = evaluation - {task.language for _, task in selected}
        if missing:
            raise ValueError(f"Manifest prompts do not match this LIBERO suite exactly: {sorted(missing)}")
    if not selected:
        raise ValueError("No LIBERO tasks selected")
    missing = {task.language for _, task in selected} - calibration
    if missing:
        raise ValueError(f"Missing separate task calibration in shared manifest: {sorted(missing)}")
    return selected


def calibration_path(capture_dir, manifest_path, manifest, prompt):
    if not capture_dir:
        raise ValueError("FlashRT closed-loop evaluation requires --capture-dir")
    capture = Path(capture_dir) / f"{prompt_key(prompt)}.pt"
    meta = json.loads(capture.with_suffix(".json").read_text())
    calibration = {r["id"]: r for r in manifest["records"] if r["split"] == "calibration"}
    record = calibration.get(meta.get("calibration_record_id"))
    if (record is None or record["prompt"] != prompt or meta.get("prompt") != prompt
            or meta.get("manifest_sha256") != file_sha256(manifest_path)
            or meta.get("capture_sha256") != file_sha256(capture)
            or meta.get("calibration_observation_sha256") != file_sha256(
                manifest_path.parent / record["observation"])):
        raise ValueError("FlashRT capture is not bound to this manifest/task calibration")
    return capture


def make_sim_policy(backend, suite_name, task_name, prompt, seed):
    """Adapt our backend to the official BasePolicy and sim wrapper interfaces."""
    from gr00t.policy.policy import BasePolicy
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    reference = getattr(backend, "policy", None) or getattr(backend, "_prep", None)
    if reference is None:
        raise TypeError("Backend does not expose its official processor modality configuration")

    class BackendPolicy(BasePolicy):
        check_observation = Gr00tPolicy.check_observation
        check_action = Gr00tPolicy.check_action

        def __init__(self):
            super().__init__(strict=True)
            self.modality_configs = reference.modality_configs
            self.state_index = None
            self.trace, self.chunks = [], []
            self.initial_processed_inputs = None

        def get_modality_config(self):
            return self.modality_configs

        def reset(self, options=None):
            if not options or "state_index" not in options:
                raise ValueError("Reset requires an explicit fixed initial-state index")
            self.state_index = int(options["state_index"])
            self.trace, self.chunks = [], []
            self.initial_processed_inputs = None
            return {}

        def _get_action(self, observation, options=None):
            if self.state_index is None:
                raise RuntimeError("Policy must be reset before each episode")
            actual_prompt = next(iter(observation["language"].values()))[0][0]
            if actual_prompt != prompt:
                raise ValueError("Simulator instruction differs from task calibration prompt")
            decision = len(self.trace)
            key = noise_record_id(suite_name, task_name, self.state_index, decision)
            noise = shared_noise(key, seed, shape=backend.noise_shape)
            if not self.trace:
                self.initial_processed_inputs = backend.input_fingerprints(observation)
            result = backend.get_action(observation, noise)
            backend.synchronize()
            self.chunks.append(action_array(result).copy())
            self.trace.append({"decision_index": decision, "noise_record_id": key,
                               "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                               "observation_sha256": _content_digest(observation)})
            return result, {}

    inner = BackendPolicy()
    return Gr00tSimPolicyWrapper(inner, strict=True), inner


def make_fixed_state_env(task, initial_state, contract, bddl_path, max_steps):
    """Install the fixed state before MultiStepWrapper fills observation history."""
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    state = np.asarray(initial_state).copy()

    class FixedStateLiberoEnv(LiberoEnv):
        def reset(self, seed=None, options=None):
            if seed is not None:
                self._env.seed(int(seed))
            self._env.reset()
            observation = self._env.set_init_state(state.copy())
            # Exact settling sequence from pinned LIBERO lifelong/metric.py.
            for _ in range(SETTLING_STEPS):
                observation, _, _, _ = self._env.step(np.zeros(7))
            self.settled_state_fingerprint = array_fingerprint(self._env.get_sim_state())
            return self._process_observation(observation), {"success": self._env.check_success()}

    base = FixedStateLiberoEnv(task_bddl_file=str(bddl_path), task_description=task.language)
    try:
        wrapped = MultiStepWrapper(base, contract=contract, max_episode_steps=max_steps,
                                   terminate_on_success=True)
    except Exception:
        base.close()
        raise
    return wrapped, base


def run_episode(env, policy, state_index, reset_seed, max_steps):
    observation, reset_info = env.reset(seed=reset_seed)
    if bool(np.any(reset_info.get("success", False))):
        raise RuntimeError("Task is already successful after initialization; refusing trivial episode")
    policy.reset({"state_index": state_index})
    steps, decisions, success = 0, 0, False
    macro_steps = []
    while steps < max_steps:
        actions, _ = policy.get_action(_batch_sim_observation(observation))
        observation, reward, terminated, truncated, info = env.step(_unbatch_sim_actions(actions))
        advanced = int(info.get("n_env_steps", 0))
        if not 1 <= advanced <= EXECUTION_HORIZON or steps + advanced > max_steps:
            raise RuntimeError("Official wrapper returned an invalid environment-step count")
        steps += advanced
        decisions += 1
        success = success or bool(np.any(info.get("success", False)))
        macro_steps.append({"decision_index": decisions - 1, "n_env_steps": advanced,
                            "success": success, "reward": float(reward)})
        if success or terminated or truncated:
            return {"success": success, "environment_steps": steps, "policy_calls": decisions,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "reached_step_budget": steps == max_steps, "macro_steps": macro_steps}
    raise RuntimeError("Episode reached the budget without official wrapper termination")


def _verify_sources(source_dir, libero_dir):
    for directory, expected in ((source_dir, ISAAC_REVISION), (libero_dir, LIBERO_REVISION)):
        actual = subprocess.check_output(["git", "-c", f"safe.directory={directory}", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"Source revision differs at {directory}: {actual} != {expected}")
    sys.path.insert(0, str(libero_dir))
    sys.path.insert(0, str(source_dir))


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = validate_manifest(manifest_path)
    camera_keys = task_camera_keys(args.checkpoint, args.allow_camera_ablation)
    source_dir = Path(args.source_dir).resolve()
    libero_dir = Path(args.libero_dir or source_dir / "external_dependencies" / "LIBERO").resolve()
    _verify_sources(source_dir, libero_dir)
    os.environ["MUJOCO_GL"] = args.render_backend
    os.environ["PYOPENGL_PLATFORM"] = args.render_backend
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    import torch

    imported = Path(benchmark.__file__).resolve()
    if libero_dir not in imported.parents:
        raise RuntimeError(f"A different LIBERO package was imported: {imported}")
    suite = benchmark.get_benchmark_dict()[args.suite](task_order_index=0)
    tasks = select_tasks(suite, manifest, args.task_ids, args.full_suite)
    captures = {}
    if args.backend == "flashrt":
        for task_id, task in tasks:
            captures[task_id] = calibration_path(args.capture_dir, manifest_path, manifest, task.language)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"format_version": 1, "backend": args.backend, "completed": False,
              "suite": args.suite, "scope": "full_suite" if len(tasks) == suite.get_num_tasks() else "task_subset",
              "isaac_revision": ISAAC_REVISION, "libero_revision": LIBERO_REVISION,
              "wrapper_sha256": {name: file_sha256(source_dir / name) for name in (
                  "gr00t/eval/sim/LIBERO/libero_env.py",
                  "gr00t/eval/sim/wrapper/multistep_wrapper.py",
                  "gr00t/policy/gr00t_policy.py", "gr00t/eval/_horizon_contract.py")},
              "manifest_sha256": file_sha256(manifest_path),
              "protocol_sha256": file_sha256(Path(__file__).resolve().parents[1] / "configs/protocol.json"),
              "weight_manifest_sha256": file_sha256(Path(__file__).resolve().parents[1] / "artifacts/weight-hashes.json"),
              "checkpoint_config_sha256": file_sha256(Path(args.checkpoint) / "config.json"),
              "processor_config_sha256": file_sha256(Path(args.checkpoint) / "processor_config.json"),
              "initialization": "LIBERO fixed .pruned_init; 5 raw zero-action settling steps",
              "settling_steps_excluded_from_budget": SETTLING_STEPS,
              "execution_horizon": EXECUTION_HORIZON, "max_episode_steps": args.max_episode_steps,
              "seed": args.seed, "initial_state_start": args.state_start,
              "episodes_per_task": args.episodes, "batch_size": 1,
              "camera_keys": list(camera_keys), "camera_ablation": camera_keys == ("image",),
              "render_backend": os.environ["MUJOCO_GL"], "latency_benchmark": False,
              "selected_tasks": [{"task_id": i, "name": t.name, "prompt": t.language} for i, t in tasks],
              "groups": [], "episodes": []}
    write_json(output / "closed_loop.json", report)
    try:
        for task_id, task in tasks:
            state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            # Pinned LIBERO stores trusted benchmark states using legacy NumPy pickle.
            # Its API calls torch.load without a weights_only argument, incompatible
            # with newer torch defaults, so make that compatibility choice explicit.
            init_states = torch.load(state_path, map_location="cpu", weights_only=False)
            if args.state_start + args.episodes > len(init_states):
                raise ValueError(f"Not enough unique fixed initial states for task {task.name}")
            bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            backend = _backend(args, captures.get(task_id))
            try:
                if tuple(backend.noise_shape) != (1, 40, 132):
                    raise ValueError("Backend noise shape differs from shared LIBERO contract")
                policy, inner = make_sim_policy(backend, args.suite, task.name, task.language, args.seed)
                contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=EXECUTION_HORIZON)
                if contract.action_horizon != 16 or contract.video_delta_indices != (0,):
                    raise ValueError("Expected shared LIBERO 16-step, single-frame policy contract")
                report["groups"].append({"task_id": task_id, "metadata": backend.metadata,
                                         "state_file_sha256": file_sha256(state_path),
                                         "bddl_sha256": file_sha256(bddl_path)})
                for offset in range(args.episodes):
                    index = args.state_start + offset
                    initial = init_states[index]
                    if hasattr(initial, "detach"):
                        initial = initial.detach().cpu().numpy()
                    state_info = array_fingerprint(initial)
                    reset_seed = case_seed(args.seed, args.suite, task.name, index)
                    env, base = make_fixed_state_env(task, initial, contract, bddl_path, args.max_episode_steps)
                    start = time.perf_counter()
                    try:
                        episode = run_episode(env, policy, index, reset_seed, args.max_episode_steps)
                        settled = base.settled_state_fingerprint
                    finally:
                        env.close()
                    name = f"task-{task_id:02d}-state-{index:03d}"
                    np.savez_compressed(output / f"{name}.npz", action_chunks=np.stack(inner.chunks))
                    episode.update(task_id=task_id, task_name=task.name, prompt=task.language,
                                   state_index=index, reset_seed=reset_seed, initial_state=state_info,
                                   settled_state=settled, elapsed_seconds=time.perf_counter() - start,
                                   initial_processed_inputs=inner.initial_processed_inputs,
                                   decisions=inner.trace, actions_file=f"{name}.npz")
                    report["episodes"].append(episode)
                    write_json(output / "closed_loop.json", report)
                    print(json.dumps({"backend": args.backend, "task_id": task_id,
                                      "state_index": index, "success": episode["success"]}), flush=True)
            finally:
                backend.close()
                write_json(output / "closed_loop.json", report)
        count = len(report["episodes"])
        wins = sum(int(e["success"]) for e in report["episodes"])
        report.update(completed=True, successful_episodes=wins, total_episodes=count,
                      task_success_rate=wins / count)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "closed_loop.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("torch", "trt", "flashrt"), required=True)
    parser.add_argument("--allow-camera-ablation", action="store_true",
                        help="Explicitly allow a one-camera diagnostic instead of default two-camera task validation")
    parser.add_argument("--engine-dir")
    parser.add_argument("--capture-dir")
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parents[1] / "vendor" / "Isaac-GR00T"))
    parser.add_argument("--libero-dir")
    parser.add_argument("--suite", choices=("libero_10",), default="libero_10")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--task-ids", type=int, nargs="+")
    subset.add_argument("--full-suite", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--state-start", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="osmesa")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.state_start < 0 or args.max_episode_steps < 1 or args.cpu_threads < 1:
        parser.error("Episodes/step budget/threads must be positive and state-start nonnegative")
    if args.backend == "trt" and not args.engine_dir:
        parser.error("TensorRT needs --engine-dir")
    if args.backend == "flashrt" and not args.capture_dir:
        parser.error("FlashRT needs --capture-dir")
    run(args)


if __name__ == "__main__":
    main()
