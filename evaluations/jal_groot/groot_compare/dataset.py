"""Export shared, unprocessed GR00T observations without loading a model.

The observation preparation function is compiled directly from the pinned JAL
standalone evaluator. Only that CPU function is loaded, avoiding the evaluator's
model and TensorRT imports. Exported observations preserve its batch/time axes.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
DEFAULT_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "Isaac-GR00T"


def _observation_arrays(observation: dict) -> dict[str, np.ndarray]:
    if set(observation) != {"video", "state", "language"}:
        raise ValueError("Observation must contain video, state and language groups")
    arrays = {}
    for group, values in observation.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Empty or invalid observation group: {group}")
        for key, value in values.items():
            if not isinstance(key, str) or not key or "/" in key:
                raise ValueError(f"Invalid observation key: {key!r}")
            arr = np.asarray(value)
            if group == "language":
                if arr.dtype.kind not in "US" or arr.shape != (1, 1):
                    raise ValueError("Language must be a single batched string [[prompt]]")
                arr = arr.astype(np.str_)
            elif group == "video":
                if arr.ndim != 5 or arr.shape[:2] != (1, 1) or arr.shape[-1] != 3:
                    raise ValueError("LIBERO video must have shape [1,1,H,W,3]")
            elif arr.ndim != 3 or arr.shape[:2] != (1, 1):
                raise ValueError("LIBERO state must have shape [1,1,D]")
            if arr.dtype.hasobject:
                raise ValueError("Object arrays are forbidden")
            if group != "language" and not np.isfinite(arr).all():
                raise ValueError("Observation contains nonfinite values")
            arrays[f"{group}/{key}"] = arr
    return arrays


def save_observation(path: str | Path, observation: dict) -> None:
    """Write raw, nested official-policy inputs as a pickle-free NPZ."""
    arrays = _observation_arrays(observation)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_observation(path: str | Path) -> dict:
    """Restore arrays and language lists exactly as the GR00T API expects."""
    observation = {"video": {}, "state": {}, "language": {}}
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            group, separator, key = name.partition("/")
            if not separator or group not in observation or not key:
                raise ValueError(f"Invalid observation archive key: {name}")
            value = data[name]
            observation[group][key] = value.tolist() if group == "language" else value
    _observation_arrays(observation)
    return observation


def _content_digest(observation: dict) -> str:
    """Hash pixels and state, ignoring prompts so relabeling cannot hide reuse."""
    digest = hashlib.sha256()
    for key, array in sorted(_observation_arrays(observation).items()):
        if key.startswith("language/"):
            continue
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _metadata_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        digest.update(name.encode())
        digest.update((path / "meta" / name).read_bytes())
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_modality_configs(checkpoint_path: str | Path, embodiment: str) -> dict:
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = (path / "processor_config.json", path / "processor" / "processor_config.json")
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    config = json.loads(path.read_text())
    configs = config.get("processor_kwargs", config).get("modality_configs")
    if configs is None or embodiment not in configs:
        raise ValueError(f"No {embodiment} modality configuration in {path}")
    return configs[embodiment]


def _vendor_dependencies(vendor_root: Path):
    """Import data-only APIs and the pinned evaluator's exact CPU function."""
    sys.path.insert(0, str(vendor_root))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig
    from gr00t.data.utils import parse_observation_gr00t

    source_path = vendor_root / "scripts/deployment/standalone_inference_script.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "prepare_observation_data")
    isolated = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ], type_ignores=[])
    namespace = {"np": np, "extract_step_data": extract_step_data,
                 "parse_observation_gr00t": parse_observation_gr00t}
    exec(compile(ast.fix_missing_locations(isolated), str(source_path), "exec"), namespace)
    return LeRobotEpisodeLoader, ModalityConfig, EmbodimentTag, namespace["prepare_observation_data"]


def _canonical_actions(trajectory, start: int, end: int) -> np.ndarray:
    columns = []
    for key in ACTION_KEYS:
        values = trajectory[f"action.{key}"].iloc[start:end]
        column = np.vstack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
        if column.shape != (end - start, 1):
            raise ValueError(f"Expected scalar LIBERO action {key}, got {column.shape}")
        columns.append(column)
    actions = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(actions).all():
        raise ValueError("Ground-truth actions contain nonfinite values")
    return actions


def export_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    calibration_dataset_path: str | Path | None = None,
    trajectory_ids=(0, 1, 2, 3, 4),
    calibration_trajectory_ids=(5,),
    execution_horizon: int = 8,
    steps: int = 300,
    embodiment_tag: str = "LIBERO_PANDA",
    modality_configs: dict | None = None,
    dataset_identity: str | None = None,
    calibration_dataset_identity: str | None = None,
    reserved_calibration_episodes=(),
    reserved_calibration_dataset_identity: str | None = None,
    vendor_root: str | Path | None = None,
) -> Path:
    """Export evaluation and episode-disjoint calibration data; return manifest.

    ``trajectory_ids`` are loader indices, as in the JAL evaluator; the actual
    dataset episode indices are recorded separately. Dataset identities should
    include the upstream repository and immutable revision. Matching metadata
    fingerprints and matching frame content also detect copied dataset paths.
    """
    if execution_horizon < 1 or steps < 1:
        raise ValueError("execution_horizon and steps must be positive")
    trajectory_ids, calibration_trajectory_ids = tuple(trajectory_ids), tuple(calibration_trajectory_ids)
    reserved_calibration_episodes = tuple(reserved_calibration_episodes)
    if reserved_calibration_episodes and not reserved_calibration_dataset_identity:
        raise ValueError("Reserved calibration episodes require their dataset identity")
    if not trajectory_ids or not calibration_trajectory_ids:
        raise ValueError("Both evaluation and separate calibration trajectories are required")
    for ids in (trajectory_ids, calibration_trajectory_ids, reserved_calibration_episodes):
        if len(ids) != len(set(ids)) or any(not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError("Trajectory IDs must be unique nonnegative integers")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty export: {output_dir}")
    vendor_root = Path(vendor_root or DEFAULT_VENDOR).resolve()
    Loader, Modality, Embodiment, prepare = _vendor_dependencies(vendor_root)
    try:
        embodiment = Embodiment[embodiment_tag]
    except KeyError:
        embodiment = Embodiment(embodiment_tag)
    if modality_configs is None:
        if checkpoint_path is None:
            raise ValueError("Provide checkpoint_path or explicit modality_configs")
        modality_configs = _read_modality_configs(checkpoint_path, embodiment.value)
    modality_configs = {key: Modality(**value) if isinstance(value, dict) else deepcopy(value)
                        for key, value in modality_configs.items()}
    if tuple(modality_configs["action"].modality_keys) != ACTION_KEYS:
        raise ValueError("Checkpoint action order must be x,y,z,roll,pitch,yaw,gripper")
    if len(modality_configs["action"].delta_indices) < execution_horizon:
        raise ValueError("Execution horizon exceeds checkpoint action horizon")
    for group in ("video", "state", "language"):
        if modality_configs[group].delta_indices != [0]:
            raise ValueError(f"LIBERO exporter requires {group} delta_indices=[0]")

    eval_path = Path(dataset_path).resolve()
    calib_path = Path(calibration_dataset_path or dataset_path).resolve()
    paths = {"evaluation": eval_path, "calibration": calib_path}
    ids_by_split = {"evaluation": trajectory_ids, "calibration": calibration_trajectory_ids}
    explicit = {"evaluation": dataset_identity, "calibration": calibration_dataset_identity}
    if calib_path == eval_path and explicit["calibration"] is None:
        explicit["calibration"] = explicit["evaluation"]
    loaders, sources = {}, {}
    for split, path in paths.items():
        loader = Loader(dataset_path=path, modality_configs=deepcopy(modality_configs))
        loaders[split] = loader
        fingerprint = _metadata_digest(path)
        sources[split] = {"path": str(path), "identity": explicit[split] or f"metadata-sha256:{fingerprint}",
                          "metadata_sha256": fingerprint, "git_revision": _git_revision(path),
                          "trajectory_ids": list(ids_by_split[split]), "episodes": []}
        for trajectory_id in ids_by_split[split]:
            if trajectory_id >= len(loader):
                raise ValueError(f"{split} trajectory {trajectory_id} exceeds dataset size {len(loader)}")
            sources[split]["episodes"].append(int(loader.episodes_metadata[trajectory_id]["episode_index"]))
    eval_source, calib_source = sources["evaluation"], sources["calibration"]
    if (reserved_calibration_dataset_identity == eval_source["identity"]
            and set(reserved_calibration_episodes) & set(eval_source["episodes"])):
        overlap = sorted(set(reserved_calibration_episodes) & set(eval_source["episodes"]))
        raise ValueError(f"Evaluation includes reserved TensorRT calibration episodes: {overlap}")
    same_dataset = (eval_path == calib_path or eval_source["identity"] == calib_source["identity"]
                    or eval_source["metadata_sha256"] == calib_source["metadata_sha256"])
    if same_dataset and set(eval_source["episodes"]) & set(calib_source["episodes"]):
        raise ValueError("Calibration and evaluation contain the same dataset episode")

    # Check calibration against every evaluated frame, including frames between
    # inference calls, so a stride offset cannot conceal calibration leakage.
    evaluation_frames: set[str] = set()
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("evaluation", "calibration"):
        loader = loaders[split]
        observation_configs = deepcopy(loader.modality_configs)
        observation_configs.pop("action", None)
        for trajectory_id, episode_id in zip(ids_by_split[split], sources[split]["episodes"]):
            trajectory = loader[trajectory_id]
            actual_steps = min(steps, len(trajectory))
            if actual_steps < 1:
                raise ValueError(f"Empty trajectory: {split}/{trajectory_id}")
            for frame in range(actual_steps):
                if split == "calibration" and frame % execution_horizon:
                    continue
                obs = prepare(trajectory, frame, observation_configs, embodiment, loader)
                digest = _content_digest(obs)
                if split == "evaluation":
                    evaluation_frames.add(digest)
                elif digest in evaluation_frames:
                    raise ValueError(f"Calibration frame {trajectory_id}/{frame} duplicates evaluation pixels and state")
                if frame % execution_horizon:
                    continue
                record_id = f"{split}-trajectory-{trajectory_id:05d}-frame-{frame:06d}"
                obs_file = f"observations/{record_id}.npz"
                gt_file = f"ground_truth/{record_id}.npz"
                save_observation(output_dir / obs_file, obs)
                gt = _canonical_actions(trajectory, frame, min(frame + execution_horizon, actual_steps))
                (output_dir / gt_file).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_dir / gt_file, actions=gt)
                prompt = next(iter(obs["language"].values()))[0][0]
                records.append({"id": record_id, "split": split, "trajectory_id": trajectory_id,
                                "episode_id": episode_id, "frame_index": frame, "prompt": prompt,
                                "observation": obs_file, "ground_truth": gt_file,
                                "content_sha256": digest, "action_steps": len(gt)})
    source = vendor_root / "scripts/deployment/standalone_inference_script.py"
    manifest = {"format_version": 1, "execution_horizon": execution_horizon, "steps_per_trajectory": steps,
                "action_keys": list(ACTION_KEYS), "embodiment_tag": embodiment.value,
                "sources": sources, "vendor_revision": _git_revision(vendor_root),
                "reserved_calibration": {"dataset_identity": reserved_calibration_dataset_identity,
                                         "episodes": list(reserved_calibration_episodes)},
                "prepare_observation_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "overlap_check": "episode identities plus calibration pixels/state against every evaluation frame",
                "records": records}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--calibration-dataset-path", type=Path)
    parser.add_argument("--trajectory-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--calibration-trajectory-ids", nargs="+", type=int, required=True)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--dataset-identity")
    parser.add_argument("--calibration-dataset-identity")
    parser.add_argument("--reserved-calibration-episodes", nargs="+", type=int, default=[])
    parser.add_argument("--reserved-calibration-dataset-identity")
    parser.add_argument("--vendor-root", type=Path)
    print(export_dataset(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
