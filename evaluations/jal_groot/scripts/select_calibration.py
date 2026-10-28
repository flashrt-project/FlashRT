#!/usr/bin/env python3
"""Select per-prompt calibration episodes using metadata only, without decoding.

This does not establish pixel-level independence. The shared exporter must still
reject matching evaluation/calibration frame content before benchmarking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_episodes(dataset: Path) -> list[dict]:
    return [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()]


def signature(episode: dict) -> tuple:
    return tuple(episode["tasks"]), int(episode["length"])


def select_calibration(evaluation_dataset: Path, calibration_dataset: Path,
                       evaluation_trajectory_ids=(0, 1, 2, 3, 4), calibration_size=10) -> dict:
    all_eval = read_episodes(evaluation_dataset)
    evaluation = [all_eval[index] for index in evaluation_trajectory_ids]
    calibration = read_episodes(calibration_dataset)
    eval_episode_ids = {int(episode["episode_index"]) for episode in evaluation}
    eval_signatures = {signature(episode) for episode in evaluation}
    prompts = list(dict.fromkeys(prompt for episode in evaluation for prompt in episode["tasks"]))
    selected = []
    for prompt in prompts:
        candidates = [(index, episode) for index, episode in enumerate(calibration)
                      if prompt in episode["tasks"]
                      and int(episode["episode_index"]) not in eval_episode_ids
                      and signature(episode) not in eval_signatures]
        if not candidates:
            raise ValueError(f"No independent metadata candidate for prompt {prompt!r}")
        index, episode = candidates[0]
        selected.append({"prompt": prompt, "trajectory_id": index,
                         "episode_id": int(episode["episode_index"]),
                         "frame_index": 0, "episode_length": int(episode["length"]),
                         "alternative_trajectory_ids": [index for index, _ in candidates[1:4]]})

    original_batches, remaining = [], calibration_size
    for index, episode in enumerate(calibration):
        count = min(remaining, int(episode["length"]))
        if count:
            matched = [eval_index for eval_index, eval_episode in zip(evaluation_trajectory_ids, evaluation)
                       if int(eval_episode["episode_index"]) == int(episode["episode_index"])
                       and signature(eval_episode) == signature(episode)]
            original_batches.append({"trajectory_id": index, "episode_id": int(episode["episode_index"]),
                                     "frame_indices": list(range(count)),
                                     "matching_evaluation_trajectory_ids_from_metadata": matched})
        remaining -= count
        if remaining <= 0:
            break
    if remaining:
        raise ValueError("Calibration dataset has fewer frames than calibration_size")
    return {
        "metadata_only": True,
        "requires_exporter_content_overlap_check": True,
        "selection_rule": "first matching-prompt episode excluding evaluation episode IDs and task/length signatures",
        "flashrt_calibration_rule": "one first-frame observation per prompt from the selected separate episode",
        "recommended_calibration_trajectory_ids": [record["trajectory_id"] for record in selected],
        "selected": selected,
        "jal_calibration_size": calibration_size,
        "jal_current_sequential_sampling": original_batches,
        "metadata_sha256": {
            name: hashlib.sha256((path / "meta" / "episodes.jsonl").read_bytes()).hexdigest()
            for name, path in (("evaluation", evaluation_dataset), ("calibration", calibration_dataset))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dataset", required=True, type=Path)
    parser.add_argument("--calibration-dataset", required=True, type=Path)
    parser.add_argument("--evaluation-trajectory-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--calibration-size", type=int, default=10)
    args = parser.parse_args()
    if args.calibration_size < 1:
        parser.error("calibration-size must be positive")
    print(json.dumps(select_calibration(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
