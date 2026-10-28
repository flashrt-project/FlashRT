#!/usr/bin/env python3
"""Recompute the published offline scores and latency statistics without a GPU."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from groot_compare.metrics import ACTION_KEYS, error_metrics


def read(name):
    return json.loads((ROOT / "results" / name).read_text())


def equal(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key in expected:
            equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f"{label}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def main():
    report = read("comparison.json")
    provenance = read("provenance.json")
    for name, key in [("comparison.json", "report_sha256"),
                      ("closed-loop-pilot.json", "pilot_report_sha256")]:
        digest = hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest()
        assert digest == provenance[key], name

    omitted_helpers = {"scripts/Dockerfile.jal", "scripts/restore_services.py"}
    for record in provenance["measured_code_files"]:
        if record["path"] in omitted_helpers:
            continue
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path

    with (ROOT / "results/action-predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    observations = read("observations.json")["records"]
    assert len(rows) == 2848 and len(observations) == 178
    expected_ids = [record["id"] for record in observations]
    assert list(dict.fromkeys(row["record_id"] for row in rows)) == expected_ids
    for index, record in enumerate(observations):
        chunk = rows[index * 16:(index + 1) * 16]
        assert [int(row["action_index"]) for row in chunk] == list(range(16))
        assert all(row["record_id"] == record["id"] and
                   int(row["trajectory_id"]) == record["trajectory_id"] and
                   int(row["frame_index"]) == record["frame_index"] for row in chunk)
    used = np.array([row["executed"] == "1" for row in rows])
    assert int(used.sum()) == 1410
    for row, executed in zip(rows, used):
        assert row["executed"] in ("0", "1")
        assert all(bool(row[f"ground_truth_{axis}"]) == executed for axis in ACTION_KEYS)

    def array(backend, selected):
        # Nine significant decimal digits round-trip the original float32 values.
        return np.array([[float(row[f"{backend}_{axis}"]) for axis in ACTION_KEYS]
                         for row in selected], dtype=np.float32)

    actions = {backend: array(backend, rows) for backend in ("torch", "trt", "flashrt")}
    truth = array("ground_truth", [row for row, keep in zip(rows, used) if keep])
    trajectories = np.array([int(row["trajectory_id"]) for row in rows])
    for expected in [report["reference_accuracy"], *report["accuracy"]]:
        backend = expected["backend"]
        equal(error_metrics(actions[backend], actions["torch"]),
              expected["against_official_pytorch"], f"{backend}.reference")
        equal(error_metrics(actions[backend][used], truth),
              expected["against_recorded_actions"], f"{backend}.demonstrations")
        per_trajectory = {}
        for trajectory in dict.fromkeys(trajectories.tolist()):
            mask = trajectories[used] == trajectory
            per_trajectory[str(trajectory)] = error_metrics(actions[backend][used][mask], truth[mask])
        equal(per_trajectory, expected["per_trajectory_ground_truth"], f"{backend}.trajectories")
        for metric in ("mae", "mse"):
            equal(float(np.mean([v[metric] for v in per_trajectory.values()])),
                  expected[f"jal_style_mean_trajectory_{metric}"], f"{backend}.trajectory_{metric}")

    timings = read("latency-samples.json")
    orders = [["trt", "flashrt"], ["flashrt", "trt"], ["trt", "flashrt"]]
    all_values = {backend: [] for backend in actions if backend != "torch"}
    round_medians = {backend: [] for backend in all_values}
    for index, recorded in enumerate(timings["rounds"]):
        assert recorded["backend_order"] == orders[index]
        paired_ids = []
        for backend, samples in recorded["samples"].items():
            assert len(samples) == 300
            paired_ids.append([(s["prompt"], s["record_id"], s["iteration"]) for s in samples])
            values = [sample["full_call_ms"] for sample in samples]
            assert np.isfinite(values).all() and min(values) > 0
            expected = report["repeated_latency"]["rounds"][index][backend]
            equal(float(np.median(values)), expected["median_ms"], f"round{index}.{backend}.median")
            equal(float(np.percentile(values, 95)), expected["p95_ms"], f"round{index}.{backend}.p95")
            all_values[backend].extend(values)
            round_medians[backend].append(float(np.median(values)))
        assert paired_ids[0] == paired_ids[1]
    for backend, values in all_values.items():
        expected = report["repeated_latency"]["aggregate"][backend]
        actual = {"median_ms": float(np.median(values)),
                  "p95_ms": float(np.percentile(values, 95)),
                  "calls": len(values), "round_medians_ms": round_medians[backend]}
        equal(actual, expected, f"{backend}.aggregate")
    equal(float(np.median(all_values["trt"]) / np.median(all_values["flashrt"])),
          report["repeated_latency"]["aggregate"]["flashrt_speedup_over_jal_trt"], "speedup")
    print("Verified original report and measured-code hashes, 178 observations, "
          "all offline error metrics, and 900 latency samples per optimized backend.")
    print("This checks published data consistency; it does not rerun inference or simulation.")


if __name__ == "__main__":
    main()
