"""Verify full task/state coverage, saved actions and paired observations/noise."""

from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from groot_compare.common import shared_noise

root = Path(sys.argv[1])
seed = int(sys.argv[2])
data = {}
summary = {}
exceptions = []
expected_keys = {(t, s) for t in range(10) for s in range(50)}
for backend in ["torch", "trt", "flashrt"]:
    reports = [
        json.loads(p.read_text()) for p in (root / backend).glob("task*/results.json")
    ]
    assert len(reports) == 10 and all(
        t["completed"] and "error" not in t and t["seed"] == seed for t in reports
    )
    rows = {}
    for task in reports:
        assert (
            task["cameras"] == ["image", "wrist_image"]
            and task["denoise_steps"] == 4
            and task["execute_actions"] == 8
            and task["settling_steps"] == 5
            and task["max_steps"] == 720
        )
        for e in task["episodes"]:
            key = (task["task_id"], e["state_index"])
            assert key not in rows
            rows[key] = e
            assert 1 <= e["environment_steps"] <= 720 and e["policy_calls"] == len(
                e["decisions"]
            )
            for decision in e["decisions"]:
                noise = shared_noise(
                    decision["noise_record_id"], seed, shape=(1, 40, 132)
                )
                assert (
                    hashlib.sha256(noise.tobytes()).hexdigest()
                    == decision["noise_sha256"]
                )
            with np.load(
                root / backend / f"task{key[0]}" / f"actions-state{key[1]}.npz"
            ) as f:
                a = f["physical_actions"]
                assert a.shape == (e["policy_calls"], 16, 7) and np.isfinite(a).all()
    assert set(rows) == expected_keys
    data[backend] = rows
    successes = sum(e["success"] for e in rows.values())
    summary[backend] = {
        "episodes": 500,
        "successes": successes,
        "success_percent": successes / 5,
        "unique_complete_task_state_keys": True,
    }
for backend in ["trt", "flashrt"]:
    for key, e in data[backend].items():
        ref = data["torch"][key]
        assert all(e[k] == ref[k] for k in ["initial_state", "settled_state"])
        n = min(len(e["decisions"]), len(ref["decisions"]))
        assert all(
            e["decisions"][i]["noise_sha256"] == ref["decisions"][i]["noise_sha256"]
            for i in range(n)
        )
        actual = e["initial_processed_inputs"]
        expected = ref["initial_processed_inputs"]
        if actual != expected:
            different = [k for k in actual if actual[k] != expected[k]]
            assert different == ["pixel_values"]
            receipt = e["input_pairing"]
            assert receipt["mode"] in [
                "render_rounding_one_lsb_sparse",
                "render_rounding_one_pixel_one_lsb",
            ]
            rb = receipt["raw_reference_backend"]
            if rb != "torch":
                assert data[rb][key]["initial_processed_inputs"] == expected
            a = torch.load(
                root
                / backend
                / f"task{key[0]}"
                / f"initial-observation-state{key[1]}.pt",
                map_location="cpu",
                weights_only=False,
            )
            b = torch.load(
                root / rb / f"task{key[0]}" / f"initial-observation-state{key[1]}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert set(a) == set(b) and set(a["video"]) == set(b["video"])
            assert a["language"] == b["language"]
            assert set(a["state"]) == set(b["state"]) and all(
                np.array_equal(a["state"][k], b["state"][k]) for k in a["state"]
            )
            count = 0
            pixels = 0
            maximum = 0
            for camera in a["video"]:
                x = np.asarray(a["video"][camera])
                y = np.asarray(b["video"][camera])
                assert x.shape == y.shape and x.dtype == y.dtype == np.uint8
                d = np.abs(x.astype(np.int16) - y.astype(np.int16))
                count += int(np.count_nonzero(np.any(d, axis=-1)))
                pixels += int(x.size // 3)
                maximum = max(maximum, int(d.max()))
            assert 0 < count and maximum <= 1 and count / pixels <= 1e-4
            assert (
                receipt["different_pixels_total"] == count
                and receipt["max_uint8_difference"] == maximum
            )
            if receipt["mode"] == "render_rounding_one_pixel_one_lsb":
                assert count <= 1
            assert (
                receipt["actual_pixel_sha256"] == actual["pixel_values"]["sha256"]
                and receipt["reference_pixel_sha256"]
                == expected["pixel_values"]["sha256"]
            )
            exceptions.append(
                {
                    "backend": backend,
                    "task_id": key[0],
                    "state_index": key[1],
                    "different_pixels": count,
                    "pixel_count": pixels,
                    "max_uint8_difference": maximum,
                }
            )
result = {
    "seed": seed,
    "verified": True,
    "total_episodes": 1500,
    "results": summary,
    "paired_state_tokens_grid_and_noise_verified": True,
    "rendering_exceptions": exceptions,
    "initial_processed_inputs_all_bit_identical": len(exceptions) == 0,
}
(root / "verified-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
