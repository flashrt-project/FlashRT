import numpy as np
import pytest

from flash_rt.models.dm05.validation import compare_actions
from tools.validate_dm05_actions import compare_records


def actions():
    return np.ones((3, 10, 7), dtype=np.float32)


def compare(a, b, **kwargs):
    return compare_actions(a, b, atol=kwargs.pop("atol", 0.01), gripper_threshold=0.0, **kwargs)


def test_tolerance_accepts_nonidentical_actions():
    a = actions()
    report = compare(a, a + 0.001)
    assert report["passed"]
    assert report["max_abs_by_component"][6] > 0


def test_arm_cosine_does_not_hide_gripper_flips():
    a = actions()
    b = a.copy()
    b[1, 4, 6] = -1
    report = compare(a, b)
    assert report["arm_cosine_min"] == pytest.approx(1)
    assert not report["passed"]
    assert report["gripper_flip_count"] == 1
    assert report["gripper_flip_indices"] == [[1, 4]]
    assert report["max_abs_by_component"][6] == 2


def test_small_absolute_difference_can_still_flip_a_decision():
    a = actions()
    a[..., 6] = 0.001
    b = a.copy()
    b[0, 0, 6] = -0.001
    report = compare(a, b)
    assert report["numeric_mismatch_count"] == 0
    assert not report["passed"]
    assert report["gripper_flip_count"] == 1


def test_same_gripper_decision_still_requires_numeric_tolerance():
    a = actions()
    b = a.copy()
    b[..., 6] += 0.1
    report = compare(a, b)
    assert report["gripper_flip_count"] == 0
    assert not report["passed"]


def test_arm_magnitude_is_checked_as_well_as_direction():
    a = actions()
    b = a.copy()
    b[..., :6] *= 2
    report = compare(a, b)
    assert report["arm_cosine_min"] == pytest.approx(1)
    assert not report["passed"]


def test_zero_vectors_do_not_produce_nan_or_spurious_failure():
    a = np.zeros((1, 2, 7))
    assert compare(a, a)["passed"]


@pytest.mark.parametrize("bad", [np.zeros((3, 10, 6)), np.zeros((0, 10, 7)),
                                  np.full((3, 10, 7), np.nan)])
def test_invalid_action_records_are_rejected(bad):
    with pytest.raises(ValueError):
        compare(actions(), bad)


@pytest.mark.parametrize("key", ["sample_ids", "input_sha256", "noise_sha256", "action_space"])
def test_different_input_or_noise_provenance_is_rejected(key):
    a = {"actions": actions(), "sample_ids": np.arange(3), "input_sha256": np.array("inputs"),
         "noise_sha256": np.array("noise"), "action_space": np.array("final_physical")}
    b = dict(a)
    b[key] = np.array("different")
    with pytest.raises(ValueError, match="record mismatch"):
        compare_records(a, b, atol=0.01, gripper_threshold=0)


def test_matching_provenance_does_not_bypass_the_action_gate():
    a = {"actions": actions(), "sample_ids": np.arange(3), "input_sha256": np.array("inputs"),
         "noise_sha256": np.array("noise"), "action_space": np.array("final_physical")}
    b = dict(a, actions=a["actions"].copy())
    b["actions"][2, 8, 6] = -1
    assert not compare_records(a, b, atol=0.01, gripper_threshold=0)["passed"]


def test_command_returns_failure_for_a_gripper_flip(tmp_path):
    import json
    from tools.validate_dm05_actions import main
    common = {"sample_ids": np.arange(3), "input_sha256": np.array("inputs"),
              "noise_sha256": np.array("noise"), "action_space": np.array("final_physical")}
    ref, candidate, report = tmp_path / "ref.npz", tmp_path / "candidate.npz", tmp_path / "report.json"
    a = actions()
    b = a.copy()
    b[1, 0, 6] = -1
    np.savez(ref, actions=a, **common)
    np.savez(candidate, actions=b, **common)
    code = main(["--reference", str(ref), "--candidate", str(candidate), "--atol", "0.01",
                 "--gripper-threshold", "0", "--output", str(report)])
    assert code == 1
    assert json.loads(report.read_text())["gripper_flip_indices"] == [[1, 0]]
