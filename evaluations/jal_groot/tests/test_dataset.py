"""CPU tests for identical shared inputs, aligned targets and split isolation."""
from enum import Enum
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from groot_compare import dataset


class Embodiment(Enum):
    LIBERO_PANDA = "libero_sim"


def observation(value=1, prompt="place the bowl"):
    return {
        "video": {"head": np.full((1, 1, 2, 3, 3), value, dtype=np.uint8)},
        "state": {"x": np.array([[[value / 10]]], dtype=np.float32)},
        "language": {"task": [[prompt]]},
    }


def configs():
    return {group: SimpleNamespace(delta_indices=list(range(16)) if group == "action" else [0],
                                   modality_keys=list(dataset.ACTION_KEYS) if group == "action" else [group])
            for group in ("video", "state", "action", "language")}


class DatasetTests(unittest.TestCase):
    def test_roundtrip_preserves_batched_pixels_state_language_without_pickle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            original = observation(prompt="put café mug down")
            dataset.save_observation(path, original)
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(stored["language/task"].dtype.kind, "U")
                self.assertTrue(all(not stored[key].dtype.hasobject for key in stored.files))
            restored = dataset.load_observation(path)
            np.testing.assert_array_equal(restored["video"]["head"], original["video"]["head"])
            np.testing.assert_array_equal(restored["state"]["x"], original["state"]["x"])
            self.assertEqual(restored["language"], original["language"])

    def test_object_and_nonfinite_observations_rejected(self):
        with TemporaryDirectory() as directory:
            invalid = observation()
            invalid["state"]["x"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                dataset.save_observation(Path(directory) / "bad.npz", invalid)
            invalid = observation()
            invalid["language"]["task"] = np.array([[object()]], dtype=object)
            with self.assertRaisesRegex(ValueError, "Language"):
                dataset.save_observation(Path(directory) / "bad.npz", invalid)

    def fixture(self, directory, *, calibration_offset=100, same_metadata=False):
        directory = Path(directory)
        for name in ("evaluation", "calibration"):
            meta = directory / name / "meta"
            meta.mkdir(parents=True)
            for filename in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
                (meta / filename).write_text(json.dumps({"name": "same" if same_metadata else name}))
        vendor = directory / "vendor"
        source = vendor / "scripts/deployment/standalone_inference_script.py"
        source.parent.mkdir(parents=True)
        source.write_text("# mock CPU-only vendor\n")

        class Loader:
            def __init__(self, dataset_path, modality_configs):
                self.path = dataset_path
                self.modality_configs = modality_configs
                self.episodes_metadata = [{"episode_index": 10}, {"episode_index": 11}]

            def __len__(self):
                return 2

            def __getitem__(self, index):
                offset = calibration_offset if self.path.name == "calibration" else 0
                frame = np.arange(11) + offset + index * 20
                return pd.DataFrame({
                    **{f"action.{key}": [np.array([i * 10 + j]) for i in frame]
                       for j, key in enumerate(dataset.ACTION_KEYS)},
                    "value": frame,
                })

        def prepare(trajectory, frame, modality_configs, embodiment, loader):
            return observation(int(trajectory["value"].iloc[frame]))

        dependencies = (Loader, lambda **kwargs: SimpleNamespace(**kwargs), Embodiment, prepare)
        kwargs = dict(dataset_path=directory / "evaluation", output_dir=directory / "output",
                      calibration_dataset_path=directory / "calibration", trajectory_ids=[0],
                      calibration_trajectory_ids=[0], modality_configs=configs(), vendor_root=vendor,
                      dataset_identity="eval@abc123", calibration_dataset_identity="calib@def456")
        return kwargs, dependencies

    def test_export_aligns_horizon_and_trailing_targets(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory)
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                manifest_path = dataset.export_dataset(**kwargs)
            manifest = json.loads(manifest_path.read_text())
            records = [record for record in manifest["records"] if record["split"] == "evaluation"]
            self.assertEqual([record["frame_index"] for record in records], [0, 8])
            self.assertEqual([record["action_steps"] for record in records], [8, 3])
            self.assertEqual([record["episode_id"] for record in records], [10, 10])
            with np.load(manifest_path.parent / records[1]["ground_truth"], allow_pickle=False) as data:
                self.assertEqual(data["actions"].shape, (3, 7))
                self.assertEqual(data["actions"].dtype, np.float32)
                np.testing.assert_array_equal(data["actions"][0], np.arange(7) + 80)
            restored = dataset.load_observation(manifest_path.parent / records[1]["observation"])
            self.assertEqual(restored["video"]["head"][0, 0, 0, 0, 0], 8)

    def test_copied_dataset_same_episode_rejected_before_writing(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory, same_metadata=True)
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                with self.assertRaisesRegex(ValueError, "same dataset episode"):
                    dataset.export_dataset(**kwargs)
            self.assertFalse(kwargs["output_dir"].exists())

    def test_identical_frames_across_different_datasets_rejected_even_between_inference_frames(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory, calibration_offset=3)
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                with self.assertRaisesRegex(ValueError, "duplicates evaluation pixels and state"):
                    dataset.export_dataset(**kwargs)
            self.assertFalse((kwargs["output_dir"] / "manifest.json").exists())

    def test_same_dataset_disjoint_episodes_allowed(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory, same_metadata=True)
            kwargs["calibration_trajectory_ids"] = [1]
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                path = dataset.export_dataset(**kwargs)
            self.assertTrue(path.is_file())

    def test_reserved_trt_episode_is_rejected_by_dataset_identity_before_export(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory)
            kwargs.update(reserved_calibration_episodes=[10],
                          reserved_calibration_dataset_identity="eval@abc123")
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                with self.assertRaisesRegex(ValueError, "reserved TensorRT calibration episodes"):
                    dataset.export_dataset(**kwargs)
            self.assertFalse(kwargs["output_dir"].exists())

    def test_reserved_trt_episode_disjoint_from_heldout_cohort_is_recorded(self):
        with TemporaryDirectory() as directory:
            kwargs, dependencies = self.fixture(directory)
            kwargs.update(reserved_calibration_episodes=[0],
                          reserved_calibration_dataset_identity="eval@abc123")
            with patch.object(dataset, "_vendor_dependencies", return_value=dependencies):
                path = dataset.export_dataset(**kwargs)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["reserved_calibration"]["episodes"], [0])
            self.assertEqual(manifest["sources"]["evaluation"]["episodes"], [10])

    def test_reserved_trt_episode_requires_explicit_source_identity(self):
        with TemporaryDirectory() as directory:
            kwargs, _ = self.fixture(directory)
            kwargs.update(reserved_calibration_episodes=[0])
            with self.assertRaisesRegex(ValueError, "require their dataset identity"):
                dataset.export_dataset(**kwargs)

    def test_checkpoint_config_uses_one_camera_overlay(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "processor_config.json"
            config = {"video": {"delta_indices": [0], "modality_keys": ["head"]}}
            path.write_text(json.dumps({"processor_kwargs": {"modality_configs": {"libero_sim": config}}}))
            self.assertEqual(dataset._read_modality_configs(directory, "libero_sim"), config)


if __name__ == "__main__":
    unittest.main()
