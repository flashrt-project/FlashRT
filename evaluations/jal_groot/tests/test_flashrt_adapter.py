"""CPU contract tests: reject incompatible inputs and consume each fresh frame."""
import contextlib
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from groot_compare.backends.flashrt import (
    ACTION_KEYS, FlashRTBackend, NOISE_SHAPE, _observation_prompt,
    observation_sha256,
)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    bfloat16 = "bfloat16"
    inference_mode = staticmethod(contextlib.nullcontext)
    from_numpy = staticmethod(Tensor)
    equal = staticmethod(lambda a, b: np.array_equal(a.value, b.value))


class Frontend:
    def __init__(self):
        self.received = []
        self.output = {k: Tensor(np.zeros((1, 16, 1), np.float32)) for k in ACTION_KEYS}

    def run_backbone_graph(self, inputs):
        return inputs["pixel_values"]

    def infer(self, state, *, initial_noise, num_inference_timesteps, action_horizon):
        assert num_inference_timesteps == 4 and action_horizon == 40
        self.received.append((self._backbone_features.value.copy(),
                              state.value.copy(), initial_noise.value.copy()))
        return Tensor(self._backbone_features.value[0, 0]
                      + state.value[0, 0, 0] + initial_noise.value[0, 0, 0])

    def denormalize_action(self, normalized, *, state_dict):
        for value in self.output.values():
            value.value.fill(float(normalized.value) + state_dict["state.x"][0, 0, 0])
        return self.output


def observation(pixel=1, state=2, prompt="pick up the mug"):
    return {
        "video": {"image": np.full((1, 1, 8, 8, 3), pixel, dtype=np.uint8)},
        "state": {k: np.full((1, 1, 2 if k == "gripper" else 1), state, np.float32)
                  for k in ACTION_KEYS},
        "language": {"annotation.human.action.task_description": [[prompt]]},
    }


def backend():
    model = FlashRTBackend.__new__(FlashRTBackend)
    model._torch = FakeTorch
    model._closed = False
    model._device = "cuda:0"
    model._frontend = Frontend()
    model._prompt = "pick up the mug"
    model._camera_keys = ("image",)
    model._expected_tokens = Tensor([[1, 2]])
    model._expected_grid = Tensor([[1, 2, 2]])
    model._pixel_shape = (1, 1)

    def prepare(obs):
        return {
            "pixel_values": Tensor([[obs["video"]["image"][0, 0, 0, 0, 0]]]),
            "state": Tensor(np.full((1, 1, 132), obs["state"]["x"][0, 0, 0])),
            "input_ids": Tensor([[1, 2]]),
            "image_grid_thw": Tensor([[1, 2, 2]]),
        }
    model._prepare = prepare
    return model


class FlashRTAdapterContractTests(unittest.TestCase):
    def test_module_import_does_not_load_gpu_dependencies(self):
        subprocess.run([
            sys.executable, "-c",
            "import sys; import groot_compare.backends.flashrt; "
            "assert not any(x in sys.modules for x in ('torch', 'gr00t', 'flash_rt'))",
        ], check=True)

    def test_new_frame_state_and_noise_are_used_and_outputs_are_owned(self):
        model = backend()
        first = model.get_action(observation(pixel=1, state=2), np.full(NOISE_SHAPE, 3, np.float32))
        second = model.get_action(observation(pixel=10, state=4), np.full(NOISE_SHAPE, 7, np.float32))
        # Include the *current* raw state again when decoding relative actions.
        np.testing.assert_array_equal(first["x"], 8)
        np.testing.assert_array_equal(second["x"], 25)
        self.assertEqual(len(model._frontend.received), 2)
        np.testing.assert_array_equal(model._frontend.received[1][0], [[10]])
        np.testing.assert_array_equal(model._frontend.received[1][1], 4)
        np.testing.assert_array_equal(model._frontend.received[1][2], 7)
        self.assertEqual(first["x"].dtype, np.float32)

    def test_changed_prompt_is_rejected_before_model_execution(self):
        model = backend()
        with self.assertRaisesRegex(ValueError, "Changed instruction"):
            model.get_action(observation(prompt="a different task"), np.zeros(NOISE_SHAPE))
        self.assertEqual(model._frontend.received, [])

    def test_changed_tokens_or_grid_are_rejected(self):
        for field, value in (("input_ids", [[1, 3]]), ("image_grid_thw", [[1, 3, 2]])):
            with self.subTest(field=field):
                model = backend()
                original = model._prepare
                def prepare(obs):
                    processed = original(obs)
                    processed[field] = Tensor(value)
                    return processed
                model._prepare = prepare
                with self.assertRaisesRegex(ValueError, field):
                    model.get_action(observation(), np.zeros(NOISE_SHAPE))
                self.assertEqual(model._frontend.received, [])

    def test_input_fingerprints_observe_new_pixels_and_state_without_inference(self):
        model = backend()
        first = model.input_fingerprints(observation(pixel=1, state=2))
        second = model.input_fingerprints(observation(pixel=10, state=4))
        self.assertEqual(set(first), {"pixel_values", "input_ids", "image_grid_thw", "state"})
        self.assertEqual(first["input_ids"], second["input_ids"])
        self.assertEqual(first["image_grid_thw"], second["image_grid_thw"])
        self.assertNotEqual(first["pixel_values"]["sha256"], second["pixel_values"]["sha256"])
        self.assertNotEqual(first["state"]["sha256"], second["state"]["sha256"])
        self.assertEqual(first["state"]["shape"], [1, 1, 132])
        self.assertEqual(model._frontend.received, [])

    def test_wrong_noise_shape_and_nonfinite_noise_are_rejected(self):
        for noise in (np.zeros((1, 16, 7)), np.full(NOISE_SHAPE, np.nan)):
            with self.assertRaisesRegex(ValueError, "finite diffusion noise"):
                backend().get_action(observation(), noise)

    def test_camera_contract_and_calibration_identity(self):
        obs = observation()
        original = observation_sha256(obs)
        obs["state"]["x"][0, 0, 0] += 1
        self.assertNotEqual(original, observation_sha256(obs))
        obs["video"]["wrist_image"] = obs["video"]["image"].copy()
        self.assertEqual(_observation_prompt(obs), "pick up the mug")
        original = observation_sha256(obs)
        obs["video"]["wrist_image"][0, 0, 0, 0, 0] += 1
        self.assertNotEqual(original, observation_sha256(obs))
        obs["video"]["extra_camera"] = obs["video"]["image"]
        with self.assertRaisesRegex(ValueError, "optionally with wrist_image"):
            _observation_prompt(obs)

    def test_wrist_shape_and_dtype_are_validated(self):
        for image in (np.zeros((1, 8, 8, 3), np.uint8),
                      np.zeros((1, 1, 8, 8, 3), np.float32)):
            obs = observation()
            obs["video"]["wrist_image"] = image
            with self.assertRaisesRegex(ValueError, "wrist_image"):
                _observation_prompt(obs)

    def test_two_camera_backend_rejects_missing_wrist_before_processing(self):
        model = backend()
        model._camera_keys = ("image", "wrist_image")
        for call in (lambda: model.get_action(observation(), np.zeros(NOISE_SHAPE)),
                     lambda: model.input_fingerprints(observation())):
            with self.assertRaisesRegex(ValueError, "Observation cameras differ"):
                call()
        self.assertEqual(model._frontend.received, [])

    def test_two_camera_calls_consume_new_wrist_pixels(self):
        model = backend()
        model._camera_keys = ("image", "wrist_image")
        original_prepare = model._prepare
        def prepare(obs):
            result = original_prepare(obs)
            result["pixel_values"].value += obs["video"]["wrist_image"][0, 0, 0, 0, 0]
            return result
        model._prepare = prepare
        obs = observation()
        obs["video"]["wrist_image"] = np.full_like(obs["video"]["image"], 2)
        first = model.get_action(obs, np.full(NOISE_SHAPE, 3, np.float32))
        obs["video"]["wrist_image"].fill(7)
        second = model.get_action(obs, np.full(NOISE_SHAPE, 3, np.float32))
        np.testing.assert_array_equal(first["x"], 10)
        np.testing.assert_array_equal(second["x"], 15)

    def test_cleanup_releases_graphs_and_model_after_cuda_error(self):
        model = backend()
        frontend = model._frontend
        frontend._k_dit_graph = object()
        frontend._kbb_graph = object()
        model._prep = model._processor = model._model = object()
        calls = []

        def synchronize(device):
            calls.append(("sync", device))
            raise RuntimeError("injected CUDA failure")

        model._torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=synchronize, empty_cache=lambda: calls.append("cache")))
        with self.assertRaisesRegex(RuntimeError, "injected CUDA failure"):
            model.close()
        self.assertTrue(model._closed)
        self.assertIsNone(model._frontend)
        self.assertIsNone(model._model)
        self.assertIsNone(model._processor)
        self.assertIsNone(model._prep)
        self.assertFalse(hasattr(frontend, "_k_dit_graph"))
        self.assertFalse(hasattr(frontend, "_kbb_graph"))
        self.assertEqual(calls, [("sync", "cuda:0"), "cache"])
        model.close()
        self.assertEqual(len(calls), 2)

    def test_closed_backend_refuses_calls(self):
        model = backend()
        model._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.get_action(observation(), np.zeros(NOISE_SHAPE))


if __name__ == "__main__":
    unittest.main()
