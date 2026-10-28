"""CPU contracts for paired task selection, initialization, and episode accounting."""
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from groot_compare.closed_loop import (
    ACTION_KEYS, _batch_sim_observation, array_fingerprint, case_seed,
    make_fixed_state_env, noise_record_id, run_episode, select_tasks,
)
from groot_compare.common import shared_noise


def observation():
    return {"video.image": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
            "state.x": np.array([[0.5]], dtype=np.float64), "annotation.task": "task"}


class FakePolicy:
    def reset(self, options):
        self.reset_options = options

    def get_action(self, obs):
        assert obs["video.image"].shape == (1, 1, 2, 2, 3)
        assert obs["state.x"].dtype == np.float32
        return {f"action.{key}": np.zeros((1, 16, 1), dtype=np.float32)
                for key in ACTION_KEYS}, {}


class FakeEnv:
    def __init__(self, steps, success=False):
        self.steps = list(steps)
        self.success = success
        self.calls = 0

    def reset(self, seed):
        self.seed = seed
        return observation(), {"success": [False]}

    def step(self, actions):
        assert set(actions) == {f"action.{key}" for key in ACTION_KEYS}
        assert all(a.shape == (16, 1) for a in actions.values())
        steps = self.steps[self.calls]
        self.calls += 1
        done = self.calls == len(self.steps)
        return observation(), 0.0, done, False, {
            "n_env_steps": steps, "success": [False, bool(done and self.success)]}


class ClosedLoopTests(unittest.TestCase):
    def test_default_task_subset_and_missing_full_suite_calibration(self):
        tasks = [SimpleNamespace(language=p, name=p) for p in ("a", "b", "c", "d")]
        suite = SimpleNamespace(get_num_tasks=lambda: 4, get_task=lambda i: tasks[i])
        manifest = {"records": [{"prompt": p, "split": split}
                                for p in ("a", "b", "c")
                                for split in ("calibration", "evaluation")]}
        self.assertEqual([i for i, _ in select_tasks(suite, manifest)], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Missing separate task calibration"):
            select_tasks(suite, manifest, full_suite=True)
        with self.assertRaisesRegex(ValueError, "unique valid"):
            select_tasks(suite, manifest, task_ids=[1, 1])

    def test_noise_schedule_is_backend_independent_and_changes_per_decision(self):
        key = noise_record_id("libero_10", "task", 0, 0)
        first = shared_noise(key, 7)
        self.assertTrue(np.array_equal(first, shared_noise(key, 7)))
        other = shared_noise(noise_record_id("libero_10", "task", 0, 1), 7)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(case_seed(7, "libero_10", "task", 0),
                         case_seed(7, "libero_10", "task", 0))
        self.assertNotEqual(case_seed(7, "libero_10", "task", 0),
                            case_seed(7, "libero_10", "task", 1))

    def test_batching_preserves_pixels_and_only_casts_state(self):
        raw = observation()
        batched = _batch_sim_observation(raw)
        self.assertTrue(np.array_equal(batched["video.image"][0], raw["video.image"]))
        self.assertEqual(batched["state.x"].shape, (1, 1, 1))
        self.assertEqual(batched["state.x"].dtype, np.float32)
        self.assertEqual(batched["annotation.task"], ["task"])

    def test_partial_final_chunk_counts_actual_steps_and_success(self):
        env, policy = FakeEnv([8, 3], success=True), FakePolicy()
        result = run_episode(env, policy, state_index=4, reset_seed=17, max_steps=720)
        self.assertTrue(result["success"])
        self.assertEqual(result["environment_steps"], 11)
        self.assertEqual(result["policy_calls"], 2)
        self.assertEqual(policy.reset_options, {"state_index": 4})
        self.assertEqual(env.seed, 17)

    def test_budget_exhaustion_is_not_task_success(self):
        result = run_episode(FakeEnv([8, 2]), FakePolicy(), 0, 17, max_steps=10)
        self.assertFalse(result["success"])
        self.assertTrue(result["reached_step_budget"])
        with self.assertRaisesRegex(RuntimeError, "invalid environment-step count"):
            run_episode(FakeEnv([9]), FakePolicy(), 0, 17, max_steps=10)

    def test_fixed_state_and_five_zero_actions_precede_history_initialization(self):
        class RawEnv:
            def seed(self, seed):
                self.seed_value = seed

            def reset(self):
                self.state = np.array([0.0])
                self.actions = []

            def set_init_state(self, state):
                self.state = state.copy()
                return {"current": self.state.copy()}

            def step(self, action):
                self.actions.append(action.copy())
                self.state += 1
                return {"current": self.state.copy()}, 0.0, False, {}

            def get_sim_state(self):
                return self.state.copy()

            def check_success(self):
                return False

        class OfficialEnvStub:
            def __init__(self, **kwargs):
                self._env = RawEnv()

            def _process_observation(self, raw):
                return {"official_transformed": raw["current"]}

            def close(self):
                pass

        class HistoryWrapperStub:
            def __init__(self, env, **kwargs):
                self.env = env
                self.kwargs = kwargs

            def reset(self, seed):
                obs, info = self.env.reset(seed=seed)
                self.history = obs
                return obs, info

        env_module = ModuleType("gr00t.eval.sim.LIBERO.libero_env")
        env_module.LiberoEnv = OfficialEnvStub
        wrapper_module = ModuleType("gr00t.eval.sim.wrapper.multistep_wrapper")
        wrapper_module.MultiStepWrapper = HistoryWrapperStub
        with patch.dict(sys.modules, {env_module.__name__: env_module,
                                     wrapper_module.__name__: wrapper_module}):
            wrapped, base = make_fixed_state_env(SimpleNamespace(language="task"),
                                                np.array([17.0]), object(), "task.bddl", 720)
            wrapped.reset(seed=23)
        self.assertEqual(wrapped.history["official_transformed"].tolist(), [22.0])
        self.assertEqual(base._env.seed_value, 23)
        self.assertEqual(len(base._env.actions), 5)
        self.assertTrue(all(np.array_equal(a, np.zeros(7)) for a in base._env.actions))
        self.assertEqual(base.settled_state_fingerprint, array_fingerprint(np.array([22.0])))


if __name__ == "__main__":
    unittest.main()
