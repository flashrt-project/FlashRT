import time

import numpy as np
import pytest

from flash_rt.runtime.rtc import AsyncChunkRunner, CallablePolicyAdapter, RTCConfig


def test_callable_policy_adapter_accepts_dict_output():
    adapter = CallablePolicyAdapter(
        lambda obs: {"actions": np.ones((4, 3), dtype=np.float32)}
    )

    out = adapter.infer_actions({"step": 0})

    assert out.shape == (4, 3)
    assert out.dtype == np.float32


def test_callable_policy_adapter_accepts_tuple_output():
    adapter = CallablePolicyAdapter(
        lambda obs: ("frames", np.ones((1, 4, 3), dtype=np.float32)),
        output_key=None,
        tuple_index=1,
    )

    out = adapter.infer_actions({"step": 0})

    assert out.shape == (4, 3)


def test_callable_policy_adapter_rejects_bad_shape():
    adapter = CallablePolicyAdapter(lambda obs: {"actions": np.ones((3,))})

    with pytest.raises(ValueError, match="expected action chunk"):
        adapter.infer_actions(None)


def test_async_runner_prefetches_next_chunk():
    calls = []

    def policy(obs):
        calls.append(obs["chunk"])
        base = obs["chunk"] * 10
        return np.arange(base, base + 4, dtype=np.float32)[:, None]

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(target_hz=1000.0, action_horizon=4, start_next_at=2),
    )
    try:
        runner.reset({"chunk": 0})
        assert runner.next_action({"chunk": 1}).item() == 0.0
        assert runner.next_action({"chunk": 1}).item() == 1.0
        time.sleep(0.02)
        action = runner.next_action({"chunk": 2}).item()
        assert action in {2.0, 10.0}
        assert runner.stats.chunks_started >= 1
    finally:
        runner.close()


def test_async_runner_hold_last_on_miss():
    def slow_policy(obs):
        time.sleep(0.03)
        return np.array([[1.0], [2.0]], dtype=np.float32)

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(slow_policy),
        RTCConfig(target_hz=1000.0, action_horizon=2, start_next_at=1),
    )
    try:
        runner.reset({"step": 0})
        assert runner.next_action({"step": 1}).item() == 1.0
        assert runner.next_action({"step": 2}).item() == 2.0
        held = runner.next_action({"step": 3}).item()
        assert held == 2.0
        assert runner.stats.deadline_misses == 1
        assert runner.stats.held_actions == 1
    finally:
        runner.close()
