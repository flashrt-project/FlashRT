import numpy as np
import pytest
from groot_compare.metrics import error_metrics, action_array, ACTION_KEYS
from groot_compare.common import shared_noise


def test_per_dimension_catches_error_hidden_by_cosine():
    reference = np.ones((16, 7))
    reference[:, :3] = 1000
    candidate = reference.copy()
    candidate[:, 6] = 0
    result = error_metrics(candidate, reference)
    assert result["cosine"] > .999
    assert result["per_dimension"]["gripper"]["max_abs"] == 1
    assert result["gripper_disagreement_at_0_5"] == 1


def test_action_order_is_not_dictionary_order():
    actions = {k: np.full((1, 16, 1), ACTION_KEYS.index(k)) for k in reversed(ACTION_KEYS)}
    assert np.array_equal(action_array(actions)[0], np.arange(7))


def test_invalid_actions_fail():
    with pytest.raises(ValueError):
        error_metrics(np.full((8, 7), np.nan), np.ones((8, 7)))
    with pytest.raises(ValueError):
        error_metrics(np.ones((7, 7)), np.ones((8, 7)))
    assert error_metrics(np.zeros((8, 7)), np.zeros((8, 7)))["cosine"] is None


def test_noise_is_case_stable_and_already_bfloat16():
    first = shared_noise("episode1/frame8")
    shared_noise("unrelated-warmup")
    assert np.array_equal(first, shared_noise("episode1/frame8"))
    assert not np.array_equal(first, shared_noise("episode1/frame16"))
    assert first.shape == (1, 40, 132)
    assert np.all((first.view(np.uint32) & 0xFFFF) == 0)
