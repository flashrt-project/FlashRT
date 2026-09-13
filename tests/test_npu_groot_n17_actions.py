"""The GR00T N1.7 action decode and state encode, against the contract they
invert.

Both take the checkpoint's own normalisation parameters from the processor and
both take a ``device``, so a fake processor and ``device="cpu"`` exercise all of
the arithmetic without an Ascend part. That arithmetic is where a silently
different trajectory would live: an unclamped block, a relative pose composed in
the wrong order, or a modality read in the wrong order.
"""
import math
import types

import pytest
import torch

from flash_rt.npu.models.groot_n17.actions import (
    ActionDecoder,
    StateEncoder,
    rot6d_to_matrix,
)


def _config(keys, *, delta=4, reps=None, types_=None, formats=None,
            mean_std=(), state_keys=None):
    configs = None
    if reps is not None:
        configs = [types.SimpleNamespace(
            rep=f"Rep.{reps[i]}", type=f"Type.{types_[i]}",
            format=f"Format.{(formats or ['XYZ_ROT6D'] * len(keys))[i]}",
            state_key=(state_keys or [None] * len(keys))[i]) for i in range(len(keys))]
    return types.SimpleNamespace(modality_keys=list(keys),
                                 delta_indices=list(range(delta)),
                                 action_configs=configs,
                                 mean_std_embedding_keys=list(mean_std))


def _processor(action_config, action_params, *, relative=True,
               state_config=None, state_params=None):
    return types.SimpleNamespace(state_action_processor=types.SimpleNamespace(
        use_relative_action=relative,
        modality_configs={"tag": {"action": action_config, "state": state_config}},
        norm_params={"tag": {"action": action_params, "state": state_params}}))


def _bounds(dim, low=-1.0, high=1.0):
    return {"dim": [dim], "min": [low] * dim, "max": [high] * dim}


# ── the decode contract ───────────────────────────────────────────────

def test_an_absolute_modality_is_the_inverse_of_the_normalisation():
    processor = _processor(
        _config(["joint"], reps=["ABSOLUTE"], types_=["JOINT"]),
        {"joint": _bounds(3, 0.0, 10.0)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    normalized = torch.tensor([[[-1.0, 0.0, 1.0]] * 4])
    out = decoder(normalized, {"joint": [[0.0, 0.0, 0.0]]})
    assert out["joint"].shape == (1, 4, 3)
    assert torch.allclose(out["joint"][0, 0], torch.tensor([0.0, 5.0, 10.0]))


def test_a_value_outside_the_normalised_range_is_clamped_not_extrapolated():
    processor = _processor(
        _config(["joint"], reps=["ABSOLUTE"], types_=["JOINT"]),
        {"joint": _bounds(1, 0.0, 10.0)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    out = decoder(torch.tensor([[[5.0]] * 4]), {"joint": [[0.0]]})
    assert float(out["joint"][0, 0, 0]) == pytest.approx(10.0)


def test_a_relative_joint_modality_adds_the_last_state_timestep():
    processor = _processor(
        _config(["joint"], reps=["RELATIVE"], types_=["JOINT"]),
        {"joint": _bounds(2, -1.0, 1.0)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    out = decoder(torch.zeros(1, 4, 2), {"joint": [[1.0, 1.0], [7.0, 9.0]]})
    assert torch.allclose(out["joint"][0, 0], torch.tensor([7.0, 9.0]))


def test_a_relative_end_effector_pose_is_composed_onto_the_reference_frame():
    """``T_ref @ T_rel``, and the identity relative pose has to come back as the
    reference exactly -- which is also what catches a transposed rotation."""
    processor = _processor(
        _config(["eef"], reps=["RELATIVE"], types_=["EEF"]),
        {"eef": _bounds(9, -1.0, 1.0)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    identity = identity.repeat(4, 1).unsqueeze(0) * 0.0
    identity[..., 3] = 1.0        # first rotation row = x
    identity[..., 7] = 1.0        # second rotation row = y
    angle = math.pi / 2
    reference = [2.0, 3.0, 4.0,
                 math.cos(angle), math.sin(angle), 0.0,
                 -math.sin(angle), math.cos(angle), 0.0]
    out = decoder(identity, {"eef": [reference]})
    assert torch.allclose(out["eef"][0, 0, :3], torch.tensor([2.0, 3.0, 4.0]),
                          atol=1e-6)
    assert torch.allclose(out["eef"][0, 0, 3:], torch.tensor(reference[3:]), atol=1e-6)


def test_the_modalities_are_laid_out_in_the_order_the_checkpoint_declares():
    processor = _processor(
        _config(["a", "b"], reps=["ABSOLUTE", "ABSOLUTE"], types_=["JOINT", "JOINT"]),
        {"a": _bounds(2, 0.0, 2.0), "b": _bounds(1, 0.0, 4.0)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    assert decoder.action_dim == 3
    out = decoder(torch.tensor([[[1.0, -1.0, 1.0]] * 4]),
                  {"a": [[0.0, 0.0]], "b": [[0.0]]})
    assert torch.allclose(out["a"][0, 0], torch.tensor([2.0, 0.0]))
    assert float(out["b"][0, 0, 0]) == pytest.approx(4.0)


def test_a_horizon_shorter_than_the_checkpoints_is_refused():
    processor = _processor(
        _config(["joint"], reps=["ABSOLUTE"], types_=["JOINT"]),
        {"joint": _bounds(1)})
    decoder = ActionDecoder(processor, "tag", device="cpu")
    with pytest.raises(ValueError, match="4-step horizon"):
        decoder(torch.zeros(1, 2, 1), {"joint": [[0.0]]})


@pytest.mark.parametrize("rep,type_,fmt,match", [
    ("ABSOLUTE", "EEF", "XYZ_ROT6D", "has not been validated"),
    ("RELATIVE", "EEF", "XYZ_QUAT", "translation-and-6D-rotation"),
])
def test_a_representation_this_decoder_has_not_been_validated_for_is_refused(
        rep, type_, fmt, match):
    processor = _processor(_config(["eef"], reps=[rep], types_=[type_], formats=[fmt]),
                           {"eef": _bounds(9)})
    with pytest.raises(NotImplementedError, match=match):
        ActionDecoder(processor, "tag", device="cpu")


def test_a_processor_with_relative_actions_disabled_is_refused():
    processor = _processor(_config(["joint"], reps=["ABSOLUTE"], types_=["JOINT"]),
                           {"joint": _bounds(1)}, relative=False)
    with pytest.raises(NotImplementedError, match="relative-action contract"):
        ActionDecoder(processor, "tag", device="cpu")


def test_a_modality_reading_another_modalitys_state_is_refused():
    processor = _processor(
        _config(["eef"], reps=["RELATIVE"], types_=["EEF"], state_keys=["other"]),
        {"eef": _bounds(9)})
    with pytest.raises(NotImplementedError, match="references state"):
        ActionDecoder(processor, "tag", device="cpu")


def test_a_missing_representation_is_refused_rather_than_defaulted():
    config = _config(["a"], reps=["ABSOLUTE"], types_=["JOINT"])
    config.modality_keys = ["a", "b"]        # one representation for two modalities
    processor = _processor(config, {"a": _bounds(1), "b": _bounds(1)})
    with pytest.raises(NotImplementedError, match="every action modality"):
        ActionDecoder(processor, "tag", device="cpu")


# ── the encode contract ───────────────────────────────────────────────

def _state_processor(keys, params, *, mean_std=()):
    return _processor(_config(["joint"], reps=["ABSOLUTE"], types_=["JOINT"]),
                      {"joint": _bounds(1)},
                      state_config=_config(keys, mean_std=mean_std),
                      state_params=params)


def test_the_state_vector_is_normalised_to_minus_one_to_one_and_zero_padded():
    processor = _state_processor(["a", "b"],
                                {"a": _bounds(2, 0.0, 10.0), "b": _bounds(1, -4.0, 4.0)})
    encoder = StateEncoder(processor, "tag", 8, device="cpu")
    out = encoder({"a": [0.0, 10.0], "b": [0.0]})
    assert out.shape == (1, 1, 8)
    assert torch.allclose(out[0, 0, :3], torch.tensor([-1.0, 1.0, 0.0]))
    assert torch.all(out[0, 0, 3:] == 0.0), "the pad has to stay zero"
    assert encoder.used == 3


def test_a_state_modality_of_the_wrong_width_is_refused():
    processor = _state_processor(["a"], {"a": _bounds(3, 0.0, 1.0)})
    encoder = StateEncoder(processor, "tag", 8, device="cpu")
    with pytest.raises(ValueError, match="declares 3"):
        encoder({"a": [0.0, 1.0]})


def test_a_state_wider_than_the_action_head_is_refused():
    processor = _state_processor(["a"], {"a": _bounds(9, 0.0, 1.0)})
    with pytest.raises(ValueError, match="wider than"):
        StateEncoder(processor, "tag", 4, device="cpu")


def test_a_mean_std_normalised_checkpoint_is_refused():
    processor = _state_processor(["a"], {"a": _bounds(1)}, mean_std=["a"])
    with pytest.raises(NotImplementedError, match="mean and standard"):
        StateEncoder(processor, "tag", 8, device="cpu")


# ── the rotation helper ───────────────────────────────────────────────

def test_rot6d_is_orthonormal_and_right_handed():
    torch.manual_seed(0)
    matrix = rot6d_to_matrix(torch.randn(5, 6))
    identity = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(matrix @ matrix.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(matrix), torch.ones(5), atol=1e-5)


def test_rot6d_takes_the_six_values_as_the_first_two_rows():
    matrix = rot6d_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)
