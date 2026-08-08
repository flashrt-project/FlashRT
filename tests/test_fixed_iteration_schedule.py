"""Public CPU contracts for fixed-iteration schedule normalization."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

from flash_rt.structures.impls.fixed_iter import (
    FixedIterationRefused,
    normalize_fixed_iteration,
)


class _Pair(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(_attn_implementation="eager")
        self.paligemma = SimpleNamespace(
            language_model=SimpleNamespace(config=config))
        self.gemma_expert = SimpleNamespace(
            model=SimpleNamespace(config=config))

    def forward(self, *, inputs_embeds, **_kwargs):
        prefix = inputs_embeds[0]
        return None, (prefix.mean(dim=1, keepdim=True),)


class _Observation:
    def __init__(self) -> None:
        self.state = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        self.image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)


class _OpenPIShape(torch.nn.Module):
    """Semantic host double; the class name is intentionally unrelated."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_horizon=3, action_dim=4)
        self.paligemma_with_expert = _Pair()

    def sample_noise(self, shape, device):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def _preprocess_observation(self, observation, *, train):
        del train
        batch = observation.state.shape[0]
        mask = torch.ones((batch,), dtype=torch.bool)
        tokens = torch.ones((batch, 2), dtype=torch.int64)
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        return [observation.image], [mask], tokens, token_mask, observation.state

    def embed_prefix(self, images, image_masks, tokens, token_masks):
        del tokens, token_masks
        prefix = images[0]
        batch, rows = prefix.shape[:2]
        pad = image_masks[0][:, None].expand(batch, rows)
        attention = torch.zeros((batch, rows), dtype=torch.int64)
        return prefix, pad, attention

    def _prepare_attention_masks_4d(self, mask):
        return mask[:, None, :, :]

    def denoise_step(self, state, prefix_pad, prefix_kv, value, timestep):
        del prefix_pad, prefix_kv
        return value * 0.125 + state[:, None, :] * 0.01 \
            + timestep[:, None, None] * 0.001

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        batch = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch, self.config.action_horizon, self.config.action_dim),
                device,
            )
        images, image_masks, tokens, token_masks, state = \
            self._preprocess_observation(observation, train=False)
        prefix, prefix_pad, prefix_ar = self.embed_prefix(
            images, image_masks, tokens, token_masks)
        cumsum = torch.cumsum(prefix_ar, dim=1)
        prefix_mask = (cumsum[:, None, :] <= cumsum[:, :, None]) \
            & (prefix_pad[:, None, :] * prefix_pad[:, :, None])
        positions = torch.cumsum(prefix_pad, dim=1) - 1
        _, prefix_kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_mask),
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        value = noise
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        while timestep >= -dt / 2:
            value = value + dt * self.denoise_step(
                state, prefix_pad, prefix_kv, value,
                timestep.expand(batch))
            timestep += dt
        return value


def _fixture():
    model = _OpenPIShape().eval()
    observation = _Observation()
    noise = torch.linspace(-1, 1, 12).reshape(1, 3, 4)
    return model, observation, noise


def test_tensor_while_partial_lowers_to_exact_fixed_iteration():
    model, observation, noise = _fixture()
    forward = functools.partial(
        model.sample_actions,
        "cpu",
        observation,
        noise=noise,
        num_steps=10,
    )
    lowering = normalize_fixed_iteration(forward)
    assert lowering is not None
    assert lowering.steps == 10
    assert lowering.family == "cond_iter_pipeline.openpi_action"
    assert set(lowering.windows) == {"noise", "observation.state"}
    assert torch.equal(lowering.forward(), lowering.reference_output)


def test_tensor_while_is_recorded_from_zero_arg_host_forward():
    model, observation, noise = _fixture()
    forward = lambda: model.sample_actions(
        "cpu", observation, noise=noise, num_steps=10)
    lowering = normalize_fixed_iteration(forward, model)
    assert lowering is not None
    assert torch.equal(lowering.forward(), lowering.reference_output)
    # Probing is transactional: the original bound method is restored.
    assert model.sample_actions.__func__ is _OpenPIShape.sample_actions


def test_missing_explicit_noise_refuses_after_family_recognition():
    model, observation, _ = _fixture()
    forward = functools.partial(
        model.sample_actions, "cpu", observation, num_steps=10)
    with pytest.raises(FixedIterationRefused, match="noise must be explicit"):
        normalize_fixed_iteration(forward)


def test_probe_restores_host_method_when_forward_raises():
    model, observation, noise = _fixture()
    original = model.sample_actions

    def broken_forward():
        model.sample_actions(
            "cpu", observation, noise=noise, num_steps=10)
        raise RuntimeError("host failure")

    with pytest.raises(RuntimeError, match="host failure"):
        normalize_fixed_iteration(broken_forward, model)
    assert model.sample_actions == original


def test_graph_safe_for_host_is_left_on_existing_capture_path():
    class StaticFor(torch.nn.Module):
        def forward(self, value):
            for _ in range(4):
                value = value + 1
            return value

    host = StaticFor()
    forward = functools.partial(host, torch.zeros(1))
    assert normalize_fixed_iteration(forward, host) is None
    assert torch.equal(forward(), torch.tensor([4.0]))


def test_incidental_sample_actions_name_is_not_a_semantic_match():
    class Unrelated(torch.nn.Module):
        def sample_actions(self, device, observation, noise=None, num_steps=10):
            del device, observation, noise, num_steps
            return torch.ones(1)

    host = Unrelated()
    forward = functools.partial(host.sample_actions, "cpu", object())
    assert normalize_fixed_iteration(forward, host) is None
