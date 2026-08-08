"""Fixed-iteration lowering for the semantic OpenPI action schedule.

The matcher deliberately does not import OpenPI or name one of its concrete
classes.  It recognizes the boundary that the host exposes: observation
preprocessing, prefix embedding/cache construction, an iterative denoise
step, and the paired PaliGemma/expert container.  This covers the OpenPI
PyTorch PI action family while an unrelated module is correctly ignored.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import torch

from .protocol import FixedIterationLowering, FixedIterationRefused


def _unwrap(call: Callable[..., Any]) -> Callable[..., Any]:
    """Unwrap ``torch.compile`` while preserving an already-bound method."""
    wrapped = getattr(call, "__wrapped__", call)
    owner = getattr(call, "__self__", None)
    if owner is not None and inspect.isfunction(wrapped):
        return wrapped.__get__(owner, type(owner))
    return wrapped


def _semantic_model(candidate: Any) -> bool:
    required_calls = (
        "_preprocess_observation",
        "_prepare_attention_masks_4d",
        "embed_prefix",
        "denoise_step",
        "sample_actions",
        "sample_noise",
    )
    if not isinstance(candidate, torch.nn.Module):
        return False
    if not all(callable(getattr(candidate, name, None))
               for name in required_calls):
        return False
    config = getattr(candidate, "config", None)
    if not all(hasattr(config, name)
               for name in ("action_horizon", "action_dim")):
        return False
    pair = getattr(candidate, "paligemma_with_expert", None)
    pali = getattr(pair, "paligemma", None)
    expert = getattr(pair, "gemma_expert", None)
    if not callable(getattr(pair, "forward", None)):
        return False
    if getattr(pali, "language_model", None) is None:
        return False
    if getattr(expert, "model", None) is None:
        return False
    try:
        names = tuple(inspect.signature(
            _unwrap(candidate.sample_actions)).parameters)
    except (TypeError, ValueError):
        return False
    return names[:2] == ("device", "observation") \
        and "noise" in names and "num_steps" in names


def _partial_invocation(
    forward: Callable[[], Any],
) -> tuple[Any, Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None:
    if not isinstance(forward, functools.partial):
        return None
    call = forward.func
    model = getattr(call, "__self__", None)
    if not _semantic_model(model):
        return None
    return model, _unwrap(call), tuple(forward.args), dict(forward.keywords or {})


def _record_invocation(
    forward: Callable[[], Any],
    model: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], Any]:
    """Record one call through ``model.sample_actions``, then restore it."""
    original = model.sample_actions
    raw = _unwrap(original)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append((tuple(args), dict(kwargs)))
        return raw(*args, **kwargs)

    owned = "sample_actions" in vars(model)
    try:
        model.sample_actions = record
        with torch.no_grad():
            reference = forward()
    finally:
        if owned:
            model.sample_actions = original
        else:
            delattr(model, "sample_actions")
    if len(calls) != 1:
        raise FixedIterationRefused(
            "fixed_iter: the host forward must call the recognized action "
            f"schedule exactly once, observed {len(calls)} call(s)")
    args, kwargs = calls[0]
    return raw, args, kwargs, reference


def _bind_arguments(
    call: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> inspect.BoundArguments:
    try:
        signature = inspect.signature(call)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound
    except (TypeError, ValueError) as exc:
        raise FixedIterationRefused(
            f"fixed_iter: cannot bind the host schedule arguments: {exc}") \
            from exc


def _attention_masks(pad_masks: torch.Tensor,
                     att_masks: torch.Tensor) -> torch.Tensor:
    """OpenPI's prefix-mask dataflow, expressed without a host import."""
    cumsum = torch.cumsum(att_masks, dim=1)
    causal = cumsum[:, None, :] <= cumsum[:, :, None]
    valid = pad_masks[:, None, :] * pad_masks[:, :, None]
    return causal & valid


def _fixed_forward(
    model: Any,
    *,
    device: Any,
    observation: Any,
    noise: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """The host schedule with only its fixed loop spelling normalized."""
    batch = observation.state.shape[0]
    images, image_masks, tokens, token_masks, state = \
        model._preprocess_observation(observation, train=False)
    prefix, prefix_pad, prefix_ar = model.embed_prefix(
        images, image_masks, tokens, token_masks)
    prefix_mask = _attention_masks(prefix_pad, prefix_ar)
    prefix_positions = torch.cumsum(prefix_pad, dim=1) - 1
    prefix_mask_4d = model._prepare_attention_masks_4d(prefix_mask)

    model.paligemma_with_expert.paligemma.language_model.config \
        ._attn_implementation = "eager"
    _, prefix_kv = model.paligemma_with_expert.forward(
        attention_mask=prefix_mask_4d,
        position_ids=prefix_positions,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )

    dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=device)
    value = noise
    timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(steps):
        velocity = model.denoise_step(
            state,
            prefix_pad,
            prefix_kv,
            value,
            timestep.expand(batch),
        )
        value = value + dt * velocity
        timestep += dt
    return value


def _observation_windows(observation: Any) -> dict[str, torch.Tensor]:
    """Expose every tensor the fixed schedule reads between replays."""
    windows: dict[str, torch.Tensor] = {}
    for attr in (
        "images",
        "image_masks",
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
    ):
        value = getattr(observation, attr, None)
        if torch.is_tensor(value):
            windows[f"observation.{attr}"] = value
        elif isinstance(value, dict):
            for key, tensor in sorted(value.items()):
                if torch.is_tensor(tensor):
                    windows[f"observation.{attr}.{key}"] = tensor
    return windows


class OpenPIFixedIterationAdapter:
    """Normalize a fixed OpenPI action loop without modifying the host."""

    family = "cond_iter_pipeline.openpi_action"

    def lower(
        self,
        forward: Callable[[], Any],
        model: Any | None,
    ) -> FixedIterationLowering | None:
        invocation = _partial_invocation(forward)
        if invocation is not None:
            found, raw, args, kwargs = invocation
            with torch.no_grad():
                reference = raw(*args, **kwargs)
            model = found
        elif _semantic_model(model):
            raw, args, kwargs, reference = _record_invocation(forward, model)
        else:
            return None

        bound = _bind_arguments(raw, args, kwargs)
        device = bound.arguments["device"]
        observation = bound.arguments["observation"]
        noise = bound.arguments.get("noise")
        steps = bound.arguments.get("num_steps")
        if noise is None:
            raise FixedIterationRefused(
                "fixed_iter: noise must be explicit so capture exposes a "
                "replayable SWAP window instead of freezing in-graph RNG")
        if not isinstance(noise, torch.Tensor):
            raise FixedIterationRefused(
                "fixed_iter: explicit noise must be a torch.Tensor")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise FixedIterationRefused(
                f"fixed_iter: num_steps must be one positive fixed int, got "
                f"{steps!r}")

        def lowered() -> torch.Tensor:
            with torch.no_grad():
                return _fixed_forward(
                    model,
                    device=device,
                    observation=observation,
                    noise=noise,
                    steps=steps,
                )

        return FixedIterationLowering(
            forward=lowered,
            reference_output=reference,
            family=self.family,
            steps=steps,
            exact=True,
            windows={"noise": noise, **_observation_windows(observation)},
            details={
                "source_form": "tensor_controlled_while",
                "canonical_form": "fixed_for",
                "noise_window": "explicit",
            },
        )
