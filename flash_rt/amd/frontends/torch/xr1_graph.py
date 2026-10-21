"""Instance-owned, fixed-profile HIP Graph for the upstream XR-1 forward.

The frontend supplies CPU processor outputs. Token/layout metadata is prepared
once; image, state, action-mask, prefix and noise buffers are refreshed before
every replay. The original vision/VLM/DiT topology and native adapters execute
during capture. Temporary method bindings remove host metadata/RNG operations
from that capture, without modifying checkpoint files or module globals.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import CellType, FunctionType, MethodType
import time

import torch


_METADATA = {"input_ids", "attention_mask", "image_grid_thw", "position_ids"}
_INPUTS = _METADATA | {"pixel_values", "state", "action_mask", "action", "prefix_length"}
_MISSING = object()


def _profile(inputs, num_steps, embodiment):
    """Freeze all Python control-flow inputs, including same-length new prompts."""
    unknown = set(inputs) - _INPUTS
    if unknown:
        raise ValueError(f"XR-1 full graph does not support input keys {sorted(unknown)}")
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("XR-1 full graph requires batch_size=1")
    key = [num_steps, embodiment]
    for name, value in sorted(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("XR-1 graph preparation requires CPU processor outputs")
            metadata = value.contiguous().numpy().tobytes() if name in _METADATA else None
            key.append((name, tuple(value.shape), str(value.dtype), metadata))
        elif name == "prefix_length" and type(value) is int:
            key.append((name, value))
        else:
            raise ValueError(f"Unsupported XR-1 graph input {name!r}")
    return tuple(key)


def _bind_helpers(function, replacements):
    """Bind helpers on a function copy, preserving its body and decorators.

    The checkpoint's RNG bookkeeping and the Transformers mask factory are
    Python helpers used inside the model forward. Binding them per instance
    preserves one semantic forward without process-wide torch monkeypatches.
    Decorator closure cells are copied as well, never changed in place.
    """
    wrapped = getattr(function, "__wrapped__", None)
    closure = function.__closure__
    if wrapped is not None:
        child = _bind_helpers(wrapped, replacements)
        closure = tuple(CellType(child) if cell.cell_contents is wrapped else cell
                        for cell in (closure or ())) or None
    copied = FunctionType(function.__code__, {**function.__globals__, **replacements},
                          function.__name__, function.__defaults__, closure)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


@contextmanager
def _temporary_bindings(bindings):
    previous = []
    try:
        for owner, name, value in bindings:
            previous.append((owner, name, vars(owner).get(name, _MISSING)))
            setattr(owner, name, value)
        yield
    finally:
        for owner, name, value in reversed(previous):
            if value is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


class _PreparedCudaRandom:
    """The host already owns noise generation and caller RNG preservation."""
    def __getattr__(self, name):
        return getattr(torch.cuda, name)

    def get_rng_state(self, *args):
        return None

    def set_rng_state(self, *args):
        return None


class _PreparedRandom:
    def __init__(self, noise):
        self.noise = noise
        self.cuda = _PreparedCudaRandom()

    def __getattr__(self, name):
        return getattr(torch, name)

    def get_rng_state(self):
        return None

    def set_rng_state(self, *args):
        return None

    def manual_seed(self, *args):
        return None

    def randn_like(self, template):
        if (template.shape != self.noise.shape or template.dtype != self.noise.dtype
                or template.device != self.noise.device):
            raise ValueError("XR-1 graph noise does not match the action profile")
        return self.noise


class XR1FullGraph:
    """One bounded graph profile per frontend; setup changes rebuild explicitly.

    A replay refreshes the full vision/VLM context. Static metadata may include
    the prompt, but cached image features and K/V are never reused across calls.
    Frontend instances are used sequentially, as in the eager XR-1 contract.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=self.device)
        self.capture_count = 0
        self.replay_count = 0
        self.last_capture_ms = 0.0
        self.key = None
        self.graph = None
        self.inputs = {}
        self.constants = []
        self.noise = None
        self.output = None
        self.stream = None

    def clear(self):
        # A profile change releases the old graph only at a safe point.
        if self.graph is not None:
            torch.cuda.synchronize(self.device)
        self.graph = self.output = self.noise = self.stream = None
        self.inputs = {}
        self.constants = []
        self.key = None

    def _set_noise(self, seed):
        self.generator.manual_seed(seed)
        self.noise.normal_(generator=self.generator)

    def _prepare_bindings(self, cpu_inputs):
        vlm = self.model.vlm.model
        vision = vlm.visual
        language = vlm.language_model
        ids = cpu_inputs["input_ids"]
        # The public image-policy prompt has no training-only modality tokens.
        if hasattr(self.model, "action_shape"):
            from flash_rt.amd.models.xr1.upstream.qwen3vl import (
                STATE_ID, ACTION_START_ID, ACTION_END_ID, SCORE_ID)
            if ((ids == STATE_ID) | (ids == SCORE_ID)
                    | ((ids >= ACTION_START_ID) & (ids < ACTION_END_ID))).any():
                raise ValueError("XR-1 full graph does not support training-only prompt tokens")
        grid = cpu_inputs["image_grid_thw"].clone()
        pos = vision.fast_pos_embed_interpolate(grid)
        rotary = vision.rot_pos_emb(grid)
        image_mask = (ids == vlm.config.image_token_id).to(self.device)
        video_mask = (ids == vlm.config.video_token_id).to(self.device)
        indices = (ids == vlm.config.image_token_id).flatten().nonzero().flatten().to(self.device)
        if "position_ids" not in self.inputs:
            positions, _ = vlm.get_rope_index(
                self.inputs["input_ids"], self.inputs["image_grid_thw"],
                attention_mask=self.inputs["attention_mask"])
            self.inputs["position_ids"] = positions
        positions = self.inputs["position_ids"]

        from transformers.masking_utils import create_causal_mask
        causal = create_causal_mask(
            config=language.config,
            input_embeds=torch.empty(1, ids.shape[1], language.config.hidden_size,
                                     device=self.device, dtype=torch.bfloat16),
            attention_mask=self.inputs["attention_mask"],
            cache_position=torch.arange(ids.shape[1], device=self.device),
            past_key_values=None, position_ids=positions[0])
        self.constants = [pos, rotary, image_mask, video_mask, indices, causal]
        # Vision splitting and image sizes are host metadata, not replay inputs.
        self.inputs["image_grid_thw"] = grid

        def placeholder(input_ids, inputs_embeds, image_features=None, video_features=None):
            return (image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    video_mask.unsqueeze(-1).expand_as(inputs_embeds))

        def deepstack(hidden_states, visual_pos_masks, visual_embeds):
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            value = flat.index_select(0, indices).clone() + visual_embeds.to(hidden_states.dtype)
            flat.index_copy_(0, indices, value)
            return hidden_states

        original_vlm_forward = vlm.forward

        def vlm_forward(*args, **kwargs):
            input_ids = kwargs.pop("input_ids")
            kwargs["inputs_embeds"] = vlm.get_input_embeddings()(input_ids)
            return original_vlm_forward(*args, input_ids=None, **kwargs)

        text_forward = _bind_helpers(language.forward.__func__,
                                     {"create_causal_mask": lambda **kwargs: causal})
        policy_forward = _bind_helpers(self.model.forward.__func__,
                                       {"torch": _PreparedRandom(self.noise)})
        return [
            (vision, "fast_pos_embed_interpolate", lambda grid_thw: pos),
            (vision, "rot_pos_emb", lambda grid_thw: rotary),
            (vlm, "get_placeholder_mask", placeholder),
            (language, "_deepstack_process", deepstack),
            (language, "forward", MethodType(text_forward, language)),
            (vlm, "forward", vlm_forward),
            (self.model, "forward", MethodType(policy_forward, self.model)),
        ]

    def _call(self, seed, num_steps, embodiment):
        if hasattr(self.model, "action_shape"):
            self.model.num_steps = num_steps
            return self.model.generate(dict(self.inputs))
        return self.model(**self.inputs, seed=seed, num_steps=num_steps,
                          task_id=embodiment).actions

    def _capture(self, cpu_inputs, key, seed, num_steps, embodiment):
        self.clear()
        started = time.perf_counter()
        try:
            self.inputs = {
                name: value.to(device=self.device, dtype=torch.bfloat16
                               if value.is_floating_point() else value.dtype).clone()
                if isinstance(value, torch.Tensor) else value
                for name, value in cpu_inputs.items()
            }
            # Validate against the unchanged production forward, including RNG.
            expected = self.pipeline.forward(self.inputs, seed=seed,
                num_steps=num_steps, embodiment=embodiment).actions.clone()
            self.noise = torch.empty_like(self.inputs["action_mask"])
            self._set_noise(seed)
            bindings = self._prepare_bindings(cpu_inputs)
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with _temporary_bindings(bindings):
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        self._call(seed, num_steps, embodiment)
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.output = self._call(seed, num_steps, embodiment)
            self.graph.replay()
            torch.cuda.synchronize(self.device)
            if not torch.equal(self.output, expected):
                delta = float((self.output.float() - expected.float()).abs().max())
                raise RuntimeError(f"XR-1 full graph failed exact eager parity (max_abs={delta})")
            self.key = key
            self.capture_count += 1
            self.last_capture_ms = (time.perf_counter() - started) * 1000
        except Exception:
            self.clear()
            raise

    @torch.inference_mode()
    def run(self, cpu_inputs, *, seed, num_steps, embodiment):
        key = _profile(cpu_inputs, num_steps, embodiment)
        if key != self.key:
            self._capture(cpu_inputs, key, seed, num_steps, embodiment)
        else:
            for name, value in cpu_inputs.items():
                if isinstance(value, torch.Tensor) and name not in _METADATA:
                    self.inputs[name].copy_(value)
            self._set_noise(seed)
            self.graph.replay()
        self.replay_count += 1
        return self.output
