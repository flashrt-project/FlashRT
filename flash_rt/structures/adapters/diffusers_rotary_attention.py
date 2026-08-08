"""Rotary Diffusers attention routed through the dense FA2 structure.

The adapter is selected by the processor boundary and module slots it can
reproduce, not by a model or processor class name.  It covers the common
video-transformer form where Q/K are normalised before an optional rotary
embedding and the processor returns a sequence-major attention result.
"""

from __future__ import annotations

import inspect

import torch

from ..impls.attention_core import bind_dense_attention_best


def _compatible_site(module, processor) -> tuple[bool, str]:
    if not callable(processor):
        return False, "processor is not callable"
    try:
        parameters = inspect.signature(processor.__call__).parameters
    except (TypeError, ValueError, AttributeError):
        return False, "processor call signature is not inspectable"
    if "rotary_emb" not in parameters:
        return False, "processor has no rotary_emb boundary"
    required_modules = ("to_q", "to_k", "to_v", "norm_q", "norm_k")
    for attr in required_modules:
        if not isinstance(getattr(module, attr, None), torch.nn.Module):
            return False, f"attention lacks callable slot {attr!r}"
    try:
        out_proj, out_drop = module.to_out[0], module.to_out[1]
    except (AttributeError, IndexError, KeyError, TypeError):
        return False, "attention lacks the to_out[projection, dropout] slots"
    if not all(isinstance(part, torch.nn.Module)
               for part in (out_proj, out_drop)):
        return False, "attention output slots are not modules"
    heads = getattr(module, "heads", None)
    if not isinstance(heads, int) or heads <= 0:
        return False, "attention lacks a positive integer head count"
    if getattr(module, "add_k_proj", None) is not None:
        return False, "added image KV is not yet an executable form"
    return True, ""


def _projections(attn, hidden_states, encoder_hidden_states):
    context = hidden_states if encoder_hidden_states is None \
        else encoder_hidden_states
    if getattr(attn, "fused_projections", False):
        if getattr(attn, "is_cross_attention", False):
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(context).chunk(2, dim=-1)
        else:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(context)
        value = attn.to_v(context)
    return query, key, value


def _apply_rotary(hidden_states, rotary_emb):
    if rotary_emb is None:
        return hidden_states
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _qkv(attn, hidden_states, encoder_hidden_states, rotary_emb):
    query, key, value = _projections(
        attn, hidden_states, encoder_hidden_states)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    query = _apply_rotary(query, rotary_emb)
    key = _apply_rotary(key, rotary_emb)
    return query, key, value


class _Recorder:
    def __init__(self, original, rows):
        self.original = original
        self.rows = rows

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, rotary_emb=None, *args, **kwargs,
    ):
        if attention_mask is None:
            query, key, value = _qkv(
                attn, hidden_states, encoder_hidden_states, rotary_emb)
            row = {
                "q": query.transpose(1, 2).detach(),
                "key": key.transpose(1, 2).detach(),
                "value": value.transpose(1, 2).detach(),
                "mask": None,
            }
            if self.rows:
                first = self.rows[0]
                expected = tuple(
                    (tuple(first[name].shape), first[name].dtype)
                    for name in ("q", "key", "value"))
                got = tuple(
                    (tuple(row[name].shape), row[name].dtype)
                    for name in ("q", "key", "value"))
                if got != expected:
                    raise ValueError(
                        "attention_core rotary: shape or dtype moved within "
                        f"one calibration call: {expected} -> {got}")
            else:
                # Binding needs one real device sample.  Subsequent calls only
                # qualify the stable signature; retaining every denoise-step
                # activation would turn calibration length into VRAM usage.
                self.rows.append(row)
        else:
            self.rows.append({"mask": attention_mask.detach()})
        return self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            rotary_emb, *args, **kwargs)


class _FlashRTRotaryAttnProcessor:
    def __init__(self, core, original):
        self.core = core
        self.original = original

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, rotary_emb=None, *args, **kwargs,
    ):
        if attention_mask is not None:
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                rotary_emb, *args, **kwargs)
        query, key, value = _qkv(
            attn, hidden_states, encoder_hidden_states, rotary_emb)
        projection_dtype = query.dtype
        guard = getattr(self.core, "_frt_guard", None)
        accepted_dtypes = tuple(getattr(guard, "dtypes", ()) or ())
        if accepted_dtypes and projection_dtype not in accepted_dtypes:
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                rotary_emb, *args, **kwargs)
        hidden_states = self.core(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class DiffusersRotaryAttentionAdapter:
    """Route capability-compatible rotary processors through Hub FA2."""

    __name__ = "diffusers_rotary_attention"

    def __call__(self, model, forward, *, prefix_cadence: bool = False):
        del prefix_cadence
        sites = []
        for path, module in model.named_modules():
            processor = getattr(module, "processor", None)
            compatible, _ = _compatible_site(module, processor)
            if compatible:
                sites.append((path, module, processor))
        if not sites:
            return None

        captures = [[] for _ in sites]
        for (_, module, original), rows in zip(sites, captures):
            module.processor = _Recorder(original, rows)
        try:
            with torch.no_grad():
                forward()
        finally:
            for _, module, original in sites:
                module.processor = original

        refused = []
        routes = []
        observed = {}
        variants = {}
        for (path, module, original), rows in zip(sites, captures):
            if not rows:
                refused.append((
                    f"{path}.processor",
                    "attention_core rotary: compatible processor was not "
                    "called during calibration",
                ))
                continue
            if any(row.get("mask") is not None for row in rows):
                refused.append((
                    f"{path}.processor",
                    "attention_core rotary: live masks are outside the "
                    "unmasked executable form",
                ))
                continue
            core = bind_dense_attention_best(rows)
            if core is None:
                refused.append((
                    f"{path}.processor",
                    "attention_core rotary: Hub FA2 does not cover the "
                    "captured head dimension",
                ))
                continue
            routed = _FlashRTRotaryAttnProcessor(core, original)
            routes.append((module, original, routed))
            observed[f"{path}.processor::fa2_core"] = core
            variants[f"{path}.processor"] = {
                "bound": getattr(core, "_frt_variant", "fa2"),
                "superseded": list(
                    getattr(core, "_frt_variant_trail", ())),
            }
        if not routes:
            return {}, None, {"refused": refused}

        def enable():
            for module, _, routed in routes:
                module.processor = routed

        def disable():
            for module, original, _ in routes:
                module.processor = original

        enable()
        return {}, None, {
            "revert": [disable],
            "observed": observed,
            "toggle": (enable, disable),
            "refused": refused,
            "attention_variants": variants,
        }
