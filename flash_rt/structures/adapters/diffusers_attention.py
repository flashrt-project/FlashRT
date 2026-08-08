"""Attention adapter for capability-compatible Diffusers attention hosts."""

from __future__ import annotations

import torch

from ..impls.attention_core import bind_dense_attention_best


def _compatible_site(module, processor) -> tuple[bool, str]:
    """Whether ``module`` exposes the processor contract reproduced below.

    Processor class names are deliberately irrelevant.  The adapter owns the
    projection/output dataflow, so it admits only sites exposing every slot it
    reads and a callable processor that accepts the ordinary Diffusers
    ``(attention, hidden_states, ...)`` boundary.
    """
    if not callable(processor):
        return False, "processor is not callable"
    for attr in ("to_q", "to_k", "to_v"):
        if not isinstance(getattr(module, attr, None), torch.nn.Module):
            return False, f"attention lacks callable projection slot {attr!r}"
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
    required_state = (
        "spatial_norm", "group_norm", "norm_cross", "norm_q", "norm_k",
        "residual_connection", "rescale_output_factor",
    )
    missing = [name for name in required_state if not hasattr(module, name)]
    if missing:
        return False, f"attention lacks processor state {missing}"
    if (getattr(module, "norm_cross", False)
            and not callable(getattr(module, "norm_encoder_hidden_states",
                                     None))):
        return False, "cross normalization is enabled but has no callable"
    return True, ""


def _qkv(attn, hidden_states, encoder_hidden_states, attention_mask, temb):
    """Reproduce the capability-compatible Diffusers projection half."""
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    if hidden_states.ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width).transpose(1, 2)
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(
            batch_size, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(
            hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    return query, key, value, attention_mask


class _Recorder:
    def __init__(self, original, rows):
        self.original = original
        self.rows = rows

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        query, key, value, mask = _qkv(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        self.rows.append({
            "q": query.detach(),
            "key": key.detach(),
            "value": value.detach(),
            "mask": mask.detach() if mask is not None else None,
        })
        return self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb,
            *args, **kwargs)


class _FlashRTDenseAttnProcessor:
    """Diffusers processor with only the SDPA body replaced by FA2."""

    def __init__(self, core, original):
        self.core = core
        self.original = original

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        # A live mask is served only by a core that baked this site's
        # fixed mask pattern at bind time (packed ranges); a maskless
        # core keeps the host path rather than reuse a frozen mask.
        if attention_mask is not None and not getattr(
                self.core, "allowed_ranges", ()):
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                temb, *args, **kwargs)
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
        query, key, value, mask = _qkv(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        projection_dtype = query.dtype
        guard = getattr(self.core, "_frt_guard", None)
        accepted_dtypes = tuple(getattr(guard, "dtypes", ()) or ())
        if accepted_dtypes and projection_dtype not in accepted_dtypes:
            # An upstream composition can change the effective projection
            # dtype after this adapter calibrated.  A hidden cast here would
            # silently change the declared boundary and add hot-path work.
            # Keep the host path; the core's zero call count makes the missed
            # route visible to the final-form gate.
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                temb, *args, **kwargs)
        hidden_states = self.core(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            query.shape[0], -1, attn.heads * query.shape[-1])
        hidden_states = hidden_states.to(projection_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class DiffusersAttentionAdapter:
    """Route called Diffusers SDPA processors through stateless Hub FA2."""

    __name__ = "diffusers_attention"

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

        refused = []
        captures = [[] for _ in sites]
        for (_, module, original), rows in zip(sites, captures):
            module.processor = _Recorder(original, rows)
        try:
            with torch.no_grad():
                forward()
        finally:
            for _, module, original in sites:
                module.processor = original

        routes = []
        observed = {}
        variants = {}
        for (path, module, original), rows in zip(sites, captures):
            if not rows:
                refused.append((
                    f"{path}.processor",
                    "attention_core dense: compatible processor was not "
                    "called during calibration",
                ))
                continue
            core = bind_dense_attention_best(rows)
            if core is None:
                refused.append((
                    f"{path}.processor",
                    "attention_core dense: published Hub artifact does not "
                    "cover the captured head dimension or mask form",
                ))
                continue
            routed = _FlashRTDenseAttnProcessor(core, original)
            routes.append((module, original, routed))
            observed[f"{path}.processor::fa2_core"] = core
            variants[f"{path}.processor"] = {
                "bound": getattr(core, "_frt_variant", "fa2"),
                "superseded": list(
                    getattr(core, "_frt_variant_trail", ())),
            }
        if not routes:
            return {}, None, {"refused": refused}

        def enable() -> None:
            for module, _, routed in routes:
                module.processor = routed

        def disable() -> None:
            for module, original, _ in routes:
                module.processor = original

        def revert() -> None:
            disable()

        enable()
        return {}, None, {
            "revert": [revert],
            "observed": observed,
            "toggle": (enable, disable),
            "refused": refused,
            "attention_variants": variants,
        }
