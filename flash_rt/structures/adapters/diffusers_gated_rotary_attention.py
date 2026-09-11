"""Attention adapter for gated dual-rotary Diffusers attention hosts.

The audio+video joint-transformer form: Q and K are RMS-normalised after
projection, rotary embeddings arrive as *separate* query/key boundaries
(cross-modal sites rotate each side with its own table), the attention
output may pass a per-head sigmoid gate computed from the pre-attention
hidden states, and the processor owns the whole half — there is no
residual or rescale state on the attention module itself. The stock
Diffusers adapter refuses this family (its processor-state contract does
not exist here), so the family gets its own adapter with the same shape:
reproduce the projection half faithfully, capture real Q/K/V at every
called site, bind the dense attention family per site, and replace only
the attention dispatch. Projections, norms, rope, gating, and the output
projection remain the host's own modules.
"""

from __future__ import annotations

import inspect

import torch

from ..impls.attention_core import bind_dense_attention_best


def _compatible_site(module, processor) -> tuple[bool, str]:
    """Whether ``module`` exposes the gated dual-rotary processor contract."""
    if not callable(processor):
        return False, "processor is not callable"
    try:
        parameters = inspect.signature(processor.__call__).parameters
    except (TypeError, ValueError, AttributeError):
        return False, "processor call signature is not inspectable"
    for name in ("query_rotary_emb", "key_rotary_emb"):
        if name not in parameters:
            return False, f"processor has no {name!r} boundary"
    for attr in ("to_q", "to_k", "to_v", "norm_q", "norm_k"):
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
    if not hasattr(module, "to_gate_logits"):
        return False, "attention lacks the gate-logits slot"
    if getattr(module, "rope_type", None) not in ("interleaved", "split"):
        return False, "attention rope type is not a recognised form"
    return True, ""


def _apply_rope(attn, query, key, query_rotary_emb, key_rotary_emb):
    if query_rotary_emb is None:
        return query, key
    from diffusers.models.transformers.transformer_ltx2 import (
        apply_interleaved_rotary_emb, apply_split_rotary_emb)
    k_rope = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
    apply = (apply_interleaved_rotary_emb if attn.rope_type == "interleaved"
             else apply_split_rotary_emb)
    return apply(query, query_rotary_emb), apply(key, k_rope)


def _qkv(attn, hidden_states, encoder_hidden_states,
         query_rotary_emb, key_rotary_emb):
    """Reproduce the host projection half; return SDPA-layout Q/K/V."""
    context = (hidden_states if encoder_hidden_states is None
               else encoder_hidden_states)
    query = attn.norm_q(attn.to_q(hidden_states))
    key = attn.norm_k(attn.to_k(context))
    value = attn.to_v(context)
    query, key = _apply_rope(attn, query, key, query_rotary_emb,
                             key_rotary_emb)
    head_dim = query.shape[-1] // attn.heads
    query = query.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
    key = key.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
    value = value.unflatten(2, (attn.heads, head_dim)).transpose(1, 2)
    return query, key, value


class _Recorder:
    def __init__(self, original, rows):
        self.original = original
        self.rows = rows

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, query_rotary_emb=None,
                 key_rotary_emb=None, *args, **kwargs):
        query, key, value = _qkv(
            attn, hidden_states, encoder_hidden_states,
            query_rotary_emb, key_rotary_emb)
        self.rows.append({
            "q": query.detach(),
            "key": key.detach(),
            "value": value.detach(),
            "mask": (attention_mask.detach()
                     if attention_mask is not None else None),
        })
        return self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask,
            query_rotary_emb, key_rotary_emb, *args, **kwargs)


class _FlashRTGatedRotaryAttnProcessor:
    """Host processor with only the attention dispatch replaced."""

    def __init__(self, core, original):
        self.core = core
        self.original = original

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, query_rotary_emb=None,
                 key_rotary_emb=None, *args, **kwargs):
        if attention_mask is not None and not getattr(
                self.core, "allowed_ranges", ()):
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                query_rotary_emb, key_rotary_emb, *args, **kwargs)
        gate_logits = None
        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)
        query, key, value = _qkv(
            attn, hidden_states, encoder_hidden_states,
            query_rotary_emb, key_rotary_emb)
        projection_dtype = query.dtype
        guard = getattr(self.core, "_frt_guard", None)
        accepted_dtypes = tuple(getattr(guard, "dtypes", ()) or ())
        if accepted_dtypes and projection_dtype not in accepted_dtypes:
            return self.original(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                query_rotary_emb, key_rotary_emb, *args, **kwargs)
        out = self.core(query, key, value)
        out = out.transpose(1, 2).flatten(2, 3).to(projection_dtype)
        if gate_logits is not None:
            out = out.unflatten(2, (attn.heads, -1))
            out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            out = out.flatten(2, 3)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class DiffusersGatedRotaryAttentionAdapter:
    """Route gated dual-rotary Diffusers processors through the family.

    Which executable form serves the seam is the family's decision, and
    preferring a quantized one is a precision decision -- so it arrives
    from the active scheme's ``attention_forms``, the same way the
    gated-delta adapter reads its projection format. ``prefer`` is the
    direct form of the same choice for a caller assembling this adapter
    by hand; the scheme wins when both are given, because the scheme is
    what the deployment selected.
    """

    __name__ = "diffusers_gated_rotary_attention"
    scheme_aware = True

    def __init__(self, prefer=()):
        self.prefer = tuple(prefer)

    def __call__(self, model, forward, *, prefix_cadence: bool = False,
                 scheme=None):
        del prefix_cadence
        prefer = tuple(getattr(scheme, "attention_forms", ()) or self.prefer)
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
                    "attention_core gated-rotary: compatible processor was "
                    "not called during calibration",
                ))
                continue
            try:
                core = bind_dense_attention_best(rows, prefer=prefer)
            except ValueError as exc:
                refused.append((f"{path}.processor", str(exc)[:160]))
                continue
            if core is None:
                refused.append((
                    f"{path}.processor",
                    "attention_core gated-rotary: no family variant serves "
                    "the captured head dimension or mask form",
                ))
                continue
            routed = _FlashRTGatedRotaryAttnProcessor(core, original)
            routes.append((module, original, routed))
            observed[f"{path}.processor::attention_core"] = core
            variants[f"{path}.processor"] = {
                "bound": getattr(core, "_frt_variant", "fa2"),
                "superseded": list(getattr(core, "_frt_variant_trail", ())),
            }
        if not routes:
            return {}, None, {"refused": refused}

        def enable() -> None:
            for module, _, routed in routes:
                module.processor = routed

        def disable() -> None:
            for module, original, _ in routes:
                module.processor = original

        def release() -> None:
            # Reverting put the host processors back; this gives back the
            # memory. The closures above keep working afterwards because
            # they close over this list rather than over its contents.
            routes.clear()

        enable()
        return {}, None, {
            "revert": [disable],
            "release": [release],
            "observed": observed,
            "toggle": (enable, disable),
            "refused": refused,
            "attention_variants": variants,
        }
