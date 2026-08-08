"""Route transformers-host attention through a hub kernel interface.

Any transformers-generation host resolves its attention through
``ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`` — the same
registry the capture lowering's pinned vision attention and the
per-head rope route already consult. Registering a hub attention
kernel there and switching the config is therefore a seat like any
other: capability-qualified (registry present, package resolvable),
fully revertible (configs restored, registry entry removed), and
receipted. Which interface actually wins on a box is a measured band
decision (``decisions.lookup("backbone_attn")``), never a default
flip: an empty cache keeps the host's own interface.
"""

from __future__ import annotations

import importlib

import torch


class TransformersAttentionInterfaceAdapter:
    __name__ = "transformers_attention_interface"

    def __call__(self, model, plan=None):
        from ..decisions import lookup
        from ..impls import KernelUnavailable, hub_kernel

        if lookup("backbone_attn", default="host") != "fa4":
            return None
        from ..impls.graph_lowering.qwen3_vl import _find_qwen3_vl
        target = _find_qwen3_vl(model)
        if target is None:
            return None
        try:
            fa4 = hub_kernel("kernels-community/flash-attn4", ">=0")
        except KernelUnavailable as missing:
            return {"refused": [("backbone_attn",
                                 f"fa4 unavailable: {missing}")]}

        def fa4_interface(module, query, key, value,
                          attention_mask=None, scaling=None,
                          dropout=0.0, is_causal=False, **kwargs):
            del module, attention_mask, dropout, kwargs
            out = fa4.flash_attn_func(
                query.transpose(1, 2), key.transpose(1, 2),
                value.transpose(1, 2), softmax_scale=scaling,
                causal=bool(is_causal))
            if isinstance(out, tuple):
                out = out[0]
            return out, None

        base = target.model
        modeling = importlib.import_module(type(base.visual).__module__)
        registry = modeling.ALL_ATTENTION_FUNCTIONS
        try:
            registry["flashrt_fa4"] = fa4_interface
        except TypeError:
            registry.register("flashrt_fa4", fa4_interface)
        configs = [base.visual.config, base.language_model.config,
                   target.config]
        saved = [(cfg, cfg._attn_implementation) for cfg in configs]
        for cfg in configs:
            cfg._attn_implementation = "flashrt_fa4"

        def revert():
            for cfg, prev in saved:
                cfg._attn_implementation = prev
            registry.pop("flashrt_fa4", None) \
                if hasattr(registry, "pop") else None

        return {"revert": [revert],
                "notes": {"backbone_attn": "fa4"}}
