"""Fused-layer adapter for Transformers-style gated-delta layers.

Where the callable-slot adapter serves the recurrent rule alone, this
one binds the whole layer's cached-decode step as the fused Hub chain
(``impls.gated_delta_core.fused_layer``). It is registered ahead of the
callable-slot adapter and refuses cleanly — letting the ladder fall
through — when the installed packages predate the chain's entries or a
layer is outside the fused profile. Recognition is by shape, never by
class or model names.
"""

from __future__ import annotations

import torch

from ..impls.gated_delta_core import fused_layer


def _fusable(module) -> bool:
    hv = getattr(module, "num_v_heads", None)
    hk = getattr(module, "num_k_heads", None)
    return (
        all(isinstance(getattr(module, name, None), torch.nn.Linear)
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                         "in_proj_a", "out_proj"))
        and getattr(module, "conv1d", None) is not None
        and getattr(module, "A_log", None) is not None
        and getattr(module, "dt_bias", None) is not None
        and getattr(module, "norm", None) is not None
        # the chain's own profile envelope: D=128, v-heads a multiple
        # of k-heads (the 48/16 host keeps its dedicated entries, other
        # profiles route the head-generic ones; the bind refuses if the
        # installed build predates them)
        and isinstance(hv, int) and isinstance(hk, int)
        and hv > 0 and hk > 0 and hv % hk == 0
        and getattr(module, "head_k_dim", None) == 128
        and getattr(module, "head_v_dim", None) == 128
    )


def _layer_index(path: str) -> int | None:
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1] == "layers" and parts[i].isdigit():
            return int(parts[i])
    return None


class TransformersGatedDeltaFusedAdapter:
    """Bind every fusable gated-delta layer module as one fused seam."""

    __name__ = "transformers_gated_delta_fused"
    scheme_aware = True

    def __call__(self, model, forward, scheme=None):
        fmt = getattr(scheme, "gdn_projection_format", None)
        release = bool(getattr(scheme, "gdn_release_host_weights",
                               False))
        sites = []
        for path, module in model.named_modules():
            for child_name, child in module.named_children():
                child_path = f"{path}.{child_name}" if path else child_name
                if not _fusable(child):
                    continue
                idx = _layer_index(child_path)
                if idx is None:
                    continue
                sites.append((module, child_name, child, idx))
        if not sites:
            return None

        routes = []
        observed = {}
        for parent, child_name, child, idx in sites:
            # a package predating the chain raises here once, and the
            # whole adapter steps aside for the callable-slot ladder
            bound = fused_layer.bind_fused_decode_layer(
                child, idx, projection_format=fmt,
                release_host_weights=release)
            routes.append((parent, child_name, child, bound))
            observed[f"{child_name}@{idx}.gated_delta_fused"] = bound

        def enable():
            for parent, child_name, _child, bound in routes:
                setattr(parent, child_name, bound)

        def disable():
            for parent, child_name, child, _bound in routes:
                setattr(parent, child_name, child)

        enable()
        return {
            "observed": observed,
            "revert": [disable],
            "toggle": (enable, disable),
        }
