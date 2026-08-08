"""The vision_tower region family: structural identification, candidates.

The identifier matches shape, never names: a module carrying a
``layers`` list whose blocks each hold two affine LayerNorms, a
*biased* attention group (``q_proj``/``k_proj``/``v_proj``/
``out_proj``) and a biased two-linear MLP — the bias is what
separates this tower from the decoder families, whose projections
are bias-free.
"""

from __future__ import annotations

import torch

from . import fp8_chain
from ... import regions


def _bias_linear(mod) -> bool:
    return isinstance(mod, torch.nn.Linear) and mod.bias is not None


def _affine_norm(mod) -> bool:
    w = getattr(mod, "weight", None)
    b = getattr(mod, "bias", None)
    return (w is not None and getattr(w, "ndim", 0) == 1
            and b is not None and getattr(b, "ndim", 0) == 1)


def _block_ok(block) -> bool:
    attn = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    if attn is None or mlp is None:
        return False
    if not all(_bias_linear(getattr(attn, a, None))
               for a in ("q_proj", "k_proj", "v_proj", "out_proj")):
        return False
    if not all(_bias_linear(getattr(mlp, a, None))
               for a in ("fc1", "fc2")):
        return False
    if not _affine_norm(getattr(block, "layer_norm1", None)):
        return False
    return _affine_norm(getattr(block, "layer_norm2", None))


def identify(model) -> list[str]:
    roots = []
    for path, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or len(layers) < 2:
            continue
        head = layers[0]
        if not hasattr(head, "layer_norm1"):
            continue
        if not all(_block_ok(b) for b in layers):
            continue
        roots.append(path)
    return roots


def _bind(model, root, probe):
    return fp8_chain.bind_vision_fp8_chain(model, root, probe)


FAMILY = regions.RegionFamily(
    family="vision_tower",
    identify=identify,
    candidates=[regions.RegionCandidate(
        name="fp8_chain",
        missing=fp8_chain.missing_symbols,
        bind=_bind,
    )],
)


def register() -> None:
    """(Re-)register the family — idempotent, import calls it once."""
    regions.register_region_family(FAMILY)


register()
