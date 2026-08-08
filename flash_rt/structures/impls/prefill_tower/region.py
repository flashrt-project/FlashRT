"""The prefill_tower region family: structural identification, candidates.

The identifier matches shape, never names: a module carrying a
``layers`` list whose blocks each hold a bias-free attention group and
gated FFN under *plain* affine RMS norms (a 1-D weight and no
conditioning projection — the conditioned twin belongs to the sibling
family), plus a rotary table and a plain stack-level norm. The key
width must be narrower than the query width: the chain's cache layout
is written for the single-KV band.
"""

from __future__ import annotations

import torch

from . import fp8_chain
from ... import regions


def _plain_linear(mod) -> bool:
    return isinstance(mod, torch.nn.Linear) and mod.bias is None


def _plain_norm(mod) -> bool:
    return fp8_chain._plain_norm_weight(mod) is not None


def _block_ok(block) -> bool:
    attn = getattr(block, "self_attn", None)
    mlp = getattr(block, "mlp", None)
    if attn is None or mlp is None:
        return False
    if not all(_plain_linear(getattr(attn, a, None))
               for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return False
    if not all(_plain_linear(getattr(mlp, a, None))
               for a in ("gate_proj", "up_proj", "down_proj")):
        return False
    if mlp.gate_proj.out_features != mlp.up_proj.out_features:
        return False
    if not _plain_norm(getattr(block, "input_layernorm", None)):
        return False
    return _plain_norm(getattr(block, "post_attention_layernorm", None))


def identify(model) -> list[str]:
    roots = []
    for path, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or len(layers) < 2:
            continue
        if not callable(getattr(mod, "rotary_emb", None)):
            continue
        head = layers[0]
        attn = getattr(head, "self_attn", None)
        if attn is None or not isinstance(
                getattr(attn, "q_proj", None), torch.nn.Linear):
            continue
        if not _plain_norm(getattr(mod, "norm", None)):
            continue
        if not all(_block_ok(b) for b in layers):
            continue
        if attn.k_proj.out_features >= attn.q_proj.out_features:
            continue
        roots.append(path)
    return roots


def _bind(model, root, probe):
    return fp8_chain.bind_prefill_fp8_chain(model, root, probe)


FAMILY = regions.RegionFamily(
    family="prefill_tower",
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
