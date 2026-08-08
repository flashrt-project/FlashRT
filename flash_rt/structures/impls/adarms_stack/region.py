"""The adarms_stack region family: structural identification, candidates.

The identifier matches shape, never names: a module carrying a
``layers`` list whose blocks each hold a bias-free attention group
(``q_proj``/``k_proj``/``v_proj``/``o_proj``), a bias-free gated FFN
(``gate_proj``/``up_proj``/``down_proj``), and *conditioned* norms — a
``dense`` projection emitting three modulation vectors per norm is
what separates this stack from an ordinary decoder tower — plus the
stack-level tail (a conditioned final norm and a rotary table). The
key width must be narrower than the query width: the fused chain's
cache layout is written for the single-KV band, and a stack outside
it is exactly what the candidate's own fact checks exist to refuse.
"""

from __future__ import annotations

import torch

from . import fp8_chain
from ... import regions


def _plain_linear(mod) -> bool:
    return isinstance(mod, torch.nn.Linear) and mod.bias is None


def _ada_norm(mod, dim: int) -> bool:
    dense = getattr(mod, "dense", None)
    return (isinstance(dense, torch.nn.Linear)
            and dense.out_features == 3 * dim)


def _block_ok(block, dim: int) -> bool:
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
    if not _ada_norm(getattr(block, "input_layernorm", None), dim):
        return False
    return _ada_norm(getattr(block, "post_attention_layernorm", None),
                     dim)


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
        dim = attn.q_proj.in_features
        if not _ada_norm(getattr(mod, "norm", None), dim):
            continue
        if not all(_block_ok(b, dim) for b in layers):
            continue
        if attn.k_proj.out_features >= attn.q_proj.out_features:
            continue
        roots.append(path)
    return roots


def _band_candidate(band: str, row: dict) -> regions.RegionCandidate:
    return regions.RegionCandidate(
        name=f"{band}_chain",
        missing=lambda band=band: fp8_chain.missing_symbols(band=band),
        bind=lambda model, root, probe, band=band:
            fp8_chain.bind_adarms_fp8_chain(model, root, probe,
                                            band=band),
        precision_rank=row["precision_rank"],
    )


#: the candidate list is generated from the band table — adding a
#: precision band to the family is a table row, not new wiring here
FAMILY = regions.RegionFamily(
    family="adarms_stack",
    identify=identify,
    candidates=[_band_candidate(band, row)
                for band, row in fp8_chain.BANDS.items()],
)


def register() -> None:
    """(Re-)register the family — idempotent, import calls it once."""
    regions.register_region_family(FAMILY)


register()
