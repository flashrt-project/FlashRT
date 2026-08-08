"""The dit_block region family: structural identification, candidates.

The identifier matches shape, never names: a module carrying a
``transformer_blocks`` list whose blocks each hold one attention group
(``to_q``/``to_k``/``to_v``/``to_out``), a two-linear gated FFN, and a
per-block AdaLN projection — with at least one cross block (key width
differs from query width) and the stack-level tail (timestep encoder,
final norm, the two output projections). That is the span the fused
chain candidate knows how to absorb; a host that merely resembles it
is exactly what the bind-time smoke gate exists to refuse.
"""

from __future__ import annotations

import torch

from . import fp4_chain
from ... import regions


def _linear(mod) -> bool:
    return isinstance(mod, torch.nn.Linear) and mod.bias is not None


def _block_ok(block) -> bool:
    attn = getattr(block, "attn1", None)
    ff = getattr(block, "ff", None)
    norm1 = getattr(block, "norm1", None)
    if attn is None or ff is None or norm1 is None:
        return False
    if not all(_linear(getattr(attn, a, None))
               for a in ("to_q", "to_k", "to_v")):
        return False
    out = getattr(attn, "to_out", None)
    if out is None or len(out) < 1 or not _linear(out[0]):
        return False
    net = getattr(ff, "net", None)
    if (net is None or len(net) < 3
            or not _linear(getattr(net[0], "proj", None))
            or not _linear(net[2])):
        return False
    if not _linear(getattr(norm1, "linear", None)):
        return False
    return isinstance(getattr(block, "dim", None), int)


def identify(model) -> list[str]:
    roots = []
    for path, mod in model.named_modules():
        blocks = getattr(mod, "transformer_blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 2:
            continue
        if not all(_block_ok(b) for b in blocks):
            continue
        if not all(hasattr(mod, a) for a in
                   ("timestep_encoder", "norm_out",
                    "proj_out_1", "proj_out_2")):
            continue
        if not any(b.attn1.to_k.in_features != b.attn1.to_q.in_features
                   for b in blocks):
            continue
        roots.append(path)
    return roots


def _bind(model, root, probe):
    return fp4_chain.bind_dit_fp4_chain(model, root, probe)


FAMILY = regions.RegionFamily(
    family="dit_block",
    identify=identify,
    candidates=[regions.RegionCandidate(
        name="fp4_chain",
        missing=fp4_chain.missing_symbols,
        bind=_bind,
    )],
)


def register() -> None:
    """(Re-)register the family — idempotent, import calls it once."""
    regions.register_region_family(FAMILY)


register()
