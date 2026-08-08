"""Quantize-on-adopt: full-precision checkpoints too big for the card.

``adopt_prequantized`` serves checkpoints that arrive already packed in
someone else's layout. This module is the door for the opposite
situation: the checkpoint is full precision and *cannot fit the card at
all*, but nearly all of its weight mass sits in one structure family —
a sparse-MoE expert bank. Quantizing that family once, at load time,
into structure impls brings the whole model into card budget while the
attention, norms, and router stay in the host's own precision.

The calling convention matches the sibling door: the model is expected
CPU-resident straight from its loader; each expert bank streams through
the GPU in slabs as it packs, so peak footprint is the dense checkpoint
plus one slab. Move the model to the device *after* adopting — by then
the dense banks are gone and the remainder fits.

Like adoption of a pre-quantized checkpoint, this is a load-time
transform, not an attachment: the dense expert weights are released as
each bank binds (holding them would defeat the footprint the door
exists for), so there is no ``detach()`` — undoing an adoption is
reloading the checkpoint. The pack-and-unpack relative L2 of every bank
is recorded in the returned report; the receipt claims a measured
conversion loss, not the absence of one.
"""

from __future__ import annotations

import torch

from .prequantized import AdoptionReport

__all__ = ["quantize_on_adopt"]

_FORMATS = ("moe_experts_nvfp4",)


def _is_moe_expert_bank(module: torch.nn.Module) -> bool:
    """An expert bank: stacked 3D projections plus the activation, the
    shape contract ``gate_up_proj [E, 2I, H]`` / ``down_proj [E, H, I]``."""
    gu = getattr(module, "gate_up_proj", None)
    dn = getattr(module, "down_proj", None)
    if not (torch.is_tensor(gu) and torch.is_tensor(dn)):
        return False
    if gu.dim() != 3 or dn.dim() != 3 or not hasattr(module, "act_fn"):
        return False
    return (gu.shape[0] == dn.shape[0]
            and gu.shape[2] == dn.shape[1]
            and gu.shape[1] == 2 * dn.shape[2])


@torch.no_grad()
def quantize_on_adopt(model: torch.nn.Module,
                      fmt: str = "moe_experts_nvfp4", *,
                      verbose: bool = False) -> AdoptionReport:
    """Quantize every discovered expert bank of ``model`` into a
    structure impl; returns the adoption report for the receipt."""
    if fmt not in _FORMATS:
        raise ValueError(
            f"unknown quantize-on-adopt format {fmt!r}; supported: "
            f"{', '.join(_FORMATS)}")

    from .impls.moe_experts import nvfp4_dynamic

    report = AdoptionReport(fmt=fmt)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not _is_moe_expert_bank(child):
                continue
            path = f"{name}.{child_name}" if name else child_name
            bound, rels = nvfp4_dynamic.bind_experts_seam(
                {"gate_up_proj": child.gate_up_proj.detach(),
                 "down_proj": child.down_proj.detach()},
                child.act_fn)
            # release the dense bank before moving on: the streaming
            # bind is only slab-peak if the retired experts actually go
            child.gate_up_proj = None
            child.down_proj = None
            setattr(module, child_name, bound)
            report.replaced.append(path)
            for stack, rel in rels.items():
                report.conversion_rel_l2[f"{path}.{stack}"] = rel
            if verbose:
                print(f"[quantize_on_adopt] {path}: "
                      + ", ".join(f"{k} relL2={v:.4f}"
                                  for k, v in rels.items()),
                      flush=True)
    if not report.replaced:
        raise ValueError(
            "no expert banks found: this model does not carry the "
            f"stacked-3D MoE structure {fmt} adopts")
    torch.cuda.empty_cache()
    if verbose:
        print(f"[quantize_on_adopt] {report.summary()}", flush=True)
    return report
