"""Adopting pre-quantized checkpoints: packed weights in, structures out.

The scheme door (``auto_swaps(scheme=...)``) quantizes at runtime from a
full-precision host — whatever the checkpoint holds, statistics are
measured and formats are chosen here. This module is the other door: a
checkpoint that is *already* quantized, in someone else's packed layout,
adopted by converting each packed projection into a structure impl that
executes it.

The first supported layout is compressed-tensors NVFP4
(``quant_method="compressed-tensors"``, ``format="nvfp4-pack-quantized"``,
the ``run_compressed`` load path where every quantized projection is an
``nn.Linear`` carrying ``weight_packed``/``weight_scale`` parameters).
The upstream execution path decompresses those weights to BF16 inside
``forward``, which multiplies the footprint back to full precision —
the 27B host that motivated this door OOMs a 32 GB card exactly that
way. Adoption converts each packed projection to the Hub NVFP4 layout
once, at load time, and the model runs in its quantized footprint.

Decompression semantics stay upstream's: the layer is unpacked by the
compressor registered for the checkpoint's own format, never by a
reimplementation here. Conversion into the Hub layout is a regrid with
a measurable relative error against those decompressed values; it is
recorded per layer in the returned report, because "the tokens still
match" is a claim about a specific conversion loss, not the absence of
one. Lossless direct transcoding between the two grids is the recorded
follow-up, not silently assumed.

Adoption is a load-time transform, not an attachment: the packed source
modules cannot execute and holding decompressed copies of every layer
would defeat the footprint this door exists for. There is no
``detach()`` — undoing an adoption is reloading the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["AdoptionReport", "adopt_prequantized"]

_FORMATS = ("ct_nvfp4",)


@dataclass
class AdoptionReport:
    """What one adoption did, per layer, for the receipt."""

    fmt: str
    replaced: list[str] = field(default_factory=list)
    conversion_rel_l2: dict[str, float] = field(default_factory=dict)

    @property
    def worst_conversion(self) -> float:
        return max(self.conversion_rel_l2.values(), default=0.0)

    def summary(self) -> dict:
        vals = sorted(self.conversion_rel_l2.values())
        return {
            "format": self.fmt,
            "replaced_projections": len(self.replaced),
            "conversion_rel_l2_median": (
                round(vals[len(vals) // 2], 5) if vals else None),
            "conversion_rel_l2_max": (
                round(vals[-1], 5) if vals else None),
        }


def _is_ct_nvfp4_linear(module: torch.nn.Module) -> bool:
    """The ``run_compressed`` load form: a plain Linear that carries the
    packed parameters alongside its (meta or absent) dense weight."""
    return isinstance(module, torch.nn.Linear) and hasattr(
        module, "weight_packed")


@torch.no_grad()
def adopt_prequantized(model: torch.nn.Module, fmt: str = "ct_nvfp4",
                       *, verbose: bool = False) -> AdoptionReport:
    """Convert every packed projection of ``model`` to a structure impl.

    ``model`` is expected CPU-resident, straight from its loader:
    conversion streams each layer through the GPU one at a time, so the
    packed checkpoint plus one decompressed layer is the peak footprint.
    Move the model to the device *after* adopting.
    """
    if fmt not in _FORMATS:
        raise ValueError(
            f"unknown pre-quantized format {fmt!r}; supported: "
            f"{', '.join(_FORMATS)}")
    try:
        from compressed_tensors.compressors import BaseCompressor
    except ImportError as exc:
        raise ImportError(
            "adopting a compressed-tensors checkpoint requires the "
            "compressed-tensors package (the checkpoint's own "
            "decompression semantics); install it rather than "
            "reimplementing the unpack") from exc

    from .impls.linear_proj import nvfp4_dynamic

    compressor = BaseCompressor.load_from_registry("nvfp4-pack-quantized")
    report = AdoptionReport(fmt=fmt)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not _is_ct_nvfp4_linear(child):
                continue
            path = f"{name}.{child_name}" if name else child_name
            # the registered compressor either mutates the layer's dense
            # weight in place or returns the tensor; take whichever form
            # this version speaks
            ret = compressor.decompress_module(child)
            w = ret if torch.is_tensor(ret) else child.weight.detach()
            bias = getattr(child, "bias", None)
            bound, rel = nvfp4_dynamic.bind_proj_seam(
                {"w": w, "b": None if bias is None else bias.detach()})
            setattr(module, child_name, bound)
            report.replaced.append(path)
            report.conversion_rel_l2[path] = rel
            del w
            if verbose:
                print(f"[prequantized] {path}: relL2={rel:.4f}",
                      flush=True)
    if not report.replaced:
        raise ValueError(
            "no packed projections found: this model does not look like "
            f"a {fmt} checkpoint loaded in its compressed form")
    torch.cuda.empty_cache()
    if verbose:
        print(f"[prequantized] {report.summary()}", flush=True)
    return report
