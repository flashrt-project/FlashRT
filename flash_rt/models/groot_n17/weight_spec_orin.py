"""BF16 weight spec for GR00T N1.7 on Jetson Orin (SM87).

SM87 has no FP8 tensor cores, so the production spec's ``Quant()`` op (FP8
E4M3 per-tensor) buys nothing here and only costs precision: the checkpoint
is natively bf16, and round-tripping it through e4m3 discards mantissa bits
for zero throughput gain. This module derives the Orin spec from the
validated N1.7 spec by rewriting each item's transform list, so
checkpoint-key coverage, sink names and item ordering cannot drift from the
spec that Thor / RTX / SM89 / Ascend all consume.

Because no ``Quant()`` survives, no ``scale_into`` list is populated: the
Orin frontend must treat every loaded weight as already dequantized (there
are no alphas to multiply back in).

Additive: ``flash_rt.models.groot_n17.weight_spec`` is not modified.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from flash_rt.executors.torch_weights import Cat, Quant, ToBf16, ToFp16
from flash_rt.executors.weight_loader import Item, ModelWeightSpec
from flash_rt.models.groot_n17.weight_spec import build_spec


def _bf16_key(key):
    """``Cat`` casts its parts to fp16 before concatenating; retarget it."""
    if isinstance(key, Cat) and key.dtype is not None:
        return Cat(list(key.keys), dim=key.dim, dtype=torch.bfloat16)
    return key


def _bf16_item(item: Item) -> Item:
    """Rewrite one item: FP16 cast -> BF16 cast, FP8 quant dropped."""
    ops = []
    for op in item.transforms:
        if isinstance(op, Quant):
            continue
        ops.append(ToBf16() if isinstance(op, ToFp16) else op)
    return replace(item, key=_bf16_key(item.key), transforms=ops,
                   scale_into=None)


def build_orin_spec() -> ModelWeightSpec:
    """Full N1.7 spec with every weight materialized as BF16 (no FP8)."""
    src = build_spec()
    spec = ModelWeightSpec(
        framework=src.framework,
        blocks=[
            replace(blk, items=[_bf16_item(it) for it in blk.items])
            for blk in src.blocks
        ],
        singletons=[_bf16_item(it) for it in src.singletons],
        buffers=list(src.buffers),
        dims=dict(src.dims),
    )
    leftover = [
        it.name
        for blk in spec.blocks
        for it in blk.items
        if any(isinstance(op, Quant) for op in it.transforms)
    ]
    leftover += [
        it.name for it in spec.singletons
        if any(isinstance(op, Quant) for op in it.transforms)
    ]
    if leftover:
        # Fail loudly rather than silently shipping an FP8 weight onto a
        # platform with no FP8 tensor cores.
        raise RuntimeError(
            f"Orin BF16 spec still contains Quant() ops: {leftover}")
    return spec


ORIN_WEIGHT_SPEC = build_orin_spec()


__all__ = ["build_orin_spec", "ORIN_WEIGHT_SPEC"]
