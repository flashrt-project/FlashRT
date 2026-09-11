"""Checkpoint loading for the Ascend GR00T N1.7 action chain.

The checkpoint's tensors are already described once, declaratively, in
``flash_rt.frontends.torch._groot_n17_thor_spec``. That description is of the
file on disk rather than of a backend, so this module *derives* the Ascend
variant from it instead of restating it: the FP16 casts become BF16 -- the
dtype this part serves and the dtype the reference policy itself serves -- the
FP8 quantization steps drop out, and the blocks the action chain never reads
are filtered away.

Deriving keeps one description of the checkpoint layout. A spec that drifts
from the file fails at load with the missing key named, which is the failure
this arrangement is meant to produce.
"""

from __future__ import annotations

import dataclasses
import pathlib

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec, WeightLoader
from flash_rt.executors.torch_weights import (MultiSafetensorsSource, Quant,
                                              SafetensorsSource, ToBf16, ToFp16)

# What each stage reads. Loading a stage's weights costs device memory the
# stage that does not run would never touch, so the two are separable and the
# whole frame is their union.
_CHAIN_BLOCKS = ("vl_self_attn", "dit")
_CHAIN_SINGLETONS = ("vlln_", "ah_pos_embed", "ts_", "proj_out_",
                     "st_enc_", "ac_enc_", "ac_dec_")
_BACKBONE_BLOCKS = ("qwen3vl_vit", "qwen3vl_llm")
_BACKBONE_SINGLETONS = ("dsm", "merger_", "llm_norm_w", "patch_embed")


def _to_bf16(transforms):
    """Rewrite one item's transform chain for a BF16 Ascend load."""
    rewritten = []
    for transform in transforms:
        if isinstance(transform, Quant):
            continue                      # no FP8 tensor hardware on this part
        rewritten.append(ToBf16() if isinstance(transform, ToFp16) else transform)
    return rewritten


def _rewrite(item: Item) -> Item:
    return dataclasses.replace(item, transforms=_to_bf16(item.transforms),
                               scale_into=None)


def _spec(block_names, singleton_prefixes) -> ModelWeightSpec:
    from flash_rt.frontends.torch._groot_n17_thor_spec import build_spec

    shared = build_spec()
    blocks = [
        LayerBlock(prefix_fmt=block.prefix_fmt, num_layers=block.num_layers,
                   items=[_rewrite(item) for item in block.items], name=block.name)
        for block in shared.blocks if block.name in block_names
    ]
    if len(blocks) != len(block_names):
        raise ImportError(
            "the shared GR00T N1.7 checkpoint spec no longer carries the "
            f"{block_names} blocks this backend derives from")
    singletons = [_rewrite(item) for item in shared.singletons
                  if item.name.startswith(singleton_prefixes)]
    return ModelWeightSpec(framework="torch", blocks=blocks, singletons=singletons)


def action_chain_spec() -> ModelWeightSpec:
    """The BF16 spec for the VL adapter, the DiT and the action encoders."""
    return _spec(_CHAIN_BLOCKS, _CHAIN_SINGLETONS)


def backbone_spec() -> ModelWeightSpec:
    """The BF16 spec for the vision tower, the mergers and the truncated LLM.

    The patch projection is here, unlike the other backends' ``aux`` boundary:
    it is a matmul wearing a Conv3d's shape, so folding it into the graph costs
    nothing and takes 8.5 ms of per-frame host work out of the frame. The token
    embedding stays with the caller, because for a fixed instruction it is a
    constant the graph reads rather than work it does.
    """
    return _spec(_BACKBONE_BLOCKS, _BACKBONE_SINGLETONS)


def frame_spec() -> ModelWeightSpec:
    """Everything a served frame reads."""
    return _spec(_BACKBONE_BLOCKS + _CHAIN_BLOCKS,
                 _BACKBONE_SINGLETONS + _CHAIN_SINGLETONS)


class ChainWeights:
    """Plain attribute holder the spec's sinks write into."""


def shard_paths(checkpoint_dir) -> list[pathlib.Path]:
    directory = pathlib.Path(checkpoint_dir)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no safetensors shard in {directory.name}; this backend loads the "
            "published GR00T N1.7 checkpoint directory")
    return shards


def load(checkpoint_dir, spec) -> ChainWeights:
    """Read one stage's weights onto the host in BF16.

    The tensors land on the host because the device copy is the caller's
    business: every projection is bound into a fractal-NZ operand or sliced to
    one embodiment first, and both of those are setup-time transforms that
    would otherwise be paid twice.
    """
    shards = shard_paths(checkpoint_dir)
    source = (SafetensorsSource(str(shards[0]), device="cpu") if len(shards) == 1
              else MultiSafetensorsSource([str(p) for p in shards], device="cpu"))
    weights = ChainWeights()
    WeightLoader(source=source, target=weights, spec=spec).run()
    return weights


def load_action_chain(checkpoint_dir) -> ChainWeights:
    return load(checkpoint_dir, action_chain_spec())


def load_backbone(checkpoint_dir) -> ChainWeights:
    return load(checkpoint_dir, backbone_spec())


def load_frame(checkpoint_dir) -> ChainWeights:
    return load(checkpoint_dir, frame_spec())
