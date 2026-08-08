"""GR00T N1.7 — the explicit structure pipeline.

This file is the hand-written counterpart of ``structures.auto_swaps``:
every seat is declared by the author, calibrated by the author's own
hooks, and bound by calling each structure family's library binder
directly. Nothing is discovered and nothing is matched — what runs is
exactly what is written below.

The same file runs unchanged across hardware generations. That works
because device differences enter through exactly one boundary — the
kernel package a binder loads — and every way that boundary can come up
empty surfaces as one catchable refusal, ``KernelUnavailable``. The
portable-explicit rules this file follows:

  1. bind attention through the family entry, never a variant module;
  2. wrap every seat in ``except (KernelUnavailable, ValueError)`` —
     record the refusal, keep the host, continue;
  3. declare seat ownership explicitly, do not infer it from names;
  4. print every refusal and every absent package: a seat served by the
     host is an outcome, a seat silently skipped is a bug.

Usage (paths point at the official Isaac-GR00T host and checkpoint):

  python groot_n17.py --host /path/to/Isaac-GR00T \\
      --checkpoint /path/to/GR00T-N1.7-3B \\
      --backbone-assets /path/to/backbone-config-assets \\
      --fixture /path/to/observation_fixture.pt [--compile]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
import time
from collections import defaultdict
from pathlib import Path

import torch


# ----------------------------------------------------------------------
# host loading
# ----------------------------------------------------------------------

def load_policy(host: Path, checkpoint: Path, backbone_assets: Path):
    """Load the official Gr00tPolicy with local backbone config assets.

    The official constructor first materialises the public backbone
    checkpoint and then overwrites every tensor with the GR00T weights.
    Pointing that redundant load at a local config keeps construction
    offline; the loaded parameters and the inference code are untouched.
    """
    sys.path.insert(0, str(host))
    from transformers import (Qwen3VLConfig,
                              Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor)

    original_processor = Qwen3VLProcessor.from_pretrained

    def from_local_config(cls, _name, *args, **kwargs):
        del args
        config = Qwen3VLConfig.from_pretrained(backbone_assets,
                                               local_files_only=True)
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs["attn_implementation"]
        return cls(config)

    Qwen3VLForConditionalGeneration.from_pretrained = classmethod(
        from_local_config)
    Qwen3VLProcessor.from_pretrained = classmethod(
        lambda cls, _n, *a, **k: original_processor(
            backbone_assets, local_files_only=True))

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(
        embodiment_tag="oxe_droid_relative_eef_relative_joint",
        model_path=str(checkpoint), device="cuda", strict=True)


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cloned = [clone_tree(v) for v in value]
        return cloned if isinstance(value, list) else tuple(cloned)
    return value


# ----------------------------------------------------------------------
# the explicit assembly
# ----------------------------------------------------------------------

class Assembly:
    """Seats, refusals and calibration for one explicit build."""

    def __init__(self):
        self.swaps: dict[str, torch.nn.Module] = {}
        self.refused: list[tuple[str, str]] = []
        self.families = defaultdict(int)

    def take(self, path: str, family: str, build):
        """Run one seat's binder; a refusal is recorded, never raised."""
        from flash_rt.structures.impls import KernelUnavailable

        try:
            bound = build()
        except (KernelUnavailable, ValueError) as refusal:
            self.refused.append((path, f"{family}: {str(refusal)[:120]}"))
            return None
        if bound is None:
            self.refused.append((path, f"{family}: binder declined"))
            return None
        self.families[family] += 1
        return bound

    def place(self, path: str, module: torch.nn.Module):
        self.swaps[path] = module


def amax(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().max())


def _dit_band() -> str:
    """Author pin > measured decision cache > precision-order default."""
    import os
    pin = os.environ.get("FRT_DIT_BAND")
    if pin:
        return pin
    from flash_rt.structures.decisions import lookup
    return lookup("groot_dit", default="fp8")


def build(model, run_once) -> tuple[Assembly, dict]:
    """Declare, calibrate and bind every seat of the GR00T N1.7 graph.

    ``run_once`` executes the model's hot path a single time; every
    calibration hook below records from that same pass.
    """
    from flash_rt.structures.adapters.diffusers_attention import (
        DiffusersAttentionAdapter)
    from flash_rt.structures.discover import Seam, seam_weights
    from flash_rt.structures.impls.adaln_producer.fused import (
        bind_adaln_producer)
    from flash_rt.structures.impls.cadence_static.cross_attention import (
        bind_cross_attention_kv, capture_cross_attention_kv,
        discover_cross_attention_kv)
    from flash_rt.structures.impls.decoder_ffn import fp8_static as dec_ffn
    from flash_rt.structures.impls.qkv_pack.fp8_static import bind_qkv_pack
    from flash_rt.structures.impls.vision_ffn import fp8_static as vis_ffn

    asm = Assembly()
    base = model.backbone.model.model
    visual_blocks = list(base.visual.blocks)
    lang_layers = list(base.language_model.layers)
    dit_blocks = list(model.action_head.model.transformer_blocks)

    # ---- seat declaration ------------------------------------------------
    # Every path named here is a claim of ownership. The qkv seats also
    # need a runtime qualification (all three siblings must consume one
    # tensor), which calibration verifies below — a masked or cross site
    # simply never satisfies it and stays with the host.
    VISION_FFN = {f"backbone.model.model.visual.blocks.{i}.mlp": b.mlp
                  for i, b in enumerate(visual_blocks)}
    LANG_FFN = {f"backbone.model.model.language_model.layers.{i}.mlp": l.mlp
                for i, l in enumerate(lang_layers)}
    LANG_QKV = {f"backbone.model.model.language_model.layers.{i}.self_attn":
                l.self_attn for i, l in enumerate(lang_layers)}
    DIT_FF = {f"action_head.model.transformer_blocks.{i}.ff": b.ff
              for i, b in enumerate(dit_blocks)}
    DIT_ADALN = {f"action_head.model.transformer_blocks.{i}.norm1": b.norm1
                 for i, b in enumerate(dit_blocks)
                 if type(b.norm1).__name__ == "AdaLayerNorm"}
    DIT_QKV = {f"action_head.model.transformer_blocks.{i}.attn1": b.attn1
               for i, b in enumerate(dit_blocks)}
    vlsa_blocks = list(
        model.action_head.vl_self_attention.transformer_blocks)
    LANG_OPROJ = {
        f"backbone.model.model.language_model.layers.{i}"
        ".self_attn.o_proj": l.self_attn.o_proj
        for i, l in enumerate(lang_layers)}
    VLSA_FF = {f"action_head.vl_self_attention.transformer_blocks.{i}.ff":
               b.ff for i, b in enumerate(vlsa_blocks)}
    VLSA_QKV = {
        f"action_head.vl_self_attention.transformer_blocks.{i}.attn1":
        b.attn1 for i, b in enumerate(vlsa_blocks)}
    VLSA_OUT = {
        f"action_head.vl_self_attention.transformer_blocks.{i}"
        ".attn1.to_out.0": b.attn1.to_out[0]
        for i, b in enumerate(vlsa_blocks)}
    PATCH = {"backbone.model.model.visual.patch_embed":
             base.visual.patch_embed}
    DIT_OUT = {
        f"action_head.model.transformer_blocks.{i}.attn1.to_out.0":
        b.attn1.to_out[0] for i, b in enumerate(dit_blocks)}
    MERGER_PROJ = {}
    for name, mod in (("merger", base.visual.merger),
                      *((f"deepstack_merger_list.{i}", m) for i, m in
                        enumerate(base.visual.deepstack_merger_list))):
        for fc in ("linear_fc1", "linear_fc2"):
            MERGER_PROJ[f"backbone.model.model.visual.{name}.{fc}"] =                 getattr(mod, fc)
    VISION_PROJ = {}
    for i, b in enumerate(visual_blocks):
        VISION_PROJ[f"backbone.model.model.visual.blocks.{i}.attn.qkv"] =             b.attn.qkv
        VISION_PROJ[f"backbone.model.model.visual.blocks.{i}.attn.proj"] =             b.attn.proj

    # ---- calibration: one pass, author-owned hooks -----------------------
    cal = defaultdict(dict)
    hooks = []

    def record_input(key, field):
        def hook(_m, args, _out):
            slot = cal[key]
            x = args[0]
            slot[field] = max(slot.get(field, 0.0), amax(x))
            slot.setdefault(field + "_rows", x.reshape(-1, x.shape[-1])
                            .shape[0])
            slot.setdefault(field + "_dtype", x.dtype)
            vec = (x.detach().float().abs()
                   .reshape(-1, x.shape[-1]).amax(0))
            prev = slot.get(field + "_cvec")
            slot[field + "_cvec"] = (vec if prev is None
                                     else torch.maximum(prev, vec))
        return hook

    def record_pair(key):
        def hook(_m, args, out):
            cal[key].setdefault("pairs", []).append(
                (args[0].detach().clone(), out.detach().clone()))
        return hook

    def record_qkv(key, attn, attrs):
        seen = {}

        def make(attr):
            def hook(_m, args, _out):
                x = args[0]
                seen[attr] = x.data_ptr()
                slot = cal[key]
                slot["x"] = max(slot.get("x", 0.0), amax(x))
                slot["rows"] = x.reshape(-1, x.shape[-1]).shape[0]
                slot["shared"] = len(set(seen.values())) == 1
            return hook
        return [getattr(attn, a).register_forward_hook(make(a))
                for a in attrs]

    for path, mlp in {**VISION_FFN, **DIT_FF, **VLSA_FF}.items():
        hooks.append(mlp.register_forward_hook(record_input(path, "x")))
        fc2 = mlp.linear_fc2 if hasattr(mlp, "linear_fc2") else mlp.net[2]
        hooks.append(fc2.register_forward_hook(record_input(path, "hidden")))
    for path, mlp in LANG_FFN.items():
        hooks.append(mlp.register_forward_hook(record_input(path, "x")))
        hooks.append(mlp.down_proj.register_forward_hook(
            record_input(path, "hidden")))
    for path, attn in LANG_QKV.items():
        hooks.extend(record_qkv(path, attn, ("q_proj", "k_proj", "v_proj")))
    for path, attn in DIT_QKV.items():
        hooks.extend(record_qkv(path, attn, ("to_q", "to_k", "to_v")))
        for attr in ("to_q", "to_k", "to_v"):
            hooks.append(getattr(attn, attr).register_forward_hook(
                record_input(f"{path}.{attr}", "x")))
    for path, attn in VLSA_QKV.items():
        hooks.extend(record_qkv(path, attn, ("to_q", "to_k", "to_v")))
    for path, norm in DIT_ADALN.items():
        hooks.append(norm.linear.register_forward_hook(record_pair(path)))
    for path, mod in {**LANG_OPROJ, **VLSA_OUT, **VISION_PROJ,
                      **DIT_OUT, **MERGER_PROJ}.items():
        hooks.append(mod.register_forward_hook(record_input(path, "x")))
    for path, mod in PATCH.items():
        hooks.append(mod.register_forward_hook(record_input(path, "x")))

    cross_sites = discover_cross_attention_kv(model)
    cross_captures = capture_cross_attention_kv(cross_sites, run_once)
    for h in hooks:
        h.remove()

    # ---- bind: direct library-binder calls -------------------------------
    for path, mlp in VISION_FFN.items():
        c = cal[path]
        seam = Seam(structure="vision_ffn", path=path,
                    parent_path=path.rsplit(".", 1)[0], norm_attr="norm2",
                    dims={}, variant={}, fc_attrs=("linear_fc1",
                                                   "linear_fc2"))
        bound = asm.take(path, "vision_ffn", lambda: vis_ffn.bind_mlp_seam(
            seam_weights(model, seam), input_scale=c["x"] / 448.0,
            hidden_scale=c["hidden"] / 448.0, original=mlp))
        if bound is not None:
            asm.place(path, bound)

    from flash_rt.structures.impls.vision_ffn import (
        nvfp4_balance as vis_w4)
    for path, mlp in {**DIT_FF, **VLSA_FF}.items():
        c = cal[path]
        seam = Seam(structure="vision_ffn", path=path,
                    parent_path=path.rsplit(".", 1)[0], norm_attr="norm3",
                    dims={}, variant={}, fc_attrs=("net.0.proj", "net.2"))
        _band = _dit_band()
        if _band == "fp4" and c.get("x_rows", 1 << 30) <= 64:
            # the denoise band is bandwidth-bound: half-width weights
            # and the FP4 FFN wire (fc1's GEMM re-quantizes its own
            # output, fc2 consumes the wire) take the seat
            bound = asm.take(path, "dit_ffn_fp4",
                             lambda: vis_w4.bind_mlp_seam(
                                 seam_weights(model, seam),
                                 channel_in=c["x_cvec"],
                                 channel_hidden=c["hidden_cvec"],
                                 original=mlp, fuse_wire=True))
        else:
            bound = asm.take(path, "dit_ffn",
                             lambda: vis_ffn.bind_mlp_seam(
                                 seam_weights(model, seam),
                                 input_scale=c["x"] / 448.0,
                                 hidden_scale=c["hidden"] / 448.0,
                                 original=mlp))
        if bound is not None:
            asm.place(path, bound)

    for path, mlp in LANG_FFN.items():
        c = cal[path]
        seam = Seam(structure="decoder_ffn", path=path,
                    parent_path=path.rsplit(".", 1)[0],
                    norm_attr="post_attention_layernorm", dims={"D":
                    mlp.down_proj.weight.shape[0]},
                    variant={"activation": "silu"})
        bound = asm.take(path, "decoder_ffn", lambda: dec_ffn.bind_mlp_seam(
            seam_weights(model, seam), variant=seam.variant,
            input_scale=c["x"] / 448.0, hidden_scale=c["hidden"] / 448.0,
            original=mlp))
        if bound is not None:
            asm.place(path, bound)

    for path, attn in LANG_QKV.items():
        c = cal[path]
        if not c.get("shared"):
            asm.refused.append(
                (path, "qkv_pack: siblings do not share one input "
                       "this tick — host keeps its projections"))
            continue
        scale = torch.tensor([max(c["x"] / 448.0, 1e-8)], device="cuda")
        parts = asm.take(path, "qkv_pack", lambda: bind_qkv_pack(
            [attn.q_proj, attn.k_proj, attn.v_proj], scale,
            rows=c["rows"], in_dtype="bf16_fused_quant"))
        if parts is not None:
            for attr, mod in zip(("q_proj", "k_proj", "v_proj"), parts):
                asm.place(f"{path}.{attr}", mod)

    # projections the packs do not cover: the language o_proj and the
    # vl-self-attention output projection each take an FP8 seat of
    # their own — the same linear_proj family the automatic path binds.
    from flash_rt.structures.impls.linear_proj.fp8_static import (
        bind_proj_seam)

    def take_proj(path, mod, c):
        if "x" not in c:
            return
        weights = {"w": mod.weight}
        if mod.bias is not None:
            weights["b"] = mod.bias
        bound = asm.take(path, "linear_proj", lambda: bind_proj_seam(
            weights, input_scale=c["x"] / 448.0,
            row_profile=[c["x_rows"]], original=mod))
        if bound is not None:
            asm.place(path, bound)

    # the vision attention projections are the one large-M band the
    # automatic book leaves to the compiler, and the compiler's
    # partitioning of them is host-form dependent (the cross-host
    # census showed the same mm landing in a 3x slower template under
    # one transformers generation). Seat them and the partitioning
    # question disappears on every host.
    for path, mod in {**LANG_OPROJ, **VLSA_OUT, **VISION_PROJ,
                      **MERGER_PROJ}.items():
        take_proj(path, mod, cal[path])

    from flash_rt.structures.impls.linear_proj import (
        nvfp4_balance as proj_w4)
    for path, mod in DIT_OUT.items():
        c = cal[path]
        if (_dit_band() != "fp4"
                or "x_cvec" not in c or c.get("x_rows", 1 << 30) > 64):
            continue
        weights = {"w": mod.weight}
        if mod.bias is not None:
            weights["b"] = mod.bias
        bound = asm.take(path, "linear_proj_fp4",
                         lambda: proj_w4.bind_proj_seam(
                             weights, channel_amax=c["x_cvec"],
                             original=mod))
        if bound is not None:
            asm.place(path, bound)

    for path, attn in VLSA_QKV.items():
        c = cal[path]
        if not c.get("shared"):
            asm.refused.append(
                (path, "qkv_pack: siblings do not share one input "
                       "this tick — host keeps its projections"))
            continue
        scale = torch.tensor([max(c["x"] / 448.0, 1e-8)], device="cuda")
        parts = asm.take(path, "qkv_pack", lambda: bind_qkv_pack(
            [attn.to_q, attn.to_k, attn.to_v], scale,
            rows=c["rows"], in_dtype="bf16_fused_quant"))
        if parts is not None:
            for attr, mod in zip(("to_q", "to_k", "to_v"), parts):
                asm.place(f"{path}.{attr}", mod)

    # the vision patch projection: one full-patch seat
    from flash_rt.structures.impls.patch_projection import (
        bind_flat_patch_projection)
    for path, mod in PATCH.items():
        c = cal[path]
        proj = getattr(mod, "proj", mod)
        if "x" not in c or not hasattr(proj, "weight"):
            continue
        weights = {"w": proj.weight.reshape(proj.weight.shape[0], -1)}
        if getattr(proj, "bias", None) is not None:
            weights["b"] = proj.bias
        bound = asm.take(path, "patch_projection",
                         lambda: bind_flat_patch_projection(
                             weights, row_profile=[c["x_rows"]],
                             host_dtypes=[c["x_dtype"]], original=mod))
        if bound is not None:
            asm.place(path, bound)

    # language attention: fold the per-head q/k norm and rope into the
    # packed projection's tail — the family composition the automatic
    # path engages, called through the same adapter entry. It requires
    # the packs above to be in the seat map already.
    from flash_rt.structures.adapters.qwen_per_head_qk_norm_rope import (
        PerHeadGqaQkNormRopeAdapter)
    reverts = []
    rope_result = PerHeadGqaQkNormRopeAdapter()(
        model, types.SimpleNamespace(swaps=asm.swaps, notes={}))
    rope_observed = {}
    if rope_result:
        rope_observed = rope_result.get("observed", {})
        asm.refused.extend(rope_result.get("refused", []))
        reverts.extend(rope_result.get("revert", ()))
        if rope_observed:
            asm.families["qk_norm_rope"] = len(rope_observed)

    # The DiT self-attention blocks carry a negotiated chain, and the
    # explicit form writes the negotiation out: the adaptive norm emits
    # FP8 at a shared activation scale, and the packed QKV consumes that
    # FP8 directly — one quantization for the pair instead of one each.
    # A cross-attention block never qualifies (its siblings read two
    # different tensors), so its norm records a refusal and stays host.
    locator = None
    for i, block in enumerate(dit_blocks):
        norm_path = f"action_head.model.transformer_blocks.{i}.norm1"
        attn_path = f"action_head.model.transformer_blocks.{i}.attn1"
        pairs = cal[norm_path].get("pairs", [])
        c = cal[attn_path]
        if norm_path not in DIT_ADALN or not pairs:
            continue
        if not c.get("shared"):
            # cross block: no pack, but the chain does not vanish — the
            # producer still emits FP8 at the query projection's scale
            # and to_q consumes it directly (the fp8-in form has no
            # work-band floor); to_k/to_v read the encoder stream and
            # take plain FP8 seats of their own
            norm = block.norm1
            cq = cal[f"{attn_path}.to_q"]
            rows_q = cq.get("x_rows", c.get("rows"))
            q_scale = torch.tensor(
                [max(cq.get("x", 0.0) / 448.0, 1e-8)], device="cuda")
            producer = asm.take(norm_path, "adaln_producer",
                                lambda: bind_adaln_producer(
                                    norm, pairs, act_scale=q_scale,
                                    rows=rows_q,
                                    dim=norm.norm.normalized_shape[0],
                                    locator=locator, norm="layer"))
            if producer is not None:
                locator = locator or producer.locator
                asm.place(norm_path, producer)
                mod = block.attn1.to_q
                weights = {"w": mod.weight}
                if mod.bias is not None:
                    weights["b"] = mod.bias
                bound = asm.take(f"{attn_path}.to_q", "linear_proj",
                                 lambda: bind_proj_seam(
                                     weights,
                                     input_scale=cq["x"] / 448.0,
                                     row_profile=[rows_q], original=mod,
                                     in_dtype="fp8_static"))
                if bound is not None:
                    asm.place(f"{attn_path}.to_q", bound)
            for attr in ("to_k", "to_v"):
                key = f"{attn_path}.{attr}"
                take_proj(key, getattr(block.attn1, attr), cal[key])
            continue
        scale = torch.tensor([max(c["x"] / 448.0, 1e-8)], device="cuda")
        norm = block.norm1
        if (_dit_band() == "fp4"
                and c.get("rows", 1 << 30) <= 64):
            # denoise band: the producer norms straight into packed
            # NVFP4 + swizzled scale factors and the pack consumes the
            # wire — the form the chain race seats automatically, taken
            # here by declaration
            from flash_rt.structures.impls.qkv_pack import (
                nvfp4_balance as pack_w4)
            producer = asm.take(norm_path, "adaln_producer",
                                lambda: bind_adaln_producer(
                                    norm, pairs, act_scale=None,
                                    rows=c["rows"],
                                    dim=norm.norm.normalized_shape[0],
                                    locator=locator, norm="layer",
                                    out_format="nvfp4"))
            if producer is None:
                continue
            parts = asm.take(attn_path, "qkv_pack_fp4",
                             lambda: pack_w4.bind_qkv_pack(
                                 [block.attn1.to_q, block.attn1.to_k,
                                  block.attn1.to_v],
                                 channel_amax=None, rows=c["rows"],
                                 wire=True))
            if parts is None:
                continue
            parts[0].accept_wire(producer.wire_sfa)
        else:
            producer = asm.take(norm_path, "adaln_producer",
                                lambda: bind_adaln_producer(
                                    norm, pairs, act_scale=scale,
                                    rows=c["rows"],
                                    dim=norm.norm.normalized_shape[0],
                                    locator=locator, norm="layer"))
            if producer is None:
                continue
            parts = asm.take(attn_path, "qkv_pack",
                             lambda: bind_qkv_pack(
                                 [block.attn1.to_q, block.attn1.to_k,
                                  block.attn1.to_v],
                                 scale, rows=c["rows"],
                                 in_dtype="fp8_static"))
            if parts is None:
                continue
        locator = locator or producer.locator
        asm.place(norm_path, producer)
        for attr, mod in zip(("to_q", "to_k", "to_v"), parts):
            asm.place(f"{attn_path}.{attr}", mod)

    # attention through the family entry: fa2 / fa4 / masked-mha / fp8
    # rank themselves; whichever the host's packages can serve binds, and
    # the losers' reasons ride along on the bound core.
    adapter_result = DiffusersAttentionAdapter()(model, run_once)
    observed, toggles, variants = dict(rope_observed), [], {}
    if adapter_result is not None:
        _, _, extras = adapter_result
        observed.update(extras.get("observed", {}))
        variants = extras.get("attention_variants", {})
        asm.refused.extend(extras.get("refused", []))
        reverts.extend(extras.get("revert", ()))
        if extras.get("toggle"):
            toggles.append(extras["toggle"])
        asm.families["attention_core"] = len(extras.get("observed", {}))

    cadence_statics = []
    if cross_sites:
        cadence_swaps, cadence_statics = bind_cross_attention_kv(
            cross_sites, cross_captures, replacements=asm.swaps)
        for p, m in cadence_swaps.items():
            asm.place(p, m)
        asm.families["cadence_static"] = len(cadence_swaps)

    extras = {"observed": observed, "toggles": toggles,
              "variants": variants, "cadence_statics": cadence_statics,
              "revert": reverts}
    return asm, extras


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------

def median_ms(fn, *, iters=5, rounds=9):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(rounds):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / iters)
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-assets", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    policy = load_policy(args.host, args.checkpoint, args.backbone_assets)
    model = policy.model
    fixture = torch.load(args.fixture, map_location="cpu",
                         weights_only=False)["inputs"]

    from flash_rt.structures import swap
    from flash_rt.structures.impls import unavailable_report

    captured = {}
    original_get_action = model.get_action

    def spy(inputs, options=None):
        captured["inputs"] = clone_tree(inputs)
        return original_get_action(inputs, options)

    model.get_action = spy
    with torch.inference_mode():
        policy.get_action(fixture)
    model.get_action = original_get_action

    backbone_inputs, action_inputs = model.prepare_input(
        dict(captured["inputs"]))
    backbone_inputs = clone_tree(backbone_inputs)
    action_inputs = clone_tree(action_inputs)

    def hot():
        torch.manual_seed(0)
        out = model.backbone(backbone_inputs)
        return model.action_head.get_action(
            out, action_inputs)["action_pred"]

    with torch.inference_mode():
        reference = hot().detach().float().cpu()
        baseline_ms = median_ms(lambda: hot())

    baseline_compiled_ms = None
    if args.compile:
        # both columns or neither: the eager pair is the community
        # convention, the compiled pair is the production form — a
        # speedup quoted across the two is not a measurement
        compiled_base = torch.compile(hot, mode="max-autotune-no-cudagraphs",
                                      fullgraph=False)
        with torch.inference_mode():
            for _ in range(8):
                compiled_base()
                torch.cuda.synchronize()
            baseline_compiled_ms = median_ms(lambda: compiled_base())
        torch._dynamo.reset()

    def run_once():
        with torch.inference_mode():
            hot()

    asm, extras = build(model, run_once)

    handle = swap.attach(model, asm.swaps, consume=False, observe=extras["observed"],
                         revert=extras["revert"], on_guard_fail="raise")

    if extras["cadence_statics"]:
        # cross-attention K/V follow the observation cadence: wire the
        # bank refresh into the producer's own forward, so every form of
        # the hot path (eager, compiled, captured) carries the current
        # observation through — and pays the refresh it would pay in
        # production
        from flash_rt.structures.impls.cadence_static.cross_attention \
            import wire_refresh_to_producer
        wire_refresh_to_producer(model, extras["cadence_statics"],
                                 run_once)
    with torch.inference_mode():
        treated = hot().detach().float().cpu()
        treated_ms = median_ms(lambda: hot())

    cos = float(torch.nn.functional.cosine_similarity(
        treated.flatten(), reference.flatten(), dim=0))
    ledger = handle.summary()

    compiled_ms = None
    if args.compile:
        torch._dynamo.reset()
        compiled = torch.compile(hot, mode="max-autotune-no-cudagraphs",
                                 fullgraph=False)
        with torch.inference_mode():
            for _ in range(8):
                compiled()
                torch.cuda.synchronize()
            compiled_ms = median_ms(lambda: compiled())

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seats_bound": dict(asm.families),
        "swaps": len(asm.swaps),
        "observed": len(extras["observed"]),
        "attention_variants": {
            path: info for path, info in
            list(extras["variants"].items())[:1]},
        "refused": len(asm.refused),
        "refusal_details": asm.refused,
        "kernel_unavailable": unavailable_report(),
        "ledger": ledger,
        "parity_cosine": cos,
        "baseline_eager_ms": baseline_ms,
        "treated_eager_ms": treated_ms,
        "baseline_compiled_ms": baseline_compiled_ms,
        "treated_compiled_ms": compiled_ms,
        "speedup_eager": baseline_ms / treated_ms,
        "speedup_compiled": (baseline_compiled_ms / compiled_ms
                             if compiled_ms and baseline_compiled_ms
                             else None),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str))

    handle.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
