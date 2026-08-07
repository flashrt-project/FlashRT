"""Declarative weight spec for HyVLATorchFrontendThor (BF16 baseline).

Hy-Embodied-0.5-VLA is a Mixture-of-Transformers dual-tower VLA:

  * ViT  ``dual_tower.vlm.model.visual.vision_tower`` — 27 SigLIP-so400m
    blocks (fused ``attn.qkv`` + bias, LayerNorm **with bias**, patch-16),
    6 of which (indices 3,7,11,15,19,23) run extra 6-frame spacetime
    attention at runtime (adopt-by-reference — no extra params).
  * merger ``dual_tower.vlm.model.visual.merger`` — proj1 → 2x2
    NormalizedDwPooler → GELU → proj2 (196→49 tokens/cam @ 2048).
  * VLM tower ``dual_tower.vlm.model.language_model.model`` — 32 layers,
    hidden 2048, GQA 16Q/4KV hd128, SwiGLU inter 6144, RMSNorm eps 1e-5,
    **QK-Norm (RMSNorm over head_dim) applied AFTER RoPE**. Every proj +
    input/post_attention_layernorm has a text and a vision (``_v``) copy
    (MoT routing); QK-norm is a single shared copy.
  * expert tower ``dual_tower.expert.model`` — 32 layers, hidden 1024,
    same head geometry, SwiGLU inter 2048. At inference runs 100% through
    the ``_v`` branch and reuses the VLM tower's QK-norm weights.
  * action head (structurally = Pi0): action_in_proj / action_out_proj /
    action_time_mlp_in/out / state_proj.
  * tied embedding: no ``embed_tokens`` in the checkpoint — the input
    embedding table is ``lm_head.weight`` (``tie_word_embeddings``).

The BF16 baseline keeps the whole model in **BF16** (no FP8 / no ``Quant``).
Weights are stored in the exact layout the reference implementation
consumes: fused QKV / Gate-Up as ``[N, D]`` (out, in) row-major so the
forward computes ``x @ w.t()``; single projections (o_proj, down_proj)
as ``[N, D]``.

All checkpoint keys are uniformly ``model.``-prefixed → strip it once at
the source. Spec-side keys therefore start ``dual_tower.*`` / ``action_*``
/ ``state_proj.*``.
"""

from __future__ import annotations

import torch

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    Attr,
    Cat,
    TensorList,
    ToBf16,
)

_BF16 = torch.bfloat16


# ════════════════════════════════════════════════════════════════════
#  ViT block (27 layers) — SigLIP-so400m isomorphic, LayerNorm WITH bias
# ════════════════════════════════════════════════════════════════════

def _vit_block() -> LayerBlock:
    bp = "dual_tower.vlm.model.visual.vision_tower.blocks.{i}"
    items = [
        Item("vit_ln1_w", f"{bp}.norm1.weight", [ToBf16()], TensorList("_vit_ln1_w")),
        Item("vit_ln1_b", f"{bp}.norm1.bias", [ToBf16()], TensorList("_vit_ln1_b")),
        Item("vit_ln2_w", f"{bp}.norm2.weight", [ToBf16()], TensorList("_vit_ln2_w")),
        Item("vit_ln2_b", f"{bp}.norm2.bias", [ToBf16()], TensorList("_vit_ln2_b")),
        # attn.qkv is already fused in the checkpoint: [3456, 1152] (+ bias).
        Item("vit_qkv_w", f"{bp}.attn.qkv.weight", [ToBf16()], TensorList("_vit_qkv_w")),
        Item("vit_qkv_b", f"{bp}.attn.qkv.bias", [ToBf16()], TensorList("_vit_qkv_b")),
        Item("vit_proj_w", f"{bp}.attn.proj.weight", [ToBf16()], TensorList("_vit_proj_w")),
        Item("vit_proj_b", f"{bp}.attn.proj.bias", [ToBf16()], TensorList("_vit_proj_b")),
        Item("vit_fc1_w", f"{bp}.mlp.fc1.weight", [ToBf16()], TensorList("_vit_fc1_w")),
        Item("vit_fc1_b", f"{bp}.mlp.fc1.bias", [ToBf16()], TensorList("_vit_fc1_b")),
        Item("vit_fc2_w", f"{bp}.mlp.fc2.weight", [ToBf16()], TensorList("_vit_fc2_w")),
        Item("vit_fc2_b", f"{bp}.mlp.fc2.bias", [ToBf16()], TensorList("_vit_fc2_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=27, items=items, name="vit")


# ════════════════════════════════════════════════════════════════════
#  VLM language tower (32 layers) — text + vision (_v) branches
# ════════════════════════════════════════════════════════════════════

def _vlm_block() -> LayerBlock:
    dp = "dual_tower.vlm.model.language_model.model.layers.{i}"
    sa = f"{dp}.self_attn"

    def branch(suffix: str, tag: str) -> list[Item]:
        # suffix "" = text branch, "_v" = vision branch.
        mlp = f"{dp}.mlp{'_v' if suffix else ''}"
        return [
            Item(f"vlm_qkv{tag}",
                 Cat([f"{sa}.q_proj{suffix}.weight",
                      f"{sa}.k_proj{suffix}.weight",
                      f"{sa}.v_proj{suffix}.weight"], dim=0, dtype=_BF16),
                 [], TensorList(f"_vlm_qkv{tag}")),
            Item(f"vlm_o{tag}", f"{sa}.o_proj{suffix}.weight",
                 [ToBf16()], TensorList(f"_vlm_o{tag}")),
            Item(f"vlm_gu{tag}",
                 Cat([f"{mlp}.gate_proj.weight", f"{mlp}.up_proj.weight"],
                     dim=0, dtype=_BF16),
                 [], TensorList(f"_vlm_gu{tag}")),
            Item(f"vlm_d{tag}", f"{mlp}.down_proj.weight",
                 [ToBf16()], TensorList(f"_vlm_d{tag}")),
            Item(f"vlm_ln_in{tag}", f"{dp}.input_layernorm{suffix}.weight",
                 [ToBf16()], TensorList(f"_vlm_ln_in{tag}")),
            Item(f"vlm_ln_post{tag}", f"{dp}.post_attention_layernorm{suffix}.weight",
                 [ToBf16()], TensorList(f"_vlm_ln_post{tag}")),
        ]

    items = branch("", "_t") + branch("_v", "_v") + [
        # Shared QK-norm (single copy, no _v twin).
        Item("qk_norm_q", f"{sa}.query_layernorm.weight",
             [ToBf16()], TensorList("_qk_norm_q")),
        Item("qk_norm_k", f"{sa}.key_layernorm.weight",
             [ToBf16()], TensorList("_qk_norm_k")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=32, items=items, name="vlm")


# ════════════════════════════════════════════════════════════════════
#  Expert tower (32 layers) — _v branch only
# ════════════════════════════════════════════════════════════════════

def _expert_block() -> LayerBlock:
    dp = "dual_tower.expert.model.layers.{i}"
    sa = f"{dp}.self_attn"
    mlp = f"{dp}.mlp_v"
    items = [
        Item("exp_qkv",
             Cat([f"{sa}.q_proj_v.weight",
                  f"{sa}.k_proj_v.weight",
                  f"{sa}.v_proj_v.weight"], dim=0, dtype=_BF16),
             [], TensorList("_exp_qkv_v")),
        Item("exp_o", f"{sa}.o_proj_v.weight", [ToBf16()], TensorList("_exp_o_v")),
        Item("exp_gu",
             Cat([f"{mlp}.gate_proj.weight", f"{mlp}.up_proj.weight"],
                 dim=0, dtype=_BF16),
             [], TensorList("_exp_gu_v")),
        Item("exp_d", f"{mlp}.down_proj.weight", [ToBf16()], TensorList("_exp_d_v")),
        Item("exp_ln_in", f"{dp}.input_layernorm_v.weight",
             [ToBf16()], TensorList("_exp_ln_in_v")),
        Item("exp_ln_post", f"{dp}.post_attention_layernorm_v.weight",
             [ToBf16()], TensorList("_exp_ln_post_v")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=32, items=items, name="expert")


# ════════════════════════════════════════════════════════════════════
#  Singletons — merger, action head, final norms, tied embed, ViT patch
# ════════════════════════════════════════════════════════════════════

def _singletons() -> list[Item]:
    mg = "dual_tower.vlm.model.visual.merger"
    vt = "dual_tower.vlm.model.visual.vision_tower"
    return [
        # --- ViT patch embed + learned pos embed (rescaled at runtime) ---
        Item("vit_patch_w", f"{vt}.patch_embed.proj.weight", [ToBf16()], Attr("_vit_patch_w")),
        Item("vit_patch_b", f"{vt}.patch_embed.proj.bias", [ToBf16()], Attr("_vit_patch_b")),
        Item("vit_pos", f"{vt}.pos_embed", [ToBf16()], Attr("_vit_pos_embed")),
        # --- merger ---
        Item("mg_proj1_w", f"{mg}.proj1.weight", [ToBf16()], Attr("_mg_proj1_w")),
        Item("mg_proj1_b", f"{mg}.proj1.bias", [ToBf16()], Attr("_mg_proj1_b")),
        Item("mg_proj2_w", f"{mg}.proj2.weight", [ToBf16()], Attr("_mg_proj2_w")),
        Item("mg_proj2_b", f"{mg}.proj2.bias", [ToBf16()], Attr("_mg_proj2_b")),
        Item("mg_pred0_w", f"{mg}.pooler.predictor.0.weight", [ToBf16()], Attr("_mg_pred0_w")),
        Item("mg_pred0_b", f"{mg}.pooler.predictor.0.bias", [ToBf16()], Attr("_mg_pred0_b")),
        Item("mg_pred2_w", f"{mg}.pooler.predictor.2.weight", [ToBf16()], Attr("_mg_pred2_w")),
        Item("mg_pred2_b", f"{mg}.pooler.predictor.2.bias", [ToBf16()], Attr("_mg_pred2_b")),
        # --- action head ---
        Item("ain_w", "action_in_proj.weight", [ToBf16()], Attr("_ain_w")),
        Item("ain_b", "action_in_proj.bias", [ToBf16()], Attr("_ain_b")),
        Item("aout_w", "action_out_proj.weight", [ToBf16()], Attr("_aout_w")),
        Item("aout_b", "action_out_proj.bias", [ToBf16()], Attr("_aout_b")),
        Item("atmlp_in_w", "action_time_mlp_in.weight", [ToBf16()], Attr("_atmlp_in_w")),
        Item("atmlp_in_b", "action_time_mlp_in.bias", [ToBf16()], Attr("_atmlp_in_b")),
        Item("atmlp_out_w", "action_time_mlp_out.weight", [ToBf16()], Attr("_atmlp_out_w")),
        Item("atmlp_out_b", "action_time_mlp_out.bias", [ToBf16()], Attr("_atmlp_out_b")),
        Item("state_w", "state_proj.weight", [ToBf16()], Attr("_state_w")),
        Item("state_b", "state_proj.bias", [ToBf16()], Attr("_state_b")),
        # --- expert final RMSNorm ---
        Item("exp_final_norm", "dual_tower.expert.model.norm.weight",
             [ToBf16()], Attr("_exp_final_norm_w")),
        # --- tied input embedding = lm_head.weight ---
        Item("embed", "dual_tower.vlm.model.language_model.lm_head.weight",
             [ToBf16()], Attr("_embed_weight")),
    ]


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="torch",
        singletons=_singletons(),
        blocks=[
            _vit_block(),
            _vlm_block(),
            _expert_block(),
        ],
    )


__all__ = ["build_spec"]
