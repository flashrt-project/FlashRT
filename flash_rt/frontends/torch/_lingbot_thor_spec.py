"""FlashRT — LingBot-VLA weight spec (Thor, torch) — declarative spec.

Source checkpoint: ``robbyant/lingbot-vla-4b`` (HuggingFace + ModelScope).
1555 fp32 tensors in a single ``model.safetensors`` (~16.7 GB).

This spec is BF16 round-trip — no Quant transforms (FP8 calibration lands
later). The on-device pipeline casts to bf16 + .contiguous() at load time;
weights themselves stay in their safetensors-native fp32 storage in the
target attributes until then.

Verified against the upstream LingBot weight inventory (1555 tensors):

  Top-level singletons                                      19
    action heads (state/in/out, time_mlp in+out, 5x{w,b})   10
    qwenvl top-level (embed_tokens, norm, vit specials,      9
                      visual.merger.ln_q + mlp.{0,2},
                      visual.patch_embed.proj,
                      qwen_expert.model.norm)
  VLM 36 layers  × 12 items                                 432
  Expert 36 layers × 20 items                               720
  ViT 32 blocks  × 12 items                                 384
                                                          ─────
                                                           1555  ✓
"""

from __future__ import annotations

from flash_rt.executors.torch_weights import Attr, TensorList
from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec


# ════════════════════════════════════════════════════════════════════
#  Top-level singletons
# ════════════════════════════════════════════════════════════════════

def _action_head_items() -> list[Item]:
    """Action-head singletons living directly under ``model.``.

    All 5 are nn.Linear, so each contributes (weight, bias):

      state_proj             75 → 768
      action_in_proj         75 → 768
      action_out_proj       768 → 75       velocity head
      action_time_mlp_in   1536 → 768      (1536 = sin+cos of timestep)
      action_time_mlp_out   768 → 768
    """
    items: list[Item] = []
    for name in (
        "state_proj", "action_in_proj", "action_out_proj",
        "action_time_mlp_in", "action_time_mlp_out",
    ):
        items.append(Item(
            name=f"{name}.weight",
            key=f"model.{name}.weight",
            sink=Attr(f"{name}_weight"),
        ))
        items.append(Item(
            name=f"{name}.bias",
            key=f"model.{name}.bias",
            sink=Attr(f"{name}_bias"),
        ))
    return items


def _qwenvl_top_singletons() -> list[Item]:
    """Top-level qwenvl + qwen_expert singletons (9 items)."""
    base = "model.qwenvl_with_expert"
    return [
        # VLM embed + final norm
        Item("vlm.embed_tokens",
             key=f"{base}.qwenvl.model.embed_tokens.weight",
             sink=Attr("vlm_embed_tokens_weight")),
        Item("vlm.norm",
             key=f"{base}.qwenvl.model.norm.weight",
             sink=Attr("vlm_norm_weight")),

        # Expert final norm (no embed_tokens — expert has no language head)
        Item("expert.norm",
             key=f"{base}.qwen_expert.model.norm.weight",
             sink=Attr("expert_norm_weight")),

        # ViT specials
        Item("vit.patch_embed.proj",
             key=f"{base}.qwenvl.visual.patch_embed.proj.weight",
             sink=Attr("vit_patch_embed_proj_weight")),
        Item("vit.merger.ln_q",
             key=f"{base}.qwenvl.visual.merger.ln_q.weight",
             sink=Attr("vit_merger_ln_q_weight")),
        Item("vit.merger.mlp.0.weight",
             key=f"{base}.qwenvl.visual.merger.mlp.0.weight",
             sink=Attr("vit_merger_mlp_0_weight")),
        Item("vit.merger.mlp.0.bias",
             key=f"{base}.qwenvl.visual.merger.mlp.0.bias",
             sink=Attr("vit_merger_mlp_0_bias")),
        Item("vit.merger.mlp.2.weight",
             key=f"{base}.qwenvl.visual.merger.mlp.2.weight",
             sink=Attr("vit_merger_mlp_2_weight")),
        Item("vit.merger.mlp.2.bias",
             key=f"{base}.qwenvl.visual.merger.mlp.2.bias",
             sink=Attr("vit_merger_mlp_2_bias")),
    ]


# ════════════════════════════════════════════════════════════════════
#  Layer blocks
# ════════════════════════════════════════════════════════════════════

# VLM (Qwen2.5-VL LLM backbone, 36 layers, hidden=2048)
#   GQA 16Q/2KV, head_dim=128
#   q/k/v_proj have bias; o_proj no bias
#   FFN gate/up/down no bias
#   Pre-LN: input_layernorm (RMS), post_attention_layernorm (RMS)
_VLM_LAYER_ITEMS: list[Item] = [
    Item("input_layernorm",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.input_layernorm.weight",
         sink=TensorList("vlm_layer_input_layernorm_weights")),
    Item("post_attention_layernorm",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.post_attention_layernorm.weight",
         sink=TensorList("vlm_layer_post_attn_layernorm_weights")),
    Item("q_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.q_proj.weight",
         sink=TensorList("vlm_layer_q_proj_weights")),
    Item("q_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.q_proj.bias",
         sink=TensorList("vlm_layer_q_proj_biases")),
    Item("k_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.k_proj.weight",
         sink=TensorList("vlm_layer_k_proj_weights")),
    Item("k_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.k_proj.bias",
         sink=TensorList("vlm_layer_k_proj_biases")),
    Item("v_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.v_proj.weight",
         sink=TensorList("vlm_layer_v_proj_weights")),
    Item("v_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.v_proj.bias",
         sink=TensorList("vlm_layer_v_proj_biases")),
    Item("o_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.self_attn.o_proj.weight",
         sink=TensorList("vlm_layer_o_proj_weights")),
    Item("mlp.gate_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.mlp.gate_proj.weight",
         sink=TensorList("vlm_layer_mlp_gate_proj_weights")),
    Item("mlp.up_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.mlp.up_proj.weight",
         sink=TensorList("vlm_layer_mlp_up_proj_weights")),
    Item("mlp.down_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.model.layers.{i}.mlp.down_proj.weight",
         sink=TensorList("vlm_layer_mlp_down_proj_weights")),
]

# Action Expert (Qwen2-768, 36 layers, hidden=768, Mixed-Head Q→2048)
#   Pre-LN is AdaRMSNorm: weight (RMS gain) + Linear beta (shift) +
#     Linear gamma (scale). Each Linear contributes weight+bias.
#   Attention shape: q_proj [2048, 768], k/v_proj [256, 768] (GQA),
#     o_proj [768, 2048] (Mixed-Head: head-space → expert hidden)
_EXPERT_LAYER_ITEMS: list[Item] = [
    # AdaRMSNorm input
    Item("input_layernorm.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.input_layernorm.weight",
         sink=TensorList("expert_layer_input_layernorm_weights")),
    Item("input_layernorm.beta.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.input_layernorm.beta.weight",
         sink=TensorList("expert_layer_input_layernorm_beta_weights")),
    Item("input_layernorm.beta.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.input_layernorm.beta.bias",
         sink=TensorList("expert_layer_input_layernorm_beta_biases")),
    Item("input_layernorm.gamma.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.input_layernorm.gamma.weight",
         sink=TensorList("expert_layer_input_layernorm_gamma_weights")),
    Item("input_layernorm.gamma.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.input_layernorm.gamma.bias",
         sink=TensorList("expert_layer_input_layernorm_gamma_biases")),
    # AdaRMSNorm post
    Item("post_attention_layernorm.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.post_attention_layernorm.weight",
         sink=TensorList("expert_layer_post_attn_layernorm_weights")),
    Item("post_attention_layernorm.beta.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.post_attention_layernorm.beta.weight",
         sink=TensorList("expert_layer_post_attn_layernorm_beta_weights")),
    Item("post_attention_layernorm.beta.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.post_attention_layernorm.beta.bias",
         sink=TensorList("expert_layer_post_attn_layernorm_beta_biases")),
    Item("post_attention_layernorm.gamma.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.post_attention_layernorm.gamma.weight",
         sink=TensorList("expert_layer_post_attn_layernorm_gamma_weights")),
    Item("post_attention_layernorm.gamma.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.post_attention_layernorm.gamma.bias",
         sink=TensorList("expert_layer_post_attn_layernorm_gamma_biases")),
    # Attention
    Item("q_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.q_proj.weight",
         sink=TensorList("expert_layer_q_proj_weights")),
    Item("q_proj.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.q_proj.bias",
         sink=TensorList("expert_layer_q_proj_biases")),
    Item("k_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.k_proj.weight",
         sink=TensorList("expert_layer_k_proj_weights")),
    Item("k_proj.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.k_proj.bias",
         sink=TensorList("expert_layer_k_proj_biases")),
    Item("v_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.v_proj.weight",
         sink=TensorList("expert_layer_v_proj_weights")),
    Item("v_proj.bias",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.v_proj.bias",
         sink=TensorList("expert_layer_v_proj_biases")),
    Item("o_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.self_attn.o_proj.weight",
         sink=TensorList("expert_layer_o_proj_weights")),
    # FFN
    Item("mlp.gate_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.mlp.gate_proj.weight",
         sink=TensorList("expert_layer_mlp_gate_proj_weights")),
    Item("mlp.up_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.mlp.up_proj.weight",
         sink=TensorList("expert_layer_mlp_up_proj_weights")),
    Item("mlp.down_proj.weight",
         key="model.qwenvl_with_expert.qwen_expert.model.layers.{i}.mlp.down_proj.weight",
         sink=TensorList("expert_layer_mlp_down_proj_weights")),
]

# Qwen2.5-VL ViT (32 blocks, hidden=1280, 16 heads, head_dim=80, MHA)
#   QKV fused: single Linear [3840, 1280] with bias
#   All MLP / proj linears have bias
#   norm1/norm2 are RMSNorm (no bias)
_VIT_BLOCK_ITEMS: list[Item] = [
    Item("attn.qkv.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.attn.qkv.weight",
         sink=TensorList("vit_block_attn_qkv_weights")),
    Item("attn.qkv.bias",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.attn.qkv.bias",
         sink=TensorList("vit_block_attn_qkv_biases")),
    Item("attn.proj.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.attn.proj.weight",
         sink=TensorList("vit_block_attn_proj_weights")),
    Item("attn.proj.bias",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.attn.proj.bias",
         sink=TensorList("vit_block_attn_proj_biases")),
    Item("mlp.gate_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.gate_proj.weight",
         sink=TensorList("vit_block_mlp_gate_proj_weights")),
    Item("mlp.gate_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.gate_proj.bias",
         sink=TensorList("vit_block_mlp_gate_proj_biases")),
    Item("mlp.up_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.up_proj.weight",
         sink=TensorList("vit_block_mlp_up_proj_weights")),
    Item("mlp.up_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.up_proj.bias",
         sink=TensorList("vit_block_mlp_up_proj_biases")),
    Item("mlp.down_proj.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.down_proj.weight",
         sink=TensorList("vit_block_mlp_down_proj_weights")),
    Item("mlp.down_proj.bias",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.mlp.down_proj.bias",
         sink=TensorList("vit_block_mlp_down_proj_biases")),
    Item("norm1.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.norm1.weight",
         sink=TensorList("vit_block_norm1_weights")),
    Item("norm2.weight",
         key="model.qwenvl_with_expert.qwenvl.visual.blocks.{i}.norm2.weight",
         sink=TensorList("vit_block_norm2_weights")),
]


# ════════════════════════════════════════════════════════════════════
#  Top-level spec builder
# ════════════════════════════════════════════════════════════════════

def build_spec(*, num_vlm_layers: int = 36,
               num_expert_layers: int = 36,
               num_vit_blocks: int = 32) -> ModelWeightSpec:
    """Build the LingBot-VLA weight-spec for the Thor torch frontend.

    Returns a ``ModelWeightSpec`` whose enumerated keys match the 1555
    tensors of ``robbyant/lingbot-vla-4b`` exactly (verified against the
    upstream weight inventory).
    """
    return ModelWeightSpec(
        framework="torch",
        singletons=_action_head_items() + _qwenvl_top_singletons(),
        blocks=[
            LayerBlock(
                name="vlm",
                prefix_fmt="",  # full keys in each Item; runner only needs {i}
                num_layers=num_vlm_layers,
                items=_VLM_LAYER_ITEMS,
            ),
            LayerBlock(
                name="expert",
                prefix_fmt="",
                num_layers=num_expert_layers,
                items=_EXPERT_LAYER_ITEMS,
            ),
            LayerBlock(
                name="vit",
                prefix_fmt="",
                num_layers=num_vit_blocks,
                items=_VIT_BLOCK_ITEMS,
            ),
        ],
    )


def enumerate_keys(spec: ModelWeightSpec) -> list[str]:
    """Enumerate every checkpoint key this spec will request.

    Mirrors the iteration order of ``WeightLoader.run``: singletons
    first, then each block in order, then each layer in each block,
    then each item per layer.
    """
    keys: list[str] = []
    for item in spec.singletons:
        keys.append(item.key)
    for block in spec.blocks:
        for i in range(block.num_layers):
            for item in block.items:
                keys.append(item.key.format(i=i))
    return keys
