"""Declarative BF16 weight spec for the LingBot-VLA-v2-6B checkpoint.

Covers ViT, DeepStack, LLM and action-expert blocks plus singletons:

* ViT block       — 24 layers (Qwen3-VL vision encoder).
* DeepStack block — 3 mergers for ViT features at layers [5, 11, 17].
* LLM block       — 36 layers (GQA + per-head Q/K RMSNorm + SwiGLU).
* Action expert  — 36 layers (AdaRMSNorm + GQA + token MoE).
* Singletons     — patch/position/token embeddings, final norms, main
                   merger, state/action/time projections and query weights.

Dense linear weights use [in, out] layout. Routed MoE weights retain
[expert, out, in]; the router uses FP32 [expert, hidden] weights.
QKV and gate/up concatenation preserves BF16; norms remain separate.

The 76 tensors in the four *_align_head training projectors are omitted.
Query embeddings and shared task projections are used by sample_actions;
query pooling is performed by the frontend/pipeline.
"""

import torch

from flash_rt.executors.torch_weights import (
    Attr,
    Cat,
    T,
    TensorList,
    ToBf16,
    ToFp32,
)
from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec

# ════════════════════════════════════════════════════════════════════
#  Block builders
# ════════════════════════════════════════════════════════════════════

_VLM = "model.qwenvl_with_expert.qwenvl.model"
_EXPERT = "model.qwenvl_with_expert.qwen_expert.model"


def _vit_block() -> LayerBlock:
    """Qwen3-VL ViT: 24 layers × (LN1, QKV+bias, O+bias, LN2, FC1+bias, FC2+bias)."""
    return LayerBlock(
        name="qwen3vl_vit", prefix_fmt=f"{_VLM}.visual.blocks.{{i}}", num_layers=24,
        items=[
            Item("ln1_w", "{prefix}.norm1.weight", [ToBf16()], TensorList("_vit_ln1_w")),
            Item("ln1_b", "{prefix}.norm1.bias", [ToBf16()], TensorList("_vit_ln1_b")),
            # Pre-fused QKV: [3072, 1024] → T → [1024, 3072].
            Item("qkv_w", "{prefix}.attn.qkv.weight", [ToBf16(), T()], TensorList("_vit_qkv_w")),
            Item("qkv_b", "{prefix}.attn.qkv.bias", [ToBf16()], TensorList("_vit_qkv_b")),
            Item("o_w", "{prefix}.attn.proj.weight", [ToBf16(), T()], TensorList("_vit_o_w")),
            Item("o_b", "{prefix}.attn.proj.bias", [ToBf16()], TensorList("_vit_o_b")),
            Item("ln2_w", "{prefix}.norm2.weight", [ToBf16()], TensorList("_vit_ln2_w")),
            Item("ln2_b", "{prefix}.norm2.bias", [ToBf16()], TensorList("_vit_ln2_b")),
            Item(
                "fc1_w", "{prefix}.mlp.linear_fc1.weight", [ToBf16(), T()], TensorList("_vit_fc1_w")
            ),
            Item("fc1_b", "{prefix}.mlp.linear_fc1.bias", [ToBf16()], TensorList("_vit_fc1_b")),
            Item(
                "fc2_w", "{prefix}.mlp.linear_fc2.weight", [ToBf16(), T()], TensorList("_vit_fc2_w")
            ),
            Item("fc2_b", "{prefix}.mlp.linear_fc2.bias", [ToBf16()], TensorList("_vit_fc2_b")),
        ],
    )


def _deepstack_block() -> LayerBlock:
    """Three DeepStack mergers: LayerNorm + FC1 + FC2, all BF16.

    LayerNorm operates on merged 4096-dim tokens; the main merger's norm
    operates on 1024-dim patches before the spatial merge.
    """
    return LayerBlock(
        name="qwen3vl_deepstack", prefix_fmt=f"{_VLM}.visual.deepstack_merger_list.{{i}}",
        num_layers=3,
        items=[
            Item("ln_w", "{prefix}.norm.weight", [ToBf16()], TensorList("_ds_ln_w")),
            Item("ln_b", "{prefix}.norm.bias", [ToBf16()], TensorList("_ds_ln_b")),
            Item("fc1_w", "{prefix}.linear_fc1.weight", [ToBf16(), T()], TensorList("_ds_fc1_w")),
            Item("fc1_b", "{prefix}.linear_fc1.bias", [ToBf16()], TensorList("_ds_fc1_b")),
            Item("fc2_w", "{prefix}.linear_fc2.weight", [ToBf16(), T()], TensorList("_ds_fc2_w")),
            Item("fc2_b", "{prefix}.linear_fc2.bias", [ToBf16()], TensorList("_ds_fc2_b")),
        ],
    )


def _llm_block() -> LayerBlock:
    """Qwen3-VL LLM: 36 layers, GQA attention + SwiGLU FFN.

    GQA: 32Q heads / 8KV heads / head_dim=128. Per-head Q/K RMSNorm
    weights have shape [128] and are applied before M-RoPE.
    """
    return LayerBlock(
        name="qwen3vl_llm", prefix_fmt=f"{_VLM}.language_model.layers.{{i}}", num_layers=36,
        items=[
            Item(
                "input_ln_w", "{prefix}.input_layernorm.weight", [ToBf16()],
                TensorList("_llm_input_ln_w"),
            ),
            # Fused QKV: cat(q, k, v) in BF16, then transpose.
            # [6144, 2560] → T → [2560, 6144].
            Item(
                "qkv_w",
                Cat(
                    [
                        "{prefix}.self_attn.q_proj.weight", "{prefix}.self_attn.k_proj.weight",
                        "{prefix}.self_attn.v_proj.weight",
                    ],
                    dtype=torch.bfloat16,
                ),
                [T()], TensorList("_llm_qkv_w"),
            ),
            # Per-head Q/K RMSNorm weights.
            Item(
                "q_norm_w", "{prefix}.self_attn.q_norm.weight", [ToBf16()],
                TensorList("_llm_q_norm_w"),
            ),
            Item(
                "k_norm_w", "{prefix}.self_attn.k_norm.weight", [ToBf16()],
                TensorList("_llm_k_norm_w"),
            ),
            Item(
                "o_w", "{prefix}.self_attn.o_proj.weight", [ToBf16(), T()], TensorList("_llm_o_w")
            ),
            Item(
                "post_ln_w", "{prefix}.post_attention_layernorm.weight", [ToBf16()],
                TensorList("_llm_post_ln_w"),
            ),
            # SwiGLU: cat(gate, up) → T → [2560, 19456].
            Item(
                "gate_up_w",
                Cat(
                    [
                        "{prefix}.mlp.gate_proj.weight", "{prefix}.mlp.up_proj.weight",
                    ],
                    dtype=torch.bfloat16,
                ),
                [T()], TensorList("_llm_gate_up_w"),
            ),
            Item(
                "down_w", "{prefix}.mlp.down_proj.weight", [ToBf16(), T()],
                TensorList("_llm_down_w"),
            ),
        ],
    )


def _action_expert_block() -> LayerBlock:
    """36-layer action expert: AdaRMSNorm + GQA attention + token MoE.

    Hidden=768; attention uses 32Q / 8KV heads with head_dim=128.
    Each MoE layer has 32 routed experts (top-k=4) and one shared expert.
    """
    items = []
    # AdaRMSNorm: norm weight + separate gamma/beta timestep projections.
    # Keep the affine terms separate to preserve BF16 rounding order.
    for tag, norm in (("input", "input_layernorm"), ("post", "post_attention_layernorm")):
        items.append(
            Item(
                f"{tag}_ln_w", f"{{prefix}}.{norm}.weight", [ToBf16()],
                TensorList(f"_expert_{tag}_ln_w"),
            )
        )
        for affine in ("gamma", "beta"):
            items.extend(
                [
                    Item(
                        f"{tag}_{affine}_w", f"{{prefix}}.{norm}.{affine}.weight", [ToBf16(), T()],
                        TensorList(f"_expert_{tag}_{affine}_w"),
                    ),
                    Item(
                        f"{tag}_{affine}_b", f"{{prefix}}.{norm}.{affine}.bias", [ToBf16()],
                        TensorList(f"_expert_{tag}_{affine}_b"),
                    ),
                ]
            )
    items.extend(
        [
            # Fused QKV: [6144, 768] → T → [768, 6144]; bias stays [6144].
            Item(
                "qkv_w",
                Cat(
                    [
                        "{prefix}.self_attn.q_proj.weight", "{prefix}.self_attn.k_proj.weight",
                        "{prefix}.self_attn.v_proj.weight",
                    ],
                    dtype=torch.bfloat16,
                ),
                [T()], TensorList("_expert_qkv_w"),
            ),
            Item(
                "qkv_b",
                Cat(
                    [
                        "{prefix}.self_attn.q_proj.bias", "{prefix}.self_attn.k_proj.bias",
                        "{prefix}.self_attn.v_proj.bias",
                    ],
                    dtype=torch.bfloat16,
                ),
                [], TensorList("_expert_qkv_b"),
            ),
            Item(
                "o_w", "{prefix}.self_attn.o_proj.weight", [ToBf16(), T()],
                TensorList("_expert_o_w"),
            ),
            # FP32 router: [32, 768] for F.linear; correction bias for top-k choice.
            Item("router_w", "{prefix}.mlp.gate.weight", [ToFp32()], TensorList("_moe_router_w")),
            Item(
                "router_bias", "{prefix}.mlp.e_score_correction_bias", [ToFp32()],
                TensorList("_moe_router_bias"),
            ),
            # Routed experts: preserve [32, 512, 768] / [32, 768, 512].
            Item("gate_w", "{prefix}.mlp.experts.gate_proj", [ToBf16()], TensorList("_moe_gate_w")),
            Item("up_w", "{prefix}.mlp.experts.up_proj", [ToBf16()], TensorList("_moe_up_w")),
            Item("down_w", "{prefix}.mlp.experts.down_proj", [ToBf16()], TensorList("_moe_down_w")),
            # Shared expert SwiGLU: gate/up [768, 1408], down [704, 768].
            Item(
                "shared_gate_up_w",
                Cat(
                    [
                        "{prefix}.mlp.shared_expert.gate_proj.weight",
                        "{prefix}.mlp.shared_expert.up_proj.weight",
                    ],
                    dtype=torch.bfloat16,
                ),
                [T()], TensorList("_moe_shared_gate_up_w"),
            ),
            Item(
                "shared_down_w", "{prefix}.mlp.shared_expert.down_proj.weight", [ToBf16(), T()],
                TensorList("_moe_shared_down_w"),
            ),
        ]
    )
    return LayerBlock(
        name="action_expert", prefix_fmt=f"{_EXPERT}.layers.{{i}}", num_layers=36, items=items,
    )


# ════════════════════════════════════════════════════════════════════
#  Singletons
# ════════════════════════════════════════════════════════════════════


def _singletons() -> list[Item]:
    """Embeddings, final norms, main merger and policy input/output weights."""
    items = [
        # ── Patch embedding (Conv3d) + position/token embedding tables ──
        Item("patch_w", f"{_VLM}.visual.patch_embed.proj.weight", [ToBf16()], Attr("_vit_patch_w")),
        Item("patch_b", f"{_VLM}.visual.patch_embed.proj.bias", [ToBf16()], Attr("_vit_patch_b")),
        Item("pos_embed", f"{_VLM}.visual.pos_embed.weight", [ToBf16()], Attr("_vit_pos_embed")),
        Item(
            "embed_tokens_w", f"{_VLM}.language_model.embed_tokens.weight", [ToBf16()],
            Attr("_embed_tokens_w"),
        ),
        # ── Final RMSNorms ──
        Item(
            "llm_final_ln_w", f"{_VLM}.language_model.norm.weight", [ToBf16()],
            Attr("_llm_final_ln_w"),
        ),
        Item("expert_final_ln_w", f"{_EXPERT}.norm.weight", [ToBf16()], Attr("_expert_final_ln_w")),
        # ── Main visual merger: LN(1024) + FC1(4096→4096) + FC2(4096→2560) ──
        Item("merger_ln_w", f"{_VLM}.visual.merger.norm.weight", [ToBf16()], Attr("_merger_ln_w")),
        Item("merger_ln_b", f"{_VLM}.visual.merger.norm.bias", [ToBf16()], Attr("_merger_ln_b")),
    ]
    for layer in ("fc1", "fc2"):
        key = f"{_VLM}.visual.merger.linear_{layer}"
        items.extend(
            [
                Item(
                    f"merger_{layer}_w", f"{key}.weight", [ToBf16(), T()],
                    Attr(f"_merger_{layer}_w"),
                ),
                Item(f"merger_{layer}_b", f"{key}.bias", [ToBf16()], Attr(f"_merger_{layer}_b")),
            ]
        )
    # ── State/action/time projections + shared task projections ──
    for projection in (
        "state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in",
        "action_time_mlp_out", "current_shared_task_proj", "future_shared_task_proj",
    ):
        items.extend(
            [
                Item(
                    f"{projection}_w", f"model.{projection}.weight", [ToBf16(), T()],
                    Attr(f"_{projection}_w"),
                ),
                Item(
                    f"{projection}_b", f"model.{projection}.bias", [ToBf16()],
                    Attr(f"_{projection}_b"),
                ),
            ]
        )
    # ── Query embeddings [256, 2560]; pool to 8 tokens during prefix setup ──
    for query in (
        "depth_align_embs", "future_depth_align_embs", "current_video_align_embs",
        "future_video_align_embs",
    ):
        items.append(Item(query, f"model.{query}", [ToBf16()], Attr(f"_{query}")))
    return items


# ════════════════════════════════════════════════════════════════════
#  Spec assembly
# ════════════════════════════════════════════════════════════════════


def build_spec() -> ModelWeightSpec:
    """Build the BF16 inference spec with FP32 router weights and biases."""
    return ModelWeightSpec(
        framework="torch",
        blocks=[_vit_block(), _deepstack_block(), _llm_block(), _action_expert_block()],
        singletons=_singletons(),
    )


WEIGHT_SPEC = build_spec()

__all__ = ["build_spec", "WEIGHT_SPEC"]
