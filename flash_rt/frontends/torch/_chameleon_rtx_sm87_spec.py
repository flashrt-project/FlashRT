"""Declarative weight spec for upstream Chameleon-7B (HF layout).

Chameleon-7B LLM — 32 layers, MHA (num_kv_heads=32, no interleave).
``attention_bias`` and ``mlp_bias`` are both ``false`` in the checkpoint
config, so no bias items are emitted; the per-head QK-Norm weight *and
bias* tensors are part of the layer block.

Per-head QK Norm prevents norm_fuse; Cat is used for QKV fusion
(not FusedQKV) because the per-head norms are handled as separate
TensorList items.

The QK-Norm tensors are loaded verbatim (``ToFp16()`` only), preserving their
``(1, 128)`` shape. That is deliberate: ``qk_norm_rope_fused_fp16`` reads the
weight as a flat ``[head_dim]`` vector shared across heads, which is exactly
what the checkpoint's ``model_parallel_size == 1`` layout means (upstream
expands it with ``repeat_interleave`` at forward time). No reshape is needed.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import Attr, Cat, FusedGateUp, T, TensorList, ToFp16


def _llm_block() -> LayerBlock:
    """Chameleon-7B LLM — 32 layers, FP16 backbone (no quantized spec)."""
    lp = "model.layers.{i}"
    items = [
        # ── Fused QKV (MHA: no interleave, no norm_fuse) ──
        Item("qkv_w",
             Cat([f"{lp}.self_attn.q_proj.weight",
                  f"{lp}.self_attn.k_proj.weight",
                  f"{lp}.self_attn.v_proj.weight"], dim=0),
             [T()],
             TensorList("_llm_qkv_w")),
        # ── O projection ──
        Item("o_w", f"{lp}.self_attn.o_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_o_w")),
        # ── Fused GateUp (no norm_fuse — per-head QK Norm incompatible) ──
        Item("gu_w",
             FusedGateUp(gate=f"{lp}.mlp.gate_proj.weight",
                         up=f"{lp}.mlp.up_proj.weight"),
             [T()],
             TensorList("_llm_gu_w")),
        # ── Down projection ──
        Item("d_w", f"{lp}.mlp.down_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_d_w")),
        # ── Layer norms ──
        Item("input_ln_w", f"{lp}.input_layernorm.weight",
             [ToFp16()], TensorList("_llm_input_ln_w")),
        Item("post_ln_w", f"{lp}.post_attention_layernorm.weight",
             [ToFp16()], TensorList("_llm_post_ln_w")),
        # ── Per-head Q/K Norm ──
        Item("q_norm_w", f"{lp}.self_attn.q_norm.weight",
             [ToFp16()], TensorList("_llm_q_norm_w")),
        Item("q_norm_b", f"{lp}.self_attn.q_norm.bias",
             [ToFp16()], TensorList("_llm_q_norm_b")),
        Item("k_norm_w", f"{lp}.self_attn.k_norm.weight",
             [ToFp16()], TensorList("_llm_k_norm_w")),
        Item("k_norm_b", f"{lp}.self_attn.k_norm.bias",
             [ToFp16()], TensorList("_llm_k_norm_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=32, items=items, name="llm")


def build_spec() -> ModelWeightSpec:
    """Chameleon-7B: 32 decoder layers + embedding, final norm, lm_head."""
    return ModelWeightSpec(
        framework="torch",
        blocks=[_llm_block()],
        singletons=[
            Item("embed_w", "model.embed_tokens.weight",
                 [ToFp16()], Attr("_llm_embed_w")),
            Item("norm_w", "model.norm.weight",
                 [ToFp16()], Attr("_llm_norm_w")),
            # Live output projection.
            Item("lm_head_w", "lm_head.weight",
                 [ToFp16()], Attr("_llm_lm_head_w")),
        ],
    )


__all__ = ["build_spec"]
