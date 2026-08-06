"""Declarative weight spec for standalone Chameleon-7B on Thor.

Standard Chameleon 32-layer backbone layout:
attention_bias=false, mlp_bias=false, per-head Q/K norm, SwiGLU FFN.
Per-head QK Norm prevents norm_fuse, so QKV is fused with ``Cat`` and the
gate/up pair with ``FusedGateUp``.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    Attr,
    Cat,
    FusedGateUp,
    Quant,
    T,
    TensorList,
    ToFp16,
)


def _llm_block(*, use_fp8: bool = True) -> LayerBlock:
    """Chameleon-7B LLM — 32 layers, MHA (num_kv_heads=32, no interleave).

    4 quantized GEMMs per layer (qkv, o, gu, d) → 32 × 4 = 128 scales
    appended to ``target._llm_w_scales``.
    """
    qkv_tx  = [T(), Quant()]            if use_fp8 else [T()]
    o_tx    = [ToFp16(), T(), Quant()]  if use_fp8 else [ToFp16(), T()]
    gu_tx   = [T(), Quant()]            if use_fp8 else [T()]
    d_tx    = [ToFp16(), T(), Quant()]  if use_fp8 else [ToFp16(), T()]
    scale_into = "_llm_w_scales" if use_fp8 else None

    lp = "model.layers.{i}"
    items = [
        # ── Fused QKV (no bias; per-head QK norm as separate items) ──
        Item("qkv_w",
             Cat([f"{lp}.self_attn.q_proj.weight",
                  f"{lp}.self_attn.k_proj.weight",
                  f"{lp}.self_attn.v_proj.weight"], dim=0),
             qkv_tx,
             TensorList("_llm_qkv_w"), scale_into=scale_into),
        # ── O projection (no bias) ──
        Item("o_w", f"{lp}.self_attn.o_proj.weight",
             o_tx,
             TensorList("_llm_o_w"), scale_into=scale_into),
        # ── Fused GateUp ──
        Item("gu_w",
             FusedGateUp(gate=f"{lp}.mlp.gate_proj.weight",
                         up=f"{lp}.mlp.up_proj.weight"),
             gu_tx,
             TensorList("_llm_gu_w"), scale_into=scale_into),
        # ── Down projection ──
        Item("d_w", f"{lp}.mlp.down_proj.weight",
             d_tx,
             TensorList("_llm_d_w"), scale_into=scale_into),
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


def build_spec(*, use_fp8: bool = True) -> ModelWeightSpec:
    """Build the standalone Chameleon-7B Thor weight spec."""
    return ModelWeightSpec(
        framework="torch",
        blocks=[_llm_block(use_fp8=use_fp8)],
        singletons=[
            Item("embed_w", "model.embed_tokens.weight",
                 [ToFp16()], Attr("_llm_embed_w")),
            Item("norm_w", "model.norm.weight",
                 [ToFp16()], Attr("_llm_norm_w")),
            Item("lm_head_w", "lm_head.weight",
                 [ToFp16()], Attr("_llm_lm_head_w")),
        ],
    )


__all__ = ["build_spec"]
