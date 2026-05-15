"""Shared LayerBlock builders for Thor torch frontends (stage 7.4).

Extracted from ``_pi05_thor_spec.py`` because Pi0.5 / Pi0 / Pi0-FAST /
GROOT (SigLIP2) share the SigLIP 27-layer block verbatim, and Pi0.5 /
Pi0 also share the Paligemma 18-layer encoder block. Keeping these in
one place removes copy-paste risk when the block schema changes.

Each factory returns a fresh ``LayerBlock`` — callers get a new instance
so spec modules don't alias mutable state across frontends.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock
from flash_rt.executors.torch_weights import (
    Cat,
    FusedGateUp,
    FusedQKV,
    Quant,
    T,
    TensorList,
    ToFp16,
)


def paligemma_siglip_block(
    *,
    model_root: str = "paligemma_with_expert.paligemma.model",
    num_layers: int = 27,
    use_fp8: bool = True,
) -> LayerBlock:
    """SigLIP encoder block used by Pi0.5 / Pi0 / Pi0-FAST torch frontends.

    27 layers × (LN1, LN2, fused QKV, O, FC1, FC2). All quantized GEMMs
    go through ``.T.contiguous()`` + per-tensor FP8 quant; scales land
    in ``target._sig_alpha`` in (q, o, up, down) order per layer.

    When ``use_fp8=False``, weights stay FP16 in the same ``[K, N]``
    row-major layout (the ``T()`` transpose is kept so ``gemm.fp16_nn``
    can read them directly). ``Quant()`` is dropped and ``_sig_alpha``
    is not populated.
    """
    qkv_tx  = [T(), Quant()]              if use_fp8 else [T()]
    o_tx    = [ToFp16(), T(), Quant()]    if use_fp8 else [ToFp16(), T()]
    up_tx   = [ToFp16(), T(), Quant()]    if use_fp8 else [ToFp16(), T()]
    down_tx = [ToFp16(), T(), Quant()]    if use_fp8 else [ToFp16(), T()]
    scale_into = "_sig_alpha" if use_fp8 else None
    vp = f"{model_root}.vision_tower.vision_model.encoder.layers.{{i}}"
    items = [
        Item("ln_attn_w", f"{vp}.layer_norm1.weight", [ToFp16()], TensorList("_sig_ln_attn_w")),
        Item("ln_attn_b", f"{vp}.layer_norm1.bias",   [ToFp16()], TensorList("_sig_ln_attn_b")),
        Item("ln_ffn_w",  f"{vp}.layer_norm2.weight", [ToFp16()], TensorList("_sig_ln_ffn_w")),
        Item("ln_ffn_b",  f"{vp}.layer_norm2.bias",   [ToFp16()], TensorList("_sig_ln_ffn_b")),

        Item("qkv_w",
             Cat([f"{vp}.self_attn.q_proj.weight",
                  f"{vp}.self_attn.k_proj.weight",
                  f"{vp}.self_attn.v_proj.weight"], dim=0),
             qkv_tx,
             TensorList("_sig_qkv_w"), scale_into=scale_into),
        Item("qkv_b",
             Cat([f"{vp}.self_attn.q_proj.bias",
                  f"{vp}.self_attn.k_proj.bias",
                  f"{vp}.self_attn.v_proj.bias"], dim=0),
             [],
             TensorList("_sig_qkv_b")),

        Item("o_w", f"{vp}.self_attn.out_proj.weight",
             o_tx,
             TensorList("_sig_o_w"), scale_into=scale_into),
        Item("o_b", f"{vp}.self_attn.out_proj.bias",
             [ToFp16()], TensorList("_sig_o_b")),

        Item("up_w", f"{vp}.mlp.fc1.weight",
             up_tx,
             TensorList("_sig_up_w"), scale_into=scale_into),
        Item("up_b", f"{vp}.mlp.fc1.bias",
             [ToFp16()], TensorList("_sig_up_b")),

        Item("down_w", f"{vp}.mlp.fc2.weight",
             down_tx,
             TensorList("_sig_down_w"), scale_into=scale_into),
        Item("down_b", f"{vp}.mlp.fc2.bias",
             [ToFp16()], TensorList("_sig_down_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="siglip")


def paligemma_encoder_block(
    *,
    model_root: str = "paligemma_with_expert.paligemma.model",
    num_layers: int = 18,
    use_fp8: bool = True,
) -> LayerBlock:
    """Paligemma encoder (18 layers, GQA, AdaRMSNorm fused into QKV/gate-up).

    Matches the encoder loader used by Pi0.5 and Pi0 torch frontends:
      qkv  : FusedQKV(interleave 8/1, norm_fuse=input_layernorm)  → [Quant()]
      o    : ToFp16 → Quant
      gu   : FusedGateUp(norm_fuse=post_attention_layernorm)      → [Quant()]
      d    : ToFp16 → Quant
    Scales append to ``target._enc_w_scales`` in (q, o, gu, d) order.

    When ``use_fp8=False``, ``Quant()`` is dropped and a ``T()``
    transpose is added so weights land in ``[K, N]`` row-major (the
    layout ``gemm.fp16_nn`` reads). FusedQKV / FusedGateUp natively
    produce ``[N, K]``; ``T()`` flips them to the row-major NN layout.
    """
    qkv_tx = [Quant()]            if use_fp8 else [T()]
    o_tx   = [ToFp16(), Quant()]  if use_fp8 else [ToFp16(), T()]
    gu_tx  = [Quant()]            if use_fp8 else [T()]
    d_tx   = [ToFp16(), Quant()]  if use_fp8 else [ToFp16(), T()]
    scale_into = "_enc_w_scales" if use_fp8 else None
    ep = f"{model_root}.language_model.layers.{{i}}"
    items = [
        Item("qkv_w",
             FusedQKV(q=f"{ep}.self_attn.q_proj.weight",
                      k=f"{ep}.self_attn.k_proj.weight",
                      v=f"{ep}.self_attn.v_proj.weight",
                      norm_fuse=f"{ep}.input_layernorm.weight",
                      interleave_q_heads=8,
                      interleave_k_heads=1),
             qkv_tx,
             TensorList("_enc_qkv_w"), scale_into=scale_into),
        Item("o_w", f"{ep}.self_attn.o_proj.weight",
             o_tx,
             TensorList("_enc_o_w"), scale_into=scale_into),
        Item("gu_w",
             FusedGateUp(gate=f"{ep}.mlp.gate_proj.weight",
                         up=f"{ep}.mlp.up_proj.weight",
                         norm_fuse=f"{ep}.post_attention_layernorm.weight"),
             gu_tx,
             TensorList("_enc_gu_w"), scale_into=scale_into),
        Item("d_w", f"{ep}.mlp.down_proj.weight",
             d_tx,
             TensorList("_enc_d_w"), scale_into=scale_into),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="encoder")


__all__ = ["paligemma_siglip_block", "paligemma_encoder_block"]
