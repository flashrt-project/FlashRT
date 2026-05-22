"""LingBot-VLA Mixed-Head joint attention — one-layer forward (reference path).

Replicates the per-layer attention block of
``QwenvlWithExpertModel.forward`` in ``lingbotvla/models/vla/pi0/
modeling_lingbot_vla.py`` (L1240-1314), starting from raw hiddens and
ending at the per-tower o_proj output (i.e. before residual + post-LN +
MLP). The full layer = this function + residual + post-LN(+ada_cond) +
MLP + residual; FFN/residual land later.

Math (per layer):

    # 1. Pre-norm
    h_vlm  = rms_norm(vlm_hidden,    weight=vlm.input_layernorm)
    h_exp  = ada_rms_norm(expert_hidden, ada_cond,
                            weight, gamma, beta from expert.input_layernorm.*)

    # 2. Q/K/V projections — Mixed-Head asymmetry: Expert hidden=768
    #    but Q_exp out = 16 × 128 = 2048 (matches VLM head space).
    Q_vlm = q_proj_vlm(h_vlm).view(B, L_vlm, 16, 128)
    K_vlm = k_proj_vlm(h_vlm).view(B, L_vlm,  2, 128)        # GQA n_rep=8
    V_vlm = v_proj_vlm(h_vlm).view(B, L_vlm,  2, 128)
    Q_exp = q_proj_exp(h_exp).view(B, L_exp, 16, 128)
    K_exp = k_proj_exp(h_exp).view(B, L_exp,  2, 128)
    V_exp = v_proj_exp(h_exp).view(B, L_exp,  2, 128)

    # 3. Joint sequence (concat on token axis)
    Q = cat([Q_vlm, Q_exp], dim=1)
    K = cat([K_vlm, K_exp], dim=1)
    V = cat([V_vlm, V_exp], dim=1)

    # 4. 1-D RoPE θ=10000 (via the adapter, bit-exact to upstream apply_rope)
    Q = apply_rope(Q, position_ids)
    K = apply_rope(K, position_ids)

    # 5. GQA: replicate KV from 2 heads → 16 heads
    K = K.repeat_interleave(8, dim=2)
    V = V.repeat_interleave(8, dim=2)

    # 6. Scaled dot-product attention with attention_mask
    A = scaled_dot_product_attention(Q.T, K.T, V.T, attn_mask=mask)
    A = A.T.reshape(B, L_total, 16*128)              # [B, L_total, 2048]

    # 7. Per-tower output projection
    out_vlm    = o_proj_vlm(A[:, :L_vlm])             # 2048 → 2048
    out_expert = o_proj_exp(A[:, L_vlm:])             # 2048 → 768  (Mixed-Head asym)

This function does NOT touch the KV cache; adds cache integration
for the denoise step where prefix-KV is reused. For now ``position_ids``
must cover both VLM and Expert tokens in one flat vector.

Reference for bit-exact tests: ``Qwen2DecoderLayer.forward`` with
``compute_kqv=True`` + manual concat + ``utils.apply_rope`` + sdpa +
``Qwen2DecoderLayer.forward`` with ``output_atten=True`` (start/end
slicing).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from flash_rt.models.lingbot.kernel_ops import (
    ada_rms_fp8_fused,
    ada_rms_residual_fp8_fused,
    attention_fmha_strided_fused,
    attention_mha_bf16_fused,
    linear_bf16, linear_fp8, linear_fp8_from_fp8,
    qkv_bias_rope_fused,
    qkv_bias_rope_merged_fused,
    silu_mul_to_fp8_fp16_fused,
    silu_mul_fp8_mpad_bf16_fused,
    silu_mul_merged_fp8_mpad_bf16_fused,
    silu_mul_merged_fp8_mpad_fp16in_fused,
    attention_fa4_fused,
    _ensure_fmha_strided,
)
from flash_rt.models.lingbot.kernel_ops import _FP8_MIN_M, _FP8_PAD_MIN_M

# fusion-budget probe (denoise FFN megakernel): skip the silu_mul kernel (gu-read +
# silu-compute + h-write + launch) and feed down a dummy h_fp8 so down still runs.
# Lower bound on the megakernel's serial DRAM-elimination budget. NOT for production.
_PROBE_SKIP_SILU = False
_PROBE_DUMMY_H = {}

# FA4 (FlashAttention-4 CuTe-DSL) denoise attention. Measured ~17% faster than
# the vendored fmha at the denoise shape (Sq=51) via pack_gqa, cos=1.0,
# CUDA-graph safe. Runs on Thor compiled for sm_101a. Opt-in; falls back to
# fmha if FA4 (cutlass-dsl/quack) isn't available.
_USE_FA4_ATTN = True

# prefer the GQA-native fp16 CUTLASS FMHA on the unmasked fused path.
# Set False to fall back to the cuBLAS bf16 attention_mha path.
_USE_FMHA_STRIDED = True

# fp16 attention island. The QKV megakernel emits fp16 q/k/v (and the
# KV cache is fp16), so the fmha wrapper consumes them with NO bf16→fp16 cast
# (eliminates ~1080 cast launches/inference @10-step). Output returns to bf16
# at the o_proj boundary. Set False to keep the bf16 attention path.
_USE_FP16_ATTN_ISLAND = True
ATTN_ISLAND_DTYPE = torch.float16

from flash_rt.models.lingbot.norms import rms_norm, ada_rms_norm
from flash_rt.models.lingbot.rope_adapter import (
    LINGBOT_ROPE_CONFIG,
    apply_rope_with_tables,
    build_cos_sin_table,
)


def _eager_attention(
    Q: torch.Tensor,     # [B, L, num_q_heads, head_dim]
    K: torch.Tensor,     # [B, L, num_kv_heads, head_dim]
    V: torch.Tensor,     # [B, L, num_kv_heads, head_dim]
    attention_mask: torch.Tensor | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    use_fused: bool = True,
) -> torch.Tensor:
    """Per-layer multi-head attention.

    fast path (``use_fused=True``, mask=None or near-trivial): one
    ``fvk.attention_mha_bf16`` call replaces the einsum chain. Reduces
    36 layers × 4 launches (Q*scale, QK^T, softmax, P@V) to 36 × 1.

    Eager fallback: bit-exact replica of ``lingbotvla.models.vla.pi0.
    utils.our_eager_attention_forward`` — explicit einsum chain in the
    input dtype (bf16), softmax in-dtype, bool mask via
    ``torch.where(big_neg)``. Used when the caller passes a non-trivial
    bool mask.

    Args:
        Q/K/V: ``[B, L, *_heads, head_dim]``.
        attention_mask: bool mask ``[B, L_q, L_kv]`` (True = keep). When
            ``None`` and ``use_fused`` is True, takes the fused path.
        num_q_heads, num_kv_heads, head_dim: shape constants.
        use_fused: opt-in (default) into the fused kernel. Set False
            from tests that need bit-exact eager .

    Returns:
        ``[B, L, num_q_heads * head_dim]`` attention output.
    """
    B, L_q, _, D = Q.shape
    L_kv = K.shape[1]
    n_groups = num_q_heads // num_kv_heads

    # Fused unmasked path. LingBot's prefix attention is non-causal full
    # (``vlm_causal=False`` upstream); denoise suffix attention drops the
    # Pi0-style state-token mask (cos delta ≤1e-4/layer — softmax is
    # dominated by the prefix keys, not the 1 state key).
    if use_fused and attention_mask is None:
        # FA4: ~17% faster than fmha at the denoise shape (pack_gqa). Returns
        # None if unavailable → fall through to fmha.
        if _USE_FA4_ATTN:
            _o = attention_fa4_fused(
                Q, K, V, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, head_dim=head_dim)
            if _o is not None:
                return _o
        # GQA-native fp16 CUTLASS FMHA — skips the 8× KV head-expand
        # and runs ~1.3-1.9× faster than the cuBLAS path (see kernel_ops).
        if _USE_FMHA_STRIDED and _ensure_fmha_strided():
            return attention_fmha_strided_fused(
                Q, K, V, num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads, head_dim=head_dim)
        # Fallback: GQA-expand + cuBLAS bf16 attention_mha.
        Ke = K.unsqueeze(3).expand(B, L_kv, num_kv_heads, n_groups, D).reshape(
            B, L_kv, num_kv_heads * n_groups, D)
        Ve = V.unsqueeze(3).expand(B, L_kv, num_kv_heads, n_groups, D).reshape(
            B, L_kv, num_kv_heads * n_groups, D)
        return attention_mha_bf16_fused(
            Q, Ke, Ve, attn_scale=head_dim ** -0.5)

    # Masked / non-fused path needs KV expanded to NHQ heads.
    # "b l h d -> b l (h g) d" is index-identical to repeat_interleave(g, dim=2).
    K = K.unsqueeze(3).expand(B, L_kv, num_kv_heads, n_groups, D).reshape(
        B, L_kv, num_kv_heads * n_groups, D)
    V = V.unsqueeze(3).expand(B, L_kv, num_kv_heads, n_groups, D).reshape(
        B, L_kv, num_kv_heads * n_groups, D)

    # Permute to [B, H, L, D] via the same einsum upstream uses (matters
    # for bit-exactness of contiguous-mem ordering).
    Q_p = torch.einsum("blhd->bhld", Q)
    K_p = torch.einsum("blhd->bhld", K)
    V_p = torch.einsum("blhd->bhld", V)

    # QK^T overflows in fp16 (Q~150 × K~30 × 128 ≈ 5e5, exceeds 65504).
    # Bf16 is fine because of its wider exponent. Promote to fp32 for the
    # matmul when running on fp16; bf16 path stays in-dtype.
    if Q.dtype == torch.float16:
        att_weights = torch.einsum(
            "bhqd,bhkd->bhqk", Q_p.float(), K_p.float())
    else:
        att_weights = torch.einsum("bhqd,bhkd->bhqk", Q_p, K_p)
    att_weights = att_weights * (head_dim ** -0.5)

    if attention_mask is None:
        # Synthesize an all-True bool mask — upstream always passes one.
        attention_mask = torch.ones(
            (B, L_q, L_kv), dtype=torch.bool, device=Q.device)

    # Dtype-safe negative infinity for the mask fill. fp16 max is
    # 65504 so the bf16-tuned -2.38e38 overflows on the fp16 path.
    big_neg = float(torch.finfo(att_weights.dtype).min)
    masked = torch.where(
        attention_mask[:, None, :, :], att_weights, big_neg)
    probs = F.softmax(masked, dim=-1)
    probs = probs.to(dtype=V.dtype)

    att_out = torch.einsum("bhqk,bhkv->bhqv", probs, V_p)
    att_out = torch.einsum("bhld->blhd", att_out)
    return att_out.reshape(B, L_q, num_q_heads * head_dim)


@dataclass(frozen=True)
class AttentionDims:
    """Mixed-Head joint attention shape constants."""

    num_q_heads: int = 16
    num_kv_heads: int = 2          # GQA n_rep = 8
    head_dim: int = 128            # Q_out = 16*128 = 2048
    vlm_hidden: int = 2048
    expert_hidden: int = 768
    rms_eps: float = 1e-6


DEFAULT_ATTN_DIMS = AttentionDims()


def compute_kqv_vlm(
    hidden_states: torch.Tensor,
    target,
    layer_idx: int,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-norm + Q/K/V projections for one VLM layer.

    Mirrors ``Qwen2DecoderLayer.forward(compute_kqv=True)`` with no ada
    conditioning. Returns Q, K, V in shape ``(B, L, num_*_heads, head_dim)``.
    """
    h = rms_norm(
        hidden_states,
        weight=target.vlm_layer_input_layernorm_weights[layer_idx],
        eps=dims.rms_eps,
    )
    B, L, _ = h.shape
    prefix = f"vlm.layer.{layer_idx}"
    q = linear_fp8(h,
                 target.vlm_layer_q_proj_weights[layer_idx],
                 target.vlm_layer_q_proj_biases[layer_idx],
                 site_id=f"{prefix}.q_proj").view(
        B, L, dims.num_q_heads, dims.head_dim)
    k = linear_fp8(h,
                 target.vlm_layer_k_proj_weights[layer_idx],
                 target.vlm_layer_k_proj_biases[layer_idx],
                 site_id=f"{prefix}.k_proj").view(
        B, L, dims.num_kv_heads, dims.head_dim)
    v = linear_fp8(h,
                 target.vlm_layer_v_proj_weights[layer_idx],
                 target.vlm_layer_v_proj_biases[layer_idx],
                 site_id=f"{prefix}.v_proj").view(
        B, L, dims.num_kv_heads, dims.head_dim)
    return q, k, v


def compute_kqv_expert(
    hidden_states: torch.Tensor,
    ada_cond: torch.Tensor,
    target,
    layer_idx: int,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    gamma_in: torch.Tensor | None = None,    # precomputed FiLM γ
    beta_in: torch.Tensor | None = None,     # precomputed FiLM β
    qk_rope: "tuple[torch.Tensor, torch.Tensor] | None" = None,  # (cos,sin)
    k_out_buf: torch.Tensor | None = None,   # write k/v here (KV cache suffix)
    v_out_buf: torch.Tensor | None = None,   # [M, NHKV*HD] views — no copy after
    residual_add: torch.Tensor | None = None,  # prev-layer mlp_out (2nd resid)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-AdaRMSNorm + Q/K/V projections for one Action Expert layer.

    Mirrors ``Qwen2DecoderLayer.forward(compute_kqv=True, ada_cond=...)``.
    The expert Q projection is Mixed-Head asymmetric: in=768, out=2048
    (matches VLM head space so the towers can share attention).
    """
    # try the fused AdaRMS+FP8 path. When static scale exists
    # for the Q-proj site, pre-compute γ/β then call our custom kernel
    # that produces FP8 input directly for the Q/K/V GEMMs. Saves the
    # 6-9 launches that eager ada_rms_norm + linear_fp8 internal quantize
    # would emit per layer.
    # γ_in/β_in may be precomputed (batched across all denoise steps);
    # only fall back to the per-site linear_fp8 when not provided.
    prefix = f"expert.layer.{layer_idx}"
    q_site = f"{prefix}.q_proj"
    if gamma_in is None:
        gamma_in = linear_fp8(
            ada_cond,
            target.expert_layer_input_layernorm_gamma_weights[layer_idx],
            target.expert_layer_input_layernorm_gamma_biases[layer_idx],
            site_id=f"{prefix}.input_ln.gamma",
        )
    if beta_in is None:
        beta_in = linear_fp8(
            ada_cond,
            target.expert_layer_input_layernorm_beta_weights[layer_idx],
            target.expert_layer_input_layernorm_beta_biases[layer_idx],
            site_id=f"{prefix}.input_ln.beta",
        )
    B, L, _ = hidden_states.shape
    NHQ, NHKV, HD = dims.num_q_heads, dims.num_kv_heads, dims.head_dim
    # M-pad the norm's FP8 output to 64 ONCE (the same buffer feeds q/k/v),
    # so the three q/k/v GEMMs read M=64 directly and skip three redundant
    # 51->64 pad copies. Only when the M=51 pad regime applies.
    M_real = B * L
    _pad = _FP8_MIN_M if _FP8_PAD_MIN_M <= M_real < _FP8_MIN_M else None
    if residual_add is not None:
        # fuse the previous layer's 2nd residual (mlp_out + afr) into this
        # input norm. ``hidden_states`` (= prev afr) is mutated in place to
        # ``hidden_states + residual_add`` (= the real layer input h), which the
        # caller then reuses as the post-attn residual — so the standalone
        # ``mlp_out + afr`` add (one launch + buffer per layer) is removed.
        fused = ada_rms_residual_fp8_fused(
            hidden_states, residual_add,
            target.expert_layer_input_layernorm_weights[layer_idx],
            gamma_in, beta_in,
            eps=dims.rms_eps, site_id=q_site, pad_to=_pad,
        )
    else:
        fused = ada_rms_fp8_fused(
            hidden_states,
            target.expert_layer_input_layernorm_weights[layer_idx],
            gamma_in, beta_in,
            eps=dims.rms_eps, site_id=q_site, pad_to=_pad,
        )
    if fused is not None:
        h_fp8, act_scale = fused
        qkv_merged = getattr(target, "expert_layer_qkv_merged_weights", None)
        if qk_rope is not None:
            cos, sin = qk_rope
            half = HD // 2
            M = B * L
            out_dtype = ATTN_ISLAND_DTYPE if _USE_FP16_ATTN_ISLAND \
                else torch.bfloat16
            if qkv_merged is not None:
                # ONE merged q/k/v GEMM ([H -> NHQ*HD+2*NHKV*HD]) instead of
                # three; the merged-input rope kernel reads q/k/v from column
                # offsets (no split). h_fp8 may be M-padded -> slice to M_real.
                qkv_raw = linear_fp8_from_fp8(
                    h_fp8, act_scale, qkv_merged[layer_idx], None,
                    site_id=q_site)
                if _pad is not None:
                    qkv_raw = qkv_raw[:M_real]
                q, k, v = qkv_bias_rope_merged_fused(
                    qkv_raw,
                    target.expert_layer_q_proj_biases[layer_idx],
                    target.expert_layer_k_proj_biases[layer_idx],
                    target.expert_layer_v_proj_biases[layer_idx],
                    cos.reshape(M, half), sin.reshape(M, half),
                    num_q_heads=NHQ, num_kv_heads=NHKV, head_dim=HD,
                    out_dtype=out_dtype,
                    k_out=k_out_buf, v_out=v_out_buf)  # k/v→cache suffix
                return (q.view(B, L, NHQ, HD), k.view(B, L, NHKV, HD),
                        v.view(B, L, NHKV, HD))
            # bias-free q/k/v GEMMs, then ONE fused kernel does
            # bias-add + RoPE (q/k) + bias (v) — replaces 3 add_bias + 2 rope.
            # When h_fp8 is M-padded ([64,K]), the GEMMs return [64,N]; slice
            # the raw outputs back to M_real (free view) before the RoPE kernel.
            q_raw = linear_fp8_from_fp8(
                h_fp8, act_scale, target.expert_layer_q_proj_weights[layer_idx],
                None, site_id=q_site)
            k_raw = linear_fp8_from_fp8(
                h_fp8, act_scale, target.expert_layer_k_proj_weights[layer_idx],
                None, site_id=f"{prefix}.k_proj")
            v_raw = linear_fp8_from_fp8(
                h_fp8, act_scale, target.expert_layer_v_proj_weights[layer_idx],
                None, site_id=f"{prefix}.v_proj")
            if _pad is not None:
                q_raw = q_raw[:M_real]; k_raw = k_raw[:M_real]; v_raw = v_raw[:M_real]
            q, k, v = qkv_bias_rope_fused(
                q_raw, k_raw, v_raw,
                target.expert_layer_q_proj_biases[layer_idx],
                target.expert_layer_k_proj_biases[layer_idx],
                target.expert_layer_v_proj_biases[layer_idx],
                cos.reshape(M, half), sin.reshape(M, half),
                num_q_heads=NHQ, num_kv_heads=NHKV, head_dim=HD,
                out_dtype=out_dtype)
            return (q.view(B, L, NHQ, HD), k.view(B, L, NHKV, HD),
                    v.view(B, L, NHKV, HD))
        q = linear_fp8_from_fp8(
            h_fp8, act_scale, target.expert_layer_q_proj_weights[layer_idx],
            target.expert_layer_q_proj_biases[layer_idx],
            site_id=q_site,
        )[:M_real].view(B, L, dims.num_q_heads, dims.head_dim)
        k = linear_fp8_from_fp8(
            h_fp8, act_scale, target.expert_layer_k_proj_weights[layer_idx],
            target.expert_layer_k_proj_biases[layer_idx],
            site_id=f"{prefix}.k_proj",
        )[:M_real].view(B, L, dims.num_kv_heads, dims.head_dim)
        v = linear_fp8_from_fp8(
            h_fp8, act_scale, target.expert_layer_v_proj_weights[layer_idx],
            target.expert_layer_v_proj_biases[layer_idx],
            site_id=f"{prefix}.v_proj",
        )[:M_real].view(B, L, dims.num_kv_heads, dims.head_dim)
        return q, k, v

    # Eager fallback when no calibration is loaded.
    h = ada_rms_norm(
        hidden_states, ada_cond,
        weight=target.expert_layer_input_layernorm_weights[layer_idx],
        gamma_weight=target.expert_layer_input_layernorm_gamma_weights[layer_idx],
        gamma_bias=target.expert_layer_input_layernorm_gamma_biases[layer_idx],
        beta_weight=target.expert_layer_input_layernorm_beta_weights[layer_idx],
        beta_bias=target.expert_layer_input_layernorm_beta_biases[layer_idx],
        eps=dims.rms_eps,
        site_prefix=f"{prefix}.input_ln",
    )
    q = linear_fp8(h,
                 target.expert_layer_q_proj_weights[layer_idx],
                 target.expert_layer_q_proj_biases[layer_idx],
                 site_id=f"{prefix}.q_proj").view(
        B, L, dims.num_q_heads, dims.head_dim)
    k = linear_fp8(h,
                 target.expert_layer_k_proj_weights[layer_idx],
                 target.expert_layer_k_proj_biases[layer_idx],
                 site_id=f"{prefix}.k_proj").view(
        B, L, dims.num_kv_heads, dims.head_dim)
    v = linear_fp8(h,
                 target.expert_layer_v_proj_weights[layer_idx],
                 target.expert_layer_v_proj_biases[layer_idx],
                 site_id=f"{prefix}.v_proj").view(
        B, L, dims.num_kv_heads, dims.head_dim)
    return q, k, v


def mixed_head_attention_layer(
    *,
    vlm_hidden: torch.Tensor,
    expert_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    ada_cond: torch.Tensor,
    target,
    layer_idx: int,
    attention_mask: torch.Tensor | None = None,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer of LingBot-VLA Mixed-Head joint attention (no FFN/residual).

    Args:
        vlm_hidden:      ``[B, L_vlm, 2048]`` pre-norm hidden state.
        expert_hidden:   ``[B, L_exp, 768]`` pre-norm hidden state.
        position_ids:    ``[B, L_total]`` long, where L_total = L_vlm + L_exp.
        ada_cond:        ``[B, 768]`` timestep embedding for AdaRMSNorm.
        target:          object loaded by the + bound by the — must have all
                         weight attributes from the lingbot WEIGHT_SPEC.
        layer_idx:       which of the 36 layers to run.
        attention_mask:  optional ``[B, 1, L_total, L_total]`` mask
                         (float fill_value, additive). None = unmasked.
        dims:            ``AttentionDims`` overrides.

    Returns:
        ``(out_vlm, out_expert)`` —
          ``out_vlm:    [B, L_vlm, 2048]``
          ``out_expert: [B, L_exp,  768]``    (Mixed-Head asymmetric o_proj)
    """
    # 1+2. Pre-norm + Q/K/V projections per tower.
    q_v, k_v, v_v = compute_kqv_vlm(vlm_hidden, target, layer_idx, dims)
    q_e, k_e, v_e = compute_kqv_expert(expert_hidden, ada_cond,
                                       target, layer_idx, dims)

    B, L_vlm = vlm_hidden.shape[:2]
    _, L_exp = expert_hidden.shape[:2]

    # 3. Concat on token axis.
    Q = torch.cat([q_v, q_e], dim=1)
    K = torch.cat([k_v, k_e], dim=1)
    V = torch.cat([v_v, v_e], dim=1)

    # 4. 1-D RoPE .
    cos, sin = build_cos_sin_table(
        position_ids, LINGBOT_ROPE_CONFIG,
        compute_dtype=torch.float32,
    )
    Q = apply_rope_with_tables(Q.to(torch.float32), cos, sin).to(Q.dtype)
    K = apply_rope_with_tables(K.to(torch.float32), cos, sin).to(K.dtype)

    # 5+6. Eager attention (matches upstream our_eager_attention_forward
    # bit-exact: explicit einsum + softmax in input-dtype + bool mask).
    A = _eager_attention(
        Q, K, V, attention_mask,
        num_q_heads=dims.num_q_heads,
        num_kv_heads=dims.num_kv_heads,
        head_dim=dims.head_dim,
    )
    assert A.shape == (B, L_vlm + L_exp, dims.num_q_heads * dims.head_dim)

    # 7. Per-tower o_proj.
    a_v, a_e = A.split([L_vlm, L_exp], dim=1)
    out_v = linear_fp8(a_v.contiguous(),
                     target.vlm_layer_o_proj_weights[layer_idx],
                     site_id=f"vlm.layer.{layer_idx}.o_proj")
    out_e = linear_fp8(a_e.contiguous(),
                     target.expert_layer_o_proj_weights[layer_idx],
                     site_id=f"expert.layer.{layer_idx}.o_proj")
    return out_v, out_e


def swiglu_mlp(
    x: torch.Tensor,
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    site_prefix: "str | None" = None,
) -> torch.Tensor:
    """Qwen2MLP: ``down(silu(gate(x)) * up(x))``.

    All linears have ``bias=False`` in Qwen2.5-VL & Qwen2-768.

    ``site_prefix``: when given, the three internal linears are
    tagged ``f"{site_prefix}.gate_proj"`` / ``up_proj`` / ``down_proj``
    so they can be calibrated and routed through ``quantize_fp8_static``.
    """
    if site_prefix is None:
        sid_g = sid_u = sid_d = None
    else:
        sid_g = f"{site_prefix}.gate_proj"
        sid_u = f"{site_prefix}.up_proj"
        sid_d = f"{site_prefix}.down_proj"
    gate = linear_fp8(x, gate_weight, site_id=sid_g)
    up = linear_fp8(x, up_weight, site_id=sid_u)

    # (fp16-only): fuse silu+mul+fp8-quant when gate/up are fp16
    # AND the down_proj has a static scale. Used by Expert hot loop.
    if gate.dtype == torch.float16 and sid_d is not None:
        from flash_rt.models.lingbot import calibration as _calib
        down_scale = _calib.get_static_scale(sid_d)
        if down_scale is not None:
            h_fp8 = silu_mul_to_fp8_fp16_fused(gate, up, down_scale)
            # linear_fp8_from_fp8 expects [M, K] 2D shape; the fused
            # output keeps the input shape so reshape and restore.
            orig_shape = gate.shape
            H = orig_shape[-1]
            M = 1
            for d in orig_shape[:-1]:
                M *= d
            out_2d = linear_fp8_from_fp8(
                h_fp8.view(M, H), down_scale, down_weight, site_id=sid_d)
            return out_2d.view(*orig_shape[:-1], down_weight.shape[0])
    return linear_fp8(F.silu(gate) * up, down_weight, site_id=sid_d)


_DENOISE_FP4_SCRATCH: dict = {}


def _denoise_fp4_scratch(M: int, K: int, twoI: int, device):
    """Lazy (eager-warmup) FP4 scratch for the denoise gate_up probe, reused
    across all denoise layers/steps: act FP4 [M,K] + gate_up fp16 out [M,2I]."""
    from flash_rt.models.lingbot import fp4_ops
    key = (M, K, twoI)
    s = _DENOISE_FP4_SCRATCH.get(key)
    if s is None:
        s = {"a": fp4_ops.FP4ActScratch(M, K),
             "gu": torch.empty(M, twoI, dtype=torch.float16, device=device)}
        _DENOISE_FP4_SCRATCH[key] = s
    return s


def swiglu_mlp_from_fp8(
    x_fp8: torch.Tensor,                # [M, K] fp8 — pre-quantized (M may be padded)
    act_scale: torch.Tensor,            # [1] fp32 — scale used to quant x_fp8
    *,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    site_prefix: str,
    m_real: int | None = None,          # real token count when x_fp8 is M-padded
    gate_up_merged_weight: torch.Tensor | None = None,  # cat([gate,up])
    gate_up_fp4: dict | None = None,    # FP4 merged gate_up weight
    x_fp16: torch.Tensor | None = None,  # fp16 norm output (skip fp8 dequant)
) -> torch.Tensor:
    """FP8-input variant of :func:`swiglu_mlp`.

    The input has already been quantized to FP8 by a fused
    ``residual_add_rms_norm_fp8`` upstream; this function reuses that
    quantization for the ``gate_proj`` and ``up_proj`` GEMMs (no
    re-quantize). The ``down_proj`` input is the bf16 product
    ``silu(gate) * up`` and goes through the normal
    :func:`linear_fp8` (dynamic or static, whichever is active).

    ``m_real``: when ``x_fp8`` is M-padded upstream (rows ``[m_real, M)`` are
    zero), the gate/up GEMMs run at M directly (no pad copy); the down output
    is sliced back to ``m_real``. Pad rows of gate/up are zero (zero input, no
    bias) so ``silu(0)*0 = 0`` — processing them is harmless.
    """
    # merged gate_up fast path. ONE GEMM ([H→2*I]) + a merged-input
    # silu_mul (reads gate/up from row offsets, no split) replaces two GEMMs +
    # the silu/mul/quant glue. Gated on a merged weight + static down scale +
    # the M=51 pad regime.
    if gate_up_merged_weight is not None:
        from flash_rt.models.lingbot import calibration as _calib
        down_sid = f"{site_prefix}.down_proj"
        down_scale = _calib.get_static_scale(down_sid)
        M_in = (x_fp16 if x_fp16 is not None else x_fp8).shape[0]
        mr_g = M_in if m_real is None else m_real
        if down_scale is not None and _FP8_PAD_MIN_M <= mr_g < _FP8_MIN_M:
            if gate_up_fp4 is not None:
                # FP4 merged gate_up. Dequant the fp8 norm output to fp16,
                # FP4-quant, FP4 GEMM → fp16, cast bf16 for the existing silu_mul.
                # Net win at M=64 (FP4 weight = half the DRAM bytes); cos held.
                from flash_rt.models.lingbot import fp4_ops
                twoI = gate_up_fp4["N"]
                if x_fp16 is not None:
                    # norm already produced fp16 — FP4-quant directly (no
                    # fp8->fp16 dequant, which was ~18us/layer of tiny kernels).
                    K_in = x_fp16.shape[1]
                    sc = _denoise_fp4_scratch(M_in, K_in, twoI, x_fp16.device)
                    fp4_ops.quant_act(x_fp16, sc["a"], M_in)
                else:
                    K_in = x_fp8.shape[1]
                    sc = _denoise_fp4_scratch(M_in, K_in, twoI, x_fp8.device)
                    x16 = (x_fp8.float() * act_scale).to(torch.float16).contiguous()
                    fp4_ops.quant_act(x16, sc["a"], M_in)
                fp4_ops.fp4_gemm(sc["a"], gate_up_fp4, sc["gu"], M_in, twoI, K_in)
                # feed the FP4 GEMM fp16 output to the fp16-input silu_mul
                # directly (no bf16 cast — that cast was ~360 calls/inference).
                if _PROBE_SKIP_SILU:
                    # fusion-budget probe: skip silu (gu-read+silu+h-write+launch);
                    # feed down a persistent dummy h_fp8 so down still runs.
                    _I = twoI // 2; _key = (_FP8_MIN_M, _I)
                    h_fp8 = _PROBE_DUMMY_H.get(_key)
                    if h_fp8 is None:
                        h_fp8 = torch.zeros(_FP8_MIN_M, _I, dtype=torch.float8_e4m3fn, device=sc["gu"].device)
                        _PROBE_DUMMY_H[_key] = h_fp8
                else:
                    h_fp8 = silu_mul_merged_fp8_mpad_fp16in_fused(
                        sc["gu"], down_scale, pad_to=_FP8_MIN_M)
            else:
                gu = linear_fp8_from_fp8(
                    x_fp8, act_scale, gate_up_merged_weight,
                    site_id=f"{site_prefix}.gate_up")            # [M_in, 2*I]
                h_fp8 = silu_mul_merged_fp8_mpad_bf16_fused(
                    gu, down_scale, pad_to=_FP8_MIN_M)
            out_pad = linear_fp8_from_fp8(
                h_fp8, down_scale, down_weight, site_id=down_sid)
            return out_pad[:mr_g].reshape(mr_g, down_weight.shape[0])

    gate = linear_fp8_from_fp8(
        x_fp8, act_scale, gate_weight,
        site_id=f"{site_prefix}.gate_proj")
    up = linear_fp8_from_fp8(
        x_fp8, act_scale, up_weight,
        site_id=f"{site_prefix}.up_proj")

    # (fp16-only): when gate/up are fp16 AND the down_proj has a
    # static scale, fuse silu+mul+fp8-quant into one launch via
    # ``silu_mul_split_fp8_fp16``. Saves 3 launches per MLP × 36 Expert
    # layers × N denoise steps + 36 VLM (prefix-once).
    if gate.dtype == torch.float16:
        from flash_rt.models.lingbot import calibration as _calib
        down_sid = f"{site_prefix}.down_proj"
        down_scale = _calib.get_static_scale(down_sid)
        if down_scale is not None:
            h_fp8 = silu_mul_to_fp8_fp16_fused(gate, up, down_scale)
            orig_shape = gate.shape
            H = orig_shape[-1]
            M = 1
            for d in orig_shape[:-1]:
                M *= d
            out_2d = linear_fp8_from_fp8(
                h_fp8.view(M, H), down_scale, down_weight, site_id=down_sid)
            return out_2d.view(*orig_shape[:-1], down_weight.shape[0])

    # bf16 path: fuse silu(gate)*up + FP8 quant + M-pad into ONE kernel that
    # writes a pre-padded [pad_to, I] FP8 buffer the down GEMM reads directly
    # (skips the eager silu/mul/quant + linear_fp8's pad copy). Gated on a
    # static down_proj scale (the FP8 quant needs it) and the M-pad regime.
    from flash_rt.models.lingbot import calibration as _calib
    down_sid = f"{site_prefix}.down_proj"
    down_scale = _calib.get_static_scale(down_sid)
    orig_shape = gate.shape
    H = orig_shape[-1]
    M = 1
    for d in orig_shape[:-1]:
        M *= d
    mr = M if m_real is None else m_real
    # Fuse when down has a static scale and the real token count is in the
    # M=51 pad regime. gate/up may already be M-padded ([64,I]); their pad
    # rows are zero so silu_mul over all M rows is exact. down reads M=pad_to
    # directly; slice the output to the real token count.
    if down_scale is not None and _FP8_PAD_MIN_M <= mr < _FP8_MIN_M:
        h_fp8 = silu_mul_fp8_mpad_bf16_fused(
            gate.view(M, H), up.view(M, H), down_scale, pad_to=_FP8_MIN_M)
        out_pad = linear_fp8_from_fp8(
            h_fp8, down_scale, down_weight, site_id=down_sid)  # [pad_to, N]
        return out_pad[:mr].view(*orig_shape[:-1], down_weight.shape[0]) \
            if M == mr else out_pad[:mr].reshape(mr, down_weight.shape[0])
    return linear_fp8(
        F.silu(gate) * up,
        down_weight, site_id=f"{site_prefix}.down_proj",
    )


def mixed_head_layer_full(
    *,
    vlm_hidden: torch.Tensor,
    expert_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    ada_cond: torch.Tensor,
    target,
    layer_idx: int,
    attention_mask: torch.Tensor | None = None,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One **full** transformer layer of LingBot-VLA Mixed-Head joint forward.

    Replicates the layer-loop body of ``QwenvlWithExpertModel.forward``:

        1. Joint Mixed-Head attention      
        2. Residual:     out += hidden    (first residual)
        3. Post-LN:      RMSNorm (VLM)   or  AdaRMSNorm(out, ada_cond) (Expert)
        4. SwiGLU MLP:   down(silu(gate(h)) * up(h))
        5. Residual:     out += after_first_residual

    Args:
        vlm_hidden / expert_hidden: pre-norm hidden states.
        position_ids: joint sequence positions.
        ada_cond:     timestep embedding for AdaRMSNorm.
        target:       device-bound weight namespace.
        layer_idx:    which of 36 layers.
        attention_mask: optional bool ``[B, L_q, L_kv]``.
        dims:         AttentionDims override.

    Returns:
        ``(out_vlm, out_expert)`` post-layer hiddens (after both residuals).
        Same shapes as inputs.
    """
    # 1. Attention.
    attn_v, attn_e = mixed_head_attention_layer(
        vlm_hidden=vlm_hidden, expert_hidden=expert_hidden,
        position_ids=position_ids, ada_cond=ada_cond,
        target=target, layer_idx=layer_idx,
        attention_mask=attention_mask, dims=dims,
    )

    # 2. First residual.
    afr_v = attn_v + vlm_hidden
    afr_e = attn_e + expert_hidden

    # 3. Post-LN (VLM: RMS, Expert: AdaRMS with ada_cond).
    h_v = rms_norm(afr_v,
                   weight=target.vlm_layer_post_attn_layernorm_weights[layer_idx],
                   eps=dims.rms_eps)
    h_e = ada_rms_norm(
        afr_e, ada_cond,
        weight=target.expert_layer_post_attn_layernorm_weights[layer_idx],
        gamma_weight=target.expert_layer_post_attn_layernorm_gamma_weights[layer_idx],
        gamma_bias=target.expert_layer_post_attn_layernorm_gamma_biases[layer_idx],
        beta_weight=target.expert_layer_post_attn_layernorm_beta_weights[layer_idx],
        beta_bias=target.expert_layer_post_attn_layernorm_beta_biases[layer_idx],
        eps=dims.rms_eps,
        site_prefix=f"expert.layer.{layer_idx}.post_attn_ln",
    )

    # 4. SwiGLU MLP (no bias).
    mlp_v = swiglu_mlp(h_v,
        gate_weight=target.vlm_layer_mlp_gate_proj_weights[layer_idx],
        up_weight=target.vlm_layer_mlp_up_proj_weights[layer_idx],
        down_weight=target.vlm_layer_mlp_down_proj_weights[layer_idx],
        site_prefix=f"vlm.layer.{layer_idx}.mlp")
    mlp_e = swiglu_mlp(h_e,
        gate_weight=target.expert_layer_mlp_gate_proj_weights[layer_idx],
        up_weight=target.expert_layer_mlp_up_proj_weights[layer_idx],
        down_weight=target.expert_layer_mlp_down_proj_weights[layer_idx],
        site_prefix=f"expert.layer.{layer_idx}.mlp")

    # 5. Second residual.
    out_v = mlp_v + afr_v
    out_e = mlp_e + afr_e

    return out_v, out_e


__all__ = [
    "AttentionDims",
    "DEFAULT_ATTN_DIMS",
    "compute_kqv_vlm",
    "compute_kqv_expert",
    "mixed_head_attention_layer",
    "mixed_head_layer_full",
    "swiglu_mlp",
    "swiglu_mlp_from_fp8",
]
