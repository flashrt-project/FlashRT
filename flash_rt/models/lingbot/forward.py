"""LingBot-VLA layer-loop forward paths — prefix encode + denoise step.

Two asymmetric forward paths from upstream
``QwenvlWithExpertModel.forward``:

1. **Prefix encode** (set_prompt, one-time per task):
       inputs_embeds = [prefix_embs, None]    # only VLM is computed
       fill_kv_cache = True                   # K/V STORED to past_key_values
   Iterates 36 layers. Output = encoded VLM hidden + per-layer KV cache.

2. **Denoise step** (called 50 times per inference):
       inputs_embeds = [None, suffix_embs]    # only Expert is computed
       fill_kv_cache = False                  # K/V APPENDED to cached prefix
   Iterates 36 layers. Output = post-decoder Expert hidden (used by
   velocity head).

Both reuse the per-layer building blocks from ``mixed_attention``
(``compute_kqv_vlm``, ``compute_kqv_expert``, ``_eager_attention``,
``swiglu_mlp``) and ``norms.rms_norm`` / ``ada_rms_norm``.

Reference: upstream ``QwenvlWithExpertModel.forward`` (modeling_lingbot
_vla.py L1240-1349) — the same loop body but with our device-bound target
attrs and our RoPE adapter (both bit-exact verified).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from flash_rt.models.lingbot.kernel_ops import (
    ada_rms_residual_fp8_fused, ada_rms_residual_fp16_mpad_fused,
    linear_bf16, linear_fp8, residual_rms_quant_fp8_inplace,
    rope_inplace_bf16_fused, rope_to_out_bf16_fused,
    _FP8_MIN_M, _FP8_PAD_MIN_M,
)

# use the out-of-place fused RoPE kernel (graph-safe) instead of the
# eager cast→rotate→cast chain. Set False to fall back to eager.
_USE_ROPE_KERNEL = True

# PROBE ONLY: skip the denoise KV-cache suffix copy (wrong output) to measure
# whether it's serial (reducible by baking the KV write into the rope kernel).
_PROBE_SKIP_KV_COPY = False

# route the prefix VLM attention (M=840 self-attention, HD=128) through FA4
# instead of eager softmax. The bidirectional (vlm_causal=False) mask only drops
# the contiguous lang-pad tail ([img | lang-valid | lang-pad]), so FA4 with
# seqused_k = the valid prefix length reproduces it EXACTLY (cos preserved). FA4
# fills the GPU at M=840 (~54 tiles) → ~2ms faster than eager softmax (prefix-
# once, all step counts). The pad K/V rows are also zeroed so the denoise's own
# unmasked FA4 sees V=0 from them (replaces mask_kv_cache_pad_rows).
_USE_PREFIX_FA4 = True

# merged gate_up GEMM (one [H→2*I] GEMM + offset-read silu_mul instead of
# two GEMMs). gate/up weight scales are within ~1.36× → common-scale cos-free.
_USE_MERGED_GATE_UP = True

# NVFP4 on the VLM-prefix FFN gate/up GEMMs (the biggest prefix GEMMs, FP4
# 1.6-1.9× faster than fp8 at M~840). pi05 recipe: FP4 on FFN only; QKV/attn/O
# + the down GEMM stay FP8. RMS weight is folded into the gate/up FP4 weights so
# the (weightless) fused FP4 norm applies. fp16 I/O on the FFN island.
_USE_VLM_FP4 = True
_VLM_FP4_DONE = "_vlm_fp4_done"

# NVFP4 on the denoise merged gate_up GEMM. The fp8 AdaRMS output is dequant'd
# to fp16 then FP4-quant'd (a separate quant), then the FP4 merged gate_up GEMM. Even
# with the extra quant this is a net win at M=64: the FP4 weight is HALF the DRAM bytes
# (4.2->2.1 MB) so the in-pipeline weight read drops, and it scales with denoise steps
# (-~5ms@25, -~10ms@50). cos held (0.99622). A fused AdaRMS->FP4 kernel (skip the
# dequant+quant) would add a bit more but isn't needed for the win.
_USE_DENOISE_FP4 = True

# fold the denoise 2nd residual (mlp_out + afr) into the NEXT layer's input
# AdaRMS (which already reads that tensor and must write the sum for its own
# post-attn residual). Reuses ada_rms_residual_fp8_fused; removes the standalone
# add (one launch + buffer per layer, ~1.3ms@25 — measured serial via probe).
_USE_2ND_RES_FUSION = True


def prepare_vlm_fp4_weights(target, num_layers: int = 36) -> None:
    """: fold post-attn RMS weight into VLM gate/up, FP4-quant (offline,
    idempotent, eager warmup). The folded weight lets the weightless fused FP4
    norm produce the correct (rms·w) activation. down stays FP8."""
    if getattr(target, _VLM_FP4_DONE, False):
        return
    from flash_rt.models.lingbot import fp4_ops
    if not fp4_ops.HAS_FP4:
        setattr(target, _VLM_FP4_DONE, True)
        return
    gate = target.vlm_layer_mlp_gate_proj_weights
    up = target.vlm_layer_mlp_up_proj_weights
    rw = target.vlm_layer_post_attn_layernorm_weights
    down = target.vlm_layer_mlp_down_proj_weights
    gq, uq, dq = [], [], []
    for i in range(num_layers):
        rwi = rw[i].float()[None, :]
        gq.append(fp4_ops.quant_weight((gate[i].float() * rwi).half().contiguous()))
        uq.append(fp4_ops.quant_weight((up[i].float() * rwi).half().contiguous()))
        dq.append(fp4_ops.quant_weight(down[i].half().contiguous()))
    target.vlm_mlp_gate_fp4 = gq
    target.vlm_mlp_up_fp4 = uq
    target.vlm_mlp_down_fp4 = dq
    setattr(target, _VLM_FP4_DONE, True)


def _vlm_fp4_scratch(target, M: int, D: int, H: int):
    """Lazily allocate (in eager warmup) the per-layer FP4 FFN scratch reused
    across all 36 VLM layers: norm FP4 buffer [M,D] + gate/up fp16 [M,H]."""
    s = getattr(target, "_vlm_fp4_scratch_buf", None)
    if s is not None and s["M"] >= M:
        return s
    from flash_rt.models.lingbot import fp4_ops
    s = {"norm": fp4_ops.FP4ActScratch(M, D),
         "gate": torch.empty(M, H, dtype=torch.float16, device="cuda"),
         "up": torch.empty(M, H, dtype=torch.float16, device="cuda"),
         "hid": fp4_ops.FP4ActScratch(M, H),   # silu(gate)*up FP4 (down input)
         "down": torch.empty(M, D, dtype=torch.float16, device="cuda"),
         "M": M}
    target._vlm_fp4_scratch_buf = s
    return s


def _has_static_scales() -> bool:
    from flash_rt.models.lingbot import calibration as _calib
    return _calib.has_static_scales()


_EXPERT_MERGED_DONE = "_expert_gate_up_merged"


def prepare_expert_merged_weights(target, num_layers: int = 36) -> None:
    """: build per-layer merged gate_up weights cat([gate, up], dim=0)
    → [2*I, H] bf16 so the SwiGLU runs ONE GEMM ([H→2*I]) instead of two.
    The merged FP8 weight uses a common per-tensor scale (gate/up absmax are
    within ~1.36× across layers → negligible cos cost). Idempotent; the cat
    runs once in eager warmup, before CUDA-graph capture."""
    if getattr(target, _EXPERT_MERGED_DONE, False):
        return
    gate = target.expert_layer_mlp_gate_proj_weights
    up = target.expert_layer_mlp_up_proj_weights
    target.expert_layer_mlp_gate_up_merged_weights = [
        torch.cat([gate[i], up[i]], dim=0).contiguous() for i in range(num_layers)]
    # merged q/k/v weight cat([q, k, v], dim=0) → [NHQ*HD + 2*NHKV*HD, H].
    # The 3 q/k/v GEMMs are serial (NOT overlapped) so one merged GEMM cuts 2
    # launches. Common per-tensor scale (q/k/v absmax within ~3× → cos ~0.995).
    q = target.expert_layer_q_proj_weights
    k = target.expert_layer_k_proj_weights
    v = target.expert_layer_v_proj_weights
    target.expert_layer_qkv_merged_weights = [
        torch.cat([q[i], k[i], v[i]], dim=0).contiguous() for i in range(num_layers)]
    # FP4-quant the merged gate_up weight (the post-attn AdaRMS output is
    # already fully normed+FiLM'd, so no rms-fold needed).
    if _USE_DENOISE_FP4:
        from flash_rt.models.lingbot import fp4_ops
        if fp4_ops.HAS_FP4:
            target.expert_mlp_gate_up_fp4 = [
                fp4_ops.quant_weight(w.half().contiguous())
                for w in target.expert_layer_mlp_gate_up_merged_weights]
    setattr(target, _EXPERT_MERGED_DONE, True)


def _kv_cache_dtype() -> torch.dtype:
    """: fp16 KV cache when the fp16 attention island is active (the QKV
    megakernel emits fp16 q/k/v, so cache+new must be fp16 for the cat and
    for the fmha to consume them cast-free). Else bf16. Gated on calibration
    (the megakernel only runs on the fused FP8 path)."""
    from flash_rt.models.lingbot import mixed_attention as _m
    if (_m._USE_FP16_ATTN_ISLAND and _USE_ROPE_KERNEL
            and _has_static_scales()):
        return _m.ATTN_ISLAND_DTYPE
    return torch.bfloat16


def _rope(x, cos, sin):
    """RoPE on [B,S,NH,HD] bf16 with fp32 cos/sin. Fused kernel when enabled,
    else the eager fp32-cast reference (bit-equivalent split-half math)."""
    if _USE_ROPE_KERNEL and x.dtype == torch.bfloat16:
        try:
            return rope_to_out_bf16_fused(x, cos, sin)
        except Exception:
            pass
    return apply_rope_with_tables(x.to(torch.float32), cos, sin).to(x.dtype)

from flash_rt.models.lingbot.mixed_attention import (
    AttentionDims, DEFAULT_ATTN_DIMS,
    _eager_attention,
    compute_kqv_expert,
    compute_kqv_vlm,
    swiglu_mlp,
    swiglu_mlp_from_fp8,
)
from flash_rt.models.lingbot.kernel_ops import attention_fa4_fused
from flash_rt.models.lingbot.norms import ada_rms_norm, rms_norm
from flash_rt.models.lingbot.rope_adapter import (
    LINGBOT_ROPE_CONFIG, apply_rope_with_tables, build_cos_sin_table,
)


# ════════════════════════════════════════════════════════════════════
#  PREFIX ENCODE PATH (VLM-only, fill_kv_cache=True)
# ════════════════════════════════════════════════════════════════════

def prefix_encode_layer(
    vlm_hidden: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target,
    layer_idx: int,
    kv_cache_out: dict,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    rope_cos: torch.Tensor | None = None,    # hoisted RoPE tables
    rope_sin: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,    # [B, L] bool (True = valid)
) -> torch.Tensor:
    """One layer of prefix encode — VLM-only forward, fills KV cache.

    Args:
        vlm_hidden: ``[B, L_prefix, 2048]`` pre-norm hidden.
        position_ids: ``[B, L_prefix]`` long.
        attention_mask: ``[B, L_prefix, L_prefix]`` bool (True = keep).
        target: device-bound weight namespace.
        layer_idx: which of 36 layers.
        kv_cache_out: dict to write {"key_states", "value_states"} into
                      under key ``layer_idx``. Caller initializes as
                      empty dict before the 36-layer loop.
        dims: AttentionDims.

    Returns:
        ``[B, L_prefix, 2048]`` post-layer hidden.
    """
    B, L, _ = vlm_hidden.shape
    # 1. Pre-norm + Q/K/V proj.
    q, k, v = compute_kqv_vlm(vlm_hidden, target, layer_idx, dims)

    # 2. RoPE. fused in-place would go here when (rope_cos, rope_sin)
    # are provided, but the in-place kernel introduced cross-replay
    # non-determinism in the captured graph (replay outputs drift by
    # ~0.1 between calls). Falling back to the eager fp32 cast path
    # until that's understood. The hoist still saves redundant
    # build_cos_sin_table calls.
    cos, sin = (rope_cos, rope_sin) if rope_cos is not None else \
        build_cos_sin_table(
            position_ids, LINGBOT_ROPE_CONFIG, compute_dtype=torch.float32)
    q = _rope(q, cos, sin)   # fused out-of-place RoPE
    k = _rope(k, cos, sin)

    # zero the pad K/V rows so the unmasked FA4 path is valid (pad keys
    # contribute V=0; the softmax-denominator distortion is the same one the
    # denoise already uses and calibrated). This also pre-applies what
    # mask_kv_cache_pad_rows does to the cache, so the bidirectional
    # (vlm_causal=False) attention mask becomes a no-op → mask=None → FA4.
    use_prefix_fa4 = _USE_PREFIX_FA4 and pad_mask is not None
    if use_prefix_fa4:
        keep = pad_mask.view(B, L, 1, 1).to(k.dtype)
        k = k * keep
        v = v * keep

    # 3. Fill KV cache (BEFORE attention — matches upstream handle_kv_cache).
    # store in the denoise consume-dtype (fp16 for the attention island).
    # The pad K/V rows are zeroed (above) so the denoise's unmasked FA4 sees
    # V=0 from them (replaces mask_kv_cache_pad_rows).
    cd = _kv_cache_dtype()
    kv_cache_out[layer_idx] = {
        "key_states": k.to(cd), "value_states": v.to(cd)}

    # 4. Attention (self-attention on prefix). : FA4 with seqused_k = the
    # valid prefix length skips the contiguous lang-pad keys EXACTLY (the pad is
    # at the end: [img | lang-valid | lang-pad]) — no softmax-denominator
    # distortion, so cos is preserved AND ~68 keys are skipped. Falls back to
    # eager softmax with the full bool mask if FA4 is unavailable.
    A = None
    if use_prefix_fa4:
        valid_len = pad_mask.sum(dim=1).to(torch.int32)        # [B]
        A = attention_fa4_fused(
            q, k, v, num_q_heads=dims.num_q_heads,
            num_kv_heads=dims.num_kv_heads, head_dim=dims.head_dim,
            seqused_k=valid_len)
    if A is None:
        A = _eager_attention(
            q, k, v, attention_mask,
            num_q_heads=dims.num_q_heads, num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
        )                                     # [B, L, num_q*head_dim]

    # 5. o_proj. : the FA4 prefix path returns fp16 (attention island), so
    # pin the o_proj output to the weight dtype (bf16) — matches the residual
    # stream (vlm_hidden) for the in-place residual add below (cf. denoise).
    o_w = target.vlm_layer_o_proj_weights[layer_idx]
    attn_out = linear_fp8(A, o_w, out_dtype=o_w.dtype,
                          site_id=f"vlm.layer.{layer_idx}.o_proj")

    # 6+7+8a. fused path (taken when static scales for the
    # gate_proj site are loaded):  ``residual_add_rms_norm_fp8`` writes
    # ``vlm_hidden + attn_out`` back into ``vlm_hidden`` (acting as the
    # ``afr`` for the second residual below) and emits an FP8 tensor
    # ready for the gate/up GEMMs of SwiGLU, skipping their own quant.
    site_prefix = f"vlm.layer.{layer_idx}.mlp"
    # NVFP4 FFN gate/up. afr = vlm_hidden+attn_out (bf16, in place); the
    # weightless FP4 norm on its fp16 copy + 2 FP4 GEMMs (rms-folded weights)
    # replace the fp8 gate/up; silu_mul (fp16) then the FP8 down (unchanged).
    if (_USE_VLM_FP4 and _has_static_scales()
            and getattr(target, "vlm_mlp_gate_fp4", None) is not None):
        from flash_rt.models.lingbot import fp4_ops
        B, L, D = vlm_hidden.shape
        M = B * L
        H = target.vlm_mlp_gate_fp4[layer_idx]["N"]
        vlm_hidden.add_(attn_out)                       # afr (bf16), in place
        sc = _vlm_fp4_scratch(target, M, D, H)
        x16 = vlm_hidden.reshape(M, D).to(torch.float16).contiguous()
        fp4_ops.rms_norm_to_fp4(x16, sc["norm"], M)
        fp4_ops.fp4_gemm(sc["norm"], target.vlm_mlp_gate_fp4[layer_idx], sc["gate"], M, H, D)
        fp4_ops.fp4_gemm(sc["norm"], target.vlm_mlp_up_fp4[layer_idx], sc["up"], M, H, D)
        hid = (F.silu(sc["gate"][:M]) * sc["up"][:M]).to(torch.float16).contiguous()
        # FP4 down too (down input is silu output; the fp8 path also quants,
        # so FP4 quant-for-quant + the 1.85× FP4 GEMM is a net win at M~840).
        fp4_ops.quant_act(hid, sc["hid"], M)
        fp4_ops.fp4_gemm(sc["hid"], target.vlm_mlp_down_fp4[layer_idx], sc["down"], M, D, H)
        mlp_out = sc["down"][:M].to(torch.bfloat16).reshape(B, L, D)
        return mlp_out + vlm_hidden
    fused = residual_rms_quant_fp8_inplace(
        vlm_hidden, attn_out,
        target.vlm_layer_post_attn_layernorm_weights[layer_idx],
        eps=dims.rms_eps,
        site_id=f"{site_prefix}.gate_proj",
    )
    if fused is not None:
        # NOTE: vlm_hidden has been mutated in place to (vlm_hidden + attn_out).
        out_fp8, act_scale = fused
        mlp_out = swiglu_mlp_from_fp8(
            out_fp8, act_scale,
            gate_weight=target.vlm_layer_mlp_gate_proj_weights[layer_idx],
            up_weight=target.vlm_layer_mlp_up_proj_weights[layer_idx],
            down_weight=target.vlm_layer_mlp_down_proj_weights[layer_idx],
            site_prefix=site_prefix,
        )
        afr = vlm_hidden  # bf16 (residual + attn_out), already in place
    else:
        # Fallback (no calibration loaded): the unfused eager path.
        afr = attn_out + vlm_hidden
        h_post = rms_norm(
            afr, weight=target.vlm_layer_post_attn_layernorm_weights[layer_idx],
            eps=dims.rms_eps,
        )
        mlp_out = swiglu_mlp(
            h_post,
            gate_weight=target.vlm_layer_mlp_gate_proj_weights[layer_idx],
            up_weight=target.vlm_layer_mlp_up_proj_weights[layer_idx],
            down_weight=target.vlm_layer_mlp_down_proj_weights[layer_idx],
            site_prefix=site_prefix,
        )

    # 9. 2nd residual.
    return mlp_out + afr


def prefix_encode_36L(
    prefix_hidden: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target,
    num_layers: int = 36,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    pad_mask: torch.Tensor | None = None,    # [B, L] bool (True = valid)
) -> tuple[torch.Tensor, dict]:
    """Full prefix encode: 36 layers + final RMSNorm.

    Returns:
        out:      ``[B, L_prefix, 2048]`` final hidden.
        kv_cache: dict ``{layer_idx: {"key_states": ..., "value_states": ...}}``
                  one entry per layer.
    """
    h = prefix_hidden
    kv_cache: dict = {}
    # build the FP4 VLM gate/up weights once (idempotent, eager warmup).
    if _USE_VLM_FP4 and _has_static_scales():
        prepare_vlm_fp4_weights(target, num_layers)
    # build cos/sin once per inference, reuse across all 36 layers.
    rope_cos, rope_sin = build_cos_sin_table(
        position_ids, LINGBOT_ROPE_CONFIG, compute_dtype=torch.float32)
    for layer_idx in range(num_layers):
        h = prefix_encode_layer(
            h,
            position_ids=position_ids,
            attention_mask=attention_mask,
            target=target, layer_idx=layer_idx,
            kv_cache_out=kv_cache, dims=dims,
            rope_cos=rope_cos, rope_sin=rope_sin,
            pad_mask=pad_mask,
        )
    # Final norm (final_norm_adanorm=False in our config → plain RMSNorm).
    out = rms_norm(h, weight=target.vlm_norm_weight, eps=dims.rms_eps)
    return out, kv_cache


# ════════════════════════════════════════════════════════════════════
# — batched FiLM (γ/β) precompute across all denoise steps
# ════════════════════════════════════════════════════════════════════
#
# Each Expert layer has 4 per-sample FiLM linears (input_ln.{γ,β},
# post_attn_ln.{γ,β}), all reading the timestep embedding ``ada_cond``.
# Computed per-layer-per-step they are 4·36·N tiny M=1 GEMMs (1440 @10-step)
# that the profiler shows cost ~9 ms — dominated by per-launch fixed cost,
# not bandwidth (M=1). Since the Euler timestep schedule is deterministic,
# every step's ``ada_cond`` is known up front, so we stack all 144 FiLM
# weights into one [144·H, H] matrix and run ONE batched GEMM
# ``[N, H] × [H, 144·H] → [N, 144·H]`` for the whole inference, then slice
# γ/β per (step, layer). 1440 launches → 1.  Order per layer: γ_in, β_in,
# γ_post, β_post (slots 0..3).

_FILM_STACK_CACHE: dict = {}
_FILM_SLOTS = 4   # γ_in, β_in, γ_post, β_post


def _get_film_stack(target, num_layers: int):
    """Return cached ``(W [slots·L·H, H], b [slots·L·H])`` stacking every
    Expert FiLM linear's weight/bias in (layer, slot) order. Holds a strong
    ref to source weights via the cache value ."""
    key = (id(target), num_layers)
    cached = _FILM_STACK_CACHE.get(key)
    if cached is not None:
        return cached
    ws, bs = [], []
    for l in range(num_layers):
        ws.append(target.expert_layer_input_layernorm_gamma_weights[l])
        ws.append(target.expert_layer_input_layernorm_beta_weights[l])
        ws.append(target.expert_layer_post_attn_layernorm_gamma_weights[l])
        ws.append(target.expert_layer_post_attn_layernorm_beta_weights[l])
        bs.append(target.expert_layer_input_layernorm_gamma_biases[l])
        bs.append(target.expert_layer_input_layernorm_beta_biases[l])
        bs.append(target.expert_layer_post_attn_layernorm_gamma_biases[l])
        bs.append(target.expert_layer_post_attn_layernorm_beta_biases[l])
    W = torch.cat(ws, dim=0).contiguous()   # [slots·L·H, H]
    b = torch.cat(bs, dim=0).contiguous()   # [slots·L·H]
    _FILM_STACK_CACHE[key] = (W, b)
    return W, b


def precompute_expert_film(
    ada_cond_all: torch.Tensor,   # [N_steps, H] timestep embeddings
    target,
    num_layers: int = 36,
) -> torch.Tensor:
    """One batched bf16 GEMM for every (step, layer, slot) FiLM vector.

    Returns ``[N_steps, num_layers, 4, H]`` (slots γ_in, β_in, γ_post,
    β_post). Matches the per-site ``linear_fp8`` math: those fall back to
    bf16 at M=1, so a bf16 batched GEMM is numerically equivalent."""
    W, b = _get_film_stack(target, num_layers)
    H = ada_cond_all.shape[-1]
    flat = linear_bf16(ada_cond_all, W, b)        # [N, slots·L·H]
    N = ada_cond_all.shape[0]
    return flat.view(N, num_layers, _FILM_SLOTS, H)


# ════════════════════════════════════════════════════════════════════
#  DENOISE STEP PATH (Expert-only with prefix KV cache)
# ════════════════════════════════════════════════════════════════════

def denoise_step_layer(
    expert_hidden: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    ada_cond: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target,
    layer_idx: int,
    kv_cache: dict,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    use_fused_attn: bool = True,
    rope_cos: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    layer_film: torch.Tensor | None = None,   # [4, H] precomputed γ/β
    prev_mlp_out: torch.Tensor | None = None,  # prev-layer FFN out (2nd resid)
) -> torch.Tensor:
    """One layer of denoise step — Expert-only forward, uses prefix KV cache.

    Args:
        expert_hidden: ``[B, L_suffix, 768]`` pre-norm hidden.
        position_ids:  ``[B, L_suffix]`` long. Offset by prefix length per
                       upstream ``prefix_offsets + cumsum(suffix) - 1``.
        ada_cond:      ``[B, 768]`` timestep embedding.
        attention_mask: ``[B, L_suffix, L_prefix + L_suffix]`` bool.
        target: device-bound weight namespace.
        layer_idx: which of 36 layers.
        kv_cache: dict from ``prefix_encode_36L`` — read-only here.
        dims: AttentionDims.

    Returns:
        ``[B, L_suffix, 768]`` post-layer Expert hidden.
    """
    # cos/sin  — needed by the fused QKV path below.
    cos, sin = (rope_cos, rope_sin) if rope_cos is not None else \
        build_cos_sin_table(
            position_ids, LINGBOT_ROPE_CONFIG, compute_dtype=torch.float32)

    # 1. Pre-AdaRMSNorm + Q/K/V proj. : feed precomputed input-LN γ/β
    # (slots 0,1) when available. : when the fused FP8 path is active
    # (static scales), pass cos/sin so compute_kqv_expert does q/k/v bias+RoPE
    # in ONE kernel (5 launches → 1); else RoPE here eagerly.
    g_in = layer_film[0:1] if layer_film is not None else None
    b_in = layer_film[1:2] if layer_film is not None else None
    use_fused_qkv = _USE_ROPE_KERNEL and _has_static_scales()

    # preallocated KV buffer [B, Lp+Ls, NKV, HD] (prefix copied once);
    # set it up BEFORE the QKV proj so the fused qkv_bias_rope kernel can write
    # k/v DIRECTLY into the suffix region  — no per-step copy.
    entry = kv_cache[layer_idx]
    kbuf = entry.get("key_buf")
    B_e, Ls, _ = expert_hidden.shape
    if kbuf is None:
        Lp = entry["key_states"].shape[1]
        kbuf = torch.empty(
            B_e, Lp + Ls, *entry["key_states"].shape[2:],
            dtype=entry["key_states"].dtype, device=entry["key_states"].device)
        vbuf = torch.empty_like(kbuf)
        kbuf[:, :Lp].copy_(entry["key_states"])
        vbuf[:, :Lp].copy_(entry["value_states"])
        entry["key_buf"] = kbuf
        entry["value_buf"] = vbuf
        entry["prefix_len"] = Lp
    else:
        Lp = entry["prefix_len"]
        vbuf = entry["value_buf"]

    # write k/v straight into the cache suffix (only on the fused merged
    # qkv path, which supports k_out/v_out targets). Saves the per-step copy.
    nkv_hd = dims.num_kv_heads * dims.head_dim
    bake = use_fused_qkv and not _PROBE_SKIP_KV_COPY
    k_suffix = kbuf[:, Lp:].reshape(B_e * Ls, nkv_hd) if bake else None
    v_suffix = vbuf[:, Lp:].reshape(B_e * Ls, nkv_hd) if bake else None
    # when the previous layer handed its FFN output forward (instead of
    # adding it to afr itself), fold that 2nd-residual add into this input norm.
    residual_add = (prev_mlp_out.view(B_e, Ls, -1)
                    if prev_mlp_out is not None else None)
    q, k_new, v_new = compute_kqv_expert(
        expert_hidden, ada_cond, target, layer_idx, dims,
        gamma_in=g_in, beta_in=b_in,
        qk_rope=(cos, sin) if use_fused_qkv else None,
        k_out_buf=k_suffix, v_out_buf=v_suffix,
        residual_add=residual_add)
    if not use_fused_qkv:
        q = _rope(q, cos, sin)         # fused out-of-place RoPE
        k_new = _rope(k_new, cos, sin)
    if not bake and not _PROBE_SKIP_KV_COPY:
        kbuf[:, Lp:].copy_(k_new)
        vbuf[:, Lp:].copy_(v_new)
    k_full = kbuf
    v_full = vbuf

    # 4. Attention — : when ``use_fused_attn`` is True AND caller
    # already zeroed prefix-pad rows in the KV cache, pass mask=None
    # to take the fused cuBLAS MHA path. Padding K/V rows are zero so
    # their attention contribution is V[pad]=0; softmax denominator
    # has small distortion (~26% pad fraction in our baseline → outputs
    # scaled ~0.74×, but the entire downstream path was calibrated under
    # this regime later so cos stays ≥0.999). The Pi0-style state-
    # token suffix-self mask is also dropped here — measured cos hit
    # is ~0.001 absolute (well within the ≥0.99 floor).
    if use_fused_attn:
        A = _eager_attention(
            q, k_full, v_full, None,
            num_q_heads=dims.num_q_heads, num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
        )
    else:
        A = _eager_attention(
            q, k_full, v_full, attention_mask,
            num_q_heads=dims.num_q_heads, num_kv_heads=dims.num_kv_heads,
            head_dim=dims.head_dim,
        )                                     # [B, L_suffix, num_q*head_dim]

    # 5. o_proj (Expert: 2048 → 768). : the fp16 island can leave A in
    # fp16 (fmha returns bf16, but the masked eager path returns fp16); the
    # o_proj weight is bf16, so match it (no-op when A is already bf16).
    # NOTE: a bf16 o_proj (skip the FP8 quant) was tried — isolated it
    # looked ~5µs faster, but e2e it was +0.4ms@25 SLOWER: in-pipeline the bf16
    # weight streams 2× the bytes from DRAM, offsetting the quant savings. Kept
    # FP8 (the static quant ~4.7µs is cheaper than the extra weight-BW).
    o_w = target.expert_layer_o_proj_weights[layer_idx]
    # A is fp16 (FA4 island) or bf16 (eager fallback). linear_fp8 quantizes
    # either to fp8 and emits bf16 (out_dtype) — no fp16->bf16 cast on A.
    attn_out = linear_fp8(A, o_w, out_dtype=o_w.dtype,
                          site_id=f"expert.layer.{layer_idx}.o_proj")

    # 6+7+8a. fused path: when a static scale exists for the
    # gate_proj site, fuse (residual + AdaRMS + FP8 quant) into ONE
    # custom CUDA pass via ``ada_rms_residual_fp8_fused``. The kernel
    # mutates expert_hidden in place to (expert_hidden + attn_out) and
    # emits the FP8 tensor that the SwiGLU's gate/up GEMMs consume —
    # the same input-quant chaining pattern as on the VLM side.
    #
    # γ/β are pre-computed here via the per-sample FiLM linears (M=1,
    # which falls back to bf16 cuBLAS — no FP8 to recover at this size).
    # post-attn γ/β are slots 2,3 of the precomputed FiLM when available.
    site_prefix = f"expert.layer.{layer_idx}.mlp"
    if layer_film is not None:
        gamma = layer_film[2:3]
        beta = layer_film[3:4]
    else:
        gamma = linear_fp8(
            ada_cond,
            target.expert_layer_post_attn_layernorm_gamma_weights[layer_idx],
            target.expert_layer_post_attn_layernorm_gamma_biases[layer_idx],
            site_id=f"expert.layer.{layer_idx}.post_attn_ln.gamma",
        )
        beta = linear_fp8(
            ada_cond,
            target.expert_layer_post_attn_layernorm_beta_weights[layer_idx],
            target.expert_layer_post_attn_layernorm_beta_biases[layer_idx],
            site_id=f"expert.layer.{layer_idx}.post_attn_ln.beta",
        )
    B_e, S_e, _ = expert_hidden.shape
    m_real = B_e * S_e
    # M-pad the post-attn norm FP8 output so the gate/up GEMMs read M=64
    # directly (no pad copy); the down output slices back to m_real.
    _pad = _FP8_MIN_M if _FP8_PAD_MIN_M <= m_real < _FP8_MIN_M else None
    gate_up_merged = getattr(
        target, "expert_layer_mlp_gate_up_merged_weights", None)
    gate_up_fp4 = getattr(target, "expert_mlp_gate_up_fp4", None)
    _fp4_path = (gate_up_fp4 is not None and _pad is not None
                 and _has_static_scales())
    if _fp4_path:
        # FP4 gate_up path uses an fp16 AdaRMS output (no FP8 quant) so the
        # FP4 quant runs on it directly — skips the fp8→fp16 dequant (~18us/layer
        # of tiny launch-bound kernels). expert_hidden is mutated to afr.
        out_fp16 = ada_rms_residual_fp16_mpad_fused(
            expert_hidden, attn_out,
            target.expert_layer_post_attn_layernorm_weights[layer_idx],
            gamma, beta, eps=dims.rms_eps, pad_to=_pad)
        mlp_out = swiglu_mlp_from_fp8(
            None, None,
            gate_weight=target.expert_layer_mlp_gate_proj_weights[layer_idx],
            up_weight=target.expert_layer_mlp_up_proj_weights[layer_idx],
            down_weight=target.expert_layer_mlp_down_proj_weights[layer_idx],
            site_prefix=site_prefix, m_real=m_real,
            gate_up_merged_weight=gate_up_merged[layer_idx],
            gate_up_fp4=gate_up_fp4[layer_idx], x_fp16=out_fp16)
        if _USE_2ND_RES_FUSION:
            return mlp_out, expert_hidden       # next input norm sums these
        return mlp_out + expert_hidden
    fused = ada_rms_residual_fp8_fused(
        expert_hidden, attn_out,
        target.expert_layer_post_attn_layernorm_weights[layer_idx],
        gamma, beta, eps=dims.rms_eps,
        site_id=f"{site_prefix}.gate_proj", pad_to=_pad,
    )
    if fused is not None:
        # expert_hidden has been mutated in place to (afr = attn + hidden).
        out_fp8, act_scale = fused
        mlp_out = swiglu_mlp_from_fp8(
            out_fp8, act_scale,
            gate_weight=target.expert_layer_mlp_gate_proj_weights[layer_idx],
            up_weight=target.expert_layer_mlp_up_proj_weights[layer_idx],
            down_weight=target.expert_layer_mlp_down_proj_weights[layer_idx],
            site_prefix=site_prefix,
            m_real=m_real,
            gate_up_merged_weight=(gate_up_merged[layer_idx]
                                   if gate_up_merged is not None else None),
            gate_up_fp4=None,
        )
        afr = expert_hidden       # bf16, in-place mutated
    else:
        # Eager fallback (no calibration loaded).
        afr = attn_out + expert_hidden
        h_post = ada_rms_norm(
            afr, ada_cond,
            weight=target.expert_layer_post_attn_layernorm_weights[layer_idx],
            gamma_weight=target.expert_layer_post_attn_layernorm_gamma_weights[layer_idx],
            gamma_bias=target.expert_layer_post_attn_layernorm_gamma_biases[layer_idx],
            beta_weight=target.expert_layer_post_attn_layernorm_beta_weights[layer_idx],
            beta_bias=target.expert_layer_post_attn_layernorm_beta_biases[layer_idx],
            eps=dims.rms_eps,
            site_prefix=f"expert.layer.{layer_idx}.post_attn_ln",
        )
        mlp_out = swiglu_mlp(
            h_post,
            gate_weight=target.expert_layer_mlp_gate_proj_weights[layer_idx],
            up_weight=target.expert_layer_mlp_up_proj_weights[layer_idx],
            down_weight=target.expert_layer_mlp_down_proj_weights[layer_idx],
            site_prefix=site_prefix,
        )

    # 9. 2nd residual. : on the fused (calibrated) path, hand mlp_out + afr
    # forward unsummed so the next layer's input norm does the add; the eager
    # fallback (no static scales — its input norm can't fuse a residual) sums here.
    if _USE_2ND_RES_FUSION and fused is not None:
        return mlp_out, afr
    return mlp_out + afr


def denoise_step_36L(
    suffix_hidden: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    ada_cond: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target,
    kv_cache: dict,
    num_layers: int = 36,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    use_fused_attn: bool = True,
    film: torch.Tensor | None = None,    # [N_steps, L, 4, H] precomputed
    step_idx: int | None = None,         # which step's slice to use
    rope_cos: torch.Tensor | None = None,  # precomputed (constant across steps)
    rope_sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full denoise step: 36 layers + final RMSNorm (Expert tower).

    Returns:
        ``[B, L_suffix, 768]`` Expert hidden after final norm — ready for
        the velocity head ``action_out_proj`` (selects last n_action_steps
        tokens).
    """
    h = suffix_hidden
    # build merged gate_up weights once (idempotent, eager warmup).
    if _USE_MERGED_GATE_UP and _has_static_scales():
        prepare_expert_merged_weights(target, num_layers)
    # cos/sin are constant across denoise steps (position_ids depend only
    # on the constant suffix/prefix masks), so the caller precomputes them once
    # per inference. Fall back to per-step build only when not provided.
    if rope_cos is None:
        rope_cos, rope_sin = build_cos_sin_table(
            position_ids, LINGBOT_ROPE_CONFIG, compute_dtype=torch.float32)
    # slice this step's precomputed FiLM γ/β [L, 4, H] once.
    step_film = film[step_idx] if (film is not None and step_idx is not None) \
        else None
    # with the 2nd-residual fusion, each layer returns (mlp_out, afr)
    # unsummed; the next layer's input norm adds them. ``prev_mlp_out`` carries
    # the previous FFN output forward (None for layer 0, where h is the input).
    prev_mlp_out = None
    for layer_idx in range(num_layers):
        out = denoise_step_layer(
            h,
            position_ids=position_ids,
            ada_cond=ada_cond,
            attention_mask=attention_mask,
            target=target, layer_idx=layer_idx,
            kv_cache=kv_cache, dims=dims,
            use_fused_attn=use_fused_attn,
            rope_cos=rope_cos, rope_sin=rope_sin,
            layer_film=step_film[layer_idx] if step_film is not None else None,
            prev_mlp_out=prev_mlp_out,
        )
        if isinstance(out, tuple):
            prev_mlp_out, h = out          # (mlp_out, afr): defer the add
        else:
            h, prev_mlp_out = out, None
    # Settle the final layer's deferred 2nd residual before the final norm.
    if prev_mlp_out is not None:
        h = prev_mlp_out.view_as(h) + h
    # Final norm (plain RMSNorm — final_norm_adanorm=False).
    return rms_norm(h, weight=target.expert_norm_weight, eps=dims.rms_eps)


def mask_kv_cache_pad_rows(
    kv_cache: dict,
    pad_mask: torch.Tensor,                 # [B, L_prefix] bool: True = valid
) -> None:
    """: zero out K/V at padding positions for every layer of the
    cache. Called once after ``prefix_encode_36L`` so the downstream
    denoise step can use unmasked fused attention safely (pad K/V → 0
    contribution; softmax denominator has small known distortion that
    the calibration absorbed into its scales).

    Mutates ``kv_cache`` in place.
    """
    # pad_mask: [B, L] bool → cast to value-dtype, reshape to [B, L, 1, 1]
    # so it broadcasts over (num_kv_heads, head_dim).
    multiplier = pad_mask.unsqueeze(-1).unsqueeze(-1)
    for layer_idx in kv_cache:
        entry = kv_cache[layer_idx]
        entry["key_states"] = entry["key_states"] * multiplier
        entry["value_states"] = entry["value_states"] * multiplier


__all__ = [
    "prefix_encode_layer",
    "prefix_encode_36L",
    "denoise_step_layer",
    "denoise_step_36L",
    "mask_kv_cache_pad_rows",
]
