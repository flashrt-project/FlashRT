"""LingBot-VLA end-to-end ``sample_actions`` — full inference pipeline.

Glues together every component to produce action chunks from raw
inputs (images + state + prompt + noise), matching the upstream
``LingbotVlaPolicy.model.sample_actions`` API.

Pipeline:

    embed_prefix(images, img_masks, lang_tokens, lang_masks)
        ↓ prefix_embs, prefix_pad_masks, prefix_att_masks
    make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = cumsum(prefix_pad_masks) - 1
    prefix_encode_36L(prefix_embs, ...)
        ↓ _, kv_cache (one entry per layer)

    x_t = noise  (or randn)
    time = 1.0
    while time >= -dt / 2:
        embed_suffix(state, x_t, time)
            ↓ time_emb (ada_cond), suffix_embs, suffix_pad_masks, suffix_att_masks
        full attention mask = [prefix_pad_2d | suffix_att_2d]
        position_ids = prefix_offsets + cumsum(suffix_pad_masks) - 1
        denoise_step_36L(suffix_embs, ..., ada_cond=time_emb, kv_cache)
            ↓ out [B, suffix_len, 768]
        v_t = action_out_proj(out[:, -50:])
        x_t += dt * v_t
        time += dt
    return x_t

Reference: upstream
``lingbotvla/models/vla/pi0/modeling_lingbot_vla.py::FlowMatching::sample_actions``
(L1744) + ``embed_prefix`` (L1551) + ``embed_suffix`` (L1616) +
``predict_velocity`` (L1796).
"""

from __future__ import annotations

import math

import einops
import torch
import torch.nn.functional as F
from flash_rt.models.lingbot.kernel_ops import linear_bf16, linear_fp8

from flash_rt.models.lingbot.forward import (
    denoise_step_36L, prefix_encode_36L, mask_kv_cache_pad_rows,
    precompute_expert_film,
)
from flash_rt.models.lingbot.mixed_attention import (
    AttentionDims, DEFAULT_ATTN_DIMS,
)
from flash_rt.models.lingbot.rope_adapter import (
    LINGBOT_ROPE_CONFIG, build_cos_sin_table,
)
from flash_rt.models.lingbot.vit import embed_image


# ════════════════════════════════════════════════════════════════════
#  Helpers ported from lingbotvla/models/vla/pi0/utils.py
# ════════════════════════════════════════════════════════════════════

def _create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int,
    min_period: float, max_period: float, device,
) -> torch.Tensor:
    """Bit-exact replica of ``utils.create_sinusoidal_pos_embedding``."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def precompute_film(
    num_steps: int, target, dims, device, dtype,
) -> torch.Tensor:
    """: the Euler timestep schedule (time 1.0 → 0, step dt=-1/N) is
    deterministic, so build every step's ``ada_cond`` (the timestep
    sinusoidal embedding) up front, [N, H], and run one batched GEMM for
    all Expert FiLM γ/β. Returns ``[N, num_layers, 4, H]``."""
    dt = -1.0 / num_steps
    # Build on-device with arange (capture-safe); torch.tensor([...]) would
    # do an illegal host->device copy inside a CUDA-graph capture.
    times = 1.0 + dt * torch.arange(num_steps, device=device, dtype=dtype)
    ada_all = _create_sinusoidal_pos_embedding(
        times, dims.expert_hidden, min_period=4e-3, max_period=4.0,
        device=device).to(dtype)                       # [N, H]
    return precompute_expert_film(ada_all, target, num_layers=36)


def make_att_2d_masks(
    pad_masks: torch.Tensor, att_masks: torch.Tensor,
) -> torch.Tensor:
    """Bit-exact replica of ``utils.make_att_2d_masks``.

    pad_masks: ``[B, N]`` bool.
    att_masks: ``[B, N]`` int — cumulative-segment block-causal coding.
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# ════════════════════════════════════════════════════════════════════
#  Embed prefix (vision + language)
# ════════════════════════════════════════════════════════════════════

def embed_prefix(
    images: torch.Tensor,         # [B, N, 256, 1176] (pre-patchified)
    img_masks: torch.Tensor,      # [B, N] bool
    lang_tokens: torch.Tensor,    # [B, L_lang] long
    lang_masks: torch.Tensor,     # [B, L_lang] bool
    *,
    target,
    vlm_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(prefix_embs, pad_masks, att_masks)`` matching upstream
    ``FlowMatching.embed_prefix`` for the inference path (no depth).
    """
    B = images.shape[0]
    if images.ndim == 5:
        images = einops.rearrange(images, "b n c h w -> (b n) c h w")
    elif images.ndim == 4:
        images = einops.rearrange(images, "b n l d -> (b n) l d")

    img_emb = embed_image(images, target=target)              # [B*N, 64, 2048]
    num_patch = img_emb.shape[1]
    img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=B)
    num_img_embs = img_emb.shape[1]
    if img_masks.ndim == 1:
        img_masks = img_masks.unsqueeze(0)
    img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

    # Language embedding — F.embedding using vlm_embed_tokens_weight.
    lang_emb = F.embedding(lang_tokens, target.vlm_embed_tokens_weight)
    num_lang_embs = lang_emb.shape[1]

    embs = torch.cat([img_emb, lang_emb], dim=1)
    pad_masks = torch.cat([img_masks, lang_masks], dim=1)
    if vlm_causal:
        att_masks = torch.ones(
            (img_emb.size(0), num_img_embs + num_lang_embs),
            device=images.device, dtype=torch.bool,
        )
    else:
        att_masks = torch.zeros(
            (img_emb.size(0), num_img_embs + num_lang_embs),
            device=images.device, dtype=torch.bool,
        )
    return embs, pad_masks, att_masks


# ════════════════════════════════════════════════════════════════════
#  Embed suffix (state + action + timestep)
# ════════════════════════════════════════════════════════════════════

def embed_suffix(
    state: torch.Tensor,                # [B, state_dim=75]
    noisy_actions: torch.Tensor,        # [B, n_steps=50, action_dim=75]
    timestep: torch.Tensor,             # [B]
    *,
    target,
    proj_width: int = 768,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(time_emb_for_ada, embs, pad_masks, att_masks)`` matching
    upstream ``FlowMatching.embed_suffix`` for the inference path
    (no ``separate_time_proj``)."""
    B = state.shape[0]
    device = state.device
    dtype = state.dtype

    # state projection
    state_emb = linear_fp8(state, target.state_proj_weight,
                         target.state_proj_bias,
                         site_id="action.state_proj")          # [B, 768]

    # sinusoidal timestep embedding (fp32 internal, then cast)
    time_emb = _create_sinusoidal_pos_embedding(
        timestep, proj_width, min_period=4e-3, max_period=4.0, device=device,
    ).to(dtype)                                              # [B, 768]
    time_emb_ori = time_emb

    # action proj
    action_emb = linear_fp8(
        noisy_actions, target.action_in_proj_weight,
        target.action_in_proj_bias,
        site_id="action.in_proj",
    )                                                        # [B, n_steps, 768]

    # broadcast time across n_steps and fuse via action_time_mlp
    time_emb_exp = einops.repeat(
        time_emb, "b d -> b n d", n=action_emb.shape[1])
    action_time_emb = torch.cat([action_emb, time_emb_exp], dim=-1)   # [B, n_steps, 1536]
    action_time_emb = linear_fp8(
        action_time_emb, target.action_time_mlp_in_weight,
        target.action_time_mlp_in_bias,
        site_id="action.time_mlp.in",
    )
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = linear_fp8(
        action_time_emb, target.action_time_mlp_out_weight,
        target.action_time_mlp_out_bias,
        site_id="action.time_mlp.out",
    )                                                        # [B, n_steps, 768]

    n_action = action_time_emb.shape[1]
    embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)   # [B, 1+n_steps, 768]
    pad_masks = torch.ones((B, n_action + 1), device=device, dtype=torch.bool)
    att_masks = torch.zeros((B, n_action + 1), device=device, dtype=torch.bool)
    att_masks[:, :2] = True                                  # state + first action are anchors

    return time_emb_ori, embs, pad_masks, att_masks


# ════════════════════════════════════════════════════════════════════
#  Velocity head + full sample_actions
# ════════════════════════════════════════════════════════════════════

def predict_velocity(
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    kv_cache: dict,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    target,
    n_action_steps: int = 50,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    use_fused_attn: bool = False,
    film: torch.Tensor | None = None,    # precomputed FiLM [N, L, 4, H]
    step_idx: int | None = None,         # which step
    rope_cos: torch.Tensor | None = None,  # precomputed (constant across steps)
    rope_sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """One denoise step — Expert-only forward + velocity head."""
    time_emb, suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(
        state, x_t, timestep, target=target, proj_width=dims.expert_hidden,
    )

    B, suffix_len = suffix_pad_masks.shape
    prefix_len = prefix_pad_masks.shape[1]

    # Full attention mask: [B, suffix_len, prefix_len + suffix_len]. The fused
    # denoise attention ignores the mask , so building it
    # is dead per-step work — skip it entirely when fused.
    if use_fused_attn:
        full_mask = None
    else:
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(B, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_mask = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

    # position_ids only feeds build_cos_sin_table, which is skipped when
    # rope_cos/sin are precomputed . So the
    # per-step sum+cumsum is dead work on the graph path; compute it only on the
    # fallback path that actually builds the RoPE table per step.
    if rope_cos is None:
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    else:
        position_ids = None

    # Denoise step (Expert-only, reads prefix KV from cache).
    out = denoise_step_36L(
        suffix_embs,
        position_ids=position_ids,
        ada_cond=time_emb,
        attention_mask=full_mask,
        target=target, kv_cache=kv_cache, dims=dims,
        use_fused_attn=use_fused_attn,
        film=film, step_idx=step_idx,
        rope_cos=rope_cos, rope_sin=rope_sin,
    )

    # Take last n_action_steps tokens → velocity head (768 → 75).
    suffix_out = out[:, -n_action_steps:]
    v_t = linear_fp8(
        suffix_out, target.action_out_proj_weight,
        target.action_out_proj_bias,
        site_id="action.out_proj",
    )
    return v_t


def sample_actions(
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    *,
    target,
    noise: torch.Tensor | None = None,
    num_steps: int = 50,
    n_action_steps: int = 50,
    action_dim: int = 75,
    vlm_causal: bool = False,
    dims: AttentionDims = DEFAULT_ATTN_DIMS,
    use_fused_denoise_attn: "bool | None" = None,
) -> torch.Tensor:
    """Full inference. Returns ``[B, n_action_steps, action_dim]`` action chunk.

    Match the upstream ABI of ``FlowMatching.sample_actions``.
    """
    # auto-enable fused denoise attention when static scales
    # are loaded. The dynamic-FP8 path's per-call absmax-reduce is
    # unstable on the slightly-shifted activation distribution that the
    # unmasked attention produces (the prior calibration captured
    # the masked-attention distribution; static scales absorb the shift,
    # dynamic scales don't). Without static scales, fall back to eager.
    if use_fused_denoise_attn is None:
        from flash_rt.models.lingbot import calibration as _calib
        use_fused_denoise_attn = _calib.has_static_scales()

    B = state.shape[0]
    device = state.device
    dtype = state.dtype

    if noise is None:
        noise = torch.randn(
            (B, n_action_steps, action_dim), device=device, dtype=dtype,
        )

    # ─── Prefix encode (one-time per task) ────────────────────────
    prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
        images, img_masks, lang_tokens, lang_masks,
        target=target, vlm_causal=vlm_causal,
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    _, kv_cache = prefix_encode_36L(
        prefix_embs,
        position_ids=prefix_position_ids,
        attention_mask=prefix_att_2d,
        target=target, dims=dims,
        pad_mask=prefix_pad_masks,           # enable unmasked FA4 prefix attn
    )

    # zero out KV cache rows for padding positions in prefix so the
    # denoise step can use unmasked fused attention (pad K/V → 0
    # contribution). Saves the eager 4-einsum + softmax + bool-where
    # chain per Expert layer × num_steps.
    if use_fused_denoise_attn:
        mask_kv_cache_pad_rows(kv_cache, prefix_pad_masks)

    # ─── Euler ODE: time 1.0 → 0.0 over num_steps ────────────────
    # Use Python-side scalars for dt/time so loop control doesn't cross
    # to device (CUDA-Graph-capture-safe). expanded_time tensor created
    # per step via torch.full from a Python float.
    dt_f = -1.0 / num_steps
    x_t = noise
    time_f = 1.0
    # the Euler schedule is deterministic, so precompute every step's
    # FiLM γ/β in ONE batched GEMM (1440 tiny M=1 launches → 1) when the
    # fused Expert path is active.
    film = precompute_film(num_steps, target, dims, device, dtype) \
        if use_fused_denoise_attn else None
    # position_ids (hence cos/sin) are constant across denoise steps —
    # the suffix pad/att masks are structural constants. Precompute the RoPE
    # tables ONCE here instead of rebuilding them every step in denoise_step_36L.
    n_action = noise.shape[1]
    suffix_pad_const = torch.ones((B, n_action + 1), device=device, dtype=torch.bool)
    prefix_offsets_c = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids_c = prefix_offsets_c + torch.cumsum(suffix_pad_const, dim=1) - 1
    rope_cos_c, rope_sin_c = build_cos_sin_table(
        position_ids_c, LINGBOT_ROPE_CONFIG, compute_dtype=torch.float32)
    for i in range(num_steps):
        expanded_time = torch.full(
            (B,), time_f, dtype=dtype, device=device,
        )
        v_t = predict_velocity(
            state, prefix_pad_masks, kv_cache, x_t, expanded_time,
            target=target, n_action_steps=n_action_steps, dims=dims,
            use_fused_attn=use_fused_denoise_attn,
            film=film, step_idx=i,
            rope_cos=rope_cos_c, rope_sin=rope_sin_c,
        )
        x_t = x_t + dt_f * v_t
        time_f += dt_f
    return x_t


__all__ = [
    "embed_prefix",
    "embed_suffix",
    "predict_velocity",
    "sample_actions",
    "make_att_2d_masks",
]
