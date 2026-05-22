"""LingBot-VLA Qwen2.5-VL Vision Tower forward.

ViT path that turns the pre-patchified image tensor produced by the
upstream image-processor into per-image token embeddings ready for the
joint prefix sequence (concatenated with the language token embeddings).

Pipeline:

    images [B, N, num_patches=256, patch_dim=1176]
        │ einops.rearrange "b n l d -> (b n) l d"
        ▼ [B*N, 256, 1176]
    patch_embed (Conv3d as kernel=(2,14,14) stride=same)
        │ proj.weight [1280, 3, 2, 14, 14] applied to view(-1, 3, 2, 14, 14)
        ▼ [B*N*256, 1280]
    32 × VitBlock
        │ each block:
        │   x + attn(RMSNorm(x))     (varlen mask via cu_seqlens, ViT M-RoPE)
        │   x + mlp(RMSNorm(x))      (SwiGLU with bias, GELU NOT used here)
        ▼ [B*N*256, 1280]
    merger
        │ RMSNorm → view(-1, 4*1280=5120) → Linear(5120,5120) → GELU → Linear(5120,2048)
        ▼ [B*N*64, 2048]                (spatial_merge_size=2: 4 tokens → 1)
    split by image, stack
        ▼ [B*N, 64, 2048]
    embed_image returns this; embed_prefix then does
    einops.rearrange "(b n) l d -> b (n l) d" → [B, N*64, 2048].

Reference upstream implementations (bit-exact targets):
    Qwen2_5_VisionPatchEmbed.forward    @ qwenvl_in_vla.py L75
    Qwen2_5_VLVisionAttention.forward   @ L148  (eager, fp32 softmax)
    Qwen2_5_VLMLP.forward               @ L46   (SwiGLU)
    Qwen2_5_VLVisionBlock.forward       @ L264
    Qwen2_5_VLPatchMerger.forward       @ L114-127
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from flash_rt.models.lingbot.kernel_ops import (
    attention_mha_bf16_fused, linear_bf16, linear_fp8,
    vit_qkv_bias_rope_fused,
)

# P1: fuse the per-block ViT qkv bias-add + 2-D M-RoPE into one custom
# kernel (replaces add_bias + the eager fp32 RoPE storm). Opt-in; the
# eager path stays the bit-exact test reference.
_USE_VIT_FUSED_QKV_ROPE = True

from flash_rt.models.lingbot.norms import rms_norm
from flash_rt.models.lingbot.vit_rope_adapter import (
    LINGBOT_VIT_ROPE_CONFIG, build_cos_sin_table,
)

# Capture-safe cache for ``image_grid_thw`` and ``split_sizes`` (which
# get re-constructed each call in the upstream API). Populated once
# during warmup, reused during CUDA-Graph capture+replay.
_GRID_CACHE: dict = {}


# ════════════════════════════════════════════════════════════════════
#  Building blocks
# ════════════════════════════════════════════════════════════════════

def vit_patch_embed(
    images: torch.Tensor,
    *,
    proj_weight: torch.Tensor,
    temporal_patch_size: int = 2,
    patch_size: int = 14,
    in_channels: int = 3,
    embed_dim: int = 1280,
) -> torch.Tensor:
    """Conv3d patch projection. Input is the pre-patchified tensor from
    the image processor: ``[N_total, num_patches, patch_dim]`` where
    ``patch_dim = in_channels * temporal_patch_size * patch_size**2``.

    Matches ``Qwen2_5_VisionPatchEmbed.forward``:
        view(-1, in_ch, t, p, p) → Conv3d(k=stride=(t,p,p)) → view(-1, embed_dim).
    """
    target_dtype = proj_weight.dtype
    x = images.view(
        -1, in_channels, temporal_patch_size, patch_size, patch_size
    ).to(target_dtype)
    x = F.conv3d(x, proj_weight, bias=None,
                 stride=(temporal_patch_size, patch_size, patch_size))
    return x.view(-1, embed_dim)


_VIT_MLP_PAD_DONE = "_vit_mlp_intermediate_padded"


def pad_vit_mlp_weights(target, *, align: int = 16) -> None:
    """Zero-pad the ViT MLP intermediate dim up to a 16-multiple, in place.

    Qwen2.5-VL ViT has intermediate=3420, which is NOT 16-aligned, so the
    gate/up/down FP8 GEMMs fall off the aligned cutlass fast path and run at
    ~15% efficiency (~6× slower). Padding gate/up output rows (N) and the
    down input cols (K) to 3424 with zeros recovers the fast path. The pad is
    mathematically exact: gate_pad = 0·x + 0 = 0 → silu(0) = 0, up_pad = 0,
    so h_pad = silu(0)·0 = 0, and the down GEMM's extra zero-input cols
    contribute nothing. absmax is unchanged → static calibration scales stay
    valid (no recalibration). Idempotent; runs once in eager warmup.
    """
    if getattr(target, _VIT_MLP_PAD_DONE, False):
        return
    gate_w = target.vit_block_mlp_gate_proj_weights
    inter = gate_w[0].shape[0]
    padded = ((inter + align - 1) // align) * align
    if padded == inter:
        setattr(target, _VIT_MLP_PAD_DONE, True)
        return
    n_pad = padded - inter

    def pad_rows(t):  # [N, K] -> [N+n_pad, K]
        return F.pad(t, (0, 0, 0, n_pad))

    def pad_vec(t):   # [N] -> [N+n_pad]
        return F.pad(t, (0, n_pad))

    def pad_cols(t):  # [N, K] -> [N, K+n_pad]
        return F.pad(t, (0, n_pad))

    target.vit_block_mlp_gate_proj_weights = [pad_rows(w) for w in gate_w]
    target.vit_block_mlp_up_proj_weights = [
        pad_rows(w) for w in target.vit_block_mlp_up_proj_weights]
    target.vit_block_mlp_gate_proj_biases = [
        pad_vec(b) for b in target.vit_block_mlp_gate_proj_biases]
    target.vit_block_mlp_up_proj_biases = [
        pad_vec(b) for b in target.vit_block_mlp_up_proj_biases]
    target.vit_block_mlp_down_proj_weights = [
        pad_cols(w) for w in target.vit_block_mlp_down_proj_weights]
    setattr(target, _VIT_MLP_PAD_DONE, True)


def vit_swiglu_mlp(
    x: torch.Tensor,
    target,
    block_idx: int,
) -> torch.Tensor:
    """ViT FFN: SwiGLU **with bias** (Qwen2_5_VLMLP, bias=True path)."""
    prefix = f"vit.block.{block_idx}.mlp"
    gate = linear_fp8(x,
        target.vit_block_mlp_gate_proj_weights[block_idx],
        target.vit_block_mlp_gate_proj_biases[block_idx],
        site_id=f"{prefix}.gate_proj")
    up = linear_fp8(x,
        target.vit_block_mlp_up_proj_weights[block_idx],
        target.vit_block_mlp_up_proj_biases[block_idx],
        site_id=f"{prefix}.up_proj")
    h = F.silu(gate) * up
    return linear_fp8(h,
        target.vit_block_mlp_down_proj_weights[block_idx],
        target.vit_block_mlp_down_proj_biases[block_idx],
        site_id=f"{prefix}.down_proj")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF rotate_half: ``cat([-x[..., D/2:], x[..., :D/2]], dim=-1)``."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact replica of ``apply_rotary_pos_emb_vision`` from
    transformers.qwen2_5_vl:

        q, k → fp32, cos/sin → fp32 with unsqueeze(-2)
        q_embed = q * cos + rotate_half(q) * sin
        cast back to input dtype.
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_emb = q * cos + _rotate_half(q) * sin
    k_emb = k * cos + _rotate_half(k) * sin
    return q_emb.to(orig_q_dtype), k_emb.to(orig_k_dtype)


def vit_attention(
    hidden_states: torch.Tensor,        # [seq, 1280]
    *,
    attn_mask: torch.Tensor,            # ignored when use_per_view_fused=True;
                                         # kept in API for the eager fallback path
    cos: torch.Tensor,                  # [seq, 80] ViT 2-D M-RoPE
    sin: torch.Tensor,
    target,
    block_idx: int,
    num_heads: int = 16,
    head_dim: int = 80,
    num_views: int = 3,
    view_len: int = 256,
    use_per_view_fused: bool = True,
) -> torch.Tensor:
    """One ViT attention block.

    fast path (``use_per_view_fused=True``, default): the upstream
    ViT mask is block-diagonal — each image's 256 tokens only attend
    within their own image. We exploit this directly by reshaping
    Q/K/V to per-view tensors and running ``attention_mha_bf16_fused``
    once per image (3 unmasked attentions @ S=256 each, vs one large
    eager softmax @ S=768 that wastes 2/3 of compute on masked-out
    cross-image terms). cuBLAS-decomposed MHA replaces the 4-einsum
    + softmax chain. Same primitive Pi0.5 uses; cos-equivalent to the
    eager + mask path (verified ≤1e-4 absolute).

    Eager fallback (``use_per_view_fused=False``): the original
    explicit-mask path. Kept for tests that need bit-exact reference.
    """
    seq_len = hidden_states.shape[0]
    use_fused = (use_per_view_fused and _USE_VIT_FUSED_QKV_ROPE
                 and seq_len == num_views * view_len)
    qkv_bias = target.vit_block_attn_qkv_biases[block_idx]
    qkv = linear_fp8(hidden_states,
        target.vit_block_attn_qkv_weights[block_idx],
        None if use_fused else qkv_bias,
        site_id=f"vit.block.{block_idx}.attn.qkv")

    if use_fused:
        # P1: fused bias-add + 2-D M-RoPE from the raw interleaved
        # [seq, 3*NH*HD] GEMM output → q/k/v [seq, NH*HD] (q/k roped).
        q, k, v = vit_qkv_bias_rope_fused(
            qkv, qkv_bias, cos, sin,
            num_heads=num_heads, head_dim=head_dim)
        q_v = q.view(num_views, view_len, num_heads, head_dim)
        k_v = k.view(num_views, view_len, num_heads, head_dim)
        v_v = v.view(num_views, view_len, num_heads, head_dim)
        outs = []
        for i in range(num_views):
            outs.append(attention_mha_bf16_fused(
                q_v[i:i + 1], k_v[i:i + 1], v_v[i:i + 1]))
        attn = torch.cat(outs, dim=1).view(seq_len, num_heads * head_dim)
        return linear_fp8(attn,
            target.vit_block_attn_proj_weights[block_idx],
            target.vit_block_attn_proj_biases[block_idx],
            site_id=f"vit.block.{block_idx}.attn.proj")

    qkv = qkv.reshape(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)             # each [seq, num_heads, head_dim]
    q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

    if use_per_view_fused and seq_len == num_views * view_len:
        # per-view fused attention. q/k/v reshape from
        # [N*L, NH, HD] to [N, L, NH, HD] and call the unmasked fused
        # kernel once per view (L=256 each). The block-diagonal mask
        # is encoded structurally by the per-view dispatch.
        q_v = q.view(num_views, view_len, num_heads, head_dim)
        k_v = k.view(num_views, view_len, num_heads, head_dim)
        v_v = v.view(num_views, view_len, num_heads, head_dim)
        outs = []
        for i in range(num_views):
            outs.append(attention_mha_bf16_fused(
                q_v[i:i + 1], k_v[i:i + 1], v_v[i:i + 1]))
        # outs: list of [1, L, NH*HD] → cat to [1, N*L, NH*HD] → [seq, NH*HD]
        attn = torch.cat(outs, dim=1).view(seq_len, num_heads * head_dim)
        return linear_fp8(attn,
            target.vit_block_attn_proj_weights[block_idx],
            target.vit_block_attn_proj_biases[block_idx],
            site_id=f"vit.block.{block_idx}.attn.proj")

    # Eager fallback (used by per-block tests; matches upstream bit-exact).
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_w = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    attn_w = attn_w + attn_mask
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = torch.matmul(attn_w, v).transpose(0, 1).reshape(
        seq_len, num_heads * head_dim,
    )
    return linear_fp8(attn,
        target.vit_block_attn_proj_weights[block_idx],
        target.vit_block_attn_proj_biases[block_idx],
        site_id=f"vit.block.{block_idx}.attn.proj")


def vit_block(
    hidden_states: torch.Tensor,
    *,
    attn_mask: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    target,
    block_idx: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """One ViT transformer block (norm1+attn+residual, norm2+mlp+residual)."""
    h = hidden_states + vit_attention(
        rms_norm(hidden_states,
                 weight=target.vit_block_norm1_weights[block_idx], eps=eps),
        attn_mask=attn_mask, cos=cos, sin=sin,
        target=target, block_idx=block_idx,
    )
    h = h + vit_swiglu_mlp(
        rms_norm(h, weight=target.vit_block_norm2_weights[block_idx], eps=eps),
        target=target, block_idx=block_idx,
    )
    return h


def vit_merger(
    hidden_states: torch.Tensor,        # [seq, 1280]
    *,
    target,
    spatial_merge_size: int = 2,
    context_dim: int = 1280,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``Qwen2_5_VLPatchMerger``: RMSNorm → view → MLP (Linear+GELU+Linear).

    Output: ``[seq / merge_size**2, 2048]`` — 4 tokens collapsed into 1 LLM
    token via the view (NOT a learned reduction; the merge happens by
    flattening 4 adjacent feature vectors).
    """
    h = rms_norm(hidden_states, weight=target.vit_merger_ln_q_weight, eps=eps)
    merged_dim = context_dim * (spatial_merge_size ** 2)
    h = h.view(-1, merged_dim)          # [seq/4, 5120]
    h = linear_fp8(h,
        target.vit_merger_mlp_0_weight, target.vit_merger_mlp_0_bias,
        site_id="vit.merger.mlp_0")
    h = F.gelu(h)                       # default approx="none" matches nn.GELU()
    h = linear_fp8(h,
        target.vit_merger_mlp_2_weight, target.vit_merger_mlp_2_bias,
        site_id="vit.merger.mlp_2")
    return h


# ════════════════════════════════════════════════════════════════════
#  Full ViT forward
# ════════════════════════════════════════════════════════════════════

def vit_forward(
    images: torch.Tensor,               # [B*N, 256, 1176]
    *,
    attn_mask: torch.Tensor,            # pre-built block-diag mask [1, seq, seq]
    cos: torch.Tensor,                  # pre-built ViT M-RoPE [seq, 80]
    sin: torch.Tensor,
    target,
    num_blocks: int = 32,
) -> torch.Tensor:
    """Full Qwen2.5-VL ViT tower forward.

    Returns ``[total_merged_tokens, 2048]`` where
    ``total_merged_tokens = sum(t*h*w/4 for each image) = N * 64`` for
    the LingBot baseline.

    capture-prep: ``attn_mask`` / ``cos`` / ``sin`` are now caller-
    provided (built once per shape in :func:`embed_image` via the
    shape-keyed ``_GRID_CACHE``). The previous in-function
    ``torch.repeat_interleave`` + ``.item()`` mask loop were capture-
    unsafe and have moved to that warmup path.
    """
    # Pad MLP intermediate to a 16-multiple (3420->3424) so gate/up/down FP8
    # GEMMs hit the aligned cutlass fast path. Idempotent; the alloc happens
    # once in eager warmup, before CUDA-Graph capture.
    pad_vit_mlp_weights(target)

    # 1. Patch embed.
    x = vit_patch_embed(images,
        proj_weight=target.vit_patch_embed_proj_weight)              # [seq, 1280]

    # 2. 32 blocks.
    for block_idx in range(num_blocks):
        x = vit_block(x,
            attn_mask=attn_mask, cos=cos, sin=sin,
            target=target, block_idx=block_idx,
        )

    # 3. Merger collapses 4 tokens into 1.
    return vit_merger(x, target=target)


def _build_vit_capture_cache(
    image_grid_thw: torch.Tensor,        # [BN, 3] long
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Pre-build the capture-unsafe constants once per shape.

    Returns ``(attn_mask [1, seq, seq] dtype, cos [seq, 80], sin [seq, 80])``.
    Called from :func:`embed_image` outside any CUDA-Graph capture.
    """
    # cu_seqlens: torch.repeat_interleave with tensor `repeats` produces a
    # variable-length output that CUDA-Graph capture cannot record. Run it
    # here, outside capture, and discard once the mask is built.
    cu = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu, (1, 0), value=0)

    # Block-diagonal attention mask: filled with -inf, then zero per image
    # block. ``.item()`` is fine here (outside capture) — the mask becomes
    # a captured constant.
    attn_mask = torch.full(
        (1, seq_len, seq_len), torch.finfo(dtype).min,
        device=device, dtype=dtype,
    )
    for i in range(1, int(cu_seqlens.shape[0])):
        s = int(cu_seqlens[i - 1].item())
        e = int(cu_seqlens[i].item())
        attn_mask[..., s:e, s:e] = 0

    # ViT 2-D M-RoPE table .
    cos, sin = build_cos_sin_table(
        image_grid_thw, LINGBOT_VIT_ROPE_CONFIG,
        compute_dtype=torch.float32,
    )
    return attn_mask, cos, sin


def embed_image(
    images: torch.Tensor,               # [B, N, 256, 1176]  OR  [B*N, 256, 1176]
    *,
    target,
    image_grid_thw: torch.Tensor | None = None,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """Wrap ``vit_forward`` to match the upstream ``embed_image`` ABI
    (modeling_lingbot_vla.py:1203):

        Input: ``images`` with optional batch dim.
            If 5-D ``[B, N, C, H, W]``: NOT supported here — caller must
            pre-patchify (matches the lingbot baseline harness).
            If 4-D ``[B, N, L, D]``: rearranged to ``[B*N, L, D]``.
            If 3-D ``[(B*N), L, D]``: used directly.
        ``image_grid_thw`` defaults to ``[[1, sqrt(L), sqrt(L)]] *
        images.shape[0]`` — matches upstream's default.

    Returns: ``[B*N, L/4, 2048]`` (4 patches collapsed into 1 LLM token).
    """
    import einops

    if images.ndim == 4:
        images = einops.rearrange(images, "b n l d -> (b n) l d")
    BN, L, _ = images.shape

    # Capture-safe cache: pre-compute everything that crosses to host
    # (grid_thw, split_sizes, cu_seqlens, attn_mask, cos, sin) once per
    # shape during the first call (warmup). Subsequent calls reuse the
    # cached tensors and never re-enter the capture-unsafe construction
    # path — required for CUDA-Graph capture.
    key = (BN, L, images.device.index)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        h = w = int(L ** 0.5)
        igtw = torch.tensor(
            [[1, h, w]] * BN, device=images.device, dtype=torch.long,
        )
        split_sizes = [int((1 * h * w) // 4)] * BN
        seq_len = BN * L  # patch_embed flattens to this seq length
        attn_mask, cos, sin = _build_vit_capture_cache(
            igtw, seq_len, images.device,
            dtype=torch.bfloat16,
        )
        cached = (igtw, split_sizes, attn_mask, cos, sin)
        _GRID_CACHE[key] = cached
    igtw, split_sizes, attn_mask, cos, sin = cached
    if image_grid_thw is None:
        image_grid_thw = igtw

    flat = vit_forward(
        images, attn_mask=attn_mask, cos=cos, sin=sin, target=target)
    parts = torch.split(flat, split_sizes)
    return torch.stack(parts, dim=0)    # [B*N, num_tokens_per_img, 2048]


__all__ = [
    "vit_patch_embed",
    "vit_swiglu_mlp",
    "pad_vit_mlp_weights",
    "vit_attention",
    "vit_block",
    "vit_merger",
    "vit_forward",
    "embed_image",
]
