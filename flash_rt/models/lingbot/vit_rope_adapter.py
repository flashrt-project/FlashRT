"""Host-side ViT 2-D M-RoPE cos/sin table construction for LingBot-VLA.

LingBot's Qwen2.5-VL ViT (32 blocks, hidden=1280, 16 heads, head_dim=80)
uses a 2-D M-RoPE over the (h, w) patch grid — NOT the 3-axis text M-RoPE
(``mrope_section=[16,24,24]``) and NOT the 1-D LLM RoPE
(``flash_rt.models.lingbot.rope_adapter`` — that's for Mixed-Head LLM).

Upstream reference (replicated bit-exact, validated against the upstream model):

    ``lingbotvla/models/vla/pi0/qwenvl_in_vla.py::Qwen2_5_VisionRotaryEmbedding``
    ``lingbotvla/models/vla/pi0/qwenvl_in_vla.py::Qwen2_5_VLVisionTransformer.rot_pos_emb``

Math:
    half_dim = head_dim // 2  (= 40 for Qwen2.5-VL ViT)
    inv_freq[i] = 1 / theta ** (2 i / half_dim)   for i ∈ [0, half_dim/2 = 20)

For each token at patch position (h_pos, w_pos):
    freqs_h = h_pos * inv_freq                    shape (half_dim/2,)
    freqs_w = w_pos * inv_freq                    shape (half_dim/2,)
    freqs   = concat([freqs_h, freqs_w], dim=-1)  shape (half_dim,) = 40

Then for the attention apply step (HF rotate_half form):
    emb = concat([freqs, freqs], dim=-1)          shape (head_dim=80)
    cos = emb.cos(),  sin = emb.sin()
    q_emb = apply_rotary_pos_emb_vision(q, k, cos, sin)

Position-ID construction (matches upstream's spatial-merge-aware layout):
    For each image with grid (t, h, w):
        hpos = arange(h).unsqueeze(1).expand(-1, w)
              .reshape(h/sm, sm, w/sm, sm).permute(0,2,1,3).flatten()
        wpos = arange(w).unsqueeze(0).expand(h, -1).
              .reshape(h/sm, sm, w/sm, sm).permute(0,2,1,3).flatten()
        pos_per_image = stack([hpos, wpos], dim=-1).repeat(t, 1)
    pos_ids = cat([pos_per_image_i for i], dim=0)

The permute(0,2,1,3) is what implements the spatial_merge_size=2 layout
where each 2×2 block of patches (which the merger collapses into one LLM
token) is kept adjacent in the flattened sequence.

This module is pure-PyTorch (CPU/GPU agnostic). It does NOT call any
flash_rt_kernels symbol. Built once per ``set_prompt`` (image layout is
fixed at that point) and reused as input to the on-device ViT attention
RoPE kernel during every graph replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LingbotVitRopeConfig:
    """LingBot-VLA Qwen2.5-VL ViT 2-D M-RoPE constants.

    Locked by the ViT submodule of Qwen2.5-VL-3B-Instruct:
        hidden_size=1280, num_heads=16 → head_dim=80,  half_dim=40
        theta=10000 (default of Qwen2_5_VisionRotaryEmbedding)
        spatial_merge_size=2

    Note ``half_dim`` is the dim passed to ``Qwen2_5_VisionRotaryEmbedding``,
    which then computes inv_freq over ``[0, half_dim/2)`` — so the actual
    inv_freq length is ``half_dim // 2 = 20``.
    """

    head_dim: int = 80                    # 1280 // 16
    half_dim: int = 40                    # head_dim // 2 (what upstream Embedding sees)
    theta: float = 10_000.0
    spatial_merge_size: int = 2


def compute_inv_freq(
    cfg: LingbotVitRopeConfig,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``1 / theta ** (2 i / half_dim)`` for ``i ∈ [0, half_dim/2)``.

    Returns: tensor of shape ``(half_dim // 2,) = (20,)``.

    Note we follow the upstream ``Qwen2_5_VisionRotaryEmbedding.__init__``
    which does ``arange(0, dim, 2)`` for ``dim = half_dim`` — so the result
    has length ``half_dim // 2``, NOT ``half_dim``.
    """
    return 1.0 / (
        cfg.theta ** (torch.arange(0, cfg.half_dim, 2, dtype=dtype, device=device) / cfg.half_dim)
    )


def build_position_ids_from_grid(
    grid_thw: torch.Tensor,
    cfg: LingbotVitRopeConfig,
    *,
    device=None,
) -> torch.Tensor:
    """Replicate upstream ``rot_pos_emb`` position-id construction.

    Args:
        grid_thw: tensor of shape ``(num_images, 3)`` with (t, h, w) per
            image. For LingBot baseline this is ``[[1, 16, 16], [1, 16, 16],
            [1, 16, 16]]`` (3 cameras × 224/14 = 16 patches per side).
        cfg: ``LingbotVitRopeConfig``.

    Returns:
        ``pos_ids`` of shape ``(sum_seqlen, 2)`` long — column 0 is h_pos,
        column 1 is w_pos. ``sum_seqlen = sum(t * h * w for each image)``.
    """
    if device is None:
        device = grid_thw.device

    sm = cfg.spatial_merge_size
    pieces: list[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        hpos = (
            torch.arange(h, dtype=torch.long, device=device)
            .unsqueeze(1)
            .expand(-1, w)
            .reshape(h // sm, sm, w // sm, sm)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos = (
            torch.arange(w, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(h, -1)
            .reshape(h // sm, sm, w // sm, sm)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        # Stack to (h*w, 2), repeat t times for video frames (T=1 for images).
        pieces.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(pieces, dim=0)


def build_rotary_pos_emb_full(
    max_grid_size: int,
    cfg: LingbotVitRopeConfig,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the full ``(max_grid_size, half_dim/2)`` freq lookup table.

    This is what the upstream ``Qwen2_5_VisionRotaryEmbedding.forward``
    produces: ``outer(arange(seqlen), inv_freq)``.
    """
    inv_freq = compute_inv_freq(cfg, device=device, dtype=dtype)
    seq = torch.arange(max_grid_size, dtype=dtype, device=device)
    return torch.outer(seq, inv_freq)


def build_freqs_from_positions(
    pos_ids: torch.Tensor,
    cfg: LingbotVitRopeConfig,
    *,
    max_grid_size: int | None = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compose the full per-token freq tensor used by the ViT attention.

    Replicates the last lines of upstream ``rot_pos_emb``::

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb      = rotary_pos_emb_full[pos_ids].flatten(1)

    Args:
        pos_ids: ``(seq, 2)`` from ``build_position_ids_from_grid``.
        cfg: ``LingbotVitRopeConfig``.
        max_grid_size: defaults to ``pos_ids.max() + 1`` — pass an explicit
            value when you want to share the lookup table across multiple
            calls of differing image sizes.

    Returns:
        ``freqs`` of shape ``(seq, half_dim)`` — the 2-D-composed M-RoPE
        freq vector per token. Each token's vector is
        ``[inv_freq * h_pos | inv_freq * w_pos]``.
    """
    if device is None:
        device = pos_ids.device
    if max_grid_size is None:
        max_grid_size = int(pos_ids.max().item()) + 1
    full = build_rotary_pos_emb_full(max_grid_size, cfg, device=device, dtype=dtype)
    # full[pos_ids] -> (seq, 2, half_dim/2); flatten(1) -> (seq, half_dim).
    return full[pos_ids].flatten(1)


def build_cos_sin_table(
    grid_thw: torch.Tensor,
    cfg: LingbotVitRopeConfig,
    *,
    compute_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype | None = None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """End-to-end host-side cos/sin builder for the ViT attention kernel.

    Args:
        grid_thw: ``(num_images, 3)`` with (t, h, w) per image.
        cfg: ``LingbotVitRopeConfig``.
        compute_dtype: dtype used inside (default fp32 — matches upstream).
        out_dtype: dtype of returned tensors. ``None`` → same as
            compute_dtype. Set to fp16 for kernel handoff.

    Returns:
        ``cos``, ``sin`` of shape ``(seq, head_dim)``.

    The returned tensors match the upstream attention path::

        emb = cat((rotary_pos_emb, rotary_pos_emb), dim=-1)   # (seq, head_dim)
        cos = emb.cos(); sin = emb.sin()
    """
    if device is None:
        device = grid_thw.device

    pos_ids = build_position_ids_from_grid(grid_thw, cfg, device=device)
    max_grid_size = int(grid_thw[:, 1:].max().item())
    freqs = build_freqs_from_positions(
        pos_ids, cfg, max_grid_size=max_grid_size,
        device=device, dtype=compute_dtype,
    )
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb)
    sin = torch.sin(emb)
    if out_dtype is not None and out_dtype != compute_dtype:
        cos = cos.to(out_dtype)
        sin = sin.to(out_dtype)
    return cos, sin


# Default config instance for the LingBot-VLA Qwen2.5-VL ViT path.
LINGBOT_VIT_ROPE_CONFIG = LingbotVitRopeConfig(
    head_dim=80, half_dim=40, theta=10_000.0, spatial_merge_size=2,
)


__all__ = [
    "LingbotVitRopeConfig",
    "LINGBOT_VIT_ROPE_CONFIG",
    "compute_inv_freq",
    "build_position_ids_from_grid",
    "build_rotary_pos_emb_full",
    "build_freqs_from_positions",
    "build_cos_sin_table",
]
