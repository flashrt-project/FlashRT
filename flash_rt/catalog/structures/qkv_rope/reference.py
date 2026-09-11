"""Reference math for packed QKV bias, split and rotate-half RoPE."""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def qkv_rope_ref(
    packed_qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add packed bias, split Q/K/V and apply pre-expanded half RoPE."""
    batch, tokens, _ = packed_qkv.shape
    q_width = q_heads * head_dim
    kv_width = kv_heads * head_dim
    packed = packed_qkv + qkv_bias
    q, k, v = packed.split((q_width, kv_width, kv_width), dim=-1)
    q = q.view(batch, tokens, q_heads, head_dim)
    k = k.view(batch, tokens, kv_heads, head_dim)
    v = v.view(batch, tokens, kv_heads, head_dim)
    if cos.shape[-1] == head_dim // 2:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    cos = cos.view(batch, tokens, 1, head_dim).float()
    sin = sin.view(batch, tokens, 1, head_dim).float()
    q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
    k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)
    return q, k, v
