"""BF16 packed-QKV bias/split/rotate-half RoPE implementation."""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardRefused, GuardedSeam


class PackedBiasQkvRope(GuardedSeam, torch.nn.Module):
    """Fixed-capacity wrapper around one formal Hub custom op."""

    _frt_can_fallback = False

    def __init__(
        self,
        qkv_bias: torch.Tensor,
        *,
        row_capacity: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if min(row_capacity, q_heads, kv_heads, head_dim) <= 0:
            raise ValueError(
                "qkv_rope: capacities and head dimensions must be positive"
            )
        if head_dim % 2 or head_dim > 256:
            raise ValueError(
                "qkv_rope: head_dim must be even and no larger than 256"
            )
        width = (q_heads + 2 * kv_heads) * head_dim
        if qkv_bias.numel() != width:
            raise ValueError(f"qkv_rope: bias must contain {width} elements")
        if qkv_bias.dtype is not torch.bfloat16 or not qkv_bias.is_cuda:
            raise ValueError("qkv_rope: bias must be CUDA BF16")

        self.row_capacity = int(row_capacity)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.width = int(width)
        self._fn = hub_kernel(
            "flashrt/flashrt-qkv-cache-rope", ">=1"
        ).qkv_split_bias_rope_bf16
        self.register_buffer("qkv_bias", qkv_bias.detach().contiguous())
        device = qkv_bias.device
        self.register_buffer(
            "q_out",
            torch.empty(
                row_capacity,
                q_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "k_out",
            torch.empty(
                row_capacity,
                kv_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "v_out", torch.empty_like(self.k_out), persistent=False
        )
        self._frt_arm(
            dtypes={torch.bfloat16}, device=device, k=width, row_capacity=row_capacity
        )

    def forward(
        self,
        packed_qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        admitted = self._frt_admit(packed_qkv, cos, sin)
        if admitted is not PROCEED:
            return admitted
        if packed_qkv.dim() != 3 or packed_qkv.shape[0] != 1:
            raise GuardRefused(
                "qkv_rope: packed QKV must have shape (1, T, width)"
            )
        _, tokens, width = packed_qkv.shape
        if width != self.width or tokens > self.row_capacity:
            raise GuardRefused("qkv_rope: packed QKV is outside the bound capacity")
        if not packed_qkv.is_contiguous():
            raise GuardRefused("qkv_rope: packed QKV must be contiguous")
        expected = {(1, tokens, self.head_dim // 2), (1, tokens, self.head_dim)}
        if tuple(cos.shape) not in expected or tuple(sin.shape) not in expected:
            raise GuardRefused(
                "qkv_rope: cos/sin shape does not match the token/head form"
            )
        if (
            cos.dtype is not torch.float32
            or sin.dtype is not torch.float32
            or cos.device != packed_qkv.device
            or sin.device != packed_qkv.device
            or not cos.is_contiguous()
            or not sin.is_contiguous()
        ):
            raise GuardRefused("qkv_rope: cos/sin must be contiguous CUDA FP32")

        q_out = self.q_out[:tokens].view(1, tokens, self.q_heads, self.head_dim)
        k_out = self.k_out[:tokens].view(1, tokens, self.kv_heads, self.head_dim)
        v_out = self.v_out[:tokens].view(1, tokens, self.kv_heads, self.head_dim)
        return self._fn(
            packed_qkv,
            self.qkv_bias,
            cos,
            sin,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            q_out=q_out,
            k_out=k_out,
            v_out=v_out,
        )


def bind_packed_bias_qkv_rope(
    qkv_bias: torch.Tensor,
    *,
    row_capacity: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> PackedBiasQkvRope:
    return PackedBiasQkvRope(
        qkv_bias,
        row_capacity=row_capacity,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
