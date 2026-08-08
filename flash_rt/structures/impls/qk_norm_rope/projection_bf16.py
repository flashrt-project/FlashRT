"""Packed-QKV implementation of projection-scope Q/K norm plus RoPE.

This is the Wan form of :mod:`qk_norm_rope`: Q and K are normalized over
the complete projection before the output is viewed as heads. It consumes
the contiguous output of a QKV pack and materializes Q/K/V attention
workspaces in one Hub kernel. Per-head Cosmos/Qwen/audio normalization is a
different implementation variant of the same catalog structure.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardRefused, GuardedSeam


class ProjectionQkNormRope(GuardedSeam, torch.nn.Module):
    """Fixed-shape packed QKV postprocess for projection-scope RMSNorm."""

    _frt_can_fallback = False

    def __init__(
        self,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        *,
        batch: int,
        tokens: int,
        heads: int,
        head_dim: int,
        qkv_bias: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(batch, tokens, heads, head_dim) <= 0:
            raise ValueError("qk_norm_rope: shape dimensions must be positive")
        if head_dim % 2:
            raise ValueError("qk_norm_rope: head_dim must be even")
        dim = int(heads) * int(head_dim)
        if q_norm_weight.numel() != dim or k_norm_weight.numel() != dim:
            raise ValueError(
                "qk_norm_rope: projection-scope norm weights must each "
                f"contain heads * head_dim = {dim} elements")
        device = q_norm_weight.device
        if k_norm_weight.device != device:
            raise ValueError("qk_norm_rope: Q/K norm weights must share device")
        if qkv_bias is None:
            qkv_bias = torch.zeros(
                3 * dim, device=device, dtype=torch.bfloat16)
        if qkv_bias.numel() != 3 * dim or qkv_bias.device != device:
            raise ValueError(
                "qk_norm_rope: qkv_bias must contain 3 * heads * head_dim "
                "elements on the norm-weight device")

        self.batch = int(batch)
        self.tokens = int(tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.dim = dim
        self.eps = float(eps)
        self._fn = hub_kernel(
            "flashrt/flashrt-qkv-cache-rope",
            ">=1",
        ).qkv_split_bias_norm_rope_v_bf16
        self.register_buffer(
            "q_norm_weight",
            q_norm_weight.detach().reshape(dim).to(torch.bfloat16).contiguous(),
        )
        self.register_buffer(
            "k_norm_weight",
            k_norm_weight.detach().reshape(dim).to(torch.bfloat16).contiguous(),
        )
        self.register_buffer(
            "qkv_bias",
            qkv_bias.detach().reshape(3 * dim).to(torch.bfloat16).contiguous(),
        )
        shape = (self.batch, self.tokens, self.heads, self.head_dim)
        self.register_buffer(
            "q_out",
            torch.empty(shape, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "k_out",
            torch.empty(shape, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "v_out",
            torch.empty(shape, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        self._frt_arm(
            dtypes={torch.bfloat16},
            device=device,
            k=3 * dim,
            rows=self.batch * self.tokens,
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
        expected = (self.batch, self.tokens, 3 * self.dim)
        if tuple(packed_qkv.shape) != expected:
            raise GuardRefused(
                f"qk_norm_rope: packed QKV shape {tuple(packed_qkv.shape)} "
                f"(bound for {expected})")
        freq_shape = (self.tokens, self.head_dim // 2)
        if tuple(cos.shape) != freq_shape or tuple(sin.shape) != freq_shape:
            raise GuardRefused(
                f"qk_norm_rope: cos/sin must have shape {freq_shape}")
        if (cos.dtype is not torch.float32 or sin.dtype is not torch.float32
                or cos.device != packed_qkv.device
                or sin.device != packed_qkv.device):
            raise GuardRefused(
                "qk_norm_rope: cos/sin must be float32 on the QKV device")
        if not packed_qkv.is_contiguous():
            raise GuardRefused("qk_norm_rope: packed QKV must be contiguous")

        return self._fn(
            packed_qkv,
            self.qkv_bias,
            self.q_norm_weight,
            self.k_norm_weight,
            cos,
            sin,
            self.heads,
            self.head_dim,
            rope_seq_len=self.tokens,
            eps=self.eps,
            q_out=self.q_out,
            k_out=self.k_out,
            v_out=self.v_out,
        )


def bind_projection_qk_norm_rope(
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    batch: int,
    tokens: int,
    heads: int,
    head_dim: int,
    qkv_bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> ProjectionQkNormRope:
    """Bind the fixed-shape projection-scope implementation."""
    return ProjectionQkNormRope(
        q_norm_weight,
        k_norm_weight,
        batch=batch,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        qkv_bias=qkv_bias,
        eps=eps,
    )
