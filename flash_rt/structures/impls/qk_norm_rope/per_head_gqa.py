"""Per-head GQA Q/K RMSNorm and rotate-half RoPE implementation."""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardRefused, GuardedSeam
from ...workspace import lease


class PerHeadGqaQkNormRope(GuardedSeam, torch.nn.Module):
    """Consume packed GQA QKV and produce attention-ready workspaces."""

    _frt_can_fallback = False

    def __init__(
        self,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        *,
        row_capacity: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        workspace_lane: str | None = None,
    ) -> None:
        super().__init__()
        if row_capacity <= 0 or q_heads <= 0 or kv_heads <= 0:
            raise ValueError(
                "qk_norm_rope: row capacity and head counts must be positive"
            )
        if head_dim != 128:
            raise ValueError(
                "qk_norm_rope: per-head GQA kernel requires head_dim == 128"
            )
        if q_norm_weight.shape != (128,) or k_norm_weight.shape != (128,):
            raise ValueError(
                "qk_norm_rope: per-head norm weights must have shape (128,)"
            )
        if q_norm_weight.device != k_norm_weight.device:
            raise ValueError("qk_norm_rope: Q/K norm weights must share device")

        self.row_capacity = int(row_capacity)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = 128
        self.eps = float(eps)
        kernel = hub_kernel("flashrt/flashrt-qkv-cache-rope", ">=1")
        try:
            self._fn = kernel.qkv_split_per_head_norm_rope_bf16
        except AttributeError as exc:
            raise ValueError(
                "qk_norm_rope: flashrt-qkv-cache-rope artifact lacks the "
                "per-head GQA entry"
            ) from exc
        self.register_buffer(
            "q_norm_weight",
            q_norm_weight.detach().to(torch.bfloat16).contiguous(),
        )
        self.register_buffer(
            "k_norm_weight",
            k_norm_weight.detach().to(torch.bfloat16).contiguous(),
        )
        device = q_norm_weight.device
        if workspace_lane is not None:
            # a caller that declares its outputs call-scoped (no cache,
            # consumed inside the layer) shares one workspace per lane
            # across every same-shape layer — the difference between a
            # 19k-token host binding 52 layers and OOMing on the 53rd
            self.q_out = lease((self.row_capacity, self.q_heads, 128),
                               torch.bfloat16, device,
                               tag=f"qkr_q|{workspace_lane}")
            self.k_out = lease((self.row_capacity, self.kv_heads, 128),
                               torch.bfloat16, device,
                               tag=f"qkr_k|{workspace_lane}")
            self.v_out = lease((self.row_capacity, self.kv_heads, 128),
                               torch.bfloat16, device,
                               tag=f"qkr_v|{workspace_lane}")
        else:
            self.register_buffer(
                "q_out",
                torch.empty(
                    self.row_capacity,
                    self.q_heads,
                    128,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                persistent=False,
            )
            self.register_buffer(
                "k_out",
                torch.empty(
                    self.row_capacity,
                    self.kv_heads,
                    128,
                    device=device,
                    dtype=torch.bfloat16,
                ),
                persistent=False,
            )
            self.register_buffer(
                "v_out",
                torch.empty_like(self.k_out),
                persistent=False,
            )
        self._frt_arm(
            dtypes={torch.bfloat16},
            device=device,
            k=(self.q_heads + 2 * self.kv_heads) * 128,
            row_capacity=self.row_capacity,
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
        if packed_qkv.dim() != 3:
            raise GuardRefused(
                "qk_norm_rope: packed QKV must have shape (B, T, width)"
            )
        batch, tokens, width = packed_qkv.shape
        rows = batch * tokens
        expected_width = (self.q_heads + 2 * self.kv_heads) * 128
        if width != expected_width or rows > self.row_capacity:
            raise GuardRefused(
                "qk_norm_rope: packed GQA QKV is outside the bound form"
            )
        expected_freq = (batch, tokens, 128)
        if cos.shape != expected_freq or sin.shape != expected_freq:
            raise GuardRefused(
                f"qk_norm_rope: cos/sin must have shape {expected_freq}"
            )
        if (
            cos.dtype is not torch.bfloat16
            or sin.dtype is not torch.bfloat16
            or cos.device != packed_qkv.device
            or sin.device != packed_qkv.device
        ):
            raise GuardRefused(
                "qk_norm_rope: cos/sin must be BF16 on the QKV device"
            )
        if not packed_qkv.is_contiguous():
            raise GuardRefused("qk_norm_rope: packed QKV must be contiguous")

        q_out = self.q_out[:rows].view(batch, tokens, self.q_heads, 128)
        k_out = self.k_out[:rows].view(batch, tokens, self.kv_heads, 128)
        v_out = self.v_out[:rows].view(batch, tokens, self.kv_heads, 128)
        return self._fn(
            packed_qkv,
            self.q_norm_weight,
            self.k_norm_weight,
            cos,
            sin,
            self.q_heads,
            self.kv_heads,
            eps=self.eps,
            q_out=q_out,
            k_out=k_out,
            v_out=v_out,
        )


def bind_per_head_gqa_qk_norm_rope(
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    *,
    row_capacity: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
    workspace_lane: str | None = None,
) -> PerHeadGqaQkNormRope:
    """Bind the capacity-guarded per-head GQA implementation.

    ``q_norm_weight`` and ``k_norm_weight`` are checkpoint-native vectors
    with layout ``[head_dim]``.  The packed activation is laid out
    ``[Q heads..., K heads..., V heads...]``, each head contiguous at
    ``head_dim`` elements; no transposition or interleaving of the norm
    weights is performed.
    """
    return PerHeadGqaQkNormRope(
        q_norm_weight,
        k_norm_weight,
        row_capacity=row_capacity,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        eps=eps,
        workspace_lane=workspace_lane,
    )
