"""FA2 implementation for factored two-way attention.

This is the common MoT form used by multimodal diffusion transformers:
the causal/understanding branch attends to itself causally, while the
full/generation branch attends to the joint causal + full sequence. The
host has already projected, normalized, and rotated Q/K/V; this structure
owns only the two attention calls and the factored output boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardRefused, GuardedSeam


@dataclass
class _AttentionScratch:
    out: torch.Tensor
    lse: torch.Tensor
    workspace: Any


def _scratch(kernel, q: torch.Tensor, k: torch.Tensor) -> _AttentionScratch:
    out, lse = kernel.allocate_outputs(q)
    return _AttentionScratch(
        out=out,
        lse=lse,
        workspace=kernel.allocate_workspace(q, k),
    )


class FactoredTwoWayAttention(GuardedSeam, torch.nn.Module):
    """Allocation-free single-sample two-way GQA attention processor."""

    _frt_can_fallback = False

    def __init__(self, capture: dict[str, Any]) -> None:
        super().__init__()
        query = capture["query"]
        key = capture["key"]
        value = capture["value"]
        causal_q = query["causal_seq"]
        full_q = query["full_only_seq"]
        causal_k = key["causal_seq"]
        full_k = key["full_only_seq"]
        causal_v = value["causal_seq"]
        full_v = value["full_only_seq"]

        if query["sample_offsets"].numel() != 2:
            raise ValueError(
                "attention_core two_way: only one-sample factored packs "
                "are qualified")
        if causal_q.ndim != 3 or full_q.ndim != 3:
            raise ValueError(
                "attention_core two_way: Q must have shape [tokens, heads, dim]")
        if causal_k.shape != causal_v.shape or full_k.shape != full_v.shape:
            raise ValueError(
                "attention_core two_way: K and V shapes differ")
        if causal_q.shape[-1] != causal_k.shape[-1]:
            raise ValueError(
                "attention_core two_way: Q and KV head dimensions differ")
        if causal_q.dtype != torch.bfloat16:
            raise ValueError(
                "attention_core two_way: current Hub FA2 path requires BF16")

        self.causal_shape = tuple(causal_q.shape)
        self.full_q_shape = tuple(full_q.shape)
        self.causal_kv_shape = tuple(causal_k.shape)
        self.full_kv_shape = tuple(full_k.shape)
        self.total_tokens = int(
            query["_causal_indices"].numel()
            + query["_full_indices"].numel())
        self.scale = float(causal_q.shape[-1] ** -0.5)
        self._kernel = hub_kernel(
            "flashrt/fa2-seqused-runtime",
            ">=1",
        )

        device = causal_q.device
        dtype = causal_q.dtype
        kv_heads = causal_k.shape[1]
        head_dim = causal_k.shape[2]
        self.register_buffer(
            "joint_k",
            torch.empty(
                1,
                self.total_tokens,
                kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "joint_v",
            torch.empty_like(self.joint_k),
            persistent=False,
        )
        self.register_buffer(
            "causal_indices",
            query["_causal_indices"].long().detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "full_indices",
            query["_full_indices"].long().detach().clone(),
            persistent=False,
        )

        causal_q4 = causal_q.unsqueeze(0)
        causal_k4 = causal_k.unsqueeze(0)
        full_q4 = full_q.unsqueeze(0)
        self._causal = _scratch(
            self._kernel,
            causal_q4,
            causal_k4,
        )
        self._full = _scratch(
            self._kernel,
            full_q4,
            self.joint_k,
        )
        self._frt_arm(
            dtypes=(dtype,),
            device=device,
            k=int(head_dim),
            rows=int(causal_q.shape[0] * causal_q.shape[1]),
        )

    def _validate(self, query, key, value) -> None:
        shapes = (
            tuple(query["causal_seq"].shape),
            tuple(query["full_only_seq"].shape),
            tuple(key["causal_seq"].shape),
            tuple(key["full_only_seq"].shape),
            tuple(value["causal_seq"].shape),
            tuple(value["full_only_seq"].shape),
        )
        expected = (
            self.causal_shape,
            self.full_q_shape,
            self.causal_kv_shape,
            self.full_kv_shape,
            self.causal_kv_shape,
            self.full_kv_shape,
        )
        if shapes != expected:
            raise GuardRefused(
                f"attention_core two_way: shapes {shapes} "
                f"(bound for {expected})")

    def forward(self, query, key, value):
        admitted = self._frt_admit(query["causal_seq"])
        if admitted is not PROCEED:
            return admitted
        self._validate(query, key, value)

        causal_q = query["causal_seq"].unsqueeze(0)
        causal_k = key["causal_seq"].unsqueeze(0)
        causal_v = value["causal_seq"].unsqueeze(0)
        cs = self._causal
        causal_out = self._kernel.forward_static(
            causal_q,
            causal_k,
            causal_v,
            out=cs.out,
            softmax_lse=cs.lse,
            workspace=cs.workspace,
            softmax_scale=self.scale,
            causal=True,
        )

        self.joint_k[0].index_copy_(
            0,
            self.causal_indices,
            key["causal_seq"],
        )
        self.joint_k[0].index_copy_(
            0,
            self.full_indices,
            key["full_only_seq"],
        )
        self.joint_v[0].index_copy_(
            0,
            self.causal_indices,
            value["causal_seq"],
        )
        self.joint_v[0].index_copy_(
            0,
            self.full_indices,
            value["full_only_seq"],
        )
        fs = self._full
        full_out = self._kernel.forward_static(
            query["full_only_seq"].unsqueeze(0),
            self.joint_k,
            self.joint_v,
            out=fs.out,
            softmax_lse=fs.lse,
            workspace=fs.workspace,
            softmax_scale=self.scale,
            causal=False,
        )

        out = dict(query)
        out["causal_seq"] = causal_out.squeeze(0).flatten(-2, -1)
        out["full_only_seq"] = full_out.squeeze(0).flatten(-2, -1)
        return out


def bind_two_way_attention(capture: dict[str, Any]
                           ) -> FactoredTwoWayAttention:
    """Bind a factored two-way attention processor from one real call."""
    return FactoredTwoWayAttention(capture)
