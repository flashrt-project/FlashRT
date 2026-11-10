# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE layer TKG stage kernels -- the megakernel building blocks.

post_attn_norm (input RMSNorm) -> {routed_experts, shared_expert, sigma_gate} -> gated sum (moe_layer).

All consumers read the SAME SBUF ``normed_sb`` (norm once, zero HBM round-trip); ``moe_layer_compose``
returns the single per-rank partial ``routed_local + g * shared_local`` for one downstream all-reduce.
"""

from .moe_layer import (
    gated_sum,
    moe_layer_compose,
    moe_layer_fwd,
    sigma_gate_compose,
)
from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts import moe_routed_compose, routed_experts_compose
from .shared_expert import (
    moe_h_shard_decision,
    moe_shared_compose,
    moe_tkg_shard_decision,
    shared_expert_compose,
)

__all__ = [
    "post_attn_rmsnorm_compose",
    "routed_experts_compose",
    "moe_routed_compose",
    "shared_expert_compose",
    "moe_shared_compose",
    "moe_tkg_shard_decision",
    "moe_h_shard_decision",
    "moe_layer_compose",
    "moe_layer_fwd",
    "sigma_gate_compose",
    "gated_sum",
]
