# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-attention RMSNorm for the GQA (full_attention) decoder layer (token generation).

The layer's input_layernorm, run so its output persists in SBUF as the [H0, T, H1] tile qkv_tkg
consumes with no HBM round-trip. Callers hand normed_sb straight to the projections via
NormType.NO_NORM.

Contract (A3B, TP=4, LNC=2, bs=1):
    hidden: [B, T, H] HBM raw pre-norm hidden, bf16/fp32. Left untouched, as the residual.
    gamma:  [1, H] HBM input_layernorm.weight in STANDARD form -- the (1+w) conversion happens once
            at checkpoint load, so there is no +1 in-kernel.
    returns [H0=128, T, H1=16] SBUF, the exact qkv_tkg SBUF-input layout.

Sharding: num_H_shards defaults to lnc, laying the H1 output columns out in the per-shard column
order qkv_tkg's NO_NORM path slices. At T <= SHARDING_THRESHOLD both cores compute the full
replicated norm, matching the unsharded runtime input_layernorm.
"""

from ...common import H0, rmsnorm_to_sbuf

# A3B GQA hidden config. input_layernorm is NOT TP-sharded -- gamma is full H, norm over full H,
# replicated on every rank/core.
H = 2048
H1 = H // H0  # 16 free H-tiles


def pre_attn_rmsnorm_compose(
    hidden, gamma, eps=1e-6, hidden_actual=None, normed_sb=None, name_prefix=""
):
    """RMSNorm raw hidden into an SBUF-resident [H0, T, H1] tile for qkv_tkg.

    gamma is the layer's input_layernorm.weight; rmsnorm_to_sbuf documents the full contract.
    Uses rmsnorm_tkg's default sharding.
    """
    return rmsnorm_to_sbuf(
        hidden,
        gamma,
        eps=eps,
        hidden_actual=hidden_actual,
        normed_sb=normed_sb,
        name_prefix=name_prefix + "prenorm_",
    )
