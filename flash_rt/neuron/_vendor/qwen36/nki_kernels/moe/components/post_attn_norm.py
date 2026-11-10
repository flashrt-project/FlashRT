# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-attention RMSNorm for the Qwen3.6-A3B MoE decoder layer (token generation).

The MoE block's post_attention_layernorm, run so its output persists in SBUF as the [H0, T, H1] tile
the router and the routed/shared experts consume with no HBM round-trip. normed_sb is shared by every
downstream composable.

single_core_forced keys rmsnorm_tkg's num_H_shards, which selects the emitted H-permutation:
  False  num_H_shards = n_prgs -- tp2013, the layout the attention kernels emit, so a megakernel can
         share one SBUF residual. This is what the MoE consumers use. At n_prgs=1 it becomes tp102.
  True   num_H_shards = 1 -- tp102 on every core.

num_H_shards is keyed off the LNC count only. It is independent of whether rmsnorm shards the BxS
work, which it does not below SHARDING_THRESHOLD -- so at T<=2 both cores compute the full norm.
"""

from ...common import H0, rmsnorm_to_sbuf

H = 2048
H1 = H // H0  # 16 free H-tiles


def post_attn_rmsnorm_compose(
    hidden,
    gamma,
    eps=1e-6,
    hidden_actual=None,
    normed_sb=None,
    single_core_forced=False,
    name_prefix="",
):
    """RMSNorm raw post-attn hidden into an SBUF-resident [H0, T, H1] tile (zero HBM round-trip).

    ``gamma`` is the layer's post_attention_layernorm.weight; see ``rmsnorm_to_sbuf`` for the full
    argument contract, and the module docstring for what ``single_core_forced`` does to the layout.
    """
    return rmsnorm_to_sbuf(
        hidden,
        gamma,
        eps=eps,
        hidden_actual=hidden_actual,
        normed_sb=normed_sb,
        single_core_forced=single_core_forced,
        name_prefix=name_prefix + "postnorm_",
    )
