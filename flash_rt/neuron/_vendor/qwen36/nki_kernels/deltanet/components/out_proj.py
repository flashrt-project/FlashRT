# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet attention output projection (o_proj) for token generation.

Thin composable over nkilib's output_projection_tkg.

The recurrence and gated norm are value-head sharded, so each core holds only its own heads of the
gated output -- but the o_proj matmul contracts over all value heads. A small cross-LNC sendrecv
gathers the attention tensor before the matmul; output_projection_tkg then H-shards the large
output across cores, so the result needs no LNC reduce.

Input is head-major [T, W_core]. The composable transposes each head to head_dim-on-partition and
assembles the [d, 1, Hv, T] layout output_projection_tkg expects.

The TP all-reduce is deferred: this returns the per-rank partial.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.utils.common_types import QuantizationType

# head_dim (value-head width); equals the partition-dim max.
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def out_proj_compose(attn_sb, out_w, sbm=None, out_in_sb=False):
    """Per-rank DeltaNet output projection.

    Gathers all heads across cores, transposes to head_dim-on-partition, then projects.

    Args:
        attn_sb:   [T, W_core] SBUF, this core's value heads head-major.
        out_w:     [value_dim, hidden] HBM, transpose of the o_proj nn.Linear weight.
        sbm:       optional BufferManager passed through to output_projection_tkg.
        out_in_sb: return the per-core H-shard as an SBUF tile instead of HBM [T, hidden].

    Returns:
        The per-rank o_proj partial; each core writes its disjoint hidden/n shard.
    """
    T, W_core = attn_sb.shape
    value_dim, hidden = out_w.shape
    d = P_MAX
    kernel_assert(W_core % d == 0, "W_core must be a multiple of head_dim")

    n = nl.num_programs(0)
    c = nl.program_id(0)
    Hv_core = W_core // d
    Hv = Hv_core * n
    kernel_assert(
        Hv * d == value_dim, "value_dim must equal Hv * head_dim (all cores' heads)"
    )
    kernel_assert(T <= P_MAX, "B*S = T must not exceed P_MAX")
    kernel_assert(hidden % n == 0, "hidden must be divisible by the LNC core count")

    # This core's heads, transposed to head_dim-on-partition.
    attn_loc = nl.ndarray((d, Hv_core, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    for h_local in nl.affine_range(Hv_core):
        # gen3+ requires the PSUM dst dtype to match the input.
        head_t = nl.ndarray((d, T), dtype=attn_sb.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=head_t, data=attn_sb[0:T, h_local * d : (h_local + 1) * d]
        )
        nisa.tensor_copy(dst=attn_loc[0:d, h_local, 0:T], src=head_t[0:d, 0:T])

    # Assemble all Hv heads at their global head positions.
    attn_full = nl.ndarray((d, 1, Hv, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=attn_full[0:d, 0, c * Hv_core : (c + 1) * Hv_core, 0:T],
        src=attn_loc[0:d, 0:Hv_core, 0:T],
    )

    if n > 1:
        # Exchange heads with the other core.
        other = 1 - c
        nisa.sendrecv(
            src=attn_loc[0:d, 0:Hv_core, 0:T],
            dst=attn_full[0:d, 0, other * Hv_core : (other + 1) * Hv_core, 0:T],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    # output_projection_tkg H-shards by program_id, each core writing its own [T, hidden/n] slice.
    return output_projection_tkg(
        attention=attn_full,
        weight=out_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=out_in_sb,
        OUT_IN_SB=out_in_sb,
        sbm=sbm,
    )
