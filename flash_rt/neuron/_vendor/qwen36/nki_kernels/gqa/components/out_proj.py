# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA attention output projection (o_proj) for token generation (head_dim=256).

Thin composable over nkilib's output_projection_tkg, which folds N sub-heads of at most 128 each and
PSUM-accumulates them. A head_dim=256 q-head is therefore presented as two 128-wide sub-heads, so
q_heads=4 gives 8 sub-heads of 128 -- structurally the same as the DeltaNet o_proj.

The attention core's output already has head_dim on the partition axis, split into D_TILES tiles of
128, so no per-head PE transpose is needed here. Each (q-head, d-tile) pair is one 128-wide sub-head
already on the partition axis. This composable only reorders the free axes into
output_projection_tkg's sub-head layout, gathers the other core's sub-heads via sendrecv, and runs
the H-sharded matmul.

Sub-head ordering is q-head major, d-tile minor, which already matches how out_w is row-indexed by
value_dim -- so the weight passes through unreshaped.

The TP all-reduce is deferred: this returns the per-rank partial.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.output_projection.output_projection_tkg import output_projection_tkg
from nkilib.core.utils.common_types import QuantizationType

# head_dim sub-head width; equals the partition-dim max (a head_dim=256 q-head -> 2 sub-heads of 128).
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def out_proj_compose(attn_sb, out_w, T, gate_sb=None, sbm=None):
    """Per-rank GQA attention output projection.

    Optional sigmoid gate, then reorder the free axes into sub-head order, gather the other core's
    sub-heads via sendrecv, and run the H-sharded matmul.

    Args:
        attn_sb: [P_MAX, D_TILES, Tq] SBUF, this core's query heads with head_dim on partition.
                 Tq = qh_local * T, head-major.
        out_w:   [value_dim, hidden] HBM, transpose of the o_proj nn.Linear weight.
        T:       decode width; qh_local = Tq // T query heads on this core.
        gate_sb: optional sigmoid output gate in the same layout as attn_sb. When given, the input
                 is gated elementwise before the matmul.
        sbm:     optional BufferManager passed through to output_projection_tkg.

    Returns:
        o_out [T, hidden] HBM, the per-rank partial; each core writes its disjoint hidden/n shard.
    """
    _, D_TILES, Tq = attn_sb.shape
    value_dim, hidden = out_w.shape

    kernel_assert(Tq % T == 0, "Tq must be a multiple of T (heads grouped, T per head)")
    qh_local = Tq // T
    N_core = qh_local * D_TILES

    n = nl.num_programs(0)
    c = nl.program_id(0)
    N = N_core * n
    kernel_assert(
        N * P_MAX == value_dim,
        "value_dim must equal N * P_MAX (all cores' sub-heads of 128)",
    )
    kernel_assert(T <= P_MAX, "B*S = T must not exceed P_MAX")
    kernel_assert(hidden % n == 0, "hidden must be divisible by the LNC core count")

    # Optional output gate: gated = attn_sb * sigmoid(gate_sb), elementwise in the head_dim-on-partition
    # layout (layout-agnostic). In a megakernel the gate projection emits this same layout.
    if gate_sb != None:
        kernel_assert(
            gate_sb.shape == attn_sb.shape, "gate_sb must match attn_sb shape/layout"
        )

        sig = nl.ndarray(attn_sb.shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sig, op=nl.sigmoid, data=gate_sb)

        gated = nl.ndarray(attn_sb.shape, dtype=attn_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gated, data1=attn_sb, data2=sig, op=nl.multiply)
        attn_src = gated
    else:
        attn_src = attn_sb

    # Reorder the free axes into sub-head order [d, N_core, T]: local sub-head n_local =
    # h_local*D_TILES + d_tile (q-head major, d-tile minor). head_dim is already on the partition
    # axis, so this is a pure SBUF reorder -- no transpose (the key simplification vs DeltaNet o_proj).
    attn_loc = nl.ndarray((P_MAX, N_core, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    for h_local in nl.affine_range(qh_local):
        for d_tile in nl.affine_range(D_TILES):
            n_local = h_local * D_TILES + d_tile
            nisa.tensor_copy(
                dst=attn_loc[0:P_MAX, n_local, 0:T],
                src=attn_src[0:P_MAX, d_tile, h_local * T : h_local * T + T],
            )

    # Assemble all N sub-heads at GLOBAL positions: core c's sub-heads at [c*N_core, (c+1)*N_core).
    # Because q-heads are sharded as contiguous blocks, global sub-head = c*N_core + n_local.
    attn_full = nl.ndarray((P_MAX, 1, N, T), dtype=attn_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=attn_full[0:P_MAX, 0, c * N_core : (c + 1) * N_core, 0:T],
        src=attn_loc[0:P_MAX, 0:N_core, 0:T],
    )

    if n > 1:
        # Exchange local sub-heads with the other core; place them at the other core's positions.
        other = 1 - c
        nisa.sendrecv(
            src=attn_loc[0:P_MAX, 0:N_core, 0:T],
            dst=attn_full[0:P_MAX, 0, other * N_core : (other + 1) * N_core, 0:T],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    # output_projection_tkg infers N=N and D=P_MAX from attn_full.shape, checks
    # weight.shape[0] == N*P_MAX == value_dim, and H-shards the [T, hidden] output across cores.
    return output_projection_tkg(
        attention=attn_full,
        weight=out_w,
        bias=None,
        quantization_type=QuantizationType.NONE,
        TRANSPOSE_OUT=False,
        OUT_IN_SB=False,
        sbm=sbm,
    )
