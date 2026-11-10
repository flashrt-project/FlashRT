# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GQA decode attention core with head_dim tiled across the partition axis.

From-scratch reference implementation, kept for cross-checking the vendored attention_tkg path in
attention.py. Not on the perf-critical path.

nkilib's TKG attention caps head_dim at 128 because D sits on the partition axis and the PE
contraction-K, stationary-free, and partition are all <= 128. This core lifts that cap by tiling D
into D_TILES = ceil(D/128) partition tiles at the two matmul sites:
  QK^T  contraction over D: split-K, the D_TILES partial products PSUM-accumulate into one scores tile
  P.V   D is the stationary-free: D_TILES stationary V-slices give D_TILES PSUM halves, one per tile
Softmax reduces over the KV-length axis, which is on free, so the tiling leaves it untouched.

Scores are produced as [Tq, L] so softmax is a plain free-axis reduce -- no partition-axis reduction,
no transpose-for-max, no cross-core online-softmax combine. The cost is one PE transpose of the
probabilities before P.V, cheap at Tq <= 8.

GQA: the query heads share one kv head, so the shared K/V tiles are reused as the matmul operand for
every query head with no explicit replication. The heads are independent, so the caller may shard
them across cores. Precision is bf16 IO with fp32 matmul/softmax accumulate.
"""

import math

import nki.isa as nisa
import nki.language as nl

# Partition-axis maximum: head_dim tile width, PE contraction-K, and stationary-free cap.
P_MAX = 128
# Per-matmul moving-free limit (== PSUM bank free width); caps the QK^T scores chunk.
MM1_FREE = 512
# Fill for masked scores. exp(scale * NEG_FILL - bias) underflows to 0 in fp32 while
# staying far from fp32 overflow when multiplied by the softmax scale.
NEG_FILL = -30000.0


def div_ceil(n, d):
    """Ceiling division for tile counts."""
    return (n + d - 1) // d


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def gqa_attention_core(q_sb, k_sb, v_sb, T):
    """Head_dim-tiled GQA decode attention core, one TP shard and one or more query heads.

    Args:
        q_sb: [P_MAX, D_TILES, Tq] SBUF bf16 query, head_dim on partition.
              Tq = T * q_heads_local, head-major so each head's T columns are contiguous.
        k_sb: [P_MAX, D_TILES, L] SBUF bf16 key, head_dim on partition. Shared across query heads.
        v_sb: [P_MAX, L_TILES, D] SBUF bf16 value, KV-length on partition in 128-row chunks; the
              last chunk is partially valid. Shared across query heads.
        T:    decode width; q_heads_local = Tq // T.

    Returns:
        out_sb [P_MAX, D_TILES, Tq] SBUF bf16, head_dim on partition.

    Note:
        L = prior_len + T, the committed prior cache followed by the T active tokens. The mask is
        causal over the active block; the entire prior region is always visible.
    """
    _, D_TILES, Tq = q_sb.shape
    L = k_sb.shape[2]
    L_TILES = v_sb.shape[1]
    D = D_TILES * P_MAX
    dtype = q_sb.dtype

    kernel_assert(Tq % T == 0, "Tq must be a multiple of T (heads grouped, T per head)")
    kernel_assert(L >= T, "L = prior_len + T must be >= T")
    kernel_assert(
        L_TILES == div_ceil(L, P_MAX), "v_sb L_TILES must equal ceil(L / P_MAX)"
    )

    qh_local = Tq // T
    prior_len = L - T
    scale = 1.0 / math.sqrt(D)

    out_sb = nl.ndarray((P_MAX, D_TILES, Tq), dtype=dtype, buffer=nl.sbuf)

    # The causal mask runs on the GpSimd engine (affine_select), which requires its tile
    # to be partition-0-based -- it cannot start mid-partition. Heads are grouped on the
    # scores partition axis, so iterate per head: each head's scores tile is a fresh
    # [T, L] (partition 0), and the head index becomes a free-dim/M slice on the matmul
    # operands (q columns, out columns), never a partition slice. T <= 2, so this is a
    # handful of tiny matmul groups. K/V are GQA-shared and reused for every head.
    for h in range(qh_local):
        hc = h * T  # this head's column base in the grouped Tq axis

        # ---- MM1: scores[t, l] = sum_d Q[t, d] * K[l, d]; split-K over the D_TILES. ----
        # stationary Q [d, T] (d on partition, T free -> output partition), moving K [d, l]
        # (d on partition, l free <= MM1_FREE -> output free). The D_TILES partials
        # accumulate in PSUM (split-K). Result scores [T, l] in fp32.
        scores = nl.ndarray((T, L), dtype=nl.float32, buffer=nl.sbuf)
        for lc in range(div_ceil(L, MM1_FREE)):
            l0 = lc * MM1_FREE
            lp = min(MM1_FREE, L - l0)

            sc_psum = nl.ndarray((T, lp), dtype=nl.float32, buffer=nl.psum)
            for dt in range(D_TILES):
                d_size = min(P_MAX, D - dt * P_MAX)
                nisa.nc_matmul(
                    dst=sc_psum[0:T, 0:lp],
                    stationary=q_sb[0:d_size, dt, hc : hc + T],
                    moving=k_sb[0:d_size, dt, l0 : l0 + lp],
                )
            nisa.tensor_copy(dst=scores[0:T, l0 : l0 + lp], src=sc_psum[0:T, 0:lp])

        # ---- Causal mask. Predicate over (t = partition, l = free):
        # prior_len + t - l >= 0  <=>  l <= prior_len + t (keep), else fill NEG_FILL.
        nisa.affine_select(
            dst=scores[0:T, 0:L],
            pattern=[[-1, L]],
            channel_multiplier=1,
            on_true_tile=scores[0:T, 0:L],
            on_false_value=NEG_FILL,
            cmp_op=nl.greater_equal,
            offset=prior_len,
        )

        # ---- Softmax over the KV-length (free axis). p = exp(scale*(s - max)) / sum. ----
        row_max = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_max, op=nl.maximum, data=scores[0:T, 0:L], axis=[1], keepdims=True
        )
        # activation bias (Nx1): -scale * max. exp(scale*data + bias) = exp(scale*(s-max)).
        neg_bias = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_bias, data=row_max, op0=nl.multiply, operand0=-scale)

        probs = nl.ndarray((T, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=probs, op=nl.exp, data=scores[0:T, 0:L], scale=scale, bias=neg_bias
        )

        row_sum = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_sum, op=nl.add, data=probs[0:T, 0:L], axis=[1], keepdims=True
        )

        row_recip = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=row_recip, data=row_sum)
        # Normalize and cast to bf16 (matmul-ready) in one op.
        probs_bf16 = nl.ndarray((T, L), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=probs_bf16, data=probs, op0=nl.multiply, operand0=row_recip
        )

        # ---- Transpose P [T, L] -> P_T [l(128-chunks), T] for the P.V contraction. ----
        # P.V contracts over l, so l must be on the partition axis. Transpose each 128-row
        # KV chunk: nc_transpose([T, cp]) -> [cp, T].
        p_t = nl.ndarray((P_MAX, L_TILES, T), dtype=dtype, buffer=nl.sbuf)
        for c in range(L_TILES):
            c0 = c * P_MAX
            cp = min(P_MAX, L - c0)

            pt_psum = nl.ndarray((cp, T), dtype=dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=pt_psum, data=probs_bf16[0:T, c0 : c0 + cp])
            nisa.tensor_copy(dst=p_t[0:cp, c, 0:T], src=pt_psum[0:cp, 0:T])

        # ---- MM2: out[d, t] = sum_l V[l, d] * P[t, l]; D_TILES stationary V-slices. ----
        # stationary V [l, d_slice] (l on partition, d_slice free <= 128 -> output
        # partition), moving P_T [l, T] (l on partition, T free -> output free). Accumulate
        # over the KV-length 128-chunks. One PSUM half [d_size, T] per head_dim tile.
        for dt in range(D_TILES):
            d_size = min(P_MAX, D - dt * P_MAX)

            out_psum = nl.ndarray((d_size, T), dtype=nl.float32, buffer=nl.psum)
            for c in range(L_TILES):
                c0 = c * P_MAX
                cp = min(P_MAX, L - c0)
                nisa.nc_matmul(
                    dst=out_psum[0:d_size, 0:T],
                    stationary=v_sb[0:cp, c, dt * P_MAX : dt * P_MAX + d_size],
                    moving=p_t[0:cp, c, 0:T],
                )
            # PSUM -> SBUF with fp32 -> bf16 cast.
            nisa.activation(
                dst=out_sb[0:d_size, dt, hc : hc + T],
                op=nl.copy,
                data=out_psum[0:d_size, 0:T],
            )

    return out_sb
