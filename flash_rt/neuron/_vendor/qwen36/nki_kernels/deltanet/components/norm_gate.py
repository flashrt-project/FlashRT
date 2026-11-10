# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated per-head RMSNorm for the DeltaNet recurrence output (matches Qwen3_5MoeRMSNormGated).

Per (token, value-head), normalized over the 128-wide head_dim:
    out = (x * rsqrt(mean(x^2) + eps)) * gamma * silu(z)

head_dim stays on the free axis, so the per-head reduce needs no transpose.

norm_gate_row is the SBUF helper used at the recurrence's per-token output seam.
deltanet_gated_rmsnorm is a thin HBM harness over the same math, for unit testing.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# head_dim (value-head width); equals the partition-dim max.
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def fold_gamma_silu(gsz_all, gamma, T, W, d):
    """In place, turn a block of raw z rows into the gate gamma*silu(z).

    gsz_all [1, T*W] holds z on entry and the gate on exit. gamma is the replicated [d] norm
    weight, free-broadcast across (token, head).
    """
    TH = T * (W // d)

    gamma_sb = nl.ndarray((1, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma.ap(pattern=[[d, 1], [1, d]], offset=0))
    nisa.activation(dst=gsz_all, op=nl.silu, data=gsz_all, bias=None, scale=1.0)
    gsz_3d = gsz_all.ap(pattern=[[T * W, 1], [d, TH], [1, d]], offset=0)
    nisa.tensor_tensor(
        dst=gsz_3d,
        data1=gsz_3d,
        data2=gamma_sb.ap(pattern=[[d, 1], [0, TH], [1, d]], offset=0),
        op=nl.multiply,
    )


def norm_gate_row(o_row, gsz_row, eps, d):
    """One token's gated per-head RMSNorm.

    o_row and gsz_row are [1, W_core] SBUF rows, gsz_row pre-folded by fold_gamma_silu.
    Returns [1, W_core] = RMSNorm(o_row) * gsz_row.
    """
    W = o_row.shape[1]
    kernel_assert(W % d == 0, "W_core must be a multiple of head_dim")
    kernel_assert(d == P_MAX, "head_dim must equal P_MAX")
    Hv = W // d

    # 3D head-major view of the [1, W] row.
    o_3d = o_row.ap(pattern=[[W, 1], [d, Hv], [1, d]], offset=0)

    # Per-head sum-of-squares: free-axis reduce over the innermost d.
    sq = nl.ndarray((1, Hv, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=o_3d, bias=None, scale=1.0)

    sumsq = nl.ndarray((1, Hv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sumsq, op=nl.add, data=sq, axis=(2,))

    # rsqrt(sumsq/d + eps), with the mean-scale and eps folded into one Scalar-engine op.
    inv = nl.ndarray((1, Hv, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=sumsq, bias=eps, scale=1.0 / d)

    # Normalize: x_j * inv_h (broadcast inv over j).
    out = nl.ndarray((1, Hv, d), dtype=nl.float32, buffer=nl.sbuf)
    inv_bc = inv.ap(pattern=[[Hv, 1], [1, Hv], [0, d]], offset=0)
    nisa.tensor_tensor(dst=out, data1=o_3d, data2=inv_bc, op=nl.multiply)

    # Gate by the pre-folded gamma*silu(z).
    out_flat = out.ap(pattern=[[W, 1], [1, W]], offset=0)

    gated = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gated, data1=out_flat, data2=gsz_row, op=nl.multiply)
    return gated


@nki.jit
def deltanet_gated_rmsnorm(attn_raw, gamma, z, eps):
    """HBM test harness: attn_raw/z (T, W) head-major, gamma (d,) -> (T, W). Launch [n]."""
    T, W_full = attn_raw.shape
    d = gamma.shape[0]
    Hv_full = W_full // d

    out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)

    # Value-head shard: this core owns a disjoint column slice of [T, W_full].
    n = nl.num_programs(0)
    c = nl.program_id(0)
    kernel_assert(Hv_full % n == 0, "v-heads must divide across cores")
    Hv = Hv_full // n
    W = Hv * d
    col_off = c * W

    # Gather this core's z columns for the block, then fold gamma in off the per-token path.
    gsz_all = nl.ndarray((1, T * W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gsz_all.ap(pattern=[[T * W, 1], [W, T], [1, W]], offset=0),
        src=z.ap(pattern=[[1, 1], [W_full, T], [1, W]], offset=col_off),
    )
    fold_gamma_silu(gsz_all, gamma, T, W, d)

    for t in nl.static_range(T):
        o_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=o_row[0:1, 0:W],
            src=attn_raw.ap(pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off),
        )
        gated = norm_gate_row(o_row, gsz_all[0:1, t * W : (t + 1) * W], eps, d)
        nisa.dma_copy(
            dst=out.ap(pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off),
            src=gated[0:1, 0:W],
        )
    return out
