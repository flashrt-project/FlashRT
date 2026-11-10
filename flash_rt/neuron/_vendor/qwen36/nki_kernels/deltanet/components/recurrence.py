# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet gated delta-rule recurrence for token generation (decode + speculative verify).

State layout: all of a core's heads live in one wide SBUF tile Sp[128, W], W = Hv*128, packed as
Sp[i, h*128+j] = state_h[i, j]. The key index sits on the partitions, head and value index on free.

Sharded by value-head across n = nl.num_programs(0) cores. Core c owns Hv = Hv_full//n value-heads
and the matching Hk = Hk_full//n k/q-heads, and writes a disjoint slice of every output.

Folds in the input glue: q/k l2norm, beta/g gating, GQA head replication.

Two kernel variants:
  deltanet_tkg_fwd        -- (attn_out, final_state)       decode / commit
  deltanet_tkg_fwd_state  -- (attn_out, candidate_states)  speculative verify

candidate_states[t] is the state after block token t; on reject the host picks [accept_count - 1].

Input contract, per-token math, and implementation rationale: specs/deltanet_tkg.md.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .norm_gate import fold_gamma_silu, norm_gate_row

# Partition dimension max (NeuronCore SBUF tile width) = d.
P_MAX = 128

# Per-matmul moving free width = one PSUM bank (512 f32).
_PSUM_FMAX = 512


def div_ceil(n, d):
    """Ceil division for tile-count computation."""
    return (n + d - 1) // d


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def partition_broadcast_psum(row_1W, width, ones_row, psum):
    """Partition-broadcast a (1, width) row to a (128, width) PSUM tile (all partitions equal)."""
    for c in nl.static_range(div_ceil(width, _PSUM_FMAX)):
        c0 = c * _PSUM_FMAX
        tile_w = min(_PSUM_FMAX, width - c0)
        nisa.nc_matmul(
            dst=psum[0:P_MAX, c0 : c0 + tile_w],
            stationary=ones_row[0:1, 0:P_MAX],
            moving=row_1W[0:1, c0 : c0 + tile_w],
            accumulate=False,
        )


def reduce_pair_by_head_group(Sp, kq_t, t, T, Hk, rep, dim, pair_p):
    """Reduce the state against token t's key and query columns in one pass.

    One matmul per GQA group. Requires rep*dim <= _PSUM_FMAX.
    pair_p [2, W] PSUM is written: key read on partition 0, query read on partition 1.
    """
    grp_w = rep * dim
    for g in nl.static_range(Hk):
        nisa.nc_matmul(
            dst=pair_p[0:2, g * grp_w : (g + 1) * grp_w],
            stationary=kq_t[0:P_MAX, g * T + t, 0:2],
            moving=Sp[0:P_MAX, g * grp_w : (g + 1) * grp_w],
            accumulate=False,
        )


def _write_state(state_hbm, Sp, Hv, dim, W, base_off, head_stride):
    """Unpack the wide state tile into the per-head HBM layout as one 3D DMA.

    head_stride = dim*dim; base_off carries the value-head offset and, for the candidate stack,
    the per-token block.
    """
    nisa.dma_copy(
        dst=state_hbm.ap(
            pattern=[[dim, P_MAX], [head_stride, Hv], [1, dim]], offset=base_off
        ),
        src=Sp.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
    )


def _load_normed_qk(src, heads, T, dim, scale, src_off, x_f_in=None):
    """Load q or k (heads, T, dim), l2-norm over d, scale, and transpose. PS = heads*T.

    scale is 1/sqrt(d) for q, 1.0 for k. x_f_in is an optional [PS, dim] SBUF conv tile, used in
    place of the HBM load. Returns (x_t [dim, PS] with dim on the partitions, x_f [PS, dim]).

    The fp32 buffer is required on both paths: gen3 nc_transpose needs dst dtype == input dtype.
    """
    PS = heads * T

    x_f = nl.ndarray((PS, dim), dtype=nl.float32, buffer=nl.sbuf)
    if x_f_in != None:
        nisa.tensor_copy(dst=x_f, src=x_f_in)
    else:
        nisa.dma_copy(
            dst=x_f, src=src.ap(pattern=[[dim, PS], [1, dim]], offset=src_off)
        )

    # l2norm over the free axis d: sum of squares -> rsqrt -> scale rows.
    sq = nl.ndarray((PS, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=x_f, data2=x_f, op=nl.multiply)

    ss = nl.ndarray((PS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, data=sq, op=nl.add, axis=(1,))

    inv = nl.ndarray((PS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, bias=None, scale=1.0)
    if scale != 1.0:
        nisa.tensor_scalar(dst=inv, data=inv, op0=nl.multiply, operand0=scale)
    nisa.tensor_scalar(dst=x_f, data=x_f, op0=nl.multiply, operand0=inv)

    # Transpose so dim lands on the partitions.
    x_t = nl.ndarray((dim, PS), dtype=nl.float32, buffer=nl.sbuf)
    x_t_p = nl.ndarray((dim, PS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=x_t_p, data=x_f)
    nisa.tensor_copy(dst=x_t, src=x_t_p)

    return x_t, x_f


def gated_delta_rule_tkg(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    write_candidates,
    q_sbuf=None,
    k_sbuf=None,
    v_sbuf=None,
    Hk_full=None,
    Hv_full=None,
    z=None,
    gamma=None,
    eps=None,
    proj_sb=None,
    a_off=None,
    b_off=None,
    z_off=None,
    attn_sb_out=None,
):
    """Value-head-sharded gated delta-rule recurrence with the input glue folded in.

    Core c of n computes its Hv = Hv_full//n value-heads from the full HBM tensors.
    write_candidates: state after every token, else the final state only.

    Optional kwargs, each swapping an HBM read for an on-chip one:
        q_sbuf/k_sbuf/v_sbuf        silu'd conv tiles; also needs Hk_full/Hv_full
        proj_sb + a_off/b_off/z_off in_proj output in SBUF, source for a/b/z
        z/gamma/eps                 gated per-head RMSNorm at the output seam
        attn_sb_out                 collect gated rows in SBUF, skip the attn_out write
    """
    from_sbuf = q_sbuf != None
    gate_from_proj = proj_sb != None
    collect_sbuf = attn_sb_out != None
    if from_sbuf:
        T = attn_out.shape[0]
        dim = attn_out.shape[1] // Hv_full
    else:
        Hk_full, T, dim = q.shape
        Hv_full = v.shape[0]
    inv_sqrt_d = 1.0 / (dim**0.5)

    # ---- Value-head shard: local counts and this core's HBM offsets ----
    n = nl.num_programs(0)
    c = nl.program_id(0)
    kernel_assert(Hv_full % n == 0, "v-heads must divide across cores")
    kernel_assert(Hk_full % n == 0, "q/k-heads must divide across cores")

    Hk = Hk_full // n
    Hv = Hv_full // n
    rep = Hv // Hk
    W = Hv * dim
    kernel_assert(
        Hv % rep == 0, "whole GQA groups per core (Hv_loc must be a multiple of rep)"
    )

    W_full = Hv_full * dim
    kv_off = c * Hk
    vh_off = c * Hv
    col_off = c * W

    PS = Hk * T
    kernel_assert(
        PS <= P_MAX,
        f"Hk_loc*T={PS} exceeds 128 (nc_transpose output partitions) -- tile the token axis",
    )
    kernel_assert(
        rep * dim <= _PSUM_FMAX,
        f"rep*dim={rep * dim} exceeds one PSUM bank ({_PSUM_FMAX} f32)",
    )

    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    qs_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    delta_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
    O_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)

    # ---- Gating tables, hoisted off the per-token path ----
    # g_{t,h} = -exp(A_log_h) * softplus(a_{t,h} + dt_bias_h), then exp(g).
    # All tables are flat (1, T*Hv) rows on partition 0, col t*Hv + h.
    TH = T * Hv

    a_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    b_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    if gate_from_proj:
        for t in range(T):
            nisa.dma_copy(
                dst=a_sb[0:1, t * Hv : (t + 1) * Hv],
                src=proj_sb[t : t + 1, a_off + vh_off : a_off + vh_off + Hv],
            )
            nisa.dma_copy(
                dst=b_sb[0:1, t * Hv : (t + 1) * Hv],
                src=proj_sb[t : t + 1, b_off + vh_off : b_off + vh_off + Hv],
            )
    else:
        nisa.dma_copy(
            dst=a_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
            src=a.ap(pattern=[[1, 1], [Hv_full, T], [1, Hv]], offset=vh_off),
        )
        nisa.dma_copy(
            dst=b_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
            src=b.ap(pattern=[[1, 1], [Hv_full, T], [1, Hv]], offset=vh_off),
        )

    # dt_bias / exp(A_log) as (1, Hv) rows, free-broadcast across t (stride 0 over t).
    dtb = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dtb, src=dt_bias.ap(pattern=[[Hv, 1], [1, Hv]], offset=vh_off))

    Alog = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Alog, src=A_log.ap(pattern=[[Hv, 1], [1, Hv]], offset=vh_off))

    expA = nl.ndarray((1, Hv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expA, op=nl.exp, data=Alog, bias=None, scale=1.0)

    dtb_bc = dtb.ap(pattern=[[Hv, 1], [0, T], [1, Hv]], offset=0)
    expA_bc = expA.ap(pattern=[[Hv, 1], [0, T], [1, Hv]], offset=0)

    # softplus(a + dt_bias)
    sp = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=sp.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data1=a_sb.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data2=dtb_bc,
        op=nl.add,
    )
    nisa.activation(dst=sp, op=nl.softplus, data=sp, bias=None, scale=1.0)

    # exp_g_all[0, t*Hv+h] = exp(g_{t,h})
    exp_g_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=exp_g_all.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data1=sp.ap(pattern=[[TH, 1], [Hv, T], [1, Hv]], offset=0),
        data2=expA_bc,
        op=nl.multiply,
    )
    nisa.activation(dst=exp_g_all, op=nl.exp, data=exp_g_all, bias=None, scale=-1.0)

    # beta = sigmoid(b)
    beta_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_all, op=nl.sigmoid, data=b_sb, bias=None, scale=1.0)

    # ---- Hoisted q/k: l2norm over d, scale q, transpose so dim lands on the partitions ----
    if from_sbuf:
        k_t, k_f = _load_normed_qk(None, Hk, T, dim, 1.0, 0, x_f_in=k_sbuf)
        q_t, _ = _load_normed_qk(None, Hk, T, dim, inv_sqrt_d, 0, x_f_in=q_sbuf)
    else:
        qk_off = kv_off * T * dim
        k_t, k_f = _load_normed_qk(k, Hk, T, dim, 1.0, qk_off)
        q_t, _ = _load_normed_qk(q, Hk, T, dim, inv_sqrt_d, qk_off)

    # Key and query interleaved into one [dim, Hk*T, 2] stationary tile, so one pass over the state
    # yields both per-token reads. Key takes PSUM partition 0.
    kq_t = nl.ndarray((dim, PS, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kq_t[0:dim, 0:PS, 0], src=k_t[0:dim, 0:PS])
    nisa.tensor_copy(dst=kq_t[0:dim, 0:PS, 1], src=q_t[0:dim, 0:PS])

    # a_{t,h} = beta_{t,h} * (q_t . k_t), one 1x1 matmul per (GQA group, token).
    qk_p = nl.ndarray((1, PS), dtype=nl.float32, buffer=nl.psum)
    for g in nl.static_range(Hk):
        for t in nl.static_range(T):
            nisa.nc_matmul(
                dst=qk_p[0:1, g * T + t : g * T + t + 1],
                stationary=q_t[0:P_MAX, g * T + t : g * T + t + 1],
                moving=k_t[0:P_MAX, g * T + t : g * T + t + 1],
                accumulate=False,
            )

    # Expand the per-group (q.k) over rep to per value-head.
    a_all = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=a_all.ap(pattern=[[TH, 1], [Hv, T], [rep, Hk], [1, rep]], offset=0),
        data1=beta_all.ap(pattern=[[TH, 1], [Hv, T], [rep, Hk], [1, rep]], offset=0),
        data2=qk_p.ap(pattern=[[PS, 1], [1, T], [T, Hk], [0, rep]], offset=0),
        op=nl.multiply,
    )

    # ---- Optional output gate: build the whole block's gamma*silu(z) ----
    apply_norm = gamma != None
    if apply_norm:
        gsz_all = nl.ndarray((1, T * W), dtype=nl.float32, buffer=nl.sbuf)
        if gate_from_proj:
            for t in range(T):
                nisa.dma_copy(
                    dst=gsz_all[0:1, t * W : (t + 1) * W],
                    src=proj_sb[t : t + 1, z_off + col_off : z_off + col_off + W],
                )
        else:
            nisa.dma_copy(
                dst=gsz_all.ap(pattern=[[T * W, 1], [W, T], [1, W]], offset=0),
                src=z.ap(pattern=[[1, 1], [W_full, T], [1, W]], offset=col_off),
            )
        fold_gamma_silu(gsz_all, gamma, T, W, dim)

    # ---- Per-token scratch and block-wide operands ----
    upd_p = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.psum)
    pair_p = nl.ndarray((2, W), dtype=nl.float32, buffer=nl.psum)

    # Decay scalars broadcast once for the block; the per-token operand is an AP at offset t*Hv.
    eg_p = nl.ndarray((P_MAX, TH), dtype=nl.float32, buffer=nl.psum)
    partition_broadcast_psum(exp_g_all, TH, ones_row, eg_p)

    # Each token/head's key row scaled by its beta gate -- the update's rank-1 stationary operand.
    # The GQA expansion is a stride-0 read of each group's key row.
    kb_rows = nl.ndarray((1, T * W), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(T):
        for g in range(Hk):
            nisa.dma_copy(
                dst=kb_rows[
                    0:1, (t * Hv + g * rep) * dim : (t * Hv + (g + 1) * rep) * dim
                ],
                src=k_f.ap(
                    pattern=[[dim, 1], [0, rep], [1, dim]], offset=(g * T + t) * dim
                ),
            )
    nisa.tensor_tensor(
        dst=kb_rows.ap(pattern=[[T * W, 1], [dim, TH], [1, dim]], offset=0),
        data1=kb_rows.ap(pattern=[[T * W, 1], [dim, TH], [1, dim]], offset=0),
        data2=beta_all.ap(pattern=[[TH, 1], [1, TH], [0, dim]], offset=0),
        op=nl.multiply,
    )

    # Ping-pong state tiles: token t reads one and writes the other.
    S0 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    S1 = nl.ndarray((P_MAX, W), dtype=nl.float32, buffer=nl.sbuf)
    bufs = [S0, S1]

    # S0[i, h*dim+j] <- init_state[vh_off+h, i, j]
    nisa.dma_copy(
        dst=S0.ap(pattern=[[W, P_MAX], [dim, Hv], [1, dim]], offset=0),
        src=init_state.ap(
            pattern=[[dim, P_MAX], [dim * dim, Hv], [1, dim]], offset=vh_off * dim * dim
        ),
    )  # 512 B DMA

    for t in nl.static_range(T):
        src = bufs[t % 2]
        Sp = bufs[(t + 1) % 2]

        # ---- Load v for token t: v_row[0, h*dim+j] = value[h, t, j] ----
        v_row = nl.ndarray((1, W), dtype=nl.float32, buffer=nl.sbuf)
        if from_sbuf:
            # Per-head: the conv tile's head partitions are token-interleaved (stride T), which one
            # multi-partition SBUF AP cannot express.
            for h in range(Hv):
                nisa.dma_copy(
                    dst=v_row[0:1, h * dim : (h + 1) * dim],
                    src=v_sbuf[h * T + t : h * T + t + 1, 0:dim],
                )  # 4 B DMA packets
        else:
            nisa.dma_copy(
                dst=v_row.ap(pattern=[[W, 1], [dim, Hv], [1, dim]], offset=0),
                src=v.ap(
                    pattern=[[W, 1], [T * dim, Hv], [1, dim]],
                    offset=vh_off * T * dim + t * dim,
                ),
            )

        # ---- Step 1: decay -- Sp = src * exp(g) ----
        eg_view = eg_p.ap(pattern=[[TH, P_MAX], [1, Hv], [0, dim]], offset=t * Hv)
        nisa.tensor_tensor(dst=Sp, data1=src, data2=eg_view, op=nl.multiply)

        # ---- Step 2: read -- kv = k^T Sp (PSUM partition 0), q^T Sp (partition 1) ----
        reduce_pair_by_head_group(Sp, kq_t, t, T, Hk, rep, dim, pair_p)

        # ---- Step 3: delta = v - kv  (*beta is folded into the update's key rows) ----
        nisa.tensor_tensor(
            dst=delta_row, data1=v_row, data2=pair_p[0:1, 0:W], op=nl.subtract
        )

        # ---- Step 4: update -- Sp[i, h*dim+j] += (k_h[i]*beta_h) * delta[h*dim+j] ----
        for h in nl.static_range(Hv):
            nisa.nc_matmul(
                dst=upd_p[0:P_MAX, h * dim : (h + 1) * dim],
                stationary=kb_rows[0:1, (t * Hv + h) * dim : (t * Hv + h + 1) * dim],
                moving=delta_row[0:1, h * dim : (h + 1) * dim],
                accumulate=False,
            )
        nisa.tensor_tensor(dst=Sp, data1=Sp, data2=upd_p[0:P_MAX, 0:W], op=nl.add)

        # ---- Step 5: output -- O = q^T Sp_post = q^T Sp + a_{t,h} * delta ----
        # Shuffle the query read onto partition 0.
        nisa.nc_stream_shuffle(
            dst=qs_row[0:1, 0:W], src=pair_p[0:2, 0:W], shuffle_mask=[1] * 32
        )
        a_view = a_all.ap(pattern=[[TH, 1], [1, Hv], [0, dim]], offset=t * Hv)
        nisa.tensor_tensor(dst=O_row, data1=delta_row, data2=a_view, op=nl.multiply)
        nisa.tensor_tensor(dst=O_row, data1=O_row, data2=qs_row, op=nl.add)

        # ---- Optional gated per-head RMSNorm ----
        if apply_norm:
            out_row = norm_gate_row(O_row, gsz_all[0:1, t * W : (t + 1) * W], eps, dim)
        else:
            out_row = O_row

        # ---- Store the token's output ----
        if collect_sbuf:
            # Keep the row SBUF-resident for the output projection (the DMA places it on partition t).
            nisa.dma_copy(dst=attn_sb_out[t : t + 1, 0:W], src=out_row[0:1, 0:W])
        else:
            nisa.dma_copy(
                dst=attn_out.ap(
                    pattern=[[W_full, 1], [1, W]], offset=t * W_full + col_off
                ),
                src=out_row[0:1, 0:W],
            )

        # Candidate state after token t: candidate_states[t, vh_off+h, i, j] <- Sp.
        if write_candidates:
            _write_state(
                state_hbm,
                Sp,
                Hv,
                dim,
                W,
                base_off=t * Hv_full * dim * dim + vh_off * dim * dim,
                head_stride=dim * dim,
            )

    if not write_candidates:
        # Iteration t=T-1 wrote bufs[T % 2].
        final_buf = bufs[T % 2]
        _write_state(
            state_hbm,
            final_buf,
            Hv,
            dim,
            W,
            base_off=vh_off * dim * dim,
            head_stride=dim * dim,
        )


@nki.jit
def deltanet_tkg_fwd(q, k, v, a, b, A_log, dt_bias, init_state):
    """Decode / commit: raw recurrence output and the final post-block state.

    Allocates FULL-shape outputs; under an LNC=n launch each core fills its disjoint head/column
    slice. Launch ``deltanet_tkg_fwd[n](...)``.

    Returns:
        attn_out:    (T, Hv*128) f32, raw head-major output (caller RMSNorms/z-gates).
        final_state: (Hv, 128, 128) f32, state after the last block token.
    """
    T, dim = q.shape[1], q.shape[2]
    Hv_full = v.shape[0]
    W_full = Hv_full * dim

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        write_candidates=False,
    )
    return attn_out, final_state


@nki.jit
def deltanet_tkg_fwd_state(q, k, v, a, b, A_log, dt_bias, init_state):
    """Speculative verify: raw recurrence output and the per-position candidate states.

    Allocates FULL-shape outputs; under an LNC=n launch each core fills its disjoint head/column
    slice. Launch ``deltanet_tkg_fwd_state[n](...)``.

    Returns:
        attn_out:         (T, Hv*128) f32, raw head-major output (caller RMSNorms/z-gates).
        candidate_states: (T, Hv, 128, 128) f32, state after each token (axis 0 = accept axis).
    """
    T, dim = q.shape[1], q.shape[2]
    Hv_full = v.shape[0]
    W_full = Hv_full * dim

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, dim, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    gated_delta_rule_tkg(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        write_candidates=True,
    )
    return attn_out, candidate_states
