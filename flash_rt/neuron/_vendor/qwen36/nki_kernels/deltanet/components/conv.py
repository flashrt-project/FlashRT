# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet causal depthwise conv1d for token generation (decode + speculative verify).

Sits between in_proj_qkv and the recurrence: conv -> silu -> split q/k/v -> l2norm.

Depthwise K-tap causal conv seeded by a carried K-1 wide state window, then SiLU, then split per head
with head_dim innermost. Channels sit on the partition axis (conv_dim 2048 -> 16 tiles of 128), K and
T on the free axis: pure VectorE, no matmul, no cross-partition reduce.

Per tile, with win = concat_free(conv_state, qkv):
    y[:, t]      = silu( sum_j w[:, j] * win[:, t+j] )   K-tap MAC, one weight col per partition
    conv_cand[t] = win[:, t+1 : t+1+(K-1)]               window after token t

Sharded by value-head: each core owns a contiguous slice of the q, k, and v channel segments.
Channels are independent, so the cores' writes are disjoint and tile the full output.

Three kernel variants, all returning qkv_out [NT, T, 128] (caller slices q/k/v via key_dim):
  deltanet_conv_tkg_fwd       -- + new_state [conv_dim, K-1]      decode / commit (T=1)
  deltanet_conv_tkg_fwd_cand  -- + conv_cand [T, conv_dim, K-1]   speculative verify (T>=2)
  deltanet_conv_tkg_fwd_sbuf  -- SBUF-resident output (megakernel demo)
Input contract and implementation rationale: specs/deltanet_tkg.md.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width) == head_dim.
P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def conv_load_tap_blocks(conv_state, conv_weight, NT, K, state_w, ch0):
    """Bulk-load one segment's per-channel taps and carried state window onto NT partitions.

    Both tensors are row-major, so each NT block is one contiguous run and lands as a single wide
    DMA that conv_load_compute then transposes into channel-on-partition. Returns (w_blk, cs_blk).
    """
    w_blk = nl.ndarray((NT, P_MAX * K), dtype=nl.float32, buffer=nl.sbuf)
    cs_blk = nl.ndarray((NT, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w_blk.ap(pattern=[[P_MAX * K, NT], [1, P_MAX * K]], offset=0),
        src=conv_weight.ap(pattern=[[P_MAX * K, NT], [1, P_MAX * K]], offset=ch0 * K),
    )
    nisa.dma_copy(
        dst=cs_blk.ap(pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=0),
        src=conv_state.ap(
            pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=ch0 * state_w
        ),
    )
    return w_blk, cs_blk


def conv_preload_taps(conv_state, conv_weight, key_dim):
    """Load this core's tap/state blocks for all 3 owned segments, in shard_segments order.

    Both tensors are layer-static, so a caller can issue these DMAs ahead of the conv; left to
    conv_qkv_sbuf they are issued at the point of use.
    """
    K = conv_weight.shape[1]
    segments, _, _ = shard_segments(conv_weight.shape[0], key_dim)
    preloaded = []
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]
        preloaded.append(
            conv_load_tap_blocks(conv_state, conv_weight, n_tiles, K, K - 1, t0 * P_MAX)
        )
    return preloaded


def conv_load_compute(
    qkv,
    tap_blocks,
    win,
    acc,
    prod,
    out,
    NT,
    K,
    T,
    state_w,
    ch0,
    qkv_cp_sbuf=None,
    t0=0,
):
    """Batched load + depthwise MAC + SiLU over one segment's NT channel-tiles, from channel ch0.

    win, acc, prod, out and tap_blocks are all caller-owned. Leaves win packed as
    [conv_state | qkv] per tile for candidate slicing, and writes out from partition 0 -- the
    Activation engine cannot target a partition offset, so the caller does any placement.

    qkv_cp_sbuf is an optional channel-on-partition qkv tile [128, NT_full*T] replacing the strided
    HBM load; t0 is then this segment's global start tile.
    """
    from_sbuf = qkv_cp_sbuf != None
    conv_dim = qkv.shape[1] if not from_sbuf else None
    img_w = state_w + T  # per-tile window width on the free axis
    prod_w = T * K  # per-tile (W_out * W_f) product width
    w_blk, cs_blk = tap_blocks

    # ---- Transpose the taps to channel-on-partition, one tap column at a time ----
    w_p = nl.ndarray((P_MAX, NT * K), dtype=nl.float32, buffer=nl.sbuf)
    w_tap = nl.ndarray((P_MAX, NT), dtype=nl.float32, buffer=nl.psum)
    for j in range(K):
        nisa.nc_transpose(
            dst=w_tap[0:P_MAX, 0:NT],
            data=w_blk.ap(pattern=[[P_MAX * K, NT], [K, P_MAX]], offset=j),
        )
        nisa.tensor_copy(
            dst=w_p.ap(pattern=[[NT * K, P_MAX], [K, NT]], offset=j),
            src=w_tap[0:P_MAX, 0:NT],
        )

    # ---- Build the window: the carried state fills each tile's first state_w columns ----
    cs_tap = nl.ndarray((P_MAX, NT), dtype=nl.float32, buffer=nl.psum)
    for w in range(state_w):
        nisa.nc_transpose(
            dst=cs_tap[0:P_MAX, 0:NT],
            data=cs_blk.ap(pattern=[[P_MAX * state_w, NT], [state_w, P_MAX]], offset=w),
        )
        nisa.tensor_copy(
            dst=win.ap(pattern=[[NT * img_w, P_MAX], [img_w, NT]], offset=w),
            src=cs_tap[0:P_MAX, 0:NT],
        )

    # ---- Fill the rest of the window with qkv, from HBM or on-chip ----
    if from_sbuf:
        cp_part_stride = qkv_cp_sbuf.shape[1]
        nisa.tensor_copy(
            dst=win.ap(
                pattern=[[NT * img_w, P_MAX], [img_w, NT], [1, T]], offset=state_w
            ),
            src=qkv_cp_sbuf.ap(
                pattern=[[cp_part_stride, P_MAX], [T, NT], [1, T]], offset=t0 * T
            ),
        )
    else:
        nisa.dma_copy(
            dst=win.ap(
                pattern=[[NT * img_w, P_MAX], [img_w, NT], [1, T]], offset=state_w
            ),
            src=qkv.ap(pattern=[[1, P_MAX], [P_MAX, NT], [conv_dim, T]], offset=ch0),
        )  # 132 B DMA packets

    # ---- Depthwise MAC: sliding-window multiply, then reduce the K-tap axis ----
    # One pass covers all T output columns; the filter broadcasts over them.
    for nt in nl.affine_range(NT):
        nisa.tensor_tensor(
            dst=prod.ap(pattern=[[prod_w, P_MAX], [1, prod_w]], offset=0),
            data1=win.ap(
                pattern=[[NT * img_w, P_MAX], [1, T], [1, K]], offset=nt * img_w
            ),
            data2=w_p.ap(pattern=[[NT * K, P_MAX], [0, T], [1, K]], offset=nt * K),
            op=nl.multiply,
        )
        nisa.tensor_reduce(
            dst=acc.ap(pattern=[[NT * T, P_MAX], [1, T]], offset=nt * T),
            data=prod.ap(pattern=[[prod_w, P_MAX], [K, T], [1, K]], offset=0),
            op=nl.add,
            axis=2,
        )

    # ---- SiLU fused with the head_dim transpose ----
    # The transpose moves head_dim onto the free axis so the caller's store is contiguous; the
    # activation doubles as the PSUM->SBUF cast to the I/O dtype.
    acc_t = nl.ndarray((NT * T, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=acc_t[0 : NT * T, 0:P_MAX], data=acc[0:P_MAX, 0 : NT * T])
    nisa.activation(
        dst=out[0 : NT * T, 0:P_MAX], op=nl.silu, data=acc_t[0 : NT * T, 0:P_MAX]
    )


def conv_state_store(win, conv_cand, NT, T, state_w, ch0, cand_is_3d):
    """Scatter one segment's per-token candidate windows to its global HBM channels.

    The window after token t is a bit-exact slice of win. Each column is transposed, then the whole
    block is scattered in one DMA per token. cand_is_3d selects the candidate stack vs the T=1 state.
    """
    conv_dim = conv_cand.shape[-2]

    cand_blk = nl.ndarray((NT, P_MAX * state_w), dtype=nl.float32, buffer=nl.sbuf)
    cand_tap = nl.ndarray((NT, P_MAX), dtype=nl.float32, buffer=nl.psum)
    img_w = state_w + T
    for t in range(T):
        cand_offset = (t * conv_dim * state_w if cand_is_3d else 0) + ch0 * state_w
        for w in range(state_w):
            nisa.nc_transpose(
                dst=cand_tap[0:NT, 0:P_MAX],
                data=win.ap(
                    pattern=[[NT * img_w, P_MAX], [img_w, NT]], offset=t + 1 + w
                ),
            )
            nisa.tensor_copy(
                dst=cand_blk.ap(
                    pattern=[[P_MAX * state_w, NT], [state_w, P_MAX]], offset=w
                ),
                src=cand_tap[0:NT, 0:P_MAX],
            )
        nisa.dma_copy(
            dst=conv_cand.ap(
                pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]],
                offset=cand_offset,
            ),
            src=cand_blk.ap(
                pattern=[[P_MAX * state_w, NT], [1, P_MAX * state_w]], offset=0
            ),
        )  # 4 B DMA packets


def conv_state_store_pending(pending_cand):
    """Run the per-segment candidate-window stores collected by conv_qkv_sbuf.

    These read only the conv windows, so the caller chooses when to drain them.
    """
    for seg in range(len(pending_cand)):
        win, conv_cand, NT, T, state_w, ch0, cand_is_3d = pending_cand[seg]
        conv_state_store(win, conv_cand, NT, T, state_w, ch0, cand_is_3d)


def qkv_to_channel_partition(proj_sb, conv_dim, T):
    """Transpose the projection's qkv sub-block to channel-on-partition [128, NT*T].

    One nc_transpose per 128-channel tile, feeding the conv's SBUF path.
    """
    NT = conv_dim // P_MAX
    # gen3 nc_transpose requires dst dtype == input dtype.
    qkv_cp = nl.ndarray((P_MAX, NT * T), dtype=proj_sb.dtype, buffer=nl.sbuf)
    tp = nl.ndarray((P_MAX, T), dtype=proj_sb.dtype, buffer=nl.psum)
    for gt in range(NT):
        nisa.nc_transpose(
            dst=tp[0:P_MAX, 0:T],
            data=proj_sb.ap(
                pattern=[[proj_sb.shape[1], T], [1, P_MAX]], offset=gt * P_MAX
            ),
        )
        nisa.tensor_copy(
            dst=qkv_cp[0:P_MAX, gt * T : (gt + 1) * T], src=tp[0:P_MAX, 0:T]
        )
    return qkv_cp


def conv_qkv_sbuf(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    conv_cand,
    cand_is_3d,
    qkv_cp_sbuf=None,
    preloaded=None,
):
    """Head-sharded conv into three separate q/k/v SBUF tiles, plus the pending state stores.

    Runs the MAC + SiLU for this core's 3 owned channel segments. Keeping q/k/v separate leaves
    every tile partition-0-based, which is the layout the recurrence consumes directly.

    The conv-state scatter is returned as a pending list so the caller can place it. It MUST be
    drained with conv_state_store_pending or the conv state is never written.

    Optional kwargs:
        qkv_cp_sbuf  channel-on-partition qkv tile; the conv reads it instead of the HBM load,
                     and qkv then carries only its shape
        preloaded    per-segment (w_blk, cs_blk) from conv_preload_taps; loaded here if omitted

    Returns (q_sbuf, k_sbuf, v_sbuf, pending_cand); the tiles are [H_loc*T, 128].
    """
    K = conv_weight.shape[1]
    if qkv_cp_sbuf == None:
        T, conv_dim = qkv.shape
        out_dtype = qkv.dtype
    else:
        conv_dim = conv_weight.shape[0]
        T = qkv_cp_sbuf.shape[1] // (conv_dim // P_MAX)
        out_dtype = qkv_cp_sbuf.dtype

    state_w = K - 1
    img_w = state_w + T

    segments, Hv_loc, _ = shard_segments(conv_dim, key_dim)
    Hk_loc = segments[0][1]
    kernel_assert(
        Hv_loc * T <= P_MAX, "Hv_loc*T must fit the transpose partition axis (<=128)"
    )

    q_sbuf = nl.ndarray((Hk_loc * T, P_MAX), dtype=out_dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((Hk_loc * T, P_MAX), dtype=out_dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((Hv_loc * T, P_MAX), dtype=out_dtype, buffer=nl.sbuf)
    seg_out = [q_sbuf, k_sbuf, v_sbuf]

    if preloaded is None:
        preloaded = conv_preload_taps(conv_state, conv_weight, key_dim)

    pending_cand = []
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]  # global start tile, tile count for this segment
        ch0 = t0 * P_MAX  # this segment's global channel offset

        # Per-segment scratch, sized at this segment's exact tile count.
        win = nl.ndarray((P_MAX, n_tiles * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, n_tiles * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)

        conv_load_compute(
            qkv,
            preloaded[seg],
            win,
            acc,
            prod,
            seg_out[seg],
            n_tiles,
            K,
            T,
            state_w,
            ch0,
            qkv_cp_sbuf=qkv_cp_sbuf,
            t0=t0,
        )

        pending_cand.append((win, conv_cand, n_tiles, T, state_w, ch0, cand_is_3d))

    return q_sbuf, k_sbuf, v_sbuf, pending_cand


def shard_segments(conv_dim, key_dim):
    """The 3 channel segments (q, k, v) this core owns under value-head sharding.

    Returns (segments, Hv_loc, NT_loc); each segment is (global_start_tile, n_tiles).
    n=1 yields the full contiguous q|k|v block.
    """
    Hk = key_dim // P_MAX  # q-heads = k-heads
    Hv = (conv_dim - 2 * key_dim) // P_MAX  # v-heads

    n = nl.num_programs(0)
    c = nl.program_id(0)

    kernel_assert(conv_dim % P_MAX == 0, "conv_dim must be a multiple of 128")
    kernel_assert(key_dim % P_MAX == 0, "key_dim must be a multiple of 128")
    kernel_assert(Hk % n == 0, "q/k-heads must divide across cores")
    kernel_assert(Hv % n == 0, "v-heads must divide across cores")

    Hk_loc = Hk // n
    Hv_loc = Hv // n
    NT_loc = 2 * Hk_loc + Hv_loc

    segments = [
        (c * Hk_loc, Hk_loc),  # q
        (Hk + c * Hk_loc, Hk_loc),  # k
        (2 * Hk + c * Hv_loc, Hv_loc),  # v
    ]
    return segments, Hv_loc, NT_loc


def deltanet_conv(
    qkv, conv_state, conv_weight, key_dim, qkv_out, conv_cand, cand_is_3d
):
    """Depthwise causal conv + SiLU into a unified qkv_out [NT, T, d], plus candidate windows.

    Value-head sharded: this core runs load/compute/store for each of its 3 channel segments and
    writes the owned tiles into their global positions. conv_cand is [T, conv_dim, K-1], or the
    T=1 [conv_dim, K-1] state.
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]

    state_w = K - 1
    img_w = state_w + T

    kernel_assert(conv_state.shape[1] == state_w, "conv_state width must be K-1")
    kernel_assert(qkv_out.dtype == qkv.dtype, "qkv_out dtype must match qkv")

    segments, Hv_loc, _ = shard_segments(conv_dim, key_dim)
    kernel_assert(
        Hv_loc * T <= P_MAX, "Hv_loc*T must fit the transpose partition axis (<=128)"
    )

    # Buffers are sized per segment at its exact tile count: nc_transpose requires the data AP
    # partition stride to equal the tensor free dim, so they cannot be over-sized.
    for seg in range(len(segments)):
        t0, NT = segments[seg]  # global start tile, tile count for this segment
        ch0 = t0 * P_MAX  # this segment's global channel offset

        win = nl.ndarray((P_MAX, NT * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, NT * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
        out = nl.ndarray((NT * T, P_MAX), dtype=qkv_out.dtype, buffer=nl.sbuf)

        taps = conv_load_tap_blocks(conv_state, conv_weight, NT, K, state_w, ch0)
        conv_load_compute(qkv, taps, win, acc, prod, out, NT, K, T, state_w, ch0)

        # Contiguous bulk store of this segment's tiles into qkv_out[NT,T,d] at its global rows.
        nisa.dma_copy(
            dst=qkv_out.ap(
                pattern=[[P_MAX, NT * T], [1, P_MAX]], offset=t0 * T * P_MAX
            ),
            src=out[0 : NT * T, 0:P_MAX],
        )

        # Scatter this segment's per-token candidate windows to its global HBM channels.
        conv_state_store(win, conv_cand, NT, T, state_w, ch0, cand_is_3d)


@nki.jit
def deltanet_conv_tkg_fwd(qkv, conv_state, conv_weight, key_dim):
    """Decode / commit (T=1): silu'd conv output and the committed new conv state.

    Returns qkv_out [NT, 1, 128] (not l2-normed) and new_state [conv_dim, K-1].
    Value-head sharded; launch [2].
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX

    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    new_state = nl.ndarray((conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm)
    deltanet_conv(
        qkv, conv_state, conv_weight, key_dim, qkv_out, new_state, cand_is_3d=False
    )
    return qkv_out, new_state


@nki.jit
def deltanet_conv_tkg_fwd_cand(qkv, conv_state, conv_weight, key_dim):
    """Speculative verify (T>=2): silu'd conv output and per-position candidate conv states.

    conv_cand[t] is the window after block token t; on reject the host commits [accept_count - 1].
    Value-head sharded; launch [2].
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    NT = conv_dim // P_MAX

    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    deltanet_conv(
        qkv, conv_state, conv_weight, key_dim, qkv_out, conv_cand, cand_is_3d=True
    )
    return qkv_out, conv_cand


@nki.jit
def deltanet_conv_tkg_fwd_sbuf(qkv, conv_state, conv_weight, key_dim):
    """SBUF-output variant (megakernel demo): q/k/v and conv_cand stay resident in SBUF.

    Builds the per-core out_sbuf in [q | k | v] partition order, then copies to HBM only so
    callers and tests can read it.
    """
    T, conv_dim = qkv.shape
    K = conv_weight.shape[1]

    state_w = K - 1
    img_w = state_w + T
    NT = conv_dim // P_MAX

    segments, _, NT_loc = shard_segments(conv_dim, key_dim)
    kernel_assert(
        NT_loc * T <= P_MAX, "NT_loc*T must fit the transpose partition axis (<=128)"
    )

    # out_sbuf is head_dim-on-free for contiguous readout; cand_sbuf stays channel-on-partition,
    # packed by global tile. Both persist across segments.
    out_sbuf = nl.ndarray((NT_loc * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)
    cand_sbuf = nl.ndarray((P_MAX, NT * T * state_w), dtype=qkv.dtype, buffer=nl.sbuf)

    row0 = 0
    for seg in range(len(segments)):
        t0, n_tiles = segments[seg]  # global start tile, tile count for this segment

        # Per-segment scratch, sized at this segment's exact tile count.
        win = nl.ndarray((P_MAX, n_tiles * img_w), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.ndarray((P_MAX, n_tiles * T), dtype=nl.float32, buffer=nl.sbuf)
        prod = nl.ndarray((P_MAX, T * K), dtype=nl.float32, buffer=nl.sbuf)
        out_seg = nl.ndarray((n_tiles * T, P_MAX), dtype=qkv.dtype, buffer=nl.sbuf)

        taps = conv_load_tap_blocks(
            conv_state, conv_weight, n_tiles, K, state_w, t0 * P_MAX
        )
        conv_load_compute(
            qkv, taps, win, acc, prod, out_seg, n_tiles, K, T, state_w, t0 * P_MAX
        )

        # Place this segment's rows into out_sbuf.
        # SBUF->SBUF DMA can target a partition offset; the Activation engine cannot.
        nisa.dma_copy(
            dst=out_sbuf[row0 : row0 + n_tiles * T, 0:P_MAX],
            src=out_seg[0 : n_tiles * T, 0:P_MAX],
        )

        # Candidate windows, packed by global tile.
        for nt in range(n_tiles):
            for t in range(T):
                c0 = ((t0 + nt) * T + t) * state_w
                nisa.tensor_copy(
                    dst=cand_sbuf[0:P_MAX, c0 : c0 + state_w],
                    src=win[0:P_MAX, nt * img_w + t + 1 : nt * img_w + t + 1 + state_w],
                )

        row0 += n_tiles * T

    qkv_out = nl.ndarray((NT, T, P_MAX), dtype=qkv.dtype, buffer=nl.shared_hbm)
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    nisa.dma_copy(
        dst=qkv_out.ap(pattern=[[P_MAX, NT_loc * T], [1, P_MAX]], offset=0),
        src=out_sbuf[0 : NT_loc * T, 0:P_MAX],
    )
    for t in range(T):
        nisa.dma_copy(
            dst=conv_cand.ap(
                pattern=[[state_w, P_MAX], [P_MAX * state_w, NT], [1, state_w]],
                offset=t * conv_dim * state_w,
            ),
            src=cand_sbuf.ap(
                pattern=[[NT * T * state_w, P_MAX], [T * state_w, NT], [1, state_w]],
                offset=t * state_w,
            ),
        )
    return qkv_out, conv_cand
