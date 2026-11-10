# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-NKI selective routed-experts loop for the Qwen3.6-A3B MoE decode kernel (SBUF-resident).

A self-contained expert loop in nki / nki.isa / nki.language, replacing the nkilib moe_tkg selective
path. Owning the loop is what makes the weight load contiguous: each selected expert's gate+up is
loaded as ONE [H0, H1, 2I] slab, then sliced into gate and up on the SBUF free axis. The slab load is
split per H-shard so the first shard's matmul can start before the second lands, and a 2-slot
prefetch ring issues expert k+1's weight DMAs while expert k computes.

Reproduces the moe_tkg selective contract exactly: top-K experts per token, SiLU activation,
POST_SCALE by the already-L1-normalized affinity with no re-normalization, summed over the K experts.
The stored [E,H,2,I] gate/up and [E,I,H] down weight layouts are unchanged.

Layout: consumes and emits the rmsnorm_tkg [H0,T,H1] tile in the tp2013 H-permutation shared with the
attention kernels' SBUF residual; at one H-shard it degenerates to tp102. Only the AP index math into
the (unmoved) HBM weights depends on it. Under an H-shard gate/up contract half of H, so one fused
[I,2] fp32 sendrecv+add reduces them together before the SiLU; down needs no reduce.

Per-rank (TP=4) A3B routed dims: H=2048 (H0=128, H1=16), E=256, K=8, I=128 (single I tile).
Why the slab load matters, and the DMA-fragmentation numbers: specs/moe_tkg.md.
"""

import nki.isa as nisa
import nki.language as nl

P_MAX = 128
PREFETCH_SLOTS = 2  # double-buffer: expert k+1 loads while expert k computes


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def expert_scalar_offset(expert_index, global_token_idx, k, K):
    """Dynamic scalar offset selecting expert_index[global_token_idx, k] for indirect DMA."""
    return expert_index.ap(pattern=[[K, 1], [1, 1]], offset=global_token_idx * K + k)


def load_expert_weight_slab(
    gate_up_flat,
    down_w,
    slab_slot,
    down_slot,
    expert_offset,
    H0,
    H2,
    s_offset,
    s_count,
    two_I,
    I0,
    H,
):
    """Load one selected expert into a ring slot: fused gate+up slab (one DMA per H-shard) + down weight.

    gate/up: ONE contiguous ``[H0, H1_local, 2I]`` slab per expert (inner run 2I, vs moe_tkg's I), issued
    as one DMA per owned H-shard ``s`` so the matmul over the first shard can begin before the next lands.
    dim1 step (2I) equals the inner extent, so each partition's H2 rows coalesce into a single ``H2 * 2I``
    run. down: ``[I0, H_local]`` -- the contiguous H-column block this core owns (inner run H_local, one
    ``H_local``-element run per partition; the full H row when not H-sharded). All three are indirect DMAs
    (dynamic expert via ``expert_offset``), so dge_mode is unknown (HW/SW DGE are unsafe for scalar_offset).

    Slab convention: slab[h0, s_local*H2 + h2, c] = gate_up_flat[e][(s_offset+s_local)*H0*H2 + h0*H2 + h2,
    c] (c<I gate, c>=I up), matching the normed tile's tp2013 H-permutation so each free-index matmul
    contracts H0 with aligned operands.
    """
    H_local = s_count * H0 * H2
    for s_local in range(s_count):
        nisa.dma_copy(
            dst=slab_slot[0:H0, s_local * H2 : (s_local + 1) * H2, 0:two_I],
            src=gate_up_flat.ap(
                pattern=[[H2 * two_I, H0], [two_I, H2], [1, two_I]],
                offset=(s_offset + s_local) * H0 * H2 * two_I,
                scalar_offset=expert_offset,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.unknown,
        )
    nisa.dma_copy(
        dst=down_slot[0:I0, 0:H_local],
        src=down_w.ap(
            pattern=[[H, I0], [1, H_local]],
            offset=s_offset * H0 * H2,
            scalar_offset=expert_offset,
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.unknown,
    )


def accumulate_expert_output(
    normed_sb,
    slab_slot,
    down_slot,
    affinity_col,
    output_temp,
    local_token_idx,
    global_token_idx,
    H0,
    H1_local,
    H2,
    f_offset,
    s_offset,
    I,
    I0,
    two_I,
    io_dtype,
    shard_on_H,
    peer_rank,
    is_first,
):
    """Compute one selected expert for one token and POST_SCALE-accumulate into output_temp.

    gate_up: contract H0 over this core's H1_local free indices -> gate/up [I,1] fp32 PSUM. Under the
    H-shard those are PARTIAL sums over half of H, so one fused [I,2] fp32 sendrecv+add reduces gate and
    up together (nkilib's use_fused_gate_up_sendrecv) before SiLU(gate)*up -> [I,1].
    down: contract I0 -> [H0,1] per free index f_offset+f via a strided stationary picking the H-columns
    {s*H0*H2 + h0*H2 + h2} (s, h2 = divmod(f_offset+f, H2)), so output_temp[h0, f] lands at the tp2013
    H-column of (h0, f_offset+f). Under the H-shard those columns are disjoint per core -- no reduce.
    Scale by the L1-normalized affinity (POST_SCALE, no re-normalization) and add to the token's running
    sum (fp32 accumulator).
    """
    H_local = H1_local * H0

    gate_psum = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.psum)
    up_psum = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.psum)
    for j in range(
        H1_local
    ):  # accumulate H0-contractions over this core's free indices
        moving = normed_sb[:, global_token_idx : global_token_idx + 1, f_offset + j]
        nisa.nc_matmul(dst=gate_psum, stationary=slab_slot[:, j, 0:I], moving=moving)
        nisa.nc_matmul(dst=up_psum, stationary=slab_slot[:, j, I:two_I], moving=moving)

    if shard_on_H:
        gate_up_sb = nl.ndarray((I, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate_up_sb[:, 0:1], src=gate_psum)
        nisa.tensor_copy(dst=gate_up_sb[:, 1:2], src=up_psum)

        gate_up_recv = nl.ndarray((I, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(
            src=gate_up_sb,
            dst=gate_up_recv,
            send_to_rank=peer_rank,
            recv_from_rank=peer_rank,
            pipe_id=0,
        )
        nisa.tensor_tensor(
            dst=gate_up_sb, data1=gate_up_sb, data2=gate_up_recv, op=nl.add
        )
        gate_src, up_src = gate_up_sb[:, 0:1], gate_up_sb[:, 1:2]
    else:
        gate_src, up_src = gate_psum, up_psum

    gate_sb = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_sb, data=gate_src, op=nl.silu)

    up_sb = nl.ndarray((I, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=up_sb, src=up_src)

    intermediate = nl.ndarray((I, 1), dtype=io_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intermediate, data1=gate_sb, data2=up_sb, op=nl.multiply)

    down_psum = nl.ndarray((H0, H1_local), dtype=nl.float32, buffer=nl.psum)
    for f in range(
        H1_local
    ):  # strided stationary picks the tp2013 H-columns of f_offset + f
        s, h2 = (f_offset + f) // H2, (f_offset + f) % H2
        stationary = down_slot.ap(
            pattern=[[H_local, I0], [H2, H0]], offset=(s - s_offset) * H0 * H2 + h2
        )
        nisa.nc_matmul(
            dst=down_psum[0:H0, f : f + 1],
            stationary=stationary,
            moving=intermediate[0:I0, 0:1],
        )

    down_sb = nl.ndarray((H0, H1_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=down_sb, src=down_psum)
    if is_first:
        nisa.tensor_scalar(
            dst=output_temp[:, :, local_token_idx],
            data=down_sb,
            op0=nl.multiply,
            operand0=affinity_col,
        )
    else:
        nisa.tensor_scalar(
            dst=down_sb, data=down_sb, op0=nl.multiply, operand0=affinity_col
        )
        nisa.tensor_tensor(
            dst=output_temp[:, :, local_token_idx],
            data1=output_temp[:, :, local_token_idx],
            data2=down_sb,
            op=nl.add,
        )


def routed_experts_selective(
    normed_sb,
    expert_gate_up_w,
    expert_down_w,
    expert_index,
    expert_affinities_eager,
    T_offset,
    T_per_shard,
    n_s=1,
    output_in_sbuf=True,
    shard_on_H=False,
    f_offset=0,
    H1_local=None,
):
    """Selective top-K routed experts on an SBUF-resident normed tile (fused-slab load + prefetch ring).

    Args:
        normed_sb:                [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp2013 layout).
        expert_gate_up_w:         [E, H, 2, I] HBM fused gate/up expert weights (stored layout, unchanged).
        expert_down_w:            [E, I, H] HBM down expert weights (stored layout, unchanged).
        expert_index:             [T, K] uint32 SBUF top-K expert indices (from router_topk).
        expert_affinities_eager:  [T, K] fp32 SBUF L1-normalized top-K affinities in index order.
        T_offset / T_per_shard:   this core's token slice (token-shard geometry from the caller).
        n_s:                      H-shards of the tp2013 permutation (== LNC core count; 1 -> tp102).
        output_in_sbuf:           True -> routed_local SBUF [H0,T,H1]; False -> HBM [T,H] natural.
        shard_on_H / f_offset / H1_local: H-shard geometry (``moe_h_shard``); the default is the
                                  unsharded full-H tile. Under the H-shard only free indices
                                  [f_offset, f_offset+H1_local) of routed_local are written by this core.

    Returns:
        routed_local: SBUF [H0,T,H1] (output_in_sbuf=True) or HBM [T,H] (False) -- per-rank routed partial.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    E = expert_gate_up_w.shape[0]
    I = expert_gate_up_w.shape[3]
    K = expert_index.shape[1]
    io_dtype = normed_sb.dtype
    two_I = 2 * I
    I0 = I  # single I tile (I <= 128 for A3B routed)

    kernel_assert(
        I <= P_MAX, "routed_experts_nki supports I <= 128 (single intermediate tile)"
    )
    kernel_assert(expert_down_w.shape == (E, I, H), "expert_down_w must be [E, I, H]")
    kernel_assert(H1 % n_s == 0, "tp2013 needs H1 divisible by the H-shard count n_s")
    H2 = H1 // n_s
    if H1_local is None:
        H1_local = H1
    # The core's owned free-index block spans whole tp2013 H-shards: shards [s_offset, s_offset+s_count)
    # <-> the contiguous H-column block of H_local columns starting at s_offset*H0*H2.
    s_offset = f_offset // H2
    s_count = H1_local // H2
    H_local = s_count * H0 * H2
    peer_rank = 1 - s_offset

    # Flatten the stored [E,H,2,I] gate/up to [E,H,2I]: gate = cols 0:I, up = cols I:2I are contiguous.
    gate_up_flat = expert_gate_up_w.reshape((E, H, two_I))

    # fp32 accumulator over the K experts, per token: [H0, H1_local, T_per_shard].
    output_temp = nl.ndarray(
        (H0, H1_local, T_per_shard), dtype=nl.float32, buffer=nl.sbuf
    )

    # Cross-expert prefetch ring: 2 slots for the fused gate/up slab + down weight (the H-shard halves
    # both, so the ring costs half the SBUF and half the expert-weight DMA bytes per core).
    slab_slot_0 = nl.ndarray((H0, H1_local, two_I), dtype=io_dtype, buffer=nl.sbuf)
    slab_slot_1 = nl.ndarray((H0, H1_local, two_I), dtype=io_dtype, buffer=nl.sbuf)
    slab_slots = [slab_slot_0, slab_slot_1]

    down_slot_0 = nl.ndarray((I0, H_local), dtype=io_dtype, buffer=nl.sbuf)
    down_slot_1 = nl.ndarray((I0, H_local), dtype=io_dtype, buffer=nl.sbuf)
    down_slots = [down_slot_0, down_slot_1]

    # Per-partition token index (token_iota[t, h0] = t), reused to build each token's one-hot selector.
    token_iota = nl.ndarray((T, H0), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=token_iota, pattern=[[0, H0]], offset=0, channel_multiplier=1)

    for local_token_idx in range(T_per_shard):
        global_token_idx = local_token_idx + T_offset

        # Broadcast eager[global_token_idx, :] to all H0 partitions via a one-hot selector matmul:
        # selector[t, h0] = (t == global_token_idx); aff_bc[h0, k] = sum_t selector[t,h0]*eager[t,k].
        selector = nl.ndarray((T, H0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=selector,
            data=token_iota,
            op0=nl.equal,
            operand0=float(global_token_idx),
        )

        aff_psum = nl.ndarray((H0, K), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=aff_psum, stationary=selector, moving=expert_affinities_eager
        )

        aff_bc = nl.ndarray((H0, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=aff_bc, src=aff_psum)

        # Prime slot 0 with expert k=0 before the K-loop.
        load_expert_weight_slab(
            gate_up_flat,
            expert_down_w,
            slab_slots[0],
            down_slots[0],
            expert_scalar_offset(expert_index, global_token_idx, 0, K),
            H0,
            H2,
            s_offset,
            s_count,
            two_I,
            I0,
            H,
        )
        for k in range(K):
            cur = k % PREFETCH_SLOTS
            # Issue expert k+1's weight DMAs before computing k -- the DMA queue works on k+1
            # while the tensor engine consumes k.
            if k + 1 < K:
                nxt = (k + 1) % PREFETCH_SLOTS
                load_expert_weight_slab(
                    gate_up_flat,
                    expert_down_w,
                    slab_slots[nxt],
                    down_slots[nxt],
                    expert_scalar_offset(expert_index, global_token_idx, k + 1, K),
                    H0,
                    H2,
                    s_offset,
                    s_count,
                    two_I,
                    I0,
                    H,
                )
            accumulate_expert_output(
                normed_sb,
                slab_slots[cur],
                down_slots[cur],
                aff_bc[:, k : k + 1],
                output_temp,
                local_token_idx,
                global_token_idx,
                H0,
                H1_local,
                H2,
                f_offset,
                s_offset,
                I,
                I0,
                two_I,
                io_dtype,
                shard_on_H,
                peer_rank,
                is_first=(k == 0),
            )

    # Store: output_temp [H0, H1_local, T] -> routed_local [H0, T, H1] (free index f, tp2013). Each core
    # writes its own token slice / free-index block.
    routed_local = nl.ndarray((H0, T, H1), dtype=io_dtype, buffer=nl.sbuf)
    for f in range(H1_local):
        nisa.tensor_copy(
            dst=routed_local[:, T_offset : T_offset + T_per_shard, f_offset + f],
            src=output_temp[:, f, :],
        )
    if output_in_sbuf:
        return routed_local

    # HBM natural [T,H]: flat = t*H + (s*H0*H2 + h0*H2 + h2); each core writes its token slice and owned
    # H-shards, one DMA per shard (a single contiguous-per-partition DMA at n_s == 1).
    out_hbm = nl.ndarray((T, H), dtype=io_dtype, buffer=nl.shared_hbm)
    store_tile_to_hbm_th(
        out_hbm,
        routed_local,
        H0,
        H2,
        n_s,
        H,
        T_offset,
        T_per_shard,
        s_offset=s_offset,
        s_count=s_count,
    )
    return out_hbm


def store_tile_to_hbm_th(
    out_hbm, tile_sb, H0, H2, n_s, H, T_offset, T_per_shard, s_offset=0, s_count=None
):
    """Store a tp2013 [H0,T,H1] SBUF tile into natural HBM [T,H], this core's owned slice only.

    dst flat for (h0, t', h2) of shard s = (T_offset + t')*H + s*H0*H2 + h0*H2 + h2. ``s_offset/s_count``
    restrict the write to the core's H-shards (all of them by default).
    """
    if s_count is None:
        s_count = n_s
    for s in range(s_offset, s_offset + s_count):
        nisa.dma_copy(
            dst=out_hbm.ap(
                pattern=[[H2, H0], [H, T_per_shard], [1, H2]],
                offset=T_offset * H + s * H0 * H2,
            ),
            src=tile_sb[:, T_offset : T_offset + T_per_shard, s * H2 : (s + 1) * H2],
        )
