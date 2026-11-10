"""Residual movement and cross-rank/cross-core reductions shared by the Qwen3.6 megakernels.

Every megakernel keeps its residual SBUF-resident in the tp2013 shard-interleaved [H0, T*H1] tile and
reduces each stage's per-rank partial back into it. The helpers here are that seam: the HBM<->SBUF
residual transfers, the two partial-gather shapes (attention H-shard, MoE free block), and the
vocab-parallel argmax fold.

``rg=None`` skips every TP collective (identity at TP=1) so the LNC gathers still run in a
single-process launch, where collectives comms are uninitialized and fail at NEFF load.
"""

import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.tensor_view import TensorView


def load_residual_to_sbuf(dst_sb, src_hbm, T, H0, H1, n_prgs):
    """[B, S, H] HBM -> [H0, T*H1] SBUF, tp2013 shard-interleaved (free = t*H1 + shard*H1_shard + h2)."""
    src_view = TensorView(src_hbm.reshape((T, H0 * H1))).rearrange(
        ("bs", ("lnc", "h0", "h1")),
        ("h0", "bs", "lnc", "h1"),
        {"lnc": n_prgs, "h0": H0},
    )
    dst = dst_sb.reshape((H0, T, n_prgs, H1 // n_prgs))
    for lnc in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src_view.slice(dim=2, start=lnc, end=lnc + 1).get_view(),
            dst=dst[:, :, lnc : lnc + 1, :],
        )


def store_residual_to_hbm(dst_hbm, src_sb, T, H0, H1, n_prgs):
    """Inverse of load_residual_to_sbuf: [H0, T*H1] SBUF -> [B, S, H] HBM."""
    src = src_sb.reshape((H0, T, n_prgs, H1 // n_prgs))
    dst_view = TensorView(dst_hbm.reshape((T, H0 * H1))).rearrange(
        ("bs", ("lnc", "h0", "h1")),
        ("h0", "bs", "lnc", "h1"),
        {"lnc": n_prgs, "h0": H0},
    )
    for lnc in nl.static_range(n_prgs):
        nisa.dma_copy(
            src=src[:, :, lnc : lnc + 1, :],
            dst=dst_view.slice(dim=2, start=lnc, end=lnc + 1).get_view(),
        )


def all_reduce_gather_h(sharded_sb, rg, prg_id, n_prgs, T):
    """Attention path: TP all-reduce the [H0, H1_shard*T] H-shard, LNC-gather to full [H0, T*H1]."""
    dtype = sharded_sb.dtype
    H0 = sharded_sb.shape[0]
    H1_shard = sharded_sb.shape[1] // T
    H1 = H1_shard * n_prgs
    if rg is None:
        reduced = sharded_sb
    else:
        reduced = nl.ndarray(sharded_sb.shape, dtype=dtype, buffer=nl.sbuf)
        nccl.all_reduce(dsts=[reduced], srcs=[sharded_sb], op=nl.add, replica_group=rg)

    gathered = nl.ndarray((H0, H1 * T), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gathered[:, nl.ds(start=prg_id * T * H1_shard, size=T * H1_shard)],
        src=reduced,
    )
    if n_prgs > 1:
        other = 1 - prg_id
        nisa.sendrecv(
            src=reduced,
            dst=gathered[:, nl.ds(start=other * T * H1_shard, size=T * H1_shard)],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )

    out = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    src_view = TensorView(gathered).rearrange(
        ("h0", ("h1", "bs")), ("h0", "bs", "h1"), {"h1": H1}
    )
    nisa.tensor_copy(dst=out.reshape((H0, T, H1)), src=src_view.get_view())
    return out


def all_reduce_gather_free_block(tile, f_offset, f_len, rg, prg_id, n_prgs):
    """MoE path: TP all-reduce this core's contiguous free block, LNC-gather the peer's -> full tile.

    Whichever way the MoE splits its work, exactly one contiguous free block of the [H0, F] output is
    valid on this core: the token block ``[T_offset*H1, ...)`` when token-sharded, the H-shard block
    ``[s*H2, ...)`` when H-sharded (T == 1), the whole tile on a single core. The two cores' blocks
    partition the free axis with core 0's first, which fixes the peer's block from this core's.

    Args:
        tile:     [H0, F] SBUF per-rank partial; only ``[f_offset, f_offset+f_len)`` is valid here.
        f_offset: start of this core's valid free block.
        f_len:    width of this core's valid free block.
        rg:       nccl replica group, or None to skip the TP all-reduce.

    Returns:
        out: [H0, F] SBUF with every free index reduced and gathered.
    """
    dtype = tile.dtype
    H0, F = tile.shape
    # A free-slice of tile inherits its partition stride, and nccl SBUF all_reduce needs a densely
    # packed operand, so copy the block into a fresh [H0, f_len] tile first.
    block_in = nl.ndarray((H0, f_len), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=block_in, src=tile[:, f_offset : f_offset + f_len])
    if rg is None:
        reduced = block_in
    else:
        reduced = nl.ndarray((H0, f_len), dtype=dtype, buffer=nl.sbuf)
        nccl.all_reduce(dsts=[reduced], srcs=[block_in], op=nl.add, replica_group=rg)

    out = nl.ndarray((H0, F), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[:, f_offset : f_offset + f_len], src=reduced)
    if n_prgs > 1:
        other = 1 - prg_id
        other_off = 0 if prg_id == 1 else f_offset + f_len
        other_len = F - f_len
        nisa.sendrecv(
            src=reduced,
            dst=out[:, other_off : other_off + other_len],
            send_to_rank=other,
            recv_from_rank=other,
            pipe_id=0,
        )
    return out


def all_reduce_gather_tokens(local_sb, rg, prg_id, n_prgs, T_offset, T_len):
    """Token-sharded MoE: the free-block gather over this core's token block of [H0, T, H1].

    Token-sharded MoE lays tokens on the free axis (f = t*H1 + h1), so a token block is exactly the
    contiguous free block ``[T_offset*H1, (T_offset+T_len)*H1)``.
    """
    H0, T, H1 = local_sb.shape
    return all_reduce_gather_free_block(
        local_sb.reshape((H0, T * H1)),
        T_offset * H1,
        T_len * H1,
        rg,
        prg_id,
        n_prgs,
    )


def all_gather_argmax(rank_max, rank_idx, rg, tp_degree, V_rank):
    """LM-head path: fold the per-rank (max, index) winners into one global greedy token id.

    The vocab is TP-sharded, so a rank-local winner is not yet the global one. Rather than gather
    [T, V_global] logits, gather only each rank's [T, 1] winner pair -- the reduction NxD's
    ``nxd_argmax`` does on the host. Gathering rather than reducing lands each rank's entry at a
    known column, so the vocab offset is a compile-time constant and no dynamic rank id is needed.

    Ties resolve to the lowest global id, matching torch.argmax: columns that do not hold the peak
    are lifted past every real id, then the row is min-reduced. The rank owning the peak always
    survives, so the reduce is never over an all-loser row.
    """
    if rg is None:
        return rank_idx

    T = rank_max.shape[0]
    # Above any real global vocab id and exact in fp32, which holds ids up to 2**24.
    loser = float(1 << 30)

    val = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=val, src=rank_max)
    nisa.tensor_copy(dst=idx, src=rank_idx)

    all_val = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    all_idx = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    nccl.all_gather(srcs=[val], dsts=[all_val], replica_group=rg, collective_dim=1)
    nccl.all_gather(srcs=[idx], dsts=[all_idx], replica_group=rg, collective_dim=1)

    peak = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=peak, op=nl.maximum, data=all_val, axis=1)

    cand = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    penalty = nl.ndarray((T, tp_degree), dtype=nl.float32, buffer=nl.sbuf)
    for r in nl.static_range(tp_degree):
        nisa.tensor_scalar(
            dst=cand[:, r : r + 1],
            data=all_idx[:, r : r + 1],
            op0=nl.add,
            operand0=float(r * V_rank),
        )
    nisa.tensor_tensor(
        dst=penalty,
        data1=all_val,
        data2=peak.ap([[1, T], [0, tp_degree]]),
        op=nl.equal,
    )
    # (1 - owns_peak) * loser, fused into one instruction.
    nisa.tensor_scalar(
        dst=penalty,
        data=penalty,
        op0=nl.multiply,
        operand0=-loser,
        op1=nl.add,
        operand1=loser,
    )
    nisa.tensor_tensor(dst=cand, data1=cand, data2=penalty, op=nl.add)

    best = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    tokens = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=best, op=nl.minimum, data=cand, axis=1)
    nisa.tensor_copy(dst=tokens, src=best)
    return tokens
