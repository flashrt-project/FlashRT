# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding for Qwen3.6-A3B: token ids in SBUF -> the tp2013 residual tile.

Replaces the XLA ``ParallelEmbedding`` + ``load_residual_to_sbuf`` pair at the front of the verify
megakernel. Ids are consumed as a ``[T, 1]`` int32 SBUF tile -- the contract the fused greedy argmax
already emits -- so a draft step can seed the trunk without a round trip through HBM or XLA.

    emb_local = gather_embed_rows(ids_sb, embed_w)      [T, H_rank]  one indirect DMA
    emb_full  = all_gather_embed_h(emb_local, rg)       [T, H]       TP concat on H
    residual  = natural_to_tp2013(emb_full)             [H0, T*H1]   H1 transposes

ParallelEmbedding is built with shard_across_embedding=True, so the per-rank table is [V, H/TP] and
a lookup is a pure row gather -- no vocab-range mask, no per-rank ownership test. Rebuilding full H
is therefore an all-gather, not a reduce: ranks hold disjoint column blocks, and collective_dim=1
lands rank r at its own block with no trace-time rank id.

Computes:
    residual[h0, t*H1 + s*H2 + h2] = embed_w_global[ids[t], s*(H0*H2) + h0*H2 + h2]

A3B per-rank config (TP=4, LNC=2): V=248320, H=2048, H_rank=512, H0=128, H1=16, H2=8, T in {1,2}.
"""

import nki
import nki.collectives as nccl
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView

from ...common import H0, kernel_assert


def load_token_ids_to_sbuf(ids_hbm, T, ids_sb=None):
    """[B, S] int32 HBM -> [T, 1] int32 SBUF: the host-ids adapter.

    Only the first step of a round needs this; once the greedy argmax is fused, its ``[T, 1]`` int32
    output feeds ``embed_compose`` directly.
    """
    if ids_sb is None:
        ids_sb = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids_sb, src=ids_hbm.reshape((T, 1)))
    return ids_sb


def gather_embed_rows(ids_sb, embed_w, emb_local=None):
    """emb_local[t, :] = embed_w[ids_sb[t, 0], :] -- one indirect DMA, one row index per partition.

    At ``indirect_dim=0`` the address is ``ids[p] * stride(dim0) + f``, so the pattern's dim-0 step
    is unused and only its extent (the partition count) matters. ``vector_offset`` rejects HWDGE;
    ``unknown`` lets the compiler settle on SWDGE. ``oob_mode`` stays at the default error -- an id
    outside [0, V) is a bug here, not a sentinel.
    """
    T = ids_sb.shape[0]
    H_rank = embed_w.shape[1]
    if emb_local is None:
        emb_local = nl.ndarray((T, H_rank), dtype=embed_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=emb_local,
        src=embed_w.ap(
            pattern=[[H_rank, T], [1, H_rank]],
            vector_offset=ids_sb.ap(pattern=[[1, T], [1, 1]]),
            indirect_dim=0,
        ),
        dge_mode=nisa.dge_mode.unknown,
    )
    return emb_local


def all_gather_embed_h(emb_local, rg, tp_degree, emb_full=None):
    """[T, H_rank] per-rank slices -> [T, H] full hidden. Disjoint slices, so gather, never reduce.

    ``rg=None`` is the identity (TP=1 launches, where collectives comms are uninitialized and fail
    at NEFF load), mirroring the megakernel's own collective helpers.
    """
    if rg is None:
        return emb_local
    T, H_rank = emb_local.shape
    if emb_full is None:
        emb_full = nl.ndarray(
            (T, H_rank * tp_degree), dtype=emb_local.dtype, buffer=nl.sbuf
        )
    nccl.all_gather(
        srcs=[emb_local], dsts=[emb_full], replica_group=rg, collective_dim=1
    )
    return emb_full


def natural_to_tp2013(emb_full, dst_sb, n_prgs):
    """[T, H] SBUF -> [H0, T*H1] SBUF, tp2013 shard-interleaved (free = t*H1 + s*H2 + h2).

    An SBUF->SBUF DMA cannot do this: SBUF access patterns must read the partition axis
    contiguously, so free elements cannot be remapped onto partitions -- which is why
    ``load_residual_to_sbuf``'s HBM-source ``TensorView.rearrange`` does not transfer here. One
    nc_transpose per free index instead, walking the tp2013 H-stride on the source AP.
    """
    T, H = emb_full.shape
    H1 = H // H0
    H2 = H1 // n_prgs
    dst4 = dst_sb.reshape((H0, T, n_prgs, H2))
    # nc_transpose is a transpose-mode matmul: on gen3+ the PSUM dtype must match the input.
    tp = nl.ndarray((H0, T), dtype=emb_full.dtype, buffer=nl.psum)
    for s in range(n_prgs):
        for h2 in range(H2):
            nisa.nc_transpose(
                dst=tp[0:H0, 0:T],
                data=emb_full.ap(
                    pattern=[[H, T], [H2, H0]],
                    offset=s * (H0 * H2) + h2,
                ),
            )
            nisa.tensor_copy(dst=dst4[:, :, s, h2], src=tp[0:H0, 0:T])
    return dst_sb


def tp2013_to_natural(src_sb, T, n_prgs, out_nat=None):
    """[H0, T*H1] SBUF -> [T, H] SBUF: the inverse of ``natural_to_tp2013``.

    Same one-transpose-per-free-index shape, run the other way: the (s, h2) source block is the
    [H0, T] tp2013 slice at free stride H1, and its transpose lands on the natural columns
    s*(H0*H2) + h0*H2 + h2.

    Lets a caller hand a tp2013 residual to a token-major consumer (the MTP draft's eh_proj front
    end) without a round trip through HBM.
    """
    H1 = src_sb.shape[1] // T
    H = H0 * H1
    H2 = H1 // n_prgs
    src4 = src_sb.reshape((H0, T, n_prgs, H2))
    if out_nat is None:
        out_nat = nl.ndarray((T, H), dtype=src_sb.dtype, buffer=nl.sbuf)

    tp = nl.ndarray((T, H0), dtype=src_sb.dtype, buffer=nl.psum)
    for s in range(n_prgs):
        for h2 in range(H2):
            nisa.nc_transpose(dst=tp[0:T, 0:H0], data=src4[:, :, s, h2])
            nisa.tensor_copy(
                dst=out_nat.ap(
                    pattern=[[H, T], [H2, H0]],
                    offset=s * (H0 * H2) + h2,
                ),
                src=tp[0:T, 0:H0],
            )
    return out_nat
    return dst_sb


def embed_compose(
    ids_sb,
    embed_w,
    rg=None,
    tp_degree=1,
    n_prgs=1,
    out_sb=None,
    return_natural=False,
):
    """Token ids in SBUF -> the tp2013 [H0, T*H1] residual tile, with no HBM or XLA round trip.

    Args:
        ids_sb:    [T, 1] int32 SBUF token ids, from the fused greedy argmax or load_token_ids_to_sbuf.
        embed_w:   [V, H/TP] HBM ParallelEmbedding weight, consumed verbatim. padding_idx needs no
                   handling: F.embedding applies it to gradients only.
        rg:        nccl replica group, or None to skip the collective (TP=1).
        tp_degree: ranks in rg; sets the gathered width H = H_rank * tp_degree.
        n_prgs:    LNC core count, which parameterizes tp2013.
        out_sb:    optional [H0, T*H1] SBUF destination; the megakernel passes its residual.
        return_natural: also return the [T, H] token-major tile, an intermediate computed anyway.

    Returns:
        out_sb [H0, T*H1] SBUF, or (out_sb, emb_full [T, H]) when return_natural.

    Note:
        Runs identically on both cores -- the residual is replicated, so no H-shard and no sendrecv.
    """
    T = ids_sb.shape[0]
    H = embed_w.shape[1] * tp_degree
    kernel_assert(H % H0 == 0, "hidden H must be divisible by 128")
    kernel_assert((H // H0) % n_prgs == 0, "H1 must be divisible by the LNC core count")

    emb_local = gather_embed_rows(ids_sb, embed_w)
    emb_full = all_gather_embed_h(emb_local, rg, tp_degree)
    if out_sb is None:
        out_sb = nl.ndarray(
            (H0, T * (H // H0)), dtype=embed_w.dtype, buffer=nl.sbuf
        )
    natural_to_tp2013(emb_full, out_sb, n_prgs)
    return (out_sb, emb_full) if return_natural else out_sb


@nki.jit
def embed_fwd(input_ids, embed_w, tp_degree=1):
    """Embedding entrypoint (LNC launch [n_prgs]): ids in HBM -> hidden [B, S, H] HBM.

    Isolation-mode twin of ``embed_compose``: the megakernel keeps the result in SBUF, this stores
    it through the tp2013 inverse so a host test reads natural [B, S, H] and the layout round trip
    is exercised end to end. The residual is replicated across cores, so core 0 owns the store and
    the others wait on it -- the same gate the megakernel puts on its own residual store.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("embed_fwd", (0, 1), 2)
    B, S = input_ids.shape
    T = B * S
    H = embed_w.shape[1] * tp_degree
    H1 = H // H0

    ids_sb = load_token_ids_to_sbuf(input_ids, T)
    residual = embed_compose(ids_sb, embed_w, tp_degree=tp_degree, n_prgs=n_prgs)

    hidden = nl.ndarray((B, S, H), dtype=embed_w.dtype, buffer=nl.shared_hbm)
    src = residual.reshape((H0, T, n_prgs, H1 // n_prgs))
    dst_view = TensorView(hidden.reshape((T, H0 * H1))).rearrange(
        ("bs", ("lnc", "h0", "h1")),
        ("h0", "bs", "lnc", "h1"),
        {"lnc": n_prgs, "h0": H0},
    )
    if prg_id == 0:
        for lnc in nl.static_range(n_prgs):
            nisa.dma_copy(
                dst=dst_view.slice(dim=2, start=lnc, end=lnc + 1).get_view(),
                src=src[:, :, lnc : lnc + 1, :],
            )
    if n_prgs > 1:
        nisa.core_barrier(data=hidden, cores=(0, 1))
    return hidden
