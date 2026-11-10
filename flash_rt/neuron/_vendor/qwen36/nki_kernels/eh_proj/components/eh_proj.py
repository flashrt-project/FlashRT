# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MTP draft front end for Qwen3.6-A3B: token ids + trunk hidden -> the tp2013 residual seed.

Replaces the XLA prologue of ``NeuronMTPHead.draft_step`` (two RMSNorms, a concat and the
``eh_proj`` ColumnParallelLinear). The output tile IS ``h_in``, the draft decoder layer's residual
base, so the draft megakernel starts here instead of round-tripping ``h_in`` through HBM.

    emb_nat  = all_gather_embed_h(gather_embed_rows(ids))        [T, H]
    hid_nat  = prev_hidden                                       [T, H]
    concat   = cat([rms(emb_nat, gamma_e), rms(hid_nat, gamma_h)], -1)   [T, 2H]
    h_in     = concat @ eh_w                                     [T, H_out]
    residual = tp2013(h_in)                                      [H0, T*H1]

Concat order is [embed | hidden], matching the checkpoint's mtp.fc with no column repacking. The
concat is assembled in tp2013 and fed to one qkv_tkg(NO_NORM) over the full 2H contraction, which is
layout-exact by construction. At LNC=2 the split lands on the half boundary; at LNC=1 it degenerates
to tp102, with no layout branch here.

Both norms run in natural [T, H] layout, since that is what the all-gather and the prev_hidden load
emit; the single natural->tp2013 transpose set then serves the assembled concat.
A3B per-rank config (TP=4, LNC=2): H=2048, 2H=4096, H_out=512, H0=128, T in {1, 2}.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.tensor_view import TensorView

from ...common import H0, kernel_assert
from ...embed import (
    all_gather_embed_h,
    gather_embed_rows,
    load_token_ids_to_sbuf,
    natural_to_tp2013,
)

REDUCE_CHUNK = 512  # free-axis reduce width; H is reduced in H//REDUCE_CHUNK passes


def rms_norm_natural(x_nat, gamma_nat, eps_t, out_nat):
    """Free-axis RMSNorm of a natural [T, H] tile over the full hidden H.

    y[t, :] = x[t, :] * rsqrt(mean_H(x[t, :]^2) + eps) * gamma, fp32 reduce with an IO-dtype store.

    Args:
        x_nat:     [T, H] SBUF, tokens on partition and hidden on free.
        gamma_nat: [T, H] SBUF norm weight, partition-broadcast to the T token rows.
        eps_t:     [T, 1] SBUF fp32 epsilon, memset once and shared across both norms.
        out_nat:   [T, H] SBUF, written.
    """
    T, H = x_nat.shape
    chunk = min(H, REDUCE_CHUNK)
    kernel_assert(H % chunk == 0, "H must be divisible by the free-reduce chunk width")
    n_chunk = H // chunk

    sq = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=x_nat)

    # H exceeds the free-reduce width, so reduce each chunk then reduce across chunks.
    part = nl.ndarray((T, n_chunk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=part, op=nl.add, data=sq.reshape((T, n_chunk, chunk)), axis=[2]
    )

    ss = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=part, axis=[1], keepdims=True)

    # inv = rsqrt(ss * (1/H) + eps) = 1 / sqrt(mean_H(x^2) + eps).
    inv = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, scale=1.0 / H, bias=eps_t)

    nisa.scalar_tensor_tensor(
        dst=out_nat,
        data=x_nat,
        op0=nl.multiply,
        operand0=inv,
        op1=nl.multiply,
        operand1=gamma_nat,
    )


def _broadcast_gamma(gamma, T, dtype):
    """[1, H] HBM norm weight -> [T, H] SBUF, replicated across the token rows (zero-stride load)."""
    H = gamma.shape[1]

    gamma_sb = nl.ndarray((T, H), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma.ap(pattern=[[0, T], [1, H]], offset=0))
    return gamma_sb


def eh_proj_compose(
    ids_sb,
    embed_w,
    prev_hidden,
    gamma_e,
    gamma_h,
    eh_w,
    eps=1e-6,
    rg=None,
    tp_degree=1,
    n_prgs=1,
    out_sb=None,
    name_prefix="",
):
    """Token ids + trunk hidden -> the tp2013 [H0, T*H1] draft residual seed, SBUF end to end.

    Args:
        ids_sb:      [T, 1] int32 SBUF token ids, from the fused greedy argmax or load_token_ids_to_sbuf.
        embed_w:     [V, H/TP] HBM ParallelEmbedding weight, consumed verbatim.
        prev_hidden: [B, T, H] HBM, or a [T, H] token-major SBUF tile handed across from verify.
        gamma_e:     [1, H] HBM embed_norm.weight, standard form.
        gamma_h:     [1, H] HBM hidden_norm.weight, standard form.
        eh_w:        [2H, H_out/TP] HBM eh_proj.weight transposed; rows [0, H) contract the normed
                     embedding, rows [H, 2H) the normed hidden.
        eps:         RMSNorm epsilon.
        rg:          nccl replica group, or None to skip the collectives (TP=1).
        tp_degree:   ranks in rg; sets the gathered widths H and H_out.
        n_prgs:      LNC core count, which parameterizes tp2013.
        out_sb:      optional [H0, T*H1] SBUF destination; the megakernel passes its residual.
        name_prefix: SBUF allocation-name prefix for the qkv_tkg call; must be unique per caller.

    Returns:
        out_sb [H0, T*(H_out//H0)] SBUF tp2013 residual tile, same dtype as embed_w.

    Note:
        Runs identically on both cores -- qkv_tkg splits the 2H contraction and combines internally,
        so the returned tile is the full result everywhere.
    """
    T = ids_sb.shape[0]
    H = embed_w.shape[1] * tp_degree
    H_out = eh_w.shape[1] * tp_degree
    io_dtype = embed_w.dtype
    kernel_assert(H % H0 == 0, "hidden H must be divisible by 128")
    kernel_assert(eh_w.shape[0] == 2 * H, "eh_w must be [2H, H_out/TP], contraction first")
    kernel_assert(
        (2 * H // H0) % n_prgs == 0, "concat H1 must be divisible by the LNC core count"
    )
    kernel_assert(
        (H_out // H0) % n_prgs == 0, "output H1 must be divisible by the LNC core count"
    )

    emb_local = gather_embed_rows(ids_sb, embed_w)
    emb_nat = all_gather_embed_h(emb_local, rg, tp_degree)

    if prev_hidden.buffer == nl.sbuf:
        hid_nat = prev_hidden  # read-only below, so the caller's tile is used in place
    else:
        hid_nat = nl.ndarray((T, H), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=hid_nat, src=prev_hidden.reshape((T, H)))

    eps_t = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=float(eps))

    # Each norm writes straight into its half of the concat: [embed | hidden] is eh_w's row order.
    concat_nat = nl.ndarray((T, 2 * H), dtype=io_dtype, buffer=nl.sbuf)
    rms_norm_natural(
        emb_nat, _broadcast_gamma(gamma_e, T, io_dtype), eps_t, concat_nat[0:T, 0:H]
    )
    rms_norm_natural(
        hid_nat,
        _broadcast_gamma(gamma_h, T, io_dtype),
        eps_t,
        concat_nat[0:T, H : 2 * H],
    )

    concat_h1 = 2 * H // H0

    concat_sb = nl.ndarray((H0, T * concat_h1), dtype=io_dtype, buffer=nl.sbuf)
    natural_to_tp2013(concat_nat, concat_sb, n_prgs)

    sbm = create_auto_alloc_manager()
    sbm.set_name_prefix(name_prefix + "eh_proj_")
    h_in_local = qkv_tkg(
        hidden=concat_sb.reshape((H0, T, concat_h1)),
        qkv_w=eh_w,
        norm_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.BSD,
        output_in_sbuf=True,
        sbm=sbm,
    )
    h_in_nat = all_gather_embed_h(h_in_local, rg, tp_degree)

    if out_sb is None:
        out_sb = nl.ndarray(
            (H0, T * (H_out // H0)), dtype=io_dtype, buffer=nl.sbuf
        )
    natural_to_tp2013(h_in_nat, out_sb, n_prgs)
    return out_sb


@nki.jit
def eh_proj_fwd(
    input_ids, embed_w, prev_hidden, gamma_e, gamma_h, eh_w, eps=1e-6, tp_degree=1
):
    """eh_proj entrypoint (LNC launch [n_prgs]): ids + prev_hidden in HBM -> h_in [B, S, H_out] HBM.

    Isolation-mode twin of ``eh_proj_compose``: the megakernel keeps the residual in SBUF, this
    stores it through the tp2013 inverse so a host test reads natural [B, S, H_out] and the layout
    round trip is exercised end to end. The residual is replicated across cores, so core 0 owns the
    store and the others wait on it.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("eh_proj_fwd", (0, 1), 2)
    B, S = input_ids.shape
    T = B * S
    H_out = eh_w.shape[1] * tp_degree
    H1 = H_out // H0

    ids_sb = load_token_ids_to_sbuf(input_ids, T)
    residual = eh_proj_compose(
        ids_sb,
        embed_w,
        prev_hidden,
        gamma_e,
        gamma_h,
        eh_w,
        eps=eps,
        tp_degree=tp_degree,
        n_prgs=n_prgs,
    )

    hidden = nl.ndarray((B, S, H_out), dtype=embed_w.dtype, buffer=nl.shared_hbm)
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
