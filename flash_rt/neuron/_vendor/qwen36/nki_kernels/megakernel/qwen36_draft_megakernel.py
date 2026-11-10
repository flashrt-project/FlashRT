"""Qwen3.6-A3B MTP draft head: the whole draft step in one LNC2 launch.

Token ids + trunk hidden in, draft token ids + carry hidden + mutated KV cache out. The residual is
SBUF-resident from the eh_proj seed to the final store, so the only HBM traffic is the weight stream,
the KV cache and the two step outputs. Structure mirrors the verify megakernel, with one layer:

    residual  = eh_proj_compose(ids, prev_hidden)          tp2013 [H0, T*H1] seed
    residual += all_reduce_gather_h(gqa_fused_compose(...))    active K/V scattered in place
    residual += all_reduce_gather_free_block(moe_layer_compose(...))
    hidden    = residual                                   PRE-final-norm carry
    tokens    = all_gather_argmax(lm_head_compose(residual))    with_lm_head build only

The KV write is in place at kv_write_idx: the scatter runs after attention's prior read, and the
mutated cache handles are returned because NCC dead-stores a mutated buffer nothing consumes.

with_lm_head is a compile-time build key -- the replay launch's logits are dead, so its build drops
the head and the weight stream that feeds it.

Not decorated -- build_draft_megakernel wraps this with nki.jit(), since a double jit overflows
the stack.
"""

import inspect

import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nki import jit as nki_jit

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from ..eh_proj.components.eh_proj import eh_proj_compose
from ..embed.components.embed import load_token_ids_to_sbuf
from ..gqa.decode.fused_layer import gqa_fused_compose
from ..lm_head.components.lm_head import lm_head_compose
from ..moe.components.moe_layer import moe_layer_compose
from ..moe.components.shared_expert import moe_h_shard_decision, moe_tkg_shard_decision
from .collectives import (
    all_gather_argmax,
    all_reduce_gather_free_block,
    all_reduce_gather_h,
    store_residual_to_hbm,
)
from .qwen36_verify_megakernel import MOE_FIELDS

# Per-launch SBUF allocation-name prefixes: the round traces both builds into one graph.
DRAFT_PREFIX = "d1_"
REPLAY_PREFIX = "d2_"

DRAFT_GQA_FIELDS = (
    "in_gamma",
    "qkv_w",
    "gate_w",
    "gamma_q",
    "gamma_k",
    "o_proj_w",
    "k_cache",
    "v_cache",
)


def moe_output_free_block(T, H, H1, moe_intermediate):
    """This core's valid free block ``(f_offset, f_len)`` of the MoE [H0, T*H1] output tile.

    Token-sharded, the block is this core's token range (free index f = t*H1 + h1). H-sharded, it is
    this core's tp2013 free-index range -- contiguous in the flat view only because the H-shard
    fallback engages at T == 1 alone. On a single core the whole tile is valid.
    """
    shard_on_T, T_offset, T_len = moe_tkg_shard_decision(T, H, moe_intermediate)
    if shard_on_T:
        return T_offset * H1, T_len * H1
    shard_on_H, f_offset, H1_local = moe_h_shard_decision(T, H, H1, moe_intermediate)
    if shard_on_H:
        return f_offset, H1_local
    return 0, T * H1


def draft_stage_compose(
    ids_sb,
    prev_hidden,
    kv_write_idx,
    cos,
    sin,
    mask,
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    gamma_in,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    o_proj_w,
    k_cache,
    v_cache,
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    final_gamma,
    lm_head_w,
    eps,
    rg,
    tp_degree,
    n_prgs,
    name_prefix="",
):
    """The draft step as one SBUF-resident stage: ids + trunk hidden -> residual + token id.

    Everything the launch shell adds around this is HBM. A fused round calls it twice with no store
    in between, threading the mutated cache handles through so the ordering edge on the shared MTP
    KV cache is explicit.

    Args:
        ids_sb:      [T, 1] int32 SBUF token ids.
        prev_hidden: [B, T, H] HBM or [T, H] SBUF -- the trunk hidden to condition on.
        final_gamma: None selects the headless stage (no vocab head, ``token_idx`` is None).

    Returns:
        ``(residual [H0, T*H1] SBUF PRE-final-norm, token_idx [T, 1] int32 SBUF or None,
        k_cache, v_cache, active_k, active_v)``.
    """
    T = ids_sb.shape[0]
    dtype = embed_w.dtype
    H0 = nl.tile_size.pmax
    H = embed_w.shape[1] * tp_degree
    H1 = H // H0
    H1_shard = H1 // n_prgs
    _, _, prg_id = get_verified_program_sharding_info(
        "qwen36_draft_stage", (0, 1), 2
    )

    residual = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    eh_proj_compose(
        ids_sb,
        embed_w,
        prev_hidden,
        gamma_e,
        gamma_h,
        eh_w,
        eps=eps,
        rg=rg,
        tp_degree=tp_degree,
        n_prgs=n_prgs,
        out_sb=residual,
        name_prefix=name_prefix,
    )

    attn_partial, active_k, active_v, k_cache, v_cache = gqa_fused_compose(
        residual.reshape((H0, T, H1)),
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        cos,
        sin,
        k_cache,
        v_cache,
        mask,
        o_proj_w,
        eps,
        kv_write_idx=kv_write_idx,
        gamma_in=gamma_in,
        out_in_sb=True,
        name_prefix=name_prefix,
    )
    attn_out = all_reduce_gather_h(
        attn_partial.reshape((H0, H1_shard * T)), rg, prg_id, n_prgs, T
    )
    nisa.tensor_tensor(dst=residual, data1=residual, data2=attn_out, op=nl.add)

    moe_partial = moe_layer_compose(
        residual.reshape((H0, T, H1)),
        moe_gamma,
        moe_router_w,
        moe_gate_up_w,
        moe_down_w,
        moe_sigma_gate_w,
        moe_shared_gate_w,
        moe_shared_up_w,
        moe_shared_down_w,
        eps=eps,
        output_in_sbuf=True,
        name_prefix=name_prefix,
    )
    f_offset, f_len = moe_output_free_block(T, H, H1, moe_gate_up_w.shape[3])
    moe_out = all_reduce_gather_free_block(
        moe_partial.reshape((H0, T * H1)), f_offset, f_len, rg, prg_id, n_prgs
    )
    nisa.tensor_tensor(dst=residual, data1=residual, data2=moe_out, op=nl.add)

    token_idx = None
    if final_gamma is not None:
        # residual stays the pre-final-norm hidden; the head norms its own copy.
        rank_max, rank_idx, _ = lm_head_compose(
            residual.reshape((H0, T, H1)),
            final_gamma,
            lm_head_w,
            eps=eps,
            name_prefix=name_prefix + "lm_",
        )
        token_idx = all_gather_argmax(
            rank_max, rank_idx, rg, tp_degree, lm_head_w.shape[1]
        )
    return residual, token_idx, k_cache, v_cache, active_k, active_v


def qwen36_draft_megakernel(
    input_ids,
    prev_hidden,
    kv_write_idx,
    cos,
    sin,
    mask,
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    gamma_in,
    qkv_w,
    gate_w,
    gamma_q,
    gamma_k,
    o_proj_w,
    k_cache,
    v_cache,
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    final_gamma,
    lm_head_w,
    eps,
    replica_groups,
    name_prefix="",
):
    """Run the draft front end, the GQA+MoE decoder layer and optionally the vocab head.

    final_gamma/lm_head_w None selects the headless build. cos/sin are indexed at token positions
    t .. t+T-1, and mask keeps the committed prior plus the trailing active slots causally -- the
    active tokens occupy the LAST T slots of the cache tile, where the in-place scatter writes.

    Returns:
        (tokens [B,S] int32, hidden [B,S,H], k_cache, v_cache, active_k, active_v) with the head,
        else the same without tokens. hidden is pre-final-norm, since the head applies its own norm
        and the next draft step seeds from the un-normed hidden. active_k/active_v are returned
        because an unreturned shared_hbm write may be overlaid by the allocator.
    """
    B, S = input_ids.shape
    dtype = embed_w.dtype
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen36_draft_megakernel", (0, 1), 2
    )
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None
    tp_degree = len(replica_groups[0]) if replica_groups is not None else 1
    H0 = nl.tile_size.pmax
    H = embed_w.shape[1] * tp_degree
    H1 = H // H0
    T = B * S

    residual, token_idx, k_cache, v_cache, active_k, active_v = draft_stage_compose(
        load_token_ids_to_sbuf(input_ids, T),
        prev_hidden,
        kv_write_idx,
        cos,
        sin,
        mask,
        embed_w,
        gamma_e,
        gamma_h,
        eh_w,
        gamma_in,
        qkv_w,
        gate_w,
        gamma_q,
        gamma_k,
        o_proj_w,
        k_cache,
        v_cache,
        moe_gamma,
        moe_router_w,
        moe_gate_up_w,
        moe_down_w,
        moe_sigma_gate_w,
        moe_shared_gate_w,
        moe_shared_up_w,
        moe_shared_down_w,
        final_gamma,
        lm_head_w,
        eps,
        rg,
        tp_degree,
        n_prgs,
        name_prefix=name_prefix,
    )

    hidden = nl.ndarray((B, S, H), dtype=dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        store_residual_to_hbm(hidden, residual, T, H0, H1, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=hidden, cores=(0, 1))

    if final_gamma is None:
        return hidden, k_cache, v_cache, active_k, active_v

    # [T, 1] SBUF is one element per partition, so the HBM row is written through a [T, 1] view
    # of it; a flat T-wide access pattern would read one partition and run off the end.
    tokens = nl.ndarray((B, S), dtype=nl.int32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=tokens.reshape((T, 1)), src=token_idx)
    if n_prgs > 1:
        nisa.core_barrier(data=tokens, cores=(0, 1))
    return tokens, hidden, k_cache, v_cache, active_k, active_v


def flatten_draft_args(
    input_ids,
    prev_hidden,
    kv_write_idx,
    cos,
    sin,
    mask,
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    gqa,
    moe,
    eps,
    replica_groups,
    final_gamma=None,
    lm_head_w=None,
):
    """The one definition of the flat positional argument order.

    ``gqa``/``moe`` map each field name to its tensor. Used both by ``build_draft_megakernel`` (over
    parameter NAMES, to check the wrapper signature) and by the caller (over tensor VALUES), so the
    two cannot drift. ``final_gamma``/``lm_head_w`` are omitted entirely by the headless build.
    """
    flat = [input_ids, prev_hidden, kv_write_idx, cos, sin, mask]
    flat += [embed_w, gamma_e, gamma_h, eh_w]
    flat += [gqa[f] for f in DRAFT_GQA_FIELDS]
    flat += [moe[f] for f in MOE_FIELDS]
    if final_gamma is not None:
        flat += [final_gamma, lm_head_w]
    flat += [eps, replica_groups]
    return flat


_DRAFT_ARG_NAMES = flatten_draft_args(
    "input_ids",
    "prev_hidden",
    "kv_write_idx",
    "cos",
    "sin",
    "mask",
    "embed_w",
    "gamma_e",
    "gamma_h",
    "eh_w",
    {f: f"gqa_{f}" for f in DRAFT_GQA_FIELDS},
    {f: f"moe_{f}" for f in MOE_FIELDS},
    "eps",
    "replica_groups",
    "final_gamma",
    "lm_head_w",
)


def _draft_with_head(
    input_ids,
    prev_hidden,
    kv_write_idx,
    cos,
    sin,
    mask,
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    gqa_in_gamma,
    gqa_qkv_w,
    gqa_gate_w,
    gqa_gamma_q,
    gqa_gamma_k,
    gqa_o_proj_w,
    gqa_k_cache,
    gqa_v_cache,
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    final_gamma,
    lm_head_w,
    eps,
    replica_groups,
):
    return qwen36_draft_megakernel(
        input_ids,
        prev_hidden,
        kv_write_idx,
        cos,
        sin,
        mask,
        embed_w,
        gamma_e,
        gamma_h,
        eh_w,
        gqa_in_gamma,
        gqa_qkv_w,
        gqa_gate_w,
        gqa_gamma_q,
        gqa_gamma_k,
        gqa_o_proj_w,
        gqa_k_cache,
        gqa_v_cache,
        moe_gamma,
        moe_router_w,
        moe_gate_up_w,
        moe_down_w,
        moe_sigma_gate_w,
        moe_shared_gate_w,
        moe_shared_up_w,
        moe_shared_down_w,
        final_gamma,
        lm_head_w,
        eps,
        replica_groups,
        name_prefix=DRAFT_PREFIX,
    )


def _draft_no_head(
    input_ids,
    prev_hidden,
    kv_write_idx,
    cos,
    sin,
    mask,
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    gqa_in_gamma,
    gqa_qkv_w,
    gqa_gate_w,
    gqa_gamma_q,
    gqa_gamma_k,
    gqa_o_proj_w,
    gqa_k_cache,
    gqa_v_cache,
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    eps,
    replica_groups,
):
    return qwen36_draft_megakernel(
        input_ids,
        prev_hidden,
        kv_write_idx,
        cos,
        sin,
        mask,
        embed_w,
        gamma_e,
        gamma_h,
        eh_w,
        gqa_in_gamma,
        gqa_qkv_w,
        gqa_gate_w,
        gqa_gamma_q,
        gqa_gamma_k,
        gqa_o_proj_w,
        gqa_k_cache,
        gqa_v_cache,
        moe_gamma,
        moe_router_w,
        moe_gate_up_w,
        moe_down_w,
        moe_sigma_gate_w,
        moe_shared_gate_w,
        moe_shared_up_w,
        moe_shared_down_w,
        None,
        None,
        eps,
        replica_groups,
        name_prefix=REPLAY_PREFIX,
    )


_DRAFT_CACHE = {}


def build_draft_megakernel(with_lm_head):
    """The jitted draft megakernel for one build key, cached.

    ``with_lm_head`` False drops ``final_gamma``/``lm_head_w`` from the signature entirely, so the
    replay launch never streams the vocab weight. Both wrappers take the flat positional order
    ``flatten_draft_args`` emits.
    """
    key = bool(with_lm_head)
    if key in _DRAFT_CACHE:
        return _DRAFT_CACHE[key]

    wrapper = _draft_with_head if key else _draft_no_head
    head = ("final_gamma", "lm_head_w")
    names = [n for n in _DRAFT_ARG_NAMES if key or n not in head]
    assert list(inspect.signature(wrapper).parameters) == names, (
        "draft wrapper signature drifted from flatten_draft_args"
    )

    jitted = nki_jit(wrapper)
    _DRAFT_CACHE[key] = jitted
    return jitted
