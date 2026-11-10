"""Qwen3.6-A3B fused speculation round: draft, verify and replay in one LNC2 launch.

Three stages that used to be three NEFFs, traced back to back so every seam stays in SBUF:

    D1  T=1  draft_stage_compose(ids=[x_t+1], h_t)         -> x_t+2^draft, writes MTP slot t
    V   T=2  verify_trunk_compose(embed([x_t+1, x_t+2]))   -> target ids, trunk hidden, candidates
    D2  T=1  draft_stage_compose(x_t+2^draft, h_t+1)       -> writes MTP slot t+1, headless

Three seams carry no HBM traffic: D1's token id reaches V's embed through a [T, 1] SBUF tile, V's
residual reaches D2's eh_proj through one tp2013->natural transpose, and D1's mutated cache handles
thread into D2 -- that last dependency orders D2's read of MTP slot t after D1's in-place write,
which is what makes the T=1 replay legal. Neither draft's hidden is stored.

The accept/commit epilogue is the caller's: this kernel returns the candidate id, the target ids,
the pre-final-norm trunk hidden and every mutated handle, and the host does the greedy compare and
the DeltaNet state select.

Not decorated -- build_round_megakernel wraps the flat signature with nki.jit(), since a double jit
overflows the stack. Why the three-call form must replay at T=2: specs/round_megakernel_perf.md.
"""

import linecache

import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nki import jit as nki_jit

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from ..embed.components.embed import (
    embed_compose,
    load_token_ids_to_sbuf,
    tp2013_to_natural,
)
from .collectives import store_residual_to_hbm
from .qwen36_draft_megakernel import (
    DRAFT_GQA_FIELDS,
    DRAFT_PREFIX,
    REPLAY_PREFIX,
    draft_stage_compose,
)
from .qwen36_verify_megakernel import (
    DN_FIELDS,
    GQA_FIELDS,
    MOE_FIELDS,
    verify_trunk_compose,
)

# The MTP weights are one set shared by both draft stages; only these differ per stage.
STAGE_FIELDS = ("kv_write_idx", "cos", "sin", "mask")
VERIFY_PREFIX = "v_"


def qwen36_round_megakernel(
    input_ids,
    prev_hidden,
    layer_is_gqa,
    key_dim,
    eps,
    replica_groups,
    # MTP draft head -- one set of weights, both stages
    embed_w,
    gamma_e,
    gamma_h,
    eh_w,
    mtp_in_gamma,
    mtp_qkv_w,
    mtp_gate_w,
    mtp_gamma_q,
    mtp_gamma_k,
    mtp_o_proj_w,
    mtp_k_cache,
    mtp_v_cache,
    mtp_moe_gamma,
    mtp_moe_router_w,
    mtp_moe_gate_up_w,
    mtp_moe_down_w,
    mtp_moe_sigma_gate_w,
    mtp_moe_shared_gate_w,
    mtp_moe_shared_up_w,
    mtp_moe_shared_down_w,
    mtp_final_gamma,
    mtp_lm_head_w,
    # per-stage rope / mask / cache write position
    d1_kv_write_idx,
    d1_cos,
    d1_sin,
    d1_mask,
    d2_kv_write_idx,
    d2_cos,
    d2_sin,
    d2_mask,
    # verify trunk
    dn_proj_w,
    dn_in_gamma,
    dn_conv_state,
    dn_conv_weight,
    dn_A_log,
    dn_dt_bias,
    dn_init_state,
    dn_out_w,
    dn_z_gamma,
    gqa_qkv_w,
    gqa_gate_w,
    gqa_gamma_q,
    gqa_gamma_k,
    gqa_in_gamma,
    gqa_o_proj_w,
    gqa_k_cache,
    gqa_v_cache,
    cos,
    sin,
    gqa_mask,
    v_kv_write_idx,
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
):
    """One greedy EAGLE round at spec_len=2.

    input_ids is the single committed token x_t+1 and prev_hidden its trunk hidden h_t. Rope, mask
    and cache-write slot come per stage, because the three read at different positions: D1 at t,
    verify at [t+1, t+2], D2 at t+1. d2_mask must count MTP slot t as committed prior -- the slot D1
    wrote. Every stage writes its own KV in place, the trunk's at v_kv_write_idx.

    Returns:
        (cand_token [B,1], tokens [B,2], hidden [B,2,H], *gqa_kv, *dn_candidates,
        mtp_k_cache, mtp_v_cache, *mtp_active_kv). hidden is the verify trunk's, pre-final-norm.
        Every mutated shared_hbm handle is returned: NCC dead-stores a write nothing consumes, and
        an unreturned buffer may be overlaid by the allocator.
    """
    B, S = input_ids.shape
    dtype = embed_w.dtype
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen36_round_megakernel", (0, 1), 2
    )
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None
    tp_degree = len(replica_groups[0]) if replica_groups is not None else 1
    H0 = nl.tile_size.pmax
    H = embed_w.shape[1] * tp_degree
    H1 = H // H0
    T_d = B * S  # one drafted token per stage
    T_v = T_d + 1  # the verify block: committed + speculative

    # -- D1: draft x_t+2 from the carry, writing MTP slot t. The MTP weights are spelled out at
    # both call sites: the NKI frontend does not accept tuple expansion in a call.
    _, cand_idx, mtp_k_cache, mtp_v_cache, d1_active_k, d1_active_v = (
        draft_stage_compose(
            load_token_ids_to_sbuf(input_ids, T_d),
            prev_hidden,
            d1_kv_write_idx,
            d1_cos,
            d1_sin,
            d1_mask,
            embed_w,
            gamma_e,
            gamma_h,
            eh_w,
            mtp_in_gamma,
            mtp_qkv_w,
            mtp_gate_w,
            mtp_gamma_q,
            mtp_gamma_k,
            mtp_o_proj_w,
            mtp_k_cache,
            mtp_v_cache,
            mtp_moe_gamma,
            mtp_moe_router_w,
            mtp_moe_gate_up_w,
            mtp_moe_down_w,
            mtp_moe_sigma_gate_w,
            mtp_moe_shared_gate_w,
            mtp_moe_shared_up_w,
            mtp_moe_shared_down_w,
            mtp_final_gamma,
            mtp_lm_head_w,
            eps,
            rg,
            tp_degree,
            n_prgs,
            name_prefix=DRAFT_PREFIX,
        )
    )

    # -- V: the block is [committed, drafted]. Partition 0 comes from HBM, partition 1 from D1's
    # argmax; a partition-BASE offset is a DMA-engine move, not a compute-engine one (tensor_copy
    # requires quadrant-aligned partition offsets).
    ids_v = nl.ndarray((T_v, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids_v[0:T_d, 0:1], src=input_ids.reshape((T_d, 1)))
    nisa.dma_copy(dst=ids_v[T_d:T_v, 0:1], src=cand_idx)

    residual = nl.ndarray((H0, T_v * H1), dtype=dtype, buffer=nl.sbuf)
    embed_compose(
        ids_v,
        embed_w,
        rg=rg,
        tp_degree=tp_degree,
        n_prgs=n_prgs,
        out_sb=residual,
    )
    token_idx, gqa_out, dn_out = verify_trunk_compose(
        residual,
        T_v,
        layer_is_gqa,
        key_dim,
        eps,
        rg,
        tp_degree,
        n_prgs,
        dn_proj_w,
        dn_in_gamma,
        dn_conv_state,
        dn_conv_weight,
        dn_A_log,
        dn_dt_bias,
        dn_init_state,
        dn_out_w,
        dn_z_gamma,
        gqa_qkv_w,
        gqa_gate_w,
        gqa_gamma_q,
        gqa_gamma_k,
        gqa_in_gamma,
        gqa_o_proj_w,
        gqa_k_cache,
        gqa_v_cache,
        cos,
        sin,
        gqa_mask,
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
        gqa_kv_write_idx=v_kv_write_idx,
        name_prefix=VERIFY_PREFIX,
    )

    # -- D2: slot t+1 is by definition built from h_t+1, a verify output, so it could not exist at
    # D1. Only that slot needs writing: at spec_len=2 the draft loop's sole input was the verified
    # carry, so D1's slot t is already correct and needs no replay.
    hidden_nat = tp2013_to_natural(residual, T_v, n_prgs)
    _, _, mtp_k_cache, mtp_v_cache, d2_active_k, d2_active_v = draft_stage_compose(
        cand_idx,
        hidden_nat[0:T_d, :],
        d2_kv_write_idx,
        d2_cos,
        d2_sin,
        d2_mask,
        embed_w,
        gamma_e,
        gamma_h,
        eh_w,
        mtp_in_gamma,
        mtp_qkv_w,
        mtp_gate_w,
        mtp_gamma_q,
        mtp_gamma_k,
        mtp_o_proj_w,
        mtp_k_cache,
        mtp_v_cache,
        mtp_moe_gamma,
        mtp_moe_router_w,
        mtp_moe_gate_up_w,
        mtp_moe_down_w,
        mtp_moe_sigma_gate_w,
        mtp_moe_shared_gate_w,
        mtp_moe_shared_up_w,
        mtp_moe_shared_down_w,
        None,
        None,
        eps,
        rg,
        tp_degree,
        n_prgs,
        name_prefix=REPLAY_PREFIX,
    )

    hidden = nl.ndarray((B, T_v, H), dtype=dtype, buffer=nl.shared_hbm)
    cand_token = nl.ndarray((B, T_d), dtype=nl.int32, buffer=nl.shared_hbm)
    tokens = nl.ndarray((B, T_v), dtype=nl.int32, buffer=nl.shared_hbm)
    if prg_id == 0:
        store_residual_to_hbm(hidden, residual, T_v, H0, H1, n_prgs)
        # [T, 1] SBUF is one element per partition, so each HBM row is written through a [T, 1]
        # view; a flat T-wide access pattern would read one partition and run off the end.
        nisa.dma_copy(dst=cand_token.reshape((T_d, 1)), src=cand_idx)
        nisa.dma_copy(dst=tokens.reshape((T_v, 1)), src=token_idx)
    if n_prgs > 1:
        nisa.core_barrier(data=hidden, cores=(0, 1))
        nisa.core_barrier(data=cand_token, cores=(0, 1))
        nisa.core_barrier(data=tokens, cores=(0, 1))

    return tuple(
        [cand_token, tokens, hidden]
        + gqa_out
        + dn_out
        + [
            mtp_k_cache,
            mtp_v_cache,
            d1_active_k,
            d1_active_v,
            d2_active_k,
            d2_active_v,
        ]
    )


MTP_FRONT_FIELDS = ("embed_w", "gamma_e", "gamma_h", "eh_w")
MTP_HEAD_FIELDS = ("final_gamma", "lm_head_w")


def flatten_round_args(
    input_ids,
    prev_hidden,
    mtp_front,
    mtp_gqa,
    mtp_moe,
    mtp_head,
    d1,
    d2,
    dn,
    gqa,
    moe,
    cos,
    sin,
    gqa_mask,
    v_kv_write_idx,
    final_gamma,
    lm_head_w,
    key_dim,
    eps,
    replica_groups,
):
    """The one definition of the flat positional argument order.

    The dict arguments map each field name to its tensor (``mtp_*``, ``d1``, ``d2``) or to its
    per-layer list (``dn``, ``gqa``, ``moe``). Used both by ``build_round_megakernel`` over
    parameter NAMES and by the caller over tensor VALUES, so the two cannot drift.
    """
    flat = [input_ids, prev_hidden]
    flat += [mtp_front[f] for f in MTP_FRONT_FIELDS]
    flat += [mtp_gqa[f] for f in DRAFT_GQA_FIELDS]
    flat += [mtp_moe[f] for f in MOE_FIELDS]
    flat += [mtp_head[f] for f in MTP_HEAD_FIELDS]
    flat += [d1[f] for f in STAGE_FIELDS]
    flat += [d2[f] for f in STAGE_FIELDS]
    for f in DN_FIELDS:
        flat += list(dn[f])
    for f in GQA_FIELDS:
        flat += list(gqa[f])
    flat += [cos, sin, gqa_mask, v_kv_write_idx]
    for f in MOE_FIELDS:
        flat += list(moe[f])
    flat += [final_gamma, lm_head_w]
    flat += [key_dim, eps, replica_groups]
    return flat


def split_round_returns(rets, n_gqa, n_dn):
    """Un-flatten into (cand_token, tokens, hidden, gqa_kv, dn_candidates, mtp_kv).

    Each GQA layer contributes four handles -- the mutated caches then the active K/V, which the
    round returns only to keep them off the allocator's overlay list. Only the caches are handed on.
    """
    gqa_end = 3 + 4 * n_gqa
    gqa_flat = rets[3:gqa_end]
    dn_flat = rets[gqa_end : gqa_end + 2 * n_dn]
    mtp_kv = rets[gqa_end + 2 * n_dn : gqa_end + 2 * n_dn + 2]
    return (
        rets[0],
        rets[1],
        rets[2],
        [(gqa_flat[4 * i], gqa_flat[4 * i + 1]) for i in range(n_gqa)],
        [(dn_flat[2 * i], dn_flat[2 * i + 1]) for i in range(n_dn)],
        tuple(mtp_kv),
    )


_ROUND_CACHE = {}


def build_round_megakernel(layer_is_gqa):
    """The jitted round megakernel for one layer-type pattern, cached.

    NKI's frontend classifies a tuple/list top-level arg as a scalar -- no HBM binding, and aliasing
    breaks -- so every weight and cache is its own positional arg and the wrapper that unpacks them
    back into per-field tuples is generated per pattern.
    """
    key = tuple(bool(x) for x in layer_is_gqa)
    if key in _ROUND_CACHE:
        return _ROUND_CACHE[key]

    n = len(key)
    n_dn = sum(1 for x in key if not x)
    n_gqa = sum(1 for x in key if x)

    def col(prefix, fields, count):
        return {f: [f"{prefix}_{f}_{j}" for j in range(count)] for f in fields}

    def one(prefix, fields):
        return {f: f"{prefix}_{f}" for f in fields}

    mtp_front = {f: f for f in MTP_FRONT_FIELDS}
    mtp_gqa = one("mtp", DRAFT_GQA_FIELDS)
    mtp_moe = one("mtp_moe", MOE_FIELDS)
    mtp_head = one("mtp", MTP_HEAD_FIELDS)
    d1 = one("d1", STAGE_FIELDS)
    d2 = one("d2", STAGE_FIELDS)
    dn = col("dn", DN_FIELDS, n_dn)
    gqa = col("gqa", GQA_FIELDS, n_gqa)
    moe = col("moe", MOE_FIELDS, n)

    flat = flatten_round_args(
        "input_ids",
        "prev_hidden",
        mtp_front,
        mtp_gqa,
        mtp_moe,
        mtp_head,
        d1,
        d2,
        dn,
        gqa,
        moe,
        "cos",
        "sin",
        "gqa_mask",
        "v_kv_write_idx",
        "final_gamma",
        "lm_head_w",
        "key_dim",
        "eps",
        "replica_groups",
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ("," if ns else "") + ")"

    lines = ["def _round_wrapper(\n    " + ",\n    ".join(flat) + ",\n):"]
    for f in DN_FIELDS:
        lines.append(f"    dn_{f} = {tup(dn[f])}")
    for f in GQA_FIELDS:
        lines.append(f"    gqa_{f} = {tup(gqa[f])}")
    for f in MOE_FIELDS:
        lines.append(f"    moe_{f} = {tup(moe[f])}")
    body_args = ["input_ids", "prev_hidden", repr(key), "key_dim", "eps", "replica_groups"]
    body_args += list(MTP_FRONT_FIELDS)
    body_args += [mtp_gqa[f] for f in DRAFT_GQA_FIELDS]
    body_args += [mtp_moe[f] for f in MOE_FIELDS]
    body_args += [mtp_head[f] for f in MTP_HEAD_FIELDS]
    body_args += [d1[f] for f in STAGE_FIELDS]
    body_args += [d2[f] for f in STAGE_FIELDS]
    body_args += [f"dn_{f}" for f in DN_FIELDS]
    body_args += [f"gqa_{f}" for f in GQA_FIELDS]
    body_args += ["cos", "sin", "gqa_mask", "v_kv_write_idx"]
    body_args += [f"moe_{f}" for f in MOE_FIELDS]
    body_args += ["final_gamma", "lm_head_w"]
    lines.append(
        "    return _body(\n        " + ",\n        ".join(body_args) + ",\n    )"
    )
    src = "\n".join(lines) + "\n"

    fname = f"<round_megakernel_L{n}_g{n_gqa}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_body": qwen36_round_megakernel}
    exec(code, ns)
    jitted = nki_jit(ns["_round_wrapper"])
    _ROUND_CACHE[key] = jitted
    return jitted
