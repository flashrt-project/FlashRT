"""Qwen3.6-A3B verify-trunk megakernel: all decoder layers in one LNC2 launch.

The residual stays SBUF-resident across every layer in the tp2013 shard-interleaved [H0, T*H1] tile,
and the kernel embeds the input token ids itself -- so only that gather and the final store touch
HBM. Each layer runs attention (DeltaNet or GQA) then the MoE FFN, each followed by an in-kernel TP
all-reduce + LNC gather of its per-rank partial back into the residual. Structure mirrors nkilib's
transformer_tkg.

Attention H-shards its o_proj output across the two cores, so it needs an H-gather; MoE
token-shards its output, so it needs a token-gather. Both reduce across the TP replica group first.

The finished residual then runs the final norm, vocab head and greedy argmax, so the kernel returns
token ids directly. The residual is still returned pre-final-norm, because the caller needs it as
the draft's rolling-buffer seed. Taking ids in and emitting ids out keeps the whole
id -> hidden -> id step inside one launch.

Not decorated -- wrap with nki.jit() at the call site, since a double jit overflows the stack.
"""

import linecache

import nki.isa as nisa
import nki.language as nl
import nki.collectives as nccl
from nki import jit as nki_jit

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from ..deltanet.decode.fused_layer import attention_layer_compose
from ..embed.components.embed import embed_compose, load_token_ids_to_sbuf
from ..gqa.decode.fused_layer import gqa_fused_compose
from ..lm_head.components.lm_head import lm_head_compose
from ..moe.components.moe_layer import moe_layer_compose
from ..moe.components.shared_expert import moe_tkg_shard_decision
from .collectives import (
    all_gather_argmax,
    all_reduce_gather_h,
    all_reduce_gather_tokens,
    load_residual_to_sbuf,  # noqa: F401  (re-exported for the layer-level tests)
    store_residual_to_hbm,
)


def verify_trunk_compose(
    residual,
    T,
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
    gqa_kv_write_idx=None,
    name_prefix="",
):
    """All decoder layers plus the vocab head over an already-seeded SBUF residual.

    The caller owns the seam on both ends: it seeds ``residual`` (from HBM hidden, or from an
    in-kernel embed) and it stores whatever it needs. ``residual`` is updated IN PLACE and is left
    PRE-final-norm, because a fused round feeds it straight to the next draft stage.

    Optional kwargs:
        gqa_kv_write_idx  [B,1] int32 write-start slot. Every GQA layer scatters its active K/V
                          into its caches in place, so the caller skips the KV scatter.

    Returns:
        ``(token_idx [T, 1] int32 SBUF, gqa_out, dn_out)`` -- the greedy ids, the flat per-GQA-layer
        (active_k, active_v) and the flat per-DeltaNet-layer (cand_state, conv_cand). With
        gqa_kv_write_idx each GQA layer contributes (k_cache, v_cache, active_k, active_v).
    """
    H0 = nl.tile_size.pmax
    H1 = residual.shape[1] // T
    H = H0 * H1
    H1_shard = H1 // n_prgs
    _, _, prg_id = get_verified_program_sharding_info(
        "qwen36_verify_trunk", (0, 1), 2
    )

    gqa_out = []
    dn_out = []
    dn = 0
    gqa = 0
    for i in range(len(layer_is_gqa)):
        pfx = f"{name_prefix}L{i}_"
        x_sb = residual.reshape((H0, T, H1))

        if layer_is_gqa[i]:
            out = gqa_fused_compose(
                x_sb,
                gqa_qkv_w[gqa],
                gqa_gate_w[gqa],
                gqa_gamma_q[gqa],
                gqa_gamma_k[gqa],
                cos,
                sin,
                gqa_k_cache[gqa],
                gqa_v_cache[gqa],
                gqa_mask,
                gqa_o_proj_w[gqa],
                eps,
                kv_write_idx=gqa_kv_write_idx,
                gamma_in=gqa_in_gamma[gqa],
                out_in_sb=True,
                name_prefix=pfx,
            )
            attn_partial = out[0]
            # Mutated caches first when writing in place; the active K/V follow either way, so the
            # allocator cannot overlay a write nothing consumes.
            gqa_out += list(out[3:5]) + list(out[1:3])
            gqa += 1
        else:
            conv_dim = dn_conv_weight[dn].shape[0]
            state_w = dn_conv_weight[dn].shape[1] - 1
            Hv = (conv_dim - 2 * key_dim) // H0

            cand_state = nl.ndarray(
                (T, Hv, H0, H0), dtype=nl.float32, buffer=nl.shared_hbm
            )
            conv_cand = nl.ndarray(
                (T, conv_dim, state_w),
                dtype=dn_conv_weight[dn].dtype,
                buffer=nl.shared_hbm,
            )
            attn_partial = attention_layer_compose(
                x_sb,
                dn_proj_w[dn],
                dn_in_gamma[dn],
                eps,
                dn_conv_state[dn],
                dn_conv_weight[dn],
                key_dim,
                dn_A_log[dn],
                dn_dt_bias[dn],
                dn_init_state[dn],
                dn_out_w[dn],
                cand_state,
                conv_cand,
                write_candidates=True,
                cand_is_3d=True,
                z_gamma=dn_z_gamma[dn],
                z_eps=eps,
                out_in_sb=True,
                name_prefix=pfx,
            )
            dn_out.append(cand_state)
            dn_out.append(conv_cand)
            dn += 1

        # attn_partial is the H-sharded o_proj output [H0, H1_shard*T] (out_in_sb + transposed_out).
        attn_out = all_reduce_gather_h(
            attn_partial.reshape((H0, H1_shard * T)), rg, prg_id, n_prgs, T
        )
        nisa.tensor_tensor(dst=residual, data1=residual, data2=attn_out, op=nl.add)

        moe_partial = moe_layer_compose(
            residual.reshape((H0, T, H1)),
            moe_gamma[i],
            moe_router_w[i],
            moe_gate_up_w[i],
            moe_down_w[i],
            moe_sigma_gate_w[i],
            moe_shared_gate_w[i],
            moe_shared_up_w[i],
            moe_shared_down_w[i],
            eps=eps,
            output_in_sbuf=True,
            name_prefix=pfx,
        )
        _, T_offset, T_len = moe_tkg_shard_decision(T, H, moe_gate_up_w[i].shape[3])
        moe_out = all_reduce_gather_tokens(
            moe_partial, rg, prg_id, n_prgs, T_offset, T_len
        )
        nisa.tensor_tensor(dst=residual, data1=residual, data2=moe_out, op=nl.add)

    # residual stays the pre-final-norm hidden; the head norms its own copy.
    rank_max, rank_idx, _ = lm_head_compose(
        residual.reshape((H0, T, H1)),
        final_gamma,
        lm_head_w,
        eps=eps,
        name_prefix=name_prefix + "lm_",
    )
    token_idx = all_gather_argmax(rank_max, rank_idx, rg, tp_degree, lm_head_w.shape[1])
    return token_idx, gqa_out, dn_out


def qwen36_verify_megakernel(
    X,
    layer_is_gqa,
    key_dim,
    eps,
    replica_groups,
    # DeltaNet layers (indexed by DeltaNet position)
    dn_proj_w,
    dn_in_gamma,
    dn_conv_state,
    dn_conv_weight,
    dn_A_log,
    dn_dt_bias,
    dn_init_state,
    dn_out_w,
    dn_z_gamma,
    # GQA layers (indexed by GQA position; cos/sin/mask shared)
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
    # MoE (every layer)
    moe_gamma,
    moe_router_w,
    moe_gate_up_w,
    moe_down_w,
    moe_sigma_gate_w,
    moe_shared_gate_w,
    moe_shared_up_w,
    moe_shared_down_w,
    # LM head
    final_gamma,
    lm_head_w,
):
    """Run all decoder layers, SBUF-resident residual, per-rank partials reduced in-kernel.

    layer_is_gqa[i] selects the attention type for layer i; the DeltaNet and GQA weight lists are
    indexed by each type's running position.

    embed_w is [V, H/TP] -- ParallelEmbedding shards on hidden, so H is recovered from it only after
    tp_degree is known, and the per-rank slices are all-gathered inside embed_compose.

    Returns:
        (tokens [B,S] int32 HBM, hidden [B,S,H] HBM, gqa_active_kv, dn_cand) -- the greedy ids, the
        trunk hidden, per-GQA-layer (active_k, active_v) for the caller's KV scatter, and
        per-DeltaNet-layer (candidate_states, conv_cand) for the accept/reject commit.
        hidden is pre-final-norm; the caller needs the un-normed hidden as the draft's seed.
    """
    B, S, H = X.shape
    dtype = X.dtype
    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "qwen36_verify_megakernel", (0, 1), 2
    )
    rg = nccl.ReplicaGroup(replica_groups) if replica_groups is not None else None
    tp_degree = len(replica_groups[0]) if replica_groups is not None else 1
    H0 = nl.tile_size.pmax
    H1 = H // H0
    T = B * S

    residual = nl.ndarray((H0, T * H1), dtype=dtype, buffer=nl.sbuf)
    load_residual_to_sbuf(residual, X, T, H0, H1, n_prgs)

    token_idx, gqa_out, dn_out = verify_trunk_compose(
        residual,
        T,
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
    )

    output = nl.ndarray((B, S, H), dtype=dtype, buffer=nl.shared_hbm)
    if prg_id == 0:
        store_residual_to_hbm(output, residual, T, H0, H1, n_prgs)
    if n_prgs > 1:
        nisa.core_barrier(data=output, cores=(0, 1))

    # [T, 1] SBUF is one element per partition, so the HBM row is written through a [T, 1] view
    # of it; a flat T-wide access pattern would read one partition and run off the end.
    tokens = nl.ndarray((B, S), dtype=nl.int32, buffer=nl.shared_hbm)
    if prg_id == 0:
        nisa.dma_copy(dst=tokens.reshape((T, 1)), src=token_idx)
    if n_prgs > 1:
        nisa.core_barrier(data=tokens, cores=(0, 1))
    return tuple([tokens, output] + gqa_out + dn_out)


DN_FIELDS = (
    "proj_w",
    "in_gamma",
    "conv_state",
    "conv_weight",
    "A_log",
    "dt_bias",
    "init_state",
    "out_w",
    "z_gamma",
)
GQA_FIELDS = (
    "qkv_w",
    "gate_w",
    "gamma_q",
    "gamma_k",
    "in_gamma",
    "o_proj_w",
    "k_cache",
    "v_cache",
)
MOE_FIELDS = (
    "gamma",
    "router_w",
    "gate_up_w",
    "down_w",
    "sigma_gate_w",
    "shared_gate_w",
    "shared_up_w",
    "shared_down_w",
)


def flatten_megakernel_args(
    X,
    dn,
    gqa,
    moe,
    cos,
    sin,
    gqa_mask,
    key_dim,
    eps,
    replica_groups,
    final_gamma,
    lm_head_w,
):
    """The one definition of the flat positional argument order.

    ``dn``/``gqa``/``moe`` map each field name to its per-layer list. Used both by
    ``build_verify_megakernel`` (over parameter NAMES, to generate the wrapper signature) and by
    the caller (over tensor VALUES), so the two cannot drift.
    """
    flat = [X]
    for f in DN_FIELDS:
        flat += list(dn[f])
    for f in GQA_FIELDS:
        flat += list(gqa[f])
    flat += [cos, sin, gqa_mask]
    for f in MOE_FIELDS:
        flat += list(moe[f])
    flat += [final_gamma, lm_head_w]
    flat += [key_dim, eps, replica_groups]
    return flat


def split_megakernel_returns(rets, n_gqa, n_dn):
    """Un-flatten the return tuple into (tokens, hidden, gqa_active_kv, dn_candidates)."""
    gqa_flat = rets[2 : 2 + 2 * n_gqa]
    dn_flat = rets[2 + 2 * n_gqa :]
    return (
        rets[0],
        rets[1],
        [(gqa_flat[2 * i], gqa_flat[2 * i + 1]) for i in range(n_gqa)],
        [(dn_flat[2 * i], dn_flat[2 * i + 1]) for i in range(n_dn)],
    )


_MEGAKERNEL_CACHE = {}


def build_verify_megakernel(layer_is_gqa):
    key = tuple(bool(x) for x in layer_is_gqa)
    if key in _MEGAKERNEL_CACHE:
        return _MEGAKERNEL_CACHE[key]

    n = len(key)
    n_dn = sum(1 for x in key if not x)
    n_gqa = sum(1 for x in key if x)

    def col(prefix, fields, count):
        return {f: [f"{prefix}_{f}_{j}" for j in range(count)] for f in fields}

    dn = col("dn", DN_FIELDS, n_dn)
    gqa = col("gqa", GQA_FIELDS, n_gqa)
    moe = col("moe", MOE_FIELDS, n)

    flat = flatten_megakernel_args(
        "X",
        dn,
        gqa,
        moe,
        "cos",
        "sin",
        "gqa_mask",
        "key_dim",
        "eps",
        "replica_groups",
        "final_gamma",
        "lm_head_w",
    )

    def tup(ns):
        return "(" + ", ".join(ns) + ("," if ns else "") + ")"

    lines = ["def _verify_wrapper(\n    " + ",\n    ".join(flat) + ",\n):"]
    for f in DN_FIELDS:
        lines.append(f"    dn_{f} = {tup(dn[f])}")
    for f in GQA_FIELDS:
        lines.append(f"    gqa_{f} = {tup(gqa[f])}")
    for f in MOE_FIELDS:
        lines.append(f"    moe_{f} = {tup(moe[f])}")
    body_args = [
        "X",
        repr(key),
        "key_dim",
        "eps",
        "replica_groups",
    ]
    body_args += [f"dn_{f}" for f in DN_FIELDS]
    body_args += [f"gqa_{f}" for f in GQA_FIELDS]
    body_args += ["cos", "sin", "gqa_mask"]
    body_args += [f"moe_{f}" for f in MOE_FIELDS]
    body_args += ["final_gamma", "lm_head_w"]
    lines.append(
        "    return _body(\n        " + ",\n        ".join(body_args) + ",\n    )"
    )
    src = "\n".join(lines) + "\n"

    fname = f"<verify_megakernel_L{n}_g{n_gqa}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    ns = {"_body": qwen36_verify_megakernel}
    exec(code, ns)
    jitted = nki_jit(ns["_verify_wrapper"])
    _MEGAKERNEL_CACHE[key] = jitted
    return jitted
