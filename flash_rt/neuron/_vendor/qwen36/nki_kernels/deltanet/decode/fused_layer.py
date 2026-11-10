# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DeltaNet causal conv + gated delta-rule recurrence for token generation.

One launch runs the conv and the recurrence back-to-back with q/k/v handed over entirely in SBUF.
Sharded by value-head: each core runs a self-contained conv->recurrence pipeline over its own heads
with zero cross-core traffic, writing a disjoint slice of every full-shape output.

Per core, conv_qkv_sbuf runs the head-sharded conv over this core's 3 owned channel segments and
returns three partition-0-based SBUF tiles plus the pending conv-state stores; gated_delta_rule_tkg
then consumes those tiles directly. The gating tables, init_state seed, attn_out columns and
state-head writes stay full HBM tensors, sliced per core.

Four kernel variants, all value-head sharded and launched [2]:
  deltanet_fused_tkg_fwd            -- conv + recurrence                       decode / commit (T=1)
  deltanet_fused_tkg_fwd_state      -- same, with per-token candidate states    verify (T>=2)
  deltanet_attention_layer          -- + in_proj, gated norm, output proj      decode / commit
  deltanet_attention_layer_state    -- same, with per-token candidate states    verify

With z/gamma provided, attn_out is the gated per-head RMSNorm'd output; without them it is the raw
head-major output and the caller applies the norm/gate.
Input contract and rationale: specs/deltanet_tkg.md.
"""

import nki
import nki.language as nl

from ..components.conv import (
    P_MAX,
    conv_preload_taps,
    conv_qkv_sbuf,
    conv_state_store_pending,
    kernel_assert,
    qkv_to_channel_partition,
)
from ..components.in_proj import in_proj_compose
from ..components.out_proj import out_proj_compose
from ..components.recurrence import gated_delta_rule_tkg


def fused_compose(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z=None,
    gamma=None,
    eps=None,
):
    """Compose the per-core conv -> recurrence pipeline, shared by both entrypoints.

    Hk_full/Hv_full are passed explicitly because the SBUF tiles carry only local heads.
    """
    conv_dim = qkv.shape[1]
    Hk_full = key_dim // P_MAX
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")

    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        qkv, conv_state, conv_weight, key_dim, conv_cand, cand_is_3d
    )
    conv_state_store_pending(pending_cand)
    gated_delta_rule_tkg(
        None,
        None,
        None,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=z,
        gamma=gamma,
        eps=eps,
    )


@nki.jit
def deltanet_fused_tkg_fwd(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    z=None,
    gamma=None,
    eps=1e-6,
):
    """Decode / commit (T=1): fused conv + recurrence in one launch.

    Returns:
        attn_out:       (T, Hv*128) f32, gated when z/gamma are given, else raw head-major.
        final_state:    (Hv, 128, 128) f32, recurrent state after the block token.
        new_conv_state: (conv_dim, K-1), the committed new conv window.
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    fused_compose(
        qkv,
        conv_state,
        conv_weight,
        key_dim,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z=z,
        gamma=gamma,
        eps=eps,
    )
    return attn_out, final_state, new_conv_state


@nki.jit
def deltanet_fused_tkg_fwd_state(
    qkv,
    conv_state,
    conv_weight,
    key_dim,
    a,
    b,
    A_log,
    dt_bias,
    init_state,
    z=None,
    gamma=None,
    eps=1e-6,
):
    """Speculative verify (T>=2): fused conv + recurrence in one launch.

    candidate_states[t] and conv_cand[t] are the recurrent and conv state after block token t;
    on reject the host selects [accept_count - 1].
    """
    T, conv_dim = qkv.shape
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=qkv.dtype, buffer=nl.shared_hbm
    )
    fused_compose(
        qkv,
        conv_state,
        conv_weight,
        key_dim,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z=z,
        gamma=gamma,
        eps=eps,
    )
    return attn_out, candidate_states, conv_cand


def in_proj_fused_compose(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    attn_out,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z_gamma,
    z_eps,
):
    """Compose in_proj -> conv -> recurrence with qkv/z/a/b kept in SBUF.

    The projection stays in SBUF as proj_sb [T, I], with the qkv/z/a/b segments at successive
    free-axis offsets. The conv reads a transposed view of the qkv segment; the recurrence sources
    a/b/z from the rest.

    in_proj is contraction-sharded while conv+recurrence are value-head sharded; the
    full-per-core projection is what bridges the two.
    """
    conv_dim = conv_weight.shape[0]
    value_dim = conv_dim - 2 * key_dim
    Hk_full = key_dim // P_MAX
    Hv_full = value_dim // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")
    # Free-axis offsets of the projection segments: qkv | z | a | b.
    z_off = conv_dim
    a_off = conv_dim + value_dim
    b_off = a_off + Hv_full

    # Layer-static taps/state: issue the DMAs before in_proj so they stream under its matmuls.
    conv_taps = conv_preload_taps(conv_state, conv_weight, key_dim)

    # Fused input RMSNorm + 4-way projection, kept in SBUF.
    proj_sb = in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf=True)
    T = proj_sb.shape[0]

    # Transpose the qkv sub-block to channel-on-partition, then run the conv off SBUF.
    qkv_cp = qkv_to_channel_partition(proj_sb, conv_dim, T)
    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
        preloaded=conv_taps,
    )

    # Drive the recurrence off the conv SBUF tiles, sourcing a/b/z from proj_sb.
    gated_delta_rule_tkg(
        None,
        None,
        None,
        None,
        None,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=None,
        gamma=z_gamma,
        eps=z_eps,
        proj_sb=proj_sb,
        a_off=a_off,
        b_off=b_off,
        z_off=z_off,
    )
    # Drained here rather than at the conv, to keep the transposes off the hand-off.
    conv_state_store_pending(pending_cand)


@nki.jit
def deltanet_in_proj_fused_tkg_fwd(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    z_eps=1e-6,
):
    """Decode / commit (T=1): in_proj + conv + recurrence + gated norm, SBUF-resident.

    gamma/eps is the input RMSNorm; z_gamma/z_eps the gated per-head RMSNorm.
    Returns (attn_out gated, final_state, new_conv_state).
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    in_proj_fused_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return attn_out, final_state, new_conv_state


@nki.jit
def deltanet_in_proj_fused_tkg_fwd_state(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    z_eps=1e-6,
):
    """Speculative verify (T>=2): like the decode path, but emits per-token candidate states.

    On reject the host selects [accept_count - 1].
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX
    W_full = Hv_full * P_MAX

    attn_out = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.shared_hbm)
    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    in_proj_fused_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        attn_out,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return attn_out, candidate_states, conv_cand


def out_proj_from_recurrence(attn_sb, out_w, T, W_core, out_in_sb=False):
    """Project this core's gated output through out_proj_compose.

    Returns HBM [T, hidden], or the per-core SBUF H-shard when out_in_sb is set.
    """
    return out_proj_compose(attn_sb[0:T, 0:W_core], out_w, out_in_sb=out_in_sb)


def attention_layer_compose(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    out_w,
    state_hbm,
    conv_cand,
    write_candidates,
    cand_is_3d,
    z_gamma,
    z_eps,
    out_in_sb=False,
    name_prefix="",
):
    """Compose in_proj -> conv -> recurrence -> gated norm into SBUF, then project to o_out.

    hidden may be HBM [B, S, H] or the megakernel's SBUF residual; qkv_tkg sniffs the buffer.
    out_in_sb returns the per-core SBUF o_proj partial instead of HBM [T, hidden].
    """
    conv_dim = conv_weight.shape[0]
    value_dim = conv_dim - 2 * key_dim
    Hk_full = key_dim // P_MAX
    Hv_full = value_dim // P_MAX
    kernel_assert(state_hbm.shape[-3] == Hv_full, "state head count must equal Hv")
    z_off = conv_dim
    a_off = conv_dim + value_dim
    b_off = a_off + Hv_full

    n = nl.num_programs(0)
    Hv_core = Hv_full // n
    W_core = Hv_core * P_MAX
    W_full = Hv_full * P_MAX

    # Layer-static taps/state: issue the DMAs before in_proj so they stream under its matmuls.
    conv_taps = conv_preload_taps(conv_state, conv_weight, key_dim)

    proj_sb = in_proj_compose(
        hidden, proj_w, gamma, eps, output_in_sbuf=True, name_prefix=name_prefix
    )
    T = proj_sb.shape[0]

    attn_shape = nl.ndarray((T, W_full), dtype=nl.float32, buffer=nl.sbuf)
    attn_sb = nl.ndarray((T, W_core), dtype=out_w.dtype, buffer=nl.sbuf)

    qkv_cp = qkv_to_channel_partition(proj_sb, conv_dim, T)
    q_sbuf, k_sbuf, v_sbuf, pending_cand = conv_qkv_sbuf(
        None,
        conv_state,
        conv_weight,
        key_dim,
        conv_cand,
        cand_is_3d,
        qkv_cp_sbuf=qkv_cp,
        preloaded=conv_taps,
    )
    gated_delta_rule_tkg(
        None,
        None,
        None,
        None,
        None,
        A_log,
        dt_bias,
        init_state,
        attn_shape,
        state_hbm,
        write_candidates,
        q_sbuf=q_sbuf,
        k_sbuf=k_sbuf,
        v_sbuf=v_sbuf,
        Hk_full=Hk_full,
        Hv_full=Hv_full,
        z=None,
        gamma=z_gamma,
        eps=z_eps,
        proj_sb=proj_sb,
        a_off=a_off,
        b_off=b_off,
        z_off=z_off,
        attn_sb_out=attn_sb,
    )
    # Drained here rather than at the conv, to keep the transposes off the hand-off.
    conv_state_store_pending(pending_cand)
    return out_proj_from_recurrence(attn_sb, out_w, T, W_core, out_in_sb=out_in_sb)


@nki.jit
def deltanet_attention_layer(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    out_w,
    z_eps=1e-6,
):
    """Decode / commit (T=1): the whole DeltaNet layer in one launch.

    in_proj + conv + recurrence + gated norm + output projection. gamma/eps is the input RMSNorm,
    z_gamma/z_eps the gated per-head RMSNorm, out_w the [value_dim, hidden] o_proj weight transpose.
    Returns (o_out per-rank partial, final_state, new_conv_state).
    """
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX

    final_state = nl.ndarray(
        (Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    new_conv_state = nl.ndarray(
        (conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    o_out = attention_layer_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        out_w,
        final_state,
        new_conv_state,
        write_candidates=False,
        cand_is_3d=False,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return o_out, final_state, new_conv_state


@nki.jit
def deltanet_attention_layer_state(
    hidden,
    proj_w,
    gamma,
    eps,
    conv_state,
    conv_weight,
    key_dim,
    A_log,
    dt_bias,
    init_state,
    z_gamma,
    out_w,
    z_eps=1e-6,
):
    """Speculative verify (T>=2): like the decode path, but emits per-token candidate states."""
    conv_dim = conv_weight.shape[0]
    state_w = conv_weight.shape[1] - 1
    T = hidden.shape[0] * hidden.shape[1]
    Hv_full = (conv_dim - 2 * key_dim) // P_MAX

    candidate_states = nl.ndarray(
        (T, Hv_full, P_MAX, P_MAX), dtype=nl.float32, buffer=nl.shared_hbm
    )
    conv_cand = nl.ndarray(
        (T, conv_dim, state_w), dtype=conv_weight.dtype, buffer=nl.shared_hbm
    )
    o_out = attention_layer_compose(
        hidden,
        proj_w,
        gamma,
        eps,
        conv_state,
        conv_weight,
        key_dim,
        A_log,
        dt_bias,
        init_state,
        out_w,
        candidate_states,
        conv_cand,
        write_candidates=True,
        cand_is_3d=True,
        z_gamma=z_gamma,
        z_eps=z_eps,
    )
    return o_out, candidate_states, conv_cand
