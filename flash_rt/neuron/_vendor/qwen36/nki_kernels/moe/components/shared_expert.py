# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert path of the Qwen3.6-A3B fused MoE layer (token generation), SBUF-resident.

A plain SwiGLU FFN -- down(silu(gate(x)) * up(x)) -- on the same SBUF-resident normed_sb the routed
path consumes. Returns the per-rank partial over H; the sigma-gate, gated sum and TP all-reduce live
in moe_layer.

Written in raw NKI, structurally one routed expert without the expert index or affinity scale.
nkilib's mlp_tkg primitives cannot be used: their gate/up loader hardcodes the tp102 H-permutation
via reshape_dim, which cannot express the permutation tp2013 needs, and no flag toggles it.

Layout: tp2013, shared with the attention kernels' SBUF residual. gate/up are separate [H, I_s]
tensors, so each partition's H2 rows coalesce into one run; down is the contiguous [I_s, H] row load
and is permutation-agnostic.

Work split matches the routed path, so shared_local keeps routed_local's per-core layout and the
gated sum is a plain per-core add. Under an H-shard, one fused [I_s, 2T] fp32 sendrecv+add reduces
the gate/up partial sums before the SiLU; down's H-columns are disjoint and need no reduce.

Per-rank (TP=4) A3B dims: H=2048 (H0=128, H1=16), I_s=128.
Layout and sharding rationale: specs/moe_tkg.md.
"""

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from .post_attn_norm import post_attn_rmsnorm_compose
from .routed_experts import NORM_SINGLE_CORE_FORCED, moe_h_shard, moe_token_shard
from .routed_experts_nki import store_tile_to_hbm_th

P_MAX = 128


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def moe_tkg_shard_decision(T, H, intermediate):
    """This core's (shard_on_T, T_offset, T_per_shard) for the MoE expert paths (routed == shared)."""
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    return moe_token_shard(T, H, intermediate, n_prgs, shard_id)


def moe_h_shard_decision(T, H, H1, intermediate):
    """This core's (shard_on_H, f_offset, H1_per_shard) over the tp2013 free axis (routed == shared)."""
    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    shard_on_T, _, _ = moe_token_shard(T, H, intermediate, n_prgs, shard_id)
    return moe_h_shard(H1, n_prgs, shard_id, shard_on_T)


def load_shared_weights(
    gate_w, up_w, down_w, H0, H1_local, H2, s_offset, s_count, I_s, H, dtype
):
    """Load the shared expert's [H,I_s] gate/up (tp2013) and [I_s,H] down (contiguous) into SBUF.

    Only this core's H-shards [s_offset, s_offset+s_count) are loaded. gate/up slabs:
    slab[h0, s_local*H2 + h2, c] = w[(s_offset+s_local)*H0*H2 + h0*H2 + h2, c]; dim1 step (I_s) equals the
    inner extent, so each partition's H2 rows coalesce into one H2*I_s run. down takes the contiguous
    H_local-column block this core owns (the whole H row when not H-sharded). Static (non-indirect) DMAs.
    """
    H_local = H1_local * H0

    gate_sb = nl.ndarray((H0, H1_local, I_s), dtype=dtype, buffer=nl.sbuf)
    up_sb = nl.ndarray((H0, H1_local, I_s), dtype=dtype, buffer=nl.sbuf)
    for s_local in range(s_count):
        pattern = [[H2 * I_s, H0], [I_s, H2], [1, I_s]]
        offset = (s_offset + s_local) * H0 * H2 * I_s
        nisa.dma_copy(
            dst=gate_sb[0:H0, s_local * H2 : (s_local + 1) * H2, 0:I_s],
            src=gate_w.ap(pattern=pattern, offset=offset),
        )
        nisa.dma_copy(
            dst=up_sb[0:H0, s_local * H2 : (s_local + 1) * H2, 0:I_s],
            src=up_w.ap(pattern=pattern, offset=offset),
        )

    down_sb = nl.ndarray((I_s, H_local), dtype=dtype, buffer=nl.sbuf)
    if H_local == H:
        nisa.dma_copy(dst=down_sb[0:I_s, 0:H], src=down_w)
    else:
        nisa.dma_copy(
            dst=down_sb[0:I_s, 0:H_local],
            src=down_w.ap(pattern=[[H, I_s], [1, H_local]], offset=s_offset * H0 * H2),
        )
    return gate_sb, up_sb, down_sb


def shared_expert_compose(normed_sb, gate_w, up_w, down_w, output_in_sbuf=True):
    """SwiGLU shared expert on an SBUF-resident normed tile (raw NKI, zero HBM round-trip).

    Args:
        normed_sb:      [H0=128, T, H1=H//128] SBUF post-attn-normed hidden (tp2013, n_s = n_prgs).
        gate_w:         [H, I_s] HBM gate weight (contraction-first; load-transposed from stored [I_s,H]).
        up_w:           [H, I_s] HBM up weight (contraction-first; load-transposed from stored [I_s,H]).
        down_w:         [I_s, H] HBM down weight (load-transposed from stored [H, I_s]).
        output_in_sbuf: SBUF [H0,T,H1] when set (matching routed_local), else HBM [T,H].

    Returns:
        shared_local, the per-rank partial in the same per-core shard layout as routed_local.
    """
    H0, T, H1 = normed_sb.shape
    H = H0 * H1
    I_s = up_w.shape[1]
    dtype = normed_sb.dtype
    kernel_assert(
        tuple(gate_w.shape) == (H, I_s), "gate_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(
        tuple(up_w.shape) == (H, I_s), "up_w must be [H, I_s] matching normed_sb H"
    )
    kernel_assert(tuple(down_w.shape) == (I_s, H), "down_w must be [I_s, H]")
    kernel_assert(
        I_s <= P_MAX, "shared expert supports I_s <= 128 (single intermediate tile)"
    )

    _, n_prgs, shard_id = get_verified_program_sharding_info("moe_shared", (0, 1), 2)
    kernel_assert(
        H1 % n_prgs == 0, "tp2013 needs H1 divisible by the H-shard count n_s"
    )
    n_s = n_prgs
    H2 = H1 // n_s
    shard_on_T, T_offset, T_per_shard = moe_token_shard(T, H, I_s, n_prgs, shard_id)
    shard_on_H, f_offset, H1_local = moe_h_shard(H1, n_prgs, shard_id, shard_on_T)
    s_offset = f_offset // H2
    s_count = H1_local // H2
    H_local = s_count * H0 * H2
    peer_rank = 1 - s_offset

    gate_sb, up_sb, down_sb = load_shared_weights(
        gate_w,
        up_w,
        down_w,
        H0,
        H1_local,
        H2,
        s_offset,
        s_count,
        I_s,
        H,
        dtype,
    )

    # gate/up: contract H0 over this core's free indices for its whole token slice at once.
    gate_psum = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.psum)
    up_psum = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.psum)
    for f in range(H1_local):
        moving = normed_sb[:, T_offset : T_offset + T_per_shard, f_offset + f]
        nisa.nc_matmul(dst=gate_psum, stationary=gate_sb[:, f, 0:I_s], moving=moving)
        nisa.nc_matmul(dst=up_psum, stationary=up_sb[:, f, 0:I_s], moving=moving)

    if shard_on_H:
        # H is the gate/up CONTRACTION dim: each core holds a partial sum. ONE fused [I_s, 2T] fp32
        # sendrecv + add reduces gate and up together before the SiLU.
        two_T = 2 * T_per_shard

        gate_up_sb = nl.ndarray((I_s, two_T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate_up_sb[:, 0:T_per_shard], src=gate_psum)
        nisa.tensor_copy(dst=gate_up_sb[:, T_per_shard:two_T], src=up_psum)

        gate_up_recv = nl.ndarray((I_s, two_T), dtype=nl.float32, buffer=nl.sbuf)
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
        gate_src = gate_up_sb[:, 0:T_per_shard]
        up_src = gate_up_sb[:, T_per_shard:two_T]
    else:
        gate_src, up_src = gate_psum, up_psum

    gate_act = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gate_act, data=gate_src, op=nl.silu)

    up_act = nl.ndarray((I_s, T_per_shard), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=up_act, src=up_src)

    intermediate = nl.ndarray((I_s, T_per_shard), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intermediate, data1=gate_act, data2=up_act, op=nl.multiply)

    # down: contract I_s -> [H0,1] per free index f_offset+f, the strided stationary picking the tp2013
    # H-columns {s*H0*H2 + h0*H2 + h2} (s, h2 = divmod(f_offset+f, H2)). H is the OUTPUT dim here, so the
    # H-shard's columns are disjoint per core -- no reduce. Each core writes ONLY its own slice.
    shared_local = nl.ndarray((H0, T, H1), dtype=dtype, buffer=nl.sbuf)
    for local_token_idx in range(T_per_shard):
        down_psum = nl.ndarray((H0, H1_local), dtype=nl.float32, buffer=nl.psum)
        for f in range(H1_local):
            s, h2 = (f_offset + f) // H2, (f_offset + f) % H2
            nisa.nc_matmul(
                dst=down_psum[0:H0, f : f + 1],
                stationary=down_sb.ap(
                    pattern=[[H_local, I_s], [H2, H0]],
                    offset=(s - s_offset) * H0 * H2 + h2,
                ),
                moving=intermediate[0:I_s, local_token_idx : local_token_idx + 1],
            )
        nisa.tensor_copy(
            dst=shared_local[
                :, T_offset + local_token_idx, f_offset : f_offset + H1_local
            ],
            src=down_psum,
        )

    if output_in_sbuf:
        return shared_local

    out_hbm = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm)
    store_tile_to_hbm_th(
        out_hbm,
        shared_local,
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


def moe_shared_compose(
    hidden,
    gamma,
    gate_w,
    up_w,
    down_w,
    eps=1e-6,
    hidden_actual=None,
    output_in_sbuf=True,
):
    """Full shared-expert chain: post-attn RMSNorm -> SwiGLU shared expert (all SBUF-resident).

    Args:
        hidden:   [B, S, H] HBM raw post-attn residual (B=1), left untouched.
        gamma:    [1, H] HBM post_attention_layernorm.weight (standard form).
        gate_w / up_w / down_w: see ``shared_expert_compose``.
        eps:      RMSNorm epsilon.
        hidden_actual: H used for the mean, when the input is padded.
        output_in_sbuf: shared_local buffer; SBUF [H0,T,H1] by default, else HBM [T,H].

    Returns:
        (shared_local, normed_sb) -- the shared partial, and the normed tile for the routed slice.
    """
    normed_sb = post_attn_rmsnorm_compose(
        hidden,
        gamma,
        eps,
        hidden_actual,
        single_core_forced=NORM_SINGLE_CORE_FORCED,
    )
    shared_local = shared_expert_compose(
        normed_sb, gate_w, up_w, down_w, output_in_sbuf=output_in_sbuf
    )
    return shared_local, normed_sb
