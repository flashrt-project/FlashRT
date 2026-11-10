# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Selective-expert MoE TKG: processes only top-K experts per token.

Vendored from nkilib selective_expert_impl.py with two megakernel-specific
additions: (1) a `name_prefix` kwarg so per-layer MoE invocations don't
collide on op names (the function builds its own internal SbufManager so
the megakernel's outer prefix doesn't reach here), and (2) a 2-slot
cross-expert prefetch ring that issues expert k+1's gate/up DMAs while
expert k's matmul runs.
"""

import os as _os

import nki.isa as nisa
import nki.language as nl

# Fused gate+up dma_copy — halves descriptor count (~13 µs win at bs=1).
_MOE_FUSION_ENABLED = _os.environ.get("NKI_MOE_ENABLE_FUSION", "0") == "1"

# Disables the cross-expert gate_up prefetch ring.
_MOE_LEGACY_GATE_UP_WEIGHT_LOAD = _os.environ.get("NKI_MOE_LEGACY_GATE_UP_WEIGHT_LOAD", "0") == "1"

# Phase 1 indirect-DMA fusion: combines the fused gate+up TensorView (one DMA per
# expert spanning the contiguous [H, 2*I] HBM slab) with the cross-expert prefetch
# ring (k+1 loads while expert k computes). On Qwen3-30B-A3B (trn3, LNC=2) the
# non-fused path caps the contiguous-inner per descriptor at I*bf16=384B because
# the second select(dim=1, GATE/UP) slices between the gate/up halves; the fused
# view's inner is 2*I*bf16=768B, halving descriptor count and doubling packet size.
# Independent of NKI_MOE_ENABLE_FUSION; implies the fused tensor view AND keeps
# the prefetch ring active. When off, behavior is unchanged from the baseline.
_MOE_INDIRECT_DMA_FUSION = _os.environ.get("NKI_MOE_INDIRECT_DMA_FUSION", "0") == "1"

# Force shard-on-H at T>1 (keeps output SBUF-resident; shard-on-T writes HBM).
_MOE_FORCE_SHARD_ON_H = _os.environ.get("NKI_MOE_FORCE_SHARD_ON_H", "0") == "1"

from nkilib.core.mlp.mlp_parameters import (
    MLPBiasParameters,
    MLPParameters,
    MLPQuantizationParameters,
)

# MLP utils
from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import (
    emit_hoisted_gate_up_dma,
    process_gate_up_projection,
)
from nkilib.core.mlp.mlp_tkg.mlp_tkg_utils import input_norm_load, transpose_store

# common utils
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import ExpertAffinityScaleMode, GateUpDim
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.tensor_view import TensorView
from .moe_tkg_utils import (
    broadcast_token_affinity,
    gather_expert_affinities,
    reshape_scale_for_mlp,
)


def _build_gate_view(initial_gate_proj_weights_tensor, expert_id_scalar_offset):
    """Construct the gate-projection weight TensorView for a given expert offset."""
    return (
        TensorView(initial_gate_proj_weights_tensor)
        .select(dim=0, index=expert_id_scalar_offset)
        .select(dim=1, index=GateUpDim.GATE.value)
    )


def _build_up_view(initial_up_proj_weights_tensor, expert_id_scalar_offset):
    """Construct the up-projection weight TensorView for a given expert offset."""
    return (
        TensorView(initial_up_proj_weights_tensor)
        .select(dim=0, index=expert_id_scalar_offset)
        .select(dim=1, index=GateUpDim.UP.value)
    )


def _build_gate_up_fused_view(initial_gate_proj_weights_tensor, expert_id_scalar_offset):
    """Construct the FUSED gate+up TensorView for a given expert offset.

    Skips the per-projection ``select(dim=1, GATE/UP)`` and flattens the (2, I)
    inner pair to ``2*I`` so the free dim is contiguous in HBM. Used by the
    indirect-DMA fusion path; the resulting view has shape ``[H, 2*I]`` with
    dynamic access (via the expert scalar_offset).
    """
    return (
        TensorView(initial_gate_proj_weights_tensor)
        .select(dim=0, index=expert_id_scalar_offset)
        .flatten_dims(start_dim=1, end_dim=2)
    )


def _selective_expert_moe_tkg(
    params: MLPParameters,
    output: nl.ndarray,
    name_prefix: str = "",
) -> nl.ndarray:
    """
    Selective-expert Mixture of Experts (MoE) kernel for token generation (TKG).

    Processes only the top-K selected experts for each token, computing MLP projections
    for the selected experts and accumulating results weighted by expert affinities.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.ndarray): Output tensor to store the final result.

    Returns:
        output (nl.ndarray): Output tensor with accumulated expert results.

    Notes:
        - Processes tokens sequentially, experts selectively based on top-K indices
        - Uses TensorView for dynamic expert weight selection
        - Column tiling is disabled for this implementation
        - SBUF I/O mode is supported

    Pseudocode:
        input_sb[H0, T, H1] = normalize(hidden_tensor[T, H])
        output_temp[H0, H1_shard, T] = zeros()

        # Gather expert affinities for efficient access
        gathered_affinities = gather_expert_affinities(expert_affinities, expert_index)

        for token_idx in range(T):
            token_affinities = broadcast_token_affinity(gathered_affinities, token_idx)

            for k in range(K):  # top-K experts
                expert_idx = expert_index[token_idx, k]
                gate_w[I, H], up_w[I, H], down_w[H, I] = weights[expert_idx]

                # Gate-Up projection: act_fn(gate(x)) * up(x)
                gate_up[I0, I1, 1] = gate_up_proj(input_sb[H0, token_idx:token_idx+1, H1], gate_w, up_w)

                # Down projection
                down[H0, H1_shard] = down_proj(gate_up[I0, I1, 1], down_w)

                # Scale by affinity if POST_SCALE
                if affinity_scaling_mode == POST_SCALE:
                    down[H0, H1_shard] *= token_affinities[k]

                # Accumulate results for this token
                if k == 0:
                    output_temp[H0, H1_shard, token_idx] = down[H0, H1_shard]
                else:
                    output_temp[H0, H1_shard, token_idx] += down[H0, H1_shard]

        output[T, H] = transpose(output_temp[H0, H1_shard, T])
    """

    # Check if input is already in SBUF
    hidden_in_sbuf = params.hidden_tensor.buffer == nl.sbuf

    # TODO: Calibrate weight tile calculations and remove auto allocation workaround
    H = params.hidden_tensor.shape[-1]
    need_auto_alloc = H >= 16 * 1024 or hidden_in_sbuf
    sbm = SbufManager(0, 200 * 1024, get_logger("selective_expert_moe_tkg"), use_auto_alloc=need_auto_alloc)
    # Vendor edit: propagate caller's per-layer prefix into the internal sbm
    # so allocations don't collide across the megakernel's per-layer calls.
    sbm.set_name_prefix(name_prefix)
    sbm.open_scope()

    io_dtype = params.hidden_tensor.dtype
    expert_index_input = params.expert_params.expert_index
    expert_affinities = params.expert_params.expert_affinities
    gate_up_weights = params.gate_proj_weights_tensor

    program_sharding_info = get_verified_program_sharding_info("moe_tkg", (0, 1))
    num_shards = program_sharding_info[1]
    shard_id = program_sharding_info[2]

    T = expert_index_input.shape[0]
    I = gate_up_weights.shape[-1]
    shard_on_T = True

    # Disable shard_on_T when:
    # 1. T == 1: Only one token, no benefit from sharding on this dimension
    # 2. H * I >= 3072 * 1536: Big config has mlp tkg tile size calculation bug (NKL-1013)
    # 3. NKI_MOE_FORCE_SHARD_ON_H=1
    if T == 1 or H * I >= 3072 * 1536 or _MOE_FORCE_SHARD_ON_H:
        shard_on_T = False

    # For odd T, use ceiling division: core 0 gets T//2, core 1 gets T - T//2
    if shard_on_T:
        T_first_shard = T // num_shards
        T_second_shard = T - T_first_shard
        T_per_shard = T_first_shard if shard_id == 0 else T_second_shard
        T_offset = 0 if shard_id == 0 else T_first_shard
    else:
        T_per_shard = T
        T_offset = 0

    params.shard_on_h_disabled = shard_on_T
    dims = MLPTKGConstants.calculate_constants(params)

    # Load input in shape of [128(H0), T, H//128(H1)]
    if hidden_in_sbuf:
        # Input is already in SBUF
        input_sb = params.hidden_tensor
    else:
        # TODO: only load for local tokens
        input_sb = sbm.alloc_stack(
            [dims.H0, T, dims.H1_shard],
            dtype=io_dtype,
            buffer=nl.sbuf,
            name="input_sb",
        )
        input_norm_load(params.hidden_tensor, input_sb, params, dims, sbm=sbm)

    # Allocate SBUF location to accumulate output
    output_temp = sbm.alloc_stack(
        (dims.H0, dims.H1_shard, T_per_shard),
        dtype=io_dtype,
        name=f"temp_output_sbuf",
        buffer=nl.sbuf,
    )

    # Allocate SBUF locations for gate/up projection result, for each token
    gate_up_output = sbm.alloc_stack(
        (dims.I0, dims.num_total_128_tiles_per_I, dims.K),
        dtype=io_dtype,
        name=f"intermediate_state_sbuf",
        buffer=nl.sbuf,
    )

    # Allocate SBUF locations for down result
    down_output_list = []
    for expert_k_idx in range(dims.K):
        down_sb = sbm.alloc_stack(
            (dims.H0, dims.H1_shard), dtype=io_dtype, name=f"down_sbuf_{expert_k_idx}", buffer=nl.sbuf
        )
        down_output_list.append(down_sb)

    # Reshape gate_up weights from [E, H, 2, I] to [E, H, 2 * I]
    E, H, i_2, I = gate_up_weights.shape
    gate_up_weights = gate_up_weights.reshape((E, H, I * i_2))

    # Load expert index
    if expert_index_input.buffer == nl.sbuf:
        expert_idx = expert_index_input
    else:
        expert_idx = sbm.alloc_stack(
            (dims.T, dims.K),
            dtype=expert_index_input.dtype,
            name=f"expert_idx_sbuf",
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=expert_idx, src=expert_index_input[0 : dims.T, 0 : dims.K])  # indices have to be in SBUF

    expert_affinities_sb = sbm.alloc_stack(
        (dims._pmax, dims.E),
        dtype=expert_affinities.dtype,
        name=f"expert_affinities_sb",
        buffer=nl.sbuf,
    )
    # Load expert affinity
    if expert_affinities.buffer == nl.sbuf:
        nisa.memset(expert_affinities_sb, value=0.0)
        nisa.tensor_copy(dst=expert_affinities_sb[0 : dims.T, 0 : dims.E], src=expert_affinities)
    else:
        # Prefetch expertIndices (Up to 128 tokens input)
        nisa.dma_copy(
            dst=expert_affinities_sb[0 : dims.T, 0 : dims.E],
            src=expert_affinities[0 : dims.T, 0 : dims.E],
        )

    # Gather expert affinities using utility function
    gathered_affinities_sb = gather_expert_affinities(expert_affinities_sb, expert_idx, dims, sbm)
    params.use_tkg_gate_up_proj_column_tiling = False
    params.use_tkg_down_proj_column_tiling = False
    # Initialize fused gate+up weight view attribute so process_gate_up_projection can
    # access it unconditionally (it is reassigned per-expert inside the K loop below).
    params.gate_up_fused_weights_tensor = None

    initial_gate_proj_weights_tensor = params.gate_proj_weights_tensor
    initial_up_proj_weights_tensor = params.up_proj_weights_tensor
    initial_down_proj_weights_tensor = params.down_proj_weights_tensor

    initial_mlp_bias_params = params.bias_params
    initial_mlp_quant_params = params.quant_params

    memory_safe_degree = 2
    if shard_on_T:
        memory_safe_degree = 2 if dims.H * dims.I < 3072 * 1024 else 1

    # convert dims.T to 1 to compute output by each token
    dims.T = 1

    # Cross-expert prefetch ring: double-buffer the hoisted gate/up DMA across
    # two slots so expert k+1's weights load concurrently with expert k's
    # matmul. Mirrors the activation conditions of the in-function hoist in
    # process_gate_up_projection.
    #
    # Two variants:
    #   - use_prefetch_ring_split: legacy per-projection rings (gate slot + up slot,
    #     two DMAs per expert). Active when NKI_MOE_INDIRECT_DMA_FUSION=0 AND
    #     NKI_MOE_ENABLE_FUSION=0.
    #   - use_prefetch_ring_fused: ONE fused [H0, H1_shard, 2*I] slot per ring
    #     entry, one DMA per expert via the fused TensorView. Active when
    #     NKI_MOE_INDIRECT_DMA_FUSION=1 (Phase 1).
    use_prefetch_ring_common = (
        (not params.use_tkg_gate_up_proj_column_tiling)
        and (not params.skip_gate_proj)
        and (dims.K >= 2)
        and (not _MOE_LEGACY_GATE_UP_WEIGHT_LOAD)
    )
    use_prefetch_ring_fused = use_prefetch_ring_common and _MOE_INDIRECT_DMA_FUSION
    use_prefetch_ring_split = (
        use_prefetch_ring_common and (not _MOE_FUSION_ENABLED) and (not _MOE_INDIRECT_DMA_FUSION)
    )
    use_prefetch_ring = use_prefetch_ring_split or use_prefetch_ring_fused

    if use_prefetch_ring:
        # Single-I-shard path: entire [H0, H1_shard, I] (split) or
        # [H0, H1_shard, 2*I] (fused) maps to one hoisted tile.
        _h_offset = dims.H1_offset * dims.H0
        _shard_dim_hidden = (_h_offset, _h_offset + dims.H_per_shard)
        _shard_dim_intr_split = (0, dims.I)
        _shard_dim_intr_fused = (0, 2 * dims.I)
        _weight_dtype = (
            nl.float8_e4m3
            if str(initial_up_proj_weights_tensor.dtype) == "float8e4"
            else initial_up_proj_weights_tensor.dtype
        )

    for local_token_idx in range(T_per_shard):
        global_token_idx = local_token_idx + T_offset
        sbm.set_name_prefix(f"{name_prefix}T{global_token_idx}_")
        # Load Expert Affinities per Token using utility function
        expert_affinity_sb = sbm.alloc_stack(
            (dims._pmax, dims.K),
            dtype=expert_affinities.dtype,
            buffer=nl.sbuf,
            name=f"expert_affinity_sb",
        )
        broadcast_token_affinity(expert_affinity_sb, gathered_affinities_sb, global_token_idx, dims, sbm)

        # 2-slot ring per (gate, up). +2×(H0, H1_shard, I) bf16 vs the in-function
        # hoist; ~+37 KB/partition at gpt-oss shapes.
        prefetch_gate_slots = None
        prefetch_up_slots = None
        prefetch_gate_up_fused_slots = None
        if use_prefetch_ring_split:
            prefetch_gate_slots = []
            prefetch_up_slots = []
            for slot_idx in range(2):
                gate_slot = sbm.alloc_stack(
                    (dims.H0, dims.H1_shard, dims.I),
                    name=f"prefetch_gate_w_tile_slot{slot_idx}",
                    dtype=_weight_dtype,
                )
                up_slot = sbm.alloc_stack(
                    (dims.H0, dims.H1_shard, dims.I),
                    name=f"prefetch_up_w_tile_slot{slot_idx}",
                    dtype=_weight_dtype,
                )
                prefetch_gate_slots.append(gate_slot)
                prefetch_up_slots.append(up_slot)

            # Prime slot 0 with expert k=0 before the K-loop.
            sbm.set_name_prefix(f"{name_prefix}T{global_token_idx}_prefetch_K0_")
            _expert0_offset = expert_idx.ap(
                pattern=[[dims.K, 1], [1, 1]], offset=global_token_idx * dims.K + 0
            )
            _gate0_view = _build_gate_view(initial_gate_proj_weights_tensor, _expert0_offset)
            _up0_view = _build_up_view(initial_up_proj_weights_tensor, _expert0_offset)
            emit_hoisted_gate_up_dma(
                unsharded_weight=_gate0_view,
                hoisted_weight=prefetch_gate_slots[0],
                dims=dims,
                shard_dim_hidden=_shard_dim_hidden,
                shard_dim_intr=_shard_dim_intr_split,
                dge_mode=nisa.dge_mode.hwdge,
            )
            emit_hoisted_gate_up_dma(
                unsharded_weight=_up0_view,
                hoisted_weight=prefetch_up_slots[0],
                dims=dims,
                shard_dim_hidden=_shard_dim_hidden,
                shard_dim_intr=_shard_dim_intr_split,
                dge_mode=nisa.dge_mode.swdge,
            )

        if use_prefetch_ring_fused:
            # ONE fused [H0, H1_shard, 2*I] slot per ring entry — gate+up in
            # the same SBUF tile, one DMA per expert via the fused view.
            prefetch_gate_up_fused_slots = []
            for slot_idx in range(2):
                fused_slot = sbm.alloc_stack(
                    (dims.H0, dims.H1_shard, 2 * dims.I),
                    name=f"prefetch_gate_up_fused_w_tile_slot{slot_idx}",
                    dtype=_weight_dtype,
                )
                prefetch_gate_up_fused_slots.append(fused_slot)

            # Prime slot 0 with expert k=0 before the K-loop.
            sbm.set_name_prefix(f"{name_prefix}T{global_token_idx}_prefetch_K0_")
            _expert0_offset = expert_idx.ap(
                pattern=[[dims.K, 1], [1, 1]], offset=global_token_idx * dims.K + 0
            )
            _gate_up0_view = _build_gate_up_fused_view(
                initial_gate_proj_weights_tensor, _expert0_offset
            )
            # The fused view has dynamic access; emit_hoisted_gate_up_dma forces
            # dge_mode=unknown (==0) internally for indirect patterns. The
            # passed-in mode is ignored for indirect views.
            emit_hoisted_gate_up_dma(
                unsharded_weight=_gate_up0_view,
                hoisted_weight=prefetch_gate_up_fused_slots[0],
                dims=dims,
                shard_dim_hidden=_shard_dim_hidden,
                shard_dim_intr=_shard_dim_intr_fused,
                dge_mode=None,
            )

        sbm.open_scope(interleave_degree=memory_safe_degree)
        for expert_k_idx in range(dims.K):
            sbm.set_name_prefix(f"{name_prefix}T{global_token_idx}_K{expert_k_idx}_")
            # Gate Up projection

            # Change back to scalar_offset=expert_idx[global_token_idx, expert_k_idx], after NKI-333 is fixed
            expert_id_scalar_offset = expert_idx.ap(
                pattern=[[dims.K, 1], [1, 1]], offset=global_token_idx * dims.K + expert_k_idx
            )
            params.gate_proj_weights_tensor = _build_gate_view(
                initial_gate_proj_weights_tensor, expert_id_scalar_offset
            )

            params.up_proj_weights_tensor = _build_up_view(
                initial_up_proj_weights_tensor, expert_id_scalar_offset
            )

            params.down_proj_weights_tensor = TensorView(initial_down_proj_weights_tensor).select(
                dim=0, index=expert_id_scalar_offset
            )

            # Fused gate+up weight view: same expert select but skips the gate/up (dim=1) select,
            # then flattens (2, I) -> 2*I so the inner free dim is contiguous in HBM. Consumed by
            # process_gate_up_projection for a single fused DMA per HTile (halves descriptor count
            # and doubles the inner contiguous chunk vs two separate gate+up loads).
            # NOTE: When NKI_MOE_INDIRECT_DMA_FUSION=1 we drive the fused view via the
            # prefetch-ring path (pre_loaded_hoisted_gate_up_fused below) instead — set
            # params.gate_up_fused_weights_tensor=None to suppress the in-function per-HTile
            # fused alloc/load inside process_gate_up_projection.
            if _MOE_FUSION_ENABLED and not _MOE_INDIRECT_DMA_FUSION:
                params.gate_up_fused_weights_tensor = _build_gate_up_fused_view(
                    initial_gate_proj_weights_tensor, expert_id_scalar_offset
                )
            else:
                params.gate_up_fused_weights_tensor = None

            gate_proj_bias_tensor_view = None
            up_proj_bias_tensor_view = None
            down_proj_bias_tensor_view = None
            if initial_mlp_bias_params.gate_proj_bias_tensor != None:
                gate_proj_bias_tensor_view = (
                    TensorView(initial_mlp_bias_params.gate_proj_bias_tensor)
                    .select(dim=0, index=expert_id_scalar_offset)
                    .select(dim=0, index=GateUpDim.GATE.value)
                )

            if initial_mlp_bias_params.up_proj_bias_tensor != None:
                up_proj_bias_tensor_view = (
                    TensorView(initial_mlp_bias_params.up_proj_bias_tensor)
                    .select(dim=0, index=expert_id_scalar_offset)
                    .select(dim=0, index=GateUpDim.UP.value)
                )

            if initial_mlp_bias_params.down_proj_bias_tensor != None:
                down_proj_bias_tensor_view = TensorView(initial_mlp_bias_params.down_proj_bias_tensor).select(
                    dim=0, index=expert_id_scalar_offset
                )

            params.bias_params = MLPBiasParameters(
                gate_proj_bias_tensor=gate_proj_bias_tensor_view,
                up_proj_bias_tensor=up_proj_bias_tensor_view,
                down_proj_bias_tensor=down_proj_bias_tensor_view,
            )

            params.quant_params = _select_quant_scales(
                initial_mlp_quant_params,
                expert_id_scalar_offset,
            )

            # Prefetch expert k+1 into the other ring slot before k's matmul
            # runs — DMA queue works on k+1 while TE consumes k.
            if use_prefetch_ring and expert_k_idx + 1 < dims.K:
                next_k = expert_k_idx + 1
                next_slot = next_k % 2
                _saved_prefix = sbm.get_name_prefix()
                sbm.set_name_prefix(f"{name_prefix}T{global_token_idx}_prefetch_K{next_k}_")
                next_expert_offset = expert_idx.ap(
                    pattern=[[dims.K, 1], [1, 1]], offset=global_token_idx * dims.K + next_k
                )
                if use_prefetch_ring_split:
                    next_gate_view = _build_gate_view(initial_gate_proj_weights_tensor, next_expert_offset)
                    next_up_view = _build_up_view(initial_up_proj_weights_tensor, next_expert_offset)
                    emit_hoisted_gate_up_dma(
                        unsharded_weight=next_gate_view,
                        hoisted_weight=prefetch_gate_slots[next_slot],
                        dims=dims,
                        shard_dim_hidden=_shard_dim_hidden,
                        shard_dim_intr=_shard_dim_intr_split,
                        dge_mode=nisa.dge_mode.hwdge,
                    )
                    emit_hoisted_gate_up_dma(
                        unsharded_weight=next_up_view,
                        hoisted_weight=prefetch_up_slots[next_slot],
                        dims=dims,
                        shard_dim_hidden=_shard_dim_hidden,
                        shard_dim_intr=_shard_dim_intr_split,
                        dge_mode=nisa.dge_mode.swdge,
                    )
                else:  # use_prefetch_ring_fused
                    next_fused_view = _build_gate_up_fused_view(
                        initial_gate_proj_weights_tensor, next_expert_offset
                    )
                    emit_hoisted_gate_up_dma(
                        unsharded_weight=next_fused_view,
                        hoisted_weight=prefetch_gate_up_fused_slots[next_slot],
                        dims=dims,
                        shard_dim_hidden=_shard_dim_hidden,
                        shard_dim_intr=_shard_dim_intr_fused,
                        dge_mode=None,
                    )
                sbm.set_name_prefix(_saved_prefix)

            if use_prefetch_ring_split:
                _cur_slot = expert_k_idx % 2
                _pre_loaded_gate = prefetch_gate_slots[_cur_slot]
                _pre_loaded_up = prefetch_up_slots[_cur_slot]
                _pre_loaded_gate_up_fused = None
            elif use_prefetch_ring_fused:
                _cur_slot = expert_k_idx % 2
                _pre_loaded_gate = None
                _pre_loaded_up = None
                _pre_loaded_gate_up_fused = prefetch_gate_up_fused_slots[_cur_slot]
            else:
                _pre_loaded_gate = None
                _pre_loaded_up = None
                _pre_loaded_gate_up_fused = None

            gate_tile_info = process_gate_up_projection(
                hidden=input_sb[:, global_token_idx : global_token_idx + 1, :],
                output=gate_up_output[:, :, expert_k_idx : expert_k_idx + 1],
                params=params,
                dims=dims,
                sbm=sbm,
                pre_loaded_hoisted_gate=_pre_loaded_gate,
                pre_loaded_hoisted_up=_pre_loaded_up,
                pre_loaded_hoisted_gate_up_fused=_pre_loaded_gate_up_fused,
            )

            # Down projection
            down_sb = down_output_list[expert_k_idx]
            process_down_projection(
                hidden=gate_up_output[:, :, expert_k_idx : expert_k_idx + 1],
                output=down_sb,
                params=params,
                dims=dims,
                gate_tile_info=gate_tile_info,
                sbm=sbm,
            )

            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                # Apply affinity and accumulate to SB
                nisa.tensor_scalar(
                    dst=down_sb,
                    data=down_sb,
                    op0=nl.multiply,
                    operand0=expert_affinity_sb[:, expert_k_idx],
                )
            if expert_k_idx == 0:
                nisa.tensor_copy(dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx], src=down_sb)
            else:
                nisa.tensor_tensor(
                    dst=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data1=output_temp[0 : dims.H0, 0 : dims.H1_shard, local_token_idx],
                    data2=down_sb,
                    op=nl.add,
                )

            sbm.increment_section()
        sbm.close_scope()

    # Save output result
    sbm.set_name_prefix(name_prefix)

    dims.T = T_per_shard

    # Store output
    if output.buffer == nl.sbuf:
        # Transpose output_temp [H0, H1_shard, T_per_shard] -> [H0, T, H1_shard] for SBUF output
        for h1_idx in range(dims.H1_shard):
            nisa.tensor_copy(dst=output[:, T_offset : T_offset + T_per_shard, h1_idx], src=output_temp[:, h1_idx, :])
    else:
        transpose_store(output_temp, output, dims, params.output_dtype, sbm, T_offset)

    sbm.close_scope()
    return output


def _select_quant_scales(quant_params: MLPQuantizationParameters, expert_id_offset: nl.ndarray):
    """
    Select and reshape quantization scales for a specific expert.

    Args:
        quant_params (MLPQuantizationParameters): Quantization parameters.
        expert_id_offset (nl.ndarray): Expert ID offset for selecting scales.

    Returns:
        MLPQuantizationParameters: Quantization parameters with scales for the specified expert.
    """
    gate_w_scale_view = None
    up_w_scale_view = None
    down_w_scale_view = None
    gate_up_in_scale_view = None
    down_in_scale_view = None

    if quant_params.gate_w_scale != None:
        gate_w_scale_view = (
            TensorView(quant_params.gate_w_scale)
            .select(dim=0, index=expert_id_offset)
            .select(dim=0, index=GateUpDim.GATE.value)
        )
        gate_w_scale_view = reshape_scale_for_mlp(gate_w_scale_view)

    if quant_params.up_w_scale != None:
        up_w_scale_view = (
            TensorView(quant_params.up_w_scale)
            .select(dim=0, index=expert_id_offset)
            .select(dim=0, index=GateUpDim.UP.value)
        )
        up_w_scale_view = reshape_scale_for_mlp(up_w_scale_view)

    if quant_params.down_w_scale != None:
        down_w_scale_view = TensorView(quant_params.down_w_scale).select(dim=0, index=expert_id_offset)
        down_w_scale_view = reshape_scale_for_mlp(down_w_scale_view)

    if quant_params.gate_up_in_scale != None:
        gate_up_in_scale_view = TensorView(quant_params.gate_up_in_scale).select(dim=0, index=expert_id_offset)
        gate_up_in_scale_view = reshape_scale_for_mlp(gate_up_in_scale_view)

    if quant_params.down_in_scale != None:
        down_in_scale_view = TensorView(quant_params.down_in_scale).select(dim=0, index=expert_id_offset)
        down_in_scale_view = reshape_scale_for_mlp(down_in_scale_view)

    return MLPQuantizationParameters(
        quantization_type=quant_params.quantization_type,
        gate_w_scale=gate_w_scale_view,
        up_w_scale=up_w_scale_view,
        down_w_scale=down_w_scale_view,
        gate_up_in_scale=gate_up_in_scale_view,
        down_in_scale=down_in_scale_view,
        clipping_bound=quant_params.clipping_bound,
    )
