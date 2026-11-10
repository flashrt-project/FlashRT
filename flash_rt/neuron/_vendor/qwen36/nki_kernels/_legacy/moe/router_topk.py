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

"""
This file implements the router top-K kernel for Mixture of Experts (MoE) models.
It computes router logits, applies activation functions, and performs top-K selection
with expert affinity scattering.

Vendored from nkilib/core/router_topk/router_topk.py @ nkilib 0.1.0+gd4920de.
Edits relative to upstream:
  1. Relative `..utils.<x>` imports rewritten to absolute
     `nkilib.core.utils.<x>`.
  2. New `name_prefix: str = ""` kwarg threaded into `router_topk`. The
     kernel allocates several SBUF buffers via direct `nl.ndarray(name=...)`
     and stores results via direct `nisa.dma_copy(name=...)` — these
     bypass any sbm prefix machinery. The `name_prefix` is prepended to
     each such literal so per-layer calls in the megakernel produce
     unique op names. The megakernel passes
     `name_prefix=f"L{layer_idx}_moe_"`.
  3. The IOTA op-name on line ~999 (`f"IOTA_offset_{offset}"`) is also
     prefixed for the same reason.
"""

import neuronxcc.nki.typing as nt
import nki.isa as nisa
import nki.language as nl
from nki.isa import core_barrier, engine, reduce_cmd

from nkilib.core.utils import (
    common_types,
    cross_partition_copy,
    kernel_helpers,
    stream_shuffle_broadcast,
    tensor_view,
    tiled_range,
)
from nkilib.core.utils.kernel_assert import kernel_assert

P_MAX = 128
F_MAX = 512
ST_F_MAX = 128  # State tile free dimension maximum
PE_COLUMN_TILE_32 = 32  # PE array column tile size for T <= 32
PE_COLUMN_TILE_64 = 64  # PE array column tile size for 32 < T <= 64
PE_COLUMN_TILE_128 = 128  # PE array column tile size (full width, disables tiling)


# TODO: the new FE is having issue with using Enum in megakernel setting
# thus we work around it by using explicit constants
XHBMLayout_H_T__0 = 0
XHBMLayout_T_H__1 = 1

XSBLayout_tp102__0 = 0
XSBLayout_tp2013__1 = 1
XSBLayout_tp201__2 = 2
XSBLayout__128_Hdiv128_T__3 = 3


def router_topk(
    x: nl.ndarray,
    w: nl.ndarray,
    w_bias: nl.ndarray,
    router_logits: nt.mutable_tensor,
    expert_affinities: nt.mutable_tensor,
    expert_index: nt.mutable_tensor,
    act_fn: common_types.RouterActFnType,
    k: int,
    x_hbm_layout: XHBMLayout_H_T__0,
    x_sb_layout: XSBLayout_tp102__0,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    use_column_tiling: bool = False,
    use_indirect_dma_scatter: bool = False,
    return_eager_affi: bool = False,
    use_PE_broadcast_w_bias: bool = False,
    shard_on_tokens: bool = False,
    skip_store_expert_index: bool = False,
    skip_store_router_logits: bool = False,
    name_prefix: str = "",
):
    """
    Router top-K kernel for Mixture of Experts (MoE) models.

    Computes router logits (x @ w + bias), applies activation functions, performs top-K selection,
    and scatters expert affinities. Supports multiple layout configurations, sharding strategies,
    and optimization modes including column tiling and indirect DMA scatter (see the Args section).

    Intended for token counts T <= 2048, expert counts E <= 512, hidden dimensions H that are
    multiples of 128, and K <= 8 top experts per token. Optimized for both context encoding (CTE)
    with larger T and token generation (TKG) with T <= 128.

    Dimensions:
        T: Total number of tokens
        H: Hidden dimension size
        E: Number of experts
        K: Number of top experts to select per token

    Args:
        x (nl.ndarray): Input tensor. Buffer type is auto-detected.
                        If in HBM: [H, T] or [T, H] depending on x_hbm_layout.
                        If in SBUF: a permutation of [128, T, H/128] depending on x_sb_layout.
        w (nl.ndarray): Weight tensor [H, E] in HBM
        w_bias (nl.ndarray): Optional bias tensor [1, E] or [E] in HBM
        router_logits (nt.mutable_tensor): Output router logits [T, E] in HBM
        expert_affinities (nt.mutable_tensor): Output expert affinities [T, E] in HBM or SBUF.
                        Buffer type is auto-detected.
        expert_index (nt.mutable_tensor): Output expert indices [T, K] in HBM or SBUF.
                        Buffer type is auto-detected.
        act_fn (common_types.RouterActFnType): Activation function (SOFTMAX or SIGMOID)
        k (int): Number of top experts to select (must be <= 8)
        x_hbm_layout (int): Layout of x in HBM (0=[H,T], 1=[T,H])
        x_sb_layout (int): Layout of x in SBUF (0-3, see router_topk_input_x_load for details)
        router_pre_norm (bool): If True, apply activation before top-K (ACT1 pipeline)
        norm_topk_prob (bool): If True, normalize top-K probabilities with L1 norm
        use_column_tiling (bool): Enable PE array column tiling for small T
        use_indirect_dma_scatter (bool): Use indirect DMA for expert affinity scatter
        return_eager_affi (bool): If True, return top-K affinities in addition to scattered
        use_PE_broadcast_w_bias (bool): Use tensor engine for bias broadcast
        shard_on_tokens (bool): Enable LNC sharding across token dimension
        skip_store_expert_index (bool): Skip storing expert indices to HBM
        skip_store_router_logits (bool): Skip storing router logits to HBM

    Returns:
        outputs (list): [router_logits, expert_index, expert_affinities, optional: expert_affinities_topk]

    Notes:
        - K must be <= 8
        - E must be <= 512 (gemm_moving_fmax)
        - H must be a multiple of 128
        - SIGMOID activation requires use_indirect_dma_scatter=True
        - With use_indirect_dma_scatter, T must be <= 128 or multiple of 128
        - shard_on_tokens requires n_prgs > 1 and T divisible by 2
        - SBUF outputs require T <= PE_COLUMN_TILE_128
        - When T <= 32 with SBUF outputs, shard_on_tokens is disabled (each core processes all T)
        - Buffer types for x, expert_affinities, and expert_index are auto-detected via .buffer attribute

    Pseudocode:
        # Load inputs
        x_sb = load_x_from_hbm_or_use_sbuf(x)
        w_sb = load_w_from_hbm(w)
        bias_sb = load_and_broadcast_bias(w_bias) if w_bias else None

        # Compute router logits
        for t_tile in range(num_t_tiles):
            router_logits_psum = zeros()
            for h_tile in range(num_h_tiles):
                x_tile = extract_tile(x_sb, h_tile, t_tile)
                w_tile = extract_tile(w_sb, h_tile)
                router_logits_psum += matmul(x_tile, w_tile)
            router_logits_sb[t_tile] = router_logits_psum + bias_sb

        # Optional ACT1 (pre-norm activation)
        if router_pre_norm:
            expert_affinities_full = activation(router_logits_sb, act_fn)
            topk_input = expert_affinities_full
        else:
            topk_input = router_logits_sb

        # Top-K selection
        for t_tile in range(num_t_tiles):
            top8_values = max8(topk_input[t_tile])
            router_logits_topk[t_tile] = top8_values[:k]
            router_indexes_topk[t_tile] = find_index8(topk_input[t_tile], top8_values)[:k]

        # Optional ACT2 (post-topk activation) and Norm
        if not router_pre_norm:
            expert_affinities_topk = activation(router_logits_topk, act_fn)
        if norm_topk_prob:
            sum_topk = reduce_sum(expert_affinities_topk, axis=expert_dim)
            expert_affinities_topk = expert_affinities_topk / sum_topk

        # Scatter to full expert dimension
        if use_indirect_dma_scatter:
            scatter_indirect_dma(expert_affinities, expert_affinities_topk, router_indexes_topk)
        else:
            scatter_one_hot(expert_affinities, expert_affinities_topk, router_indexes_topk)

        return [router_logits, expert_index, expert_affinities]
    """
    # The following code is split into sections correlated to the pseudo-code above

    """Get input shapes and validate."""

    # Validate layout
    kernel_assert(x_hbm_layout in (0, 1), f"x_hbm_layout must be 0 or 1, got {x_hbm_layout}")
    kernel_assert(x_sb_layout in (0, 1, 2), f"x_sb_layout must be 0, 1 or 2, got {x_sb_layout}")

    # Detect buffer types from tensor attributes
    x_input_in_sbuf = x.buffer == nl.sbuf
    expert_affin_in_sb = expert_affinities.buffer == nl.sbuf
    expert_index_in_sb = expert_index.buffer == nl.sbuf

    T: int = -1
    H: int = -1

    # Validate 'x' shape
    H = w.shape[0]
    if not x_input_in_sbuf:  # 'x' in HBM
        kernel_assert(len(x.shape) == 2, f"Expect the input 'x' HBM tensor to be 2-D, got shape of {x.shape}")
        x_H: int = -1
        if x_hbm_layout == XHBMLayout_H_T__0:  # [H,T]
            x_H, T = x.shape
        elif x_hbm_layout == XHBMLayout_T_H__1:  # [T,H]
            T, x_H = x.shape
        else:
            kernel_assert(False, f"Unsupported x_hbm_layout: {x_hbm_layout}. Expected 0,1")

        kernel_assert(x_H == H, f"x shape {x.shape} has an H: {x_H}) that mismatches w's H {H}. w.shape: {w.shape}")
    else:  # 'x' in SB with layout [128, T, H/128]
        kernel_assert(len(x.shape) == 3, f"Expect the input 'x' SBUF tensor to be 3-D, got shape of {x.shape}")
        kernel_assert(
            x.shape[0] == P_MAX, f"Expect input x.shape[0] to be partition-dim max {P_MAX} but got {x.shape[0]}"
        )
        # Sanity check 'x' dim-2 (3rd dim) is H/pmax
        kernel_assert(x.shape[2] == H // P_MAX, f"Expect x.shape[2] to be H/pmax ({H // P_MAX}) but got {x.shape[2]}")
        T = x.shape[1]

    E = w.shape[1]

    # Validate LNC sharding config -- it's only supported in certain scenarios.
    _, n_prgs, prg_id = kernel_helpers.get_verified_program_sharding_info("router_topk_kernel", (0, 1), 2)

    if shard_on_tokens:
        kernel_assert(n_prgs > 1, f"LNC sharding only supported with n_prgs>1, got {shard_on_tokens=} {n_prgs=}")
        T_first_shard = T // n_prgs
        T_second_shard = T - T_first_shard
        T_local = T_first_shard if prg_id == 0 else T_second_shard
        T_offset = 0 if prg_id == 0 else T_first_shard
        other_T_local = T_second_shard if prg_id == 0 else T_first_shard
        # Beta 3: sendrecv requires src and dst to have matching partition dims.
        # Use max of T_local and other_T_local for sendrecv buffer allocation.
        T_sendrecv_size = max(T_local, other_T_local)
    else:  # If not sharding, process the full T on this core.
        T_local = T
        T_offset = 0
        other_T_local = T_local
        T_sendrecv_size = T_local

    # Validate when indirect-DMA scatter must be used
    # SIGMOID with router_pre_norm=False requires indirect-DMA scatter as one-hot scatter ACT2 path doesn't support SIGMOID
    # TODO: this could be supported, however.
    kernel_assert(
        not (act_fn == common_types.RouterActFnType.SIGMOID and not use_indirect_dma_scatter and not router_pre_norm),
        f"SIGMOID activation with router_pre_norm=False requires use_indirect_dma_scatter=True",
    )

    # If using indirect-DMA, T must be a multiple of 128
    if use_indirect_dma_scatter:
        kernel_assert(
            T_local <= PE_COLUMN_TILE_128 or T_local % PE_COLUMN_TILE_128 == 0,
            (f"T_local ({T_local}) must be <= 128 or a multiple of 128 with {use_indirect_dma_scatter=}. "),
        )
        T_pad = 0
    else:
        T_pad = (
            0 if T <= PE_COLUMN_TILE_128 else kernel_helpers.div_ceil(T_local, P_MAX) * P_MAX - T_local
        )  # T_pad is the amount, on the T dimension, required to pad to pmax.

    # E is expected to be less than max moving free-dim
    kernel_assert(E <= F_MAX, f"E ({E}) must be <= gemm_moving_fmax ({F_MAX})")

    # skip_store_router_logits is not yet supported due to a compiler limitation:
    # the compiler requires every mutable_tensor to have at least one store operation,
    # but router_logits is always returned as a mutable_tensor output.
    kernel_assert(
        not skip_store_router_logits,
        "skip_store_router_logits=True is not currently supported due to a compiler limitation "
        "(NCC_IGCA090: mutable_tensor must have at least one store). "
        "Set skip_store_router_logits=False as a workaround.",
    )

    # Check 'w_bias'
    has_bias = w_bias != None
    if has_bias:
        # w_bias should be [1,E], this will assert if we cannot do the reshape
        w_bias = w_bias.reshape((1, E))

    # We tile on H, which is the contraction dim = partition dim.
    # And we tile on T.
    kernel_assert(H % P_MAX == 0, f"H ({H}) must be a multiple of pmax ({P_MAX})")
    num_h_tiles = H // P_MAX
    num_t_tiles_total = kernel_helpers.div_ceil(T, ST_F_MAX)

    num_t_tiles = kernel_helpers.div_ceil(
        T_local, ST_F_MAX
    )  # returns minimum value of 1 (i.e. we will have at least 1 tile)
    num_t_whole_tiles = T_local // ST_F_MAX
    t_remainder = T_local % ST_F_MAX
    t_tile_size = T_local if (T_local < ST_F_MAX) else ST_F_MAX  # Used when num_t_tiles==1 (i.e. not tiling)

    # Check that input data types match
    kernel_assert(x.dtype == w.dtype, f"x dtype ({x.dtype}) must match w dtype ({w.dtype})")

    input_dtype = x.dtype

    """Compute router-logits (x.T @ w) + w_bias."""

    """
    TensorEngine/PEArray column tiling setup.
    
    The 'x' tensor's T is on the free-dim in SBUF. If T<128 then some of the PEArray columns
    are unused. Hence we engage the column-tiling feature to split the array in multiple tiles
    (column-wise), each of which can execute an independent matmul in parallel.
    """

    pe_array_column_tiling_size = None
    # Column tile width must be multiple of 32 so round up to the nearest 32 depending on T.
    if use_column_tiling:
        if T_local <= PE_COLUMN_TILE_32:
            pe_array_column_tiling_size = PE_COLUMN_TILE_32
        elif T_local <= PE_COLUMN_TILE_64:
            pe_array_column_tiling_size = PE_COLUMN_TILE_64
        else:
            pe_array_column_tiling_size = PE_COLUMN_TILE_128
    else:
        pe_array_column_tiling_size = (
            128  # Essentially disables column-tiling by setting the column-tile-size to the full width
        )

    num_pe_array_column_tiles = PE_COLUMN_TILE_128 // pe_array_column_tiling_size
    tile_size = (P_MAX, pe_array_column_tiling_size)

    t_p_dim = T_local if num_t_tiles == 1 else P_MAX

    # Result of the x@w matmul, shape [t_p_dim,num_t_tiles,E]
    # This will be oversized if T>pmax and T is not a multiple of pmax.
    router_logits_sb = nl.ndarray((t_p_dim, num_t_tiles, E), dtype=nl.float32, buffer=nl.sbuf)

    """
    Internal x_sb_layout defaults to x_sb_layout, which makes sense when the 'x' input is in SBUF.
    
    But when we load 'x' from HBM, we override internal_x_sb_layout to be the SBUF layout we load into.
    To be clear, when loading from HBM, the SB layout is chosen by this kernel, below. It is *not*
    specified by the user via x_sb_layout.
    """
    internal_x_sb_layout = x_sb_layout

    """Load from HBM."""

    # Load 'x'. See the comments in router_topk_input_x_load() to understand the HBM-to-SBUF layouts/access patterns.
    x_sb = None
    if not x_input_in_sbuf:  # Obviously only load if 'x' is in HBM.
        if x_hbm_layout == XHBMLayout_H_T__0:
            internal_x_sb_layout = XSBLayout__128_Hdiv128_T__3
        else:  # x_hbm_layout==1
            internal_x_sb_layout = XSBLayout_tp102__0
        x_sb = router_topk_input_x_load(x, hbm_layout=x_hbm_layout, sb_layout=internal_x_sb_layout)
    else:  # 'x' is already in SB.
        x_sb = x

    # Load 'w'. See the comments in router_topk_input_w_load() to understand the HBM-to-SBUF
    # layouts/access patterns as they depend on the 'x' layout.
    w_sb = router_topk_input_w_load(w, x_sb_layout=internal_x_sb_layout, name=f'{name_prefix}router_w_sb')

    # Load optional bias
    router_logits_bias_broadcasted_sb = None
    if has_bias:
        """
        Load the bias vector, shape [1,E]. Broadcast into a tensor of shape [t_tile_size,E]
        (i.e. every row contains a copy of w_bias).
        """
        router_logits_bias_vector_sb = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf, name=f'{name_prefix}router_bias_sb')
        router_logits_bias_broadcasted_sb = nl.ndarray(
            shape=(t_tile_size, E), dtype=router_logits_bias_vector_sb.dtype, buffer=nl.sbuf
        )
        # router_logits_bias_vector_sb = nl.load(w_bias)
        nisa.dma_copy(
            dst=tensor_view.TensorView(router_logits_bias_vector_sb).get_view(),
            src=tensor_view.TensorView(w_bias).get_view(),
        )
        if use_PE_broadcast_w_bias:  # Use TensorE/matmul to do the broadcast
            ones_mask = nl.ndarray(
                (1, t_tile_size), dtype=router_logits_bias_vector_sb.dtype, name=f"{name_prefix}ones_mask_sb", buffer=nl.sbuf
            )
            nisa.memset(dst=ones_mask, value=1.0, engine=engine.gpsimd)
            bias_bc_psum = nl.ndarray((t_tile_size, E), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=bias_bc_psum,
                stationary=ones_mask[...],
                moving=router_logits_bias_vector_sb[...],
                is_stationary_onezero=True,
            )
            nisa.tensor_copy(dst=router_logits_bias_broadcasted_sb, src=bias_bc_psum[...], engine=engine.scalar)
        else:
            stream_shuffle_broadcast.stream_shuffle_broadcast(
                router_logits_bias_vector_sb, router_logits_bias_broadcasted_sb
            )
    """Tiled matmul."""

    # for t_tile_idx in range(num_t_tiles):
    for t_tile in tiled_range.TiledRange(T_local, ST_F_MAX):
        t_tile_idx = t_tile.index
        t_tile_size_actual = t_tile.size
        """
        Initialize PSUM buffer.
        
        We allocate [pmax, E] (i.e. the full pmax) because each column-tile writes into
        a separate p-dim range. If T < pmax we'll use an access pattern to get the correct data.
        """
        router_logits_psum = nl.ndarray((P_MAX, E), nl.float32, buffer=nl.psum)

        for h_tile_idx in range(num_h_tiles):
            # Logically we are multiplying x @ w = [T,H] @ [H,E] = [Hp,Tf] @ [Hp,Ef] (using P/F notation).
            w_tile_sb = w_sb[:, h_tile_idx, :]

            """
            Form 'x' tile that is a multiple of t_tile_size.
            
            This will be oversized on the last tile if T is not a multiple of pmax.
            But the correct remainder amount is selected using the mask in the matmul.
            Compute the start/end indexes for this T-tile slice where T_offset is the LNC shard offset.
            """
            start_t = T_offset + t_tile_idx * t_tile_size
            end_t = start_t + t_tile_size_actual
            if internal_x_sb_layout in [
                XSBLayout_tp102__0,
                XSBLayout_tp2013__1,
                XSBLayout_tp201__2,
            ]:  # x_sb = [128, T, num_h_tiles]
                x_tile_sb = (
                    tensor_view.TensorView(x_sb)
                    .slice(dim=1, start=start_t, end=start_t + t_tile_size_actual)
                    .select(dim=2, index=h_tile_idx)
                    .get_view()
                )
            else:  # internal_x_sb_layout==3, x_sb = [128, num_h_tiles, T]
                x_tile_sb = x_sb[:, h_tile_idx, start_t:end_t]

            # column-tiling setup
            current_column_tile_index = h_tile_idx % num_pe_array_column_tiles
            # Calculate the starting column number for this slot (it's used to set the tile_position in nc_matmul)
            current_column_tile_column_offset = current_column_tile_index * pe_array_column_tiling_size
            tile_position = (0, current_column_tile_column_offset)  # will be (0,0) if use_column_tiling==false

            # matmul ([P,F] notation): x_tile_sb [pmax, t_tile_size] @ w_tile_sb [pmax, E] = router_logits_psum [t_tile_size, E]
            nisa.nc_matmul(
                # dst=router_logits_psum[nl.ds(current_column_tile_column_offset, t_tile_size_actual), :],
                dst=router_logits_psum[nl.ds(current_column_tile_column_offset, t_tile_size_actual), :],
                stationary=x_tile_sb,
                moving=w_tile_sb,
                tile_position=tile_position,
                tile_size=tile_size,
            )

        """Apply bias."""

        if has_bias:
            """
            Apply the bias to router-logits.
            
            Element-wise add with the broadcasted bias tensor while copying from PSUM->SBUF.
            Tensor shapes:
            - router_logits_sb: [t_p_dim, num_t_tiles, E]
            - router_logits_psum: [pmax, E]
            - router_logits_bias_broadcasted_sb: [t_tile_size, E]
            """
            nisa.tensor_tensor(
                dst=router_logits_sb[:t_tile_size_actual, t_tile_idx, :],
                data1=router_logits_psum[:t_tile_size_actual, :E],
                data2=router_logits_bias_broadcasted_sb[:t_tile_size_actual, :E],
                op=nl.add,
            )

        """Merge column tiles."""
        # Straight copy the first tile result, but not if we have bias because we already spilled PSUM using the above tensor_tensor
        if not has_bias:
            nisa.tensor_copy(
                dst=router_logits_sb[:t_tile_size_actual, t_tile_idx, :],
                src=router_logits_psum[:t_tile_size_actual, :E],
            )

        # Accumulate the remaining tile results. This loop does not execute if use_column_tiling==false (i.e. num_pe_array_column_tiles==1)
        # Only accumulate column tiles that received matmul data. When num_h_tiles < num_pe_array_column_tiles,
        # some column tile slots are empty (never written by a matmul), so we must not accumulate them.
        num_active_column_tiles = min(num_h_tiles, num_pe_array_column_tiles)
        for column_tile_idx in range(1, num_active_column_tiles):
            current_column_tile_column_offset = column_tile_idx * pe_array_column_tiling_size
            nisa.tensor_tensor(
                dst=router_logits_sb[:t_tile_size_actual, t_tile_idx, :],
                data1=router_logits_sb[:t_tile_size_actual, t_tile_idx, :],
                data2=router_logits_psum[nl.ds(current_column_tile_column_offset, t_tile_size_actual), :],
                op=nl.add,
            )

    """Store router_logits."""
    if not skip_store_router_logits:
        kernel_assert(router_logits != None, "router_logits must be provided when skip_store_router_logits is False")
        """
        Caller provides router_logits and therefore specifies its dtype. A cast is implied if
        data types mismatch between routers_logits and router_logits_sb.
        """

        # copy from
        # router_logits_sb = [t_p_dim, num_t_tiles, E]
        # to
        # router_logits = [T,E]
        if num_t_whole_tiles > 0:
            nisa.dma_copy(
                src=router_logits_sb[:t_p_dim, :num_t_whole_tiles, :E],
                dst=_hbm_tiled_store_view(router_logits, T_offset, num_t_whole_tiles, t_p_dim),
            )
        if t_remainder > 0:
            nisa.dma_copy(
                src=router_logits_sb[:t_remainder, num_t_whole_tiles, :E],
                dst=_hbm_remainder_store_view(router_logits, T_offset, num_t_whole_tiles, t_p_dim, t_remainder),
                name=f"{name_prefix}store_router_logits_sb_remainder",
            )

    """Configure Pipeline Flags."""

    """
    After computing router-logits, the subsequent operations in this kernel are set up in a pipeline
    where individual stages can be enabled/disabled. Only certain combinations are valid.
    
    The pipeline is:
    ACT1 --> topK --> ACT2 --> Norm --> Scatter
    
    ACT* is an activation function. Norm is an L1 norm (normalize by sum of all values).
    
    Valid combinations:
      (topK, ACT2, Scatter)
      (ACT1, topK)
      (ACT1, topK, Norm, Scatter)
    """

    """
    The following flags specify which stages to enable. topK is always enabled.
    
    Rather than simply accept these flags as kernel arguments, they are derived from
    the existing arguments to maintain kernel interface backwards compatibility.
    """

    pipeline_enable_act1 = router_pre_norm
    pipeline_enable_act2 = not router_pre_norm
    pipeline_enable_norm = pipeline_enable_act1 and norm_topk_prob
    pipeline_enable_scatter = pipeline_enable_act2 or (pipeline_enable_act1 and pipeline_enable_norm)

    # Validate pipeline configurations: (pipeline_enable_act1, pipeline_enable_act2, pipeline_enable_norm)
    valid_combinations = [
        (False, True, False),  # (topK, ACT2)
        (True, False, False),  # (ACT1, topK)
        (True, False, True),  # (ACT1, topK, Norm)
    ]

    current_combination = (pipeline_enable_act1, pipeline_enable_act2, pipeline_enable_norm)
    kernel_assert(
        current_combination in valid_combinations,
        (
            f"Invalid pipeline combination: ACT1={pipeline_enable_act1}, ACT2={pipeline_enable_act2}, "
            f"Norm={pipeline_enable_norm}. Valid combinations are: {valid_combinations}"
        ),
    )

    """ACT1 -- Activation Function on router-logits."""

    if pipeline_enable_act1:
        # 'Full' means the full E.
        expert_affinities_full_sb = nl.ndarray((t_p_dim, num_t_tiles, E), dtype=nl.float32, buffer=nl.sbuf)

        for t_tile in tiled_range.TiledRange(T_local, ST_F_MAX):
            compute_activation(
                expert_affinities_full_sb,
                router_logits_sb,
                t_tile,
                act_fn,
                complete_activation=True,
                input_dtype=input_dtype,
            )

        if not pipeline_enable_scatter:
            # Store the full expert_affinities.
            # expert_affinities = [T,E] whereas expert_affinities_full_sb = [128, T/128, E]

            if num_t_whole_tiles > 0:
                nisa.dma_copy(
                    src=expert_affinities_full_sb[:t_p_dim, :num_t_whole_tiles, :E],
                    dst=_hbm_tiled_store_view(expert_affinities, T_offset, num_t_whole_tiles, t_p_dim),
                )
            if t_remainder > 0:
                nisa.dma_copy(
                    src=expert_affinities_full_sb[:t_remainder, num_t_whole_tiles, :E],
                    dst=_hbm_remainder_store_view(expert_affinities, T_offset, num_t_whole_tiles, t_p_dim, t_remainder),
                    name=f"{name_prefix}store_expert_affinities_1hot_scattered_remainder_full_sb",
                )

            # core_barrier is only meaningful at LNC≥2. nki.isa.core_barrier
            # asserts num_progs>1 (lnc.py:73), so guard for LNC=1 callers
            # (gpt-oss megakernel runs at LNC=1).
            if n_prgs > 1:
                core_barrier(expert_affinities, (0, 1))

    """topK."""

    """
    Top-K operation finds the largest K values in each partition along with their indexes.
    
    Select input for topK operation. If ACT1 is enabled, use its output. Otherwise use router-logits.
    """
    topk_input_sb = router_logits_sb if (not pipeline_enable_act1) else expert_affinities_full_sb

    # Simplify by setting max K=8.
    kernel_assert(k >= 1, f"K ({k}) must be >= 1")
    kernel_assert(k <= 8, f"K ({k}) must be <= 8")

    # The nisa instructions for topK operate in chunks of 8.
    # Get the top 8 router_logit values, in each partition (each token). Keep this around for subsequent use in nc_find_index8
    router_logits_topk_sb = nl.ndarray((t_p_dim, num_t_tiles, k), dtype=nl.float32, buffer=nl.sbuf)
    router_indexes_topk_sb = nl.ndarray((t_p_dim, num_t_tiles, k), dtype=nl.uint32, buffer=nl.sbuf)
    router_indexes_topk_truek_sb_fp32 = nl.ndarray((t_p_dim, num_t_tiles, k), dtype=nl.float32, buffer=nl.sbuf)

    tmp_buffer = nl.ndarray((t_p_dim, 8), dtype=router_indexes_topk_sb.dtype, buffer=nl.sbuf)

    for t_tile in tiled_range.TiledRange(T_local, ST_F_MAX):
        t_tile_size_actual = t_tile.size
        t_tile_idx = t_tile.index
        # topk_input_sb has nominal shape [T,E], folded into SBUF shape [128,T/128,E] = [128,num_t_tiles,k]
        router_logits_top8_sb = nl.ndarray((t_tile_size_actual, 8), dtype=topk_input_sb.dtype, buffer=nl.sbuf)
        nisa.max8(
            dst=router_logits_top8_sb[:t_tile_size_actual, :], src=topk_input_sb[:t_tile_size_actual, t_tile_idx, :]
        )  # returns [t_tile_size,8]
        nisa.tensor_copy(
            dst=router_logits_topk_sb[:t_tile_size_actual, t_tile_idx, :],
            src=router_logits_top8_sb[:t_tile_size_actual, :k],
        )  # Take the topK values from the 8 --> [t_tile_size:k]

        # Now get the indexes (locations) of those top-8 values.
        # returns [t_tile_size,8], take [t_tile_size,K] slice

        nisa.nc_find_index8(
            dst=tmp_buffer[:t_tile_size_actual, :],
            data=topk_input_sb[:t_tile_size_actual, t_tile_idx, :],
            vals=router_logits_top8_sb,
        )
        nisa.tensor_copy(
            dst=router_indexes_topk_sb[:t_tile_size_actual, t_tile_idx, :k], src=tmp_buffer[:t_tile_size_actual, :k]
        )

        # Cast to fp32 for later use in the one-hot scatter. Nominal shape (T,k), but in SB we must fold to [128,num_t_tiles,k]
        nisa.tensor_copy(
            dst=router_indexes_topk_truek_sb_fp32[:t_tile_size_actual, t_tile_idx, :],
            src=router_indexes_topk_sb[:t_tile_size_actual, t_tile_idx, :k],
        )

    """Store expert_index."""

    if expert_index_in_sb and not skip_store_expert_index:
        if shard_on_tokens:
            if T <= PE_COLUMN_TILE_128:
                expert_index_send = nl.ndarray((T_sendrecv_size, k), dtype=nl.uint32, buffer=nl.sbuf)
                expert_index_recv = nl.ndarray((T_sendrecv_size, k), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(src=router_indexes_topk_sb[:T_local, 0, :], dst=expert_index_send[:T_local, :])
                nisa.sendrecv(
                    dst=expert_index_recv,
                    src=expert_index_send,
                    send_to_rank=1 - prg_id,
                    recv_from_rank=1 - prg_id,
                    pipe_id=0,
                )
                other_offset = T_first_shard if prg_id == 0 else 0
                # Core 0: T_offset=0, other_offset=T_first_shard → write local first
                # Core 1: T_offset=T_first_shard, other_offset=0 → write recv first
                if T_offset == 0:
                    cross_partition_copy.cross_partition_copy(
                        src=expert_index_send,
                        dst=expert_index,
                        src_start_partition=0,
                        dst_start_partition=0,
                        num_partitions_to_copy=T_local,
                        free_dim_size=k,
                    )
                    cross_partition_copy.cross_partition_copy(
                        src=expert_index_recv,
                        dst=expert_index,
                        src_start_partition=0,
                        dst_start_partition=other_offset,
                        num_partitions_to_copy=other_T_local,
                        free_dim_size=k,
                    )
                else:
                    cross_partition_copy.cross_partition_copy(
                        src=expert_index_recv,
                        dst=expert_index,
                        src_start_partition=0,
                        dst_start_partition=0,
                        num_partitions_to_copy=other_T_local,
                        free_dim_size=k,
                    )
                    cross_partition_copy.cross_partition_copy(
                        src=expert_index_send,
                        dst=expert_index,
                        src_start_partition=0,
                        dst_start_partition=T_offset,
                        num_partitions_to_copy=T_local,
                        free_dim_size=k,
                    )
            else:
                # Tile-based sendrecv requires T to be a multiple of 128 for correct alignment
                kernel_assert(
                    T % PE_COLUMN_TILE_128 == 0,
                    f"T ({T}) must be a multiple of 128 when T > 128 with shard_on_tokens and expert_index in SBUF",
                )
                other_num_t_tiles = num_t_tiles_total - num_t_tiles
                t_local_slice = nl.ds(0, num_t_tiles) if prg_id == 0 else nl.ds(other_num_t_tiles, num_t_tiles)
                t_remote_slice = nl.ds(num_t_tiles, other_num_t_tiles) if prg_id == 0 else nl.ds(0, other_num_t_tiles)

                # Copy local results to expert_index
                nisa.tensor_copy(
                    src=router_indexes_topk_sb[:, :, :],
                    dst=expert_index[:t_p_dim, t_local_slice, :],
                )
                # Exchange data between cores via sendrecv on tile dimension
                nisa.sendrecv(
                    dst=expert_index[:t_p_dim, t_remote_slice, :],
                    src=expert_index[:t_p_dim, t_local_slice, :],
                    send_to_rank=1 - prg_id,
                    recv_from_rank=1 - prg_id,
                    pipe_id=0,
                )
        else:
            # For SBUF output, copy router_indexes_topk_sb to expert_index
            if num_t_tiles == 1:
                router_indexes_topk_sb = router_indexes_topk_sb.reshape((t_p_dim, k))
            nisa.tensor_copy(src=router_indexes_topk_sb, dst=expert_index)
    elif not skip_store_expert_index:
        # copy from
        # router_indexes_topk_sb has SB shape [t_p_dim, num_t_tiles, k]
        # to
        # expert_index has HBM shape [T,k]

        if num_t_whole_tiles > 0:
            nisa.dma_copy(
                src=router_indexes_topk_sb[:t_p_dim, :num_t_whole_tiles, :k],
                dst=_hbm_tiled_store_view(expert_index, T_offset, num_t_whole_tiles, t_p_dim),
            )
        if t_remainder > 0:
            nisa.dma_copy(
                src=router_indexes_topk_sb[:t_remainder, num_t_whole_tiles, :k],
                dst=_hbm_remainder_store_view(expert_index, T_offset, num_t_whole_tiles, t_p_dim, t_remainder),
            )
    """ACT2 -- Activation Function on top-K values, tensor declarations."""

    # If doing softmax we split into two stages to facilitate the one-hot scatter technique below.
    # Declare tensors for all the intermediate steps of the tiled softmax.
    router_logits_topk_negmax_sb = nl.ndarray((t_p_dim, num_t_tiles), dtype=nl.float32, buffer=nl.sbuf)
    router_logits_topk_exp_sum_sb = nl.ndarray((t_p_dim, num_t_tiles, 1), dtype=nl.float32, buffer=nl.sbuf)
    router_logits_exp_sb = nl.ndarray((t_p_dim, num_t_tiles, E), dtype=nl.float32, buffer=nl.sbuf)
    router_logits_softmax_sb = nl.ndarray((t_p_dim, num_t_tiles, E), dtype=nl.float32, buffer=nl.sbuf)
    expert_affinities_topk_sb = nl.ndarray((t_p_dim, num_t_tiles, k), dtype=nl.float32, buffer=nl.sbuf)
    # When we have T>128 and not divisible by 128, init buffer with zeros.
    expert_affinities_one_hot_scattered_sb = nl.ndarray((t_p_dim, num_t_tiles, E), dtype=nl.float32, buffer=nl.sbuf)

    if pipeline_enable_scatter:
        # In case of using indirect DMA for storing expert_affinities, we need to memset the output buffer on HBM to zero
        # We need to init this zero tile here and keep reusing them
        expert_affinities_zero_tile = nl.ndarray(shape=(t_p_dim, E), dtype=input_dtype)
        nisa.memset(dst=expert_affinities_zero_tile[:t_p_dim, :E], value=0.0)

    if T_pad > 0:
        nisa.memset(dst=expert_affinities_topk_sb[:t_p_dim, num_t_tiles - 1, :], value=0.0)
        nisa.memset(dst=expert_affinities_one_hot_scattered_sb, value=0.0, engine=engine.gpsimd)

    # for t_tile_idx in range(num_t_tiles):
    for t_tile in tiled_range.TiledRange(T_local, ST_F_MAX):
        t_tile_idx = t_tile.index
        t_tile_size = t_tile.size
        """ACT2 -- Activation."""
        if pipeline_enable_act2:
            if return_eager_affi or use_indirect_dma_scatter:
                kernel_assert(
                    (not return_eager_affi or (T <= PE_COLUMN_TILE_128)),
                    "If return_eager_affi, then T must be <=128 because expert_affinities shape is [T,k]",
                )
                # Compute complete activation, optionally preserving intermediates
                compute_activation(
                    expert_affinities_topk_sb,
                    router_logits_topk_sb,
                    t_tile,
                    act_fn,
                    router_logits_topk_negmax_sb if act_fn == common_types.RouterActFnType.SOFTMAX else None,
                    router_logits_topk_exp_sum_sb if act_fn == common_types.RouterActFnType.SOFTMAX else None,
                    complete_activation=True,
                    return_intermediates=(act_fn == common_types.RouterActFnType.SOFTMAX),
                    input_dtype=input_dtype,
                )

            elif act_fn == common_types.RouterActFnType.SOFTMAX:
                # Compute partial activation only
                compute_activation(
                    expert_affinities_topk_sb,
                    router_logits_topk_sb,
                    t_tile,
                    act_fn,
                    router_logits_topk_negmax_sb,
                    router_logits_topk_exp_sum_sb,
                    complete_activation=False,
                    input_dtype=input_dtype,
                )
            else:
                kernel_assert(
                    False,
                    "Should not reach this code branch. Check for valid combination of use_indirect_dma_scatter, act_fn, return_eager_affi.",
                )

        """Norm."""

        if pipeline_enable_norm:
            # Input is router_logits_topk_sb

            sum_of_max_sb = nl.ndarray((t_tile_size, num_t_tiles, 1), dtype=nl.float32, buffer=nl.sbuf)

            # Sum all topK values for each token (reduce along expert dimension)
            # sum_of_max_sb[:t_tile_size, t_tile_idx, :] =
            nisa.tensor_reduce(
                dst=sum_of_max_sb[:t_tile_size, t_tile_idx, :],
                op=nl.add,
                data=router_logits_topk_sb[:t_tile_size, t_tile_idx, :],
                axis=1,
                keepdims=True,
            )

            # Take reciprocal to convert sum into scaling factor
            nisa.reciprocal(
                dst=sum_of_max_sb[:t_tile_size, t_tile_idx, :], data=sum_of_max_sb[:t_tile_size, t_tile_idx, :]
            )

            # Multiply each topK value by the scaling factor (L1 normalization)
            # expert_affinities_topk_sb[:t_tile_size, t_tile_idx, :] =
            nisa.tensor_scalar(
                dst=expert_affinities_topk_sb[:t_tile_size, t_tile_idx, :],
                data=router_logits_topk_sb[:t_tile_size, t_tile_idx, :],
                op0=nl.multiply,
                operand0=sum_of_max_sb[:t_tile_size, t_tile_idx, :],
            )

        """Scatter -- expert_affinities."""

        if pipeline_enable_scatter:
            # At this point we either fully (if return_eager_affi or use_indirect_dma_scatter) or partially (if !use_indirect_dma_scatter) have the
            # topK expert-affinities and their indexes.
            # Now we create the final expert_affinities by scattering the topK values into their index positions, with zeroes everywhere else.
            # There are two methods for scattering: one-hot and indirect-dma.

            # one-hot scatter
            if not use_indirect_dma_scatter:
                """
                In this technique we use a one-hot mask (a tensor of 1s in the expert_index positions)
                to grab only the values we want, with zeroes everywhere else.

                For ACT2 path (pipeline_enable_act2): We softmax the entire router-logits but use the
                negmax and denominator from top-K so that, post-mask, it's equivalent to having directly
                softmax'ed the topK.

                For ACT1 path (pipeline_enable_act1): Activation is already computed on full tensor
                (expert_affinities_full_sb), so we just multiply with the mask. If norm is enabled,
                we also scale by sum_of_max_sb (reciprocal of sum of top-K affinities).
                """

                if pipeline_enable_act2:
                    """Complete the softmax (ACT2 path)."""
                    # Calculate numerator.
                    # Apply exponential to the full router-logits, but use the negmax computed from topk values.
                    nisa.activation(
                        dst=router_logits_exp_sb[:t_tile_size, t_tile_idx, :],
                        op=nl.exp,
                        data=router_logits_sb[:t_tile_size, t_tile_idx, :],
                        bias=router_logits_topk_negmax_sb[:t_tile_size, t_tile_idx : t_tile_idx + 1],
                    )
                    # Still use the denominator computed above: router_logits_topk_exp_sum_sb
                    # Apply the denominator
                    nisa.tensor_scalar(
                        dst=router_logits_softmax_sb[:t_tile_size, t_tile_idx, :],
                        data=router_logits_exp_sb[:t_tile_size, t_tile_idx, :],
                        op0=nl.multiply,
                        operand0=router_logits_topk_exp_sum_sb[:t_tile_size, t_tile_idx, :],
                    )  # [T,K]
                    # At this point we have a softmax on router-logits where the negmax and the denominator
                    # came from the topk values.

                """
                One-hot technique.

                We construct a [1,E] tensor of incrementing integers, meaning every expert position simply
                contains its expert number. We then compare it against the router-indexes, yielding a '1' in
                every position there is a match, which means we get a '1' in the chosen expert positions.
                """

                # mask_sbuf = nisa.memset((t_p_dim, E), 0.0, dtype=router_logits_sb.dtype, engine=nisa.engine.gpsimd)
                mask_sbuf = nl.ndarray(shape=(t_tile_size, E), dtype=router_logits_sb.dtype, buffer=nl.sbuf)
                nisa.memset(dst=mask_sbuf, value=0.0, engine=nisa.gpsimd_engine)

                # [1,E] tensor of incrementing ints
                expert_num_idx_arr_sbuf = nl.ndarray((t_p_dim, E), dtype=router_indexes_topk_sb.dtype, buffer=nl.sbuf)
                nisa.iota(dst=expert_num_idx_arr_sbuf, pattern=[[1, E]], offset=0, channel_multiplier=0)

                """
                nl.equal is a tensor_scalar where a broadcastable vector is compared against a tensor.

                So one side of the equals must be a vector, hence we iterate through the 'k' column-vectors
                of router_indexes_topk_truek_sb_fp32.
                """
                for expert_idx in range(k):
                    # router_indexes_topk_truek_sb_fp32 is [t_p_dim, num_t_tiles,k].
                    # We take [T,1] column slices and compare against the [1,E] expert_num_idx_arr_sbuf tensor.
                    # Broadcast rules mean that this becomes a [t_p_dim,E] vs [t_p_dim,E] comparison.
                    # To be clear, the broadcasted expert_num_idx_arr_sbuf is comprised of identical rows,
                    # each containing incrementing ints.
                    # The broadcasted router_indexes_topk_truek_sb_fp32 column slice contains a single expert
                    # index, per token (row) repeated across all columns.
                    # Therefore the equals operator returns a '1' precisely in the expert index location.

                    export_check = nl.ndarray(shape=(t_tile_size, E), dtype=mask_sbuf.dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=export_check[:t_tile_size, :],
                        op0=nl.equal,
                        data=expert_num_idx_arr_sbuf[:t_tile_size, :],
                        operand0=router_indexes_topk_truek_sb_fp32[
                            :t_tile_size, t_tile_idx, expert_idx
                        ],  # column vector [t_p_dim, 1] should be the operand
                    )

                    nisa.tensor_tensor(
                        dst=mask_sbuf[:t_tile_size, :],
                        data1=mask_sbuf[:t_tile_size, :],
                        op=nl.add,
                        data2=export_check[:t_tile_size, :],
                    )

                # Now multiply with the mask to select the affinities we want in the topK positions
                if pipeline_enable_act1:
                    # ACT1 path: activation already computed on full tensor (expert_affinities_full_sb)
                    if pipeline_enable_norm:
                        # Apply L1 normalization in-place: scale by 1/sum(topk_affinities)
                        nisa.tensor_scalar(
                            dst=expert_affinities_full_sb[:t_tile_size, t_tile_idx, :],
                            data=expert_affinities_full_sb[:t_tile_size, t_tile_idx, :],
                            op0=nl.multiply,
                            operand0=sum_of_max_sb[:t_tile_size, t_tile_idx, :],
                        )
                    # Apply mask to expert_affinities_full_sb
                    nisa.tensor_tensor(
                        dst=expert_affinities_one_hot_scattered_sb[:t_tile_size, t_tile_idx, :],
                        data1=mask_sbuf[:t_tile_size, :],
                        op=nl.multiply,
                        data2=expert_affinities_full_sb[:t_tile_size, t_tile_idx, :],
                    )
                else:
                    # ACT2 path: multiply mask with router_logits_softmax_sb
                    nisa.tensor_tensor(
                        dst=expert_affinities_one_hot_scattered_sb[:t_tile_size, t_tile_idx, :],
                        data1=mask_sbuf[:t_tile_size, :],
                        op=nl.multiply,
                        data2=router_logits_softmax_sb[:t_tile_size, t_tile_idx, :],
                    )

                if expert_affin_in_sb:
                    kernel_assert(
                        T <= PE_COLUMN_TILE_128,
                        "If expert_affin_in_sb, then T must be <=128",
                    )
                    if shard_on_tokens:
                        # With shard_on_tokens, local data copy is deferred until after sendrecv
                        # to ensure dst_start=0 is written first
                        pass
                    else:
                        expert_affinities = expert_affinities_one_hot_scattered_sb.reshape((t_p_dim, E))

            # indirect-dma scatter
            else:
                """
                Here we perform the scatter using indirect DMA which allows the destination HBM tensor
                (expert_affinities_hbm) to be accessed using an index tensor that is dynamically constructed
                during execution (router_indexes_topk_sb).
                
                But indirect-DMA does not support a 2D index tensor (ie. we cannot directly use router_indexes_topk_sb).
                We must transform both expert_affinities_hbm and router_indexes_topk_sb into 1D.
                
                The procedure is:
                - Create expert_affinities_hbm = [T,E] in HBM as a tensor of zeroes
                - Reshape it to a 1D tensor of shape [T*E] = expert_affinities_hbm_1d (just a view change)
                - Transform router_indexes_topk_sb into a 2D tensor containing indexes into the 1D expert_affinities_hbm_1d
                - Use 1D columns of router_indexes_topk_sb to perform indirect-DMA into the 1D expert_affinities_hbm_1d
                """
                kernel_assert(
                    not expert_affin_in_sb,
                    "expert_affinities cannot be a SBUF tensor if we use indirect dma to perform scatter",
                )
                # Store the zeroes to HBM. When sharding is enabled, T_offset accounts for the global token offset.
                nisa.dma_copy(
                    dst=expert_affinities[T_offset + t_tile_idx * t_p_dim : T_offset + (t_tile_idx + 1) * t_p_dim, :],
                    src=expert_affinities_zero_tile,
                )
                # Prepare the funky index vector.

                """
                Indirect-DMA requires us to take the indexes from router_indexes_topk_sb and transform them
                into equivalent indexes into a flattened 1D HBM tensor of shape [T*E].
                
                Create a column-vector in SBUF with values (0, E, 2E, 3E, 4E, ...) but offset into the
                current T-tile. When sharding is enabled, T_offset accounts for the global token offset.
                """
                index_offset_sb = nl.ndarray(
                    shape=(t_p_dim, 1),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                offset = (T_offset + t_tile_idx * t_p_dim) * E

                nisa.iota(
                    dst=index_offset_sb.ap([[1, t_p_dim], [1, 1], [1, 1]]),
                    pattern=[[1, 1]],
                    offset=offset,
                    channel_multiplier=E,
                    name=f"{name_prefix}IOTA_offset_{offset}",
                )

                """
                Recall that router_indexes_topk_sb = [128,T/128,K], meaning each Row (token) contains a list
                of K indexes (the indexes of the top-K experts).
                
                index_offset_sb = [T/128,1]
                tensor_scalar will broadcast index_offset_sb in the free-dim.
                
                This op will therefore take router_indexes_topk_sb and:
                - add 0 to all values in Row T=0
                - add E to all values in Row T=1
                - add 2E to all values in Row T=2
                - etc
                
                The result is that each expert index gets mapped into the corresponding index into the
                flattened 1D tensor shape [T*E]. Not shown above is the additional offset, contained in
                index_offset_sb, to shift us into the current T-tile.
                """
                router_indexes_topk_flattened_sb = nl.ndarray(
                    shape=(t_p_dim, k),
                    dtype=nl.uint32,
                    buffer=nl.sbuf,
                    # , name=f"router_indexes_topk_flattened_sb_{t_tile_idx}"
                )

                nisa.tensor_scalar(
                    dst=router_indexes_topk_flattened_sb.ap([[k, t_p_dim], [1, 1], [1, k]]),
                    data=router_indexes_topk_sb[:, t_tile_idx, :],
                    op0=nl.add,
                    operand0=index_offset_sb,
                )

                # Now that we have the funky index vector prepared, perform the indirect-DMA.

                expert_affinities_hbm_1d = expert_affinities.reshape((T * E,))

                for k_idx in range(k):
                    # router_indexes_topk_flattened_sb has shape [T,k]
                    index_column = router_indexes_topk_flattened_sb.ap(
                        pattern=[[k, t_p_dim], [1, 1], [1, 1]], offset=k_idx
                    )

                    # Do the indirect DMA using the column vector as the dynamic indexes.
                    # Copy the k'th column of expert_affinities_topk_sb into the scattered positions given by index_column.
                    nisa.dma_copy(
                        dst=expert_affinities_hbm_1d.ap(
                            [[k * num_t_tiles, t_p_dim], [1, 1]], offset=0, vector_offset=index_column, indirect_dim=0
                        ),
                        src=expert_affinities_topk_sb.ap(
                            [[k * num_t_tiles, t_p_dim], [1, 1]], offset=k_idx + t_tile_idx * k
                        ),
                    )

    # when using one-hot scatter method, spill expert affinities to HBM outside of t_tile loop
    if pipeline_enable_scatter:
        if (not use_indirect_dma_scatter) and (not expert_affin_in_sb):
            if num_t_whole_tiles > 0:
                nisa.dma_copy(
                    src=expert_affinities_one_hot_scattered_sb[:t_p_dim, :num_t_whole_tiles, :E],
                    dst=_hbm_tiled_store_view(expert_affinities, T_offset, num_t_whole_tiles, t_p_dim),
                )
            if t_remainder > 0:
                nisa.dma_copy(
                    src=expert_affinities_one_hot_scattered_sb[:t_remainder, num_t_whole_tiles, :E],
                    dst=_hbm_remainder_store_view(expert_affinities, T_offset, num_t_whole_tiles, t_p_dim, t_remainder),
                    name=f"{name_prefix}store_expert_affinities_1hot_scattered_remainder_one_hot",
                )

            core_barrier(expert_affinities, cores=[0, 1])

        # When expert_affin_in_sb and shard_on_tokens, exchange data between cores via sendrecv
        if expert_affin_in_sb and shard_on_tokens and (not use_indirect_dma_scatter):
            expert_affin_send = nl.ndarray((T_sendrecv_size, E), dtype=nl.float32, buffer=nl.sbuf)
            expert_affin_recv = nl.ndarray((T_sendrecv_size, E), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                src=expert_affinities_one_hot_scattered_sb[:T_local, 0, :],
                dst=expert_affin_send[:T_local, :],
            )
            nisa.sendrecv(
                dst=expert_affin_recv,
                src=expert_affin_send,
                send_to_rank=1 - prg_id,
                recv_from_rank=1 - prg_id,
                pipe_id=0,
            )
            other_offset = T_first_shard if prg_id == 0 else 0
            if T_offset == 0:
                cross_partition_copy.cross_partition_copy(
                    src=expert_affin_send,
                    dst=expert_affinities,
                    src_start_partition=0,
                    dst_start_partition=0,
                    num_partitions_to_copy=T_local,
                    free_dim_size=E,
                )
                cross_partition_copy.cross_partition_copy(
                    src=expert_affin_recv,
                    dst=expert_affinities,
                    src_start_partition=0,
                    dst_start_partition=other_offset,
                    num_partitions_to_copy=other_T_local,
                    free_dim_size=E,
                )
            else:
                cross_partition_copy.cross_partition_copy(
                    src=expert_affin_recv,
                    dst=expert_affinities,
                    src_start_partition=0,
                    dst_start_partition=0,
                    num_partitions_to_copy=other_T_local,
                    free_dim_size=E,
                )
                cross_partition_copy.cross_partition_copy(
                    src=expert_affin_send,
                    dst=expert_affinities,
                    src_start_partition=0,
                    dst_start_partition=T_offset,
                    num_partitions_to_copy=T_local,
                    free_dim_size=E,
                )

    """Return."""

    outputs = [router_logits, expert_index, expert_affinities]

    if return_eager_affi:
        if shard_on_tokens:
            kernel_assert(
                T <= PE_COLUMN_TILE_128,
                "If return_eager_affi with shard_on_tokens, then T must be <=128",
            )
            expert_affinities_topk_full = nl.ndarray((T, k), dtype=nl.float32, buffer=nl.sbuf)
            eager_affi_send = nl.ndarray((T_sendrecv_size, k), dtype=nl.float32, buffer=nl.sbuf)
            eager_affi_recv = nl.ndarray((T_sendrecv_size, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(src=expert_affinities_topk_sb[:T_local, 0, :], dst=eager_affi_send[:T_local, :])
            nisa.sendrecv(
                dst=eager_affi_recv,
                src=eager_affi_send,
                send_to_rank=1 - prg_id,
                recv_from_rank=1 - prg_id,
                pipe_id=0,
            )
            other_offset = T_first_shard if prg_id == 0 else 0
            if T_offset == 0:
                cross_partition_copy.cross_partition_copy(
                    src=eager_affi_send,
                    dst=expert_affinities_topk_full,
                    src_start_partition=0,
                    dst_start_partition=0,
                    num_partitions_to_copy=T_local,
                    free_dim_size=k,
                )
                cross_partition_copy.cross_partition_copy(
                    src=eager_affi_recv,
                    dst=expert_affinities_topk_full,
                    src_start_partition=0,
                    dst_start_partition=other_offset,
                    num_partitions_to_copy=other_T_local,
                    free_dim_size=k,
                )
            else:
                cross_partition_copy.cross_partition_copy(
                    src=eager_affi_recv,
                    dst=expert_affinities_topk_full,
                    src_start_partition=0,
                    dst_start_partition=0,
                    num_partitions_to_copy=other_T_local,
                    free_dim_size=k,
                )
                cross_partition_copy.cross_partition_copy(
                    src=eager_affi_send,
                    dst=expert_affinities_topk_full,
                    src_start_partition=0,
                    dst_start_partition=T_offset,
                    num_partitions_to_copy=T_local,
                    free_dim_size=k,
                )
            outputs.append(expert_affinities_topk_full)
        else:
            expert_affinities_topk_sb = expert_affinities_topk_sb.reshape((t_p_dim, num_t_tiles * k))
            outputs.append(expert_affinities_topk_sb)

    return outputs


def _hbm_tiled_store_view(tensor, T_offset, num_t_whole_tiles, t_p_dim):
    """Create TensorView for tiled HBM store: slice -> reshape -> permute.

    Transforms tensor[T, X] to view compatible with SBUF layout [t_p_dim, num_tiles, X].
    Shape transformations:
        [T, X] -> slice -> [num_t_whole_tiles * t_p_dim, X]
                -> reshape -> [num_t_whole_tiles, t_p_dim, X]
                -> permute -> [t_p_dim, num_t_whole_tiles, X]
    """
    return (
        tensor_view.TensorView(tensor)
        .slice(dim=0, start=T_offset, end=T_offset + num_t_whole_tiles * t_p_dim)
        .reshape_dim(dim=0, shape=(num_t_whole_tiles, t_p_dim))
        .permute((1, 0, 2))
        .get_view()
    )


def _hbm_remainder_store_view(tensor, T_offset, num_t_whole_tiles, t_p_dim, t_remainder):
    """Create TensorView for remainder HBM store: slice -> expand_dim.

    Shape transformations:
        [T, X] -> slice -> [t_remainder, X]
                -> expand_dim -> [t_remainder, 1, X]

    expand_dim is needed to match the 3D SBUF layout [t_p_dim, num_tiles, X].
    The remainder has only 1 tile, so we insert a dimension of size 1 at dim=1.
    """
    return (
        tensor_view.TensorView(tensor)
        .slice(
            dim=0,
            start=T_offset + num_t_whole_tiles * t_p_dim,
            end=T_offset + num_t_whole_tiles * t_p_dim + t_remainder,
        )
        .expand_dim(dim=1)
        .get_view()
    )


def router_topk_input_x_load(x: nl.ndarray, hbm_layout=XHBMLayout_H_T__0, sb_layout=XSBLayout_tp2013__1):
    """
    Load input tensor x from HBM to SBUF with specified layout transformations.

    Performs DMA transfer from HBM to SBUF with layout conversion based on hbm_layout
    and sb_layout parameters. Supports multiple layout combinations optimized for
    different access patterns in subsequent matmul operations.

    hbm_layout:
        0 = [H,T]
        1 = [T,H]

    sb_layout (same meaning as router_topk(x_sb_layout)). num_h_tiles=H/128:
        0 = [128, T, H/128] where p-dim contains H elements with stride of num_h_tiles
        1 = [128, T, H/128] where p-dim contains H elements with stride of num_h_tiles, but H is further
                            interleaved inside dim-2 with a chunk size of H/256.
        2 = [128, T, H/128] where p-dim contains H elements with stride of 1 (i.e. consecutive)
        3 = [128, H/128, T/128, 128] where p-dim contains H elements with stride of 1 (i.e. consecutive)

    Supported combos (Y=yes supported. dash=No not supported)::

        ┌─────────────────────────────────┐
        │               sb_layout         │
        │ hbm_layout    0    1    2    3  │
        │     0         -    -    -    Y  │
        │     1         Y    Y    Y    -  │
        └─────────────────────────────────┘

    Returned shapes (terms: num_h_tiles=H/128)::

        ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │               sb_layout                                                                                      │
        │ hbm_layout    0                    1                    2                     3                              │
        │     0         -                    -                    -                     [128,num_h_tiles,T]            │
        │     1         [128,T,num_h_tiles]  [128,T,num_h_tiles]  [128,T,num_h_tiles]    -                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

    Args:
        x (nl.ndarray): Input tensor in HBM. Shape [H, T] if hbm_layout=0, [T, H] if hbm_layout=1
        hbm_layout (int): Layout of x in HBM (0=[H,T], 1=[T,H])
        sb_layout (int): Target layout in SBUF (0-3). See layout descriptions above

    Returns:
        x_sb (nl.ndarray): Input tensor in SBUF with transformed layout.
                          Shape depends on sb_layout: [128, T, H/128] for layouts 0-2,
                          [128, H/128, T] for layout 3

    Notes:
        - H must be a multiple of 128
        - Supported combinations: (hbm_layout=0, sb_layout=3) and (hbm_layout=1, sb_layout=0/1/2)
        - Layout 0: P-dim contains H elements with stride of H/128
        - Layout 1: P-dim contains H elements with stride of H/128, interleaved by H/256 chunks
        - Layout 2: P-dim contains consecutive H elements
        - Layout 3: H-tiles arranged in dim-1, T in dim-2
    """
    # Input validation
    if hbm_layout not in (XHBMLayout_H_T__0, XHBMLayout_T_H__1):
        kernel_assert(
            False, f"router_topk_input_x_load only hbm_layout=0,1 are supported. Specified layout value: {hbm_layout}"
        )

    if sb_layout not in (0, 1, 2, 3):
        kernel_assert(
            False,
            f"router_topk_input_x_load only sb_layout=0,1,2,3 are supported. Specified layout value: {sb_layout}",
        )

    # Check for unsupported layout combinations
    if hbm_layout == XHBMLayout_H_T__0 and sb_layout != XSBLayout__128_Hdiv128_T__3:
        kernel_assert(False, f"hbm_layout=0 only supports sb_layout=3, got sb_layout={sb_layout}")
    if hbm_layout == XHBMLayout_T_H__1 and sb_layout not in [
        XSBLayout_tp102__0,
        XSBLayout_tp2013__1,
        XSBLayout_tp201__2,
    ]:
        kernel_assert(False, f"hbm_layout=1 only supports sb_layout=0,1,2 got sb_layout={sb_layout}")

    # Get input shapes
    if hbm_layout == XHBMLayout_H_T__0:
        H, T = x.shape
    else:
        T, H = x.shape

    num_h_tiles = H // P_MAX
    kernel_assert(H % P_MAX == 0, f"Hidden dimension must be multiples of {P_MAX}")
    num_h_tiles_by_2 = num_h_tiles // 2

    x_sb = None

    # sb_layout == 0

    if hbm_layout == XHBMLayout_T_H__1 and sb_layout == XSBLayout_tp102__0:
        # HBM tensor [T,H] reshaped (new view) into 128 H-tiles of size num_h_tiles=H/128.
        # A note about the terminology. In HBM we have an H tile-count of 128 and an H tile-size of H/128.
        # Effectively, a transpose is happening (to turn the [T,H] shape into [H,T]) when we combine
        # the HBM->SB DMA here with the eventual matmul access pattern into SB. So the tile-count becomes
        # the tile-size and vice versa. We want to use the same terminology in both diagrams to make it clear
        # how the HBM data is rearranged in SB. So we choose to use the SB terminology to describe the data
        # in HBM. Therefore the SB tile-count name (num_h_tiles) is used as the HBM tile-size name.
        #
        # HBM Layout [T, 128, H/128] = [T, 128, num_h_tiles]
        #
        #         H-tiles (128 total) →
        # ┌─────────────┬─────────────┬─────────────┬─────┐
        # │     H0      │     H1      │     H2      │ ... │ ← Token 0  ↕ 1 elem
        # ├─────────────┼─────────────┼─────────────┼─────┤
        # │     H0      │     H1      │     H2      │ ... │ ← Token 1  ↕ 1 elem
        # ├─────────────┼─────────────┼─────────────┼─────┤
        # │     H0      │     H1      │     H2      │ ... │ ← Token 2  ↕ 1 elem
        # └─────────────┴─────────────┴─────────────┴─────┘
        # ←num_h_tiles→ ←num_h_tiles→ ←num_h_tiles→ ←num_h_tiles→
        #     elem         elem         elem         elem
        #
        # SBUF layout [128, T, H/128] = [128, T, num_h_tiles]
        #
        #   Token0      Token1      Token2      Token3      Token4
        #     ↓           ↓           ↓           ↓           ↓
        # ┌──────────────────────────────────────────────────────────┐
        # │     H0    │     H0    │     H0    │     H0    │  H0  ... │ ← 128 rows
        # │     H1    │     H1    │     H1    │     H1    │  H1      │   (P-dim)
        # │     H2    │     H2    │     H2    │     H2    │  H2      │
        # │     ..    │     ..    │     ..    │     ..    │  ..      │
        # └──────────────────────────────────────────────────────────┘
        # ←num_h_tiles→ ←num_h_tiles→ ←num_h_tiles→ ←num_h_tiles→ ←num_h_tiles→
        #     elem         elem         elem         elem         elem
        #
        # Each token occupies num_h_tiles columns in free-dim
        # Therefore the P-dim (i.e. each column of SBUF) contains H elements with a stride of num_h_tiles.

        x_sb = nl.ndarray((P_MAX, T, num_h_tiles), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=(
                tensor_view.TensorView(x).reshape_dim(dim=1, shape=(P_MAX, num_h_tiles)).permute((1, 0, 2)).get_view()
            ),
            dst=tensor_view.TensorView(x_sb).get_view(),
        )

    # sb_layout == 1

    elif hbm_layout == XHBMLayout_T_H__1 and sb_layout == XSBLayout_tp2013__1:
        # First, understand the above diagram for sb_layout == 0.
        # sb_layout==1 shrinks the H-tile size by half (num_h_tiles_by_2) and introduces an additional
        # dimension of 2 on the H dimension.
        # num_h_tiles = nht = H/128
        # num_h_tiles_by_2 = nht2 = nht/2 = H/256
        #
        # Original HBM tensor [T,H] reshaped (new view) to [T,2,128,H/256].
        #
        #           H-half-a                                           H-half-b
        # ┌─────────────────────────────────────────┬─────────────────────────────────────────┐
        # │ H0a │ H1a │ H2a │...│H63a│H64a│...│H127a│ H0b │ H1b │ H2b │...│H63b│H64b│...│H127b│ ← Token 0
        # ├─────────────────────────────────────────┼─────────────────────────────────────────┤
        # │ H0a │ H1a │ H2a │...│H63a│H64a│...│H127a│ H0b │ H1b │ H2b │...│H63b│H64b│...│H127b│ ← Token 1
        # ├─────────────────────────────────────────┴─────────────────────────────────────────┤
        # │ H0a │ H1a │ H2a │...│H63a│H64a│...│H127a│ H0b │ H1b │ H2b │...│H63b│H64b│...│H127b│ ← Token 2
        # └─────────────────────────────────────────┴─────────────────────────────────────────┘
        #                    ↑                                       ↑
        #                  Half-a: 128 H-tiles                   Half-b: 128 H-tiles
        #                  Each H-tile = nht2 elements          Each H-tile = nht2 elements
        #
        # SBUF layout [128, T, 2, H/256] where p-dim contains H elements with stride of H/256.
        # But returned as [128, T, H/128] (see below).
        #
        #   Token0  Token1  Token2  Token3  Token4
        #     ↓       ↓       ↓       ↓       ↓
        # ┌─────────────────────────────────────────────┐
        # │H0a,H0b│H0a,H0b│H0a,H0b│H0a,H0b│H0a,H0b...   │ ← 128 rows
        # │H1a,H1b│H1a,H1b│H1a,H1b│H1a,H1b│H1a,H1b      │   (P-dim)
        # │H2a,H2b│H2a,H2b│H2a,H2b│H2a,H2b│H2a,H2b      │   interleaved
        # │  ..   │  ..   │  ..   │  ..   │  ..         │   H/256 chunks
        # └─────────────────────────────────────────────┘
        #   ←nht→  ←nht→    ←nht→  ←nht→   ←nht→
        #   elem   elem     elem   elem    elem
        #
        # Each token occupies nht=H/128 columns, but H data is interleaved within with a chunk size of H/256.
        # Therefore the P-dim (i.e. each column of SBUF) contains H elements with stride of H/256.
        # This layout is amenable to LNC2 operation (though not necessary) sharded on H where each core could operate
        # on one half of the H data (i.e. NC0 operates on [:,:,0,:] and NC1 on [:,:,1,:]

        x_reshape = x.reshape((T, 2, P_MAX, num_h_tiles_by_2))  # HBM reshape
        x_sb = nl.ndarray(
            (P_MAX, T, 2, num_h_tiles_by_2), dtype=x.dtype, buffer=nl.sbuf
        )  # Declare SB tensor as 4D for now.
        nisa.dma_copy(
            src=tensor_view.TensorView(x_reshape).permute([2, 0, 1, 3]).get_view(),
            dst=tensor_view.TensorView(x_sb).get_view(),
        )
        # We reshape to 3D to maintain compatibility with existing upstream kernels who produce this this layout in this 3D shape:
        x_sb = x_sb.reshape((P_MAX, T, num_h_tiles))

    # sb_layout == 2

    elif hbm_layout == XHBMLayout_T_H__1 and sb_layout == XSBLayout_tp201__2:
        # Original HBM tensor [T,H] reshaped (new view) into num_h_tiles H-tiles of size 128.
        # num_h_tiles = H/128.
        #
        #      H-tiles (num_h_tiles total) →
        # ┌─────────┬─────────┬─────────┬─────┐
        # │   H0    │   H1    │   H2    │ ... │ ← Token 0  ↕ 1 elem
        # ├─────────┼─────────┼─────────┼─────┤
        # │   H0    │   H1    │   H2    │ ... │ ← Token 1  ↕ 1 elem
        # ├─────────┼─────────┼─────────┼─────┤
        # │   H0    │   H1    │   H2    │ ... │ ← Token 2  ↕ 1 elem
        # └─────────┴─────────┴─────────┴─────┘
        #   ←─128─→   ←─128─→   ←─128─→   ←─128─→
        #    elem      elem      elem      elem

        # SBUF shape [128, T, H/128]
        #
        #        Token0           Token1             Token2              ...
        #   ←─num_h_tiles─→   ←─num_h_tiles─→    ←─num_h_tiles─→
        # ┌─────────────────┬─────────────────┬─────────────────┬─────────────┐
        # │                 │                 │                 │             │
        # │                 │                 │                 │             │
        # │H0 H1 H2...H_n-1 │H0 H1 H2...H_n-1 │H0 H1 H2...H_n-1 │     ...     │ ← 128 rows (P-dim)
        # │                 │                 │                 │             │
        # │                 │                 │                 │             │
        # └─────────────────┴─────────────────┴─────────────────┴─────────────┘
        #
        # Each H-tile from HBM (of shape [1,128]) is written into SBUF, transposed, as [128,1].
        # This means that each column of SBUF (as you travel down the p-dim) contains consecutive H elements.

        x_reshape = x.reshape((T, num_h_tiles, P_MAX))
        x_sb = nl.ndarray((P_MAX, T, num_h_tiles), dtype=x.dtype, buffer=nl.sbuf)  # [128,T,H/128]
        # nisa.dma_copy(src=x_reshape[i_x_t, i_x_h, i_x_p], dst=x_sb[i_x_p, i_x_t, i_x_h])
        nisa.dma_copy(
            src=tensor_view.TensorView(x_reshape).permute([2, 0, 1]).get_view(),
            dst=tensor_view.TensorView(x_sb).get_view(),
        )
    # sb_layout == 3

    elif hbm_layout == XHBMLayout_H_T__0 and sb_layout == XSBLayout__128_Hdiv128_T__3:
        # HBM tensor [H,T] reshaped (new view) to [num_h_tiles, 128, T] where num_h_tiles = H/128.
        # In other words, tile on H-dimension in groups of 128.
        #
        # H-dim ↓     T columns →
        # ┌─────────────────────────────────┐
        # │ H0                              │ ← H-tile 0 (128 rows)
        # │                                 │
        # │                                 │
        # │ H127                            │
        # ├─────────────────────────────────┤
        # │ H128                            │ ← H-tile 1 (128 rows)
        # │                                 │
        # │                                 │
        # │ H255                            │
        # ├─────────────────────────────────┤
        # │ H256                            │ ← H-tile 2 (128 rows)
        # │                                 │
        # │                                 │
        # │ H383                            │
        # ├─────────────────────────────────┤
        # │              ...                │
        # ├─────────────────────────────────┤
        # │ H_(num_h_tiles-1)*128           │ ← H-tile num_h_tiles-1 (128 rows)
        # │                                 │
        # │                                 │
        # │ H-1                             │
        # └─────────────────────────────────┘

        # SBUF layout [128, num_h_tiles, T] where num_h_tiles = H/128.
        #
        # Dimension breakdown:
        # - dim 0: 128
        # - dim 1: num_h_tiles
        # - dim 2: T
        #
        #   ←─────T─────→       ←─────T─────→   ←─────T─────→          ←─────T─────→
        # ┌─────────────────┬─────────────────┬─────────────────┬───┬─────────────────────────┐
        # │ H0              │ H128            │ H256            │   │H_(num_h_tiles-1)*128    │
        # │                 │                 │                 │   │                         │ ← 128 rows
        # │   H-tile 0      │   H-tile 1      │   H-tile 2      │...│  H-tile num_h_tiles-1   │   (P-dim)
        # │                 │                 │                 │   │                         │
        # │ H127            │ H255            │ H383            │   │ H-1                     │
        # └─────────────────┴─────────────────┴─────────────────┴───┴─────────────────────────┘

        x_sb = nl.ndarray((P_MAX, num_h_tiles, T), dtype=x.dtype)

        nisa.dma_copy(
            src=(
                tensor_view.TensorView(x).reshape_dim(dim=0, shape=(num_h_tiles, P_MAX)).permute((1, 0, 2)).get_view()
            ),
            dst=tensor_view.TensorView(x_sb).get_view(),
        )

    else:
        # Should never reach here.
        kernel_assert(False, f"Combination of hbm_layout {hbm_layout} and sb_layout {sb_layout} not supported.")

    return x_sb


def router_topk_input_w_load(w: nl.ndarray, x_sb_layout, name=''):
    """
    Load weight tensor w from HBM to SBUF with layout matching x tensor.

    Performs DMA transfer from HBM to SBUF with layout conversion that matches
    the H-dimension stride pattern of the x tensor in SBUF, enabling efficient
    matmul operations.

    Args:
        w (nl.ndarray): Weight tensor [H, E] in HBM
        x_sb_layout (int): Layout of x in SBUF (0-3), determines w layout
        name (str): Optional tensor annotation name for debugging

    Returns:
        w_sb (nl.ndarray): Weight tensor in SBUF [128, H/128, E] with layout
                          matching x_sb_layout H-dimension stride pattern

    Notes:
        - H must be a multiple of 128
        - Layout must match x tensor layout for correct matmul contraction
        - Layout 0: H-tiles arranged horizontally with stride H/128
        - Layout 1: H-tiles with interleaved halves, stride H/256
        - Layouts 2-3: H-tiles with consecutive H elements
    """
    H, E = w.shape
    num_h_tiles = H // P_MAX
    num_h_tiles_by_2 = num_h_tiles // 2

    w_sb = None

    if x_sb_layout not in [0, 1, 2, 3]:
        kernel_assert(
            False,
            f"router_topk_input_w_load only x_sb_layout=0,1,2,3 are supported. Specified layout value: {x_sb_layout}",
        )

    if x_sb_layout == XSBLayout_tp102__0:
        # HBM tensor shape [H,E] reshaped (new view) as [128, num_h_tiles, E] where num_h_tiles = H/128.
        # A note about the terminology. In SBUF we aim to create num_h_tiles H-Tiles along the free-dim.
        # This is done by taking num_h_tiles rows from HBM and arranging them horizontally in the SB free-dim.
        # And we must do this for each SB partition-dimension row. We want to use common terminology between the
        # HBM and SB diagrams to show how HBM data is rearranged in SB.
        # Therefore we say that, in HBM, we have an H-tile-count of 128 and an H-tile-size of num_h_tiles=H/128.
        #
        # H-dim ↓     E columns →
        # ┌─────────────────────┐
        # │ H0                  │ ← H-Tile 0 (num_h_tiles rows)
        # │ H1                  │
        # │ H2                  │
        # │ ..                  │
        # │ H_num_h_tiles-1     │
        # ├─────────────────────┤
        # │ H_num_h_tiles       │ ← H-Tile 1 (num_h_tiles rows)
        # │ H_num_h_tiles+1     │
        # │ H_num_h_tiles+2     │
        # │ ..                  │
        # │ H_2*num_h_tiles-1   │
        # ├─────────────────────┤
        # │        ...          │ ← ... (more H-Tiles)
        # ├─────────────────────┤
        # │ H_127*num_h_tiles   │ ← H-Tile 127 (num_h_tiles rows)
        # │ H_127*num_h_tiles+1 │
        # │ H_127*num_h_tiles+2 │
        # │ ..                  │
        # │ H-1                 │
        # └─────────────────────┘

        # SBUF layout [128, num_h_tiles, E]
        #
        #        ←─E cols─→         ←─E cols─→         ←─E cols─→               ←─E cols─→
        # ┌───────────────────┬───────────────────┬───────────────────┬───┬───────────────────┐
        # │        H0         │        H1         │        H2         │...│  H_num_h_tiles-1  │ ← P-dim row 0 (H-Tile 0)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │   H_num_h_tiles   │  H_num_h_tiles+1  │  H_num_h_tiles+2  │...│ H_2*num_h_tiles-1 │ ← P-dim row 1 (H-Tile 1)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │        ...        │        ...        │        ...        │...│        ...        │ ← ... (more P-dim rows)
        # ├───────────────────┼───────────────────┼───────────────────┼───┼───────────────────┤
        # │ H_127*num_h_tiles │H_127*num_h_tiles+1│H_127*num_h_tiles+2│...│       H-1         │ ← P-dim row 127 (H-Tile 127)
        # └───────────────────┴───────────────────┴───────────────────┴───┴───────────────────┘
        #
        # Each P-dim row contains one H-Tile spread horizontally.
        # As we move down one column of SBUF we get H data with a stride of num_h_tiles, which matches the
        # the stride of the corresponding 'x' layout.

        w_reshape = w.reshape((P_MAX, num_h_tiles, E))
        w_sb = nl.ndarray((P_MAX, num_h_tiles, E), dtype=w.dtype, buffer=nl.sbuf, name=name)
        nisa.dma_copy(src=w_reshape, dst=w_sb)

    elif x_sb_layout == XSBLayout_tp2013__1:
        # First, understand the above diagram for x_sb_layout==0.
        # x_sb_layout==1 shrinks the H-tile size by half (num_h_tiles_by_2) and introduces an additional
        # dimension of 2 on the H dimension.
        # num_h_tiles = nht = H/128
        # num_h_tiles_by_2 = nht2 = nht/2 = H/256

        # HBM tensor shape [H,E] reshaped (new view) as [2, 128, num_h_tiles_by_2, E]
        # In other words, this is similar to x_sb_layout=0 except the H dimension is further divided into 2 halves.
        # Within each half, we take the same view as x_sb_layout=0 but each H-Tile contains half as many rows.

        # Half-a (upper half - 128 H-Tiles):
        # ┌─────────────────┐
        # │  H0             │ ← H-Tile 0 (nht2 rows)
        # │  H1             │
        # │  H2             │
        # │  ..             │
        # │  H_nht2-1       │
        # ├─────────────────┤
        # │  H_nht2         │ ← H-Tile 1 (nht2 rows)
        # │  H_nht2+1       │
        # │  H_nht2+2       │
        # │  ..             │
        # │  H_2*nht2-1     │
        # ├─────────────────┤
        # │        ...      │ ← ... (more H-Tiles)
        # ├─────────────────┤
        # │ H_127*nht2      │ ← H-Tile 127 (nht2 rows)
        # │ H_127*nht2+1    │
        # │ H_127*nht2+2    │
        # │  ..             │
        # │ H_128*nht2-1    │
        # ├─────────────────┤
        # Half-b (lower half - 128 H-Tiles):
        # ├─────────────────┤
        # │ H_128*nht2      │ ← H-Tile 0 (nht2 rows)
        # │ H_128*nht2+1    │
        # │ H_128*nht2+2    │
        # │  ..             │
        # │ H_129*nht2-1    │
        # ├─────────────────┤
        # │ H_129*nht2      │ ← H-Tile 1 (nht2 rows)
        # │ H_129*nht2+1    │
        # │ H_129*nht2+2    │
        # │  ..             │
        # │ H_130*nht2-1    │
        # ├─────────────────┤
        # │        ...      │ ← ... (more H-Tiles)
        # ├─────────────────┤
        # │ H_255*nht2      │ ← H-Tile 127 (nht2 rows)
        # │ H_255*nht2+1    │
        # │ H_255*nht2+2    │
        # │  ..             │
        # │        H-1      │
        # └─────────────────┘
        #
        # Total: 2 halves × 128 H-Tiles, each H-Tile containing num_h_tiles_by_2×E elements

        # SBUF layout [128, 2, nht2, E] where halves are in separate dim-1 slices.
        # But returned as [128, H/128, E] (see below).
        #
        #     Half-a (dim-1=0)                                          Half-b (dim-1=1)
        #      ←─E─→        ←─E─→        ←─E─→                   ←─E─→           ←─E─→            ←─E─→
        # ┌─────────────┬─────────────┬─────────────┬───┐ ┌─────────────────┬─────────────────┬─────────────────┬───┐
        # │     H0      │     H1      │     H2      │...│ │   H_128*nht2    │  H_128*nht2+1   │  H_128*nht2+2   │...│ P-dim row 0 (H-Tile 0)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │   H_nht2    │  H_nht2+1   │  H_nht2+2   │...│ │   H_129*nht2    │  H_129*nht2+1   │  H_129*nht2+2   │...│ P-dim row 1 (H-Tile 1)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │     ...     │     ...     │     ...     │...│ │       ...       │       ...       │       ...       │...│ ... (more H-Tiles)
        # ├─────────────┼─────────────┼─────────────┼───┤ ├─────────────────┼─────────────────┼─────────────────┼───┤
        # │ H_127*nht2  │H_127*nht2+1 │H_127*nht2+2 │...│ │   H_255*nht2    │  H_255*nht2+1   │  H_255*nht2+2   │...│ P-dim row 127 (H-Tile 127)
        # └─────────────┴─────────────┴─────────────┴───┘ └─────────────────┴─────────────────┴─────────────────┴───┘
        #
        # Each P-dim row contains one HBM H-Tile from Half-a followed by one HBM H-Tile from Half-b.
        # As we move down one column of SBUF we get H data with a stride of num_h_tiles_by_2, which matches the
        #   the stride of the corresponding 'x' layout.

        w_reshape = w.reshape((2, P_MAX, num_h_tiles_by_2, E))
        w_sb = nl.ndarray((P_MAX, 2, num_h_tiles_by_2, E), dtype=w.dtype, buffer=nl.sbuf, name=name)
        nisa.dma_copy(
            src=tensor_view.TensorView(w_reshape).permute([1, 0, 2, 3]).get_view(),
            dst=tensor_view.TensorView(w_sb).get_view(),
        )
        # We always return a 3D shape. The extra H-dimension of [0:2], while necessary for loading from HBM->SBUF above,
        # is not necessary when an eventual matmul reads from SBUF. It will simply read the entire free-dim in order.
        w_sb = w_sb.reshape((P_MAX, num_h_tiles, E))

    else:  # x_sb_layout = 2,3
        # HBM tensor shape [H,E] reshaped (new view) as [num_h_tiles, 128, E].
        # Divide the H dimension into num_h_tiles groups of 128 rows.
        #
        # H-dim ↓     E columns →
        # ┌─────────────────────────────────┐
        # │ H0          H-tile 0            │
        # │                                 │
        # │                                 │
        # │ H127                            │
        # ├─────────────────────────────────┤
        # │ H128        H-tile 1            │
        # │                                 │
        # │                                 │
        # │ H255                            │
        # ├─────────────────────────────────┤
        # │ H256        H-tile 2            │
        # │                                 │
        # │                                 │
        # │ H383                            │
        # ├─────────────────────────────────┤
        # │              ...                │
        # ├─────────────────────────────────┤
        # │ H_(num_h_tiles-1)*128           │
        # │         H-tile num_h_tiles-1    │
        # │                                 │
        # │ H-1                             │
        # └─────────────────────────────────┘
        #
        # SBUF layout [128, num_h_tiles, E]
        #
        #   ←─────E─────→     ←─────E─────→     ←─────E─────→             ←─────E─────→
        # ┌─────────────────┬─────────────────┬─────────────────┬───┬─────────────────────────┐
        # │ H0              │ H128            │ H256            │   │ H_(num_h_tiles-1)*128   │
        # │                 │                 │                 │   │                         │
        # │   H-tile 0      │   H-tile 1      │   H-tile 2      │...│ H-tile num_h_tiles-1    │ ← 128 rows
        # │                 │                 │                 │   │                         │   (P-dim)
        # │ H127            │ H255            │ H383            │   │ H-1                     │
        # └─────────────────┴─────────────────┴─────────────────┴───┴─────────────────────────┘
        #
        # H-tiles arranged left to right.
        # Therefore each column of SBUF contains H data with a stride of 1 (i.e. consecutive H data).

        w_sb = nl.ndarray((P_MAX, num_h_tiles, E), dtype=w.dtype, buffer=nl.sbuf, name=name)

        nisa.dma_copy(
            src=(
                tensor_view.TensorView(w).reshape_dim(dim=0, shape=(num_h_tiles, P_MAX)).permute((1, 0, 2)).get_view()
            ),
            dst=tensor_view.TensorView(w_sb).get_view(),
        )

    return w_sb


def compute_activation(
    dst_tensor,
    input_tensor,
    t_tile: tiled_range.TiledRangeIterator,
    act_fn,
    negmax_sb=None,
    exp_sum_sb=None,
    complete_activation=False,
    return_intermediates=False,
    input_dtype=nl.float32,
) -> None:
    """
    Compute activation function intermediates and optionally complete activation result.

    Args:
      input_tensor: Input tensor to apply activation to. Shape [P, num_t_tiles, F] where dim-1 is indexed by t_tile_idx.
      t_tile_idx: Tile index for current iteration
      act_fn: Activation function (common_types.RouterActFnType.SOFTMAX or common_types.RouterActFnType.SIGMOID)
      negmax_sb: Output tensor for negative max values (required if complete_activation=False or return_intermediates=True, when act_fn=SOFTMAX)
      exp_sum_sb: Output tensor for exponential sum (required if complete_activation=False or return_intermediates=True, when act_fn=SOFTMAX)
      complete_activation: If True, complete the activation and return final values (ignored for sigmoid)
      return_intermediates: If True, return intermediate tensors even when complete_activation=True
      input_dtype: Data type for final output

    Returns:
      activation output tensor (always returns result for sigmoid, conditionally for softmax)
      When return_intermediates=True and complete_activation=True, intermediates are stored in provided tensors
    """
    t_tile_idx = t_tile.index
    t_tile_size = t_tile.size
    if act_fn == common_types.RouterActFnType.SIGMOID:
        nisa.activation(
            dst=dst_tensor[:t_tile_size, t_tile_idx, :], op=nl.sigmoid, data=input_tensor[:t_tile_size, t_tile_idx, :]
        )

    elif act_fn == common_types.RouterActFnType.SOFTMAX:
        # Use provided tensors or create internal ones
        local_negmax_sb = negmax_sb
        local_exp_sum_sb = exp_sum_sb

        if complete_activation and not return_intermediates:
            # Must create local tensors for intermediates
            local_negmax_sb = nl.ndarray((t_tile_size, input_tensor.shape[1]), dtype=nl.float32, buffer=nl.sbuf)
            local_exp_sum_sb = nl.ndarray((t_tile_size, input_tensor.shape[1], 1), dtype=nl.float32, buffer=nl.sbuf)

        # Compute intermediate values
        nisa.tensor_reduce(
            dst=local_negmax_sb[:t_tile_size, t_tile_idx : t_tile_idx + 1],
            op=nl.maximum,
            data=input_tensor[:t_tile_size, t_tile_idx, :],
            axis=1,
            negate=True,
            keepdims=True,
        )
        result_exp = nl.ndarray((t_tile_size, 1, input_tensor.shape[2]), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=result_exp,
            op=nl.exp,
            data=input_tensor[:t_tile_size, t_tile_idx : t_tile_idx + 1, :],
            bias=local_negmax_sb[:t_tile_size, t_tile_idx : t_tile_idx + 1],
            reduce_op=nl.add,
            reduce_res=(
                tensor_view.TensorView(local_exp_sum_sb)
                .slice(dim=0, start=0, end=t_tile_size)
                .select(dim=1, index=t_tile_idx)
                .get_view()
            ),
            reduce_cmd=reduce_cmd.reset_reduce,
        )

        nisa.reciprocal(
            dst=local_exp_sum_sb[:t_tile_size, t_tile_idx, :], data=local_exp_sum_sb[:t_tile_size, t_tile_idx, :]
        )

        # Complete softmax if requested
        if complete_activation:
            nisa.tensor_scalar(
                dst=dst_tensor[:t_tile_size, t_tile_idx, :],
                data=result_exp,
                op0=nl.multiply,
                operand0=local_exp_sum_sb[:t_tile_size, t_tile_idx, :],
            )

        return None

    else:
        kernel_assert(False, f"Unsupported activation function: {act_fn}")
