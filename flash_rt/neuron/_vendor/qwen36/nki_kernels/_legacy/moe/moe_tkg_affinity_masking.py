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

"""Expert affinity masking utilities for all-expert MoE kernels.

Vendored from nkilib/core/moe/moe_tkg/moe_tkg_affinity_masking.py @
nkilib 0.1.0+gd4920de. Only edit relative to upstream: relative `...`
imports rewritten to absolute `nkilib.core.*` imports.

This file is dead code on the gpt-oss megakernel path (`is_all_expert=False`
is always passed). Vendored anyway because moe_tkg.py imports it at module
load time, but the inner functions are never invoked. The hardcoded
`name=` literals on lines 145/184 therefore never produce ops; no
prefix transform is required.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa import engine
from nki.isa.constants import dge_mode

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# Hardware constants
_PARTITION_ALIGNMENT = 32  # Partition alignment for tensor operations
_DGE_ALIGNMENT = 16  # DMA gather engine alignment requirement


def mask_expert_affinities(
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    rank_id: nl.ndarray,
    E_L: int,
    T: int,
    K: int,
    io_dtype,
    mask_unselected_experts: bool,
    output_in_sbuf: bool,
) -> nl.ndarray:
    """
    Mask expert affinities for all-expert MoE computation.

    Slices the global expert affinities to local experts based on rank_id, and optionally
    masks affinities by checking expert_index against each local expert (when mask_unselected_experts=True).

    Args:
        expert_affinities (nl.ndarray): [T, E], Global expert affinities in HBM.
        expert_index (nl.ndarray): [T, K], Top-K expert indices per token in SBUF.
        rank_id (nl.ndarray): [1, 1], Rank ID tensor in HBM, specifies which experts this rank processes.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        io_dtype: Data type for computation.
        mask_unselected_experts (bool): If True, apply masking based on expert_index.
        output_in_sbuf (bool): If True, output affinities in SBUF; otherwise in shared HBM.

    Returns:
        expert_affinities_masked (nl.ndarray): [T, E_L]@SBUF or [128_T, T/128, E_L]@SBUF or [T, E_L]@HBM

    Notes:
        - For T <= 128, returns 2D tensor [T, E_L]
        - For T > 128, returns 3D tiled tensor [T_par, n_T128_tiles, E_L]
        - Uses indirect DMA with scalar_offset for rank-based expert slicing
    """
    kernel_assert(
        output_in_sbuf or not mask_unselected_experts,
        f"mask_unselected_experts=True only supports output_in_sbuf=True, "
        f"but got {mask_unselected_experts=}, {output_in_sbuf=}",
    )

    # Load rank_id to SBUF
    rank_id_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(src=rank_id[0:1, 0:1], dst=rank_id_sbuf[0:1, 0:1])

    # Calculate expert offset: expert_offset = rank_id * E_L
    expert_offset_sbuf = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=expert_offset_sbuf[0:1, 0:1],
        data=rank_id_sbuf[0:1, 0:1],
        op0=nl.multiply,
        operand0=E_L,
    )

    # Slice expert affinities based on rank_id, load to SBUF
    if output_in_sbuf:
        expert_affinities_masked = _load_slice_affinities_sbuf(
            expert_affinities=expert_affinities,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            io_dtype=io_dtype,
        )
    # Slice expert affinities into HBM intermediate buffer
    else:
        expert_affinities_masked = _slice_affinities_hbm(
            expert_affinities=expert_affinities,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            io_dtype=io_dtype,
        )

    # Apply masking based on expert_index when mask_unselected_experts=True
    if mask_unselected_experts:
        _apply_expert_index_mask(
            expert_affinities_masked=expert_affinities_masked,
            expert_index=expert_index,
            expert_offset_sbuf=expert_offset_sbuf,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=io_dtype,
        )

    return expert_affinities_masked


def _load_slice_affinities_sbuf(expert_affinities, expert_offset_sbuf, E_L, T, io_dtype):
    """Load and slice expert affinities from HBM to SBUF based on rank offset."""
    E = expert_affinities.shape[1]

    if T <= 128:
        # 2D output [T, E_L]
        expert_affinities_masked = nl.ndarray((T, E_L), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=expert_affinities.ap(
                pattern=[[E, T], [1, E_L]],
                offset=0,
                scalar_offset=expert_offset_sbuf,
                indirect_dim=1,
            ),
            dst=expert_affinities_masked[0:T, 0:E_L],
            dge_mode=dge_mode.unknown if T % _DGE_ALIGNMENT == 0 else dge_mode.swdge,
        )
    else:
        # 3D tiled output [T_par, n_T128_tiles, E_L] for T > 128
        T_par = nl.tile_size.pmax
        n_T128_tiles = div_ceil(T, T_par)
        last_tile_T = T % T_par
        has_partial_tile = last_tile_T != 0
        n_full_tiles = n_T128_tiles - 1 if has_partial_tile else n_T128_tiles

        expert_affinities_masked = nl.ndarray(
            (T_par, n_T128_tiles, E_L), dtype=io_dtype, buffer=nl.sbuf, name="expert_affinities_masked"
        )

        if n_full_tiles > 0:
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, T_par], [T_par * E, n_full_tiles], [1, E_L]],
                    offset=0,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:T_par, 0:n_full_tiles, 0:E_L],
                dge_mode=dge_mode.unknown,
            )

        if has_partial_tile:
            partial_tile_offset = n_full_tiles * T_par * E
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[E, last_tile_T], [T_par * E, 1], [1, E_L]],
                    offset=partial_tile_offset,
                    scalar_offset=expert_offset_sbuf,
                    indirect_dim=1,
                ),
                dst=expert_affinities_masked[0:last_tile_T, n_full_tiles : n_full_tiles + 1, 0:E_L],
                dge_mode=dge_mode.unknown if last_tile_T % _DGE_ALIGNMENT == 0 else dge_mode.swdge,
            )

    return expert_affinities_masked


def _slice_affinities_hbm(expert_affinities, expert_offset_sbuf, E_L, T, io_dtype):
    """Load and slice expert affinities to shared HBM with LNC sharding."""
    E = expert_affinities.shape[1]
    _, n_prgs, prg_id = get_verified_program_sharding_info()
    T_shard = T // 2 if n_prgs > 1 else T
    T_offset = T_shard * prg_id

    expert_affinities_masked = nl.ndarray(
        (T, E_L), dtype=io_dtype, buffer=nl.shared_hbm, name="expert_affinities_masked_after_E_L_slice"
    )
    nisa.dma_copy(
        src=expert_affinities.ap(
            pattern=[[E, T_shard], [1, E_L]],
            offset=T_offset,
            scalar_offset=expert_offset_sbuf,
            indirect_dim=1,
        ),
        dst=expert_affinities_masked[nl.ds(T_offset, T_shard), :],
        dge_mode=dge_mode.unknown if T % _DGE_ALIGNMENT == 0 else dge_mode.swdge,
    )
    nisa.core_barrier(expert_affinities_masked, (0, 1))

    return expert_affinities_masked


def _apply_expert_index_mask(
    expert_affinities_masked: nl.ndarray,
    expert_index: nl.ndarray,
    expert_offset_sbuf: nl.ndarray,
    E_L: int,
    T: int,
    K: int,
    io_dtype,
):
    """
    Apply masking to expert affinities based on expert_index.

    For each local expert, checks if it was selected in expert_index for each token.
    Zeros out affinities for experts not selected by each token.

    Args:
        expert_affinities_masked (nl.ndarray): [T, E_L], Affinities to mask in-place in SBUF.
        expert_index (nl.ndarray): [T, K], Top-K expert indices in SBUF.
        expert_offset_sbuf (nl.ndarray): [1, 1], Starting expert index for this rank in SBUF.
        E_L (int): Number of local experts.
        T (int): Number of tokens.
        K (int): Top-K value.
        io_dtype: Data type for computation.

    Notes:
        - Modifies expert_affinities_masked in-place
        - Iterates through each local expert and applies element-wise masking
    """
    T_32s = _PARTITION_ALIGNMENT * div_ceil(T, _PARTITION_ALIGNMENT)

    # Load expert_index to SBUF if it's in HBM
    if expert_index.buffer == nl.shared_hbm:
        expert_index_sbuf = nl.ndarray((T, K), dtype=expert_index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=expert_index[0:T, 0:K], dst=expert_index_sbuf[0:T, 0:K])
    else:
        expert_index_sbuf = expert_index

    # Broadcast expert_offset to [T_32s, K] for comparison
    expert_offset_broadcast = nl.ndarray((T_32s, 1), dtype=nl.uint32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=expert_offset_sbuf[0:1, 0:1], dst=expert_offset_broadcast[0:T_32s, 0:1])

    expert_offset_f = nl.ndarray((T_32s, K), dtype=nl.float32, buffer=nl.sbuf)
    for k_idx in nl.affine_range(K):
        if k_idx % 2 == 0:
            nisa.tensor_copy(
                src=expert_offset_broadcast[0:T_32s, 0:1],
                dst=expert_offset_f[0:T_32s, k_idx : k_idx + 1],
                engine=engine.vector,
            )
        else:
            nisa.tensor_copy(
                src=expert_offset_broadcast[0:T_32s, 0:1],
                dst=expert_offset_f[0:T_32s, k_idx : k_idx + 1],
                engine=engine.scalar,
            )

    # For each local expert, mask affinities
    for expert_idx in nl.affine_range(E_L):
        # Check expert_index against current expert: [T, K] comparison
        expert_check = nl.ndarray((T, K), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=expert_check[0:T, 0:K],
            data1=expert_index_sbuf[0:T, 0:K],
            data2=expert_offset_f[0:T, 0:K],
            op=nl.equal,
        )

        # Sum across K dimension to get match indicator [T, 1]
        expert_match = nl.ndarray((T, 1), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=expert_match,
            op=nl.add,
            data=expert_check,
            axis=1,
        )

        # Multiply affinities by match indicator
        nisa.tensor_tensor(
            dst=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data1=expert_affinities_masked[0:T, expert_idx : expert_idx + 1],
            data2=expert_match[0:T, 0:1],
            op=nl.multiply,
        )

        # Increment expert_offset_f by 1 for next iteration
        nisa.tensor_scalar(
            dst=expert_offset_f[0:T, 0:K],
            data=expert_offset_f[0:T, 0:K],
            op0=nl.add,
            operand0=1,
        )
