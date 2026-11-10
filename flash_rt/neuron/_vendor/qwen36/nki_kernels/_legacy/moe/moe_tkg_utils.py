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

"""Utility functions for MoE token generation kernels including expert affinity gathering and broadcasting.

Vendored from nkilib/core/moe/moe_tkg/moe_tkg_utils.py @ nkilib 0.1.0+gd4920de.
Only edit relative to upstream: relative `...` imports rewritten to absolute
`nkilib.core.*` imports. All `name=` literals in this file are on
`sbm.alloc_*` which auto-prefix via the megakernel's outer
`sbm.set_name_prefix(...)`.
"""

import nki.isa as nisa
import nki.language as nl

from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstantsDimensionSizes
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nkilib.core.utils.tensor_view import TensorView


def gather_expert_affinities(
    expert_affinities_sb: nl.ndarray,
    expert_idx: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Gathers expert affinities based on expert indices using local_gather operation.

    This function collects expert affinities for each token based on the expert indices.
    It handles different token count scenarios and performs necessary transpositions
    and local gather operations to prepare affinities for broadcasting.

    Args:
        expert_affinities_sb (nl.ndarray): [_pmax, E], Tensor containing expert affinities in SBUF.
        expert_idx (nl.ndarray): [T, K], Expert indices for each token.
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object containing T, K, _pmax and other constants.
        sbm (SbufManager): SBUF memory manager for allocation.

    Returns:
        gathered_affinities (nl.ndarray): [_pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
            Gathered affinities tensor.

    Notes:
        - Uses different strategies for T <= 16 vs T > 16 for optimization
        - PARTITIONS_PER_GPSIMD_CORE = 16 (partitions per GPSIMD core)
        - PARTITIONS_PER_QUADRANT = 32 (partitions per quadrant)
    """
    # Hardware-specific constants
    PARTITIONS_PER_GPSIMD_CORE = 16  # Number of partitions per GPSIMD core
    kernel_assert(dims.K <= PARTITIONS_PER_GPSIMD_CORE, f"top_k {dims.K} exceeds {PARTITIONS_PER_GPSIMD_CORE}")
    kernel_assert(dims.E > 1, f"E={dims.E} must be > 1 for MoE (local_gather requires src_buffer_size > 1)")

    if dims.T <= PARTITIONS_PER_GPSIMD_CORE:
        # Optimized path for small token counts (T <= 16)

        # Convert expert indices to uint16 for local_gather operation
        expert_idx_u16 = sbm.alloc_stack(
            (dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="expert_idx_u16"
        )
        nisa.memset(dst=expert_idx_u16, value=0)
        nisa.tensor_copy(dst=expert_idx_u16[0 : dims.T, 0 : dims.K], src=expert_idx[0 : dims.T, 0 : dims.K])

        # Prepare index values for gathering
        index_values = sbm.alloc_stack(
            (dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="index_values"
        )
        nisa.memset(dst=index_values, value=0)
        expert_indices_trans = sbm.alloc_stack(
            (PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE),
            dtype=nl.uint16,
            buffer=nl.sbuf,
            name="expert_indices_trans",
        )
        nisa.nc_transpose(
            dst=expert_indices_trans,
            data=expert_idx_u16[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(
            dst=index_values[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
            src=expert_indices_trans,
        )

    else:
        # Path for larger token counts (T > 16)
        # Use DMA_copy to avoid partition alignment problems
        active_channels = (dims.T + PARTITIONS_PER_GPSIMD_CORE - 1) // PARTITIONS_PER_GPSIMD_CORE

        # Convert expert indices to uint16 for local_gather operation
        expert_idx_u16 = sbm.alloc_stack(
            (128, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="expert_idx_u16"
        )
        nisa.memset(dst=expert_idx_u16, value=0)
        nisa.tensor_copy(dst=expert_idx_u16[0 : dims.T, 0 : dims.K], src=expert_idx[0 : dims.T, 0 : dims.K])

        # Fill out 16 partition layout requirement in blocks of 16 partitions up to 128 partitions total
        index_values = sbm.alloc_stack(
            (dims._pmax, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="index_values", align=32
        )
        nisa.memset(dst=index_values, value=0)
        for channel_idx in range(active_channels):
            # Use DMA Transpose for better performance with larger token counts
            nisa.dma_transpose(
                dst=index_values.ap(
                    pattern=[
                        [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                        [1, 1],
                        [1, 1],
                        [1, PARTITIONS_PER_GPSIMD_CORE],
                    ],
                    offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                ),
                src=expert_idx_u16.ap(
                    pattern=[
                        [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                        [1, 1],
                        [1, 1],
                        [1, PARTITIONS_PER_GPSIMD_CORE],
                    ],
                    offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                ),
            )

    # Perform local gather to collect affinities based on indices
    gathered_affinities_sb = sbm.alloc_stack(
        (dims._pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE),
        dtype=expert_affinities_sb.dtype,
        buffer=nl.sbuf,
        name="gathered_affinities_sb",
    )
    ga_sb_fdim = PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE

    # num_valid_indices is hard-coded due to compiler limitation
    nisa.memset(dst=gathered_affinities_sb, value=0.0)
    nisa.local_gather(
        dst=gathered_affinities_sb.ap([[ga_sb_fdim, dims._pmax], [1, ga_sb_fdim]]),
        src_buffer=expert_affinities_sb,
        index=index_values[:, :],
        num_elem_per_idx=1,
        num_valid_indices=ga_sb_fdim,
    )

    return gathered_affinities_sb


def broadcast_token_affinity(
    dst: nl.ndarray,
    gathered_affinities_sb: nl.ndarray,
    token_index: int,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Broadcasts expert affinities for a specific token across partitions.

    This function takes gathered affinities and broadcasts the affinities for a specific
    token across all partitions, ensuring proper alignment with hardware constraints.

    Args:
        dst (nl.ndarray): Destination tensor for broadcasted affinities.
        gathered_affinities_sb (nl.ndarray): [_pmax, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
            Gathered affinities tensor.
        token_index (int): Index of the current token being processed (i_t).
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object containing K, _pmax and other constants.
        sbm (SbufManager): SBUF memory manager for allocation.

    Returns:
        broadcasted_affinities (nl.ndarray): [_pmax, K], Broadcasted token affinities ready for computation.

    Notes:
        - PARTITIONS_PER_GPSIMD_CORE = 16 (partitions per GPSIMD core)
        - PARTITIONS_PER_QUADRANT = 32 (partitions per quadrant)
        - Uses stream shuffle for proper partition alignment
    """
    # Hardware-specific constants
    PARTITIONS_PER_GPSIMD_CORE = 16  # Number of partitions per GPSIMD core
    PARTITIONS_PER_QUADRANT = 32  # Number of partitions per quadrant

    # Calculate partition and quadrant positions for the current token
    current_partition_channel = token_index % PARTITIONS_PER_GPSIMD_CORE  # Active Partition Channel 0..15
    current_quadrant_group = token_index // PARTITIONS_PER_QUADRANT  # Partition groups of 32
    current_quadrant_channel = token_index % PARTITIONS_PER_QUADRANT  # Active Quadrant Channel 0..31

    # Select token affinities from gathered data [T, PARTITIONS_PER_GPSIMD_CORE]
    token_affinities = gathered_affinities_sb[:, current_partition_channel, :]

    # Create shuffle mask for partition alignment
    shuffle_mask = [current_quadrant_channel] * PARTITIONS_PER_QUADRANT

    # Perform stream shuffle to align token affinities with partitions
    token_affinities_partition_aligned = sbm.alloc_stack(
        (dims._pmax, PARTITIONS_PER_GPSIMD_CORE),
        dtype=gathered_affinities_sb.dtype,
        buffer=nl.sbuf,
        name="token_affinities_partition_aligned",
    )
    nisa.nc_stream_shuffle(src=token_affinities, dst=token_affinities_partition_aligned, shuffle_mask=shuffle_mask)

    # Select the appropriate quadrant group and broadcast across all partitions
    quadrant_start = current_quadrant_group * PARTITIONS_PER_QUADRANT
    quadrant_end = quadrant_start + 1
    stream_shuffle_broadcast(src=token_affinities_partition_aligned[quadrant_start:quadrant_end, : dims.K], dst=dst)


def reshape_scale_for_mlp(scale_tensor: TensorView):
    """
    Reshapes scale tensor for MLP operations by expanding and broadcasting.

    Args:
        scale_tensor (TensorView): Scale tensor to reshape.

    Returns:
        TensorView: Reshaped scale tensor with expanded dimension 0 and broadcasted to size 128.

    Notes:
        - Expands dimension 0 and broadcasts to partition size (128)
    """
    return scale_tensor.expand_dim(dim=0).broadcast(dim=0, size=128)


def safe_tensor_view(tensor: nl.ndarray):
    """
    Creates a TensorView wrapper for a tensor if it is not None.

    Args:
        tensor (nl.ndarray): Tensor to wrap, or None.

    Returns:
        TensorView or None: TensorView wrapper if tensor != None, otherwise None.

    Notes:
        - Safe wrapper to handle optional tensor inputs
    """
    return TensorView(tensor) if tensor != None else None
