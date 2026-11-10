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

"""Gate and Up projection sub-kernels for MLP TKG with column tiling and LHS/RHS swap modes."""

import os as _os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import SbufManager, sizeinbytes
from nkilib.core.utils.interleave_copy import interleave_copy
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.tiled_range import TiledRange
from nkilib.core.mlp.mlp_parameters import (
    MLPParameters,
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)
from nkilib.core.mlp.mlp_tkg.mlp_tkg_utils import adaptive_dge_mode

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode

# Falls back to upstream's per-HTile weight ring (no hoist, no cross-expert prefetch).
_MOE_LEGACY_GATE_UP_WEIGHT_LOAD = _os.environ.get("NKI_MOE_LEGACY_GATE_UP_WEIGHT_LOAD", "0") == "1"


def emit_hoisted_gate_up_dma(
    unsharded_weight: TensorView,
    hoisted_weight: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    shard_dim_hidden: tuple[int, int],
    shard_dim_intr: tuple[int, int],
    dge_mode: int = None,
):
    """
    Issues a hoisted gate/up weight DMA into a pre-allocated SBUF tile.

    Split into two H1-halves so the matmul on the first half can start before the
    second half lands. Shared between the in-function hoist and the cross-expert
    prefetch ring in selective_expert_impl.

    When the source view has dynamic (indirect) access, falls back to the unknown
    DGE mode — HWDGE/SWDGE are not safe for scalar_offset/indirect_dim patterns.
    """
    H0 = dims.H0
    shared_I = shard_dim_intr[1] - shard_dim_intr[0]
    full_weight_view = (
        unsharded_weight.slice(dim=0, start=shard_dim_hidden[0], end=shard_dim_hidden[1])
        .reshape_dim(dim=0, shape=(H0, dims.H1_shard))
        .slice(dim=2, start=shard_dim_intr[0], end=shard_dim_intr[1])
    )
    if full_weight_view.has_dynamic_access():
        _hoisted_dge = _DGE_MODE_UNKNOWN
    else:
        _hoisted_dge = dge_mode if dge_mode is not None else nisa.dge_mode.hwdge
    half_h1 = dims.H1_shard // 2
    nisa.dma_copy(
        dst=hoisted_weight[0:H0, 0:half_h1, 0:shared_I],
        src=full_weight_view.slice(dim=1, start=0, end=half_h1).get_view(),
        dge_mode=_hoisted_dge,
    )
    nisa.dma_copy(
        dst=hoisted_weight[0:H0, half_h1:dims.H1_shard, 0:shared_I],
        src=full_weight_view.slice(dim=1, start=half_h1, end=dims.H1_shard).get_view(),
        dge_mode=_hoisted_dge,
    )


def gate_up_projection(
    hidden: nl.ndarray,
    unsharded_weight: nl.ndarray,
    shard_dim_hidden: tuple[int, int],
    shard_dim_intr: tuple[int, int],
    bias: nl.ndarray,
    dequant_scale: TensorView,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    op_name: str,
    sbm: SbufManager,
):
    """
    Performs a single Gate or Up projection shard on the H.

    Computes: Hidden[H, T] @ Weight[H, I] + Optional(Bias[1, I]) → [T, I]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    Weight tiles are loaded [HTile, I] at a time for efficient memory access:
    H/HTile * [ HTile/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ] ]

    Column Tiling Optimization:
    For small T, column tiling improves performance by fully utilizing PE engine space.
    E.g., if T=32, the hidden tile [128, 32] leaves unused 32:128 column space in PE engine.

    After Column Tiling:
    ---------------------------
    | col_tile_1 | col_tile_2 | col_tile_3 | col_tile_4 |
    | 32 columns | 32 columns | 32 columns | 32 columns |
    ---------------------------
    - `column_tiling_dim` = [32, 64, 128], chosen based on T.
    - `column_tiling_factor` = 128 / column_tiling_dim, with a maximum factor of 4 → up to 4× speedup.
    - `column_tile` = HTile / column_tiling_factor
    H/HTile * HTile/column_tiling_factor(parallel execution) * column_tile/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ]

    Key Points:
    -----------
    - Intermediate projection tensors are always fp32 for better numerical accuracy
    - Bias is applied on one core only to avoid double-counting (sharding along H)
    - Matrix multiplication is tiled along H and I
    - Column tiling improves PE utilization for small T

    Returns:
        Output tensor with shape [T, I]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0, T, H1 = hidden.shape
    H = shard_dim_hidden[1] - shard_dim_hidden[0]
    I = shard_dim_intr[1] - shard_dim_intr[0]
    i_offset = shard_dim_intr[0]
    num_allocated_w_tile = tiles.num_allocated_w_tile
    is_up_proj = "up" in op_name

    # Sanity checks for sharding
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection supports I <= {dims.max_I_shard_size}",
    )
    kernel_assert(
        H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles as a ring buffer so up weights load after gate weights.
    weight_base_idx = tiles.num_HTiles % num_allocated_w_tile if is_up_proj else 0

    # ---------- Allocate PSUM buffers ----------
    result_psums = []
    for psum_idx in range(tiles.num_allocated_psums):
        psum_bank = psum_idx + tiles.up_psum_base_bank if is_up_proj else psum_idx
        result_psum = nl.ndarray(
            (dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_{sbm.get_name_prefix()}_psum_{i_offset}_{psum_bank}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_bank * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along H)
    is_bias = bias is not None and dims.shard_id == 0
    if is_bias:
        # Load bias with broadcast across T dimension using TensorView
        # [1, I_total] -> slice to [1, I] -> broadcast to [T, I]
        bias_hbm_view = TensorView(bias).slice(dim=1, start=i_offset, end=i_offset + I).broadcast(dim=0, size=T)
        bias_tile_view = TensorView(bias_tile).slice(dim=1, start=0, end=I)
        nisa.dma_copy(
            dst=bias_tile_view.get_view(),
            src=bias_hbm_view.get_view(),
            dge_mode=_DGE_MODE_NONE,
        )

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        dequant_scale_view = dequant_scale.slice(dim=0, start=0, end=T).slice(dim=1, start=i_offset, end=i_offset + I)
        nisa.dma_copy(
            dst=dequant_tile[0:T, 0:I],
            src=dequant_scale_view.get_view(),
            dge_mode=_DGE_MODE_NONE,
        )

    # ---------- Matrix multiplication ----------
    used_columns = 0

    # Gate Up Projection
    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Compute start offset
        h_offset = hidden_tiles.index * tiles.num_128_tiles_per_HTile
        weight_shard_offset = shard_dim_hidden[0] * unsharded_weight.shape[1] + i_offset
        h1_tiles = hidden_tiles.size // H0

        # Load weight tile [HTile, I] → SBUF layout [H0, HTile/H0, I]
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        nisa.dma_copy(
            dst=weight_tiles[weight_idx][0:H0, 0:h1_tiles, 0:I],
            src=unsharded_weight.ap(
                pattern=[
                    [H1 * unsharded_weight.shape[1], H0],
                    [unsharded_weight.shape[1], h1_tiles],
                    [1, I],
                ],
                offset=h_offset * dims.I + weight_shard_offset,
                dtype=nl.float8_e4m3 if str(unsharded_weight.dtype) == "float8e4" else unsharded_weight.dtype,
            ),
            dge_mode=_DGE_MODE_NONE,
        )

        # Matmult
        for column_tile in TiledRange(h1_tiles, dims.column_tiling_factor):
            for column_idx in range(column_tile.size):
                column_tile_offset = dims.column_tiling_factor * column_tile.index + column_idx
                for i_tiles in TiledRange(I, dims._psum_fmax):
                    nisa.nc_matmul(
                        dst=result_psums[i_tiles.index][
                            nl.ds(dims.column_tiling_dim * column_idx, T),
                            0 : i_tiles.size,
                        ],
                        stationary=hidden.ap(
                            pattern=[[T * H1, H0], [H1, T]],
                            offset=h_offset + column_tile_offset,
                        ),
                        moving=weight_tiles[weight_idx][
                            0:H0,
                            column_tile_offset,
                            nl.ds(i_tiles.start_offset, i_tiles.size),
                        ],
                        tile_position=(0, dims.column_tiling_dim * column_idx),
                        tile_size=(H0, dims.column_tiling_dim),
                    )
            # Update used column numbers
            used_columns = max(used_columns, column_tile.size)

    # ---------- Accumulate PSUMs into output ----------
    for i_tiles in TiledRange(I, dims._psum_fmax):
        dst_offset = shard_dim_intr[0] + i_tiles.start_offset
        # Copy PSUM to SBUF
        nisa.activation(
            dst=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
            data=result_psums[i_tiles.index][0:T, 0 : i_tiles.size],
            op=nl.copy,
        )

        # Accumulate PSUMs to SBUF
        for factor_idx in range(1, used_columns):
            nisa.tensor_tensor(
                dst=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
                data1=result_psums[i_tiles.index][nl.ds(dims.column_tiling_dim * factor_idx, T), 0 : i_tiles.size],
                data2=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
                op=nl.add,
            )

    if params.quant_params.is_quant():
        dequant_tile_view = TensorView(dequant_tile)
        if params.quant_params.is_quant_row():
            dequant_tile_view = dequant_tile_view.slice(dim=1, start=0, end=I)

        interleave_copy(
            dst=output_tile[0:T, nl.ds(i_offset, I)],
            src=output_tile[0:T, nl.ds(i_offset, I)],
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias separately from matmul pipeline ----------
    if is_bias:
        bias_tile_view = TensorView(bias_tile).slice(dim=1, start=0, end=I)
        output_tile_view = TensorView(output_tile).slice(dim=1, start=i_offset, end=i_offset + I)
        nisa.tensor_tensor(
            dst=output_tile_view.get_view(),
            data1=output_tile_view.get_view(),
            data2=bias_tile_view.get_view(),
            op=nl.add,
        )


def gate_up_projection_lhs_rhs_swap(
    hidden: nl.ndarray,
    unsharded_weight: TensorView,
    shard_dim_hidden: tuple[int, int],
    shard_dim_intr: tuple[int, int],
    bias: TensorView,
    dequant_scale: TensorView,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    op_name: str,
    sbm: SbufManager,
    T_offset: int = 0,
    fused_weight_tiles: list = None,
    i_offset_in_fused: int = 0,
    hoisted_weight: nl.ndarray = None,
    hoisted_dge_mode: int = None,
    skip_hoisted_dma: bool = False,
    pre_loaded_fused_hoist: nl.ndarray = None,
    fused_hoist_i_offset: int = 0,
):
    """
    Performs a single Gate or Up projection shard on the H using regular matmult with operands swapped

    Computes: Weight[H, I] @ Hidden[H, T] + Optional(Bias[1, I]) → [T, I]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Weight source (mutually exclusive, listed in precedence order):
        - pre_loaded_fused_hoist: caller pre-loaded a single [H0, H1_shard, 2*I]
          tile with gate+up concatenated on the free dim. Indexed via
          fused_hoist_i_offset (0 for gate, I for up).
        - fused_weight_tiles: caller pre-loaded per-HTile fused tiles. Indexed via
          i_offset_in_fused.
        - hoisted_weight: a single [H0, H1_shard, I] tile. DMA is issued here if
          skip_hoisted_dma is False, else assumed pre-loaded by caller.
        - default: per-HTile weight ring loaded inline.

    Returns:
        Output tensor with shape [128, I/128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0, _, _ = hidden.shape
    # Use dims.T (tile size) instead of hidden.shape[1], which may be T_total when hidden is in SBUF
    T = dims.T
    shared_H = shard_dim_hidden[1] - shard_dim_hidden[0]
    shared_I = shard_dim_intr[1] - shard_dim_intr[0]
    I0 = dims.I0
    i_offset = shard_dim_intr[0]
    i1_offset = shard_dim_intr[0] // I0
    num_allocated_w_tile = tiles.num_allocated_w_tile
    use_caller_fused_hoist = pre_loaded_fused_hoist is not None
    use_fused_load = (fused_weight_tiles is not None) and (not use_caller_fused_hoist)

    # Sanity checks for sharding
    kernel_assert(
        shared_I <= dims.max_I_shard_size,
        f"{op_name}_projection only supports shared_I <= {dims.max_I_shard_size}",
    )
    kernel_assert(
        shared_H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {shared_H}",
    )

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along shared_H)
    is_bias = bias != None and dims.shard_id == 0

    # Number of full/res 128(_pmax)-elements tiles along shared_I
    num_128_I_tiles = shared_I // I0
    res_128_I_tiles = shared_I % I0
    num_total_128_I_tiles = num_128_I_tiles + (res_128_I_tiles != 0)

    if is_bias:
        # Load bias tensor with proper reshaping based on intermediate dimension alignment
        bias_i_dim = 0 if bias.get_dim() == 1 else 1

        if num_128_I_tiles > 0:
            bias_view = bias.slice(
                dim=bias_i_dim, start=shard_dim_intr[0], end=shard_dim_intr[1] - res_128_I_tiles
            ).reshape_dim(dim=bias_i_dim, shape=(num_128_I_tiles, I0))
            if not bias_view.has_dynamic_access():
                while bias_view.get_dim() < 4:
                    bias_view = bias_view.expand_dim(1)

                nisa.dma_transpose(
                    src=bias_view.slice(dim=0, start=0, end=num_128_I_tiles).get_view(),
                    dst=bias_tile.ap(
                        pattern=[
                            [num_total_128_I_tiles, I0],
                            [1, 1],
                            [1, 1],
                            [1, num_128_I_tiles],
                        ]
                    ),
                    dge_mode=_DGE_MODE_NONE,
                )

            else:
                # WA for dynamic access not supported by dma_transpose, issue: NKI-415
                # Using dma_copy + PE transpose on tensor_engine
                bias_tile_sbuf = sbm.alloc_stack(
                    shape=(num_128_I_tiles, I0),
                    dtype=bias.dtype,
                    buffer=nl.sbuf,
                    name=f"{op_name}_{sbm.get_name_prefix()}_ishard_{i_offset}_bias_tile_sbuf",
                )
                bias_tile_psum = nl.ndarray((I0, num_128_I_tiles), dtype=bias.dtype, buffer=nl.psum)
                nisa.dma_copy(
                    src=bias_view.base_tensor.ap(
                        pattern=[[I0, num_128_I_tiles], [1, I0]],
                        offset=bias_view.offset,
                        indirect_dim=bias_view.indirect_dim,
                        scalar_offset=bias_view.scalar_offset,
                    ),
                    dst=bias_tile_sbuf.ap(
                        pattern=[[I0, num_128_I_tiles], [1, I0]],
                        offset=0,
                    ),
                    dge_mode=adaptive_dge_mode(bias_view),
                )
                nisa.nc_transpose(dst=bias_tile_psum, data=bias_tile_sbuf)
                nisa.tensor_copy(dst=bias_tile[0:I0, 0:num_128_I_tiles], src=bias_tile_psum)

        if res_128_I_tiles > 0:
            bias_view = bias.slice(
                dim=bias_i_dim, start=shard_dim_intr[1] - res_128_I_tiles, end=shard_dim_intr[1]
            ).expand_dim(1)
            nisa.dma_copy(
                src=bias_view.get_view(),
                dst=bias_tile.ap(
                    pattern=[[num_total_128_I_tiles, res_128_I_tiles], [1, 1]],
                    offset=num_128_I_tiles,
                ),
                dge_mode=adaptive_dge_mode(bias_view),
            )

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        # Load dequant_scale tensor with proper reshaping based on intermediate dimension alignment
        dequant_scale = dequant_scale.select(dim=0, index=0)
        if num_128_I_tiles > 0:
            dequant_scale_view = dequant_scale.slice(
                dim=0, start=shard_dim_intr[0], end=shard_dim_intr[1] - res_128_I_tiles
            ).reshape_dim(dim=0, shape=(num_128_I_tiles, I0))
            if not dequant_scale_view.has_dynamic_access():
                while dequant_scale_view.get_dim() < 4:
                    dequant_scale_view = dequant_scale_view.expand_dim(1)
                nisa.dma_transpose(
                    src=dequant_scale_view.slice(dim=0, start=0, end=num_128_I_tiles).get_view(),
                    dst=dequant_tile.ap(
                        pattern=[
                            [dequant_tile.shape[1], I0],
                            [1, 1],
                            [1, 1],
                            [1, num_128_I_tiles],
                        ]
                    ),
                    dge_mode=_DGE_MODE_NONE,
                )
            else:
                # WA for dynamic access not supported by dma_transpose, issue: NKI-415
                dequant_tile_sbuf = sbm.alloc_stack(
                    shape=(num_128_I_tiles, I0),
                    dtype=dequant_scale.dtype,
                    buffer=nl.sbuf,
                    name=f"{op_name}_{sbm.get_name_prefix()}_ishard_{i_offset}_dequant_tile_sbuf",
                )
                nisa.dma_copy(src=dequant_scale_view.get_view(), dst=dequant_tile_sbuf)
                stream_tile_size = 32
                for stream_tile_i in range(I0 // stream_tile_size):
                    nisa.nc_transpose(
                        dst=dequant_tile.ap(
                            pattern=[
                                [dequant_tile.shape[1], stream_tile_size],
                                [1, num_128_I_tiles],
                            ],
                            offset=stream_tile_i * 32 * num_total_128_I_tiles,
                        ),
                        data=dequant_tile_sbuf.ap(
                            pattern=[
                                [I0, num_128_I_tiles],
                                [1, stream_tile_size],
                            ],
                            offset=stream_tile_i * stream_tile_size,
                        ),
                    )

        if res_128_I_tiles > 0:
            dequant_scale_view = dequant_scale.slice(
                dim=0, start=shard_dim_intr[1] - res_128_I_tiles, end=shard_dim_intr[1]
            ).expand_dim(1)
            nisa.dma_copy(
                src=dequant_scale_view.get_view(),
                dst=dequant_tile.ap(
                    pattern=[[dequant_tile.shape[1], res_128_I_tiles], [1, 1]],
                    offset=num_128_I_tiles,
                ),
                dge_mode=adaptive_dge_mode(dequant_scale_view),
            )
    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles for gate and up projection
    # this treats them as a ring buffer so up weights load after gate weights for efficient reuse.
    # When using fused weight tiles, both gate and up read the SAME tiles (already loaded by the
    # wrapper), so no offset is needed.
    if use_fused_load or use_caller_fused_hoist:
        weight_base_idx = 0
    else:
        weight_base_idx = tiles.num_HTiles % num_allocated_w_tile if op_name == "up" else 0

    # Allocate PSUM buffers to store output
    result_psums = []
    for i_tiles in TiledRange(shared_I, I0):
        result_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_{sbm.get_name_prefix()}_psum_ishard_{i_offset}_{i_tiles.index}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, i_tiles.index * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # Hoisted weight: emit the single-DMA load here unless the caller already did.
    # Wrapper routes gate via HWDGE and up via SWDGE so both loads run in parallel.
    use_hoisted_weight = (hoisted_weight is not None) and (not use_fused_load) and (not use_caller_fused_hoist)
    if use_hoisted_weight and not skip_hoisted_dma:
        emit_hoisted_gate_up_dma(
            unsharded_weight=unsharded_weight,
            hoisted_weight=hoisted_weight,
            dims=dims,
            shard_dim_hidden=shard_dim_hidden,
            shard_dim_intr=shard_dim_intr,
            dge_mode=hoisted_dge_mode,
        )

    # ---------- Matrix multiplication ----------
    # Gate Up Projection
    for hidden_tiles in TiledRange(shared_H, tiles.HTile):
        # Compute start offset
        h_start_offset = hidden_tiles.index * (tiles.HTile // H0)

        # Load weight tile [HTile, shared_I] → SBUF layout [H0, HTile/H0, shared_I]
        # When using fused weight tiles, this load is skipped (already done by the wrapper).
        h1_size = hidden_tiles.size // H0
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        if use_caller_fused_hoist:
            # Caller pre-loaded one [H0, H1_shard, 2*I] tile; walk H1 via
            # h_start_offset and select gate/up via fused_hoist_i_offset.
            weight_for_matmul = pre_loaded_fused_hoist
            weight_i_base = fused_hoist_i_offset
            h_idx_offset = h_start_offset
        elif use_fused_load:
            weight_for_matmul = fused_weight_tiles[weight_idx]
            weight_i_base = i_offset_in_fused
            h_idx_offset = 0
        elif use_hoisted_weight:
            # Index into the single pre-loaded tile, offset by h_start_offset
            # to preserve the per-HTile slice math.
            weight_for_matmul = hoisted_weight
            weight_i_base = 0
            h_idx_offset = h_start_offset
        else:
            weight_view = (
                unsharded_weight.slice(dim=0, start=shard_dim_hidden[0], end=shard_dim_hidden[1])  # LNC shard
                .reshape_dim(dim=0, shape=(H0, dims.H1_shard))  # shared_H -> H0, h1_tiles
                .slice(dim=1, start=h_start_offset, end=h_start_offset + h1_size)  # Local shared_H tiling
                .slice(dim=2, start=shard_dim_intr[0], end=shard_dim_intr[1])  # Slice on shared_I dim
            )
            nisa.dma_copy(
                dst=weight_tiles[weight_idx][0:H0, 0:h1_size, 0:shared_I],
                src=weight_view.get_view(),
                dge_mode=nisa.dge_mode.hwdge,
            )
            weight_for_matmul = weight_tiles[weight_idx]
            weight_i_base = 0
            h_idx_offset = 0

        # Matmult
        for h1_tiles in TiledRange(hidden_tiles.size, H0):
            for i_tiles in TiledRange(shared_I, I0):
                nisa.nc_matmul(
                    result_psums[i_tiles.index][0 : i_tiles.size, 0:T],
                    weight_for_matmul[
                        0:H0,
                        h_idx_offset + h1_tiles.index,
                        nl.ds(weight_i_base + i_tiles.index * I0, i_tiles.size),
                    ],
                    hidden[0:H0, nl.ds(T_offset, T), h_start_offset + h1_tiles.index],
                )

    # ---------- Accumulate partial PSUMs to output ----------
    for i_tiles in TiledRange(shared_I, I0):
        # Set tile view for dequant tile
        dequant_tile_view = None
        if params.quant_params.is_quant():
            dequant_tile_view = TensorView(dequant_tile).slice(dim=0, start=0, end=i_tiles.size)
            if params.quant_params.is_quant_row():
                dequant_tile_view = dequant_tile_view.slice(
                    dim=1, start=i_tiles.index, end=i_tiles.index + 1
                ).broadcast(dim=1, size=T)

        # Create output tile view for this I tile
        output_tile_view = (
            TensorView(output_tile)
            .slice(dim=0, start=0, end=i_tiles.size)
            .slice(dim=1, start=i1_offset + i_tiles.index, end=i1_offset + i_tiles.index + 1)
            .squeeze_dim(dim=1)
        )

        # PSUM to SBUF copy while applying dequant tensor optionally
        interleave_copy(
            index=i_tiles.index,
            dst=output_tile_view.get_view(),
            src=result_psums[i_tiles.index][0 : i_tiles.size, 0:T],
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias separately from matmul pipeline ----------
    if is_bias:
        for i_tiles in TiledRange(shared_I, I0):
            bias_tile_view = (
                TensorView(bias_tile)
                .slice(dim=0, start=0, end=i_tiles.size)
                .slice(dim=1, start=i_tiles.index, end=i_tiles.index + 1)
                .broadcast(dim=1, size=T)
            )
            output_tile_view = (
                TensorView(output_tile)
                .slice(dim=0, start=0, end=i_tiles.size)
                .slice(dim=1, start=i1_offset + i_tiles.index, end=i1_offset + i_tiles.index + 1)
                .squeeze_dim(dim=1)
            )
            nisa.tensor_tensor(
                dst=output_tile_view.get_view(),
                data1=output_tile_view.get_view(),
                data2=bias_tile_view.get_view(),
                op=nl.add,
            )


def _load_fused_gate_up_weights(
    fused_unsharded_weight: TensorView,
    fused_weight_tiles: list,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    shard_dim_hidden: tuple[int, int],
    shard_dim_intr_total: tuple[int, int],
):
    """
    Loads gate+up fused weights once per HTile.

    The fused HBM view has shape [H, 2*I_total] where the inner 2*I dim is
    contiguous (original layout [E, H, 2, I] is row-major). One dma_copy per HTile
    instead of two, with a 2x larger inner contiguous chunk per row.
    """
    H0 = dims.H0
    fused_I_total = shard_dim_intr_total[1] - shard_dim_intr_total[0]
    num_allocated_w_tile = tiles.num_allocated_w_tile

    for hidden_tiles in TiledRange(shard_dim_hidden[1] - shard_dim_hidden[0], tiles.HTile):
        h_start_offset = hidden_tiles.index * (tiles.HTile // H0)
        h1_size = hidden_tiles.size // H0
        weight_idx = hidden_tiles.index % num_allocated_w_tile

        weight_view = (
            fused_unsharded_weight.slice(
                dim=0, start=shard_dim_hidden[0], end=shard_dim_hidden[1]
            )  # LNC shard on H
            .reshape_dim(dim=0, shape=(H0, dims.H1_shard))  # shared_H -> H0, h1
            .slice(dim=1, start=h_start_offset, end=h_start_offset + h1_size)  # local H1 tile
            .slice(dim=2, start=shard_dim_intr_total[0], end=shard_dim_intr_total[1])  # 2*I slice
        )
        nisa.dma_copy(
            dst=fused_weight_tiles[weight_idx][0:H0, 0:h1_size, 0:fused_I_total],
            src=weight_view.get_view(),
            dge_mode=adaptive_dge_mode(weight_view),
        )


def process_gate_up_projection(
    hidden: nl.ndarray,
    output: nl.ndarray,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    T_offset: int = 0,
    pre_loaded_hoisted_gate: nl.ndarray = None,
    pre_loaded_hoisted_up: nl.ndarray = None,
    pre_loaded_hoisted_gate_up_fused: nl.ndarray = None,
):
    """
    Performs the Gate/Up projection for MLP (T = BxS).
    Expected hidden tensor shape is [128(H0), T, H//128]

    Overview:
    ---------
    gate_proj_out [T, I] = hidden [H, T] @ gate_weight [H, I] + optional(gate_bias [1, I])
    act_gate_proj [T, I] = Activation_Fn(gate_proj_out [T, I])
    up_proj_out [T, I]   = hidden [H, T] @ up_weight [H, I] + optional(up_bias)
    hidden[T, I] = act_gate_proj [T, I] * up_proj_out [T, I]  # elementwise multiplication

    Hardware constraints (max partition size of 128) require tiling along the H dimension:
    # hidden [128, BxS, H//128] @ gate/up_weight [128, H//128, I]

    Behavior based on `use_tkg_gate_up_proj_column_tiling`:
    ------------------------------------------
    - True: column tiling(`gate_up_projection`)
        hidden[128, BxS] @ gate/up_weight[128, I] → [T, I]
    - False: regular matmult with operands swapped(`gate_up_projection_lhs_rhs_swap`)
        gate/up_weight[128, I] @ hidden[128, BxS] → [I, T]
        Further tiling along I: [128, I//128, T]

    DMA mode:
    ---------
    Based on experiments, Static DMA provides better performance.
    The MLP TKG implementation therefore uses Static DMA for tensor loads.
    If HBM out-of-memory (OOM) issues arise, we can fall back to DGE mode.

    Caller-preloaded weight tiles (lhs_rhs_swap path only):
        - pre_loaded_hoisted_gate/_up: caller emitted the per-projection hoist
          DMAs; the inner kernel skips them and matmuls against the tiles.
        - pre_loaded_hoisted_gate_up_fused: caller pre-loaded a single
          [H0, H1_shard, 2*I] tile with gate+up concatenated along the free dim
          (one DMA per expert vs. two). Takes precedence over the per-projection
          tiles when set.

    Note:
    -----
    Intermediate gate/up projection tensors are always fp32 to improve numerical accuracy.
    Hidden tensor in SBUF has layout [H, T], tiled as [128(H0), T, H//128] to fully utilize the partition dimension.
    Caller will have the flexibility to manage sbm:sbufManager's scope and interleave degree.
    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor
    gate_b, up_b = (
        params.bias_params.gate_proj_bias_tensor,
        params.bias_params.up_proj_bias_tensor,
    )
    gate_w_scale, up_w_scale = (
        params.quant_params.gate_w_scale,
        params.quant_params.up_w_scale,
    )
    input_scale = params.quant_params.gate_up_in_scale

    # ---------------- Allocate Gate/Up/Bias/DequantScale Tiles ----------------
    # Note: intermediate tiles are fp32 for better numerical accuracy
    bias_tile = None
    bias_size = 0

    # Fused gate+up sendrecv optimization: use a single sendrecv instead of two when LNC > 1.
    # Allocates a combined gate+up buffer and a 2X receive buffer to perform one sendrecv for
    # both projections, reducing inter-core communication overhead. Currently enabled by default
    # for column tiling mode.
    use_fused_gate_up_sendrecv = (
        dims.num_shards > 1 and not params.skip_gate_proj and params.use_tkg_gate_up_proj_column_tiling
    )
    if params.use_tkg_gate_up_proj_column_tiling:
        if not params.skip_gate_proj:
            if use_fused_gate_up_sendrecv:
                # Allocate a single combined gate+up buffer with 2X the I dimension
                gate_up_sb_fp32 = sbm.alloc_stack(
                    (dims.T, 2 * dims.I),
                    dtype=nl.float32,
                    name="gate_up_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                gate_up_tv = TensorView(gate_up_sb_fp32)
                gate_sb_view = gate_up_tv.slice(dim=1, start=0, end=dims.I)
                up_sb_view = gate_up_tv.slice(dim=1, start=dims.I, end=2 * dims.I)
            else:
                gate_sb_fp32 = sbm.alloc_stack(
                    (dims.T, dims.I),
                    dtype=nl.float32,
                    name="gate_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                up_sb_fp32 = sbm.alloc_stack(
                    (dims.T, dims.I),
                    dtype=nl.float32,
                    name="up_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                gate_sb_view = TensorView(gate_sb_fp32)
                up_sb_view = TensorView(up_sb_fp32)
        else:
            up_sb_fp32 = sbm.alloc_stack(
                (dims.T, dims.I),
                dtype=nl.float32,
                name="up_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
            up_sb_view = TensorView(up_sb_fp32)
        if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
            bias_tile = sbm.alloc_stack(
                (dims.T, dims.max_I_shard_size),
                dtype=gate_b.dtype,
                name="gate_up_broadcasted_bias",
                buffer=nl.sbuf,
            )
    else:
        if not params.skip_gate_proj:
            if use_fused_gate_up_sendrecv:
                # Allocate a single combined gate+up buffer with 2X the I tile dimension
                gate_up_sb_fp32 = sbm.alloc_stack(
                    (dims.I0, 2 * dims.num_total_128_tiles_per_I, dims.T),
                    dtype=nl.float32,
                    name="gate_up_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                gate_up_tv = TensorView(gate_up_sb_fp32)
                gate_sb_view = gate_up_tv.slice(dim=1, start=0, end=dims.num_total_128_tiles_per_I)
                up_sb_view = gate_up_tv.slice(
                    dim=1, start=dims.num_total_128_tiles_per_I, end=2 * dims.num_total_128_tiles_per_I
                )
            else:
                gate_sb_fp32 = sbm.alloc_stack(
                    (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
                    dtype=nl.float32,
                    name="gate_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                up_sb_fp32 = sbm.alloc_stack(
                    (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
                    dtype=nl.float32,
                    name="up_sbuf_fp32",
                    buffer=nl.sbuf,
                    align=4,
                )
                gate_sb_view = TensorView(gate_sb_fp32)
                up_sb_view = TensorView(up_sb_fp32)
        else:
            up_sb_fp32 = sbm.alloc_stack(
                (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
                dtype=nl.float32,
                name="up_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
            up_sb_view = TensorView(up_sb_fp32)
            gate_sb_view = None
        # Allocate the bias/dequant tile inside the loop due to the DMA transpose 32-byte address alignment requirement.
        if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
            bias_size = dims.num_total_128_tiles_per_I * sizeinbytes(gate_b.dtype)

    # ---------------- Static quantization ----------------
    gate_dequant_tile = up_dequant_tile = None
    if params.quant_params.is_quant_static():
        par_dim = dims.T if params.use_tkg_gate_up_proj_column_tiling else dims.I0

        # Allocate Dequant tile
        gate_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=gate_w_scale.dtype,
            name=f"gate_w_scale_sb",
            buffer=nl.sbuf,
            align=4,
        )
        up_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=up_w_scale.dtype,
            name=f"up_w_scale_sb",
            buffer=nl.sbuf,
            align=4,
        )

        # Load gate up dequantization scale
        gate_w_scale_view = gate_w_scale.slice(dim=0, start=0, end=par_dim)
        up_w_scale_view = up_w_scale.slice(dim=0, start=0, end=par_dim)
        nisa.dma_copy(
            dst=gate_dequant_tile[0:par_dim, :],
            src=gate_w_scale_view.get_view(),
            dge_mode=adaptive_dge_mode(gate_w_scale_view),
        )
        nisa.dma_copy(
            dst=up_dequant_tile[0:par_dim, :],
            src=up_w_scale_view.get_view(),
            dge_mode=adaptive_dge_mode(up_w_scale_view),
        )

    # ---------------- Row quantization ----------------
    # Scale data is loaded in projection function
    elif params.quant_params.is_quant_row():
        if params.use_tkg_gate_up_proj_column_tiling:
            gate_dequant_tile = sbm.alloc_stack(
                (dims.T, min(dims.max_I_shard_size, dims.I)),
                dtype=gate_w_scale.dtype,
                name=f"gate_w_scale_sb",
                buffer=nl.sbuf,
                align=4,
            )
            up_dequant_tile = sbm.alloc_stack(
                gate_dequant_tile.shape,
                dtype=up_w_scale.dtype,
                name=f"up_w_scale_sb",
                buffer=nl.sbuf,
                align=4,
            )
        else:
            gate_dequant_tile = sbm.alloc_stack(
                (dims.I0, min(dims.max_I_shard_size // dims.I0, dims.num_total_128_tiles_per_I)),
                dtype=gate_w_scale.dtype,
                name=f"gate_w_scale_sb",
                buffer=nl.sbuf,
                align=32,
            )
            up_dequant_tile = sbm.alloc_stack(
                gate_dequant_tile.shape,
                dtype=up_w_scale.dtype,
                name=f"up_w_scale_sb",
                buffer=nl.sbuf,
                align=32,
            )

    # ---------------- Allocate Receive Buffer for LNC > 1 ----------------
    gate_up_recv = None
    if dims.num_shards > 1:
        if use_fused_gate_up_sendrecv:
            # 2X receive buffer to receive both gate and up projections in a single sendrecv
            gate_up_recv = sbm.alloc_stack(
                gate_up_sb_fp32.shape,
                dtype=nl.float32,
                buffer=nl.sbuf,
                name="gate_up_recv_buffer_fp32",
            )
        else:
            gate_up_recv = sbm.alloc_stack(
                up_sb_fp32.shape,
                dtype=nl.float32,
                buffer=nl.sbuf,
                name="gate_up_recv_buffer_fp32",
            )

    # ---------------- Decide whether to fuse gate+up weight loads ----------------
    # Requires lhs_rhs_swap (no column tiling), both projections active, and a
    # fused [H, 2*I_total] HBM view stashed by selective_expert_impl. If the
    # caller pre-loaded a fused hoist tile, take that path instead.
    fused_unsharded_weight = params.gate_up_fused_weights_tensor
    use_caller_preloaded_fused_hoist = (
        pre_loaded_hoisted_gate_up_fused is not None
        and not params.use_tkg_gate_up_proj_column_tiling
        and not params.skip_gate_proj
    )
    use_fused_gate_up_load = (
        not params.use_tkg_gate_up_proj_column_tiling
        and not params.skip_gate_proj
        and fused_unsharded_weight is not None
        and not use_caller_preloaded_fused_hoist
    )

    # ---------------- Allocate Weight Tiles ----------------
    # By calculating the remaining SBUF space, we allocate as many weight tiles as possible
    if sbm.is_auto_alloc():
        remaining_space = 0
        current_address = 0
    else:
        remaining_space = sbm.get_free_space() - bias_size
        kernel_assert(remaining_space > 0, "Not enough memory for gate/up weights")
        current_address = sbm.get_stack_curr_addr()
    tiles = MLPTKGConstants.calculate_gate_up_tiles(current_address, remaining_space, params, dims, sbm.is_auto_alloc())

    # Fused tiles are 2x the size of a regular tile, so we allocate half as many
    # for the same SBUF budget. Need at least num_HTiles fused tiles so each
    # HTile's load isn't overwritten before both gate and up consume it; if not,
    # fall back to the non-fused path.
    fused_weight_tiles = None
    weight_tiles = []
    if use_fused_gate_up_load:
        num_fused_w_tile = tiles.num_allocated_w_tile // 2
        if num_fused_w_tile >= tiles.num_HTiles:
            tiles.num_allocated_w_tile = num_fused_w_tile
            fused_weight_tiles = []
            for w_tile_idx in range(num_fused_w_tile):
                fused_tile = sbm.alloc_stack(
                    (dims.H0, tiles.num_128_tiles_per_HTile, 2 * dims.I),
                    name=f"gate_up_fused_w_tile_{w_tile_idx}",
                    dtype=nl.float8_e4m3 if str(up_w.dtype) == "float8e4" else up_w.dtype,
                )
                fused_weight_tiles.append(fused_tile)
        else:
            use_fused_gate_up_load = False

    # Hoisted gate/up weight tiles (lhs_rhs_swap path, no fused load): one
    # [H0, H1_shard, I] tile per projection, single-DMA load. Caller-preloaded
    # tiles take precedence (cross-expert prefetch ring); else allocate here.
    use_caller_preloaded_hoist = (
        pre_loaded_hoisted_gate is not None
        and pre_loaded_hoisted_up is not None
        and not params.use_tkg_gate_up_proj_column_tiling
        and not use_fused_gate_up_load
        and not use_caller_preloaded_fused_hoist
        and not params.skip_gate_proj
    )

    gate_hoisted_weight = None
    up_hoisted_weight = None
    use_hoisted_gate_up_load = (
        (not params.use_tkg_gate_up_proj_column_tiling)
        and (not use_fused_gate_up_load)
        and (not use_caller_preloaded_fused_hoist)
        and (not _MOE_LEGACY_GATE_UP_WEIGHT_LOAD)
    )

    if (not use_fused_gate_up_load) and (not use_caller_preloaded_fused_hoist):
        if use_caller_preloaded_hoist:
            gate_hoisted_weight = pre_loaded_hoisted_gate
            up_hoisted_weight = pre_loaded_hoisted_up
        elif use_hoisted_gate_up_load:
            weight_dtype = nl.float8_e4m3 if str(up_w.dtype) == "float8e4" else up_w.dtype
            if not params.skip_gate_proj:
                gate_hoisted_weight = sbm.alloc_stack(
                    (dims.H0, dims.H1_shard, dims.I),
                    name="gate_hoisted_w_tile",
                    dtype=weight_dtype,
                )
            up_hoisted_weight = sbm.alloc_stack(
                (dims.H0, dims.H1_shard, dims.I),
                name="up_hoisted_w_tile",
                dtype=weight_dtype,
            )
        else:
            for w_tile_idx in range(tiles.num_allocated_w_tile):
                weight_tile = sbm.alloc_stack(
                    (dims.H0, tiles.num_128_tiles_per_HTile, dims.I),
                    name=f"gate_up_w_tile_{w_tile_idx}",
                    dtype=nl.float8_e4m3 if str(up_w.dtype) == "float8e4" else up_w.dtype,
                )
                weight_tiles.append(weight_tile)

    # ---------------- Gate/Up Projection ----------------
    if params.use_tkg_gate_up_proj_column_tiling:
        for i_tiles in TiledRange(dims.I, dims.max_I_shard_size):
            h_offset = dims.H1_offset * dims.H0
            I_start = i_tiles.start_offset
            I_end = min(I_start + dims.max_I_shard_size, dims.I)

            if not params.skip_gate_proj:
                # Gate projection
                gate_up_projection(
                    hidden=hidden,
                    unsharded_weight=gate_w,
                    shard_dim_hidden=(h_offset, h_offset + dims.H_per_shard),
                    shard_dim_intr=(I_start, I_end),
                    bias=gate_b,
                    dequant_scale=gate_w_scale,
                    output_tile=gate_sb_view.get_view(),
                    weight_tiles=weight_tiles,
                    bias_tile=bias_tile,
                    dequant_tile=gate_dequant_tile,
                    dims=dims,
                    tiles=tiles,
                    params=params,
                    op_name="gate",
                    sbm=sbm,
                )

            # Up projection
            gate_up_projection(
                hidden=hidden,
                unsharded_weight=up_w,
                shard_dim_hidden=(h_offset, h_offset + dims.H_per_shard),
                shard_dim_intr=(I_start, I_end),
                bias=up_b,
                dequant_scale=up_w_scale,
                output_tile=up_sb_view.get_view(),
                weight_tiles=weight_tiles,
                bias_tile=bias_tile,
                dequant_tile=up_dequant_tile,
                dims=dims,
                tiles=tiles,
                params=params,
                op_name="up",
                sbm=sbm,
            )
    else:
        for i_tiles in TiledRange(dims.I, dims.max_I_shard_size):
            h_offset = dims.H1_offset * dims.H0
            I_start = i_tiles.start_offset
            I_end = min(I_start + dims.max_I_shard_size, dims.I)
            num_total_128_I_tiles = div_ceil(i_tiles.size, dims.I0)

            if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
                bias_tile = sbm.alloc_stack(
                    (dims.I0, num_total_128_I_tiles),
                    dtype=nl.float32 if gate_b.has_dynamic_access() else gate_b.dtype,
                    name=f"gate_up_bias_{i_tiles.index}",
                    buffer=nl.sbuf,
                    align=32,
                )

            # ---------- Pre-load fused gate+up weights once for this I shard ----------
            # The fused HBM view has shape [H, 2*I_total]; slice the relevant 2*I window.
            if use_fused_gate_up_load:
                fused_I_start = 2 * I_start
                fused_I_end = 2 * I_end
                _load_fused_gate_up_weights(
                    fused_unsharded_weight=fused_unsharded_weight,
                    fused_weight_tiles=fused_weight_tiles,
                    dims=dims,
                    tiles=tiles,
                    shard_dim_hidden=(h_offset, h_offset + dims.H_per_shard),
                    shard_dim_intr_total=(fused_I_start, fused_I_end),
                )

            if not params.skip_gate_proj:
                # Gate projection
                gate_up_projection_lhs_rhs_swap(
                    hidden=hidden,
                    unsharded_weight=gate_w,
                    shard_dim_hidden=(h_offset, h_offset + dims.H_per_shard),
                    shard_dim_intr=(I_start, I_end),
                    bias=gate_b,
                    dequant_scale=gate_w_scale,
                    output_tile=gate_sb_view.get_view(),
                    weight_tiles=weight_tiles,
                    bias_tile=bias_tile,
                    dequant_tile=gate_dequant_tile,
                    dims=dims,
                    tiles=tiles,
                    params=params,
                    op_name="gate",
                    sbm=sbm,
                    T_offset=T_offset,
                    fused_weight_tiles=fused_weight_tiles,
                    i_offset_in_fused=0,
                    hoisted_weight=gate_hoisted_weight,
                    # gate -> HWDGE, up -> SWDGE: equal-sized loads run in parallel.
                    hoisted_dge_mode=nisa.dge_mode.hwdge,
                    skip_hoisted_dma=use_caller_preloaded_hoist,
                    pre_loaded_fused_hoist=pre_loaded_hoisted_gate_up_fused
                    if use_caller_preloaded_fused_hoist
                    else None,
                    fused_hoist_i_offset=0,
                )

            # Up projection
            gate_up_projection_lhs_rhs_swap(
                hidden=hidden,
                unsharded_weight=up_w,
                shard_dim_hidden=(h_offset, h_offset + dims.H_per_shard),
                shard_dim_intr=(I_start, I_end),
                bias=up_b,
                dequant_scale=up_w_scale,
                output_tile=up_sb_view.get_view(),
                weight_tiles=weight_tiles,
                bias_tile=bias_tile,
                dequant_tile=up_dequant_tile,
                dims=dims,
                tiles=tiles,
                params=params,
                op_name="up",
                sbm=sbm,
                T_offset=T_offset,
                fused_weight_tiles=fused_weight_tiles,
                i_offset_in_fused=(I_end - I_start) if use_fused_gate_up_load else 0,
                hoisted_weight=up_hoisted_weight,
                hoisted_dge_mode=nisa.dge_mode.swdge,
                skip_hoisted_dma=use_caller_preloaded_hoist,
                pre_loaded_fused_hoist=pre_loaded_hoisted_gate_up_fused
                if use_caller_preloaded_fused_hoist
                else None,
                fused_hoist_i_offset=(I_end - I_start) if use_caller_preloaded_fused_hoist else 0,
            )

    if params.skip_gate_proj:
        # ---------------- Up Projection Multi-Shard Communication ----------------
        # Receive up projection output from the other neuron core when LNC > 1
        if dims.num_shards > 1:
            nisa.sendrecv(
                src=up_sb_fp32,
                dst=gate_up_recv,
                send_to_rank=(1 - dims.shard_id),
                recv_from_rank=(1 - dims.shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=up_sb_fp32, data1=up_sb_fp32, data2=gate_up_recv, op=nl.add)

        #  ---------------- Optionally perform clamping on up projection results  ----------------
        if params.up_clamp_upper_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.minimum, operand0=params.up_clamp_upper_limit)
        if params.up_clamp_lower_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.maximum, operand0=params.up_clamp_lower_limit)

        # ---------------- Up Activation ----------------
        nisa.activation(
            dst=up_sb_fp32[:, :],
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=up_sb_fp32,
            scale=1.0,
        )

        if not params.use_tkg_gate_up_proj_column_tiling:
            nisa.tensor_copy(dst=output, src=up_sb_fp32, engine=nki.isa.vector_engine)

    else:
        # ---------------- Gate/Up Projection Multi-Shard Communication ----------------
        if dims.num_shards > 1:
            if use_fused_gate_up_sendrecv:
                # Single sendrecv for both gate and up projections using the combined buffer
                nisa.sendrecv(
                    src=gate_up_sb_fp32,
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(dst=gate_up_sb_fp32, data1=gate_up_sb_fp32, data2=gate_up_recv, op=nl.add)
            else:
                # Separate sendrecv for gate projection
                nisa.sendrecv(
                    src=gate_sb_view.get_view(),
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(
                    dst=gate_sb_view.get_view(), data1=gate_sb_view.get_view(), data2=gate_up_recv, op=nl.add
                )

                # Separate sendrecv for up projection
                nisa.sendrecv(
                    src=up_sb_view.get_view(),
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(
                    dst=up_sb_view.get_view(), data1=up_sb_view.get_view(), data2=gate_up_recv, op=nl.add
                )

        #  ---------------- Optionally perform clamping on gate projection results  ----------------
        if params.gate_clamp_upper_limit is not None:
            nisa.tensor_scalar(
                data=gate_sb_view.get_view(),
                dst=gate_sb_view.get_view(),
                op0=nl.minimum,
                operand0=params.gate_clamp_upper_limit,
            )
        if params.gate_clamp_lower_limit is not None:
            nisa.tensor_scalar(
                data=gate_sb_view.get_view(),
                dst=gate_sb_view.get_view(),
                op0=nl.maximum,
                operand0=params.gate_clamp_lower_limit,
            )

        # ---------------- Gate Activation ----------------
        nisa.activation(
            dst=gate_sb_view.get_view(),
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_sb_view.get_view(),
            scale=1.0,
        )

        #  ---------------- Optionally perform clamping on up projection results  ----------------
        if params.up_clamp_upper_limit is not None:
            nisa.tensor_scalar(
                data=up_sb_view.get_view(),
                dst=up_sb_view.get_view(),
                op0=nl.minimum,
                operand0=params.up_clamp_upper_limit,
            )
        if params.up_clamp_lower_limit is not None:
            nisa.tensor_scalar(
                data=up_sb_view.get_view(),
                dst=up_sb_view.get_view(),
                op0=nl.maximum,
                operand0=params.up_clamp_lower_limit,
            )

        # ---------------- Multiply Gate and Up Outputs ----------------
        if params.use_tkg_gate_up_proj_column_tiling:
            nisa.tensor_tensor(
                dst=up_sb_view.get_view(), data1=gate_sb_view.get_view(), data2=up_sb_view.get_view(), op=nl.multiply
            )
        else:
            nisa.tensor_tensor(dst=output, data1=gate_sb_view.get_view(), data2=up_sb_view.get_view(), op=nl.multiply)

    # ---------- Transpose hidden if column tiling is enabled ----------
    if params.use_tkg_gate_up_proj_column_tiling:
        # Transpose hidden [T, I] to [I1, I0, T]
        for i_tile in TiledRange(dims.I, dims.I0):
            psum_idx = i_tile.index % dims._psum_bmax
            tp_psum = nl.ndarray(
                (i_tile.size, dims.T),
                dtype=up_sb_view.dtype,
                buffer=nl.psum,
                name=f"{sbm.get_name_prefix()}transpose_psum_{i_tile.index}",
                address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
            )
            nisa.nc_transpose(
                dst=tp_psum,
                data=up_sb_view.slice(dim=0, start=0, end=dims.T)
                .slice(dim=1, start=i_tile.index * dims.I0, end=i_tile.index * dims.I0 + i_tile.size)
                .get_view(),
            )
            nisa.tensor_copy(dst=output[0 : i_tile.size, i_tile.index, 0 : dims.T], src=tp_psum)

    return tiles
