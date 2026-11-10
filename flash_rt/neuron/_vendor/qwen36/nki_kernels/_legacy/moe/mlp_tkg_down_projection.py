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

"""Down projection sub-kernels for MLP TKG with column tiling and LHS/RHS swap modes."""

import os as _os

import nki.isa as nisa
import nki.language as nl

# Bypass the down-weight hoist (reshape on I needs I % 128 == 0).
_MOE_LEGACY_DOWN_WEIGHT_LOAD = _os.environ.get("NKI_MOE_LEGACY_DOWN_WEIGHT_LOAD", "0") == "1"

from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.interleave_copy import interleave_copy
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.tiled_range import TiledRange
from nkilib.core.mlp.mlp_parameters import MLPParameters, mlpp_has_down_projection_bias
from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsDownTileCounts,
    MLPTKGConstantsGateUpTileCounts,
)
from nkilib.core.mlp.mlp_tkg.mlp_tkg_utils import adaptive_dge_mode

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def down_projection(
    hidden: nl.ndarray,
    weight: nl.ndarray,
    bias: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
):
    """
    Performs a single Down projection shard on the H.

    Computes: Hidden[I, T] @ Weight[I, H] + Optional(Bias[1, H]) → [T, H]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    Weight tiles are loaded [HTile, I] at a time for efficient memory access:
    H/HTile * [ HTile/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ] ]

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
    H/HTile * HTile/column_tiling_factor(parallel execution) * column_tile/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ]

    Key Points:
    -----------
    - Matrix multiplication is tiled along H and I
    - Column tiling improves PE utilization for small T

    Returns:
        Output tensor with shape [T, H]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I, H = weight.shape[0], dims.H_per_shard
    kernel_assert(
        H == dims.H_per_shard,
        f"Weight Sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # Calculate the starting weight index for the down projection.
    # offset the index to avoid anti-dependencies when loading weights in sequence.
    weight_base_idx = tiles.weight_base_idx

    # ---------- Bias handling ----------
    is_bias = bias is not None
    if is_bias:
        # Load bias with broadcast across T dimension using TensorView
        # [1, H_total] -> slice to [1, H_per_shard] -> broadcast to [T, H_per_shard]
        bias_hbm_view = (
            TensorView(bias).slice(dim=1, start=dims.shard_id * H, end=dims.shard_id * H + H).broadcast(dim=0, size=T)
        )
        bias_tile_view = TensorView(bias_tile)
        nisa.dma_copy(
            dst=bias_tile_view.get_view(),
            src=bias_hbm_view.get_view(),
            dge_mode=_DGE_MODE_NONE,
        )

    # ---------- Compute matmul  ----------
    # compute the number of required PSUM banks per HTile
    num_required_psums_per_HTile = div_ceil(tiles.HTile, dims._psum_fmax)
    num_required_psum_after_column_tiling = div_ceil(num_required_psums_per_HTile, dims.column_tiling_factor)

    # Calculate how many HTiles can be computed in parallel
    # Total PSUM banks (_psum_bmax=8) divided by PSUMs needed per HTile
    # Each PSUM group processes one HTile independently
    num_available_psum_group = dims._psum_bmax // num_required_psum_after_column_tiling

    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Calculate start offset for H
        h_offset = hidden_tiles.start_offset

        # Allocate PSUM
        psum_offset = (hidden_tiles.index % num_available_psum_group) * num_required_psum_after_column_tiling
        result_psums = []
        for psum_idx in range(num_required_psum_after_column_tiling):
            result_psum = nl.ndarray(
                shape=(dims._pmax, dims._psum_fmax),
                dtype=nl.float32,
                name=f"down_psum_{sbm.get_name_prefix()}_{hidden_tiles.index}_{psum_offset + psum_idx}",
                buffer=nl.psum,
                address=None if sbm.is_auto_alloc() else (0, (psum_offset + psum_idx) * dims._psum_fmax * 4),
            )
            result_psums.append(result_psum)

        for i_tiles in TiledRange(I, I0):
            # Load weight of [I, HTile] elements into [I0, HTile]
            weight_idx = (
                weight_base_idx + hidden_tiles.index * dims.num_total_128_tiles_per_I + i_tiles.index
            ) % tiles.num_allocated_w_tile
            nisa.dma_copy(
                dst=weight_tiles[weight_idx].ap(
                    pattern=[[weight_tiles[weight_idx].shape[1], i_tiles.size], [1, hidden_tiles.size]],
                    dtype=nl.float8_e4m3 if str(weight.dtype) == "float8e4" else weight.dtype,
                ),
                src=weight.ap(
                    pattern=[[weight.shape[1], i_tiles.size], [1, hidden_tiles.size]],
                    offset=i_tiles.start_offset * weight.shape[1] + dims.shard_id * dims.H_per_shard + h_offset,
                    dtype=nl.float8_e4m3 if str(weight.dtype) == "float8e4" else weight.dtype,
                ),
                dge_mode=_DGE_MODE_NONE,
            )

            # Matmult
            for compute_tile in TiledRange(hidden_tiles.size, dims._psum_fmax):
                # Distribute tiles across columns and PSUM banks for optimal memory access
                # Example: with column_tiling_factor=4, tiles 0-3 go to columns 0-3 in bank 0,
                #          tiles 4-7 go to columns 0-3 in bank 1, etc.
                column_tile_index = compute_tile.index % dims.column_tiling_factor
                psum_bank_index = compute_tile.index // dims.column_tiling_factor
                nisa.nc_matmul(
                    dst=result_psums[psum_bank_index][
                        nl.ds(dims.column_tiling_dim * column_tile_index, T),
                        0 : compute_tile.size,
                    ],
                    stationary=hidden[0 : i_tiles.size, i_tiles.index, 0:T],
                    moving=weight_tiles[weight_idx][
                        0 : i_tiles.size, nl.ds(compute_tile.index * dims._psum_fmax, compute_tile.size)
                    ],
                    tile_position=(0, dims.column_tiling_dim * column_tile_index),
                    tile_size=(I0, dims.column_tiling_dim),
                )

        # ---------- Accumulate partial PSUM output to SB and optionally apply dequant scale ----------
        dequant_tile_view = TensorView(dequant_tile) if params.quant_params.is_quant() else None
        for compute_tile in TiledRange(hidden_tiles.size, dims._psum_fmax):
            # Read tiles across columns and PSUM banks
            column_tile_index = compute_tile.index % dims.column_tiling_factor
            psum_bank_index = compute_tile.index // dims.column_tiling_factor
            dst_offset = h_offset + compute_tile.index * dims._psum_fmax
            interleave_copy(
                index=column_tile_index,
                dst=output_tile[0:T, nl.ds(dst_offset, compute_tile.size)],
                src=result_psums[psum_bank_index][
                    nl.ds(dims.column_tiling_dim * column_tile_index, T),
                    0 : compute_tile.size,
                ],
                scale=dequant_tile_view.slice(dim=1, start=dst_offset, end=dst_offset + compute_tile.size)
                if params.quant_params.is_quant_row()
                else dequant_tile_view,
                bias=None,
            )

    # ---------- Apply bias separately from matmul pipeline ----------
    if is_bias:
        bias_tile_view = TensorView(bias_tile)
        output_tile_view = TensorView(output_tile)
        nisa.tensor_tensor(
            dst=output_tile_view.get_view(),
            data1=output_tile_view.get_view(),
            data2=bias_tile_view.get_view(),
            op=nl.add,
        )


def down_projection_lhs_rhs_swap(
    hidden: nl.ndarray,
    weight: nl.ndarray,
    bias: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
    hoisted_weight: nl.ndarray = None,
):
    """
    Performs a single Down projection shard on the H using regular matmult with operands swapped

    Computes: Weight[I, H] @ Hidden[I, T] + Optional(Bias[1, H]) → [T, H]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Optional hoisted weight load:
        When `hoisted_weight` is provided (shape
        [I0, num_total_128_tiles_per_I, H_per_shard]), the per-(HTile, i_tile)
        DMA is replaced with a single dma_copy pulling the full weight slice.
        Drops descriptor count from (num_HTiles * num_total_128_tiles_per_I)
        to 1 per call.

    Returns:
        Output tensor with shape [128, H//128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I, H = weight.shape[0], dims.H_per_shard
    H0 = dims.H0

    kernel_assert(
        H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # Calculate the starting weight index for the down projection.
    # Offset to prevent anti-dependencies with gate/up weight loads, enabling efficient weight tile loading.
    weight_base_idx = tiles.weight_base_idx

    # ---------- Load Bias ----------
    is_bias = bias != None
    if is_bias:
        bias_hbm_view = bias.slice(
            dim=0, start=dims.shard_id * dims.H_per_shard, end=(dims.shard_id + 1) * dims.H_per_shard
        ).reshape_dim(dim=0, shape=(H0, dims.H1_shard))
        bias_tile_view = TensorView(bias_tile)
        nisa.dma_copy(
            src=bias_hbm_view.get_view(),
            dst=bias_tile_view.get_view(),
            dge_mode=adaptive_dge_mode(bias),
        )

    # ---------- Compute matmul ----------
    perBankT = dims._psum_fmax // T
    num_required_down_psum_banks = div_ceil(dims.H1_shard, perBankT)
    kernel_assert(
        num_required_down_psum_banks <= dims._psum_bmax,
        f"Required psum banks for down projection: {num_required_down_psum_banks}, which exceeds hardware limit of {dims._psum_bmax}, please use CTE mode",
    )

    # Allocate PSUM buffers to store output
    result_psums = []
    for psum_idx in range(num_required_down_psum_banks):
        result_psum = nl.ndarray(
            (dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"down_psum_{sbm.get_name_prefix()}_{psum_idx}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # Hoisted weight load: one dma_copy for the full down-projection weight
    # slice. The naive reshape_dim((I0, num_total_128_tiles_per_I)) would put
    # slot as the inner sub-dim; we need partition inner (stride H_total) so
    # reshape to (num_total_128_tiles_per_I, I0) then permute (1, 0, 2).
    use_hoisted_weight = hoisted_weight is not None
    if use_hoisted_weight:
        full_weight_view = (
            weight.slice(
                dim=1,
                start=dims.shard_id * dims.H_per_shard,
                end=(dims.shard_id + 1) * dims.H_per_shard,
            )
            .reshape_dim(dim=0, shape=(dims.num_total_128_tiles_per_I, I0))
            .permute((1, 0, 2))
        )
        nisa.dma_copy(
            dst=hoisted_weight[0:I0, 0 : dims.num_total_128_tiles_per_I, 0 : dims.H_per_shard],
            src=full_weight_view.get_view(),
            dge_mode=adaptive_dge_mode(weight),
        )

    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Calculate starting offset
        h_offset = hidden_tiles.start_offset
        h1_offset = h_offset // H0

        i_tiles = TiledRange(I, I0)
        for i_tile in i_tiles:
            # Load weight of [I, HTile] elements into [128, HTile]
            weight_idx = (
                weight_base_idx + hidden_tiles.index * len(i_tiles) + i_tile.index
            ) % tiles.num_allocated_w_tile
            if use_hoisted_weight:
                # Hoisted path: matmul indexes into the single pre-loaded tile.
                # Skip DMA entirely.
                pass
            else:
                nisa.dma_copy(
                    dst=weight_tiles[weight_idx][0 : i_tile.size, 0 : hidden_tiles.size],
                    src=weight.slice(dim=0, start=i_tile.start_offset, end=i_tile.start_offset + i_tile.size)
                    .slice(
                        dim=1,
                        start=dims.shard_id * dims.H_per_shard + h_offset,
                        end=dims.shard_id * dims.H_per_shard + h_offset + hidden_tiles.size,
                    )
                    .get_view(),
                    dge_mode=adaptive_dge_mode(weight),
                )

            # When use_tkg_down_proj_optimized_layout is disabled,
            # the weight-tile access pattern uses a stride of H1_shard.
            # When it is enabled, the framework is expected to permute the weights
            # so that the weight tile can be accessed without any stride.
            h1_tiles = TiledRange(hidden_tiles.size, H0)
            if params.use_tkg_down_proj_optimized_layout:
                for h1_tile in h1_tiles:
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    if use_hoisted_weight:
                        # Hoisted tile shape [I0, num_total_128_tiles_per_I, H_per_shard];
                        # the per-HTile, per-h1_tile slot in canonical maps to
                        # h_offset + h1_tile.index*H0 in the full H_per_shard axis.
                        stationary_view = hoisted_weight[
                            0 : i_tile.size,
                            i_tile.index,
                            nl.ds(h_offset + h1_tile.index * H0, H0),
                        ]
                    else:
                        stationary_view = weight_tiles[weight_idx][
                            0 : i_tile.size, nl.ds(h1_tile.index * H0, H0)
                        ]
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=stationary_view,
                        moving=hidden[0 : i_tile.size, i_tile.index, 0:T],
                    )
            else:
                for h1_tile in h1_tiles:
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    if use_hoisted_weight:
                        # Hoisted tile shape [I0, num_total_128_tiles_per_I, H_per_shard].
                        # Non-optimized layout interleaves stride=len(h1_tiles) within
                        # the HTile band of free-dim positions, with h1_tile.index as
                        # the in-band offset. With the hoisted tile we keep the same
                        # interleave AND step into the right i_tile slot and HTile
                        # base via the offset.
                        stationary_view = hoisted_weight.ap(
                            pattern=[
                                [dims.num_total_128_tiles_per_I * dims.H_per_shard, i_tile.size],
                                [len(h1_tiles), H0],
                            ],
                            offset=i_tile.index * dims.H_per_shard + h_offset + h1_tile.index,
                        )
                    else:
                        stationary_view = weight_tiles[weight_idx].ap(
                            pattern=[
                                [hidden_tiles.size, i_tile.size],
                                [len(h1_tiles), H0],
                            ],
                            offset=h1_tile.index,
                        )
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=stationary_view,
                        moving=hidden[0 : i_tile.size, i_tile.index, 0:T],
                    )

    # Reshape output to 2D
    output_tile = output_tile.reshape((H0, dims.H1_shard * T))

    # Copy PSUM output to SB
    bias_tile_view = TensorView(bias_tile) if is_bias else None
    for psum_tiles in TiledRange(dims.H1_shard, perBankT):
        # Number of elements that each PSUM can hold
        perBankElem = perBankT * T
        # Actual number of elements in the current PSUM
        numElements = psum_tiles.size * T

        if params.quant_params.is_quant_row():
            dequant_tile_view = (
                TensorView(dequant_tile)
                .slice(dim=1, start=psum_tiles.index * psum_tiles.size, end=(psum_tiles.index + 1) * psum_tiles.size)
                .expand_dim(dim=2)
                .broadcast(dim=2, size=T)
            )
        else:
            dequant_tile_view = TensorView(dequant_tile) if params.quant_params.is_quant() else None

        interleave_copy(
            index=psum_tiles.index,
            dst=output_tile[0:H0, nl.ds(psum_tiles.index * perBankElem, numElements)],
            src=result_psums[psum_tiles.index][0:H0, 0:numElements],
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias separately from matmul pipeline ----------
    if is_bias:
        # Broadcast bias across T dimension and add to output
        # output_tile is [H0, H1_shard * T], bias_tile is [H0, H1_shard]
        bias_tile_broadcasted = TensorView(bias_tile).expand_dim(dim=2).broadcast(dim=2, size=T)
        output_tile_view = (
            TensorView(output_tile).slice(dim=0, start=0, end=H0).slice(dim=1, start=0, end=dims.H1_shard * T)
        )
        nisa.tensor_tensor(
            dst=output_tile_view.get_view(),
            data1=output_tile_view.get_view(),
            data2=bias_tile_broadcasted.get_view(),
            op=nl.add,
        )

    # Reshape output back to 3D
    output_tile = output_tile.reshape((H0, dims.H1_shard, T))


def process_down_projection(
    hidden: nl.ndarray,
    output: nl.ndarray,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    gate_tile_info: MLPTKGConstantsGateUpTileCounts,
    sbm: SbufManager,
):
    """
    Performs the Down projection for MLP (T = BxS).
    Expected hidden tensor shape is [128(I0), I/128, T],
    with a remainder tile shape of [res_I, I/128, T] if I is not a multiple of 128.

    Overview:
    ---------
    hidden @ down_weight + optional(down_bias)
    # [T, H] = [T, I] @ [I, H] + optional([1, H])

    Hardware constraints (max partition size of 128) require tiling along the I dimension:
    # hidden [128, I//128, T] @ down_weight [128, I//128, H]

    Behavior based on `use_tkg_down_proj_column_tiling`:
    ---------------------------------------
    - False: column tiling(`down_projection`)
        hidden[128, T] @ down_weight[128, H] → [T, H]
        Output shape: [T, H]

    - True: operands swapped(`down_projection_lhs_rhs_swap`)
        down_weight[128, H] @ hidden[128, T] → [H, T]
        Further tiling along H: [128, H//128, T]
        Output shape: [128, H//128, T]

    DMA mode:
    ---------
    Based on experiments, Static DMA provides better performance.
    The MLP TKG implementation therefore uses Static DMA for tensor loads.
    If HBM out-of-memory (OOM) issues arise, we can fall back to DGE mode.

    Note:
    ---------
    Caller will have the flexibility to manage sbm:sbufManager's scope and interleave degree.

    """
    down_w, down_b, down_w_scale, down_in_scale = (
        params.down_proj_weights_tensor,
        params.bias_params.down_proj_bias_tensor,
        params.quant_params.down_w_scale,
        params.quant_params.down_in_scale,
    )

    # ---------------- Allocate Bias Tile ----------------
    bias_tile = None
    if mlpp_has_down_projection_bias(params):
        if params.use_tkg_down_proj_column_tiling:
            bias_tile = sbm.alloc_stack(
                (dims.T, dims.H_per_shard),
                dtype=down_b.dtype,
                name=f"down_broadcasted_bias",
                buffer=nl.sbuf,
            )
        else:
            bias_tile = sbm.alloc_stack(
                (dims.H0, dims.H1_shard),
                dtype=down_b.dtype,
                name=f"down_bias",
                buffer=nl.sbuf,
            )

    dequant_tile = None
    # ---------------- Static quantization ----------------
    if params.quant_params.is_quant_static():
        par_dims = dims.T if params.use_tkg_down_proj_column_tiling else dims.H0
        down_w_scale_view = down_w_scale.slice(dim=0, start=0, end=par_dims)
        dequant_tile = sbm.alloc_stack((par_dims, 1), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4)
        nisa.dma_copy(
            dst=dequant_tile[:par_dims, :],
            src=down_w_scale_view.get_view(),
            dge_mode=adaptive_dge_mode(down_w_scale_view),
        )

    # ---------------- Row quantization ----------------
    elif params.quant_params.is_quant_row():
        slice_dim = dims.T if params.use_tkg_down_proj_column_tiling else 1
        down_w_scale_view = down_w_scale.slice(dim=0, start=0, end=slice_dim).slice(
            dim=1, start=dims.shard_id * dims.H_per_shard, end=dims.shard_id * dims.H_per_shard + dims.H_per_shard
        )

        if params.use_tkg_down_proj_column_tiling:
            dequant_tile = sbm.alloc_stack(
                (dims.T, dims.H_per_shard), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4
            )
            nisa.dma_copy(
                dst=dequant_tile[0 : dims.T, 0 : dims.H_per_shard],
                src=down_w_scale_view.get_view(),
                dge_mode=_DGE_MODE_NONE,
            )
        else:
            dequant_tile = sbm.alloc_stack(
                (dims.H0, dims.H1_shard), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4
            )
            down_w_scale_view = down_w_scale_view.squeeze_dim(dim=0)
            nisa.dma_copy(
                dst=dequant_tile[0 : dims.H0, 0 : dims.H1_shard],
                src=down_w_scale_view.reshape_dim(dim=0, shape=(dims.H0, dims.H1_shard)).get_view(),
                dge_mode=adaptive_dge_mode(down_w_scale_view),
            )

    # ---------------- Allocate Weight Tiles ----------------
    # By calculating the remaining SBUF space, we allocate as many weight tiles as possible
    if sbm.is_auto_alloc():
        remaining_space = 0
        current_address = 0
    else:
        remaining_space = sbm.get_free_space()
        kernel_assert(remaining_space > 0, f"Not enough memory for down projection weight")
        current_address = sbm.get_stack_curr_addr()

    # Calculate tile info
    tiles = MLPTKGConstants.calculate_down_tiles(
        current_address, remaining_space, params, dims, gate_tile_info, sbm.is_auto_alloc()
    )

    # Hoisted down weight tile (lhs_rhs_swap path): replace the per-HTile,
    # per-i_tile ring buffer with a single [I0, num_total_128_tiles_per_I,
    # H_per_shard] tile loaded by one DMA. Bytes-equivalent to the ring tiles
    # when HTile divides H_per_shard.
    down_hoisted_w_tile = None
    use_hoisted_down_load = (
        (not params.use_tkg_down_proj_column_tiling)
        and (not _MOE_LEGACY_DOWN_WEIGHT_LOAD)
    )

    weight_tiles = []
    if use_hoisted_down_load:
        weight_dtype = nl.float8_e4m3 if str(down_w.dtype) == "float8e4" else down_w.dtype
        down_hoisted_w_tile = sbm.alloc_stack(
            (dims.I0, dims.num_total_128_tiles_per_I, dims.H_per_shard),
            name="down_hoisted_w_tile",
            dtype=weight_dtype,
            buffer=nl.sbuf,
        )
    else:
        for w_tile_idx in range(tiles.num_allocated_w_tile):
            weight_tile = sbm.alloc_stack(
                (dims.I0, tiles.HTile),
                name=f"down_w_tile_{w_tile_idx}",
                dtype=nl.float8_e4m3 if str(down_w.dtype) == "float8e4" else down_w.dtype,
                buffer=nl.sbuf,
            )
            weight_tiles.append(weight_tile)

    # ---------------- Down Projection ----------------
    if params.use_tkg_down_proj_column_tiling:
        down_projection(
            hidden=hidden,
            weight=down_w,
            bias=down_b,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            sbm=sbm,
        )

    else:
        down_projection_lhs_rhs_swap(
            hidden=hidden,
            weight=down_w,
            bias=down_b,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            sbm=sbm,
            hoisted_weight=down_hoisted_w_tile,
        )

    return output, tiles
