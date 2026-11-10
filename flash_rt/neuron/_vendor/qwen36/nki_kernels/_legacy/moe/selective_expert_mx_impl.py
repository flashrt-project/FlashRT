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

"""Selective-expert MoE token generation implementation with MX (microscaling) FP4 quantization support.

Vendored from nkilib/core/moe/moe_tkg/selective_expert_mx_impl.py. The only
edit relative to upstream is a `name_prefix: str = ""` kwarg threaded into
the impl. Without it, calling this kernel multiple times in one trace (e.g.
24 layers in the megakernel) trips NKI's duplicate-op-name checker on the
four hardcoded `nl.ndarray(name=...)` allocations + subkernel allocations
under the internal sbm. The prefix scopes them to a unique namespace per
call site.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from nkilib.core.mlp.mlp_parameters import MLPParameters
from .down_projection_mx_shard_H import (
    ProjConfig,
    down_projection_mx_tp_shard_H,
)
from nkilib.core.mlp.mlp_tkg.gate_up_projection_mx_shard_H import (
    process_fused_gate_up_projection_mxfp4,
)

# MLP utils
from nkilib.core.mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from nkilib.core.mlp.mlp_tkg.mlp_tkg_utils import (
    _layout_adapter_hbm,
    _layout_adapter_sb,
)
from nkilib.core.mlp.mlp_tkg.projection_mx_constants import (
    _pmax,
    _q_height,
    _q_width,
)
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil

# Common utils
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.moe.moe_tkg.moe_tkg_utils import broadcast_token_affinity, gather_expert_affinities


def _selective_expert_moe_tkg_mxfp4(
    params: MLPParameters,
    output: nl.ndarray,
    name_prefix: str = "",
) -> nl.ndarray:
    """
    Perform selective-expert MoE MLP token generation with MXFP4 quantization.

    The input first goes through a layout adapter for desired MX-quantizable layout.

    Args:
        params (MLPParameters): MLPParameters containing all input tensors and configuration.
        output (nl.ndarray): [T, H], Output tensor in HBM.

    Returns:
        output (nl.ndarray): [T, H], Output tensor with MoE computation results in HBM.

    Notes:
        - This kernel only supports gate/up and down proj both swapped
        - gate_up_weights: mxfp4[E, _pmax, 2, n_H512_tiles, I] in HBM (2 dim means up & gate weights stacked)
        - down_weights: mxfp4[E, I_p, ceil(I/512), H] in HBM, where I_p = I//4 if I <= 512 else _pmax
        - gate_up_weights_scale: uint8[E, _pmax // _q_height, 2, n_H512_tiles, I] in HBM
        - down_weights_scale: uint8[E, I_p // _q_height, ceil(I/512), H] in HBM
        - gate_up_weights_bias: bf16[E, I_p, 2, ceil(I/512), 4] in HBM
        - down_weights_bias: bf16[E, H] in HBM (needs offline shuffling for down_lhs_rhs_swap)

    Pseudocode:
        # Layout adapter and quantization
        input_qtz, input_scale = layout_adapter(input)

        # Process each token
        for token_idx in range(T):
            for expert_k_idx in range(K):
                expert_idx = expert_index[token_idx, expert_k_idx]

                # Gate/up projection
                intermediate = gate_up_projection(input_qtz[token_idx], weights[expert_idx])

                # Down projection
                expert_out = down_projection(intermediate, down_weights[expert_idx])

                # Apply affinity and accumulate
                expert_out *= expert_affinities[token_idx, expert_idx]
                if expert_k_idx == 0:
                    output[token_idx] = expert_out
                else:
                    output[token_idx] += expert_out
    """
    # Init dims
    dims = MLPTKGConstants.calculate_constants(params)

    # This kernel uses auto allocation, init an auto allocator for subkernels that requires a sbm
    auto_sbm = SbufManager(0, 200 * 1024, use_auto_alloc=True)
    auto_sbm.set_name_prefix(name_prefix)
    auto_sbm.open_scope()

    kernel_assert(not params.store_output_in_sbuf, "_all_token_mlp_mxfp4_kernel does not support sbuf output")
    kernel_assert(dims.T <= _pmax, "_all_token_mlp_mxfp4_kernel does not support T > 128")

    shard_on_K = True

    # Get intermediate dims
    kernel_assert(dims.H_shard % (_pmax * _q_width) == 0, "Expect H after sharding to be divisible by 512")
    n_H512_tile_sharded = dims.H_shard // (_pmax * _q_width)
    n_I512_tile = div_ceil(dims.I, (_pmax * _q_width))
    T_padded = div_ceil(dims.T, 4) * 4

    # This is used iff. shard_on_K
    num_shards, shard_id = nl.num_programs(0), nl.program_id(0)
    K_sharded = dims.K
    if shard_on_K:
        kernel_assert(dims.K % num_shards == 0, "Selective load shard on K requires K divisible by num NC")
        K_sharded = dims.K // num_shards

    io_dtype = params.output_dtype

    # Use layout adapter to get quantizable layout for Gate/Up projection, runs this unsharded since we shard on K
    input_sb_shfl = None  # always bf16[_pmax, n_H512_tile_sharded, T_padded, _q_width]@SB
    if params.input_in_sbuf:
        input_sb_shfl = _layout_adapter_sb(params.hidden_tensor, n_prgs=1, prg_id=0)
    else:
        input_sb_shfl = _layout_adapter_hbm(params.hidden_tensor, n_prgs=1, prg_id=0)

    input_flat = input_sb_shfl.reshape((_pmax, n_H512_tile_sharded * T_padded * 4))
    inp_qtz = nl.ndarray((_pmax, n_H512_tile_sharded * T_padded), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    inp_scale = nl.ndarray(inp_qtz.shape, dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=inp_qtz, src=input_flat, dst_scale=inp_scale)

    inp_qtz = inp_qtz.reshape((_pmax, n_H512_tile_sharded, T_padded))
    inp_scale = inp_scale.reshape(inp_qtz.shape)

    # Allocate SBUF location to accumulate output which has shape [128, H_per_shard] to store the outputs for
    # four tokens on each of the four SBUF quadrants. This is to save sendrecvs (reduced by 4x).
    output_temp_shape = (dims.H0, dims.T, dims.H1_shard)
    output_temp = nl.ndarray(output_temp_shape, dtype=io_dtype, name=f"{name_prefix}temp_output_sbuf", buffer=nl.sbuf)

    # Determine tiling on T. When down is not swapped (producing [T, H] output), it's tiled by 4. Otherwise we don't tile.
    sz_T_tile, n_T_tile = (dims.T, 1)

    # Allocate SBUF locations for gate/up projection results. NOTE: likely won't need fp32 precision, but keep this in mind
    intermediate_state_sb = nl.ndarray(
        (_pmax, n_I512_tile, 4, _q_width), dtype=io_dtype, name=f"{name_prefix}intermediate_state_sbuf", buffer=nl.sbuf
    )

    # Load expert index
    if params.expert_params.expert_index.buffer == nl.sbuf:
        expert_idx = params.expert_params.expert_index
    else:
        expert_idx = nl.ndarray((dims.T, dims.K), dtype=params.expert_params.expert_index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=expert_idx, src=params.expert_params.expert_index[0 : dims.T, 0 : dims.K]
        )  # indices have to be in SBUF

    # Prepare expert index into broadcasted form for generating DGE indices
    # These scalars are broadcasted 4 times on the pdim for DGE indices
    expert_idx_f32 = nl.ndarray(expert_idx.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expert_idx_f32, op=nl.copy, data=expert_idx)
    expert_idx_scalar_broadcasted = nl.ndarray(
        (4, dims.T, K_sharded), dtype=params.expert_params.expert_index.dtype, buffer=nl.sbuf
    )
    for i_k in range(K_sharded):
        i_k_lnc_adjusted = (i_k + shard_id * K_sharded) if shard_on_K else i_k

        # Transpose a slice of [T, 1 (i_k)] such that data starts on par 0
        expert_idx_cur_k_tp_psum = nl.ndarray((4, dims.T), dtype=expert_idx_f32.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=expert_idx_cur_k_tp_psum,
            data=expert_idx_f32.ap(
                pattern=[[dims.K, dims.T], [0, 4]], offset=i_k_lnc_adjusted
            ),  # repeated (by 4 times) read on f-dim
        )
        nisa.activation(
            dst=expert_idx_scalar_broadcasted[:, :, i_k],
            op=nl.copy,
            data=expert_idx_cur_k_tp_psum[:, :],
        )

    # Prepare expert index into vector DGE indices format
    p_idx_vector_gup = nl.ndarray((_pmax, dims.T, K_sharded), dtype=nl.float32, buffer=nl.sbuf, name=f"{name_prefix}p_idx_vector_gup")
    nisa.memset(dst=p_idx_vector_gup, value=-1.0)

    p_idx_vector_down = nl.ndarray(
        (_pmax, dims.T, K_sharded), dtype=nl.float32, buffer=nl.sbuf, name=f"{name_prefix}p_idx_vector_down"
    )
    nisa.memset(dst=p_idx_vector_down, value=-1.0)

    n_quadrants_needed = 4
    for i_quad in range(n_quadrants_needed):
        arange_4P = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=arange_4P, pattern=[[1, 1]], offset=i_quad * 4, channel_multiplier=1)

        # Generate indices for gate and up
        nisa.activation(
            dst=p_idx_vector_gup[i_quad * 32 : i_quad * 32 + 4, :, :],
            op=nl.copy,
            data=expert_idx_scalar_broadcasted,
            scale=float(16),
            bias=arange_4P,
        )

        # Generate indices for down
        nisa.activation(
            dst=p_idx_vector_down[i_quad * 32 : i_quad * 32 + 4, :, :],
            op=nl.copy,
            data=expert_idx_scalar_broadcasted,
            scale=float(params.quant_params.down_w_scale.shape[1]),
            bias=arange_4P,
        )

    # Load expert affinity
    expert_affinities_sb = nl.ndarray(
        (dims._pmax, dims.E), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.sbuf
    )
    nisa.memset(dst=expert_affinities_sb, value=0.0)
    if params.expert_params.expert_affinities.buffer == nl.sbuf:
        nisa.tensor_copy(dst=expert_affinities_sb[: dims.T, :], src=params.expert_params.expert_affinities)
    else:
        # Prefetch expertIndices (Up to 128 tokens input)
        nisa.dma_copy(dst=expert_affinities_sb[: dims.T, :], src=params.expert_params.expert_affinities)

    # Gather expert affinities using utility function
    if params.expert_params.expert_affinities_eager != None:
        # broadcast expert_affinities_eager into a [128(P), K(F), T(F)] tensor
        expert_affi_eager_sb = nl.ndarray(
            (dims._pmax, K_sharded, dims.T), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.sbuf
        )
        for i_k in range(K_sharded):
            i_k_lnc_adjusted = (i_k + shard_id * K_sharded) if shard_on_K else i_k
            expert_affi_eager_tp = nl.ndarray(
                (1, dims.T), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=expert_affi_eager_tp, data=params.expert_params.expert_affinities_eager[: dims.T, i_k_lnc_adjusted]
            )
            nisa.tensor_copy(
                dst=expert_affi_eager_sb[0:1, i_k, 0 : dims.T],
                src=expert_affi_eager_tp[0:1, 0 : dims.T],
                engine=nisa.vector_engine,
            )
        expert_affi_eager_sb = expert_affi_eager_sb.reshape((dims._pmax, K_sharded * dims.T))
        stream_shuffle_broadcast(src=expert_affi_eager_sb, dst=expert_affi_eager_sb)
        expert_affi_eager_sb = expert_affi_eager_sb.reshape((dims._pmax, K_sharded, dims.T))

    else:
        gathered_affinities_sb = gather_expert_affinities(expert_affinities_sb, expert_idx, dims, auto_sbm)
        expert_affinity_sb = nl.ndarray((_pmax, dims.T, dims.K), dtype=gathered_affinities_sb.dtype, buffer=nl.sbuf)
        for i_t in range(dims.T):
            # Set SBM prefix to deduplicate (keep outer name_prefix so per-layer
            # invocations don't collide across the 24-layer megakernel trace).
            auto_sbm.set_name_prefix(f"{name_prefix}T{i_t}_")

            # In the new FE, t[:, i_t, :] is 3D instead of 2D. Reshape as a workaround
            expert_affinity_sb = expert_affinity_sb.reshape((_pmax, dims.T * dims.K))
            broadcast_token_affinity(
                dst=expert_affinity_sb[:, i_t * dims.K : (i_t + 1) * dims.K],
                gathered_affinities_sb=gathered_affinities_sb,
                token_index=i_t,
                dims=dims,
                sbm=auto_sbm,
            )
            expert_affinity_sb = expert_affinity_sb.reshape((_pmax, dims.T, dims.K))
        # Reset SBM prefix back to the outer name_prefix.
        auto_sbm.set_name_prefix(name_prefix)

    p_I = _pmax if dims.I > 512 else dims.I // 4

    # The SBUF scale tensor for gate_up/down projection can be reused by different expert iterations to avoid redundant memset
    gate_up_scale_sb = nl.ndarray(
        (_pmax, 2, n_H512_tile_sharded, dims.I), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name_prefix}gate_up_w_scale_sb"
    )
    nisa.memset(dst=gate_up_scale_sb, value=0)
    down_scale_sb = nl.ndarray((_pmax, n_I512_tile, dims.H_shard), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=down_scale_sb, value=0)

    for i_T_tile in range(n_T_tile):
        # For down proj with [T, H] layout, all four (at most) token outputs will write to the same output_temp on each of the four quadrants,
        # then one local CC + one DMA store is needed for saving these four (at most) outputs.
        for i_T_sub_tile in range(sz_T_tile):
            # Get true token index
            i_t = i_T_tile * sz_T_tile + i_T_sub_tile

            # Even with static ranges, NKI has undefined behaviour when using breaks
            if i_t < dims.T:
                inp_qtz_cur_t = nl.ndarray((_pmax, n_H512_tile_sharded, 4), dtype=inp_qtz.dtype, buffer=nl.sbuf)
                inp_scale_cur_t = nl.ndarray((_pmax, n_H512_tile_sharded, 4), dtype=inp_scale.dtype, buffer=nl.sbuf)
                nisa.memset(dst=inp_scale_cur_t, value=0)
                nisa.tensor_copy(
                    dst=inp_qtz_cur_t.ap(
                        pattern=[[n_H512_tile_sharded * 4, _pmax], [4, n_H512_tile_sharded]], offset=0, dtype=nl.float32
                    ),
                    src=inp_qtz.ap(
                        pattern=[[n_H512_tile_sharded * T_padded, _pmax], [T_padded, n_H512_tile_sharded]],
                        offset=i_t,
                        dtype=nl.float32,
                    ),
                    engine=nisa.vector_engine,
                )
                nisa.tensor_copy(
                    dst=inp_scale_cur_t[:, :, :1], src=inp_scale[:, :, i_t : i_t + 1], engine=nisa.vector_engine
                )

                for i_k in range(K_sharded):
                    i_k_lnc_adjusted = (i_k + shard_id * K_sharded) if shard_on_K else i_k

                    # Gate and Up projection
                    process_fused_gate_up_projection_mxfp4(
                        hidden=inp_qtz_cur_t,  # [_pmax, n_H512_tile_sharded, 4_padded_from_1_t]
                        hidden_scale=inp_scale_cur_t,  # [_pmax, n_H512_tile_sharded, 4_padded_from_1_t]
                        gate_up_weights=params.gate_proj_weights_tensor,  # [E, _pmax, 2, n_H512_tiles, I]
                        gate_up_scale=params.quant_params.gate_w_scale,  # [E, _pmax // _q_height, 2, n_H512_tiles, I]
                        gate_up_bias=params.bias_params.gate_proj_bias_tensor,  # [E, I_p, 2, ceil(I/512), 4], I_p = I//4 if I <= 512 else _pmax
                        p_idx_vector=p_idx_vector_gup[
                            :, i_t, i_k : i_k + 1
                        ],  # prepared expert idx for vec dge, note it only contains data for cur K-shard
                        gate_up_scale_sb=gate_up_scale_sb,  # [_pmax, 2, n_H512_tile_sharded, dims.I]
                        output=intermediate_state_sb,  # [_pmax, ceil(I/512), 4_padded_from_1_t, _q_width]
                        attrs=params,
                        dims=dims,
                        gate_up_weights_E_offset=expert_idx.ap(
                            pattern=[[dims.K, 1], [1, 1]], offset=i_t * dims.K + i_k_lnc_adjusted
                        ),
                        gate_up_bias_E_offset=expert_idx.ap(
                            pattern=[[dims.K, 1], [1, 1]], offset=i_t * dims.K + i_k_lnc_adjusted
                        ),
                    )

                    # Efficiently load down projection scales using vector DGE indexing
                    scale_shape = params.quant_params.down_w_scale.shape

                    down_scale_view = params.quant_params.down_w_scale.reshape(
                        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3])
                    )

                    n_quadrants_needed, n_remaining_partition = divmod(p_I, 32)
                    n_remaining_partition = n_remaining_partition // _q_height

                    # Handle remaining partitions if they exist
                    token_indices_on_p = nl.ndarray((_pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=token_indices_on_p, src=p_idx_vector_down[:, i_t, i_k])
                    nisa.dma_copy(
                        dst=down_scale_sb,
                        src=down_scale_view.ap(
                            pattern=[[n_I512_tile * dims.H, _pmax], [dims.H, n_I512_tile], [1, dims.H_shard]],
                            offset=0 if shard_on_K else (dims.shard_id * dims.H_shard),
                            vector_offset=token_indices_on_p,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                    )

                    # Load down proj weights into [I0, ceil(I/512), H_sharded] NOTE: this is pre-quantized and each elt is mx_x4 (packed I)
                    down_weight_qtz_sb = nl.ndarray(
                        (_pmax, n_I512_tile, dims.H_shard),
                        dtype=params.down_proj_weights_tensor.base_tensor.dtype,
                        buffer=nl.sbuf,
                    )
                    # Memset weight if input weight HBM does not pad on par dim
                    if p_I != _pmax:
                        nisa.memset(dst=down_weight_qtz_sb[:, n_I512_tile - 1, :], value=0)

                    # down_proj_weights_tensor shape: (E, p_I, n_I512_tile, H)
                    H_offset = 0 if shard_on_K else (dims.shard_id * dims.H_shard)
                    expert_scalar = (
                        TensorView(expert_idx)
                        .slice(dim=0, start=i_t, end=i_t + 1)
                        .slice(dim=1, start=i_k_lnc_adjusted, end=i_k_lnc_adjusted + 1)
                        .get_view()
                    )
                    down_weights_view = (
                        TensorView(params.down_proj_weights_tensor.base_tensor)
                        .select(dim=0, index=expert_scalar)
                        .slice(dim=2, start=H_offset, end=H_offset + dims.H_shard)
                    )
                    nisa.dma_copy(
                        dst=down_weight_qtz_sb[:p_I, :, :],
                        src=down_weights_view.get_view(),
                        dge_mode=nisa.dge_mode.hwdge,
                    )
                    down_weight_qtz_sb = down_weight_qtz_sb.view(params.down_proj_weights_tensor.dtype)

                    # Call down proj with out_p_offset (to ensure the data in cur_down_out is on the same partition as output_temp)
                    down_cfg = ProjConfig(
                        H=dims.H_shard,
                        I=dims.I,
                        BxS=4,
                        n_prgs=1,
                        prg_id=0,  # perform as LNC1
                        out_p_offset=0,
                    )
                    cur_down_out = down_projection_mx_tp_shard_H(
                        inter_sb=intermediate_state_sb,
                        weight=down_weight_qtz_sb,
                        weight_scale=down_scale_sb,
                        bias_sb=None,  # NOTE: we assert down swap layout, postpone down bias to after down projection
                        cfg=down_cfg,
                        name_prefix=f"{name_prefix}T{i_t}_K{i_k}_",
                    )  # NOTE: only the first 1_T partition has value

                    # cur_down_out has shape [H0, H1_shard, 4], slice out the part that has value, shape [H0, H1]
                    cur_down_out_view = cur_down_out[:, :, 0]

                    # Apply affinity and accumulate to SB
                    # output_temp shape: [H0, T, H1_shard]
                    cur_affinity = (
                        expert_affinity_sb[:, i_t, i_k_lnc_adjusted]
                        if params.expert_params.expert_affinities_eager == None
                        else expert_affi_eager_sb[:, i_k, i_t]
                    )
                    if i_k == 0:
                        nisa.tensor_scalar(
                            dst=output_temp[:, i_t, :], data=cur_down_out_view, op0=nl.multiply, operand0=cur_affinity
                        )
                    else:
                        nisa.scalar_tensor_tensor(
                            dst=output_temp[:, i_t, :],
                            data=cur_down_out_view,
                            op0=nl.multiply,
                            operand0=cur_affinity,
                            op1=nl.add,
                            operand1=output_temp[:, i_t, :],
                        )

        # If we shard on K, reduce result between two NCs
        if shard_on_K and (num_shards > 1):
            output_temp_recv = nl.ndarray(output_temp.shape, dtype=output_temp.dtype, buffer=nl.sbuf)
            nisa.sendrecv(
                src=output_temp,
                dst=output_temp_recv,
                send_to_rank=(1 - shard_id),
                recv_from_rank=(1 - shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=output_temp, data1=output_temp, data2=output_temp_recv, op=nl.add)

        # Now we have all tokens processed with all K experts, we shard on T to transpose and save (with optionally adding final down proj bias).
        T_sharded, T_has_remainder = dims.T // num_shards, (dims.T % num_shards > 0)

        # Compute expert_affinities scaled down projection bias with the pseudo code below:
        # (1) weighted_bias[T, H] = expert_affinities[T, E] @ down_bias[E, H]
        # (2) output_temp[T, H] += weighted_bias[T, H]
        if params.bias_params.down_proj_bias_tensor != None:
            kernel_assert(
                (dims.E <= _pmax),
                "MXFP4 down projection with LHS/RHS swapped only supports E <= 128 when down projection bias exists",
            )
            down_bias_sb = nl.ndarray(
                (dims.E, dims.H), dtype=params.bias_params.down_proj_bias_tensor.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=down_bias_sb, src=params.bias_params.down_proj_bias_tensor)

            expert_affinity_psum = nl.ndarray(
                (dims.E, dims.T), dtype=params.expert_params.expert_affinities.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(dst=expert_affinity_psum, data=expert_affinities_sb[0 : dims.T, 0 : dims.E])

            # Down cast the transposed expert affinities from FP32 to the same dtype as the down proj bias
            expert_affinity_tp = nl.ndarray(
                (dims.E, dims.T), dtype=params.bias_params.down_proj_bias_tensor.dtype, buffer=nl.sbuf
            )
            nisa.activation(dst=expert_affinity_tp, op=nl.copy, data=expert_affinity_psum)

            # perform matmul with expert_affinity_tp to be LHS and down_bias_sb to be RHS
            # result has the layout of [dims.H0, dims.T, dims.H1_shard] layout to match output_temp
            scaled_bias_psum = nl.ndarray((dims.H0, dims.H1_shard, dims.T), dtype=nl.float32, buffer=nl.psum)
            for i_h1 in range(dims.H1_shard):
                nisa.nc_matmul(
                    dst=scaled_bias_psum[0 : dims.H0, i_h1, 0 : dims.T],
                    stationary=down_bias_sb[0 : dims.E, i_h1 * dims.H0 : (i_h1 + 1) * dims.H0],
                    moving=expert_affinity_tp[0 : dims.E, 0 : dims.T],
                )

            for i_t in range(dims.T):
                nisa.tensor_tensor(
                    dst=output_temp[:, i_t, :],
                    data1=output_temp[:, i_t, :],
                    data2=scaled_bias_psum[:, :, i_t],
                    op=nl.add,
                )

        # Transpose output since down proj is lhs/rhs swapped and producing HT layout
        output_temp_tp = nl.ndarray((dims.H1_shard, dims.T, dims.H0), dtype=output_temp.dtype, buffer=nl.sbuf)
        for i_T_sharded in range(T_sharded):
            i_t = T_sharded * shard_id + i_T_sharded
            out_tp_psum = nl.ndarray((dims.H1_shard, dims.H0), dtype=output_temp.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=out_tp_psum, data=output_temp[:, i_t, :])
            nisa.activation(dst=output_temp_tp[:, i_t, :], op=nl.copy, data=out_tp_psum)

        if T_has_remainder and (shard_id == 0):
            i_t = T_sharded * num_shards
            out_tp_psum_r = nl.ndarray((dims.H1_shard, dims.H0), dtype=output_temp.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=out_tp_psum_r, data=output_temp[:, i_t, :])
            nisa.activation(dst=output_temp_tp[:, i_t, :], op=nl.copy, data=out_tp_psum_r)

        # Save output for each token.
        for i_T_sharded in range(T_sharded):
            i_t = T_sharded * shard_id + i_T_sharded
            # output_temp has shape [H1, T, H0] with H0 being the contig dim
            nisa.dma_copy(
                dst=output.ap(pattern=[[dims.H0, dims.H1], [1, dims.H0]], offset=i_t * dims.H),
                src=output_temp_tp[:, i_t, :],
            )

        if T_has_remainder and (shard_id == 0):
            i_t = T_sharded * num_shards
            # output_temp has shape [H1, T, H0] with H0 being the contig dim
            nisa.dma_copy(
                dst=output.ap(pattern=[[dims.H0, dims.H1], [1, dims.H0]], offset=i_t * dims.H),
                src=output_temp_tp[:, i_t, :],
            )

    auto_sbm.close_scope()
    return output
