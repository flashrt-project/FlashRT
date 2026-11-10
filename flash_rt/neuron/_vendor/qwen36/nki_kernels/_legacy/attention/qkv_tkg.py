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
QKV Projection TKG Kernel

This kernel implements the fused QKV (Query-Key-Value) projection operation with
optional residual addition and normalization (RMSNorm/LayerNorm), commonly used
before the attention block in transformer models. The kernel is specifically optimized
for Token Generation (TKG, also known as Decoding) scenarios where batch_size * seqlen
is small.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded across cores.
Multiple output layouts (BSD, NBSd) are supported to match downstream kernel requirements.

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import nki.isa as nisa
import nki.language as nl

from nkilib.core.subkernels.layernorm_tkg import (
    SHARDING_THRESHOLD as layernorm_sharding_threshold,
)
from nkilib.core.subkernels.layernorm_tkg import layernorm_tkg as _layernorm_tkg
from nkilib.core.subkernels.rmsnorm_tkg import SHARDING_THRESHOLD as rmsnorm_sharding_threshold
from nkilib.core.subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from nkilib.core.utils.allocator import (
    SbufManager,
    create_auto_alloc_manager,
    sizeinbytes,
)
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.tiled_range import TiledRange, TiledRangeIterator
from nkilib.core.qkv.qkv_tkg_mx_impl import _qkv_tkg_mx_impl

P_MAX = 128
F_MAX = 512
NUM_PSUM_BANKS = 8

I_TILE_SIZE = F_MAX
I_BLOCK_SIZE = NUM_PSUM_BANKS * I_TILE_SIZE

# Heuristic tile size for H dimension weight loading
H_BLOCK_SIZE = 2048
NUM_TILES_PER_H_BLOCK = H_BLOCK_SIZE // P_MAX

_DGE_MODE_NONE = 3

logger = get_logger("qkv_tkg")


def qkv_tkg(
    hidden: nl.ndarray,
    qkv_w: nl.ndarray,
    norm_w: Optional[nl.ndarray] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attn_prev: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    is_h_dim_4h_transposed: bool = False,
    qkv_w_scale: Optional[nl.ndarray] = None,
    qkv_in_scale: Optional[nl.ndarray] = None,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[nl.ndarray] = None,
    norm_bias: Optional[nl.ndarray] = None,
    hidden_actual: Optional[int] = None,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray | Tuple[nl.ndarray, nl.ndarray]:
    """
    QKV Projection Kernel for Token Generation

    This kernel computes the fused QKV projection operation:
        hidden' = norm(hidden + attn_prev + mlp_prev)  # optional fused add and norm
        output = hidden' @ qkv_w + qkv_bias
    typically used before the attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. Using this kernel with B*S > 128 may result in
    degraded performance - use the CTE variant for large sequence lengths.

    The kernel supports optional fused residual addition and normalization (RMSNorm/LayerNorm)
    to reduce HBM traffic and improve performance.

    Data Types:
        This kernel supports nl.float32, nl.float16, and nl.bfloat16 data types.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: Fused QKV output dimension ((Q + K + V) * N * D)
        N: Number of heads (for NBSd output layout)
        D: Head dimension size (for NBSd output layout)

    Args:
        hidden (nl.ndarray):
            Input hidden states tensor in HBM or SBUF.
            Shape:
                [B, S, H]         when in HBM
                [H0=128, BxS, H1] when in SBUF
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        qkv_w (nl.ndarray):
            QKV projection weight tensor in HBM.
            Shape:    [H, I]
            Dtype:
                - QuantizationType.NONE: nl.float32, nl.float16, or nl.bfloat16
                - QuantizationType.STATIC: nl.float8_e4m3
                - QuantizationType.ROW: nl.float8_e4m3
        norm_w (nl.ndarray, optional):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        fused_add (bool):
            Enable fused residual addition (hidden + attn_prev + mlp_prev). Default: False.
        mlp_prev (nl.ndarray, optional):
            Previous MLP residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        attn_prev (nl.ndarray, optional):
            Previous attention residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        d_head (int, optional):
            Head dimension size D. Required for static quantization and NBSd and NBdS output layouts.
        num_q_heads : Optional[int], default=None
            Number of query heads (required for FP8 quantization)
        num_kv_heads : Optional[int], default=None
            Number of key/value heads (required for FP8 quantization)
        output_layout (QKVOutputLayout):
            Output tensor layout format. BSD: [B, S, I] or NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float):
            Epsilon value to maintain numerical stability in normalization. Default: 1e-6.
        norm_type (NormType):
            Type of normalization to apply (NO_NORM, RMS_NORM, or LAYER_NORM). Default: NormType.RMS_NORM.
        quantization_type (QuantizationType):
            Type of quantization to apply (NONE, ROW, STATIC). Default: QuantizationType.NONE.
        is_h_dim_4h_transposed: bool, default=False
            Whether the H-dim (in input and gamma) has been pre-transposed by 4 (only applicable with MX Quantization).
            If is_h_dim_4h_transposed = False,
                * input has typical shape [B, S, H], viewed as [B, S, H//512, 128_H, 4_H].
            If is_h_dim_4h_transposed = True,
                * input has shape [B, S, H] but is pre-shuffled from
                  [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].
                * IMPORTANT: H-dim in both input and gamma weights (for RMSNorm) must be pre-shuffled.
                    * For input, this is achieved by offline pre-shuffling weights of upstream projection (in real model).
                    * For gamma, this is achieved by offline pre-shuffling of gamma tensor.
                Purpose: More efficent for obtaining the required swizzled layout for quantize_mx instruction.
        qkv_w_scale (nl.ndarray, optional):
            Weight dequantization scale tensor in HBM.
                - QuantizationType.STATIC: [1, 3] or [P_MAX, 3] pre-broadcasted
                - QuantizationType.ROW: [1, I] or [P_MAX, I] pre-broadcasted
                - QuantizationType.MX: [H//32, I]
            Dtype: nl.float32
        qkv_in_scale (nl.ndarray, optional):
            Input quantization and dequantization scale tensor in HBM. Only required for STATIC quantization.
            Shape:    [1, 1] or [P_MAX, 1] pre-broadcasted
            Dtype: nl.float32
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
        qkv_bias (nl.ndarray, optional):
            Bias tensor in HBM for QKV projection.
            Shape:    [1, I]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        norm_bias (nl.ndarray, optional):
            LayerNorm beta parameter tensor in HBM. Required when norm_type is LAYER_NORM.
            Shape:    [1, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        hidden_actual (int, optional):
            Actual hidden dimension for padded input tensors. If specified, normalization
            uses this value instead of H for mean calculation.
        sbm (SbufManager, optional):
            Instance of SbufManager responsible for handling SBUF allocation.
            If None, auto-allocation manager is created.

    Returns:
        output (nl.ndarray | Tuple[nl.ndarray, nl.ndarray]):
            QKV projection output tensor. The tensor can reside in either SBUF or HBM.
            Shape:    [B, S, I] for BSD layout, [N, B, S, D] for NBSd layout.
            When fused_add is True, returns tuple (output, fused_hidden) where
            fused_hidden is the result of the fused residual addition.

    Notes:
        - H must be divisible by 128 (nl.tile_size.pmax).
        - H1 (H//128) must be divisible by number of shards for multi-core execution.

    Pseudocode:
        # Optional fused residual add
        if fused_add:
            hidden = hidden + attn_prev + mlp_prev

        # Optional normalization
        if norm_type != NO_NORM:
            hidden = norm(hidden, norm_w, norm_bias, eps)

        # QKV projection with tiled matmul
        output = zeros(B, S, I)
        for i_block in range(0, I, I_BLOCK_SIZE):
            for h_block in range(0, H_shard, H_BLOCK_SIZE):
                output[:, i_block:i_block+I_BLOCK_SIZE] += hidden[:, h_block:h_block+H_BLOCK_SIZE] @ qkv_w[h_block:h_block+H_BLOCK_SIZE, i_block:i_block+I_BLOCK_SIZE]
            output[:, i_block:i_block+I_BLOCK_SIZE] += qkv_bias[i_block:i_block+I_BLOCK_SIZE]
    """

    if quantization_type == QuantizationType.MX:
        return _qkv_tkg_mx_impl(
            hidden=hidden,
            weights_qtz_hbm=qkv_w,
            norm_w=norm_w,
            fused_add=fused_add,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            quantization_type=quantization_type,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            weight_scales_hbm=qkv_w_scale,
            output_in_sbuf=output_in_sbuf,
            qkv_bias=qkv_bias,
            norm_bias=norm_bias,
            hidden_actual=hidden_actual,
            sbm=sbm,
        )

    if not sbm:
        sbm = create_auto_alloc_manager()

    sbm.open_scope(name="qkv_tkg")

    # Validate inputs and create config
    cfg = _validate_and_create_config(
        hidden=hidden,
        qkv_w=qkv_w,
        qkv_bias=qkv_bias,
        norm_w=norm_w,
        norm_bias=norm_bias,
        norm_type=norm_type,
        output_layout=output_layout,
        output_in_sbuf=output_in_sbuf,
        fused_add=fused_add,
        attn_prev=attn_prev,
        mlp_prev=mlp_prev,
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )

    io_dtype = hidden.dtype
    quant_dtype = qkv_w.dtype if quantization_type != QuantizationType.NONE else None
    if hidden_actual == None:
        hidden_actual = cfg.H

    # Perform optional fused residual add in HBM
    if fused_add:
        hidden = _fused_residual_add_hbm2hbm(
            hidden_hbm=hidden, attn_prev_hbm=attn_prev, mlp_prev_hbm=mlp_prev, cfg=cfg, norm_type=norm_type
        )

    # Load quantization scales
    if quantization_type == QuantizationType.STATIC:
        w_scale_tile = sbm.alloc_stack(shape=(P_MAX, 3), dtype=qkv_w_scale.dtype, name="qkv_w_scale_sb", buffer=nl.sbuf)
        if qkv_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=w_scale_tile[0, :], src=qkv_w_scale[0, :], dge_mode=_DGE_MODE_NONE)
            stream_shuffle_broadcast(w_scale_tile, w_scale_tile)
        else:
            nisa.dma_copy(dst=w_scale_tile, src=qkv_w_scale, dge_mode=_DGE_MODE_NONE)

        in_scale_tile = sbm.alloc_heap(shape=(P_MAX, 1), dtype=qkv_in_scale.dtype)
        if qkv_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_tile[0, :], src=qkv_in_scale[0, :], dge_mode=_DGE_MODE_NONE)
            stream_shuffle_broadcast(in_scale_tile, in_scale_tile)
        else:
            nisa.dma_copy(dst=in_scale_tile, src=qkv_in_scale, dge_mode=_DGE_MODE_NONE)
        nisa.activation(dst=w_scale_tile, op=nl.copy, data=w_scale_tile, scale=in_scale_tile)
        quant_config = StaticQuantConfig(combined_scale_sb=w_scale_tile, in_scale_tile=in_scale_tile)
    elif quantization_type == QuantizationType.ROW:
        quant_config = RowQuantConfig(weight_scale_hbm=qkv_w_scale)
    else:
        quant_config = None

    # Perform optional fused norm and load
    hidden_sb = _fused_norm_and_load(
        hidden=hidden,
        norm_type=norm_type,
        norm_w=norm_w,
        norm_bias=norm_bias,
        eps=eps,
        hidden_actual=hidden_actual,
        cfg=cfg,
        sbm=sbm,
        quantization_type=quantization_type,
        quant_config=quant_config,
        quant_dtype=quant_dtype,
    )

    # Shard on H for qkv_w: (H0, H1_sharded, I)
    qkv_w_hbm = (
        TensorView(qkv_w)
        .reshape_dim(dim=0, shape=(cfg.num_shards, cfg.H0, cfg.H1_shard))
        .select(dim=0, index=cfg.shard_id)
    )

    # Dispatch to appropriate projection path based output buffer
    if output_in_sbuf:
        output = _qkv_projection_sbuf_output(
            hidden_sb=hidden_sb,
            qkv_w=qkv_w_hbm,
            qkv_bias=qkv_bias,
            cfg=cfg,
            sbm=sbm,
            io_dtype=io_dtype,
            quantization_type=quantization_type,
            quant_config=quant_config,
        )
    else:
        output = _qkv_projection_hbm_output(
            hidden_sb=hidden_sb,
            qkv_w=qkv_w_hbm,
            qkv_bias=qkv_bias,
            cfg=cfg,
            output_layout=output_layout,
            sbm=sbm,
            io_dtype=io_dtype,
            quantization_type=quantization_type,
            quant_config=quant_config,
        )

    sbm.close_scope()

    if fused_add:
        return output, hidden
    else:
        return output


@dataclass
class StaticQuantConfig(nl.NKIObject):
    """Configuration for STATIC quantization.

    Holds pre-computed combined scale (weight_scale * input_scale) in SBUF.
    """

    combined_scale_sb: nl.ndarray  # [P_MAX, 3], pre-computed in SBUF
    in_scale_tile: nl.ndarray  # [P_MAX, 1], input scale in SBUF (for norm quantization)


@dataclass
class RowQuantConfig(nl.NKIObject):
    """Configuration for ROW quantization.

    Holds weight scale tensor in HBM to be loaded inside impl functions.
    """

    weight_scale_hbm: nl.ndarray  # [1, I] or [P_MAX, I], in HBM


@dataclass
class QkvTkgConfig(nl.NKIObject):
    """Configuration for QKV TKG kernel containing input dimensions, sharding, and tiling parameters."""

    # Input dimensions
    B: Optional[int]
    S: Optional[int]
    BxS: int
    H: int
    I: int
    H0: int
    H1: int
    d_head: int
    n_q_heads: int
    n_kv_heads: int
    # Sharding
    num_shards: int
    shard_id: int
    H_shard: int
    H1_shard: int
    H1_offset: int
    # Array tiling
    array_tiling_dim: int
    array_tiling_factor: int
    remainder_array_tiling_dim: int
    remainder_array_tiling_factor: int
    array_tiled_H1: int
    remainder_array_tiled_H1: int


def _validate_and_create_config(
    hidden: nl.ndarray,
    qkv_w: nl.ndarray,
    qkv_bias: Optional[nl.ndarray],
    norm_w: Optional[nl.ndarray],
    norm_bias: Optional[nl.ndarray],
    norm_type: NormType,
    output_layout: QKVOutputLayout,
    output_in_sbuf: bool,
    fused_add: bool,
    attn_prev: Optional[nl.ndarray],
    mlp_prev: Optional[nl.ndarray],
    quantization_type: QuantizationType,
    qkv_w_scale: Optional[nl.ndarray],
    qkv_in_scale: Optional[nl.ndarray],
    d_head: Optional[int],
    num_q_heads: Optional[int],
    num_kv_heads: Optional[int],
) -> QkvTkgConfig:
    """
    Validate inputs and create kernel configuration.

    Performs comprehensive validation of input tensor shapes, quantization settings,
    and layout constraints. Computes derived tiling parameters.

    Args:
        hidden: Input hidden states tensor
        qkv_w: QKV projection weight tensor
        qkv_bias: Optional bias tensor
        norm_w: Optional normalization weight tensor
        norm_bias: Optional normalization bias tensor (LayerNorm only)
        norm_type: Type of normalization
        output_layout: Output tensor layout
        output_in_sbuf: Whether output stays in SBUF
        fused_add: Whether to fuse residual addition
        attn_prev: Previous attention residual (required if fused_add)
        mlp_prev: Previous MLP residual (required if fused_add)
        quantization_type: Quantization mode
        qkv_w_scale: Weight scale for quantization
        qkv_in_scale: Input scale for quantization
        d_head: Head dimension
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        QkvTkgConfig with validated configuration and computed tiling parameters
    """
    # Get sharding info
    _, num_shards, shard_id = get_verified_program_sharding_info("qkv_tkg", (0, 1))

    input_in_sbuf = hidden.buffer == nl.sbuf
    if input_in_sbuf:
        kernel_assert(not fused_add, "fused residual add is only supported when input is in hbm")
        kernel_assert(output_in_sbuf, "sb2hbm is not yet supported for qkv_tkg")

    # Get input shapes
    if input_in_sbuf:
        H0, BxS, H1 = hidden.shape
        kernel_assert(H0 == nl.tile_size.pmax, f"invalid input shape - H0 (first dimension) must equal 128, got {H0}")
        H = H0 * H1
        B = None
        S = None
    else:
        B, S, H = hidden.shape
        H0 = nl.tile_size.pmax
        H1 = H // H0
        BxS = B * S
    _H, I = qkv_w.shape

    # Validate kernel inputs
    kernel_assert(
        H % nl.tile_size.pmax == 0,
        f"H must be divisible by {nl.tile_size.pmax}, got {H} % {nl.tile_size.pmax}={H % nl.tile_size.pmax}",
    )
    kernel_assert(
        _H == H,
        f"Weight tensor reduction dimension must match hidden dimension, got weight H={_H}, hidden H={H}",
    )
    if qkv_bias != None:
        kernel_assert(
            qkv_bias.shape == (1, I),
            f"Bias shape must be [1, I], got {qkv_bias.shape}, expected {(1, I)}",
        )
    kernel_assert(
        norm_type == NormType.LAYER_NORM or not norm_bias,
        f"norm_bias only supported for LAYER_NORM, got norm_type={norm_type} with norm_bias={norm_bias != None}",
    )
    kernel_assert(
        output_layout != QKVOutputLayout.NBdS,
        f"NBdS output layout is not supported in QKV TKG, got output_layout={output_layout}",
    )
    kernel_assert(
        H1 % num_shards == 0,
        f"H1 must be divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, H1 % num_shards={H1 % num_shards}",
    )
    kernel_assert(
        not fused_add or (attn_prev != None and mlp_prev != None),
        f"attn_prev and mlp_prev must be provided when fused_add=True, got fused_add={fused_add}, "
        f"attn_prev provided={attn_prev != None}, mlp_prev provided={mlp_prev != None}",
    )

    # Validate quantization inputs
    kernel_assert(
        quantization_type == QuantizationType.NONE
        or quantization_type == QuantizationType.STATIC
        or quantization_type == QuantizationType.ROW,
        f"Only QuantizationType.NONE, QuantizationType.STATIC, and QuantizationType.ROW are supported, got {quantization_type}",
    )
    if quantization_type == QuantizationType.STATIC:
        kernel_assert(qkv_w_scale != None, "qkv_w_scales must be provided when quantization_type is STATIC")
        kernel_assert(qkv_in_scale != None, "qkv_in_scales must be provided when quantization_type is STATIC")
        if qkv_w_scale.shape[0] == 1:
            logger.warn(
                f"For static quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, 3) for better performance"
            )
        else:
            kernel_assert(
                qkv_w_scale.shape == (nl.tile_size.pmax, 3),
                f"Incorrect shape for qkv weight scale for static per tensor quantization, expected ({nl.tile_size.pmax}, 3), got {qkv_w_scale.shape}",
            )
        if qkv_in_scale.shape[0] == 1:
            logger.warn(
                f"For static quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, 3) for better performance"
            )
        else:
            kernel_assert(
                qkv_in_scale.shape == (nl.tile_size.pmax, 1),
                f"Incorrect shape for qkv input scale for static per tensor quantization, expected ({nl.tile_size.pmax}, 1), got {qkv_in_scale.shape}",
            )
        kernel_assert(
            d_head != None and num_kv_heads != None and num_q_heads != None,
            f"d_head, num_q_heads, num_kv_heads must be provided when quant_type is STATIC, got d_head={d_head}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}",
        )
    elif quantization_type == QuantizationType.ROW:
        kernel_assert(qkv_w_scale != None, "qkv_w_scale must be provided when quantization_type is ROW")
        if qkv_w_scale.shape[0] == 1:
            logger.warn(
                f"For row quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, {I}) for better performance"
            )
        else:
            kernel_assert(
                qkv_w_scale.shape == (nl.tile_size.pmax, I),
                f"Incorrect shape for qkv weight scale for row quantization, expected ({nl.tile_size.pmax}, {I}), got {qkv_w_scale.shape}",
            )

    # Explicit Enum checks
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM and norm_type != NormType.LAYER_NORM:
        kernel_assert(False, f"NormType {norm_type} is not supported")
    if output_layout != QKVOutputLayout.BSD and output_layout != QKVOutputLayout.NBSd:
        kernel_assert(False, f"OutputLayout {output_layout} is not supported")

    # Validate HBM output requirements
    if not output_in_sbuf:
        kernel_assert(
            B != None and S != None,
            "B and S must be present when output is in HBM (input must be in HBM)",
        )
        if output_layout == QKVOutputLayout.NBSd:
            kernel_assert(
                d_head != None,
                f"d_head must be specified for NBSd output layout, got output_layout={output_layout}, d_head={d_head}",
            )
            kernel_assert(
                I % d_head == 0,
                f"I must be divisible by d_head for NBSd output layout, got I={I}, d_head={d_head}, I % d_head = {I % d_head}",
            )

    # Compute sharding
    H1_sharded = H1 // num_shards
    H1_remainder = H1 % num_shards

    kernel_assert(
        H1_remainder == 0,
        f"H1 must be evenly divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, "
        f"H1 % num_shards = {H1_remainder}",
    )
    if H1_remainder == 0:
        H1_shard = H1_sharded
        H1_offset = shard_id * H1_sharded
    else:
        if shard_id < H1_remainder:
            H1_shard = H1_sharded + 1
            H1_offset = shard_id * (H1_sharded + 1)
        else:
            H1_shard = H1_sharded
            H1_offset = H1_remainder * (H1_sharded + 1) + (shard_id - H1_remainder) * H1_sharded

    H_shard = H1_shard * H0

    # Intermediate dimension tiling
    remainder_H_block = H_shard % H_BLOCK_SIZE
    num_128_tiles_per_remainder_H_block = remainder_H_block // 128

    # Choose Array tiling strategy depending on BxS
    if BxS <= 32:  # do 4x 128P*32F PE array tiling
        array_tiling_dim = 32
    elif BxS <= 64:  # do 2x 128P*64F PE array tiling
        array_tiling_dim = 64
    else:
        array_tiling_dim = 128

    # Adjust hardware-specific logic for column tiling on NeuronCore-v2
    if nisa.get_nc_version() == nisa.nc_version.gen2:
        # Both the row and column sizes in tile_size cannot be 32
        array_tiling_dim = 64

    array_tiling_factor = 128 // array_tiling_dim
    array_tiled_H1 = NUM_TILES_PER_H_BLOCK // array_tiling_factor

    # If H is not multiple of H_BLOCK_SIZE and num_128_tiles_per_remainder_H_block is not multiple of array_tiling_factor,
    # kernel won't use array tiling
    remainder_array_tiling_dim = array_tiling_dim
    remainder_array_tiling_factor = array_tiling_factor
    if num_128_tiles_per_remainder_H_block % array_tiling_factor != 0:
        remainder_array_tiling_dim = 128
        remainder_array_tiling_factor = 1
    remainder_array_tiled_H1 = num_128_tiles_per_remainder_H_block // remainder_array_tiling_factor

    return QkvTkgConfig(
        B=B,
        S=S,
        BxS=BxS,
        H=H,
        I=I,
        H0=H0,
        H1=H1,
        d_head=d_head,
        n_q_heads=num_q_heads,
        n_kv_heads=num_kv_heads,
        num_shards=num_shards,
        shard_id=shard_id,
        H_shard=H_shard,
        H1_shard=H1_shard,
        H1_offset=H1_offset,
        array_tiling_dim=array_tiling_dim,
        array_tiling_factor=array_tiling_factor,
        remainder_array_tiling_dim=remainder_array_tiling_dim,
        remainder_array_tiling_factor=remainder_array_tiling_factor,
        array_tiled_H1=array_tiled_H1,
        remainder_array_tiled_H1=remainder_array_tiled_H1,
    )


def _fused_residual_add_hbm2hbm(
    hidden_hbm: nl.ndarray,
    attn_prev_hbm: nl.ndarray,
    mlp_prev_hbm: nl.ndarray,
    cfg: QkvTkgConfig,
    norm_type: NormType,
) -> nl.ndarray:
    """
    Perform fused residual addition in HBM: hidden + attn_prev + mlp_prev.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        attn_prev_hbm: Previous attention residual in HBM. Shape: (B, S, H)
        mlp_prev_hbm: Previous MLP residual in HBM. Shape: (B, S, H)
        cfg: QKV TKG config
        norm_type: Type of normalization (affects sharding strategy)

    Returns:
        Fused hidden states in HBM. Shape: (B, S, H)
    """

    sharding_threshold = rmsnorm_sharding_threshold if norm_type == NormType.RMS_NORM else layernorm_sharding_threshold

    B, S, BxS, H = cfg.B, cfg.S, cfg.BxS, cfg.H
    num_shards, shard_id = cfg.num_shards, cfg.shard_id

    # Allocate Fused hidden hbm tensor
    if num_shards > 1 and (norm_type == NormType.NO_NORM or BxS > sharding_threshold and norm_type != NormType.NO_NORM):
        fused_hidden = nl.ndarray(
            (BxS, H),
            dtype=hidden_hbm.dtype,
            buffer=nl.shared_hbm,
            name="fused_hidden_shared_hbm",
        )
    else:
        fused_hidden = nl.ndarray((BxS, H), dtype=hidden_hbm.dtype, buffer=nl.shared_hbm, name="fused_hidden_hbm")

    # To prevent non-determinism, a different access pattern is needed for offloaded FMA instruction
    if num_shards > 1 and norm_type == NormType.NO_NORM:
        hidden_hbm = hidden_hbm.reshape((BxS, 2, H // 2))
        attn_prev_hbm = attn_prev_hbm.reshape((BxS, 2, H // 2))
        mlp_prev_hbm = mlp_prev_hbm.reshape((BxS, 2, H // 2))
        fused_hidden = fused_hidden.reshape((BxS, 2, H // 2))
        nisa.dma_compute(
            fused_hidden[0:BxS, shard_id, 0 : (H // 2)],
            (
                hidden_hbm[0:BxS, shard_id, 0 : (H // 2)],
                attn_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
                mlp_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    elif norm_type != NormType.NO_NORM and num_shards > 1 and BxS > sharding_threshold:
        kernel_assert(
            BxS % 2 == 0,
            f"expected BxS divisible by 2 when BxS={BxS} > {sharding_threshold}",
        )
        hidden_hbm = hidden_hbm.reshape((2, BxS // 2, H))
        attn_prev_hbm = attn_prev_hbm.reshape((2, BxS // 2, H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((2, BxS // 2, H))
        fused_hidden = fused_hidden.reshape((2, BxS // 2, H))
        nisa.dma_compute(
            fused_hidden[shard_id, 0 : (BxS // 2), 0:H],
            (
                hidden_hbm[shard_id, 0 : (BxS // 2), 0:H],
                attn_prev_hbm[shard_id, 0 : (BxS // 2), 0:H],
                mlp_prev_hbm[shard_id, 0 : (BxS // 2), 0:H],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    else:
        hidden_hbm = hidden_hbm.reshape((1, BxS * H))
        attn_prev_hbm = attn_prev_hbm.reshape((1, BxS * H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((1, BxS * H))
        fused_hidden = fused_hidden.reshape((1, BxS * H))
        nisa.dma_compute(
            fused_hidden,
            (hidden_hbm, attn_prev_hbm, mlp_prev_hbm),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )

    return fused_hidden.reshape((B, S, H))


def _fused_norm_and_load(
    hidden: nl.ndarray,
    norm_type: NormType,
    norm_w: Optional[nl.ndarray],
    norm_bias: Optional[nl.ndarray],
    eps: float,
    hidden_actual: int,
    cfg: QkvTkgConfig,
    sbm: SbufManager,
    quantization_type: QuantizationType,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    quant_dtype=None,
) -> TensorView:
    """
    Perform fused normalization and load from HBM to SBUF when input is in HBM.

    Handles three normalization paths:
    - NO_NORM:
        Input in HBM:  Direct load with shape (H0, BxS, H1_sharded)
        Input in SBUF: Return as-is with shape (H0, BxS, H1)
    - RMS_NORM: RMSNorm (with HBM -> SBUF load if needed) with shape (H0, BxS, H1)
    - LAYER_NORM: LayerNorm (with HBM -> SBUF load if needed) with shape (H0, BxS, H1)

    Args:
        hidden: Input hidden states in HBM or SBUF.
            Shape:
                (B, S, H)         when in HBM
                (H0=128, BxS, H1) when in SBUF
        norm_type: Type of normalization (NO_NORM, RMS_NORM, or LAYER_NORM)
        norm_w: Normalization weights (required for RMS/LAYER_NORM). Shape: (1, H)
        norm_bias: LayerNorm bias (required for LAYER_NORM). Shape: (1, H)
        eps: Epsilon for numerical stability
        hidden_actual: Actual hidden dimension for padded tensors
        cfg: QKV TKG config
        sbm: SbufManager object for SBUF allocation
        quantization_type: Type of quantization
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        TensorView wrapping hidden states in SBUF:
          Shape: (H0, BxS, H1_sharded)
    """

    BxS, H0, H1 = cfg.BxS, cfg.H0, cfg.H1
    num_shards, shard_id = cfg.num_shards, cfg.shard_id
    H1_sharded = cfg.H1_shard

    hidden_in_sbuf = hidden.buffer == nl.sbuf

    hidden_sb = None
    hidden_sb_quantized = None

    hidden_shape = (H0, BxS, H1)
    hidden_sharded_shape = (H0, BxS, H1_sharded)

    if quantization_type != QuantizationType.NONE:
        hidden_sb_quantized = sbm.alloc_stack(hidden_sharded_shape, dtype=quant_dtype, buffer=nl.sbuf)

    if norm_type == NormType.NO_NORM:
        if hidden_in_sbuf:
            hidden_sb = TensorView(hidden).slice(dim=2, start=shard_id * H1_sharded, end=(shard_id + 1) * H1_sharded)
        else:
            if quantization_type != QuantizationType.NONE:
                hidden_sb = sbm.alloc_heap(hidden_sharded_shape, dtype=hidden.dtype, buffer=nl.sbuf)
            else:
                hidden_sb = sbm.alloc_stack(hidden_sharded_shape, dtype=hidden.dtype, buffer=nl.sbuf)
            # Perform direct input load with no norm
            # hidden_sb: (H0, BxS, H1_sharded)
            hidden_sb = _input_load(hidden, hidden_sb, cfg, sbm)
            hidden_sb = TensorView(hidden_sb)
    elif norm_type == NormType.RMS_NORM or norm_type == NormType.LAYER_NORM:
        if hidden_in_sbuf:
            hidden_sb = hidden
        elif quantization_type != QuantizationType.NONE:
            hidden_sb = sbm.alloc_heap(hidden_shape, dtype=hidden.dtype, buffer=nl.sbuf)
        else:
            hidden_sb = sbm.alloc_stack(hidden_shape, dtype=hidden.dtype, buffer=nl.sbuf)
        # Perform norm with load
        if norm_type == NormType.RMS_NORM:
            # Perform rmsnorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _rmsnorm_tkg(
                input=hidden,
                gamma=norm_w,
                output=hidden_sb,
                eps=eps,
                hidden_actual=hidden_actual,
                sbm=sbm,
            )
        elif norm_type == NormType.LAYER_NORM:
            # Perform layernorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _layernorm_tkg(
                input=hidden,
                gamma=norm_w,
                beta=norm_bias,
                output=hidden_sb,
                eps=eps,
                sbm=sbm,
            )
        hidden_sb = TensorView(hidden_sb).slice(dim=2, start=shard_id * H1_sharded, end=(shard_id + 1) * H1_sharded)

    # optionally quantize the inputs
    if quantization_type == QuantizationType.STATIC:
        in_scale_tile = quant_config.in_scale_tile
        nisa.reciprocal(dst=in_scale_tile, data=in_scale_tile)
        nisa.activation(dst=hidden_sb.get_view(), op=nl.copy, data=hidden_sb.get_view(), scale=in_scale_tile[:H0, :])
        max_pos_val = get_max_positive_value_for_dtype(quant_dtype)
        nisa.tensor_scalar(
            dst=hidden_sb_quantized,
            data=hidden_sb.get_view(),
            op0=nl.minimum,
            operand0=max_pos_val,
            op1=nl.maximum,
            operand1=-max_pos_val,
        )
        if not hidden_in_sbuf:
            sbm.pop_heap()  # hidden_sb
        sbm.pop_heap()  # in_scale_tile
        hidden_sb = TensorView(hidden_sb_quantized)

    return hidden_sb


def _input_load(
    hidden_hbm: nl.ndarray,
    hidden_sb: nl.ndarray,
    cfg: QkvTkgConfig,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Load hidden states from HBM to SBUF without normalization.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        hidden_sb: Hidden states in SBUF to be loaded to. Shape: (H0, BxS, H1_sharded)
        cfg: QKV TKG config
        sbm: SbufManager for SBUF allocation

    Returns:
        Hidden states in SBUF. Shape: (H0, BxS, H1_sharded)
    """
    BxS, H0, H1 = cfg.BxS, cfg.H0, cfg.H1
    num_shards, shard_id = cfg.num_shards, cfg.shard_id
    H1_sharded = cfg.H1_shard

    if num_shards > 1:
        # Reshape (B, S, H) to (BxS, num_shards, H0, H1_sharded), select shard, permute to (H0, BxS, H1_sharded)
        hidden_hbm = hidden_hbm.reshape((BxS, num_shards, H0, H1_sharded))
        hidden_hbm = TensorView(hidden_hbm).select(dim=1, index=shard_id).permute((1, 0, 2))
        nisa.dma_copy(hidden_sb, hidden_hbm.get_view())
    else:
        # Reshape (B, S, H) to (BxS, H0, H1), permute to (H0, BxS, H1)
        hidden_hbm = hidden_hbm.reshape((BxS, H0, H1))
        hidden_hbm = TensorView(hidden_hbm).permute((1, 0, 2))
        nisa.dma_copy(hidden_sb, hidden_hbm.get_view())

    return hidden_sb


def _initialize_qkv_out_with_bias(
    qkv_out_sb: nl.ndarray,
    qkv_bias: nl.ndarray,
    cfg: QkvTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
) -> None:
    """
    Initialize QKV output buffer with bias or zeros.

    When bias is provided and this is shard_id 0, loads the bias into SBUF
    and broadcasts it across all BxS partitions. Otherwise initializes to zeros.

    Args:
        qkv_out_sb: Output buffer to initialize. Shape: (BxS, I_block_size)
        qkv_bias: Optional bias tensor. Shape: (1, I_block_size) or None
        cfg: QKV TKG config
        sbm: SbufManager for SBUF allocation
        i_block_idx: I-block index for scope/buffer naming
    """
    if qkv_bias != None and cfg.shard_id == 0:
        scope_name = f"qkv_bias_init_block_{i_block_idx}" if i_block_idx != None else "qkv_bias_init_block"
        sbm.open_scope(name=scope_name)

        buffer_name = f"qkv_bias_sb_{i_block_idx}" if i_block_idx != None else "qkv_bias_sb"
        qkv_bias_sb = sbm.alloc_stack(
            qkv_bias.shape,
            dtype=qkv_bias.dtype,
            buffer=nl.sbuf,
            name=buffer_name,
        )
        nisa.dma_copy(qkv_bias_sb, qkv_bias)
        # Broadcast bias to all BxS partitions
        stream_shuffle_broadcast(qkv_bias_sb, qkv_out_sb)
        sbm.close_scope()
    else:
        nisa.memset(qkv_out_sb, value=0)


def _static_dequantize(
    output_sb: nl.ndarray,
    dequant_scale_sb: nl.ndarray,
    cfg: QkvTkgConfig,
    I_start: int = 0,
):
    pdim, I_size = output_sb.shape
    I_end = I_start + I_size

    # Q heads dequant
    q_start = 0
    q_end = min(I_end, cfg.d_head * cfg.n_q_heads) - I_start
    if q_start < q_end:
        nisa.tensor_scalar(
            dst=output_sb[:pdim, q_start:q_end],
            data=output_sb[:pdim, q_start:q_end],
            op0=nl.multiply,
            operand0=dequant_scale_sb[:pdim, 0:1],
            engine=nisa.vector_engine,
        )

    # K head dequant
    k_start = max(I_start, cfg.d_head * cfg.n_q_heads) - I_start
    k_end = min(I_end, cfg.d_head * (cfg.n_q_heads + cfg.n_kv_heads)) - I_start
    if k_start < k_end:
        nisa.activation(
            dst=output_sb[:pdim, k_start:k_end],
            data=output_sb[:pdim, k_start:k_end],
            op=nl.copy,
            scale=dequant_scale_sb[:pdim, 1:2],
        )
    # V head dequant
    v_start = max(I_start, cfg.d_head * (cfg.n_q_heads + cfg.n_kv_heads)) - I_start
    v_end = min(I_end, cfg.d_head * (cfg.n_q_heads + 2 * cfg.n_kv_heads)) - I_start
    if v_start < v_end:
        nisa.activation(
            dst=output_sb[:pdim, v_start:v_end],
            data=output_sb[:pdim, v_start:v_end],
            op=nl.copy,
            scale=dequant_scale_sb[:pdim, 2:3],
        )
    return output_sb


def _row_dequantize(
    output_sb: nl.ndarray,
    weight_scale: TensorView,
    cfg: QkvTkgConfig,
    sbm: SbufManager,
) -> None:
    """
    Apply row dequantization to QKV projection output.

    Loads weight scale from HBM, broadcasts if needed, and applies element-wise multiply.

    Args:
        output_sb: QKV projection output in SBUF. Shape: (BxS, i_size)
        weight_scale: Pre-sliced weight scale TensorView in HBM. Shape: (P_MAX, i_size) or (1, i_size)
        cfg: QKV TKG config
        sbm: SbufManager for allocation
    """
    BxS = cfg.BxS
    i_size = output_sb.shape[1]

    weight_scale_sb = sbm.alloc_heap((P_MAX, i_size), dtype=weight_scale.dtype, buffer=nl.sbuf)
    if weight_scale.shape[0] == 1:
        nisa.dma_copy(dst=weight_scale_sb[0:1, :], src=weight_scale.get_view())
        stream_shuffle_broadcast(weight_scale_sb, weight_scale_sb)
    else:
        nisa.dma_copy(dst=weight_scale_sb, src=weight_scale.get_view())
    nisa.tensor_tensor(dst=output_sb, data1=output_sb, data2=weight_scale_sb[:BxS, :], op=nl.multiply)
    sbm.pop_heap()  # weight_scale_sb


def _apply_bias(
    output_sb: nl.ndarray,
    qkv_bias: TensorView,
    sbm: SbufManager,
) -> None:
    """
    Apply bias to QKV projection output.

    Loads bias from HBM, broadcasts to all BxS partitions, and adds to output.

    Args:
        output_sb: QKV projection output in SBUF. Shape: (BxS, i_size)
        qkv_bias: Pre-sliced bias TensorView in HBM. Shape: (1, i_size)
        sbm: SbufManager for allocation
    """
    qkv_bias_sb = sbm.alloc_heap(output_sb.shape, dtype=qkv_bias.dtype, buffer=nl.sbuf)
    nisa.dma_copy(qkv_bias_sb[0:1, :], qkv_bias.get_view())
    stream_shuffle_broadcast(qkv_bias_sb, qkv_bias_sb)
    nisa.tensor_tensor(dst=output_sb, data1=output_sb, data2=qkv_bias_sb, op=nl.add)
    sbm.pop_heap()  # qkv_bias_sb


def _compute_qkv_i_block(
    hidden_sb: TensorView,
    qkv_w: TensorView,
    qkv_bias: nl.ndarray,
    qkv_out_sb: nl.ndarray,
    i_block: TiledRangeIterator,
    cfg: QkvTkgConfig,
    sbm: SbufManager,
    quantization_type: QuantizationType,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
) -> nl.ndarray:
    """
    Compute a single I-block of QKV projection: bias init, matmul, optional dequant, and cross-core reduce.

    Args:
        hidden_sb: Input hidden states in SBUF (TensorView). Shape: (H0, BxS, H1_sharded)
        qkv_w: QKV projection weights (TensorView). Shape: (H0, H1_sharded, I)
        qkv_bias: Optional bias in HBM. Shape: (1, I) or None
        qkv_out_sb: Pre-allocated SBUF output for this I-block. Shape: (BxS, i_block.size)
        cfg: QKV TKG config
        i_block: Current I-block from TiledRange
        sbm: SbufManager for SBUF allocation
        quantization_type: Quantization mode
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        Computed output in SBUF. Shape: (BxS, i_block.size)
    """
    num_shards, shard_id = cfg.num_shards, cfg.shard_id

    # Create bias slice for this I block if bias exists
    if qkv_bias != None and quantization_type == QuantizationType.NONE:
        qkv_bias_block = qkv_bias[:, i_block.start_offset : i_block.end_offset]
    else:
        qkv_bias_block = None

    _initialize_qkv_out_with_bias(qkv_out_sb, qkv_bias_block, cfg, i_block.index, sbm)

    # Slice qkv_w for this I block
    qkv_w_block = qkv_w.slice(dim=2, start=i_block.start_offset, end=i_block.end_offset)

    # Perform QKV projection for this I block
    # output_sb: (BxS, i_block.size)
    output_sb = _qkv_projection(
        hidden_sb=hidden_sb,
        qkv_w_hbm=qkv_w_block,
        qkv_out_sb=qkv_out_sb,
        cfg=cfg,
        i_block_idx=i_block.index,
        sbm=sbm,
    )

    if quantization_type == QuantizationType.STATIC:
        output_sb = _static_dequantize(output_sb, quant_config.combined_scale_sb, cfg, I_start=i_block.start_offset)
        if qkv_bias != None and cfg.shard_id == 0:
            qkv_bias_block = TensorView(qkv_bias).slice(dim=1, start=i_block.start_offset, end=i_block.end_offset)
            _apply_bias(output_sb, qkv_bias_block, sbm)
    elif quantization_type == QuantizationType.ROW:
        weight_scale_block = TensorView(quant_config.weight_scale_hbm).slice(
            dim=1, start=i_block.start_offset, end=i_block.end_offset
        )
        _row_dequantize(output_sb, weight_scale_block, cfg, sbm)
        if qkv_bias != None and cfg.shard_id == 0:
            qkv_bias_block = TensorView(qkv_bias).slice(dim=1, start=i_block.start_offset, end=i_block.end_offset)
            _apply_bias(output_sb, qkv_bias_block, sbm)

    # Receive qkv projection output from the other neuron core when LNC > 1
    if num_shards > 1:
        sbm.open_scope(name=f"output_store_sendrecv_block_{i_block.index}")
        qkv_recv = sbm.alloc_stack((cfg.BxS, i_block.size), dtype=output_sb.dtype, buffer=nl.sbuf)
        other_core = 1 - shard_id
        nisa.sendrecv(
            src=output_sb,
            dst=qkv_recv,
            send_to_rank=other_core,
            recv_from_rank=other_core,
            pipe_id=0,
        )
        nisa.tensor_tensor(output_sb, output_sb, qkv_recv, op=nl.add)
        sbm.close_scope()

    return output_sb


def _qkv_projection_sbuf_output(
    hidden_sb: TensorView,
    qkv_w: TensorView,
    qkv_bias: nl.ndarray,
    cfg: QkvTkgConfig,
    sbm: SbufManager,
    io_dtype,
    quantization_type: QuantizationType = QuantizationType.NONE,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]] = None,
) -> nl.ndarray:
    """
    QKV projection with SBUF output (output_in_sbuf=True path).

    Loops over I-blocks (each up to I_BLOCK_SIZE) with output kept in SBUF.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF (TensorView). Shape: (H0, BxS, H1_sharded)
        qkv_w: QKV projection weights (TensorView). Shape: (H0, H1_sharded, I)
        qkv_bias: Optional bias in HBM. Shape: (1, I)
        cfg: QKV TKG config
        sbm: SbufManager for SBUF allocation
        io_dtype: Data type for input/output tensors
        quantization_type: Quantization mode
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        QKV projection output in SBUF. Shape: (BxS, I)
    """

    BxS, I = cfg.BxS, cfg.I

    # Allocate full (BxS, I) output on heap — persists for caller
    qkv_out_sb = sbm.alloc_heap((BxS, I), dtype=io_dtype, buffer=nl.sbuf)

    # Process each I-block
    for i_block in TiledRange(I, I_BLOCK_SIZE):
        sbm.open_scope(name=f"qkv_sbuf_output_i_block_{i_block.index}")

        # Output slice for this I-block
        output_slice = qkv_out_sb[:, i_block.start_offset : i_block.end_offset]

        _compute_qkv_i_block(
            hidden_sb, qkv_w, qkv_bias, output_slice, i_block, cfg, sbm, quantization_type, quant_config
        )

        sbm.close_scope()

    return qkv_out_sb


def _qkv_projection_hbm_output(
    hidden_sb: TensorView,
    qkv_w: TensorView,
    qkv_bias: nl.ndarray,
    cfg: QkvTkgConfig,
    output_layout: QKVOutputLayout,
    sbm: SbufManager,
    io_dtype,
    quantization_type: QuantizationType = QuantizationType.NONE,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]] = None,
) -> nl.ndarray:
    """
    QKV projection with HBM output (output_in_sbuf=False path).

    Handles multiple I-blocks when I > I_BLOCK_SIZE. Each block is computed
    in SBUF then stored to HBM with layout-specific transformation.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF (TensorView). Shape: (H0, BxS, H1_sharded)
        qkv_w: QKV projection weights (TensorView). Shape: (H0, H1_sharded, I)
        qkv_bias: Optional bias in HBM. Shape: (1, I) or None
        cfg: QKV TKG config
        output_layout: Target layout (BSD or NBSd)
        sbm: SbufManager for SBUF allocation
        io_dtype: Dtype of the input hidden, which should also be the output dtype
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        QKV projection output tensor. Shape depends on output_layout:
        - BSD: (B, S, I)
        - NBSd: (num_heads, B, S, d_head)
    """

    B, S, BxS, I = cfg.B, cfg.S, cfg.BxS, cfg.I

    # Allocate output tensor with layout-specific shape
    if output_layout == QKVOutputLayout.BSD:
        output = nl.ndarray(
            (BxS, I), dtype=io_dtype, buffer=nl.shared_hbm, name="qkv_output_bsd"
        )
    elif output_layout == QKVOutputLayout.NBSd:
        nh = I // cfg.d_head
        output = nl.ndarray(
            (nh, BxS, cfg.d_head), dtype=io_dtype, buffer=nl.shared_hbm, name="qkv_output_nbsd"
        )

    # Process each I block
    for i_block in TiledRange(I, I_BLOCK_SIZE):
        sbm.open_scope(name=f"qkv_hbm_output_i_block_{i_block.index}")

        # Allocate output SB that gets accumulated in HBM
        qkv_out_sb = sbm.alloc_stack((BxS, i_block.size), dtype=io_dtype, buffer=nl.sbuf)

        output_sb = _compute_qkv_i_block(
            hidden_sb, qkv_w, qkv_bias, qkv_out_sb, i_block, cfg, sbm, quantization_type, quant_config
        )

        # Store to HBM with layout-specific transformation
        _store_qkv_output_to_hbm(
            output_hbm=output,
            output_sb=output_sb,
            I_block_start=i_block.start_offset,
            I_block_end=i_block.end_offset,
            output_layout=output_layout,
            cfg=cfg,
        )

        sbm.close_scope()

    # Reshape to expected output shape
    if output_layout == QKVOutputLayout.BSD:
        output = output.reshape((B, S, I))
    elif output_layout == QKVOutputLayout.NBSd:
        n_heads = I // cfg.d_head
        output = output.reshape((n_heads, B, S, cfg.d_head))

    # Return output in HBM
    return output


def _store_qkv_output_to_hbm(
    output_hbm: nl.ndarray,
    output_sb: nl.ndarray,
    I_block_start: int,
    I_block_end: int,
    output_layout: QKVOutputLayout,
    cfg: QkvTkgConfig,
) -> None:
    """
    Store QKV output from SBUF to HBM with layout-specific transformation.

    Handles two output layouts:
    - BSD: (Batch, Seqlen, fused_qkv_dim) -> direct 2D copy
    - NBSd: (num_heads, Batch, Seqlen, d_head) -> reshape with head dimension split

    Args:
        output_hbm: Destination HBM tensor
                    Shape: (BxS, I) for BSD layout
                           (num_heads, BxS, d_head) for NBSd layout
        output_sb: Source SBUF tensor with shape (BxS, I_block_size)
        I_block_start: Starting index in I dimension for this block
        I_block_end: Ending index in I dimension for this block (exclusive)
        output_layout: Target layout (BSD or NBSd)
        cfg: QKV TKG config
    """
    if output_layout == QKVOutputLayout.BSD:
        nisa.dma_copy(output_hbm[:, I_block_start:I_block_end], output_sb)

    elif output_layout == QKVOutputLayout.NBSd:
        # TODO: Change to TensorView
        I_block_size = I_block_end - I_block_start
        ns = I_block_start // cfg.d_head
        ne = I_block_end // cfg.d_head

        for i_n in range(ns, ne):
            output_pattern = [[cfg.d_head, cfg.BxS], [1, cfg.d_head]]
            output_offset = i_n * cfg.BxS * cfg.d_head
            output_sb_pattern = [[I_block_size, cfg.BxS], [1, cfg.d_head]]
            output_sb_offset = (i_n - ns) * cfg.d_head
            nisa.dma_copy(
                output_hbm.ap(pattern=output_pattern, offset=output_offset),
                output_sb.ap(pattern=output_sb_pattern, offset=output_sb_offset),
            )


def _qkv_projection(
    hidden_sb: TensorView,
    qkv_w_hbm: TensorView,
    qkv_out_sb: nl.ndarray,
    cfg: QkvTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
) -> nl.ndarray:
    _, _, I = qkv_w_hbm.shape  # I-block size from tensor
    output_dtype = hidden_sb.dtype
    weight_dtype = qkv_w_hbm.dtype

    sbm.open_scope(name=f"qkv_projection_block_{i_block_idx}")

    # Allocate all temp buffers: weights, PSUMs
    qkv_w_sb, num_w_blocks, result_psum = _allocate_qkv_buffers(
        I=I,
        qkv_out_shape=qkv_out_sb.shape,
        output_dtype=output_dtype,
        weight_dtype=weight_dtype,
        cfg=cfg,
        i_block_idx=i_block_idx,
        sbm=sbm,
    )

    # Process all H blocks (full + remainder)
    for h_block in TiledRange(cfg.H1_shard, NUM_TILES_PER_H_BLOCK):
        is_remainder = h_block.size < NUM_TILES_PER_H_BLOCK
        hidden_block = hidden_sb.slice(dim=2, start=h_block.start_offset, end=h_block.end_offset)
        qkv_w_block = qkv_w_hbm.slice(dim=1, start=h_block.start_offset, end=h_block.end_offset)

        _process_h_block(
            H_block_idx=h_block.index,
            num_128_tiles=h_block.size,
            array_tiled_H1=cfg.remainder_array_tiled_H1 if is_remainder else cfg.array_tiled_H1,
            array_tiling_dim=cfg.remainder_array_tiling_dim if is_remainder else cfg.array_tiling_dim,
            array_tiling_factor=cfg.remainder_array_tiling_factor if is_remainder else cfg.array_tiling_factor,
            qkv_w_hbm=qkv_w_block,
            qkv_w_sb=qkv_w_sb,
            hidden_sb=hidden_block,
            result_psum=result_psum,
            cfg=cfg,
            num_w_blocks=num_w_blocks,
        )

    # Accumulate PSUMs into output
    _accumulate_psum_to_output(qkv_out_sb=qkv_out_sb, result_psum=result_psum, cfg=cfg)

    sbm.close_scope()

    return qkv_out_sb


def _allocate_qkv_buffers(
    I: int,
    qkv_out_shape: Tuple[int, ...],
    output_dtype,
    weight_dtype,
    cfg: QkvTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
) -> Tuple[nl.ndarray, int, list]:
    """
    Allocate all buffers needed for QKV projection: weights, and PSUMs.

    Allocates:
    1. Weight tile buffer (qkv_w_sb) sized to fit remaining SBUF space
    2. PSUM tiles for accumulation (one per I tile)

    Args:
        I: I-block size (from tensor shape)
        qkv_out_shape: Shape of the output slice, used to reserve space for post-projection
            buffers (bias broadcast, sendrecv) that coexist at the same scope level
        output_dtype: Data type for output tensor
        weight_dtype: Data type for weight tensor
        cfg: QKV TKG config
        i_block_idx: I-block index for buffer naming
        sbm: SbufManager for SBUF allocation

    Returns:
        Tuple of (qkv_w_sb, num_w_blocks, result_psum):
        - qkv_w_sb: Weight tile buffer
        - num_w_blocks: Number of weight tiles allocated
        - result_psum: List of PSUM tiles
    """

    # Reserve space for post-projection buffers (bias broadcast, sendrecv) that will be
    # allocated at the same scope level after the projection scope closes.
    # get_free_space() already accounts for all prior allocations (stack and heap),
    # but not for these future sibling-scope allocations.
    extra_space_needed = sizeinbytes(output_dtype)
    for i in range(len(qkv_out_shape)):
        extra_space_needed *= qkv_out_shape[i]

    remaining_space = sbm.get_free_space() - extra_space_needed
    size_of_qkv_w_block = I * NUM_TILES_PER_H_BLOCK * sizeinbytes(weight_dtype)
    num_available_w_blocks = remaining_space // size_of_qkv_w_block
    num_H_blocks = div_ceil(cfg.H_shard, H_BLOCK_SIZE)
    num_w_blocks = min(num_H_blocks, num_available_w_blocks)
    # With auto_alloc, remaining_space underestimates available memory due to automatic reuse,
    # so ensure at least one tile can be allocated
    if sbm.is_auto_alloc():
        num_w_blocks = max(1, num_w_blocks)
    kernel_assert(
        num_w_blocks > 0,
        f"Not enough SBUF space for qkv projection weight, need {size_of_qkv_w_block}, got {remaining_space}",
    )

    # Allocate weight tiles
    qkv_w_sb = sbm.alloc_stack(
        (cfg.H0, num_w_blocks, NUM_TILES_PER_H_BLOCK, I),
        name=f"qkv_w_sb_block_{i_block_idx}",
        dtype=weight_dtype,
        buffer=nl.sbuf,
    )
    qkv_w_sb = TensorView(qkv_w_sb)

    # Allocate PSUM tiles - one per I tile in this I-block
    n_psum = div_ceil(I, I_TILE_SIZE)
    result_psum = []
    prefix = sbm.get_name_prefix()
    for psum_idx in range(n_psum):
        psum_tensor = nl.ndarray(
            (128, I_TILE_SIZE),
            dtype=nl.float32,
            name=f"{prefix}batch_result_psum_{i_block_idx}_{psum_idx}",
            buffer=nl.psum,
            address=None
            if sbm.is_auto_alloc()
            else (
                0,
                (psum_idx % NUM_PSUM_BANKS) * (F_MAX * sizeinbytes(nl.float32)),
            ),
        )
        result_psum.append(psum_tensor)

    return qkv_w_sb, num_w_blocks, result_psum


def _process_h_block(
    H_block_idx: int,
    num_128_tiles: int,
    array_tiled_H1: int,
    array_tiling_dim: int,
    array_tiling_factor: int,
    qkv_w_hbm: TensorView,
    qkv_w_sb: TensorView,
    hidden_sb: TensorView,
    result_psum: list,
    cfg: QkvTkgConfig,
    num_w_blocks: int,
) -> None:
    """
    Process a single H block: load weights and perform tiled matrix multiplication.

    Unified implementation for both full and remainder H blocks.
    Caller determines block type and provides appropriate parameters.

    Steps:
    1. Load weight tile from HBM to SBUF using TensorView
    2. Perform nested tiled matmul with array tiling optimization:
       - Outer loop: h1_tile (array tiling chunks)
       - Middle loop: factor (array tiling factor)
       - Inner loop: i_tile (I tiles within this I-block)

    Args:
        H_block_idx: Index of current H block
        num_128_tiles: Number of 128-element tiles in this H block
        array_tiled_H1: Number of array-tiled H1 chunks for this tile
        array_tiling_dim: Array tiling dimension (32, 64, or 128)
        array_tiling_factor: Array tiling factor (128 / array_tiling_dim)
        qkv_w_hbm: QKV projection weights (TensorView, already sliced to H and I). Shape: (H0, num_128_tiles, I_block_size)
        qkv_w_sb: Weight tile buffer in SBUF (TensorView). Shape: (H0, num_w_blocks, NUM_TILES_PER_H_BLOCK, I_block_size)
        hidden_sb: Hidden states in SBUF (TensorView, already sliced to H block). Shape: (H0, BxS, num_128_tiles)
        result_psum: List of PSUM tensors for accumulation
        cfg: QKV TKG config
        num_w_blocks: Number of weight tiles allocated in SBUF
    """

    I = qkv_w_sb.shape[3]  # I-block size from tensor shape

    # Select the weight tile slot for this H_block (circular buffer), slice to actual num_128_tiles
    w_block_slot = H_block_idx % num_w_blocks
    qkv_w_sb_block = qkv_w_sb.select(dim=1, index=w_block_slot).slice(dim=1, start=0, end=num_128_tiles)

    nisa.dma_copy(qkv_w_sb_block.get_view(), qkv_w_hbm.get_view())

    # Perform tiled matrix multiplication with array tiling
    for h1_tile in range(array_tiled_H1):
        array_tile_offset = array_tiling_factor * h1_tile

        for factor in range(array_tiling_factor):
            h1_tile_idx = array_tile_offset + factor
            hidden_tile = hidden_sb.select(dim=2, index=h1_tile_idx)

            # Process all I tiles (full + remainder) within this I-block
            for i_tile in TiledRange(I, I_TILE_SIZE):
                qkv_w_sb_tile = qkv_w_sb_block.select(dim=1, index=h1_tile_idx).slice(
                    dim=1, start=i_tile.start_offset, end=i_tile.end_offset
                )

                psum_row_start = array_tiling_dim * factor
                result_slice = result_psum[i_tile.index][psum_row_start : psum_row_start + cfg.BxS, 0 : i_tile.size]

                nisa.nc_matmul(
                    result_slice,
                    hidden_tile.get_view(),
                    qkv_w_sb_tile.get_view(),
                    tile_position=(0, array_tiling_dim * factor),
                    tile_size=(cfg.H0, array_tiling_dim),
                )


def _accumulate_psum_to_output(
    qkv_out_sb: nl.ndarray,
    result_psum: list,
    cfg: QkvTkgConfig,
) -> None:
    """
    Accumulate PSUM tiles into final output tensor.

    Args:
        qkv_out_sb: Output buffer to accumulate into. Shape: (BxS, I_block_size)
        result_psum: List of PSUM tensors. Shape: (128, I_TILE_SIZE) each
        cfg: QKV TKG config
    """

    array_tiling_factor = cfg.array_tiling_factor
    array_tiling_dim = cfg.array_tiling_dim
    has_only_remainder_H_block = cfg.H_shard < H_BLOCK_SIZE
    if has_only_remainder_H_block:
        # When there is only H remainder tile, use the remainder tiling level
        array_tiling_factor = cfg.remainder_array_tiling_factor
        array_tiling_dim = cfg.remainder_array_tiling_dim

    I = qkv_out_sb.shape[1]  # I-block size from output shape

    # Accumulate all I tiles (full + remainder)
    for i_tile in TiledRange(I, I_TILE_SIZE):
        for factor in range(array_tiling_factor):
            result_psum_slice_start = array_tiling_dim * factor
            result_psum_slice_end = array_tiling_dim * factor + cfg.BxS

            nisa.tensor_tensor(
                qkv_out_sb[0 : cfg.BxS, i_tile.start_offset : i_tile.end_offset],
                qkv_out_sb[0 : cfg.BxS, i_tile.start_offset : i_tile.end_offset],
                result_psum[i_tile.index][result_psum_slice_start:result_psum_slice_end, 0 : i_tile.size],
                op=nl.add,
            )
