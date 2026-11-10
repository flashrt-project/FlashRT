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
Output Projection TKG Kernel

This kernel implements the output projection operation (attention @ weight +
bias) commonly used after attention blocks in transformer models. The kernel is
specifically optimized for Token Generation (TKG, also known as Decode) scenarios
where the sequence length S is small (often 1 or a small number for spec. decode).

Remark: The input layouts expected for this kernel are different from those for the
CTE kernel. The reason for this is the broader impacts of such layouts on performance
(not just on this kernel but also on other kernels).

In CTE workloads, where sequence length is large, we generally expect to have to reload
it from HBM more frequently. Placing the large S dimension at the end allows more efficient
HBM loads.

For TKG workloads, the S dimension is small, so placing the N dimension next to it
allows more efficient GQA implementations by loading multiple heads at once.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded
across the cores. We choose to shard on H as this avoids the need for any
inter-core collective operations, as each core produces part of the output tensor.

"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import matmul_perf_mode
from nki.language import affine_range, static_range

from nkilib.core.utils.allocator import BufferManager, create_auto_alloc_manager
from nkilib.core.utils.common_types import QuantizationType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_program_sharding_info
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from nkilib.core.utils.tiled_range import TiledRange
from nkilib.core.output_projection.output_projection_utils import calculate_head_packing

P_MAX = 128
F_MAX = 512

# Conservative limit assuming 2 bytes per scalar and LNC=2, ~20MB SBUF per core for projection weights
MAX_VALIDATED_N_TIMES_H_SIZE = 163840
MAX_VALIDATED_N_TIMES_H_SIZE_FP32 = MAX_VALIDATED_N_TIMES_H_SIZE // 2


@nki.jit
def output_projection_tkg(
    attention: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.ndarray] = None,
    input_scale: Optional[nl.ndarray] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm: Optional[BufferManager] = None,
) -> nl.ndarray:
    """
    Output Projection Kernel

    This kernel computes
      out = attention @ weight + bias
    typically used to project the output scores after an attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decode) use cases where sequence length S is
    small.

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        attention (nl.ndarray): Input tensor in HBM or SBUF, typically the scores output from an attention block.
            Shape:    [D, B, N, S]
            Indexing: [d, b, n, s]
            Dtype:    nl.float32, nl.float16, or nl.bfloat16
        weight (nl.ndarray): Weight tensor in HBM
            Shape:    [N * D,     H]
            Indexing: [n * D + d, h]
            Dtype:
                - QuantizationType.NONE: nl.float32, nl.float16, or nl.bfloat16
                - QuantizationType.STATIC: nl.float8_e4m3
                - QuantizationType.ROW: nl.float8_e4m3
        bias (Optional[nl.ndarray]): Optional bias tensor in HBM
            Shape:    [1, H]
            Indexing: [1, h]
            Dtype:    nl.float32, nl.float16, or nl.bfloat16
        quantization_type (QuantizationType): Type of quantization to apply (NONE, STATIC, ROW).
            Default: QuantizationType.NONE.
        weight_scale (Optional[nl.ndarray]): Weight dequantization scale tensor in HBM
            Shape:    [P_MAX, 1] for STATIC, [P_MAX, H] for ROW (all P_MAX rows identical)
            Dtype:    nl.float32
        input_scale (Optional[nl.ndarray]): Input dequantization scale tensor in HBM
            Shape:    [P_MAX, 1] for STATIC (not used for ROW)
            Dtype:    nl.float32
        TRANSPOSE_OUT (bool): Whether to store the output in transposed shape.
            If False, the output tensor has the following shape and indexing:
              Shape:    [B * S,     H]
              Indexing: [b * S + s, h]
            If True, the output is instead kept in a different shape, which may be
            advantageous for other kernels' performance.
              Shape:    [H_1, H_0, H_2, B * S    ]
              Indexing: [h_1, h_0, h_2, b * S + s]
            where
              H_0 = logical core size (LNC = 1 or LNC = 2),
              H_1 = 128,
              H_2 = H // H_0 // H_1,
            such that h = h_0 * H_1 * H_2 + h_1 * H_2 + h_2.
        OUT_IN_SB (bool): If True, output is in SBUF. Else, it is written out to HBM.
        sbm (BufferManager): Optional BufferManager for tensor allocation with consistent naming.

    Returns:
        out (nl.ndarray): Output tensor in HBM. Shape depends on `TRANSPOSE_OUT` parameter.

    Notes:
        - This kernel supports nl.float32, nl.float16 and nl.bfloat16 data types.
          However, for nl.float32, large inputs may not fit in SBUF.
        - The product B * S must not exceed 128.
        - Head dimension D must not exceed 128.
        - When TRANSPOSE_OUT is False: H must be divisible by LNC.
        - When TRANSPOSE_OUT is True: H must be divisible by 128 * LNC.
        - When TRANSPOSE_OUT is True with float32 dtype: N * H must not exceed 81920.
        - When TRANSPOSE_OUT is True with float16/bfloat16 dtype: N * H must not exceed 163840.

    Pseudocode:
        # Load attention scores
        attn_sb = load_to_sbuf(attention)

        # Shuffle attention from [D, B, N, S] to [D, N * B * S]
        attn_shuffled = shuffle(attn_sb)

        # Compute projection
        if TRANSPOSE_OUT:
            for h_tile in range(H // (128 * LNC)):
                result[h_tile] = attn_shuffled @ weight[h_tile]
                if bias is not None:
                    result[h_tile] += bias[h_tile]
        else:
            for h_block in range(H // (512 * LNC)):
                result[h_block] = attn_shuffled @ weight[h_block]
                if bias is not None:
                    result[h_block] += bias[h_block]

        return result
    """

    cfg = _validate_and_create_config(
        attention=attention,
        weight=weight,
        bias=bias,
        quantization_type=quantization_type,
        weight_scale=weight_scale,
        input_scale=input_scale,
        transpose_out=TRANSPOSE_OUT,
        out_in_sb=OUT_IN_SB,
    )

    # Create local sbm if not provided
    if sbm == None:
        sbm = create_auto_alloc_manager()

    # Load quantization scales
    if quantization_type == QuantizationType.STATIC:
        weight_scale_sb = sbm.alloc_heap(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, name="weight_scale_sb"
        )
        input_scale_sb = sbm.alloc_heap(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, name="input_scale_sb"
        )
        nisa.dma_copy(dst=weight_scale_sb, src=weight_scale)
        nisa.dma_copy(dst=input_scale_sb, src=input_scale)
        # Pre-apply input scales onto the weight scaling to get combined dequant scale
        combined_scale_sb = sbm.alloc_heap(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, name="combined_scale_sb"
        )
        nisa.dma_copy(dst=combined_scale_sb, src=weight_scale)
        nisa.activation(dst=combined_scale_sb, op=nl.copy, data=combined_scale_sb, scale=input_scale_sb)
        quant_config = StaticQuantConfig(combined_scale_sb=combined_scale_sb)
    elif quantization_type == QuantizationType.ROW:
        quant_config = RowQuantConfig(weight_scale_hbm=weight_scale)
        input_scale_sb = None
    else:
        input_scale_sb = None
        quant_config = None

    attn_shuffled = _load_and_shuffle_attn(
        attention=attention,
        input_scale_sb=input_scale_sb,
        cfg=cfg,
    )

    w_reshaped = weight.reshape((cfg.n_size, cfg.d_size, cfg.h_size))

    if not cfg.transpose_out:
        out = (
            sbm.alloc(
                (cfg.b_size * cfg.s_size, cfg.h_size),
                dtype=cfg.io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not cfg.out_in_sb
            else None
        )

        return _output_projection_tkg_impl(
            out_hbm_buffer=out,
            bias=bias,
            w_reshaped=w_reshaped,
            quant_config=quant_config,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
        )

    else:  # TRANSPOSE_OUT == True
        """
        Notes on iteration order:
        
        cfg.h_0_size corresponds to the outermost logical iterator h_0 = prg_id from 0 to cfg.num_prgs - 1. This corresponds to LNC sharding.
        cfg.h_1_size corresponds to the mid logical iterator h_1 from 0 to P_MAX - 1. This is placed in partition dim.
        cfg.h_2_size corresponds to the innermost logical iterator h_2. This is placed in free dim.
        
        Check for h_size % (P_MAX * n_prgs) == 0 above should cover this
        """
        kernel_assert(cfg.h_size == cfg.h_0_size * cfg.h_1_size * cfg.h_2_size, "")
        out = (
            sbm.alloc(
                (cfg.h_1_size, cfg.h_0_size, cfg.h_2_size, cfg.b_size * cfg.s_size),
                dtype=cfg.io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not cfg.out_in_sb
            else None
        )
        return _output_projection_tkg_transpose_out_impl(
            out_hbm_buffer=out,
            bias=bias,
            w_reshaped=w_reshaped,
            quant_config=quant_config,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
        )


@dataclass()
class StaticQuantConfig(nl.NKIObject):
    """Configuration for STATIC quantization.

    Holds pre-computed combined scale (weight_scale * input_scale) in SBUF.
    """

    combined_scale_sb: nl.ndarray  # [P_MAX, 1], pre-computed in SBUF


@dataclass()
class RowQuantConfig(nl.NKIObject):
    """Configuration for ROW (per-output-channel) quantization.

    Holds weight scale tensor in HBM to be loaded inside impl functions.
    """

    weight_scale_hbm: nl.ndarray  # [P_MAX, H], in HBM


@dataclass()
class OutputProjectionTkgConfig(nl.NKIObject):
    """Configuration and validation for output projection TKG kernel."""

    # Input dimensions
    d_original_size: int
    b_size: int
    n_original_size: int
    s_size: int
    h_size: int

    # Execution parameters
    num_prgs: int
    prg_id: int

    # Kernel options
    transpose_out: bool
    out_in_sb: bool
    is_quantized: bool
    quantization_type: QuantizationType
    use_double_row: bool

    # Optional tensors (for validation)
    has_bias: bool = False
    has_weight_scale: bool = False
    has_input_scale: bool = False

    # Data types
    io_dtype: Any = None
    quant_dtype: Any = None

    # Computed fields (set by compute_derived_config)
    n_size: int = None
    d_size: int = None
    group_size: int = None
    h_sharded: int = None
    h_0_size: int = None
    h_1_size: int = None
    h_2_size: int = None


def _validate_and_create_config(
    attention: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.ndarray] = None,
    input_scale: Optional[nl.ndarray] = None,
    transpose_out: bool = False,
    out_in_sb: bool = False,
) -> OutputProjectionTkgConfig:
    """
    Validate inputs and create kernel configuration.

    Performs comprehensive validation of input tensor shapes, quantization settings,
    and layout constraints. Computes derived tiling parameters including head packing.

    Args:
        attention (nl.ndarray): [D, B, N, S], Input attention tensor.
        weight (nl.ndarray): [N*D, H], Weight tensor.
        bias (Optional[nl.ndarray]): [1, H], Optional bias tensor.
        quantization_type (QuantizationType): Quantization mode (NONE, STATIC, or ROW).
        weight_scale (Optional[nl.ndarray]): [P_MAX, 1] for STATIC, [P_MAX, H] for ROW.
        input_scale (Optional[nl.ndarray]): [P_MAX, 1] for STATIC (not used for ROW).
        transpose_out (bool): Whether to produce transposed output layout.
        out_in_sb (bool): Whether output stays in SBUF.

    Returns:
        OutputProjectionTkgConfig: Validated configuration with computed tiling parameters.
    """
    d_original_size, b_size, n_original_size, s_size = attention.shape
    n_d, h_size = weight.shape
    io_dtype = attention.dtype
    _, n_prgs, prg_id = get_program_sharding_info()

    # Hardware constraints
    kernel_assert(nl.tile_size.pmax == nl.tile_size.gemm_stationary_fmax, "")
    kernel_assert(nl.tile_size.psum_fmax == nl.tile_size.gemm_moving_fmax, "")

    # Validate weight shape
    kernel_assert(
        n_d == n_original_size * d_original_size,
        f"output_projection_tkg kernel requires weight in shape (N * D = {n_original_size * d_original_size}, H = {h_size}), but got {weight.shape}.\n"
        f"Note: N and D inferred from attention score shape: {attention.shape}: {n_original_size} * {d_original_size} = {n_original_size * d_original_size}.",
    )

    # Validate bias shape
    if bias != None:
        kernel_assert(
            bias.shape[0] == 1,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )
        kernel_assert(
            bias.shape[1] == h_size,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )

    # Validate quantization type
    kernel_assert(
        quantization_type == QuantizationType.NONE
        or quantization_type == QuantizationType.STATIC
        or quantization_type == QuantizationType.ROW,
        f"Only QuantizationType.NONE, QuantizationType.STATIC, and QuantizationType.ROW are supported, but got {quantization_type}",
    )

    # Validate quantization scale shapes
    if quantization_type == QuantizationType.STATIC:
        kernel_assert(weight_scale != None, f"Weight scale must be provided for quantization type {quantization_type}")
        kernel_assert(input_scale != None, f"Input scale must be provided for quantization type {quantization_type}")
        kernel_assert(
            weight_scale.shape == (P_MAX, 1),
            f"Incorrect shape for weight scale for static per tensor quantization, expected ({P_MAX}, 1), got {weight_scale.shape}",
        )
        kernel_assert(
            input_scale.shape == (P_MAX, 1),
            f"Incorrect shape for input scale for static per tensor quantization, expected ({P_MAX}, 1), got {input_scale.shape}",
        )
    elif quantization_type == QuantizationType.ROW:
        kernel_assert(weight_scale != None, f"Weight scale must be provided for quantization type {quantization_type}")
        kernel_assert(
            weight_scale.shape == (P_MAX, h_size),
            f"Incorrect shape for weight scale for row quantization, expected ({P_MAX}, {h_size}), got {weight_scale.shape}",
        )

    # Kernel shape validation
    kernel_assert(
        h_size % n_prgs == 0,
        f"output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be divisible by logical core size of {n_prgs}.",
    )

    # Dimension constraints
    if out_in_sb and not transpose_out:
        kernel_assert(
            b_size * s_size <= P_MAX,
            f"When OUT_IN_SB=True and TRANSPOSE_OUT=False, output_projection_tkg kernel does not support (B * S = {b_size * s_size}) > {P_MAX}.",
        )

    kernel_assert(
        d_original_size <= P_MAX,
        f"output_projection_tkg kernel does not support head dimension (D = {d_original_size}) greater than {P_MAX}.",
    )

    # Layout-specific validation
    if not transpose_out:
        kernel_assert(
            h_size % n_prgs == 0,
            f"When `TRANSPOSE_OUT` is False, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of logical core size, where logical core size is {n_prgs}.",
        )
    else:
        kernel_assert(
            h_size % (P_MAX * n_prgs) == 0,
            f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of {P_MAX} * logical core size, where logical core size is {n_prgs}.",
        )

        # Size limits for transpose mode
        if weight.dtype == nl.float32:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE_FP32,
                f"When `TRANSPOSE_OUT` is True and using 32bit floats, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE_FP32}.",
            )
        else:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE,
                f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE}.",
            )

    # Compute head packing and tiling parameters
    # Head packing optimization
    if d_original_size % 32 == 0:
        n_size, d_size, group_size = calculate_head_packing(n_original_size, d_original_size, P_MAX)
    else:
        n_size, d_size, group_size = n_original_size, d_original_size, 1

    # Tiling parameters
    h_sharded = h_size // n_prgs

    if transpose_out:
        h_0_size = n_prgs
        h_1_size = P_MAX
        h_2_size = h_size // n_prgs // P_MAX
    else:
        h_0_size = -1
        h_1_size = -1
        h_2_size = -1

    is_quantized = quantization_type == QuantizationType.STATIC or quantization_type == QuantizationType.ROW

    # TODO: support padding for odd number of heads
    # n_heads // 2 * batch needs to be multiple of 16 for double row stride access
    use_double_row = (
        quantization_type == QuantizationType.STATIC
        and n_size % 2 == 0
        and n_size * b_size % 32 == 0
        and b_size * s_size >= 64
    )

    return OutputProjectionTkgConfig(
        d_original_size=d_original_size,
        b_size=b_size,
        n_original_size=n_original_size,
        s_size=s_size,
        h_size=h_size,
        num_prgs=n_prgs,
        prg_id=prg_id,
        transpose_out=transpose_out,
        out_in_sb=out_in_sb,
        is_quantized=is_quantized,
        quantization_type=quantization_type,
        use_double_row=use_double_row,
        has_bias=bias != None,
        has_weight_scale=weight_scale != None,
        has_input_scale=input_scale != None,
        io_dtype=io_dtype,
        quant_dtype=weight.dtype,
        n_size=n_size,
        d_size=d_size,
        group_size=group_size,
        h_sharded=h_sharded,
        h_0_size=h_0_size,
        h_1_size=h_1_size,
        h_2_size=h_2_size,
    )


def _load_and_shuffle_attn(
    attention: nl.ndarray,
    input_scale_sb: Optional[nl.ndarray],
    cfg: OutputProjectionTkgConfig,
) -> nl.ndarray:
    """
    Load attention tensor to SBUF, optionally quantize, and shuffle to optimized layout.

    Transforms attention from [d_original_size, B, n_original_size, S] to [D, N * B * S] or
    [D, 2, N//2 * B * S] for efficient matrix multiplication. When group_size > 1, multiple heads
    are packed into the partition dimension D (head packing optimization). For STATIC quantization,
    applies input scaling and quantizes activations to FP8. ROW quantization does not quantize activations.

    Args:
        attention (nl.ndarray): [d_original_size, B, n_original_size, S], Input attention tensor.
        input_scale_sb (Optional[nl.ndarray]): [P_MAX, 1], Input scale tensor in SBUF for quantization.
        cfg (OutputProjectionTkgConfig): Configuration with D=d_original_size*group_size, N=n_original_size//group_size.

    Returns:
        nl.ndarray: Shuffled attention tensor in SBUF. Shape [D, N * B * S] or [D, 2, N//2 * B * S].

    Notes:
        - When group_size > 1, cfg.d_size = d_original_size * group_size and cfg.n_size = n_original_size // group_size
        - The shuffle operation maps n_orig to (n_group, n_offset) where n_group = n_orig // group_size
    """
    if attention.buffer == nl.sbuf:
        attn_sb = attention
    else:
        attn_sb = nl.ndarray(
            (cfg.d_original_size, cfg.b_size, cfg.n_original_size, cfg.s_size),
            dtype=attention.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=attn_sb[...], src=attention[...])

    if cfg.quantization_type == QuantizationType.STATIC:
        attn_quantized = nl.ndarray(
            (cfg.d_original_size, cfg.b_size, cfg.n_original_size, cfg.s_size),
            dtype=cfg.quant_dtype,
            buffer=nl.sbuf,
        )
        nisa.reciprocal(dst=input_scale_sb, data=input_scale_sb)
        nisa.activation(dst=attn_sb, op=nl.copy, data=attn_sb, scale=input_scale_sb[: cfg.d_original_size, :])
        max_pos_val = get_max_positive_value_for_dtype(cfg.quant_dtype)
        nisa.tensor_scalar(
            dst=attn_quantized,
            data=attn_sb,
            op0=nl.minimum,
            operand0=max_pos_val,
            op1=nl.maximum,
            operand1=-max_pos_val,
        )
        attn_sb = attn_quantized

    """
    Shuffle from attn_sb[d_original_size, B, n_original_size, S] to attn_shuffled[D, N * B * S]
    Indexing is attn_shuffled[d, n * B * S + b * S + s]
    Combined reshape + shuffle when group_size > 1

    For double row, the shape is attn_shuffled[D, 2, N // 2 * B * S] and we need to
    interleave heads so that adjacent head pairs are processed together:
    - Head 0 -> row 0, pair 0: attn_shuffled[d, 0, 0:bxs]
    - Head 1 -> row 1, pair 0: attn_shuffled[d, 1, 0:bxs]
    - Head 2 -> row 0, pair 1: attn_shuffled[d, 0, bxs:2*bxs]
    - Head 3 -> row 1, pair 1: attn_shuffled[d, 1, bxs:2*bxs]
    """
    tensor_shape = (
        (cfg.d_size, 2, cfg.n_size // 2 * cfg.b_size * cfg.s_size)
        if cfg.use_double_row
        else (cfg.d_size, cfg.n_size * cfg.b_size * cfg.s_size)
    )

    attn_shuffled = nl.ndarray(
        tensor_shape,
        dtype=attn_sb.dtype,
        buffer=nl.sbuf,
    )

    bxs_size = cfg.b_size * cfg.s_size
    # Use original n_original_size before it was divided by group_siz
    for n_orig in static_range(cfg.n_original_size):
        # Map original n to new (n_group, n_offset) coordinates
        n_group, n_offset = divmod(n_orig, cfg.group_size)
        nisa.tensor_copy(
            dst=attn_shuffled.ap(
                pattern=[
                    [cfg.n_size * cfg.b_size * cfg.s_size, cfg.d_original_size],
                    [cfg.s_size, cfg.b_size],
                    [1, cfg.s_size],
                ],
                offset=n_offset * cfg.d_original_size * cfg.n_size * bxs_size
                + (
                    # For double_row, interleave heads so adjacent pairs are together:
                    # pair_idx, row_idx = divmod(n_group, 2)
                    # offset = row_idx * (n_size // 2 * bxs) + pair_idx * bxs
                    (n_group % 2) * (cfg.n_size // 2 * bxs_size) + (n_group // 2) * bxs_size
                    if cfg.use_double_row
                    else n_group * bxs_size
                ),
            ),
            src=attn_sb.ap(
                pattern=[
                    [cfg.b_size * cfg.n_original_size * cfg.s_size, cfg.d_original_size],
                    [cfg.n_original_size * cfg.s_size, cfg.b_size],
                    [1, cfg.s_size],
                ],
                offset=cfg.s_size * n_orig,
            ),
        )

    return attn_shuffled


def _output_projection_tkg_impl(
    out_hbm_buffer: Optional[nl.ndarray],
    bias: Optional[nl.ndarray],
    w_reshaped: nl.ndarray,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    attn_shuffled: nl.ndarray,
    cfg: OutputProjectionTkgConfig,
) -> nl.ndarray:
    """
    Core implementation for regular (non-transposed) output projection.

    Computes attention @ weight + bias with output shape [B*S, H]. Tiles computation
    across H dimension in F_MAX-sized blocks and B*S dimension in P_MAX-sized blocks.

    Args:
        out_hbm_buffer (Optional[nl.ndarray]): Output buffer in HBM or None if output in SBUF.
        bias (Optional[nl.ndarray]): [1, H], Optional bias tensor in HBM.
        w_reshaped (nl.ndarray): [N, D, H], Weight tensor reshaped.
        quant_config (Optional[Union[StaticQuantConfig, RowQuantConfig]]): Quantization config.
        attn_shuffled (nl.ndarray): [D, N*B*S], Shuffled attention tensor in SBUF.
        cfg (OutputProjectionTkgConfig): Kernel configuration containing dimensions and options.

    Returns:
        nl.ndarray: Output tensor with shape [B*S, H], either from HBM buffer or SBUF.
    """
    bxs_size = cfg.b_size * cfg.s_size
    BxS_block_size = min(P_MAX, bxs_size)
    if bias != None:
        # Load bias at once, this can be improved if cfg.h_size is large.
        bias_sb_1d = nl.ndarray((1, cfg.h_sharded), dtype=bias.dtype)
        nisa.dma_copy(
            src=bias.ap(
                pattern=[[cfg.h_sharded, 1], [1, cfg.h_sharded]],
                offset=cfg.prg_id * cfg.h_sharded,
            ),
            dst=bias_sb_1d,
        )
        bias_sb = nl.ndarray((BxS_block_size, cfg.h_sharded), dtype=bias.dtype, buffer=nl.sbuf)
        # Broadcast bias from [1, cfg.h_sharded] to [B*S, cfg.h_sharded] to match out_sb shape below.
        stream_shuffle_broadcast(bias_sb_1d, bias_sb)

    """
    Potentially load w_reshaped into separate blocks in sbuf to allow prefetching.
    2K block size, otherwise just 1 block.
    TODO: Fine tune this number.
    """
    h_block_size = 2048 if cfg.h_sharded % 2048 == 0 else cfg.h_sharded
    num_h_blocks_per_prg = cfg.h_sharded // h_block_size
    kernel_assert(num_h_blocks_per_prg * h_block_size * cfg.num_prgs == cfg.h_size, "")

    if cfg.quantization_type == QuantizationType.STATIC:
        weight_scale_sb = quant_config.combined_scale_sb
    elif cfg.quantization_type == QuantizationType.ROW:
        weight_scale_hbm = quant_config.weight_scale_hbm
        weight_scale_blocks = []
        for h_block_idx in affine_range(num_h_blocks_per_prg):
            scale_tensor = nl.ndarray((P_MAX, h_block_size), dtype=weight_scale_hbm.dtype, buffer=nl.sbuf)
            h_offset = (cfg.prg_id * num_h_blocks_per_prg + h_block_idx) * h_block_size
            nisa.dma_copy(
                src=weight_scale_hbm[:, nl.ds(h_offset, h_block_size)],
                dst=scale_tensor,
            )
            weight_scale_blocks.append(scale_tensor)

    """
    By loading into separate sbuf tensors, we give the compiler finer control over pre-fetching.
    overall shape is [num_h_blocks_per_prg][cfg.n_size][cfg.d_size, h_block_size]
    for double row, shape is [num_h_blocks_per_prg][cfg.n_size // 2][cfg.d_size, 2, h_block_size]
    """
    w_sbuf_blocks = []
    for h_block_idx in affine_range(num_h_blocks_per_prg):
        w_row = []
        if not cfg.use_double_row:
            for head_idx in affine_range(cfg.n_size):
                w_tensor = nl.ndarray((cfg.d_size, h_block_size), dtype=w_reshaped.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=head_idx * cfg.d_size * cfg.h_size
                        + (cfg.prg_id * num_h_blocks_per_prg + h_block_idx) * h_block_size,
                    ),
                    dst=w_tensor,
                )
                w_row.append(w_tensor)
        else:
            for head_idx in affine_range(0, cfg.n_size, 2):
                w_tensor = nl.ndarray((cfg.d_size, 2, h_block_size), dtype=w_reshaped.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=head_idx * cfg.d_size * cfg.h_size
                        + (cfg.prg_id * num_h_blocks_per_prg + h_block_idx) * h_block_size,
                    ),
                    dst=w_tensor[:, 0, :],
                )
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=(head_idx + 1) * cfg.d_size * cfg.h_size
                        + (cfg.prg_id * num_h_blocks_per_prg + h_block_idx) * h_block_size,
                    ),
                    dst=w_tensor[:, 1, :],
                )
                w_row.append(w_tensor)
        w_sbuf_blocks.append(w_row)

    # Compute and write out attention @ weight (+ bias) blocks
    matmul_idx = 0  # for load balancing tensor_copy
    for bxs_block in TiledRange(bxs_size, P_MAX):
        out_sb = nl.ndarray(
            (bxs_block.size, cfg.h_sharded),
            dtype=cfg.io_dtype,
            buffer=nl.sbuf,
        )

        for h_block_idx in affine_range(num_h_blocks_per_prg):
            for h_block_f_tile in TiledRange(h_block_size, F_MAX):
                matmul_idx += 1
                res_psum = nl.ndarray((bxs_block.size, h_block_f_tile.size), dtype=nl.float32, buffer=nl.psum)

                # Accumulate (B*S, F_MAX) tiled attn @ weight blocks for all cfg.n_size heads
                if not cfg.use_double_row:
                    for head_idx in affine_range(cfg.n_size):
                        stationary = attn_shuffled[
                            :, nl.ds(head_idx * bxs_size + bxs_block.start_offset, bxs_block.size)
                        ]
                        moving = w_sbuf_blocks[h_block_idx][head_idx][
                            :, nl.ds(h_block_f_tile.start_offset, h_block_f_tile.size)
                        ]
                        nisa.nc_matmul(res_psum, stationary, moving)
                else:
                    # For double row we use the leading free dimension of 2 and double the data per matmul
                    for head_idx in affine_range(cfg.n_size // 2):
                        stationary = attn_shuffled[
                            :, :, nl.ds(head_idx * bxs_size + bxs_block.start_offset, bxs_block.size)
                        ]
                        moving = w_sbuf_blocks[h_block_idx][head_idx][
                            :, :, nl.ds(h_block_f_tile.start_offset, h_block_f_tile.size)
                        ]
                        nisa.nc_matmul(res_psum, stationary, moving, perf_mode=matmul_perf_mode.double_row)

                h_offset = h_block_idx * h_block_size + h_block_f_tile.start_offset
                out_sb_slice = out_sb[:, nl.ds(h_offset, h_block_f_tile.size)]
                res_psum_slice = res_psum[:, : h_block_f_tile.size]

                # Apply dequant
                if cfg.quantization_type == QuantizationType.ROW:
                    nisa.tensor_tensor(
                        dst=out_sb_slice,
                        data1=res_psum_slice,
                        data2=weight_scale_blocks[h_block_idx][
                            : bxs_block.size, nl.ds(h_block_f_tile.start_offset, h_block_f_tile.size)
                        ],
                        op=nl.multiply,
                    )
                elif cfg.quantization_type == QuantizationType.STATIC:
                    nisa.activation(
                        dst=out_sb_slice,
                        data=res_psum_slice,
                        op=nl.copy,
                        scale=weight_scale_sb[: bxs_block.size, :],
                    )

                # Apply bias
                res_slice = out_sb_slice if cfg.is_quantized else res_psum_slice
                if cfg.has_bias:
                    nisa.tensor_tensor(
                        dst=out_sb_slice,
                        data1=res_slice,
                        data2=bias_sb[: bxs_block.size, nl.ds(h_offset, h_block_f_tile.size)],
                        op=nl.add,
                    )
                else:
                    if matmul_idx % 2 == 0:
                        nisa.tensor_copy(dst=out_sb_slice, src=res_slice, engine=nisa.scalar_engine)
                    else:
                        nisa.tensor_copy(dst=out_sb_slice, src=res_slice, engine=nisa.vector_engine)

        if out_hbm_buffer != None:
            nisa.dma_copy(
                dst=out_hbm_buffer[
                    nl.ds(bxs_block.start_offset, bxs_block.size), nl.ds(cfg.prg_id * cfg.h_sharded, cfg.h_sharded)
                ],
                src=out_sb,
            )
        else:
            return out_sb

    return out_hbm_buffer


def _output_projection_tkg_transpose_out_impl(
    out_hbm_buffer: Optional[nl.ndarray],
    bias: Optional[nl.ndarray],
    w_reshaped: nl.ndarray,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    attn_shuffled: nl.ndarray,
    cfg: OutputProjectionTkgConfig,
) -> nl.ndarray:
    """
    Core implementation for transposed output projection.

    Computes attention @ weight + bias with transposed output shape [H_1, H_0, H_2, B*S].
    Uses weights-stationary matmul with attention moving. Packs multiple B*S groups in
    PSUM banks for efficiency.

    Args:
        out_hbm_buffer (Optional[nl.ndarray]): Output buffer in HBM or None if output in SBUF.
        bias (Optional[nl.ndarray]): [1, H], Optional bias tensor in HBM.
        w_reshaped (nl.ndarray): [N, D, H], Weight tensor reshaped.
        quant_config (Optional[Union[StaticQuantConfig, RowQuantConfig]]): Quantization config.
        attn_shuffled (nl.ndarray): [D, N*B*S], Shuffled attention tensor in SBUF.
        cfg (OutputProjectionTkgConfig): Kernel configuration containing dimensions and options.

    Returns:
        nl.ndarray: Output tensor in transposed layout, either from HBM buffer or SBUF.
    """
    bxs_size = cfg.b_size * cfg.s_size
    out_sb = nl.ndarray(
        (cfg.h_1_size, cfg.h_2_size * bxs_size),
        dtype=cfg.io_dtype,
        buffer=nl.sbuf,
    )

    if bias != None:
        bias_sb = nl.ndarray((cfg.h_1_size, cfg.h_2_size), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_sb,
            src=bias.ap(
                pattern=[[cfg.h_2_size, cfg.h_1_size], [1, cfg.h_2_size]],
                offset=cfg.prg_id * cfg.h_1_size * cfg.h_2_size,
            ),
        )

    if cfg.quantization_type == QuantizationType.STATIC:
        weight_scale_sb = quant_config.combined_scale_sb
    elif cfg.quantization_type == QuantizationType.ROW:
        weight_scale_hbm = quant_config.weight_scale_hbm
        weight_scale_sb = nl.ndarray(
            (cfg.h_1_size, cfg.h_2_size), dtype=weight_scale_hbm.dtype, buffer=nl.sbuf, name="weight_scale_sb"
        )
        nisa.dma_copy(
            dst=weight_scale_sb,
            src=weight_scale_hbm.ap(
                pattern=[[cfg.h_2_size, cfg.h_1_size], [1, cfg.h_2_size]],
                offset=cfg.prg_id * cfg.h_1_size * cfg.h_2_size,
            ),
        )

    # Load weights at once.
    w_sbuf = nl.ndarray(
        (cfg.d_size, cfg.n_size, cfg.h_1_size * cfg.h_2_size),
        dtype=w_reshaped.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=w_sbuf.ap(
            pattern=[
                [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                [cfg.h_1_size * cfg.h_2_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ]
        ),
        src=w_reshaped.ap(
            pattern=[
                [cfg.h_size, cfg.d_size],
                [cfg.d_size * cfg.h_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ],
            offset=(cfg.h_1_size * cfg.h_2_size) * cfg.prg_id,
        ),
    )

    """
    Tiling Strategy:
    - Weights stationary, input (attn_shuffled) moving in, with B*S on free-dim each time
    - When B*S <= F_MAX: Pack multiple B*S groups (different h_2 values) in one PSUM bank
    - When B*S > F_MAX: Tile B*S across ceil(B*S / F_MAX) PSUM tiles

    Memory Usage:
    - PSUM: cfg.h_1_size * F_MAX * sizeof(float32) per tile
    """
    if bxs_size <= F_MAX:
        NUM_BS_PER_PSUM_BANK = F_MAX // bxs_size
        NUM_PSUM_BANKS_PER_BS = 1
        NUM_PSUM_TILES = div_ceil(cfg.h_2_size, NUM_BS_PER_PSUM_BANK)
    else:
        NUM_PSUM_BANKS_PER_BS = div_ceil(bxs_size, F_MAX)
        NUM_BS_PER_PSUM_BANK = 1
        NUM_PSUM_TILES = cfg.h_2_size * NUM_PSUM_BANKS_PER_BS

    for psum_tile_idx in affine_range(NUM_PSUM_TILES):
        # Compute h_2_base and bs_tile_idx from psum_tile_idx
        if bxs_size <= F_MAX:
            h_2_base = psum_tile_idx * NUM_BS_PER_PSUM_BANK
            bs_tile_idx = 0
        else:
            h_2_base, bs_tile_idx = divmod(psum_tile_idx, NUM_PSUM_BANKS_PER_BS)

        bxs_tile_offset = bs_tile_idx * F_MAX
        bxs_tile_size = min(F_MAX, bxs_size - bxs_tile_offset)

        res_psum = nl.ndarray((cfg.h_1_size, F_MAX), dtype=nl.float32, buffer=nl.psum)
        # Accumulate (cfg.h_1_size, F_MAX) sized attn @ weight blocks for all cfg.n_size heads.
        # When B*S <= F_MAX: pack multiple B*S groups in one PSUM row. Iterates once when B*S > F_MAX.
        for bs_group_idx in affine_range(NUM_BS_PER_PSUM_BANK):
            h_2_idx = h_2_base + bs_group_idx

            psum_f_offset = bs_group_idx * bxs_size if bxs_size <= F_MAX else 0
            curr_bxs_size = bxs_size if bxs_size <= F_MAX else bxs_tile_size

            if not cfg.use_double_row:
                for head_idx in affine_range(cfg.n_size):
                    moving = attn_shuffled[:, nl.ds(head_idx * bxs_size + bxs_tile_offset, curr_bxs_size)]
                    stationary = w_sbuf.ap(
                        pattern=[
                            [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                            [cfg.h_2_size, cfg.h_1_size],
                        ],
                        offset=head_idx * (cfg.h_1_size * cfg.h_2_size) + h_2_idx,
                    )
                    # if cfg.h_2_size is not divisible by NUM_BS_PER_PSUM_BANK,
                    # the last psum tile may be "incomplete".
                    if h_2_idx < cfg.h_2_size:
                        nisa.nc_matmul(
                            dst=res_psum[:, nl.ds(psum_f_offset, curr_bxs_size)],
                            stationary=stationary,
                            moving=moving,
                        )
            else:
                for head_idx in affine_range(cfg.n_size // 2):
                    moving = attn_shuffled[:, :, nl.ds(head_idx * bxs_size + bxs_tile_offset, curr_bxs_size)]
                    stationary = w_sbuf.ap(
                        pattern=[
                            [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                            [cfg.h_1_size * cfg.h_2_size, 2],
                            [cfg.h_2_size, cfg.h_1_size],
                        ],
                        offset=head_idx * 2 * (cfg.h_1_size * cfg.h_2_size) + h_2_idx,
                    )
                    # if cfg.h_2_size is not divisible by NUM_BS_PER_PSUM_BANK,
                    # the last psum tile may be "incomplete".
                    if h_2_idx < cfg.h_2_size:
                        nisa.nc_matmul(
                            dst=res_psum[:, nl.ds(psum_f_offset, curr_bxs_size)],
                            stationary=stationary,
                            moving=moving,
                            perf_mode=matmul_perf_mode.double_row,
                        )

        # Last psum tile may be "incomplete" if cfg.h_2_size is not divisible by NUM_BS_PER_PSUM_BANK
        num_BS_for_current_psum_tile = min(NUM_BS_PER_PSUM_BANK, cfg.h_2_size - h_2_base)

        dst_sb_access_pattern = [
            [cfg.h_2_size * bxs_size, cfg.h_1_size],
            [bxs_size, num_BS_for_current_psum_tile] if bxs_size <= F_MAX else [1, bxs_tile_size],
            [1, bxs_tile_size] if bxs_size <= F_MAX else [0, 1],
        ]
        dst_sb_access_offset = h_2_base * bxs_size + bxs_tile_offset
        psum_access_pattern = [
            [F_MAX, cfg.h_1_size],
            [bxs_tile_size, num_BS_for_current_psum_tile] if bxs_size <= F_MAX else [1, bxs_tile_size],
            [1, bxs_tile_size] if bxs_size <= F_MAX else [0, 1],
        ]
        bias_access_pattern = [
            [cfg.h_2_size, cfg.h_1_size],
            [1, num_BS_for_current_psum_tile],
            [0, bxs_tile_size],
        ]
        bias_access_offset = h_2_base

        # Apply dequant
        if cfg.quantization_type == QuantizationType.ROW:
            # Access pattern for row scale: [h_1_size, h_2_size] -> broadcast across B*S
            weight_scale_access_pattern = [
                [cfg.h_2_size, cfg.h_1_size],
                [1, num_BS_for_current_psum_tile],
                [0, bxs_tile_size],
            ]
            weight_scale_access_offset = bias_access_offset
            nisa.tensor_tensor(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                data1=res_psum.ap(pattern=psum_access_pattern),
                data2=weight_scale_sb.ap(
                    pattern=weight_scale_access_pattern,
                    offset=weight_scale_access_offset,
                ),
                op=nl.multiply,
            )
        elif cfg.quantization_type == QuantizationType.STATIC:
            nisa.activation(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                op=nl.copy,
                data=res_psum.ap(pattern=psum_access_pattern),
                scale=weight_scale_sb[: cfg.h_1_size, :],
            )

        # Apply bias
        res_slice = (
            out_sb.ap(pattern=dst_sb_access_pattern, offset=dst_sb_access_offset)
            if cfg.is_quantized
            else res_psum.ap(pattern=psum_access_pattern)
        )
        if cfg.has_bias:
            nisa.tensor_tensor(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                data1=res_slice,
                data2=bias_sb.ap(
                    pattern=bias_access_pattern,
                    offset=bias_access_offset,
                ),
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                src=res_slice,
            )

    # Store out as transposed.
    out_sb = out_sb.reshape((cfg.h_1_size, cfg.h_2_size, bxs_size))

    if out_hbm_buffer != None:
        nisa.dma_copy(
            dst=out_hbm_buffer.ap(
                pattern=[
                    [cfg.h_0_size * cfg.h_2_size * bxs_size, cfg.h_1_size],
                    [bxs_size, cfg.h_2_size],
                    [1, bxs_size],
                ],
                offset=cfg.prg_id * cfg.h_2_size * bxs_size,
            ),
            src=out_sb,
        )
        return out_hbm_buffer
    else:
        return out_sb
