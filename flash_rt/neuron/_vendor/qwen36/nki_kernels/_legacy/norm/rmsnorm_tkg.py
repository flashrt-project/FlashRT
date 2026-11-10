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

"""RMSNorm kernel optimized for token generation (decoding) phase with efficient sharding and memory management.

Vendored from nkilib/core/subkernels/rmsnorm_tkg.py @ nkilib 0.1.0+gd4920de
(Mar 17 2026 build). Only edits relative to upstream:
  1. `from ..utils.<x>` rewritten to absolute `from nkilib.core.utils.<x>` so
     the file works outside the `nkilib.core.subkernels` package.
  2. `from .norm_tkg_utils import …` rewritten to absolute path.

No `name=` argument changes required: every `name=` literal in this file
is on `sbm.alloc_*`, which `BufferManager._get_prefixed_name` already
auto-prefixes via `sbm.set_name_prefix(...)`. The megakernel sets
`L{layer_idx}_moe_` on the outer sbm before calling rmsnorm_tkg.
"""

from typing import Optional, Tuple, Union

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.logging import get_logger
from nkilib.core.utils.tensor_view import TensorView
from nkilib.core.utils.tiled_range import TiledRange
from nkilib.core.subkernels.norm_tkg_utils import load_gamma_to_sbuf, load_input_to_sbuf, validate_shapes, validate_shapes_shard_on_h

# Minimum BxS size to enable sharding (balances computation vs communication overhead)
SHARDING_THRESHOLD = 18

# Tile size for BxS dimension processing
BxS_FULL_TILE_SIZE = 512


def rmsnorm_tkg(
    input: Union[TensorView, nl.ndarray],
    gamma: Union[TensorView, nl.ndarray],
    output: Union[TensorView, nl.ndarray],
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm implementation optimized for inference token generation (decoding) phase.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size

    Args:
        input (Union[TensorView, nl.ndarray]): [B, S, H] when in HBM or [128, B×S, H//128] when in SBUF, Input tensor
        gamma (Union[TensorView, nl.ndarray]): [1, H], Gamma tensor used in normalization on HBM
        output (Union[TensorView, nl.ndarray]): [128, B×S, H//128], Output tensor
        eps (float): Epsilon to maintain numerical stability. Default is 1e-6
        hidden_actual (Optional[int]): Actual hidden dimension size for mean calculation
            when input is padded. Default is None
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128); if False, (128, H/128). Default is False
        single_core_forced (bool): If True, force single-core execution. Default is False
        shard_on_h (bool): If True, shard computation along H dimension instead of BxS. Default is False
        use_heap_memory (bool): Indicates whether to allocate memory on heap instead of stack. Default is False
        sbm (Optional[SbufManager]): Instance of SbufManager responsible for handling sbuf allocation. Default is None

    Returns:
        output (nl.ndarray): [128, BxS, H//128], Output tensor with RMSNorm applied

    Notes:
        - H must be divisible by 128 (partition dimension).
        - When LNC=2 and BxS > SHARDING_THRESHOLD, computation is sharded across cores.
        - Output layout is transposed for efficient downstream sharded matmul.
        - shard_on_h=True requires LNC=2 and is incompatible with hidden_dim_tp and single_core_forced.

    Pseudocode:
        result = RMSNorm(hidden, gamma, eps)
        result = result.reshape((BxS, -1))
        t0 = result[:, 0:H//LNC_SIZE]
        t1 = result[:, H//LNC_SIZE:]
        t0 = t0.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        t1 = t1.reshape((BxS, 128, H//128//LNC_SIZE)).transpose((1, 0, 2))
        result = np.concatenate([t0, t1], axis=2)
    """

    input_view = TensorView(input) if not isinstance(input, TensorView) else input
    gamma_view = TensorView(gamma) if not isinstance(gamma, TensorView) else gamma
    output_view = TensorView(output) if not isinstance(output, TensorView) else output

    if not sbm:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            logger=get_logger("rmsnorm_tkg"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="rmsnorm_tkg")

    if shard_on_h:
        kernel_assert(not hidden_dim_tp, "hidden_dim_tp is not supported in shard on H implementation")
        kernel_assert(not single_core_forced, "single_core_forced is not supported in shard on H implementation")
        _rmsnorm_tkg_shard_on_h(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=False,
            single_core_forced=False,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )
    else:
        _rmsnorm_tkg_shard_on_bxs(
            input_view=input_view,
            gamma_view=gamma_view,
            output_view=output_view,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=hidden_dim_tp,
            single_core_forced=single_core_forced,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    sbm.close_scope()

    if isinstance(output, TensorView):
        return output_view
    else:
        return output


def _rmsnorm_tkg_shard_on_bxs(
    input_view: TensorView,
    gamma_view: TensorView,
    output_view: TensorView,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm with sharding on the BxS dimension.

    Distributes work across LNC cores by splitting the batch-sequence dimension.
    Each core processes its shard independently, then results are exchanged if
    the output is in SBUF.

    Args:
        input_view (TensorView): [B, S, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view.
        gamma_view (TensorView): [1, H], Gamma tensor view.
        output_view (TensorView): [H0, BxS, H1], Output tensor view.
        eps (float): Epsilon for numerical stability.
        hidden_actual (Optional[int]): Actual hidden dimension size for padded inputs.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        single_core_forced (bool): If True, force single-core execution.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1 = validate_shapes(input_view, gamma_view, output_view)

    if output_view.is_sbuf():
        output_sb_view = output_view
    else:
        alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack
        output_sb = alloc_tensor((H0, BxS, H1), dtype=input_view.dtype, buffer=nl.sbuf, name="rmsnorm_output_sb")
        output_sb_view = TensorView(output_sb)

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    num_shards = lnc if not single_core_forced else 1
    do_shard = num_shards == 2 and BxS > SHARDING_THRESHOLD and BxS % lnc == 0
    if not do_shard:
        num_shards, shard_id = 1, 0

    # When single_core_forced=True, use num_shards (1) for H sharding
    # When single_core_forced=False, use lnc for H sharding (original behavior)
    num_H_shards_in_output_H_dim = 1 if hidden_dim_tp or single_core_forced else lnc

    shard_size = BxS // num_shards

    if not input_view.is_sbuf():
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=0, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
    else:
        input_view_sharded = input_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    output_view_sharded = output_sb_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)

    _rmsnorm_tkg_llama_impl(
        input=input_view_sharded,
        gamma=gamma_view,
        output=output_view_sharded,
        num_H_shards=num_H_shards_in_output_H_dim,
        hidden_actual=H if not hidden_actual else hidden_actual,
        eps=eps,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=False,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.is_sbuf():
        if do_shard:
            output_view_sharded_other_core = output_sb_view.slice(
                dim=1, start=(1 - shard_id) * shard_size, end=(2 - shard_id) * shard_size
            )
            nisa.sendrecv(
                dst=output_view_sharded_other_core.get_view(),
                src=output_view_sharded.get_view(),
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
            )
    else:
        output_hbm_view_sharded = output_view.slice(dim=1, start=shard_id * shard_size, end=(shard_id + 1) * shard_size)
        nisa.dma_copy(dst=output_hbm_view_sharded.get_view(), src=output_view_sharded.get_view())
        if use_heap_memory:
            sbm.pop_heap()  # dealloc output_sb


def _rmsnorm_tkg_shard_on_h(
    input_view: TensorView,
    gamma_view: TensorView,
    output_view: TensorView,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = False,
    single_core_forced: bool = False,
    use_heap_memory: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm with sharding on the H (hidden) dimension.

    Each LNC core processes a shard of the hidden dimension. Partial sums for
    RMS are exchanged between cores via sendrecv to compute the full statistics
    before normalization.

    Args:
        input_view (TensorView): [B, S, H] when in HBM or [H0, BxS, sharded_H1] when in SBUF, Input tensor view.
        gamma_view (TensorView): [1, H], Gamma tensor view.
        output_view (TensorView): [H0, BxS, H1] or [H0, BxS, sharded_H1] if in SBUF, Output tensor view.
        eps (float): Epsilon for numerical stability.
        hidden_actual (Optional[int]): Actual hidden dimension size for padded inputs.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        single_core_forced (bool): If True, force single-core execution.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack.
        sbm (Optional[SbufManager]): SBUF memory manager instance.

    Returns:
        None: Results written to output_view.
    """
    BxS, H, H0, H1, shard_H, shard_H1 = validate_shapes_shard_on_h(input_view, gamma_view, output_view)

    if output_view.is_sbuf():
        output_sb_view = output_view
    else:
        alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack
        output_sb = alloc_tensor((H0, BxS, shard_H1), dtype=input_view.dtype, buffer=nl.sbuf, name="rmsnorm_output_sb")
        output_sb_view = TensorView(output_sb)

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    input_view_sharded = input_view
    if not input_view.is_sbuf():
        input_view_flat = input_view.flatten_dims(start_dim=0, end_dim=1)
        input_view_sharded = input_view_flat.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    if not output_view.is_sbuf():
        output_view_flat = output_view.flatten_dims(start_dim=0, end_dim=1)
        output_view_sharded = output_view_flat.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    gamma_view = gamma_view.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    _rmsnorm_tkg_llama_impl(
        input=input_view_sharded,
        gamma=gamma_view,
        output=output_sb_view,
        num_H_shards=lnc,
        hidden_actual=H if not hidden_actual else hidden_actual,
        eps=eps,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=True,
        use_heap_memory=use_heap_memory,
        sbm=sbm,
    )

    if output_view.is_sbuf():
        output_view = output_sb_view
    else:
        output_hbm_view_sharded = output_view.slice(dim=2, start=shard_id * shard_H1, end=(shard_id + 1) * shard_H1)
        nisa.dma_copy(dst=output_hbm_view_sharded.get_view(), src=output_sb_view.get_view())


def _process_rmsnorm_tile(
    input_sb_view: TensorView,
    gamma_sb_view: TensorView,
    output_sb_view: TensorView,
    eps_view: TensorView,
    matmul_reduction_const_view: TensorView,
    bxs_tile: Tuple,
    hidden_actual: int,
    shard_on_h: bool,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Process a single tile of RMSNorm computation.

    Args:
        input_sb_view (TensorView): [H0, BxS, H1], Input tensor view in SBUF
        gamma_sb_view (TensorView): [H0, BxS, H1], Gamma tensor view in SBUF (broadcasted)
        output_sb_view (TensorView): [H0, BxS, H1], Output tensor view in SBUF
        eps_view (TensorView): [H0, 1], Epsilon value for numerical stability
        matmul_reduction_const_view (TensorView): [H0, H0], Constant matrix for reduction
        bxs_tile (Tuple): Tile information containing index and size
        hidden_actual (int): Actual hidden dimension size
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack
        sbm (SbufManager): SBUF memory manager instance

    Returns:
        None: Results written directly to output_sb_view

    Notes:
        - Computes RMSNorm: output = (input * gamma) / sqrt(mean(input^2) + eps)
        - Uses intermediate float32 precision for numerical stability
    """
    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    sbm.open_scope()

    num_allocated_tensor = 0
    inter_dtype = nl.float32

    kernel_assert(
        input_sb_view.shape == output_sb_view.shape, "Input and output tensor shapes must match for RMSNorm processing"
    )
    H0, BxS, H1 = input_sb_view.shape

    # Compute x^2 for RMS calculation
    rmsnorm_square = alloc_tensor(
        shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name=f"rmsnorm_square_{bxs_tile.index}"
    )
    num_allocated_tensor += 1
    nisa.activation(rmsnorm_square[...], op=nl.square, data=input_sb_view.get_view())

    # Reduce along H1 dimension to compute sum(x^2)
    rmsnorm_reduced_square = alloc_tensor(
        shape=(H0, BxS),
        dtype=inter_dtype,
        buffer=nl.sbuf,
        name=f"rmsnorm_reduced_square_{bxs_tile.index}",
    )
    num_allocated_tensor += 1
    nisa.tensor_reduce(rmsnorm_reduced_square[...], nl.add, rmsnorm_square[...], axis=2)

    if shard_on_h:
        _, _, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))
        # Exchange partial sums with the other core via sendrecv
        remote_reduced_square = alloc_tensor(
            shape=(H0, BxS),
            dtype=inter_dtype,
            buffer=nl.sbuf,
            name=f"rmsnorm_remote_reduced_square_{bxs_tile.index}",
        )
        num_allocated_tensor += 1
        nisa.sendrecv(
            dst=remote_reduced_square[...],
            src=rmsnorm_reduced_square[...],
            send_to_rank=1 - shard_id,
            recv_from_rank=1 - shard_id,
            pipe_id=0,
        )

        # Combine partial sums: full_sq = local_sq + remote_sq
        nisa.tensor_tensor(
            rmsnorm_reduced_square[...],
            rmsnorm_reduced_square[...],
            remote_reduced_square[...],
            nl.add,
        )

    # Apply gamma scaling: input * gamma
    gamma_mult = alloc_tensor(
        shape=(H0, BxS, H1), dtype=inter_dtype, buffer=nl.sbuf, name=f"rmsnorm_gamma_mult_{bxs_tile.index}"
    )
    num_allocated_tensor += 1
    gamma_mult_view = TensorView(gamma_mult)
    nisa.tensor_tensor(
        gamma_mult_view.get_view(),
        input_sb_view.get_view(),
        gamma_sb_view.get_view(),
        nl.multiply,
    )

    # Complete reduction across H0 dimension using matmul
    if sbm.is_auto_alloc():
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum)
    else:
        final_reduced = nl.ndarray((H0, BxS), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(
        stationary=matmul_reduction_const_view.get_view(),
        moving=rmsnorm_reduced_square,
        dst=final_reduced,
    )

    # Compute normalization factor: 1/sqrt(mean(x^2) + eps)
    hidden_scale = 1.0 / hidden_actual
    nisa.activation(
        rmsnorm_reduced_square[...],
        op=nl.rsqrt,
        data=final_reduced[...],
        scale=hidden_scale,
        bias=eps_view.get_view(),
    )

    # Final RMSNorm: (input * gamma) * normalization_factor
    reduced_view = TensorView(rmsnorm_reduced_square).expand_dim(dim=2).broadcast(dim=2, size=H1)
    nisa.tensor_tensor(
        output_sb_view.get_view(),
        gamma_mult_view.get_view(),
        reduced_view.get_view(),
        nl.multiply,
    )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()

    sbm.close_scope()


def _rmsnorm_tkg_llama_impl(
    input: TensorView,
    gamma: TensorView,
    output: TensorView,
    num_H_shards: int,
    hidden_actual: Optional[int],
    eps: float,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    use_heap_memory: bool = False,
    sbm: SbufManager = None,
):
    """
    Perform RMSNorm on input tensor with sharding support.

    Args:
        input (TensorView): [BxS, H] when in HBM or [H0, BxS, H1] when in SBUF, Input tensor view
        gamma (TensorView): [1, H], Gamma tensor view
        output (TensorView): [H0, BxS, H1], Output tensor view in SBUF
        num_H_shards (int): Number of shards along H dimension
        hidden_actual (Optional[int]): Actual hidden dimension size
        eps (float): Epsilon for numerical stability
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128); if False, (128, H/128)
        shard_on_h (bool): If True, exchange partial sums between cores.
        use_heap_memory (bool): If True, allocate on heap; otherwise on stack
        sbm (SbufManager): SBUF memory manager instance

    Returns:
        None: Results written directly to output tensor view

    Notes:
        - Uses Static DMA for input data reads (superior performance vs DGE)
        - For LNC sharding: input [B, S, lnc, H//lnc] reshaped to [BxS, lnc, H0, H1//lnc]
        - After transpose: [H0, BxS, lnc, H1//lnc] reshaped back to [H0, BxS, H1]
        - RMSNorm performed on combined [H0, lnc, H1//lnc] dimension
    """

    if input.is_sbuf():
        H0, BxS, H1 = input.shape
        H = H0 * H1
    else:
        BxS, H = input.shape
        H0 = nl.tile_size.pmax
        H1 = H // H0

    if hidden_dim_tp:
        kernel_assert(num_H_shards == 1, "When hidden_dim_tp is True, num_H_shards must be 1")

    inter_dtype = nl.float32

    alloc_tensor = sbm.alloc_heap if use_heap_memory else sbm.alloc_stack

    num_allocated_tensor = 0

    # Load input and reuse output_buffer
    if input.is_sbuf():
        input_sb_view = input
    else:
        input_sb_view = load_input_to_sbuf(
            input_hbm=input,
            input_sb=output,
            num_H_shards=num_H_shards,
            hidden_dim_tp=hidden_dim_tp,
            shard_on_h=shard_on_h,
            sbm=sbm,
        )

    # Load gamma
    # if hidden_dim_tp is on, rmsnorm_gamma offset needs to be 32B aligned
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(shape=(H0, H1), dtype=gamma.dtype, name="rmsnorm_gamma", align=gamma_align)
    num_allocated_tensor += 1
    gamma_sb_view = load_gamma_to_sbuf(
        gamma_hbm=gamma,
        gamma_sb=TensorView(gamma_sb),
        num_H_shards=num_H_shards,
        hidden_dim_tp=hidden_dim_tp,
        shard_on_h=shard_on_h,
    )

    # Load eps
    eps_sb = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_eps")
    num_allocated_tensor += 1
    nisa.memset(eps_sb, value=eps)

    # Load matmul reduction const
    matmul_reduction_const = alloc_tensor(
        shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_mm_reduced_const"
    )
    num_allocated_tensor += 1
    nisa.memset(matmul_reduction_const, value=1.0)

    for bxs_tile in TiledRange(BxS, BxS_FULL_TILE_SIZE):
        input_sb_view_tile = input_sb_view.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        gamma_sb_view_tile = gamma_sb_view.expand_dim(dim=1).broadcast(dim=1, size=bxs_tile.size)
        output_sb_view_tile = output.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        _process_rmsnorm_tile(
            input_sb_view=input_sb_view_tile,
            gamma_sb_view=gamma_sb_view_tile,
            output_sb_view=output_sb_view_tile,
            eps_view=TensorView(eps_sb),
            matmul_reduction_const_view=TensorView(matmul_reduction_const),
            bxs_tile=bxs_tile,
            hidden_actual=hidden_actual,
            shard_on_h=shard_on_h,
            use_heap_memory=use_heap_memory,
            sbm=sbm,
        )

    if use_heap_memory:
        for _ in range(num_allocated_tensor):
            sbm.pop_heap()
