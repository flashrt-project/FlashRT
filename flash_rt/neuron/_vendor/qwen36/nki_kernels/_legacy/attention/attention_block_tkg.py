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
Attention Block TKG Kernel

This kernel implements the attention block for token generation (TKG), fusing all stages in SBUF to avoid HBM round-trips.

It performs:
        +-------------------+   +-------------------+   +-------------------+   +-------------------+
        | Input X (HBM)     |-->|   RMSNorm X       |-->| QKV Projection    |-->|  Split QKV → Q,K  |-->
        | [B, S_tkg, H]     |   | (optional)        |   |                   |   |   (transpose)     |
        +-------------------+   +-------------------+   +-------------------+   +-------------------+

        +-------------------+   +-------------------+   +-------------------+   +-------------------+
    --> | RMSNorm Q/K       |-->|   RoPE Embedding  |-->| RMSNorm Q/K       |-->| Quantize K/V      |-->
        | (optional)        |   | (optional)        |   | (optional)        |   | to FP8 (optional) |
        +-------------------+   +-------------------+   +-------------------+   +-------------------+

        +-------------------+   +-----------------------+   +--------------------+   +-------------------+   +-------------------+
    --> | KVDP Input Gather |-->| Attention TKG         |-->| KVDP Output Gather |-->| KV-Cache Update   |-->| Output Projection |
        | (optional)        |   | (softmax(Q·Kᵀ/√d) @ V)|   | (optional)         |   | (optional)        |   | (optional)        |
        +-------------------+   +-----------------------+   +--------------------+   +-------------------+   +-------------------+

Features:
- Supports grouped-query attention (GQA) with a single key/value head
- LNC-2 sharding support
- KV data parallelism (KVDP) for multi-rank inference with sharded KV cache
- Operates with or without output projection
- Optimized for small batch_size * sequence length typical in decoding
- Optional FP8 KV cache quantization for memory-efficient inference
"""

from typing import Any, Dict, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

try:
    from nki.collectives import ReplicaGroup
except ImportError:
    ReplicaGroup = None  # not available in simulation runtime
from nki.isa.constants import oob_mode

from .attention_tkg import AttnTKGConfig, attention_tkg
from nkilib.core.attention.attention_tkg_utils import is_fp8_e4m3
from nkilib.core.embeddings.rope import RoPE_sbuf
from .output_projection_tkg import output_projection_tkg
from nkilib.core.qkv.qkv import qkv
from nkilib.core.utils.allocator import SbufManager, create_auto_alloc_manager
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_max_positive_value_for_dtype,
    get_verified_program_sharding_info,
    is_hbm_buffer,
)
from nkilib.core.utils.logging import Logger
from nkilib.core.utils.tensor_view import TensorView
from nkilib.experimental.transformer.attention_block_tkg_sharding import (
    _KVDP_attention_input_collectives,
    _KVDP_attention_output_collectives,
)


# TODO(NKI-699): Refactor API to use configuration dataclasses for better clarity
# Note: Using keyword-only args (via *) to avoid breaking callers when adding/reordering
# parameters, and to improve readability given the large number of arguments.
@nki.jit
def attention_block_tkg(
    # -- input
    X: nl.ndarray,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool,
    rmsnorm_QK_pre_rope_eps: float,
    rmsnorm_QK_pre_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_pre_rope_W_K: Optional[nl.ndarray],
    # -- Q/K processing: RoPE
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    rmsnorm_QK_post_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_post_rope_W_K: Optional[nl.ndarray],
    # -- attention
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: Optional[nl.ndarray],
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    transposed_out: bool,
    # -- output
    out_in_sb: bool,
    # -- optional params with defaults
    softmax_scale: Optional[float] = None,
    enable_fa_s_prior_tiling: bool = True,
    k_scale: Optional[nl.ndarray] = None,
    v_scale: Optional[nl.ndarray] = None,
    sbm: Optional[SbufManager] = None,
    skip_attention: bool = False,
    KVDP: int = 1,
    KVDP_replica_group: Optional[ReplicaGroup] = None,
):
    """
    Fused Attention Block for Token Generation (TKG).

    Performs end-to-end attention block computation optimized for autoregressive
    decoding with all stages fused in SBUF to avoid HBM round-trips. Intended for
    small batch sizes (B ≤ 16) and short sequence lengths (S_tkg ≤ 8) typical in
    token generation workloads.

    Dimensions:
        B: Batch size (≤ 16 recommended)
        B_attn: Batch size for attention = B/KVDP when KV data parallelism enabled, otherwise B
        S_tkg: Number of new tokens to generate (≤ 8 required)
        S_ctx: KV cache sequence length in current bucket
        S_max_ctx: Maximum KV cache capacity of current bucket
        H: Hidden dimension (must be multiple of 128)
        d_head: Head dimension (must be even)
        q_heads: Number of query heads
        kv_heads: 1 (GQA with single KV head)
        num_blocks: Number of blocks in block KV cache
        block_len: Block length for block KV cache

    Args:
        X (nl.ndarray): Input hidden states
            Shape:
                [B, S_tkg, H]                                when in HBM
                [H0=pmax, BxS, H1] where H1=lnc x (H//lnc//pmax) when in SBUF

            When in SBUF, the layout is obtained by rearranging HBM data:
                HBM: (BxS, lnc, H0, H1//lnc) -> SBUF: (H0, BxS, (lnc, H1//lnc))
            This interleaves H1//lnc values from each lnc chunk along the H dimension,
            matching qkv_tkg() kernel's expected SBUF input format.
        X_hidden_dim_actual (Optional[int]): Actual hidden dim if X is padded

        rmsnorm_X_enabled (bool): Apply RMSNorm to X before QKV projection
        rmsnorm_X_eps (Optional[float]): RMSNorm epsilon (default 1e-3)
        rmsnorm_X_gamma (Optional[nl.ndarray]): [1, H] @ HBM, RMSNorm weights

        W_qkv (nl.ndarray): [H, d_head*(q_heads+2)] @ HBM, QKV projection weights
        bias_qkv (Optional[nl.ndarray]): [1, d_head*(q_heads+2)] @ HBM, QKV bias
        quantization_type_qkv (QuantizationType): Type of quantization for QKV projection (NONE, STATIC).
        weight_dequant_scale_qkv (Optional[nl.ndarray]): Weight dequantization scale for QKV projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_qkv is STATIC.
        input_dequant_scale_qkv (Optional[nl.ndarray]): Input dequantization scale for QKV projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_qkv is STATIC.

        rmsnorm_QK_pre_rope_enabled (bool): Apply RMSNorm to Q/K before RoPE
        rmsnorm_QK_pre_rope_eps (float): Pre-RoPE RMSNorm epsilon
        rmsnorm_QK_pre_rope_W_Q (Optional[nl.ndarray]): [1, d_head] @ HBM, Pre-RoPE Q gamma weights
        rmsnorm_QK_pre_rope_W_K (Optional[nl.ndarray]): [1, d_head] @ HBM, Pre-RoPE K gamma weights
        cos (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE cosine embeddings (None = skip RoPE)
        sin (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout (bool): True for contiguous halves, False for interleaved
        rmsnorm_QK_post_rope_enabled (bool): Apply RMSNorm to Q/K after RoPE
        rmsnorm_QK_post_rope_eps (float): Post-RoPE RMSNorm epsilon
        rmsnorm_QK_post_rope_W_Q (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE Q weights
        rmsnorm_QK_post_rope_W_K (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE K weights

        K_cache_transposed (bool): Whether K cache is stored transposed in HBM.
            If True: K cache is [B, d_head, S_ctx]. If False: K cache is [B, S_ctx, d_head].
            Must be False for block KV cache.
        active_blocks_table (Optional[nl.ndarray]): [B, num_blocks] @ HBM, Block indices for block KV cache
        K_cache (nl.ndarray): Key cache @ HBM.
            Flat KV: [B, d_head, S_max_ctx] if K_cache_transposed else [B, S_max_ctx, d_head].
            Block KV: [num_blocks, block_len, d_head].
        V_cache (nl.ndarray): Value cache @ HBM.
            Flat KV: [B, S_max_ctx, d_head].
            Block KV: [num_blocks, block_len, d_head].
        attention_mask (nl.ndarray): [S_ctx, B, q_heads, S_tkg] @ HBM, Attention mask
        sink (Optional[nl.ndarray]): [H, 1] @ HBM, Attention sink tokens
        softmax_scale (Optional[float]): Scaling factor for attention scores. If None, defaults to 1/sqrt(d_head).
        enable_fa_s_prior_tiling: bool: Whether to enable flash attention (FA) for attention computation.
            When enabled, the attention computation is tiled along the context (s_prior) to reduce peak memory usage.

        k_scale (Optional[nl.ndarray]): Scale for K quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM.
            Must contain a single scalar value (replicated or scalar). When provided with v_scale,
            enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.
        v_scale (Optional[nl.ndarray]): Scale for V quantization to FP8. Shape (PMAX, 1) or (1, 1) @ HBM.
            Must contain a single scalar value (replicated or scalar). When provided with k_scale,
            enables FP8 KV cache quantization. Supported dtypes: float32, float16, bfloat16.

        update_cache (bool): Update KV cache with new tokens
        kv_cache_update_idx (Optional[nl.ndarray]): [B, 1], Cache write positions (uint32_max = skip)

        W_out (Optional[nl.ndarray]): [q_heads*d_head, H] @ HBM, Output projection weights
        bias_out (Optional[nl.ndarray]): [1, H] @ HBM, Output projection bias
        quantization_type_out (QuantizationType): Type of quantization for output projection (NONE, STATIC).
        weight_dequant_scale_out (Optional[nl.ndarray]): Weight dequantization scale for output projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_out is STATIC.
        input_dequant_scale_out (Optional[nl.ndarray]): Input dequantization scale for output projection.
            Shape: [PMAX, 1] @ HBM when quantization_type_out is STATIC.
        transposed_out (bool): Transpose output layout (requires W_out)
        out_in_sb (bool): Return output in SBUF instead of HBM
        sbm (Optional[SbufManager]): SBUF memory manager (otherwise auto-allocated)
        skip_attention (bool): Skip attention computation (for testing)

        KVDP (int): KV cache data parallelism degree - number of ranks that shard the KV cache
            across the batch dimension (1 = disabled). Each rank processes B/KVDP batches.
        KVDP_replica_group (Optional[ReplicaGroup]): Replica group for collective ops

    KV Data Parallelism (KVDP > 1):
        KV-DP partitions the KV cache across ranks along the batch dimension. Each rank holds
        B/KVDP batches of the KV cache. Before attention: all_gather Q heads, slice Q/K/V batch.
        After attention: all_gather output batch, slice heads.

        When KV data parallelism is enabled, input/output shapes change:
        - B_attn = B / KVDP (batches per rank for attention)
        - q_heads_attn = q_heads * KVDP (query heads per rank after gather)

        Input shape changes:
        - K_cache, V_cache: [B_attn, ...] instead of [B, ...]
        - attention_mask: [S_ctx, B_attn, q_heads_attn, S_tkg]
        - kv_cache_update_idx: [B_attn, 1] (caller must slice per rank)

        Output shape changes (when update_cache=False):
        - K_out: [d_head, B_attn, S_tkg]
        - V_out: [B_attn, 1, S_tkg, d_head]

    Returns:
        out (nl.ndarray): Output tensor with shape depending on projection and output location:
            - Without projection (W_out=None):
                - out_in_sb=False: [B, q_heads, d_head, S_tkg] @ HBM
                - out_in_sb=True: [d_head, B*q_heads*S_tkg] @ SBUF
            - With projection (W_out provided):
                - transposed_out=False, out_in_sb=False: [B*S_tkg, H] @ HBM
                - transposed_out=False, out_in_sb=True: [B*S_tkg, H//lnc] @ SBUF
                - transposed_out=True, out_in_sb=False: [128, lnc, H//lnc//128, B*S_tkg] @ HBM
                - transposed_out=True, out_in_sb=True: [128, H//lnc//128, B*S_tkg] @ SBUF
        K_out (nl.ndarray):
            - If update_cache=True: Updated K cache (shape matches K_cache input)
            - If update_cache=False: New K tokens [d_head, B_attn, S_tkg] @ HBM
        V_out (nl.ndarray):
            - If update_cache=True: Updated V cache (shape matches V_cache input)
            - If update_cache=False: New V tokens [B_attn, 1, S_tkg, d_head] @ HBM

    Notes:
        - Requires NeuronCore v3+
        - d_head must be even
        - H must be multiple of pmax
        - Requires batch * sequence_tkg * q_heads <= pmax (=pmax)
        - Supports grouped-query attention (GQA) with single key/value head
        - LNC-2 sharding support for KV cache updates

    Pseudocode:
        # Stage 1: QKV Projection
        if rmsnorm_X_enabled:
            X_norm = rms_norm(X, rmsnorm_X_gamma, rmsnorm_X_eps)
        QKV = matmul(X_norm, W_qkv) + bias_qkv

        # Stage 2: Q/K Processing
        Q, K = split_and_transpose(QKV)
        if rmsnorm_QK_pre_rope_enabled:
            Q = rms_norm(Q, rmsnorm_QK_pre_rope_W_Q)
            K = rms_norm(K, rmsnorm_QK_pre_rope_W_K)
        if cos != None and sin != None:
            Q, K = rope(Q, cos, sin), rope(K, cos, sin)
        if rmsnorm_QK_post_rope_enabled:
            Q = rms_norm(Q, rmsnorm_QK_post_rope_W_Q)
            K = rms_norm(K, rmsnorm_QK_post_rope_W_K)
        V = extract_V(QKV)

        # Stage 3: Attention
        Q_scaled = Q / sqrt(d_head)
        attn_out = attention_tkg(Q_scaled, K, V, K_cache, V_cache, attention_mask)

        # Stage 4: KV Cache Update
        if update_cache:
            update_kv_cache(K_cache, V_cache, K, V, kv_cache_update_idx)
            K_out, V_out = K_cache, V_cache  # Return updated caches
        else:
            K_out, V_out = K, V  # Return new tokens

        # Stage 5: Output Projection
        if W_out is not None:
            output = matmul(attn_out, W_out) + bias_out
        else:
            output = attn_out

        return output, K_out, V_out
    """

    # ========== Validation and Setup ==========
    config = _validate_and_extract_config(
        X,
        W_qkv,
        K_cache,
        V_cache,
        attention_mask,
        cos,
        sin,
        rmsnorm_X_gamma,
        K_cache_transposed,
        active_blocks_table,
        W_out,
        k_scale,
        v_scale,
        KVDP,
        KVDP_replica_group,
        out_in_sb,
        skip_attention,
    )

    B, S_tkg = config['B'], config['S_tkg']
    d_head, q_heads = config['d_head'], config['q_heads']
    S_ctx, S_max_ctx = config['S_ctx'], config['S_max_ctx']
    is_block_kv, blk_len = config['is_block_kv'], config['blk_len']
    cache_had_head_dim = config['cache_had_head_dim']
    do_out_proj = config['do_out_proj']
    K_cache, V_cache = config['K_cache'], config['V_cache']
    kv_quant = config['kv_quant']
    B, B_attn, q_heads_attn = config['B'], config['B_attn'], config['q_heads_attn']
    is_KVDP = config['is_KVDP']
    n_bxs_tiles = config['n_bxs_tiles']
    bxs_tile = config['bxs_tile']
    kv_heads = 1
    I = d_head * (q_heads + 2 * kv_heads)

    sbm = sbm if sbm != None else create_auto_alloc_manager(logger=Logger("attn-block-tkg"))
    sbm.open_scope(name="attn-blk-tkg-scope")

    # ========== QKV Projection ==========
    # Input:  X [B, S_tkg, H] @ HBM
    # Output: QKV [B*S_tkg, I] @ SBUF (small batch) or HBM (large batch)
    # where I = d_head * (q_heads + 2)
    # qkv() routes to qkv_tkg (SBUF output) or qkv_cte (HBM output) based on B*S_tkg
    rmsnorm_X_eps = 1e-3 if rmsnorm_X_eps == None else rmsnorm_X_eps
    output_in_sbuf = n_bxs_tiles == 1
    QKV_out = qkv(
        input=X,
        fused_qkv_weights=W_qkv,
        output_layout=QKVOutputLayout.BSD,
        bias=bias_qkv,
        fused_norm_type=NormType.RMS_NORM if rmsnorm_X_enabled else NormType.NO_NORM,
        gamma_norm_weights=rmsnorm_X_gamma,
        norm_eps=rmsnorm_X_eps,
        hidden_actual=X_hidden_dim_actual,
        quantization_type=quantization_type_qkv,
        qkv_w_scale=weight_dequant_scale_qkv,
        qkv_in_scale=input_dequant_scale_qkv,
        d_head=d_head,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        store_output_in_sbuf=output_in_sbuf,
        sbm=sbm,
        use_auto_allocation=True,
    )
    QKV_out = QKV_out.reshape((B * S_tkg, I))

    # ========== Q/K Processing + V extraction: Transpose + RMSNorm pre + RoPE + RMSNorm post, K/V quantization ==========
    # Handles tiling internally: for each tile, loads QKV from HBM → SBUF (large batch)
    # or uses QKV directly from SBUF (small batch), then processes Q/K/V per tile.
    # Input:  QKV_out [B*S_tkg, I] @ SBUF or HBM
    # Output: Q_tkg_sb [d_head, B*q_heads*S_tkg] @ SBUF, K_tkg_sb [d_head, B*S_tkg] @ SBUF,
    #         V_tkg_hbm [B, 1, S_tkg, d_head] @ HBM, V_tkg_sb [B*S_tkg, d_head] @ SBUF (small batch only)
    Q_tkg_sb, K_tkg_sb, V_tkg_hbm, V_tkg_sb = _QKV_processing(
        QKV=QKV_out,
        q_heads=q_heads,
        kv_heads=kv_heads,
        B=B,
        S_tkg=S_tkg,
        d_head=d_head,
        n_bxs_tiles=n_bxs_tiles,
        bxs_tile=bxs_tile,
        rmsnorm_pre_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_pre_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_pre_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_pre_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_post_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_post_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_post_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_post_W_K=rmsnorm_QK_post_rope_W_K,
        kv_quant=kv_quant,
        k_scale=k_scale,
        v_scale=v_scale,
        io_dtype=X.dtype,
        sbm=sbm,
    )

    # ========== KV Data Parallelism: Input Collectives ==========
    if is_KVDP:
        # Gather Q heads, slice Q/K/V batch
        #   B -> B_attn (B/KVDP)
        #   q_heads -> q_heads_attn (q_heads*KVDP)
        # Q: [d, B*q_heads*S] @ SBUF -> [d, B_attn*q_heads_attn*S] @ SBUF
        # K: [d, B*S] @ SBUF -> [d, B_attn*S] @ SBUF
        # V: [B, 1, S, d] @ HBM -> [B_attn, 1, S, d] @ HBM
        Q_tkg_sb, K_tkg_sb, V_tkg_hbm = _KVDP_attention_input_collectives(
            Q_tkg_sb,
            K_tkg_sb,
            V_tkg_hbm,
            q_heads,
            kv_heads,
            d_head,
            KVDP,
            B,
            B_attn,
            S_tkg,
            KVDP_replica_group,
            sbm,
        )

    # ========== Attention Computation ==========
    # Input:  Q_tkg_sb [d_head, B_attn*q_heads_attn*S_tkg] @ SBUF
    #         K_tkg_sb [d_head, B_attn*S_tkg] @ SBUF
    #         V_tkg_hbm [B_attn, 1, S_tkg, d_head] @ HBM
    # Output: attn_out [d_head, B_attn*q_heads_attn*S_tkg] @ SBUF or [B_attn, q_heads_attn, d_head, S_tkg] @ HBM
    if skip_attention:
        attn_out = Q_tkg_sb
    else:
        # Scale Q by softmax_scale (default: 1/sqrt(d_head))
        _softmax_scale = softmax_scale if softmax_scale != None else d_head ** (-0.5)
        nisa.tensor_scalar(dst=Q_tkg_sb, data=Q_tkg_sb, op0=nl.multiply, operand0=_softmax_scale)
        # Allocate attention output buffer
        allocate_attn_out_on_HBM = not do_out_proj and not out_in_sb and not is_KVDP
        if allocate_attn_out_on_HBM:
            attn_out = nl.ndarray(
                (B_attn, q_heads_attn, d_head, S_tkg),
                dtype=X.dtype,
                buffer=nl.shared_hbm,
                name=f"{sbm.get_name_prefix()}attn_v_active_hbm",
            )
        else:  # attn_out @ SBUF
            attn_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=X.dtype, buffer=nl.sbuf)

        # Prepare KV cache views for attention
        if is_block_kv:
            k_prior, v_prior = K_cache, V_cache
        else:
            k_shape = (B_attn, 1, d_head, S_max_ctx) if K_cache_transposed else (B_attn, 1, S_max_ctx, d_head)
            k_prior = K_cache.reshape(k_shape)
            v_prior = V_cache.reshape((B_attn, 1, S_max_ctx, d_head))

        attn_cfg = AttnTKGConfig(
            bs=B_attn,
            q_head=q_heads_attn,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=blk_len if is_block_kv else 0,
            # tp_k_prior = "kernel needs to transpose K_prior". K_cache_transposed means
            # K is already transposed in HBM, so the kernel does NOT need to transpose it.
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=False,
            fuse_rope=False,
            use_gpsimd_sb2sb=True,
            qk_in_sb=True,
            k_out_in_sb=False,
            out_in_sb=do_out_proj or out_in_sb or is_KVDP,
            enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
        )

        attention_tkg(
            q=Q_tkg_sb,
            k_active=K_tkg_sb,
            v_active=V_tkg_hbm,  # Attention_tkg() wants V @ HBM
            k_prior=k_prior,
            v_prior=v_prior,
            mask=attention_mask,
            out=attn_out,  # OUT
            cfg=attn_cfg,
            sbm=sbm,
            sink=sink,
            active_blocks_table=active_blocks_table,
        )

    # ========== KV Data Parallelism: Output Gather ==========
    if is_KVDP:
        # Gather batch, slice heads (restore for output projection):
        #   B_attn (B/KVDP) -> B
        #   q_heads_attn (q_heads*KVDP) -> q_heads
        # attn_out: [d, B_attn*q_heads_attn*S] @ SBUF -> [d, B*q_heads*S] @ SBUF
        # V: [B_attn, 1, S, d] @ HBM -> V_tkg_sb [B_attn*S, d] @ SBUF
        attn_out, V_tkg_sb = _KVDP_attention_output_collectives(
            attn_out,
            V_tkg_hbm,
            KVDP,
            B_attn,
            q_heads,
            d_head,
            S_tkg,
            KVDP_replica_group,
            sbm,
        )

    # ========== KV Cache Update ==========
    # Input:  K_tkg_sb [d_head, B_attn*S_tkg] @ SBUF
    #         V_tkg_sb [B_attn*S_tkg, d_head] @ SBUF
    # Output: K_hbm_out, V_hbm_out (updated caches or new tokens) @ HBM
    if update_cache:
        _kv_cache_update(
            K_cache=K_cache,
            V_cache=V_cache,
            K_tkg=K_tkg_sb,
            V_tkg=V_tkg_sb if V_tkg_sb is not None else V_tkg_hbm,
            kv_cache_update_idx=kv_cache_update_idx,
            B=B_attn,
            d_head=d_head,
            S_tkg=S_tkg,
            S_max_ctx=S_max_ctx,
            K_cache_transposed=K_cache_transposed,
            is_block_kv=is_block_kv,
        )
        K_cache, V_cache = __internal_unsqueeze_head_dim(K_cache, V_cache, cache_had_head_dim, is_block_kv)
    else:  # No cache update: return new K/V tokens
        K_tkg_hbm = sbm.alloc((d_head, B_attn, S_tkg), dtype=K_tkg_sb.dtype, buffer=nl.shared_hbm, name="K_hbm")
        nisa.dma_copy(K_tkg_hbm.reshape(K_tkg_sb.shape), K_tkg_sb)

    # ========== Output Projection (Optional) ==========
    # Input:  attn_out [d_head, B, q_heads, S_tkg] @ SBUF/HBM
    # Output: kernel_output layout depends on transposed_out and out_in_sb
    if do_out_proj:
        kernel_output = output_projection_tkg(
            attention=attn_out.reshape((d_head, B, q_heads, S_tkg)),
            weight=W_out,
            bias=bias_out,
            quantization_type=quantization_type_out,
            weight_scale=weight_dequant_scale_out,
            input_scale=input_dequant_scale_out,
            TRANSPOSE_OUT=transposed_out,
            OUT_IN_SB=out_in_sb,
            sbm=sbm,
        )
    else:
        kernel_assert(not transposed_out, "transposed_out requires output projection (W_out must be provided)")
        kernel_output = attn_out

    # Copy output to HBM if caller expects it on HBM but it's on SBUF. This is only used for debug when skipping both attention and output-projection.
    if out_in_sb == False and kernel_output.buffer == nl.sbuf:
        kernel_output_hbm = nl.ndarray(
            kernel_output.shape, kernel_output.dtype, nl.shared_hbm, name=f"{sbm.get_name_prefix()}kernel_output_hbm"
        )
        nisa.dma_copy(kernel_output_hbm, kernel_output)
        kernel_output = kernel_output_hbm

    # ========== Cleanup and Return ==========
    sbm.close_scope()
    if update_cache:
        return kernel_output, K_cache, V_cache
    else:
        return kernel_output, K_tkg_hbm, V_tkg_hbm


############### Internal ###############


def _validate_and_extract_config(
    X: nl.ndarray,
    W_qkv: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    W_out: Optional[nl.ndarray],
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    KVDP: int,
    KVDP_replica_group: Optional[ReplicaGroup],
    out_in_sb: bool,
    skip_attention: bool,
) -> Dict[str, Any]:
    """
    Validate inputs and extract configuration parameters for attention block.

    Args:
        X (nl.ndarray): Input hidden states
        W_qkv (nl.ndarray): QKV projection weights
        K_cache (nl.ndarray): Key cache
        V_cache (nl.ndarray): Value cache
        attention_mask (nl.ndarray): Attention mask
        cos (Optional[nl.ndarray]): RoPE cosine embeddings
        sin (Optional[nl.ndarray]): RoPE sine embeddings
        rmsnorm_X_gamma (Optional[nl.ndarray]): RMSNorm weights
        K_cache_transposed (bool): K cache layout flag
        active_blocks_table (Optional[nl.ndarray]): Block indices for block KV cache
        W_out (Optional[nl.ndarray]): Output projection weights

    Returns:
        Dict[str, Any]: Configuration dictionary with keys: B, S_tkg, H, d_head, half_d,
            q_heads, S_ctx, S_max_ctx, is_block_kv, blk_len, cache_had_head_dim,
            do_out_proj, K_cache, V_cache

    Notes:
        - Validates tensor shapes and dimensions
        - Extracts batch size, sequence lengths, and head dimensions
        - Handles both block and flat KV cache layouts
    """

    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen3,
        f"Kernel requires nc-version >= gen3, got {nisa.get_nc_version()}",
    )

    _, B, _, _ = attention_mask.shape
    if X.buffer == nl.sbuf:
        # X.shape = (pmax, B*S, H // pmax) @ SBUF
        kernel_assert(len(X.shape) == 3, "SBUF input X must have 3 dimensions")
        kernel_assert(X.shape[0] == nl.tile_size.pmax, f"SBUF input X dim0 must be {nl.tile_size.pmax}")
        kernel_assert(X.shape[1] % B == 0, f"SBUF input X dim1 must be divisible by B={B}")
        H = X.shape[2] * nl.tile_size.pmax
        S_tkg = X.shape[1] // B
    else:
        # X.shape = (B,S,H) @ HBM
        kernel_assert(is_hbm_buffer(X), "Input X must be in HBM or SBUF")
        B, S_tkg, H = X.shape

    d_head = V_cache.shape[-1]
    I = W_qkv.shape[1]
    kv_heads = 1

    # This limitation can be relaxed
    is_KVDP = KVDP > 1
    if is_KVDP:
        kernel_assert(KVDP_replica_group != None, "KVDP_replica_group is required when KVDP > 1")

    # Compute tiling for large batch support (B * S_tkg > pmax)
    pmax = nl.tile_size.pmax
    bxs_tile = min(B * S_tkg, (pmax // S_tkg) * S_tkg)  # multiple of S_tkg to allow tiling on batch dim
    n_bxs_tiles = div_ceil(B * S_tkg, bxs_tile)
    # Guards for unsupported combinations when B * S_tkg > pmax
    if n_bxs_tiles > 1:
        kernel_assert(
            not out_in_sb,
            f"out_in_sb is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
        kernel_assert(
            not skip_attention,
            f"skip_attention is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
        kernel_assert(
            X.buffer != nl.sbuf,
            f"SBUF input is not supported when B * S_tkg > {pmax}, got B * S_tkg = {B * S_tkg}",
        )
    kernel_assert(d_head % 2 == 0, f"d_head must be even, got {d_head}")
    kernel_assert(
        d_head > 0 and I % d_head == 0,
        f"QKV weights must be packed as (q_heads + 2) * d_head, got I={I}, d_head={d_head}",
    )

    q_heads = I // d_head - 2 * kv_heads
    half_d = d_head // 2

    # Compute KV data parallelism dimensions early for use in validation
    if is_KVDP:
        B_attn = B // KVDP
        q_heads_attn = q_heads * KVDP
        kernel_assert(B % KVDP == 0, f"B must be divisible by KVDP, got B={B}, KVDP={KVDP}")
    else:
        B_attn = B
        q_heads_attn = q_heads

    # Process KV cache
    is_block_kv = active_blocks_table != None
    K_cache, V_cache, cache_had_head_dim = __internal_squeeze_head_dim(K_cache, V_cache, is_block_kv)

    if is_block_kv:
        blk_len = V_cache.shape[1]
        S_ctx = S_max_ctx = active_blocks_table.shape[1] * blk_len
        kernel_assert(
            V_cache.shape == K_cache.shape,
            f"Block KV cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
        )
    else:
        S_ctx = attention_mask.shape[0]
        S_max_ctx = V_cache.shape[1]
        blk_len = 0
        kernel_assert(
            V_cache.shape[0] == B_attn,
            f"V_cache batch mismatch: expected {B_attn}, got {V_cache.shape[0]}",
        )
        expected_K_shape = (B_attn, d_head, S_max_ctx) if K_cache_transposed else (B_attn, S_max_ctx, d_head)
        kernel_assert(
            tuple(K_cache.shape) == expected_K_shape,
            f"K_cache shape mismatch: expected {expected_K_shape}, got {K_cache.shape}",
        )

    # Validate attention mask
    expected_mask_shape = (S_ctx, B_attn, q_heads_attn, S_tkg)
    kernel_assert(
        tuple(attention_mask.shape) == expected_mask_shape,
        f"attention_mask shape mismatch: expected {expected_mask_shape}, got {attention_mask.shape}",
    )

    # Validate RMSNorm weights
    if rmsnorm_X_gamma != None:
        kernel_assert(
            tuple(rmsnorm_X_gamma.shape) == (1, H),
            f"rmsnorm_X_gamma must be (1, {H}), got {rmsnorm_X_gamma.shape}",
        )

    # Validate RoPE embeddings
    if cos != None and sin != None:
        kernel_assert(
            tuple(cos.shape) == (half_d, B, S_tkg),
            f"cos shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {cos.shape}",
        )
        kernel_assert(
            tuple(sin.shape) == (half_d, B, S_tkg),
            f"sin shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {sin.shape}",
        )

    # KV Quantization
    if k_scale != None and v_scale != None:
        kernel_assert(is_fp8_e4m3(K_cache.dtype), f'KV quantization requires float8_e4m3 K_cache, got {K_cache.dtype}')
        kernel_assert(is_fp8_e4m3(V_cache.dtype), f'KV quantization requires float8_e4m3 V_cache, got {V_cache.dtype}')
        kv_quant = True
    else:
        kv_quant = False

    return {
        'B': B,
        'S_tkg': S_tkg,
        'H': H,
        'd_head': d_head,
        'half_d': half_d,
        'q_heads': q_heads,
        'S_ctx': S_ctx,
        'S_max_ctx': S_max_ctx,
        'is_block_kv': is_block_kv,
        'blk_len': blk_len,
        'cache_had_head_dim': cache_had_head_dim,
        'do_out_proj': W_out != None,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'kv_quant': kv_quant,
        'B': B,
        'B_attn': B_attn,
        'q_heads_attn': q_heads_attn,
        'is_KVDP': is_KVDP,
        'n_bxs_tiles': n_bxs_tiles,
        'bxs_tile': bxs_tile,
    }


def __internal_squeeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, is_block_kv: bool
) -> Tuple[nl.ndarray, nl.ndarray, bool]:
    """
    Remove head dimension from 4D cache tensors.

    Args:
        K_cache (nl.ndarray): Key cache (3D or 4D)
        V_cache (nl.ndarray): Value cache (3D or 4D)
        is_block_kv (bool): Block KV cache flag

    Returns:
        Tuple[nl.ndarray, nl.ndarray, bool]: (K_squeezed, V_squeezed, had_head_dim)

    Notes:
        - If is_block_kv is False, removes dim-1 (N) from BNSd or BNdS
        - If is_block_kv is True, removes dim-2 from (blocks, block_len, heads, d_head)
        - Returns original tensors if already 3D
    """
    if len(K_cache.shape) != 4:
        kernel_assert(len(K_cache.shape) == len(V_cache.shape) == 3, "Expecting KV cache to have 3 or 4 dims")
        return K_cache, V_cache, False

    head_dim = 2 if is_block_kv else 1
    kernel_assert(len(K_cache.shape) == len(V_cache.shape) == 4, "Expecting KV cache to have 3 or 4 dims")
    kernel_assert(K_cache.shape[head_dim] == V_cache.shape[head_dim] == 1, "Expecting single head for KV")
    K_shape = list(K_cache.shape[:head_dim]) + list(K_cache.shape[head_dim + 1 :])
    V_shape = list(V_cache.shape[:head_dim]) + list(V_cache.shape[head_dim + 1 :])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape)), True


def __internal_unsqueeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, cache_had_head_dim: bool, is_block_kv: bool
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Add back head dimension if cache originally had one.

    Args:
        K_cache (nl.ndarray): Key cache (3D)
        V_cache (nl.ndarray): Value cache (3D)
        cache_had_head_dim (bool): Whether cache originally had head dimension
        is_block_kv (bool): Block KV cache flag

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: (K_cache, V_cache) with head dimension restored

    Notes:
        - Inverse operation of __internal_squeeze_head_dim
        - Returns original tensors if cache_had_head_dim is False
    """
    if not cache_had_head_dim:
        return K_cache, V_cache

    head_dim = 2 if is_block_kv else 1
    K_shape = list(K_cache.shape[:head_dim]) + [1] + list(K_cache.shape[head_dim:])
    V_shape = list(V_cache.shape[:head_dim]) + [1] + list(V_cache.shape[head_dim:])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape))


def _to_sbuf(buf: nl.ndarray, sbm: SbufManager) -> nl.ndarray:
    """
    Ensure buffer is in SBUF; copy from HBM if needed.

    Args:
        buf (nl.ndarray): Input buffer (HBM or SBUF)
        sbm (SbufManager): SBUF memory manager

    Returns:
        nl.ndarray: Buffer in SBUF

    Notes:
        - Returns original buffer if already in SBUF
        - Allocates and copies if buffer is in HBM
    """
    if buf.buffer == nl.sbuf:
        return buf
    else:
        sb = sbm.alloc_stack(buf.shape, dtype=buf.dtype, buffer=nl.sbuf)
        nisa.dma_copy(sb, buf)
        return sb


def _process_head_group(
    QKV: nl.ndarray,
    qkv_offset: int,
    n_heads: int,
    d: int,
    B: int,
    S: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    rmsnorm_pre_W: Optional[nl.ndarray],
    enable_rope: bool,
    sb_cos: Optional[nl.ndarray],
    sb_sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W: Optional[nl.ndarray],
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Process Q or K for a single tile: extract heads, transpose to [d, B*n_heads*S],
    apply optional RMSNorm pre/post and RoPE.
    For Q: n_heads=q_heads, qkv_offset=0
    For K: n_heads=1, qkv_offset=d*q_heads

    QKV must be in SBUF with shape (B*S, I) where B*S <= pmax.
    """
    # Transpose heads from [B*S, n_heads*d] to [d, B*n_heads*S]
    out = sbm.alloc_stack(shape=(d, B * n_heads * S), dtype=QKV.dtype, buffer=nl.sbuf)
    for head_idx in range(n_heads):
        psum = nl.ndarray((d, B * S), dtype=QKV.dtype, buffer=nl.psum)
        nisa.nc_transpose(psum, QKV[:, nl.ds(qkv_offset + head_idx * d, d)])
        nisa.tensor_copy(out.reshape((d, B, n_heads, S))[:, :, head_idx, :], psum.reshape((d, B, S)))

    # Pre-RoPE RMSNorm
    if rmsnorm_pre_enabled:
        _rms_norm_inplace(out, rmsnorm_pre_eps, w=rmsnorm_pre_W, sbm=sbm)

    # RoPE
    if enable_rope:
        out_4d = out.reshape((d, B, n_heads, S))
        out_rope = sbm.alloc_stack(out_4d.shape, dtype=out.dtype, buffer=nl.sbuf)
        RoPE_sbuf(out_4d, sb_cos, sb_sin, out_rope, convert_from_interleaved=not rope_contiguous_layout)
        out = out_rope.reshape((d, B * n_heads * S))

    # Post-RoPE RMSNorm
    if rmsnorm_post_enabled:
        _rms_norm_inplace(out, rmsnorm_post_eps, rmsnorm_post_W, sbm)

    return out


def _QKV_processing(
    QKV: nl.ndarray,
    q_heads: int,
    kv_heads: int,
    B: int,
    S_tkg: int,
    d_head: int,
    n_bxs_tiles: int,
    bxs_tile: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    rmsnorm_pre_W_Q: Optional[nl.ndarray],
    rmsnorm_pre_W_K: Optional[nl.ndarray],
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W_Q: Optional[nl.ndarray],
    rmsnorm_post_W_K: Optional[nl.ndarray],
    kv_quant: bool,
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    io_dtype,
    sbm: SbufManager,
) -> Tuple[nl.ndarray, nl.ndarray, nl.ndarray, Optional[nl.ndarray]]:
    """
    Unified Q/K/V processing with tiling. For each tile:
      1. Load QKV tile to SBUF (from HBM for large batch, or use QKV directly if already in SBUF)
      2. Process Q and K via _process_head_group (transpose, optional pre-RoPE RMSNorm, optional RoPE, optional post-RoPE RMSNorm)
      3. Extract V and copy to HBM

    Args:
        QKV: [B*S, I] @ SBUF or HBM - concatenated Q/K/V projections where I = d*(q_heads+2)
        q_heads: number of query heads
        kv_heads: number of key/value heads
        B: batch size
        S_tkg: sequence length (tokens per batch)
        d_head: head dimension
        n_bxs_tiles: number of tiles for B*S dimension (1 = small batch, >1 = large batch)
        bxs_tile: tile size for B*S dimension
        rmsnorm_pre_enabled: Apply RMSNorm before RoPE
        rmsnorm_pre_eps: Pre-RoPE RMSNorm epsilon
        rmsnorm_pre_W_Q: Pre-RoPE Q gamma weights (optional)
        rmsnorm_pre_W_K: Pre-RoPE K gamma weights (optional)
        cos: RoPE cosine embeddings (None = skip RoPE)
        sin: RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout: True for contiguous halves, False for interleaved
        rmsnorm_post_enabled: Apply RMSNorm after RoPE
        rmsnorm_post_eps: Post-RoPE RMSNorm epsilon
        rmsnorm_post_W_Q: Post-RoPE Q weights (optional)
        rmsnorm_post_W_K: Post-RoPE K weights (optional)
        kv_quant: whether to quantize K/V to FP8
        k_scale: scale for K quantization (optional)
        v_scale: scale for V quantization (optional)
        io_dtype: input/output dtype
        sbm: SBUF memory manager

    Returns:
        Q_sb: [d_head, B*q_heads*S_tkg] @ SBUF
        K_sb: [d_head, B*S_tkg] @ SBUF (fp8 if kv_quant)
        V_hbm: [B, 1, S_tkg, d_head] @ HBM (fp8 if kv_quant)
        V_sb: [B*S_tkg, d_head] @ SBUF for small batch KV cache update, None for large batch
    """
    I = d_head * (q_heads + 2 * kv_heads)

    # Allocate full output buffers
    Q_sb = sbm.alloc_stack((d_head, B * q_heads * S_tkg), dtype=io_dtype, buffer=nl.sbuf)
    K_sb = sbm.alloc_stack((d_head, B * S_tkg), dtype=io_dtype, buffer=nl.sbuf)
    V_hbm = nl.ndarray(
        (B, 1, S_tkg, d_head),
        dtype=nl.float8_e4m3 if kv_quant else io_dtype,
        buffer=nl.shared_hbm,
        name=f"{sbm.get_name_prefix()}v_attention_hbm",
    )

    enable_rope = cos != None and sin != None

    # Load RoPE embeddings to SBUF if needed
    sb_cos, sb_sin = None, None
    if enable_rope:
        sb_cos = _to_sbuf(cos, sbm)
        sb_sin = _to_sbuf(sin, sbm)

    for tile_idx in range(n_bxs_tiles):
        tile_start = tile_idx * bxs_tile
        tile_size = min(bxs_tile, B * S_tkg - tile_start)
        tile_B = tile_size // S_tkg

        # Load QKV tile to SBUF if QKV is on HBM, otherwise use directly
        if QKV.buffer != nl.sbuf:
            qkv_sb = sbm.alloc_stack((tile_size, I), dtype=io_dtype, buffer=nl.sbuf)
            qkv_tile = TensorView(QKV).slice(0, start=tile_start, end=tile_start + tile_size)
            nisa.dma_copy(qkv_sb, qkv_tile.get_view())
        else:
            qkv_sb = QKV

        # Slice cos/sin for this tile (copy to fresh buffer — RoPE_sbuf can't use indexed tensors)
        tile_cos, tile_sin = None, None
        if enable_rope:
            half_d = d_head // 2
            tile_cos = sbm.alloc_stack((half_d, tile_B, S_tkg), dtype=sb_cos.dtype, buffer=nl.sbuf)
            tile_sin = sbm.alloc_stack((half_d, tile_B, S_tkg), dtype=sb_sin.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(tile_cos, sb_cos[:, nl.ds(tile_start // S_tkg, tile_B), :])
            nisa.tensor_copy(tile_sin, sb_sin[:, nl.ds(tile_start // S_tkg, tile_B), :])

        # Process Q tile
        Q_tile = _process_head_group(
            qkv_sb,
            qkv_offset=0,
            n_heads=q_heads,
            d=d_head,
            B=tile_B,
            S=S_tkg,
            rmsnorm_pre_enabled=rmsnorm_pre_enabled,
            rmsnorm_pre_eps=rmsnorm_pre_eps,
            rmsnorm_pre_W=rmsnorm_pre_W_Q,
            enable_rope=enable_rope,
            sb_cos=tile_cos,
            sb_sin=tile_sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_post_enabled=rmsnorm_post_enabled,
            rmsnorm_post_eps=rmsnorm_post_eps,
            rmsnorm_post_W=rmsnorm_post_W_Q,
            sbm=sbm,
        )
        # Scatter Q tile into full buffer [d_head, B, q_heads, S_tkg]
        Q_tile_4d = Q_tile.reshape((d_head, tile_B, q_heads, S_tkg))
        Q_4d = Q_sb.reshape((d_head, B, q_heads, S_tkg))
        for h in range(q_heads):
            nisa.tensor_copy(Q_4d[:, nl.ds(tile_start // S_tkg, tile_B), h, :], Q_tile_4d[:, :, h, :])

        # Process K tile
        K_tile = _process_head_group(
            qkv_sb,
            qkv_offset=d_head * q_heads,
            n_heads=1,
            d=d_head,
            B=tile_B,
            S=S_tkg,
            rmsnorm_pre_enabled=rmsnorm_pre_enabled,
            rmsnorm_pre_eps=rmsnorm_pre_eps,
            rmsnorm_pre_W=rmsnorm_pre_W_K,
            enable_rope=enable_rope,
            sb_cos=tile_cos,
            sb_sin=tile_sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_post_enabled=rmsnorm_post_enabled,
            rmsnorm_post_eps=rmsnorm_post_eps,
            rmsnorm_post_W=rmsnorm_post_W_K,
            sbm=sbm,
        )
        nisa.tensor_copy(K_sb[:, nl.ds(tile_start, tile_size)], K_tile)

        # Extract V from QKV to SBUF, then copy to HBM for attention_tkg
        # attention_tkg expects V input from HBM
        V_tile_sb = sbm.alloc_stack((tile_size, d_head), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(V_tile_sb, qkv_sb[:, nl.ds(d_head * (q_heads + kv_heads), d_head)])
        # Quantize V to FP8 for attention when kv_quant=True
        if kv_quant:
            V_tile_sb = _quantize_to_fp8(V_tile_sb, v_scale, sbm)
        # Write V tile to HBM
        V_hbm_view = TensorView(V_hbm.reshape((B * S_tkg, d_head))).slice(
            0, start=tile_start, end=tile_start + tile_size
        )
        nisa.dma_copy(V_hbm_view.get_view(), V_tile_sb)

    # Quantize K to FP8 for attention when kv_quant=True
    if kv_quant:
        K_sb = _quantize_to_fp8(K_sb, k_scale, sbm)

    # V_tile_sb from the last (or only) tile is kept for KV cache update (small batch only).
    # For large batch (n_bxs_tiles > 1), V is only on HBM.
    V_sb = V_tile_sb if n_bxs_tiles == 1 else None

    return Q_sb, K_sb, V_hbm, V_sb


def _rms_norm_inplace(
    x: nl.ndarray, eps: float, w: Optional[nl.ndarray] = None, sbm: Optional[SbufManager] = None
) -> None:
    """
    RMS normalization in-place: x / sqrt(mean(x^2) + eps), optionally scaled by w.
    Computed in fp32, result written back to x in original dtype.

    Args:
        x: [d_head, BnS] @ SBUF - input tensor (d_head must be nl.tile_size.pmax), modified in-place
        eps: epsilon for numerical stability
        w: [d_head, 1] @ HBM - optional scale weights
        sbm: SBUF memory manager
    """
    d_head, BnS = x.shape
    kernel_assert(d_head == nl.tile_size.pmax, f"d_head must be {nl.tile_size.pmax}, got {d_head}")

    # Setup constants
    ones_sb = sbm.alloc_stack((d_head, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    eps_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)

    # Compute x^2 in fp32
    x_squared = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(x_squared, x, x, nl.multiply)

    # Compute sum(x^2) via matmul with all-ones matrix
    psum_sb = nl.ndarray((d_head, BnS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(psum_sb, stationary=ones_sb, moving=x_squared)

    # Compute rsqrt(mean(x^2) + eps)
    rsqrt_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rsqrt_sb, op=nl.rsqrt, data=psum_sb, bias=eps_sb, scale=1.0 / d_head)

    # Normalize: x * rsqrt
    out_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(out_sb, x, rsqrt_sb, nl.multiply)

    # Optional scaling by weights
    if w != None:
        w_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(w_sb, w.reshape((d_head, 1)))
        nisa.tensor_scalar(dst=out_sb, data=out_sb, op0=nl.multiply, operand0=w_sb)

    # Copy result back to x with original dtype
    nisa.tensor_copy(dst=x, src=out_sb)


############################# KV cache update logic #############################


def _kv_cache_update(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    B: int,
    d_head: int,
    S_tkg: int,
    S_max_ctx: int,
    K_cache_transposed: bool,
    is_block_kv: bool,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Update KV cache with new tokens for token generation.

    Args:
        K_cache: K cache @ HBM
            - Block KV: [num_blocks, block_len, d_head]
            - Flat transposed: [B, d_head, S_max_ctx]
            - Flat: [B, S_max_ctx, d_head]
        V_cache: V cache @ HBM
            - Block KV: [num_blocks, block_len, d_head]
            - Flat: [B, S_max_ctx, d_head]
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B] slot indices for cache writes
        B: batch size
        d_head: head dimension
        S_tkg: number of new tokens
        S_max_ctx: max cache sequence length
        K_cache_transposed: K cache layout flag
        is_block_kv: block KV cache flag

    Returns:
        Updated (K_cache, V_cache) - modified in-place
    """

    # TODO: oob_mode.skip not supported for flat cache. Using oob_mode.skip causes accuracy failures (root cause unknown).

    if is_block_kv:
        _update_block_cache(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)
    elif S_tkg == 1 and B > 1 and (not K_cache_transposed or B > 16):
        # vector DMA with indirect addressing, tiled over batch dim. Bug for S_tkg > 1.
        _update_flat_cache_batched(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
            K_cache_transposed=K_cache_transposed,
        )
    else:
        # per-batch scalar DMA
        _update_flat_cache(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            K_cache_transposed,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
        )


def _update_flat_cache_batched(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
    K_cache_transposed: bool = False,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using batched DMA operations.

    This optimized version writes batches in vector DMA operations using vector_offset
    for indirect addressing.
    Tiles over the batch dimension in chunks of pmax to support B > pmax.
    Currently limited to S_tkg=1 due to access pattern bug.

    Args:
        K_cache: [B, S_max_ctx, d_head] or [B, d_head, S_max_ctx] K cache in HBM
        V_cache: [B, S_max_ctx, d_head] V cache in HBM
        K_tkg: [d_head, B*S_tkg] new K tokens in SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, 1] per-batch write positions
        S_tkg: number of new tokens per batch (must be 1)
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    # Validate sharding configuration
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(S_tkg == 1, f"_update_flat_cache_batched() only supports S_tkg=1, got {S_tkg}")

    # Validate tensor shapes
    if K_cache_transposed:
        kernel_assert(
            K_cache.shape == (B, d_head, S_max_ctx),
            f"K_cache shape mismatch: expected {(B, d_head, S_max_ctx)}, got {K_cache.shape}",
        )
    else:
        kernel_assert(
            K_cache.shape == (B, S_max_ctx, d_head),
            f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
        )
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )
    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )

    tile_sz = nl.tile_size.pmax
    v_on_hbm = V_tkg.buffer != nl.sbuf

    # token_indices are used by V update and non-transposed K update.
    # When K_cache_transposed=True and lnc=2, lnc=1 computes its own k_token_indices,
    # so token_indices would be unused on lnc=1, causing a compiler error.
    needs_token_indices = n_prgs == 1 or prg_id == 0 or not K_cache_transposed

    for b_start in range(0, B, tile_sz):
        tile_B = min(tile_sz, B - b_start)

        # Compute absolute token indices for this tile:
        #   token_indices[b] = kv_cache_update_idx[b] + b * S_max_ctx
        if needs_token_indices:
            token_indices = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(token_indices, kv_cache_update_idx[nl.ds(b_start, tile_B)])
            batch_offset = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.iota(batch_offset, [[0, 1]], offset=b_start * S_max_ctx, channel_multiplier=S_max_ctx)
            nisa.tensor_tensor(token_indices, token_indices, batch_offset, nl.add)

        # V tile: Vector DGE requires source in SBUF, so load from HBM if needed
        if v_on_hbm:
            v_tile = nl.ndarray((tile_B, d_head), dtype=V_tkg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(v_tile, V_tkg.reshape((B * S_tkg, d_head))[nl.ds(b_start, tile_B), :])
        else:
            v_tile = V_tkg  # single tile

        # Vector DMA with indirect addressing:
        # - Reshape cache to (B*S_max_ctx, d_head) so each row has stride d_head
        # - vector_offset provides per-batch row indices: token_indices[b]
        # - DMA engine scales token_indices[b] by d_head (stride of indirect_dim=0)

        # Update V_cache on lnc=0
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=V_cache.reshape((B * S_max_ctx, d_head)).ap(
                    pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                    offset=0,
                    vector_offset=token_indices,
                    indirect_dim=0,
                ),
                src=v_tile.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
            )

        # Update K_cache on lnc=1
        if n_prgs == 1 or prg_id == 1:
            # Transpose K tile: (d_head, tile_B) → (tile_B, d_head)
            K_tile_sb = nl.ndarray((tile_B * S_tkg, d_head), dtype=K_tkg.dtype, buffer=nl.sbuf)
            _transpose_sbuf(K_tkg[:, nl.ds(b_start * S_tkg, tile_B * S_tkg)], K_tile_sb)

            if K_cache_transposed:
                # K_cache [B, d_head, S_max_ctx] — strided scatter
                # k_token_indices[b] = b * d_head * S_max_ctx + idx[b]
                k_token_indices = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
                k_batch_offset = nl.ndarray((tile_B, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.iota(
                    k_batch_offset, [[0, 1]], offset=b_start * d_head * S_max_ctx, channel_multiplier=d_head * S_max_ctx
                )
                nisa.dma_copy(k_token_indices, kv_cache_update_idx[nl.ds(b_start, tile_B)])
                nisa.tensor_tensor(k_token_indices, k_token_indices, k_batch_offset, nl.add)

                nisa.dma_copy(
                    dst=K_cache.reshape((B * d_head * S_max_ctx,)).ap(
                        pattern=[[1, tile_B], [S_max_ctx, d_head], [1, S_tkg]],
                        offset=0,
                        vector_offset=k_token_indices,
                        indirect_dim=0,
                    ),
                    src=K_tile_sb.ap(pattern=[[d_head, tile_B], [1, d_head], [1, S_tkg]]),
                )
            else:
                # K_cache [B, S_max_ctx, d_head] — same pattern as V
                nisa.dma_copy(
                    dst=K_cache.reshape((B * S_max_ctx, d_head)).ap(
                        pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                        offset=0,
                        vector_offset=token_indices,
                        indirect_dim=0,
                    ),
                    src=K_tile_sb.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
                )


def _update_flat_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    K_cache_transposed: bool,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using per-batch scalar_offset.

    This version iterates over batches and uses scalar_offset for indirect addressing.
    Supports any B*S_tkg via tiled K transpose. When V_tkg is on HBM, V is loaded
    per-batch.

    Args:
        K_cache: [B, d_head, S_max_ctx] if transposed else [B, S_max_ctx, d_head] @ HBM
        V_cache: [B, S_max_ctx, d_head] @ HBM
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        K_cache_transposed: K cache layout flag
        kv_cache_update_idx: [B, 1] per-batch write positions
        S_tkg: number of new tokens per batch
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")

    v_on_hbm = V_tkg.buffer != nl.sbuf

    # Validate tensor shapes
    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Tiled K transpose for non-transposed K cache
    if not K_cache_transposed and (n_prgs == 1 or prg_id == 1):
        K_transposed_sb, tile_sz = _tiled_k_transpose(K_tkg, B, S_tkg)

    # Update V_cache on lnc=0
    if n_prgs == 1 or prg_id == 0:
        start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        for batch_idx in range(B):
            nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
            if v_on_hbm:
                v_src = V_tkg.reshape((B * S_tkg, d_head))[nl.ds(batch_idx * S_tkg, S_tkg), :]
            else:
                v_src = V_tkg[nl.ds(batch_idx * S_tkg, S_tkg), :]
            nisa.dma_copy(
                dst=V_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=batch_idx * S_max_ctx * d_head,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=v_src,
            )

    # Update K_cache on lnc=1
    if n_prgs == 1 or prg_id == 1:
        if K_cache_transposed:
            kernel_assert(
                K_cache.shape == (B, d_head, S_max_ctx),
                f"K_cache shape mismatch: expected {(B, d_head, S_max_ctx)}, got {K_cache.shape}",
            )
            # K_tkg is already in correct layout [d_head, B*S_tkg]
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[S_max_ctx, d_head], [1, S_tkg]],
                        offset=batch_idx * d_head * S_max_ctx,
                        scalar_offset=start_position,
                        indirect_dim=2,
                    ),
                    src=K_tkg[:, nl.ds(batch_idx * S_tkg, S_tkg)],
                )
        else:
            kernel_assert(
                K_cache.shape == (B, S_max_ctx, d_head),
                f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
            )
            # Write transposed K to cache — index into tiled K_transposed_sb
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
                k_src = _get_k_transposed_slice(K_transposed_sb, tile_sz, batch_idx, S_tkg)
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[d_head, S_tkg], [1, d_head]],
                        offset=batch_idx * S_max_ctx * d_head,
                        scalar_offset=start_position,
                        indirect_dim=1,
                    ),
                    src=k_src,
                )


def _update_block_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens.

    Routes to batched or scalar_offset implementation based on S_tkg.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head]
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
    """
    # TODO: Use batched case for all S_tkg values once vector DMA access pattern supports S_tkg > 1
    if S_tkg == 1:
        # one vector DMA, S_tkg = 1
        _update_block_cache_batched(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)
    else:
        # B scalar DMAs, with any S_tkg
        _update_block_cache_scalar(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)


def _update_block_cache_batched(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens using batched DMA operations.

    Writes batches in vector DMA operations using vector_offset for indirect addressing.
    Tiles over the batch dimension in chunks of pmax to support B > pmax.
    Currently limited to S_tkg=1 due to access pattern bug.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens (must be 1)
        B: batch size
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(S_tkg == 1, f"_update_block_cache_batched() only supports S_tkg=1, got {S_tkg}")

    num_blocks, blk_len, d_head = K_cache.shape

    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        K_cache.shape == V_cache.shape,
        f"K/V cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    tile_sz = nl.tile_size.pmax
    v_on_hbm = V_tkg.buffer != nl.sbuf

    for b_start in range(0, B, tile_sz):
        tile_B = min(tile_sz, B - b_start)

        # Load cache update indices for this tile
        idx_tile = nl.ndarray((tile_B, 1), dtype=kv_cache_update_idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(idx_tile, kv_cache_update_idx[nl.ds(b_start, tile_B)])

        # V tile: Vector DGE requires source in SBUF, so load from HBM if needed
        if v_on_hbm:
            v_tile = nl.ndarray((tile_B, d_head), dtype=V_tkg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(v_tile, V_tkg.reshape((B * S_tkg, d_head))[nl.ds(b_start, tile_B), :])
        else:
            v_tile = V_tkg  # single tile

        # Vector DMA with indirect addressing:
        # - Reshape cache to (num_blocks*blk_len, d_head) so each row has stride d_head
        # - vector_offset provides per-batch global token indices: idx_tile[b]
        # - DMA engine scales by d_head

        # Update V_cache on lnc=0
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=V_cache.reshape((num_blocks * blk_len, d_head)).ap(
                    pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                    offset=0,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=v_tile.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )

        # Update K_cache on lnc=1
        if n_prgs == 1 or prg_id == 1:
            # Transpose K tile: (d_head, tile_B) → (tile_B, d_head)
            K_tile_sb = nl.ndarray((tile_B * S_tkg, d_head), dtype=K_tkg.dtype, buffer=nl.sbuf)
            _transpose_sbuf(K_tkg[:, nl.ds(b_start * S_tkg, tile_B * S_tkg)], K_tile_sb)

            nisa.dma_copy(
                dst=K_cache.reshape((num_blocks * blk_len, d_head)).ap(
                    pattern=[[1, tile_B], [d_head, S_tkg], [1, d_head]],
                    offset=0,
                    vector_offset=idx_tile,
                    indirect_dim=0,
                ),
                src=K_tile_sb.ap(pattern=[[S_tkg * d_head, tile_B], [d_head, S_tkg], [1, d_head]]),
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )


def _update_block_cache_scalar(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens using per-batch scalar_offset.

    This version iterates over batches and uses scalar_offset for indirect addressing.
    Supports any B*S_tkg via tiling, and handles both transposed and non-transposed
    K cache layouts. When V_tkg is on HBM, V is loaded per-batch.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head] @ SBUF or [B, 1, S_tkg, d_head] @ HBM
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")

    v_on_hbm = V_tkg.buffer != nl.sbuf

    num_blocks, blk_len, d_head = K_cache.shape

    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        K_cache.shape == V_cache.shape,
        f"K/V cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Tiled K transpose
    if n_prgs == 1 or prg_id == 1:
        K_transposed_sb, tile_sz = _tiled_k_transpose(K_tkg, B, S_tkg)

    # Update cache per batch element using scalar_offset
    for batch_idx in range(B):
        start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])

        # Update V_cache on lnc=0
        if n_prgs == 1 or prg_id == 0:
            if v_on_hbm:
                v_src = V_tkg.reshape((B * S_tkg, d_head))[nl.ds(batch_idx * S_tkg, S_tkg), :]
            else:
                v_src = V_tkg[nl.ds(batch_idx * S_tkg, S_tkg), :]
            nisa.dma_copy(
                dst=V_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=0,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=v_src,
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )

        # Update K_cache on lnc=1
        if n_prgs == 1 or prg_id == 1:
            k_src = _get_k_transposed_slice(K_transposed_sb, tile_sz, batch_idx, S_tkg)
            nisa.dma_copy(
                dst=K_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=0,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=k_src,
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )


############################# FP8 Quantization Helpers #############################

_FP8_E4M3_MAX = get_max_positive_value_for_dtype(nl.float8_e4m3)
_FP8_E4M3_MIN = -_FP8_E4M3_MAX


def _quantize_to_fp8(tensor, scale, sbm):
    """
    Quantize a tensor to FP8 E4M3 format using a single scalar scale.

    Computes: output = cast_to_fp8(clip(tensor * scale, [-240, 240]))

    The scale must represent a single scalar value. Two shapes are supported for
    compatibility with different APIs:
    - (1, 1): scalar, broadcast to partition dim
    - (PMAX, 1): assumed to contain identical values, copied directly

    Args:
        tensor: Input tensor in SBUF, shape (P, F), dtype bf16 or f32
        scale: Scale tensor in HBM, shape (PMAX, 1) or (1, 1).
               Must contain a single scalar value (broadcast or replicated).
               Supported dtypes: float32, float16, bfloat16.
        sbm: SbufManager for allocations

    Returns:
        FP8 E4M3 quantized tensor in SBUF, same shape as input
    """
    kernel_assert(tensor.buffer == nl.sbuf, "quantize_to_fp8 requires tensor in SBUF")
    kernel_assert(not is_fp8_e4m3(tensor.dtype), f"quantize_to_fp8 input already FP8: {tensor.dtype}")

    partition_dim = tensor.shape[0]

    # Copy scale to SBUF
    # ndarray avoids anti-dependency with other stack values
    scale_sb = nl.ndarray(shape=(partition_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    if scale.shape == (nl.tile_size.pmax, 1):
        nisa.dma_copy(dst=scale_sb, src=scale[0:partition_dim, :])
    else:
        kernel_assert(scale.shape == (1, 1), f"scale must be (pmax, 1) or (1, 1), got {scale.shape}")
        nisa.dma_copy(dst=scale_sb, src=TensorView(scale).broadcast(dim=0, size=partition_dim).get_view())

    # Scale: multiply by scale
    tensor_scaled = sbm.alloc_stack(tensor.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(tensor_scaled, tensor, nl.multiply, scale_sb)

    # Clip to FP8 range and cast
    tensor_fp8 = sbm.alloc_stack(tensor.shape, dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_scalar(tensor_fp8, tensor_scaled, nl.minimum, _FP8_E4M3_MAX, op1=nl.maximum, operand1=_FP8_E4M3_MIN)

    return tensor_fp8


def _transpose_sbuf(src, dst):
    """
    Transpose tensor from SBUF to SBUF via PSUM.

    For FP8: nc_transpose doesn't support FP8, so we cast to bf16, transpose, cast back.

    Args:
        src: Source tensor in SBUF (P, F)
        dst: Destination tensor in SBUF (F, P) - must be pre-allocated
    """
    if is_fp8_e4m3(src.dtype):
        # FP8 workaround: cast to bf16, transpose, cast back
        src_bf16 = nl.ndarray(src.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(src_bf16, src)
        psum = nl.ndarray(dst.shape, dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=src_bf16)
        nisa.tensor_copy(dst=dst, src=psum)
    else:
        kernel_assert(
            src.dtype in (nl.bfloat16, nl.float16),
            f"_transpose_sbuf only supports bf16, fp16, or fp8, got {src.dtype}",
        )
        psum = nl.ndarray(dst.shape, dtype=src.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum, data=src)
        nisa.tensor_copy(dst=dst, src=psum)


def _tiled_k_transpose(K_tkg: nl.ndarray, B: int, S_tkg: int) -> Tuple[nl.ndarray, int]:
    """
    Transpose K_tkg from (d_head, B*S_tkg) to tiled (tile_sz, n_tiles, d_head) in SBUF.

    Tiles in chunks of tile_sz (multiple of S_tkg, <= pmax) so batch boundaries
    align with tile boundaries. Index result as in _get_k_transposed_slice.

    Args:
        K_tkg: [d_head, B*S_tkg] @ SBUF
        B: batch size
        S_tkg: tokens per batch

    Returns:
        K_transposed_sb: [tile_sz, n_tiles, d_head] @ SBUF
        tile_sz: tile size used (multiple of S_tkg)
    """
    total_bxs = B * S_tkg
    tile_sz = min(total_bxs, (nl.tile_size.pmax // S_tkg) * S_tkg)
    n_k_tiles = div_ceil(total_bxs, tile_sz)
    d_head = K_tkg.shape[0]
    K_transposed_sb = nl.ndarray((tile_sz, n_k_tiles, d_head), K_tkg.dtype, nl.sbuf)
    for t_idx in range(n_k_tiles):
        t_start = t_idx * tile_sz
        t_size = min(tile_sz, total_bxs - t_start)
        k_dst = TensorView(K_transposed_sb).select(1, t_idx).slice(0, start=0, end=t_size).get_view()
        _transpose_sbuf(K_tkg[:, nl.ds(t_start, t_size)], k_dst)
    return K_transposed_sb, tile_sz


def _get_k_transposed_slice(K_transposed_sb: nl.ndarray, tile_sz: int, batch_idx: int, S_tkg: int):
    """Index into tiled K transpose buffer for a given batch.

    Args:
        K_transposed_sb: [tile_sz, n_tiles, d_head] from _tiled_k_transpose
        tile_sz: tile size used (returned by _tiled_k_transpose)
        batch_idx: batch index
        S_tkg: tokens per batch

    Returns:
        View of (S_tkg, d_head) for this batch's K data
    """
    flat_idx = batch_idx * S_tkg
    tile_idx = flat_idx // tile_sz
    tile_off = flat_idx % tile_sz
    return TensorView(K_transposed_sb).select(1, tile_idx).slice(0, start=tile_off, end=tile_off + S_tkg).get_view()
