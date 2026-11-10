# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused input RMSNorm + GQA QKV projection for token generation (head_dim=256).

Thin wrapper over nkilib's qkv_tkg -- one fused call computes:

    hidden' = RMSNorm(hidden, norm_w, eps)   # input_layernorm over H
    qkv     = hidden' @ qkv_w                # fused Q/K/V projection

Per rank (TP=4) the block has q_heads=4, kv_heads=1, head_dim=256, so the fused output dim is
I = 6*256 = 1536, head-major on the free axis as [q0|q1|q2|q3|k0|v0]. qkv_w is [H, I], the
transpose of the nn.Linear weight; norm_w is the [1, H] input_layernorm weight.

Two output forms, both bf16 IO with fp32 accumulate:
  output_in_sbuf=True   -- stays in SBUF as [B*S, I], head_dim on the free axis so D=256 needs no
                           partition tiling. Downstream slices per head.
  output_in_sbuf=False  -- NBSd HBM [N, B, S, D], N = 6, same head ordering.

The model's separate sigmoid output-gate projection is not computed here; q/k/v only.
"""

import nki

from nkilib.core.qkv.qkv_tkg import qkv_tkg
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType

# Qwen3.6 GQA per-rank (TP=4) head config: 16 Q / 4 (replicated 2->4) KV heads sharded over 4 ranks.
HEAD_DIM = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = (
    NUM_Q_HEADS + 2 * NUM_KV_HEADS
)  # 6 (q heads, then k, then v) on the I/N axis
I_DIM = NUM_HEADS * HEAD_DIM  # 1536


def qkv_proj_compose(hidden, qkv_w, norm_w, eps=1e-6, output_in_sbuf=True):
    """Fused input RMSNorm + GQA QKV projection via qkv_tkg, head-major q -> k -> v.

    Args:
        hidden:         [B, S, H] HBM, or [128, B*S, H//128] SBUF, bf16.
        qkv_w:          [H, I] HBM fused QKV weight, columns head-major.
        norm_w:         [1, H] HBM input_layernorm weight, bf16.
        eps:            RMSNorm epsilon.
        output_in_sbuf: SBUF [B*S, I] when set, else NBSd HBM [N, B, S, HEAD_DIM].
    """
    return qkv_tkg(
        hidden=hidden,
        qkv_w=qkv_w,
        norm_w=norm_w,
        norm_type=NormType.RMS_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.NBSd,
        eps=eps,
        d_head=HEAD_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        fused_add=False,
        output_in_sbuf=output_in_sbuf,
    )


@nki.jit
def gqa_qkv_proj_fwd(hidden, qkv_w, norm_w, eps=1e-6):
    """Standalone HBM-output path: NBSd [N, B, S, HEAD_DIM] fused QKV. Launch [2] or [1]."""
    return qkv_proj_compose(hidden, qkv_w, norm_w, eps, output_in_sbuf=False)
