# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DeltaNet input RMSNorm + 4-way input projection for token generation.

Thin @nki.jit wrapper over the vendored qkv_tkg: norm(hidden) @ proj_w in one call.

proj_w is [H, I], the transpose of the nn.Linear weights, concatenating in_proj_qkv|z|a|b on the
output axis. The caller slices the output back into qkv/z/a/b.

Per rank (TP=4): hidden 2048, I 3088, T <= 2.
The SBUF-resident fusion into conv+recurrence lives in deltanet/decode/fused_layer.py.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType

from ..vendored.qkv_tkg import qkv_tkg


def in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf, name_prefix=""):
    """Fused input RMSNorm + 4-way projection; returns [B, S, I] HBM or [B*S, I] SBUF."""
    if hidden.buffer == nl.sbuf:
        norm_in = nl.ndarray(hidden.shape, dtype=hidden.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=norm_in, src=hidden)
    else:
        norm_in = hidden
    sbm = create_auto_alloc_manager()
    sbm.set_name_prefix(name_prefix + "in_proj_")
    return qkv_tkg(
        hidden=norm_in,
        qkv_w=proj_w,
        norm_w=gamma,
        norm_type=NormType.RMS_NORM,
        quantization_type=QuantizationType.NONE,
        output_layout=QKVOutputLayout.BSD,
        eps=eps,
        d_head=None,
        num_q_heads=None,
        num_kv_heads=None,
        fused_add=False,
        output_in_sbuf=output_in_sbuf,
        sbm=sbm,
        i_column_tiling=True,
    )


@nki.jit
def deltanet_in_proj_fwd(hidden, proj_w, gamma, eps=1e-6):
    """Standalone HBM-output path: [B, S, I] = norm(hidden) @ proj_w. Launch [2] or [1]."""
    return in_proj_compose(hidden, proj_w, gamma, eps, output_in_sbuf=False)
