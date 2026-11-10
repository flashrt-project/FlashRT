# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fully fused GQA token-generation NKI kernels (head_dim=256).

One @nki.jit LNC2 launch fuses the five validated GQA composables --
qkv projection, q/k RMSNorm, partial RoPE, attention, output projection -- with
every intermediate (qkv / normed / roped / attn-out / gate) kept in SBUF: no HBM
round-trip for intermediates. The only HBM writes are the KV-cache persistence and
the final o_proj partial.
"""

from .fused_layer import gqa_fused_tkg_fwd

__all__ = ["gqa_fused_tkg_fwd"]
