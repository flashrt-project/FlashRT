# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom NKI kernels for the Qwen3.6-27B DeltaNet layers.

Stable public surface. Consumers import entrypoints from this package root
(``from ...nki_kernels import deltanet_fused_tkg_fwd``) and stay decoupled from
the internal module layout — add new kernels here as the megakernel grows.

Layout (see README.md for the full map):
- ``deltanet/components/`` reusable stage kernels (in_proj, conv, recurrence,
  norm_gate, out_proj) — the megakernel building blocks.
- ``deltanet/decode/``     token-generation (TKG) path; ``fused_layer`` is the
  assembled DeltaNet attention megakernel (current); ``recurrent`` is the
  per-token fallback.
- ``deltanet/prefill/``    context-encoding (CTE) path; ``chunked_step`` is the
  stable default; ``chunked_fused`` is a faster single-kernel variant that
  overflows fp32 for this checkpoint (gated off — see README).
- ``specs/``               design specs for individual kernels.
"""

# Decode (TKG).
from .deltanet.decode.fused_layer import (
    deltanet_fused_tkg_fwd,
    deltanet_fused_tkg_fwd_state,
    deltanet_attention_layer_state,
)
from .deltanet.decode.recurrent import (
    deltanet_recurrent_fwd,
    deltanet_recurrent_fwd_state,
)

# Prefill (CTE).
from .deltanet.prefill.chunked_step import deltanet_chunk_step
from .deltanet.prefill.chunked_fused import (
    deltanet_fused_chunked_fwd,
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
)

# GQA (full-attention) decode/verify TKG megakernel.
from .gqa import gqa_fused_tkg_fwd

# MoE (post-attention FFN) fused-layer verify kernel.
from .moe import moe_layer_fwd

# LM head (final norm + vocab projection + per-rank greedy argmax), shared by both heads.
from .lm_head import lm_head_fwd

__all__ = [
    "deltanet_fused_tkg_fwd",
    "deltanet_fused_tkg_fwd_state",
    "deltanet_attention_layer_state",
    "deltanet_recurrent_fwd",
    "deltanet_recurrent_fwd_state",
    "deltanet_chunk_step",
    "deltanet_fused_chunked_fwd",
    "_make_lower_mask",
    "_make_lower_mask_diag",
    "_make_identity",
    "gqa_fused_tkg_fwd",
    "moe_layer_fwd",
    "lm_head_fwd",
]
