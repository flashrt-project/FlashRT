# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head_dim=256 GQA token-generation NKI kernels.

Qwen3.6's GQA attention runs at head_dim=256, above nkilib's 128-partition cap.
These phased composables tile head_dim into ceil(d/128) partition tiles. Pipeline:

    qkv_proj -> qk_norm -> rope -> attention -> out_proj

- qkv_proj  : wraps nkilib qkv_tkg (NBSd layout, fused input RMSNorm)
- qk_norm   : free-axis RMSNorm over head_dim=256 (q/k heads; head_dim on free)
- rope      : partial 64/256 + mRoPE, free-axis
- attention : vendored nkilib attention_tkg, patched (D_TILES) for head_dim=256
- out_proj  : wraps nkilib output_projection_tkg (256 as 2x128 sub-heads)

The ``decode`` package fuses all five into one @nki.jit kernel (``gqa_fused_tkg_fwd``)
with no HBM round-trip for intermediates.

Each phase has an isolated PyTorch-reference test under
``models/qwen3_6_moe/tests/test_gqa_<phase>_kernel.py``. Not yet integrated into
the model.
"""

from .components.qkv_proj import qkv_proj_compose, gqa_qkv_proj_fwd
from .components.qk_norm import qk_norm_compose, rms_norm_over_free
from .components.rope import rope_partial_compose
from .components.attention import gqa_attention_d256, build_attention_tkg_config
from .components.out_proj import out_proj_compose
from .decode.fused_layer import gqa_fused_tkg_fwd

__all__ = [
    "qkv_proj_compose",
    "gqa_qkv_proj_fwd",
    "qk_norm_compose",
    "rms_norm_over_free",
    "rope_partial_compose",
    "gqa_attention_d256",
    "build_attention_tkg_config",
    "out_proj_compose",
    "gqa_fused_tkg_fwd",
]
