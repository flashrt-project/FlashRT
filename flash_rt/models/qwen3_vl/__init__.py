"""FlashRT Qwen3-VL (multimodal) model namespace.

Currently targeted: Qwen3-VL-8B-Instruct (RTX SM120).

The language backbone is a strict superset of ``flash_rt.models.qwen3``
(dense Qwen3-8B: hidden 4096 / 36L / GQA 32:8 / head_dim 128 /
SwiGLU 12288 / vocab 151936) and reuses that path's NVFP4 W4A4 decoder
kernels. Qwen3-VL adds a SigLIP-style ViT tower (DeepStack +
interleaved-MRoPE) ahead of the decoder; that tower lives in the
RTX frontend / pipeline alongside the reused decoder.

Public re-exports keep the model-namespace surface in line with the
``flash_rt.models.qwen3`` sibling.
"""
from __future__ import annotations

from .pipeline_rtx import Qwen3VlDims, Qwen3VlPipeline

__all__ = ['Qwen3VlDims', 'Qwen3VlPipeline']
