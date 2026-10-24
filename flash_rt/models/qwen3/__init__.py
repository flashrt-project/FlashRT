"""FlashRT Qwen3 (plain dense) model namespace.

Shipped:
  * Qwen3-8B-Instruct-2512-SFT-NVFP4 (RTX SM120) — ``Qwen3Dims``
  * Qwen/Qwen3-4B BF16 prefill (RTX SM120 / SM89, Jetson Thor) — ``Qwen3_4BDims``,
    shared sequence in ``prefill_bf16``

Public re-exports keep the model-namespace surface in line with the
``flash_rt.models.qwen36`` sibling.
"""
from __future__ import annotations

from .pipeline_rtx import Qwen3Dims, Qwen3Pipeline
from .prefill_bf16 import Qwen3_4BDims

__all__ = ['Qwen3Dims', 'Qwen3Pipeline', 'Qwen3_4BDims']
