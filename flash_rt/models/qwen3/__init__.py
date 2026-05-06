"""FlashRT Qwen3 (plain dense) model namespace.

Currently shipped: Qwen3-8B-Instruct-2512-SFT-NVFP4 (RTX SM120).
Public re-exports keep the model-namespace surface in line with the
``flash_rt.models.qwen36`` sibling.
"""
from __future__ import annotations

from .pipeline_rtx import Qwen3Dims, Qwen3Pipeline

__all__ = ['Qwen3Dims', 'Qwen3Pipeline']
