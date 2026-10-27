"""FlashRT -- GR00T N1.5 model pipeline.

``forward`` holds the stage sequence (vision tower, Qwen3 language model,
VL self-attention, DiT flow-matching head) every target shares;
``pipeline_rtx`` binds it to RTX SM89. The torch frontend is
``flash_rt.frontends.torch.groot_n15_rtx_sm89``.

Importing this package needs neither the compiled kernels nor the upstream
``gr00t`` package; the frontend checks both when it is constructed.
"""
from __future__ import annotations

from flash_rt.models.groot_n15.forward import (
    GROOT_N15_SYMBOLS,
    KERNEL_SYMBOLS,
    GrootN15Dims,
    GrootN15Forward,
    GrootN15Target,
    require_kernels,
)

__all__ = [
    "GROOT_N15_SYMBOLS",
    "KERNEL_SYMBOLS",
    "GrootN15Dims",
    "GrootN15Forward",
    "GrootN15Target",
    "require_kernels",
]
