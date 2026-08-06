"""Compatibility import path for the Qwen3.6-35B-A3B text frontend.

The frontend moved to :mod:`flash_rt.frontends.torch.qwen36_moe` when it stopped
being RTX-only. This module re-exports the public names so an existing import
keeps working; new code should import from the module above.
"""

from __future__ import annotations

from flash_rt.frontends.torch.qwen36_moe import (
    Qwen36MoeTextFrontend,
    Qwen36MoeTextFrontendRtx,
    validate_qwen36_moe_checkpoint,
)

__all__ = [
    "Qwen36MoeTextFrontend",
    "Qwen36MoeTextFrontendRtx",
    "validate_qwen36_moe_checkpoint",
]
