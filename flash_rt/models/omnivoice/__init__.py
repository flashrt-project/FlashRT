"""FlashRT — OmniVoice TTS model pipeline.

Mixed BF16 CFG + FP4 noCFG acceleration preserves audio quality at 5.0x
throughput on Blackwell SM120 GPUs.

Per the unified API contract:
    inject()       — patch OmniVoice model for FlashRT acceleration
    free_encoder() — release encoder weights (~600 MB VRAM saved)
    eject()        — restore original forward and generate methods

See docs/PERFORMANCE_OMNIVOICE.md for performance specifications.
"""

from flash_rt.models.omnivoice.pipeline_rtx import (
    FlashRTLlm,
    FlashRTLlmBF16,
    inject,
    free_encoder,
    eject,
    _check_kernels,
)

__all__ = [
    "FlashRTLlm",
    "FlashRTLlmBF16",
    "inject",
    "free_encoder",
    "eject",
    "_check_kernels",
]
