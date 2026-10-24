"""X-VLA — soft-prompted flow-matching VLA (``lerobot/xvla-libero``).

Architecture: Florence-2 DaViT-L vision tower + BART encoder feeding a
24-layer soft-prompted flow-matching transformer that runs 10 denoising steps
to emit a 30 x 20 action chunk. RTX SM120 compute path in ``pipeline_rtx``.
"""

from flash_rt.models.xvla.pipeline_rtx import XVLARTXPipeline

__all__ = ["XVLARTXPipeline"]
