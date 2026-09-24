"""Plain-torch reference for ``audio_codec``.

The parity ground truth is the host codec's own eager fp32 decode.
"""

from __future__ import annotations

import torch


def audio_codec_ref(module: torch.nn.Module,
                    codes: torch.Tensor) -> torch.Tensor:
    """Run the host codec decode eagerly (fp32); returns audio_values."""
    out = module.decode(codes)
    return out.audio_values
