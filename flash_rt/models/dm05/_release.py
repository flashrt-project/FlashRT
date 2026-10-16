"""Free the bf16 storage of a weight once a quantized copy has replaced it."""
from __future__ import annotations

import os

import torch


def release(param: torch.Tensor) -> None:
    """Drop the tensor's storage in place (the Parameter object stays valid, empty).

    The native pipeline never reads a quantized site's bf16 weight again, and at
    this model's size keeping both copies does not fit a 16 GB card. The HF modules
    that own these parameters are then unusable for those layers, which only the
    diagnostic ablation paths would notice. DM05_NATIVE_KEEP_BF16=1 keeps them.
    """
    if os.environ.get("DM05_NATIVE_KEEP_BF16") == "1":
        return
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
