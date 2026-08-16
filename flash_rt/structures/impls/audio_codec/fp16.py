"""FP16 implementation of the ``audio_codec`` structure.

The host codec decode is compute-bound on fp32 cuDNN convolutions;
fp16 autocast is the only measured lever (1.22x, waveform cosine
1.00000 vs the fp32 host — torch.compile and CUDA graph both measure
~1.0x here). The impl replaces the codec module's ``decode`` and keeps
the host module for fallback and attribute delegation.

Boundary: codes [B, C, T] (int64) -> audio_values [B, 1, N] (fp32).
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam


class AudioCodecDecodeFp16(GuardedSeam, torch.nn.Module):
    """Codec decode seam: fp16 autocast over the host's eager decode."""

    _frt_host_attr = "host_codec"
    _frt_can_fallback = True

    def __init__(self, host_codec, device):
        super().__init__()
        self.host_codec = host_codec
        self._device = device
        self._frt_arm(dtypes=(torch.long,), device=device)
        self._frt_guard.notes["backend"] = "fp16_codec"

    def decode(self, codes: torch.Tensor):
        admitted = self._frt_admit(codes)
        if admitted is not PROCEED:
            return admitted
        with torch.autocast("cuda", dtype=torch.float16):
            return self.host_codec.decode(codes)

    forward = decode

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_codec":
                raise
            return getattr(super().__getattr__("host_codec"), name)


@torch.no_grad()
def bind_codec_decode(host_codec, *, device=None, original=None):
    """Bind the codec decode seam to the fp16 path."""
    dev = device or host_codec.device
    seam = AudioCodecDecodeFp16(host_codec, dev)
    if original is not None:
        seam.host_codec = original
    # bind-time smoke (AGENTS.md §2.8): one real decode, zero codes is
    # not a valid shape — use a tiny all-zero frame set
    probe = seam.decode(torch.zeros(1, 1, 4, device=dev, dtype=torch.long))
    audio = probe.audio_values
    if not torch.isfinite(audio.float()).all():
        raise ValueError(
            f"refused: audio_codec fp16 bind smoke produced non-finite "
            f"audio {tuple(audio.shape)}")
    return seam
