"""Higgs Audio v3 neural codec — decode (codes -> 24 kHz waveform).

The codec is a DAC-style convolutional decoder bundled inside the TTS
checkpoint's ``model.safetensors`` under the
``tied.embedding.modality_embeddings.0.model.`` prefix. The model definition is
vendored under ``_codec/tokenizer_model.py`` (upstream:
``bosonai/higgs-audio`` v2 tokenizer). Only the decode path is used here;
synthesis runs in fp32 (ConvTranspose is unstable in low precision).

Acoustic codes are produced by :class:`HiggsAudioV3TorchFrontendRtx.predict`.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import torch

from ._codec.env_guard import apply as _apply_env_guard

_apply_env_guard()

from ._codec.tokenizer_model import (  # noqa: E402
    HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel,
)

_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "_codec", "tokenizer_config.json")


class HiggsAudioV3Codec:
    """Frozen decode-only wrapper around the bundled Higgs audio tokenizer."""

    SAMPLE_RATE = 24_000

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, *,
                        device: str = "cuda:0",
                        dtype: torch.dtype = torch.float32) -> "HiggsAudioV3Codec":
        """Build the codec from a TTS checkpoint dir (config.json + shards)."""
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        for k in ("architectures", "torch_dtype", "transformers_version"):
            cfg.pop(k, None)
        model = HiggsAudioV2TokenizerModel(
            HiggsAudioV2TokenizerConfig(**cfg)).to(dtype=dtype).eval()

        state: dict[str, torch.Tensor] = {}
        for shard in sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))):
            from safetensors import safe_open
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(_CODEC_PREFIX):
                        state[key[len(_CODEC_PREFIX):]] = f.get_tensor(key)
        if not state:
            raise FileNotFoundError(
                f"no codec weights under {_CODEC_PREFIX!r} in {checkpoint_path}")
        missing, _ = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"codec load too sparse: {len(missing)} missing / {len(state)}")
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model.to(device), device)

    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """``[T, num_codebooks]`` int codes -> mono waveform ``[L]`` (cpu f32)."""
        if codes_TN.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got {tuple(codes_TN.shape)}")
        codes_BNT = codes_TN.transpose(0, 1).unsqueeze(0).to(
            device=self.device, dtype=torch.long)
        return self.model.decode(codes_BNT).audio_values.squeeze(0).squeeze(0).float().cpu()
