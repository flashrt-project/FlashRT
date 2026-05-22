"""Host-side RoPE cos/sin table construction for LingBot-VLA.

LingBot's Mixed-Head LLM (the hot 36 layers × 50 Euler steps path) does NOT
use the Qwen2.5-VL multimodal M-RoPE despite the ckpt config exposing one.
Instead, ``lingbotvla/models/vla/pi0/utils.py::apply_rope`` is called with
flat 1D ``position_ids`` (constructed in modeling_lingbot_vla.py via
``cumsum(pad_masks) - 1`` and ``prefix_offsets + cumsum(suffix_pad_masks) - 1``).

That apply_rope is plain 1-D RoPE in **split-half** layout:

    inv_freq[i] = 1.0 / 10000 ** (2 i / head_dim)     for i in [0, head_dim/2)
    radians[b, l, i] = positions[b, l] * inv_freq[i]
    cos = cos(radians), sin = sin(radians)           shape (B, L, head_dim/2)
    x1, x2 = x.split(head_dim/2, dim=-1)
    out = concat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)

This is **NOT** the HF ``rotate_half`` form (which interleaves dims by 1)
and **NOT** M-RoPE (which selects axis frequencies per chunk). For the
ViT path (which DOES use a different 2-D M-RoPE over image patches),
see ``flash_rt.models.lingbot.rope_adapter_vit`` (added later).

Built once per ``set_prompt`` (when the prefix layout is fixed) and
once per denoise step (when the suffix position_ids extend). Stored as
device tensors and reused as inputs to the existing flash_rt RoPE
kernels during every graph replay.

Math source: ``lingbotvla/models/vla/pi0/utils.py::apply_rope`` —
validated bit-exact against the upstream reference.

This module is pure-PyTorch (CPU/GPU agnostic). It does NOT call any
flash_rt_kernels symbol — the kernel consumes the tensors produced here
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LingbotRopeConfig:
    """LingBot-VLA Mixed-Head LLM 1-D RoPE constants.

    Locked by ``utils.apply_rope`` defaults + the QwenvlWithExpertConfig
    head_dim. NOT to be confused with Qwen2.5-VL's M-RoPE config on the
    ViT side (which has rope_theta=1e6 and mrope_section=[16,24,24]) — the
    Mixed-Head LLM forward overrides the HF path and uses these constants.
    """

    head_dim: int = 128
    max_wavelength: float = 10_000.0


def compute_inv_freq(
    cfg: LingbotRopeConfig,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``1 / max_wavelength ** (2 i / head_dim)`` for ``i ∈ [0, head_dim/2)``.

    Returns: tensor of shape ``(head_dim/2,)``, dtype as requested.
    """
    half = cfg.head_dim // 2
    freq_exponents = (2.0 / cfg.head_dim) * torch.arange(
        half, dtype=dtype, device=device
    )
    timescale = cfg.max_wavelength ** freq_exponents
    return 1.0 / timescale


def build_cos_sin_table(
    positions: torch.Tensor,
    cfg: LingbotRopeConfig,
    *,
    compute_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype | None = None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` of shape ``(B, L, head_dim/2)`` from position ids.

    Args:
        positions: ``(B, L)`` integer or float positions. Will be cast to
            ``compute_dtype`` internally to match the upstream apply_rope
            arithmetic exactly.
        cfg: ``LingbotRopeConfig``.
        compute_dtype: dtype used for the sin/cos compute. Default fp32
            (matches lingbot apply_rope's default ``dtype=torch.float32``).
        out_dtype: dtype of returned tensors. ``None`` means same as
            ``compute_dtype``. Set to fp16 for kernel handoff.
        device: device for the output. Defaults to positions' device.

    Returns:
        ``cos``, ``sin`` of shape ``(B, L, head_dim/2)``.

    Notes:
        The returned tables are **split-half** — they cover the first
        ``head_dim/2`` slots only. The downstream apply step treats the
        head_dim as ``[x1 | x2]`` where ``x1, x2`` each have ``head_dim/2``
        elements and uses the SAME cos/sin for both halves. (This is the
        layout used by ``utils.apply_rope`` — NOT HF's rotate_half form.)
    """
    if device is None:
        device = positions.device
    inv_freq = compute_inv_freq(cfg, device=device, dtype=compute_dtype)
    pos_cast = positions.to(compute_dtype)
    # radians: (B, L, head_dim/2)
    radians = torch.einsum("bl,h->blh", pos_cast, inv_freq)
    cos = torch.cos(radians)
    sin = torch.sin(radians)
    if out_dtype is not None and out_dtype != compute_dtype:
        cos = cos.to(out_dtype)
        sin = sin.to(out_dtype)
    return cos, sin


def apply_rope_with_tables(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-built cos/sin to ``x: (B, L, H, D)`` — pure-PyTorch reference.

    Used by the bit-exact test to verify our cos/sin path matches the
    upstream apply_rope. The flash_rt kernel will do the same math
    on-device.

    cos, sin should have shape ``(B, L, head_dim/2)``; we add a head
    broadcast axis automatically (the original ``utils.apply_rope`` does
    the same via ``radians[..., None, :]``).
    """
    d = x.shape[-1]
    d_half = d // 2
    cos = cos[..., None, :]  # (B, L, 1, head_dim/2)
    sin = sin[..., None, :]
    x1, x2 = x.split(d_half, dim=-1)
    return torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
    )


# Default config instance for the LingBot-VLA Mixed-Head LLM path.
# Importers should use this constant rather than re-instantiating, so
# that the values stay in lock-step with the model file.
LINGBOT_ROPE_CONFIG = LingbotRopeConfig(head_dim=128, max_wavelength=10_000.0)


__all__ = [
    "LingbotRopeConfig",
    "LINGBOT_ROPE_CONFIG",
    "compute_inv_freq",
    "build_cos_sin_table",
    "apply_rope_with_tables",
]
