"""Single source of truth for compiled AMD backend capabilities."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


_ARCH_TO_HARDWARE = {"gfx942": "amd_cdna3", "gfx950": "amd_cdna4"}


@dataclass(frozen=True)
class AmdCapabilities:
    hardware: str
    gpu_arch: str
    fp8_format: str
    fp8_max_finite: float
    supports_mxfp4: bool
    supports_packed_fp8_mfma: bool
    supports_packed_bf16_mfma: bool
    supports_fused_attention_fp8_output: bool
    supports_aiter: bool
    aiter_installed: bool

    @property
    def torch_fp8_dtype(self):
        import torch
        if self.fp8_format == "e4m3fnuz":
            return torch.float8_e4m3fnuz
        if self.fp8_format == "e4m3fn":
            return torch.float8_e4m3fn
        raise RuntimeError(f"unsupported AMD FP8 format {self.fp8_format!r}")


def load_capabilities(fvk: Any, expected_hardware: str | None = None) -> AmdCapabilities:
    """Validate extension/device identity and return declared capabilities."""
    info = dict(fvk.build_info())
    build_arch = str(info.get("gpu_arch", "unknown")).split(":", 1)[0]
    device_arch = str(fvk.device_arch()).split(":", 1)[0]
    build_hardware = _ARCH_TO_HARDWARE.get(build_arch)
    device_hardware = _ARCH_TO_HARDWARE.get(device_arch)
    declared_hardware = str(info.get("hardware", "unknown"))
    if build_hardware is None:
        raise RuntimeError(
            f"unsupported AMD extension gpu_arch {build_arch!r}; expected gfx942 or gfx950")
    if device_hardware is None:
        raise RuntimeError(
            f"unsupported AMD device arch {device_arch!r}; expected gfx942 or gfx950")
    if build_arch != device_arch or declared_hardware != build_hardware:
        raise RuntimeError(
            "AMD extension/device mismatch: extension was built for "
            f"{build_arch!r} ({declared_hardware!r}), device is "
            f"{device_arch!r} ({device_hardware!r})")
    if expected_hardware is not None and expected_hardware != build_hardware:
        raise RuntimeError(
            f"requested hardware {expected_hardware!r}, but the validated "
            f"AMD extension/device pair is {build_hardware!r}")

    required = (
        "fp8_format", "fp8_max_finite", "supports_mxfp4",
        "supports_packed_fp8_mfma", "supports_packed_bf16_mfma",
        "supports_fused_attention_fp8_output", "supports_aiter",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise RuntimeError(
            "AMD extension is missing architecture capability metadata: "
            + ", ".join(missing))
    supports_aiter = bool(info["supports_aiter"])
    return AmdCapabilities(
        hardware=build_hardware,
        gpu_arch=build_arch,
        fp8_format=str(info["fp8_format"]),
        fp8_max_finite=float(info["fp8_max_finite"]),
        supports_mxfp4=bool(info["supports_mxfp4"]),
        supports_packed_fp8_mfma=bool(info["supports_packed_fp8_mfma"]),
        supports_packed_bf16_mfma=bool(info["supports_packed_bf16_mfma"]),
        supports_fused_attention_fp8_output=bool(info["supports_fused_attention_fp8_output"]),
        supports_aiter=supports_aiter,
        aiter_installed=(supports_aiter and importlib.util.find_spec("aiter") is not None),
    )
