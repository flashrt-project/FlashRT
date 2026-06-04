"""Quantization manifest utilities.

The manifest is a small JSON contract between an external quantization
pipeline (for example ModelOpt exporting ONNX Q/DQ scales) and FlashRT's
static low-precision runtime.  It deliberately uses FlashRT internal GEMM
site names so frontend code does not have to reason over an ONNX graph at
inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_POLICIES = {"strict", "compatible", "fallback_bf16"}


@dataclass(frozen=True)
class QuantTensorSpec:
    dtype: str = "fp8_e4m3fn"
    granularity: str = "per_tensor"
    scale: float | list[float] | None = None
    scale_name: str | list[str] | None = None
    axis: int | None = None
    layout: str | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "QuantTensorSpec | None":
        if data is None:
            return None
        return cls(
            dtype=str(data.get("dtype", "fp8_e4m3fn")),
            granularity=str(data.get("granularity", "per_tensor")),
            scale=data.get("scale"),
            scale_name=data.get("scale_name"),
            axis=data.get("axis"),
            layout=data.get("layout"),
        )

    def scalar_scale(self) -> float | None:
        if self.scale is None:
            return None
        if isinstance(self.scale, list):
            if len(self.scale) != 1:
                raise ValueError(
                    f"expected scalar scale for {self.granularity}, got {len(self.scale)} values")
            return float(self.scale[0])
        return float(self.scale)


@dataclass(frozen=True)
class QuantSite:
    name: str
    precision: str = "fp8"
    weight: QuantTensorSpec | None = None
    activation: QuantTensorSpec | None = None
    onnx_nodes: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_json(cls, name: str, data: Mapping[str, Any]) -> "QuantSite":
        return cls(
            name=name,
            precision=str(data.get("precision", "fp8")),
            weight=QuantTensorSpec.from_json(data.get("weight")),
            activation=QuantTensorSpec.from_json(data.get("activation")),
            onnx_nodes=[str(x) for x in data.get("onnx_nodes", [])],
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class QuantManifest:
    model: str
    format: str = "flashrt_quant_manifest_v1"
    sites: dict[str, QuantSite] = field(default_factory=dict)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "QuantManifest":
        sites_raw = data.get("sites", {})
        if not isinstance(sites_raw, Mapping):
            raise ValueError("quant manifest 'sites' must be an object")
        return cls(
            model=str(data.get("model", "")),
            format=str(data.get("format", "flashrt_quant_manifest_v1")),
            source=data.get("source"),
            metadata=dict(data.get("metadata", {})),
            sites={
                str(name): QuantSite.from_json(str(name), site)
                for name, site in sites_raw.items()
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "QuantManifest":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def site(self, name: str) -> QuantSite | None:
        return self.sites.get(name)

    def require_model(self, model: str) -> None:
        if self.model and self.model != model:
            raise ValueError(f"quant manifest model={self.model!r} does not match {model!r}")


def load_quant_manifest(path: str | Path | None) -> QuantManifest | None:
    if path is None:
        return None
    return QuantManifest.load(path)


def validate_quant_policy(policy: str) -> str:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"quant_policy must be one of {sorted(SUPPORTED_POLICIES)}, got {policy!r}")
    return policy
