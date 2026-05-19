"""Declarative quantization descriptor.

This module defines the data structures that record *how* a tensor is
quantized. The goal is to give the framework (and the user) a single
place to read and write quantization metadata — so that today's
calibration-produced FP8 scales and tomorrow's QAT-checkpoint-loaded
scales share one representation.

Scope for v1:
    * The current shipped kernels support experimental
      ``dtype in {"fp8_e4m3", "int8"}`` with
      ``granularity="per_tensor"`` + ``scheme="symmetric"``.
    * ``PrecisionSpec.validate()`` raises ``NotImplementedError`` for any
      other combination — this is intentional. It forces future extensions
      (QAT, per-channel, asymmetric) to touch this file first, keeping
      the kernel dispatcher honest.

No kernel dispatch, no runtime behaviour, no implicit conversions here.
This module is pure data + validation, imported by frontends so their
``precision_spec`` attribute is introspectable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

DType = Literal[
    "fp32", "fp16", "bf16",
    "fp8_e4m3", "fp8_e5m2",
    "nvfp4", "int8", "int4",
]
Granularity = Literal["per_tensor", "per_channel", "per_group"]
Scheme = Literal["symmetric", "asymmetric"]
ScaleSource = Literal["calibration", "qat_checkpoint", "runtime_dynamic", "manual"]

_SUPPORTED_DTYPES = {"fp8_e4m3", "int8"}
_SUPPORTED_GRANULARITIES = {"per_tensor"}
_SUPPORTED_SCHEMES = {"symmetric"}


@dataclass
class PrecisionSpec:
    """Quantization descriptor for a single tensor."""

    dtype: DType = "fp8_e4m3"
    granularity: Granularity = "per_tensor"
    scheme: Scheme = "symmetric"
    group_size: Optional[int] = None

    scale_source: ScaleSource = "calibration"
    scale: Optional[np.ndarray] = None
    zero_point: Optional[np.ndarray] = None

    calibration_method: Optional[str] = None
    calibration_samples: Optional[int] = None
    calibration_percentile: Optional[float] = None

    def validate(self) -> None:
        """Assert the spec is expressible by currently-shipped kernels.

        Hard-fails with NotImplementedError on any combination outside
        the v1 capability set. This is deliberate — we'd rather force
        the user (or a future dev) to extend this file than silently
        fall back to something that doesn't match their QAT scheme.
        """
        if self.group_size is not None and self.granularity != "per_group":
            raise ValueError("group_size is only valid for per_group granularity")
        if self.granularity == "per_group" and self.group_size is None:
            raise ValueError("per_group granularity requires group_size")
        if self.scheme == "asymmetric" and self.zero_point is None:
            raise ValueError("asymmetric scheme requires zero_point")

        if self.dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError(
                f"dtype={self.dtype!r} is not supported by shipped kernels. "
                f"v1 supports only {sorted(_SUPPORTED_DTYPES)}.")
        if self.granularity not in _SUPPORTED_GRANULARITIES:
            raise NotImplementedError(
                f"granularity={self.granularity!r} is not supported by shipped "
                f"kernels. v1 supports only {sorted(_SUPPORTED_GRANULARITIES)}.")
        if self.scheme not in _SUPPORTED_SCHEMES:
            raise NotImplementedError(
                f"scheme={self.scheme!r} is not supported by shipped kernels. "
                f"v1 supports only {sorted(_SUPPORTED_SCHEMES)}.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # numpy arrays → lists for JSON
        if d["scale"] is not None:
            d["scale"] = np.asarray(d["scale"]).tolist()
        if d["zero_point"] is not None:
            d["zero_point"] = np.asarray(d["zero_point"]).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrecisionSpec":
        d = dict(d)
        if d.get("scale") is not None:
            d["scale"] = np.asarray(d["scale"], dtype=np.float32)
        if d.get("zero_point") is not None:
            d["zero_point"] = np.asarray(d["zero_point"])
        return cls(**d)


@dataclass
class ModelPrecisionSpec:
    """Aggregate quantization metadata for a whole model.

    Layer specs are keyed by (layer_idx, tensor_name) strings to keep
    JSON serialization trivial. Callers that want structured access can
    build their own views on top.
    """

    encoder_layer_specs: Dict[str, PrecisionSpec] = field(default_factory=dict)
    decoder_layer_specs: Dict[str, PrecisionSpec] = field(default_factory=dict)
    weight_specs: Dict[str, PrecisionSpec] = field(default_factory=dict)
    activation_specs: Dict[str, PrecisionSpec] = field(default_factory=dict)
    source: Literal["calibration", "qat_checkpoint", "manual"] = "calibration"

    def validate(self) -> None:
        for d in (self.encoder_layer_specs, self.decoder_layer_specs,
                  self.weight_specs, self.activation_specs):
            for spec in d.values():
                spec.validate()

    def to_json(self, path: str) -> None:
        payload = {
            "source": self.source,
            "encoder_layer_specs": {k: v.to_dict()
                                    for k, v in self.encoder_layer_specs.items()},
            "decoder_layer_specs": {k: v.to_dict()
                                    for k, v in self.decoder_layer_specs.items()},
            "weight_specs": {k: v.to_dict()
                             for k, v in self.weight_specs.items()},
            "activation_specs": {k: v.to_dict()
                                 for k, v in self.activation_specs.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelPrecisionSpec":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            source=payload.get("source", "calibration"),
            encoder_layer_specs={k: PrecisionSpec.from_dict(v)
                                 for k, v in payload.get("encoder_layer_specs", {}).items()},
            decoder_layer_specs={k: PrecisionSpec.from_dict(v)
                                 for k, v in payload.get("decoder_layer_specs", {}).items()},
            weight_specs={k: PrecisionSpec.from_dict(v)
                          for k, v in payload.get("weight_specs", {}).items()},
            activation_specs={k: PrecisionSpec.from_dict(v)
                              for k, v in payload.get("activation_specs", {}).items()},
        )
