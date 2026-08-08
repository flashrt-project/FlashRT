"""Structure catalog registry — pure lookup, no execution logic.

Loads structure specifications from the on-disk catalog and resolves
their reference implementations. Dispatch, tuning, calibration, and
activation live in separate layers; the registry only answers "what is
structure X and where is its ground truth".
"""

from __future__ import annotations

import importlib
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import yaml

_CATALOG_DIR = pathlib.Path(__file__).resolve().parent / "catalog"


@dataclass(frozen=True)
class StructureSpec:
    """One catalog entry, as declared in ``catalog/<name>/structure.yaml``.

    Two kinds exist. ``region`` structures declare a tensor boundary,
    weight slots and a torch reference. ``stage_pipeline`` structures
    declare a stage graph with cadence attributes; their parity reference
    is the host's own eager path, so ``boundary``/``weights``/``reference``
    are empty for them, while ``stages`` and the allowed
    ``embedded_regions`` are populated instead.
    """

    name: str
    version: int
    description: str
    boundary: Mapping[str, Any]
    weights: Sequence[Mapping[str, Any]]
    variants: Mapping[str, Sequence[str]]
    calibration: Mapping[str, Any]
    gates: Mapping[str, Any]
    _reference: Mapping[str, str] = field(repr=False)
    kind: str = "region"
    family: str = ""
    stages: Sequence[Mapping[str, Any]] = ()
    embedded_regions: Sequence[str] = ()
    conformance: Sequence[str] = ()

    @property
    def symbolic_dims(self) -> Sequence[str]:
        return tuple(self.boundary.get("symbolic_dims", ()))

    @property
    def weight_slots(self) -> Sequence[str]:
        return tuple(entry["slot"] for entry in self.weights)

    def reference(self) -> Callable[..., Any]:
        """Resolve the reference entrypoint (ground truth for gates)."""
        if not self._reference:
            raise LookupError(
                f"structure {self.name!r} (kind={self.kind!r}) has no "
                "standalone reference; its parity ground truth is the "
                "host's own eager path under the same noise window"
            )
        module = importlib.import_module(
            f"{__package__}.catalog.{self._reference['module']}"
        )
        return getattr(module, self._reference["entrypoint"])


def list_structures() -> list[str]:
    """Names of all structures present in the catalog."""
    return sorted(
        path.parent.name for path in _CATALOG_DIR.glob("*/structure.yaml")
    )


def load(name: str) -> StructureSpec:
    """Load one structure specification by catalog name."""
    path = _CATALOG_DIR / name / "structure.yaml"
    if not path.is_file():
        raise KeyError(f"unknown structure: {name!r}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    kind = data.get("kind", "region")
    spec = StructureSpec(
        name=data["structure"],
        version=int(data["version"]),
        description=str(data.get("description", "")).strip(),
        boundary=data.get("boundary", {}) if kind != "region"
        else data["boundary"],
        weights=data.get("weights", ()) if kind != "region"
        else data["weights"],
        variants=data.get("variants", {}),
        calibration=data.get("calibration", {}),
        gates=data["gates"],
        _reference=data.get("reference", {}) if kind != "region"
        else data["reference"],
        kind=kind,
        family=str(data.get("family", "")),
        stages=data.get("stages", ()),
        embedded_regions=data.get("embedded_regions", ()),
        conformance=data.get("conformance", ()),
    )
    if spec.name != name:
        raise ValueError(
            f"catalog directory {name!r} declares structure {spec.name!r}"
        )
    return spec
