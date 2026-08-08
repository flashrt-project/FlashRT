"""Load and validate host bindings.

Region bindings map framework-neutral slots onto one host. Pipeline
bindings additionally describe stage seams and may carry a complete-hot-path
coverage contract. The contract is deliberately descriptive: it classifies
composition and state ownership while leaving kernel dispatch to structure
implementations.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml

from flash_rt.structures.registry import StructureSpec, load

_BINDINGS_DIR = pathlib.Path(__file__).resolve().parent / "bindings"
_BINDING_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_CLASSIFICATIONS = frozenset({
    "structure",
    "state_region",
    "host_stage",
    "control",
})
_COVERAGE_KEYS = frozenset({"contract", "hot_path", "segments"})
_SEGMENT_KEYS = frozenset({
    "name",
    "stage",
    "classification",
    "seam",
    "structures",
})


@dataclass(frozen=True)
class CoverageSegment:
    """One classified region of a native pipeline."""

    name: str
    stage: str
    classification: str
    seam: str
    structures: Sequence[str] = ()
    hot: bool = False

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        out: dict[str, Any] = {
            "name": self.name,
            "stage": self.stage,
            "classification": self.classification,
            "seam": self.seam,
            "hot": self.hot,
        }
        if self.structures:
            out["structures"] = list(self.structures)
        return out


@dataclass(frozen=True)
class BindingSpec:
    """One validated host binding."""

    name: str
    structure: StructureSpec
    data: Mapping[str, Any]
    stages: Mapping[str, Mapping[str, Any]]
    coverage_contract: str = ""
    hot_path: Sequence[str] = ()
    segments: Sequence[CoverageSegment] = ()

    @property
    def is_pipeline(self) -> bool:
        return self.structure.kind == "stage_pipeline"

    @property
    def coverage_declared(self) -> bool:
        return bool(self.coverage_contract)

    def manifest(self) -> dict[str, Any]:
        """Return the normalized portion consumed by runtime/tooling layers."""
        out: dict[str, Any] = {
            "binding": self.name,
            "structure": {
                "name": self.structure.name,
                "version": self.structure.version,
                "kind": self.structure.kind,
                "family": self.structure.family,
            },
        }
        if self.is_pipeline:
            out["stages"] = list(self.stages)
        if self.coverage_declared:
            out["coverage"] = {
                "contract": self.coverage_contract,
                "hot_path": list(self.hot_path),
                "segments": [segment.manifest() for segment in self.segments],
            }
        return out


def list_bindings(structure: str | None = None) -> list[str]:
    """List binding names, optionally filtered by catalog structure."""
    names = sorted(path.stem for path in _BINDINGS_DIR.glob("*.yaml"))
    if structure is None:
        return names
    return [
        name for name in names
        if load_binding(name).structure.name == structure
    ]


def load_binding(
    name: str,
    *,
    require_pipeline_coverage: bool = False,
) -> BindingSpec:
    """Load one binding and fail loudly on an invalid declaration."""
    if not isinstance(name, str) or not _BINDING_NAME.fullmatch(name):
        raise KeyError(f"invalid binding name: {name!r}")
    path = _BINDINGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown binding: {name!r}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return validate_binding(
        data,
        expected_name=name,
        require_pipeline_coverage=require_pipeline_coverage,
    )


def validate_binding(
    data: Mapping[str, Any],
    *,
    expected_name: str | None = None,
    require_pipeline_coverage: bool = False,
) -> BindingSpec:
    """Validate binding data and return its normalized contract."""
    if not isinstance(data, Mapping):
        raise ValueError("binding document must be a mapping")

    name = _required_text(data, "binding")
    if not _BINDING_NAME.fullmatch(name):
        raise ValueError(f"invalid binding name: {name!r}")
    if expected_name is not None and name != expected_name:
        raise ValueError(
            f"binding file {expected_name!r} declares binding {name!r}"
        )

    structure_name = _required_text(data, "structure")
    try:
        structure = load(structure_name)
    except KeyError as exc:
        raise ValueError(
            f"binding {name!r} references unknown structure "
            f"{structure_name!r}"
        ) from exc

    if structure.kind != "stage_pipeline":
        if "coverage" in data:
            raise ValueError(
                f"region binding {name!r} cannot declare pipeline coverage"
            )
        return BindingSpec(
            name=name,
            structure=structure,
            data=data,
            stages={},
        )

    stages = _validate_stages(name, structure, data.get("stages"))
    coverage = data.get("coverage")
    if coverage is None:
        if require_pipeline_coverage:
            raise ValueError(
                f"pipeline binding {name!r} has no complete-hot-path "
                "coverage contract"
            )
        return BindingSpec(
            name=name,
            structure=structure,
            data=data,
            stages=stages,
        )

    contract, hot_path, segments = _validate_coverage(
        name,
        coverage,
        stages,
        structure.embedded_regions,
    )
    return BindingSpec(
        name=name,
        structure=structure,
        data=data,
        stages=stages,
        coverage_contract=contract,
        hot_path=hot_path,
        segments=segments,
    )


def _validate_stages(
    binding_name: str,
    structure: StructureSpec,
    stages: Any,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(stages, Mapping):
        raise ValueError(
            f"pipeline binding {binding_name!r} must declare a stages mapping"
        )
    declared = {
        _required_text(stage, "name"): stage
        for stage in structure.stages
    }
    required = {
        name for name, stage in declared.items()
        if not bool(stage.get("optional", False))
    }
    names = set(stages)
    unknown = sorted(names - set(declared))
    missing = sorted(required - names)
    if unknown:
        raise ValueError(
            f"pipeline binding {binding_name!r} declares unknown stages: "
            f"{unknown}"
        )
    if missing:
        raise ValueError(
            f"pipeline binding {binding_name!r} is missing required stages: "
            f"{missing}"
        )
    normalized: dict[str, Mapping[str, Any]] = {}
    for stage_name, stage_binding in stages.items():
        if not isinstance(stage_name, str) or not stage_name:
            raise ValueError("pipeline stage names must be non-empty strings")
        if not isinstance(stage_binding, Mapping):
            raise ValueError(
                f"pipeline stage {stage_name!r} must be a mapping"
            )
        normalized[stage_name] = stage_binding
    return normalized


def _validate_coverage(
    binding_name: str,
    coverage: Any,
    stages: Mapping[str, Mapping[str, Any]],
    embedded_regions: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[CoverageSegment, ...]]:
    if not isinstance(coverage, Mapping):
        raise ValueError(
            f"pipeline binding {binding_name!r} coverage must be a mapping"
        )
    unknown_keys = sorted(set(coverage) - _COVERAGE_KEYS)
    if unknown_keys:
        raise ValueError(
            f"pipeline binding {binding_name!r} coverage has unknown keys: "
            f"{unknown_keys}"
        )
    contract = _required_text(coverage, "contract")
    if contract != "complete_hot_path":
        raise ValueError(
            f"pipeline binding {binding_name!r} has unsupported coverage "
            f"contract {contract!r}"
        )

    hot_path_data = coverage.get("hot_path")
    if not isinstance(hot_path_data, list) or not hot_path_data:
        raise ValueError(
            f"pipeline binding {binding_name!r} hot_path must be a non-empty list"
        )
    hot_path = tuple(_nonempty_text(value, "hot_path entry")
                     for value in hot_path_data)
    if len(set(hot_path)) != len(hot_path):
        raise ValueError(
            f"pipeline binding {binding_name!r} hot_path contains duplicates"
        )

    segment_data = coverage.get("segments")
    if not isinstance(segment_data, list) or not segment_data:
        raise ValueError(
            f"pipeline binding {binding_name!r} segments must be a non-empty list"
        )
    parsed: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for item in segment_data:
        if not isinstance(item, Mapping):
            raise ValueError("pipeline coverage segment must be a mapping")
        unknown = sorted(set(item) - _SEGMENT_KEYS)
        if unknown:
            raise ValueError(
                f"pipeline coverage segment has unknown keys: {unknown}"
            )
        name = _required_text(item, "name")
        if name in seen:
            raise ValueError(
                f"pipeline binding {binding_name!r} repeats segment {name!r}"
            )
        seen.add(name)
        stage = _required_text(item, "stage")
        if stage not in stages:
            raise ValueError(
                f"pipeline segment {name!r} references unbound stage {stage!r}"
            )
        classification = _required_text(item, "classification")
        if classification not in _CLASSIFICATIONS:
            raise ValueError(
                f"pipeline segment {name!r} has unsupported classification "
                f"{classification!r}"
            )
        seam = _required_text(item, "seam")
        structures = _validate_segment_structures(
            name,
            classification,
            item.get("structures"),
            embedded_regions,
        )
        parsed.append((name, stage, classification, seam, structures))

    missing = [name for name in hot_path if name not in seen]
    if missing:
        raise ValueError(
            f"pipeline binding {binding_name!r} leaves hot-path segments "
            f"unclassified: {missing}"
        )
    hot = set(hot_path)
    segments = tuple(
        CoverageSegment(
            name=name,
            stage=stage,
            classification=classification,
            seam=seam,
            structures=structures,
            hot=name in hot,
        )
        for name, stage, classification, seam, structures in parsed
    )
    return contract, hot_path, segments


def _validate_segment_structures(
    segment_name: str,
    classification: str,
    value: Any,
    embedded_regions: Sequence[str],
) -> tuple[str, ...]:
    if value is None:
        structures: tuple[str, ...] = ()
    elif isinstance(value, list) and value:
        structures = tuple(
            _nonempty_text(name, f"segment {segment_name!r} structure")
            for name in value
        )
    else:
        raise ValueError(
            f"pipeline segment {segment_name!r} structures must be a "
            "non-empty list when declared"
        )

    if classification == "structure" and not structures:
        raise ValueError(
            f"pipeline segment {segment_name!r} classified as structure "
            "must reference catalog structures"
        )
    if classification in {"host_stage", "control"} and structures:
        raise ValueError(
            f"pipeline segment {segment_name!r} classified as "
            f"{classification!r} cannot reference catalog structures"
        )
    if len(set(structures)) != len(structures):
        raise ValueError(
            f"pipeline segment {segment_name!r} repeats catalog structures"
        )
    allowed = set(embedded_regions)
    for structure_name in structures:
        try:
            spec = load(structure_name)
        except KeyError as exc:
            raise ValueError(
                f"pipeline segment {segment_name!r} references unknown "
                f"structure {structure_name!r}"
            ) from exc
        if spec.kind == "stage_pipeline":
            raise ValueError(
                f"pipeline segment {segment_name!r} cannot embed pipeline "
                f"structure {structure_name!r}"
            )
        if classification == "structure" and structure_name not in allowed:
            raise ValueError(
                f"pipeline segment {segment_name!r} references structure "
                f"{structure_name!r}, which is not embedded by its pipeline "
                "family"
            )
    return structures


def _required_text(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f"missing required binding field {key!r}")
    return _nonempty_text(data[key], key)


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
