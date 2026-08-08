"""Catalog and release-facing contracts for pipeline structures."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import yaml

from flash_rt.structures import registry
from flash_rt.structures.binding import list_bindings, load_binding
from flash_rt.structures.registry import load as load_structure

_COMPLETE_PIPELINES = {
    "pi05_tick": {
        "structure": "vla_tick_pipeline",
        "stages": {"obs_encode", "action_denoise"},
        "counts": {
            "control": 1,
            "host_stage": 1,
            "state_region": 2,
            "structure": 7,
        },
        "host_gaps": {"observation_inputs"},
    },
    "qwen3_8b_pipeline": {
        "structure": "autoregressive_decode_pipeline",
        "stages": {"input_prepare", "prefill", "decode", "token_select"},
        "counts": {
            "control": 2,
            "host_stage": 5,
            "state_region": 2,
            "structure": 4,
        },
        "host_gaps": {
            "request_tokenize",
            "prefill_embedding",
            "prefill_norm_rope",
            "decode_embedding",
            "decode_norm_rope",
        },
    },
    "qwen3_vl_8b_pipeline": {
        "structure": "autoregressive_decode_pipeline",
        "stages": {
            "input_prepare",
            "modality_encode",
            "prefill",
            "decode",
            "token_select",
        },
        "counts": {
            "control": 2,
            "host_stage": 4,
            "state_region": 2,
            "structure": 10,
        },
        "host_gaps": {
            "message_preprocess",
            "multimodal_scatter",
            "deepstack_injection",
            "decode_embedding",
        },
    },
    "motus_tick": {
        "structure": "vla_tick_pipeline",
        "stages": {"obs_encode", "action_denoise", "readout"},
        "counts": {
            "control": 1,
            "host_stage": 4,
            "state_region": 4,
            "structure": 7,
        },
        "host_gaps": {
            "first_frame_encode",
            "understanding_tokens",
            "video_decode",
            "action_readout",
        },
    },
}


def test_pipeline_catalogs_declare_shared_schedule_families():
    autoregressive = load_structure("autoregressive_decode_pipeline")
    vla = load_structure("vla_tick_pipeline")

    assert autoregressive.kind == "stage_pipeline"
    assert autoregressive.family == "autoregressive_decode"
    assert autoregressive.version == 2
    assert _stage_sets(autoregressive) == (
        {"prefill", "decode", "token_select"},
        {"input_prepare", "modality_encode"},
    )

    assert vla.kind == "stage_pipeline"
    assert vla.family == "cond_iter_pipeline"
    assert vla.version == 2
    assert _stage_sets(vla) == (
        {"obs_encode", "action_denoise"},
        {"readout"},
    )


def test_every_binding_document_loads_with_filename_identity():
    names = list_bindings()

    assert names == sorted(names)
    assert len(names) == len(set(names))
    for name in names:
        assert load_binding(name).name == name


def test_complete_pipeline_manifests_cover_each_hot_segment_once():
    for name, expected in _COMPLETE_PIPELINES.items():
        binding = load_binding(name, require_pipeline_coverage=True)
        segments = {segment.name: segment for segment in binding.segments}

        assert binding.structure.name == expected["structure"]
        assert set(binding.stages) == expected["stages"]
        assert len(segments) == len(binding.segments)
        assert set(binding.hot_path) == set(segments)
        assert all(segment.hot for segment in binding.segments)
        assert all(segment.stage in binding.stages
                   for segment in binding.segments)
        assert Counter(segment.classification
                       for segment in binding.segments) == expected["counts"]


def test_pipeline_structure_segments_are_declared_by_their_family():
    for name in _COMPLETE_PIPELINES:
        binding = load_binding(name, require_pipeline_coverage=True)
        allowed = set(binding.structure.embedded_regions)

        for segment in binding.segments:
            if segment.classification == "structure":
                assert segment.structures
                assert set(segment.structures) <= allowed


def test_known_host_gaps_stay_explicit_until_a_structure_lands():
    for name, expected in _COMPLETE_PIPELINES.items():
        binding = load_binding(name, require_pipeline_coverage=True)
        gaps = {
            segment.name
            for segment in binding.segments
            if segment.classification == "host_stage"
        }

        assert gaps == expected["host_gaps"]


def test_text_and_multimodal_qwen_share_one_language_schedule():
    text = load_binding("qwen3_8b_pipeline",
                        require_pipeline_coverage=True)
    multimodal = load_binding("qwen3_vl_8b_pipeline",
                              require_pipeline_coverage=True)

    assert text.structure.name == multimodal.structure.name
    assert {"prefill", "decode", "token_select"} <= set(text.stages)
    assert {"prefill", "decode", "token_select"} <= set(multimodal.stages)
    assert "modality_encode" not in text.stages
    assert "modality_encode" in multimodal.stages


def test_vla_v2_optional_readout_keeps_legacy_bindings_valid():
    for name in ("smolvla_tick", "groot_n16_tick", "pi05_tick"):
        binding = load_binding(name)
        assert binding.structure.version == 2
        assert "readout" not in binding.stages

    motus = load_binding("motus_tick", require_pipeline_coverage=True)
    assert "readout" in motus.stages


def test_normalized_pipeline_manifest_excludes_dispatch_and_host_details():
    prohibited = {
        "arch",
        "backend",
        "device",
        "hardware",
        "hosts",
        "kernel",
        "runtime_benchmark",
    }
    for name in _COMPLETE_PIPELINES:
        manifest = load_binding(
            name,
            require_pipeline_coverage=True,
        ).manifest()

        json.dumps(manifest)
        assert not (_mapping_keys(manifest) & prohibited)


def test_pipeline_catalogs_do_not_own_hardware_dispatch():
    prohibited = {
        "arch",
        "backend",
        "device",
        "hardware",
        "kernel",
        "runtime_benchmark",
    }
    catalog_root = Path(registry.__file__).resolve().parent / "catalog"

    for name in ("autoregressive_decode_pipeline", "vla_tick_pipeline"):
        data = yaml.safe_load(
            (catalog_root / name / "structure.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert not (_mapping_keys(data) & prohibited)


def test_every_embedded_region_resolves_to_a_region_structure():
    for pipeline_name in (
        "autoregressive_decode_pipeline",
        "vla_tick_pipeline",
    ):
        pipeline = load_structure(pipeline_name)
        assert pipeline.embedded_regions
        assert len(pipeline.embedded_regions) == len(
            set(pipeline.embedded_regions)
        )
        for region_name in pipeline.embedded_regions:
            assert load_structure(region_name).kind == "region"


def _stage_sets(spec):
    required = {
        stage["name"] for stage in spec.stages
        if not stage.get("optional", False)
    }
    optional = {
        stage["name"] for stage in spec.stages
        if stage.get("optional", False)
    }
    return required, optional


def _mapping_keys(value):
    keys = set()
    if isinstance(value, Mapping):
        keys.update(value)
        for child in value.values():
            keys.update(_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys
