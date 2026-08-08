"""Binding loader and pipeline coverage contract."""

import json

import pytest

from flash_rt.structures.binding import (
    list_bindings,
    load_binding,
    validate_binding,
)


def test_region_binding_loads_with_catalog_identity():
    binding = load_binding("pi05")

    assert binding.name == "pi05"
    assert binding.structure.name == "decoder_ffn"
    assert not binding.is_pipeline
    assert not binding.coverage_declared


def test_pi05_pipeline_binding_covers_every_declared_hot_segment():
    binding = load_binding(
        "pi05_tick",
        require_pipeline_coverage=True,
    )

    assert binding.structure.name == "vla_tick_pipeline"
    assert tuple(binding.stages) == ("obs_encode", "action_denoise")
    assert binding.coverage_contract == "complete_hot_path"
    assert {segment.name for segment in binding.segments if segment.hot} == {
        "observation_inputs",
        "vision_encoder",
        "multimodal_projector",
        "prefix_transformer",
        "prefix_kv_state",
        "denoise_control",
        "time_conditioning",
        "action_in_projection",
        "denoise_transformer",
        "action_readout",
        "action_window",
    }
    assert all(segment.classification != "unclassified"
               for segment in binding.segments)
    json.dumps(binding.manifest())


def test_binding_listing_can_filter_pipeline_family():
    assert "pi05_tick" in list_bindings("vla_tick_pipeline")
    assert "pi05" not in list_bindings("vla_tick_pipeline")


@pytest.mark.parametrize(
    ("name", "structure", "stages"),
    [
        (
            "qwen3_8b_pipeline",
            "autoregressive_decode_pipeline",
            {"input_prepare", "prefill", "decode", "token_select"},
        ),
        (
            "qwen3_vl_8b_pipeline",
            "autoregressive_decode_pipeline",
            {
                "input_prepare",
                "modality_encode",
                "prefill",
                "decode",
                "token_select",
            },
        ),
        (
            "motus_tick",
            "vla_tick_pipeline",
            {"obs_encode", "action_denoise", "readout"},
        ),
    ],
)
def test_extended_pipeline_bindings_have_complete_coverage(
    name,
    structure,
    stages,
):
    binding = load_binding(name, require_pipeline_coverage=True)

    assert binding.structure.name == structure
    assert set(binding.stages) == stages
    assert set(binding.hot_path) == {
        segment.name for segment in binding.segments if segment.hot
    }
    assert all(segment.classification != "unclassified"
               for segment in binding.segments)


def test_qwen_text_and_vl_share_pipeline_family_not_modality_stage():
    text = load_binding("qwen3_8b_pipeline", require_pipeline_coverage=True)
    vl = load_binding("qwen3_vl_8b_pipeline", require_pipeline_coverage=True)

    assert text.structure.name == vl.structure.name
    assert "modality_encode" not in text.stages
    assert "modality_encode" in vl.stages


def test_native_pipeline_gaps_remain_explicit_host_stages():
    qwen = load_binding("qwen3_8b_pipeline",
                        require_pipeline_coverage=True)
    motus = load_binding("motus_tick", require_pipeline_coverage=True)
    qwen_by_name = {segment.name: segment for segment in qwen.segments}
    motus_by_name = {segment.name: segment for segment in motus.segments}

    assert qwen_by_name["decode_norm_rope"].classification == "host_stage"
    assert motus_by_name["video_decode"].classification == "host_stage"


def test_legacy_pipeline_binding_is_visible_but_not_complete():
    binding = load_binding("smolvla_tick")
    assert binding.is_pipeline
    assert not binding.coverage_declared

    with pytest.raises(ValueError, match="no complete-hot-path"):
        load_binding("smolvla_tick", require_pipeline_coverage=True)


def test_pipeline_binding_rejects_unknown_stage():
    data = _minimal_pipeline()
    data["stages"]["unknown"] = {}

    with pytest.raises(ValueError, match="unknown stages"):
        validate_binding(data)


def test_pipeline_binding_rejects_missing_required_stage():
    data = _minimal_pipeline()
    del data["stages"]["action_denoise"]

    with pytest.raises(ValueError, match="missing required stages"):
        validate_binding(data)


def test_pipeline_binding_rejects_unclassified_hot_path():
    data = _minimal_pipeline()
    data["coverage"]["hot_path"].append("missing")

    with pytest.raises(ValueError, match="unclassified"):
        validate_binding(data)


def test_pipeline_binding_rejects_duplicate_segments():
    data = _minimal_pipeline()
    data["coverage"]["segments"].append(
        dict(data["coverage"]["segments"][0])
    )

    with pytest.raises(ValueError, match="repeats segment"):
        validate_binding(data)


def test_pipeline_binding_rejects_duplicate_hot_path_entries():
    data = _minimal_pipeline()
    data["coverage"]["hot_path"].append("loop")

    with pytest.raises(ValueError, match="hot_path contains duplicates"):
        validate_binding(data)


def test_pipeline_binding_rejects_unknown_coverage_field():
    data = _minimal_pipeline()
    data["coverage"]["runtime_benchmark"] = True

    with pytest.raises(ValueError, match="coverage has unknown keys"):
        validate_binding(data)


def test_pipeline_binding_rejects_unknown_classification():
    data = _minimal_pipeline()
    data["coverage"]["segments"][0]["classification"] = "kernel"

    with pytest.raises(ValueError, match="unsupported classification"):
        validate_binding(data)


def test_structure_segment_requires_catalog_references():
    data = _minimal_pipeline()
    data["coverage"]["segments"][0]["classification"] = "structure"

    with pytest.raises(ValueError, match="must reference catalog structures"):
        validate_binding(data)


def test_structure_segment_requires_known_catalog_structures():
    data = _minimal_pipeline()
    segment = data["coverage"]["segments"][0]
    segment["classification"] = "structure"
    segment["structures"] = ["not_a_structure"]

    with pytest.raises(ValueError, match="unknown structure"):
        validate_binding(data)


def test_structure_segment_must_belong_to_pipeline_family():
    data = _minimal_pipeline()
    segment = data["coverage"]["segments"][0]
    segment["classification"] = "structure"
    segment["structures"] = ["cadence_static"]

    with pytest.raises(ValueError, match="not embedded by its pipeline"):
        validate_binding(data)


def test_host_and_control_segments_cannot_claim_structures():
    for classification in ("host_stage", "control"):
        data = _minimal_pipeline()
        segment = data["coverage"]["segments"][0]
        segment["classification"] = classification
        segment["structures"] = ["linear_proj"]

        with pytest.raises(ValueError, match="cannot reference"):
            validate_binding(data)


def test_pipeline_structure_cannot_be_nested_as_region():
    data = _minimal_pipeline()
    segment = data["coverage"]["segments"][0]
    segment["classification"] = "structure"
    segment["structures"] = ["vla_tick_pipeline"]

    with pytest.raises(ValueError, match="cannot embed pipeline"):
        validate_binding(data)


def test_region_binding_cannot_declare_pipeline_coverage():
    data = {
        "binding": "region_fixture",
        "structure": "decoder_ffn",
        "coverage": {
            "contract": "complete_hot_path",
            "hot_path": ["ffn"],
            "segments": [],
        },
    }

    with pytest.raises(ValueError, match="cannot declare pipeline coverage"):
        validate_binding(data)


def test_binding_filename_identity_and_name_safety_are_enforced():
    data = {
        "binding": "declared_name",
        "structure": "decoder_ffn",
    }
    with pytest.raises(ValueError, match="declares binding"):
        validate_binding(data, expected_name="file_name")
    with pytest.raises(KeyError, match="invalid binding name"):
        load_binding("../pi05")


def _minimal_pipeline():
    return {
        "binding": "contract_fixture",
        "structure": "vla_tick_pipeline",
        "stages": {
            "obs_encode": {"seam": "fixture.encode"},
            "action_denoise": {"seam": "fixture.denoise"},
        },
        "coverage": {
            "contract": "complete_hot_path",
            "hot_path": ["loop"],
            "segments": [{
                "name": "loop",
                "stage": "action_denoise",
                "classification": "control",
                "seam": "fixture.loop",
            }],
        },
    }
