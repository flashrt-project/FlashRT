"""Shared pipeline contract for Cosmos and Wan video generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from flash_rt.structures.binding import load_binding
from flash_rt.structures.registry import load as load_structure


_ROOT = Path(__file__).resolve().parents[1]
_BINDINGS = ("cosmos3_video_pipeline", "wan22_video_pipeline")


def test_video_generation_is_a_cond_iter_sibling_not_a_vla_alias():
    video = load_structure("video_generation_pipeline")
    vla = load_structure("vla_tick_pipeline")

    assert video.kind == "stage_pipeline"
    assert video.family == vla.family == "cond_iter_pipeline"
    assert video.name != vla.name
    assert {stage["name"] for stage in video.stages} == {
        "input_prepare",
        "condition_encode",
        "latent_prepare",
        "denoise",
        "decode",
    }


def test_cosmos_and_wan_bind_the_same_complete_generation_schedule():
    cosmos = load_binding("cosmos3_video_pipeline",
                          require_pipeline_coverage=True)
    wan = load_binding("wan22_video_pipeline",
                       require_pipeline_coverage=True)

    assert cosmos.structure.name == wan.structure.name == (
        "video_generation_pipeline"
    )
    assert set(cosmos.stages) == {
        "input_prepare", "condition_encode", "latent_prepare", "denoise"
    }
    assert set(wan.stages) == {
        "input_prepare", "condition_encode", "latent_prepare", "denoise",
        "decode",
    }
    for binding in (cosmos, wan):
        assert set(binding.hot_path) == {
            segment.name for segment in binding.segments
        }
        json.dumps(binding.manifest())


def test_landed_qk_region_and_remaining_video_gaps_are_explicit():
    cosmos = load_binding("cosmos3_video_pipeline",
                          require_pipeline_coverage=True)
    wan = load_binding("wan22_video_pipeline",
                       require_pipeline_coverage=True)
    cosmos_segments = {segment.name: segment for segment in cosmos.segments}
    wan_segments = {segment.name: segment for segment in wan.segments}

    assert cosmos_segments["qk_norm_rope"].structures == ("qk_norm_rope",)
    assert wan_segments["qk_norm_rope"].structures == ("qk_norm_rope",)
    assert wan_segments["six_chunk_modulation"].classification == "host_stage"
    # covered since vision_ffn learned the diffusers FeedForward shape;
    # the segment's modulation and gated residual stay host
    assert wan_segments["modulated_ffn"].classification == "structure"
    assert wan_segments["modulated_ffn"].structures == ("vision_ffn",)
    assert wan_segments["vae_decode"].classification == "host_stage"


def test_cosmos_binding_names_the_official_host_and_factored_qk_boundary():
    cosmos = load_binding("cosmos3_video_pipeline",
                          require_pipeline_coverage=True)

    assert cosmos.data["hosts"]["official"]["module_path"] == (
        "diffusers_cosmos3.transformer.Cosmos3OmniTransformer"
    )
    qk = {segment.name: segment for segment in cosmos.segments}[
        "qk_norm_rope"
    ]
    assert "factored causal/full QKV" in qk.seam


def test_video_pipeline_source_seams_are_real():
    expected = {
        "flash_rt/models/cosmos3_video/pipeline_rtx.py": (
            "class CosmosVideo",
            "def precompute_und",
            "def embed_gen",
            "def forward",
            "qk_norm_rope",
        ),
        "flash_rt/frontends/torch/cosmos3_video_rtx.py": (
            "class Cosmos3VideoTorchFrontendRtx",
            "def set_prompt",
            "def _denoise",
            "def infer",
        ),
        "flash_rt/frontends/torch/wan22_rtx.py": (
            "class Wan22TorchFrontendRtx",
            "def set_prompt",
            "def infer",
        ),
    }

    for relative, needles in expected.items():
        source = (_ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in source


def test_video_pipeline_catalog_does_not_own_hardware_dispatch():
    data = yaml.safe_load(
        (
            _ROOT
            / "flash_rt/structures/catalog/video_generation_pipeline/"
            "structure.yaml"
        ).read_text(encoding="utf-8")
    )
    prohibited = {
        "arch",
        "backend",
        "device",
        "hardware",
        "kernel",
        "runtime_benchmark",
    }

    assert not (_mapping_keys(data) & prohibited)


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
