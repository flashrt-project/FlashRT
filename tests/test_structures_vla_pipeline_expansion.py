"""Complete coverage contracts for the expanded VLA host set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flash_rt.structures.binding import load_binding


_ROOT = Path(__file__).resolve().parents[1]
_BINDINGS = (
    "smolvla_pipeline",
    "groot_n16_pipeline",
    "groot_n17_pipeline",
    "lingbot_vla_pipeline",
)


@pytest.mark.parametrize("name", _BINDINGS)
def test_expanded_vla_bindings_cover_the_complete_hot_path(name):
    binding = load_binding(name, require_pipeline_coverage=True)

    assert binding.structure.name == "vla_tick_pipeline"
    assert set(binding.stages) == {"obs_encode", "action_denoise"}
    assert set(binding.hot_path) == {
        segment.name for segment in binding.segments
    }
    assert len(binding.hot_path) == len(set(binding.hot_path))
    assert all(segment.hot for segment in binding.segments)
    json.dumps(binding.manifest())


@pytest.mark.parametrize("name", _BINDINGS)
def test_expanded_vla_bindings_keep_control_state_and_host_gaps_explicit(name):
    binding = load_binding(name, require_pipeline_coverage=True)
    classes = {segment.classification for segment in binding.segments}

    assert {"structure", "state_region", "host_stage", "control"} <= classes


def test_vla_bindings_share_schedule_without_sharing_host_seams():
    bindings = [
        load_binding(name, require_pipeline_coverage=True)
        for name in _BINDINGS
    ]

    assert {binding.structure.family for binding in bindings} == {
        "cond_iter_pipeline"
    }
    assert len({
        binding.data["stages"]["obs_encode"]["seam"]
        for binding in bindings
    }) == len(bindings)


def test_key_vla_seams_are_present_in_the_native_sources():
    expected = {
        "flash_rt/models/groot/pipeline_rtx.py": (
            "class GrootSigLIP2",
            "class GrootQwen3",
            "class GrootDiT",
            "def precompute_cross_kv",
            "def run_steps",
        ),
        "flash_rt/frontends/torch/groot_n17_thor.py": (
            "class GrootN17TorchFrontendThor",
            "def set_prompt",
            "def infer",
            "def _cross_kv_fwd",
            "def _precompute_diffusion_modulators",
        ),
        "flash_rt/models/lingbot/sample_actions.py": (
            "def embed_prefix",
            "def embed_suffix",
            "def predict_velocity",
            "def sample_actions",
        ),
        "flash_rt/models/lingbot/forward.py": (
            "def prefix_encode_36L",
            "def denoise_step_36L",
        ),
    }

    for relative, needles in expected.items():
        source = (_ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in source

