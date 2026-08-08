"""Pipeline coverage for hybrid-attention Qwen3.6 and Nex-N2."""

from __future__ import annotations

from pathlib import Path

import pytest

from flash_rt.structures.binding import load_binding


_ROOT = Path(__file__).resolve().parents[1]
_BINDINGS = ("qwen36_27b_pipeline", "nexn2_pipeline")


@pytest.mark.parametrize("name", _BINDINGS)
def test_hybrid_qwen_bindings_cover_the_complete_language_schedule(name):
    binding = load_binding(name, require_pipeline_coverage=True)

    assert binding.structure.name == "autoregressive_decode_pipeline"
    assert set(binding.stages) == {
        "input_prepare",
        "prefill",
        "decode",
        "token_select",
    }
    assert set(binding.hot_path) == {
        segment.name for segment in binding.segments
    }


@pytest.mark.parametrize("name", _BINDINGS)
def test_gdn_decode_is_owned_while_prefill_stays_an_explicit_gap(name):
    binding = load_binding(name, require_pipeline_coverage=True)
    segments = {segment.name: segment for segment in binding.segments}

    assert segments["prefill_gdn"].classification == "host_stage"
    assert segments["decode_gdn"].classification == "structure"
    assert segments["decode_gdn"].structures == ("gated_delta_core",)


def test_nexn2_moe_remains_a_gap_until_moe_ffn_lands():
    binding = load_binding("nexn2_pipeline",
                           require_pipeline_coverage=True)
    segments = {segment.name: segment for segment in binding.segments}

    assert segments["prefill_moe"].classification == "host_stage"
    assert segments["decode_moe"].classification == "host_stage"


def test_qwen36_and_nexn2_source_seams_are_real():
    expected = {
        "flash_rt/frontends/torch/qwen36_rtx.py": (
            "class Qwen36TorchFrontendRtx",
            "def set_prompt",
            "def generate_own",
            "def generate_own_speculative_KN_nvfp4",
            "def _layer_forward_full_nvfp4",
            "def _layer_forward_lin_nvfp4",
        ),
        "flash_rt/frontends/torch/_nexn2_rtx_forward.py": (
            "def _gdn_layer",
            "def _full_attn_layer",
            "def _moe_layer",
            "def nexn2_forward_nvfp4",
        ),
        "flash_rt/frontends/torch/_nexn2_rtx_decode.py": (
            "def _decode_gdn",
            "def _decode_full",
            "def _moe_layer_decode",
            "def decode_step",
            "def seed_prefill_batched",
        ),
    }

    for relative, needles in expected.items():
        source = (_ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in source
