from __future__ import annotations

import os

import pytest
import torch

from flash_rt.frontends.torch.groot_n17_rtx import GrootN17TorchFrontendRtx
from flash_rt.frontends.torch.groot_n17_rtx_fp8 import (
    GrootN17TorchFrontendRtxFP8,
)


def test_fp8_infer_replays_backbone_before_action_graph(monkeypatch):
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    features = object()
    calls = []

    def capture_backbone_graph():
        calls.append(("capture",))
        frontend._kbb_graph = object()

    def run_backbone_graph(aux):
        calls.append(("backbone", aux))
        return features

    def action_infer(self, state, **kwargs):
        calls.append(("action", self._backbone_features, state, kwargs))
        return "actions"

    frontend.run_backbone_graph = run_backbone_graph
    frontend._capture_backbone_graph = capture_backbone_graph
    frontend._validate_backbone_graph_contract = lambda aux: None
    monkeypatch.setattr(GrootN17TorchFrontendRtx, "infer", action_infer)

    state = torch.empty(1, 1, 132)
    aux = {"pixel_features": object()}
    result = frontend.infer(state, aux=aux)

    assert result == "actions"
    assert calls[0] == ("capture",)
    assert calls[1] == ("backbone", aux)
    assert calls[2][0:3] == ("action", features, state)
    assert calls[2][3]["use_dit_graph"] is True

    frontend.infer(state, aux=aux)
    assert [call for call in calls if call == ("capture",)] == [("capture",)]


def test_fp8_infer_without_aux_reuses_prompt_backbone(monkeypatch):
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    features = object()
    frontend._backbone_features = features

    def fail_if_captured():
        raise AssertionError("backbone graph should not capture without aux")

    def fail_if_replayed(aux):
        raise AssertionError("backbone graph should not replay without aux")

    def action_infer(self, state, **kwargs):
        assert self._backbone_features is features
        return "actions"

    frontend.run_backbone_graph = fail_if_replayed
    frontend._capture_backbone_graph = fail_if_captured
    monkeypatch.setattr(GrootN17TorchFrontendRtx, "infer", action_infer)

    assert frontend.infer(torch.empty(1, 1, 132)) == "actions"


def test_run_backbone_graph_refreshes_persistent_inputs():
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    frontend.device = torch.device("cpu")
    frontend._S_vit = 2
    frontend.Se = 3
    frontend._kbb_vit_h = torch.zeros(2, 1024, dtype=torch.float16)
    frontend._kbb_llm_h = torch.zeros(3, 2048, dtype=torch.float16)
    frontend._kbb_vlsa_h = torch.ones(3, 2048, dtype=torch.float16)
    frontend._validate_backbone_graph_contract = lambda aux: None

    class Graph:
        replay_count = 0

        def replay(self):
            self.replay_count += 1

    graph = Graph()
    frontend._kbb_graph = graph
    aux = {
        "pixel_features": torch.full((1, 2, 1024), 2.0),
        "llm_input_embeds": torch.full((1, 3, 2048), 3.0),
    }

    result = frontend.run_backbone_graph(aux)

    assert graph.replay_count == 1
    assert torch.all(frontend._kbb_vit_h == 2)
    assert torch.all(frontend._kbb_llm_h == 3)
    assert result.data_ptr() == frontend._kbb_vlsa_h.data_ptr()
    assert result.shape == (1, 3, 2048)


def _contract_aux():
    return {
        "pixel_features": torch.zeros(2, 1024),
        "llm_input_embeds": torch.zeros(1, 3, 2048),
        "grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "visual_pos_masks": torch.tensor([[False, True, True]]),
        "rope_cos": torch.zeros(1, 3, 128, dtype=torch.bfloat16),
        "rope_sin": torch.ones(1, 3, 128, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pixel_features", torch.zeros(3, 1024)),
        ("llm_input_embeds", torch.zeros(1, 4, 2048)),
        ("grid_thw", torch.tensor([[1, 4, 2]])),
        ("visual_pos_masks", torch.tensor([[True, False, True]])),
        ("rope_cos", torch.ones(1, 3, 128, dtype=torch.bfloat16)),
        ("rope_sin", torch.zeros(1, 3, 128, dtype=torch.bfloat16)),
    ],
)
def test_backbone_graph_contract_rejects_structural_changes(field, replacement):
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    original = _contract_aux()
    frontend._backbone_graph_contract = frontend._snapshot_backbone_graph_contract(
        original)
    changed = dict(original)
    changed[field] = replacement

    with pytest.raises(ValueError, match=field):
        frontend._validate_backbone_graph_contract(changed)


def test_backbone_graph_contract_allows_fresh_feature_values():
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    original = _contract_aux()
    frontend._backbone_graph_contract = frontend._snapshot_backbone_graph_contract(
        original)
    fresh = dict(original)
    fresh["pixel_features"] = torch.ones_like(original["pixel_features"])
    fresh["llm_input_embeds"] = torch.ones_like(original["llm_input_embeds"])

    frontend._validate_backbone_graph_contract(fresh)


def test_backbone_graph_contract_detects_in_place_metadata_mutation():
    frontend = object.__new__(GrootN17TorchFrontendRtxFP8)
    aux = _contract_aux()
    frontend._backbone_graph_contract = frontend._snapshot_backbone_graph_contract(aux)
    aux["rope_cos"].add_(1)

    with pytest.raises(ValueError, match="rope_cos"):
        frontend._validate_backbone_graph_contract(aux)


@pytest.mark.skipif(
    os.environ.get("FLASH_RT_RUN_GROOT_N17_GRAPH_GPU_TEST") != "1",
    reason="set FLASH_RT_RUN_GROOT_N17_GRAPH_GPU_TEST=1 for the fixture GPU test",
)
def test_gpu_backbone_graph_replay_matches_eager_for_fresh_input():
    checkpoint = os.environ["FLASH_RT_GROOT_N17_CKPT"]
    aux_path = os.environ["FLASH_RT_GROOT_N17_AUX"]
    aux = torch.load(aux_path, map_location="cpu", weights_only=False)

    frontend = GrootN17TorchFrontendRtxFP8(
        checkpoint,
        num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint",
    )
    frontend.set_prompt(aux=aux, prompt="Put the blue block in the green bowl")
    assert not hasattr(frontend, "_kbb_graph")
    frontend._capture_backbone_graph()

    fresh = dict(aux)
    fresh["pixel_features"] = aux["pixel_features"].clone().add_(0.01)
    graph_first = frontend.run_backbone_graph(fresh).clone()
    graph_second = frontend.run_backbone_graph(fresh).clone()

    frontend._kbb_vit_h.copy_(fresh["pixel_features"].cuda().half())
    frontend._kbb_llm_h.copy_(fresh["llm_input_embeds"].cuda().half()[0])
    eager = frontend._kbb_forward(0).unsqueeze(0).clone()
    torch.cuda.synchronize()

    assert torch.equal(graph_first, graph_second)
    assert torch.equal(graph_second, eager)
