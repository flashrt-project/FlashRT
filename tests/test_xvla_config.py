"""X-VLA metadata YAML must match the pipeline's hard-coded geometry."""

from pathlib import Path

import yaml

from flash_rt.models.xvla import pipeline_rtx as p

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_xvla_yaml_matches_pipeline_constants():
    cfg = yaml.safe_load((REPO_ROOT / "flash_rt/configs/xvla.yaml").read_text())

    assert cfg["vlm_layers"] == p.ENC_LAYERS
    assert cfg["expert_layers"] == p.ACT_LAYERS
    assert cfg["expert_hidden"] == p.HIDDEN
    assert cfg["expert_mlp"] == p.ACT_MLP
    assert cfg["num_heads"] == p.ACT_HEADS
    assert cfg["head_dim"] == p.HIDDEN // p.ACT_HEADS
    assert cfg["action_dim"] == p.ACTION_DIM
    assert cfg["chunk_size"] == p.CHUNK
    assert cfg["state_dim"] == p.DIM_PROPRIO
    assert cfg["soft_prompts"] == p.N_SOFT
    assert cfg["decoder_steps"] == p.N_DENOISE
    assert cfg["image_views"] == p.N_VIEWS
