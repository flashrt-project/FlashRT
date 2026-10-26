"""GROOT N1.5 on RTX SM89 -- routing, observation building, K/V cache, GPU e2e.

What this file protects:

  * the ``("groot_n15", "torch", "rtx_sm89")`` entry in
    ``flash_rt.hardware._PIPELINE_MAP`` (resolution is lazy, so a rename would
    only break at runtime) and that no other arch resolves it;
  * the frontend's configuration errors (missing source root / data config),
    which must fire before any import of the upstream ``gr00t`` package;
  * ``_build_observation``: GR00T-native key layout (``[1, H, W, 3]`` video,
    ``[1, D]`` state, ``[prompt]`` language) from both ``images=[...]`` and
    native-key observations, and clear errors for missing inputs;
  * ``_CachedProjection``: memoised only while the graph forward is active.

Environment matrix:
  - CPU / no checkpoint: everything except the e2e test runs (no torch GPU,
    no ``gr00t`` needed).
  - RTX 4090 + checkpoint: set ``GROOT_N15_CHECKPOINT``, ``GROOT_N15_ROOT``
    and ``GROOT_N15_DATA_CONFIG`` (and install ``flash_attn``) to also run
    ``test_load_model_graph_matches_eager``.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

_MODULE = "flash_rt.frontends.torch.groot_n15_rtx_sm89"
_CLASS = "GrootN15TorchFrontendRtxSm89"
_VIDEO = ("video.a", "video.b")
_STATE = ("state.pos", "state.grip")


def _frontend():
    cls = resolve_pipeline_class("groot_n15", "torch", "rtx_sm89")
    fe = object.__new__(cls)
    fe._video_keys = _VIDEO
    fe._state_keys = _STATE
    fe._prompt = "pick the cup"
    return fe


def test_pipeline_map_pins_the_sm89_frontend():
    assert _PIPELINE_MAP[("groot_n15", "torch", "rtx_sm89")] == (_MODULE, _CLASS)
    assert resolve_pipeline_class("groot_n15", "torch", "rtx_sm89").__name__ == _CLASS
    for arch in ("thor", "rtx_sm120"):
        with pytest.raises(RuntimeError, match="rtx_sm89"):
            resolve_pipeline_class("groot_n15", "torch", arch)


def test_missing_root_or_data_config_is_refused(monkeypatch, tmp_path):
    cls = resolve_pipeline_class("groot_n15", "torch", "rtx_sm89")
    monkeypatch.delenv("GROOT_N15_ROOT", raising=False)
    monkeypatch.delenv("GROOT_N15_DATA_CONFIG", raising=False)
    with pytest.raises(ValueError, match="GROOT_N15_ROOT"):
        cls(tmp_path, data_config="x")
    with pytest.raises(ValueError, match="GROOT_N15_DATA_CONFIG"):
        cls(tmp_path, groot_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        cls(tmp_path / "missing", groot_root=tmp_path, data_config="x")


def test_observation_from_image_list_and_state_dict():
    fe = _frontend()
    img = np.zeros((8, 8, 3), np.uint8)
    obs = fe._build_observation({
        "images": [img, img + 1],
        "state": {"state.pos": [1, 2, 3], "grip": [0.5, 0.5]},
    })
    assert obs["video.a"].shape == (1, 8, 8, 3) and obs["video.a"].dtype == np.uint8
    assert obs["video.b"].max() == 1
    assert obs["state.pos"].shape == (1, 3)
    assert obs["state.grip"].tolist() == [[0.5, 0.5]]
    assert obs["annotation.human.task_description"] == ["pick the cup"]


def test_observation_accepts_native_keys():
    fe = _frontend()
    img = np.zeros((4, 4, 3), np.uint8)
    obs = fe._build_observation({
        "video.a": img, "video.b": img,
        "state.pos": np.zeros(3), "state.grip": np.zeros(2),
    })
    assert set(obs) == {*_VIDEO, *_STATE, "annotation.human.task_description"}


def test_observation_reports_missing_inputs():
    fe = _frontend()
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError, match="video.b"):
        fe._build_observation({"images": [img], "state": {}})
    with pytest.raises(ValueError, match="state.pos"):
        fe._build_observation({"images": [img, img], "state": {}})


def test_infer_requires_prompt():
    fe = _frontend()
    fe._prompt = None
    with pytest.raises(RuntimeError, match="set_prompt"):
        fe.infer({})


def test_cached_projection_only_memoises_while_active():
    from flash_rt.frontends.torch._groot_n15_graph import _CachedProjection

    calls = []
    state = types.SimpleNamespace(active=False)

    def linear(x):
        calls.append(x)
        return x * 2

    proj = _CachedProjection(linear, state)
    assert proj(1) == 2 and proj(1) == 2 and len(calls) == 2  # pass-through
    state.active = True
    assert proj(3) == 6 and proj(99) == 6 and len(calls) == 3  # memoised
    proj.value = None
    assert proj(5) == 10


_E2E = ("GROOT_N15_CHECKPOINT", "GROOT_N15_ROOT", "GROOT_N15_DATA_CONFIG")


@pytest.mark.skipif(
    not all(os.environ.get(k) for k in _E2E),
    reason="needs GROOT_N15_CHECKPOINT/ROOT/DATA_CONFIG and an RTX 4090")
def test_load_model_graph_matches_eager():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import flash_rt

    model = flash_rt.load_model(
        os.environ["GROOT_N15_CHECKPOINT"], config="groot_n15",
        hardware="rtx_sm89", num_views=int(os.environ.get("GROOT_N15_VIEWS", "3")))
    fe = model.pipeline
    rng = np.random.default_rng(0)
    obs = {
        "images": [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                   for _ in fe._video_keys],
        "state": {k: rng.normal(size=dim).astype(np.float64) * 0.1
                  for k, dim in _state_dims(fe).items()},
    }
    torch.manual_seed(0)
    graph = model.predict(obs["images"], prompt="pick up the cup", state=obs["state"])
    runner = fe.graph_runner
    assert runner.captures == 1 and runner.eager_fallbacks == 0
    torch.manual_seed(0)
    fe._policy._get_action_from_normalized_input = runner._eager
    try:
        eager = fe._policy.get_action(fe._build_observation(obs))
    finally:
        fe._policy._get_action_from_normalized_input = runner.get_action
    for key, value in graph.items():
        assert np.isfinite(value).all()
        assert np.abs(value - eager[key]).max() < 0.05, key
    model.predict(obs["images"], state=obs["state"])
    assert runner.replays >= 2


def _state_dims(fe):
    import json
    import pathlib

    meta = json.loads((pathlib.Path(os.environ["GROOT_N15_CHECKPOINT"])
                       / "experiment_cfg" / "metadata.json").read_text())
    stats = next(iter(meta.values()))["statistics"]["state"]
    return {k: len(stats[k.removeprefix("state.")]["min"])
            for k in fe._state_keys}
