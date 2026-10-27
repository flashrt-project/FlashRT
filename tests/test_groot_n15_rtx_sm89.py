"""GR00T N1.5 on RTX SM89: routing, contracts, fail-fast paths, and parity.

Runs without a GPU or the upstream package:
  - routing, model-package import, the target contract;
  - checkpoint-config parsing and its refusals;
  - required-symbol checks, the attention split rule, the lowerings' guards;
  - observation assembly and the IO adapter's early refusals.

RTX 4090 + checkpoint: set ``GROOT_N15_CHECKPOINT``, ``GROOT_N15_ROOT`` (the
Isaac-GR00T source providing ``gr00t``) and ``GROOT_N15_DATA_CONFIG`` to also
run the parity tests. They build the unmodified upstream ``Gr00tPolicy`` as an
independent reference (it needs ``flash_attn`` importable), feed the
FlashRT frontend the same normalized inputs and initial noise, and check the
actions, the discrete outputs, CUDA-graph replay against eager execution,
A -> B -> A observations through one graph, and recapture after eviction.

Usage:
  PYTHONPATH=. pytest -q tests/test_groot_n15_rtx_sm89.py
"""
from __future__ import annotations

import copy
import json
import os
import types

import numpy as np
import pytest

from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

KEY = ("groot_n15", "torch", "rtx_sm89")
FRONTEND = ("flash_rt.frontends.torch.groot_n15_rtx_sm89", "GrootN15TorchFrontendRtxSm89")

# The geometry of the released GR00T-N1.5-3B checkpoints.
CONFIG = {
    "model_type": "gr00t_n1_5",
    "backbone_cfg": {"select_layer": 12, "project_to_dim": None, "reproject_vision": False},
    "action_head_cfg": {
        "action_dim": 32, "action_horizon": 16, "add_pos_embed": True,
        "backbone_embedding_dim": 2048, "hidden_size": 1024, "input_embedding_dim": 1536,
        "max_state_dim": 64, "num_inference_timesteps": 4, "num_target_vision_tokens": 32,
        "num_timestep_buckets": 1000, "use_vlln": True,
        "diffusion_model_cfg": {
            "attention_head_dim": 48, "cross_attention_dim": 2048,
            "interleave_self_attention": True, "norm_type": "ada_norm",
            "num_attention_heads": 32, "num_layers": 16, "output_dim": 1024,
            "positional_embeddings": None},
        "vl_self_attention_cfg": {
            "attention_head_dim": 64, "num_attention_heads": 32, "num_layers": 4,
            "positional_embeddings": None},
    },
}
EAGLE = {
    "use_pixel_shuffle": False, "mlp_connector_layers": 1, "select_layer": -1,
    "force_image_size": 224, "image_token_index": 151669,
    "vision_config": {"model_type": "siglip_vision_model", "hidden_act": "gelu_pytorch_tanh",
                      "hidden_size": 1152, "image_size": 224, "intermediate_size": 4304,
                      "layer_norm_eps": 1e-6, "num_attention_heads": 16,
                      "num_hidden_layers": 27, "patch_size": 14},
    "text_config": {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False,
                    "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
                    "intermediate_size": 6144, "num_attention_heads": 16,
                    "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-6,
                    "rope_scaling": None, "rope_theta": 1000000, "vocab_size": 151680},
}


# ── routing and contracts ────────────────────────────────────────────────────

def test_pipeline_map_pins_the_sm89_frontend():
    assert _PIPELINE_MAP[KEY] == FRONTEND
    assert [k for k in _PIPELINE_MAP if k[0] == "groot_n15"] == [KEY]
    assert resolve_pipeline_class(*KEY).__name__ == FRONTEND[1]


def test_model_package_imports_without_kernels_or_gr00t():
    import flash_rt.models.groot_n15 as m
    from flash_rt.models.groot_n15.pipeline_rtx import GROOT_N15_RTX_SM89 as t
    assert m.GrootN15Dims().dit_tokens == 1 + 32 + 16
    assert t.compute_capability == (8, 9) and not t.pdl
    assert t.precisions == ("bf16", "fp8")
    assert t.fp8_projections == {"vision", "vl", "cross_kv", "llm_attention"}
    assert "FLASHRT_ENABLE_GROOT_N15=ON" in t.build_hint


def test_parse_dims_reads_the_checkpoint_geometry():
    from flash_rt.frontends.torch._groot_n15_weights import parse_dims
    d = parse_dims(CONFIG, EAGLE, action_horizon=16)
    assert (d.vision_layers, d.vision_heads, d.vision_head_dim) == (27, 16, 72)
    assert (d.llm_layers, d.llm_q_heads, d.llm_kv_heads, d.llm_head_dim) == (12, 16, 8, 128)
    assert (d.vl_hidden, d.dit_hidden, d.dit_layers, d.denoising_steps) == (2048, 1536, 16, 4)
    assert d.patches_per_tile == 256 and d.patch_features == 588
    assert [d.step_bucket(s) for s in range(4)] == [0, 250, 500, 750]


@pytest.mark.parametrize("path,value,match", [
    (("model_type",), "gr00t_n1_6", "model_type"),
    (("backbone_cfg", "project_to_dim"), 1536, "project_to_dim"),
    (("action_head_cfg", "diffusion_model_cfg", "interleave_self_attention"), False,
     "interleave_self_attention"),
    (("action_head_cfg", "use_vlln"), False, "use_vlln"),
    (("eagle", "use_pixel_shuffle"), True, "use_pixel_shuffle"),
    (("eagle", "mlp_connector_layers"), 2, "mlp_connector_layers"),
    (("eagle", "text_config", "rope_scaling"), {"type": "linear"}, "rope_scaling"),
])
def test_parse_dims_refuses_what_the_forward_does_not_implement(path, value, match):
    from flash_rt.frontends.torch._groot_n15_weights import parse_dims
    cfg, eagle = copy.deepcopy(CONFIG), copy.deepcopy(EAGLE)
    node = eagle if path[0] == "eagle" else cfg
    keys = path[1:] if path[0] == "eagle" else path
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    with pytest.raises(ValueError, match=match):
        parse_dims(cfg, eagle, action_horizon=16)


def test_require_kernels_names_what_is_missing():
    from flash_rt.models.groot_n15 import GROOT_N15_SYMBOLS, KERNEL_SYMBOLS, require_kernels
    from flash_rt.models.groot_n15.pipeline_rtx import GROOT_N15_RTX_SM89 as t
    fvk = types.SimpleNamespace(**{n: object() for n in KERNEL_SYMBOLS + t.linear_symbols("fp8")})
    g = types.SimpleNamespace(**{n: object() for n in GROOT_N15_SYMBOLS})
    require_kernels(fvk, g, t, "fp8")
    del g.groot_n15_rescale_rows_bf16
    with pytest.raises(RuntimeError, match="groot_n15_rescale_rows_bf16.*FLASHRT_ENABLE_GROOT_N15"):
        require_kernels(fvk, g, t, "fp8")
    fvk2 = types.SimpleNamespace(**{n: object() for n in KERNEL_SYMBOLS})
    g2 = types.SimpleNamespace(**{n: object() for n in GROOT_N15_SYMBOLS})
    with pytest.raises(RuntimeError, match="GemmRunner"):
        require_kernels(fvk2, g2, t, "fp8")


def test_attention_sites_at_head_dim_64_and_below_run_unsplit():
    from flash_rt.models.groot_n15._attention import _max_splits
    assert _max_splits(48, 1024) == 1 and _max_splits(64, 1024) == 1
    assert _max_splits(72, 256) == 2 and _max_splits(128, 1024) == 8


def test_lowerings_refuse_mismatched_weights_and_stale_activations():
    from flash_rt.models.groot_n15._linear import Bf16Linear, Fp8Linear, LinearWeights
    g = types.SimpleNamespace(groot_n15_linear_bias_bf16=lambda *a: None,
                              groot_n15_quant_rows_fp8=lambda *a: None,
                              groot_n15_rescale_rows_bf16=lambda *a: None)
    fvk = types.SimpleNamespace(bf16_matmul_cublaslt_ref_bf16=lambda *a: None,
                                GemmRunner=lambda: types.SimpleNamespace(
                                    fp8_nt_dev=lambda *a: None,
                                    autotune_fp8_nt_dev=lambda *a: None))
    packed = LinearWeights(0, 1, 8, 8, w8=2, scale=3)
    with pytest.raises(ValueError, match="packed to FP8"):
        Bf16Linear(g, fvk)(Bf16Linear(g, fvk).input(1, 4, 8, 0), packed, 4, 4, 0)
    lin = Fp8Linear(g, fvk, x8=10, x8_bytes=64, row_scale=11, rows=8, one=12)
    old = lin.input(1, 4, 8, 0)
    lin.input(2, 4, 8, 0)
    with pytest.raises(RuntimeError, match="older than the last input"):
        lin.matmul(old, packed, 4, 4, 0)
    with pytest.raises(ValueError, match="exceeds the quantization scratch"):
        lin(lin.input(3, 16, 8, 0), packed, 4, 16, 0)


# ── IO and observation assembly ──────────────────────────────────────────────

def test_io_refuses_a_missing_data_config_or_source_root(monkeypatch, tmp_path):
    from flash_rt.frontends.torch._groot_n15_io import GrootN15IO
    monkeypatch.delenv("GROOT_N15_DATA_CONFIG", raising=False)
    with pytest.raises(ValueError, match="GROOT_N15_DATA_CONFIG"):
        GrootN15IO(str(tmp_path), None, "new_embodiment", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no gr00t package"):
        GrootN15IO(str(tmp_path), "panda_omron", "new_embodiment", str(tmp_path))


def _frontend_without_model():
    module = __import__(FRONTEND[0], fromlist=[FRONTEND[1]])
    fe = object.__new__(getattr(module, FRONTEND[1]))
    fe.io = types.SimpleNamespace(
        video_keys=("video.left", "video.wrist"),
        state_keys=("state.eef", "state.gripper"),
        language_key="annotation.task")
    fe._prompt = "pick the cup"
    return fe


def test_observation_from_image_list_and_state_dict():
    fe = _frontend_without_model()
    img = np.zeros((256, 256, 3), np.uint8)
    obs = fe._observation({"images": [img, img + 1],
                           "state": {"eef": [1, 2, 3], "state.gripper": [4.0]}})
    assert obs["video.left"].shape == (1, 256, 256, 3) and obs["video.wrist"][0, 0, 0, 0] == 1
    assert obs["state.eef"].shape == (1, 3) and obs["state.gripper"].dtype == np.float64
    assert obs["annotation.task"] == ["pick the cup"]


def test_observation_accepts_native_keys_and_reports_missing_inputs():
    fe = _frontend_without_model()
    img = np.zeros((64, 64, 3), np.uint8)
    obs = fe._observation({"video.left": img, "video.wrist": img,
                           "state.eef": [0, 0, 0], "state.gripper": [0]})
    assert set(obs) == {"video.left", "video.wrist", "state.eef", "state.gripper",
                        "annotation.task"}
    with pytest.raises(ValueError, match="missing camera frame"):
        fe._observation({"images": [img]})
    with pytest.raises(ValueError, match="missing state"):
        fe._observation({"images": [img, img], "state": {"eef": [0, 0, 0]}})
    with pytest.raises(ValueError, match="uint8"):
        fe._observation({"images": [img.astype(np.float32), img], "state": {}})


def test_infer_requires_a_prompt():
    fe = _frontend_without_model()
    fe._prompt = None
    with pytest.raises(RuntimeError, match="set_prompt"):
        fe.infer({})


# ── load_model routing ───────────────────────────────────────────────────────

def _needs_kernels():
    from flash_rt import _extensions
    if not _extensions.present("flash_rt_kernels"):
        pytest.skip("flash_rt_kernels not built")


def test_load_model_refuses_other_hardware_and_the_fp16_switch(tmp_path):
    _needs_kernels()
    import flash_rt
    with pytest.raises(RuntimeError, match="groot_n15"):
        flash_rt.load_model(str(tmp_path), config="groot_n15", hardware="rtx_sm120",
                            num_views=3)
    with pytest.raises(ValueError, match="use_fp16"):
        flash_rt.load_model(str(tmp_path), config="groot_n15", hardware="rtx_sm89",
                            num_views=3, use_fp16=True, use_fp8=False)


# ── RTX 4090 parity against the unmodified upstream policy ───────────────────

_E2E = ("GROOT_N15_CHECKPOINT", "GROOT_N15_ROOT", "GROOT_N15_DATA_CONFIG")
needs_e2e = pytest.mark.skipif(
    not all(os.environ.get(k) for k in _E2E),
    reason="needs GROOT_N15_CHECKPOINT / GROOT_N15_ROOT / GROOT_N15_DATA_CONFIG and an RTX 4090")


def _torch_sm89():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("needs an SM89 GPU")
    return torch


@pytest.fixture(scope="module")
def upstream():
    torch = _torch_sm89()
    import sys
    pytest.importorskip("flash_attn", reason="the upstream Eagle backbone needs flash_attn")
    sys.path.insert(0, os.environ["GROOT_N15_ROOT"])
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    cfg = DATA_CONFIG_MAP[os.environ["GROOT_N15_DATA_CONFIG"]]
    policy = Gr00tPolicy(model_path=os.environ["GROOT_N15_CHECKPOINT"],
                         embodiment_tag="new_embodiment", modality_config=cfg.modality_config(),
                         modality_transform=cfg.transform(), denoising_steps=4)
    yield policy, cfg
    del policy
    torch.cuda.empty_cache()


def _observation(policy, seed: int):
    """A deterministic observation: smooth images, states inside the checkpoint's range."""
    rng = np.random.default_rng(seed)
    meta = json.load(open(os.path.join(os.environ["GROOT_N15_CHECKPOINT"], "experiment_cfg",
                                       "metadata.json")))["new_embodiment"]
    obs = {}
    for key in policy.modality_config["video"].modality_keys:
        h, w = meta["modalities"]["video"][key.split(".", 1)[1]]["resolution"]
        low = rng.integers(0, 256, size=(8, 8, 3)).astype(np.float32)
        img = np.kron(low, np.ones((h // 8 + 1, w // 8 + 1, 1)))[:h, :w]
        obs[key] = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)[None]
    for key in policy.modality_config["state"].modality_keys:
        st = meta["statistics"]["state"][key.split(".", 1)[1]]
        lo, hi = np.asarray(st["q01"]), np.asarray(st["q99"])
        obs[key] = (lo + (hi - lo) * rng.random(lo.shape))[None].astype(np.float64)
    return obs


def _reference(policy, obs, prompt):
    """Upstream normalized inputs, the noise it drew, and its normalized actions."""
    torch = pytest.importorskip("torch")
    seen = {}
    lang = policy.modality_config["language"].modality_keys[0]
    obs = {**obs, lang: [prompt]}
    head = policy.model.action_head

    def grab_noise(mod, args):
        seen.setdefault("noise", args[0].detach().clone())

    handle = head.action_encoder.register_forward_pre_hook(grab_noise)
    orig = policy._get_action_from_normalized_input

    def grab(normalized):
        seen["inputs"] = normalized
        seen["actions"] = orig(normalized)
        return seen["actions"]

    policy._get_action_from_normalized_input = grab
    try:
        out = policy.get_action(obs)
    finally:
        policy._get_action_from_normalized_input = orig
        handle.remove()
    return seen["inputs"], seen["noise"], seen["actions"].float(), out, obs


def _frontend(use_fp8, **kw):
    cls = resolve_pipeline_class(*KEY)
    return cls(os.environ["GROOT_N15_CHECKPOINT"], num_views=3,
               data_config=os.environ["GROOT_N15_DATA_CONFIG"],
               groot_root=os.environ["GROOT_N15_ROOT"], use_fp8=use_fp8, **kw)


def _binary_keys(cfg):
    return [k for k, m in getattr(cfg, "action_normalization_modes", {}).items() if m == "binary"]


def _real_action_dims(policy):
    meta = json.load(open(os.path.join(os.environ["GROOT_N15_CHECKPOINT"], "experiment_cfg",
                                       "metadata.json")))["new_embodiment"]
    return sum(meta["modalities"]["action"][k.split(".", 1)[1]]["shape"][0]
               for k in policy.modality_config["action"].modality_keys)


@needs_e2e
@pytest.mark.parametrize("use_fp8,min_cos,max_err", [(False, 0.9999, 0.1), (True, 0.998, 0.4)])
def test_actions_match_the_upstream_policy(upstream, use_fp8, min_cos, max_err):
    torch = _torch_sm89()
    policy, cfg = upstream
    real = _real_action_dims(policy)
    fe = _frontend(use_fp8)
    try:
        for seed, prompt in ((0, "open the left drawer"),
                             (1, "pick the apple from the counter and place it in the sink")):
            inputs, noise, ref, ref_dict, _ = _reference(policy, _observation(policy, seed), prompt)
            out = fe.predict_normalized(inputs, noise).cpu()
            ref = ref.cpu()
            cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
            assert cos.item() >= min_cos, f"seed {seed}: cos {cos.item():.6f}"
            err = (out[..., :real] - ref[..., :real]).abs().max().item()
            assert err <= max_err, f"seed {seed}: real-dim max error {err:.3e}"
            ours = fe.io.unnormalize(out)
            for key in _binary_keys(cfg):
                assert np.array_equal(ours[key], ref_dict[key]), key
    finally:
        del fe
        torch.cuda.empty_cache()


@needs_e2e
@pytest.mark.parametrize("use_fp8", [False, True])
def test_graph_replay_reuse_and_recapture(upstream, use_fp8):
    torch = _torch_sm89()
    policy, _ = upstream
    prompt_a, prompt_b = "open the left drawer", "press the start button on the microwave"
    runs = {}
    for name, seed, prompt in (("A", 2, prompt_a), ("B", 3, prompt_a), ("C", 4, prompt_b)):
        inputs, noise, _, _, _ = _reference(policy, _observation(policy, seed), prompt)
        runs[name] = (inputs, noise)
    s_a = int(runs["A"][0]["eagle_input_ids"].shape[1])
    s_c = int(runs["C"][0]["eagle_input_ids"].shape[1])
    assert int(runs["B"][0]["eagle_input_ids"].shape[1]) == s_a and s_c != s_a
    fe = _frontend(use_fp8, graph_cache_size=1)
    try:
        fe.cuda_graphs = False
        eager = {k: fe.predict_normalized(*v) for k, v in runs.items()}
        fe.cuda_graphs = True
        # A -> B -> A through one graph: the staged inputs change, the graph does not.
        for name in ("A", "B", "A"):
            assert torch.equal(fe.predict_normalized(*runs[name]), eager[name]), name
        assert fe.graph_lengths == (s_a,)
        # A second length evicts the first; coming back recaptures it.
        assert torch.equal(fe.predict_normalized(*runs["C"]), eager["C"])
        assert fe.graph_lengths == (s_c,)
        assert torch.equal(fe.predict_normalized(*runs["A"]), eager["A"])
        assert fe.graph_lengths == (s_a,)
    finally:
        del fe
        torch.cuda.empty_cache()


@needs_e2e
def test_infer_returns_the_per_key_action_dict(upstream):
    torch = _torch_sm89()
    policy, _ = upstream
    fe = _frontend(True)
    try:
        obs = _observation(policy, 5)
        fe.set_prompt("open the left drawer")
        out = fe.infer({k: v[0] for k, v in obs.items()})["actions"]
        assert set(out) == set(policy.modality_config["action"].modality_keys)
        assert all(np.isfinite(v).all() and v.shape[0] == 16 for v in out.values())
    finally:
        del fe
        torch.cuda.empty_cache()


@needs_e2e
def test_frontend_refuses_another_device_generation(monkeypatch):
    torch = _torch_sm89()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    with pytest.raises(RuntimeError, match="validated on SM89"):
        _frontend(False)


@needs_e2e
def test_staging_refuses_inputs_outside_the_contract(upstream):
    torch = _torch_sm89()
    policy, _ = upstream
    inputs, noise, _, _, _ = _reference(policy, _observation(policy, 6), "open the left drawer")
    fe = _frontend(False, max_prompt_tokens=900)
    try:
        def stage(**changes):
            return fe.predict_normalized({**inputs, **changes}, noise)

        ids = inputs["eagle_input_ids"]
        image = ids[0] == fe.io.image_token_id()
        fewer = ids.clone()
        fewer[0, int(image.nonzero()[0])] = ids[0, -1]
        with pytest.raises(ValueError, match="image tokens"):
            stage(eagle_input_ids=fewer)
        longer = torch.cat([ids, ids[:, -1:].expand(1, 200)], dim=1)
        with pytest.raises(ValueError, match="max_prompt_tokens"):
            stage(eagle_input_ids=longer)
        mask = torch.ones_like(ids)
        mask[0, 0] = 0
        with pytest.raises(ValueError, match="padded"):
            stage(eagle_attention_mask=mask)
        with pytest.raises(ValueError, match="embodiment"):
            stage(embodiment_id=torch.tensor([0]))
        with pytest.raises(ValueError, match="eagle_pixel_values"):
            stage(eagle_pixel_values=inputs["eagle_pixel_values"][:2])
        with pytest.raises(ValueError, match="noise"):
            fe.predict_normalized(inputs, noise[..., :8])
        with pytest.raises(ValueError, match="action_horizon"):
            _frontend(False, action_horizon=8)
    finally:
        del fe
        torch.cuda.empty_cache()
