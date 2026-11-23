"""CPU checks for the Inkling 2 Mini RTX path; no GPU, checkpoint or network required."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from flash_rt.frontends.torch import _inkling2_rtx_runtime as rtm  # noqa: E402
from flash_rt.frontends.torch._inkling2_rtx_mtp import MTP_GEOMETRY, check_mtp_config  # noqa: E402


def _config():
    text = dict(rtm.GEOMETRY, model_type="inkling2_loop_text", loop_config=dict(rtm.LOOP),
                local_layer_ids=[l for l in range(24) if l % 6 != 5], unpadded_vocab_size=200058)
    mtp = dict(MTP_GEOMETRY, rope_theta=1.0e6, target_layer_ids=[5, 23, 41, 48])
    return {"model_type": "inkling2_loop", "text_config": text, "mtp_config": mtp, "eos_token_id": 200006}


def test_routing_entry():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("inkling2", "torch", "rtx_sm120")] == (
        "flash_rt.frontends.torch.inkling2_rtx", "Inkling2TorchFrontendRtx")


def test_released_geometry_is_accepted():
    raw = _config()
    rtm.check_config(raw)
    assert check_mtp_config(raw, visits=48)["target_layer_ids"] == [5, 23, 41, 48]


@pytest.mark.parametrize("key,value", [("hidden_size", 4096), ("n_routed_experts", 64),
                                       ("sliding_window_size", 4096), ("n_shared_experts", 1)])
def test_other_geometry_is_rejected(key, value):
    raw = _config()
    raw["text_config"][key] = value
    with pytest.raises(rtm.Inkling2ConfigError, match=key):
        rtm.check_config(raw)


def test_loop_and_layer_layout_are_checked():
    raw = _config()
    raw["text_config"]["loop_config"]["num_passes"] = 2
    with pytest.raises(rtm.Inkling2ConfigError, match="num_passes"):
        rtm.check_config(raw)
    raw = _config()
    raw["text_config"]["local_layer_ids"] = list(range(24))
    with pytest.raises(rtm.Inkling2ConfigError, match="global layers"):
        rtm.check_config(raw)


def test_drafter_config_is_checked():
    raw = _config()
    raw["mtp_config"]["block_size"] = 16
    with pytest.raises(rtm.Inkling2ConfigError, match="block_size"):
        check_mtp_config(raw, visits=48)
    raw = _config()
    raw["mtp_config"]["target_layer_ids"] = [5, 23, 41, 47]
    with pytest.raises(rtm.Inkling2ConfigError, match="target_layer_ids"):
        check_mtp_config(raw, visits=48)
    with pytest.raises(rtm.Inkling2ConfigError, match="mtp_config"):
        check_mtp_config({"text_config": {}}, visits=48)


def test_read_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    raw, text = rtm.read_config(tmp_path)
    assert text["hidden_size"] == 2304 and raw["eos_token_id"] == 200006
    with pytest.raises(FileNotFoundError):
        rtm.read_config(tmp_path / "missing")


def test_checkpoint_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        rtm.checkpoint_shards(tmp_path)
    for name in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors", "mtp.safetensors"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in rtm.checkpoint_shards(tmp_path)] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00002-of-00002.safetensors"}}))
    assert [p.name for p in rtm.checkpoint_shards(tmp_path)] == ["model-00002-of-00002.safetensors"]


@pytest.mark.parametrize("n", [1, 7, 8, 15, 16, 40, 511, 512, 513, 700, 1100, 4000])
@pytest.mark.parametrize("exact", [False, True])
def test_plan_chunks_covers_the_prompt(n, exact):
    sizes = (512, 256, 128, 64, 32, 16)
    plan = rtm.plan_chunks(n, sizes, 8, exact)
    assert sum(r for _, r in plan) == n
    kinds = [k for k, _ in plan]
    assert kinds == sorted(kinds, key=("chunk", "exact", "step").index)       # chunks, then exact, then steps
    assert kinds.count("step") < 8
    if exact:
        assert "chunk" not in kinds
    else:
        chunk = [r for k, r in plan if k == "chunk"]
        assert chunk == sorted(chunk, reverse=True) and all(r in sizes for r in chunk)
        assert kinds.count("exact") <= 1


def test_plan_chunks_rejects_an_empty_prompt():
    with pytest.raises(ValueError):
        rtm.plan_chunks(0, (512,), 8, False)


def test_visit_count():
    assert rtm.visit_count(_config()["text_config"]) == 48


def test_chat_template_tojson_matches_transformers():
    """tojson takes transformers' arguments and defaults (ensure_ascii=False): tool declarations render with sorted
    keys and compact separators as the checkpoint's template asks."""
    pytest.importorskip("jinja2")
    from flash_rt.frontends.torch.inkling2_rtx import compile_chat_template
    t = compile_chat_template('{{ x | tojson }}|{{ x | tojson(sort_keys=true, separators=(",", ":")) }}|'
                              '{{ x | tojson(indent=1) }}')
    x = {"b": 1, "a": "é"}
    assert t.render(x=x) == '{"b": 1, "a": "é"}|{"a":"é","b":1}|{\n "b": 1,\n "a": "é"\n}'
    with pytest.raises(ValueError, match="stop"):
        compile_chat_template("{{ raise_exception('stop') }}").render()
    assert compile_chat_template("{% generation %}A{% endgeneration %}B").render() == "AB"


def test_missing_extension_names_the_build_flag(monkeypatch):
    monkeypatch.setattr(rtm, "_ik", None)
    with pytest.raises(RuntimeError, match="FLASHRT_ENABLE_INKLING2"):
        rtm.Inkling2RtxRuntime("/nonexistent")
