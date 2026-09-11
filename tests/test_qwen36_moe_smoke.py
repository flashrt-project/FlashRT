"""Smoke tests for the Qwen3.6-35B-A3B text frontend."""

from __future__ import annotations

import inspect
import json
import os

import pytest


def _config():
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(40)
    ]
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "vocab_size": 248320,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "full_attention_interval": 4,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            "layer_types": layer_types,
        },
    }


def _checkpoint(tmp_path):
    from flash_rt.frontends.torch.qwen36_moe import (
        _MTP_KEYS,
        _required_text_keys,
    )

    config = _config()
    layer_types = tuple(config["text_config"]["layer_types"])
    keys = _required_text_keys(layer_types) | _MTP_KEYS
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / shard).write_bytes(b"checkpoint")
    (tmp_path / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in keys}}),
        encoding="utf-8",
    )
    return tmp_path


def _mock_checkpoint_shapes(monkeypatch, overrides=None):
    from flash_rt.frontends.torch import qwen36_moe

    shapes = qwen36_moe._expected_text_shapes(
        qwen36_moe._EXPECTED_LAYER_TYPES)
    shapes.update(qwen36_moe._expected_mtp_shapes())
    shapes.update(overrides or {})
    monkeypatch.setattr(
        qwen36_moe,
        "_read_tensor_shapes",
        lambda checkpoint_path, weight_map, tensor_names: {
            name: shapes[name] for name in tensor_names
        },
    )


def test_frontend_is_a_thin_qwen_entry():
    from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx
    from flash_rt.frontends.torch.qwen36_moe import (
        Qwen36MoeTextFrontend,
    )

    assert issubclass(Qwen36MoeTextFrontend, Nexn2TorchFrontendRtx)
    assert Qwen36MoeTextFrontend._MODEL_LABEL == (
        "Qwen3.6-35B-A3B text")
    assert inspect.signature(
        Qwen36MoeTextFrontend).parameters["kernelized"].default is True


@pytest.mark.parametrize("arch", ["rtx_sm120", "thor"])
def test_registry_resolves_qwen36_moe(arch):
    from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

    assert _PIPELINE_MAP[("qwen36_moe", "torch", arch)] == (
        "flash_rt.frontends.torch.qwen36_moe",
        "Qwen36MoeTextFrontend",
    )
    cls = resolve_pipeline_class("qwen36_moe", "torch", arch)
    assert cls.__name__ == "Qwen36MoeTextFrontend"


def test_both_architectures_resolve_to_one_frontend():
    """Thor and RTX run the same code, so they must resolve to one class.

    Two entries pointing at two classes would be two paths to keep in step;
    what differs between the targets is which kernel tiers the build has, not
    which Python runs.
    """
    from flash_rt.hardware import resolve_pipeline_class

    assert (resolve_pipeline_class("qwen36_moe", "torch", "thor")
            is resolve_pipeline_class("qwen36_moe", "torch", "rtx_sm120"))


def test_previous_import_path_still_works():
    """The module was renamed when it stopped being RTX-only."""
    from flash_rt.frontends.torch.qwen36_moe import Qwen36MoeTextFrontend
    from flash_rt.frontends.torch.qwen36_moe_rtx import (
        Qwen36MoeTextFrontendRtx,
    )

    assert Qwen36MoeTextFrontendRtx is Qwen36MoeTextFrontend


def test_load_model_redirects_to_text_frontend():
    import flash_rt

    with pytest.raises(NotImplementedError) as exc:
        flash_rt.load_model("/nonexistent", config="qwen36_moe")
    message = str(exc.value)
    assert "Qwen36MoeTextFrontend" in message
    assert "text LLM" in message


def test_constructor_rejects_quant_before_checkpoint_access():
    from flash_rt.frontends.torch.qwen36_moe import (
        Qwen36MoeTextFrontend,
    )

    with pytest.raises(NotImplementedError, match="only 'nvfp4'"):
        Qwen36MoeTextFrontend("/nonexistent", quant="fp8")


def test_constructor_rejects_reference_path_before_checkpoint_access():
    from flash_rt.frontends.torch.qwen36_moe import (
        Qwen36MoeTextFrontend,
    )

    with pytest.raises(NotImplementedError, match="kernelized=True"):
        Qwen36MoeTextFrontend("/nonexistent", kernelized=False)


def test_checkpoint_contract_accepts_complete_layout(tmp_path, monkeypatch):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    _mock_checkpoint_shapes(monkeypatch)
    result = validate_qwen36_moe_checkpoint(str(_checkpoint(tmp_path)))
    assert result["text_tensor_count"] == 693
    assert result["mtp_tensor_count"] == 19
    assert result["tensor_count"] == 712
    assert result["shard_count"] == 1


@pytest.mark.parametrize(
    ("path", "invalid", "message"),
    [
        (("moe_intermediate_size",), 1024, "moe_intermediate_size=1024"),
        (
            ("shared_expert_intermediate_size",),
            1024,
            "shared_expert_intermediate_size=1024",
        ),
        (("linear_conv_kernel_dim",), 3, "linear_conv_kernel_dim=3"),
        (("partial_rotary_factor",), 0.5, "partial_rotary_factor=0.5"),
        (("rms_norm_eps",), 1e-5, "rms_norm_eps=1e-05"),
        (("attn_output_gate",), False, "attn_output_gate=False"),
        (
            ("rope_parameters", "rope_theta"),
            10000,
            "rope_parameters.rope_theta=10000",
        ),
    ],
)
def test_checkpoint_contract_rejects_wrong_geometry(
        tmp_path, path, invalid, message):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    checkpoint = _checkpoint(tmp_path)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["text_config"]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = invalid
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_qwen36_moe_checkpoint(str(checkpoint))


def test_checkpoint_contract_rejects_missing_text_tensor(tmp_path):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    checkpoint = _checkpoint(tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["lm_head.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="lm_head.weight"):
        validate_qwen36_moe_checkpoint(str(checkpoint))


def test_checkpoint_contract_rejects_partial_mtp(tmp_path):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    checkpoint = _checkpoint(tmp_path)
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["mtp.fc.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="MTP tensor group"):
        validate_qwen36_moe_checkpoint(str(checkpoint))


def test_checkpoint_contract_rejects_missing_shard(tmp_path):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "model-00001-of-00001.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="missing or empty shards"):
        validate_qwen36_moe_checkpoint(str(checkpoint))


def test_checkpoint_contract_rejects_wrong_tensor_shape(
        tmp_path, monkeypatch):
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    _mock_checkpoint_shapes(monkeypatch, {name: (8192, 1, 3)})

    with pytest.raises(ValueError, match=r"conv1d\.weight=.*8192, 1, 3"):
        validate_qwen36_moe_checkpoint(str(_checkpoint(tmp_path)))


def test_generic_env_names_precede_legacy_aliases(monkeypatch):
    from flash_rt.frontends.torch._nexn2_rtx_decode import _qwen35moe_env

    monkeypatch.setenv("FLASHRT_NEXN2_PREFILL_CHUNK", "4096")
    assert _qwen35moe_env("PREFILL_CHUNK", "8192") == "4096"
    monkeypatch.setenv("FLASHRT_QWEN35MOE_PREFILL_CHUNK", "2048")
    assert _qwen35moe_env("PREFILL_CHUNK", "8192") == "2048"


def test_kernelized_generate_uses_shared_graph_path(monkeypatch):
    from flash_rt.frontends.torch import _nexn2_rtx_decode as decode
    from flash_rt.frontends.torch.qwen36_moe import (
        Qwen36MoeTextFrontend,
    )

    calls = {}

    class FakeState:
        def __init__(self, weights, max_seq, device, *,
                     spec_graph_cache_max=None):
            calls["state"] = (weights, max_seq, device)
            calls["spec_cap"] = spec_graph_cache_max

    def fake_generate(state, prompt_ids, count, fvk, device):
        calls["generate"] = (state, prompt_ids, count, fvk, device)
        return [7] * count

    monkeypatch.setattr(decode, "Nexn2DecodeState", FakeState)
    monkeypatch.setattr(decode, "generate_greedy_graph", fake_generate)

    frontend = Qwen36MoeTextFrontend.__new__(
        Qwen36MoeTextFrontend)
    frontend._kernelized = True
    frontend._prompt_ids = object()
    frontend._decode_state = None
    frontend._weights = object()
    frontend._user_max_seq = 128
    frontend.device = "cuda:0"
    frontend._fvk = object()
    frontend._spec_graph_cache_max = None

    assert frontend.generate(3) == [7, 7, 7]
    assert calls["state"] == (
        frontend._weights, frontend._user_max_seq, frontend.device)
    assert calls["generate"][1:] == (
        frontend._prompt_ids, 3, frontend._fvk, frontend.device)
    with pytest.raises(NotImplementedError, match="greedy"):
        frontend.generate(1, do_sample=True)


@pytest.mark.skipif(
    not os.environ.get("FLASHRT_QWEN36_MOE_CKPT_DIR"),
    reason="set FLASHRT_QWEN36_MOE_CKPT_DIR for checkpoint validation",
)
def test_real_checkpoint_contract():
    from flash_rt.frontends.torch.qwen36_moe import (
        validate_qwen36_moe_checkpoint,
    )

    result = validate_qwen36_moe_checkpoint(
        os.environ["FLASHRT_QWEN36_MOE_CKPT_DIR"])
    assert result == {
        "checkpoint_path": os.path.abspath(
            os.environ["FLASHRT_QWEN36_MOE_CKPT_DIR"]),
        "text_tensor_count": 693,
        "mtp_tensor_count": 19,
        "vision_tensor_count": 333,
        "tensor_count": 1045,
        "shard_count": 26,
    }
