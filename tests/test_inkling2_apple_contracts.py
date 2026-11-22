"""CPU checks for the Inkling 2 Mini Apple path; no checkpoint, GPU or network required."""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from flash_rt.models.inkling2.checkpoint import (
    Inkling2ConfigError, check_constants, check_geometry, check_mtp_config, config_from_json)


def _config():
    """config.json fields of the released checkpoint that the reader consults."""
    local = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
    return {
        "model_type": "inkling2_loop", "eos_token_id": 200006,
        "text_config": {
            "num_attention_heads": 24, "swa_num_attention_heads": 24, "num_key_value_heads": 8,
            "swa_num_key_value_heads": 8, "head_dim": 128, "swa_head_dim": 128, "n_shared_experts": 0,
            "q_bias": False, "o_bias": False, "num_hidden_layers": 24, "hidden_size": 2304, "vocab_size": 201024,
            "unpadded_vocab_size": 200058, "d_rel": 16, "rel_extent": 1024, "sliding_window_size": 1024,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6, "n_routed_experts": 128,
            "num_experts_per_tok": 8, "intermediate_size": 1152, "local_layer_ids": local,
            "model_max_length": 1048576, "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0, "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0, "final_logit_softcapping": 30.0,
            "use_gate_bias": True, "use_value_norm_no_gamma": True, "attn_res_normalize_values": True,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False},
        },
        "mtp_config": {
            "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 9216, "max_position_embeddings": 1048576,
            "torch_dtype": "bfloat16", "q_bias": False, "o_bias": False, "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0, "local_layer_ids": [0, 2], "sliding_window_size": 1024, "block_size": 8,
            "markov_rank": 256, "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48],
        },
    }


def test_import_needs_no_framework():
    code = '''
import json, sys
import flash_rt.models.inkling2.checkpoint
import flash_rt.frontends.torch._inkling2_mps_runtime
import flash_rt.frontends.torch._inkling2_mtp
print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in
    {'torch', 'mlx', 'mlx_lm', 'tokenizers', 'jinja2'})))
'''
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_routing_entry():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("inkling2", "torch", "apple_mps")] == (
        "flash_rt.frontends.torch.inkling2_mps", "Inkling2TorchFrontendMps")


def test_load_model_names_the_frontend():
    from flash_rt.api import load_model
    with pytest.raises(NotImplementedError, match="Inkling2TorchFrontendMps"):
        load_model("/nonexistent", config="inkling2")


def test_config_geometry_and_visits():
    raw = _config()
    cfg = config_from_json(raw, {"eos_token_id": 200006, "temperature": 1.0})
    check_geometry(cfg)
    check_constants(raw)
    check_mtp_config(raw)
    visits = cfg.visits()
    assert len(visits) == 48 and cfg.eos_token_ids == (200006,)
    assert visits[18] == (6, 1) and visits[42] == (18, 0)
    # drafter features: the hidden state entering visits 5, 23, 41 (rows capture it after visits 4, 22, 40)
    assert [visits[v] for v in (5, 23, 41)] == [(5, 0), (11, 1), (17, 2)]
    assert [l for l in range(24) if cfg.is_global(l)] == [5, 11, 17, 23]


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(model_type="inkling2"), "model_type"),
    (lambda r: r["text_config"].update(swa_head_dim=64), "head_dim"),
    (lambda r: r["text_config"].update(n_shared_experts=1), "shared expert"),
    (lambda r: r["text_config"]["loop_config"].update(is_varloop=True), "loop count"),
    (lambda r: r["text_config"].pop("d_rel"), "d_rel"),
])
def test_config_refusals(edit, message):
    raw = _config()
    edit(raw)
    with pytest.raises(Inkling2ConfigError, match=message):
        config_from_json(raw)


@pytest.mark.parametrize("key,value", [("num_experts_per_tok", 6), ("intermediate_size", 2048),
                                       ("sliding_window_size", 4096), ("num_hidden_layers", 26)])
def test_other_geometries_are_refused(key, value):
    raw = _config()
    raw["text_config"][key] = value
    with pytest.raises(Inkling2ConfigError):
        check_geometry(config_from_json(raw))


@pytest.mark.parametrize("key,value", [("final_logit_softcapping", 50.0), ("route_scale", 2.5),
                                       ("use_gate_bias", False)])
def test_other_constants_are_refused(key, value):
    raw = _config()
    raw["text_config"][key] = value
    with pytest.raises(Inkling2ConfigError, match=key):
        check_constants(raw)


@pytest.mark.parametrize("key,value", [("block_size", 16), ("markov_rank", 128), ("rope_theta", 10000.0),
                                       ("target_layer_ids", [5, 23, 41]), ("local_layer_ids", [0])])
def test_other_drafters_are_refused(key, value):
    raw = _config()
    raw["mtp_config"][key] = value
    with pytest.raises(Inkling2ConfigError, match=key):
        check_mtp_config(raw)
    raw = _config()
    del raw["mtp_config"]
    with pytest.raises(Inkling2ConfigError, match="mtp_config"):
        check_mtp_config(raw)


def test_nvfp4_global_scale_is_mlx_lm_fp32():
    from flash_rt.frontends.torch._inkling2_mps_runtime import global_scale
    amax = np.array([1.5, 3.25, 1e-3], np.float32)
    den = np.array([3.0], np.float32)
    want = (amax * (np.float32(2688.0) / den)).astype(np.float32)
    assert np.array_equal(global_scale(amax, den), want)


def test_bf16_rounding_helpers():
    from flash_rt.frontends.torch._inkling2_mtp import _bf16_round, _f32_to_bf16
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 0.0], np.float32)
    # ties to even: 1 + 2^-8 -> 1.0, 1 + 3 * 2^-8 -> 1 + 2^-6
    assert _f32_to_bf16(x).tolist() == [0x3F80, 0x3F80, 0x3F82, 0xC020, 0x0000]
    assert np.array_equal(_bf16_round(_bf16_round(x)), _bf16_round(x))


def test_chunk_plan():
    from flash_rt.frontends.torch._inkling2_mps_runtime import chunk_plan
    sizes = (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)
    assert chunk_plan(1031, sizes) == [1024, 4, 2, 1]
    assert chunk_plan(1, sizes) == [1]
    assert chunk_plan(3000, sizes) == [1024, 1024, 512, 256, 128, 32, 16, 8]
    assert sum(chunk_plan(777, sizes)) == 777
    with pytest.raises(ValueError):
        chunk_plan(5, (8, 4))


def test_mtp_width_by_batch():
    from flash_rt.frontends.torch._inkling2_mtp import _MTPMixin
    m = _MTPMixin()
    assert [m.mtp_width(c) for c in (1, 2, 4, 8, 16, 32, 64, 128)] == [8, 8, 8, 4, 4, 3, 3, 3]
    m.mtp_widths = {1 << 20: 5}
    assert m.mtp_width(17) == 5


def test_mtp_widths_fit_the_drafter_block():
    from flash_rt.frontends.torch._inkling2_mtp import MTP_BLOCK, _MTPMixin
    assert all(2 <= w <= MTP_BLOCK for w in _MTPMixin.mtp_widths.values())


def test_config_is_not_mutated():
    raw = _config()
    snapshot = copy.deepcopy(raw)
    config_from_json(raw)
    check_mtp_config(raw)
    assert raw == snapshot


@pytest.mark.parametrize("top_k,top_p,temperature", [(0, 1.0, 1.0), (33, 1.0, 1.0), (20, 0.0, 1.0),
                                                      (20, 1.5, 1.0), (20, 0.9, 0.0)])
def test_sampling_parameters_are_checked(top_k, top_p, temperature):
    from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime
    with pytest.raises(ValueError):
        Inkling2Runtime._check_sampling(top_k, top_p, temperature)
