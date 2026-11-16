"""CPU checks for the Qwen3.6-35B-A3B Apple path; no checkpoint, GPU or network required."""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flash_rt.models.flint.mlx_checkpoint import f32_to_bf16
from flash_rt.models.qwen36_moe import pipeline_mps as P
from flash_rt.models.qwen36_moe.mlx_checkpoint import (
    Qwen35MoeConfigError, check_quantization, config_from_json)


def _config():
    """config.json fields of the MLX 4-bit export that the reader consults."""
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    for i in range(40):
        for name in ("gate", "shared_expert_gate"):
            q[f"language_model.model.layers.{i}.mlp.{name}"] = {"group_size": 64, "bits": 8}
    return {
        "model_type": "qwen3_5_moe", "eos_token_id": [248046, 248044], "quantization": q,
        "text_config": {
            "attention_bias": False, "attn_output_gate": True, "full_attention_interval": 4, "head_dim": 256,
            "hidden_act": "silu", "hidden_size": 2048,
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)],
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
            "linear_num_value_heads": 32, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
            "moe_intermediate_size": 512, "num_attention_heads": 16, "num_experts": 256, "num_experts_per_tok": 8,
            "num_hidden_layers": 40, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
            "shared_expert_intermediate_size": 512, "tie_word_embeddings": False, "vocab_size": 248320,
        },
    }


def test_import_needs_no_framework():
    code = '''
import json, sys
import flash_rt.models.qwen36_moe.pipeline_mps
import flash_rt.frontends.torch._qwen36_moe_mps_runtime
print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in
    {'torch', 'mlx', 'mlx_lm', 'tokenizers', 'jinja2'})))
'''
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_config_and_geometry():
    raw = _config()
    cfg = config_from_json(raw, {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "eos_token_id": [248046, 248044]})
    P.check_geometry(cfg)
    check_quantization(raw, cfg)
    assert cfg.full_layers == tuple(range(3, 40, 4)) and len(cfg.linear_layers) == 30
    assert (cfg.conv_dim, cfg.value_dim, cfg.q_dim, cfg.kv_dim, cfg.rotary_dims) == (8192, 4096, 4096, 512, 64)
    assert cfg.eos_token_ids == (248046, 248044) and (cfg.top_k, cfg.top_p) == (20, 0.95)


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(model_type="qwen3_5"), "model_type"),
    (lambda r: r["text_config"].update(num_experts_per_tok=6), "num_experts_per_tok"),
    (lambda r: r["text_config"].update(head_dim=128), "head_dim"),
    (lambda r: r["text_config"]["layer_types"].__setitem__(0, "full_attention"), "layer_types"),
    (lambda r: r["text_config"].update(attn_output_gate=False), "output gate"),
])
def test_other_geometries_are_refused(edit, message):
    raw = _config()
    edit(raw)
    with pytest.raises(Qwen35MoeConfigError, match=message):
        P.check_geometry(config_from_json(raw))


@pytest.mark.parametrize("edit", [
    lambda q: q.update(bits=8),
    lambda q: q.pop("language_model.model.layers.7.mlp.gate"),
    lambda q: q.update({"language_model.lm_head": {"group_size": 64, "bits": 8}}),
])
def test_other_quantizations_are_refused(edit):
    raw = _config()
    cfg = config_from_json(raw)
    edit(raw["quantization"])
    with pytest.raises(Qwen35MoeConfigError):
        check_quantization(raw, cfg)


def test_descriptor_slots_do_not_overlap():
    four = [P.D_M0, P.D_M1, P.D_M2, P.D_M3, P.D_OUT, P.D_ROUTER, P.D_SGATE, P.D_GATE, P.D_UP, P.D_DOWN,
            P.D_SH_GATE, P.D_SH_UP, P.D_SH_DOWN]
    used = [P.D_FLAGS, P.D_IN_NORM, P.D_POST_NORM, P.D_GDN_NORM, P.D_CONV, P.D_ALOG, P.D_DT, P.D_STATE, P.D_AUX]
    for base in four:
        used += list(range(base, base + 4))
    assert len(used) == len(set(used)) and max(used) < P.DESCRIPTOR
    assert (P.D_Q_NORM, P.D_K_NORM) == (P.D_GDN_NORM, P.D_CONV)   # attention layers reuse the DeltaNet slots


def _fake_runtime(logits=None, noise=None):
    from flash_rt.frontends.torch._qwen36_moe_mps_runtime import Qwen36MoeRuntime
    rt = Qwen36MoeRuntime.__new__(Qwen36MoeRuntime)
    rt.max_seq, rt.prefill_chunk = 16384, 2048
    rt.buf = {"tokens": 0, "noise": 1}
    rt.programs = {name: name for name in ("prefill", "prefill_small", "prefill_head", "noise")}
    rt._state = np.zeros(4, np.uint32)
    rt._params = np.zeros(16, np.uint32)
    runs, resets = [], []
    noise = np.zeros(64, np.float32) if noise is None else noise

    def run(program):
        if program == "noise":
            noise[1] = 5.0
        runs.append((program, tuple(int(x) for x in rt._state)))
        return {"gpu_us": 0.0}
    rt.plan = SimpleNamespace(write=lambda *a: None, run=run, read=lambda b, o, dt, k: noise[:k].copy())
    rt.reset = lambda: resets.append(1)
    if logits is None:
        rt.select = lambda *a, **k: 7
    else:
        rt.logits = lambda step=0: f32_to_bf16(logits)
    return rt, runs, resets, noise


@pytest.mark.parametrize("n,start,expect", [
    (5000, 0, [("prefill", 0, 2048), ("prefill", 2048, 2048), ("prefill", 4096, 904)]),
    (2560, 0, [("prefill", 0, 2048), ("prefill_small", 2048, 512)]),
    (2561, 0, [("prefill", 0, 2048), ("prefill", 2048, 513)]),
    (300, 0, [("prefill_small", 0, 300)]),
    (2100, 2000, [("prefill_small", 2000, 100)]),
])
def test_prefill_chunking(n, start, expect):
    rt, runs, resets, _ = _fake_runtime()
    assert rt.prefill(list(range(n)), start=start) == 7
    assert [(p, s[0], s[1]) for p, s in runs[:-1]] == expect
    assert runs[-1][0] == "prefill_head" and rt.pos == n
    assert all(s[3] == s[0] + s[1] for _, s in runs[:-1])
    assert len(resets) == (1 if start == 0 else 0)


@pytest.mark.parametrize("n", [300, 2048, 5000])
def test_prefill_samples_the_first_token(n):
    logits = np.full(64, -10.0, np.float32)
    logits[[0, 1]] = [3.0, 2.5]
    sampling = dict(top_k=20, top_p=1.0, temperature=1.0)
    rt, runs, _, _ = _fake_runtime(logits)
    assert rt.prefill(list(range(n)), program="sampled", seed=123, **sampling) == 1
    assert [p for p, _ in runs][-2:] == ["prefill_head", "noise"]
    assert (rt._params[1], rt._params[2], rt._params[8], rt._params[12], rt._params[14]) == (n - 1, 1, 20, 123, 1)
    rt, runs, _, _ = _fake_runtime(logits)
    assert rt.prefill(list(range(n)), seed=123, **sampling) == 0          # greedy ignores the noise
    assert "noise" not in [p for p, _ in runs]


def test_first_token_selection_rule():
    logits = np.full(64, -10.0, np.float32)
    logits[[5, 9, 12]] = [3.0, 3.0, 2.5]
    noise = np.zeros(64, np.float32)
    rt, _, _, _ = _fake_runtime(logits, noise)
    assert rt.select("greedy") == 5                                   # lowest id among equal maxima
    assert rt.select("sampled", top_k=1) == 9                         # top-k ties rank the larger id first
    noise[12] = 5.0
    assert rt.select("sampled", top_k=3, top_p=1.0) == 12
    assert rt.select("sampled", top_k=3, top_p=0.3) == 9              # nucleus keeps the first only


def test_parameter_blocks_follow_the_kernel_structs():
    from flash_rt.frontends.torch._qwen36_moe_mps_runtime import Qwen36MoeRuntime
    rt = Qwen36MoeRuntime.__new__(Qwen36MoeRuntime)
    rt._param_blob = bytearray()
    a = rt.param("qk", cap=4096, q_norm=11, k_norm=22, kv=33, logbase=23.25)
    b = rt.param("gemm", rows=8, columns=32, inner=2048, x_stride=2048, y_stride=64, w=[1, 2, 3], w2=[4, 5, 6])
    blob = bytes(rt._param_blob)
    assert (a, b) == (0, 256)
    assert np.frombuffer(blob, np.uint32, 2, 0).tolist() == [4096, 0]
    assert np.frombuffer(blob, np.uint64, 3, 8).tolist() == [11, 22, 33]
    assert np.frombuffer(blob, np.float32, 1, 32)[0] == np.float32(23.25)
    assert np.frombuffer(blob, np.uint32, 8, 256).tolist() == [8, 32, 2048, 2048, 64, 0, 0, 0]
    assert np.frombuffer(blob, np.uint64, 6, 288).tolist() == [1, 2, 3, 4, 5, 6]


def test_frontend_reuses_only_an_extended_context():
    from flash_rt.frontends.torch.qwen36_moe_mps import Qwen36MoeTorchFrontendMps
    fe = Qwen36MoeTorchFrontendMps.__new__(Qwen36MoeTorchFrontendMps)
    starts = []

    def prefill(ids, start=0, **kw):
        starts.append(start)
        fe.rt.pos = len(ids)
        return 100

    def decode(token, position, steps, **kw):
        fe.rt.pos = position + steps
        return [101] * steps
    fe.rt = SimpleNamespace(max_seq=4096, prefill=prefill, decode=decode, pos=0)
    fe.cfg = SimpleNamespace(vocab_size=1000)
    fe.eog, fe._state_ids, fe.last_reused = set(), [], 0
    first = fe.generate([1, 2, 3], max_new_tokens=5, reuse_prefix=True)
    assert first == [100, 101, 101, 101, 101] and fe._state_ids == [1, 2, 3, 100, 101, 101, 101]
    fe.generate([1, 2, 3, 100, 101, 101, 101, 101, 9, 9], max_new_tokens=2, reuse_prefix=True)   # next turn
    fe.generate([1, 2, 4], max_new_tokens=2, reuse_prefix=True)                                 # edited prompt
    fe.generate([1, 2, 4, 100, 5], max_new_tokens=2)                                           # reuse not requested
    assert starts == [0, 7, 0, 0]
