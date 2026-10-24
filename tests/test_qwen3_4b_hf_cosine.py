#!/usr/bin/env python3
"""Qwen3-4B BF16 prefill: target contracts, and full-logits parity with HF.

CPU tests check the routing and the parts of the numerical contract that do
not need a GPU (config refusals, RoPE tables, the reduction-tree switch).

GPU tests compare full-sequence logits against HuggingFace
``Qwen3ForCausalLM`` in BF16 (cosine >= 0.999) with the frontend of the
device's target, at short / sensitive / long prompt lengths, on random
weights: ``Qwen3ForCausalLM(config).to(bf16)`` -- that cast rounds RoPE
``inv_freq`` to bf16, so ``rope_inv_freq="bf16"``. The real-checkpoint
comparison is ``tests/test_qwen3_4b_hf_checkpoint.py``.

Usage:
  PYTHONPATH=. pytest -q tests/test_qwen3_4b_hf_cosine.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from _helpers.qwen3_4b import cosine, frontend_cls  # noqa: E402

COSINE_FLOOR = 0.999
SEQS = [8, 256, 1024]


# ── contracts (no GPU) ──

def test_pipeline_map_binds_one_frontend_per_target():
    from flash_rt.hardware import resolve_pipeline_class
    from flash_rt.frontends.torch.qwen3_4b_rtx import (
        Qwen3_4BTorchFrontendRtx, Qwen3_4BTorchFrontendRtxSm89)
    from flash_rt.frontends.torch.qwen3_4b_thor import Qwen3_4BTorchFrontendThor

    expected = {
        "rtx_sm120": (Qwen3_4BTorchFrontendRtx, (12, 0), "fa2", True),
        "rtx_sm89": (Qwen3_4BTorchFrontendRtxSm89, (8, 9), "fa2", False),
        "thor": (Qwen3_4BTorchFrontendThor, (11, 0), "fa4", True),
    }
    for arch, (cls, cc, attention, pdl) in expected.items():
        resolved = resolve_pipeline_class("qwen3_4b", "torch", arch)
        assert resolved is cls
        t = cls.TARGET
        assert (t.name, t.compute_capability, t.attention, t.pdl) == (arch, cc, attention, pdl)


def test_base_frontend_needs_a_target():
    from flash_rt.frontends.torch._qwen3_4b_bf16_runtime import Qwen3_4BBf16Frontend

    with pytest.raises(TypeError):
        Qwen3_4BBf16Frontend(None)


@pytest.mark.parametrize("override", [
    {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
    {"attention_bias": True},
    {"use_sliding_window": True},
    {"hidden_act": "gelu"},
    {"hidden_size": 4096},
])
def test_config_refuses_what_the_path_does_not_compute(override):
    from flash_rt.frontends.torch._qwen3_4b_bf16_weights import (
        parse_qwen3_config, qwen3_4b_config_dict, qwen3_4b_dims)

    cfg = dict(qwen3_4b_config_dict(), **override)
    with pytest.raises(ValueError):
        qwen3_4b_dims(parse_qwen3_config(cfg))


def test_reduction_tree_follows_the_pytorch_release():
    from flash_rt.models.qwen3.prefill_bf16 import torch_reduce_tree

    assert torch_reduce_tree("2.8.0+cu128") == 0
    assert torch_reduce_tree("2.9.0a0+145a3a7") == 0
    assert torch_reduce_tree("2.10.0") == 1
    assert torch_reduce_tree("2.14.0+cu130") == 1


@pytest.mark.parametrize("inv_freq", ["fp32", "bf16"])
def test_rope_tables_match_hf_rotary_embedding(inv_freq):
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from flash_rt.frontends.torch._qwen3_4b_bf16_runtime import rope_half_tables
    from flash_rt.frontends.torch._qwen3_4b_bf16_weights import qwen3_4b_config_dict
    from flash_rt.models.qwen3.prefill_bf16 import Qwen3_4BDims

    cfg = transformers.Qwen3Config(**{
        k: v for k, v in qwen3_4b_config_dict().items() if k != "architectures"})
    rot = Qwen3RotaryEmbedding(cfg)
    if inv_freq == "bf16":
        rot = rot.to(torch.bfloat16)
    S = 4096
    x = torch.zeros(1, S, 8, dtype=torch.bfloat16)
    cos, sin = rot(x, torch.arange(S)[None])
    c, s = rope_half_tables(Qwen3_4BDims(), S, inv_freq, torch.device("cpu"))
    assert torch.equal(c, cos[0, :, :64]) and torch.equal(s, sin[0, :, :64])
    assert torch.equal(cos[0, :, :64], cos[0, :, 64:])


# ── GPU parity ──

@pytest.fixture(scope="module")
def random_pair():
    cls = frontend_cls()
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from flash_rt.frontends.torch._qwen3_4b_bf16_weights import qwen3_4b_config_dict

    cfg = Qwen3Config(**{k: v for k, v in qwen3_4b_config_dict().items()
                         if k != "architectures"})
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    fe = cls(None, device="cuda", max_seq=max(SEQS), state_dict=hf.state_dict(),
             config=hf.config, rope_inv_freq="bf16")
    yield hf, fe
    del hf, fe
    torch.cuda.empty_cache()


@pytest.mark.parametrize("S", SEQS)
def test_logits_match_hf_random_weights(random_pair, S):
    hf, fe = random_pair
    g = torch.Generator(device="cuda").manual_seed(123)
    ids = torch.randint(0, hf.config.vocab_size, (S,), device="cuda", generator=g)
    with torch.no_grad():
        ref = hf(input_ids=ids[None]).logits[0].float()
    out = fe.forward_prefill(ids).float()
    assert out.shape == ref.shape
    cos = cosine(out, ref)
    assert cos >= COSINE_FLOOR, f"S={S}: full-logits cosine {cos:.6f}"


def test_graph_replay_follows_new_inputs(random_pair):
    # The first call for a length captures its graph; the second, with other
    # token ids, is a pure replay and must equal the eager result for them.
    _, fe = random_pair
    S = 77
    first = torch.randint(0, 151936, (S,), device="cuda")
    second = torch.randint(0, 151936, (S,), device="cuda")
    captured = fe.forward_prefill(first)
    replayed = fe.forward_prefill(second)
    fe._rt.cuda_graphs = False
    try:
        eager = fe.forward_prefill(second)
    finally:
        fe._rt.cuda_graphs = True
    assert torch.equal(replayed, eager)
    assert not torch.equal(replayed, captured)
    # The LM head over one row is its own (M=1) GEMM, so close, not equal.
    last = fe.forward_prefill(second, logits_to_keep=1)
    assert last.shape == (1, eager.shape[1])
    assert cosine(last[0].float(), eager[-1].float()) > 0.99999


def test_autotuned_gemms_match_the_strict_path(random_pair):
    # The strict path's GEMMs are torch's algorithm; the autotuned ones were
    # only checked on probes, so compare the two end to end.
    hf, fe = random_pair
    cls = type(fe)
    tuned = cls(None, device="cuda", max_seq=max(SEQS), state_dict=hf.state_dict(),
                config=hf.config, rope_inv_freq="bf16", autotune_gemms=True)
    g = torch.Generator(device="cuda").manual_seed(5)
    for S in SEQS:
        ids = torch.randint(0, hf.config.vocab_size, (S,), device="cuda", generator=g)
        assert torch.equal(tuned.forward_prefill(ids), fe.forward_prefill(ids)), S
    del tuned
    torch.cuda.empty_cache()


def test_input_forms(random_pair):
    _, fe = random_pair
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    ref = fe.forward_prefill(ids)
    for form in (torch.tensor(ids), torch.tensor([ids], device="cuda"),
                 torch.tensor(ids, dtype=torch.int32, device="cuda")):
        assert torch.equal(fe.forward_prefill(form), ref)
    with pytest.raises(ValueError):
        fe.forward_prefill([0, 151936])
    with pytest.raises(ValueError):
        fe.forward_prefill(torch.zeros(2, 4, dtype=torch.int64))
    with pytest.raises(TypeError):
        fe.forward_prefill(torch.zeros(4))
    with pytest.raises(ValueError):
        fe.forward_prefill(list(range(fe.max_seq + 1)))


def test_refuses_other_targets_device():
    cls = frontend_cls()
    from flash_rt.frontends.torch.qwen3_4b_rtx import (
        Qwen3_4BTorchFrontendRtx, Qwen3_4BTorchFrontendRtxSm89)
    from flash_rt.frontends.torch.qwen3_4b_thor import Qwen3_4BTorchFrontendThor

    for other in (Qwen3_4BTorchFrontendRtx, Qwen3_4BTorchFrontendRtxSm89,
                  Qwen3_4BTorchFrontendThor):
        if other is cls:
            continue
        with pytest.raises(RuntimeError, match="validated on"):
            other(None, state_dict={}, config={})


def test_refuses_a_build_without_the_kernels(monkeypatch):
    cls = frontend_cls()
    import types
    import flash_rt
    from flash_rt import flash_rt_kernels as real

    stub = types.SimpleNamespace(**{n: getattr(real, n) for n in dir(real)
                                    if n != "qk_rms_norm_rope_hf_bf16"})
    monkeypatch.setattr(flash_rt, "flash_rt_kernels", stub)
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_kernels", stub)
    with pytest.raises(RuntimeError, match="qk_rms_norm_rope_hf_bf16"):
        cls(None, state_dict={}, config={})


def test_attention_backend_is_required(monkeypatch):
    cls = frontend_cls()
    from flash_rt.models.qwen3.prefill_bf16 import Qwen3_4BDims

    dims = Qwen3_4BDims()
    buf = torch.empty(16 * 32 * 128, device="cuda", dtype=torch.bfloat16)
    make = cls.TARGET.make_attention
    if cls.TARGET.attention == "fa2":
        import flash_rt
        monkeypatch.delattr(flash_rt, "flash_rt_fa2", raising=False)
        monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fa2", None)
    else:
        from flash_rt.hardware.thor import fa4_backend
        monkeypatch.setattr(fa4_backend, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        make(dims, 16, buf, buf, buf, buf, torch.device("cuda"))
