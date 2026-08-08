"""The adarms_stack region family: identification, and the seated floor.

Same stability contract as the first region family, pinned on a box
that cannot run the chain: the identifier claims a conditioned-norm
decoder stack and nothing else — an ordinary decoder tower with plain
norms, a stack with biased projections, or one with square K/Q widths
walks past unclaimed; a cold box resolves to seated; a foreign
receipt falls through when the symbols are absent; a bind failure
leaves every seam in place. The chain's numerics live in the GPU
suite on the device whose hub packages carry the symbols.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _mods():
    """The *current* module instances, via sys.modules (the
    install-smoke purge makes collection-time references stale)."""
    return SimpleNamespace(
        autobuild=importlib.import_module(
            "flash_rt.structures.autobuild"),
        regions=importlib.import_module("flash_rt.structures.regions"),
        fp8_chain=importlib.import_module(
            "flash_rt.structures.impls.adarms_stack.fp8_chain"),
        adarms_region=importlib.import_module(
            "flash_rt.structures.impls.adarms_stack.region"))


DIM, HEADS, KV_HEADS, HEAD_DIM, COND = 64, 4, 1, 16, 32


class _AdaNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(COND, 3 * DIM)
        self.eps = 1e-6


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.head_dim = HEAD_DIM
        self.self_attn.scaling = HEAD_DIM ** -0.5
        self.self_attn.q_proj = nn.Linear(DIM, HEADS * HEAD_DIM,
                                          bias=False)
        self.self_attn.k_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM,
                                          bias=False)
        self.self_attn.v_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM,
                                          bias=False)
        self.self_attn.o_proj = nn.Linear(HEADS * HEAD_DIM, DIM,
                                          bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(DIM, 4 * DIM, bias=False)
        self.mlp.up_proj = nn.Linear(DIM, 4 * DIM, bias=False)
        self.mlp.down_proj = nn.Linear(4 * DIM, DIM, bias=False)
        self.mlp.act_fn = nn.GELU(approximate="tanh")
        self.input_layernorm = _AdaNorm()
        self.post_attention_layernorm = _AdaNorm()


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(3)])
        self.norm = _AdaNorm()

    def rotary_emb(self, x, position_ids):
        seq = position_ids.shape[1]
        angle = torch.zeros(1, seq, HEAD_DIM)
        return angle.cos(), angle.sin()

    def forward(self, *args, **kwargs):
        embs = kwargs.get("inputs_embeds", args[0] if args else None)
        return SimpleNamespace(last_hidden_state=embs)


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = _Stack()

    def forward(self, x):
        return self.expert(inputs_embeds=x).last_hidden_state


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("FRT_DECISION_CACHE",
                       str(tmp_path / "decisions.json"))
    monkeypatch.delenv("FRT_REGION_ADARMS_STACK", raising=False)
    _mods().adarms_region.register()
    yield
    _mods().adarms_region.register()


def _seams(*paths):
    return [SimpleNamespace(path=p) for p in paths]


def test_identify_matches_the_stack_shape_and_nothing_else():
    m = _mods()
    assert m.adarms_region.identify(_Host()) == ["expert"]
    assert m.adarms_region.identify(
        nn.Sequential(nn.Linear(4, 4))) == []


def test_identify_refuses_plain_norms_and_biased_projections():
    m = _mods()
    plain = _Host()
    for block in plain.expert.layers:
        block.input_layernorm = nn.LayerNorm(DIM)
    assert m.adarms_region.identify(plain) == []
    biased = _Host()
    biased.expert.layers[0].self_attn.q_proj = nn.Linear(
        DIM, HEADS * HEAD_DIM, bias=True)
    assert m.adarms_region.identify(biased) == []


def test_identify_refuses_square_key_widths():
    """A stack whose keys are as wide as its queries is outside the
    single-KV cache layout the chain is written for."""
    m = _mods()
    host = _Host()
    for block in host.expert.layers:
        block.self_attn.k_proj = nn.Linear(DIM, HEADS * HEAD_DIM,
                                           bias=False)
        block.self_attn.v_proj = nn.Linear(DIM, HEADS * HEAD_DIM,
                                           bias=False)
    assert m.adarms_region.identify(host) == []


def test_cold_box_keeps_every_seam_and_leaves_a_trail():
    m = _mods()
    host = _Host()
    seams = _seams("expert.layers.0.self_attn.q_proj", "backbone.fc")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras is not None
    assert extras["seams"] == seams
    trails = {t["family"]: t for t in extras["notes"]["regions"]}
    assert trails["adarms_stack"]["winner"] == "seated"


def test_foreign_receipt_falls_to_seated_on_a_box_without_symbols():
    m = _mods()
    if not m.fp8_chain.missing_symbols():
        pytest.skip("this box qualifies the chain; the fall-through "
                    "cannot be exercised here")
    m.regions.record("adarms_stack", "fp8_chain", {"fp8_chain": 60.0})
    host = _Host()
    seams = _seams("expert.layers.0.self_attn.q_proj")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras["seams"] == seams
    trails = {t["family"]: t for t in extras["notes"]["regions"]}
    trail = trails["adarms_stack"]
    assert trail["winner"] == "seated"
    assert trail["fell_through"][0]["name"] == "fp8_chain"
    assert "missing" in trail["fell_through"][0]["reason"]


def test_bind_failure_leaves_the_seats_in_place(monkeypatch):
    m = _mods()
    cand = m.regions.family("adarms_stack").candidate("fp8_chain")
    monkeypatch.setattr(cand, "missing", lambda: [])
    monkeypatch.setenv("FRT_REGION_ADARMS_STACK", "fp8_chain")
    host = _Host()
    seams = _seams("expert.layers.0.self_attn.q_proj", "backbone.fc")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras["seams"] == seams
    refused = dict(extras["notes"]["regions_refused"])
    assert "expert::fp8_chain" in refused
    # and the host still runs, untouched
    host(torch.randn(1, 5, DIM))


def test_mask_facts_read_the_prefix_pad_suffix_band():
    fp8_chain = _mods().fp8_chain
    seq, p_used, p_raw = 4, 6, 8
    total = p_raw + seq
    row = torch.full((total,), float("-inf"))
    row[:p_used] = 0.0
    row[p_raw:] = 0.0
    mask = row.expand(seq, total).reshape(1, 1, seq, total)
    assert fp8_chain._mask_facts(mask, seq) == (p_used, p_raw)
    # a hole inside the prefix run is outside the band
    holey = mask.clone()
    holey[0, 0, :, 2] = float("-inf")
    assert fp8_chain._mask_facts(holey, seq) is None
    # rows that disagree are outside the band
    ragged = mask.clone()
    ragged[0, 0, 1, 0] = float("-inf")
    assert fp8_chain._mask_facts(ragged, seq) is None
    # a batched mask is not this contract
    assert fp8_chain._mask_facts(mask.expand(2, 1, seq, total),
                                 seq) is None


def test_interleave_rows_carries_rotate_half_into_adjacent_pairs():
    fp8_chain = _mods().fp8_chain
    heads, hd, cols = 2, 8, 5
    w = torch.arange(heads * hd * cols, dtype=torch.float32)
    w = w.reshape(heads * hd, cols)
    out = fp8_chain._interleave_rows(w, heads, hd)
    ref = w.reshape(heads, hd, cols)
    for h in range(heads):
        for i in range(hd // 2):
            assert torch.equal(out.reshape(heads, hd, cols)[h, 2 * i],
                               ref[h, i])
            assert torch.equal(
                out.reshape(heads, hd, cols)[h, 2 * i + 1],
                ref[h, i + hd // 2])
