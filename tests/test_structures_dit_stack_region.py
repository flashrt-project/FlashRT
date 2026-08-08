"""The dit_block region family: identification, and the seated floor.

What these tests pin is the stability contract of the first region
family, on a box that cannot run the chain: the identifier claims the
stack shape and nothing else; a cold box resolves to seated; a
foreign receipt naming the chain falls through to seated because this
box does not qualify it; and a candidate whose bind fails leaves
every seam exactly where the scan put it. The chain's numerics live
in the GPU suite and on the device whose hub packages carry the
symbols — here the only thing that must be true is that nothing
degrades and every outcome leaves a trail.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _mods():
    """The *current* module instances, via sys.modules.

    The install-smoke test purges and re-imports the whole flash_rt
    package mid-suite; a reference taken at collection time then talks
    to a different registry than the one ``autobuild`` uses. Resolving
    at call time is the same route the mechanism itself takes.
    """
    return SimpleNamespace(
        autobuild=importlib.import_module(
            "flash_rt.structures.autobuild"),
        regions=importlib.import_module("flash_rt.structures.regions"),
        fp4_chain=importlib.import_module(
            "flash_rt.structures.impls.dit_stack.fp4_chain"),
        dit_region=importlib.import_module(
            "flash_rt.structures.impls.dit_stack.region"))

DIM, XDIM = 64, 48


class _FfProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, 4 * DIM)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x))


class _Block(nn.Module):
    def __init__(self, cross: bool):
        super().__init__()
        self.dim = DIM
        self.num_attention_heads = 2
        self.attention_head_dim = DIM // 2
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(DIM, DIM)
        self.attn1.to_k = nn.Linear(XDIM if cross else DIM, DIM)
        self.attn1.to_v = nn.Linear(XDIM if cross else DIM, DIM)
        self.attn1.to_out = nn.ModuleList([nn.Linear(DIM, DIM),
                                           nn.Dropout(0.0)])
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([_FfProj(), nn.Identity(),
                                     nn.Linear(4 * DIM, DIM)])
        self.norm1 = nn.Module()
        self.norm1.linear = nn.Linear(DIM, 2 * DIM)


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_Block(cross=(i % 2 == 0)) for i in range(4)])
        self.timestep_encoder = nn.Linear(1, DIM)
        self.norm_out = nn.LayerNorm(DIM)
        self.proj_out_1 = nn.Linear(DIM, 2 * DIM)
        self.proj_out_2 = nn.Linear(DIM, DIM)

    def forward(self, hidden_states, timestep):
        temb = self.timestep_encoder(timestep)
        h = hidden_states + temb
        for block in self.transformer_blocks:
            h = h + 0.0 * block.attn1.to_q(h)
        return self.proj_out_2(self.norm_out(h))


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = _Stack()

    def forward(self, x, t):
        return self.dit(x, t)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("FRT_DECISION_CACHE",
                       str(tmp_path / "decisions.json"))
    monkeypatch.delenv("FRT_REGION_DIT_BLOCK", raising=False)
    _mods().dit_region.register()
    yield
    _mods().dit_region.register()


def _seams(*paths):
    return [SimpleNamespace(path=p) for p in paths]


def test_identify_matches_the_stack_shape_and_nothing_else():
    m = _mods()
    assert m.dit_region.identify(_Host()) == ["dit"]
    assert m.dit_region.identify(nn.Sequential(nn.Linear(4, 4))) == []
    # all-self stacks are not this region: no cross block, no claim
    host = _Host()
    for block in host.dit.transformer_blocks:
        block.attn1.to_k = nn.Linear(DIM, DIM)
        block.attn1.to_v = nn.Linear(DIM, DIM)
    assert m.dit_region.identify(host) == []


def test_cold_box_keeps_every_seam_and_leaves_a_trail():
    m = _mods()
    host = _Host()
    seams = _seams("dit.transformer_blocks.0.attn1.to_q", "backbone.fc")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras is not None
    assert extras["seams"] == seams
    trail = extras["notes"]["regions"][0]
    assert (trail["family"], trail["winner"]) == ("dit_block", "seated")


def test_no_region_host_pays_nothing():
    extras = _mods().autobuild._bind_regions(
        nn.Sequential(nn.Linear(4, 4)), _seams("0"),
        probe=lambda: None, say=lambda m: None)
    assert extras is None


def test_foreign_receipt_falls_to_seated_on_a_box_without_symbols():
    """The transport scenario: a Thor-measured receipt reaches a box
    whose hub package lacks the chain symbols. Speed degrades to the
    floor; nothing else happens."""
    m = _mods()
    if not m.fp4_chain.missing_symbols():
        pytest.skip("this box qualifies the chain; the fall-through "
                    "cannot be exercised here")
    m.regions.record("dit_block", "fp4_chain", {"fp4_chain": 34.17})
    host = _Host()
    seams = _seams("dit.transformer_blocks.0.attn1.to_q")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras["seams"] == seams
    trail = extras["notes"]["regions"][0]
    assert trail["winner"] == "seated"
    assert trail["fell_through"][0]["name"] == "fp4_chain"
    assert "missing" in trail["fell_through"][0]["reason"]


def test_bind_failure_leaves_the_seats_in_place(monkeypatch):
    """A candidate that qualifies on paper but dies at bind must not
    have claimed anything: the scan proceeds over every seam."""
    m = _mods()
    cand = m.regions.family("dit_block").candidate("fp4_chain")
    monkeypatch.setattr(cand, "missing", lambda: [])
    monkeypatch.setenv("FRT_REGION_DIT_BLOCK", "fp4_chain")
    host = _Host()
    seams = _seams("dit.transformer_blocks.0.attn1.to_q", "backbone.fc")
    extras = m.autobuild._bind_regions(
        host, seams, probe=lambda: None, say=lambda m: None)
    assert extras["seams"] == seams
    refused = extras["notes"]["regions_refused"]
    assert refused and refused[0][0] == "dit::fp4_chain"
    # and the host still runs, untouched
    host(torch.randn(1, 3, DIM), torch.randn(1, 1))


def test_parse_call_refuses_out_of_contract_shapes():
    fp4_chain = _mods().fp4_chain
    h = torch.randn(1, 8, DIM)
    enc = torch.randn(1, 4, XDIM)
    t = torch.tensor([5.0])
    assert fp4_chain._parse_call((h, enc, t), {}) is not None
    assert fp4_chain._parse_call((h, enc), {"timestep": t}) is not None
    # batched, missing timestep, unknown kwarg, double-given arg
    assert fp4_chain._parse_call((h.repeat(2, 1, 1), enc, t), {}) is None
    assert fp4_chain._parse_call((h, enc), {}) is None
    assert fp4_chain._parse_call((h, enc, t), {"surprise": 1}) is None
    assert fp4_chain._parse_call((h, enc, t),
                                 {"hidden_states": h}) is None


def test_auto_swaps_threads_the_region_trail(monkeypatch):
    host = _Host()
    x, t = torch.randn(1, 3, DIM), torch.randn(1, 1)
    ref = host(x, t)
    plan = _mods().autobuild.auto_swaps(host, lambda: host(x, t),
                                        scheme="none")
    assert any(e["family"] == "dit_block"
               for e in plan.notes.get("regions", []))
    got = host(x, t)
    torch.testing.assert_close(got, ref)
