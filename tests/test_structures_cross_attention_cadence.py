from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_rt.structures.impls.cadence_static.cross_attention import (
    bind_cross_attention_kv,
    capture_cross_attention_kv,
    discover_cross_attention_kv,
    refresh_cross_attention_kv,
)


class AttentionSite(nn.Module):
    def __init__(self, hidden: int, encoder: int):
        super().__init__()
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(encoder, hidden, bias=False)
        self.to_v = nn.Linear(encoder, hidden, bias=False)


class FamilyOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionSite(8, 12), AttentionSite(8, 8)]
        )


class FamilyTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleDict(
            {"conditioning": AttentionSite(16, 20)}
        )


@pytest.mark.parametrize(
    ("host", "paths"),
    (
        (
            FamilyOne(),
            ("blocks.0.to_k", "blocks.0.to_v"),
        ),
        (
            FamilyTwo(),
            (
                "stages.conditioning.to_k",
                "stages.conditioning.to_v",
            ),
        ),
    ),
)
def test_cross_kv_discovery_is_semantic_across_host_layouts(host, paths):
    candidates = discover_cross_attention_kv(host)
    assert tuple(candidate.path for candidate in candidates) == paths


def test_cross_kv_cadence_refreshes_once_for_repeated_reads():
    torch.manual_seed(0)
    host = FamilyOne()
    candidates = discover_cross_attention_kv(host)
    encoder = torch.randn(1, 5, 12)

    def repeated_loop():
        for _ in range(4):
            host.blocks[0].to_k(encoder)
            host.blocks[0].to_v(encoder)

    captures = capture_cross_attention_kv(candidates, repeated_loop)
    assert all(len(rows) == 4 for rows in captures)
    swaps, statics = bind_cross_attention_kv(candidates, captures)
    assert tuple(swaps) == ("blocks.0.to_k", "blocks.0.to_v")

    fresh = torch.randn_like(encoder)
    refresh_cross_attention_kv(statics, fresh)
    for candidate, static in zip(candidates, statics):
        expected = candidate.module(fresh)
        assert torch.equal(static(fresh + 1), expected)
        assert static._frt_guard.notes["refreshes"] == 1


def test_cross_kv_cadence_refuses_a_moving_encoder_source():
    host = FamilyOne()
    candidates = discover_cross_attention_kv(host)
    step = [0]

    def moving_loop():
        for _ in range(4):
            encoder = torch.full((1, 5, 12), float(step[0]))
            step[0] += 1
            host.blocks[0].to_k(encoder)
            host.blocks[0].to_v(encoder)

    captures = capture_cross_attention_kv(candidates, moving_loop)
    with pytest.raises(ValueError, match="varies within the hot loop"):
        bind_cross_attention_kv(candidates, captures)


def test_cross_kv_refresh_does_not_mutate_ledger_while_compiling(monkeypatch):
    host = FamilyOne()
    candidates = discover_cross_attention_kv(host)
    encoder = torch.randn(1, 5, 12)

    def one_loop():
        host.blocks[0].to_k(encoder)
        host.blocks[0].to_v(encoder)

    captures = capture_cross_attention_kv(candidates, one_loop)
    _, statics = bind_cross_attention_kv(candidates, captures)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    refresh_cross_attention_kv(statics, encoder)

    assert all(
        static._frt_guard.notes["refreshes"] == 0 for static in statics
    )


def test_cross_kv_refresh_never_uses_a_sibling_ordered_replacement():
    host = FamilyOne()
    candidates = discover_cross_attention_kv(host)
    encoder = torch.randn(1, 5, 12)
    captures = capture_cross_attention_kv(
        candidates,
        lambda: [
            (host.blocks[0].to_k(encoder), host.blocks[0].to_v(encoder))
            for _ in range(2)
        ],
    )

    class StaleSiblingReader(nn.Module):
        _frt_requires_sibling_order = True

        def forward(self, value):
            return torch.full(
                (*value.shape[:-1], 8), -999.0, dtype=value.dtype)

    replacements = {
        candidate.path: StaleSiblingReader() for candidate in candidates
    }
    _, statics = bind_cross_attention_kv(
        candidates, captures, replacements=replacements)
    fresh = torch.randn_like(encoder)
    refresh_cross_attention_kv(statics, fresh)

    for candidate, static in zip(candidates, statics):
        torch.testing.assert_close(
            static.buffer, candidate.module(fresh))
        assert not bool((static.buffer == -999).all())
