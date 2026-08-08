from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flash_rt.structures.impls.qkv_pack import fp8_static
from flash_rt.structures.guard import GuardRefused
from flash_rt.structures.swap import attach


class _FakeKernel:
    @staticmethod
    def bf16_fp8_linear_bias_bf16(
        x, weight, bias, input_scale, weight_scale, *,
        input_fp8, out,
    ):
        del input_scale, input_fp8
        logical_rows = x.shape[0]
        y = torch.nn.functional.linear(
            x.float(),
            weight.float() * weight_scale.float(),
            bias.float(),
        ).to(torch.bfloat16)
        out[:logical_rows].copy_(y)
        return out[:logical_rows]


class _PackHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
        self.k = nn.Linear(4, 2, bias=True, dtype=torch.bfloat16)
        self.v = nn.Linear(4, 1, bias=True, dtype=torch.bfloat16)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


def _bind(monkeypatch, capacity=8):
    monkeypatch.setattr(
        fp8_static, "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=(
                _FakeKernel.bf16_fp8_linear_bias_bf16)))
    torch.manual_seed(7)
    host = _PackHost().eval()
    originals = (host.q, host.k, host.v)
    parts = fp8_static.bind_qkv_pack(
        originals, torch.ones(1), rows=capacity,
        in_dtype="bf16_fused_quant")
    handle = attach(host, {
        "q": parts[0],
        "k": parts[1],
        "v": parts[2],
    })
    return host, originals, handle


def _reference(originals, x):
    return tuple(module(x) for module in originals)


def test_qkv_pack_uses_one_capacity_for_multiple_logical_rows(monkeypatch):
    host, originals, handle = _bind(monkeypatch, capacity=8)
    for rows in (8, 3, 6):
        x = torch.randn(rows, 4, dtype=torch.bfloat16)
        got = host(x)
        want = _reference(originals, x)
        for actual, expected in zip(got, want):
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected,
                                       rtol=0.06, atol=0.06)

    report = handle.report()
    assert sum(entry["fallbacks"] for entry in report.values()) == 0
    assert sum(entry["calls"] for entry in report.values()) == 9
    assert {entry["form"]["row_capacity"]
            for entry in report.values()} == {8}


def test_qkv_pack_falls_back_as_a_group_above_capacity(monkeypatch):
    host, originals, handle = _bind(monkeypatch, capacity=8)
    x = torch.randn(9, 4, dtype=torch.bfloat16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = host(x)
    want = _reference(originals, x)
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    report = handle.report()
    assert sum(entry["fallbacks"] for entry in report.values()) == 3
    assert all("capacity 8" in entry["last_reason"]
               for entry in report.values())


def test_qkv_joint_view_and_remaining_stash_follow_logical_rows(monkeypatch):
    host, _, _ = _bind(monkeypatch, capacity=8)
    head = host.q
    head.enable_joint(2)
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    qk = head.joint(x)
    v = host.v(x)
    assert qk.shape == (3, 5)
    assert v.shape == (3, 1)


@pytest.mark.parametrize(
    "value",
    (
        torch.randn(9, 4, dtype=torch.bfloat16),
        torch.randn(3, 5, dtype=torch.bfloat16),
        torch.randn(3, 4, dtype=torch.float64),
    ),
)
def test_qkv_joint_refuses_every_bound_form_mismatch(monkeypatch, value):
    host, _, handle = _bind(monkeypatch, capacity=8)
    host.q.enable_joint(3)

    with warnings.catch_warnings(), pytest.raises(GuardRefused):
        warnings.simplefilter("ignore")
        host.q.joint(value)

    assert handle.report()["q"]["fallbacks"] == 1


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
        self.head_dim = 2
        self.scale = 2 ** -0.5


def test_qkv_module_form_uses_row_capacity(monkeypatch):
    monkeypatch.setattr(
        fp8_static, "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=(
                _FakeKernel.bf16_fp8_linear_bias_bf16)))
    bound = fp8_static.bind_attn_block(
        _Attention().eval(), torch.ones(1), rows=8)
    for seq in (8, 5):
        out, aux = bound(torch.randn(
            1, seq, 4, dtype=torch.bfloat16))
        assert out.shape == (1, seq, 4)
        assert aux is None
    assert bound._frt_guard.fallbacks == 0
    assert bound._frt_guard.row_capacity == 8
