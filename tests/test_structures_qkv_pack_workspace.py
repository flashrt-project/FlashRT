"""The pack's big scratch belongs to the pool, not to the layer.

The stash fix shared the sibling buffers; the two buffers that
actually dominate a large host stayed per-layer (`y_buf` ~792 MiB and
`x8_buf` ~99 MiB per layer on a 19k-token host — a 52-layer bind was
~51 GiB of scratch for buffers whose lifetime is one forward call).
These tests pin the pooled form: same-shape packs share one
allocation, and — the collision discipline — a later pack's write
must not corrupt what an earlier pack already handed out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flash_rt.structures import workspace
from flash_rt.structures.impls.qkv_pack import fp8_static


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


def _mods(seed):
    torch.manual_seed(seed)
    return (nn.Linear(4, 3, bias=True, dtype=torch.bfloat16),
            nn.Linear(4, 2, bias=True, dtype=torch.bfloat16),
            nn.Linear(4, 1, bias=True, dtype=torch.bfloat16))


def _bind(monkeypatch, seed, rows=8):
    monkeypatch.setattr(
        fp8_static, "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=(
                _FakeKernel.bf16_fp8_linear_bias_bf16)))
    mods = _mods(seed)
    parts = fp8_static.bind_qkv_pack(
        mods, torch.ones(1), rows=rows, in_dtype="bf16_fused_quant")
    return mods, parts


@pytest.fixture(autouse=True)
def _fresh_pool():
    workspace.clear()
    yield
    workspace.clear()


def test_same_shape_packs_share_scratch(monkeypatch):
    _, parts_a = _bind(monkeypatch, seed=1)
    _, parts_b = _bind(monkeypatch, seed=2)
    assert parts_a[0].y_buf.data_ptr() == parts_b[0].y_buf.data_ptr()
    assert parts_a[0].x8_buf.data_ptr() == parts_b[0].x8_buf.data_ptr()


def test_different_rows_do_not_share(monkeypatch):
    _, parts_a = _bind(monkeypatch, seed=1, rows=8)
    _, parts_b = _bind(monkeypatch, seed=2, rows=16)
    assert parts_a[0].y_buf.data_ptr() != parts_b[0].y_buf.data_ptr()


def test_sequential_layers_stay_correct(monkeypatch):
    """Layer B overwrites the shared scratch only after layer A's
    outputs left it — the copies handed out must survive."""
    mods_a, parts_a = _bind(monkeypatch, seed=1)
    mods_b, parts_b = _bind(monkeypatch, seed=2)
    x_a = torch.randn(5, 4).to(torch.bfloat16)
    x_b = torch.randn(5, 4).to(torch.bfloat16)

    def run(mods, parts, x):
        q = parts[0](x)
        k = parts[1](x)
        v = parts[2](x)
        return q.clone(), k.clone(), v.clone()

    got_a = run(mods_a, parts_a, x_a)
    got_b = run(mods_b, parts_b, x_b)

    for mods, x, got in ((mods_a, x_a, got_a), (mods_b, x_b, got_b)):
        want = [m(x.float().to(torch.bfloat16)).float() for m in mods]
        for w, g in zip(want, got):
            torch.testing.assert_close(g.float(), w, atol=2e-2, rtol=2e-2)


def test_layer_a_outputs_survive_layer_b(monkeypatch):
    """The strict collision case: read A's q, run B, then read A's
    stashed siblings — the stash slots are per-slot pooled and B ran in
    between, so any cross-layer aliasing bug shows here."""
    mods_a, parts_a = _bind(monkeypatch, seed=1)
    _, parts_b = _bind(monkeypatch, seed=2)
    x = torch.randn(5, 4).to(torch.bfloat16)
    q_a = parts_a[0](x).clone()
    _ = parts_b[0](x)
    want_q = mods_a[0](x).float()
    torch.testing.assert_close(q_a.float(), want_q, atol=2e-2, rtol=2e-2)


def test_host_form_reads_stay_fresh_after_joint_enable(monkeypatch):
    """The H3 0.714 lesson: a joint-enabled head whose module is NOT
    routed still serves host-form calls — forward must stash, and a
    genuinely skipped slot must refuse loudly, never read stale."""
    mods, parts = _bind(monkeypatch, seed=1)
    head, k_reader, _ = parts
    x1 = torch.randn(5, 4).to(torch.bfloat16)
    x2 = torch.randn(5, 4).to(torch.bfloat16)
    _ = head(x1)
    _ = k_reader(x1)
    head.enable_joint(3)
    _ = head(x2)                      # host-form call: must stash
    got = k_reader(x2)
    want = mods[1](x2).float()
    torch.testing.assert_close(got.float(), want, atol=2e-2, rtol=2e-2)

    _ = head.joint(x1)                # joint consumer: skips stashes
    with pytest.raises(Exception, match="stale"):
        k_reader(x1)
