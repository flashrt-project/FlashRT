from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flash_rt.structures.impls.attention_core import fa2_seqused


class _FakePackedKVAttention:
    def __init__(
        self,
        plan,
        q_shape,
        kv_heads,
        dtype,
        device,
        prefix_kv=None,
        scratch=None,
    ):
        del q_shape, kv_heads, dtype, device, prefix_kv
        self.plan = plan
        self._scratch = scratch or SimpleNamespace()


class _FakeFa2:
    SUPPORTED_HEAD_DIMS = tuple(range(8, 257, 8))

    @staticmethod
    def allocate_outputs(q):
        return torch.empty_like(q), torch.empty(
            q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)

    @staticmethod
    def allocate_workspace(q, k):
        del q, k
        return None

    @staticmethod
    def forward_static(
        q, k, v, *, out, softmax_lse, workspace, softmax_scale,
    ):
        del softmax_lse, workspace
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=softmax_scale,
        ).transpose(1, 2)
        out.copy_(ref)
        return out


def _capture(head_dim: int):
    q = torch.randn(1, 4, 7, head_dim, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 13, head_dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    return {
        "q": q,
        "keys": [key, key.clone()],
        "values": [value, value.clone()],
        "mask": None,
    }


@pytest.fixture(autouse=True)
def _artifact_capabilities(monkeypatch):
    monkeypatch.setattr(
        fa2_seqused, "supported_head_dims",
        lambda: _FakeFa2.SUPPORTED_HEAD_DIMS,
    )


@pytest.mark.parametrize("head_dim", [48, 64, 72, 80, 128, 256])
def test_attention_core_admits_production_logical_head_dims(
    monkeypatch, head_dim
):
    monkeypatch.setattr(
        fa2_seqused, "PackedKVAttention", _FakePackedKVAttention
    )
    bound = fa2_seqused.bind_attention_core([_capture(head_dim)])
    assert bound is not None
    modules, update = bound
    assert len(modules) == 1
    assert callable(update)


@pytest.mark.parametrize("head_dim", [7, 44, 264])
def test_attention_core_refuses_unaligned_or_oversized_head_dims(head_dim):
    assert fa2_seqused.bind_attention_core([_capture(head_dim)]) is None


def test_dense_attention_uses_complete_per_call_kv(monkeypatch):
    monkeypatch.setattr(
        fa2_seqused, "hub_kernel", lambda *args: _FakeFa2())
    query = torch.randn(1, 4, 7, 48, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 13, 48, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    captures = [
        {"q": query, "key": key, "value": value, "mask": None},
        {
            "q": query.clone(),
            "key": key.add(1),
            "value": value.add(1),
            "mask": None,
        },
    ]
    core = fa2_seqused.bind_dense_attention(captures)
    got0 = core(query, key, value).clone()
    got1 = core(query, key.add(1), value.add(1)).clone()
    ref0 = torch.nn.functional.scaled_dot_product_attention(
        query, key, value)
    ref1 = torch.nn.functional.scaled_dot_product_attention(
        query, key.add(1), value.add(1))
    torch.testing.assert_close(got0, ref0)
    torch.testing.assert_close(got1, ref1)
    assert not torch.equal(got0, got1)


@pytest.mark.parametrize(
    "allowed",
    [
        list(range(3, 11)),
        list(range(0, 4)) + list(range(10, 13)),
    ],
)
def test_dense_attention_packs_one_or_two_allowed_runs(
    monkeypatch, allowed
):
    torch.manual_seed(0)
    monkeypatch.setattr(
        fa2_seqused, "hub_kernel", lambda *args: _FakeFa2())
    query = torch.randn(1, 4, 7, 48, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 13, 48, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.zeros(1, 4, 1, 13, dtype=torch.bool)
    mask[..., allowed] = True
    captures = [{
        "q": query, "key": key, "value": value, "mask": mask,
    }]
    core = fa2_seqused.bind_dense_attention(captures)
    got = core(query, key, value).clone()
    reference = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask)
    torch.testing.assert_close(got, reference, rtol=2e-2, atol=1e-3)
