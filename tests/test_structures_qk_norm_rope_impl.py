"""Mechanism tests for the projection-scope Hub implementation."""

from __future__ import annotations

import types

import pytest
import torch

from flash_rt.structures.guard import GuardRefused
from flash_rt.structures.impls.qk_norm_rope import projection_bf16
from flash_rt.structures.impls.qk_norm_rope import per_head_gqa


class _FakeKernel:
    def __init__(self):
        self.calls = 0

    def qkv_split_bias_norm_rope_v_bf16(
        self,
        packed,
        bias,
        q_weight,
        k_weight,
        cos,
        sin,
        heads,
        head_dim,
        *,
        q_out,
        k_out,
        v_out,
        **_,
    ):
        self.calls += 1
        dim = heads * head_dim
        q, k, v = (packed + bias).split(dim, dim=-1)
        q_out.copy_(q.view_as(q_out))
        k_out.copy_(k.view_as(k_out))
        v_out.copy_(v.view_as(v_out))
        return q_out, k_out, v_out


@pytest.fixture()
def fake_hub(monkeypatch):
    kernel = _FakeKernel()
    monkeypatch.setattr(
        projection_bf16,
        "hub_kernel",
        lambda *_: types.SimpleNamespace(
            qkv_split_bias_norm_rope_v_bf16=(
                kernel.qkv_split_bias_norm_rope_v_bf16
            )
        ),
    )
    return kernel


def _bound(fake_hub):
    del fake_hub
    return projection_bf16.bind_projection_qk_norm_rope(
        torch.ones(32),
        torch.ones(32),
        batch=2,
        tokens=3,
        heads=4,
        head_dim=8,
    )


def test_projection_impl_calls_one_kernel_and_reuses_outputs(fake_hub):
    bound = _bound(fake_hub)
    packed = torch.randn(2, 3, 96, dtype=torch.bfloat16)
    theta = torch.randn(3, 4)

    first = bound(packed, theta.cos(), theta.sin())
    pointers = tuple(out.data_ptr() for out in first)
    second = bound(packed, theta.cos(), theta.sin())

    assert fake_hub.calls == 2
    assert tuple(out.data_ptr() for out in second) == pointers
    q, k, v = packed.split(32, dim=-1)
    torch.testing.assert_close(second[0], q.view(2, 3, 4, 8))
    torch.testing.assert_close(second[1], k.view(2, 3, 4, 8))
    torch.testing.assert_close(second[2], v.view(2, 3, 4, 8))


def test_projection_impl_refuses_wrong_secondary_form(fake_hub):
    bound = _bound(fake_hub)
    packed = torch.randn(2, 3, 96, dtype=torch.bfloat16)

    with pytest.raises(GuardRefused, match="cos/sin"):
        bound(packed, torch.randn(3, 8), torch.randn(3, 8))
    with pytest.raises(GuardRefused, match="float32"):
        bound(
            packed,
            torch.randn(3, 4, dtype=torch.bfloat16),
            torch.randn(3, 4, dtype=torch.bfloat16),
        )


def test_projection_impl_refuses_head_scope_weights(fake_hub):
    del fake_hub
    with pytest.raises(ValueError, match="projection-scope"):
        projection_bf16.ProjectionQkNormRope(
            torch.ones(8),
            torch.ones(8),
            batch=1,
            tokens=3,
            heads=4,
            head_dim=8,
        )


def test_per_head_gqa_impl_reuses_capacity_buffers(monkeypatch):
    class Kernel:
        calls = 0

        @classmethod
        def run(
            cls,
            packed,
            q_weight,
            k_weight,
            cos,
            sin,
            q_heads,
            kv_heads,
            *,
            q_out,
            k_out,
            v_out,
            **kwargs,
        ):
            del q_weight, k_weight, cos, sin, kwargs
            cls.calls += 1
            q_width = q_heads * 128
            kv_width = kv_heads * 128
            q_out.copy_(packed[..., :q_width].view_as(q_out))
            k_out.copy_(
                packed[..., q_width : q_width + kv_width].view_as(k_out)
            )
            v_out.copy_(packed[..., q_width + kv_width :].view_as(v_out))
            return q_out, k_out, v_out

    monkeypatch.setattr(
        per_head_gqa,
        "hub_kernel",
        lambda *_: types.SimpleNamespace(
            qkv_split_per_head_norm_rope_bf16=Kernel.run
        ),
    )
    bound = per_head_gqa.bind_per_head_gqa_qk_norm_rope(
        torch.ones(128, dtype=torch.bfloat16),
        torch.ones(128, dtype=torch.bfloat16),
        row_capacity=8,
        q_heads=4,
        kv_heads=2,
        head_dim=128,
    )
    pointers = None
    for tokens in (8, 3):
        packed = torch.randn(
            1, tokens, (4 + 2 * 2) * 128, dtype=torch.bfloat16
        )
        cos = torch.ones(1, tokens, 128, dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        outputs = bound(packed, cos, sin)
        current = tuple(tensor.data_ptr() for tensor in outputs)
        if pointers is None:
            pointers = current
        else:
            assert current == pointers
        assert outputs[0].shape == (1, tokens, 4, 128)
        assert outputs[1].shape == (1, tokens, 2, 128)
    assert Kernel.calls == 2
