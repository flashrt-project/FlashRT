from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from flash_rt.structures.adapters import packed_qkv_rope as adapter_mod
from flash_rt.structures.catalog.qkv_rope.reference import qkv_rope_ref
from flash_rt.structures.registry import load


def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, **kwargs
):
    del module, kwargs
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scaling
    )
    return output.transpose(1, 2).contiguous(), None


class _Dispatch:
    @staticmethod
    def get_interface(name, eager):
        assert name == "eager"
        return eager


ALL_ATTENTION_FUNCTIONS = _Dispatch()


class _VisionMlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, 2 * dim, dtype=torch.bfloat16)
        self.linear_fc2 = nn.Linear(2 * dim, dim, dtype=torch.bfloat16)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class _PackedVisionAttention(nn.Module):
    def __init__(self, heads=4, head_dim=8):
        super().__init__()
        dim = heads * head_dim
        self.num_heads = heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.qkv = nn.Linear(dim, 3 * dim, dtype=torch.bfloat16)
        self.proj = nn.Linear(dim, dim, dtype=torch.bfloat16)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        del kwargs
        tokens = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).view(
            tokens, 3, self.num_heads, self.head_dim
        ).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1).float()
        sin = sin.unsqueeze(1).float()

        def rotate(value):
            first, second = value.float().chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        q = (q.float() * cos + rotate(q) * sin).to(q.dtype)
        k = (k.float() * cos + rotate(k) * sin).to(k.dtype)
        q, k, v = (tensor.transpose(0, 1).unsqueeze(0) for tensor in (q, k, v))
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        outputs = [
            eager_attention_forward(
                self, qs, ks, vs, None, self.scaling, is_causal=False
            )[0]
            for qs, ks, vs in zip(*[
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (q, k, v)
            ])
        ]
        return self.proj(torch.cat(outputs, dim=1).reshape(tokens, -1).contiguous())


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(32, dtype=torch.bfloat16)
        self.attn = _PackedVisionAttention()
        self.mlp = _VisionMlp(32)


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()


class _FakeRope(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias
        self.calls = 0

    def forward(self, packed, cos, sin):
        self.calls += 1
        return qkv_rope_ref(
            packed,
            self.bias,
            cos,
            sin,
            q_heads=4,
            kv_heads=4,
            head_dim=8,
        )


class _FakeDenseAttention(nn.Module):
    instances = []

    def __init__(self, q_shape, kv_shape, dtype, device, *, scratch=None):
        super().__init__()
        del q_shape, kv_shape, dtype, device
        self._scratch = object() if scratch is None else scratch
        self._frt_guard = None
        self.calls = 0
        self.__class__.instances.append(self)

    def forward(self, query, key, value, *, scale=None):
        self.calls += 1
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=scale
        )


def test_qkv_rope_catalog_excludes_norm_and_cache_state():
    spec = load("qkv_rope")
    assert spec.version == 1
    assert spec.weight_slots == ("qkv_bias",)
    assert "without q/k normalization" in spec.description.lower()
    assert "cache" in spec.description.lower()
    assert "cache" not in {
        item["name"]
        for side in ("inputs", "outputs")
        for item in spec.boundary[side]
    }


def test_packed_vision_adapter_is_capability_based_and_reversible(monkeypatch):
    host = _Host().eval()
    fake = _FakeRope(host.block.attn.qkv.bias)
    monkeypatch.setattr(
        adapter_mod,
        "bind_packed_bias_qkv_rope",
        lambda *args, **kwargs: fake,
    )
    tokens = 7
    hidden = torch.randn(tokens, 32, dtype=torch.bfloat16)
    angle = torch.randn(tokens, 4, dtype=torch.float32)
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1).contiguous()
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1).contiguous()
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    expected = host.block.attn(hidden, cu, (cos, sin))

    adapter = adapter_mod.PackedQkvRopeAdapter()
    extras = adapter(
        host,
        SimpleNamespace(seams=[]),
        {"block.mlp": {"rows": tokens}},
    )
    assert extras is not None
    actual = host.block.attn(hidden, cu, (cos, sin))
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert fake.calls == 1

    enable, disable = extras["toggle"]
    disable()
    host.block.attn(hidden, cu, (cos, sin))
    assert fake.calls == 1
    enable()
    host.block.attn(hidden, cu, (cos, sin))
    assert fake.calls == 2
    extras["revert"][0]()


def test_packed_vision_adapter_composes_single_segment_attention(monkeypatch):
    host = _Host().eval()
    fake_rope = _FakeRope(host.block.attn.qkv.bias)
    _FakeDenseAttention.instances = []
    monkeypatch.setattr(
        adapter_mod,
        "bind_packed_bias_qkv_rope",
        lambda *args, **kwargs: fake_rope,
    )
    monkeypatch.setattr(
        adapter_mod, "DenseAttention", _FakeDenseAttention
    )
    tokens = 7
    hidden = torch.randn(tokens, 32, dtype=torch.bfloat16)
    angle = torch.randn(tokens, 4, dtype=torch.float32)
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1).contiguous()
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1).contiguous()
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    expected = host.block.attn(hidden, cu, (cos, sin))

    plan = SimpleNamespace(seams=[], notes={})
    extras = adapter_mod.PackedQkvRopeAdapter()(
        host,
        plan,
        {"block.mlp": {"rows": tokens}},
        compose_attention=True,
    )
    assert extras is not None
    actual = host.block.attn(hidden, cu, (cos, sin))
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert len(_FakeDenseAttention.instances) == 1
    assert _FakeDenseAttention.instances[0].calls == 2  # bind smoke + route
    assert "block.attn::attention_core" in extras["observed"]
    assert plan.notes["composed_structures"] == [
        "qkv_rope->attention_core"
    ]

    multi_segment = torch.tensor([0, 3, tokens], dtype=torch.int32)
    multi_expected = _PackedVisionAttention.forward(
        host.block.attn, hidden, multi_segment, (cos, sin)
    )
    multi_actual = host.block.attn(
        hidden, multi_segment, (cos, sin)
    )
    torch.testing.assert_close(
        multi_actual, multi_expected, atol=2e-2, rtol=2e-2
    )
    assert _FakeDenseAttention.instances[0].calls == 2
    extras["revert"][0]()


def test_packed_vision_adapter_refuses_causal_packed_attention(monkeypatch):
    host = _Host().eval()
    host.block.attn.is_causal = True
    monkeypatch.setattr(
        adapter_mod,
        "bind_packed_bias_qkv_rope",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("causal site must refuse before binding")
        ),
    )
    extras = adapter_mod.PackedQkvRopeAdapter()(
        host,
        SimpleNamespace(seams=[]),
        {"block.mlp": {"rows": 7}},
    )
    assert extras is not None
    assert "causal attention" in extras["refused"][0][1]
