from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from flash_rt.structures.adapters import qwen_per_head_qk_norm_rope as adapter_mod
from flash_rt.structures.impls.qkv_pack import fp8_static


def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, **kwargs
):
    del module, kwargs
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scaling
    )
    return output.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS = {}


class _FakeGemm:
    @staticmethod
    def bf16_fp8_linear_bias_bf16(
        x,
        weight,
        bias,
        input_scale,
        weight_scale,
        *,
        input_fp8,
        out,
    ):
        del input_scale, input_fp8
        value = torch.nn.functional.linear(
            x.float(), weight.float() * weight_scale.float(), bias.float()
        ).to(torch.bfloat16)
        out[: x.shape[0]].copy_(value)
        return out[: x.shape[0]]


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        scale = torch.rsqrt(
            value.float().square().mean(-1, keepdim=True)
            + self.variance_epsilon
        )
        return (value.float() * scale * self.weight.float()).to(value.dtype)


class _QwenLikeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 4 * 128, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(128, 2 * 128, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(128, 2 * 128, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(4 * 128, 128, bias=False, dtype=torch.bfloat16)
        self.q_norm = _Norm()
        self.k_norm = _Norm()
        self.head_dim = 128
        self.num_key_value_groups = 2
        self.scaling = 128**-0.5
        self.attention_dropout = 0.0
        self.layer_idx = 3
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        del cache_position, kwargs
        assert past_key_values is None
        batch, tokens, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, tokens, 4, 128)
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(batch, tokens, 2, 128)
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, tokens, 2, 128
        ).transpose(1, 2)
        cos, sin = position_embeddings
        rotate_query = torch.cat((-query[..., 64:], query[..., :64]), -1)
        rotate_key = torch.cat((-key[..., 64:], key[..., :64]), -1)
        query = query * cos.unsqueeze(1) + rotate_query * sin.unsqueeze(1)
        key = key * cos.unsqueeze(1) + rotate_key * sin.unsqueeze(1)
        output, weights = eager_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=self.scaling,
        )
        return self.o_proj(output.reshape(batch, tokens, -1)), weights


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _QwenLikeAttention()


class _FakePostprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, packed, cos, sin):
        self.calls += 1
        batch, tokens, _ = packed.shape
        q, k, v = packed.split((4 * 128, 2 * 128, 2 * 128), dim=-1)

        def norm_rope(value, heads):
            value = value.view(batch, tokens, heads, 128).float()
            value = value * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + 1e-6
            )
            rotated = torch.cat((-value[..., 64:], value[..., :64]), -1)
            return (value * cos.unsqueeze(2) + rotated * sin.unsqueeze(2)).to(
                torch.bfloat16
            )

        return (
            norm_rope(q, 4),
            norm_rope(k, 2),
            v.view(batch, tokens, 2, 128),
        )


def test_qwen_adapter_composes_pack_and_is_reversible(monkeypatch):
    monkeypatch.setattr(
        fp8_static,
        "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=_FakeGemm.bf16_fp8_linear_bias_bf16
        ),
    )
    fake_postprocess = _FakePostprocess()
    monkeypatch.setattr(
        adapter_mod,
        "bind_per_head_gqa_qk_norm_rope",
        lambda *args, **kwargs: fake_postprocess,
    )
    host = _Host().eval()
    parts = fp8_static.bind_qkv_pack(
        (host.attn.q_proj, host.attn.k_proj, host.attn.v_proj),
        torch.ones(1),
        rows=7,
        in_dtype="bf16_fused_quant",
    )
    plan = SimpleNamespace(
        swaps={
            "attn.q_proj": parts[0],
            "attn.k_proj": parts[1],
            "attn.v_proj": parts[2],
        }
    )
    adapter = adapter_mod.QwenPerHeadQkNormRopeAdapter()
    extras = adapter(host, plan)
    assert extras is not None
    enable, disable = extras["toggle"]

    hidden = torch.randn(1, 7, 128, dtype=torch.bfloat16)
    angle = torch.randn(1, 7, 128, dtype=torch.bfloat16)
    position = (angle.cos(), angle.sin())
    routed, _ = host.attn(hidden, position, None)
    assert routed.shape == (1, 7, 128)
    assert fake_postprocess.calls == 1
    assert parts[0].joint_slots == 3

    disable()
    assert parts[0].joint_slots == 0
    host.attn(hidden, position, None)
    assert fake_postprocess.calls == 1

    enable()
    host.attn(hidden, position, None)
    assert fake_postprocess.calls == 2

    class Cache:
        def __init__(self):
            self.seen = None

        def update(self, key, value, layer_idx, cache_kwargs):
            self.seen = (layer_idx, cache_kwargs)
            return key, value

    cache = Cache()
    host.attn(
        hidden,
        position,
        None,
        past_key_values=cache,
        cache_position=torch.arange(7),
    )
    assert fake_postprocess.calls == 3
    assert cache.seen[0] == 3
    assert cache.seen[1]["cos"] is position[0]
    extras["revert"][0]()
    assert parts[0].joint_slots == 0


def test_qwen_adapter_reports_a_recognised_but_incompatible_site(monkeypatch):
    monkeypatch.setattr(
        fp8_static,
        "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=_FakeGemm.bf16_fp8_linear_bias_bf16
        ),
    )
    host = _Host().eval()
    parts = fp8_static.bind_qkv_pack(
        (host.attn.q_proj, host.attn.k_proj, host.attn.v_proj),
        torch.ones(1),
        rows=7,
        in_dtype="bf16_fused_quant",
    )
    del host.attn.config
    plan = SimpleNamespace(
        swaps={
            "attn.q_proj": parts[0],
            "attn.k_proj": parts[1],
            "attn.v_proj": parts[2],
        }
    )

    extras = adapter_mod.PerHeadGqaQkNormRopeAdapter()(host, plan)

    assert extras is not None
    assert not extras.get("observed")
    assert "host lacks the complete" in extras["refused"][0][1]
