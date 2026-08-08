from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flash_rt.structures.adapters import factored_qk_norm_rope as adapter_mod
from flash_rt.structures.impls.qkv_pack import fp8_static


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


class _Postprocess(nn.Module):
    def __init__(self, q_heads=4, kv_heads=2):
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.calls = 0

    def forward(self, packed, cos, sin):
        self.calls += 1
        batch, tokens, _ = packed.shape
        q, k, v = packed.split(
            (
                self.q_heads * 128,
                self.kv_heads * 128,
                self.kv_heads * 128,
            ),
            dim=-1,
        )
        # Keep the fake observable without reproducing the qualified kernel.
        q = q.view(batch, tokens, self.q_heads, 128) + cos.unsqueeze(2)
        k = k.view(batch, tokens, self.kv_heads, 128) + sin.unsqueeze(2)
        v = v.view(batch, tokens, self.kv_heads, 128)
        return q, k, v


class _FactoredAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(128, 4 * 128, bias=False, dtype=torch.bfloat16)
        self.to_k = nn.Linear(128, 2 * 128, bias=False, dtype=torch.bfloat16)
        self.to_v = nn.Linear(128, 2 * 128, bias=False, dtype=torch.bfloat16)
        self.add_q_proj = nn.Linear(
            128, 4 * 128, bias=False, dtype=torch.bfloat16
        )
        self.add_k_proj = nn.Linear(
            128, 2 * 128, bias=False, dtype=torch.bfloat16
        )
        self.add_v_proj = nn.Linear(
            128, 2 * 128, bias=False, dtype=torch.bfloat16
        )
        self.to_out = nn.Linear(
            4 * 128, 128, bias=False, dtype=torch.bfloat16
        )
        self.to_add_out = nn.Linear(
            4 * 128, 128, bias=False, dtype=torch.bfloat16
        )
        self.norm_q = _Norm()
        self.norm_k = _Norm()
        self.norm_added_q = _Norm()
        self.norm_added_k = _Norm()
        self.head_dim = 128
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.cp_mesh = None
        self.config = SimpleNamespace(freeze_und=False)
        self.dispatch_attention_fn = self._attention
        self.original_calls = 0

    @staticmethod
    def _attention(query, key, value):
        del key, value
        return {
            **query,
            "causal_seq": query["causal_seq"].flatten(-2),
            "full_only_seq": query["full_only_seq"].flatten(-2),
        }

    def forward(
        self,
        pack,
        attention_mask,
        packed_position_embeddings,
        dual_kv_cache=None,
        natten_metadata=None,
    ):
        del attention_mask, packed_position_embeddings, dual_kv_cache
        del natten_metadata
        self.original_calls += 1
        return pack


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FactoredAttention()


def _bind_packs(host, rows=5):
    und = fp8_static.bind_qkv_pack(
        (host.attn.to_q, host.attn.to_k, host.attn.to_v),
        torch.ones(1),
        rows=rows,
        in_dtype="bf16_fused_quant",
    )
    gen = fp8_static.bind_qkv_pack(
        (host.attn.add_q_proj, host.attn.add_k_proj, host.attn.add_v_proj),
        torch.ones(1),
        rows=rows,
        in_dtype="bf16_fused_quant",
    )
    return und, gen


def _plan(und, gen):
    return SimpleNamespace(
        swaps={
            "attn.to_q": und[0],
            "attn.to_k": und[1],
            "attn.to_v": und[2],
            "attn.add_q_proj": gen[0],
            "attn.add_k_proj": gen[1],
            "attn.add_v_proj": gen[2],
        }
    )


def _position(tokens):
    angle = torch.randn(tokens, 128, dtype=torch.bfloat16)
    return angle.cos(), angle.sin()


def test_factored_qk_norm_rope_composes_both_packs_and_restores(monkeypatch):
    monkeypatch.setattr(
        fp8_static,
        "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=_FakeGemm.bf16_fp8_linear_bias_bf16
        ),
    )
    bounds = []

    def bind(*args, **kwargs):
        del args, kwargs
        bound = _Postprocess()
        bounds.append(bound)
        return bound

    monkeypatch.setattr(adapter_mod, "bind_per_head_gqa_qk_norm_rope", bind)
    host = _Host().eval()
    und, gen = _bind_packs(host)
    original = host.attn.forward
    extras = adapter_mod.FactoredQkNormRopeAdapter()(host, _plan(und, gen))

    assert extras is not None
    assert len(extras["observed"]) == 2
    assert und[0].joint_slots == gen[0].joint_slots == 3
    pack = {
        "causal_seq": torch.randn(3, 128, dtype=torch.bfloat16),
        "full_only_seq": torch.randn(2, 128, dtype=torch.bfloat16),
        "metadata": "preserved",
    }
    position = (
        {
            "causal_seq": _position(3)[0],
            "full_only_seq": _position(2)[0],
        },
        {
            "causal_seq": _position(3)[1],
            "full_only_seq": _position(2)[1],
        },
    )
    result = host.attn(pack, None, position)
    assert result["causal_seq"].shape == (3, 128)
    assert result["full_only_seq"].shape == (2, 128)
    assert result["metadata"] == "preserved"
    assert [bound.calls for bound in bounds] == [1, 1]

    enable, disable = extras["toggle"]
    disable()
    assert und[0].joint_slots == gen[0].joint_slots == 0
    assert host.attn(pack, None, position) is pack
    assert host.attn.original_calls == 1
    assert [bound.calls for bound in bounds] == [1, 1]

    enable()
    host.attn(pack, None, position)
    assert [bound.calls for bound in bounds] == [2, 2]
    extras["revert"][0]()
    assert und[0].joint_slots == gen[0].joint_slots == 0
    assert "forward" not in host.attn.__dict__
    assert host.attn.forward.__func__ is original.__func__


def test_factored_qk_norm_rope_refuses_an_incomplete_pair(monkeypatch):
    monkeypatch.setattr(
        fp8_static,
        "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=_FakeGemm.bf16_fp8_linear_bias_bf16
        ),
    )
    host = _Host().eval()
    und, gen = _bind_packs(host)
    plan = _plan(und, gen)
    del plan.swaps["attn.add_v_proj"]

    extras = adapter_mod.FactoredQkNormRopeAdapter()(host, plan)

    assert extras is not None
    assert not extras.get("observed")
    assert "both causal and full QKV groups" in extras["refused"][0][1]
    assert "forward" not in host.attn.__dict__


def test_factored_qk_norm_rope_refuses_mutating_cache_at_runtime(monkeypatch):
    monkeypatch.setattr(
        fp8_static,
        "hub_kernel",
        lambda *args, **kwargs: SimpleNamespace(
            bf16_fp8_linear_bias_bf16=_FakeGemm.bf16_fp8_linear_bias_bf16
        ),
    )
    monkeypatch.setattr(
        adapter_mod,
        "bind_per_head_gqa_qk_norm_rope",
        lambda *args, **kwargs: _Postprocess(),
    )
    host = _Host().eval()
    und, gen = _bind_packs(host)
    adapter_mod.FactoredQkNormRopeAdapter()(host, _plan(und, gen))
    pack = {
        "causal_seq": torch.randn(3, 128, dtype=torch.bfloat16),
        "full_only_seq": torch.randn(2, 128, dtype=torch.bfloat16),
    }
    position = (
        {"causal_seq": _position(3)[0], "full_only_seq": _position(2)[0]},
        {"causal_seq": _position(3)[1], "full_only_seq": _position(2)[1]},
    )

    with pytest.raises(adapter_mod.GuardRefused, match="cache mutation"):
        host.attn(pack, None, position, dual_kv_cache=object())
