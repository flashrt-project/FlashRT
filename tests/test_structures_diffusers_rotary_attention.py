from __future__ import annotations

import torch
from torch import nn

from flash_rt.structures.adapters.diffusers_rotary_attention import (
    DiffusersRotaryAttentionAdapter,
)


class _Processor:
    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, rotary_emb=None,
    ):
        del attention_mask
        context = hidden_states if encoder_hidden_states is None \
            else encoder_hidden_states
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(context))
        value = attn.to_v(context)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            from flash_rt.structures.adapters.diffusers_rotary_attention \
                import _apply_rotary
            query = _apply_rotary(query, rotary_emb)
            key = _apply_rotary(key, rotary_emb)
        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2),
            value.transpose(1, 2)).transpose(1, 2)
        return attn.to_out[1](attn.to_out[0](out.flatten(2, 3)))


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.norm_q = nn.RMSNorm(8)
        self.norm_k = nn.RMSNorm(8)
        self.to_out = nn.ModuleList((nn.Linear(8, 8), nn.Identity()))
        self.add_k_proj = None
        self.fused_projections = False
        self.is_cross_attention = False
        self.processor = _Processor()

    def forward(self, hidden_states, rotary_emb):
        return self.processor(self, hidden_states, rotary_emb=rotary_emb)


class _Core(nn.Module):
    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value)


def test_rotary_adapter_preserves_processor_semantics(monkeypatch):
    host = nn.ModuleDict({"attention": _Attention()}).eval()
    x = torch.randn(1, 4, 8)
    cos = torch.ones(1, 4, 1, 4)
    sin = torch.zeros_like(cos)
    rotary = (cos, sin)
    expected = host.attention(x, rotary)

    monkeypatch.setattr(
        "flash_rt.structures.adapters.diffusers_rotary_attention."
        "bind_dense_attention_best",
        lambda rows: _Core(),
    )
    result = DiffusersRotaryAttentionAdapter()(
        host, lambda: host.attention(x, rotary))
    assert result is not None
    swaps, update, extras = result
    assert not swaps and update is None
    assert extras["observed"]
    got = host.attention(x, rotary)
    torch.testing.assert_close(got, expected)
    extras["revert"][0]()
    assert torch.equal(host.attention(x, rotary), expected)


def test_rotary_adapter_refuses_added_image_kv():
    host = nn.ModuleDict({"attention": _Attention()}).eval()
    host.attention.add_k_proj = nn.Linear(8, 8)
    x = torch.randn(1, 4, 8)
    assert DiffusersRotaryAttentionAdapter()(
        host, lambda: host.attention(x, None)) is None
