from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt.structures.adapters.diffusers_attention import (
    DiffusersAttentionAdapter,
)


class AttnProcessor2_0:
    def __init__(self):
        self.calls = 0

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        self.calls += 1
        del attention_mask, temb, args, kwargs
        source = (
            hidden_states
            if encoder_hidden_states is None else encoder_hidden_states)
        batch, seq_q, _ = hidden_states.shape
        seq_kv = source.shape[1]
        dim = attn.to_q.out_features // attn.heads
        q = attn.to_q(hidden_states).view(
            batch, seq_q, attn.heads, dim).transpose(1, 2)
        k = attn.to_k(source).view(
            batch, seq_kv, attn.heads, dim).transpose(1, 2)
        v = attn.to_v(source).view(
            batch, seq_kv, attn.heads, dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_q, -1)
        return attn.to_out[1](attn.to_out[0](out))


class FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q = torch.nn.Linear(16, 16)
        self.to_k = torch.nn.Linear(16, 16)
        self.to_v = torch.nn.Linear(16, 16)
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(16, 16), torch.nn.Identity()])
        self.processor = AttnProcessor2_0()
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.norm_q = None
        self.norm_k = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0

    def prepare_attention_mask(self, attention_mask, sequence_length,
                               batch_size):
        # mirror of the diffusers helper this adapter consumes
        return attention_mask.reshape(
            -1, attention_mask.shape[-2], attention_mask.shape[-1]
        ).expand(batch_size * self.heads, -1, attention_mask.shape[-1])

    def forward(self, hidden, encoder):
        return self.processor(self, hidden, encoder)


class Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()
        self.hidden = torch.randn(1, 7, 16)
        self.encoder = torch.randn(1, 13, 16)

    def forward(self):
        return self.attention(self.hidden, self.encoder)


class Core(torch.nn.Module):
    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(query, key, value)


class BF16Core(Core):
    def __init__(self):
        super().__init__()
        self._frt_guard = type(
            "Guard", (), {"dtypes": frozenset({torch.bfloat16})}
        )()
        self.seen_dtype = None

    def forward(self, query, key, value):
        self.seen_dtype = query.dtype
        return super().forward(query, key, value)


def test_diffusers_adapter_routes_and_restores(monkeypatch):
    monkeypatch.setattr(
        "flash_rt.structures.adapters.diffusers_attention."
        "bind_dense_attention_best",
        lambda captures: Core(),
    )
    root = Root()
    original = root.attention.processor
    reference = root()
    result = DiffusersAttentionAdapter()(root, root.forward)
    assert result is not None
    _, _, extras = result
    torch.testing.assert_close(root(), reference)
    assert set(extras["observed"]) == {
        "attention.processor::fa2_core"}
    extras["toggle"][1]()
    assert root.attention.processor is original
    extras["toggle"][0]()
    torch.testing.assert_close(root(), reference)
    extras["revert"][0]()
    assert root.attention.processor is original


def test_diffusers_adapter_uses_host_instead_of_hidden_dtype_cast(monkeypatch):
    core = BF16Core()
    monkeypatch.setattr(
        "flash_rt.structures.adapters.diffusers_attention."
        "bind_dense_attention_best",
        lambda captures: core,
    )
    root = Root()
    reference = root()
    original = root.attention.processor
    result = DiffusersAttentionAdapter()(root, root.forward)
    assert result is not None
    actual = root()
    assert core.seen_dtype is None
    assert original.calls == 3  # reference, calibration, routed fallback
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(actual, reference)


def test_diffusers_adapter_matches_capability_not_processor_class_name(
        monkeypatch):
    class RenamedCompatibleProcessor(AttnProcessor2_0):
        pass

    monkeypatch.setattr(
        "flash_rt.structures.adapters.diffusers_attention."
        "bind_dense_attention_best",
        lambda captures: Core(),
    )
    root = Root()
    root.attention.processor = RenamedCompatibleProcessor()

    result = DiffusersAttentionAdapter()(root, root.forward)

    assert result is not None
    assert result[2]["observed"]


def test_diffusers_adapter_hands_masked_sites_to_the_family(monkeypatch):
    monkeypatch.setattr(
        "flash_rt.structures.adapters.diffusers_attention."
        "bind_dense_attention_best",
        lambda captures: Core(),
    )
    root = Root()
    original = root.attention.processor

    result = DiffusersAttentionAdapter()(
        root,
        lambda: root.attention.processor(
            root.attention, root.hidden, root.encoder,
            torch.zeros(1, 1, 7, 13)),
    )

    assert result is not None
    swaps, update, extras = result
    assert swaps == {} and update is None
    # masked sites now reach the family: the outcome is either a bound
    # core (a variant covered the fixed mask pattern) or the family's
    # recorded refusal — never a silent skip
    assert extras.get("observed") or extras["refused"], \
        "masked site must bind or record a family verdict"
    if extras.get("observed"):
        # a variant covered the fixed mask pattern: the routed
        # processor stands until the toggle disables it
        assert root.attention.processor is not original
        enable, disable = extras["toggle"]
        disable()
        assert root.attention.processor is original
    else:
        assert root.attention.processor is original
