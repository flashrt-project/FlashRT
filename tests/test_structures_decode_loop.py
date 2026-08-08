"""Contract pins for the decode_loop family (CPU, no CUDA needed)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_rt.structures.impls.decode_loop.whole_step import (
    _StaticHybridCache,
    _find_stack,
)


def test_static_hybrid_cache_serves_the_layer_surface():
    c = _StaticHybridCache(4, [1, 3], 2, 8, 32, "cpu")
    k = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    c._cp = torch.tensor([5])
    ko, vo = c.update(k, k, 1)
    assert ko.shape == (1, 2, 32, 8)
    assert torch.equal(ko[:, :, 5], k[:, :, 0])
    # untouched attention slots stay empty; gated-delta slots are plain
    assert c.key_cache[0] is None and c.conv_states[2] is None
    # decode is signalled by a filled conv slot, the host convention
    assert not c.has_previous_state
    c.conv_states[0] = torch.zeros(1)
    assert c.has_previous_state
    assert c.get_mask_sizes(1, 0) == (32, 0)
    # mask sizing is the static window; seq length is TRUE progress —
    # host glue branches on it (a fake length sends multimodal hosts
    # down their continuation path, receipts on record)
    assert c.get_seq_length() == 0
    c._seen = 7
    assert c.get_seq_length() == 7


def test_stack_discovery_is_by_slots_not_names():
    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            self.embed_tokens = nn.Embedding(8, 4)
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _LM()

    lm = _find_stack(_Host())
    assert hasattr(lm, "rotary_emb")

    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.language_model = _LM()
            self.model = inner

    assert hasattr(_find_stack(_Wrapped()), "embed_tokens")

    with pytest.raises(ValueError, match="refused"):
        _find_stack(nn.Linear(4, 4))


def test_mtp_and_release_arms_are_scheme_decisions():
    from flash_rt.structures import schemes

    assert schemes.QuantScheme.mtp_projection_format is None
    assert schemes.QuantScheme.gdn_projection_format is None
    base = schemes.get("w4a4_decode")
    rel = schemes.get("w4a4_decode_release")
    assert not getattr(base, "gdn_release_host_weights", False)
    assert rel.gdn_release_host_weights is True
    assert rel.gdn_projection_format == "nvfp4_dynamic"
    # the release arm is its own registered name, never a mutation of
    # the default arm
    assert base is not rel


def test_decode_loop_door_is_exported():
    from flash_rt import structures

    assert callable(structures.decode_loop)
    assert "decode_loop" in structures.__all__


def test_explain_renders_a_plan_without_a_model():
    from types import SimpleNamespace

    from flash_rt import structures

    plan = SimpleNamespace(
        swaps={"a.mlp": object(), "b.mlp": object()},
        observed={"c@1.core": object()},
        seams=[SimpleNamespace(path="a.mlp", structure="decoder_ffn"),
               SimpleNamespace(path="b.mlp", structure="decoder_ffn")],
        notes={
            "scheme": {"name": "w8a16_decode",
                       "keep_host": {"d.proj": "amax outlier"},
                       "formats": {"a.mlp": "w8a16_static"}},
            "refused": [("qkv_pack", "siblings did not share input")],
        },
    )
    text = structures.explain(plan)
    assert "w8a16_decode" in text
    assert "2 swapped seam(s)" in text
    assert "decoder_ffn: 2" in text
    assert "amax outlier" in text
    assert "siblings did not share input" in text


def test_mtp_tensor_loader_names_its_refusals(tmp_path):
    import json

    import pytest

    from flash_rt.structures.impls.decode_loop.mtp_speculative import (
        _load_mtp_tensors)

    # neither shipping form present
    with pytest.raises(ValueError, match="neither mtp.safetensors"):
        _load_mtp_tensors(tmp_path)

    # a sharded index that carries no draft head
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed.weight": "s1"}}))
    with pytest.raises(ValueError, match="no mtp"):
        _load_mtp_tensors(tmp_path)


def test_draft_precision_axes_are_explicit_and_refuse_by_name():
    import pytest

    from flash_rt.structures.impls.decode_loop.mtp_speculative import (
        DRAFT_FORMATS, check_draft_formats)

    # the measured arms are the whole vocabulary
    assert DRAFT_FORMATS == {"head": ("w8a16_static", "host"),
                             "experts": ("bf16", "nvfp4_dynamic")}
    check_draft_formats("w8a16_static", "bf16")
    check_draft_formats("host", "nvfp4_dynamic")
    with pytest.raises(ValueError, match="unknown draft head"):
        check_draft_formats("w4", "bf16")
    with pytest.raises(ValueError, match="unknown draft experts"):
        check_draft_formats("host", "int4")
