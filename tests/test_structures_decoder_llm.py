"""Contract pins for the ``decoder_llm`` structure and ``nvfp4_static``
scheme. CPU-safe: nothing here requires kernels, a GPU, or a model —
the native-build tier refuses cleanly without ``flash_rt_kernels``, and
that refusal is itself the contract pin.
"""

from __future__ import annotations

import pytest

from flash_rt.structures import schemes
from flash_rt.structures.binding import load_binding
from flash_rt.structures.registry import load


def test_autoplan_exposes_one_call_attach_door():
    """The article's ``plan.attach()``: an AutoPlan carries its root and
    commits its swaps atomically (refuses cleanly without a root)."""
    from flash_rt.structures.autobuild import AutoPlan

    plan = AutoPlan()
    assert callable(getattr(plan, "attach", None))
    with pytest.raises(ValueError, match="no root host"):
        plan.attach()
    plan.root = object()
    with pytest.raises(ValueError, match="no swaps or routed seams"):
        plan.attach()


def test_maskgit_loop_door_is_exported():
    from flash_rt import structures

    assert callable(structures.maskgit_loop)
    assert "maskgit_loop" in structures.__all__


def test_decoder_llm_catalog_entry_loads():
    spec = load("decoder_llm")
    assert spec.kind == "region"
    assert spec.symbolic_dims == ("B", "S", "D")
    # a whole-stack seam binds weights in place: no remapped slots
    assert spec.weight_slots == ()
    assert spec.reference() is not None


def test_omnivoice_llm_binding_loads():
    binding = load_binding("omnivoice_llm")
    assert binding.name == "omnivoice_llm"
    assert binding.structure.name == "decoder_llm"
    assert binding.data["dims"]["D"] == 1024
    assert binding.data["dims"]["L"] == 28
    assert not binding.is_pipeline


def test_nvfp4_static_scheme_registered():
    assert "nvfp4_static" in schemes.names()
    scheme = schemes.get("nvfp4_static")
    assert scheme._format == "nvfp4_static"
    assert scheme._linear_format == "nvfp4_static"


class _FakeStats(dict):
    def __init__(self, *args, structure=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.structure = structure


def test_nvfp4_static_routes_decoder_ffn_and_linear_proj():
    scheme = schemes.get("nvfp4_static")

    report = {
        "llm": _FakeStats({}, structure="decoder_llm"),
        "llm.layers.0.mlp": _FakeStats(
            {"llm.layers.0.mlp|act_after_mul": 0.5},
            structure="decoder_ffn"),
        "llm.layers.0.self_attn.q_proj": _FakeStats(
            {}, structure="linear_proj"),
        "llm.layers.0.self_attn.o_proj": _FakeStats(
            {}, structure="linear_proj"),
        "some.other.seam": _FakeStats(
            {"some.other.seam|x": 0.1}, structure="other"),
    }
    decision = scheme.decide(report)
    assert decision.formats == {
        "llm": "nvfp4_static",
        "llm.layers.0.mlp": "nvfp4_static",
        "llm.layers.0.self_attn.q_proj": "nvfp4_static",
        "llm.layers.0.self_attn.o_proj": "nvfp4_static",
    }
    assert decision.keep_host == ("some.other.seam",)


def test_decoder_llm_native_refuses_without_local_build():
    from flash_rt.structures.impls.decoder_llm import nvfp4 as llm_impl

    if llm_impl._native() is not None:
        pytest.skip("local flash_rt_kernels build present")
    with pytest.raises(ValueError, match="refused: decoder_llm nvfp4"):
        llm_impl.bind_decoder_stack(None)


def test_decoder_ffn_nvfp4_refuses_without_local_build():
    from flash_rt.structures.impls.decoder_ffn import (
        nvfp4_static as ffn_impl)

    if ffn_impl._native() is not None:
        pytest.skip("local flash_rt_kernels build present")
    with pytest.raises(ValueError, match="refused: nvfp4_static needs"):
        ffn_impl.bind_mlp_seam(
            {"w_gate": None, "w_up": None, "w_down": None},
            variant={"activation": "silu"})


def test_linear_proj_nvfp4_refuses_without_local_build():
    from flash_rt.structures.impls.linear_proj import (
        nvfp4_static as proj_impl)

    if proj_impl._native() is not None:
        pytest.skip("local flash_rt_kernels build present")
    with pytest.raises(ValueError, match="refused: linear_proj"):
        proj_impl.bind_proj_seam({"w": None})


def test_audio_codec_catalog_entry_loads():
    spec = load("audio_codec")
    assert spec.kind == "region"
    assert spec.symbolic_dims == ("B", "C", "T", "N")


def test_audio_codec_binding_loads():
    binding = load_binding("omnivoice_audio_codec")
    assert binding.structure.name == "audio_codec"
    assert binding.data["hosts"]["omnivoice"]["module_path"] == \
        "audio_tokenizer"


def test_nvfp4_static_routes_audio_codec():
    scheme = schemes.get("nvfp4_static")
    report = {
        "audio_tokenizer": _FakeStats({}, structure="audio_codec"),
    }
    decision = scheme.decide(report)
    assert decision.formats == {"audio_tokenizer": "fp16_codec"}


def test_discovery_finds_decoder_llm_seam_on_stack_slots():
    """The discovery rule keys on slots (layers/embed_tokens/norm/
    rotary_emb), never on model names."""
    import torch

    from flash_rt.structures import discover

    class DummyRotary(torch.nn.Module):
        pass

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            self.embed_tokens = torch.nn.Embedding(100, 16)
            self.norm = torch.nn.LayerNorm(16)
            self.rotary_emb = DummyRotary()

    class Host(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = Stack()

    seams = discover.discover(Host(), structures=("decoder_llm",))
    assert len(seams) == 1
    assert seams[0].structure == "decoder_llm"
    assert seams[0].path == "llm"
    assert seams[0].dims["D"] == 16
