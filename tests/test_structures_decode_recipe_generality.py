"""The decode recipe is family-table driven, not model-named.

Discovery finds attention projections by the sibling-group tables in
``discover.py`` and the w8a16_decode scheme routes them by structure
name — no host model name appears anywhere on that path. These tests
pin that on config-constructed models from two families the recipe was
never run against on a GPU, so a family regression shows up here as a
CPU failure instead of as a silent non-discovery on someone's host.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from flash_rt.structures.discover import discover  # noqa: E402
from flash_rt.structures.schemes import W8A16Decode  # noqa: E402


def _tiny(cfg_name, model_name, **kw):
    try:
        # resolution is inside the guard: these classes import lazily,
        # and some environments carry a kernels-library version whose
        # hub decorator breaks that import. That is the environment's
        # issue, not a discovery result, so the family is skipped with
        # the reason instead of read as a failure
        cfg = getattr(transformers, cfg_name)(
            hidden_size=512, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=128, **kw)
        with torch.no_grad():
            return getattr(transformers, model_name)(cfg)
    except ValueError as exc:
        pytest.skip(f"{model_name} cannot construct here: {exc}")


FAMILIES = [
    ("llama", "LlamaConfig", "LlamaForCausalLM"),
    ("gemma", "GemmaConfig", "GemmaForCausalLM"),
    ("qwen3", "Qwen3Config", "Qwen3ForCausalLM"),
]


@pytest.mark.parametrize("name,cfg_name,model_name", FAMILIES)
def test_projection_family_is_discovered_across_architectures(
        name, cfg_name, model_name):
    model = _tiny(cfg_name, model_name)
    seams = discover(model, ("linear_proj", "decoder_ffn"))
    projs = [s for s in seams if s.structure == "linear_proj"]
    ffns = [s for s in seams if s.structure == "decoder_ffn"]
    # 2 layers x q/k/v/o
    assert len(projs) == 8, f"{name}: {[s.path for s in projs]}"
    assert {s.proj_attr for s in projs} == {
        "q_proj", "k_proj", "v_proj", "o_proj"}
    assert len(ffns) == 2


@pytest.mark.parametrize("name,cfg_name,model_name", FAMILIES)
def test_decode_recipe_routes_both_seam_kinds(name, cfg_name, model_name):
    model = _tiny(cfg_name, model_name)
    seams = discover(model, ("linear_proj", "decoder_ffn"))

    class Stats(dict):
        def __init__(self, structure, values):
            super().__init__(values)
            self.structure = structure

    report = {}
    for s in seams:
        point = ("act_after_mul" if s.structure == "decoder_ffn" else "x")
        report[s.path] = Stats(s.structure, {f"{s.path}|{point}": None})
    d = W8A16Decode().decide(report)
    routed = set(d.formats.values())
    assert routed == {"w8a16_static"}
    assert len(d.formats) == len(seams), (
        f"{name}: kept {d.keep_host} at host — the recipe missed a "
        "seam kind it claims to cover")
    assert not d.keep_host
