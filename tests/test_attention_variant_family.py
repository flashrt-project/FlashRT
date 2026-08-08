"""The dense attention family records why each preferred variant lost.

A host that silently binds a slower or lower-precision variant looks,
in the receipt, exactly like a host where the preferred one was weighed
and rejected. They need opposite fixes — one is a distribution gap, the
other a real capability limit — so the trail has to survive the
selection.
"""

from __future__ import annotations

import torch

from flash_rt.structures.impls import attention_core


class _Core(torch.nn.Module):
    def forward(self, query, key, value, *, scale=None):
        del scale
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value)


def test_family_names_the_bound_variant_and_keeps_the_trail(monkeypatch):
    monkeypatch.setattr(
        attention_core, "bind_dense_attention",
        lambda captures: (_ for _ in ()).throw(
            OSError("no build variant for this host")))
    from flash_rt.structures.impls.attention_core import fa4_cute

    bound = _Core()
    monkeypatch.setattr(fa4_cute, "bind_dense_attention",
                        lambda captures: bound)

    core = attention_core.bind_dense_attention_best([{"q": None}])

    assert core is bound
    assert core._frt_variant == "fa4_cute"
    assert len(core._frt_variant_trail) == 1
    assert core._frt_variant_trail[0].startswith("fa2: ")
    assert "no build variant" in core._frt_variant_trail[0]


def test_trail_separates_an_absent_package_from_a_shape_decline(
        monkeypatch):
    from flash_rt.structures.impls.attention_core import (fa4_cute,
                                                          masked_mha)

    # fa2 executes its qualification and declines the captured form
    monkeypatch.setattr(attention_core, "bind_dense_attention",
                        lambda captures: None)
    # fa4 is simply not distributed to this host
    monkeypatch.setattr(
        fa4_cute, "bind_dense_attention",
        lambda captures: (_ for _ in ()).throw(
            RuntimeError("kernel 'kernels-community/flash-attn4' is "
                         "not staged")))
    bound = _Core()
    monkeypatch.setattr(masked_mha, "bind_dense_attention",
                        lambda captures: bound)

    core = attention_core.bind_dense_attention_best([{"q": None}])

    assert core._frt_variant == "masked_mha"
    trail = core._frt_variant_trail
    assert trail[0] == "fa2: declined the captured shape form"
    assert trail[1].startswith("fa4_cute: ") and "not staged" in trail[1]


def test_a_site_no_variant_serves_still_declines_rather_than_raises(
        monkeypatch):
    from flash_rt.structures.impls.attention_core import (fa4_cute,
                                                          fa4_fp8,
                                                          masked_mha)

    monkeypatch.setattr(attention_core, "bind_dense_attention",
                        lambda captures: None)
    for module in (fa4_cute, masked_mha, fa4_fp8):
        monkeypatch.setattr(module, "bind_dense_attention",
                            lambda captures: None)

    assert attention_core.bind_dense_attention_best([{"q": None}]) is None
