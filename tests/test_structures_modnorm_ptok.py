"""Contract surface of the per-token-table ``modnorm_qkv_chain`` form.

The video-DiT block whose modulation is inline in ``forward`` from a
per-token timestep table. These tests pin what is checkable without a
GPU: shape-based discovery (positive and correctly-negative), the
variant-specific calibration point set and its placement, the router
branch, and the wire projection's off-wire fallback accounting.
"""

import pytest
import torch
from torch import nn

from flash_rt.structures.discover import discover
from flash_rt.structures.points import resolve as resolve_points


class _Attn(nn.Module):
    def __init__(self, dim, cross=False):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])


class _Ffn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        act = nn.Module()
        act.proj = nn.Linear(dim, 4 * dim)
        self.net = nn.ModuleList(
            [act, nn.Identity(), nn.Linear(4 * dim, dim)])


class _Block(nn.Module):
    """The shape the predicate looks for — no host class names."""

    def __init__(self, dim=512, chunks=6, affine=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=affine)
        self.attn1 = _Attn(dim)
        self.attn2 = _Attn(dim, cross=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.ffn = _Ffn(dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=affine)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, chunks, dim) / dim**0.5)


class _Model(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(**kw)])


def test_table_block_is_discovered_by_shape():
    seams = discover(_Model(), ("modnorm_qkv_chain",))
    assert len(seams) == 1
    seam = seams[0]
    assert seam.structure == "modnorm_qkv_chain"
    assert seam.variant["modulation"] == "per_token_table"
    assert seam.dims == {"D": 512, "C": 6}


def test_wrong_shapes_are_correctly_not_discovered():
    # an affine norm1 is a different composition (the affine fold would
    # be silently lost) — must not be discovered
    assert not discover(_Model(affine=True), ("modnorm_qkv_chain",))
    # no modulation table -> not this structure
    plain = _Model()
    del plain.blocks[0].scale_shift_table
    assert not discover(plain, ("modnorm_qkv_chain",))


def test_variant_specific_calibration_points():
    from flash_rt.structures.autobuild import _spec_points

    seam = discover(_Model(), ("modnorm_qkv_chain",))[0]
    assert _spec_points(seam) == ("attn_in", "o_in", "ffn_in", "ffn_hid")
    placed = resolve_points(seam, _spec_points(seam))
    by_name = {p.name: p.path for p in placed}
    assert by_name["attn_in"].endswith(".attn1.to_q")
    assert by_name["o_in"].endswith(".attn1.to_out.0")
    assert by_name["ffn_in"].endswith(".ffn")
    assert by_name["ffn_hid"].endswith(".ffn.net.2")


def test_chain_owns_its_producer_fed_members():
    # the block chain excludes exactly the members its producer feeds
    # (self-attention pack, FFN); cross-attention and the output
    # projection stay individually bindable
    import inspect

    from flash_rt.structures import autobuild

    src = inspect.getsource(autobuild.auto_swaps)
    assert "per_token_table" in src
    assert '".attn1"' in src and '".ffn"' in src


def test_router_binds_the_table_form():
    import inspect

    from flash_rt.structures import autobuild

    src = inspect.getsource(autobuild._bind_auto)
    assert "fp8_ptok_table" in src


def test_wire_projection_off_wire_fallback_is_counted():
    from flash_rt.structures.impls.modnorm_qkv_chain.fp8_ptok_table import (
        WireProj)

    lin = nn.Linear(64, 64)
    wire = WireProj(lin, gemm=None)
    x = torch.randn(3, 64)
    out = wire(x)
    # no wire armed: the host projection ran, and the dispatch is counted
    assert torch.allclose(out, lin(x))
    assert wire.off_wire_calls == 1


def test_binder_refuses_unreleased_producer_build(monkeypatch):
    from flash_rt.structures.impls.modnorm_qkv_chain import fp8_ptok_table

    class _OldPkg:
        pass

    fp8_ptok_table._producer.cache_clear()
    monkeypatch.setattr(
        "flash_rt.structures.impls.hub_kernel",
        lambda repo, version: _OldPkg())
    with pytest.raises(ValueError, match="predates the per-token"):
        fp8_ptok_table._producer()
    fp8_ptok_table._producer.cache_clear()
