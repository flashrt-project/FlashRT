"""The precision entry: one parameter, profiles by name, auto default.

``scheme=`` is the only precision door. ``"auto"`` resolves to the
highest-performing profile the device can execute; ``"none"`` is the
explicit quantisation off-switch (fusion structures are unaffected —
they never consult a scheme decision); ``"w4a16_decode"`` is the 4-bit
weight-only twin of the validated INT8 recipe. These tests pin the
resolution table, the two new profiles' decisions, and the 4-bit impl's
contract surface that is checkable without a GPU.
"""

import pytest

from flash_rt.structures import schemes
from flash_rt.structures.schemes import (Bf16Structural, Decision, NoQuant, PointStat,
                                         W4A16Decode, W8A16Decode,
                                         resolve_auto, validate_request)


class _Pt:
    def __init__(self, path, name):
        self.path, self.name = path, name


def test_new_profiles_are_registered():
    assert isinstance(schemes.get("none"), NoQuant)
    assert isinstance(schemes.get("bf16_structural"), Bf16Structural)
    assert isinstance(schemes.get("w4a16_decode"), W4A16Decode)
    assert {"none", "bf16_structural", "w4a16_decode"} <= set(schemes.names())


def test_bf16_structural_routes_only_qkv_pack():
    class Stats(dict):
        def __init__(self, structure, values):
            super().__init__(values)
            self.structure = structure

    report = {
        "block.attn.q": Stats("qkv_pack", {"block.attn.q|x": None}),
        "block.attn.o": Stats("linear_proj", {"block.attn.o|x": None}),
        "block.norm": Stats("norm_fused", {"block.norm|x": None}),
    }
    decision = Bf16Structural().decide(report)
    assert decision.formats == {"block.attn.q": "bf16_pack"}
    assert decision.keep_host == ("block.attn.o",)


def test_auto_resolves_by_fp8_capability(monkeypatch):
    from flash_rt.core.utils import hardware

    monkeypatch.setattr(hardware, "supports_fp8", lambda: True)
    assert resolve_auto() == "fp8_static"
    monkeypatch.setattr(hardware, "supports_fp8", lambda: False)
    assert resolve_auto() == "none"


def test_auto_resolves_to_none_when_probe_unavailable(monkeypatch):
    from flash_rt.core.utils import hardware

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(hardware, "supports_fp8", boom)
    assert resolve_auto() == "none"


def test_none_profile_wants_no_statistics_and_keeps_everything():
    pts = [_Pt("a.mlp", "x_after_norm"), _Pt("b.proj", "x")]
    req = NoQuant().statistics(pts)
    assert all(ps == PointStat(None) for ps in req.values())
    validate_request(req)  # executable: nothing to measure

    report = {"a.mlp": {"a.mlp|x_after_norm": None},
              "b.proj": {"b.proj|x": None}}
    d = NoQuant().decide(report)
    assert set(d.keep_host) == {"a.mlp", "b.proj"}
    assert not d.formats
    assert all("off" in r for r in d.reasons.values())


def test_w4a16_scheme_routes_ffn_and_keeps_the_rest():
    report = {
        "layers.0.mlp": {"layers.0.mlp|x_after_norm": None,
                         "layers.0.mlp.down_proj|act_after_mul": None},
        "layers.0.self_attn": {"layers.0.self_attn|x": None},
    }
    d = W4A16Decode().decide(report)
    assert d.formats == {"layers.0.mlp": "w4a16_static"}
    assert d.keep_host == ("layers.0.self_attn",)
    assert "w4a16_decode" in d.reasons["layers.0.self_attn"]


def test_w8a16_routing_is_unchanged_by_the_subclass():
    report = {"layers.0.mlp": {
        "layers.0.mlp.down_proj|act_after_mul": None}}
    assert W8A16Decode().decide(report).formats == {
        "layers.0.mlp": "w8a16_static"}


class _Stats(dict):
    def __init__(self, structure, values):
        super().__init__(values)
        self.structure = structure


def test_w8a16_scheme_routes_linear_proj_seams():
    # the decode recipe covers the attention projection family too: a
    # report entry carrying the linear_proj structure name is routed to
    # the projection twin, everything unrecognised stays at host
    report = {
        "layers.0.mlp": _Stats("decoder_ffn", {
            "layers.0.mlp.down_proj|act_after_mul": None}),
        "layers.0.self_attn.q_proj": _Stats("linear_proj", {
            "layers.0.self_attn.q_proj|x": None}),
        "layers.0.norm": _Stats("norm_fused", {"layers.0.norm|x": None}),
    }
    d = W8A16Decode().decide(report)
    assert d.formats == {"layers.0.mlp": "w8a16_static",
                         "layers.0.self_attn.q_proj": "w8a16_static"}
    assert d.keep_host == ("layers.0.norm",)


def test_w4a16_scheme_keeps_linear_proj_at_host():
    # the 4-bit twin's linear auto band is too narrow to route blind;
    # its projections stay at host precision until a measured table
    # says otherwise
    report = {"layers.0.self_attn.q_proj": _Stats("linear_proj", {
        "layers.0.self_attn.q_proj|x": None})}
    d = W4A16Decode().decide(report)
    assert not d.formats
    assert d.keep_host == ("layers.0.self_attn.q_proj",)


def test_w8a16_linear_band_mirrors_the_kernel_qualification():
    # the kernel's auto dispatch accepts M in [1,4] with an N/K shape
    # table (w8_auto_linear_supported in the package's binding); the
    # impl's predicate must agree with it, not rediscover it as
    # runtime errors
    from flash_rt.structures.impls.linear_proj.w8a16_static import _qualified

    assert _qualified(1, 512, 1024)          # K <= 1024 always
    assert _qualified(4, 128, 1024)
    assert _qualified(1, 1024, 4096)         # K <= 4096 needs N >= 1024
    assert not _qualified(1, 960, 4096)
    assert _qualified(2, 1024, 8192)         # K > 4096, M <= 2: N >= 1024
    assert not _qualified(3, 1024, 8192)     # K > 4096, M in {3,4}: N >= 2048
    assert _qualified(3, 2048, 8192)
    assert not _qualified(5, 16384, 4096)    # M=5 never qualifies
    assert not _qualified(0, 16384, 4096)


def test_w8a16_linear_impl_contract_surface():
    import torch

    from flash_rt.structures.impls.linear_proj import w8a16_static

    assert w8a16_static.KERNEL_DEP["repo"] == "flashrt/weight-only-ffn"
    assert w8a16_static._check({"w": torch.zeros(1024, 4096)}) == (1024, 4096)
    with pytest.raises(ValueError, match="outside support envelope"):
        w8a16_static._check({"w": torch.zeros(1024, 96)})
    with pytest.raises(ValueError, match="bias shape"):
        w8a16_static._check({"w": torch.zeros(1024, 4096),
                             "b": torch.zeros(960)})


def test_bind_router_accepts_w8a16_linear_format():
    import inspect

    from flash_rt.structures import autobuild

    src = inspect.getsource(autobuild._bind_auto)
    assert 'fmt == "w8a16_static"' in src


def test_w4a16_impl_contract_surface():
    import torch

    from flash_rt.structures.impls.decoder_ffn import w4a16_static

    # entry points name real exports of the pinned Hub package; the
    # activation map refuses what the kernel does not implement
    assert w4a16_static.KERNEL_DEP["repo"] == "flashrt/weight-only-ffn"
    with pytest.raises(ValueError, match="unsupported activation"):
        w4a16_static._entrypoint({"activation": "swish"})

    # the dim envelope walls off shapes the kernel does not cover,
    # before any weight leaves the host
    good = {"w_gate": torch.zeros(1024, 512),
            "w_up": torch.zeros(1024, 512),
            "w_down": torch.zeros(512, 1024)}
    assert w4a16_static._check(good) == (512, 1024)
    bad = dict(good, w_down=torch.zeros(512, 960))
    with pytest.raises(ValueError, match="inconsistent"):
        w4a16_static._check(bad)
    narrow = {"w_gate": torch.zeros(1024, 96),
              "w_up": torch.zeros(1024, 96),
              "w_down": torch.zeros(96, 1024)}
    with pytest.raises(ValueError, match="outside support envelope"):
        w4a16_static._check(narrow)


def test_w4a16_band_mirrors_the_kernel_qualification():
    # the kernel's auto dispatch accepts M in [1,3] with a per-M floor
    # on total weight elements; the impl's band table must agree with
    # it, not rediscover it as runtime errors
    from flash_rt.structures.impls.decoder_ffn.w4a16_static import (
        _AUTO_FLOOR, _in_band)

    assert _AUTO_FLOOR == {1: 12 << 20, 2: 32 << 20, 3: 64 << 20}
    assert _in_band(1, 12 << 20)
    assert not _in_band(1, (12 << 20) - 1)
    assert _in_band(2, 32 << 20)
    assert not _in_band(2, (32 << 20) - 1)
    assert _in_band(3, 64 << 20)
    assert not _in_band(4, 1 << 40)      # M=4 never qualifies


def test_bind_router_accepts_w4a16_format():
    # the router must not wall the 4-bit twin the way it walls unknown
    # formats; unknown names still fail loudly (pinned elsewhere)
    import inspect

    from flash_rt.structures import autobuild

    src = inspect.getsource(autobuild._bind_auto)
    assert "w4a16_static" in src


def test_default_decision_shape_is_stable():
    # the none profile's Decision is an ordinary Decision — no new
    # vocabulary for "do nothing"
    assert NoQuant().decide({}) == Decision()
