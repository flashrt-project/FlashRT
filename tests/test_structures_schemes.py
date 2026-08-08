"""The quantisation-scheme boundary: registry, the loud wall, decisions.

A scheme owns two things — what statistic each point needs, and what
happens at each seam — and nothing else. These tests pin the contract:
the registry resolves and refuses by name, a request the collector
cannot measure fails loudly instead of silently degrading to per-tensor,
and keep-host is a decision with a reason attached, not a refusal.
"""

import pytest

from flash_rt.structures import schemes
from flash_rt.structures.schemes import (Decision, Fp8Static, PointStat,
                                         QuantScheme, validate_request)


class _Pt:
    def __init__(self, path, name):
        self.path, self.name = path, name


def test_registry_resolves_and_refuses_by_name():
    assert isinstance(schemes.get("fp8_static"), Fp8Static)
    assert "fp8_static" in schemes.names()
    with pytest.raises(KeyError, match="registered"):
        schemes.get("no_such_scheme")


def test_default_scheme_requests_per_tensor_amax_everywhere():
    pts = [_Pt("a.mlp", "x_after_norm"), _Pt("a.mlp.down_proj",
                                             "act_after_mul")]
    req = Fp8Static().statistics(pts)
    assert set(req) == {"a.mlp|x_after_norm",
                        "a.mlp.down_proj|act_after_mul"}
    assert all(ps == PointStat("amax", "tensor") for ps in req.values())
    validate_request(req)  # executable today, must not raise


def test_unmeasurable_granularity_fails_loudly():
    # NVFP4 weight scale factors are per-16-block; until the collector
    # measures that, the request must hit a wall — not silently receive
    # per-tensor numbers of the wrong shape
    with pytest.raises(NotImplementedError, match="block16"):
        validate_request({"w|weight_sf": PointStat("amax", "block16")})
    with pytest.raises(NotImplementedError, match="histogram"):
        validate_request({"w|hist": PointStat("histogram", "tensor")})


def test_dynamic_no_statistic_is_a_legal_declaration():
    # a format that quantises activations in-kernel needs nothing here
    validate_request({"a|x": PointStat(None, "tensor")})


def test_default_decides_to_bind_everything():
    report = {"layers.0.mlp": {"layers.0.mlp|x_after_norm": 0.02}}
    assert Fp8Static().decide(report) == Decision()


def test_keep_outliers_consumes_the_house_criterion():
    report = {
        "layers.0.mlp": {"layers.0.mlp|x": 0.010},
        "layers.1.mlp": {"layers.1.mlp|x": 0.012},
        "layers.2.mlp": {"layers.2.mlp|x": 0.011},
        "layers.16.mlp": {"layers.16.mlp|act": 0.90},   # 75x the median
    }
    decision = Fp8Static(keep_outliers=20.0).decide(report)
    assert decision.keep_host == ("layers.16.mlp",)
    assert "kept at host precision" in decision.reasons["layers.16.mlp"]
    # below the ceiling nothing is kept
    assert Fp8Static(keep_outliers=100.0).decide(report).keep_host == ()


def test_channel_statistics_are_measurable_now():
    validate_request({"a|x": PointStat("amax", "channel"),
                      "b|x": PointStat("second_moment", "channel")})
    # block16 stays walled until the collector measures it
    with pytest.raises(NotImplementedError):
        validate_request({"a|x": PointStat("amax", "block16")})


def test_collector_channel_amax_and_second_moment():
    import numpy as np
    import torch

    from flash_rt.structures.points import Collector, Point

    pt = Point(name="x", path="m.proj")
    key = f"{pt.path}|{pt.name}"
    for stat, expect in (("amax", None), ("second_moment", None)):
        col = Collector(points=[pt], request={key: PointStat(stat,
                                                             "channel")})
        a = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
        b = torch.tensor([[-3.0, 1.0]])
        col._record(pt, a)
        col._record(pt, b)          # second call, same sample
        col.end_sample()
        col.reduce(99.9)
        if stat == "amax":
            got = col.channel_amax(pt.path, pt.name)
            assert np.allclose(got, [3.0, 4.0])   # max over both calls
            assert col.second_moment(pt.path, pt.name) is None
        else:
            got = col.second_moment(pt.path, pt.name)
            # E[x^2] per channel over all 3 rows of the sample
            assert np.allclose(got, [(1 + 0.25 + 9) / 3,
                                     (4 + 16 + 1) / 3])
        # the scalar per-tensor amax rides along regardless — it keeps
        # the per-sample vectors aligned
        assert col.amax(pt.path, pt.name) == 4.0


def test_w8a16_scheme_routes_ffn_and_keeps_the_rest():
    scheme = schemes.get("w8a16_decode")
    req = scheme.statistics([_Pt("l.mlp", "x_after_norm")])
    assert all(ps.stat is None for ps in req.values())   # zero calibration
    decision = scheme.decide({
        "l.mlp": {"l.mlp|x_after_norm": None,
                  "l.mlp.down_proj|act_after_mul": None},
        "l.self_attn": {"l.self_attn.q_proj|x": None},
    })
    assert decision.formats == {"l.mlp": "w8a16_static"}
    assert decision.keep_host == ("l.self_attn",)


def test_scheme_subclass_owns_only_two_methods():
    class Imatrix(QuantScheme):
        name = "imatrix_demo"

        def statistics(self, points):
            return {f"{p.path}|{p.name}": PointStat("second_moment",
                                                    "channel")
                    for p in points}

    schemes.register("imatrix_demo", Imatrix())
    try:
        got = schemes.get("imatrix_demo")
        # the imatrix statistic (per-channel second moment) is now
        # measurable, so the demo's request validates
        validate_request(got.statistics([_Pt("a", "x")]))
    finally:
        schemes._REGISTRY.pop("imatrix_demo", None)
