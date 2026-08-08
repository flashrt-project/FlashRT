from __future__ import annotations

import torch
from torch import nn

from flash_rt.structures.adapters.transformers_gated_delta import (
    TransformersGatedDeltaAdapter,
)
from flash_rt.structures.catalog.gated_delta_core.reference import (
    gated_delta_core_ref,
)
from flash_rt.structures.registry import load


def test_gated_delta_reference_carries_state_across_calls():
    torch.manual_seed(4)
    q = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 3, 2, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    whole, state = gated_delta_core_ref(q, k, v, g, beta)
    first, state1 = gated_delta_core_ref(
        q[:, :2], k[:, :2], v[:, :2], g[:, :2], beta[:, :2])
    last, state2 = gated_delta_core_ref(
        q[:, 2:], k[:, 2:], v[:, 2:], g[:, 2:], beta[:, 2:], state1)
    # The public state boundary is BF16, so splitting the call introduces one
    # intentional state cast that a single sequence call does not.
    torch.testing.assert_close(
        torch.cat((first, last), dim=1), whole, rtol=5e-2, atol=1e-4)
    torch.testing.assert_close(state2, state, rtol=5e-2, atol=1e-4)


def test_gated_delta_catalog_declares_state_boundary():
    spec = load("gated_delta_core")
    inputs = {entry["name"] for entry in spec.boundary["inputs"]}
    outputs = {entry["name"] for entry in spec.boundary["outputs"]}
    assert {"q", "k", "v", "log_decay", "beta", "state"} <= inputs
    assert {"out", "final_state"} <= outputs


class _HostCore:
    def __call__(
        self, query, key, value, g, beta, *, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True, **kwargs,
    ):
        del kwargs
        out, state = gated_delta_core_ref(
            query, key, value, g, beta, initial_state,
            qk_l2norm=use_qk_l2norm_in_kernel)
        return out, state if output_final_state else None


class _FakeLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_v_heads = 48
        self.head_k_dim = 128
        self.head_v_dim = 128
        self.recurrent_gated_delta_rule = _HostCore()
        self.chunk_gated_delta_rule = _HostCore()

    def forward(self, q, k, v, g, beta, state=None):
        return self.recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)


class _BoundCore(nn.Module):
    def forward(
        self, query, key, value, g, beta, state, *,
        output_final_state, use_qk_l2norm,
    ):
        out, state = gated_delta_core_ref(
            query, key, value, g, beta, state,
            qk_l2norm=use_qk_l2norm)
        return out, state if output_final_state else None


def test_transformers_gated_delta_adapter_routes_and_reverts(monkeypatch):
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    q = torch.randn(1, 1, 48, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 1, 48, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.zeros(1, 48, 128, 128, dtype=torch.bfloat16)
    expected = host.linear_attn(q, k, v, g, beta, initial_state)
    original = host.linear_attn.recurrent_gated_delta_rule

    monkeypatch.setitem(
        TransformersGatedDeltaAdapter.__call__.__globals__,
        "bind_gated_delta_core", lambda row: _BoundCore())
    result = TransformersGatedDeltaAdapter()(
        host, lambda: host.linear_attn(q, k, v, g, beta, initial_state))
    assert result is not None and result["observed"]
    got = host.linear_attn(q, k, v, g, beta, initial_state)
    torch.testing.assert_close(got[0], expected[0])
    torch.testing.assert_close(got[1], expected[1])
    result["revert"][0]()
    assert host.linear_attn.recurrent_gated_delta_rule is original


def test_transformers_gated_delta_adapter_rejects_other_head_shapes():
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    host.linear_attn.num_v_heads = 8
    assert TransformersGatedDeltaAdapter()(host, lambda: None) is None


def test_transformers_gated_delta_adapter_accepts_second_head_profile(
        monkeypatch):
    host = nn.ModuleDict({"linear_attn": _FakeLinearAttention()}).eval()
    host.linear_attn.num_v_heads = 32
    q = torch.randn(1, 1, 32, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = -torch.rand(1, 1, 32, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn_like(g))
    initial_state = torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16)
    monkeypatch.setitem(
        TransformersGatedDeltaAdapter.__call__.__globals__,
        "bind_gated_delta_core", lambda row: _BoundCore())

    result = TransformersGatedDeltaAdapter()(
        host, lambda: host.linear_attn(q, k, v, g, beta, initial_state))

    assert result is not None and result["observed"]
    out, state = host.linear_attn(q, k, v, g, beta, initial_state)
    assert out.shape == q.shape
    assert state.shape == (1, 32, 128, 128)


def test_hub_v3_binds_the_log_decay_dtype_the_host_exposes(monkeypatch):
    # the 27B-class cached-decode hosts keep g in FP32; the impl must
    # bind the entry matching the observed dtype, refuse builds that
    # predate the FP32 entry, and hold the bound dtype at call time
    import torch

    from flash_rt.structures.impls.gated_delta_core import hub_v3

    calls = {}

    class _Ops:
        def gated_delta_recurrent_inout_bf16(self, *a, **kw):
            calls["entry"] = "bf16"
            return kw["out"], kw["state_out"]

        def gated_delta_recurrent_inout_gf32_bf16(self, *a, **kw):
            calls["entry"] = "gf32"
            return kw["out"], kw["state_out"]

    monkeypatch.setattr(hub_v3, "hub_kernel", lambda repo, ver: _Ops())
    q = torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)
    sample = {"query": q, "key": q.clone(), "value": q.clone(),
              "g": torch.zeros(1, 1, 32, dtype=torch.float32),
              "beta": torch.zeros(1, 1, 32, dtype=torch.bfloat16),
              "state": torch.zeros(1, 32, 128, 128, dtype=torch.bfloat16),
              "output_final_state": True, "use_qk_l2norm": True}
    core = hub_v3.bind_gated_delta_core(sample)
    assert calls["entry"] == "gf32"

    # a bf16-g host still binds the original entry
    sample_bf = dict(sample, g=sample["g"].to(torch.bfloat16))
    hub_v3.bind_gated_delta_core(sample_bf)
    assert calls["entry"] == "bf16"

    # the bound dtype is a contract: a host that changes g dtype after
    # binding is refused, not silently served
    import pytest as _pytest
    with _pytest.raises(ValueError, match="dtypes moved"):
        core(q, q, q, sample["g"].to(torch.bfloat16), sample["beta"],
             sample["state"], output_final_state=True, use_qk_l2norm=True)


def test_hub_v3_refuses_builds_without_the_fp32_entry(monkeypatch):
    import torch
    import pytest as _pytest

    from flash_rt.structures.impls.gated_delta_core import hub_v3

    class _OldOps:
        def gated_delta_recurrent_inout_bf16(self, *a, **kw):
            return kw["out"], kw["state_out"]

    monkeypatch.setattr(hub_v3, "hub_kernel", lambda repo, ver: _OldOps())
    q = torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)
    with _pytest.raises(ValueError, match="predates the .*gf32"):
        hub_v3.HubV3GatedDeltaCore(q, g_dtype=torch.float32)


def test_fused_adapter_recognises_by_shape_and_ladders_cleanly(monkeypatch):
    # the fused-layer adapter is shape-recognised (no class names) and
    # steps aside — for the callable-slot ladder — when the installed
    # packages predate the chain entries
    import torch
    import pytest as _pytest
    from torch import nn

    from flash_rt.structures.adapters.transformers_gated_delta_fused import (
        TransformersGatedDeltaFusedAdapter, _fusable, _layer_index)
    from flash_rt.structures.impls.gated_delta_core import fused_layer

    class _Gdn(nn.Module):
        def __init__(self, profile=True):
            super().__init__()
            d = 5120
            self.in_proj_qkv = nn.Linear(d, 10240, bias=False)
            self.in_proj_z = nn.Linear(d, 6144, bias=False)
            self.in_proj_b = nn.Linear(d, 48, bias=False)
            self.in_proj_a = nn.Linear(d, 48, bias=False)
            self.out_proj = nn.Linear(6144, d, bias=False)
            self.conv1d = nn.Conv1d(10240, 10240, 4, groups=10240)
            self.A_log = nn.Parameter(torch.zeros(48))
            self.dt_bias = nn.Parameter(torch.zeros(48))
            self.norm = nn.RMSNorm(128)
            # profile=True is the original 48/16 host; "h32" is the
            # 32/16 MoE host the head-generic entries serve; False is
            # a genuinely unservable profile (v-heads not a multiple
            # of k-heads)
            self.num_v_heads = {True: 48, "h32": 32, False: 40}[profile]
            self.num_k_heads = 16
            self.head_k_dim = 128
            self.head_v_dim = 128

    assert _fusable(_Gdn())
    assert _fusable(_Gdn(profile="h32"))       # head-generic profile
    assert not _fusable(_Gdn(profile=False))   # out of the fused profile
    assert not _fusable(nn.Linear(8, 8))
    assert _layer_index("model.layers.7.linear_attn") == 7
    assert _layer_index("model.embed") is None

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].linear_attn = _Gdn()

    class _OldPkg:
        pass

    fused_layer._packages.cache_clear()
    monkeypatch.setattr(
        "flash_rt.structures.impls.hub_kernel",
        lambda repo, ver: _OldPkg())
    with _pytest.raises(ValueError, match="lacks"):
        TransformersGatedDeltaFusedAdapter()(_Model(), lambda: None)
    fused_layer._packages.cache_clear()


def test_w4a4_scheme_routes_the_gdn_projection_band():
    from flash_rt.structures import schemes
    from flash_rt.structures.adapters.transformers_gated_delta_fused \
        import TransformersGatedDeltaFusedAdapter

    assert schemes.QuantScheme.gdn_projection_format is None
    assert schemes.get("none").gdn_projection_format is None
    assert (schemes.get("w4a4_decode").gdn_projection_format
            == "nvfp4_dynamic")
    # the fused adapter declares scheme awareness so autobuild hands
    # the active scheme through; older adapters keep the two-arg call
    assert TransformersGatedDeltaFusedAdapter.scheme_aware is True
