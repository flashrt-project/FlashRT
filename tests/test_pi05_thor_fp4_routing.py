"""``load_model`` routing contract for the Pi0.5 Thor NVFP4 / FA4 options.

Covers the public behaviors the Thor NVFP4 tier adds to
``flash_rt.load_model``:

* ``use_fa4`` is an attention-backend choice, independent of ``use_fp4``:
  it reaches both the FP8 frontend and the NVFP4 subclass, and is rejected
  on any other config / framework / hardware;
* ``use_fp4=True`` alone resolves the encoder-only preset, while adding
  ``use_fp4_decoder=True`` resolves the full tier the published latency
  table is measured with — asserted against the benchmark suite's own
  preset table so the two cannot drift apart;
* ``use_fp4_encoder_attn`` / ``use_fp4_siglip_ffn`` require ``use_fp4`` and
  Pi0.5 torch on Thor, and an explicit value overrides the preset.

The frontends are stubbed out, so these run without a GPU or checkpoint.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest

import flash_rt
from flash_rt import api as flash_rt_api


def _load_bench_module():
    """Import the benchmark suite for its published-preset table."""
    path = Path(__file__).with_name("bench_pi05_decoder_fp4_e2e.py")
    spec = importlib.util.spec_from_file_location(
        "_bench_pi05_decoder_fp4_e2e", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = _load_bench_module()
PRESET = BENCH.PUBLIC_API_PRESET


def test_benchmark_defaults_distinguish_reference_from_regression_gate():
    assert BENCH.DEFAULT_WARMUP == 300
    assert BENCH.RESULT_SCHEMA_VERSION == 2
    assert BENCH.README_REFERENCE_P50_MS == {1: 23.01, 2: 27.17, 3: 31.74}
    assert BENCH.REGRESSION_BASELINE_P50_MS == {1: 30.5, 2: 36.3, 3: 42.8}
    assert BENCH.REQUIRED_REGRESSION_MARGIN_MS == 2.0


def test_benchmark_latency_groups_preserve_measurement_order():
    samples_ms = [
        value
        for group in range(10)
        for value in (float(group), float(group + 2))
    ]
    assert BENCH.latency_group_medians(samples_ms) == [
        float(group + 1) for group in range(10)
    ]


def test_benchmark_latency_groups_require_ten_samples():
    with pytest.raises(ValueError, match="at least 10 samples"):
        BENCH.latency_group_medians([1.0] * 9)


# The constructor parameters the Pi0.5 Thor frontends expose. load_model
# feature-detects with inspect.signature, so the stubs must declare them for
# the routing to be exercised at all. test_stub_signatures_match_frontends
# keeps this list honest wherever the real modules are importable.
_FP8_PARAMS = (
    "num_views", "use_cuda_graph", "autotune", "use_fp8",
    "state_prompt_mode", "state_prompt_fixed_max_len", "use_fa4",
)
_FP4_EXTRA_PARAMS = (
    "use_fp4_encoder_ffn", "use_fp4_decoder", "fp4_layers", "use_awq",
    "awq_alpha", "use_p1_split_gu", "encoder_p1_combiner",
    "encoder_down_variant", "encoder_down_x_variant",
    "decoder_gate_up_variant", "use_fp4_encoder_attn",
    "use_fp4_encoder_attn_qkv", "use_fp4_siglip_ffn",
)


def _make_stub(cls_name, params):
    """Build a stub frontend that records the kwargs the router passed."""

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs

    # load_model filters kwargs through inspect.signature(pipe_cls), so give
    # the stub the real parameter names with keyword-only placeholders.
    signature = inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
         inspect.Parameter("checkpoint", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
           for name in params])
    __init__.__signature__ = signature
    return type(cls_name, (), {"__init__": __init__, "_stub_name": cls_name})


def _install_stub_frontends(monkeypatch):
    for mod_name, cls_name, params in (
        ("flash_rt.frontends.torch.pi05_thor", "Pi05TorchFrontendThor",
         _FP8_PARAMS),
        ("flash_rt.frontends.torch.pi05_thor_fp4", "Pi05TorchFrontendThorFP4",
         _FP8_PARAMS + _FP4_EXTRA_PARAMS),
    ):
        module = types.ModuleType(mod_name)
        setattr(module, cls_name, _make_stub(cls_name, params))
        monkeypatch.setitem(sys.modules, mod_name, module)


def _stub_fp4_extension(monkeypatch, *, available: bool = True,
                        has_nvfp4: bool = True):
    """Stand in for the optional NVFP4 extension.

    ``import flash_rt.flash_rt_fp4 as x`` resolves through the parent
    package's attribute once the real submodule has been imported, so patch
    the attribute as well as the ``sys.modules`` entry.
    """
    if not available:
        monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", None)
        monkeypatch.delattr(flash_rt, "flash_rt_fp4", raising=False)
        return
    module = types.ModuleType("flash_rt.flash_rt_fp4")
    module.has_nvfp4 = lambda: has_nvfp4
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", module)
    monkeypatch.setattr(flash_rt, "flash_rt_fp4", module, raising=False)


def _load(monkeypatch, *, config="pi05", framework="torch", hardware="thor",
          **kwargs):
    monkeypatch.setattr(flash_rt_api, "detect_arch", lambda: "thor",
                        raising=False)
    return flash_rt.load_model(
        "/nonexistent/checkpoint",
        framework=framework,
        config=config,
        hardware=hardware,
        **kwargs,
    )


def _built(monkeypatch, **kwargs):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch)
    model = _load(monkeypatch, **kwargs)
    inner = getattr(model, "_pipe", model)
    return type(inner)._stub_name, inner.kwargs


# ── use_fa4 contract ────────────────────────────────────────────────────


def test_fa4_reaches_the_fp8_frontend_without_fp4(monkeypatch):
    """FP8 + FA4 is a supported combination, not an accident.

    It is the baseline the published NVFP4 speedups are measured against,
    so load_model must forward it rather than require use_fp4.
    """
    name, kwargs = _built(monkeypatch, use_fa4=True)
    assert name == "Pi05TorchFrontendThor"
    assert kwargs["use_fa4"] is True


def test_fa4_reaches_the_fp4_frontend(monkeypatch):
    name, kwargs = _built(monkeypatch, use_fp4=True, use_fa4=True)
    assert name == "Pi05TorchFrontendThorFP4"
    assert kwargs["use_fa4"] is True


def test_fa4_is_not_passed_when_not_requested(monkeypatch):
    _, kwargs = _built(monkeypatch)
    assert "use_fa4" not in kwargs


@pytest.mark.parametrize("where", ["config", "framework", "hardware"])
def test_fa4_is_rejected_off_pi05_torch_thor(monkeypatch, where):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch)
    overrides = {
        "config": {"config": "groot_n17", "use_fp8": True},
        "framework": {"framework": "jax"},
        "hardware": {"hardware": "rtx_sm120"},
    }[where]
    with pytest.raises(ValueError, match="use_fa4"):
        _load(monkeypatch, use_fa4=True, **overrides)


# ── NVFP4 presets ───────────────────────────────────────────────────────


def test_published_tier_matches_the_benchmark_preset(monkeypatch):
    """load_model must produce exactly the benchmarked configuration.

    The expectations are derived from the benchmark suite's own preset
    table, so a change to either side fails here instead of silently
    publishing latencies the public API cannot reproduce.
    """
    name, kwargs = _built(
        monkeypatch, use_fp4=True, use_fp4_decoder=True, use_fa4=True)
    assert name == "Pi05TorchFrontendThorFP4"
    expected = {
        "use_fp4_encoder_ffn": True,
        "fp4_layers": tuple(range(PRESET["encoder_fp4_layer_count"])),
        "use_fp4_decoder": True,
        "use_fp4_encoder_attn": bool(PRESET["encoder_attn_o_fp4"]),
        "use_fp4_siglip_ffn": bool(PRESET["siglip_ffn_fp4"]),
        "use_awq": True,
        "awq_alpha": PRESET["awq_alpha"],
        "use_p1_split_gu": PRESET["encoder_gu_mode"] == "p1",
        "encoder_p1_combiner": PRESET["encoder_p1_combiner"],
        "encoder_down_variant": PRESET["encoder_down_variant"],
        "decoder_gate_up_variant": PRESET["decoder_gate_up_variant"],
        "use_fa4": True,
    }
    missing = {k: (kwargs.get(k), v) for k, v in expected.items()
               if kwargs.get(k) != v}
    assert not missing, f"load_model deviates from the published preset: {missing}"


def test_encoder_only_preset_is_unchanged(monkeypatch):
    """use_fp4=True alone keeps the pre-existing encoder-only defaults."""
    _, kwargs = _built(monkeypatch, use_fp4=True)
    assert kwargs["use_fp4_encoder_attn"] is False
    assert kwargs["use_fp4_siglip_ffn"] is False
    assert kwargs["encoder_p1_combiner"] == "lut_native"
    assert kwargs["awq_alpha"] == 0.5
    assert "use_fp4_decoder" not in kwargs


def test_explicit_values_override_the_preset(monkeypatch):
    _, kwargs = _built(
        monkeypatch, use_fp4=True, use_fp4_decoder=True,
        use_fp4_siglip_ffn=False, encoder_p1_combiner="lut_native")
    assert kwargs["use_fp4_siglip_ffn"] is False
    assert kwargs["use_fp4_encoder_attn"] is True   # still from the preset
    assert kwargs["encoder_p1_combiner"] == "lut_native"


@pytest.mark.parametrize(
    "flag", ["use_fp4_decoder", "use_fp4_encoder_attn", "use_fp4_siglip_ffn"])
def test_fp4_subflags_require_use_fp4(monkeypatch, flag):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch)
    with pytest.raises(ValueError, match=f"{flag}=True requires use_fp4=True"):
        _load(monkeypatch, **{flag: True})


@pytest.mark.parametrize(
    "flag", ["use_fp4_decoder", "use_fp4_encoder_attn", "use_fp4_siglip_ffn"])
def test_fp4_subflags_are_rejected_off_thor(monkeypatch, flag):
    _install_stub_frontends(monkeypatch)
    _stub_fp4_extension(monkeypatch)
    with pytest.raises(ValueError, match=flag):
        _load(monkeypatch, hardware="rtx_sm120", use_fp4=True,
              **{flag: True})


# ── stub fidelity ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mod_name, cls_name, params",
    [("flash_rt.frontends.torch.pi05_thor", "Pi05TorchFrontendThor",
      _FP8_PARAMS),
     ("flash_rt.frontends.torch.pi05_thor_fp4", "Pi05TorchFrontendThorFP4",
      _FP8_PARAMS + _FP4_EXTRA_PARAMS)])
def test_stub_signatures_match_frontends(mod_name, cls_name, params):
    """The stubs only prove routing if they mirror the real signatures."""
    module = pytest.importorskip(
        mod_name, reason="Pi0.5 Thor frontend needs the compiled extensions")
    real = set(inspect.signature(getattr(module, cls_name)).parameters)
    assert set(params) <= real, f"stub declares parameters {cls_name} lacks"


@pytest.mark.parametrize(
    "knob, param",
    [("encoder_attn_qkv_fp4", "use_fp4_encoder_attn_qkv"),
     ("encoder_down_x_variant", "encoder_down_x_variant"),
     ("decoder_weight_format", "decoder_weight_format"),
     ("decoder_act_format", "decoder_act_format"),
     ("decoder_rht", "decoder_rht"),
     ("decoder_fused_attn", "decoder_fused_attn"),
     ("decoder_fused_geglu", "decoder_fused_geglu")])
def test_frontend_defaults_cover_the_unexposed_preset_knobs(knob, param):
    """Knobs load_model does not forward must default to the published value.

    The benchmark preset lists them, the public API leaves them alone, so
    the frontend default is what reproduces the published latency.
    """
    module = pytest.importorskip(
        "flash_rt.frontends.torch.pi05_thor_fp4",
        reason="Pi0.5 Thor NVFP4 frontend needs the compiled extensions")
    default = inspect.signature(
        module.Pi05TorchFrontendThorFP4).parameters[param].default
    want = PRESET[knob]
    if isinstance(default, bool) or isinstance(want, bool):
        assert bool(default) == bool(want)
    else:
        assert default == want
