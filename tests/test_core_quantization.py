"""The shared channel-balance algorithm: one implementation, provable.

``fit_input_channel_balance`` is the single home of the
activation-aware per-input-channel balance the production NVFP4 paths
use. These tests pin three things: the output is bit-identical to the
production formula it was extracted from (transcribed here from the
Pi0.5 Thor FP4 frontend's ``_awq_scale_weight``, operation for
operation), the reparameterisation preserves the mathematics
(``x' @ W'.T == x @ W.T``), and the edges — zero channels, the clamp
bounds, the weight-only fallback — behave as declared.
"""

import pytest
import torch

from flash_rt.core.quantization import fit_input_channel_balance


def _native_reference(W, activation_amax, alpha):
    # transcription of the production formula (Pi0.5 Thor FP4 frontend,
    # `_awq_scale_weight`), kept operation-for-operation so equality
    # below is bit-exact, not approximate
    if activation_amax is not None:
        a = activation_amax.float().clamp(min=1e-6)
    else:
        a = W.abs().amax(dim=0).float().clamp(min=1e-6)
    s = (a / a.mean()).pow(alpha).clamp(min=0.25, max=4.0)
    inv_s = (1.0 / s).to(torch.float16).contiguous()
    W_scaled = (W.float() * s.unsqueeze(0)).to(torch.float16).contiguous()
    return W_scaled, inv_s


def test_bit_identical_to_the_production_formula():
    torch.manual_seed(0)
    W = torch.randn(64, 128, dtype=torch.float16)
    amax = torch.rand(128) * 3
    for alpha in (0.25, 0.5, 0.85):
        want_w, want_inv = _native_reference(W, amax, alpha)
        got_w, got_inv = fit_input_channel_balance(
            W, amax, alpha=alpha, out_dtype=torch.float16)
        assert torch.equal(got_w, want_w)
        assert torch.equal(got_inv, want_inv)


def test_weight_only_fallback_matches_the_production_fallback():
    torch.manual_seed(1)
    W = torch.randn(32, 64, dtype=torch.float16)
    want_w, want_inv = _native_reference(W, None, 0.5)
    got_w, got_inv = fit_input_channel_balance(
        W, None, out_dtype=torch.float16)
    assert torch.equal(got_w, want_w)
    assert torch.equal(got_inv, want_inv)


def test_reparameterisation_preserves_the_mathematics():
    torch.manual_seed(2)
    W = torch.randn(48, 96, dtype=torch.float32)
    x = torch.randn(5, 96, dtype=torch.float32)
    amax = x.abs().amax(dim=0)
    sw, inv = fit_input_channel_balance(W, amax, out_dtype=torch.float32)
    ref = x @ W.t()
    got = (x * inv) @ sw.t()
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_clamp_bounds_and_zero_channels():
    # one dominant channel among 64 near-zero ones: its amax/mean ratio
    # is ~64, so s = ratio**0.5 = 8 before the clamp — both bounds are
    # genuinely exercised (a zero channel lands at the eps floor and
    # clamps low, the dominant one clamps high)
    W = torch.ones(4, 64)
    amax = torch.zeros(64)
    amax[-1] = 1000.0
    sw, inv = fit_input_channel_balance(W, amax, out_dtype=torch.float32)
    s = 1.0 / inv
    assert torch.all(s >= 0.25 - 1e-6) and torch.all(s <= 4.0 + 1e-6)
    assert s[0] == pytest.approx(0.25)
    assert s[-1] == pytest.approx(4.0)


def test_shape_contract_is_enforced():
    with pytest.raises(ValueError, match=r"\[N, K\]"):
        fit_input_channel_balance(torch.zeros(4, 4, 4))
    with pytest.raises(ValueError, match="channels"):
        fit_input_channel_balance(torch.zeros(4, 8), torch.zeros(7))


def test_default_out_dtype_is_the_weights():
    W = torch.randn(8, 16, dtype=torch.bfloat16)
    sw, inv = fit_input_channel_balance(W, torch.rand(16))
    assert sw.dtype == torch.bfloat16 and inv.dtype == torch.bfloat16


def test_nvfp4_awq_scheme_decides_with_the_recipe_payload():
    from flash_rt.structures import schemes
    from flash_rt.structures.schemes import Nvfp4Awq, PointStat, \
        validate_request

    scheme = schemes.get("nvfp4_awq")
    assert isinstance(scheme, Nvfp4Awq)

    class _Pt:
        def __init__(self, path, name):
            self.path, self.name = path, name

    req = scheme.statistics([_Pt("a.mlp", "x_after_norm"),
                             _Pt("a.mlp.down_proj", "act_after_mul")])
    assert all(ps == PointStat("amax", "channel") for ps in req.values())
    validate_request(req)          # the collector measures this today

    report = {
        "layers.0.mlp": {"layers.0.mlp.down_proj|act_after_mul": None},
        "layers.0.self_attn": {"layers.0.self_attn|x": None},
    }
    d = scheme.decide(report)
    assert d.formats == {"layers.0.mlp": "nvfp4_awq"}
    assert d.params["layers.0.mlp"] == {
        "alpha": 0.5, "clamp": [0.25, 4.0], "recipe": "balance"}
    assert d.keep_host == ("layers.0.self_attn",)

    with pytest.raises(ValueError, match="recipe"):
        Nvfp4Awq(recipe="magic")
