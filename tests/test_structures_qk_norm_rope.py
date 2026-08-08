"""Catalog boundary tests for the cross-family Q/K postprocess region."""

from __future__ import annotations

import torch

from flash_rt.structures.registry import load


def _inputs(*, q_heads=4, kv_heads=2, head_dim=8, tokens=5):
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(2, tokens, q_heads, head_dim, generator=gen)
    k = torch.randn(2, tokens, kv_heads, head_dim, generator=gen)
    v = torch.randn(2, tokens, kv_heads, head_dim, generator=gen)
    qw = torch.randn(head_dim, generator=gen)
    kw = torch.randn(head_dim, generator=gen)
    theta = torch.randn(2, tokens, head_dim // 2, generator=gen)
    return q, k, v, qw, kw, theta.cos(), theta.sin()


def test_qk_norm_rope_catalog_owns_math_but_not_cache_state():
    spec = load("qk_norm_rope")

    assert spec.kind == "region"
    assert spec.version == 1
    assert spec.weight_slots == ("q_norm_weight", "k_norm_weight")
    assert tuple(spec.calibration["points"]) == (
        "q_after_rope",
        "k_after_rope",
    )
    assert "cache" not in {
        item["name"]
        for side in ("inputs", "outputs")
        for item in spec.boundary[side]
    }


def test_half_rotation_matches_direct_float_reference_and_preserves_v():
    q, k, v, qw, kw, cos, sin = _inputs()
    ref = load("qk_norm_rope").reference()
    q_out, k_out, v_out = ref(q, k, v, qw, kw, cos, sin)

    def direct(x, w):
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        cf = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sf = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        return x * cf + rotated * sf

    torch.testing.assert_close(q_out, direct(q, qw))
    torch.testing.assert_close(k_out, direct(k, kw))
    assert v_out is v


def test_interleaved_partial_rope_keeps_unrotated_suffix_and_gqa_shape():
    q, k, v, qw, kw, cos, sin = _inputs(head_dim=12)
    cos, sin = cos[..., :4], sin[..., :4]
    ref = load("qk_norm_rope").reference()
    q_out, k_out, _ = ref(
        q,
        k,
        v,
        qw,
        kw,
        cos,
        sin,
        rotary_dim=8,
        variant={"rope_layout": "interleaved"},
    )

    qn = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6) * qw
    kn = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + 1e-6) * kw
    torch.testing.assert_close(q_out[..., 8:], qn[..., 8:])
    torch.testing.assert_close(k_out[..., 8:], kn[..., 8:])
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert q_out.shape[-2] != k_out.shape[-2]


def test_projection_scope_matches_wan_pre_reshape_rmsnorm():
    q, k, v, _, _, cos, sin = _inputs()
    gen = torch.Generator().manual_seed(11)
    qw = torch.randn(q.shape[-2] * q.shape[-1], generator=gen)
    kw = torch.randn(k.shape[-2] * k.shape[-1], generator=gen)
    ref = load("qk_norm_rope").reference()
    q_out, k_out, _ = ref(
        q,
        k,
        v,
        qw,
        kw,
        cos,
        sin,
        variant={"normalization_scope": "projection"},
    )

    def wan_norm(x, weight):
        flat = x.flatten(-2)
        flat = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-6)
        return (flat * weight).view_as(x)

    qn = wan_norm(q, qw)
    kn = wan_norm(k, kw)
    half = q.shape[-1] // 2
    qrot = torch.cat((-qn[..., half:], qn[..., :half]), dim=-1)
    krot = torch.cat((-kn[..., half:], kn[..., :half]), dim=-1)
    cf = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sf = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    torch.testing.assert_close(q_out, qn * cf + qrot * sf)
    torch.testing.assert_close(k_out, kn * cf + krot * sf)


def test_preexpanded_position_table_makes_geometry_source_irrelevant():
    q, k, v, qw, kw, cos, sin = _inputs()
    ref = load("qk_norm_rope").reference()

    from_1d = ref(q, k, v, qw, kw, cos, sin)
    from_3d = ref(q, k, v, qw, kw, cos.clone(), sin.clone())

    torch.testing.assert_close(from_1d[0], from_3d[0])
    torch.testing.assert_close(from_1d[1], from_3d[1])
