#!/usr/bin/env python3
"""flash_rt_groot_n15: symbol surface, numerics against torch, shape guards.

The module is built only with -DFLASHRT_ENABLE_GROOT_N15=ON on GPU_ARCH=89;
these tests skip cleanly without it. Projections, embodiment GEMMs and the
elementwise ops must equal the torch ops of the reference bit for bit;
LayerNorm, which reduces in a different order, must agree to an ulp on all
but a vanishing fraction of elements.

Usage:
  PYTHONPATH=. pytest -q tests/test_groot_n15_kernels.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
F = torch.nn.functional
BF16 = torch.bfloat16


def _g():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_groot_n15 as g
    except ImportError as exc:
        pytest.skip(f"flash_rt_groot_n15 not built (FLASHRT_ENABLE_GROOT_N15=OFF?): {exc}")
    return g


def _s():
    return torch.cuda.current_stream().cuda_stream


def _ulp_mismatch(a, b):
    return (a.float() != b.float()).float().mean().item()


def test_default_import_does_not_need_the_module():
    import flash_rt
    import flash_rt.models.groot_n15 as m
    assert flash_rt is not None and m.GrootN15Dims().dit_tokens == 49


def test_published_symbols_match_the_required_list():
    g = _g()
    from flash_rt.models.groot_n15 import GROOT_N15_SYMBOLS
    exported = {n for n in dir(g) if n.startswith("groot_n15_")}
    assert exported == set(GROOT_N15_SYMBOLS)


@pytest.mark.parametrize("M,N,K", [(768, 1152, 1152), (768, 4304, 1152), (768, 1152, 588),
                                   (814, 2048, 8192), (814, 1536, 2048), (300, 2048, 2048),
                                   (49, 6144, 1536), (49, 1024, 1536)])
def test_linear_bias_is_torch_linear(M, N, K):
    g = _g()
    torch.manual_seed(M + N + K)
    x = (torch.randn(M, K, device="cuda") * 0.5).to(BF16)
    W = (torch.randn(N, K, device="cuda") * 0.03).to(BF16)
    b = (torch.randn(N, device="cuda") * 0.1).to(BF16)
    out = torch.empty(M, N, device="cuda", dtype=BF16)
    g.groot_n15_linear_bias_bf16(x.data_ptr(), W.data_ptr(), b.data_ptr(), out.data_ptr(),
                                 M, N, K, _s())
    assert torch.equal(out, F.linear(x[None], W, b)[0])


@pytest.mark.parametrize("M,N,K", [(1, 1024, 64), (1, 1536, 1024), (16, 1536, 32),
                                   (16, 1536, 3072), (49, 1024, 1024), (49, 32, 1024)])
def test_bmm_is_torch_bmm_of_an_embodiment_slice(M, N, K):
    g = _g()
    torch.manual_seed(M * N + K)
    x = (torch.randn(1, M, K, device="cuda") * 0.5).to(BF16)
    W = (torch.randn(32, K, N, device="cuda") * 0.03).to(BF16)
    cat = torch.tensor([31], device="cuda")
    ref = torch.bmm(x, W[cat])[0]
    Wc = W[31].contiguous()
    out = torch.empty(M, N, device="cuda", dtype=BF16)
    g.groot_n15_bmm_bf16(x.data_ptr(), Wc.data_ptr(), out.data_ptr(), M, N, K, _s())
    assert torch.equal(out, ref)


def test_gemm_rejects_null_pointers_and_degenerate_weights():
    g = _g()
    with pytest.raises(RuntimeError, match="null pointer"):
        g.groot_n15_linear_bias_bf16(0, 0, 0, 0, 4, 4, 4, _s())
    x = torch.zeros(4, 1, device="cuda", dtype=BF16)
    with pytest.raises(RuntimeError, match="at least 2"):
        g.groot_n15_linear_bias_bf16(x.data_ptr(), x.data_ptr(), x.data_ptr(), x.data_ptr(),
                                     4, 4, 1, _s())


def _rows(rows, dim, scale=1.0, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(rows, dim, device="cuda") * scale
    return (base + torch.randn(1, dim, device="cuda") * 3 * scale).to(BF16)


def test_layer_norm_affine_with_residual():
    g = _g()
    h, d = _rows(768, 1152), (_rows(768, 1152, 0.3, 1))
    gm = (1 + 0.1 * torch.randn(1152, device="cuda")).to(BF16)
    bt = (0.1 * torch.randn(1152, device="cuda")).to(BF16)
    r = torch.empty_like(h)
    out = torch.empty_like(h)
    g.groot_n15_layer_norm_bf16(h.data_ptr(), d.data_ptr(), r.data_ptr(), gm.data_ptr(),
                                bt.data_ptr(), out.data_ptr(), 768, 1152, 1e-6, _s())
    assert torch.equal(r, h + d)
    ref = F.layer_norm((h + d).float(), (1152,), gm.float(), bt.float(), 1e-6).to(BF16)
    assert _ulp_mismatch(out, ref) < 1e-3
    assert (out.float() - ref.float()).abs().max().item() <= 2 * 2 ** -7 * ref.float().abs().max()


def test_layer_norm_requires_both_affine_tensors():
    g = _g()
    h = _rows(4, 64)
    with pytest.raises(RuntimeError, match="both gamma and beta"):
        g.groot_n15_layer_norm_bf16(h.data_ptr(), 0, 0, h.data_ptr(), 0, h.data_ptr(), 4, 64,
                                    1e-6, _s())
    with pytest.raises(RuntimeError, match="dim must be"):
        big = torch.zeros(1, 4096, device="cuda", dtype=BF16)
        g.groot_n15_layer_norm_bf16(big.data_ptr(), 0, 0, 0, 0, big.data_ptr(), 1, 4096, 1e-6,
                                    _s())


def test_fp32_residual_layer_norm():
    g = _g()
    x = _rows(814, 2048)
    gm = (1 + 0.1 * torch.randn(2048, device="cuda")).to(BF16)
    bt = (0.1 * torch.randn(2048, device="cuda")).to(BF16)
    h32 = torch.empty(814, 2048, device="cuda")
    g.groot_n15_layer_norm_to_f32(x.data_ptr(), gm.data_ptr(), bt.data_ptr(), h32.data_ptr(),
                                  814, 2048, 1e-5, _s())
    ref32 = F.layer_norm(x.float(), (2048,), gm.float(), bt.float(), 1e-5)
    assert torch.allclose(h32, ref32, rtol=1e-5, atol=1e-5)
    d = _rows(814, 2048, 0.3, 2)
    expect = d.float() + h32
    out = torch.empty_like(x)
    g.groot_n15_residual_layer_norm_f32(h32.data_ptr(), d.data_ptr(), gm.data_ptr(),
                                        bt.data_ptr(), out.data_ptr(), 814, 2048, 1e-5, _s())
    assert torch.equal(h32, expect)
    ref = F.layer_norm(expect, (2048,), gm.float(), bt.float(), 1e-5).to(BF16)
    assert _ulp_mismatch(out, ref) < 1e-3
    final = torch.empty_like(x)
    g.groot_n15_residual_add_f32_to_bf16(h32.data_ptr(), d.data_ptr(), final.data_ptr(),
                                         h32.numel(), _s())
    assert torch.equal(final, (d.float() + h32).to(BF16))


def test_ada_layer_norm_rounds_like_the_reference():
    g = _g()
    h = _rows(49, 1536)
    scale = (0.2 * torch.randn(1, 1536, device="cuda")).to(BF16)
    shift = (0.2 * torch.randn(1, 1536, device="cuda")).to(BF16)
    ref = (F.layer_norm(h.float(), (1536,), None, None, 1e-5) * (1 + scale) + shift).to(BF16)
    a1 = (1 + scale).reshape(-1).contiguous()
    out = torch.empty_like(h)
    g.groot_n15_ada_layer_norm_bf16(h.data_ptr(), 0, 0, a1.data_ptr(), shift.data_ptr(),
                                    out.data_ptr(), 49, 1536, 1e-5, _s())
    assert _ulp_mismatch(out, ref) < 1e-3


def test_gelu_tanh_is_torch_gelu():
    g = _g()
    x = (torch.randn(814, 8192, device="cuda") * 3).to(BF16)
    y = x.clone()
    g.groot_n15_gelu_tanh_bf16(y.data_ptr(), y.numel(), _s())
    assert torch.equal(y, F.gelu(x, approximate="tanh"))


@pytest.mark.parametrize("act", [0, 1, 2])
def test_bias_activations(act):
    g = _g()
    x = (torch.randn(16, 1536, device="cuda") * 3).to(BF16)
    b = torch.randn(1536, device="cuda").to(BF16)
    t = x + b
    ref = (t, F.relu(t), t * torch.sigmoid(t))[act]
    out = torch.empty_like(x)
    g.groot_n15_bias_act_bf16(x.data_ptr(), b.data_ptr(), out.data_ptr(), 16, 1536, act, _s())
    assert torch.equal(out, ref)
    with pytest.raises(RuntimeError, match="act must be"):
        g.groot_n15_bias_act_bf16(x.data_ptr(), b.data_ptr(), out.data_ptr(), 16, 1536, 3, _s())


def test_rows_concat_and_euler():
    g = _g()
    x = torch.randn(16, 1536, device="cuda").to(BF16)
    b = torch.randn(1536, device="cuda").to(BF16)
    pos = torch.randn(16, 1536, device="cuda").to(BF16)
    out = torch.empty_like(x)
    g.groot_n15_add_rows_bf16(x.data_ptr(), b.data_ptr(), pos.data_ptr(), out.data_ptr(), 16,
                              1536, 16, _s())
    assert torch.equal(out, (x + b) + pos)
    tail = torch.randn(16, 1536, device="cuda").to(BF16)
    cat = torch.empty(16, 3072, device="cuda", dtype=BF16)
    g.groot_n15_concat_bias_bf16(x.data_ptr(), b.data_ptr(), tail.data_ptr(), cat.data_ptr(), 16,
                                 1536, 1536, _s())
    assert torch.equal(cat, torch.cat([x + b, tail], dim=1))
    noise = torch.randn(16, 32, device="cuda")
    act32 = torch.empty_like(noise)
    act = torch.empty(16, 32, device="cuda", dtype=BF16)
    g.groot_n15_actions_init(noise.data_ptr(), act32.data_ptr(), act.data_ptr(), 512, _s())
    assert torch.equal(act32, noise) and torch.equal(act, noise.to(BF16))
    pred = torch.randn(16, 32, device="cuda").to(BF16)
    g.groot_n15_euler_step(pred.data_ptr(), act32.data_ptr(), act.data_ptr(), 0.25, 512, _s())
    expect = noise + 0.25 * pred
    assert torch.equal(act32, expect) and torch.equal(act, expect.to(BF16))


def test_fp8_row_quantization_and_rescale():
    g = _g()
    x = _rows(814, 2048, 2.0)
    q = torch.empty(814, 2048, device="cuda", dtype=torch.uint8)
    s = torch.empty(814, device="cuda")
    g.groot_n15_quant_rows_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), 814, 2048, _s())
    amax = x.float().abs().amax(dim=1)
    assert torch.allclose(s, amax / 448.0, rtol=1e-6)
    ref = (x.float() / s[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(q, ref)
    d = torch.randn(814, 1536, device="cuda").to(BF16)
    cs = torch.rand(1536, device="cuda") + 0.5
    b = torch.randn(1536, device="cuda").to(BF16)
    out = d.clone()
    g.groot_n15_rescale_rows_bf16(out.data_ptr(), s.data_ptr(), cs.data_ptr(), b.data_ptr(), 814,
                                  1536, _s())
    expect = (d.float() * (s[:, None] * cs[None, :]) + b.float()).to(BF16)
    assert torch.equal(out, expect)
