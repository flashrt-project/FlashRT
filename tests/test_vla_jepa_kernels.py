"""Numerical checks for the VLA-JEPA model-specific kernels.

Covers the boundary/post-processing helpers added with the model. The tests
skip cleanly when the VLA-JEPA object was not built
(FLASHRT_ENABLE_VLA_JEPA=OFF) or when no CUDA device is present.

Run:
    PYTHONPATH=. python -m pytest tests/test_vla_jepa_kernels.py -q
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _fvk_or_skip():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:                  # pragma: no cover
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    if not hasattr(fvk, "vla_action_postprocess"):
        pytest.skip("VLA-JEPA kernels not built "
                    "(FLASHRT_ENABLE_VLA_JEPA=OFF)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return fvk


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def test_action_postprocess_matches_reference():
    fvk = _fvk_or_skip()
    rows, cols, grip = 7, 7, 6
    thr = 0.5
    gen = torch.Generator(device="cuda").manual_seed(0)
    actions = (torch.rand(rows, cols, generator=gen, device="cuda") * 3 - 1.5
               ).to(torch.bfloat16)
    amin = torch.tensor([-1.0, -0.5, 0.0, 0.1, -0.2, 0.3, -0.9],
                        device="cuda", dtype=torch.float32)
    amax = torch.tensor([1.0, 0.5, 0.0, 0.9, 0.8, 0.7, 0.9],
                        device="cuda", dtype=torch.float32)
    out = torch.empty(rows, cols, device="cuda", dtype=torch.float32)

    fvk.vla_action_postprocess(actions.data_ptr(), out.data_ptr(),
                               amin.data_ptr(), amax.data_ptr(),
                               rows, cols, grip, thr, _stream())
    torch.cuda.synchronize()

    x = actions.float().clamp(-1.0, 1.0)
    x[:, grip] = (x[:, grip] >= thr).float()
    ref = (x + 1.0) / 2.0 * (amax - amin) + amin
    ref[:, grip] = 1.0 - 2.0 * (ref[:, grip] > thr).float()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_state_normalize_matches_reference():
    fvk = _fvk_or_skip()
    n = 8
    gen = torch.Generator(device="cuda").manual_seed(1)
    raw = torch.randn(n, generator=gen, device="cuda") * 2
    mean = torch.randn(n, generator=gen, device="cuda") * 0.1
    std = torch.rand(n, generator=gen, device="cuda") + 0.5
    eps = 1e-6
    out = torch.empty(n, device="cuda", dtype=torch.bfloat16)

    fvk.vla_state_normalize(raw.data_ptr(), mean.data_ptr(), std.data_ptr(),
                            eps, n, out.data_ptr(), _stream())
    torch.cuda.synchronize()

    ref = ((raw - mean) / (std + eps)).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-3)


def test_fa2_qnvfp4_matches_standalone_quantizer():
    """The FA2 tile64 qnvfp4 epilogue must emit the same operand bytes as the
    bf16 tile output followed by the standalone NVFP4 quantizer."""
    try:
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_fa2 as fa2
    except ImportError as exc:                  # pragma: no cover
        pytest.skip(f"FA2/kernel extensions not built: {exc}")
    if not hasattr(fa2, "fwd_bf16_tile64m32w2_qnvfp4"):
        pytest.skip("FA2 tile64 qnvfp4 entry not built")
    if not hasattr(fa2, "fwd_bf16_tile64m32w2"):
        pytest.skip("FA2 tile64m32w2 entry not built")
    if hasattr(fa2, "has_xvla_tile64") and not fa2.has_xvla_tile64():
        pytest.skip("FA2 tile64 objects not compiled")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    Lq, Hq, hd = 40, 12, 64
    width = Hq * hd
    torch.manual_seed(3)
    q = torch.randn(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(Hq, Lq, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    strides = (Lq * width, width, hd)
    scale = 1.0 / (hd ** 0.5)

    fa2.fwd_bf16_tile64m32w2(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(),
        softmax_lse=lse.data_ptr(), seqused_k=0,
        batch=1, seqlen_q=Lq, seqlen_k=Lq, num_heads_q=Hq,
        num_heads_kv=Hq, head_dim=hd,
        q_strides=strides, k_strides=strides, v_strides=strides,
        o_strides=strides, softmax_scale=scale, num_sms=0,
        stream=stream, enable_pdl=False)

    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(Lq, width)
    fp4_ref = torch.empty(Lq * width // 2, device="cuda", dtype=torch.uint8)
    sf_ref = torch.zeros(sf_bytes, device="cuda", dtype=torch.uint8)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        o.data_ptr(), fp4_ref.data_ptr(), sf_ref.data_ptr(), Lq, width, stream)

    fp4_out = torch.empty_like(fp4_ref)
    sf_out = torch.zeros_like(sf_ref)
    o2 = torch.empty_like(o)
    lse2 = torch.empty_like(lse)
    fa2.fwd_bf16_tile64m32w2_qnvfp4(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o2.data_ptr(),
        softmax_lse=lse2.data_ptr(), seqused_k=0,
        o_packed=fp4_out.data_ptr(), o_sfa=sf_out.data_ptr(),
        o_packed_row_bytes=width // 2,
        o_sfa_n_col_blocks=(width // 16 + 3) // 4,
        batch=1, seqlen_q=Lq, seqlen_k=Lq, num_heads_q=Hq,
        num_heads_kv=Hq, head_dim=hd,
        q_strides=strides, k_strides=strides, v_strides=strides,
        o_strides=strides, softmax_scale=scale, num_sms=0,
        stream=stream, enable_pdl=False)
    torch.cuda.synchronize()

    assert torch.equal(fp4_out, fp4_ref), "FA2 packed FP4 operand differs"
    assert torch.equal(sf_out, sf_ref), "FA2 swizzled SF differs"


def test_dit_step_rows_matches_reference():
    fvk = _fvk_or_skip()
    head_rows, act_rows, dim = 3, 4, 16
    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(head_rows + act_rows, dim, generator=gen,
                    device="cuda").to(torch.bfloat16)
    head = torch.randn(head_rows, dim, generator=gen,
                       device="cuda").to(torch.bfloat16)
    pos = torch.randn(act_rows, dim, generator=gen,
                      device="cuda").to(torch.bfloat16)
    x0 = x.clone()

    rc = fvk.vla_dit_step_rows(x.data_ptr(), head.data_ptr(), pos.data_ptr(),
                               head_rows, act_rows, dim, _stream())
    assert rc == 0
    torch.cuda.synchronize()

    # bf16 add written as bf16(bf16(a) + bf16(b)) in the reference.
    ref = x0.clone()
    ref[:head_rows] = head
    ref[head_rows:] = (x0[head_rows:].float() + pos.float()).to(torch.bfloat16)
    torch.testing.assert_close(x, ref, rtol=0, atol=0)


def test_fa2_qnvfp4_batched_matches_per_view():
    """The additive batched NVFP4 epilogue must emit the same bytes as the
    original one-call-per-view path (the delivered VLA ViT route)."""
    try:
        from flash_rt import flash_rt_fa2 as fa2
    except ImportError as exc:                  # pragma: no cover
        pytest.skip(f"FA2 not built: {exc}")
    if not hasattr(fa2, "fwd_bf16_tile64_qnvfp4_batched"):
        pytest.skip("batched FA2 qnvfp4 entry not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    nviews, P, nh, hd = 2, 256, 16, 64
    H = nh * hd
    ncb = (H // 16 + 3) // 4
    torch.manual_seed(5)
    q = torch.randn(nviews, P, nh, hd, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.zeros_like(q)
    lse = torch.zeros(nviews, nh, P, device="cuda", dtype=torch.float32)
    pak_ref = torch.zeros(nviews * P * (H // 2), device="cuda",
                          dtype=torch.uint8)
    sf_ref = torch.zeros(nviews * (P // 128) * ncb * 512, device="cuda",
                         dtype=torch.uint8)
    pak_out = torch.zeros_like(pak_ref)
    sf_out = torch.zeros_like(sf_ref)
    stream = torch.cuda.current_stream().cuda_stream
    row_q = P * H
    strides = (row_q, H, hd)
    scale = hd ** -0.5
    for vi in range(nviews):
        fa2.fwd_bf16_tile64_qnvfp4(
            Q=q.data_ptr() + vi * row_q * 2,
            K=k.data_ptr() + vi * row_q * 2,
            V=v.data_ptr() + vi * row_q * 2,
            O=o.data_ptr() + vi * row_q * 2,
            softmax_lse=lse.data_ptr() + vi * nh * P * 4, seqused_k=0,
            o_packed=pak_ref.data_ptr() + vi * P * (H // 2),
            o_sfa=sf_ref.data_ptr() + vi * (P // 128) * ncb * 512,
            o_packed_row_bytes=H // 2, o_sfa_n_col_blocks=ncb,
            batch=1, seqlen_q=P, seqlen_k=P, num_heads_q=nh,
            num_heads_kv=nh, head_dim=hd,
            q_strides=strides, k_strides=strides, v_strides=strides,
            o_strides=strides, softmax_scale=scale, num_sms=0,
            stream=stream, enable_pdl=False)
    fa2.fwd_bf16_tile64_qnvfp4_batched(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(),
        softmax_lse=lse.data_ptr(), seqused_k=0,
        o_packed=pak_out.data_ptr(), o_sfa=sf_out.data_ptr(),
        o_packed_row_bytes=H // 2, o_sfa_n_col_blocks=ncb,
        o_packed_batch_stride=P * (H // 2),
        o_sfa_batch_stride=(P // 128) * ncb * 512,
        batch=nviews, seqlen_q=P, seqlen_k=P, num_heads_q=nh,
        num_heads_kv=nh, head_dim=hd,
        q_strides=strides, k_strides=strides, v_strides=strides,
        o_strides=strides, softmax_scale=scale, num_sms=0,
        stream=stream, enable_pdl=False)
    torch.cuda.synchronize()
    assert torch.equal(pak_out, pak_ref), "batched packed FP4 differs"
    assert torch.equal(sf_out, sf_ref), "batched swizzled SF differs"


def test_residual_add_tail_is_opt_in():
    """The odd-tail fix is opt-in: the original binding keeps its old
    behaviour, and the VLA Euler update passes include_tail=True."""
    fvk = _fvk_or_skip()
    n = 49
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(n, generator=gen, device="cuda").to(torch.bfloat16)
    base = torch.randn(n, generator=gen, device="cuda").to(torch.bfloat16)
    old = base.clone()
    fvk.residual_add(old.data_ptr(), x.data_ptr(), n, _stream())
    new = base.clone()
    fvk.residual_add(new.data_ptr(), x.data_ptr(), n, _stream(),
                     include_tail=True)
    torch.cuda.synchronize()
    # The vectorized body covers the first 48 elements in both cases.
    assert torch.equal(old[:48], new[:48])
    # Default path leaves the odd tail untouched; the opt-in updates it.
    assert torch.equal(old[48], base[48])
    ref_tail = (base[48].float() + x[48].float()).to(torch.bfloat16)
    assert torch.equal(new[48], ref_tail)


def test_fa2_smallq_fp4out_matches_standalone_quantizer():
    """Target-validated smallq fp4out epilogue must equal the bf16 tile
    output followed by the standalone quantizer (batch > 1 included)."""
    try:
        from flash_rt import flash_rt_fa2 as fa2
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:                  # pragma: no cover
        pytest.skip(f"FA2/kernels not built: {exc}")
    if not hasattr(fa2, "fwd_bf16_tile_fp4out"):
        pytest.skip("target smallq fp4out entry not built")
    if not hasattr(fa2, "fwd_bf16_tile_vla"):
        pytest.skip("VLA smallq bf16 entry not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    B, P, nh, hd = 2, 256, 16, 64
    H = nh * hd
    torch.manual_seed(9)
    q = torch.randn(B, P, nh, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.empty_like(q)
    lse = torch.empty(B, nh, P, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream
    strides = (P * H, H, hd)
    scale = hd ** -0.5
    fa2.fwd_bf16_tile_vla(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(),
        softmax_lse=lse.data_ptr(), seqused_k=0,
        batch=B, seqlen_q=P, seqlen_k=P, num_heads_q=nh, num_heads_kv=nh,
        head_dim=hd, q_strides=strides, k_strides=strides,
        v_strides=strides, o_strides=strides, softmax_scale=scale,
        stream=stream)
    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(B * P, H)
    fp4_ref = torch.empty(B * P * H // 2, device="cuda", dtype=torch.uint8)
    sf_ref = torch.zeros(sf_bytes, device="cuda", dtype=torch.uint8)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        o.data_ptr(), fp4_ref.data_ptr(), sf_ref.data_ptr(), B * P, H, stream)
    fp4_out = torch.empty_like(fp4_ref)
    sf_out = torch.zeros_like(sf_ref)
    fa2.fwd_bf16_tile_fp4out(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
        O_fp4=fp4_out.data_ptr(), O_sf=sf_out.data_ptr(),
        softmax_lse=lse.data_ptr(),
        batch=B, seqlen_q=P, seqlen_k=P, num_heads_q=nh, num_heads_kv=nh,
        head_dim=hd, q_strides=strides, k_strides=strides,
        v_strides=strides, o_strides=strides, softmax_scale=scale,
        stream=stream)
    torch.cuda.synchronize()
    assert torch.equal(fp4_out, fp4_ref), "smallq packed FP4 differs"
    assert torch.equal(sf_out, sf_ref), "smallq swizzled SF differs"


def test_w4a16_k128_matches_shared_tile():
    """The model-scoped K128 twins must be bit-identical to the shared tile.

    The shared entry keeps the upstream 128x128x256 tile; VLA-JEPA selects
    the additive *_k128 entries. Same operands and epilogues -> same bytes.
    """
    fvk = _fvk_or_skip()
    if not hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_k128"):
        pytest.skip("VLA K128 GEMM entries not built")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = "cuda"
    stream = _stream()

    def pack_weight(w):
        n, k = w.shape
        gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12)
        gs = gs.reshape(1).contiguous()
        q = torch.empty(n, k // 2, dtype=torch.uint8, device=dev)
        sf = torch.empty(n, k // 16, dtype=torch.uint8, device=dev)
        assert fvk.nvfp4_skinny_quantize_weight(
            w.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(), n, k,
            0, 4) == 0
        swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(n, k),
                          dtype=torch.uint8, device=dev)
        fvk.nvfp4_sf_linear_to_swizzled(sf.data_ptr(), swz.data_ptr(), n, k,
                                        True, 0)
        return q, swz, float(gs.item())

    def quant_act(x):
        m, k = x.shape
        q = torch.empty(m, k // 2, dtype=torch.uint8, device=dev)
        sf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(m, k),
                         dtype=torch.uint8, device=dev)
        fvk.quantize_bf16_to_nvfp4_swizzled_v2(
            x.data_ptr(), q.data_ptr(), sf.data_ptr(), m, k, 0)
        return q, sf

    gen = torch.Generator(device=dev).manual_seed(3)
    m, n, k = 128, 256, 1024
    x = (torch.randn(m, k, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=gen, device=dev) * 0.05).to(torch.bfloat16)
    aq, asf = quant_act(x)
    wq, wsf, alpha = pack_weight(w)
    c = (torch.randn(m, n, generator=gen, device=dev) * 0.1).to(torch.bfloat16)
    bias = (torch.randn(n, generator=gen, device=dev) * 0.1).to(torch.bfloat16)

    def run(fn, out, extra=()):
        fn(aq.data_ptr(), wq.data_ptr(), out.data_ptr(), *extra, m, n, k,
           asf.data_ptr(), wsf.data_ptr(), alpha, stream)
        return out

    base0 = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    base1 = torch.empty_like(base0)
    run(fvk.fp4_w4a16_gemm_sm120_bf16out, base0)
    run(fvk.fp4_w4a16_gemm_sm120_bf16out_k128, base1)
    res0 = torch.empty_like(base0)
    res1 = torch.empty_like(base0)
    fvk.fp4_w4a16_gemm_residual_sm120_bf16out(
        aq.data_ptr(), wq.data_ptr(), c.data_ptr(), res0.data_ptr(), m, n, k,
        asf.data_ptr(), wsf.data_ptr(), alpha, stream)
    fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
        aq.data_ptr(), wq.data_ptr(), c.data_ptr(), res1.data_ptr(), m, n, k,
        asf.data_ptr(), wsf.data_ptr(), alpha, stream)
    bias0 = torch.empty_like(base0)
    bias1 = torch.empty_like(base0)
    run(fvk.fp4_w4a16_gemm_sm120_bf16out_bias, bias0, (bias.data_ptr(),))
    run(fvk.fp4_w4a16_gemm_sm120_bf16out_bias_k128, bias1, (bias.data_ptr(),))
    torch.cuda.synchronize()
    assert torch.equal(base0, base1), "base K128 differs from the shared tile"
    assert torch.equal(res0, res1), "residual K128 differs from the shared tile"
    assert torch.equal(bias0, bias1), "bias K128 differs from the shared tile"
