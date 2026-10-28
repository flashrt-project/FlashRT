"""LaWAM RTX SM120: dispatch contract and flash_rt_lawam kernel parity (no checkpoint)."""
import pytest

torch = pytest.importorskip("torch")


def test_dispatch_resolves_the_dedicated_frontend():
    from flash_rt.hardware import resolve_pipeline_class
    from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx
    assert resolve_pipeline_class("lawam", "torch", "rtx_sm120") is LawamTorchFrontendRtx


@pytest.mark.parametrize("framework,hardware", [("jax", "rtx_sm120"), ("torch", "thor"), ("torch", "rtx_sm89")])
def test_unsupported_routes_fail_before_extension_loading(framework, hardware):
    from flash_rt import load_model
    with pytest.raises(ValueError, match="LaWAM supports"):
        load_model("checkpoint", config="lawam", framework=framework, hardware=hardware)


def _lawam():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        from flash_rt import flash_rt_lawam
    except ImportError:
        pytest.skip("flash_rt_lawam not built (-DFLASHRT_ENABLE_LAWAM=ON)")
    return flash_rt_lawam


def test_w8a16_skinny_gemm_matches_dequantized_matmul():
    lk = _lawam()
    torch.manual_seed(0)
    M, N, K = 24, 1024, 1024
    a = torch.randn(M, K, device="cuda").bfloat16()
    w = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
    ws = torch.rand(N, device="cuda") * 0.01
    for cfg in range(lk.w8a16_skinny_config_count()):
        kc = lk.w8a16_skinny_config_k_chunk(cfg)
        part = torch.empty(K // kc, M, N, device="cuda")
        assert lk.w8a16_skinny_gemm(a.data_ptr(), w.data_ptr(), ws.data_ptr(), part.data_ptr(),
                                    M, N, K, cfg, False, 0) == 0
        ref = a.float() @ (w.float() * ws[:, None]).t()
        torch.testing.assert_close(part.sum(0), ref, rtol=1e-4, atol=1e-3)


def test_gated_residual_norm_matches_torch():
    lk = _lawam()
    torch.manual_seed(0)
    rows, dim = 7, 1024
    x = torch.randn(rows, dim, device="cuda").half()
    delta, bias, gate = (torch.randn(rows, dim, device="cuda").half(), torch.randn(dim, device="cuda").half(),
                         torch.randn(dim, device="cuda").half())
    scale, shift = torch.randn(dim, device="cuda").half() * 0.1, torch.randn(dim, device="cuda").half()
    out = torch.empty(rows, dim, device="cuda").half()
    x_ref = (x.float() + (delta.float() + bias.float()) * gate.float()).half().float()
    y_ref = torch.nn.functional.layer_norm(x_ref, (dim,), eps=1e-6) * (1 + scale.float()) + shift.float()
    assert lk.gated_residual_norm(x.data_ptr(), delta.data_ptr(), bias.data_ptr(), gate.data_ptr(), 0, 0, 1e-6, 0, 0,
                                  scale.data_ptr(), shift.data_ptr(), 0, 0, out.data_ptr(), 0, rows, dim, 0) == 0
    torch.testing.assert_close(x.float(), x_ref, rtol=0, atol=0)
    torch.testing.assert_close(out.float(), y_ref, rtol=2e-3, atol=2e-2)


def test_preprocess_matches_reference_patchify():
    lk = _lawam()
    torch.manual_seed(0)
    frames = torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8, device="cuda")
    qwen = torch.empty(512, 768, device="cuda").half()
    dino = torch.empty(256, 768, device="cuda").half()
    assert lk.preprocess_u8(frames.data_ptr(), 2, qwen.data_ptr(), dino.data_ptr(), 0) == 0
    x = frames.permute(0, 3, 1, 2).float() / 255.0                        # [views, 3, 256, 256]
    q = (x - 0.5) / 0.5
    # Qwen3-VL merge order: (block row, block col, sub row, sub col), features (c, py, px)
    q = q.view(2, 3, 8, 2, 16, 8, 2, 16).permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(512, 768)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)
    d = ((x[0] - mean) / std).view(3, 16, 16, 16, 16).permute(1, 3, 0, 2, 4).reshape(256, 768)
    torch.testing.assert_close(qwen.float(), q, rtol=0, atol=2e-3)
    torch.testing.assert_close(dino.float(), d, rtol=0, atol=5e-3)


@pytest.mark.parametrize("M,N,K", [(512, 1024, 4096), (219, 4096, 2048), (261, 768, 3072)])
def test_cutlass_f16acc_gemm_matches_cublaslt_plan(M, N, K):
    # cuBLASLt may choose a different FP16 reduction on another SM120 GPU.
    # Check both implementations against FP32, rather than requiring equal bits.
    lk = _lawam()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, N, device="cuda") / K ** 0.5).half()
    ref, out = torch.empty(M, N, device="cuda").half(), torch.empty(M, N, device="cuda").half()
    assert lk.gemm_f16acc(a.data_ptr(), w.data_ptr(), ref.data_ptr(), 0, M, N, K, False, 0) == 0
    split = max(1, lk.gemm_f16acc_split(M, N, K, False, False))
    assert lk.gemm_f16acc_act(a.data_ptr(), w.data_ptr(), out.data_ptr(), 0, M, N, K, 0, split, 0) == 0
    exact = a.float() @ w.float()
    for actual in (out, ref):
        error = actual.float() - exact
        assert (error.norm() / exact.norm()).item() < 3e-3
        assert error.abs().max().item() < 0.04


def test_cutlass_f16acc_erf_gelu_epilogue_matches_bias_gelu():
    lk = _lawam()
    torch.manual_seed(0)
    M, N, K = 256, 4096, 1024
    a = (torch.randn(M, K, device="cuda") * 0.5).half()
    w = (torch.randn(K, N, device="cuda") / K ** 0.5).half()
    b = (torch.randn(N, device="cuda") * 0.1).half()
    ref, out = torch.empty(M, N, device="cuda").half(), torch.empty(M, N, device="cuda").half()
    assert lk.gemm_f16acc(a.data_ptr(), w.data_ptr(), ref.data_ptr(), 0, M, N, K, False, 0) == 0
    assert lk.bias_gelu_erf(ref.data_ptr(), N, b.data_ptr(), 0, 0, ref.data_ptr(), M, N, 0) == 0
    split = max(1, lk.gemm_f16acc_split(M, N, K, False, False))
    assert lk.gemm_f16acc_act(a.data_ptr(), w.data_ptr(), out.data_ptr(), b.data_ptr(), M, N, K, 2, split, 0) == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_cutlass_swiglu_dual_gemm_matches_gateup_and_silu_mul():
    lk = _lawam()
    torch.manual_seed(0)
    M, FF, K = 219, 6144, 2048
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, 2 * FF, device="cuda") / K ** 0.5).half()
    gu, ref = torch.empty(M, 2 * FF, device="cuda").half(), torch.empty(M, FF, device="cuda").half()
    sums, out = torch.empty(M, 2 * FF, device="cuda").half(), torch.empty(M, FF, device="cuda").half()
    assert lk.gemm_f16acc(a.data_ptr(), w.data_ptr(), gu.data_ptr(), 0, M, 2 * FF, K, False, 0) == 0
    assert lk.silu_mul(gu.data_ptr(), gu.data_ptr() + 2 * FF, 2 * FF, ref.data_ptr(), M, FF, 0) == 0
    split = max(1, lk.gemm_f16acc_split(M, 2 * FF, K, False, False))
    assert lk.gemm_f16acc_swiglu(a.data_ptr(), w.data_ptr(), w.data_ptr() + 2 * FF, 2 * FF, sums.data_ptr(),
                                 sums.data_ptr() + 2 * FF, 2 * FF, out.data_ptr(), M, FF, K, split, 0) == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(sums, gu, rtol=0, atol=0)


@pytest.mark.parametrize("slices", [2, 4])
def test_dit_cross_attention_matches_softmax_attention(slices):
    lk = _lawam()
    torch.manual_seed(0)
    rows, heads, hd, keys, splits, ld = 24, 16, 64, 512, 4, 8192
    d = heads * hd
    part = torch.randn(splits, rows, d, device="cuda") * 0.5
    bias = (torch.randn(d, device="cuda") * 0.1).bfloat16()
    one = torch.ones(1, device="cuda")
    kv = torch.randn(keys, ld, device="cuda").bfloat16()
    k, v = kv[:, :d], kv[:, d:2 * d]
    out = torch.empty(rows, d, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(lk.dit_cross_attention_state_floats(rows, heads, slices), device="cuda")
    counters = torch.zeros(heads * 2, dtype=torch.int32, device="cuda")
    assert lk.dit_cross_attention(part.data_ptr(), splits, one.data_ptr(), one.data_ptr(), bias.data_ptr(),
                                  k.data_ptr(), v.data_ptr(), ld, keys, slices, out.data_ptr(), rows, heads, hd,
                                  hd ** -0.5, state.data_ptr(), counters.data_ptr(), False, 0) == 0
    q = (part.sum(0) + bias.float()).view(rows, heads, hd).transpose(0, 1)
    att = torch.softmax(q @ k.float().view(keys, heads, hd).permute(1, 2, 0) * hd ** -0.5, -1)
    ref = (att @ v.float().view(keys, heads, hd).transpose(0, 1)).transpose(0, 1).reshape(rows, d)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=1e-2)
    assert int(counters.abs().sum()) == 0


def test_gemv_nvfp4w_matches_dequantized_weights():
    lk = _lawam()
    from flash_rt.frontends.torch.lawam_rtx import _nvfp4_weight_only
    torch.manual_seed(0)
    N, K = 4096, 1024
    w = torch.randn(N, K, device="cuda") * 0.02
    x, b = torch.randn(K, device="cuda").half(), (torch.randn(N, device="cuda") * 0.1).half()
    w4, sf, gs = _nvfp4_weight_only(w)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    q = torch.stack([lut[(w4 & 15).long()], lut[(w4 >> 4).long()]], -1).view(N, K)
    deq = q * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * gs
    assert ((deq - w).norm() / w.norm()).item() < 0.12
    out = torch.empty(N, dtype=torch.half, device="cuda")
    assert lk.gemv_nvfp4w(x.data_ptr(), w4.data_ptr(), sf.data_ptr(), gs.data_ptr(), b.data_ptr(), out.data_ptr(),
                          N, K, 0) == 0
    torch.testing.assert_close(out.float(), deq @ x.float() + b.float(), rtol=2e-3, atol=2e-3)


def test_gemv_nvfp4w_row_ranges_and_grid_cap_are_bit_identical():
    lk = _lawam()
    from flash_rt.frontends.torch.lawam_rtx import _nvfp4_weight_only
    torch.manual_seed(0)
    N, K, R = 3 * 6144, 1024, 6144
    w4, sf, gs = _nvfp4_weight_only(torch.randn(N, K, device="cuda") * 0.02)
    x, b = torch.randn(K, device="cuda").half(), (torch.randn(N, device="cuda") * 0.1).half()
    ref, out = (torch.empty(N, dtype=torch.half, device="cuda") for _ in range(2))
    assert lk.gemv_nvfp4w(x.data_ptr(), w4.data_ptr(), sf.data_ptr(), gs.data_ptr(), b.data_ptr(), ref.data_ptr(),
                          N, K, 0) == 0
    for r0, cap in ((0, 0), (R, 36), (2 * R, 5)):   # one launch per row range, as the LAM modulation
        assert lk.gemv_nvfp4w(x.data_ptr(), w4[r0:].data_ptr(), sf[r0:].data_ptr(), gs.data_ptr(), b[r0:].data_ptr(),
                              out[r0:].data_ptr(), R, K, 0, cap) == 0
    assert torch.equal(out, ref)


def test_qformer_latent_matches_torch():
    lk = _lawam()
    F = torch.nn.functional
    torch.manual_seed(0)
    nq, vdim, code, heads, ctx, S = 8, 2048, 32, 8, 1024, 40
    r = lambda *s, sc=0.05: torch.randn(*s, device="cuda") * sc   # noqa: E731
    w = {"query": r(code, sc=1.0), "norm_q_w": 1 + r(code), "norm_q_b": r(code), "norm_kv_w": 1 + r(vdim),
         "norm_kv_b": r(vdim), "wq": r(code, code), "wk": r(code, vdim, sc=0.02), "wv": r(code, vdim, sc=0.02),
         "in_bias": r(3 * code), "wo": r(code, code), "bo": r(code), "ffn_ln_w": 1 + r(code), "ffn_ln_b": r(code),
         "w1": r(4 * code, code), "b1": r(4 * code), "w2": r(code, 4 * code), "b2": r(code),
         "final_w": 1 + r(code), "final_b": r(code), "act_w": r(ctx, code), "act_b": r(ctx)}
    vlm = r(S, vdim, sc=1.0).half()
    idx = torch.arange(3, 3 + nq, device="cuda", dtype=torch.int32)
    dev = {k: v.contiguous() for k, v in w.items()}
    for k in ("wq", "wo", "w1", "w2", "act_w"):
        dev[k + "_t"] = dev.pop(k).t().contiguous()
    scratch = torch.empty(nq * 2 * code, device="cuda")
    latent = torch.empty(code, device="cuda")
    cond = torch.empty(ctx, dtype=torch.half, device="cuda")
    assert lk.qformer_latent(vlm.data_ptr(), idx.data_ptr(), nq, vdim, code, heads, ctx,
                             {k: v.data_ptr() for k, v in dev.items()}, scratch.data_ptr(), latent.data_ptr(),
                             cond.data_ptr(), 0) == 0
    kvn = F.layer_norm(vlm[idx.long()].float(), (vdim,), w["norm_kv_w"], w["norm_kv_b"], 1e-5)
    k_, v_ = kvn @ w["wk"].t() + w["in_bias"][code:2 * code], kvn @ w["wv"].t() + w["in_bias"][2 * code:]
    q = F.layer_norm(w["query"], (code,), w["norm_q_w"], w["norm_q_b"], 1e-5) @ w["wq"].t() + w["in_bias"][:code]
    hd = code // heads
    p = torch.softmax((q.view(heads, 1, hd) @ k_.view(nq, heads, hd).permute(1, 2, 0)) * hd ** -0.5, -1)
    o = (p @ v_.view(nq, heads, hd).transpose(0, 1)).reshape(code)
    h = w["query"] + o @ w["wo"].t() + w["bo"]
    f = F.gelu(F.layer_norm(h, (code,), w["ffn_ln_w"], w["ffn_ln_b"], 1e-5) @ w["w1"].t() + w["b1"])
    h = h + f @ w["w2"].t() + w["b2"]
    lat = F.layer_norm(h, (code,), w["final_w"], w["final_b"], 1e-5)
    torch.testing.assert_close(latent, lat, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cond.float(), F.silu(lat @ w["act_w"].t() + w["act_b"]), rtol=2e-3, atol=2e-3)


def test_w4a16_skinny_gemm_matches_dequantized_matmul():
    lk = _lawam()
    from flash_rt.frontends.torch.lawam_rtx import _nvfp4_weight_only
    torch.manual_seed(0)
    M, N, K = 24, 1024, 4096
    a = torch.randn(M, K, device="cuda").bfloat16()
    w4, sf, gs = _nvfp4_weight_only(torch.randn(N, K, device="cuda") * 0.02)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    q = torch.stack([lut[(w4 & 15).long()], lut[(w4 >> 4).long()]], -1).view(N, K)
    ref = a.float() @ (q * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1)).t()
    for cfg in range(lk.w4a16_skinny_config_count()):
        kc = lk.w4a16_skinny_config_k_chunk(cfg)
        part = torch.empty(K // kc, M, N, device="cuda")
        assert lk.w4a16_skinny_gemm(a.data_ptr(), w4.data_ptr(), sf.data_ptr(), part.data_ptr(),
                                    M, N, K, cfg, False, 0) == 0
        # partials exclude the per-tensor scale (the consumers' alpha)
        torch.testing.assert_close(part.sum(0) * gs, ref * gs, rtol=1e-4, atol=1e-4)


def test_qkv_rope_split_matches_flashrt_qkv_split():
    lk = _lawam()
    fvk = pytest.importorskip("flash_rt.flash_rt_kernels")
    torch.manual_seed(0)
    S, H = 512, 16
    D = H * 64
    qkv = torch.randn(S, 3 * D, device="cuda").half()
    b = (torch.randn(3 * D, device="cuda") * 0.1).half()
    cos, sin = torch.randn(S, 64, device="cuda").half(), torch.randn(S, 64, device="cuda").half()
    ref = [torch.empty(S, D, device="cuda").half() for _ in range(3)]
    out = [torch.empty(S, D, device="cuda").half() for _ in range(3)]
    p = qkv.data_ptr()
    assert fvk.fp8_qkv_split(p, p + 2 * D, p + 4 * D, 3 * D, b.data_ptr(), b.data_ptr() + 2 * D, b.data_ptr() + 4 * D,
                             1.0, 1.0, 0, 0, 1e-6, cos.data_ptr(), sin.data_ptr(), *[t.data_ptr() for t in ref],
                             False, S, H, H, 64, 0) == 0
    assert lk.qkv_rope_split(p, 3 * D, b.data_ptr(), b.data_ptr() + 2 * D, b.data_ptr() + 4 * D, 1.0, 1.0,
                             cos.data_ptr(), sin.data_ptr(), *[t.data_ptr() for t in out], False, S, H, H, 64, 0) == 0
    for r, o in zip(ref, out):
        torch.testing.assert_close(o, r, rtol=0, atol=0)


@pytest.mark.parametrize("M,N,K", [(219, 2048, 6144), (219, 12288, 2048)])
def test_cutlass_f16acc_streamk_tile_matches_fp32_reference(M, N, K):
    # stream-K partial tiles meet in a fixed order: FP16 sums, reproducible run to run
    lk = _lawam()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, N, device="cuda") / K ** 0.5).half()
    out, again = torch.empty(M, N, device="cuda").half(), torch.empty(M, N, device="cuda").half()
    assert lk.gemm_f16acc_act_cfg(a.data_ptr(), w.data_ptr(), out.data_ptr(), 0, M, N, K, 0, 8, 1, 0) == 0
    assert lk.gemm_f16acc_act_cfg(a.data_ptr(), w.data_ptr(), again.data_ptr(), 0, M, N, K, 0, 8, 1, 0) == 0
    ref = a.float() @ w.float()
    dp = torch.empty(M, N, device="cuda").half()   # data-parallel FP16-accumulate tile for scale
    assert lk.gemm_f16acc_act_cfg(a.data_ptr(), w.data_ptr(), dp.data_ptr(), 0, M, N, K, 0, 0, 1, 0) == 0
    err = ((out.float() - ref).norm() / ref.norm()).item()
    err_dp = ((dp.float() - ref).norm() / ref.norm()).item()
    assert err <= 1.05 * err_dp   # shorter FP16 accumulation chains than the data-parallel tile
    torch.testing.assert_close(again, out, rtol=0, atol=0)


@pytest.mark.parametrize("M,N,K,act,cfg,sched", [(219, 4096, 2048, 0, 0, 0), (261, 3072, 768, 2, 1, 2),
                                                 (512, 4096, 1024, 1, 2, 1), (256, 1024, 4096, 0, 4, 0)])
def test_persistent_f16acc_gemm_matches_fp32_reference(M, N, K, act, cfg, sched):
    # data-parallel + stream-K schedule: split tiles summed in k order, reproducible run to run
    lk = _lawam()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, N, device="cuda") / K ** 0.5).half()
    bias = torch.randn(N, device="cuda").half() if act else None
    bp = bias.data_ptr() if act else 0
    out, again = torch.empty(M, N, device="cuda").half(), torch.empty(M, N, device="cuda").half()
    for o in (out, again):
        assert lk.gemm_f16acc_persistent(a.data_ptr(), w.data_ptr(), o.data_ptr(), bp, M, N, K, N, act, cfg, sched,
                                         0) == 0
    ref = a.float() @ w.float() + (bias.float() if act else 0)
    if act:
        ref = torch.nn.functional.gelu(ref, approximate="tanh" if act == 1 else "none")
    dp = torch.empty(M, N, device="cuda").half()   # CUTLASS data-parallel FP16-accumulate tile for scale
    assert lk.gemm_f16acc_act_cfg(a.data_ptr(), w.data_ptr(), dp.data_ptr(), bp, M, N, K, act, 0, 1, 0) == 0
    err = ((out.float() - ref).norm() / ref.norm()).item()
    err_dp = ((dp.float() - ref).norm() / ref.norm()).item()
    assert err <= 1.1 * err_dp
    torch.testing.assert_close(again, out, rtol=0, atol=0)


def test_persistent_swiglu_matches_gateup_and_silu_mul():
    from flash_rt.models.lawam import pipeline_rtx as P

    lk = _lawam()
    torch.manual_seed(0)
    M, FF, K = 219, 6144, 2048
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, 2 * FF, device="cuda") / K ** 0.5).half()
    out = torch.empty(M, FF, device="cuda").half()
    wi = P.interleave_gate_up(w)
    assert lk.gemm_f16acc_persistent(a.data_ptr(), wi.data_ptr(), out.data_ptr(), 0, M, 2 * FF, K, FF, 3, 0, 0,
                                     0) == 0
    gu = a.float() @ w.float()
    ref = torch.nn.functional.silu(gu[:, :FF]) * gu[:, FF:]
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_build_gate_is_opt_in_and_owned_by_the_lawam_module():
    """The module is not part of the default build: its sources exist only in
    the FLASHRT_ENABLE_LAWAM block and the option defaults to OFF."""
    from pathlib import Path

    cmake = (Path(__file__).resolve().parents[1] / "CMakeLists.txt").read_text()
    assert ('option(FLASHRT_ENABLE_LAWAM\n'
            '       "Build LaWAM RTX SM120 fused kernels in flash_rt_lawam" OFF)') in cmake
    gate = cmake.index("if(FLASHRT_ENABLE_LAWAM)")
    block = cmake[gate:cmake.index("# ── MiniMax-Remover", gate)]
    assert "pybind11_add_module(flash_rt_lawam" in block
    assert "message(STATUS \"LaWAM SM120 kernels: ENABLED\")" in block
    for src in ("csrc/lawam_bindings.cpp",
                "csrc/kernels/lawam/lawam_dit_persistent_sm120.cu"):
        assert src in block
        assert src not in cmake[:gate]
    # the config-time prerequisites and the opt-in FA2 short-tile dispatch
    assert "FLASHRT_ENABLE_LAWAM requires the FA2 FP16 causal" in cmake
    assert cmake.count("FLASHRT_FA2_SM12X_SHORT_TILES=1") == 2
    header = (Path(__file__).resolve().parents[1] /
              "csrc/attention/flash_attn_2_src/flash_attn/flash_fwd_launch_template.h").read_text()
    assert header.count("#if defined(FLASHRT_FA2_SM12X_SHORT_TILES)") == 2
    assert header.count("params.lawam_short_tiles &&") == 2


def test_missing_module_fails_fast(monkeypatch):
    """Without -DFLASHRT_ENABLE_LAWAM=ON the frontend names the build flag
    instead of surfacing a bare ModuleNotFoundError after loading weights."""
    import sys

    from flash_rt.frontends.torch import lawam_rtx

    monkeypatch.delattr(sys.modules["flash_rt"], "flash_rt_lawam", raising=False)
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_lawam", None)
    with pytest.raises(RuntimeError, match="FLASHRT_ENABLE_LAWAM=ON"):
        lawam_rtx.LawamTorchFrontendRtx("/nonexistent")


def test_required_symbols_fail_fast():
    """A stale flash_rt_lawam names the missing entries instead of raising
    AttributeError midway through graph capture."""
    from flash_rt.frontends.torch.lawam_rtx import _check_lawam_symbols

    class _Empty:
        pass

    with pytest.raises(RuntimeError) as ei:
        _check_lawam_symbols(_Empty())
    assert "FLASHRT_ENABLE_LAWAM" in str(ei.value)
    assert "dit_flow_persistent" in str(ei.value)


def test_frames_shape_and_dtype_guard():
    import numpy as np
    from types import SimpleNamespace

    from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx

    fe = SimpleNamespace(num_views=2)          # _frames only reads num_views
    with pytest.raises(ValueError, match="uint8 frames"):
        LawamTorchFrontendRtx._frames(fe, np.zeros((2, 224, 224, 3), np.uint8))
    with pytest.raises(ValueError, match="uint8 frames"):
        LawamTorchFrontendRtx._frames(fe, np.zeros((2, 256, 256, 3), np.float32))
    assert LawamTorchFrontendRtx._frames(fe, np.zeros((2, 256, 256, 3), np.uint8)).shape == (2, 256, 256, 3)


def test_constructor_rejects_view_counts():
    _lawam()
    from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx
    with pytest.raises(ValueError, match="primary \\+ one wrist"):
        LawamTorchFrontendRtx("/nonexistent", num_views=1)


def test_model_packages_import_without_the_extension():
    import importlib
    assert importlib.import_module("flash_rt.models.lawam") is not None
    assert importlib.import_module("flash_rt.models.lawam.pipeline_rtx") is not None
    assert importlib.import_module("flash_rt.frontends.torch.lawam_rtx") is not None


def test_checkpoint_smoke_is_deterministic():
    """Local smoke, skipped in CI: set FLASHRT_LAWAM_CHECKPOINT to the LeRobot
    checkpoint. Checks the documented predict path, finite output and that a
    fixed flow sample reproduces bit-identically."""
    import os
    import numpy as np

    ckpt = os.environ.get("FLASHRT_LAWAM_CHECKPOINT")
    if not ckpt:
        pytest.skip("set FLASHRT_LAWAM_CHECKPOINT=<lawam-pretrain-lerobot> for the checkpoint smoke")
    from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx

    fe = LawamTorchFrontendRtx(ckpt)
    fe.set_prompt("put the bowl on the plate")
    frames = np.zeros((2, 256, 256, 3), np.uint8)
    noise = np.zeros((fe.horizon, 32), np.float32)
    a = fe.infer(frames, noise=noise)["actions"]
    b = fe.infer(frames, noise=noise)["actions"]
    assert a.shape == (fe.horizon, fe.action_dim)
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)


@pytest.mark.parametrize("num_ctas", [32, 36, 64])
def test_persistent_layernorm_legal_cta_profiles(num_ctas):
    """Exercise the actual cooperative LN reduction, including the 5090 default."""
    import numpy as np
    from flash_rt.models.lawam import pipeline_rtx as P
    lk = _lawam()
    if torch.cuda.get_device_properties(0).multi_processor_count < num_ctas:
        pytest.skip("profile needs more resident SMs")
    torch.manual_seed(7)
    h = torch.randn(24, 1024, device="cuda").bfloat16()
    out = torch.empty_like(h, dtype=torch.float16)
    stats = torch.zeros(num_ctas, 24, 2, device="cuda")
    # Per-owner column sums, exactly as produced by the preceding GEMM.
    for owner, chunk in enumerate(torch.tensor_split(h.float(), num_ctas, dim=1)):
        stats[owner, :, 0] = chunk.sum(1)
        stats[owner, :, 1] = chunk.square().sum(1)
    i, f, p = np.zeros(16, np.int32), np.zeros(8, np.float32), np.zeros(16, np.uint64)
    i[P._I_KIND], i[P._I_BAR], i[P._I_OWNERS] = P._OP_LN, 1, num_ctas
    f[P._F_EPS] = 1e-6
    p[P._P_ABUF], p[P._P_OUT0], p[P._P_STATS] = h.data_ptr(), out.data_ptr(), stats.data_ptr()
    desc = torch.from_numpy(np.concatenate([i.view(np.uint8), f.view(np.uint8), p.view(np.uint8)])).cuda()
    tab = torch.zeros(num_ctas, 4, dtype=torch.int32, device="cuda")
    units = torch.zeros(num_ctas, dtype=torch.int32, device="cuda")
    stream = torch.zeros(num_ctas, 16, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(4, dtype=torch.int32, device="cuda")
    args = (desc.data_ptr(), 1, tab.data_ptr(), 1, units.data_ptr(), 1,
            stream.data_ptr(), 16, counter.data_ptr())
    assert lk.dit_flow_persistent(*args, num_ctas, 0, True) == 0
    expected = torch.nn.functional.layer_norm(h.float(), (1024,), eps=1e-6).bfloat16().half()
    torch.testing.assert_close(out, expected, rtol=0, atol=0.015625)
    if num_ctas == 64:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream):
            status = lk.dit_flow_persistent(*args, num_ctas, capture_stream.cuda_stream, True)
        assert status == 0
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0.015625)
    # A device-sized launch must be rejected before it reaches the 64-owner reduction.
    assert lk.dit_flow_persistent(*args, 170, 0, True) != 0


@pytest.mark.parametrize("num_ctas", [32, 36, 64])
@pytest.mark.parametrize("keys", [24, 219, 512])
def test_persistent_attention_covers_every_head_and_query_tile(num_ctas, keys):
    import numpy as np
    from flash_rt.models.lawam import pipeline_rtx as P
    lk = _lawam()
    if torch.cuda.get_device_properties(0).multi_processor_count < num_ctas:
        pytest.skip("profile needs more resident SMs")
    torch.manual_seed(23)
    q = (torch.randn(24,16,64,device="cuda") / 8).bfloat16()
    k = torch.randn(keys,16,64,device="cuda").bfloat16()
    v = torch.randn(keys,16,64,device="cuda").bfloat16()
    padded_keys = (keys + 15) // 16 * 16
    vt = torch.zeros(16,64,padded_keys,device="cuda",dtype=torch.bfloat16)
    vt[:,:,:keys] = v.permute(1,2,0)
    out = torch.full((24,16,64),float("nan"),device="cuda",dtype=torch.float16)
    i, f, p = np.zeros(16,np.int32), np.zeros(8,np.float32), np.zeros(16,np.uint64)
    i[P._I_KIND], i[P._I_BAR] = P._OP_ATTN, 1
    i[P._I_KVLEN], i[P._I_KLD], i[P._I_VTLD] = keys, 1024, padded_keys
    p[P._P_Q], p[P._P_K], p[P._P_VT], p[P._P_OUT0] = q.data_ptr(), k.data_ptr(), vt.data_ptr(), out.data_ptr()
    desc = torch.from_numpy(np.concatenate([i.view(np.uint8),f.view(np.uint8),p.view(np.uint8)])).cuda()
    tab = torch.zeros(num_ctas,4,device="cuda",dtype=torch.int32)
    units = torch.zeros(num_ctas,device="cuda",dtype=torch.int32)
    stream = torch.zeros(num_ctas,16,device="cuda",dtype=torch.uint8)
    counter = torch.zeros(4,device="cuda",dtype=torch.int32)
    args = (desc.data_ptr(),1,tab.data_ptr(),1,units.data_ptr(),1,stream.data_ptr(),16,counter.data_ptr())
    for invalid in (24,28):
        assert lk.dit_flow_persistent(*args,invalid,0,True) != 0
    assert torch.isnan(out).all()
    assert lk.dit_flow_persistent(*args,num_ctas,0,True) == 0
    assert torch.isfinite(out).all(), "attention left at least one output element unwritten"
    # Q already includes the base-2 softmax scale; convert logits back to
    # natural-log units for the PyTorch reference.
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(1,0,2).float(),k.permute(1,0,2).float(),v.permute(1,0,2).float(),scale=0.6931471805599453).permute(1,0,2)
    torch.testing.assert_close(out.float(),ref,rtol=0.02,atol=0.015)
