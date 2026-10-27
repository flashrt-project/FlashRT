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
    # same serial split-K as the cuBLASLt COMPUTE_16F plan -> the same FP16 sums
    lk = _lawam()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda").half()
    w = (torch.randn(K, N, device="cuda") / K ** 0.5).half()
    ref, out = torch.empty(M, N, device="cuda").half(), torch.empty(M, N, device="cuda").half()
    assert lk.gemm_f16acc(a.data_ptr(), w.data_ptr(), ref.data_ptr(), 0, M, N, K, False, 0) == 0
    split = max(1, lk.gemm_f16acc_split(M, N, K, False, False))
    assert lk.gemm_f16acc_act(a.data_ptr(), w.data_ptr(), out.data_ptr(), 0, M, N, K, 0, split, 0) == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


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
