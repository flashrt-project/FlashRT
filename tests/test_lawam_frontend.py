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
