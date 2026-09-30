"""Parity gates for the gfx942 packed small-M BF16 GEMM."""
from __future__ import annotations

import importlib

import pytest


SHAPES = (
    ("decoder_qkv", 10, 2560, 1024),
    ("decoder_o", 10, 1024, 2048),
    ("decoder_gate_up", 10, 8192, 1024),
    ("decoder_down", 10, 1024, 4096),
)


@pytest.fixture(scope="module")
def env():
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        pytest.skip("requires a visible ROCm device")
    ext = importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    if dict(ext.build_info())["hardware"] != "amd_cdna3":
        pytest.skip("gfx942-specific parity gate")
    return torch, ext


def _pack_gfx942(weight):
    """Pack KxN BF16 for gfx942's 16-deep, four-BF16/lane MFMA."""
    k, n = weight.shape
    return (weight.view(k // 16, 4, 4, n // 16, 16)
            .permute(3, 0, 1, 4, 2).contiguous())


@pytest.mark.parametrize("label,m,n,k", SHAPES, ids=[s[0] for s in SHAPES])
def test_all_variants_match_torch(env, label, m, n, k):
    torch, ext = env
    torch.manual_seed(13)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = (0.02 * torch.randn(k, n, device="cuda")).to(torch.bfloat16)
    packed = _pack_gfx942(weight)
    bias = (0.1 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    reference = (a.float() @ weight.float() + bias.float()).to(torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream

    names = list(ext.smallm_mfma_bf16_variants())
    assert names == ["auto", "w4_fused", "w8_fused", "w4_split", "w8_split"]
    for variant, name in enumerate(names):
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ext.smallm_mfma_bf16_nn_bias(
            a.data_ptr(), packed.data_ptr(), bias.data_ptr(), out.data_ptr(),
            m, n, k, variant, stream,
        )
        torch.cuda.synchronize()
        cosine = torch.nn.functional.cosine_similarity(
            out.float().flatten(), reference.float().flatten(), dim=0,
        ).item()
        assert cosine >= 0.9999, f"{label}/{name}: cosine={cosine}"
        torch.testing.assert_close(out.float(), reference.float(),
                                   atol=2e-2, rtol=0.0)
