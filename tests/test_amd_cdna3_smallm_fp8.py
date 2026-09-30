"""Parity gates for the gfx942 packed small-M FNUZ GEMM."""
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


def _pack(weight):
    n, k = weight.shape
    return (weight.view(n // 16, 16, k // 64, 2, 4, 8)
            .permute(0, 2, 4, 1, 3, 5).contiguous())


@pytest.mark.parametrize("label,m,n,k", SHAPES, ids=[s[0] for s in SHAPES])
def test_packed_fnuz_matches_quantized_torch(env, label, m, n, k):
    torch, ext = env
    torch.manual_seed(17)
    dtype = torch.float8_e4m3fnuz
    a = (0.2 * torch.randn(m, k, device="cuda")).to(dtype)
    weight = (0.2 * torch.randn(n, k, device="cuda")).to(dtype)
    packed = _pack(weight)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    scale_a = torch.ones(1, device="cuda", dtype=torch.float32)
    scale_w = torch.ones(1, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream().cuda_stream

    ext.smallm_mfma_nt_packed(
        a.data_ptr(), packed.data_ptr(), out.data_ptr(), m, n, k,
        scale_a.data_ptr(), scale_w.data_ptr(), stream,
    )
    torch.cuda.synchronize()

    reference = (a.float() @ weight.float().t()).to(torch.bfloat16)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), reference.float().flatten(), dim=0,
    ).item()
    assert cosine >= 0.9999, f"{label}: cosine={cosine}"
    torch.testing.assert_close(out.float(), reference.float(),
                               atol=2e-2, rtol=0.0)
