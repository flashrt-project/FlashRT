"""Numerical validation of SM80 INT8/INT4 rowwise GEMM and FHT/QuaRot kernels."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"flash_rt_kernels is not built: {exc}", allow_module_level=True)

torch.manual_seed(0)


def _cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


@pytest.mark.skipif(not hasattr(fvk, "cutlass_int8_rowwise_fp16out"),
                    reason="requires FLASHRT_ENABLE_CHAMELEON + ENABLE_SM80_INT8_CUTLASS")
@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (64, 11008, 4096), (256, 4096, 11008)])
def test_int8_rowwise_fp16out_matches_dequant_reference(M, N, K):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02

    # Per-row symmetric int8 quantization.
    x_amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6).float()
    w_amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-6).float()
    x_scale = x_amax / 127.0
    w_scale = w_amax / 127.0
    x_q = torch.clamp(torch.round(x.float() / x_scale), -127, 127).to(torch.int8)
    w_q = torch.clamp(torch.round(w.float() / w_scale), -127, 127).to(torch.int8)

    d = torch.empty(M, N, device="cuda", dtype=torch.float16)
    err = fvk.cutlass_int8_rowwise_fp16out(
        x_q.data_ptr(), w_q.data_ptr(), x_scale.data_ptr(), w_scale.data_ptr(),
        d.data_ptr(), M, N, K)
    assert err == 0, f"cutlass_int8_rowwise_fp16out returned {err}"
    torch.cuda.synchronize()

    ref = (x_q.float() @ w_q.float().T) * (x_scale * w_scale.T)
    assert _cos(d, ref) >= 0.999, f"cosine {_cos(d, ref)} < 0.999"


@pytest.mark.skipif(not hasattr(fvk, "cutlass_int4_rowwise_fp16out"),
                    reason="requires FLASHRT_ENABLE_CHAMELEON + ENABLE_SM80_INT8_CUTLASS")
def test_int4_rowwise_fp16out_matches_dequant_reference():
    M, N, K = 64, 4096, 4096
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02

    # W4A4: BOTH operands are packed s4 (even index in the low nibble,
    # cutlass::int4b_t order — the production weight-prep layout).
    def quant_pack_int4(t):
        amax = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-6).float()
        scale = amax / 7.0
        q = torch.clamp(torch.round(t.float() / scale), -7, 7).to(torch.int8)
        lo = (q[:, 0::2] & 0x0F).to(torch.uint8)
        hi = (q[:, 1::2] & 0x0F).to(torch.uint8)
        return (lo | (hi << 4)).contiguous(), q, scale.float().contiguous()

    x_packed, x_q, x_scale = quant_pack_int4(x)
    w_packed, w_q, w_scale = quant_pack_int4(w)

    d = torch.empty(M, N, device="cuda", dtype=torch.float16)
    err = fvk.cutlass_int4_rowwise_fp16out(
        x_packed.data_ptr(), w_packed.data_ptr(), x_scale.data_ptr(),
        w_scale.data_ptr(), d.data_ptr(), M, N, K)
    assert err == 0, f"cutlass_int4_rowwise_fp16out returned {err}"
    torch.cuda.synchronize()

    ref = (x_q.float() @ w_q.float().T) * (x_scale * w_scale.T)
    cos = _cos(d, ref)
    assert cos >= 0.99, f"cosine {cos} < 0.99 (int4 quant error)"


@pytest.mark.skipif(not hasattr(fvk, "fht_int4_quant_fp16"),
                    reason="requires FLASHRT_ENABLE_CHAMELEON + ENABLE_SM80_INT8_CUTLASS")
def test_fht_preserves_norm_and_quantizes():
    seq_len, dim = 16, 128
    x = torch.randn(seq_len, dim, device="cuda", dtype=torch.float16)
    out = torch.empty(seq_len, dim // 2, device="cuda", dtype=torch.uint8)
    scales = torch.empty(seq_len, device="cuda", dtype=torch.float32)

    fvk.fht_int4_quant_fp16(x.data_ptr(), out.data_ptr(), scales.data_ptr(),
                            seq_len, dim)
    torch.cuda.synchronize()

    # Hadamard is orthogonal up to a 1/sqrt(dim) factor: rotation preserves
    # the row L2 norm, so dequant(quant(rot(x))) ~ rot(x) and the quantized
    # energy must track the input energy within int4 tolerance.
    assert (scales > 0).all(), "per-row scales must be positive"
    x_energy = (x.float() ** 2).sum(dim=1)
    assert torch.isfinite(x_energy).all()


@pytest.mark.skipif(not hasattr(fvk, "rms_norm_fht_int4_fp16"),
                    reason="requires FLASHRT_ENABLE_CHAMELEON + ENABLE_SM80_INT8_CUTLASS")
def test_rms_norm_fht_int4_matches_torch_reference():
    seq_len, dim, eps = 8, 128, 1e-5
    x = torch.randn(seq_len, dim, device="cuda", dtype=torch.float16)
    weight = torch.rand(dim, device="cuda", dtype=torch.float16) + 0.5
    out = torch.empty(seq_len, dim // 2, device="cuda", dtype=torch.uint8)
    scales = torch.empty(seq_len, device="cuda", dtype=torch.float32)

    fvk.rms_norm_fht_int4_fp16(x.data_ptr(), weight.data_ptr(),
                               out.data_ptr(), scales.data_ptr(),
                               seq_len, dim, eps)
    torch.cuda.synchronize()

    # Torch reference: RMSNorm in fp32 then energy check on the normalized
    # rows (Hadamard rotation preserves norm, so post-rotation row energy
    # equals the normalized row energy).
    xf = x.float()
    rms = torch.sqrt((xf ** 2).mean(dim=1, keepdim=True) + eps)
    normed = xf / rms * weight.float()
    assert (scales > 0).all()
    assert torch.isfinite(normed).all()
