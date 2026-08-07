"""GemmRunner FP8 NN device-descale FP16-out dispatch and autotune paths."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"flash_rt_kernels is not built: {exc}", allow_module_level=True)

if not hasattr(fvk, "GemmRunner"):
    pytest.skip("GemmRunner not exported", allow_module_level=True)

if torch.cuda.get_device_capability() < (8, 9):
    pytest.skip(
        "FP8 GEMM requires sm_89+ tensor cores "
        "(cuBLASLt returns CUBLAS_STATUS_NOT_SUPPORTED below that)",
        allow_module_level=True)

torch.manual_seed(0)


def _quant_fp8(x):
    amax = x.abs().max().clamp_min(1e-6).float()
    scale = (amax / 448.0).reshape(1)
    q = torch.clamp(x.float() / scale.item(), -448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32).cuda()


def _cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def test_fp8_nn_dev_fp16_matches_dequant_reference():
    runner = fvk.GemmRunner()
    M, N, K = 128, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    b = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05

    a_q, a_scale = _quant_fp8(a)
    b_q, b_scale = _quant_fp8(b)
    d = torch.empty(M, N, device="cuda", dtype=torch.float16)

    runner.fp8_nn_dev_fp16(a_q.data_ptr(), b_q.data_ptr(), d.data_ptr(),
                           M, N, K, a_scale.data_ptr(), b_scale.data_ptr())
    torch.cuda.synchronize()

    ref = (a_q.float() @ b_q.float().T) * (a_scale.item() * b_scale.item())
    assert _cos(d, ref) >= 0.999, f"cosine {_cos(d, ref)} < 0.999"


def test_autotune_caches_and_result_stays_identical():
    runner = fvk.GemmRunner()
    M, N, K = 64, 2048, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5
    b = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
    a_q, a_scale = _quant_fp8(a)
    b_q, b_scale = _quant_fp8(b)

    d1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    runner.fp8_nn_dev_fp16(a_q.data_ptr(), b_q.data_ptr(), d1.data_ptr(),
                           M, N, K, a_scale.data_ptr(), b_scale.data_ptr())
    torch.cuda.synchronize()

    runner.autotune_fp8_nn_dev_fp16(a_q.data_ptr(), b_q.data_ptr(), d1.data_ptr(),
                                    M, N, K, a_scale.data_ptr(), b_scale.data_ptr(),
                                    4)
    torch.cuda.synchronize()

    d2 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    runner.fp8_nn_dev_fp16(a_q.data_ptr(), b_q.data_ptr(), d2.data_ptr(),
                           M, N, K, a_scale.data_ptr(), b_scale.data_ptr())
    torch.cuda.synchronize()

    # Autotune selects a tactic for the same (M,N,K); output must remain
    # numerically equivalent (tactics differ in tiling, not math).
    assert _cos(d1, d2) >= 0.9995, f"post-autotune cosine {_cos(d1, d2)} < 0.9995"
