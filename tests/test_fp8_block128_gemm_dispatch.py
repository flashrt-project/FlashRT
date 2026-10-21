#!/usr/bin/env python3
"""Block-128 FP8 GEMM (SM120): either tile computes the same result.

``fp8_block128_gemm_cutlass_sm120_bf16out`` dispatches between a 64-row
(Pingpong) and a 128-row (Cooperative) tile by grid size. Both accumulate each
output element over the same K blocks in the same order, so the tile must not
change a single bit: a row of the product is the same whatever M the call has,
and whichever tile that M selects. The tests pin that, and the product against
a dequantized FP32 reference.

Usage:
  PYTHONPATH=. pytest -q tests/test_fp8_block128_gemm_dispatch.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

# (M, N, K): at M = 512 an N = 2560 grid is under one wave of 128-row tiles and
# takes the 64-row tile; N = 19456 fills several waves and keeps the 128-row one.
SHAPES = [(512, 2560, 4096), (512, 19456, 2560), (300, 6144, 2560), (200, 1024, 1024)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("fp8_block128_gemm_cutlass_sm120_bf16out is an SM120 kernel")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _quantize_activation(x):
    """Per token, per 128 K: (fp8 values (M, K), dequant scales (M, K/128) fp32)."""
    M, K = x.shape
    blocks = x.float().view(M, K // 128, 128)
    scale = (blocks.abs().amax(-1) / 448.0).clamp_min(1e-12)
    values = (blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return values.view(M, K).contiguous(), scale.contiguous()


def _quantize_weight(w):
    """Per 128 x 128 block: (fp8 values (N, K), dequant scales (N/128, K/128) fp32)."""
    N, K = w.shape
    blocks = w.float().view(N // 128, 128, K // 128, 128)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-12)
    values = (blocks / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return values.view(N, K).contiguous(), scale.contiguous()


def _gemm(fvk, xq, xs, wq, ws, M):
    N, K = wq.shape
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
        xq.data_ptr(), wq.data_ptr(), out.data_ptr(), M, N, K, xs.data_ptr(), ws.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return out


def _operands(M, N, K, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    return (*_quantize_activation(x), *_quantize_weight(w))


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matches_dequantized_reference(M, N, K):
    fvk = _fvk()
    xq, xs, wq, ws = _operands(M, N, K)
    out = _gemm(fvk, xq, xs, wq, ws, M)
    x = (xq.float().view(M, K // 128, 128) * xs[..., None]).view(M, K)
    w = (wq.float().view(N // 128, 128, K // 128, 128) * ws[:, None, :, None]).view(N, K)
    reference = x @ w.t()
    assert torch.isfinite(out).all()
    assert float((out.float() - reference).norm() / reference.norm()) < 1e-2


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_rows_do_not_depend_on_the_tile(M, N, K):
    """Rows of an M-row product equal the same rows computed alone (M <= 64 always takes the 64-row tile)."""
    fvk = _fvk()
    xq, xs, wq, ws = _operands(M, N, K)
    full = _gemm(fvk, xq, xs, wq, ws, M)
    for start in (0, 64, M - 64):
        rows = slice(start, start + 64)
        alone = _gemm(fvk, xq[rows].contiguous(), xs[rows].contiguous(), wq, ws, 64)
        assert torch.equal(full[rows], alone), f"rows {start}..{start + 63} differ at M={M}, N={N}, K={K}"
