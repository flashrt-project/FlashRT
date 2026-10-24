#!/usr/bin/env python3
"""bf16_matmul_cublaslt_bitexact_bf16 reproduces torch.mm bit for bit.

The entry may run a different cuBLASLt algorithm than the heuristic's first
choice, but only one that was checked to produce the same bits. The first
call for a shape does that check on random probes; later calls are compared
here against torch.mm on fresh activations, at the Qwen3-4B projection shapes
and at row counts on either side of where cuBLASLt switches algorithm.

Usage:
  PYTHONPATH=. pytest -q tests/test_bf16_matmul_cublaslt_bitexact.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16

# (N, K) of the Qwen3-4B projections: q, k/v, o, gate/up, down.
SHAPES = [(4096, 2560), (1024, 2560), (2560, 4096), (9728, 2560), (2560, 9728)]


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    if not hasattr(fvk, "bf16_matmul_cublaslt_bitexact_bf16"):
        pytest.skip("flash_rt_kernels predates bf16_matmul_cublaslt_bitexact_bf16")
    return fvk


@pytest.mark.parametrize("M", [1, 8, 33, 256, 1024])
@pytest.mark.parametrize("N,K", SHAPES)
def test_matches_torch_mm(M, N, K):
    fvk = _fvk()
    torch.manual_seed(M * 7 + N)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    out = torch.empty(M, N, device="cuda", dtype=BF16)
    s = torch.cuda.current_stream().cuda_stream
    for scale in (1.5, 0.7):
        x = (torch.randn(M, K, device="cuda") * scale).to(BF16)
        fvk.bf16_matmul_cublaslt_bitexact_bf16(x.data_ptr(), w.data_ptr(),
                                               out.data_ptr(), M, N, K, s)
        assert torch.equal(out, torch.mm(x, w.t()))
