#!/usr/bin/env python3
"""bf16_matmul_cublaslt_ref_bf16 reproduces torch.mm bit for bit.

The strict path (tuned=False) runs cuBLASLt's heuristic first choice, the
algorithm torch's matmul runs. The tuned path may run another candidate that
matched it on random probe operands drawn by the kernel, so its choice for a
shape must not depend on the weights of the call that settled it: a first
call with all-zero weights (on which every candidate agrees) and later calls
with other weights of the same shape are compared against torch.mm here.

Usage:
  PYTHONPATH=. pytest -q tests/test_bf16_matmul_cublaslt_ref.py
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
    if not hasattr(fvk, "bf16_matmul_cublaslt_ref_bf16"):
        pytest.skip("flash_rt_kernels predates bf16_matmul_cublaslt_ref_bf16")
    return fvk


def _gemm(fvk, x, w, tuned):
    out = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=BF16)
    fvk.bf16_matmul_cublaslt_ref_bf16(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                                      x.shape[0], w.shape[0], x.shape[1], tuned,
                                      torch.cuda.current_stream().cuda_stream)
    return out


@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("M", [1, 8, 33, 256, 1024])
@pytest.mark.parametrize("N,K", SHAPES)
def test_matches_torch_mm(M, N, K, tuned):
    fvk = _fvk()
    torch.manual_seed(M * 7 + N)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    for scale in (1.5, 0.7):
        x = (torch.randn(M, K, device="cuda") * scale).to(BF16)
        assert torch.equal(_gemm(fvk, x, w, tuned), torch.mm(x, w.t()))


# Shapes no other test uses, so the first call below is the one that settles
# the plan.
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("M,N,K", [(40, 1536, 2560), (1000, 1536, 2560)])
def test_choice_does_not_depend_on_the_first_weights(M, N, K, tuned):
    fvk = _fvk()
    zero_w = torch.zeros(N, K, device="cuda", dtype=BF16)
    zero_x = torch.zeros(M, K, device="cuda", dtype=BF16)
    assert torch.equal(_gemm(fvk, zero_x, zero_w, tuned), torch.zeros(M, N, device="cuda", dtype=BF16))
    torch.manual_seed(N + M)
    weights = [(torch.randn(N, K, device="cuda") * s).to(BF16) for s in (0.02, 0.3)]
    for step in range(4):
        w = weights[step % 2]
        x = (torch.randn(M, K, device="cuda") * (step + 1)).to(BF16)
        assert torch.equal(_gemm(fvk, x, w, tuned), torch.mm(x, w.t()))


@pytest.mark.parametrize("tuned", [False, True])
def test_same_shape_different_weights(tuned):
    # k_proj and v_proj share a shape and a cached plan.
    fvk = _fvk()
    torch.manual_seed(3)
    M, N, K = 257, 1024, 2560
    wk = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    wv = (torch.randn(N, K, device="cuda") * 0.05).to(BF16)
    for step in range(4):
        x = torch.randn(M, K, device="cuda").to(BF16)
        for w in (wk, wv):
            assert torch.equal(_gemm(fvk, x, w, tuned), torch.mm(x, w.t()))
