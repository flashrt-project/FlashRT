#!/usr/bin/env python3
"""BF16 GEMM with a fused bias epilogue over the (N,K) weight layout.

`bf16_nn_bias` computes D = A @ B(K,N) + bias from a (K,N) B. This covers the
other convention, an nn.Linear weight kept as stored at (N,K), which is what a
caller holding the checkpoint layout has in hand and what `bf16_run` already
accepts without a bias.

The reason this needs its own entry rather than a flag: the row-order
formulation with a BIAS epilogue returns CUBLAS_STATUS_NOT_SUPPORTED for some
BF16 shapes on sm_120, which two other call sites in this tree document and
work around with a separate bias add. `test_row_order_path_is_the_one_that_fails`
records whether this host is one of the affected ones; it is informational, and
reports rather than asserts, because the failure is shape- and driver-specific.

Usage:
  PYTHONPATH=. pytest -q tests/test_bf16_nt_bias.py
"""
import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

BF16 = torch.bfloat16


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    return fvk


def _run(fvk, M, N, K, seed=0):
    torch.manual_seed(seed)
    A = (torch.randn(M, K, device="cuda") * 0.5).to(BF16)
    W = (torch.randn(N, K, device="cuda") * 0.5).to(BF16)   # nn.Linear layout
    b = (torch.randn(N, device="cuda") * 0.1).to(BF16)
    D = torch.empty(M, N, device="cuda", dtype=BF16)
    g = fvk.GemmRunner()
    g.bf16_nt_bias(A.data_ptr(), W.data_ptr(), D.data_ptr(), b.data_ptr(),
                   M, N, K, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    # the epilogue sums in fp32 and rounds to bf16 exactly once
    ref = (A.float() @ W.float().t() + b.float()).to(BF16)
    return D, ref


class TestBf16NtBias:
    @pytest.mark.parametrize("M,N,K", [
        (2048, 1152, 1152),    # a SigLIP projection at its production shape
        (2048, 4304, 1152),    # the same tower's wider MLP
        (2048, 1152, 4304),
        (16, 64, 32),          # a small shape, where the row-order path is likeliest to fail
        (1, 256, 256),         # a single row
        (7, 129, 257),         # deliberately unaligned
    ])
    def test_matches_a_torch_reference(self, M, N, K):
        fvk = _fvk()
        D, ref = _run(fvk, M, N, K)
        frac = (D != ref).float().mean().item()
        # an fp32 sum over K in a different order can round the other way in bf16
        assert frac <= 0.01, f"{frac:.4f} of elements differ from the reference"

    def test_the_gemm_numerator_is_right(self):
        """A transposed or mis-strided operand would move the values by a lot."""
        fvk = _fvk()
        D, ref = _run(fvk, 64, 128, 96, seed=3)
        rel = ((D.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).mean().item()
        assert rel < 1e-2, f"mean relative error {rel}"

    def test_a_null_bias_is_not_silently_accepted(self):
        """The bias is fused into the epilogue, so a null pointer is a caller bug."""
        fvk = _fvk()
        torch.manual_seed(0)
        A = torch.zeros(4, 8, device="cuda", dtype=BF16)
        W = torch.zeros(16, 8, device="cuda", dtype=BF16)
        D = torch.zeros(4, 16, device="cuda", dtype=BF16)
        import ctypes
        with pytest.raises(ValueError, match="bf16_nt_bias.*non-null"):
            fvk.GemmRunner().bf16_nt_bias(
                A.data_ptr(), W.data_ptr(), D.data_ptr(), 0, 4, 16, 8,
                torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()

    @pytest.mark.parametrize("shape", [(0, 16, 8), (4, -1, 8), (4, 16, 0)])
    def test_invalid_dimensions_rejected_before_launch(self, shape):
        fvk = _fvk()
        with pytest.raises(ValueError, match="bf16_nt_bias.*positive"):
            fvk.GemmRunner().bf16_nt_bias(0, 0, 0, 0, *shape, 0)

    def test_graph_replay_reads_refreshed_inputs(self):
        fvk = _fvk()
        M, N, K = 16, 64, 32
        A = torch.randn(M, K, device="cuda", dtype=BF16)
        W = torch.randn(N, K, device="cuda", dtype=BF16)
        b = torch.randn(N, device="cuda", dtype=BF16)
        D = torch.empty(M, N, device="cuda", dtype=BF16)
        runner = fvk.GemmRunner()
        def launch():
            runner.bf16_nt_bias(A.data_ptr(), W.data_ptr(), D.data_ptr(), b.data_ptr(),
                               M, N, K, torch.cuda.current_stream().cuda_stream)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for offset in (0.0, 0.25):
            A.add_(offset)
            b.add_(offset)
            graph.replay()
            torch.cuda.synchronize()
            ref = (A.float() @ W.float().t() + b.float()).to(BF16)
            assert (D != ref).float().mean().item() <= 0.01

    def test_row_order_path_is_the_one_that_fails(self):
        """Informational: records whether bf16_nn_bias covers these shapes here."""
        fvk = _fvk()
        torch.manual_seed(0)
        M, N, K = 2048, 1152, 1152
        A = (torch.randn(M, K, device="cuda") * 0.5).to(BF16)
        W = (torch.randn(K, N, device="cuda") * 0.5).to(BF16)   # (K,N) for bf16_nn_bias
        b = (torch.randn(N, device="cuda") * 0.1).to(BF16)
        D = torch.empty(M, N, device="cuda", dtype=BF16)
        g = fvk.GemmRunner()
        try:
            g.bf16_nn_bias(A.data_ptr(), W.data_ptr(), D.data_ptr(), b.data_ptr(),
                           M, N, K, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            worked = True
        except Exception as exc:
            worked = False
            print(f"\nbf16_nn_bias (row-order, (K,N) B) raised on this host: {exc}")
        print(f"bf16_nn_bias covered (M,N,K)=({M},{N},{K}): {worked}")
        # both paths must agree where both work
        D2, ref = _run(fvk, M, N, K)
        if worked:
            got = (A.float() @ W.float() + b.float()).to(BF16)
            assert (D != got).float().mean().item() <= 0.01
        assert (D2 != ref).float().mean().item() <= 0.01


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
