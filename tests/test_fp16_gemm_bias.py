"""Verify the FP16 bias epilogue's column layout and CUDA graph replay."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("shape", [(8, 64, 64), (256, 2048, 1152)])
def test_fp16_bias_matches_float_reference_and_replays(shape):
    from flash_rt import flash_rt_kernels
    torch.manual_seed(7)
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * .1
    b = torch.randn(k, n, device="cuda", dtype=torch.float16) * .1
    bias = torch.linspace(-.25, .25, n, device="cuda", dtype=torch.float16)
    output = torch.empty(m, n, device="cuda", dtype=torch.float16)
    runner = flash_rt_kernels.GemmRunner()

    def run():
        runner.fp16_nn_bias(a.data_ptr(), b.data_ptr(), output.data_ptr(), bias.data_ptr(),
                           m, n, k, torch.cuda.current_stream().cuda_stream)

    run()
    expected = (a.float() @ b.float() + bias.float()).half()
    torch.testing.assert_close(output, expected, atol=.002, rtol=.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a.mul_(.5)
    graph.replay()
    torch.testing.assert_close(output, (a.float() @ b.float() + bias.float()).half(),
                               atol=.002, rtol=.002)
