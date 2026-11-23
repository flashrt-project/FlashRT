"""Contracts and GPU regression for the optional SM87 gated GEMM tile."""
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[1]

def test_tile_option_is_opt_in_and_requires_sm87():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert '"Use the 64-row INT8 gated GEMM tile for Pi0.5 M15/N4096/K1024 on SM87" OFF)' in cmake
    assert 'NOT GPU_ARCH STREQUAL "87" OR NOT ENABLE_SM80_INT8_CUTLASS' in cmake

def test_only_measured_shape_selects_small_tile():
    source = (ROOT / "csrc/gemm/cutlass_sm80_int8_silu_gated.cu").read_text()
    assert "M == 15 && N == 4096 && K == 1024" in source
    assert "KernelConfig<64>::run" in source
    assert "KernelConfig<128>::run" in source
    assert "#ifdef FLASHRT_PI05_GATED_TILE_SM87" in source

@pytest.mark.parametrize("shape", [(1,128,256), (15,4096,1024), (31,256,256)])
def test_gated_gemm_math_and_mutable_graph(shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8,7):
        pytest.skip("validated SM87 path")
    from flash_rt import flash_rt_kernels as fvk
    m,n,k = shape
    torch.manual_seed(47)
    a = torch.randint(-16,17,(m,k),device="cuda",dtype=torch.int8)
    w = torch.randint(-16,17,(n,k),device="cuda",dtype=torch.int8)
    sa = torch.full((m,),0.01,device="cuda")
    sw = torch.full((n,),0.02,device="cuda")
    gate = torch.randn(m,n,device="cuda",dtype=torch.bfloat16)
    out = torch.empty_like(gate)
    def run():
        stream = torch.cuda.current_stream().cuda_stream
        status = fvk.cutlass_int8_silu_gated_bf16out(
            a.data_ptr(),w.data_ptr(),sa.data_ptr(),sw.data_ptr(),
            gate.data_ptr(),out.data_ptr(),m,n,k,stream)
        assert status == 0
    def check():
        # Small integers have an exactly representable FP32 dot product here.
        acc = a.float() @ w.float().T
        up = (acc * sa[:,None]) * sw[None,:]
        g = gate.float()
        expected = (up * g / (1 + torch.exp(-g))).bfloat16()
        torch.testing.assert_close(out,expected,rtol=0.016,atol=0.0001)
    run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a.neg_()
    gate.mul_(0.5)
    sa.mul_(0.75)
    graph.replay()
    torch.cuda.synchronize()
    check()
