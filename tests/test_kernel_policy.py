"""Scoped kernel policy: defaults, exceptions, nesting and concurrent models."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest

kernels = pytest.importorskip("flash_rt.flash_rt_kernels")
pytestmark = pytest.mark.skipif(not hasattr(kernels,"get_kernel_fast_paths"),reason="rebuild kernel policy bindings")
ROOT = Path(__file__).resolve().parents[1]


def policy(fast=True,pdl=True):
    if not hasattr(kernels,"VlaJepaKernelPolicy"):
        pytest.skip("model policy needs FLASHRT_ENABLE_VLA_JEPA=ON")
    return kernels.VlaJepaKernelPolicy(fast,pdl)


def test_shared_default_is_unchanged():
    assert kernels.get_kernel_fast_paths() == (False,False)
    assert not hasattr(kernels,"set_kernel_fast_paths")


def test_nested_model_calls_restore_the_previous_policy():
    first,second=policy(),policy(False,True)
    def run():
        assert kernels.get_kernel_fast_paths()==(True,True)
        assert second.run(kernels.get_kernel_fast_paths)==(False,True)
        assert kernels.get_kernel_fast_paths()==(True,True)
    first.run(run)
    assert kernels.get_kernel_fast_paths()==(False,False)


def test_failed_model_call_does_not_affect_next_model():
    def fail():
        assert kernels.get_kernel_fast_paths()==(True,True)
        raise ValueError("failed model call")
    with pytest.raises(ValueError,match="failed model"):
        policy().run(fail)
    assert kernels.get_kernel_fast_paths()==(False,False)


def test_models_on_other_threads_keep_independent_policies():
    first,second=policy(True,False),policy(False,True)
    barrier=threading.Barrier(2)
    def run(instance,expected):
        def inspect():
            barrier.wait(timeout=10)
            assert kernels.get_kernel_fast_paths()==expected
        instance.run(inspect)
        assert kernels.get_kernel_fast_paths()==(False,False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[pool.submit(run,first,(True,False)),pool.submit(run,second,(False,True))]
        for result in results:result.result(timeout=20)
    assert kernels.get_kernel_fast_paths()==(False,False)


@pytest.mark.parametrize("override,expected",[(None,"False"),("1","True"),("0","False")])
def test_pdl_environment_override_remains_explicit(override,expected):
    env=dict(os.environ);env.pop("FLASHRT_PDL",None)
    if override is not None:env["FLASHRT_PDL"]=override
    code=f"import sys;sys.path.insert(0,{str(ROOT)!r});import flash_rt.flash_rt_kernels as k;print(k.pdl_enabled())"
    assert subprocess.check_output([sys.executable,"-I","-c",code],env=env,text=True).strip()==expected


def test_quantizer_and_graph_do_not_leak_model_policy():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():pytest.skip("needs CUDA")
    selected=policy()
    x=torch.randn(17,256,device="cuda",dtype=torch.bfloat16)
    out=torch.empty(17,256,device="cuda",dtype=torch.float8_e4m3fn)
    scales=torch.empty(17,2,device="cuda")
    def quantize():
        kernels.fp8_per_token_block128_quant_bf16(x.data_ptr(),out.data_ptr(),scales.data_ptr(),17,256,torch.cuda.current_stream().cuda_stream)
    quantize();reference=out.view(torch.uint8).clone();ref_scales=scales.clone()
    selected.run(quantize)
    assert kernels.get_kernel_fast_paths()==(False,False)
    assert torch.equal(out.view(torch.uint8),reference)
    assert torch.equal(scales,ref_scales)
    capture_stream=torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):selected.run(quantize)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=capture_stream):selected.run(quantize)
    assert kernels.get_kernel_fast_paths()==(False,False)
    graph.replay()
    assert torch.equal(out.view(torch.uint8),reference)
    quantize()
    assert torch.equal(out.view(torch.uint8),reference)


def test_constructor_call_restores_shared_defaults():
    class Model:
        def __init__(self,value):
            self.value=value
            self.policy=kernels.get_kernel_fast_paths()
    model=policy().run(Model,value=7)
    assert model.value==7 and model.policy==(True,True)
    assert kernels.get_kernel_fast_paths()==(False,False)


def test_pdl_override_wins_inside_an_active_model_call():
    policy()  # skip if the model gate is OFF
    env=dict(os.environ,FLASHRT_PDL="0")
    code=f"import sys;sys.path.insert(0,{str(ROOT)!r});import flash_rt.flash_rt_kernels as k;print(k.VlaJepaKernelPolicy().run(k.pdl_enabled))"
    assert subprocess.check_output([sys.executable,"-I","-c",code],env=env,text=True).strip()=="False"


def test_bf16_gemm_can_alternate_model_and_shared_calls():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():pytest.skip("needs CUDA")
    selected=policy()
    x=torch.randn(17,64,device="cuda",dtype=torch.bfloat16)*0.1
    w=torch.randn(128,64,device="cuda",dtype=torch.bfloat16)*0.1
    out=torch.empty(17,128,device="cuda",dtype=torch.bfloat16)
    def gemm():
        kernels.bf16_matmul_cublaslt_bf16(x.data_ptr(),w.data_ptr(),out.data_ptr(),17,128,64,torch.cuda.current_stream().cuda_stream,2)
    selected.run(gemm)
    assert kernels.get_kernel_fast_paths()==(False,False)
    ref=x.float()@w.float().t()
    torch.testing.assert_close(out.float(),ref,rtol=0.02,atol=0.01)
    gemm()
    torch.testing.assert_close(out.float(),ref,rtol=0.02,atol=0.01)
    assert kernels.get_kernel_fast_paths()==(False,False)
