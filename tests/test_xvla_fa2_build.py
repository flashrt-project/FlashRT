"""Run against either the FP16-only stub build or the enabled BF16 profile."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


def adapter():
    value = os.environ.get("FLASHRT_FA2_ADAPTER_LIBRARY")
    if not value:
        pytest.skip("set FLASHRT_FA2_ADAPTER_LIBRARY to validate an actual build")
    spec = importlib.util.spec_from_file_location("flash_rt_fa2", value)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NAMES = ("fwd_bf16_tile64", "fwd_bf16_tile64m32", "fwd_bf16_tile64m32w2",
         "fwd_bf16_tile64_qnvfp4", "fwd_bf16_tile64m32w2_qnvfp4")


def test_native_exports_match_the_adapter_surface():
    raw = os.environ.get("FLASHRT_FA2_RAW_LIBRARY")
    if not raw:
        pytest.skip("set FLASHRT_FA2_RAW_LIBRARY to check native exports")
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", raw], text=True)
    expected = ["fvk_attention_fa2_" + name for name in NAMES] + ["fvk_attention_fa2_has_xvla_tile64"]
    for name in expected:
        assert any(line.split()[-1] == name for line in symbols.splitlines()), name
    module = adapter()
    assert all(hasattr(module, name) for name in NAMES)
    assert not hasattr(module, "set_pdl")


@pytest.mark.parametrize("name", NAMES)
def test_unavailable_profile_raises_before_launch(name):
    module = adapter()
    if module.has_xvla_tile64():
        pytest.skip("supported profile needs actual GPU buffers")
    args = [0]*6
    if name.endswith("qnvfp4"):
        args += [0, 0, 1, 1]
    args += [1, 1, 1, 1, 1, 64] + [(1, 1, 1)]*4
    for enabled in (False, True):
        with pytest.raises(RuntimeError, match="not compiled"):
            getattr(module, name)(*args, enable_pdl=enabled)


def test_pdl_is_per_call_and_preserves_legacy_attention():
    module = adapter()
    if not module.has_xvla_tile64():
        pytest.skip("requires enabled BF16 X-VLA profile")
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL needs a supported CUDA device")
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 17, 2, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    before, enabled, disabled, after = [torch.empty_like(q) for _ in range(4)]
    lse = torch.empty(1, 2, 17, device="cuda", dtype=torch.float32)
    strides = (17*128, 128, 64)
    args = [1, 17, 17, 2, 2, 64] + [strides]*4
    kw = dict(softmax_scale=0.125, stream=torch.cuda.current_stream().cuda_stream)
    def old(out):
        module.fwd_bf16(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                        lse.data_ptr(), 0, 0, *args, **kw)
    old(before)
    for out, pdl in ((enabled, True), (disabled, False)):
        module.fwd_bf16_tile64(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                               lse.data_ptr(), 0, *args, enable_pdl=pdl, **kw)
    old(after)
    torch.cuda.synchronize()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(enabled, disabled, rtol=0.01, atol=0.01)
    torch.testing.assert_close(enabled, before, rtol=0.01, atol=0.01)
    # Null quantized-output pointers exercise both fused-entry adapters and
    # their fallback, including propagation of the explicit PDL argument.
    for name in NAMES[-2:]:
        getattr(module, name)(q.data_ptr(), k.data_ptr(), v.data_ptr(), enabled.data_ptr(),
                              lse.data_ptr(), 0, 0, 0, 1, 1, *args, enable_pdl=True, **kw)
        torch.cuda.synchronize()
        torch.testing.assert_close(enabled, before, rtol=0.01, atol=0.01)
