"""Strict visual-fusion build, shape and numerical contracts."""
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).parents[1]

def test_build_and_bindings_have_matching_ownership():
    cmake = (ROOT/"CMakeLists.txt").read_text()
    bindings = (ROOT/"csrc/bindings.cpp").read_text()
    assert '"Build Pi0.5 strict visual consumer fusion for SM87" OFF)' in cmake
    assert 'csrc/kernels/pi05/pi05_projector_norm_int8.cu' in cmake
    assert 'FLASHRT_PI05_STRICT_VISUAL_SM87=1' in cmake
    assert 'm.def("pi05_strict_visual_available"' in bindings
    assert 'm.def("pi05_projector_bias_rms_norm_int8"' in bindings
    assert 'm.def("qkv_split_bias_bf16"' in bindings

def test_pipeline_uses_direct_calls_and_rejects_unvalidated_modes():
    source = (ROOT/"flash_rt/models/pi05/pipeline_rtx.py").read_text()
    assert 'FLASHRT_PI05_STRICT_VISUAL' in source
    assert 'fvk.qkv_split_bias_bf16(' in source
    assert 'fvk.pi05_projector_bias_rms_norm_int8(' in source
    assert 'self._strict_visual_fusion and i == 0' in source
    assert 'or self._fixed_shape or self.vision_pool_factor != 1' in source


def _construct_with_probe(probe, **kwargs):
    from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    options = dict(num_views=2, max_prompt_len=50, use_fp8=False,
                   use_int8_encoder=True, use_int8_decoder=True)
    options.update(kwargs)
    return Pi05Pipeline(
        gemm=object(), fvk=probe,
        attn_backend=SimpleNamespace(set_fixed_shape=lambda value: None),
        weights={}, **options)


def test_not_built_fails_before_allocating_buffers(monkeypatch):
    monkeypatch.setenv("FLASHRT_PI05_STRICT_VISUAL", "1")
    with pytest.raises(RuntimeError, match="rebuild the extension"):
        _construct_with_probe(SimpleNamespace())
    probe = SimpleNamespace(pi05_strict_visual_available=lambda: False,
                            qkv_split_bias_bf16=lambda: None,
                            pi05_projector_bias_rms_norm_int8=lambda: None)
    with pytest.raises(RuntimeError, match="was not built"):
        _construct_with_probe(probe)


@pytest.mark.parametrize("options", [dict(use_fp8=True), dict(use_int8_vision=True),
                                    dict(use_int8_encoder=False), dict(fixed_shape=True),
                                    dict(vision_pool_factor=2)])
def test_unvalidated_precision_or_layout_is_rejected(monkeypatch, options):
    monkeypatch.setenv("FLASHRT_PI05_STRICT_VISUAL", "1")
    probe = SimpleNamespace(pi05_strict_visual_available=lambda: True,
                            qkv_split_bias_bf16=lambda: None,
                            pi05_projector_bias_rms_norm_int8=lambda: None)
    with pytest.raises(ValueError, match="strict visual fusion requires"):
        _construct_with_probe(probe, **options)

def _cuda_kernel():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8, 7):
        pytest.skip("requires SM87 (Jetson Orin)")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except (ImportError, OSError) as exc:
        pytest.skip(f"optional CUDA extension is not available: {exc}")
    if not hasattr(fvk,"pi05_strict_visual_available"):
        pytest.skip("Rebuild with FLASHRT_ENABLE_PI05_STRICT_VISUAL_SM87=ON")
    if not fvk.pi05_strict_visual_available():
        pytest.skip("Build with FLASHRT_ENABLE_PI05_STRICT_VISUAL_SM87=ON")
    return torch,fvk

@pytest.mark.parametrize("rows,cols,image_rows", [(2,256,1),(7,64,0),(558,2048,512),(565,2048,512)])
def test_projector_residual_quantization_scales_and_graph(rows,cols,image_rows):
    torch,k = _cuda_kernel()
    torch.manual_seed(61)
    x = torch.randn(rows,cols,device="cuda",dtype=torch.bfloat16)
    bias = torch.randn(cols,device="cuda",dtype=torch.bfloat16)
    weight = torch.randn_like(bias)
    a,b = torch.empty_like(x),torch.empty_like(x)
    qa = torch.empty(rows,cols,device="cuda",dtype=torch.int8)
    qb = torch.empty_like(qa)
    sa,sb = torch.empty(rows,device="cuda"),torch.empty(rows,device="cuda")
    def original():
        stream = torch.cuda.current_stream().cuda_stream
        a.copy_(x)
        if image_rows:
            k.add_bias_bf16(a.data_ptr(),bias.data_ptr(),image_rows,cols,stream)
        k.rms_norm_int8_rowwise(a.data_ptr(),weight.data_ptr(),qa.data_ptr(),sa.data_ptr(),
                               rows,cols,1e-6,stream)
    def candidate():
        b.copy_(x)
        k.pi05_projector_bias_rms_norm_int8(
            b.data_ptr(),bias.data_ptr(),weight.data_ptr(),qb.data_ptr(),sb.data_ptr(),
            rows,cols,image_rows,1e-6,torch.cuda.current_stream().cuda_stream)
    def check():
        torch.cuda.synchronize()
        assert torch.equal(a,b)
        assert torch.equal(qa,qb)
        assert torch.equal(sa,sb)
        assert torch.equal(b[image_rows:],x[image_rows:])
    original()
    candidate()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    x.mul_(0.3)
    bias.neg_()
    weight.mul_(0.75)
    original()
    graph.replay()
    check()

@pytest.mark.parametrize("rows,hq,hk,hv", [(3,7,5,3),(512,1152,1152,1152)])
def test_qkv_bias_split_matches_unfused(rows,hq,hk,hv):
    torch,k = _cuda_kernel()
    x = torch.randn(rows,hq+hk+hv,device="cuda",dtype=torch.bfloat16)
    bias = torch.randn(hq+hk+hv,device="cuda",dtype=torch.bfloat16)
    z = x.clone()
    ref = [torch.empty(rows,h,device="cuda",dtype=torch.bfloat16) for h in (hq,hk,hv)]
    out = [torch.empty_like(t) for t in ref]
    stream = torch.cuda.current_stream().cuda_stream
    k.add_bias_bf16(z.data_ptr(),bias.data_ptr(),rows,hq+hk+hv,stream)
    k.qkv_split(z.data_ptr(),*[t.data_ptr() for t in ref],rows,hq,hk,hv,stream)
    k.qkv_split_bias_bf16(x.data_ptr(),bias.data_ptr(),*[t.data_ptr() for t in out],
                         rows,hq,hk,hv,stream)
    torch.cuda.synchronize()
    assert all(torch.equal(a,b) for a,b in zip(ref,out))

@pytest.mark.parametrize("rows,cols,image_rows", [(0,64,0),(2,63,1),(2,9000,1),(2,64,3)])
def test_projector_rejects_invalid_shapes(rows,cols,image_rows):
    torch,k = _cuda_kernel()
    storage = torch.empty(20000,device="cuda",dtype=torch.bfloat16)
    ptr = storage.data_ptr()
    with pytest.raises(ValueError,match="pi05_projector_bias_rms_norm_int8"):
        k.pi05_projector_bias_rms_norm_int8(ptr,ptr,ptr,ptr,ptr,rows,cols,image_rows,1e-6,0)


@pytest.mark.parametrize("available,capability", [(False, None), (True, (12, 0))])
def test_non_target_environment_skips_before_extension_import(monkeypatch, available, capability):
    import builtins
    torch = pytest.importorskip("torch")
    original = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_rt" and "flash_rt_kernels" in (fromlist or ()):
            pytest.fail("non-target environment attempted to import the CUDA extension")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(pytest.skip.Exception):
        _cuda_kernel()


@pytest.mark.parametrize("feature", ["missing_extension", "missing_probe", "disabled", "enabled"])
def test_sm87_optional_feature_probe(monkeypatch, feature):
    import builtins
    torch = pytest.importorskip("torch")
    original = builtins.__import__
    kernels = SimpleNamespace()
    if feature in ("disabled", "enabled"):
        kernels.pi05_strict_visual_available = lambda: feature == "enabled"

    def import_extension(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_rt" and "flash_rt_kernels" in (fromlist or ()):
            if feature == "missing_extension":
                raise ImportError("optional extension not built")
            return SimpleNamespace(flash_rt_kernels=kernels)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 7))
    monkeypatch.setattr(builtins, "__import__", import_extension)
    if feature == "enabled":
        assert _cuda_kernel() == (torch, kernels)
    else:
        with pytest.raises(pytest.skip.Exception):
            _cuda_kernel()
