"""N1.7 geometries and shared Pi0.5/RDNA transformer numerical contracts."""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def rdna_device():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm and an RDNA GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1100", "gfx1151", "gfx1201"):
        pytest.skip("requires gfx1100, gfx1151 or gfx1201")


@pytest.mark.parametrize("batch,sq,sk,hq,hk,dim,causal", [
    (2, 256, 256, 16, 16, 64, False),
    (1, 461, 461, 16, 8, 128, True),
    (1, 461, 461, 32, 32, 64, False),
    (1, 41, 41, 32, 32, 48, False),
    (1, 41, 13, 32, 32, 48, False),
    (1, 41, 448, 32, 32, 48, False),
    (1, 1, 1, 16, 8, 128, True),
    (1, 17, 65, 4, 2, 48, False),
])
def test_attention_reference_and_graph(batch, sq, sk, hq, hk, dim, causal):
    from flash_rt.amd.hardware.rdna.attention import attention_bf16
    torch.manual_seed(17)
    q = torch.randn(batch, sq, hq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, sk, hk, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_heads = k.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    v_heads = v.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    scores = q.float().transpose(1, 2) @ k_heads.transpose(-1, -2) * dim ** -0.5
    if causal:
        mask = torch.ones(sq, sk, device="cuda", dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -torch.inf)
    reference = (scores.softmax(-1) @ v_heads).transpose(1, 2)
    actual = attention_bf16(q, k, v, causal=causal)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    assert torch.nn.functional.cosine_similarity(actual.float().flatten(), reference.flatten(), dim=0) > 0.9999
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attention_bf16(q, k, v, causal=causal)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = attention_bf16(q, k, v, causal=causal)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, actual)


@pytest.mark.parametrize('width,mode', [(128, 0), (2048, 0), (1536, 1), (1536, 2), (4096, 2)])
def test_fused_normalization(width, mode):
    from flash_rt.amd.hardware.rdna.normalization import normalize
    torch.manual_seed(21)
    x = torch.randn(41, width, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    for use_residual in (False, True):
        summed = (x + residual) if use_residual else x
        if mode == 0:
            ref = (summed.float() * (summed.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).bfloat16() * weight
        else:
            ref = torch.nn.functional.layer_norm(summed, (width,), eps=1e-5)
            if mode == 2:
                ref = ref * (1 + weight) + bias
        actual, total = normalize(x, weight if mode != 1 else None, bias if mode == 2 else None, residual=residual if use_residual else None, kind="rms" if mode == 0 else "layer",
                                  weight_offset=1.0 if mode == 2 else 0.0,
                                  round_affine=True)
        if use_residual:
            assert torch.equal(total, summed)
        torch.testing.assert_close(actual.float(), ref.float(), atol=0.032, rtol=0.016)
        assert torch.nn.functional.cosine_similarity(actual.double().flatten(), ref.double().flatten(), dim=0) > 0.99999


def test_strided_rope():
    from flash_rt.amd.hardware.rdna.normalization import rope
    x = torch.randn(67, 3, 16, 64, device='cuda', dtype=torch.bfloat16)[:, 0]
    cos, sin = [torch.randn(67, 64, device='cuda') for _ in range(2)]
    a, b = x.float().chunk(2, dim=-1)
    ref = (x.float() * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]).bfloat16()
    assert torch.equal(rope(x, cos, sin), ref)


def _check_graph_updates(run, update, reference, *, atol=0.032, rtol=0.016):
    """Warm, capture, and check changed inputs against an independent formula."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    update()
    expected = reference()
    eager = run().clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    torch.testing.assert_close(captured.float(), expected.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(41, 1536), (2, 17, 128)])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("strided_output", [False, True])
def test_shared_gemm_transposed_weight_layout(shape, biased, strided_output):
    from flash_rt.amd import flash_rt_amd_kernels as kernels
    from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend

    torch.manual_seed(59)
    gemm = Rdna35GemmBackend(kernels, autotune=False)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, shape[-1], device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16) if biased else None
    rows = x.numel() // shape[-1]
    storage = torch.empty(rows, 512 if strided_output else 256, device="cuda", dtype=x.dtype)
    out = storage[:, :256]

    def run():
        gemm.linear(out, x.reshape(rows, -1), weight, bias, weight_transposed=True)
        return out

    def reference():
        return torch.nn.functional.linear(x, weight, bias).reshape(rows, -1)

    _check_graph_updates(run, lambda: x.mul_(0.75).add_(0.25), reference, atol=0, rtol=0)
    with pytest.raises(ValueError, match="linear shape"):
        gemm.linear(out, x.reshape(rows, -1), weight[:, :-1], bias, weight_transposed=True)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("offset,rounded", [(0.0, True), (1.0, False)])
def test_shared_rms_contracts_and_graph_updates(fused, offset, rounded):
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
    # Exercise the shared primitive without a native extension or model weights.
    ops = Rdna35TensorOps(None)
    torch.manual_seed(43)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    def run():
        ops.rms_norm(out, x, weight, weight_offset=offset,
                     round_affine=rounded, fused=fused)
        return out

    def reference():
        norm = x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return norm.bfloat16() * weight if rounded else (norm * (1 + weight.float())).bfloat16()

    _check_graph_updates(run, lambda: x.mul_(0.7).add_(0.4), reference)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_shared_residual_layer_norm_graph_updates(fused, adaptive):
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
    ops = Rdna35TensorOps(None)
    torch.manual_seed(51)
    x, residual = [torch.randn(7, 1536, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    scale, shift = [torch.randn(1536, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    out, total = torch.empty_like(x), torch.empty_like(x)

    def run():
        ops.residual_layer_norm(out, total, x, residual,
                                scale if adaptive else None, shift if adaptive else None,
                                fused=fused)
        return out

    def reference():
        norm = torch.nn.functional.layer_norm(x + residual, (1536,), eps=1e-5)
        return norm * (1 + scale) + shift if adaptive else norm

    _check_graph_updates(run, lambda: residual.add_(0.3), reference)
    torch.testing.assert_close(total, x + residual, atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_rotate_half_graph_updates(fused):
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
    x = torch.randn(67, 3, 4, 64, device="cuda", dtype=torch.bfloat16)[:, 1]
    cos, sin = [torch.randn(67, 64, device="cuda") for _ in range(2)]
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    def run():
        Rdna35TensorOps.rotate_half(out, x, cos, sin, fused=fused)
        return out

    def reference():
        a, b = x.float().chunk(2, -1)
        return (x.float() * cos[:, None] + torch.cat((-b, a), -1) * sin[:, None]).bfloat16()

    _check_graph_updates(run, lambda: sin.add_(0.2), reference, atol=0, rtol=0)


def test_shared_sdpa_mask_and_graph_updates():
    from flash_rt.amd.hardware.rdna35.attention import sdpa
    torch.manual_seed(53)
    q = torch.randn(1, 9, 8, 256, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 17, 1, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    mask = torch.arange(17, device="cuda") < 13

    def reference():
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float().expand(-1, -1, 8, -1)) / 16
        scores.masked_fill_(~mask, -torch.inf)
        return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float().expand(-1, -1, 8, -1))

    _check_graph_updates(lambda: sdpa(q, k, v, mask=mask), lambda: mask.fill_(True), reference)


def test_shared_fused_output_contracts():
    from flash_rt.amd.hardware.rdna.normalization import normalize, rope
    x = torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="distinct buffers"):
        normalize(x, residual=x, out=x, out_sum=x)
    with pytest.raises(ValueError, match="output must be contiguous"):
        normalize(x, out=torch.empty(4, 128, device=x.device, dtype=x.dtype)[:, ::2])
    with pytest.raises(ValueError, match="must not alias"):
        rope(x[:, None], x, x, out=x[:, None])


@pytest.mark.parametrize("width", [1024, 1536, 4096])
@pytest.mark.parametrize("affine", [False, True])
def test_shared_layer_norm_destinations_and_fp32_reference(width, affine):
    from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps
    ops = Rdna35TensorOps(None)
    torch.manual_seed(59)
    x = torch.randn(17, width, device="cuda", dtype=torch.bfloat16)
    weight, bias = [torch.randn(width, device=x.device, dtype=x.dtype) if affine else None
                    for _ in range(2)]
    out = torch.empty_like(x)

    def reference():
        return torch.nn.functional.layer_norm(
            x.float(), (width,), None if weight is None else weight.float(),
            None if bias is None else bias.float(), eps=1e-6).bfloat16()

    _check_graph_updates(lambda: ops.layer_norm(None, x, weight, bias),
                         lambda: x.add_(0.5), reference)
    returned = ops.layer_norm(out, x, weight, bias)
    assert returned is out
    torch.testing.assert_close(out, ops.layer_norm(None, x, weight, bias), atol=0, rtol=0)


@pytest.mark.parametrize("dtype_cos,dtype_sin", [(torch.bfloat16,torch.float32),(torch.float32,torch.bfloat16)])
def test_rope_mixed_table_dtypes(dtype_cos,dtype_sin):
    from flash_rt.amd.hardware.rdna.normalization import rope
    x=torch.randn(17,4,48,device="cuda",dtype=torch.bfloat16)
    c=torch.randn(17,48,device="cuda",dtype=dtype_cos)
    s=torch.randn(17,48,device="cuda",dtype=dtype_sin)
    a,b=x.float().chunk(2,-1)
    reference=(x.float()*c.float()[:,None]+torch.cat((-b,a),-1)*s.float()[:,None]).bfloat16()
    assert torch.equal(rope(x,c,s),reference)


def test_normalization_sum_can_alias_input():
    from flash_rt.amd.hardware.rdna.normalization import normalize
    x=torch.randn(13,1536,device="cuda",dtype=torch.bfloat16)
    residual=torch.randn_like(x)
    summed=x+residual
    reference=torch.nn.functional.layer_norm(summed,(1536,),eps=1e-5)
    actual,total=normalize(x,residual=residual,kind="layer",eps=1e-5,round_affine=True,out_sum=x)
    assert torch.equal(total,summed)
    torch.testing.assert_close(actual,reference,atol=0.032,rtol=0.016)
