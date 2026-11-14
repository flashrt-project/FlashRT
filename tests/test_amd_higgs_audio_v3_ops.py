"""RDNA numerical contracts, checked against independent Torch/CPU math."""
import math

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def rdna_device():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a supported RDNA ROCm device")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1100", "gfx1151", "gfx1201"):
        pytest.skip("requires gfx1100/gfx1151/gfx1201")
    pytest.importorskip("triton")
    torch.manual_seed(1234)


def test_ocp_fp8_encoding_and_all_byte_values():
    import triton
    import triton.language as tl
    from flash_rt.amd.hardware.rdna.linear import _decode_e4m3, _encode_e4m3

    @triton.jit
    def convert(B, X, D, E):
        i = tl.arange(0, 256)
        tl.store(D + i, _decode_e4m3(tl.load(B + i)))
        tl.store(E + i, _encode_e4m3(tl.load(X + i)))

    bits = torch.arange(256, dtype=torch.uint8)
    expected = bits.view(torch.float8_e4m3fn).float()
    values = torch.linspace(-448, 448, 256)
    values[:12] = torch.tensor([0.0, -0.0, 1/1024, 3/1024, 5/1024,
                                15/1024, -1/1024, -3/1024, 1, 1.0625, 1.1875, 448])
    decoded = torch.empty(256, device="cuda", dtype=torch.float32)
    encoded = torch.empty(256, device="cuda", dtype=torch.uint8)
    convert[(1,)](bits.cuda(), values.cuda(), decoded, encoded)
    torch.testing.assert_close(decoded.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(encoded.cpu(), values.to(torch.float8_e4m3fn).view(torch.uint8))


def test_higgs_per_tensor_weights_match_cuda_bytes_and_scale():
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _pack_weight as pack_weight
    from flash_rt.frontends.torch._higgs_audio_v3_fp8 import _qw

    shape = (6144, 2560)
    weight = (torch.randn(*shape, device="cuda") / math.sqrt(shape[1])).bfloat16()
    # Unequal row ranges detect accidental per-channel quantization.
    weight[0].mul_(4)
    expected, scale = _qw(weight)
    actual = pack_weight(weight, fp8=True)
    assert torch.equal(actual.values, expected.view(torch.uint8))
    assert torch.equal(actual.scale, torch.full((shape[0],), scale, device="cuda"))
    unchanged = pack_weight(weight, fp8=False)
    assert unchanged.values is weight and unchanged.scale is None


def test_higgs_zero_weight_quantization_is_finite():
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _pack_weight as pack_weight

    packed = pack_weight(torch.zeros(3, 2560, device="cuda", dtype=torch.bfloat16), fp8=True)
    assert bool(torch.isfinite(packed.scale).all()) and bool((packed.scale > 0).all())
    assert not bool(packed.values.any())


def test_higgs_small_nonzero_weight_scale_matches_cuda():
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _pack_weight as pack_weight
    from flash_rt.frontends.torch._higgs_audio_v3_fp8 import _qw

    weight = (torch.randn(17, 32, device="cuda") * 1e-15).bfloat16()
    expected, scale = _qw(weight)
    actual = pack_weight(weight, fp8=True)
    assert torch.equal(actual.values, expected.view(torch.uint8))
    assert torch.equal(actual.scale, torch.full((17,), scale, device="cuda"))


def test_fused_residual_norm_preserves_sum_until_normalization():
    from flash_rt.amd.hardware.rdna.transformer import rms_norm

    # Half-way BF16 additions make premature residual rounding observable.
    x = torch.ones(2, 2560, dtype=torch.bfloat16)
    x[:, 1::2] = 0.5
    update = torch.full_like(x, 1 / 256)
    update[1].neg_()
    weight = torch.ones(2560, dtype=torch.bfloat16)
    total = x.double() + update.double()
    expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    rounded = total.bfloat16().double()
    premature = (rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    assert not torch.equal(expected, premature)
    device_x, device_update, device_weight = x.cuda(), update.cuda(), weight.cuda()
    out = torch.empty_like(device_x)
    rms_norm(device_x, device_weight, out, eps=1e-6, residual=device_update)
    torch.testing.assert_close(device_x.cpu(), total.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_rdna_precision_selection_matches_actual_device(monkeypatch):
    from contextlib import nullcontext
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import HiggsAudioV3TorchFrontendAmdRdna

    # Exercise real device/target selection without loading a checkpoint.
    monkeypatch.setattr(HiggsAudioV3TorchFrontendAmdRdna, "_load_weights", lambda self: None)
    native = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"
    for requested, expected in ((None, native), (True, native), (False, False)):
        warning = pytest.warns(RuntimeWarning, match="falling back to BF16") if requested and not native else nullcontext()
        with warning:
            frontend = HiggsAudioV3TorchFrontendAmdRdna(
                "unused", fp8=requested, alloc_own_forward_buffers=False)
        assert frontend.fp8 is expected
        assert frontend.precision_spec["weights"] == ("float8_e4m3fn" if expected else "bfloat16")
        assert frontend._linear.fp8_activations is expected


@pytest.mark.parametrize("rows,n,k", [(1, 257, 2560), (1, 6144, 2560), (1, 2560, 9728),
                                      (17, 257, 2560), (128, 257, 2560)])
def test_w8a8_linear_independent_quantization_and_graph(rows, n, k):
    from flash_rt.amd.hardware.rdna.linear import RdnaLinear
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _pack_weight

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201":
        pytest.skip("Higgs W8A8 requires RDNA4")
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).bfloat16()
    packed = _pack_weight(w, fp8=True)
    scratch = (torch.empty_like(x, dtype=torch.uint8), torch.empty(rows, device="cuda"))
    out = torch.empty(rows, n, device="cuda", dtype=torch.float32)
    linear = RdnaLinear(fp8_activations=True, fp8_long_k_rows=4)
    with pytest.raises(ValueError, match="quantization scratch"):
        linear(x, packed, out)
    linear(x, packed, out, quant_scratch=scratch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear(x, packed, out, quant_scratch=scratch)
    # Independent CPU quantization: do not derive the expected dot from the
    # implementation's output scratch. A BF16 activation accidentally used by
    # the M=1 branch fails this test, as does a stale graph scale.
    for multiplier in (1.0, 7.0, 0.0):
        x.mul_(multiplier)
        a = x.cpu().float()
        scale = (a.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
        aq = (a / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        wf = w.cpu().float()
        ws = float(wf.abs().amax() / 448)
        wq = (wf / ws).clamp(-448, 448).to(torch.float8_e4m3fn)
        expected = (aq.double() @ wq.double().t()) * (scale * ws).double()
        graph.replay()
        torch.testing.assert_close(scratch[0].cpu(), aq.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scratch[1].cpu(), scale[:, 0], rtol=1e-6, atol=0)
        torch.testing.assert_close(out.cpu(), expected.float(), rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("n,k", [(257, 2560), (6144, 2560), (2560, 9728)])
def test_real_bf16_projection_gemv(n, k):
    from flash_rt.amd.hardware.rdna.linear import RdnaLinear, pack_weight
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") / math.sqrt(k)).bfloat16()
    packed = pack_weight(w, fp8=False)
    reconstructed = w.cpu().float()
    expected = torch.mv(reconstructed, x[0].cpu().float()).bfloat16()[None]
    out = torch.empty(1, n, device="cuda", dtype=torch.bfloat16)
    RdnaLinear()(x, packed, out)
    torch.testing.assert_close(out.cpu(), expected, atol=0.016, rtol=0.01)


@pytest.mark.parametrize("rows", [1, 17])
def test_bf16_linear_supports_fp32_output(rows):
    from flash_rt.amd.hardware.rdna.linear import RdnaLinear, pack_weight
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(257, 2560, device="cuda") / 50).bfloat16()
    out = torch.empty(rows, 257, device="cuda", dtype=torch.float32)
    expected = (x.cpu().double() @ w.cpu().double().t()).float()
    RdnaLinear()(x, pack_weight(w, fp8=False), out)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=2e-5)
    assert not torch.equal(out, out.bfloat16().float())


@pytest.mark.parametrize("native_fp8", [False, True])
def test_prefill_capacity_reuse_preserves_actual_rows_and_decode_storage(native_fp8):
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import HiggsAudioV3TorchFrontendAmdRdna
    from flash_rt.amd.hardware.rdna.linear import RdnaLinear
    frontend = object.__new__(HiggsAudioV3TorchFrontendAmdRdna)
    frontend.device = torch.device("cuda")
    frontend._cfg = dict(hidden=2560, num_q_heads=32, num_kv_heads=8,
                         head_dim=128, intermediate=9728, num_codebooks=8, codebook_vocab=1026)
    frontend._linear = RdnaLinear(fp8_activations=native_fp8)
    frontend._prefill_linear = frontend._linear
    frontend._prefill_storage = frontend._prefill_buffers = None
    frontend._decode_buffers = frontend._allocate_batch(1)
    decode_pointer = frontend._decode_buffers["h"].data_ptr()
    large = frontend._prefill_batch(17)
    for rows in (2, 1, 17):
        current = frontend._prefill_batch(rows)
        assert current["h"].shape == (rows, 2560)
        assert current["q"].shape == (rows, 32, 128)
        assert current["logits"].shape == (rows, 8208)
        assert current["logits"].dtype == torch.bfloat16
        for name in ("h", "q", "logits", "activation", "gate_up"):
            assert current[name].is_contiguous()
            assert current[name].data_ptr() == (frontend._decode_buffers if rows == 1 else large)[name].data_ptr()
        if native_fp8 and rows > 1:
            assert current["quant_h"][0].shape == (rows, 2560)
            assert current["quant_h"][1].shape == (rows,)
        if rows == 1:
            assert current["attention_workspace"] is frontend._decode_buffers["attention_workspace"]
    grown = frontend._prefill_batch(32)
    assert grown["h"].shape == (32, 2560)
    assert frontend._decode_buffers["h"].data_ptr() == decode_pointer


@pytest.mark.parametrize("rows", [1, 17])
def test_native_head_rounds_before_greedy_sampling(rows):
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import HiggsAudioV3TorchFrontendAmdRdna
    from flash_rt.amd.hardware.rdna.linear import RdnaLinear, pack_weight
    frontend = object.__new__(HiggsAudioV3TorchFrontendAmdRdna)
    frontend.device = torch.device("cuda")
    frontend._cfg = dict(hidden=2560, num_q_heads=32, num_kv_heads=8,
                         head_dim=128, intermediate=9728, num_codebooks=1, codebook_vocab=2)
    frontend._linear = RdnaLinear()
    buffers = frontend._allocate_batch(rows)
    x = torch.zeros(rows, 2560, device="cuda", dtype=torch.bfloat16)
    x[:, :2] = 1
    weight = torch.zeros(2, 2560, device="cuda", dtype=torch.bfloat16)
    weight[:, 0] = 1
    weight[1, 1] = 1 / 256
    # The second logit is larger in FP32, but the native BF16 boundary
    # rounds [1, 1 + 1/256] to a tie. Greedy must select the first token.
    expected_fp32 = x.cpu().float() @ weight.cpu().float().t()
    assert bool((expected_fp32.argmax(-1) == 1).all())
    frontend._linear(x, pack_weight(weight, fp8=False), buffers["logits"])
    assert torch.equal(buffers["logits"].cpu(), expected_fp32.bfloat16())
    assert bool((buffers["logits"].argmax(-1) == 0).all())


@pytest.mark.parametrize("rows,base,split", [
    (1, 0, False), (1, 511, False), (13, 37, False), (17, 37, False),
    (1, 0, True), (1, 2047, True),
])
def test_causal_gqa_and_stale_cache(rows, base, split):
    from flash_rt.amd.hardware.rdna.cached_attention import cached_attention, allocate_decode_workspace
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.full((4096, 8, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k).fill_(float("nan"))
    end = base + rows
    k[:end].normal_(); v[:end].normal_()
    position = torch.tensor(base, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    workspace = allocate_decode_workspace(32, 128, q.device) if split else None
    cached_attention(q, k, v, position, out, workspace=workspace)
    kc = k[:end].cpu().float().repeat_interleave(4, 1)
    vc = v[:end].cpu().float().repeat_interleave(4, 1)
    score = torch.einsum("qhd,khd->hqk", q.cpu().float(), kc) / math.sqrt(128)
    mask = torch.arange(end)[None, :] > base + torch.arange(rows)[:, None]
    score.masked_fill_(mask[None], -float("inf"))
    expected = torch.einsum("hqk,khd->qhd", score.softmax(-1), vc).bfloat16()
    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out.cpu(), expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("split", [False, True])
def test_attention_graph_reads_changed_device_position(split):
    from flash_rt.amd.hardware.rdna.cached_attention import cached_attention, allocate_decode_workspace
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out, reference = torch.empty_like(q), torch.empty_like(q)
    position = torch.tensor(0, device="cuda", dtype=torch.int32)
    workspace = allocate_decode_workspace(32, 128, q.device) if split else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cached_attention(q, k, v, position, out, workspace=workspace)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cached_attention(q, k, v, position, out, workspace=workspace)
    torch.cuda.current_stream().wait_stream(stream)
    first = None
    for base in (0, 79, 193):
        position.fill_(base)
        graph.replay()
        cached_attention(q, k, v, position, reference, workspace=workspace)
        assert torch.equal(out, reference)
        if first is None:
            first = out.clone()
        else:
            assert not torch.equal(out, first)


def _uniform_reference(seed, step, cb):
    mask = (1 << 64) - 1
    x = seed ^ ((step * 0xD2B74407B1CE6E93) & mask) ^ ((cb * 0xCA5A826395121157) & mask)
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return ((x >> 40) + 0.5) / (1 << 24)


@pytest.mark.parametrize("temperature,seed,step", [
    (0.0, 0, 0), (1.0, 1234, 0), (1.0, 1234, 1), (1.0, (1 << 64) - 1, 7),
])
def test_sampling_delay_seed_and_embedding(temperature, seed, step):
    from flash_rt.amd.models.higgs_audio_v3.sampling import delayed_codebook_sample_embed_bf16
    logits = torch.zeros((8, 1026), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((8 * 1026, 2560), device="cuda", dtype=torch.bfloat16)
    codes = torch.empty(8, device="cuda", dtype=torch.long)
    embed = torch.empty(1, 2560, device="cuda", dtype=torch.bfloat16)
    delayed_codebook_sample_embed_bf16(
        logits, table, codes, embed, delay=3, boc=1024, temperature=temperature, seed=seed, step=step)
    expected = [min(1025, int(_uniform_reference(seed, step, cb) * 1026))
                if temperature else 0 for cb in range(8)]
    expected[4:] = [1024] * 4
    assert codes.cpu().tolist() == expected
    accumulator = torch.zeros(2560)
    cpu_table = table.cpu()
    for cb, code in enumerate(expected):
        accumulator += cpu_table[cb * 1026 + code].float()
    assert torch.equal(embed.cpu()[0], accumulator.bfloat16())


@pytest.mark.parametrize("temperature,dtype", [(0.2, torch.bfloat16), (1.0, torch.float32)])
def test_nonuniform_categorical_sampling_matches_cpu(temperature, dtype):
    from flash_rt.amd.models.higgs_audio_v3.sampling import delayed_codebook_sample_embed_bf16
    logits = torch.tensor([[1.001, -0.5001, 2.004, -4.0001]] * 8, dtype=dtype, device="cuda")
    codes = torch.empty(8, device="cuda", dtype=torch.long)
    embed = torch.empty(1, 16, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros(32, 16, device="cuda", dtype=torch.bfloat16)
    delayed_codebook_sample_embed_bf16(
        logits, table, codes, embed, delay=8, boc=1024,
        temperature=temperature, seed=1234, step=0)
    cumulative = (logits.cpu().float()[0] / temperature).softmax(0).cumsum(0)
    expected = [int(torch.searchsorted(cumulative, _uniform_reference(1234, 0, cb))) for cb in range(8)]
    assert codes.cpu().tolist() == expected


def test_swiglu_rounding_boundary():
    from flash_rt.amd.hardware.rdna.transformer import silu_mul
    gate = torch.linspace(-6, 6, 256).bfloat16()
    up = torch.linspace(0.5, 3, 256).bfloat16()
    expected = torch.nn.functional.silu(gate) * up
    single_round = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    assert not torch.equal(expected, single_round)
    merged = torch.cat([gate, up])[None].cuda()
    out = torch.empty(1, 256, device="cuda", dtype=torch.bfloat16)
    silu_mul(merged, out)
    assert torch.equal(out.cpu()[0], expected)


def test_qk_norm_rope_writes_only_the_requested_kv_positions():
    from flash_rt.amd.hardware.rdna.transformer import qk_norm_rope_cache
    rows, nq, nk, d, base = 3, 32, 8, 128, 19
    merged = torch.randn(rows, (nq + 2 * nk) * d, device="cuda", dtype=torch.bfloat16)
    qw = torch.randn(d, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    angles = torch.randn(64, 64, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    q = torch.empty(rows, nq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.full((64, nk, d), float("nan"), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k).fill_(float("nan"))
    pos = torch.tensor(base, device="cuda", dtype=torch.int32)
    qk_norm_rope_cache(merged, qw, kw, cos, sin, pos, q, k, v, eps=1e-6)
    raw = merged.cpu().float().view(rows, nq + 2 * nk, d)
    for values, norm, actual in ((raw[:, :nq], qw, q), (raw[:, nq:nq + nk], kw, k[base:base + rows])):
        normalized = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                      * norm.cpu().float()).bfloat16().float()
        rotate = torch.cat([-normalized[..., d // 2:], normalized[..., :d // 2]], -1)
        c = cos[base:base + rows].cpu().float().repeat(1, 2)[:, None]
        s = sin[base:base + rows].cpu().float().repeat(1, 2)[:, None]
        expected = (normalized * c + rotate * s).bfloat16()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.016)
    assert torch.equal(v[base:base + rows].cpu(), raw[:, nq + nk:].bfloat16())
    assert bool(torch.isnan(k[:base]).all()) and bool(torch.isnan(k[base + rows:]).all())


def test_qk_norm_keeps_bf16_boundary_before_rope():
    """An exact staged CPU reference distinguishes the missing norm rounding."""
    from flash_rt.amd.hardware.rdna.transformer import qk_norm_rope_cache
    rows, nq, nk, d = 1, 32, 8, 128
    raw = torch.tensor([1., 2., 3., 4.]).repeat(rows, nq + 2 * nk, d // 4).bfloat16()
    weight = torch.linspace(.5, 1.5, d).bfloat16()
    cos = torch.full((1, d // 2), .765625, dtype=torch.bfloat16)
    sin = torch.full_like(cos, .64453125)
    q = torch.empty(rows, nq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(rows, nk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    qk_norm_rope_cache(raw.flatten(1).cuda(), weight.cuda(), weight.cuda(),
                       cos.cuda(), sin.cuda(), torch.zeros((), device="cuda", dtype=torch.int32),
                       q, k, v, eps=1e-6)
    values = raw.double()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6) * weight.double()
    def rotate(x):
        partner = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
        return (x * cos.double().repeat(1, 2) + partner * sin.double().repeat(1, 2)).bfloat16()
    expected = rotate(normalized.bfloat16().double())
    assert not torch.equal(expected, rotate(normalized)), "fixture must detect omitted rounding"
    assert torch.equal(q.cpu(), expected[:, :nq])
    assert torch.equal(k.cpu(), expected[:, nq:nq + nk])
    assert torch.equal(v.cpu(), raw[:, nq + nk:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sampling_cdf_endpoint_uses_last_positive_probability(dtype):
    from flash_rt.amd.models.higgs_audio_v3.sampling import delayed_codebook_sample_embed_bf16

    logits = torch.tensor([[0., 0., -200., -200.]], device="cuda", dtype=dtype)
    table = torch.arange(4, device="cuda", dtype=torch.bfloat16).reshape(4, 1)
    codes = torch.empty(1, device="cuda", dtype=torch.long)
    embed = torch.empty(1, 1, device="cuda", dtype=torch.bfloat16)
    for temperature, expected in ((1.0, 1), (0.0, 0)):
        delayed_codebook_sample_embed_bf16(
            logits, table, codes, embed, delay=1, boc=1024,
            temperature=temperature, seed=3558559446808474027, step=0)
        assert codes.item() == expected
        assert embed.item() == expected
