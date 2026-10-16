"""Alpamayo 1.5 Thor: request contracts and kernel parity with the official op sequence."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from flash_rt.models.alpamayo15.pipeline_thor import (  # noqa: E402
    ALPAMAYO15_KERNEL_SYMBOLS,
    _TrajTokenMask,
    FP8_E4M3_MAX,
    check_generation_config,
    expert_attention_spans,
    fp8_fragment_layout,
    merge_attention,
    quantize_fp8_rows,
    require_alpamayo15_kernels,
    update_stop_state,
)

BF16 = torch.bfloat16


def test_required_symbols_fail_with_build_hint():
    with pytest.raises(RuntimeError, match=r"FLASHRT_ENABLE_ALPAMAYO15=ON") as exc:
        require_alpamayo15_kernels(SimpleNamespace())
    for name in ALPAMAYO15_KERNEL_SYMBOLS:
        assert name in str(exc.value)


def test_symbol_inventory_is_atomic():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    found = {name for name in dir(fvk) if name.startswith("alpamayo15_")}
    assert found in (set(), set(ALPAMAYO15_KERNEL_SYMBOLS))
    if os.environ.get("FLASHRT_TEST_ALPAMAYO15_REQUIRED") == "1":
        assert found == set(ALPAMAYO15_KERNEL_SYMBOLS)


def _run_stop(seqs, max_new=6, traj=7, eos=frozenset({9})):
    """Feed per-step tokens (one list per step) and return the step after which generation ends."""
    n = len(seqs[0])
    traj_found, unfinished = [False] * n, [True] * n
    for step, last in enumerate(seqs, start=1):
        update_stop_state(list(last), traj_found, unfinished, step, traj_start_id=traj,
                          max_new_tokens=max_new, eos_ids=set(eos))
        if not any(unfinished):
            return step, unfinished
    return None, unfinished


def test_stop_criteria_match_official_sequence():
    # StopAfterEOS: one more token after <|traj_future_start|> (7), then stop.
    assert _run_stop([[1], [7], [3], [4]])[0] == 3
    # Generation-config EOS stops a row immediately; the budget bounds the loop.
    assert _run_stop([[1], [9]])[0] == 2
    assert _run_stop([[1]] * 6)[0] == 6
    # Batched: all rows must have produced 7 before the extra step; EOS rows finish alone.
    assert _run_stop([[7, 1], [2, 1], [3, 7], [4, 5], [5, 5]])[0] == 4
    step, unfinished = _run_stop([[9, 1], [0, 2], [0, 3]])
    assert step is None and unfinished == [False, True]


def test_expert_spans_match_official_offsets_and_mask():
    off_mod = pytest.importorskip("alpamayo1_5.models.alpamayo1_5")
    official = off_mod.Alpamayo1_5
    P, traj, pad = 5, 7, 0
    # rows: found early (later tokens masked), found at the next-to-last step, never found
    tokens = [[1, 7, 3, 4, 2], [1, 2, 3, 7, 4], [1, 2, 3, 4, 5]]
    offsets, visible = expert_attention_spans(tokens, P, traj)
    seq = torch.tensor([[9] * P + row for row in tokens])
    ref_off = official._find_eos_offset(seq, traj, torch.device("cpu"), warn=False)
    assert offsets == ref_off.tolist()
    cache_len = P + len(tokens[0]) - 1
    n_act = 4
    _, mask = official._build_expert_pos_ids_and_attn_mask(
        offset=ref_off, rope_deltas=torch.zeros(3, 1, dtype=torch.long), kv_cache_seq_len=cache_len,
        n_diffusion_tokens=n_act, b_star=3, device=torch.device("cpu"), prefix_mask=None)
    for b in range(3):  # visible cache rows of the official mask == our visible prefix
        seen = (mask[b, 0, 0, :cache_len] == 0).nonzero().squeeze(1).tolist()
        assert seen == list(range(visible[b]))
        assert (mask[b, 0, :, cache_len:] == 0).all()
    del pad


def test_merge_attention_equals_single_pass():
    torch.manual_seed(1)
    rows, heads, D, n1, n2 = 6, 4, 16, 9, 5
    q = torch.randn(rows, heads, D)
    k = torch.randn(n1 + n2, heads, D)
    v = torch.randn(n1 + n2, heads, D)

    def attend(kk, vv):
        s = torch.einsum("rhd,khd->hrk", q, kk) / D ** 0.5
        return torch.einsum("hrk,khd->rhd", s.softmax(-1), vv), torch.logsumexp(s, -1)

    full, _ = attend(k, v)
    o1, l1 = attend(k[:n1], v[:n1])
    o2, l2 = attend(k[n1:], v[n1:])
    merged = merge_attention(o1.to(BF16), l1, o2.to(BF16), l2)
    assert torch.allclose(merged.float(), full, atol=2e-2, rtol=2e-2)


def test_generation_config_guard():
    ok = SimpleNamespace(repetition_penalty=1.0, typical_p=1.0, top_k=50, min_length=0, min_new_tokens=None)
    check_generation_config(ok)
    with pytest.raises(ValueError, match="repetition_penalty"):
        check_generation_config(SimpleNamespace(repetition_penalty=1.1))
    with pytest.raises(ValueError, match="min_new_tokens"):
        check_generation_config(SimpleNamespace(min_new_tokens=4))


def test_resolver_entry_and_load_model_redirect():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP[("alpamayo15", "torch", "thor")] == (
        "flash_rt.frontends.torch.alpamayo15_thor", "Alpamayo15TorchFrontendThor")
    import flash_rt
    with pytest.raises(NotImplementedError, match="Alpamayo15TorchFrontendThor"):
        flash_rt.load_model("unused", config="alpamayo15")


def test_traj_token_mask_range():
    scores = torch.zeros(2, 20)
    out = _TrajTokenMask(5, 4)(None, scores)
    assert torch.isinf(out[:, 5:9]).all() and (out[:, 5:9] < 0).all()
    assert torch.equal(out[:, :5], torch.zeros(2, 5)) and torch.equal(out[:, 9:], torch.zeros(2, 11))


def test_fp8_row_quantization():
    w = torch.randn(5, 64) * torch.tensor([1e-3, 1.0, 30.0, 0.0, 2.0])[:, None]
    codes, scale = quantize_fp8_rows(w, chunk_rows=2)
    assert codes.dtype == torch.uint8 and scale.dtype == torch.float32
    assert scale[3] == 1.0 and not codes[3].view(torch.float8_e4m3fn).float().any()   # +-0 codes
    deq = codes.view(torch.float8_e4m3fn).float() * scale[:, None]
    nz = [0, 1, 2, 4]
    torch.testing.assert_close(deq[nz].abs().amax(1), w[nz].abs().amax(1), rtol=4e-7, atol=0)  # row max -> 448
    rel = ((deq - w).abs() / w.abs().clamp_min(1e-30))[nz]
    normal = w[nz].abs() >= scale[nz, None] * 2.0 ** -6
    assert float(rel[normal].max()) <= 2.0 ** -4 + 1e-6                     # 3 mantissa bits, RNE
    assert float(deq.abs().max() / scale.max()) <= FP8_E4M3_MAX
    # Parameters require grad: the quantized copies must not carry an autograd graph (it would pin FP32 temporaries).
    codes_g, scale_g = quantize_fp8_rows(torch.nn.Parameter(w.clone()))
    frag_g, _ = fp8_fragment_layout(codes_g, scale_g)
    assert all(t.grad_fn is None and not t.requires_grad for t in (codes_g, scale_g, frag_g))


def test_fp8_fragment_layout_matches_mma_a_fragments():
    N, K = 21, 64
    codes = torch.randint(0, 256, (N, K), dtype=torch.uint8)
    frag, scale = fp8_fragment_layout(codes, torch.arange(N, dtype=torch.float32))
    T = 2
    assert frag.numel() == T * 16 * K and scale.shape == (T * 16,) and not scale[N:].any()
    f = frag.view(T, K // 32, 32, 16)
    for t in range(T):
        for kc in range(K // 32):
            for lane in range(32):
                g, c = lane // 4, lane % 4
                for j in range(16):
                    kt, ri, e = j // 8, (j % 8) // 2, j % 2
                    row = 16 * t + g + 8 * (ri & 1)
                    col = 32 * kc + 16 * kt + 8 * (ri >> 1) + 2 * c + e
                    want = int(codes[row, col]) if row < N else 0
                    assert int(f[t, kc, lane, j]) == want


def _kernels():
    if not torch.cuda.is_available():
        pytest.skip("Alpamayo 1.5 kernel parity requires CUDA")
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels is unavailable: {exc}")
    if not all(hasattr(fvk, n) for n in ALPAMAYO15_KERNEL_SYMBOLS):
        pytest.skip("built without -DFLASHRT_ENABLE_ALPAMAYO15=ON")
    return fvk


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _rot(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor, max_frac: float = 0.01):
    """Equal up to one BF16 ulp on a small fraction of elements (FP32 reduction order)."""
    a, e = actual.float(), expected.float()
    ulp = torch.where(e == 0, torch.full_like(e, 2.0 ** -133), e.abs() * 2.0 ** -7)
    diff = (a - e).abs()
    assert bool((diff <= ulp * 1.01).all()), f"max diff {diff.max().item()}"
    assert float((diff > 0).float().mean()) <= max_frac


def _qwen3_head_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    n1 = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)
    return w * n1


def test_vision_rope_matches_official_rotary():
    fvk = _kernels()
    torch.manual_seed(3)
    S, H, D = 37, 16, 72
    qkv = (torch.randn(S, 3, H, D, device="cuda") * 0.5).to(BF16)
    angles = torch.randn(S, D // 2, device="cuda") * 3
    emb = torch.cat((angles, angles), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    ref = qkv.clone()
    for i in (0, 1):  # apply_rotary_pos_emb_vision: FP32 math, one BF16 rounding
        x = ref[:, i].float()
        ref[:, i] = (x * cos[:, None] + _rot(x) * sin[:, None]).to(BF16)
    out = qkv.clone()
    fvk.alpamayo15_vision_rope_qk_bf16(out.data_ptr(), cos.data_ptr(), sin.data_ptr(), S, H, D, _stream())
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("with_delta", [False, True])
def test_vlm_rms_matches_qwen3_rmsnorm(with_delta: bool):
    fvk = _kernels()
    torch.manual_seed(5)
    rows, dim, eps = 9, 4096, 1e-6
    res = torch.randn(rows, dim, device="cuda").to(BF16)
    delta = torch.randn(rows, dim, device="cuda").to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    ref_res = (res + delta) if with_delta else res.clone()
    ref = _qwen3_head_norm(ref_res, w, eps)
    out = torch.empty_like(res)
    fvk.alpamayo15_rms_bf16(res.data_ptr(), delta.data_ptr() if with_delta else 0, w.data_ptr(), out.data_ptr(),
                            rows, dim, eps, _stream())
    torch.cuda.synchronize()
    assert torch.equal(res, ref_res)
    _assert_bf16_close(out, ref)


def test_expert_fp32_residual_rms():
    fvk = _kernels()
    torch.manual_seed(6)
    rows, dim, eps = 64, 2048, 1e-6
    res = torch.randn(rows, dim, device="cuda")
    delta = torch.randn(rows, dim, device="cuda").to(BF16)
    w = (torch.rand(dim, device="cuda") + 0.5).to(BF16)
    ref_res = res + delta.float()                                     # fp32 + bf16 -> fp32
    hs = ref_res * torch.rsqrt(ref_res.pow(2).mean(-1, keepdim=True) + eps)
    ref = (w.float() * hs).to(BF16)                                   # autocast cast at the next linear
    out = torch.empty(rows, dim, device="cuda", dtype=BF16)
    fvk.alpamayo15_res_rms_f32_bf16(res.data_ptr(), delta.data_ptr(), w.data_ptr(), out.data_ptr(),
                                    rows, dim, eps, _stream())
    torch.cuda.synchronize()
    assert torch.equal(res, ref_res)
    _assert_bf16_close(out, ref)


@pytest.mark.parametrize("n_samples", [1, 3])
def test_expert_qk_prep_matches_autocast_sequence(n_samples):
    fvk = _kernels()
    torch.manual_seed(8)
    A, Hq, Hk, D, eps, S = 64, 16, 8, 128, 1e-6, 80
    rows = n_samples * A
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)
    qw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    kw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    q = qkv[:, :Hq * D].view(rows, Hq, D)
    k = qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D)
    v = qkv[:, (Hq + Hk) * D:].view(rows, Hk, D)

    def ref(x, w):  # BF16 head norm, then FP32 RoPE, cast to BF16 at SDPA
        n = _qwen3_head_norm(x, w, eps)
        return (n.float() * cos[:, None] + _rot(n).float() * sin[:, None]).to(BF16)

    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    kc = torch.zeros(n_samples, S, Hk, D, device="cuda", dtype=BF16)
    vc = torch.zeros_like(kc)
    slots = torch.tensor([3 + 5 * b for b in range(n_samples)], device="cuda", dtype=torch.long)
    fvk.alpamayo15_expert_qk_prep_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                       q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(), slots.data_ptr(), rows,
                                       Hq, Hk, Hk * D, A, S, eps, _stream())
    torch.cuda.synchronize()
    _assert_bf16_close(q_out, ref(q, qw))
    k_ref, written = ref(k, kw), torch.zeros(n_samples, S, dtype=torch.bool, device="cuda")
    for b, st in enumerate(slots.tolist()):  # sample b's action rows start at its slot
        _assert_bf16_close(kc[b, st:st + A], k_ref[b * A:(b + 1) * A])
        assert torch.equal(vc[b, st:st + A], v[b * A:(b + 1) * A])
        written[b, st:st + A] = True
    assert not kc[~written].any() and not vc[~written].any()


@pytest.mark.parametrize("rows,slot,batch_stride", [(11, None, 0), (1, 5, 0), (3, 5, 16)])
def test_vlm_qk_prep_matches_bf16_rope_and_cache_write(rows, slot, batch_stride):
    fvk = _kernels()
    torch.manual_seed(9)
    Hq, Hk, D, eps, S = 32, 8, 128, 1e-6, 16
    qkv = (torch.randn(rows, (Hq + 2 * Hk) * D, device="cuda") * 2).to(BF16)
    qw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    kw = (torch.rand(D, device="cuda") + 0.5).to(BF16)
    ang = torch.randn(rows, D // 2, device="cuda") * 3
    emb = torch.cat((ang, ang), -1)
    cos, sin = emb.cos().to(BF16).contiguous(), emb.sin().to(BF16).contiguous()
    q = qkv[:, :Hq * D].view(rows, Hq, D)
    k = qkv[:, Hq * D:(Hq + Hk) * D].view(rows, Hk, D)
    v = qkv[:, (Hq + Hk) * D:].view(rows, Hk, D)

    def ref(x, w):  # official BF16 decode RoPE: every product and the sum rounded to BF16
        n = _qwen3_head_norm(x, w, eps)
        return n * cos[:, None] + _rot(n) * sin[:, None]

    nb = rows if batch_stride else 1
    q_out = torch.empty(rows, Hq, D, device="cuda", dtype=BF16)
    kc = torch.zeros(nb, S, Hk, D, device="cuda", dtype=BF16)
    vc = torch.zeros_like(kc)
    d_slot = torch.tensor([slot or 0], device="cuda", dtype=torch.long)
    rpb = 1 if batch_stride else rows
    fvk.alpamayo15_vlm_qk_prep_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                    q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(),
                                    d_slot.data_ptr() if slot is not None else 0, rows, Hq, Hk, Hk * D,
                                    rpb, batch_stride, eps, _stream())
    torch.cuda.synchronize()
    start = slot or 0
    written = torch.zeros(nb, S, dtype=torch.bool, device="cuda")
    if batch_stride:  # row b -> sample b, cache row `slot`
        k_got, v_got = kc[:, start], vc[:, start]
        written[:, start] = True
    else:             # rows land contiguously from `slot`
        k_got, v_got = kc[0, start:start + rows], vc[0, start:start + rows]
        written[0, start:start + rows] = True
    _assert_bf16_close(q_out, ref(q, qw))
    _assert_bf16_close(k_got, ref(k, kw))
    assert torch.equal(v_got, v)
    assert not kc[~written].any() and not vc[~written].any()


@pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (3, 40, 64), (8, 6144, 4096), (9, 1000, 12288), (16, 155, 4096),
                                   (1, 65537, 64), (12, 65553, 32)])   # last two: wide (2 tiles/warp) path, odd tile count
def test_w8a16_gemm_matches_dequantized_reference(M, N, K):
    fvk = _kernels()
    torch.manual_seed(M * 7 + N)
    w = (torch.randn(N, K, device="cuda") * 0.02).to(BF16)
    x = torch.randn(M, K, device="cuda").to(BF16)
    codes, scale = quantize_fp8_rows(w)
    frag, scale_p = fp8_fragment_layout(codes, scale)
    y = torch.full((M, N), float("nan"), device="cuda", dtype=BF16)
    fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr(), scale_p.data_ptr(), x.data_ptr(), y.data_ptr(), M, N, K, _stream())
    torch.cuda.synchronize()
    ref = x.float() @ (codes.view(torch.float8_e4m3fn).float() * scale[:, None]).t()
    err = (y.float() - ref).abs()
    # FP32 accumulation in a different order, then one BF16 rounding of the output.
    assert bool((err <= ref.abs() * 2.0 ** -8 + 1e-3 * ref.abs().max()).all()), float(err.max())


def test_w8a16_gemm_rejects_unsupported_shapes():
    fvk = _kernels()
    frag = torch.zeros(16 * 64, device="cuda", dtype=torch.uint8)
    scale = torch.zeros(16, device="cuda")
    x = torch.zeros(17, 64, device="cuda", dtype=BF16)
    y = torch.zeros(17, 16, device="cuda", dtype=BF16)
    for M, K, off in ((17, 64, 0), (1, 48, 0), (1, 64, 8)):   # too many rows, K % 32, misaligned weights
        with pytest.raises(Exception, match="alpamayo15_w8a16_gemm_bf16"):
            fvk.alpamayo15_w8a16_gemm_bf16(frag.data_ptr() + off, scale.data_ptr(), x.data_ptr(), y.data_ptr(),
                                           M, 16, K, _stream())
