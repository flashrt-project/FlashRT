"""Skinny FP8 decoder GEMM family on the Pi0.5 RTX frontend (sm_120a builds).

Runs on RTX 5090 with a GPU_ARCH=120 kernel build. Skipped when the GPU,
the kernel family or the pi05_libero PyTorch checkpoint is unavailable.
Set ``PI05_LIBERO_PYTORCH_CHECKPOINT`` to override the checkpoint directory.

Invariants covered:
  - the K-split GEMM plus its residual consumer reproduces an FP32 matmul
    of the same FP8 operands within FP32 summation-order noise, for the
    four decoder shapes at chunk-sized and batch-folded row counts, with
    and without programmatic dependent launch;
  - the fused consumers are bit-identical to the unfused kernels they
    replace (gated residual + adaptive norm + FP8 quantize, GeGLU + FP8
    quantize, QKV split + RoPE + cache write) when fed the same values;
  - the frontend on the family matches the cuBLASLt decoder on the same
    prompt, images and noise (cosine gate), and 300 graph replays of the
    same input are bit-identical (no race through the PDL chain);
  - the batched pipeline on the family matches the single pipeline per
    slot.

Run::

    python -m pytest tests/test_pi05_decoder_skinny.py -v
"""

import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
PROMPT = "pick up the black bowl and place it on the plate"

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)


def _family_available() -> bool:
    if not _GPU_AVAILABLE:
        return False
    import importlib
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except ModuleNotFoundError as exc:
        if exc.name == "flash_rt.flash_rt_kernels":
            return False
        raise
    probe = getattr(fvk, "pi05_dec_skinny_available", None)
    return bool(probe is not None and probe())


requires_family = pytest.mark.skipif(
    not _family_available(), reason="needs CUDA sm_120a and the skinny decoder build")
requires_ckpt = pytest.mark.skipif(
    not (_family_available() and _CKPT_AVAILABLE),
    reason=f"needs the skinny decoder build and the pi05 ckpt at {CKPT_PI05}")

SHAPES = [("qkv", 2560, 1024, 3), ("o", 1024, 2048, 6), ("gate_up", 8192, 1024, 5), ("down", 1024, 4096, 5)]


def _fp8(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _make_image(rng: np.random.Generator) -> np.ndarray:
    """Smooth synthetic camera frame: low-frequency shading plus a few
    blocks, closer to a scene than white noise (which drives the FP8
    calibration to unrepresentative scales)."""
    yy, xx = np.mgrid[0:224, 0:224] / 224.0
    img = np.zeros((224, 224, 3), np.float32)
    for c in range(3):
        fx, fy, ph = rng.uniform(0.5, 3.0, 3)
        img[..., c] = 0.5 + 0.35 * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    for _ in range(4):
        y0, x0 = rng.integers(0, 160, 2)
        h, w = rng.integers(20, 64, 2)
        img[y0:y0 + h, x0:x0 + w] = rng.uniform(0.1, 0.9, 3)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "image": _make_image(rng),
        "wrist_image": _make_image(rng),
        "state": rng.random(8, dtype=np.float32),
    }


@requires_family
@pytest.mark.parametrize("rows", [10, 80])
@pytest.mark.parametrize("pdl", [False, True])
def test_gemm_matches_fp32_matmul(rows, pdl):
    from flash_rt import flash_rt_kernels as fvk
    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    for name, N, K, cfg in SHAPES:
        assert fvk.pi05_dec_skinny_config_supports(cfg, N, K)
        splits = K // fvk.pi05_dec_skinny_config_k_chunk(cfg)
        A = _fp8(torch.randn(rows, K, device="cuda") * 0.5)
        W = _fp8(torch.randn(N, K, device="cuda") * 0.05)
        a_scale = torch.tensor([0.02], device="cuda"); w_scale = torch.tensor([0.01], device="cuda")
        partials = torch.empty(splits, rows, N, device="cuda")
        residual = torch.zeros(rows, N, dtype=torch.bfloat16, device="cuda")
        gate = torch.ones(rows, N, dtype=torch.bfloat16, device="cuda")
        with torch.cuda.stream(stream):
            rc = fvk.pi05_dec_skinny_gemm(A.data_ptr(), W.data_ptr(), partials.data_ptr(), rows, N, K, cfg, pdl, stream.cuda_stream)
            assert rc == 0, (name, rc)
            rc = fvk.pi05_dec_skinny_residual_gate_mul(partials.data_ptr(), splits, a_scale.data_ptr(), w_scale.data_ptr(),
                                                  residual.data_ptr(), gate.data_ptr(), rows, N, pdl, stream.cuda_stream)
            assert rc == 0, (name, rc)
        torch.cuda.synchronize()
        ref = (A.float() @ W.float().T) * (a_scale * w_scale)
        err = (residual.float() - ref).abs().max().item() / ref.abs().max().item()
        assert err < 5e-3, (name, rows, pdl, err)
        # BF16 activation quantized on load == quantizing first.
        A16 = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
        Aq = _fp8(torch.clamp(A16.float() / a_scale, -448, 448))
        with torch.cuda.stream(stream):
            rc = fvk.pi05_dec_skinny_gemm_bf16_act(A16.data_ptr(), a_scale.data_ptr(), W.data_ptr(), partials.data_ptr(),
                                              rows, N, K, cfg, pdl, stream.cuda_stream)
            assert rc == 0
            residual.zero_()
            fvk.pi05_dec_skinny_residual_gate_mul(partials.data_ptr(), splits, a_scale.data_ptr(), w_scale.data_ptr(),
                                             residual.data_ptr(), gate.data_ptr(), rows, N, pdl, stream.cuda_stream)
        torch.cuda.synchronize()
        ref = (Aq.float() @ W.float().T) * (a_scale * w_scale)
        err = (residual.float() - ref).abs().max().item() / ref.abs().max().item()
        assert err < 5e-3, (name, "bf16 act", err)


@requires_family
def test_consumers_bit_identical_to_unfused_kernels():
    """Feed the fused consumers BF16-exact partial sums with unit scales and
    compare against the kernels of the cuBLASLt decoder path."""
    from flash_rt import flash_rt_kernels as fvk
    torch.manual_seed(1)
    rows, D, H = 10, 1024, 4096
    one = torch.ones(1, device="cuda")
    bf = lambda *shape: torch.randn(*shape, device="cuda").to(torch.bfloat16)

    # gated residual + adaptive norm + FP8 quantize
    x = bf(rows, D)
    residual_a = bf(rows, D); residual_b = residual_a.clone()
    gate_a = bf(rows, D); gate_b = gate_a.clone()
    weight = bf(D); style = bf(rows, 3 * D)
    out_scale = torch.tensor([0.05], device="cuda")
    out_a = torch.empty(rows, D, dtype=torch.float8_e4m3fn, device="cuda")
    out_b = torch.empty_like(out_a)
    fvk.gate_residual_ada_norm_fp8(residual_a.data_ptr(), x.data_ptr(), gate_a.data_ptr(), weight.data_ptr(),
                                   style.data_ptr(), out_a.data_ptr(), gate_a.data_ptr(), rows, D, 1e-6,
                                   out_scale.data_ptr())
    partials = x.float().contiguous()  # splits = 1, alpha = 1
    rc = fvk.pi05_dec_skinny_residual_ada_norm(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), residual_b.data_ptr(),
                                          gate_b.data_ptr(), weight.data_ptr(), style.data_ptr(), out_b.data_ptr(), 0,
                                          out_scale.data_ptr(), gate_b.data_ptr(), rows, D, 1e-6, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    assert torch.equal(residual_a, residual_b)
    assert torch.equal(gate_a, gate_b)
    assert torch.equal(out_a.view(torch.uint8), out_b.view(torch.uint8))

    # last layer: gate*residual then the final adaptive norm in BF16
    residual_a = bf(rows, D); residual_b = residual_a.clone()
    gate_a = bf(rows, D); gate_b = gate_a.clone()
    normed_a = torch.empty(rows, D, dtype=torch.bfloat16, device="cuda"); normed_b = torch.empty_like(normed_a)
    fvk.gate_mul_residual(residual_a.data_ptr(), x.data_ptr(), gate_a.data_ptr(), rows * D)
    fvk.ada_rms_norm_style(residual_a.data_ptr(), weight.data_ptr(), style.data_ptr(), normed_a.data_ptr(),
                           gate_a.data_ptr(), rows, D, 1e-6)
    rc = fvk.pi05_dec_skinny_residual_ada_norm(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), residual_b.data_ptr(),
                                          gate_b.data_ptr(), weight.data_ptr(), style.data_ptr(), 0, normed_b.data_ptr(),
                                          0, gate_b.data_ptr(), rows, D, 1e-6, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    assert torch.equal(residual_a, residual_b)
    assert torch.equal(gate_a, gate_b)
    assert torch.equal(normed_a, normed_b)

    # GeGLU + FP8 quantize over merged [gate | up]
    merged = bf(rows, 2 * H)
    out_a = torch.empty(rows, H, dtype=torch.float8_e4m3fn, device="cuda"); out_b = torch.empty_like(out_a)
    fvk.gate_geglu_merged_fp8(merged.data_ptr(), out_a.data_ptr(), rows, H, out_scale.data_ptr())
    partials = merged.float().contiguous()
    rc = fvk.pi05_dec_skinny_gate_gelu_fp8(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), out_b.data_ptr(),
                                      rows, H, out_scale.data_ptr(), False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    assert torch.equal(out_a.view(torch.uint8), out_b.view(torch.uint8))

    # QKV split + RoPE + cache write (two samples with a cache stride)
    q_dim, kv_dim, hd, enc = 2048, 256, 256, 37
    per_sample = enc + rows + 5
    qkv = bf(2 * rows, q_dim + 2 * kv_dim)
    rope = bf(rows, hd)
    Q_a = torch.zeros(2 * rows, q_dim, dtype=torch.bfloat16, device="cuda"); Q_b = Q_a.clone()
    K_a = torch.zeros(2 * per_sample, kv_dim, dtype=torch.bfloat16, device="cuda"); K_b = K_a.clone()
    V_a = K_a.clone(); V_b = K_a.clone()
    row_bytes = (q_dim + 2 * kv_dim) * 2
    for b in range(2):
        fvk.qkv_split_rope(qkv.data_ptr() + b * rows * row_bytes, rope.data_ptr(), Q_a.data_ptr() + b * rows * q_dim * 2,
                           K_a.data_ptr() + (b * per_sample + enc) * kv_dim * 2,
                           V_a.data_ptr() + (b * per_sample + enc) * kv_dim * 2,
                           rows, q_dim, kv_dim, kv_dim, hd)
    partials = qkv.float().contiguous()
    rc = fvk.pi05_dec_skinny_sum_rope(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), rope.data_ptr(),
                                 Q_b.data_ptr(), K_b.data_ptr() + enc * kv_dim * 2, V_b.data_ptr() + enc * kv_dim * 2,
                                 0, 2 * rows, q_dim, kv_dim, kv_dim, hd, rows, per_sample, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    assert torch.equal(Q_a, Q_b)
    assert torch.equal(K_a, K_b)
    assert torch.equal(V_a, V_b)


@requires_family
@pytest.mark.parametrize("pdl", [False, True])
def test_attention_matches_torch(pdl):
    """Split-KV cross-attention against torch on random Q/K/V: two samples
    with a cache stride, ten query rows, valid keys from a device count,
    stale rows past the count filled with NaN to prove they are masked;
    two launches in a row to exercise the self-resetting counters."""
    from flash_rt import flash_rt_kernels as fvk
    torch.manual_seed(3)
    rows, heads, hd, samples = 10, 8, 256, 2
    kv_len, valid = 530, 517
    stride_rows = kv_len + 7
    Q = torch.randn(samples, rows, heads, hd, device="cuda").to(torch.bfloat16)
    K = torch.randn(samples, stride_rows, hd, device="cuda").to(torch.bfloat16)
    V = torch.randn(samples, stride_rows, hd, device="cuda").to(torch.bfloat16)
    K[:, valid:] = float("nan"); V[:, valid:] = float("nan")
    O = torch.zeros(samples, rows, heads, hd, device="cuda", dtype=torch.bfloat16)
    seqused = torch.tensor([valid], dtype=torch.int32, device="cuda")
    splits = fvk.pi05_dec_skinny_attn_splits(kv_len)
    scratch = torch.zeros(fvk.pi05_dec_skinny_attn_scratch_floats(splits, samples, heads), device="cuda")
    counters = torch.zeros(samples * heads, dtype=torch.int32, device="cuda")
    for _ in range(2):
        rc = fvk.pi05_dec_skinny_attn(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), rows, samples, heads,
                                 heads * hd, kv_len, seqused.data_ptr(), stride_rows, 1.0 / hd ** 0.5,
                                 scratch.data_ptr(), counters.data_ptr(), pdl, 0)
        torch.cuda.synchronize()
        assert rc == 0
    assert int(counters.sum().item()) == 0
    q = Q.float().permute(0, 2, 1, 3)                       # (B, H, rows, hd)
    k = K[:, :valid].float().unsqueeze(1)                    # (B, 1, valid, hd)
    v = V[:, :valid].float().unsqueeze(1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / hd ** 0.5)
    ref = ref.permute(0, 2, 1, 3)
    assert torch.isfinite(O.float()).all()
    err = (O.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-2, err
    cos = torch.nn.functional.cosine_similarity(O.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.9999, cos
    # without a device count every key up to kv_len counts (no NaN rows then)
    K[:, valid:] = 0; V[:, valid:] = 0
    rc = fvk.pi05_dec_skinny_attn(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), rows, samples, heads,
                             heads * hd, kv_len, 0, stride_rows, 1.0 / hd ** 0.5,
                             scratch.data_ptr(), counters.data_ptr(), pdl, 0)
    torch.cuda.synchronize()
    assert rc == 0
    k = K[:, :kv_len].float().unsqueeze(1); v = V[:, :kv_len].float().unsqueeze(1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / hd ** 0.5).permute(0, 2, 1, 3)
    cos = torch.nn.functional.cosine_similarity(O.float().flatten(), ref.flatten(), dim=0).item()
    assert cos > 0.9999, cos


@requires_family
def test_action_projection_kernels():
    """Fused action projections against the arithmetic they replace: BF16
    GEMM rounding, bias add rounding, first adaptive norm (bit-identical to
    ada_rms_norm_style_fp8 on the projected rows), trace copies and the
    in-place BF16 Euler update."""
    from flash_rt import flash_rt_kernels as fvk
    torch.manual_seed(5)
    rows, D, A = 10, 1024, 32
    bf = lambda *shape: torch.randn(*shape, device="cuda").to(torch.bfloat16)
    noise = bf(rows, A); w_in = bf(A, D) * 0.1; b_in = bf(D) * 0.1
    x = torch.empty(rows, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(D, dtype=torch.bfloat16, device="cuda"); style = bf(rows, 3 * D)
    out_scale = torch.tensor([0.05], device="cuda")
    out = torch.empty(rows, D, dtype=torch.float8_e4m3fn, device="cuda")
    gate = torch.empty(rows, D, dtype=torch.bfloat16, device="cuda")
    rc = fvk.pi05_dec_skinny_action_in_norm(noise.data_ptr(), w_in.data_ptr(), b_in.data_ptr(), x.data_ptr(),
                                       weight.data_ptr(), style.data_ptr(), out.data_ptr(), gate.data_ptr(),
                                       out_scale.data_ptr(), rows, 1e-6, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    x_ref = ((noise.float() @ w_in.float()).to(torch.bfloat16).float() + b_in.float()).to(torch.bfloat16)
    assert (x.float() - x_ref.float()).abs().max().item() <= 2 * 2 ** -8 * x_ref.float().abs().max().item()
    out_ref = torch.empty_like(out); gate_ref = torch.empty_like(gate)
    fvk.ada_rms_norm_style_fp8(x.data_ptr(), weight.data_ptr(), style.data_ptr(), out_ref.data_ptr(),
                               gate_ref.data_ptr(), rows, D, 1e-6, out_scale.data_ptr())
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), out_ref.view(torch.uint8))
    assert torch.equal(gate, gate_ref)

    xn = bf(rows, D); w_out = bf(D, A) * 0.02; b_out = bf(A) * 0.01
    noise0 = noise.clone(); noise1 = noise.clone()
    action = torch.empty(rows, A, dtype=torch.bfloat16, device="cuda")
    tx = torch.empty(rows, A, dtype=torch.bfloat16, device="cuda"); td = torch.empty_like(tx)
    rc = fvk.pi05_dec_skinny_action_out_residual(xn.data_ptr(), w_out.data_ptr(), b_out.data_ptr(), action.data_ptr(),
                                            noise1.data_ptr(), tx.data_ptr(), td.data_ptr(), rows, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    a_ref = ((xn.float() @ w_out.float()).to(torch.bfloat16).float() + b_out.float()).to(torch.bfloat16)
    tol = 2 * 2 ** -8 * a_ref.float().abs().max().item()
    assert (action.float() - a_ref.float()).abs().max().item() <= tol
    assert torch.equal(tx, noise0)
    assert torch.equal(td, action)
    assert torch.equal(noise1, (noise0.float() + action.float()).to(torch.bfloat16))


@requires_ckpt
def test_default_keeps_cublaslt_without_transposed_copies(monkeypatch):
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    monkeypatch.delenv("FLASHRT_PI05_DECODER_KERNEL", raising=False)
    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    assert rt._decoder_kernel == "cublaslt"
    assert not any(key.endswith("__nk") for key in rt._fp8_weights)
    rt.set_prompt(PROMPT)
    assert not rt.pipeline._skinny


@requires_ckpt
def test_frontend_matches_cublaslt_and_replays_bit_identical():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    obs = [_make_obs(s) for s in range(3)]
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    ref = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, decoder_kernel="cublaslt")
    ref.set_prompt(PROMPT); ref.calibrate(obs)
    ref_actions = [ref.infer(o, noise=noise)["actions"] for o in obs]
    del ref; torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, decoder_kernel="skinny")
    assert rt.pipeline is None
    rt.set_prompt(PROMPT); rt.calibrate(obs)
    assert rt.pipeline._skinny
    for o, ra in zip(obs, ref_actions):
        a = rt.infer(o, noise=noise)["actions"]
        assert _cos(a, ra) > 0.9999, _cos(a, ra)

    first = rt.infer(obs[0], noise=noise)["actions"]
    for _ in range(300):
        again = rt.infer(obs[0], noise=noise)["actions"]
        assert np.array_equal(first, again)


@requires_ckpt
def test_batched_family_matches_batched_cublaslt():
    """The batched pipeline on the family against the batched cuBLASLt
    decoder, slot by slot, same prompts, images and noise. (Batched versus
    single-sample agreement is the existing B=N test's concern; the FP8
    path is sensitive to synthetic images there, so it is not re-gated
    here.)"""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    B = 4
    obs = [_make_obs(s) for s in range(B)]
    g = torch.Generator(device="cuda"); g.manual_seed(11)
    noise = torch.randn(B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    outs = {}
    for kernel in ("cublaslt", "skinny"):
        rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, decoder_kernel=kernel)
        rt.set_batched_mode(enable=True, batch_size=B)
        rt.set_prompt_batch([PROMPT] * B)
        rt.calibrate_batch(obs)
        assert rt.pipeline._skinny == (kernel == "skinny")
        outs[kernel] = [r["actions"] for r in rt.infer_batch(obs, noise=noise)]
        if kernel == "skinny":
            again = [r["actions"] for r in rt.infer_batch(obs, noise=noise)]
            for b in range(B):
                assert np.array_equal(outs[kernel][b], again[b])
        del rt; torch.cuda.empty_cache()
    # The batched pipeline quantizes with per-layer (not per-step) FP8
    # scales, and calibrating on synthetic frames leaves little headroom,
    # so two summation orders flip visibly more FP8 bins than at B = 1:
    # with these frames the slots agree to 0.98-0.9999, with real LIBERO
    # frames to 0.99997-0.99999 (docs/pi05_decoder_skinny.md). This is a
    # smoke gate; the real-frame numbers are the acceptance criterion.
    for b in range(B):
        c = _cos(outs["skinny"][b], outs["cublaslt"][b])
        assert c > 0.98, (b, c)
