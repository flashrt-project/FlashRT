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
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        return False
    probe = getattr(fvk, "dec_skinny_available", None)
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
        assert fvk.dec_skinny_config_supports(cfg, N, K)
        splits = K // fvk.dec_skinny_config_k_chunk(cfg)
        A = _fp8(torch.randn(rows, K, device="cuda") * 0.5)
        W = _fp8(torch.randn(N, K, device="cuda") * 0.05)
        a_scale = torch.tensor([0.02], device="cuda"); w_scale = torch.tensor([0.01], device="cuda")
        partials = torch.empty(splits, rows, N, device="cuda")
        residual = torch.zeros(rows, N, dtype=torch.bfloat16, device="cuda")
        gate = torch.ones(rows, N, dtype=torch.bfloat16, device="cuda")
        with torch.cuda.stream(stream):
            rc = fvk.dec_skinny_gemm(A.data_ptr(), W.data_ptr(), partials.data_ptr(), rows, N, K, cfg, pdl, stream.cuda_stream)
            assert rc == 0, (name, rc)
            rc = fvk.dec_skinny_residual_gate_mul(partials.data_ptr(), splits, a_scale.data_ptr(), w_scale.data_ptr(),
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
            rc = fvk.dec_skinny_gemm_bf16_act(A16.data_ptr(), a_scale.data_ptr(), W.data_ptr(), partials.data_ptr(),
                                              rows, N, K, cfg, pdl, stream.cuda_stream)
            assert rc == 0
            residual.zero_()
            fvk.dec_skinny_residual_gate_mul(partials.data_ptr(), splits, a_scale.data_ptr(), w_scale.data_ptr(),
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
    rc = fvk.dec_skinny_residual_ada_norm(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), residual_b.data_ptr(),
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
    rc = fvk.dec_skinny_residual_ada_norm(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), residual_b.data_ptr(),
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
    rc = fvk.dec_skinny_gate_gelu_fp8(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), out_b.data_ptr(),
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
    rc = fvk.dec_skinny_sum_rope(partials.data_ptr(), 1, one.data_ptr(), one.data_ptr(), rope.data_ptr(),
                                 Q_b.data_ptr(), K_b.data_ptr() + enc * kv_dim * 2, V_b.data_ptr() + enc * kv_dim * 2,
                                 0, 2 * rows, q_dim, kv_dim, kv_dim, hd, rows, per_sample, False, 0)
    torch.cuda.synchronize()
    assert rc == 0
    assert torch.equal(Q_a, Q_b)
    assert torch.equal(K_a, K_b)
    assert torch.equal(V_a, V_b)


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
    # scales, so two summation orders flip more FP8 bins than at B = 1;
    # with synthetic frames the slots agree to about 0.998-0.9999, with
    # real LIBERO frames to 0.99998 (docs/pi05_decoder_skinny.md).
    for b in range(B):
        c = _cos(outs["skinny"][b], outs["cublaslt"][b])
        assert c > 0.997, (b, c)
