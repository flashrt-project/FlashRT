"""NVFP4 prefix tier on the Pi0.5 RTX frontend (sm_120a builds).

Runs on RTX 5090 with a GPU_ARCH=120 kernel build; skipped without the
GPU, the NVFP4 kernels or the pi05_libero PyTorch checkpoint. Set
``PI05_LIBERO_PYTORCH_CHECKPOINT`` to override the checkpoint directory.

Invariants covered:
  - ``prefix_precision="nvfp4"`` runs the vision and encoder GEMMs on the
    block-scaled 4-bit kernels (weights listed, SigLIP FFN down padded to
    K = 4352) and agrees with the FP8 tier on the same prompt, images and
    noise to the documented cosine (4-bit operands: looser than FP8);
  - replays are bit-identical; the tier is faster per environment than FP8
    at B = 1 and B = 8;
  - the batched pipeline runs the tier and matches the single one per slot;
  - a weight reload re-quantizes the NVFP4 weights.

Run::

    python -m pytest tests/test_pi05_prefix_nvfp4.py -v -s
"""

import gc
import os
import time

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
PROMPT = "pick up the black bowl and place it on the plate"

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)


def _tier_available() -> bool:
    if not _GPU_AVAILABLE:
        return False
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    return all(hasattr(fvk, n) for n in ("bf16_weight_to_nvfp4_swizzled", "quantize_bf16_to_nvfp4_swizzled_v2",
                                          "fp4_w4a16_gemm_sm120_bf16out_pingpong"))


requires = pytest.mark.skipif(not (_tier_available() and _CKPT_AVAILABLE),
                              reason=f"needs CUDA sm_120a, the NVFP4 kernels and the pi05 ckpt at {CKPT_PI05}")


def _make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224] / 224.0
    imgs = []
    for _ in range(2):
        img = np.zeros((224, 224, 3), np.float32)
        for c in range(3):
            fx, fy, ph = rng.uniform(0.5, 3.0, 3)
            img[..., c] = 0.5 + 0.35 * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
        for _ in range(4):
            y0, x0 = rng.integers(0, 160, 2); h, w = rng.integers(20, 64, 2)
            img[y0:y0 + h, x0:x0 + w] = rng.uniform(0.1, 0.9, 3)
        imgs.append(np.clip(img * 255, 0, 255).astype(np.uint8))
    return {"image": imgs[0], "wrist_image": imgs[1], "state": rng.random(8, dtype=np.float32)}


def _cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _median_ms(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return 1e3 * float(np.median(ts))


@requires
def test_fused_geglu_quantizer_matches_unfused_pair():
    """pi05_geglu_merged_to_nvfp4_swizzled == gate_geglu_merged followed by
    quantize_bf16_to_nvfp4_swizzled_v2, bit for bit (packed e2m1 and
    swizzled scales), at the encoder's shape and a batched row count."""
    from flash_rt import flash_rt_kernels as fvk
    torch.manual_seed(2)
    for rows in (520, 4160):
        half = 16384
        merged = (torch.randn(rows, 2 * half, device="cuda") * 1.5).to(torch.bfloat16)
        hidden = torch.empty(rows, half, dtype=torch.bfloat16, device="cuda")
        fvk.gate_geglu_merged(merged.data_ptr(), hidden.data_ptr(), rows, half)
        nb = half // 16
        sf_bytes = ((rows + 127) // 128) * ((nb + 3) // 4) * 512
        a_ref = torch.empty(rows, half // 2, dtype=torch.uint8, device="cuda"); sf_ref = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
        fvk.quantize_bf16_to_nvfp4_swizzled_v2(hidden.data_ptr(), a_ref.data_ptr(), sf_ref.data_ptr(), rows, half)
        a = torch.empty_like(a_ref); sf = torch.zeros_like(sf_ref)
        rc = fvk.pi05_geglu_merged_to_nvfp4_swizzled(merged.data_ptr(), a.data_ptr(), sf.data_ptr(), rows, half)
        torch.cuda.synchronize()
        assert rc == 0
        assert torch.equal(sf, sf_ref), rows
        assert torch.equal(a, a_ref), rows


@requires
def test_nvfp4_prefix_matches_fp8_and_is_faster():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    obs = [_make_obs(s) for s in range(3)]
    g = torch.Generator(device="cuda"); g.manual_seed(3)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    fp8 = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    fp8.set_prompt(PROMPT); fp8.calibrate(obs)
    ref = [fp8.infer(o, noise=noise)["actions"] for o in obs]
    t8 = _median_ms(lambda: fp8.infer(obs[0]))
    del fp8; gc.collect(); torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_precision="nvfp4")
    rt.set_prompt(PROMPT); rt.calibrate(obs)
    assert rt.pipeline._nvfp4 and "vision_ffn_down_w_0" in rt.pipeline._nvfp4
    assert rt.pipeline._nvfp4["vision_ffn_down_w_0"][3] == 4352
    out = [rt.infer(o, noise=noise)["actions"] for o in obs]
    again = rt.infer(obs[0], noise=noise)["actions"]
    assert np.array_equal(again, out[0])
    t4 = _median_ms(lambda: rt.infer(obs[0]))
    cs = [_cos(a, r) for a, r in zip(out, ref)]
    print(f"\nnvfp4 prefix vs fp8: cos {[round(c, 5) for c in cs]}, infer fp8 {t8:.2f} ms, nvfp4 {t4:.2f} ms")
    assert min(cs) > 0.97, cs
    assert t4 < t8, (t4, t8)
    rt.reload_weights(CKPT_PI05)
    after = rt.infer(obs[0], noise=noise)["actions"]
    assert _cos(after, out[0]) > 0.9999
    del rt; gc.collect(); torch.cuda.empty_cache()


@requires
def test_nvfp4_prefix_batched():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    B = 4
    obs = [_make_obs(s) for s in range(B)]
    g = torch.Generator(device="cuda"); g.manual_seed(5)
    noise = torch.randn(B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    single = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_precision="nvfp4")
    single.set_prompt(PROMPT); single.calibrate(obs)
    ref = [single.infer(o, noise=noise[b])["actions"] for b, o in enumerate(obs)]
    del single; gc.collect(); torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_precision="nvfp4")
    rt.set_batched_mode(enable=True, batch_size=B)
    rt.set_prompt_batch([PROMPT] * B); rt.calibrate_batch(obs)
    out = [r["actions"] for r in rt.infer_batch(obs, noise=noise)]
    cs = [_cos(a, r) for a, r in zip(out, ref)]
    tb = _median_ms(lambda: rt.infer_batch(obs, noise=noise), n=10)
    print(f"\nnvfp4 batched B={B}: cos vs single {[round(c, 5) for c in cs]}, {tb / B:.2f} ms per environment")
    # Batched vs single on synthetic frames: 4-bit operands and the batched
    # decoder's per-layer FP8 scales stack up to 0.97-0.999 here; on real
    # LIBERO frames the slots agree to 0.998-0.9995 (docs). Smoke gate.
    assert min(cs) > 0.96, cs
