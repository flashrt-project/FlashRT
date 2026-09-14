"""Stochastic (SDE) sampler on the Pi0.5 RTX frontend.

``Pi05TorchFrontendRtx(..., sde=True)`` builds pipelines whose denoising
step adds ``sigma[s] * eps[s]`` after the flow increment:
``x[s+1] = bf16(x[s] + fma(sigma[s], eps[s], delta[s]))``. The per-step
noise and sigma live in device buffers written per call, so one captured
graph serves the ODE sampler (sigma zero, bit-identical to a frontend
built without ``sde``) and the SDE sampler. Runs on RTX 5090 / 4090 with
the pi05_libero PyTorch checkpoint (``PI05_LIBERO_PYTORCH_CHECKPOINT``
overrides the directory).

Invariants:
  - sde frontend with no sigma matches a plain frontend (cosine 0.999 FP8 /
    0.9999 BF16: two builds calibrate and autotune separately, so this is
    the cross-build band, not bit-equality) for the skinny FP8 decoder, the
    library FP8 decoder and the BF16 engine; within one sde frontend a zero
    schedule and no schedule are bit-identical (below);
  - with a sigma schedule the trace is self-consistent under the SDE rule
    (exact where sigma is 0, within one bf16 ulp elsewhere), the same
    (noise, step_noise) reproduces the actions bit for bit, a different
    step noise or a zero schedule gives different actions, and the last
    step with sigma 0 leaves the final action at the step mean;
  - the batched pipeline does the same per slot;
  - inputs are validated (sigma length, sign, sde on a plain frontend).

Run::

    python -m pytest tests/test_pi05_sde_sampler.py -v
"""

import gc
import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
PROMPT = "pick up the black bowl and place it on the plate"

requires_gpu_ckpt = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT_PI05)),
    reason=f"needs CUDA and the pi05 ckpt at {CKPT_PI05}")


def _make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224] / 224.0
    imgs = []
    for _ in range(2):
        img = np.zeros((224, 224, 3), np.float32)
        for c in range(3):
            fx, fy, ph = rng.uniform(0.5, 3.0, 3)
            img[..., c] = 0.5 + 0.35 * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
        imgs.append(np.clip(img * 255, 0, 255).astype(np.uint8))
    return {"image": imgs[0], "wrist_image": imgs[1], "state": rng.random(8, dtype=np.float32)}


def _cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _bf16(a: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def _check_sde_trace(x, delta, eps, sigma, raw_actions):
    """x[s+1] == bf16(x[s] + sigma[s]*eps[s] + delta[s]); exact where sigma is 0."""
    steps = x.shape[0]
    for s in range(steps):
        nxt = x[s + 1] if s + 1 < steps else raw_actions
        want = _bf16(x[s] + (np.float32(sigma[s]) * eps[s] + delta[s]).astype(np.float32))
        if sigma[s] == 0:
            assert np.array_equal(nxt, want), f"step {s} (sigma 0) not exact"
        else:
            ulp = np.abs(want) * 2.0 ** -7 + 1e-6
            assert np.all(np.abs(nxt - want) <= ulp), f"step {s} off by more than one bf16 ulp"


def _build(sde: bool, **kw):
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    fe = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, denoise_trace=True, sde=sde, **kw)
    fe.set_prompt(PROMPT)
    return fe


@requires_gpu_ckpt
@pytest.mark.parametrize("engine", ["fp8-skinny", "fp8-cublaslt", "bf16"])
def test_sde_frontend_without_sigma_is_the_ode_sampler(engine):
    gc.collect(); torch.cuda.empty_cache()
    kw = {"use_fp8": engine != "bf16"}
    if engine == "fp8-cublaslt":
        kw["decoder_kernel"] = "cublaslt"
    elif engine == "fp8-skinny":
        kw["decoder_kernel"] = "auto"
    obs = [_make_obs(s) for s in range(2)]
    g = torch.Generator(device="cuda"); g.manual_seed(3)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    plain = _build(False, **kw); plain.calibrate(obs)
    ref = [plain.infer(o, noise=noise) for o in obs]
    del plain; gc.collect(); torch.cuda.empty_cache()

    fe = _build(True, **kw); fe.calibrate(obs)
    out = [fe.infer(o, noise=noise) for o in obs]
    gate = 0.9999 if engine == "bf16" else 0.999
    for r, o in zip(ref, out):
        c = _cos(r["raw_actions"], o["raw_actions"])
        assert c > gate, f"{engine}: sde frontend vs plain frontend cosine {c:.5f}"
        assert "step_noise" not in o
    # within the sde frontend: no schedule, a zero schedule and a replay are the same sample
    z = fe.infer(obs[0], noise=noise, sde_sigma=[0.0] * 10)
    assert np.array_equal(z["raw_actions"], out[0]["raw_actions"])
    assert np.array_equal(fe.infer(obs[0], noise=noise)["raw_actions"], out[0]["raw_actions"])
    with pytest.raises(ValueError):
        fe.infer(obs[0], noise=noise, sde_sigma=[0.1] * 3)
    with pytest.raises(ValueError):
        fe.infer(obs[0], noise=noise, sde_sigma=[-0.1] + [0.0] * 9)
    with pytest.raises(ValueError):
        fe.infer(obs[0], noise=noise, step_noise=np.zeros((10, 10, 32), np.float32))


@requires_gpu_ckpt
def test_sde_trace_reproducible_and_consistent():
    gc.collect(); torch.cuda.empty_cache()
    obs = _make_obs(1)
    g = torch.Generator(device="cuda"); g.manual_seed(5)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    sigma = [0.08] * 9 + [0.0]
    fe = _build(True, use_fp8=True); fe.calibrate([obs])
    ode = fe.infer(obs, noise=noise)
    gen = torch.Generator(device="cuda"); gen.manual_seed(9)
    a = fe.infer(obs, noise=noise, sde_sigma=sigma, generator=None,
                 step_noise=torch.randn(10, 10, 32, generator=gen, device="cuda", dtype=torch.bfloat16))
    assert a["step_noise"].shape == (10, 10, 32) and np.allclose(a["sde_sigma"], sigma)
    b = fe.infer(obs, noise=noise, sde_sigma=sigma, step_noise=a["step_noise"])
    assert np.array_equal(a["raw_actions"], b["raw_actions"])
    assert np.array_equal(a["denoise_trace"]["x"], b["denoise_trace"]["x"])
    assert not np.array_equal(a["raw_actions"], ode["raw_actions"])
    assert np.array_equal(a["denoise_trace"]["x"][0], ode["denoise_trace"]["x"][0])   # same start
    _check_sde_trace(a["denoise_trace"]["x"], a["denoise_trace"]["delta"], a["step_noise"], sigma, a["raw_actions"])
    # the last step has sigma 0: the final action is exactly the step mean
    assert np.array_equal(a["raw_actions"], _bf16(a["denoise_trace"]["x"][-1] + a["denoise_trace"]["delta"][-1]))
    # a fresh draw from a generator differs and is reported
    c = fe.infer(obs, noise=noise, sde_sigma=sigma, generator=gen)
    assert not np.array_equal(c["step_noise"], a["step_noise"]) and not np.array_equal(c["raw_actions"], a["raw_actions"])
    # zero schedule through the sde path == ODE
    z = fe.infer(obs, noise=noise, sde_sigma=[0.0] * 10, step_noise=a["step_noise"])
    assert np.array_equal(z["raw_actions"], ode["raw_actions"])
    # the plain kwargs afterwards are the ODE again
    assert np.array_equal(fe.infer(obs, noise=noise)["raw_actions"], ode["raw_actions"])


@requires_gpu_ckpt
def test_sde_batched_per_slot():
    gc.collect(); torch.cuda.empty_cache()
    B = 4
    obs = [_make_obs(s) for s in range(B)]
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    noise = torch.randn(B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    eps = torch.randn(10, B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    sigma = [0.05] * 9 + [0.0]
    fe = _build(True, use_fp8=True)
    fe.set_batched_mode(enable=True, batch_size=B)
    fe.set_prompt_batch([PROMPT] * B); fe.calibrate_batch(obs)
    ode = fe.infer_batch(obs, noise=noise)
    out = fe.infer_batch(obs, noise=noise, sde_sigma=sigma, step_noise=eps)
    again = fe.infer_batch(obs, noise=noise, sde_sigma=sigma, step_noise=eps)
    for b in range(B):
        assert np.array_equal(out[b]["raw_actions"], again[b]["raw_actions"])
        assert not np.array_equal(out[b]["raw_actions"], ode[b]["raw_actions"])
        assert np.array_equal(out[b]["step_noise"], eps[:, b].float().cpu().numpy())
        _check_sde_trace(out[b]["denoise_trace"]["x"], out[b]["denoise_trace"]["delta"],
                         out[b]["step_noise"], sigma, out[b]["raw_actions"])
    back = fe.infer_batch(obs, noise=noise)
    for b in range(B):
        assert np.array_equal(back[b]["raw_actions"], ode[b]["raw_actions"])
