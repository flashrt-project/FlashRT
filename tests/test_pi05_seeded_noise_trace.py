"""Reproducible sampling and denoise-trace export on the Pi0.5 RTX frontend.

Runs on RTX 5090 / 4090. Skipped if GPU or the pi05_libero PyTorch
checkpoint are unavailable. Set ``PI05_LIBERO_PYTORCH_CHECKPOINT`` to
override the default checkpoint directory.

Invariants covered:
  - ``infer(noise=...)`` with the same noise gives bit-identical
    actions; a different noise gives different actions; the returned
    ``"noise"`` is the noise that was used.
  - ``infer(generator=...)`` with equal seeds reproduces the sample.
  - Without ``noise``/``generator`` the default path still works and
    the extra result keys are additive.
  - With ``denoise_trace=True``: ``x[0]`` is the input noise,
    ``x[s+1] == x[s] + delta[s]`` at every step, ``x[-1] + delta[-1]``
    is the raw output, and the trace has ``num_steps`` entries with
    the documented timesteps.
  - The batched path honours ``noise`` per slot and records a per-slot
    trace; identical inputs on both slots give bit-identical traces.

Run::

    python -m pytest tests/test_pi05_seeded_noise_trace.py -v
"""

import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)

requires_gpu_ckpt = pytest.mark.skipif(
    not (_GPU_AVAILABLE and _CKPT_AVAILABLE),
    reason=f"needs CUDA and the pi05 ckpt at {CKPT_PI05}")


def _make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "state": rng.random(8, dtype=np.float32),
    }


def _bf16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two float32 views of bf16 values the way the kernel does (round to bf16)."""
    return (torch.from_numpy(a).to(torch.bfloat16) + torch.from_numpy(b).to(torch.bfloat16)).float().numpy()


def _check_trace(trace: dict, noise: np.ndarray, raw_actions: np.ndarray, num_steps: int):
    x, delta, ts = trace["x"], trace["delta"], trace["timesteps"]
    assert x.shape == (num_steps,) + noise.shape
    assert delta.shape == x.shape
    assert len(ts) == num_steps and ts[0] == pytest.approx(1.0)
    assert ts[-1] == pytest.approx(1.0 / num_steps)
    np.testing.assert_array_equal(x[0], noise)
    for s in range(num_steps - 1):
        np.testing.assert_array_equal(x[s + 1], _bf16_add(x[s], delta[s]), err_msg=f"step {s}")
    np.testing.assert_array_equal(_bf16_add(x[-1], delta[-1]), raw_actions)
    assert np.isfinite(x).all() and np.isfinite(delta).all()


@requires_gpu_ckpt
def test_noise_injection_is_reproducible_and_reported():
    from flash_rt.frontends.torch.pi05_rtx import ACTION_DIM, Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_prompt("pick up the cup")
    obs = _make_obs()
    rt.calibrate([obs])
    noise = np.random.default_rng(1).standard_normal((rt.chunk_size, ACTION_DIM)).astype(np.float32)

    a = rt.infer(obs, noise=noise, return_noise=True)
    b = rt.infer(obs, noise=noise, return_noise=True)
    np.testing.assert_array_equal(a["actions"], b["actions"])
    # the reported noise is the bf16-rounded input and round-trips exactly
    np.testing.assert_array_equal(a["noise"], b["noise"])
    c = rt.infer(obs, noise=a["noise"])
    np.testing.assert_array_equal(a["actions"], c["actions"])

    other = rt.infer(obs, noise=-noise)
    assert not np.array_equal(a["actions"], other["actions"])

    g1 = torch.Generator(device="cuda").manual_seed(1234)
    g2 = torch.Generator(device="cuda").manual_seed(1234)
    s1 = rt.infer(obs, generator=g1, return_noise=True)
    s2 = rt.infer(obs, generator=g2, return_noise=True)
    np.testing.assert_array_equal(s1["noise"], s2["noise"])
    np.testing.assert_array_equal(s1["actions"], s2["actions"])

    default = rt.infer(obs)
    assert set(default) == {"actions"}
    assert np.isfinite(default["actions"]).all()

    with pytest.raises(ValueError):
        rt.infer(obs, noise=noise[:1])
    with pytest.raises(ValueError):
        rt.infer(obs, noise=noise, generator=g1)


@requires_gpu_ckpt
def test_denoise_trace_is_self_consistent():
    from flash_rt.frontends.torch.pi05_rtx import ACTION_DIM, Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, denoise_trace=True)
    rt.set_prompt("put the bowl on the plate")
    obs = _make_obs(3)
    rt.calibrate([obs])
    noise = np.random.default_rng(5).standard_normal((rt.chunk_size, ACTION_DIM)).astype(np.float32)

    res = rt.infer(obs, noise=noise, return_noise=True)
    assert {"actions", "noise", "raw_actions", "denoise_trace"} <= set(res)
    _check_trace(res["denoise_trace"], res["noise"], res["raw_actions"], rt._num_steps)

    # a second call overwrites the trace consistently
    res2 = rt.infer(obs, noise=-noise, return_noise=True)
    _check_trace(res2["denoise_trace"], res2["noise"], res2["raw_actions"], rt._num_steps)
    assert not np.array_equal(res["denoise_trace"]["x"][1], res2["denoise_trace"]["x"][1])

    with pytest.raises(NotImplementedError):
        rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)


@requires_gpu_ckpt
def test_trace_off_by_default_and_batched_trace_per_slot():
    from flash_rt.frontends.torch.pi05_rtx import ACTION_DIM, PI05_BATCH_SIZE, Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_prompt("pick up the cup")
    rt.calibrate([_make_obs()])
    assert "denoise_trace" not in rt.infer(_make_obs())

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, denoise_trace=True)
    rt.set_batched_mode(enable=True)
    rt.set_prompt_batch(["pick up the cup"] * PI05_BATCH_SIZE)
    obs = _make_obs(7)
    rt.calibrate_batch([obs])
    noise = np.random.default_rng(9).standard_normal(
        (PI05_BATCH_SIZE, rt.chunk_size, ACTION_DIM)).astype(np.float32)
    noise[1] = noise[0]

    out = rt.infer_batch([obs] * PI05_BATCH_SIZE, noise=noise, return_noise=True)
    assert len(out) == PI05_BATCH_SIZE
    for b in range(PI05_BATCH_SIZE):
        _check_trace(out[b]["denoise_trace"], out[b]["noise"], out[b]["raw_actions"], rt._num_steps)
    np.testing.assert_array_equal(out[0]["actions"], out[1]["actions"])
    np.testing.assert_array_equal(out[0]["denoise_trace"]["x"], out[1]["denoise_trace"]["x"])

    again = rt.infer_batch([obs] * PI05_BATCH_SIZE, noise=noise)
    np.testing.assert_array_equal(out[0]["actions"], again[0]["actions"])
