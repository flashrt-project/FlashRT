"""Batched inference at B > 2 and the prefix hidden-state export on the
Pi0.5 RTX frontend.

Runs on RTX 5090 / 4090; skipped without GPU or the pi05_libero PyTorch
checkpoint (``PI05_LIBERO_PYTORCH_CHECKPOINT`` overrides the path).

Covers:
  - ``set_batched_mode(batch_size=4)`` builds a 4-slot backend and
    pipeline; identical inputs in every slot give bit-identical outputs
    per slot; each slot agrees with the B=1 path on the same noise
    (cosine gate, since GEMM tactics differ with M).
  - The batched per-environment forward time drops with B (printed).
  - ``prefix_features=True``: the pooled encoder hidden state has the
    encoder width, is finite, differs between observations, and agrees
    between the B=1 and the batched path.
  - The CFG batched pipeline refuses a backend wider than 2.

Run::

    python -m pytest tests/test_pi05_batched_n.py -v -s
"""

import os
import time

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

PROMPT = "put the bowl on the plate"


def _make_obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "state": rng.random(8, dtype=np.float32),
    }


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _median_ms(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


@requires_gpu_ckpt
def test_unequal_prompt_prefix_matches_single():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    obs = _make_obs(11)
    prompts = ["pick up bowl", "pick up the black bowl and place it on the plate"]
    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_features=True)
    noise = np.random.default_rng(12).standard_normal((rt.chunk_size, 32)).astype(np.float32)
    expected = []
    for prompt in prompts:
        rt.set_prompt(prompt)
        rt.calibrate([obs])
        expected.append(rt.infer(obs, noise=noise)["prefix_features"])
    rt.set_batched_mode(enable=True, batch_size=2)
    rt.set_prompt_batch(prompts)
    assert len(set(rt._batch_prompt_lens)) == 2
    rt.calibrate_batch([obs])
    outputs = rt.infer_batch([obs, obs], noise=np.stack([noise, noise]))
    for output, reference in zip(outputs, expected):
        assert _cos(output["prefix_features"], reference) > 0.999


@requires_gpu_ckpt
def test_batched_b4_matches_b1_and_scales():
    from flash_rt.frontends.torch.pi05_rtx import ACTION_DIM, Pi05TorchFrontendRtx
    from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline

    obs = _make_obs(1)

    ref = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    ref.set_prompt(PROMPT); ref.calibrate([obs])
    noise = np.random.default_rng(2).standard_normal((ref.chunk_size, ACTION_DIM)).astype(np.float32)
    single = ref.infer(obs, noise=noise)
    t1 = _median_ms(lambda: ref.infer(obs, noise=noise))
    del ref; torch.cuda.empty_cache()

    per_env = {1: t1}
    for B in (4, 8):
        rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
        rt.set_batched_mode(enable=True, batch_size=B)
        assert rt.attn_backend.batch_size == B
        rt.set_prompt_batch([PROMPT] * B)
        assert isinstance(rt.pipeline, Pi05BatchedPipeline) and rt.pipeline.B == B
        rt.calibrate_batch([obs])
        out = rt.infer_batch([obs] * B, noise=np.stack([noise] * B))
        assert len(out) == B
        for b in range(1, B):
            np.testing.assert_array_equal(out[0]["actions"], out[b]["actions"])
        c = _cos(out[0]["actions"], single["actions"])
        assert c >= 0.999, f"B={B}: slot vs B=1 cos={c:.5f}"
        tb = _median_ms(lambda: rt.infer_batch([obs] * B, noise=np.stack([noise] * B)))
        per_env[B] = tb / B
        print(f"\nB={B}: {tb:.1f} ms per call, {tb / B:.1f} ms per env, cos vs B=1 {c:.5f}")
        del rt; torch.cuda.empty_cache()
    print(f"per-env ms: {per_env}")
    assert per_env[8] < per_env[1], per_env


@requires_gpu_ckpt
def test_b4_mixed_slots_match_independent_b1_and_permutation():
    from flash_rt.frontends.torch.pi05_rtx import ACTION_DIM, Pi05TorchFrontendRtx

    observations = [_make_obs(30 + slot) for slot in range(4)]
    ref = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    ref.set_prompt(PROMPT)
    # Compare slot addressing under the same calibration input and noise.
    torch.manual_seed(41)
    ref.calibrate([observations[0]])
    noises = np.random.default_rng(40).standard_normal(
        (4, ref.chunk_size, ACTION_DIM)).astype(np.float32)
    expected = [ref.infer(obs, noise=noise)["actions"].copy()
                for obs, noise in zip(observations, noises)]
    scales = {key: value.download_new((1,), np.float32)
              for key, value in ref.pipeline.fp8_act_scales.items()}
    del ref
    torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True, batch_size=4)
    rt.set_prompt_batch([PROMPT] * 4)
    torch.manual_seed(41)
    rt.calibrate_batch([observations[0]])
    assert rt.pipeline.fp8_act_scales.keys() == scales.keys()
    for key, value in rt.pipeline.fp8_act_scales.items():
        np.testing.assert_array_equal(scales[key], value.download_new((1,), np.float32))
    outputs = rt.infer_batch(observations, noise=noises)
    cosines = []
    for slot, (output, reference) in enumerate(zip(outputs, expected)):
        assert np.isfinite(output["actions"]).all()
        cosine = _cos(output["actions"], reference)
        cosines.append(cosine)
    for slot in range(1, 4):
        assert not np.array_equal(outputs[0]["actions"], outputs[slot]["actions"])

    # Reusing the graph after moving every sample catches stale per-slot data.
    order = [2, 0, 3, 1]
    permuted = rt.infer_batch([observations[i] for i in order], noise=noises[order])
    for slot, original in enumerate(order):
        np.testing.assert_array_equal(permuted[slot]["actions"], outputs[original]["actions"])
    # Same synthetic-input numerical contract as test_pi05_batched_precision.
    # Different M changes GEMM reductions; slot isolation itself is exact above.
    assert min(cosines) >= 0.99, cosines


@requires_gpu_ckpt
def test_prefix_features_single_and_batched():
    from flash_rt.frontends.torch.pi05_rtx import ENC_D, Pi05TorchFrontendRtx

    o1, o2 = _make_obs(3), _make_obs(4)
    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_features=True)
    rt.set_prompt(PROMPT); rt.calibrate([o1])
    f1 = rt.infer(o1)["prefix_features"]
    f2 = rt.infer(o2)["prefix_features"]
    assert f1.shape == (ENC_D,) and np.isfinite(f1).all()
    assert not np.array_equal(f1, f2)     # different observations, different features
    again = rt.infer(o1)["prefix_features"]
    np.testing.assert_array_equal(f1, again)
    with pytest.raises(NotImplementedError):
        rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    del rt; torch.cuda.empty_cache()

    # The FP8 path leaves the last layer's residual pending and the export
    # adds it; the BF16 path applies it in-layer. Both must describe the
    # same hidden state, so the pooled features have to agree closely.
    rt16 = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_features=True, use_fp8=False)
    rt16.set_prompt(PROMPT); rt16.calibrate([o1])
    f1_16 = rt16.infer(o1)["prefix_features"]
    c16 = _cos(f1, f1_16)
    print(f"\nprefix features: fp8 vs bf16 path cos {c16:.5f}, "
          f"rel diff {np.linalg.norm(f1 - f1_16) / (np.linalg.norm(f1_16) + 1e-9):.4f}")
    assert c16 >= 0.995, c16
    del rt16; torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, prefix_features=True)
    rt.set_batched_mode(enable=True, batch_size=4)
    rt.set_prompt_batch([PROMPT] * 4)
    rt.calibrate_batch([o1])
    out = rt.infer_batch([o1, o2, o1, o2])
    fb = np.stack([e["prefix_features"] for e in out])
    assert fb.shape == (4, ENC_D)
    np.testing.assert_array_equal(fb[0], fb[2])
    np.testing.assert_array_equal(fb[1], fb[3])
    assert _cos(fb[0], f1) >= 0.999 and _cos(fb[1], f2) >= 0.999
    print(f"\nprefix features: batched vs single cos {_cos(fb[0], f1):.5f} / {_cos(fb[1], f2):.5f}")


@requires_gpu_ckpt
def test_cfg_batched_requires_two_slots():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True, batch_size=4)
    with pytest.raises(ValueError, match="batch_size=2"):
        rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
