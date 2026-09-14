"""Several batched pipelines of different widths in one Pi0.5 RTX frontend.

``select_batch_size`` parks the active batched pipeline and activates (or
builds) the one of the requested width; all widths share the weight
buffers. A fleet server runs the smallest width that fits the pending
requests. Runs on RTX 5090 / 4090 with the pi05_libero PyTorch checkpoint
(``PI05_LIBERO_PYTORCH_CHECKPOINT`` overrides the directory).

Invariants:
  - a width-2 pipeline built next to a width-4 one gives, for the same
    observations and noise, actions within cosine 0.999 of the width-4
    slots (same FP8 scales from the same calibration sample, different GEMM
    tiling), and its own replays are bit-identical;
  - swapping back to width 4 replays the same graph object bit-identically;
  - a weight reload while width 2 is parked reaches it (mapping reload of
    the original weights keeps the outputs; the styles and prompts of the
    parked width are refreshed on the swap);
  - the widths list reports both.

Run::

    python -m pytest tests/test_pi05_batched_widths.py -v
"""

import gc
import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
PROMPTS = ["put the red block in the box", "open the top drawer",
           "pick up the blue cup", "close the middle drawer"]

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


@requires_gpu_ckpt
def test_two_widths_share_weights_and_swap_exactly():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    obs = [_make_obs(s) for s in range(4)]
    g = torch.Generator(device="cuda"); g.manual_seed(11)
    noise4 = torch.randn(4, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    fe = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, use_fp8=True)
    fe.set_batched_mode(enable=True, batch_size=4)
    fe.set_prompt_batch(PROMPTS); fe.calibrate_batch(obs)
    out4 = [np.asarray(r["actions"]) for r in fe.infer_batch(obs, noise=noise4)]
    graph4 = fe.pipeline._graph
    free_before = torch.cuda.mem_get_info()[0]

    fe.select_batch_size(2)
    assert fe.batch_sizes == (2, 4) and fe.pipeline is None
    fe.set_prompt_batch(PROMPTS[:2]); fe.calibrate_batch(obs[:2])
    assert fe.pipeline is not None and fe.pipeline._graph is not graph4
    out2 = [np.asarray(r["actions"]) for r in fe.infer_batch(obs[:2], noise=noise4[:2])]
    out2b = [np.asarray(r["actions"]) for r in fe.infer_batch(obs[:2], noise=noise4[:2])]
    for b in range(2):
        assert np.array_equal(out2[b], out2b[b])
        c = _cos(out2[b], out4[b])
        assert c > 0.999, f"width 2 slot {b} vs width 4: cos {c:.5f}"
    extra_gib = (free_before - torch.cuda.mem_get_info()[0]) / 2**30
    assert extra_gib < 4.0, f"width-2 pipeline took {extra_gib:.2f} GiB; weights are not shared?"

    # swap back: same graph, bit-identical replay, prompts kept
    fe.select_batch_size(4)
    assert fe.pipeline._graph is graph4 and fe.batch_sizes == (2, 4)
    out4b = [np.asarray(r["actions"]) for r in fe.infer_batch(obs, noise=noise4)]
    for b in range(4):
        assert np.array_equal(out4[b], out4b[b])

    # reload while width 2 is parked: same weights, so outputs are unchanged on both widths
    fe.reload_weights(CKPT_PI05)
    out4c = [np.asarray(r["actions"]) for r in fe.infer_batch(obs, noise=noise4)]
    for b in range(4):
        assert np.array_equal(out4[b], out4c[b])
    fe.select_batch_size(2)
    out2c = [np.asarray(r["actions"]) for r in fe.infer_batch(obs[:2], noise=noise4[:2])]
    for b in range(2):
        assert np.array_equal(out2[b], out2c[b])
    with pytest.raises(ValueError):
        fe.infer_batch(obs, noise=noise4)
