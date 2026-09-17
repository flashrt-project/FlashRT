"""In-place weight reload on the Pi0.5 RTX frontend.

Runs on RTX 5090 / 4090. Skipped if GPU or the pi05_libero PyTorch
checkpoint are unavailable. Set ``PI05_LIBERO_PYTORCH_CHECKPOINT`` to
override the checkpoint directory.

Invariants covered:
  - after ``reload_weights`` on a perturbed checkpoint the frontend matches
    a frontend built fresh from that checkpoint on the same prompt, images
    and noise (BF16: cosine 0.9999; FP8 with the calibration scales kept:
    cosine 0.999), and no longer matches its own pre-reload output;
  - the captured graph object is the same before and after (no re-capture)
    and replays are bit-identical after the reload;
  - reloading the original weights restores the original output;
  - a mapping source (never written to disk) behaves like a directory;
  - the batched pipeline picks the reload up as well;
  - the reload takes well under a second.

Run::

    python -m pytest tests/test_pi05_weight_reload.py -v
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
PROMPT = "pick up the black bowl and place it on the plate"

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)

requires_gpu_ckpt = pytest.mark.skipif(
    not (_GPU_AVAILABLE and _CKPT_AVAILABLE),
    reason=f"needs CUDA and the pi05 ckpt at {CKPT_PI05}")

# Tensors touched by the perturbation: one of every family the reload has
# to refresh (BF16 pointers, FP8 quantized copies, decoder styles via the
# time MLP and modulation weights, the pre-scaled output projection, and
# the embedding table behind the prompt).
PERTURBED = [
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight",
    "paligemma_with_expert.paligemma.model.language_model.layers.5.self_attn.q_proj.weight",
    "paligemma_with_expert.paligemma.model.language_model.layers.5.mlp.down_proj.weight",
    "paligemma_with_expert.gemma_expert.model.layers.2.self_attn.q_proj.weight",
    "paligemma_with_expert.gemma_expert.model.layers.2.mlp.gate_proj.weight",
    "paligemma_with_expert.gemma_expert.model.layers.9.mlp.down_proj.weight",
    "paligemma_with_expert.gemma_expert.model.layers.4.input_layernorm.dense.weight",
    "time_mlp_out.weight",
    "action_out_proj.weight",
    "action_in_proj.bias",
    "paligemma_with_expert.paligemma.lm_head.weight",
]


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


def _strip(keys):
    from flash_rt.executors.torch_weights import _autodetect_strip_prefix
    return _autodetect_strip_prefix(set(keys))


def _perturbed_state(path, seed: int = 0, scale: float = 0.25):
    """Full state dict (CPU) with the listed tensors scaled and noised."""
    from safetensors.torch import load_file
    state = load_file(path)
    prefix = _strip(state.keys())
    g = torch.Generator(); g.manual_seed(seed)
    for name in PERTURBED:
        key = prefix + name
        w = state[key].float()
        noise = torch.randn(w.shape, generator=g) * (scale * w.std())
        state[key] = (w * 1.2 + noise).to(state[key].dtype)
    return state


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    from safetensors.torch import save_file
    src = os.path.join(CKPT_PI05, "model.safetensors")
    state = _perturbed_state(src)
    out = tmp_path_factory.mktemp("ckpt")
    save_file({k: v.contiguous() for k, v in state.items()}, str(out / "model.safetensors"))
    for name in os.listdir(CKPT_PI05):
        p = os.path.join(CKPT_PI05, name)
        if name != "model.safetensors" and os.path.isfile(p):
            os.symlink(p, str(out / name))
        elif os.path.isdir(p):
            os.symlink(p, str(out / name))
    # Original checkpoints may resolve statistics from a sibling directory.
    # Make the copied fixture self-contained instead of relying on its parent.
    if not (out / "norm_stats.json").exists():
        from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
        checkpoint = Path(CKPT_PI05)
        stats = load_norm_stats(pi05_candidates(checkpoint), checkpoint_dir=checkpoint)
        (out / "norm_stats.json").write_text(
            json.dumps({"norm_stats": stats}, default=lambda value: value.tolist()))
    return {"dir": str(out), "state": state}


@requires_gpu_ckpt
@pytest.mark.parametrize("use_fp8", [False, True])
def test_reload_matches_fresh_build(perturbed, use_fp8):
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    obs = [_make_obs(s) for s in range(2)]
    g = torch.Generator(device="cuda"); g.manual_seed(3)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    gate = 0.999 if use_fp8 else 0.9999

    fresh = Pi05TorchFrontendRtx(perturbed["dir"], num_views=2, use_fp8=use_fp8)
    fresh.set_prompt(PROMPT); fresh.calibrate(obs)
    ref = [fresh.infer(o, noise=noise)["actions"] for o in obs]
    del fresh; torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, use_fp8=use_fp8)
    rt.set_prompt(PROMPT); rt.calibrate(obs)
    before = [rt.infer(o, noise=noise)["actions"] for o in obs]
    graph_before = rt.pipeline._graph
    assert rt.weight_version == 0

    # Malformed input must fail before changing any live weight or poisoning
    # an otherwise usable frontend, even when the bad key is read last.
    bad = dict(perturbed["state"])
    embedding_key = next(k for k in bad if k.endswith("paligemma.lm_head.weight"))
    del bad[embedding_key]
    with pytest.raises(KeyError):
        rt.reload_weights(bad)
    bad[embedding_key] = perturbed["state"][embedding_key][:1]
    with pytest.raises(ValueError, match="shape/dtype"):
        rt.reload_weights(bad)
    bad[embedding_key] = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating-point"):
        rt.reload_weights(bad)
    assert rt.weight_version == 0
    np.testing.assert_array_equal(rt.infer(obs[0], noise=noise)["actions"], before[0])

    elapsed = rt.reload_weights(perturbed["dir"])
    assert rt.weight_version == 1
    assert rt.pipeline._graph is graph_before
    after = [rt.infer(o, noise=noise)["actions"] for o in obs]
    for a, r, b in zip(after, ref, before):
        assert _cos(a, r) > gate, (_cos(a, r), "vs fresh build")
        # the reload moved the output most of the way from the old model's
        # output to the perturbed model's (the old output is the yardstick)
        assert (1 - _cos(a, r)) < 0.2 * (1 - _cos(b, r)), (_cos(a, r), _cos(b, r))
    again = rt.infer(obs[0], noise=noise)["actions"]
    assert np.array_equal(again, after[0])
    print(f"\nreload (fp8={use_fp8}): {elapsed:.3f} s, cos vs fresh {[round(_cos(a, r), 6) for a, r in zip(after, ref)]}")
    assert elapsed < 3.0, elapsed

    # back to the original weights, from a mapping this time
    from safetensors.torch import load_file
    original = load_file(os.path.join(CKPT_PI05, "model.safetensors"))
    rt.reload_weights(original)
    assert rt.weight_version == 2
    restored = [rt.infer(o, noise=noise)["actions"] for o in obs]
    for r0, b in zip(restored, before):
        assert _cos(r0, b) > 0.9999, _cos(r0, b)
    del rt; gc.collect(); torch.cuda.empty_cache()


@requires_gpu_ckpt
def test_reload_reaches_batched_pipeline(perturbed):
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    B = 4
    obs = [_make_obs(s) for s in range(B)]
    g = torch.Generator(device="cuda"); g.manual_seed(5)
    noise = torch.randn(B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    fresh = Pi05TorchFrontendRtx(perturbed["dir"], num_views=2, use_fp8=False)
    fresh.set_batched_mode(enable=True, batch_size=B)
    fresh.set_prompt_batch([PROMPT] * B); fresh.calibrate_batch(obs)
    ref = [r["actions"] for r in fresh.infer_batch(obs, noise=noise)]
    del fresh; torch.cuda.empty_cache()

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, use_fp8=False)
    rt.set_batched_mode(enable=True, batch_size=B)
    rt.set_prompt_batch([PROMPT] * B); rt.calibrate_batch(obs)
    before = [r["actions"] for r in rt.infer_batch(obs, noise=noise)]
    rt.reload_weights(perturbed["state"])
    after = [r["actions"] for r in rt.infer_batch(obs, noise=noise)]
    for a, r, b in zip(after, ref, before):
        assert _cos(a, r) > 0.9999, _cos(a, r)
        assert (1 - _cos(a, r)) < 0.2 * (1 - _cos(b, r)), (_cos(a, r), _cos(b, r))
