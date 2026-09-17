"""Prompt embedding cache on the Pi0.5 RTX frontend.

The tokenizer is built once per process and each prompt's embedding rows
are cached per frontend, so a fleet whose tasks rotate at episode
boundaries does not re-tokenise, re-embed and synchronise every slot of
the batch on every rotation. The batched pipeline uploads only the slots
whose prompt changed.

Host-only tests cover the cache and tokenizer helpers; the GPU tests
(RTX 5090 / 4090, pi05_libero PyTorch checkpoint, set
``PI05_LIBERO_PYTORCH_CHECKPOINT`` to override) check that rotated
prompts produce bit-identical actions to a cold re-embedding, that the
rotation is cheap, and that a weight reload drops the cache.

Run::

    python -m pytest tests/test_pi05_prompt_cache.py -v
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

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)
requires_gpu_ckpt = pytest.mark.skipif(
    not (_GPU_AVAILABLE and _CKPT_AVAILABLE),
    reason=f"needs CUDA and the pi05 ckpt at {CKPT_PI05}")

CANDIDATES = [
    "put the red block in the box", "put the blue block in the box",
    "put the green block in the box", "put the black bowl on the plate",
    "put the white bowl on the plate", "pick up the red cup", "pick up the blue cup",
    "pick up the green cup", "open the top drawer", "open the middle drawer",
    "close the top drawer", "close the middle drawer",
]


def test_cache_is_lru_and_keys_on_text_len_state():
    from flash_rt.frontends.torch.pi05_rtx import _PromptEmbedCache
    c = _PromptEmbedCache(capacity=2)
    k1 = _PromptEmbedCache.key("a", 48, None)
    k2 = _PromptEmbedCache.key("a", 48, np.zeros(8, np.float32))
    k3 = _PromptEmbedCache.key("a", 200, None)
    assert len({k1, k2, k3}) == 3
    assert _PromptEmbedCache.key("a", 48, np.zeros(8, np.float32)) == k2
    c.put(k1, 1); c.put(k2, 2)
    assert c.get(k1) == 1          # k1 is now most recently used
    c.put(k3, 3)                   # evicts k2
    assert c.get(k2) is None and c.get(k1) == 1 and c.get(k3) == 3 and len(c) == 2
    c.clear(); assert len(c) == 0


def test_changing_state_does_not_grow_embedding_cache(monkeypatch):
    from types import SimpleNamespace
    from flash_rt.frontends.torch import pi05_rtx as module

    calls = []
    def embed(text, weight, max_len, state):
        calls.append(state)
        value = 0.0 if state is None else float(state[0])
        return torch.full((2, 4), value, dtype=torch.bfloat16), 2

    monkeypatch.setattr(module, "_embed_prompt", embed)
    frontend = SimpleNamespace(embedding_weight=None)
    run = module.Pi05TorchFrontendRtx._embed_prompt_cached
    task = run(frontend, "task", 200)
    for i in range(600):
        result = run(frontend, "task", 200, np.array([i], dtype=np.float32))
        assert float(result[0][0, 0]) == float(torch.tensor(i).bfloat16())
        assert len(frontend._prompt_embed_cache) == 1
    assert run(frontend, "task", 200) is task
    assert len(calls) == 601


def test_batch_width_switch_uses_lifecycle_guard():
    from types import SimpleNamespace
    from threading import RLock
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    class RecordingLock:
        def __init__(self):
            self.lock = RLock()
            self.entries = 0

        def __enter__(self):
            self.lock.acquire()
            self.entries += 1

        def __exit__(self, *args):
            self.lock.release()

    lock = RecordingLock()
    frontend = SimpleNamespace(_lifecycle_lock=lock, _reload_failed=False,
                               _batched_active=True, _batch_size=4)
    Pi05TorchFrontendRtx.select_batch_size(frontend, 4)
    assert lock.entries == 1
    frontend._reload_failed = True
    with pytest.raises(RuntimeError, match="reload failed after mutation"):
        Pi05TorchFrontendRtx.select_batch_size(frontend, 4)
    assert lock.entries == 2


def _tokenizer_available() -> bool:
    try:
        from flash_rt.frontends.torch.pi05_rtx import _get_tokenizer
        _get_tokenizer(48)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tokenizer_available(), reason="PaliGemma tokenizer not available")
def test_tokenizer_built_once_and_ids_match_direct_encoding():
    from flash_rt.frontends.torch.pi05_rtx import _get_tokenizer, _prompt_token_ids
    a = _get_tokenizer(48); b = _get_tokenizer(48)
    assert a[1] is b[1]
    ids = _prompt_token_ids("pick_up the\nbowl ", 48)
    assert ids[0] > 0 and ids[-1] == 108 and len(ids) > 3
    assert ids == _prompt_token_ids("pick_up the\nbowl ", 48)
    if a[0] == "sp":
        sp = a[1]
        assert ids == [sp.bos_id()] + list(sp.Encode("pick up the bowl")) + [108]
    t0 = time.perf_counter()
    for _ in range(20):
        _prompt_token_ids("open the top drawer", 48)
    assert (time.perf_counter() - t0) / 20 < 0.005


def _equal_length_prompts(n: int):
    from flash_rt.frontends.torch.pi05_rtx import _prompt_token_ids
    by_len = {}
    for p in CANDIDATES:
        by_len.setdefault(len(_prompt_token_ids(p, 48)), []).append(p)
    for group in by_len.values():
        if len(group) >= n:
            return group[:n]
    return None


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


@requires_gpu_ckpt
def test_batched_prompt_rotation_is_exact_and_cheap():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    B = 4
    prompts = _equal_length_prompts(3)
    assert prompts is not None, "need three candidate prompts with equal token counts"
    obs = [_make_obs(s) for s in range(B)]
    g = torch.Generator(device="cuda"); g.manual_seed(5)
    noise = torch.randn(B, 10, 32, generator=g, device="cuda", dtype=torch.bfloat16)

    fe = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, use_fp8=True)
    fe.set_batched_mode(enable=True, batch_size=B)
    first = [prompts[0], prompts[1], prompts[0], prompts[2]]
    fe.set_prompt_batch(first); fe.calibrate_batch(obs)
    graph = fe.pipeline._graph
    out_first = [r["actions"] for r in fe.infer_batch(obs, noise=noise)]
    assert len(fe._prompt_embed_cache) == 3

    # rotate the tasks across the slots: no rebuild, cache hits only
    second = [prompts[1], prompts[0], prompts[2], prompts[0]]
    t0 = time.perf_counter(); fe.set_prompt_batch(second); rot = time.perf_counter() - t0
    assert fe.pipeline._graph is graph and fe.current_prompt_len == fe.pipeline._current_prompt_len_b2
    assert len(fe._prompt_embed_cache) == 3
    out_rot = [r["actions"] for r in fe.infer_batch(obs, noise=noise)]

    # same rotation with a cold cache and a full upload is the reference
    fe._prompt_embed_cache.clear(); fe._batch_prompt_texts = None
    fe.set_prompt_batch(second)
    out_cold = [r["actions"] for r in fe.infer_batch(obs, noise=noise)]
    for b in range(B):
        assert np.array_equal(np.asarray(out_rot[b]), np.asarray(out_cold[b])), f"slot {b} differs from cold re-embed"
    # slots that kept their prompt (none here) or whose prompt moved must
    # differ from the first arrangement where the prompt changed
    for b in range(B):
        if first[b] != second[b]:
            assert not np.array_equal(np.asarray(out_first[b]), np.asarray(out_rot[b]))
    assert rot < 0.02, f"rotation took {rot * 1e3:.1f} ms"

    # a reload drops the cache (embedding table may change) and re-embeds the live prompts
    fe.reload_weights(CKPT_PI05)
    assert len(fe._prompt_embed_cache) == 3
    out_reload = [r["actions"] for r in fe.infer_batch(obs, noise=noise)]
    for b in range(B):
        assert np.array_equal(np.asarray(out_reload[b]), np.asarray(out_rot[b]))


@requires_gpu_ckpt
def test_single_prompt_switch_hits_cache():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    gc.collect(); torch.cuda.empty_cache()
    prompts = _equal_length_prompts(2)
    assert prompts is not None
    obs = _make_obs(1)
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    noise = torch.randn(10, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    fe = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2, use_fp8=True)
    fe.set_prompt(prompts[0]); fe.calibrate([obs])
    a0 = np.asarray(fe.infer(obs, noise=noise)["actions"])
    fe.set_prompt(prompts[1]); a1 = np.asarray(fe.infer(obs, noise=noise)["actions"])
    t0 = time.perf_counter(); fe.set_prompt(prompts[0]); sw = time.perf_counter() - t0
    a0b = np.asarray(fe.infer(obs, noise=noise)["actions"])
    assert np.array_equal(a0, a0b) and not np.array_equal(a0, a1)
    assert len(fe._prompt_embed_cache) == 2 and sw < 0.02
