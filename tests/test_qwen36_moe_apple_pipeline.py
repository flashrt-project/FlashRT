#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of the Qwen3.6-35B-A3B native Metal runtime on the MLX affine int4 checkpoint.

Skipped unless ``FLASHRT_QWEN36_MOE_MLX_DIR`` points at a ``mlx-community/Qwen3.6-35B-A3B-4bit`` directory, the
host is an Apple Silicon Mac and the plan library is built (``scripts/apple/build_native.sh``).

* one-token steps from an empty context reproduce mlx-lm's logits bit for bit (needs ``mlx_lm``; skipped
  otherwise; both engines are loaded, about 40 GB), below and above the 1024-key switch of the attention kernel;
* a block of eight greedy steps equals eight single steps (tokens and logits);
* chunked prefills (small-chunk, single-chunk and multi-chunk programs) are deterministic, a prefill continued
  at a chunk boundary equals the full prefill, and the chunked prefill agrees with stepping the prompt through
  the decode program up to reordering noise;
* a decode repeated from a restored recurrent state repeats its tokens;
* seeded sampling is reproducible and depends on the seed, from the first token on;
* generation stops after a stop token, inside a decode block; a follow-up request reuses the held context.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform

import numpy as np
import pytest

MLX_DIR = os.environ.get("FLASHRT_QWEN36_MOE_MLX_DIR")
LIBRARY = Path(os.environ.get("FLASHRT_APPLE_PLAN_LIBRARY") or
               Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib")
if not MLX_DIR or not Path(MLX_DIR, "config.json").is_file():
    pytest.skip("set FLASHRT_QWEN36_MOE_MLX_DIR to the Qwen3.6-35B-A3B MLX int4 checkpoint", allow_module_level=True)
if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("needs an Apple Silicon Mac", allow_module_level=True)
if not LIBRARY.is_file():
    pytest.skip("build the Apple plan library with scripts/apple/build_native.sh", allow_module_level=True)

PROMPT = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."}]


@pytest.fixture(scope="module")
def fe():
    from flash_rt.frontends.torch.qwen36_moe_mps import Qwen36MoeTorchFrontendMps
    return Qwen36MoeTorchFrontendMps(MLX_DIR, max_seq=8192, library=str(LIBRARY))


@pytest.fixture(scope="module")
def chat_ids(fe):
    return fe.tokenizer.encode(fe.tokenizer.apply_chat_template(PROMPT))


def code_ids(fe, n):
    text = "".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(4 * n))
    ids = fe.tokenizer.encode(text)
    assert len(ids) >= n
    return ids[:n]


def log_probs(raw):
    from flash_rt.models.flint.mlx_checkpoint import bf16_to_f32
    x = bf16_to_f32(raw).astype(np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def test_single_steps_reproduce_mlx_lm(fe):
    mx = pytest.importorskip("mlx.core")
    mlx_lm = pytest.importorskip("mlx_lm")
    model, _ = mlx_lm.load(MLX_DIR)
    ids = code_ids(fe, 1040)
    cache = model.make_cache()
    rt = fe.rt
    rt.reset()
    checked = 0
    for t, tok in enumerate(ids):
        ref = model(mx.array([[tok]]), cache=cache)[0, -1]
        rt.decode(tok, t, 1)
        if t < 12 or t >= 1020:   # single-pass attention, then the two-pass path from 1024 keys
            raw = (np.array(ref.astype(mx.float32)).view(np.uint32) >> 16).astype(np.uint16)
            assert np.array_equal(rt.logits(0), raw), f"logits differ at step {t}"
            checked += 1
        else:
            mx.eval(ref)
    assert checked == 32
    del model, cache
    mx.clear_cache()


def test_block_equals_single_steps(fe, chat_ids):
    rt = fe.rt
    first = rt.prefill(chat_ids)
    state = rt.recurrent_state()
    block = rt.decode(first, len(chat_ids), 8)
    block_logits = [rt.logits(i).copy() for i in range(8)]
    rt.restore_recurrent_state(state)
    token = first
    for i in range(8):
        token = rt.decode(token, len(chat_ids) + i, 1)[0]
        assert token == block[i]
        assert np.array_equal(rt.logits(0), block_logits[i])


@pytest.mark.parametrize("n", [300, 1500, 2600])
def test_prefill_is_deterministic(fe, n):
    rt = fe.rt
    ids = code_ids(fe, n)
    first = rt.prefill(ids)
    logits = rt.logits(0).copy()
    state = rt.recurrent_state()
    assert rt.prefill(ids) == first
    assert np.array_equal(rt.logits(0), logits)
    again = rt.recurrent_state()
    assert np.array_equal(state[0], again[0]) and np.array_equal(state[1], again[1])


def test_continued_prefill_matches_full_prefill_at_a_chunk_boundary(fe):
    rt = fe.rt
    ids = code_ids(fe, 2600)
    rt.prefill(ids)
    logits, state = rt.logits(0).copy(), rt.recurrent_state()
    rt.prefill(ids[:2048])
    rt.prefill(ids, start=2048)
    assert np.array_equal(rt.logits(0), logits)
    assert np.array_equal(rt.recurrent_state()[0], state[0])


@pytest.mark.parametrize("n", [200, 700])
def test_chunked_prefill_agrees_with_stepwise(fe, n):
    rt = fe.rt
    ids = code_ids(fe, n)
    rt.prefill(ids)
    chunked = log_probs(rt.logits(0))
    rt.prefill_stepwise(ids)
    stepwise = log_probs(rt.logits(0))
    kl = float((np.exp(stepwise) * (stepwise - chunked)).sum())
    assert kl < 0.5                                    # reordering noise of the 4-bit model, not a different result
    assert int(stepwise.argmax()) in np.argsort(chunked)[-5:]


def test_restored_state_repeats_a_decode(fe, chat_ids):
    rt = fe.rt
    first = rt.prefill(chat_ids)
    state = rt.recurrent_state()
    def two_blocks():
        block = rt.decode(first, len(chat_ids), 8)
        return block + rt.decode(block[-1], len(chat_ids) + 8, 8)
    a = two_blocks()
    rt.restore_recurrent_state(state)
    assert two_blocks() == a


def test_seeded_sampling(fe, chat_ids):
    sampling = dict(temperature=1.0, top_p=0.95, top_k=20)
    a = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=7, **sampling)
    b = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=7, **sampling)
    c = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=8, **sampling)
    assert a == b
    assert a != c


@pytest.mark.parametrize("n", [300, 2048, 2600])
def test_seeded_first_token(fe, n):
    ids = code_ids(fe, n)
    sampling = dict(temperature=50.0, top_p=1.0, top_k=20)
    first = [fe.generate(ids, max_new_tokens=1, stop_ids=[], seed=s, **sampling)[0] for s in range(8)]
    assert first == [fe.generate(ids, max_new_tokens=1, stop_ids=[], seed=s, **sampling)[0] for s in range(8)]
    assert len(set(first)) > 1          # near-uniform over the 20 largest logits, not the greedy token


def test_stop_token_inside_a_block(fe, chat_ids):
    tokens = fe.generate(chat_ids, max_new_tokens=16, stop_ids=[])
    stop = next(t for i, t in enumerate(tokens) if i >= 3 and t not in tokens[:i])
    cut = tokens.index(stop) + 1
    assert fe.generate(chat_ids, max_new_tokens=16, stop_ids=[stop]) == tokens[:cut]


def test_follow_up_reuses_the_held_context(fe, chat_ids):
    tokens = fe.generate(chat_ids, max_new_tokens=24, stop_ids=[])
    follow = chat_ids + tokens + fe.tokenizer.encode("\nNow explain it.")
    out = fe.generate(follow, max_new_tokens=8, stop_ids=[], reuse_prefix=True)
    assert fe.last_reused == len(chat_ids) + len(tokens) - 1
    assert len(out) == 8 and all(0 <= t < fe.cfg.vocab_size for t in out)
    fe.generate(chat_ids[:-1] + [chat_ids[-1] ^ 1], max_new_tokens=2, stop_ids=[], reuse_prefix=True)
    assert fe.last_reused == 0
