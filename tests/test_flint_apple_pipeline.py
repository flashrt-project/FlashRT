#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of the Flint native Metal runtime on the MLX affine int4 checkpoint.

Skipped unless ``FLASHRT_FLINT_MLX_DIR`` points at a ``RecursiveSI/flint-int4`` directory, the host is
an Apple Silicon Mac and the plan library is built (``scripts/apple/build_native.sh``).

* the production ``greedy`` decode program is identical to the ``reference`` program (logits of every
  step) while every global layer has at most 1024 keys;
* single-chunk, small-chunk and multi-chunk prefills are deterministic;
* chunked prefill + decode agrees with stepping the prompt through the decode program;
* a decode repeated from a restored sliding-window state repeats its tokens; a prefill continued at a chunk
  boundary equals the full prefill; a follow-up request reuses the cached
  prefix, and a regenerate whose sliding-window history was overwritten recomputes the prompt;
* seeded sampling is reproducible and depends on the seed;
* generation stops after a stop token, inside a decode block.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform

import numpy as np
import pytest

MLX_DIR = os.environ.get("FLASHRT_FLINT_MLX_DIR")
LIBRARY = Path(os.environ.get("FLASHRT_APPLE_PLAN_LIBRARY") or
               Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib")
if not MLX_DIR or not Path(MLX_DIR, "config.json").is_file():
    pytest.skip("set FLASHRT_FLINT_MLX_DIR to the Flint MLX int4 checkpoint", allow_module_level=True)
if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("needs an Apple Silicon Mac", allow_module_level=True)
if not LIBRARY.is_file():
    pytest.skip("build the Apple plan library with scripts/apple/build_native.sh", allow_module_level=True)

PROMPT = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."}]


@pytest.fixture(scope="module")
def fe():
    from flash_rt.frontends.torch.flint_mps import FlintTorchFrontendMps
    return FlintTorchFrontendMps(MLX_DIR, max_seq=8192, library=str(LIBRARY))


@pytest.fixture(scope="module")
def chat_ids(fe):
    return fe.tokenizer.encode(fe.tokenizer.apply_chat_template(PROMPT))


def code_ids(fe, n):
    text = "".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(4 * n))
    ids = fe.tokenizer.encode(text)
    assert len(ids) >= n
    return ids[:n]


def greedy(rt, first, position, n):
    out, token = [first], first
    while len(out) < n:
        block = rt.decode(token, position, min(8, n - len(out)), program="greedy")
        out += block
        position += len(block)
        token = block[-1]
    return out


def test_greedy_program_matches_reference(fe, chat_ids):
    rt, n = fe.rt, len(chat_ids)
    first = rt.prefill(chat_ids)
    runs = {}
    for program in ("reference", "greedy"):
        tokens = rt.decode(first, n, 8, program=program)
        runs[program] = (tokens, [rt.logits(s).copy() for s in range(len(tokens))])
    assert runs["greedy"][0] == runs["reference"][0]
    for a, b in zip(runs["greedy"][1], runs["reference"][1]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [300, 2048, 2600])
def test_prefill_is_deterministic(fe, n):
    rt, ids = fe.rt, code_ids(fe, n)
    first = rt.prefill(ids)
    logits = rt.logits(0).copy()
    assert rt.prefill(ids) == first
    np.testing.assert_array_equal(rt.logits(0), logits)


def test_chunked_prefill_matches_stepwise(fe, chat_ids):
    rt, n = fe.rt, len(chat_ids)
    chunked = greedy(rt, rt.prefill(chat_ids), n, 32)
    stepwise = greedy(rt, rt.prefill_stepwise(chat_ids), n, 32)
    assert chunked == stepwise


def test_restored_sliding_state_repeats_a_decode(fe):
    rt, ids = fe.rt, code_ids(fe, 300)
    first = rt.prefill(ids)
    state = rt.sliding_state()
    tokens = greedy(rt, first, len(ids), 600)   # decodes past the ring: the prompt's sliding history is gone
    rt.restore_sliding_state(state)
    assert greedy(rt, first, len(ids), 600) == tokens


def test_continued_prefill_matches_full_prefill_at_a_chunk_boundary(fe):
    rt, ids = fe.rt, code_ids(fe, 2600)
    first = rt.prefill(ids)
    logits = rt.logits(0).copy()
    rt.prefill(ids[:2048])
    assert rt.prefill(ids, start=2048) == first   # the same chunks as one 2600-token prefill
    np.testing.assert_array_equal(rt.logits(0), logits)


def test_prefix_reuse(fe, chat_ids):
    first = fe.generate(chat_ids, max_new_tokens=64, stop_ids=[])
    follow = chat_ids + first + fe.tokenizer.encode(" Now add memoization.")
    fe.generate(follow, max_new_tokens=8, stop_ids=[], reuse_prefix=True)
    assert fe.last_reused == len(chat_ids) + 63   # prompt and generated tokens except the last one


def test_regenerate_recomputes_overwritten_sliding_history(fe):
    ids = code_ids(fe, 700)
    tokens = fe.generate(ids, max_new_tokens=32, stop_ids=[])
    # The 31 decoded positions overwrote ring slots the reuse position would need: recompute from 0.
    assert fe.generate(ids, max_new_tokens=32, stop_ids=[], reuse_prefix=True) == tokens
    assert fe.last_reused == 0


def test_seeded_sampling(fe, chat_ids):
    sampling = dict(temperature=1.0, top_p=0.95, top_k=20)
    a = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=7, **sampling)
    b = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=7, **sampling)
    c = fe.generate(chat_ids, max_new_tokens=48, stop_ids=[], seed=8, **sampling)
    assert a == b
    assert a != c


def test_stop_token_inside_a_block(fe, chat_ids):
    tokens = fe.generate(chat_ids, max_new_tokens=16, stop_ids=[])
    stop = next(t for i, t in enumerate(tokens) if i >= 3 and t not in tokens[:i])
    cut = tokens.index(stop) + 1
    assert fe.generate(chat_ids, max_new_tokens=16, stop_ids=[stop]) == tokens[:cut]
