#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of the Flint runtime on the official NVFP4 GGUF.

Skipped unless ``FLASHRT_FLINT_GGUF`` points at the checkpoint (and the SM120
``flash_rt_flint`` module is built). Each configuration runs in a fresh process
because the launch switches are read once per process.

* fused decode with programmatic dependent launch must produce the same greedy
  tokens as the same step without PDL (catches PDL ordering bugs);
* the fused step must agree with the unfused reference step;
* chunked prefill + decode must agree with stepping the prompt through the
  decode graph;
* a follow-up request that reuses the cached prefix must continue like a full
  prefill of the same tokens;
* prefill chunks replayed from CUDA graphs (rows padded to the graph shape) must
  give the same tokens as eager launches of the exact chunk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

GGUF = os.environ.get("FLASHRT_FLINT_GGUF")
if not GGUF or not os.path.exists(GGUF):
    pytest.skip("set FLASHRT_FLINT_GGUF to the official Flint NVFP4 GGUF", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("flash_rt.flash_rt_flint")

PROMPT = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."}]

_CHILD = r"""
import json, os, sys
from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
mode = sys.argv[3]
fe = FlintTorchFrontendRtx(sys.argv[1], max_seq=32768 if mode == "reuse_long_fork" else 2048)
ids = fe.tokenizer.encode(fe.tokenizer.apply_chat_template(json.loads(sys.argv[2])))
if mode == "stepwise":
    rt = fe.rt
    rt.reset(ids)
    rt.replay(len(ids) + 47)
    out = rt.tokens[len(ids):len(ids) + 48].tolist()
elif mode == "reuse":
    first = fe.generate(ids, max_new_tokens=64, stop_ids=[])
    follow = ids + first + fe.tokenizer.encode(" Now add memoization.")
    out = [int(t) for b in fe.stream(follow, 32, stop_ids=(), reuse_prefix=True) for t in b] + [fe.last_reused]
elif mode == "evict":   # three graph shapes through a 2-entry cache, then a cleared cache
    fe.rt.pf_graph_max = 2
    out = []
    for rep in (1, 2, 3, 1):
        out += fe.generate(ids * rep, max_new_tokens=4, stop_ids=[])
    fe.rt.pf_graphs.clear()
    out += fe.generate(ids * 2, max_new_tokens=4, stop_ids=[])
elif mode == "follow_full":
    first = fe.generate(ids, max_new_tokens=64, stop_ids=[])
    follow = ids + first + fe.tokenizer.encode(" Now add memoization.")
    out = fe.generate(follow, max_new_tokens=32, stop_ids=[]) + [0]
elif mode == "reuse_long_fork":
    import torch
    old = (ids * (20000 // len(ids) + 1))[:20000]
    fe.generate(old, max_new_tokens=8, stop_ids=())
    retained_start = fe.rt.swa_kv_start
    follow = old[:1000] + fe.tokenizer.encode(" Instead, explain binary search.")
    cached = [t for block in fe.stream(follow, 32, stop_ids=(), reuse_prefix=True) for t in block]
    reused = fe.last_reused
    logits = fe.rt.logits.clone()
    full = fe.generate(follow, max_new_tokens=32, stop_ids=())
    out = [retained_start > 0, reused, cached == full, torch.equal(logits, fe.rt.logits)]
else:
    out = fe.generate(ids, max_new_tokens=48, stop_ids=[])
print(json.dumps(out))
"""


def _generate(env_extra, mode="prefill"):
    env = dict(os.environ, **env_extra)
    r = subprocess.run([sys.executable, "-c", _CHILD, GGUF, json.dumps(PROMPT), mode], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_pdl_does_not_change_tokens():
    assert _generate({"FLINT_PDL": "1"}) == _generate({"FLINT_PDL": "0"})


def test_fused_step_matches_basic_step():
    fused = _generate({"FLINT_DECODE": "fused"})
    basic = _generate({"FLINT_DECODE": "basic"})
    # The two step implementations differ only in fp32 summation order / bf16 P rounding layout;
    # greedy tokens agree for the opening of the reply.
    assert fused[:16] == basic[:16]


def test_prefill_matches_stepwise_prompt():
    assert _generate({}, "prefill")[:16] == _generate({}, "stepwise")[:16]


def test_prefix_reuse_matches_full_prefill():
    reuse = _generate({}, "reuse")
    full = _generate({}, "follow_full")
    assert reuse[-1] >= 64          # the cached prompt + first reply were reused
    assert reuse[:16] == full[:16]


@pytest.mark.parametrize("graphs", ["0", "1"])
def test_long_request_early_fork_recomputes_evicted_history(graphs):
    assert _generate({"FLINT_PF_GRAPH": graphs}, "reuse_long_fork") == [True, 0, True, True]


def test_prefill_graphs_match_eager_launches():
    for mode in ("prefill", "reuse", "evict"):
        assert _generate({"FLINT_PF_GRAPH": "1"}, mode) == _generate({"FLINT_PF_GRAPH": "0"}, mode)
