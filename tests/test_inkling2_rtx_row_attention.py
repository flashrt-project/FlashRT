"""Inkling 2 Mini on RTX SM120: prompt chunks on the per-row attention kernels (``flash_prefill=False``; a block
merges several rows). Needs FLASHRT_INKLING2_DIR = the released checkpoint directory, an SM120 GPU and the module
built with FLASHRT_ENABLE_INKLING2; skipped otherwise."""
from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest

CKPT = os.environ.get("FLASHRT_INKLING2_DIR", "")
if not CKPT or not Path(CKPT).is_dir():
    pytest.skip("set FLASHRT_INKLING2_DIR to the Inkling-2-Mini checkpoint", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("needs an SM120 GPU", allow_module_level=True)
pytest.importorskip("flash_rt.flash_rt_inkling2")

TEXT = ("A bounded worker queue applies updates in timestamp order, retries transient failures and reports "
        "backpressure, idempotency and fairness metrics; the service keys events by user and sequence number. ")


@pytest.fixture(scope="module")
def rt():
    from flash_rt.frontends.torch._inkling2_rtx_runtime import Inkling2RtxRuntime
    r = Inkling2RtxRuntime(CKPT, max_seq=4096, flash_prefill=False)
    yield r
    r.graphs.clear()
    r.programs.clear()
    del r
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def ids():
    from flash_rt.models.flint.tokenizer import FlintTokenizer
    out = FlintTokenizer.from_directory(CKPT).encode(TEXT * 40)
    assert len(out) >= 700
    return out[:700]


def test_row_attention_chunks_are_deterministic(rt, ids):
    """Chunks of 512, 128, 32 and 16 rows: the same logits bit for bit on every run."""
    out = []
    for _ in range(3):
        rt.reset()
        first = rt.prefill(ids)
        out.append((first, rt.logits[0].clone()))
    assert all(t == out[0][0] and torch.equal(lg, out[0][1]) for t, lg in out[1:])


def test_row_attention_chunks_track_exact_prefill(rt, ids):
    rt.reset()
    exact = rt.prefill(ids, exact=True)
    ref = rt.logits[0].clone()
    rt.reset()
    assert rt.prefill(ids) == exact
    p, q = torch.log_softmax(ref.float(), -1), torch.log_softmax(rt.logits[0].float(), -1)
    assert float((p.exp() * (p - q)).sum()) < 0.1
