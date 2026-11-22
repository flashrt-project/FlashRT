"""Inkling 2 Mini on the native Metal plan (macOS, Apple Silicon). Needs FLASHRT_INKLING2_DIR = the released
checkpoint directory (thinkingmachines/Inkling-2-Mini); skipped otherwise. Self-consistency checks between the
resident decode step, the row programs and the speculative path (no reference data)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

CKPT = os.environ.get("FLASHRT_INKLING2_DIR", "")
pytestmark = pytest.mark.skipif(sys.platform != "darwin" or not CKPT or not Path(CKPT).is_dir(),
                                reason="needs macOS and FLASHRT_INKLING2_DIR")

TEXT = ("FlashRT runs the released checkpoint in its own precisions: BF16 attention and NVFP4 or MXFP8 experts, "
        "with a resident decode step and chunked prefill row programs on the native Metal plan.")


@pytest.fixture(scope="module")
def tokens():
    from flash_rt.models.flint.tokenizer import FlintTokenizer
    ids = FlintTokenizer.from_directory(CKPT).encode(TEXT)
    assert len(ids) >= 16
    return ids


def _runtime(**kw):
    from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime
    return Inkling2Runtime(CKPT, **kw)


def test_one_row_program_matches_the_resident_step(tokens):
    """The few-row path (multi-row gemv / per-expert qmv) runs the decode step's arithmetic: at position 0 every
    visit's hidden state equals the resident step's bit for bit."""
    rt = _runtime(max_seq=64, row_capacity=1, row_shapes=[(1, 1)], trace_rows=True)
    try:
        rt.decode(tokens[0], 0, 1, program="reference")
        ref = rt.trace(0).copy()
        rt.run_rows([tokens[0]], np.array([[0, 0, 0, 0]], np.uint32), 1)
        rows = rt.plan.read(rt.buf["r_trace"], 0, np.uint16, ref.size).reshape(ref.shape)
        assert np.array_equal(rows, ref)
    finally:
        rt.close()


def test_batched_rows_do_not_depend_on_the_batch(tokens):
    """Four sequences in one decode step produce, row by row, the hidden states each one produces alone."""
    rt = _runtime(max_seq=64, sequences=4, row_capacity=4, row_shapes=[(1, 1), (4, 4)], trace_rows=True)
    try:
        H = rt.cfg.hidden_size
        batch = tokens[1:5]
        rt.run_rows(batch, np.array([[s, 0, 0, 0] for s in range(4)], np.uint32), 4)
        both = rt.plan.read(rt.buf["r_trace"], 0, np.uint16, 48 * 4 * H).reshape(48, 4, H).copy()
        for s, tok in enumerate(batch):
            rt.run_rows([tok], np.array([[s, 0, 0, 0]], np.uint32), 1)
            alone = rt.plan.read(rt.buf["r_trace"], 0, np.uint16, 48 * H).reshape(48, H)
            assert np.array_equal(both[:, s], alone), f"sequence {s}"
    finally:
        rt.close()


def test_chunked_prefill_agrees_with_one_chunk(tokens):
    n = 37
    ids = (tokens * 4)[:n]
    rt = _runtime(max_seq=128, row_capacity=n, row_shapes=[(n, 1), (32, 1), (4, 1), (1, 1)])
    try:
        rt.run_rows(ids, np.array([[0, i, 0, n - 1] for i in range(n)], np.uint32), 1)
        whole = rt.rows_logits(0).copy()
        first = rt.prefill(ids)          # 32 + 4 + 1 rows
        chunked = rt.rows_logits(0)
        assert int(np.argmax(whole)) == first == int(np.argmax(chunked))
        p = np.exp(whole - whole.max()); p /= p.sum()
        q = np.exp(chunked - chunked.max()); q /= q.sum()
        assert float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)))) < 1e-3
    finally:
        rt.close()


def test_speculative_decoding_follows_greedy(tokens):
    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    plain = Inkling2TorchFrontendMps(CKPT, max_seq=256)
    want = plain.generate(tokens, max_new_tokens=8, stop_ids=())
    plain.rt.close()
    spec = Inkling2TorchFrontendMps(CKPT, max_seq=256, speculative=True)
    got = spec.generate(tokens, max_new_tokens=8, stop_ids=())
    spec.rt.close()
    assert got == want


def test_chat_and_input_checks():
    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    fe = Inkling2TorchFrontendMps(CKPT, max_seq=512)
    try:
        text = fe.chat([{"role": "user", "content": "Name three primary colors."}], max_new_tokens=24,
                       reasoning_effort="none")
        assert isinstance(text, str) and text
        with pytest.raises(ValueError):
            fe.generate([fe.cfg.unpadded_vocab_size], max_new_tokens=4)
        with pytest.raises(ValueError):
            fe.generate([1] * 500, max_new_tokens=64)
    finally:
        fe.rt.close()
