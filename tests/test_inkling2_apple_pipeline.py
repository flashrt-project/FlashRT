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


def test_row_program_input_checks(tokens):
    rt = _runtime(max_seq=64, sequences=2, row_capacity=4, row_shapes=[(4, 1), (2, 2)])
    try:
        ok = np.array([[0, i, 0, 3] for i in range(4)], np.uint32)
        rt.run_rows(tokens[:4], ok, 1)
        bad = [np.array([[0, i, 0, 3] for i in (0, 2, 1, 3)], np.uint32),     # positions out of row order
               np.array([[0, i, 0, 3] for i in (0, 1, 2, 64)], np.uint32),    # beyond max_seq
               np.array([[2, i, 0, 3] for i in range(4)], np.uint32)]         # no such sequence slot
        for info in bad:
            with pytest.raises(ValueError):
                rt.run_rows(tokens[:4], info, 1)
        with pytest.raises(ValueError):
            rt.run_rows([rt.cfg.unpadded_vocab_size, 1], np.array([[0, 0, 0, 0], [1, 0, 0, 0]], np.uint32), 2)
        with pytest.raises(ValueError):
            rt.mtp_generate([tokens], 8)                                       # no MTP programs recorded
    finally:
        rt.close()


def test_sampling(tokens):
    """top_k = 1 samples a largest logit (on an exact tie the sampler ranks the larger id first, as mlx-lm's top-p
    ordering does; greedy takes the lowest id); a seed reproduces its sequence; another seed draws another one."""
    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    fe = Inkling2TorchFrontendMps(CKPT, max_seq=256)
    try:
        rt, n = fe.rt, len(tokens)
        first = rt.prefill(tokens)
        for step in range(4):
            greedy = rt.decode(first, n + step, 1, program="greedy")[0]
            logits = rt.logits(0).copy()
            pick = rt.decode(first, n + step, 1, program="sampled", top_k=1, top_p=1.0, temperature=1.0, seed=3)[0]
            assert logits[pick] == logits.max() == logits[greedy]
            first = greedy
        a = fe.generate(tokens, max_new_tokens=20, stop_ids=(), temperature=1.5, top_k=32, top_p=0.95, seed=7)
        b = fe.generate(tokens, max_new_tokens=20, stop_ids=(), temperature=1.5, top_k=32, top_p=0.95, seed=7)
        c = fe.generate(tokens, max_new_tokens=20, stop_ids=(), temperature=1.5, top_k=32, top_p=0.95, seed=8)
        assert a == b and a != c
        assert all(0 <= t < fe.cfg.unpadded_vocab_size for t in a + c)
    finally:
        fe.rt.close()


def test_mma_expert_copy_matches_the_reference_arrangement():
    """The load-time re-arranged NVFP4 experts (ink_nv_arrange) against nvfp4_mma_arrangement, first and last
    NVFP4 layer, first and last expert, gate/up and down."""
    from flash_rt.frontends.torch import _inkling2_mps_runtime as m
    rt = _runtime(max_seq=64, row_capacity=8, row_shapes=[(8, 1)])
    try:
        cfg = rt.cfg
        H, I, E = cfg.hidden_size, cfg.expert_intermediate_size, cfg.num_experts
        layers = sorted(rt._arranged)
        for layer in (layers[0], layers[-1]):
            d, offs = rt.table[layer], rt._arranged[layer]
            for (rows, k), (w, s), (ao, so) in zip(((2 * I, H), (H, I)), ((m.D_W13, m.D_S13), (m.D_W2, m.D_S2)),
                                                   ((offs[0], offs[1]), (offs[2], offs[3]))):
                cb, sb = rows * k // 2, rows * k // 16
                for e in (0, E - 1):
                    codes = rt.plan.read(rt.buf["weights"], int(d[w]) + e * cb, np.uint8, cb).reshape(rows, k // 2)
                    scales = rt.plan.read(rt.buf["weights"], int(d[s]) + e * sb, np.uint8, sb).reshape(rows, k // 16)
                    want_c, want_s = m.nvfp4_mma_arrangement(codes, scales)
                    got_c = rt.plan.read(rt.buf["arranged"], ao + e * cb, np.uint8, cb).reshape(want_c.shape)
                    got_s = rt.plan.read(rt.buf["arranged"], so + e * sb, np.uint8, sb).reshape(want_s.shape)
                    assert np.array_equal(got_c, want_c) and np.array_equal(got_s, want_s), (layer, rows, k, e)
    finally:
        rt.close()
