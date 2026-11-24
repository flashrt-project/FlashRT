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
    """With the exact verification the speculative output is the greedy decode's (the prompt prefilled by the same
    chunk programs). The fast verification follows it up to a near-tie: where they part, the greedy step's two
    leading logits are within 0.5."""
    from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps
    spec = Inkling2TorchFrontendMps(CKPT, max_seq=256, speculative=True, verify="exact")
    n = 24
    try:
        got = spec.generate(tokens, max_new_tokens=n, stop_ids=())
        rt = spec.rt
        tok = rt.prefill(list(tokens), features=True)
        want, pos = [tok], len(tokens)
        while len(want) < n:
            out = rt.decode(want[-1], pos, min(8, n - len(want)))
            want += out
            pos += len(out)
        assert got == want
        rt.mtp_verify = "fast"
        fast = spec.generate(tokens, max_new_tokens=n, stop_ids=())
        part = next((i for i, (a, b) in enumerate(zip(fast, want)) if a != b), n)
        assert part >= 4
        if part < n:
            rt.prefill(list(tokens), features=True)
            for i in range(part):
                rt.decode(want[i], len(tokens) + i, 1)
            lg = rt.plan.read(rt.buf["logits"], 0, np.float32, rt.cfg.vocab_size)[:rt.cfg.unpadded_vocab_size]
            top = np.sort(lg)[-2:]
            assert top[1] - top[0] < 0.5
    finally:
        spec.rt.close()


def test_resident_verification_matches_decode_steps(tokens):
    """The exact resident verification of 8 rows repeats the decode step's arithmetic: its logits equal those of 8
    one-row decode steps at the same positions bit for bit. The fast verification selects the same tokens here,
    and its logits stay closer to the exact ones than the row program's (the shipped verification's) do, row by row
    in the maximum and the mean KL (an evaluation side by side, no fixed threshold)."""
    ids = (list(tokens) * 3)[:40]
    rt = _runtime(max_seq=128, row_capacity=32, row_shapes=[(32, 1), (1, 1), (8, 8)], mtp_batches=[1])
    try:
        V, L, R = rt.cfg.vocab_size, 32, 8
        t0 = rt.prefill(ids[:L])
        rows = [t0] + ids[L:L + R - 1]
        exact = rt.verify_resident(rows, L)
        lg = rt.plan.read(rt.buf["r_logits"], 0, np.float32, R * V).reshape(R, V).copy()
        fast = rt.verify_resident(rows, L, kind="fast")
        lf = rt.plan.read(rt.buf["r_logits"], 0, np.float32, R * V).reshape(R, V).copy()
        ref = []
        for r in range(R):
            rt.decode(rows[r], L + r, 1)
            ref.append(rt.plan.read(rt.buf["logits"], 0, np.float32, V).copy())
        ref = np.stack(ref)
        assert np.array_equal(lg.view(np.uint32), ref.view(np.uint32))
        assert exact.tolist() == [int(np.argmax(r[:rt.cfg.unpadded_vocab_size])) for r in ref]
        assert fast.tolist() == exact.tolist()
        # the row program on the same rows (after the decode steps: it reads the same caches below L)
        rt.run_rows(rows, np.array([[0, L + j, L, L + R - 1] for j in range(R)], np.uint32), R)
        lr = rt.plan.read(rt.buf["r_logits"], 0, np.float32, R * V).reshape(R, V).copy()
        vu = rt.cfg.unpadded_vocab_size

        def kl(a, b):
            p = a[:, :vu].astype(np.float64); q = b[:, :vu].astype(np.float64)
            p -= p.max(1, keepdims=True); q -= q.max(1, keepdims=True)
            lp = p - np.log(np.exp(p).sum(1, keepdims=True)); lq = q - np.log(np.exp(q).sum(1, keepdims=True))
            return (np.exp(lp) * (lp - lq)).sum(1)
        kf, kr = kl(lg, lf), kl(lg, lr)
        assert kf.max() <= kr.max() and kf.mean() <= kr.mean(), (kf, kr)
    finally:
        rt.close()


def test_q8_decode_stays_close(tokens):
    """The Q8_0 option (attention projections and LM head from Q8_0 copies) against the BF16 step, teacher forced on
    the BF16 greedy continuation."""
    ids = list(tokens)
    rt = _runtime(max_seq=96, row_capacity=32, row_shapes=[(32, 1), (1, 1)])
    try:
        V, vu = rt.cfg.vocab_size, rt.cfg.unpadded_vocab_size
        tok = rt.prefill(ids)
        seq, ref = [tok], []
        for i in range(8):
            nxt = rt.decode(seq[-1], len(ids) + i, 1)[0]
            ref.append(rt.plan.read(rt.buf["logits"], 0, np.float32, V)[:vu].copy())
            seq.append(nxt)
    finally:
        rt.close()
    rq = _runtime(max_seq=96, row_capacity=32, row_shapes=[(32, 1), (1, 1)], q8=True)
    try:
        assert rq.prefill(ids) == tok
        kls = []
        for i in range(8):
            rq.decode(seq[i], len(ids) + i, 1)
            q = rq.plan.read(rq.buf["logits"], 0, np.float32, V)[:vu].astype(np.float64)
            p = ref[i].astype(np.float64)
            lp = p - p.max(); lp -= np.log(np.exp(lp).sum()); lq = q - q.max(); lq -= np.log(np.exp(lq).sum())
            kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        assert max(kls) < 0.05 and float(np.mean(kls)) < 0.01
    finally:
        rq.close()


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


def test_mma_tiles_match_per_expert_passes(tokens):
    """Above mma_tile_rows the expert MMAs run one 32-row pass per threadgroup (scan tiles) instead of all of an
    expert's passes in one threadgroup: the same passes, so the same logits bit for bit. A repeated token gives a
    few experts most of the rows (several tiles each)."""
    from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime
    n = 96
    ids = [tokens[0]] * 64 + (tokens * 8)[:n - 64]
    info = np.array([[0, i, 0, n - 1] for i in range(n)], np.uint32)
    saved, out = Inkling2Runtime.mma_tile_rows, []
    try:
        for tile_rows in (saved, 1 << 30):
            Inkling2Runtime.mma_tile_rows = tile_rows
            rt = _runtime(max_seq=128, row_capacity=n, row_shapes=[(n, 1)])
            try:
                rt.run_rows(ids, info, 1)
                out.append(rt.rows_logits(0).copy())
                starts = rt.plan.read(rt.buf["r_starts"], 0, np.uint32, rt.cfg.num_experts + 1)
                assert int(np.diff(starts.astype(np.int64)).max()) > 32       # the last visit: an expert with 2+ tiles
            finally:
                rt.close()
    finally:
        Inkling2Runtime.mma_tile_rows = saved
    assert np.array_equal(out[0].view(np.uint32), out[1].view(np.uint32))


def test_concurrent_expert_stages_match_barriers(tokens):
    """Up to expert_overlap_rows rows the qmv experts and the MMA (NVFP4 layers) / TensorOps (MXFP8 layers) experts
    of a stage run as concurrent dispatches: every visit's hidden state of every row equals the program with a
    barrier before each dispatch, bit for bit, on repeated runs (a race would show as a run-to-run difference)."""
    from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime
    shapes, H = (8, 32), None
    ids = (tokens * 8)[:32]
    saved, traces = Inkling2Runtime.expert_overlap_rows, {}
    try:
        for overlap in (0, saved):
            Inkling2Runtime.expert_overlap_rows = overlap
            rt = _runtime(max_seq=64, row_capacity=32, row_shapes=[(n, 1) for n in shapes], trace_rows=True)
            try:
                H = rt.cfg.hidden_size
                for n in shapes:
                    info = np.array([[0, i, 0, n - 1] for i in range(n)], np.uint32)
                    for rep in range(3):
                        rt.run_rows(ids[:n], info, 1)
                        traces[(overlap, n, rep)] = rt.plan.read(rt.buf["r_trace"], 0, np.uint16, 48 * n * H).copy()
            finally:
                rt.close()
    finally:
        Inkling2Runtime.expert_overlap_rows = saved
    assert saved >= max(shapes)
    for n in shapes:
        ref = traces[(0, n, 0)]
        for overlap in (0, saved):
            for rep in range(3):
                assert np.array_equal(traces[(overlap, n, rep)], ref), f"{n} rows, overlap {overlap}, run {rep}"
