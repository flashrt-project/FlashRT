"""Inkling 2 Mini on RTX SM120 (``flash_rt_inkling2``). Needs FLASHRT_INKLING2_DIR = the released checkpoint
directory (thinkingmachines/Inkling-2-Mini), an SM120 GPU and the module built with FLASHRT_ENABLE_INKLING2;
skipped otherwise. Self-consistency checks between the decode step, the multi-row programs and the speculative
path (no reference data)."""
from __future__ import annotations

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

TEXT = ("FlashRT runs the released checkpoint in its own precisions: BF16 attention and NVFP4 or MXFP8 experts, "
        "with recorded kernel programs replayed as CUDA graphs. A bounded worker queue applies updates in "
        "timestamp order, retries transient failures and reports backpressure, idempotency and fairness metrics. ")


@pytest.fixture(scope="module")
def fe():
    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    f = Inkling2TorchFrontendRtx(CKPT, max_seq=4096, speculative=True)
    yield f
    f.close()


@pytest.fixture(scope="module")
def text_ids(fe):
    ids = fe.tokenizer.encode(TEXT * 40)
    assert len(ids) >= 1200
    return ids


def _same_state(a, b):
    """KV rows below the position (later rows of a global-attention cache may hold rejected verification rows,
    never read and rewritten later), every sliding-window slot, every convolution state, the position."""
    (kv_a, conv_a, pos_a), (kv_b, conv_b, pos_b) = a, b
    n = int(pos_b.item())
    for (ka, va), (kb, vb) in zip(kv_a, kv_b):
        for x, y in ((ka, kb), (va, vb)):
            if x.shape[0] > 1024:
                x, y = x[:n], y[:n]
            if not torch.equal(x, y):
                return False
    return all(torch.equal(x, y) for ca, cb in zip(conv_a, conv_b) for x, y in zip(ca, cb)) and \
        int(pos_a.item()) == n


def _kl(p_logits, q_logits):
    p, q = torch.log_softmax(p_logits.float(), -1), torch.log_softmax(q_logits.float(), -1)
    return float((p.exp() * (p - q)).sum())


def test_verification_rows_match_decode_steps(fe, text_ids):
    """Eight verification rows give the logits of eight decode steps bit for bit, accept every correct draft and
    commit the state of the decode steps; a wrong draft at row 4 accepts three and commits the state after four."""
    rt = fe.rt
    rt.reset()
    anchor = rt.prefill(text_ids[:300])
    s0 = rt.state()
    toks, logits, snaps = [anchor], [], {}
    for i in range(8):
        rt.step(toks[-1])
        logits.append(rt.logits[0].clone())
        toks.append(int(rt.token.item()))
        if i in (3, 7):
            snaps[i + 1] = rt.state()
    verify = rt.program("verify", 8)
    rt.restore(s0)
    rt.vtokens[:8].copy_(torch.tensor(toks[:8], dtype=torch.int32))
    rt.run(verify)
    assert all(torch.equal(rt.logits[r], logits[r]) for r in range(8))
    assert rt.targets.tolist() == toks[1:9]
    assert int(rt.accepted.item()) == 7
    assert _same_state(rt.state(), snaps[8])
    rt.restore(s0)
    bad = list(toks[:8])
    bad[4] = (bad[4] + 1) % rt.vocab
    rt.vtokens[:8].copy_(torch.tensor(bad, dtype=torch.int32))
    rt.run(verify)
    assert int(rt.accepted.item()) == 3
    assert rt.step_tokens.tolist()[:4] == toks[1:5]
    assert _same_state(rt.state(), snaps[4])


def test_exact_prefill_matches_token_steps(fe, text_ids):
    """Per-row prompt chunks reproduce token-by-token decoding bit for bit past the 1024-slot sliding window."""
    rt = fe.rt
    ids = text_ids[:1100]
    rt.reset()
    for t in ids:
        rt.step(t)
    first, logits, st = int(rt.token.item()), rt.logits[0].clone(), rt.state()
    rt.reset()
    assert rt.prefill(ids, exact=True) == first
    assert torch.equal(rt.logits[0], logits)
    assert _same_state(rt.state(), st)


@pytest.mark.parametrize("n", [40, 700, 1100])
def test_fast_prefill_tracks_exact_prefill(fe, text_ids, n):
    """Tensor-core chunks (several sizes, past the sliding window) against the bit-exact path: a coarse guard on
    the last row's logits (a misplaced chunk or a stale tile gives KL far above 1; the checkpoint's FP4 rounding
    keeps the two paths within a few 1e-2)."""
    rt = fe.rt
    rt.reset()
    exact = rt.prefill(text_ids[:n], exact=True)
    ref = rt.logits[0].clone()
    rt.reset()
    fast = rt.prefill(text_ids[:n])
    got = rt.logits[0]
    assert torch.isfinite(got).all()
    assert fast == exact
    assert _kl(ref, got) < 0.1


def test_grouped_mma_experts_track_flat_passes(fe, text_ids):
    """A 512-row chunk with the grouped block-scaled MMA experts against the same chunk on the flat expert passes
    (same quantities, different accumulation order)."""
    rt = fe.rt
    flat_prog = "prefill512_flat"
    rt.mma_experts = False
    try:
        rt.programs[flat_prog] = rt._record(512, verify=False, prefill="last")
    finally:
        rt.mma_experts = True
    out = []
    for name in (rt.program("chunk_last", 512), flat_prog):
        rt.reset()
        rt.vtokens[:512].copy_(torch.tensor(text_ids[:512], dtype=torch.int32))
        rt.run(name)
        out.append((int(rt.token.item()), rt.logits[0].clone()))
    rt.graphs.pop(flat_prog, None)
    rt.programs.pop(flat_prog)
    assert out[0][0] == out[1][0]
    assert _kl(out[1][1], out[0][1]) < 0.1


@pytest.mark.parametrize("prefill", ["fast", "exact"])
def test_speculative_output_equals_greedy(fe, text_ids, prefill):
    ids = text_ids[:500]
    plain = fe.generate(ids, 160, stop_ids=(), speculative=False, prefill=prefill)
    assert len(plain) == 160
    assert fe.generate(ids, 160, stop_ids=(), speculative=True, prefill=prefill) == plain


@pytest.mark.parametrize("width", [2, 8])
def test_speculative_widths(fe, text_ids, width):
    dr = fe.drafter
    ids = text_ids[:200]
    plain = fe.generate(ids, 96, stop_ids=(), speculative=False)
    dr.set_width(width)
    try:
        assert fe.generate(ids, 96, stop_ids=(), speculative=True) == plain
    finally:
        dr.set_width(4)


@pytest.mark.parametrize("speculative", [False, True])
def test_stop_tokens_and_lengths(fe, text_ids, speculative):
    ids = text_ids[:64]
    ref = fe.generate(ids, 40, stop_ids=(), speculative=False)
    assert fe.generate(ids, 1, stop_ids=(), speculative=speculative) == ref[:1]
    assert fe.generate(ids, 37, stop_ids=(), speculative=speculative) == ref[:37]
    stop = ref[20]
    cut = ref.index(stop) + 1
    assert fe.generate(ids, 40, stop_ids=(stop,), speculative=speculative) == ref[:cut]


def test_prefill_is_independent_of_torch_matmul_settings(fe, text_ids):
    """The products run on the module's own cuBLAS handle: a tensor-core prefill gives the same logits bit for bit
    whatever PyTorch's process-wide matmul settings are, and leaves them untouched (also while another thread
    reads them)."""
    import threading
    rt, mm = fe.rt, torch.backends.cuda.matmul
    flags = lambda: (mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction)
    saved, out = flags(), []
    seen, stop = set(), threading.Event()

    def watch():
        while not stop.is_set():
            seen.add(flags())

    try:
        for setting in (("ieee", False), ("tf32", True)):
            mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction = setting
            seen.clear()
            stop.clear()
            t = threading.Thread(target=watch)
            t.start()
            for graphs in (True, False):
                rt.use_graphs = graphs
                rt.reset()
                rt.prefill(text_ids[:700])
                out.append(rt.logits[0].clone())
            torch.cuda.synchronize()
            stop.set()
            t.join()
            assert seen == {setting} and flags() == setting
    finally:
        rt.use_graphs = True
        mm.fp32_precision, mm.allow_bf16_reduced_precision_reduction = saved
    assert all(torch.equal(x, out[0]) for x in out[1:])


def test_bad_weights_are_refused_before_loading(tmp_path):
    """A tensor of another dtype or shape is refused from the safetensors headers, before any upload."""
    import shutil
    from safetensors.torch import save_file
    from flash_rt.frontends.torch._inkling2_rtx_runtime import Inkling2ConfigError, Inkling2RtxRuntime
    shutil.copy(Path(CKPT) / "config.json", tmp_path / "config.json")
    save_file({"model.llm.layers.0.mlp.experts.w2_weight": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(Inkling2ConfigError, match="layers.0.mlp.experts.w2_weight: F32 \\[1\\]"):
        Inkling2RtxRuntime(str(tmp_path), max_seq=4096)


def test_one_drafter_per_runtime(fe):
    from flash_rt.frontends.torch._inkling2_rtx_mtp import Inkling2RtxDrafter
    with pytest.raises(RuntimeError, match="already has a drafter"):
        Inkling2RtxDrafter(fe.rt)


def test_request_validation(fe):
    with pytest.raises(ValueError):
        fe.generate([], 4)
    with pytest.raises(ValueError):
        fe.generate([fe.vocab], 4)
    with pytest.raises(ValueError):
        fe.generate([1, 2, 3], fe.rt.max_seq)


def test_chat(fe):
    reply = fe.chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                    max_new_tokens=96, reasoning_effort="none")
    assert "391" in reply
