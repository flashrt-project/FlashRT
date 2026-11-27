"""Inkling 2 Mini on RTX SM120 with the DFlash 2 drafter, BF16 and ``fp8=True``. Needs FLASHRT_INKLING2_DIR = the
released checkpoint directory (thinkingmachines/Inkling-2-Mini), FLASHRT_INKLING2_DFLASH2_DIR = the drafter
directory (incoai/Inkling-2-Mini-DFlash2), an SM120 GPU and the module built with FLASHRT_ENABLE_INKLING2; skipped
otherwise. Self-consistency checks between the decode step, the verification rows and the speculative path (no
reference data)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

CKPT = os.environ.get("FLASHRT_INKLING2_DIR", "")
DRAFTER = os.environ.get("FLASHRT_INKLING2_DFLASH2_DIR", "")
if not CKPT or not Path(CKPT).is_dir() or not DRAFTER or not Path(DRAFTER).is_dir():
    pytest.skip("set FLASHRT_INKLING2_DIR and FLASHRT_INKLING2_DFLASH2_DIR", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("needs an SM120 GPU", allow_module_level=True)
pytest.importorskip("flash_rt.flash_rt_inkling2")

TEXT = ("FlashRT runs the released checkpoint in its own precisions: BF16 attention and NVFP4 or MXFP8 experts, "
        "with recorded kernel programs replayed as CUDA graphs. A bounded worker queue applies updates in "
        "timestamp order, retries transient failures and reports backpressure, idempotency and fairness metrics. ")


@pytest.fixture(scope="module", params=[False, True], ids=["bf16", "fp8"])
def fe(request):
    from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx
    f = Inkling2TorchFrontendRtx(CKPT, max_seq=4096, speculative=True, dflash2=DRAFTER, fp8=request.param)
    yield f
    f.close()


@pytest.fixture(scope="module")
def text_ids(fe):
    ids = fe.tokenizer.encode(TEXT * 40)
    assert len(ids) >= 1200
    return ids


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
def test_verification_rows_match_decode_steps(fe, text_ids, width):
    """W verification rows give the logits of W decode steps bit for bit (FP8: of its own decode steps)."""
    rt = fe.rt
    rt.reset()
    anchor = rt.prefill(text_ids[:300])
    s0 = rt.state()
    toks, logits = [anchor], []
    for _ in range(width):
        rt.step(toks[-1])
        logits.append(rt.logits[0].clone())
        toks.append(int(rt.token.item()))
    rt.restore(s0)
    rt.vtokens[:width].copy_(torch.tensor(toks[:width], dtype=torch.int32))
    rt.run(rt.program("verify", width))
    assert all(torch.equal(rt.logits[r], logits[r]) for r in range(width))
    assert rt.targets[:width].tolist() == toks[1:width + 1]
    assert int(rt.accepted.item()) == width - 1


@pytest.mark.parametrize("prefill", ["fast", "exact"])
def test_adaptive_widths_output_equals_greedy(fe, text_ids, prefill):
    ids = text_ids[:500]
    plain = fe.generate(ids, 192, stop_ids=(), speculative=False, prefill=prefill)
    assert len(plain) == 192
    assert fe.generate(ids, 192, stop_ids=(), speculative=True, prefill=prefill) == plain
    dr = fe.drafter
    assert dr.rounds < len(plain) / 2           # the drafts are accepted (a repetitive text: long runs)
    assert set(dr.widths) <= set(dr.widths_all)


@pytest.mark.parametrize("width", [2, 8])
def test_fixed_widths_output_equals_greedy(fe, text_ids, width):
    dr = fe.drafter
    ids = text_ids[:200]
    plain = fe.generate(ids, 96, stop_ids=(), speculative=False)
    dr.adaptive = False
    dr.set_width(width)
    try:
        assert fe.generate(ids, 96, stop_ids=(), speculative=True) == plain
    finally:
        dr.adaptive = True


def test_stop_tokens_and_lengths(fe, text_ids):
    ids = text_ids[:64]
    ref = fe.generate(ids, 40, stop_ids=(), speculative=False)
    assert fe.generate(ids, 1, stop_ids=(), speculative=True) == ref[:1]
    assert fe.generate(ids, 37, stop_ids=(), speculative=True) == ref[:37]
    stop = ref[20]
    cut = ref.index(stop) + 1
    assert fe.generate(ids, 40, stop_ids=(stop,), speculative=True) == ref[:cut]


def test_kernel_context_tracks_the_reference(fe, text_ids):
    """The drafter's context K / V of a prompt (tensor-core chunks, a per-row chunk, decode steps) from the kernels
    against the PyTorch reference over the BF16 weights (a wrong row, position or layer gives errors near 1)."""
    rt, dr = fe.rt, fe.drafter
    ids = text_ids[:300]
    n = len(ids)
    rt.reset()
    dr.kv.zero_()
    dr.prefill(ids)
    got = dr.kv[:, :, :, :n].float().clone()
    names = [k for k in rt.programs if k.startswith("df_ctx")]
    saved = {k: rt.programs.pop(k) for k in names}
    graphs = {k: rt.graphs.pop(k) for k in names if k in rt.graphs}
    dr.use_kernels = False
    try:
        rt.reset()
        dr.kv.zero_()
        rt.prefill(ids, after=lambda kind, rows: rt._run(rt.programs[dr.context_program(rows)]))
        ref = dr.kv[:, :, :, :n].float().clone()
    finally:
        dr.use_kernels = True
        for k in [k for k in rt.programs if k.startswith("df_ctx")]:
            rt.programs.pop(k)
            rt.graphs.pop(k, None)
        rt.programs.update(saved)
        rt.graphs.update(graphs)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    assert rel(got, ref) < 0.02
    # the prompt's last 4 rows (decode steps: the context products from the E4M3 copies)
    assert rel(got[..., -4:, :], ref[..., -4:, :]) < 0.1


def test_one_drafter_per_runtime(fe):
    from flash_rt.frontends.torch._inkling2_rtx_dflash2 import Inkling2RtxDFlash2
    with pytest.raises(RuntimeError, match="already has a drafter"):
        Inkling2RtxDFlash2(fe.rt, DRAFTER)
