"""Optional GPU smoke tests for Qwen3.6 agent split serving.

Run manually with:

    FLASHRT_QWEN36_NVFP4_CKPT_DIR=... \
    FLASHRT_QWEN36_MTP_CKPT_DIR=... \
    pytest -q tests/test_qwen36_agent_gpu_split.py -s

The default CI path skips these tests because they load the 27B checkpoint.
"""

from __future__ import annotations

import os

import pytest


CKPT = os.environ.get("FLASHRT_QWEN36_NVFP4_CKPT_DIR", "")
MTP = os.environ.get("FLASHRT_QWEN36_MTP_CKPT_DIR", "")


pytestmark = pytest.mark.skipif(
    not CKPT or not MTP,
    reason=(
        "set FLASHRT_QWEN36_NVFP4_CKPT_DIR and "
        "FLASHRT_QWEN36_MTP_CKPT_DIR to run Qwen3.6 GPU split smoke"
    ),
)


def _load_frontend(max_seq: int):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    return Qwen36TorchFrontendRtx(
        CKPT, quant="nvfp4", device="cuda", max_seq=max_seq)


def _token_ids(fe, prompt_len: int):
    import torch

    token_ids = fe._tokenizer(
        " validation", add_special_tokens=False).input_ids
    token = int(token_ids[0] if token_ids else 1)
    return torch.full(
        (1, prompt_len), token, device="cuda", dtype=torch.long)


def _assert_split_matches_full(fe, prompt_len: int, *,
                               max_new: int = 8, K: int = 3):
    import torch

    ids = _token_ids(fe, prompt_len)
    torch.cuda.synchronize()
    full = fe.generate_own_speculative_KN_nvfp4(
        ids, max_new_tokens=max_new, K=K)
    torch.cuda.synchronize()
    expected = full[0, prompt_len:prompt_len + max_new].tolist()

    if fe._should_use_long_ctx_route(prompt_len, max_new):
        fe.prefill_long_ctx_nvfp4_agent(
            ids, max_new_tokens=max_new, K=K)
        chunks = list(fe.decode_long_ctx_nvfp4_committed_stream(
            max_new_tokens=max_new, K=K))
    else:
        chunks = list(fe.generate_own_speculative_KN_nvfp4_committed_stream(
            ids, max_new_tokens=max_new, K=K))
    torch.cuda.synchronize()
    actual = [tok for chunk in chunks for tok in chunk]
    assert actual == expected
    if fe._should_use_long_ctx_route(prompt_len, max_new):
        start = int(getattr(fe, "_agent_long_h_tail_start", -1))
        rows = int(getattr(fe, "_agent_long_h_tail_rows", 0))
        assert start >= 0
        assert start + rows == prompt_len + max_new
        assert rows >= max_new


def test_qwen36_short_agent_split_matches_full_generate():
    fe = _load_frontend(max_seq=4096)
    _assert_split_matches_full(fe, 64)


def test_qwen36_long_agent_split_matches_full_generate(monkeypatch):
    monkeypatch.setenv("FLASHRT_QWEN36_LONG_KV_CACHE", "fp8")
    fe = _load_frontend(max_seq=32768)
    assert fe._long_ctx_mode
    _assert_split_matches_full(fe, 128)
    _assert_split_matches_full(fe, 512)


def test_qwen36_long_agent_append_matches_full_generate(monkeypatch):
    import torch

    monkeypatch.setenv("FLASHRT_QWEN36_LONG_KV_CACHE", "fp8")
    fe = _load_frontend(max_seq=32768)
    prompt_len = 128
    first_new = 4
    next_new = 8
    K = 3
    base = _token_ids(fe, prompt_len)

    fe.prefill_long_ctx_nvfp4_agent(base, max_new_tokens=first_new, K=K)
    first_chunks = list(fe.decode_long_ctx_nvfp4_committed_stream(
        max_new_tokens=first_new, K=K))
    first_tokens = [tok for chunk in first_chunks for tok in chunk]
    start_pos = prompt_len + len(first_tokens)

    suffix = _token_ids(fe, 4)
    prompt2 = torch.cat([
        base,
        torch.tensor([first_tokens], device="cuda", dtype=torch.long),
        suffix,
    ], dim=1)
    assert int(prompt2.shape[1]) == start_pos + 4

    full = fe.generate_own_speculative_KN_nvfp4(
        prompt2, max_new_tokens=next_new, K=K)
    expected = full[0, prompt2.shape[1]:prompt2.shape[1] + next_new].tolist()

    fe.prefill_long_ctx_nvfp4_agent(base, max_new_tokens=first_new, K=K)
    _ = list(fe.decode_long_ctx_nvfp4_committed_stream(
        max_new_tokens=first_new, K=K))
    fe.append_long_ctx_nvfp4_agent(
        prompt2, start_pos=start_pos, max_new_tokens=next_new, K=K)
    chunks = list(fe.decode_long_ctx_nvfp4_committed_stream(
        max_new_tokens=next_new, K=K))
    actual = [tok for chunk in chunks for tok in chunk]
    assert actual == expected
