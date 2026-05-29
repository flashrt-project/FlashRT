"""Optional GPU tests for Qwen3.6 execution-state capsules (short route).

A capsule freezes a committed-stream boundary so a shared prefix can be
cold-prefilled once and then restored, instead of re-prefilled, on every later
turn/session. These tests are the correctness gate for that mechanism: restore
must be bit-identical (token-exact) to a cold prefill of the same prefix. See
docs/serving_design.md.

Run manually with:

    FLASHRT_QWEN36_NVFP4_CKPT_DIR=... \
    FLASHRT_QWEN36_MTP_CKPT_DIR=... \
    pytest -q tests/test_qwen36_agent_capsule.py -s
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
        "FLASHRT_QWEN36_MTP_CKPT_DIR to run Qwen3.6 capsule tests"
    ),
)


def _load_frontend(max_seq: int = 4096):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    return Qwen36TorchFrontendRtx(
        CKPT, quant="nvfp4", device="cuda", max_seq=max_seq)


def _token_ids(fe, prompt_len: int, *, word: str = " validation"):
    import torch

    token_ids = fe._tokenizer(word, add_special_tokens=False).input_ids
    token = int(token_ids[0] if token_ids else 1)
    return torch.full((1, prompt_len), token, device="cuda", dtype=torch.long)


def _decode_tokens(fe, *, max_new: int, K: int):
    chunks = list(fe.decode_own_speculative_nvfp4_committed_stream(
        max_new_tokens=max_new, K=K))
    return [tok for chunk in chunks for tok in chunk]


def test_capsule_restore_then_decode_is_bit_exact():
    """restore(capsule) + decode == prefill-boundary + decode, token-exact."""
    fe = _load_frontend()
    ids = _token_ids(fe, 64)
    max_new, K = 16, 3

    fe.prefill_own_speculative_nvfp4_agent(ids, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    assert cap["nbytes"] > 0
    cold = _decode_tokens(fe, max_new=max_new, K=K)

    fe.restore_capsule(cap)
    restored = _decode_tokens(fe, max_new=max_new, K=K)
    assert restored == cold


def test_capsule_restore_survives_dirty_state():
    """A different prompt between snapshot and restore must not leak in."""
    fe = _load_frontend()
    ids = _token_ids(fe, 48)
    max_new, K = 12, 3

    fe.prefill_own_speculative_nvfp4_agent(ids, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    cold = _decode_tokens(fe, max_new=max_new, K=K)

    # Fully overwrite every live buffer with an unrelated prompt + decode.
    other = _token_ids(fe, 80, word=" interruption")
    fe.prefill_own_speculative_nvfp4_agent(other, max_new_tokens=max_new, K=K)
    _ = _decode_tokens(fe, max_new=max_new, K=K)

    fe.restore_capsule(cap)
    restored = _decode_tokens(fe, max_new=max_new, K=K)
    assert restored == cold


def test_capsule_restore_then_append_matches_cold_prefill():
    """The coding-agent flow: restore a pinned prefix, append the new suffix.

    restore(prefix capsule) + append(prefix+suffix) + decode must equal a cold
    prefill(prefix+suffix) + decode, token-exact.
    """
    import torch

    fe = _load_frontend()
    prefix_len, suffix_len = 48, 16
    max_new, K = 16, 3
    prefix = _token_ids(fe, prefix_len)
    suffix = _token_ids(fe, suffix_len, word=" suffix")
    full = torch.cat([prefix, suffix], dim=1)

    # Cold reference: prefill the whole thing, decode.
    fe.prefill_own_speculative_nvfp4_agent(full, max_new_tokens=max_new, K=K)
    cold = _decode_tokens(fe, max_new=max_new, K=K)

    # Warm: prefill only the prefix, snapshot, restore, append the suffix.
    fe.prefill_own_speculative_nvfp4_agent(
        prefix, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    assert cap["cur_pos"] == prefix_len
    fe.restore_capsule(cap)
    fe.append_own_speculative_nvfp4_agent(
        full, start_pos=prefix_len, max_new_tokens=max_new, K=K)
    warm = _decode_tokens(fe, max_new=max_new, K=K)
    assert warm == cold


def test_capsule_fork_two_branches_match_independent_runs():
    """Fork: one prefix capsule restored into two independent continuations."""
    fe = _load_frontend()
    ids = _token_ids(fe, 56)
    max_new, K = 16, 3

    fe.prefill_own_speculative_nvfp4_agent(ids, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    branch_a = _decode_tokens(fe, max_new=max_new, K=K)

    fe.restore_capsule(cap)
    branch_b = _decode_tokens(fe, max_new=max_new, K=K)
    # Greedy decode from the same boundary is deterministic: both branches match.
    assert branch_a == branch_b


def _decode_long(fe, *, max_new, K):
    chunks = list(fe.decode_long_ctx_nvfp4_committed_stream(
        max_new_tokens=max_new, K=K))
    return [tok for chunk in chunks for tok in chunk]


def _load_long_frontend(monkeypatch, max_seq=32768):
    monkeypatch.setenv("FLASHRT_QWEN36_LONG_KV_CACHE", "fp8")
    fe = _load_frontend(max_seq=max_seq)
    if not getattr(fe, "_long_ctx_mode", False):
        pytest.skip("long-context mode not enabled in this build")
    if getattr(fe, "_long_kv_cache_mode", "tq") != "fp8":
        pytest.skip("long route is not in fp8 KV mode")
    return fe


def test_long_capsule_restore_then_decode_is_bit_exact(monkeypatch):
    """Long FP8-KV route: restore + decode == prefill-boundary + decode."""
    fe = _load_long_frontend(monkeypatch)
    prompt_len, max_new, K = 600, 16, 3
    if not fe._should_use_long_ctx_route(prompt_len, max_new):
        pytest.skip("prompt did not select the long route")
    ids = _token_ids(fe, prompt_len)

    fe.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    assert cap["is_long"] and cap["kv_mode"] == "fp8" and cap["nbytes"] > 0
    cold = _decode_long(fe, max_new=max_new, K=K)

    fe.restore_capsule(cap)
    restored = _decode_long(fe, max_new=max_new, K=K)
    assert restored == cold


def test_long_capsule_restore_survives_dirty_state(monkeypatch):
    """A different long prompt between snapshot and restore must not leak in."""
    fe = _load_long_frontend(monkeypatch)
    prompt_len, max_new, K = 600, 12, 3
    if not fe._should_use_long_ctx_route(prompt_len, max_new):
        pytest.skip("prompt did not select the long route")
    ids = _token_ids(fe, prompt_len)

    fe.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    cold = _decode_long(fe, max_new=max_new, K=K)

    other = _token_ids(fe, 700, word=" interruption")
    fe.prefill_long_ctx_nvfp4_agent(other, max_new_tokens=max_new, K=K)
    _ = _decode_long(fe, max_new=max_new, K=K)

    fe.restore_capsule(cap)
    restored = _decode_long(fe, max_new=max_new, K=K)
    assert restored == cold


def test_long_capsule_restore_then_append_matches_noncapsule_append(monkeypatch):
    """Long-route coding-agent flow: restore a pinned prefix + append the suffix
    must equal the non-capsule append path (prefill prefix + append) token-exact.

    The capsule's contract is that it reproduces the path it replaces with zero
    added error. (The long-append path itself differs from a cold full prefill at
    scale because of FP8-KV rounding at the append boundary -- a pre-existing
    property of append_long, orthogonal to capsules; see capsules.md.)
    """
    import torch

    fe = _load_long_frontend(monkeypatch)
    prefix_len, suffix_len, max_new, K = 600, 64, 16, 3
    if not fe._should_use_long_ctx_route(prefix_len + suffix_len, max_new):
        pytest.skip("prompt did not select the long route")
    prefix = _token_ids(fe, prefix_len)
    suffix = _token_ids(fe, suffix_len, word=" suffix")
    full = torch.cat([prefix, suffix], dim=1)

    # Non-capsule append: prefill the prefix, append the suffix, decode.
    fe.prefill_long_ctx_nvfp4_agent(prefix, max_new_tokens=max_new, K=K)
    fe.append_long_ctx_nvfp4_agent(
        full, start_pos=prefix_len, max_new_tokens=max_new, K=K)
    nocap = _decode_long(fe, max_new=max_new, K=K)

    # Capsule append: snapshot the prefix boundary, restore, append, decode.
    fe.prefill_long_ctx_nvfp4_agent(prefix, max_new_tokens=max_new, K=K)
    cap = fe.snapshot_capsule()
    assert cap["cur_pos"] == prefix_len
    fe.restore_capsule(cap)
    fe.append_long_ctx_nvfp4_agent(
        full, start_pos=prefix_len, max_new_tokens=max_new, K=K)
    warm = _decode_long(fe, max_new=max_new, K=K)
    assert warm == nocap


def test_long_capsule_tq_mode_raises(monkeypatch):
    """The long TQ KV mode capsule is not wired yet; fail loudly, not silently."""
    monkeypatch.setenv("FLASHRT_QWEN36_LONG_KV_CACHE", "tq")
    fe = _load_frontend(max_seq=32768)
    if not getattr(fe, "_long_ctx_mode", False):
        pytest.skip("long-context mode not enabled in this build")
    if getattr(fe, "_long_kv_cache_mode", "tq") != "tq":
        pytest.skip("long route is not in tq KV mode")
    ids = _token_ids(fe, 600)
    if not fe._should_use_long_ctx_route(600, 8):
        pytest.skip("600-token prompt did not select the long route")
    fe.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=8, K=3)
    with pytest.raises(NotImplementedError):
        fe.snapshot_capsule()
