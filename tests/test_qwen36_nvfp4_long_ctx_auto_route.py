"""Regression test for Qwen3.6-27B NVFP4 long-context auto-routing.

History
-------

Reported by a user on RTX PRO 4500 32 GB: ``Qwen36TorchFrontendRtx``
constructed with ``max_seq=32768`` would OOM during the FIRST
generation if the prompt was longer than ~1 K tokens. We reproduced
the same behaviour on RTX 5090 32 GB.

Two root causes were live at the same time:

1. ``_ensure_graph_for_pos_nvfp4`` cloned the *entire* BF16 KV cache
   for state-snap on every per-cur_pos graph capture. At
   ``max_seq=32768`` that is a 2 GB transient on top of an already
   tight ~30 GB baseline, so prefill OOMed once it touched a fresh
   cur_pos value. Every sister method (``_ensure_verify_graph_*``,
   ``_ensure_mtp_graph_*``, ``_ensure_chain_graph_*``) already did
   partial snap of just the rows they wrote — this method was the
   outlier.

2. The TurboQuant (TQ) packed-cache path was the only way to push
   past ~16 K context, but it had no public on-ramp: every
   ``_load_turboquant_*`` / ``_shrink_bf16_kv_cache`` /
   ``forward_own_decode_nvfp4_tq*`` method was underscore-prefixed,
   the constructor accepted ``max_seq`` up to 256 K without changing
   path, and the OpenAI server example never invoked the TQ helpers.
   Users who set ``max_seq=32768`` got a silently broken BF16 spec
   path instead.

Coverage
--------

Three scenarios. Each one independently failed before this fix lands;
all three must pass after.

  * ``test_short_ctx_spec_path_unchanged`` — ``max_seq=2048`` with the
    documented spec workload still works at ~100 tok/s warm. Pure
    regression net: ensures the auto-route logic does not silently
    rewire short-context.

  * ``test_long_prompt_doesnt_oom`` — ``max_seq=32768``,
    ``prompt_len=1024``, ``max_new=16``. This is the exact failure
    the bug report described; on main it OOMs, post-fix it must
    succeed and produce ``max_new`` tokens.

  * ``test_auto_route_init_at_each_tier`` — constructor at
    ``max_seq ∈ [2K, 8K, 16K, 32K, 64K, 128K]`` all succeed, all
    expose a ``_long_ctx_mode`` boolean that flips at the documented
    threshold, and all complete a tiny end-to-end generate without
    OOM. (256 K is omitted because it sometimes pushes the test
    runner's residual VRAM over the line; the path is exercised by
    the 128 K case which is structurally identical.)

Skipped if the NVFP4 + MTP checkpoints are not present.

Run:
    PYTHONPATH=. python -m pytest \
        tests/test_qwen36_nvfp4_long_ctx_auto_route.py -v -s
"""
from __future__ import annotations

import gc
import os

import pytest

CKPT_NVFP4 = os.environ.get('FLASHRT_QWEN36_NVFP4_CKPT_DIR', '')
CKPT_MTP = os.environ.get('FLASHRT_QWEN36_MTP_CKPT_DIR', '')


def _have_ckpts() -> bool:
    if not CKPT_NVFP4 or not os.path.isdir(CKPT_NVFP4):
        return False
    if not CKPT_MTP or not os.path.isfile(
            os.path.join(CKPT_MTP, 'mtp.safetensors')):
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _have_ckpts(),
    reason=(
        'Needs NVFP4 + MTP checkpoints + CUDA. Set '
        'FLASHRT_QWEN36_NVFP4_CKPT_DIR / FLASHRT_QWEN36_MTP_CKPT_DIR '
        'to override.'
    ),
)


def _used_vram_bytes() -> int:
    """Return driver-reported used VRAM (total - free)."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return total - free


def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def _free_frontend(fe):
    """Aggressively free a frontend so the next test starts clean."""
    import torch

    if hasattr(fe, 'clear_graphs'):
        fe.clear_graphs()
    del fe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_short_ctx_spec_path_unchanged():
    """max_seq=2048 stays on BF16 spec path, decode in documented range."""
    import time
    import torch

    os.environ.setdefault('FLASHRT_QWEN36_MTP_CKPT_DIR', CKPT_MTP)
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx(CKPT_NVFP4, quant='nvfp4', max_seq=2048)
    try:
        assert getattr(fe, '_long_ctx_mode', False) is False, (
            'short-ctx instance must NOT be in long-ctx mode'
        )
        prompt = 'Explain quantum entanglement in one short paragraph.'
        ids = fe._tokenizer(
            prompt, return_tensors='pt').input_ids.cuda()

        # Warmup.
        _ = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=128, K=6)
        torch.cuda.synchronize()

        # Timed.
        fe.reset_state()
        fe.reset_mtp_state()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=128, K=6)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new_tokens = out.shape[1] - ids.shape[1]
        tps = new_tokens / dt

        print(
            f'\n[short-ctx] prompt={ids.shape[1]} new={new_tokens} '
            f'time={dt:.3f}s tok/s={tps:.1f}'
        )

        # Documented range is 90-130 tok/s; allow ≥ 70 to absorb
        # measurement noise on contended GPUs.
        assert tps >= 70.0, (
            f'short-ctx warm decode degraded: {tps:.1f} tok/s '
            f'(expected ≥70, docs claim 90-130)'
        )
        assert new_tokens == 128
    finally:
        _free_frontend(fe)


def test_long_prompt_doesnt_oom():
    """max_seq=32768, prompt_len=1024 — the exact bug-report repro."""
    import torch

    os.environ.setdefault('FLASHRT_QWEN36_MTP_CKPT_DIR', CKPT_MTP)
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx(
        CKPT_NVFP4, quant='nvfp4', max_seq=32768)
    try:
        # Auto-route should put us in long-ctx mode.
        assert getattr(fe, '_long_ctx_mode', False) is True, (
            'max_seq=32768 must auto-route into long-ctx mode'
        )

        used_post_init = _used_vram_bytes()
        print(
            f'\n[long-prompt] post-init VRAM used='
            f'{_bytes_to_mb(used_post_init):.0f} MB'
        )

        # Build a 1024-token prompt by padding a real prompt.
        pad = fe._tokenizer.pad_token_id or 0
        base_ids = fe._tokenizer(
            'Tell me a long story about a robot.',
            return_tensors='pt').input_ids.cuda()
        target_len = 1024
        if base_ids.shape[1] >= target_len:
            ids = base_ids[:, :target_len]
        else:
            extra = torch.full(
                (1, target_len - base_ids.shape[1]), pad,
                device='cuda', dtype=torch.long,
            )
            ids = torch.cat([base_ids, extra], dim=1)
        assert ids.shape[1] == target_len

        # The pre-fix bug: this OOMs at 32 GB on the very first call.
        out = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=16, K=6)

        new_tokens = out.shape[1] - ids.shape[1]
        used_post_gen = _used_vram_bytes()
        print(
            f'[long-prompt] generated {new_tokens} tokens; '
            f'VRAM used={_bytes_to_mb(used_post_gen):.0f} MB '
            f'(Δ={_bytes_to_mb(used_post_gen - used_post_init):+.0f} MB)'
        )

        assert new_tokens == 16, (
            f'expected 16 new tokens, got {new_tokens}'
        )
    finally:
        _free_frontend(fe)


@pytest.mark.parametrize('max_seq', [2048, 8192, 16384, 32768, 65536, 131072])
def test_auto_route_init_at_each_tier(max_seq):
    """Constructor at every tier succeeds and produces tokens."""
    import torch

    os.environ.setdefault('FLASHRT_QWEN36_MTP_CKPT_DIR', CKPT_MTP)
    from flash_rt.frontends.torch.qwen36_rtx import (
        Qwen36TorchFrontendRtx,
    )

    fe = Qwen36TorchFrontendRtx(
        CKPT_NVFP4, quant='nvfp4', max_seq=max_seq)
    try:
        threshold = getattr(
            Qwen36TorchFrontendRtx, 'LONG_CTX_THRESHOLD', 16384)
        long_ctx_mode = getattr(fe, '_long_ctx_mode', False)
        used = _used_vram_bytes()
        print(
            f'\n[tier max_seq={max_seq:>7}] long_ctx_mode={long_ctx_mode} '
            f'(threshold={threshold}) VRAM={_bytes_to_mb(used):.0f} MB'
        )

        # Mode flips at the threshold.
        if max_seq > threshold:
            assert long_ctx_mode is True, (
                f'max_seq={max_seq} > threshold={threshold} should '
                f'auto-route to long-ctx mode'
            )
        else:
            assert long_ctx_mode is False, (
                f'max_seq={max_seq} <= threshold={threshold} should '
                f'stay on short-ctx (BF16 spec) path'
            )

        # End-to-end smoke: short prompt, few tokens. Just must not OOM
        # and must return a tensor of the expected shape.
        prompt = 'Hi there.'
        ids = fe._tokenizer(
            prompt, return_tensors='pt').input_ids.cuda()
        out = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=8, K=6)
        new_tokens = out.shape[1] - ids.shape[1]
        assert new_tokens == 8, (
            f'expected 8 new tokens, got {new_tokens}'
        )
    finally:
        _free_frontend(fe)
