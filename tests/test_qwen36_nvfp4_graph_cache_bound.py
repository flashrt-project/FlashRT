"""Regression test for issue: NVFP4 OpenAI server VRAM grows unbounded.

Symptom (reported on a 32 GB RTX 5090): after a handful of chat
requests with varied prompt lengths, ``torch.cuda.CUDAGraph().capture_end()``
raised ``cudaErrorMemoryAllocation``. Root cause: the per-``cur_pos``
graph cache (``_captured_graphs`` / ``_captured_verify_graphs`` /
``_captured_chain_graphs`` / ``_captured_mtp_graphs``) grew without
bound, and each captured graph held its own private CUDA mempool.

This test exercises the spec-decode path with diverse short prompts —
the same shape mix that triggered the production OOM — and asserts:

  1. ``_used_vram_bytes()`` plateaus instead of growing
     linearly with the number of distinct (prompt_len, max_new_tokens)
     shapes seen.
  2. The combined size of all per-position graph caches stays bounded
     (the LRU cap is honored).

The test deliberately runs in a single process so the steady-state
memory after warmup is measured, not first-call capture cost.

Skipped if the NVFP4 + MTP checkpoints are not present.

Run:
    PYTHONPATH=. python -m pytest \
        tests/test_qwen36_nvfp4_graph_cache_bound.py -v -s
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


# Mix of short prompts at different lengths — mirrors the real-traffic
# pattern from the bug report (prompt_len ∈ {11, 12, 23, 30, 35} ...).
# Distinct prompt_len count drives distinct cur_pos values seen.
PROMPT_TEMPLATES = [
    'Hi.',
    'Hello there!',
    'What is 2+2?',
    'Explain quantum entanglement briefly.',
    'Write a haiku about autumn leaves falling.',
    'List three primes greater than 100 and explain why.',
    'Summarize the plot of Hamlet in two short sentences please.',
    'Translate to French: The quick brown fox jumps over the lazy dog.',
]


def _bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)


def _used_vram_bytes() -> int:
    """Return driver-reported used VRAM (total - free).

    ``_used_vram_bytes()`` only covers PyTorch's caching
    allocator; captured CUDA Graphs hold their own (per-graph private
    or shared) mempool that the caching allocator does not see. The
    leak is in *that* pool, so the test must read the driver counter.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return total - free


@pytest.fixture(scope='module')
def fe():
    """Load the NVFP4 frontend once for the whole module."""
    import torch

    os.environ.setdefault('FLASHRT_QWEN36_MTP_CKPT_DIR', CKPT_MTP)
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

    fe = Qwen36TorchFrontendRtx(CKPT_NVFP4, quant='nvfp4', max_seq=2048)
    yield fe
    del fe
    gc.collect()
    torch.cuda.empty_cache()


def _total_captured_graph_count(fe) -> int:
    n = 0
    for attr in (
        '_captured_graphs',
        '_captured_verify_graphs',
        '_captured_mtp_graphs',
        '_captured_chain_graphs',
    ):
        n += len(getattr(fe, attr, {}) or {})
    return n


def test_graph_cache_bounded_under_varied_prompts(fe):
    """Assert VRAM and graph count plateau under repeated traffic.

    Drives the spec path with the same set of varied short prompts
    repeatedly; after an initial growth phase, both VRAM and the
    captured-graph count must stop growing — i.e., the LRU works and
    the shared mempool keeps per-graph footprint small.
    """
    import torch

    fe.reset_state()
    fe.reset_mtp_state()

    # Tokenize a fixed pool of prompts (ascending prompt_len).
    pools = []
    for text in PROMPT_TEMPLATES:
        ids = fe._tokenizer(text, return_tensors='pt').input_ids.cuda()
        pools.append((int(ids.shape[1]), ids))
    pools.sort(key=lambda x: x[0])

    # Warmup pass — first time through each prompt pays capture cost.
    for plen, ids in pools:
        fe.reset_state()
        fe.reset_mtp_state()
        _ = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=64, K=6,
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base_alloc = _used_vram_bytes()
    base_graphs = _total_captured_graph_count(fe)
    print(
        f'\n[soak] post-warmup baseline: '
        f'mem={_bytes_to_mb(base_alloc):.1f} MB, '
        f'graphs={base_graphs}'
    )

    # Steady-state pass — same prompt shapes repeated. Should not grow.
    samples_alloc = []
    samples_graphs = []
    for cycle in range(6):
        for plen, ids in pools:
            fe.reset_state()
            fe.reset_mtp_state()
            _ = fe.generate_own_speculative_KN_nvfp4(
                ids, max_new_tokens=64, K=6,
            )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        samples_alloc.append(_used_vram_bytes())
        samples_graphs.append(_total_captured_graph_count(fe))
        print(
            f'[soak] cycle {cycle}: '
            f'mem={_bytes_to_mb(samples_alloc[-1]):.1f} MB '
            f'(Δ={_bytes_to_mb(samples_alloc[-1] - base_alloc):+.1f} MB), '
            f'graphs={samples_graphs[-1]}'
        )

    # Steady-state assertion: total drift across all 6 cycles must be
    # under 200 MB. Pre-fix this drifts unbounded (the OOM source).
    final = samples_alloc[-1]
    drift = final - base_alloc
    assert drift < 200 * 1024 * 1024, (
        f'memory drifted {_bytes_to_mb(drift):.1f} MB across 6 '
        f'steady-state cycles — graph cache is leaking'
    )

    # Cap assertion: graph count must stop growing across cycles.
    # (Same prompt shapes → no new cur_pos → cache count constant.)
    assert samples_graphs[-1] == samples_graphs[0], (
        f'captured-graph count grew across steady-state cycles: '
        f'{samples_graphs[0]} → {samples_graphs[-1]}'
    )


def test_graph_cache_lru_cap_under_unique_shapes(fe):
    """Assert the LRU cap holds when many distinct shapes are seen.

    Drives the spec path with many distinct prompt lengths so the set
    of cur_pos values seen exceeds the LRU cap. The cache must cap,
    not grow without bound; VRAM must plateau after the cap is full.
    """
    import torch

    fe.reset_state()
    fe.reset_mtp_state()

    # Force a small cap on this instance so the test can exercise
    # eviction in a reasonable wall time. The production default
    # (256) would need 500+ generations to cross, taking 25+ min.
    fe.GRAPH_CACHE_MAX = 32
    fe.clear_graphs()

    # Build a large set of unique prompt_len values by padding a base
    # prompt with extra tokens. The tokenizer's pad token is fine —
    # only the length matters for graph capture.
    pad = fe._tokenizer.pad_token_id or 0
    base_ids = fe._tokenizer(
        'Tell me a short story.',
        return_tensors='pt',
    ).input_ids.cuda()
    base_len = int(base_ids.shape[1])

    cap = fe.GRAPH_CACHE_MAX

    # Generate 2× the cap worth of distinct prompt lengths to force
    # eviction. Cap each generation small so total wall time stays
    # tractable in CI.
    n_distinct = cap * 2
    print(
        f'\n[lru] driving {n_distinct} distinct prompt_len values '
        f'(cap={cap})'
    )

    base_alloc = None
    for i in range(n_distinct):
        target_len = base_len + (i % 200)  # 0..199 extra pad tokens
        if target_len > base_len:
            extra = torch.full(
                (1, target_len - base_len), pad,
                device='cuda', dtype=torch.long,
            )
            ids = torch.cat([base_ids, extra], dim=1)
        else:
            ids = base_ids
        fe.reset_state()
        fe.reset_mtp_state()
        _ = fe.generate_own_speculative_KN_nvfp4(
            ids, max_new_tokens=16, K=6,
        )
        if i == cap:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base_alloc = _used_vram_bytes()
            print(
                f'[lru] post-cap-fill: '
                f'mem={_bytes_to_mb(base_alloc):.1f} MB, '
                f'graphs={_total_captured_graph_count(fe)}'
            )

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    final_alloc = _used_vram_bytes()
    final_graphs = _total_captured_graph_count(fe)
    print(
        f'[lru] final: mem={_bytes_to_mb(final_alloc):.1f} MB, '
        f'graphs={final_graphs}'
    )

    # Cache must not exceed cap on a per-cache basis. There are 4
    # NVFP4 caches that can hold entries (decode / verify / mtp /
    # chain), so total ≤ cap × 4 + small slack for startup buckets.
    assert final_graphs <= cap * 4 + 16, (
        f'captured-graph count {final_graphs} exceeds cap*4+16 '
        f'({cap * 4 + 16})'
    )

    # Memory drift after the cap was full should be near zero — LRU
    # eviction reclaims one graph per insertion.
    if base_alloc is not None:
        drift = final_alloc - base_alloc
        assert drift < 500 * 1024 * 1024, (
            f'memory drifted {_bytes_to_mb(drift):.1f} MB after '
            f'the cap was reached — eviction is not freeing memory'
        )
