# Execution-state capsules — usage

A **capsule** is the full, restorable execution state of a Qwen3.6 session at a
committed token boundary (linear-attention recurrent/conv state, full-attention
KV valid range, the hidden journal, MTP cache, and boundary metadata). Capsules
let a serving host **cold-prefill a shared prefix once and restore it** — instead
of re-prefilling it — on every later turn, session, or branch.

This is FlashRT's graph-replay-native answer to prefix caching; the design
rationale and how it differs from vLLM/SGLang block/radix KV caching is in
[`docs/serving_design.md`](../../docs/serving_design.md).

## API (Qwen3.6 frontend)

```python
fe = Qwen36TorchFrontendRtx(ckpt, quant="nvfp4", device="cuda", max_seq=4096)

# 1) Cold-prefill a shared prefix once and freeze it.
fe.prefill_own_speculative_nvfp4_agent(prefix_ids, max_new_tokens=64, K=3)
capsule = fe.snapshot_capsule()           # an opaque object; capsule["nbytes"] = footprint

# 2) Per turn: restore the prefix, append only the new suffix, decode.
fe.restore_capsule(capsule)
fe.append_own_speculative_nvfp4_agent(full_ids, start_pos=len(prefix_ids),
                                      max_new_tokens=64, K=3)
for chunk in fe.decode_own_speculative_nvfp4_committed_stream(max_new_tokens=64, K=3):
    ...                                   # committed tokens, ready to stream

# Fork: restore the same capsule into several independent continuations.
# Time-travel: restore an earlier boundary of the same session (undo a turn).
```

`snapshot_capsule()` clones the boundary state device-to-device and returns it as
a stable object. `restore_capsule()` copies it back into the live buffers and
rebuilds the boundary, so the next decode reuses the *same captured CUDA graphs*
— no recapture. The capsule decision logic (when to pin, evict, restore vs
rebuild) is serving-layer policy and stays out of the execution contract.

## Correctness contract

A capsule restore is **bit-identical to the path it replaces** — verified
token-exact in `tests/test_qwen36_agent_capsule.py`:

- **Pure restore == cold prefill.** `restore + decode` produces the same tokens
  as a cold `prefill + decode` of the same prefix (short and long routes, real
  text), including restore after the buffers were dirtied by another prompt, and
  fork (two branches from one capsule).
- **Restore + append == cold full prefill** (long route, chunk-aligned boundary).
  The coding-agent flow (`restore + append(suffix) + decode`) is token-identical
  to a cold `prefill(prefix + suffix) + decode` when the capsule is snapshotted
  at a chunk-aligned boundary (see below).

Decode throughput is unchanged — capsules touch prefill / TTFT only, never
steady-state decode.

### Long route: snapshot at a chunk-aligned boundary

The long chunked-GDN prefill folds its linear-attention recurrent state **per
chunk**, so the state at a position depends on where the chunk boundaries fall. A
cold full prefill of length F places boundaries at multiples of the prefill chunk
size. If a capsule/append boundary is *not* a multiple of that size, append adds
a chunk split the cold prefill never had, and the two diverge under FP8 rounding
(the divergence is small but compounds through greedy decode).

The fix is to snapshot at a chunk-aligned boundary:

```python
aligned = fe.capsule_aligned_len(prefix_len)   # floor to fe.long_prefill_chunk_size()
fe.prefill_long_ctx_nvfp4_agent(prefix_ids[:, :aligned], max_new_tokens=..., K=...)
cap = fe.snapshot_capsule()
...
fe.restore_capsule(cap)
fe.append_long_ctx_nvfp4_agent(full_ids, start_pos=aligned, ...)   # == cold full prefill
```

At an aligned boundary, `restore + append + decode` is token-identical to a cold
full prefill (verified in `test_long_capsule_chunk_aligned_matches_cold_full_prefill`).
The remainder of the prefix after the aligned boundary (< one chunk) is re-prefilled
by the append, which is cheap. The short route has no chunking and needs no
alignment.

## Status

- **Short committed-stream route: supported** (`snapshot_capsule` /
  `restore_capsule`, in-GPU device-to-device).
- **Long FP8-KV route: supported** — the production agent path
  (`--route-min-seq 0`, `FLASHRT_QWEN36_LONG_KV_CACHE=fp8`). The capsule covers
  the packed FP8 KV valid range, linear recurrent/conv state, MTP cache + long
  MTP hidden tail, and metadata; restore re-dequantizes the BF16 stage from the
  restored FP8 cache.
- **Long TQ KV mode: not wired** — `snapshot_capsule()` raises
  `NotImplementedError` rather than producing a partial capsule.

## Measured benefit (short route)

Real coding-agent workload on RTX 5090 (`pi0-stablehlo-test`): one shared prefix
(coding-assistant system prompt + project context, **185 shared tokens**) reused
across three tasks, served two ways — `cold` (re-prefill prefix+suffix every
turn) vs `capsule` (restore prefix + append suffix). `max_new=64`, `K=3`, median
of 7 repeats, stable to < 1% across 3 full runs:

| task | full / suffix tok | cold TTFT | capsule TTFT | TTFT speedup | decode tok/s (cold → capsule) | token-exact |
| --- | --- | --- | --- | --- | --- | --- |
| fill-doc | 258 / 73 | ~5.47 s | ~1.85 s | **2.96x** | 97.1 → 97.1 | yes |
| write-code | 223 / 38 | ~4.77 s | ~1.10 s | **4.33x** | 113.1 → 113.1 | yes |
| algorithm | 225 / 40 | ~4.82 s | ~1.13 s | **4.26x** | 112.8 → 112.8 | yes |

- **Mean TTFT speedup ~3.85x** (scales with the shared-prefix / suffix ratio).
- **Decode throughput unchanged** (capsule == cold to 0.1 tok/s) — by design.
- **Token-exact** cold vs capsule on every task and repeat.
- **Capsule footprint 89.85 MB** for the 185-token prefix.

Honest reading: the short route prefills sequentially, one position at a time, so
even a ~220-token cold prefill takes seconds — that absolute cost is a property of
the short route, and is exactly what the capsule removes for the shared prefix.

## Measured benefit (long FP8-KV route — production agent path)

Same workload on the chunked long FP8-KV route, shared prefix snapshotted at a
chunk-aligned 2k / 4k / 8k boundary. `cold` = prefill_long(prefix+suffix) every
turn; `capsule` = restore + append(suffix). Median of 5 repeats, stable to
< 0.5% across runs:

| shared prefix | cold TTFT | capsule TTFT | TTFT speedup | capsule MB | capsule==cold |
| --- | --- | --- | --- | --- | --- |
| 2048 tok | ~288 ms | ~138 ms | **2.08x** | 168 MB | yes |
| 4096 tok | ~388 ms | ~73 ms | **5.28x** | 211 MB | yes |
| 8192 tok | ~816 ms | ~142 ms | **5.72x** | 360 MB | yes |

- **Cold TTFT grows with prefix length** (288 → 388 → 816 ms); **capsule TTFT
  stays roughly flat** (restore is a ~0.1 ms device-to-device copy — bandwidth on
  the capsule bytes — so you pay essentially only for the suffix append). The
  speedup therefore **widens with prefix length**, and keeps widening past 8k
  toward the 10k–50k shared prefixes a real coding agent resends each turn.
- Capsule output is **token-identical to a cold full prefill** at every size.
- Decode throughput is unchanged by the capsule.

A single continuous hot session is unaffected — the shipped contiguous-append
path already reuses its own prefix. Capsules help fresh / multi-session / fork
reuse.
