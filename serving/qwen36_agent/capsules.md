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
- **Restore + append == non-capsule append.** The coding-agent flow
  (`restore + append(suffix) + decode`) equals the existing
  `prefill(prefix) + append(suffix) + decode` path token-exact: the capsule adds
  **zero** error to the path it stands in for.

Decode throughput is unchanged — capsules touch prefill / TTFT only, never
steady-state decode.

> Note (long route, pre-existing and orthogonal to capsules): the long-context
> `append` path itself does **not** reproduce a cold *full* prefill token-for-token
> at scale, because FP8-KV rounding at the append boundary perturbs the committed
> state. This is a property of `append_long_ctx_*`, not of capsules — a capsule
> reproduces whichever path it replaces exactly. Pure restore (no append) is
> bit-identical to a cold prefill.

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

Same workload on the chunked long FP8-KV route, with the shared prefix padded to
2k / 4k / 8k tokens (system + tool schema + repo index). `cold` =
prefill_long(prefix+suffix) every turn; `capsule` = restore + append(suffix).
Median of 5 repeats, stable to < 0.5% across runs:

| shared prefix | cold TTFT | capsule TTFT | TTFT speedup | capsule MB |
| --- | --- | --- | --- | --- |
| 2064 tok | ~289 ms | ~139 ms | **2.09x** | 168 MB |
| 4112 tok | ~389 ms | ~73 ms | **5.31x** | 211 MB |
| 8236 tok | ~817 ms | ~140 ms | **5.82x** | 361 MB |

- **Cold TTFT grows with prefix length** (289 → 389 → 817 ms); **capsule TTFT
  stays roughly flat** (restore is a ~0.1 ms device-to-device copy — bandwidth on
  the capsule bytes — so you pay essentially only for the suffix append). The
  speedup therefore **widens with prefix length**, and keeps widening past 8k
  toward the 10k–50k shared prefixes a real coding agent resends each turn.
- Capsule output is token-exact to the non-capsule append path at every size.
- Decode throughput is unchanged by the capsule.

A single continuous hot session is unaffected — the shipped contiguous-append
path already reuses its own prefix. Capsules help fresh / multi-session / fork
reuse.
