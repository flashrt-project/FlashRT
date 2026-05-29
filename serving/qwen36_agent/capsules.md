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

Restore is **bit-identical** to a cold prefill of the same prefix. The gate lives
in `tests/test_qwen36_agent_capsule.py` and asserts token-exact output for:
restore + decode, restore after the buffers were dirtied by another prompt,
restore + append + decode (the coding-agent flow), and fork (two branches from
one capsule). Decode throughput is unchanged — capsules touch prefill / TTFT
only, never steady-state decode.

## Status

- **Short committed-stream route: supported** (`snapshot_capsule` /
  `restore_capsule`, in-GPU device-to-device).
- **Long FP8-KV/TQ route: in progress** — the production agent path (`--route-min-seq 0`)
  uses the chunked long-context route; its capsule covers a larger state surface
  (TQ/FP8 dequant stage + long MTP tail) and is being wired next. On the long
  route `snapshot_capsule()` raises `NotImplementedError` rather than producing a
  partial capsule.

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
The long route (production agent path) prefills faster per token; its capsule
advantage carries over and **grows with prefix length**, where the seconds →
milliseconds win lands for 10k–50k-token shared prefixes.

A single continuous hot session is unaffected — the shipped contiguous-append
path already reuses its own prefix. Capsules help fresh / multi-session / fork
reuse.
