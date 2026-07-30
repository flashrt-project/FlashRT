# Qwen3.6-MoE edge experiments

This directory contains checkpoint-independent development utilities for a
memory-constrained Qwen3.6-35B-A3B runtime. It is not a production frontend.

The intended runtime layout follows the MiniMax-M3 Spark prototype:

- non-routed weights remain resident in a mixed-precision format;
- each routed expert is stored as one fixed-size block;
- a bounded per-layer LRU holds hot expert blocks;
- misses are read from local storage into reusable staging buffers.

Inspect projected checkpoint sizes and sampled expert quality:

```bash
PYTHONPATH=. python qwen36_moe_edge/probe.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --mode memory \
  --group-size 16

PYTHONPATH=. python qwen36_moe_edge/probe.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --mode quality \
  --group-size 16
```

Generate fixed-size routed-expert blocks for a layer range:

```bash
PYTHONPATH=. python qwen36_moe_edge/quantize_experts.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --output /models/Qwen3.6-35B-A3B-INT8E \
  --format int8 \
  --layers 0:40

PYTHONPATH=. python qwen36_moe_edge/quantize_experts.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --output /models/Qwen3.6-35B-A3B-INT4E-RHT16 \
  --format int4-rht \
  --group-size 16 \
  --layers 0:40
```

INT8 uses symmetric per-output-channel FP16 scales. INT4 follows the Thor
Pi0.5 numerical contract: sign-magnitude values, one UE4M3 scale per 16 K
values, and two values per byte with the low nibble first. `int4-rht` applies
the same orthonormal H16/4 transform to every K block that the runtime applies
to activations. Scale bytes in these edge block files are linear; a Thor
loader must convert them to the SM1xx SFB tile-interleaved layout before
calling the native block-scaled MMA kernels.

An SM120 machine can collect real router selections for cache sizing:

```bash
PYTHONPATH=. python qwen36_moe_edge/route_trace.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --prompt "Explain edge mixture-of-experts inference. " \
  --prompt-tokens 32 \
  --new-tokens 64 \
  --quotas 16,27,32,43,64 \
  --output qwen36_moe_route_trace.json
```

Tracing deliberately uses eager per-token prefill. It must not be enabled
during CUDA Graph capture.

Each quota is scored under three policies:

- `single_lru` — one per-layer LRU behind both prefill and decode. Prefill
  touches every expert in a layer, so this measures what survives prompt
  churn.
- `two_tier` — a per-layer warm set pinned from prompt-phase selection counts
  plus an evictable ring sized by `--stream-fraction`. Prefill cannot displace
  the warm set.
- `two_tier_oracle_warm` — the same split with the warm set chosen from the
  decode phase. Not implementable; it bounds what a better warm-set heuristic
  could add.

`--block-bytes` (default: the INT4 group-16 block) and `--bandwidths` turn
misses per token into a read volume and the token rate each storage bandwidth
would allow, which is the number that decides whether a memory budget is
viable.
