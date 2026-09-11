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

Check that the gated kernels compute the same thing on another architecture.
Compiling for a target says nothing about what it computes there, so record a
reference where the kernels are known good and replay it elsewhere:

```bash
# On a known-good target:
PYTHONPATH=. python qwen36_moe_edge/kernel_parity.py \
  --output parity_sm120.json

# On the target under test, with the reference alongside it:
PYTHONPATH=. python qwen36_moe_edge/kernel_parity.py \
  --output parity_sm110.json \
  --reference parity_sm120.json
```

No checkpoint is involved: shapes come from the Qwen3.6 geometry and inputs
from a fixed generator. Inputs are stored in the reference and replayed rather
than regenerated, because CUDA RNG is not bit-reproducible across
architectures — the Philox thread mapping follows occupancy, so regenerating on
the target compares kernels on different data and reads as a kernel failure.
Divergence appears only past the first launch block, which is why small tensors
appear to agree and large ones do not.

Each case also checks its kernel against a Torch expression on the local
device, so a genuine kernel fault is distinguishable from a harness or input
problem: a broken kernel fails its local check first.

Score the quantization schemes against the activations the router actually
sends each expert, and optionally save the references a device can check
itself against:

```bash
PYTHONPATH=. python qwen36_moe_edge/expert_quality.py \
  --checkpoint /models/Qwen3.6-35B-A3B \
  --prompt "Explain edge mixture-of-experts inference. " \
  --prompt-tokens 32 \
  --new-tokens 32 \
  --output qwen36_expert_quality.json \
  --golden qwen36_expert_golden.safetensors
```

Prefer this over `probe.py --mode quality` when deciding what to generate.
The probe uses `torch.randn` activations and quantizes the activations too;
neither matches the runtime, where the activation is 4 KiB against a 1.7 MiB
weight block and so is left in BF16. Random activations also hide errors that
real inputs expose, because a scale calibrated against noise is not the scale
real inputs need.

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
to activations. Scale bytes in these edge block files are linear; a loader
must convert them to the SM1xx SFB tile-interleaved layout before calling the
native block-scaled MMA kernels.

Each block carries a trailing pad so its offset and length are multiples of
`BLOCK_ALIGNMENT`. On a device whose memory holds only a fraction of the
experts, the expert stream cannot go through the page cache — it would compete
with the resident weights for the same physical memory — so the reader has to
use `O_DIRECT`, which requires aligned offsets and lengths. The INT4 group-16
payload is already a multiple of 4096; the INT8 payload is 3,151,872 bytes and
takes 2048 bytes of pad. `manifest.json` records `block_bytes`,
`block_alignment`, and the padding entry in `block_sizes`.

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

On Qwen3.6-35B-A3B the plain LRU wins from 16 slots per layer up, and its
margin grows with prompt length — at 43 slots per layer, 0.745 against 0.731
for a 32-token prompt and 0.711 against 0.664 for a 128-token prompt. Once a
per-layer quota exists, recency predicts this router's next selections better
than prompt-phase frequency, and a longer prompt makes the frequency estimate
more diffuse rather than more reliable. The oracle variant stays ahead of both
(0.776 at 43 slots for the 128-token prompt), so pinning is sound and the
prompt-derived choice of what to pin is what falls short. Treat the policy as
something to measure per checkpoint, not to assume.

Capacity dominates policy either way: going from 43 to 64 slots per layer cuts
read volume by a third, while any policy change at a fixed quota moves it by a
few percent.

`--block-bytes` (default: the INT4 group-16 block) and `--bandwidths` turn
misses per token into a read volume and the token rate each storage bandwidth
would allow, which is the number that decides whether a memory budget is
viable.
