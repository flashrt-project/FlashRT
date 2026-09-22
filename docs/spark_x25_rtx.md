# Spark-X2.5-4B on RTX SM120 (16 GB)

Single RTX 5060 Ti, 16 GB, SM120, CUDA 13.0. Batch 1. NVFP4 W4A4 weights, an
E4M3 KV cache, and a decode loop captured whole into one CUDA Graph.

See `docs/spark_x25_usage.md` for the parameter reference.

## Optimization structure

The decode step is a weight stream plus a KV stream, and which one dominates
depends entirely on context. That is the whole shape of this port:

| stage | kernel | source |
|---|---|---|
| weight stream (NVFP4 W4A4 GEMMs) | `fvk.fp4_w4a4_mma_sm120*` | `flash_rt_kernels` |
| prefill attention | FA2 (`fwd_bf16_causal`, `fwd_bf16_window`) | `flash_rt_fa2` |
| **decode attention** | two-pass over the KV cache | `flash_rt_sparkx25` |
| KV write (partial RoPE + E4M3 quantise) | `qkv_post_rope_kvwrite` | `flash_rt_sparkx25` |
| boundary norm + NVFP4 quantise, output gate, gproj, argmax, position advance | — | `flash_rt_sparkx25` |

### Why the decode attention is not FA2

FA2's decode entry launches one block per (query block, KV head, batch). A decode
step is a single query row, so that grid is `1 x 4 x 1` — **four blocks on a
36-SM part**. Its `num_sms` split recovers part of that but saturates early.
Swept over the entry's whole configuration space at the 131 072-key shape:

| entry | GB/s | % of the 425.8 GB/s ceiling |
|---|---|---|
| `fwd_bf16_seqused` (plain, 4 blocks) | 78.6 | 18% |
| `fwd_bf16_seqused_splitkv`, `num_sms`=16 (64 blocks) | 154.3 | 36% |
| `fwd_bf16_seqused_splitkv`, `num_sms`=96 (384 blocks) | 168.1 | 39% |
| this module's two-pass, bf16 KV | 383 | 90% |
| this module's two-pass, E4M3 KV | 323 | 76% |

The shape is not the obstacle: the same tensor, layout and bytes through a plain
GEMM run at 404 GB/s. FA2 still serves prefill, where it wins.

The replacement materialises the whole score row `S[head][key]`, so pass 2 is a
plain weighted sum of V rows with no online softmax and no per-split maxima:

```
state_init -> scores (S = q K^T, + tile atomicMax) -> softmax (P = exp(S-m), + row sums)
           -> pv (per-split weighted sum of V rows) -> pv_combine (sum splits / l)
```

Two implementation details carry the speed: the scores pass pads its
shared-memory row pitch by 8 elements (unpadded, every lane of a warp reads a
different 512-byte-apart row at the same column — a 32-way bank conflict), and a
pass-2 block owns all four KV heads so its V rows are the full 2048-byte row
rather than a 512-byte quarter.

### The KV split count

Both passes split along KV, and the count is bounded in both directions: a
segment must be long enough to stream, and the combine reads
`nsplit x heads x head_dim` partials per layer whatever the key count is. The
optimum segment runs from ~16 keys at 512 to ~512 at 128k, i.e. one split per
~32 tokens of budget, floored at 32 and capped at 256. Extrapolating from the
128k point alone under-parallelises the middle by 2.5x:

| klen | best `nsplit` | us | one-split-per-512-tokens gives |
|---:|---:|---:|---:|
| 512 | 32 | 18.7 | 29.0 |
| 2 048 | 64 | 29.0 | 78.1 |
| 8 192 | 128 | 61.7 | 157.9 |
| 32 768 | 256 | 228.7 | 359.6 |
| 131 072 | 256 | 843.5 | 843.5 |

## Measurements

Repeated-text prompt, 128 decode steps replayed from the captured graph,
`benchmarks/spark_x25_rtx_latency.py`:

| prompt ctx | 128 | 512 | 2 048 | 32 k | 64 k | 128 k | 256 k |
|---|---|---|---|---|---|---|---|
| TTFT ms | 14 | 29 | 114 | 3 457 | 10 628 | 36 105 | 132 762 |
| decode tok/s | 138.7 | 136.4 | 135.9 | 111.3 | 92.7 | 69.7 | 46.7 |

Against Ollama serving the same checkpoint on the same card, same prompts
(decode rate from the API's own `eval_count`/`eval_duration`, `load_duration`
excluded):

| prompt ctx | 128 | 512 | 2 048 | 32 k | 64 k | 128 k | 256 k |
|---|---|---|---|---|---|---|---|
| Ollama tok/s | 46.4 | 46.1 | 45.9 | 40.5 | 36.3 | 29.9 | 1.87 |
| this frontend | 138.7 | 136.4 | 135.9 | 111.3 | 92.7 | 69.7 | 46.7 |
| | **2.99x** | **2.96x** | **2.96x** | **2.75x** | **2.56x** | **2.33x** | **25x** |

The 256k row is not a like-for-like comparison: Ollama does not fit there and
moves 24% of the weights to host memory (`/api/ps` reports 10.02 GB resident,
7.59 GB of it on the GPU), so its 1.87 tok/s is a PCIe-bound reading. The
comparable long-context point is 128k at 2.33x.

## Why the rate falls with context, and what would flatten it

A full-attention decoder must read its whole KV every step, so the step cost is

```
per-step bytes = 9 full layers x 4 KV heads x 256 head_dim x 2 (K,V) x 1 byte x context
               + 2.31 GB of NVFP4 weights
```

At 262 144 tokens the KV term is 4.83 GB against 2.31 GB of weights — the KV is
twice the weight stream, which is why decode falls from 138.7 to 46.7 tok/s
across the range. The marginal cost is 52-57 ns per context token, i.e.
322-352 GB/s against the ceiling.

Two things flatten that curve elsewhere and neither is available here:

- **Speculative decoding**, which is what Qwen3.6's MTP/DFlash path actually
  buys. The KV is read once per *step* regardless of how many query rows the
  step verifies, so a step that emits R tokens divides the KV cost by R. That
  checkpoint ships a drafter (`z-lab/Qwen3.6-27B-DFlash`); Spark-X2.5-4B does
  not, and FlashRT ships no frontend for it, so the missing piece is a trained
  artefact rather than an inference implementation.
- **A larger weight stream.** Qwen3.6-27B at NVFP4 streams 13.5 GB/token, so at
  the same context its KV (8.59 GB) is still smaller than its weights and stays
  a minority of the step. At 4B the KV overtakes the weights early.

It is not a VRAM limit. With bf16 KV and unlimited memory this model would read
9.7 GB/token at 262k and the curve would be **steeper**; the E4M3 cache is what
halves it. VRAM decides which KV precision fits and whether prefill can stay
exact, not the slope.

## The E4M3 load path costs ~18%

The same scores kernel reads bf16 at 405 GB/s (95% of the ceiling) and E4M3 at
327 (77%). Five hypotheses were tested:

| hypothesis | result |
|---|---|
| arithmetic | removing the whole compute loop leaves the time unchanged (425.7 vs 425.3 us) |
| dequant cost, fp32 -> half2 | 425.4 -> 422.8 us (0.6%) |
| insufficient ILP (4 chains, second accumulator) | 422.6 -> 425.8 us |
| one scale load per row | **425.4 -> 409.9 us** (shipped) |
| row scales via shared-memory broadcast | **409.9 -> 447.1 us** (worse, reverted) |

The decisive experiment puts the data in L2 (klen 4096-16384, fully resident):
bf16 reaches 569 GB/s, E4M3 314. **E4M3 reaches only 55% of bf16 with no DRAM in
the path**, so the tax is SM-side instruction throughput: expanding 16 E4M3 bytes
into 16 bf16 costs two shared-memory stores per input word where bf16 costs one.
Keeping the tile in E4M3 and dequantising inside the dot product was costed and
rejected — compute instructions per thread go 448 -> 704 while the load path
saves 4.

This is a bounded residual of ~12% at 262k, not a tuning knob: it needs a
different tile layout, not different parameters.

## Prefill attention

Sliding layers run one FA2 call per layer with an explicit 512-key window
(`fvk.attention_fa2_fwd_bf16_window`). FA2 statically forbids
`Is_causal && Is_local`, so a bounded *past* window cannot go through the causal
entry at all; the window entry is the non-causal path with `window_left = W-1`
and `window_right = 0` over `seqlen_k == seqlen_q`, which clamps the mask's right
edge to `row + 1` and yields exactly the causal window.

Without that entry the same window is expressible through FA2's batch-stride
trick (one batch element per query), which is exact but makes each query re-read
its whole window: on a 2263-token prompt the 27 sliding layers then move ~128 GB
of K/V and take 594 ms of a 660-680 ms TTFT.

## A wider SM120 part: RTX 5090

The same code on one RTX 5090 (32 GB, 170 SMs, CUDA 12.8), same repeated-text
prompt and 128-step captured graph (`benchmarks/spark_x25_rtx_latency.py`):

| prompt ctx | 128 | 512 | 2 048 | 8 k | 32 k | 128 k | 512 k | 1 M |
|---|---|---|---|---|---|---|---|---|
| KV mode | bf16+E4M3 | bf16+E4M3 | bf16+E4M3 | bf16+E4M3 | bf16+E4M3 | bf16+E4M3 | E4M3 only | E4M3 only |
| full-layer split | 128 | 128 | 128 | 128 | 257 | 1 024 | 1 024 | 1 024 |
| TTFT ms | 5.7* | 8.7* | 27.7 | 131 | 878 | 9 208 | 131 108 | 511 961 |
| decode tok/s | 360.6 | 354.7 | 347.0 | 319.0 | 276.7 | 198.7 | 95.4 | 56.4 |

\* Warm median; the benchmark's single cold first prefill of a fresh process
reports 16-63 ms there. Below ~8k the whole prompt is one prefill chunk, so
TTFT is fixed-cost bound and not a throughput reading; from 8k up it is the
benchmark's own number. 1M is the checkpoint's native maximum and fits in 32 GB
(E4M3-only KV, ~30.4 GB peak).

At short context the step is a weight stream (2.31 GB/token) plus 9 full layers'
KV read. Both stream at the DRAM limit when the decode GEMMs run alone: a plain
2-stream bf16 copy reaches **1 527 GB/s** here (nominal 1 792), and the isolated
`lm_head` decode GEMM -- the one DRAM-resident shape at 167 MB -- reaches
**1 635 GB/s**. The measured short-context 360 tok/s (2.777 ms) lands between
the extrapolation table's 70% and 90% rows below, so the weight-bound end is
behaving as predicted. What was *not* at speed was the KV attention.

### The split count is a per-part knob

The 36-SM rule above (one split per 32 tokens of budget, capped at 256)
under-parallelises a 170-SM part. The split count is a partition of the key
range and the combine is a sum, so re-sweeping it changes no arithmetic --
only the block count. Swept per length on the 5090:

| klen | optimum nsplit | tok/s at optimum | capped-256 tok/s |
|---:|---:|---:|---:|
| 8 192 | 256 | 321.6 | 321.6 |
| 32 768 | 512 | 280.9 | 272.8 |
| 131 072 | 768 | 197.6 | 167.1 |
| 524 288 | 1 024 | 95.0 | 66.9 |
| 1 048 576 | 1 024 | 56.3 | 37.3 |

The cap cost **19% at 128k and 51% at 1M**. The default now scales with the
part -- `min(1024, max(128, max_seq // 128))` on >64-SM parts, the 36-SM rule
kept unchanged below that -- and the sliding-window split moves 32 -> 64 (a
512-key window leaves most of a 170-SM part idle at 32 blocks). End to end on
the same benchmark, before -> after the change:

| ctx | 128 | 512 | 2 048 | 8 k | 32 k | 128 k | 512 k | 1 M |
|---|---|---|---|---|---|---|---|---|
| before | 360.1 | 345.8 | 336.7 | 320.3 | 272.4 | 167.3 | 67.0 | 37.4 |
| after | 360.6 | 354.7 | 347.0 | 319.0 | 276.7 | 198.7 | 95.4 | 56.4 |
| | +0% | **+3%** | **+3%** | -0% | +2% | **+19%** | **+42%** | **+51%** |

After tuning the 1M step moves 21.6 GB in 17.73 ms = **1 218 GB/s**, 80% of the
measured copy ceiling, against 808 GB/s before. The remaining long-context gap
is the E4M3 load path (SM-side instruction throughput, not DRAM) and the
attention tile; at short context it is the fixed ~1.4 ms of launches, norms, KV
write, gate and argmax that caps a 5090-class step at ~989 tok/s even with free
bandwidth:

| GEMM efficiency | step | decode |
|---|---|---|
| 100% | 2.30 ms | 435 tok/s |
| 90% (what this card reaches) | 2.44 ms | 409 tok/s |
| 70% | 2.85 ms | 351 tok/s |

The attention tile and `_DECODE_SPLIT`'s per-shape warp/stage table are still
tuned for 36 SMs and are the next two things to re-sweep on a wider part.

## Scope

Batch 1, SM120, greedy only. No speculative decoding — see above for why that is
a checkpoint limitation rather than an implementation one.
