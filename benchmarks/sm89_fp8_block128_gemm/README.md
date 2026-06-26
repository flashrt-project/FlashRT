# SM89 FP8 Block-128 GEMM Kernel Bench

Standalone kernel-iteration harness for the Qwen3-VL SM89 FP8 block-128
prefill GEMM. Kept separate from the FlashRT Python runtime so a low-context
agent can iterate on the CUDA kernel quickly.

## Target Boundary

```text
A_fp8[M,K] + B_fp8[N,K] + act_scale[M,K/128] + w_scale[N/128,K/128]
    -> D_bf16[M,N]
```

The production hotspot is `fp8_bs_gemm_kernel<64,64,4,1,4>`
(`fp8_block128_gemm_bs_sm89_64x64x128_w4_s1`).

## Zero-drift baseline (read this first)

The `baseline` kernel is **not a copy** — `--mode baseline` runs the *exact*
production kernel by `#include`-ing `csrc/gemm/fp8_bs_gemm_device.cuh`, the same
header the production `.cu` compiles. So the bench baseline cannot lag behind
production, which is the bug that previously made the C1-C4 deltas look ~10x
bigger than they were: the old frozen baseline was the pre-C1 kernel, ~19%
slower than what production actually shipped, so every reported delta was
"cumulative vs a stale kernel" instead of "marginal vs current production".

The `candidate`:
- **Default build** (`./build.sh`): candidate aliases the production kernel, so
  `--mode both` reports **~0%**. This is a built-in honesty check — if a fresh
  bench shows a nonzero delta, the harness itself is wrong.
- **Experiment build** (`./build.sh --experiment`): compiles the editable
  `fp8_bs_gemm_kernel_cand` (seeded identical to production). Edit *that* kernel
  for a structural experiment; baseline stays pinned to production.

After an experiment is accepted, fold it into
`csrc/gemm/fp8_bs_gemm_device.cuh`. Production and the bench baseline both pick
it up automatically — no second copy to keep in sync.

**Always read candidate deltas as marginal-over-current-production**, never as
cumulative-over-some-old-baseline.

## Quick Start

```bash
cd benchmarks/sm89_fp8_block128_gemm

# Faithful baseline (candidate == production): expect ~0% delta.
./build.sh
./build/bench_sm89_fp8_block128_gemm --shape gate --mode both

# Iterate on a structural change:
./build.sh --experiment            # then edit fp8_bs_gemm_kernel_cand
./build/bench_sm89_fp8_block128_gemm --shape gate --mode both
```

Representative shapes (M=1581 ~ the real language-prefill M; vision uses
M=6256, pass it explicitly with `--M`):

```bash
--shape gate  # M=1581,N=12288,K=4096   (gate/up)
--shape down  # M=1581,N=4096,K=12288
--shape qkv   # M=1581,N=6144,K=4096
--M 6256 --N 3456 --K 1536   # representative vision linear
```

NCU on one candidate launch:

```bash
./profile_ncu.sh candidate gate
python parse_ncu.py profiles/candidate_gate_details.csv
```

## Methodology notes

- **Cold-L2 is the right default** (`--flush-l2-mb 256`). In real multimodal
  prefill each layer's FP8 weights (tens of MB) are cold per-GEMM — they don't
  fit alongside the next layer's in the 4090's 72 MB L2. The cold-bench gate
  time (~0.61 ms) matches the production end-to-end gate kernel time, so the
  flush models reality. Warm timing (`--flush-l2-mb 0`) understates weight
  traffic and is only useful for isolating compute-bound effects.
- A candidate is **not** accepted on timing alone. It must pass sampled
  correctness, beat baseline beyond noise on cold-L2, and move the relevant NCU
  metric in the predicted direction. If NCU contradicts the hypothesis, reject.
- One structural change at a time. No broad tile/dispatch sweeps.

## Iteration history (deltas are vs the ORIGINAL pre-C1 kernel)

These shipped and are now folded into the shared header, i.e. they ARE the
current baseline. A new candidate must beat this baseline, not re-bank these.

| step | change | gate cold | vs pre-C1 |
|------|--------|-----------|-----------|
| pre-C1 | original scalar scale-load + scalar store + padded smem | 0.720 ms | — |
| C1 | scale-load coalescing (stage SCALE_KTILE scales in smem) | 0.650 ms | -9.7% |
| C2 | store-pattern coalescing (bfloat162 pair store) | 0.619 ms | -14.1% |
| C4 | ldmatrix.x4 + 128B-swizzle smem (no pad) | 0.612 ms | -15.7% |

Each step's INCREMENTAL value (not cumulative):
- C1: -9.7%   (the big one — global scale-load was the top NCU bottleneck)
- C2: ~-4.5%  (store sectors 8B -> 16B)
- C4: ~-1.3%  (LSU 67.7% -> 29.8%, but LSU was no longer binding after C1)

End-to-end (8B full-res prefill, nsys): GEMM total 125.3 ms (clean PR111) ->
107.7 ms (C4) = **-14%**, faithful to the cold-bench per-shape -15.6%. Total
prefill 208.6 -> 190.9 ms = **-8.5%** (Amdahl: GEMM is only 56% of prefill;
FA2 vision attn is 29%, untouched on 8B).

### Rejected

- **C5 smem-staged epilogue** (fully-coalesce global stores 16B->32B/sector):
  NCU on the C4 baseline ranks the global-store pattern as the #1 actionable
  rule (24.5% estimated). C5 stages the output tile in the (now-free) A/B smem
  and streams it out as 128-bit uint4 stores. **Hypothesis mechanically
  confirmed** — NCU shows the store-pattern rule eliminated (stores now 32/32).
  But duration only moved 686.3 -> 682.5 us (**-0.6%**, bit-exact), because the
  kernel is latency-bound and stores were never on the critical path; the smem
  round-trip even adds +5.3% instructions (249.7M -> 262.9M). Same lesson as C3:
  NCU per-rule estimates are theoretical and don't realize on this latency-bound
  kernel. Not worth a `__syncthreads` + 8 KB epilogue + only-tested-at-M=1581
  risk on small-M tiles for ~0.5%. **Conclusion: the obvious memory-pattern
  levers (load coalescing C1, store coalescing C2/C5, ldmatrix C4) are
  exhausted; the kernel sits at its latency wall.** Further GEMM gains need a
  fundamentally different accumulator/pipeline design, not another coalescing
  pass. End-to-end, the bigger lever is now vision attention (29% of 8B prefill).
- **C3 raise occupancy** (force `MIN_BLOCKS_PER_SM=5`): ptxas drops to 96 regs
  but spills and runs +4.8% slower. The kernel is **latency-bound, not
  occupancy-bound** (Warp Cycles/Issued-Inst ~14.7, DRAM ~10%). Do not pursue
  register/occupancy tuning without a fundamentally different accumulator
  design.
- **Temp accumulator at M-atom scope**: fewer instructions but 120->128 regs,
  worse eligible warps, no occupancy gain.
