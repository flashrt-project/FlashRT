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

- **C6 multi-stage cp.async pipeline (STAGES=2/3, SM120a-style)**: the SM120a
  reference GEMM (`fp8_smallM_handtuned_ldmatrix_sm120`) defaults to a 2-stage
  (and tunes up to 5-stage) cp.async pipeline to hide load latency, and this
  kernel is latency-bound, so S>=2 looked like the natural next step. It
  **regresses**: 64x64 gate +12.4%, down +10.1%, qkv +12.0% at S=2; S=3 is
  +60-73%. NCU (gate, S=2) shows the mechanism clearly and why it differs from
  SM120a: the second smem stage grows dynamic smem 18.5 -> 34.9 KB, which drops
  the shared-mem block limit 5 -> 2 and so halves occupancy (33.3% -> 16.7%,
  4 -> 2 CTA/SM). Per-warp latency hiding *does* improve (Warp-Cycles/Inst
  11.74 -> 6.17), but eligible warps/sched still *fall* (0.46 -> 0.37) because
  losing half the resident CTAs removes more warps than double-buffering gains.
  This kernel is **register-limited to 4 CTA/SM at 128 regs** with no smem
  headroom on the 4090 (4 x 34.9 KB = 139 KB > 100 KB/SM). SM120a affords S>=2
  because its motus kernel uses per-tensor `alpha` (no block-128 scale staging)
  -> fewer regs + less smem -> occupancy headroom the block-128 Qwen3-VL kernel
  does not have. **S=1 is correct for this kernel; the SM120a multi-stage design
  does not port because the block-128 scaling changes the occupancy budget.**
  (Bench harness keeps `./build.sh --experiment --stages N --bm M --bn N --warps W`
  for future re-tests.)
  - **Small-tile S=2 (hold occupancy)**: to keep >=3 CTA/SM at S=2 the tile must
    shrink (`BM+BN<=91`). Tested 32x64 and 32x128: 32x64 S=1 is already +36% vs
    64x64 S=1 (arithmetic intensity drops), and S=2 over S=1 at 32x64 is a wash
    (+0.7%) -- the overlap buys nothing once the tile is small. 32x128 S=2 is
    +23.6%. **Two-sided squeeze: big tile -> S=2 halves occupancy; small tile ->
    arithmetic-intensity loss dwarfs any S=2 gain. No tile wins.** This is the
    no-TMA penalty: on Ada a deeper cp.async pipeline always costs either
    occupancy or intensity.
  - **CUTLASS won't help on SM89**: the SM120 Qwen3-VL GEMM
    (`fp8_block128_gemm_cutlass_sm120_bf16out`) wins via `KernelTmaWarpSpecialized
    BlockwisePingpongSm120` -- TMA + warp specialization + auto multi-stage.
    Ada (sm89) has **no TMA and no warp-specialized/TMA collective in CUTLASS**
    (only Sm90+/Sm120). CUTLASS's Ada blockwise (example 94) is the same cp.async
    `device::GemmBlockwise` class we hand-wrote, and was already benched at
    ~64 TFLOPS (slower than this kernel) with an activation-scale precision
    downgrade. The only way to get TMA-style pipelining benefit on sm89 is a
    hand-written **warp-specialized producer/consumer** kernel (deferred).
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
