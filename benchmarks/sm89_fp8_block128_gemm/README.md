# SM89 FP8 Block-128 GEMM Kernel Bench

This is a standalone kernel-iteration harness for the Qwen3-VL SM89 FP8
block-128 prefill GEMM. It is intentionally separate from the FlashRT Python
runtime so a low-context agent can iterate on the CUDA kernel quickly.

## Target Boundary

```text
A_fp8[M,K] + B_fp8[N,K] + act_scale[M,K/128] + w_scale[N/128,K/128]
    -> D_bf16[M,N]
```

The main production hotspot is equivalent to:

```text
fp8_bs_gemm_kernel<64,64,4,1,4>
```

The harness includes:

- `baseline`: current FlashRT SM89 kernel structure.
- `candidate`: a separate template instantiation for structural experiments.
- deterministic device-side input initialization.
- sampled FP32 reference checks.
- cold-L2 timing with an explicit flush buffer.
- NCU profile wrapper and CSV parser.

## Quick Start

```bash
cd /data/home/tianjianyang/code/FlashRT/benchmarks/sm89_fp8_block128_gemm
./build.sh
./build/bench_sm89_fp8_block128_gemm --shape gate --mode both
```

Representative shapes:

```bash
./build/bench_sm89_fp8_block128_gemm --shape gate  # M=1581,N=12288,K=4096
./build/bench_sm89_fp8_block128_gemm --shape up    # same as gate
./build/bench_sm89_fp8_block128_gemm --shape down  # M=1581,N=4096,K=12288
./build/bench_sm89_fp8_block128_gemm --shape qkv   # M=1581,N=6144,K=4096
```

Run NCU on one candidate launch:

```bash
./profile_ncu.sh candidate gate
python parse_ncu.py profiles/candidate_gate_details.csv
```

## Iteration Rules

- Make one structural change at a time in the candidate path.
- Do not start with broad tile or dispatch sweeps.
- Keep the baseline path unchanged unless intentionally updating the imported
  FlashRT baseline after validating a runtime change.
- A candidate is not accepted from timing alone. It must pass sampled
  correctness, improve cold-L2 timing beyond noise, and move NCU metrics in the
  expected direction.
- If NCU contradicts the hypothesis, reject or redesign the candidate even if
  raw timing is slightly faster.

## Current Baseline Profile Summary

Standalone bench baseline for `--shape gate` (`M=1581,N=12288,K=4096`) shows:

- target kernel: `64x64x128_w4_s1`
- cold-L2 event timing: about 0.716 ms median
- NCU duration: about 810 us
- registers/thread: 120
- spill load/store: 0 bytes in ptxas output
- dynamic shared memory/block: 18.43 KB
- theoretical occupancy: 33.33%, register-limited
- achieved occupancy: about 32.27%
- eligible warps/scheduler: about 0.45
- no eligible: about 71.7%
- issue slots busy: about 28.3%
- executed instructions: about 254.0 M

The first rejected structural experiment moved the temporary accumulator to an
M-atom scope. It reduced instruction count but increased registers from 120 to
128, did not improve occupancy, and worsened eligible warps. Do not repeat that
candidate without a new reason.

Use the NCU wrapper for comparable profiles:

```bash
./profile_ncu.sh baseline gate
./parse_ncu.py profiles/baseline_gate_details.csv
```

## Candidate C1: scale-load coalescing

NCU's top bottleneck on the baseline (~51% estimated speedup) is the global
load access pattern: each 32-byte sector delivers only 4 used bytes, caused by
the row-strided scalar `act_scale[row*K128+kb]` reads in the scale fold.

C1 stages the scales in shared memory with a coalesced load. To keep the smem
footprint independent of K (so occupancy does not regress on large-K shapes
like `down`, K128=96), it stages only `SCALE_KTILE=8` scale-block columns at a
time (`BLOCK_M*8 + 8` floats = ~2 KB), re-staged each time the k-loop crosses a
tile boundary.

A first attempt that staged the *entire* `[BLOCK_M, K128]` slice at once won on
`gate`/`qkv` (K128=32) but regressed `down` +17% (24 KB scale smem dropped
occupancy 33%→25%). The tiled version below is the fix.

Cold-L2 timing (`--flush-l2-mb 256`, both mode):

```text
gate M=1581,N=12288,K=4096:  baseline 0.7199 ms  candidate 0.6502 ms  (-9.7%)
down M=1581,N=4096,K=12288:  baseline 0.6636 ms  candidate 0.6154 ms  (-7.3%)
qkv  M=1581,N=6144,K=4096:   baseline 0.3359 ms  candidate 0.3052 ms  (-9.1%)
```

NCU (gate): duration 813→731 us, executed instructions 254.0M→200.4M (-21%),
theoretical occupancy held at 33.3% (smem footprint kept small), registers
120→124. Sampled correctness unchanged (max_rel ~6e-4).

## Candidate C2: store-pattern coalescing

After C1, NCU's top remaining bottleneck (~47%) is the global store pattern:
the scalar 16-bit BF16 epilogue stores deliver only 8 of 32 bytes per sector.
The m16n8 accumulator layout has each lane holding two adjacent output columns
per row (`acc[0,1]` = row0 cols {2l,2l+1}, `acc[2,3]` = row1), so C2 emits one
32-bit `bfloat162` store per row instead of two scalar stores (tail columns
keep scalar stores).

Cold-L2 timing (C1+C2 vs baseline):

```text
gate: baseline 0.7199 ms  candidate 0.6185 ms  (-14.1%)
down: baseline 0.6615 ms  candidate 0.6062 ms  (-8.4%)
qkv:  baseline 0.3400 ms  candidate 0.2929 ms  (-13.8%)
```

NCU (gate): duration 731→696 us, store sector utilization 8→16 bytes, store
bottleneck ~47%→~33%. Correctness unchanged. (Reaching 32/32 would need a
wider vector store across the discontiguous row halves — diminishing return,
left for later.)

## Rejected: C3 raise occupancy (register/CTA tuning)

After C1+C2, NCU still lists occupancy (~19% est.) as a bottleneck: 124
registers limit the kernel to 4 CTAs/SM (33.3% theoretical occupancy). Forcing
`MIN_BLOCKS_PER_SM=5` makes ptxas drop to 96 registers — but it spills (4 B
spill stores) and runs **slower**: gate candidate 0.6185 → 0.6482 ms (+4.8%).

This confirms the kernel is **latency-bound, not occupancy-bound**: Warp Cycles
Per Issued Instruction is ~14.7 and DRAM throughput is only ~10%, so the low
eligible-warp count comes from instruction/dependency latency, not from too few
resident warps. Cutting registers to add a CTA removes the latency-hiding
headroom each warp needs. NCU's occupancy estimate is theoretical and does not
realize here. Do not pursue register/occupancy tuning for this kernel without a
fundamentally different accumulator design.

## Candidate C4: ldmatrix.x4 + 128B-swizzle smem (structural)

Raw-page NCU on C1+C2 reveals the real top pipe: **LSU at 67.7%** (the highest),
with **54.7M shared-load instructions** (27% of all instructions) — the scalar
32-bit `LDS` reads of A/B fragments in the MMA inner loop. Tensor pipe is only
20.5%, so the kernel is bound by the load/store unit, not compute.

C4 ports the sm120 ldmatrix structure (`fp8_smallM_handtuned_ldmatrix_sm120`):
a 128B-swizzle smem layout (no `SMEM_K_PAD`) written by swizzled cp.async and
read by `ldmatrix.sync.aligned.x4.m8n8.shared.b16`, which loads four 8x8 b16
fragments per lane in one instruction. The MMA uses the sm89 e4m3 variant (the
sm120 `kind::f8f6f4` is sm120a-only); C1 scale staging and C2 pair-store are
kept (orthogonal). Build with `-DUSE_LDMATRIX`.

Cold-L2 timing (C4 vs baseline):

```text
gate: baseline 0.7199 ms  candidate 0.6072 ms  (-15.7%)
down: baseline 0.6636 ms  candidate 0.5827 ms  (-12.2%)
qkv:  baseline 0.3400 ms  candidate 0.2866 ms  (-15.7%)
```

NCU (gate): **LSU pipe 67.7%->29.8%**, **shared-load instructions 54.7M->5.5M
(-10x)**, Tensor pipe unchanged (~21%). Correctness unchanged (max_rel matches
baseline exactly — swizzle + ldmatrix addressing is bit-correct). All
production tiles have an even N_ATOMS_PW, so the ldmatrix pairing constraint
holds for the whole dispatch table.
