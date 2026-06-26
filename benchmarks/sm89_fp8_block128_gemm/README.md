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
