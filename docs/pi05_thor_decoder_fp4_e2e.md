# Pi0.5 Thor NVFP4 End-to-End Results

## Current Result (2026-08-05)

The configuration is the full Thor NVFP4 tier, which
`load_model(..., use_fp4=True, use_fp4_decoder=True, use_fa4=True)`
resolves in one call (see [Implemented Path](#implemented-path)). Each row
is one same-session A/B; rows are separate sessions, so compare within a
row only. These are the numbers the README quotes.

| Views | Same-run FP8 p50 | NVFP4 + FA4 p50 | Speedup | Throughput | Gates |
|---:|---:|---:|---:|---:|---|
| 1 † | 32.92 ms | **23.01 ms** | 1.431 | 43 Hz | fidelity gates fail |
| 2 | 38.70 ms | **27.17 ms** | 1.424 | 37 Hz | all pass |
| 3 | 49.02 ms | **31.74 ms** | 1.544 | 32 Hz | all pass |

Matched-noise fidelity against the same-run FP8 reference:

| Views | Raw cosine / worst | Final action cosine / worst |
|---:|---:|---:|
| 1 † | 0.991367 / 0.965022 | 0.993474 / 0.971119 |
| 2 | 0.999206 / 0.998030 | 0.999720 / 0.999157 |
| 3 | 0.999036 / 0.997660 | 0.999736 / 0.999435 |

† The 1-view row does not clear the per-sample cosine gates. See
[One-View Fidelity](#one-view-fidelity-diagnosis-and-passing-configuration-2026-07-27)
for the diagnosis and a configuration that does clear them.

Both children run FA4, so these speedups isolate NVFP4 against FP8 with the
attention backend held fixed.

### Public API confirmation run

The table above was measured by constructing the frontends directly. A
second session re-ran all three view counts through the harness default
`--construct load_model`, which builds both children with
`flash_rt.load_model()` (see [Verification Contract](#verification-contract)).
Every cosine came out identical to the table above at every view count,
which is the evidence that the two construction paths produce the same
configuration:

| Views | Same-run FP8 p50 | NVFP4 + FA4 p50 / p95 | Speedup | FP4 100-sample spread |
|---:|---:|---:|---:|---:|
| 1 † | 32.80 ms | 23.07 / 23.14 ms | 1.422 | 0.18 ms |
| 2 | 38.62 ms | 27.29 / 27.34 ms | 1.415 | 0.12 ms |
| 3 | 45.75 ms | 31.89 / 31.94 ms | 1.435 | 0.11 ms |

The FP4 latencies land within 0.15 ms of the table above. The FP8 baseline
drifts a few ms between sessions on this hardware, which moves the 3-view
ratio without moving the FP4 result — the reason only within-row ratios are
meaningful.

Artifacts (confirmation run):

- `flash_rt_fp4`: `22216c9514bb16c8a3be1578f1f3c814ec9d3a4a02ec77472a911a919984ae79`
- `flash_rt_kernels`: `c40f72766d8c16def8758b2d2590250de0970f09bf6071c54da96b97b52bfccf`
- 1-view `result.json`: `5b159a002c61d8de215f82d291bb3a7db1474897f5ea768587ec371d1616ec86`
- 2-view `result.json`: `cd8bf27a8f16c05e600684f2918d07d6e058c672e22a0680501ec205eaa3d216`
- 3-view `result.json`: `7f0ad414b7538c9d7208ad39c6e0d4e5f0adbfb7d76d5b1dabc780fb103f81af`

The sections below are the chronological development log that produced this
configuration; their absolute latencies belong to their own sessions.

## Vectorized SigLIP LayerNorms (2026-07-27)

The two per-layer SigLIP LayerNorms were the largest non-GEMM item in
the steady-state kernel profile (1.3 ms of the 2.7 ms normalization
bucket at 3 views). Register-resident single-pass variants (16-byte
loads, one 16-element block per thread, all three stages from registers)
replace them with fallback to the originals on unsupported dims:
LayerNorm-to-FP8 23.9 -> 11.1 us (2.16x), LayerNorm-to-NVFP4
28.8 -> 24.7 us at the production shape, about 0.35-0.44 ms per 3-view
frame. Outputs agree byte-for-byte on real shapes (reduction order
differs at floating-point rounding level). Formal 3-view run passes all
gates (raw min 0.99708, action min 0.99904); the end-to-end delta is
within run-to-run regime noise, consistent with the kernel-level
measurement.

## One-View Fidelity: Diagnosis and Passing Configuration (2026-07-27)

The long-standing 1-view blocker is now characterized. It is not an
implementation defect, and the original "gripper sign flip" description
no longer holds: on the current branch the gripper dimension matches the
FP8 reference exactly (cosine +1.0000). The failing samples instead show
the dominant translation component of the whole action trajectory
changing direction — a denoising-trajectory bifurcation, not noise.

Ablations pin the mechanism. With the full FP4 stack, one sample fails
(worst 0.843); with the encoder returned to FP8 and only the decoder in
FP4, that sample recovers to 0.999 but a *different* sample fails
(0.816). Two configurations from the same quantization family flipping
different samples means several 1-view observations sit near decision
boundaries of the flow-matching velocity field — with single-view
information the model itself is uncertain, and any small perturbation,
regardless of source, can push a trajectory into a different action
basin. Per-sample cosine against one FP8 reference is therefore an
over-strict acceptance criterion for 1-view; task-level evaluation is
the meaningful judge. Decoder precision still helps monotonically: the
rotated full-INT4 decoder alone lifts the worst sample from 0.843 to
0.930.

A fully passing configuration exists for deployments that require the
cosine gates at 1 view — encoder in FP8, decoder in rotated full-INT4:

```
--num-views 1 --construct frontend \
--encoder-fp4-layer-count 0 --siglip-ffn-fp4 0 --encoder-attn-o-fp4 0 \
--decoder-weight-format e0m3 --decoder-act-format e0m3 --decoder-rht 1
```

It deviates from the published preset, so it needs `--construct frontend`;
`load_model()` does not expose the e0m3 / Hadamard decoder knobs.

Formal 1-view result for this configuration: raw cosine 0.99974 with
worst sample 0.99954, action cosine 0.99990 with worst sample 0.99973
(8/8 samples, every precision gate PASS) at 28.99 ms — about 1.9 ms over
the all-FP4 1-view pipeline from running the encoder in FP8.

## Full-INT4 Decoder Activations + Hadamard Rotation (2026-07-27)

`--decoder-act-format e0m3` extends the uniform-INT4 grid to all five
decoder activation quantize exits (AdaRMS entry, both gated-residual
AdaRMS sites, attention context, GeGLU), and `--decoder-rht 1` rotates
every 16-value block by the orthonormal 16x16 Hadamard matrix on both
operands before quantization (lane-parallel kernels use a 4-stage
shfl_xor butterfly, vectorized kernels an in-register FWHT; the rotation
is mathematically inert on the GEMM). Formal 3-view, all gates PASS,
same-session NVFP4 baseline for reference:

| decoder config | p50 (ms) | raw cos | raw min | act min |
|---|---|---|---|---|
| W4A4 INT4 + RHT | 35.061 | 0.99937 | 0.99904 | 0.99942 |
| W4A4 INT4 | 34.902 | 0.99861 | 0.99795 | 0.99891 |
| NVFP4 (default) | 34.707 | 0.99906 | 0.99833 | 0.99899 |

The rotated full-INT4 configuration is the most accurate decoder
quantization measured to date — the Hadamard rotation gaussianizes the
per-block distributions, and the uniform grid then beats E2M1-with-MSE
on every cosine metric — at ~0.35 ms from the software E0M3 encoders.
The default remains `nvfp4`; the tiers exist for task-level evaluation
and as precision headroom for further compression.
## E0M3 Uniform-INT4 Decoder Weights (2026-07-27)

The SM110 tcgen05 block-scaled MMA reads its 3-bit element-format field
from the runtime instruction descriptor; value 0 decodes a sign-magnitude
uniform INT4 grid (E0M3, magnitudes 0..7) alongside the documented E2M1
at value 1. An element-level canary confirms the decode exactly,
including negative and mixed-nibble payloads, with no binary patching.

`--decoder-weight-format e0m3` quantizes all four decoder projection
weights to E0M3 (per-16 UE4M3 scales, amax/7) and routes them through a
runtime-datatype GEMM on the production 128x64x256 tile, with E2M1
activations unchanged. The quantizer is bit-exact against a host
reference; the GEMM matches an fp32 reference at 0.999998+ cosine on all
four decoder shapes.

Back-to-back formal 3-view runs, both all gates PASS:

| decoder weights | p50 (ms) | FP8 ref | raw min | act min |
|---|---|---|---|---|
| E0M3 (amax/7) | 36.829 | 49.211 | 0.99834 | 0.99896 |
| NVFP4 (MSE, default) | 36.796 | 49.370 | 0.99833 | 0.99899 |

Speed and precision are equivalent at this operating point — decoder
weight quantization is not the current end-to-end error bottleneck, and
the runtime-datatype kernel carries no performance penalty. The default
stays `nvfp4`; the E0M3 path is the foundation for more aggressive
low-bit configurations (coarser scales, INT4 activations, Hadamard
rotations) where the uniform grid's lower quantization error pays off.

## Encoder Attention QKV NVFP4 + Scale-Factor Zero-Init (2026-07-27)

Two changes landed together in this round.

**Scale-factor buffer zero-init (fix, `cefb181`).** The tile-interleaved
SFA/SFB layouts round K up to 64-element atoms while the quantize kernels
and fp4out epilogues only write real (row, block) entries. With K not a
multiple of 64 — the SigLIP FFN's padded hidden dim 4320 is the only such
shape in the pipeline — the padding entries kept allocation garbage. Bytes
0x7F/0xFF there decode as UE4M3 NaN and poison the block-scaled GEMM
accumulator, so whether the pipeline produced NaN vision embeddings
depended on allocator history (reusing freed weight-staging memory
triggered it deterministically; fresh pages hid it). All six SFA/SFB
allocation sites are now zero-initialized. Init-time cost only; healthy
runs are bit-identical.

**Encoder attention QKV NVFP4 with AWQ (`3b2b678`, off by default).** The
QKV projections quantize to NVFP4 with the input LayerNorm folded and a
fused weightless-RMSNorm x AWQ-inverse-scale -> NVFP4 kernel replacing the
FP8 RMSNorm. Plain 4-bit Q/K weights break the raw-cosine gate (worst
sample 0.97), and per-channel AWQ collected at the QKV input during
multi-sample calibration recovers it. Formal 3-view result with the full
FP4 stack and SigLIP FFN enabled, all gates PASS:

| config | p50 (ms) | FP8 ref | speedup | raw min | act min |
|---|---|---|---|---|---|
| QKV NVFP4 on | 37.132 | 49.321 | 1.3283 | 0.99661 | 0.99889 |
| QKV FP8 (default) | 36.773 | 49.652 | 1.3502 | 0.99833 | 0.99899 |

The flag ships **disabled**: at encoder sequence length the QKV GEMM is
compute-bound rather than weight-bandwidth-bound, so NVFP4 only matches
the FP8 GEMM while the extra fused quantize kernel costs ~18 us per layer
(~0.33 ms per frame) and the quantization consumes raw-cosine margin.
The kernels and calibration path remain available behind
`--encoder-attn-qkv-fp4 1` (with `--construct frontend`) for shapes where
the projection is bandwidth-bound.

## SigLIP AWQ, 27-Layer Preset (2026-07-27)

Commit `dde9809` adds activation-aware requantization for the SigLIP FFN
Up weights (see the commit message for the mechanism) and moves the
SigLIP preset from 16 to all 27 layers. Formal strict-harness runs, all
gates passed, both children fully regime-stable (FP4 sample spread under
0.15 ms across 100 samples):

| Views | Same-run FP8 p50 | FP4+FP4 p50 / p95 | Speedup |
|---:|---:|---:|---:|
| 2 | 38.633 ms | **29.587 / 29.655 ms** | 1.3057 |
| 3 | 47.676 ms | **34.864 / 34.922 ms** | 1.3675 |

Matched-noise fidelity versus the same-run FP8 reference (better than
the 16-layer preset on every metric):

| Views | Raw cosine / worst | Final action cosine / worst |
|---:|---:|---:|
| 2 | 0.999102 / 0.997126 | 0.999659 / 0.998810 |
| 3 | 0.999061 / 0.998330 | 0.999631 / 0.998995 |

Artifacts:

- `flash_rt_fp4`: `3cded1e3d46f9e789686485c4081e61f185049079e252195c51293ff5469e41f`
- `flash_rt_kernels`: `b9173596a459cec26fc9044ba796ab042e40ada89709eec6854dbdee8c486b37`
- 2-view `result.json`: `d1f08fe8642cabd41b48990bfda1375c845a7f3cdac46b4c113bfd26da88c8cb`
- 3-view `result.json`: `bf047a04b779e7c5270df849c90b2416c8a3a69b3910b354bf864d26e126e5a2`

## Decoder v10 Tiles (2026-07-27)

Commit `d12fbfc` switches the decoder qkv/o/down GEMMs to the narrow-N
v10 tile (gate_up already used it); per-kernel evidence is in the commit
message. Formal strict-harness runs (all gates passed; fidelity is
bit-identical to the previous section, as expected from a
schedule-only change):

| Views | Same-run FP8 p50 | FP4+FP4 p50 / p95 | Speedup |
|---:|---:|---:|---:|
| 2 | 38.576 ms | **29.782 / 29.855 ms** | 1.2953 |
| 3 | 49.594 ms | **34.922 / 37.154 ms** | 1.4201 |

The 2-view run held one clock regime throughout; its speedup gain over
the previous section (1.2839 to 1.2953) matches the kernel-level
prediction of roughly 0.3 ms. The 3-view FP4 child crossed from the
slow into the fast sustained-load regime mid-run while its FP8
reference stayed slow, which inflates that speedup ratio; read it as
FP4 p50 34.9 ms in the fast regime with a 37.2 ms slow-regime tail
(p95), not as a like-for-like 1.42x.

## Fused GeGLU Epilogue for the P1 Encoder FFN (2026-07-27)

Commits `68e82ae` + `5909b71` add a single-GEMM alternative to the P1
split-GU chain, parameter-isolated behind
`encoder_p1_combiner='epilogue'` (default `lut_native` unchanged). The
gate and up projection rows are pairwise interleaved along N and a
forked SM100 block-scale row-store visitor computes `gelu(gate)*up` on
adjacent accumulator column pairs before scale-factor generation
(`csrc/gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp`), emitting the
FP4+SFD result directly. The separate gate/up fp4out GEMMs and the
GeGLU combiner kernel disappear. The down-projection AWQ `inv_s` is
folded into the up weight rows at quantization time (algebraically
exact), so the epilogue needs no per-column vector.

Full-width stage: the folded value is duplicated into both columns of
each pair, so the down projection consumes N=16384 through a K-expanded
weight with zero odd columns.

Measured verdict on Thor (3v, single-frame kernel attribution via
`nsys --cuda-graph-trace=node` + `--cuda-profile`, regime-immune):

| chain component        | `lut_native` | `epilogue` |
|------------------------|--------------|------------|
| gate+up / interleaved GEMM | 4.89 ms  | 4.91 ms    |
| GeGLU combiner         | 2.22 ms      | —          |
| down (K=8192 / 16384)  | 3.07 ms      | 5.55 ms    |
| P1 FFN chain total     | 10.18 ms     | 10.45 ms   |

The fused epilogue itself is free (the interleaved GEMM matches the two
split GEMMs within 0.5%), and precision improves slightly (single
quantization: formal 3v raw_min 0.99713 vs 0.99708, action_min 0.99928
vs 0.99904; all gates pass at 1v-3v tier parity). But the K-expanded
down projection is DRAM-weight-bandwidth-bound: doubling the streamed
weight costs 1.8x even with the best tile, which cancels the combiner
saving exactly. Formal 3v/2v A/B confirms a wash (3v 34.49 vs 34.31,
2v 28.94 vs 29.08, speedups 1.342/1.360 and 1.327/1.323).

Two tile findings worth keeping:

- Cluster-launch GEMM variants invert between isolated and pipeline
  benchmarks on Thor: in isolation the 2x1-cluster tile was fastest for
  the K=16384 down, but in the pipeline it costs +2.2 ms e2e, and
  2x2/2x4 clusters cost +11-14 ms. The plain 128x256x128 tile
  (`encoder_down_x_variant=6`, now the default) is the only sane
  choice.
- Absolute latencies from different benchmark batches must not be
  compared: the whole-machine regime moved ~2 ms between batches during
  this work while back-to-back in-batch comparisons stayed consistent.

The path is kept as the foundation for a half-width store stage, which
would restore the down projection to its K=8192 cost and turn the
chain into a ~2 ms net win.

## Half-Width Fused GeGLU Epilogue — New Default (2026-07-28)

Commit `ccf35b9` delivers that half-width stage and makes it the default
P1 combiner (`encoder_p1_combiner='epilogue_hw'`; `lut_native` and the
full-width `epilogue` remain selectable). A second store-node variant
quantizes `gelu(gate)*up` at compact granularity — 16 unique values per
scale block, the same block geometry the standalone combiner produced —
and writes the packed FP4 + SFA buffers for the down projection directly
from the visitor. The collective's own D path lands in one small
reusable dummy buffer, and the down projection keeps its original
weight, K extent, and tile variant. Same-batch formal A/B (identical
FP8 references):

| views | `lut_native` | `epilogue_hw` | speedup | raw_min | act_min |
|-------|--------------|---------------|---------|---------|---------|
| 3v    | 34.30 ms     | **32.25 ms**  | 1.3478 → **1.4355** | 0.99708 → 0.99754 | 0.99904 → 0.99947 |
| 2v    | 29.13 ms     | **27.74 ms**  | 1.3207 → **1.3856** | 0.99764 → 0.99794 | 0.99898 → 0.99906 |

All gates pass on both view counts, and both cosine floors improve
(the chain quantizes the FFN hidden state once instead of twice). A
same-batch A/B/A sandwich (34.32 / 32.23 / 34.33 ms) bounds regime
drift at 0.01 ms over the comparison window.

Implementation notes for future epilogue work of this kind:

- The visitor's coordinate tensor is thread-relative: its layout is
  rebuilt without the per-thread iterator offset, so every thread reads
  the same local coordinates. The thread's global base coordinate is
  recovered as the problem extent minus the per-thread residue, which
  is computed from the offset-carrying tensor.
- The SFA tile-atom layout helper takes the quantized axis in the K
  slot of its (M, N, K, L) shape argument.
- Quantize with the hardware e2m1 converter (`NumericArrayConverter`
  over a packed subbyte `Array`). A branch-ladder scalar conversion at
  this volume (7.9M values per GEMM) cost 2.9x kernel time for
  bit-identical output.

## Fused GeGLU FFN in the Decoder — New Default (2026-07-28)

The decoder FFN reuses the compact GeGLU store epilogue on the decoder
GEMM tile (128x64x256): a pairwise-interleaved gate/up weight
(MSE-quantized like the other decoder projections) turns the gate_up
GEMM + GeGLU-quantize pair into one GEMM that writes the
down-projection input directly. NVFP4 weight tier only — the e0m3 and
e0m3+RHT tiers keep the separate kernels. `--decoder-fused-geglu 0`
restores the pair.

Formal same-batch A/B with matched FP8 references:

| views | off | on | speedup | raw_min |
|-------|-----|-----|---------|---------|
| 3v    | 32.26 ms | **31.64 ms** | 1.4417 → **1.4726** | 0.99754 → 0.99766 |
| 2v    | 27.70 ms | **27.14 ms** | 1.3917 → **1.4169** | 0.99794 → 0.99803 |

All gates pass and cosine floors are equal or better (the fused FFN
applies GELU to the fp32 accumulator instead of the fp16-rounded GEMM
output). A single-frame kernel trace confirms the attribution: the
standalone GeGLU-quantize kernel disappears from the decoder's 180
layer-steps and the gate_up GEMM carries the fused epilogue.

`attention_qkv_fp16_seqused_v2` folds the seqused `-inf` mask into the
softmax kernel — the register-to-column mapping and reduction order are
copied from the reference kernel and out-of-range columns take -1e30
instead of the -65504 a separate kernel would have written, so both
exponentiate to zero and the stored probabilities are bit-identical
(`tests/test_pi05_fp4_fusion_kernels.py` asserts exact equality of the
attention output). It saves one launch per attention call, but only the
fixed-shape state-prompt path routes through the seqused kernels, and
this suite does not exercise that path, so the flag ships off
(`--decoder-fused-attn 1` to enable).

A full replacement of the decoder attention chain (QK^T + softmax + AV
+ quantize in one SIMT kernel) was implemented, validated for
numerics, and measured at 5-7x the chain's latency across three
schedule designs: the skinny attention shape leaves the GEMM work
tensor-core-bound (the two cuBLAS calls are ~1 us of tensor-core math
that SIMT arithmetic cannot approach), and per-(head, row) grids
multiply KV cache re-reads past the L2 budget. FlashAttention-4 at
this shape measures 24.6 us (head_dim 256 has no KV-split path).
Fusing the glue between GEMMs — not the GEMMs — is the viable pattern
here.

## SigLIP FFN NVFP4, 16-Layer Preset (2026-07-27)

Commit `1e07ea2` moves the SigLIP vision-tower FFN to NVFP4 on the first
16 of 27 layers: a fused LayerNorm + NVFP4/SFA quantize kernel, an Up
GEMM with fused per-channel bias + tanh-GELU + fp4/SFA output, and a
Down GEMM with fused bias + residual accumulate. The hidden dimension is
zero-padded from 4304 to 4320 for the 32-element fp4 TMA alignment; the
padding carries zero weights and biases. Quantizing deeper layers
(20/23/27 of 27) pushes the worst-sample raw cosine below the 0.995
gate, so the remaining layers keep the FP8 FFN.

Formal strict-harness runs at `1e07ea2` (all gates passed):

| Views | Same-run FP8 p50 | FP4+FP4 p50 / p95 | Speedup |
|---:|---:|---:|---:|
| 2 | 38.616 ms | **30.076 / 31.948 ms** | 1.2839 |
| 3 | 49.401 ms | **37.442 / 37.530 ms** | 1.3194 |

Matched-noise fidelity versus the same-run FP8 reference:

| Views | Raw cosine / worst | Final action cosine / worst |
|---:|---:|---:|
| 2 | 0.999123 / 0.998523 | 0.999735 / 0.999427 |
| 3 | 0.998744 / 0.996937 | 0.999672 / 0.999231 |

The 3-view run executed in the slower sustained-load clock regime (FP8
reference 49.4 ms; see the measurement note below) — the speedup ratio
is the like-for-like metric, and it improves from 1.2925 to 1.3194 over
the previous section's configuration.

Artifacts:

- `flash_rt_fp4`: `5ffd0736eb1e20497cb875a8135db60a113de800c69a1328f9b8ac50633ed9c8`
- `flash_rt_kernels`: `b9173596a459cec26fc9044ba796ab042e40ada89709eec6854dbdee8c486b37`
- 2-view `result.json`: `21df99f986635c6f1fca11aafd2a0d8a08328d672561cdde82e46236120be6db`
- 3-view `result.json`: `dbb1370794f434e2d39254018b80a118c48f04ea7eaf67a48dfba4d4199541ef`

## Vectorized Kernels + Encoder Attention O NVFP4 (2026-07-26)

Three follow-up optimizations on top of the merged FP4+FP4 path, at
commits `e2a3f99` (vectorized QKV split-RoPE and FP4 activation
quantize), `73b5c2f` (vectorized decoder GeGLU quantize), and `1ee2d37`
(NVFP4 encoder attention O projections; QKV stays FP8 because 4-bit Q/K
weights break the raw-cosine gate). The kernel rewrites are bit-exact
with their scalar predecessors; the O-projection quantization is the
only numerical change.

Formal runs with the strict harness defaults (20 warmups, 100 samples,
locked GPC/NVD clocks, separate FP8/FP4 processes) at commit `1ee2d37`:

| Views | Same-run FP8 p50 | FP4+FP4 p50 / p95 | Speedup | Gates |
|---:|---:|---:|---:|---|
| 2 | 38.584 ms | **30.277 / 30.335 ms** | 1.2744 | all pass |
| 3 | 46.372 ms | **35.877 / 35.956 ms** | 1.2925 | all pass |

Matched-noise fidelity versus the same-run FP8 reference:

| Views | Raw cosine / worst | Final action cosine / worst |
|---:|---:|---:|
| 2 | 0.999213 / 0.998429 | 0.999674 / 0.999164 |
| 3 | 0.998692 / 0.996296 | 0.999512 / 0.998731 |

Measurement note: Thor drifts between two sustained-load clock regimes
roughly 3 ms apart even with GPC/NVD locked (the EMC cap is a ceiling,
not a lock), which also moves the FP8 reference — earlier formal runs
recorded FP8 at 41.5/49.5 ms where these runs recorded 38.6/46.4 ms.
Both children of each run above executed entirely in the same regime,
so the speedup ratios and cosine comparisons are like-for-like; compare
absolute milliseconds only against the same-run FP8 column. At commit
`e2a3f99` a mixed-regime formal 3-view run recorded FP4 38.426 ms
against FP8 49.211 ms (speedup 1.2807, all gates passed).

Artifacts for the runs above:

- `flash_rt_fp4`: `5d40c96503938b89cc6d68ae00ef16bde4a0cfa6ecda053f2c04525a94502b1a`
- `flash_rt_kernels`: `b9173596a459cec26fc9044ba796ab042e40ada89709eec6854dbdee8c486b37`
- 2-view `result.json`: `fa87026d5737fe307b84ead943d4f6a6305fa63332ca932c600e379729ddefdd`
- 3-view `result.json`: `1fb0eb90f959ce902431626df9a2f506902a62cc20658fd204a647457768dcb3`

The 1-view fidelity blocker and the LIBERO rollout validation recorded
below remain outstanding and are unaffected by these changes.

This document records the strict end-to-end development of the Pi0.5 NVFP4
path on NVIDIA Thor SM110. The first run isolated the action-expert decoder;
the current candidate combines all 17 live encoder FFN NVFP4 layers with the
decoder NVFP4 path. For 1/2/3 views, p50 must be at least two milliseconds
faster than the published encoder-FP4 + decoder-FP8 results: at most
28.5/34.3/40.8 ms respectively. The earlier explicit 3-view 40 ms target is
also retained.

Latency covers the full `infer()` call: image preprocessing and upload, SigLIP,
encoder, all 18 decoder layers across 10 denoising steps, CUDA Graph replay,
synchronization, action download, and postprocessing.

## Implemented Path

The explicit `use_fp4_decoder=True` path replaces all four decoder projection
GEMMs at the production `M=10` shape:

| Projection | M | N | K | CUTLASS variant |
|---|---:|---:|---:|---:|
| `qkv` | 10 | 2560 | 1024 | v10 |
| `o` | 10 | 1024 | 2048 | v10 |
| `gate_up` | 10 | 8192 | 1024 | v10 |
| `down` | 10 | 1024 | 4096 | v10 |

All four projections use the narrow-N v10 tile (128x64x256): per-kernel
nsys times inside the CUDA-graph pipeline (regime-checked against the
fixed gate_up kernel) measure qkv 10.6 to 10.1 us, o 9.4 to 9.0 us, and
down 14.3 to 13.5 us against the earlier v7 schedules.

Weights are loaded directly from the FP16 safetensors checkpoint, transformed
with the same Q/K head interleave and Gate+Up concatenation as the FP8 loader,
then quantized once to NVFP4 E2M1 plus CUTLASS SFB. There is no FP8-dequantized
weight path.

Runtime activation preprocessing is CUDA-Graph capturable and uses:

- Pi0.5 AdaRMSNorm + gate output + NVFP4/SFA in one launch for QKV.
- Dynamic NVFP4/SFA quantization for the attention output before O.
- Gated residual update + Pi0.5 AdaRMSNorm + gate output + NVFP4/SFA in one
  launch before Gate+Up and the next layer's QKV.
- Existing fused GeGLU + NVFP4/SFA before Down.

The current candidate additionally uses register-only decoder AdaRMSNorm
preprocessing and native SM110 E2M1x2 conversion. The encoder P1 path uses two
FP4-output Gate/Up GEMMs, a gate LUT plus native E2M1x2 combiner, and encoder
Down variant v7. Native FP4 conversion uses round-to-nearest-even, so it is an
explicit numerical mode rather than a bit-exact alias for the historical
midpoint implementation.

The candidate also enables the established 17-layer encoder FFN NVFP4 preset
with AWQ and P1 split-GU. Its AWQ exponent is 0.8. The standard uint8 image hot
path uses a precomputed 256-entry FP16 normalization table and a reused host
buffer. This replaces per-frame uint8-to-FP32-to-FP16 allocations while
producing bit-identical normalized images and bit-identical model outputs.

The production FP8 frontend remains the default. The complete measured tier is
exposed through the public API as:

```python
model = flash_rt.load_model(
    checkpoint, config="pi05", framework="torch", hardware="thor",
    num_views=3, use_fp4=True, use_fp4_decoder=True, use_fa4=True)
```

`use_fp4_decoder=True` resolves the remaining sub-flags to the measured
values — `use_fp4_encoder_attn=True`, `use_fp4_siglip_ffn=True`,
`encoder_p1_combiner="epilogue_hw"`, `awq_alpha=0.8` — and each can still be
overridden explicitly. `use_fp4=True` on its own keeps the earlier
encoder-only preset. `tests/test_pi05_thor_fp4_routing.py` asserts the
resolved constructor arguments against the harness's own preset table, so the
published configuration and the public API cannot drift apart.

The decoder FP4 path
currently supports standard Torch B=1 inference only. CFG, batched inference,
and model-runtime export raise explicit errors when enabled. Unsupported
hardware, shapes, missing kernels, invalid variants, or failed launches also
raise; none select FP8 implicitly.

## Verification Contract

The committed harness is `tests/bench_pi05_decoder_fp4_e2e.py`. It defaults to
`--construct load_model`, which builds both processes through the public
`flash_rt.load_model()` API and refuses to run when any sweep knob deviates
from the published preset; `--construct frontend` instantiates the frontend
classes directly and is required for the exploratory A/B knobs, and results
from that mode are not public-API numbers. The recorded `result.json` carries
the mode in `construction` and the exact call in `children.*.public_api_call`.

### Warmup and latency references

The harness defaults to 300 warmup calls per mode. On Thor, 20 calls can leave
the system warming up during the timed run: a reproduced 2-view run moved from
41.38 / 29.05 ms FP8 / FP4 p50 with 20 warmups to 38.62 / 27.25 ms with 300.
Each child result also records ten ordered group medians in
`latency_group_medians_ms`. A clear step between groups suggests the timing
regime changed while samples were being collected.

The latest README values and the performance gate serve different purposes:

| Views | README reference | Earlier-tier regression baseline | Gate limit |
|---:|---:|---:|---:|
| 1 | 23.01 ms | 30.5 ms | 28.5 ms |
| 2 | 27.17 ms | 36.3 ms | 34.3 ms |
| 3 | 31.74 ms | 42.8 ms | 40.8 ms |

The README column is the current comparison point. The regression baseline is
retained only for the existing acceptance gate, which requires a 2 ms margin
over that earlier tier. Passing the gate therefore does not claim that a run
reproduced the latest README latency. These names replace the ambiguous
`published_sota_p50_ms` fields and gate in earlier artifacts, so new
`result.json` files carry `schema_version: 2`.

Both processes run FA4: the comparison isolates NVFP4 against FP8 with the
attention backend held fixed, so the reported speedup is not an FA4 speedup.

The 2026-08-05 multi-view run used:

- NVIDIA Thor, compute capability 11.0, MAXN.
- GPC min/max/current 1.575 GHz.
- NVD min/max/current 1.692 GHz.
- EMC cap 4.266 GHz.
- Torch 2.10.0 with CUDA 13.0.
- Production graph autotune level 3.
- One, two, or three camera views and the explicit 13-token prompt in the
  harness.
- Eight LIBERO observations with N=8, percentile 99.9 calibration.
- Matched NumPy noise seeds for action comparison.
- Separate FP8 and FP4 processes.
- 20 warmup calls and 100 complete `infer()` samples per mode.

The suite requires a clean tracked worktree and fails unless clocks and device
identity match, all outputs are finite, FP4 is faster than FP8, and the
per-view earlier-tier regression baseline minus 2 ms limit passes. It also
requires 2-view p95 at most 40 ms and 3-view p50 at most 40 ms. Final 7D action
cosine must be at least 0.999 globally and 0.995 for every sample; internal raw
cosine must be at least 0.995 globally and for every sample.

## Encoder Down v7 Multi-View Result (commit `8424808`, 2026-07-26)

The encoder Down GEMM was re-swept with complete `infer()` calls at the actual
per-view encoder shapes. Variant v7 (`tile128x128x256`, cluster `1x1x1`) was
the fastest end-to-end choice. It changes only the CUTLASS tile schedule; the
FP4 inputs, weights, scale layouts, and outputs are unchanged.

Locked-clock results use 20 warmups and 100 retained samples per mode:

| Views | Same-run FP8 p50 / p95 | FP4+FP4 p50 / p95 | Published FP4+FA4 p50 | Gain vs published |
|---:|---:|---:|---:|---:|
| 1 | 35.315 / 35.588 ms | **28.035 / 28.269 ms** | 30.5 ms | **2.465 ms** |
| 2 | 41.485 / 41.652 ms | **32.986 / 33.138 ms** | 36.3 ms | **3.314 ms** |
| 3 | 49.461 / 49.726 ms | **39.393 / 39.743 ms** | 42.8 ms | **3.407 ms** |

The same matched-noise run recorded the following raw and final-action
cosines, measured against the FP16 reference path; the Down variant changes
only the GEMM schedule.

| Views | Raw cosine / worst sample | Final action cosine / worst sample |
|---:|---:|---:|
| 1 | 0.954346 / 0.720291 | 0.870413 / 0.021278 |
| 2 | 0.997625 / 0.995526 | 0.999380 / 0.998825 |
| 3 | 0.997289 / 0.995878 | 0.999062 / 0.996514 |

The variant sweep used complete FP4 child processes with the 1-view fixture,
5 warmups, and 20 retained calls. The screened p50 values were v0 28.352 ms,
v1 28.686 ms, v3 28.791 ms, v4 28.358 ms, v6 28.897 ms, v7 28.022 ms,
v8 28.962 ms, and v10 28.541 ms. The formal multi-view measurements above
then confirmed v7 with the full sampling contract. The tree also retains the decoder MSE
quantization diagnostic used for that precision comparison; it does not add
a runtime fallback.

## 40 ms Production Result

| Metric | Production FP8 | Full NVFP4 | Change |
|---|---:|---:|---:|
| p50 latency | 44.1315 ms | 39.1045 ms | -5.0270 ms |
| p95 latency | 44.4170 ms | 39.2420 ms | -5.1749 ms |
| p50 speedup | 1.0000x | 1.1286x | +12.86% |

Both absolute latency gates passed with 0.8955 ms of p50 headroom and 0.7580 ms
of p95 headroom.

Matched-noise fidelity across all eight observations:

| Metric | Result |
|---|---:|
| Internal raw 32D action cosine | 0.99764686 |
| Worst raw per-sample cosine | 0.99506149 |
| Raw max absolute difference | 0.24609375 |
| Final returned 7D action cosine | 0.99913207 |
| Worst final-action per-sample cosine | 0.99635148 |
| Final-action max absolute difference | 0.16438568 |

The full encoder FP4 preset has a documented raw-output cosine around 0.998.
The final 7D action is the API output consumed by the LIBERO robot, so it keeps
the stricter 0.999 global gate; the full 32D tensor remains a recorded internal
diagnostic. These checks establish matched-input numerical fidelity, not robot
task success rate.

The measured artifacts were:

- `flash_rt_fp4`: `a944449f4a1f763461fb92b6e87d3796c6b6dfbde58e8550eaa62bf15d61a345`
- `flash_rt_kernels`: `c16f817c9ea924b1d88c97e9b510bd61cdbecf3422f483609bb9de8e38b0292b`
- `result.json`: `0bc9e539cd0225d254cff4f674e8befdcd00acae8264af28a336d1ddb66bbcb3`

## Decoder-Isolation Baseline

The earlier run at commit `bc070ae5ae3764d872efced263c401d3c05f91fb`
kept the encoder in FP8 and changed only the decoder. It established the
decoder contribution independently of the production encoder preset:

| Metric | FP8 | Decoder FP4 | Change |
|---|---:|---:|---:|
| p50 latency | 44.7473 ms | 43.1039 ms | -1.6433 ms |
| p95 latency | 44.9634 ms | 43.2874 ms | -1.6760 ms |
| p50 speedup | 1.0000x | 1.0381x | +3.81% |

Its final 7D action cosine was 0.99980575 and raw 32D cosine was 0.99956287.
The result JSON SHA-256 was
`cf64f3e470448881a35dfbcb7219609413633ca55197e07379f929999492fc83`.

The local result files contain all 200 retained latency samples and are not
committed because they include machine-local paths. The reproducible method,
precision configuration, and acceptance thresholds are committed in the
harness.

## FP4+FP4 Multi-View Candidate (2026-07-25)

The formal runs use commit
`8c09371586b70e7a0c53fb79cc017f16100cbeab` and the strict default
configuration in `tests/bench_pi05_decoder_fp4_e2e.py`:

- Encoder layers 0-16: NVFP4 Gate, Up, and Down FFN projections with AWQ
  alpha 0.8 and P1 split-GU. Encoder attention projections remain FP8.
- Decoder layers 0-17 across all 10 denoising steps: NVFP4 QKV, O, Gate+Up,
  and Down projections. No decoder projection selects FP8 implicitly.
- FA4 is active for SigLIP and encoder attention.
- Each view count uses its matching eight-observation fixture, 20 warmups, and
  100 complete `infer()` samples in separate FP8 and FP4 processes.

Locked-clock latency:

| Views | Same-run FP8 p50 / p95 | FP4+FP4 p50 / p95 | Published FP4+FA4 p50 | Published delta | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 35.614 / 35.878 ms | **28.884 / 29.025 ms** | 30.5 ms | -1.616 ms | **fail**, target <=28.5 ms |
| 2 | 41.727 / 41.948 ms | **33.095 / 33.340 ms** | 36.3 ms | **-3.205 ms** | pass |
| 3 | 49.816 / 50.076 ms | **39.821 / 39.985 ms** | 42.8 ms | **-2.979 ms** | pass, including <40 ms |

Matched-noise fidelity across eight observations per view count:

| Views | Raw cosine / worst sample | Raw max abs | Final action cosine / worst sample | Action max abs | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 0.902456 / 0.480090 | 2.005371 | 0.697944 / -0.358415 | 1.887612 | **fail** |
| 2 | 0.997764 / 0.995729 | 0.380005 | 0.999295 / 0.998271 | 0.096136 | pass |
| 3 | 0.998375 / 0.997610 | 0.211670 | 0.999444 / 0.998854 | 0.095402 | pass |

The 1-view failure is reproducible and is not caused by an unstable FP8
reference: an independent FP8 rerun was elementwise identical. A decoder-only
FP4 diagnostic, with every encoder FFN left on FP8, still produced final-action
cosine 0.859115 because one sample's gripper sign changed. Quantizing encoder
layers 0-15 or all 0-16 produced nearly the same failure, so excluding the last
live encoder layer is not a fix. AWQ alpha 0.5 also failed. These diagnostic
configurations are not runtime fallback paths and are not the proposed preset.

This candidate must not merge yet. Two blockers remain:

1. Fix 1-view FP4+FP4 numerical fidelity without relaxing the cosine gates.
   *(Resolved 2026-07-27 — diagnosed as denoising-trajectory bifurcation
   rather than an implementation defect, with a fully passing 1-view
   configuration available; see "One-View Fidelity: Diagnosis and Passing
   Configuration" above.)*
2. Run task-level LIBERO rollouts. The local environments do not currently
   contain the `libero` Python package, so this validation has not been run.

Reproduction command, repeated with `--num-views 1`, `2`, and `3`:

```bash
PYTHONPATH=<repo-root> python tests/bench_pi05_decoder_fp4_e2e.py \
  --num-views 2 \
  --checkpoint <pi05-safetensors-checkpoint-dir> \
  --fixture <dir>/libero_obs_2v_n8.npz \
  --output-dir <output-dir>
```

`--checkpoint` and `--fixture` also read `$PI05_CHECKPOINT` and
`$PI05_FIXTURE_DIR` (the latter holding `libero_obs_<views>v_n8.npz`). The
default `--construct load_model` is what produces the table above.

The FA4 path additionally needs its runtime dependencies on the
interpreter's path (the `thor-fa4` pip extra) and the CUDA runtime
libraries of the active torch install on `LD_LIBRARY_PATH`.

Current shared artifacts:

- `flash_rt_fp4`: `2c66b308661a142765af9cad8ee6a54eff465665829964359d0cada1c4a0ec96`
- `flash_rt_kernels`: `30270002a9646ec230fd69f2cb76ef33acbb5d683872c5833796aa15e10c0c91`
- 1-view `result.json`: `9a5c911dd3a867d7b58abf25bcfeb7201e3a6649baafc7d529a2c0f92bd53267`
- 2-view `result.json`: `43d7c80ca06528a76c183e3e51a018b27eabbb4cb44f38406fd8dacf3b0e4df1`
- 3-view `result.json`: `0780386b4281bf057425a710fcb821dee0dd8cc552d045e3426cf56f38fb6ade`
