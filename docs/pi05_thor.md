# Pi0.5 on Jetson AGX Thor — Usage and Performance

Canonical guide for running Pi0.5 (PaliGemma vision-language encoder +
Gemma action-expert decoder) on Jetson AGX Thor (SM110). It covers the
four supported precision tiers, how to select them, what each one costs
and delivers, and how to reproduce every number.

`pi05_thor_decoder_fp4_e2e.md` is the chronological engineering record
behind these results — individual sections there describe intermediate
states and are superseded by this document.

---

## 1. What runs on the device

| stage | shape | precision (default tier) |
|---|---|---|
| SigLIP vision tower, 27 layers | `num_views × 256` tokens, d=1152 | FFN NVFP4 + AWQ, attention FA4 |
| Encoder (PaliGemma), 18 layers | `num_views × 256 + prompt` tokens, d=2048, H=16384, GQA 8/1, head_dim 256 | 17 live FFNs NVFP4 + AWQ, attention-O NVFP4, QKV FP8, attention FA4 |
| Decoder (Gemma action expert), 18 layers × 10 denoise steps | 10 action tokens, d=1024, H=4096 | all four projections NVFP4, attention cuBLAS FP16 |

One `infer()` call runs the vision tower, the encoder prefill, and ten
denoise steps of the action expert, and returns a `(10, 7)` action chunk.
The whole path is CUDA-graph captured; there is no Python in the hot loop.

---

## 2. Requirements

- Jetson AGX Thor, compute capability `(11, 0)`, MAXN power mode
- CUDA 13, PyTorch ≥ 2.10
- CUTLASS v4.4.2 at `third_party/cutlass` (vendored or symlinked)
- FA4 (`flashrt_fa4.cute`) importable — required for the SigLIP and
  encoder attention path used by every number below

```bash
cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j8          # .so files land in flash_rt/
```

Locked clocks are required for reproducible timing:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
```

---

## 3. Quick start

```python
from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4

pipe = Pi05TorchFrontendThorFP4(
    checkpoint_dir,
    num_views=3,
    use_fa4=True,
    # ---- production NVFP4 tier ----
    use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
    use_awq=True, awq_alpha=0.8, use_p1_split_gu=True,
    use_fp4_encoder_attn=True,      # attention-O projections
    use_fp4_siglip_ffn=True,        # all 27 SigLIP FFNs
    use_fp4_decoder=True,           # all four decoder projections
)

pipe.set_prompt("pick up the black bowl and place it on the plate")
pipe.calibrate(observations, percentile=99.9)   # 8 real observations
actions = pipe.infer(observation)["actions"]    # (10, 7)
```

`observation` is a dict with `image` (`(224, 224, 3)` uint8/float16),
`state`, and — depending on `num_views` — `wrist_image` and
`wrist_image_right`.

The FP8 baseline is the same call on the base class:

```python
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
pipe = Pi05TorchFrontendThor(checkpoint_dir, num_views=3, use_fa4=True)
```

**Calibration is not optional.** The FP4 tiers derive activation scales
and AWQ per-channel weight scales from real observations; calibrating on
synthetic data or skipping it degrades accuracy well past the gates below.

---

## 4. The four precision tiers

All four share the same encoder and SigLIP configuration; they differ in
how the **decoder** stores weights and activations.

| tier | selection | decoder weights | decoder activations |
|---|---|---|---|
| **FP8** | `Pi05TorchFrontendThor` (no FP4 flags) | E4M3, per-tensor static scales | E4M3 |
| **NVFP4** (default) | `use_fp4_decoder=True` | E2M1 + per-16 UE4M3 block scale, per-block MSE scale search | dynamic E2M1 + per-16 scale |
| **INT4** | `decoder_weight_format="e0m3"`, `decoder_act_format="e0m3"` | uniform INT4 (E0M3, ±0..7), scale = amax/7 | uniform INT4 |
| **INT4+RHT** | above plus `decoder_rht=True` | INT4 after a per-16 orthonormal Hadamard rotation | INT4, rotation fused into the quantize kernel |

**NVFP4** uses NVIDIA's block-scaled FP4: a 4-bit E2M1 element with a
UE4M3 scale per 16 elements. Weight scales come from a per-block MSE
search; activation scales are computed at run time in the fused
norm/quantize kernels.

**INT4 (E0M3)** exploits the SM110 tcgen05 block-scaled MMA: the
instruction descriptor's 3-bit element-format field selects
sign-magnitude uniform INT4 at *run time*, so the same tensor-core path
serves both formats with no binary patching. The packed and scale-factor
layouts are identical to NVFP4, so buffers are interchangeable.

**INT4+RHT** rotates every 16-element block by an orthonormal Hadamard
matrix (H16/4, symmetric) — weights offline, activations fused into the
quantize kernel. The rotation is mathematically inert for the GEMM
(verified at kernel level and end to end) but gaussianizes the
per-block distribution, which suits a uniform grid. It buys accuracy at
roughly +0.35 ms.

An unquantized FP16 path also exists (`use_fp8=False`). It is far too
slow for deployment but serves as the common accuracy reference in §5.3,
where FP8 itself is measured against it.

---

## 5. Performance and accuracy

### 5.1 How these numbers were taken

Thor exhibits **day-scale whole-machine drift** (thermal / EMC state):
the same binary measured hours apart can differ by ~2.5 ms, and the FP8
reference moves with it. Therefore:

- absolute milliseconds are only comparable **within one measurement
  batch**;
- **speedup against the same-run FP8 reference is the stable metric**;
- every table below comes from one batch (the FP8 references land within
  0.2 ms of each other at each view count).

Each run is a separate process pair — FP8 child then FP4 child — over
eight real LIBERO observations with matched noise seeds, 20 warmup
iterations and 100 timed iterations, with locked clocks verified from
`/sys` at start. Within-run spread is small (p95 − p50 ≤ 0.09 ms).

### 5.2 Latency and accuracy by tier

Cosines are against the FP8 reference, per sample over the eight
observations. `raw` is the pre-unnormalization action tensor; `act` is
the final action chunk.

**3 views**

| tier | p50 (ms) | speedup | raw cos | raw min | act cos | act min | gates |
|---|---|---|---|---|---|---|---|
| FP8 (reference) | 46.62 | 1.000 | — | — | — | — | — |
| **NVFP4 (default)** | **31.98** | **1.458** | 0.99904 | 0.99766 | 0.99974 | 0.99944 | PASS |
| NVFP4 + row kernels v2 (§8.1) | 30.74 | 1.615 | 0.99801 | 0.99683 | 0.99931 | 0.99828 | PASS |
| + programmatic dependent launch (§8.2) | 29.7 | 1.67 | 0.99816 | 0.99683 | 0.99925 | 0.99828 | PASS |
| **+ operand-swapped decoder GEMMs, early weight stream** (current default, §8.3) | **28.6** | **1.72** | 0.99816 | 0.99683 | 0.99925 | 0.99828 | PASS |
| INT4 | 32.54 | 1.461 | 0.99838 | 0.99512 | 0.99961 | 0.99939 | PASS |
| INT4+RHT | 32.60 | 1.452 | 0.99918 | 0.99742 | 0.99983 | 0.99970 | PASS |

**2 views**

| tier | p50 (ms) | speedup | raw cos | raw min | act cos | act min | gates |
|---|---|---|---|---|---|---|---|
| FP8 (reference) | 38.50 | 1.000 | — | — | — | — | — |
| **NVFP4 (default)** | **27.25** | **1.413** | 0.99921 | 0.99803 | 0.99972 | 0.99916 | PASS |
| INT4 | 27.72 | 1.387 | 0.99879 | 0.99751 | 0.99965 | 0.99928 | PASS |
| INT4+RHT | 27.88 | 1.381 | 0.99941 | 0.99828 | 0.99977 | 0.99922 | PASS |

**1 view** — the accuracy gates do not pass at any fully-quantized tier,
for reasons that are not implementation defects (§6).

| tier | p50 (ms) | speedup | raw cos | raw min | act cos | act min | gates |
|---|---|---|---|---|---|---|---|
| FP8 (reference) | 32.65 | 1.000 | — | — | — | — | — |
| NVFP4 (default) | **22.98** | **1.421** | 0.99137 | 0.96502 | 0.99347 | 0.97112 | accuracy FAIL |
| INT4 | 23.65 | 1.381 | 0.99101 | 0.96112 | 0.99332 | 0.96827 | accuracy FAIL |
| INT4+RHT | 23.75 | 1.375 | 0.99222 | 0.96385 | 0.99406 | 0.97021 | accuracy FAIL |
| FP8 encoder + INT4+RHT decoder | 29.06 | 1.122 | 0.99974 | 0.99954 | 0.99990 | 0.99973 | accuracy PASS |

Gates: `raw cos ≥ 0.995`, worst-sample `raw cos ≥ 0.995`,
`action cos ≥ 0.999`, worst-sample `action cos ≥ 0.995`, plus a latency
gate (3-view p50 ≤ 40 ms, 2-view p95 ≤ 40 ms).

### 5.3 Cosine against a common FP16 reference

The tables above measure each quantized tier against FP8, which leaves
FP8's own error unmeasured. Running the same protocol against the FP16
path (`use_fp8=False`) puts every tier on one yardstick:

| views | tier | raw cos | raw min | act cos | act min |
|---|---|---|---|---|---|
| 3 | FP8 | 0.99994 | 0.99992 | 0.99997 | 0.99995 |
| 3 | NVFP4 | 0.99913 | 0.99812 | 0.99976 | 0.99948 |
| 3 | INT4 | 0.99848 | 0.99518 | 0.99963 | 0.99936 |
| 3 | INT4+RHT | 0.99928 | 0.99742 | 0.99985 | 0.99972 |
| 2 | FP8 | 0.99995 | 0.99994 | 0.99998 | 0.99997 |
| 2 | NVFP4 | 0.99929 | 0.99830 | 0.99976 | 0.99931 |
| 2 | INT4 | 0.99887 | 0.99775 | 0.99969 | 0.99943 |
| 2 | INT4+RHT | 0.99950 | 0.99854 | 0.99982 | 0.99937 |
| 1 | FP8 | 0.99876 | **0.99421** | 0.99905 | 0.99529 |
| 1 | NVFP4 | 0.99187 | 0.96338 | 0.99380 | 0.96939 |
| 1 | INT4 | 0.99118 | 0.95940 | 0.99342 | 0.96644 |
| 1 | INT4+RHT | 0.99204 | 0.96243 | 0.99381 | 0.96856 |

Two things follow.

**FP8 is essentially exact at two and three views** (0.9999+ on every
metric), so using it as the reference in §5.2 costs nothing — those
numbers are within 1e-4 of the same measurement against FP16.

The same harness also times each tier, which places the unquantized path
on the scale. One locked-clock batch at three views:

| tier | p50 (ms) | p95 (ms) | vs FP16 |
|---|---|---|---|
| FP16 | 80.230 | 80.605 | 1.000 |
| FP8 | 47.325 | 49.418 | 1.695 |
| NVFP4 | **31.842** | 31.893 | **2.520** |
| INT4 | 32.646 | 32.718 | 2.458 |
| INT4+RHT | 32.545 | 32.596 | 2.465 |

**At one view even FP8 loses its worst sample**, to 0.99421 — below the
0.995 gate that the quantized tiers also miss. FP8 differs from FP16 by
a very small perturbation, so a sample that moves this much under it is
not being broken by 4-bit quantization; it is sitting somewhere that any
perturbation moves it. See §6.

Reproduce with `tests/bench_pi05_precision_vs_fp16.py` (one subprocess
per tier, same prompt / observations / seeds as the strict suite).

### 5.4 Choosing a tier

- **Default to NVFP4.** It is both the fastest tier and, on 3 views, has
  the best worst-sample raw cosine. The decoder FFN fusion (§8) only
  applies to NVFP4 weights, which is why the INT4 tiers now sit ~0.7 ms
  behind.
- **INT4+RHT when accuracy matters most.** It leads on aggregate cosine
  and on worst-sample action cosine at both view counts, for ~0.9 ms.
- **Plain INT4 has no niche today** — it is slower than NVFP4 and less
  accurate than INT4+RHT. It exists because it is the base the rotation
  is applied to, and because it demonstrates the runtime-descriptor path.
- **FP8** remains the reference for correctness comparisons and for any
  deployment that cannot calibrate on real observations.

---

## 6. One-view accuracy

At one view the per-sample cosine gates fail for every quantized tier.
This is characterized, not open:

- **FP8 fails the same gate at one view** (worst-sample raw cosine 0.99421
  against FP16, §5.3) while being exact to 0.9999 at two and three views.
  FP8 is a far smaller perturbation than 4-bit quantization, so whatever
  moves that sample is not a property of the FP4 kernels.
- Ablations flip **different** samples under different quantization
  configurations (full FP4 flips sample 0; FP8-encoder + FP4-decoder
  flips sample 3 instead).
- The failure mode is a whole-trajectory direction change in the worst
  sample's dominant motion component, not a magnitude error. The gripper
  dimension is exact (cos = 1.0).
- Decoder accuracy improvements move it monotonically: INT4+RHT lifts the
  worst sample from 0.843 to 0.930.

The reading: with one view the observation underdetermines some samples,
which sit near a decision boundary of the flow-matching velocity field,
and any small perturbation — including FP8's — selects the other branch.
Such a sample produces *a different valid action candidate*, which a
per-sample cosine gate cannot distinguish from an error. Task success
rate is the meaningful judge; that evaluation is out of scope here.

A configuration that passes every **accuracy** gate at one view — FP8
encoder with an INT4+RHT decoder — at 29.06 ms (8/8 samples, worst-sample
raw cosine 0.99954, worst-sample action cosine 0.99973):

```bash
--num-views 1 \
--encoder-fp4-layer-count 0 --siglip-ffn-fp4 0 --encoder-attn-o-fp4 0 \
--decoder-weight-format e0m3 --decoder-act-format e0m3 --decoder-rht 1
```

It still trips the suite's published-SOTA latency gate (which wants
≤ 28.5 ms at one view), so `result.json` reports `passed: false` with
every accuracy gate green. Keeping the encoder in FP8 is what buys the
fidelity: it costs 6.1 ms against the fully quantized tier.

---

## 7. Knobs

Constructor keyword / bench flag pairs. Defaults are the production tier.

| knob | default | effect |
|---|---|---|
| `decoder_weight_format` / `--decoder-weight-format` | `nvfp4` | `nvfp4` or `e0m3` |
| `decoder_act_format` / `--decoder-act-format` | `nvfp4` | `e0m3` requires `e0m3` weights |
| `decoder_rht` / `--decoder-rht` | `False` | per-16 Hadamard rotation; requires `e0m3` activations |
| `decoder_fused_geglu` / `--decoder-fused-geglu` | `True` | fuse the decoder GeGLU into the gate_up GEMM epilogue (NVFP4 weights only) |
| `encoder_p1_combiner` / `--encoder-p1-combiner` | `epilogue_hw_nod` | `epilogue_hw_nod` (fused, compact store, collective D store elided), `epilogue_hw` (fused, compact store), `epilogue` (fused, full width — parity with the old path), `lut_native` (separate GEMMs + combiner kernel) |
| `use_fp4_encoder_attn_qkv` / `--encoder-attn-qkv-fp4` | `False` | implemented and passing, but that GEMM is not weight-bandwidth-bound, so FP4 only matches FP8 while costing an extra quantize step |
| `decoder_fused_attn` / `--decoder-fused-attn` | `False` | folds the seqused mask into softmax (bit-identical, one fewer launch). Only the fixed-shape state-prompt path takes the seqused kernels, which this suite does not exercise |
| `rowops_v2` / `--rowops-v2` | `True` | warp-per-row encoder/SigLIP norm + quantize kernels (hardware e2m1/e4m3 conversions, reduction order and tie rounding identical to the originals — bit-exact end to end) |
| `rowops_res_epilogue` / `--rowops-res-epilogue` | `True` | with `rowops_v2`: the encoder O and Down projections accumulate into the residual stream inside the GEMM epilogue (`beta = 1`), so the following norm reads `x` once. Single rounding instead of double; worst-sample raw cosine 0.99683 vs 0.99681 |
| `siglip_up_variant` / `--siglip-up-variant` | `2` | SigLIP Up GEMM tile: 0 = 128x256x256, 1 = 128x128x256, 2 = 128x128x128, 3 = 128x64x256, 4–6 = 2-SM UMMA 256x128x128 / 256x256x128 / 256x128x256 (outputs identical; U4 34.1 µs vs 34.6 for v2) |
| `siglip_down_variant` / `--siglip-down-variant` | `0` | SigLIP Down GEMM tile: 0 = 128x128x256, 1 = 128x64x256, 2 = 128x128x128, 3 = 128x256x256, 4–6 = 2-SM UMMA 256x{128,256,64}x256 (isolated: D4 22.6 µs vs 28.0 for the base) |
| `encoder_attn_o_variant` / `--encoder-attn-o-variant` | `1` | NVFP4 encoder attention-O projection GEMM variant |
| `decoder_qkv_variant`, `decoder_o_variant`, `decoder_down_variant` | `28` | decoder projection GEMM variants (bench flags of the same names). `28` = operand-swapped 2-SM tile with three weight k-tiles streamed before the PDL wait (§8.3); `10` = the previous 128x64x256 tile |
| `decoder_fused_geglu_swap` / `--decoder-fused-geglu-swap` | `False` | operand-swapped fused GeGLU (column compact store, byte-identical output); faster hot but slower in the pipeline (§8.3), off |
| `decoder_rowops_quant` / `--decoder-rowops-quant` | `True` | warp-per-row NVFP4 quantize (row kernels v2) for the decoder attention output; bit-identical, within noise end to end (the launch was already hidden by PDL) |
| `decoder_seq` / `--decoder-seq` | `False` | persistent decoder GEMM sequence: one launch per layer for O, gate_up (GeGLU), down and the next qkv with the AdaRMS phases inside (§8.4); bit-identical, slower than the six launches, off |
| `decoder_attn_mqa` / `--decoder-attn-mqa` | `False` | single-kernel MQA decoder attention with the split merge and NVFP4 quantize fused (`mma.sync`); numerically equivalent, slower than the cuBLAS chain on Thor (§8.3), off |
| `pdl` / `--pdl` | `True` | programmatic dependent launch for the NVFP4 GEMMs and the activation kernels of this module (§8.2) |
| `pdl_fvk` / `--pdl-fvk` | `False` | also PDL-launch the rope/softmax/FP8-quantize kernels and the FP8 encoder GEMM; measured within noise |
| decoder/encoder GEMM variants `15`–`17` | opt-in | mainloop fork that streams the weight tiles before the PDL wait; helps only when a GEMM directly follows a GEMM (72-GEMM chain 0.901 → 0.837 ms/step), within noise in the pipeline |
| decoder GEMM variants `21`–`24` | opt-in | operand-swapped tiles through the stock kernel (`cutlass_fp4_gemm_variants_swap.cu`): 128x64x256, 256x64x256 (2-SM), 128x128x256, 128x64x128 |
| GEMM variants `35`–`38` | opt-in | 2-SM UMMA tiles 256x{128,256}x{128,256} cluster 2x1 for the large-M encoder projections (cold, Se=968: down v36 80 µs vs 105 for v8; attention-O v38 23 µs vs 25 for v1) |
| decoder GEMM variants `25`–`32` | opt-in | operand-swapped tiles through the mainloop fork: all / 2 / 3 weight k-tiles before the PDL wait, dependents triggered from the load warp or the MMA warp (the §8.3 matrix); `28` is the default |
| `awq_alpha` / `--awq-alpha` | `0.8` | AWQ per-channel scale exponent |
| `encoder_down_variant`, `decoder_*_variant` | `8`, `28` | GEMM tile selection |

**Tile selection warning.** Cluster-launch GEMM variants invert between
isolated and in-pipeline benchmarks on Thor: the isolated-best tile for
one projection cost +2.2 ms end to end, and larger clusters +11–14 ms.
The 2-SM UMMA tiles repeat it: encoder down v36 measures 80 µs against
105 in isolation (cold weights, Se=968) and +1.0 ms in the pipeline;
SigLIP down D4 22.6 vs 28.0 µs isolated and within noise end to end.
Always A/B tiles inside the pipeline.

---

## 8. Where the time goes

Single-frame kernel trace, 3 views, NVFP4 tier
(`nsys --cuda-graph-trace=node` around one `cudaProfilerStart/Stop`
window). Frame 31.85 ms under the profiler; 99% is GPU kernel time.

| component | ms/frame | share |
|---|---|---|
| block-scaled FP4 GEMM (incl. fused GeGLU epilogues) | 21.3 | 67% |
| normalization / AdaRMS (incl. 350 decoder AdaRMS calls) | 3.3 | 10% |
| FA4 attention (SigLIP + encoder) | 2.1 | 7% |
| decoder attention cuBLAS chain (QK^T / softmax / AV) | 1.9 | 6% |
| encoder FP8 GEMM (attention QKV / O) | 1.2 | 4% |
| activation quantize | 1.1 | 3% |
| RoPE / QKV split | 0.6 | 2% |
| other | 0.4 | 1% |

Headroom is thin and mostly hard floors:

1. The decoder's 9.7 ms of GEMM sits against a ~7.3–7.8 ms weight-bandwidth
   floor; the gap is fixed overhead across 720 GEMM launches, which would
   need a bespoke persistent mainloop to recover.
2. Decoder AdaRMS (350 × 2.92 µs) and RoPE (180 × 1.58 µs) are at the
   kernel-launch floor.
3. The SigLIP attention projections were characterized with L2-sector
   counters and ruled out: at M=768 they are L2-bandwidth/compute bound
   (32.4 MB of L2 reads per qkv call at ~1.16 TB/s, with the measured
   time bracketed between the weight-only and all-DRAM rooflines), so
   FP4 conversion has no bandwidth dividend and the extra quantize step
   makes it a net loss. AWQ for the SigLIP up-projection remains the one
   accuracy-gated candidate.

### 8.1 Row kernels v2 and the remaining floor

Nsight Compute on the original one-CTA-per-row kernels showed them
instruction-bound (IPC 3.06, ~50 instructions per element in the eight-
compare e2m1 chain), not bandwidth-bound. The v2 kernels
(`csrc/fused_fp4/pi05_rowops_v2.cu`) use one warp per row, 16-byte loads,
shuffle reductions and the hardware `cvt` for e2m1x2 / e4m3x2. Two details
keep them bit-exact against the originals in the real pipeline (random-data
parity is not enough — SigLIP activations flipped the worst LIBERO sample
from 0.99681 to 0.99287 before this was fixed): the fp32 reduction
follows the originals' thread partition and xor-tree order, and exact
e2m1 midpoints (|v| ∈ {0.75, 1.75, 3.5}) are rounded toward zero as the
compare chain does, where the hardware rounds to even.

| kernel (782×2048 / 768×1152, L2-hot) | before | after |
|---|---|---|
| encoder quantize → NVFP4 + SFA | 18.3 µs | 8.3 µs |
| encoder RMSNorm → FP8 | 12.8 µs | 7.2 µs |
| encoder residual + RMSNorm × inv_s → NVFP4 | 33.9 µs | 19.4 µs |
| SigLIP LayerNorm × inv_s → NVFP4 | 22.5 µs | 14.3 µs |
| SigLIP LayerNorm → FP8 | 9.0 µs | 7.2 µs |

Together with the SigLIP Up tile and the encoder Down (v8) / O-projection
(v1) variants, one alternating four-leg A/B measures 31.91 → 31.22 →
31.03 → 30.79 ms; the strict suite through `load_model` lands at 30.74 ms.

Measured device ceilings on this part (locked clocks): device copy
256 GB/s; cuBLAS FP16 ~100 TFLOPS, FP8 ~205 TFLOPS, CUTLASS block-scaled
NVFP4 ~360 TFLOPS — about 40% of the nominal figures. Against those, the
encoder gate_up GEMM (370 TFLOPS) and the FP8 QKV GEMMs are at the
kernel-family ceiling, and the decoder's 720 GEMMs stream 175 MB/step at
185 GB/s (72% of copy). The hard floor for 3 views is therefore ~20 ms
(SigLIP 2.5 + encoder 9.5 + decoder 7.5); what remains above it is the
decoder's per-launch ramp/tail and its ~4 ms of launch-floor kernels,
which only a persistent decoder kernel can remove.

### 8.2 Programmatic dependent launch

With ~2450 kernels per frame, most of them a few microseconds, the launch
ramp of each kernel (CTA scheduling, TMA descriptor prefetch, barrier
init) is a large share of the decoder. The CUTLASS runners are now launched
with `launch_with_pdl` (built with `CUTLASS_ENABLE_GDC_FOR_SM100`) and the
decoder AdaRMS / quantize / row kernels with the programmatic-stream-
serialization attribute; each executes `griddepcontrol.wait` before
touching its inputs and `launch_dependents` right after, so a kernel's
prologue overlaps the previous kernel's tail. CUDA-graph capture keeps
the programmatic edges. Outputs are bit-identical. Three alternating legs
at 3 views: 30.76 → 29.94 ms; the decoder's 72-GEMM chain alone goes
0.949 → 0.902 ms per denoise step. Extending PDL to the rope / softmax /
FP8 kernels measured no further change.

Under the profiler the kernels now overlap (busy time exceeds the frame
span): frame 29.8 ms span, SigLIP 5.0, encoder 11.9, decoder 12.9
(with §8.3: 28.7 span, decoder 12.2).

### 8.3 Operand-swapped decoder GEMMs and the early weight stream

Aliasing all 18 decoder layers onto one layer's weights (everything
L2-hot) only saves 1.9 ms of the decoder's 12.9, so its GEMMs were not
DRAM-bound. The M=10 tile was paying for its A operand: TMA fetches the
full 128-row activation box every k-tile and the 118 out-of-bounds rows
are zero-filled through the same L2/TMA path (Nsight counts 2.1 MB of A
traffic against 1.1 MB of weights for the O projection), so the L2 → SM
path (~1 TB/s measured with plain loads, ~0.45 µs per 128x64x256 k-tile
at 16 CTAs) was the limiter. Computing D^T = W X^T instead makes the
weights the A operand (every streamed row useful) and the activations
the 64-wide B tile; a column-major D is byte-for-byte the row-major
buffer the pipeline already uses and the Sm1xx SFA/SFB layouts coincide,
so no data moves. Outputs are bit-identical. Hot, the four projections
go 41.0 → 19.3 µs (2-SM 256x64x256 tile best).

Cold, a single GEMM is bound by its own ramp, so the mainloop fork now
streams the weight k-tiles before the PDL wait in the swapped
orientation. Two details matter (variants 25–32 sweep): only 2–3 k-tiles
may go ahead of the wait — all seven put 126 KB of weights ahead of the
activation tile in the TMA queue and the kernel gets slower — and
`griddepcontrol.launch_dependents` must come from the load warp as soon
as its loads are issued; issuing it from the MMA warp (the stock
placement) costs 30%. The 72-GEMM decoder chain goes 0.900 → 0.761 ms per
denoise step, 230 GB/s = 94% of the measured 246 GB/s read peak. Two
alternating pipeline rounds at 3 views: 29.45 → 28.56 ms with identical
outputs, which makes variant 28 the default for the qkv, O and down
projections. The fused GeGLU gate_up stays on the 128x64x256 tile: its
swapped form (column compact store, `--decoder-fused-geglu-swap`) is
byte-identical and 20% faster hot but 1.1 ms slower in the pipeline —
with one tile per CTA its cost is waves × per-CTA latency, and the
heavier epilogue lengthens every wave.

### 8.4 Persistent decoder GEMM sequence (opt-in)

`csrc/gemm/fp4/sm100_gemm_seq_persistent_kernel.hpp` runs a list of NVFP4
GEMM problems in one launch on a resident grid (20 CTAs, the operand-swapped
2-SM tile of §8.3, static persistent scheduler): the CTAs synchronise on a
global counter between problems, the load warp streams the next problem's
weight k-tiles before it waits, and the epilogue warps run the decoder's
AdaRMS phase (`sm100_seq_phases.hpp`, a bit-exact port of the row kernel)
between GEMMs. With four independent problems per launch the 72-GEMM
decoder chain reaches 0.721 ms per denoise step = 243 GB/s, 99% of the
measured read peak (0.774 for the separate launches), and a full FFN
half-layer — O, AdaRMS, gate_up with the column GeGLU store, down, AdaRMS,
next qkv — is bit-identical to the pipeline's kernels.

It does not win: with the phases the sequence measures 1.07 ms per step
against 0.92 for the six launches. A phase needs two grid sync points
(all stores visible, then all phase rows visible) and blocks the epilogue
warps meanwhile; the probes attribute ~0.25 ms/step to the second point
and ~0.10 to the blocked epilogue warps, the phase work itself is 0.07.
(Measurement note: a probe that skips a wait leaves the counter dirty and
every later run silently unsynchronised — check the counter after each
timed run.) The second form (`--decoder-seq` today) folds the gated
residual add and per-row sum-of-squares partials into the O/down epilogue
(mode 2 of the column store) and lets every cluster quantize the AdaRMS
rows privately into its own activation slot after a single sync point;
still bit-identical, but 1.41 ms per step: the mode-2 epilogue and the
private phase each cost ~0.2 ms/step of exposed work on the four epilogue
warps, which the weight prefetch across the barrier cannot hide because
the next GEMM's MMAs wait for the phase output rather than for DRAM. Even
the phase-free pair gate_up → down as one launch measures 1.03–1.06
against 0.92. The lesson: on this part the epilogue warps are the serial
resource of a persistent kernel, and the six-kernel pipeline spreads the
same epilogue work over many CTAs in parallel with everything else.

### Approaches measured and rejected

- **Single-kernel decoder attention.** Implemented and numerically
  validated, then measured at 5–7× the existing chain across three
  schedule designs. The skinny attention shape leaves the GEMM work
  tensor-core-bound (the two cuBLAS calls are ~1 µs of tensor-core math),
  and per-(head, row) grids multiply KV re-reads past the L2 budget.
  FlashAttention-4 at this shape measures 24.6 µs (head_dim 256 has no
  KV-split path). Fuse the glue *between* GEMMs, not the GEMMs.
- **Full-width fused GeGLU epilogue.** The combiner kernel it removes is
  exactly cancelled by the doubled weight streaming of the K-expanded
  down projection. The half-width compact store is the form that wins;
  with its unread D store elided outright (`epilogue_hw_nod`, a fork of
  the SM100 epilogue collective with the same `is_destination_supported`
  guards the SM90 and SM100 ptr-array collectives already carry) it is
  the default: a five-leg alternating sandwich measures −0.57 ms
  (32.513/32.550/32.547 vs 31.956/31.967, drift ≤ 0.037 ms). Two
  measurement caveats worth keeping: the isolated kernel benchmark scored
  the elision as a regression (one more entry for the tile-selection
  warning above), and under `nsys --cuda-graph-trace=node` the two
  variants converge entirely — the win only exists unprofiled, so
  per-kernel traces cannot attribute it.
- **Split-KV decoder attention with fused NVFP4 output.** Three designs
  (`csrc/fused_fp4/pi05_dec_attn_splitkv.cu`, opt-in
  `--decoder-attn-splitkv`): 26–37 µs against 10.7 µs for the cuBLAS
  QKᵀ/softmax/PV chain. Nsight: 17% occupancy, 84% of cycles with no
  eligible warp — the cross-CTA merge serialises behind this part's
  ~700-cycle L2 latency. Numerics are right (e2m1 codes 99.6% identical).
- **Stream-K / split-K decoder GEMMs** (variants v11–v14): 18–68 µs cold
  against 10 µs for the static tile; the reduction workspace round trip
  costs more than the extra CTAs gain. Narrower N tiles are ruled out by
  the SM100 block-scaled mainloop (`Cta N` must be 64/128/192/256).
- **L2 prefetch of the next layer's decoder weights**
  (`csrc/fused_fp4/l2_prefetch.cu`, `cp.async.bulk.prefetch.L2` or real
  loads on a side stream): every variant slower than none
  (0.951 → 1.034–1.641 ms per denoise step on the 72-GEMM chain). The chain
  is DRAM-throughput bound as a whole; prefetching only reorders traffic.
- **FA4 tiling.** The SM100 hd256 kernel accepts only tile 128×128 and no
  SplitKV; SigLIP hd72 is already at the best tile.
- **M≤16 NVFP4 GEMM on `mma.sync`** (`csrc/gemm/fp4/nvfp4_m16_gemm_sm110.cu`):
  bit-identical to the CUTLASS runner, but ALU-bound on the dequantisation
  (~45 instructions per k-block per warp) and not faster; kept as the
  reference implementation of the scale-factor layout.
- **No-D-store decoder GeGLU.** The same elision applied to the decoder
  tile is a wash (the dummy store there is 0.04 MB), so it ships opt-in
  (`--decoder-fused-geglu-nod`) and stays off by default.
- **Persistent L2 weight pump** (`csrc/fused_fp4/l2_pump.cu`: resident
  CTAs issuing `cp.async.bulk.prefetch.L2` one layer ahead, paced by a
  progress counter, on a forked graph branch). On Thor the bulk prefetch
  path delivers at most ~160 GB/s (a prefetch followed by a read takes
  longer than the cold read alone; more CTAs only issue faster, the lines
  arrive at the same rate), and a concurrent prefetch stream slows the
  latency-bound hot GEMMs by 27%. The decoder chain went 0.87 → 1.25 ms
  per step — every weight byte read twice.
- **Single-kernel MQA decoder attention on `mma.sync`**
  (`csrc/fused_fp4/attn_mqa_s16_fp4out.cu`, `--decoder-attn-mqa`):
  QKᵀ/softmax/PV per head pair with a 64-key `cp.async` pipeline, split
  merge and the NVFP4 quantize fused. Output bytes match the cuBLAS chain
  plus quantize at 99.7–99.9% (scale factors 100%), but 21.6 µs against
  9.7: on this part `mma.sync` runs at roughly a quarter of the UMMA rate
  the cuBLAS kernels use (1.7 µs per 64-key tile of tensor-core work), and
  the merge/partial traffic adds a fixed ~11 µs. A tcgen05 version would
  be needed to beat the chain.

---

## 9. Reproducing

```bash
# kernel numerical contracts
pytest tests/test_pi05_fp4_fusion_kernels.py tests/test_pi05_decoder_fp4_kernels.py

# every tier against a common FP16 reference (§5.3)
python tests/bench_pi05_precision_vs_fp16.py \
  --checkpoint <CHECKPOINT_DIR> \
  --fixture <FIXTURE_DIR>/libero_obs3v_n8.npz \
  --num-views 3 --output-dir <OUT_DIR>

# strict end-to-end suite (requires a clean tracked worktree)
python tests/bench_pi05_decoder_fp4_e2e.py \
  --checkpoint <CHECKPOINT_DIR> \
  --num-views 3 \
  --fixture <FIXTURE_DIR>/libero_obs3v_n8.npz \
  --output-dir <OUT_DIR>

# tier switches
  --decoder-weight-format e0m3 --decoder-act-format e0m3                 # INT4
  --decoder-weight-format e0m3 --decoder-act-format e0m3 --decoder-rht 1 # INT4+RHT
```

The fixture is an npz of eight real LIBERO observations
(keys `n`, `img_i`, `state_i`, `wrist_i`, `wrist_right_i`). The suite
writes `result.json` with per-iteration timings, the verified clock
state, per-gate verdicts, and `.so` SHA256s, plus the FP4 and FP8 action
tensors.

Measurement discipline:

- compare back-to-back within one batch; across batches use speedup;
- warm up at least 20 iterations before timing;
- one process, exclusive GPU, no concurrent load.

---

## 10. Implementation map

| area | files |
|---|---|
| frontend, tier selection, weight prep | `flash_rt/frontends/torch/pi05_thor_fp4.py` |
| benchmarks | `tests/bench_pi05_decoder_fp4_e2e.py` (strict E2E), `tests/bench_pi05_precision_vs_fp16.py` (common-reference accuracy) |
| decoder pipeline | `flash_rt/models/pi05/pipeline_thor.py` |
| SigLIP / encoder pipeline | `flash_rt/hardware/thor/shared_primitives_fp4.py` |
| attention dispatch (FA4, cuBLAS, seqused) | `flash_rt/hardware/thor/attn_backend.py` |
| NVFP4 / INT4 GEMM runners | `csrc/gemm/fp4/` |
| fused GeGLU store epilogue (row compact store; column compact store for the swapped orientation) | `csrc/gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp`, `csrc/gemm/fp4/cutlass_fp4_gemm_geglu_il_swap_sm100.cu` |
| operand-swapped decoder GEMMs, early-weight mainloop fork, forked kernel layer | `csrc/gemm/fp4/cutlass_fp4_gemm_variants_swap.cu`, `csrc/gemm/fp4/cutlass_fp4_gemm_variants_earlyb.cu`, `csrc/gemm/fp4/sm100_blockscaled_mma_earlyb.hpp`, `csrc/gemm/fp4/sm100_gemm_seq_kernel.hpp` |
| measured-and-rejected kernels kept opt-in | `csrc/fused_fp4/l2_pump.cu`, `csrc/fused_fp4/attn_mqa_s16_fp4out.cu`, `csrc/fused_fp4/pi05_dec_attn_splitkv.cu` |
| persistent decoder GEMM sequence (§8.4) | `csrc/gemm/fp4/sm100_gemm_seq_persistent_kernel.hpp`, `csrc/gemm/fp4/sm100_seq_phases.hpp`, `csrc/gemm/fp4/cutlass_fp4_gemm_seq_sm100.cu` |
| fused norm / quantize / activation kernels | `csrc/fused_fp4/`, `csrc/quantize/` |
| E0M3 quantizer and activation kernels | `csrc/quantize/quantize_e0m3_sfa.cu`, `csrc/fused_fp4/pi05_e0m3_act.cu` |

---

## 11. Known limitations

- One-view accuracy gates fail at every quantized tier (§6); a passing
  configuration exists at reduced speed.
- Task-level (rollout) validation is out of scope for this document.
- The benchmark requires `--checkpoint` and `--fixture` explicitly; there
  are no default dataset paths.
- Numbers here are Thor-specific. The SM110 runtime-descriptor INT4 path
  in particular has no equivalent on other architectures.
