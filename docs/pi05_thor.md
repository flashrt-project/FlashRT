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
| `awq_alpha` / `--awq-alpha` | `0.8` | AWQ per-channel scale exponent |
| `encoder_down_variant`, `decoder_*_variant` | `7`, `10` | GEMM tile selection |

**Tile selection warning.** Cluster-launch GEMM variants invert between
isolated and in-pipeline benchmarks on Thor: the isolated-best tile for
one projection cost +2.2 ms end to end, and larger clusters +11–14 ms.
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
- **No-D-store decoder GeGLU.** The same elision applied to the decoder
  tile is a wash (the dummy store there is 0.04 MB), so it ships opt-in
  (`--decoder-fused-geglu-nod`) and stays off by default.

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
| fused GeGLU store epilogue | `csrc/gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp` |
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
