# FP8 Calibration Mechanics

> **Target audience**: engineers who need to implement `_calibrate` when adding a new model, want to understand whether a precision regression is a calibration issue or a weight issue, or are debugging `alpha = scale * weight_scale` style bugs.
>
> **TL;DR**:
> - FP8 E4M3 only has a dynamic range of ±448, so per-tensor scaling is mandatory to avoid overflow.
> - **Weight scales** are precomputed once during `_load_weights` (via `quant_fp8`) and stored alongside the weight tensors.
> - **Activation scales** are computed during `set_prompt` by running one forward pass and measuring the amax at each GEMM input, then cached to disk.
> - `alpha = act_scale * weight_scale` is the CUTLASS FP8 GEMM descale multiplier and **must be multiplied in f32** — not f64.
> - Calibration cache location: `~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json`.

---

## 1. Why calibration is necessary

The FP8 E4M3 format has 1 sign bit + 4 exponent bits + 3 mantissa bits. Its largest representable magnitude is **±448** and its smallest normal positive value is roughly **2^-6** — about five decimal digits of dynamic range.

FP16/BF16 weights and activations are typically distributed over roughly [-5, 5]. Casting them directly to FP8 causes two failure modes:
- Large values (> 448) saturate and lose precision.
- Small values (< 2^-9) truncate to zero and distort the sparsity pattern.

Every FP8 tensor therefore needs an accompanying **per-tensor scale** (float32):

```
fp8_value = clip(fp32_value / scale, -448, 448)
fp32_restored = fp8_value * scale
```

FlashRT's policy:
- **Weights** — static per-tensor scale, computed once at load time and never updated.
- **Activations** — static per-GEMM-input scale, computed once during calibration and held constant at runtime.

The activation scale can only be obtained by running a full forward pass and recording the amax at each GEMM input. That is exactly what `_calibrate` does.

**How this compares to vLLM and similar frameworks**: vLLM defaults to eager BF16 with no FP8 quantization. When AWQ or GPTQ is enabled, the quantization is **weight-only** (activations stay in FP16). FlashRT runs **W8A8**, which requires calibrating both sides.

---

## 2. Two scales, three stages

### 2.1 Weight scales (`*_w_scales`)

Computed when loading the checkpoint, inside [`_load_weights`](../flash_rt/frontends/torch/pi05_thor.py) via the `Quant()` transform in WEIGHT_SPEC:

```python
# flash_rt/core/thor_frontend_utils.py
def quant_fp8(w):
    w = w.contiguous()
    a = w.float().abs().max().item()
    s = max(a / 448.0, 1e-12)
    return (w.float() / s).clamp(-448, 448).to(_FP8), s
```

Every quantized weight tensor has its own scale float. The `scale_into="_enc_w_scales"` directive in the spec gathers these scales into a python list in spec order; the frontend then wraps it as a device tensor:

```python
self._enc_w_dev = torch.tensor(self._enc_w_scales, dtype=torch.float32, device='cuda')
```

**Storage**: bound to the weight tensor itself, persisted across restarts with the checkpoint, **never updated dynamically**.

### 2.2 Activation scales (`*_calib_scales`)

Computed during `set_prompt` — on the first call, or whenever `recalibrate=True` is passed.

The distribution at each GEMM input depends on:
- The statistics of every preceding layer's output (which are a function of the model weights).
- The input sequence length `Se` (prompt tokens + SigLIP patches).
- The prompt content itself — different text produces different distributions, but **the difference is small enough to ignore**. Verified empirically that calibrating with zero inputs still yields cos > 0.998 on real tasks.

`_calibrate` therefore uses a fixed **zero input** plus a representative prompt of length Se, then runs `encoder_forward_calibrate` followed by `decoder_forward_calibrate`:

```python
# insert an amax reduction at each GEMM input
fvk.rms_norm_fp16(x, ones_buf, norm_scratch, Se, D, 1e-6, stream)
_measure_scale_gpu(fvk, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
# d_scale now equals max(|norm_scratch|) / 448 with clamp
# written to calib_buf[l*4 + 0]; the next GEMM reads it directly
fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, D, d_scale, stream)
```

Reference implementations: [`shared_primitives.py::encoder_forward_calibrate`](../flash_rt/hardware/thor/shared_primitives.py) (lines 491-610) and [`decoder_forward_calibrate`](../flash_rt/hardware/thor/shared_primitives.py) (from line 610 on).

Each Pi0.5 / Pi0 encoder layer has 4 FP8 GEMMs (QKV / O / Gate+Up / Down), so `enc_calib_scales.shape == (L * 4,) == (72,)`. The decoder is identical: `(La * 4,) == (72,)`.

### 2.3 `alpha = act_scale * weight_scale`

A CUTLASS FP8 GEMM by itself produces `fp8_a × fp8_b = fp32` without any scaling. To recover the true value, the accumulator must be multiplied by the descale factor:

```
output = fp32_accumulator × act_scale × weight_scale
       = fp32_accumulator × alpha
```

`alpha` is a per-GEMM f32 scalar. It **must be computed on the host in f32** (not f64), because the kernel that Myelin emits performs the multiplication in f32 — a mismatch introduces last-bit differences:

```python
# Historical bug: computing float(...) * float(...) (implicit f64) produced
# different bits than the production C code, regressing cos to 0.9878.
self._enc_alpha_host = [
    float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
    for i in range(Le * 4)
]
```

Always use **`np.float32(a) * np.float32(b)`** — never plain `a * b`.

---

## 3. The calibration cache

### 3.1 Cache key

`~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json`, where:
- `ckpt_hash` = SHA256(first 64 KB + file_size), first 16 hex chars (see [`calibrator.py::_checkpoint_hash`](../flash_rt/core/quant/calibrator.py)).
- `Se` = encoder input sequence length (a function of num_views and prompt length).

**Why both parts are needed**:
- Different fine-tunes of the same base → different activation distributions → different scales.
- Same checkpoint at a different `Se` → different GEMM shapes → potentially different optimal scales (the impact is small but nonzero).

### 3.2 Cache format

```json
{
  "version": 1,
  "ckpt_hash": "a1b2c3d4e5f67890",
  "Se": 776,
  "num_enc_scales": 72,
  "num_ae_scales": 72,
  "enc_scales":    [x0, x1, ...],  // activation scales, len = L * 4
  "enc_alpha":     [x0, x1, ...],  // = enc_scales * enc_w_scales, len = L * 4
  "enc_w_scales":  [x0, x1, ...],  // weight scales (for alpha recomputation)
  "ae_scales":     [x0, x1, ...]   // decoder activation scales
}
```

**Why this redundancy**: `enc_w_scales` is also stored in the cache so that if the weight quantizer is ever upgraded, `alpha` can be recomputed from the cache alone without rerunning the forward pass. The current production path does not exercise this, but the option is there.

### 3.3 Invalidation

Automatic invalidation conditions in [`load_calibration`](../flash_rt/core/quant/calibrator.py):
- `version` mismatch → recalibrate.
- `ckpt_hash` mismatch (user swapped the checkpoint) → recalibrate.
- `Se` mismatch (prompt length changed) → recalibrate.

Manual invalidation:
```bash
rm ~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json
# or wipe everything:
rm -rf ~/.flash_rt/calibration/
```

`load_model(..., recalibrate=True)` also forces a rerun.

---

<a name="precision-history"></a>
## 4. Precision history — Pi0.5 from 0.9992 to 1.0000 bit-identical

Every step in this progression was caused by a calibration bug. Keep the list as a **reference checklist**: if your new model shows cos < 0.995, check these four regressions first.

### 4.1 Weight-scale ordering mismatch (0.999 → 0.933)

**Symptom**: cos collapsed overnight from 0.999 to 0.933 even though neither the weights nor the activation calibration had changed.

**Root cause**: `_override_weight_scale` indexed scales by a running counter, but the item order inside WEIGHT_SPEC drifted. The counter got out of sync, so layer L ended up using the weight scale belonging to layer L+N.

**Fix**: replace the counter with a **fingerprint lookup** that matches scales to tensors by identity.

**How to avoid in a new model**: the `scale_into` order in WEIGHT_SPEC **must match the order the legacy C++ loader expects**. Walk the old loader and verify every Quant site's scale index.

### 4.2 Per-step vs per-GEMM aggregation (Myelin crash)

**Symptom**: Pi0.5 flow matching takes 10 steps, each with a slightly different activation distribution. Calibrating each step independently produced 10 distinct scale sets, which gave Myelin inconsistent KV-cache shapes at compile time and crashed it.

**Fix**: aggregate the max **across all steps inside `_calibrate`**. One calibration forward already covers every step because the decoder is iterative.

**How to avoid in a new model**: if your model is flow-matching or autoregressive-decode, **do not** compute per-step scales inside `_calibrate`. Use the final max from a single forward.

### 4.3 `np.float32 * np.float32` instead of implicit f64 (0.9878 → 0.9992)

See §2.3 — force f32 multiplication.

### 4.4 30-layer weight-scale fallback (0.9878 → 0.9992)

**Symptom**: some layers had normal cos, others regressed.

**Root cause**: the old `_override_weight_scale` fell back to the legacy path for 30 special layers; only the remainder went through the fingerprint lookup. Those 30 fallback layers computed their scales incorrectly.

**Fix**: precompute inside `_override_weight_scale` and route every layer through fingerprint lookup uniformly.

**How to avoid in a new model**: do not branch. Either every layer goes through the new scale path or every layer stays on legacy — never mix.

---

## 5. Runtime recalibration (`_recalibrate_with_real_data`)

The scales built on the first `infer` call are derived from **zero inputs plus a placeholder prompt**. A real task's distribution may differ. Pi0.5 exposes an optional `_recalibrate_with_real_data` method for this case:

```python
# Pi05TorchFrontendThor.infer()
if not self._real_data_calibrated and <first real infer>:
    self._recalibrate_with_real_data()
    self._real_data_calibrated = True
```

What it does: take the **first real input** (real image + real prompt) and rerun `encoder_forward_calibrate` + `decoder_forward_calibrate`, overwriting the cached scales.

Measured impact: Pi0.5 LIBERO Spatial success rate went from **91.8% to 98.2%** thanks to this mechanism. Any task with significant domain shift from zero inputs should turn it on.

**Should your new model do this?** If the target task's distribution is far from a zero tensor (for example mobile manipulation), strongly recommended. The simplest path is to copy Pi0.5's `_recalibrate_with_real_data` verbatim and rename the bufs/weights/dims.

---

## 6. Common pitfalls

### 6.1 "cos is only 0.001 off — scale bug or tactic noise?"

- Tactic noise: spread is usually ≤ 0.002 and **fluctuates up and down** across runs — not monotonic.
- Scale bug: typically **consistently low**, hovering around the same value on every run.

Run 3-5 A/B trials to disambiguate.

### 6.2 `_gpu_sync` and `_d2h_float` inside the calibrate loop

The calibrate function contains many `_gpu_sync(stream); as_o = _d2h_float(d_scale)` calls. These are **required**: the next GEMM's alpha depends on the host-side value of `as_o`. **Do not** remove these syncs in favor of async variants.

This is also why `_calibrate` is roughly 10x slower than `_capture_*_graph` (3-4s vs 0.3s) — four D2H transfers per layer.

### 6.3 Is `enc_alpha_host` a python list or a device tensor?

- `enc_alpha_host`: python `list[float]` — used as a scalar argument to each GEMM in forward (`cutlass_fp8_sq(... alpha=ENC_ALPHA_HOST[idx] ...)`).
- `enc_calib_scales`: `torch.tensor(..., device='cuda')` — passed to `quantize_fp8_static_fp16` as the d_scale pointer.

You need both. Do not pack alpha into a device tensor — the kernel expects a host scalar parameter.

### 6.4 Do not call `jax.block_until_ready` prematurely on the JAX side

`CudaBufferFlat.finalize` already contains a single `jax.block_until_ready(flat)`, which is sufficient. Adding another `block_until_ready` inside the calibration loop breaks the async pipeline and doubles the first `set_prompt` time.

---

## 7. Calibration checklist for a new model

- [ ] Every `Quant()` item in WEIGHT_SPEC has a `scale_into` name ordered to match what the C++ pipeline expects.
- [ ] `_load_weights` finishes by wrapping the scale list as a device tensor (`self._enc_w_dev = torch.tensor(self._enc_w_scales, ...)`).
- [ ] The `enc_bufs` / `enc_weights` / `enc_dims` dictionaries in `_calibrate` use **exactly the keys that `shared_primitives.encoder_forward_calibrate` expects** — a missing key is a segfault.
- [ ] `_enc_alpha_host` is computed with **`np.float32 * np.float32`**.
- [ ] Call `load_calibration(ckpt, Se)` first; only run the calibration forward on miss. The first cold start may take 3-4s, but a second start must be under 0.5s.
- [ ] For flow-matching or iterative-decode models: a **single** `decoder_forward_calibrate` call must cover every step — do not calibrate per step.
- [ ] In `set_prompt`, set `self._enc_calib_scales` / `self._ae_calib_scales` as device tensors before graph capture.
- [ ] Optional: implement `_recalibrate_with_real_data` and trigger it on the first `infer` call.

---

## 8. Code-location quick reference

| Content | File |
|------|------|
| `quant_fp8` weight quantization function | [`flash_rt/core/thor_frontend_utils.py`](../flash_rt/core/thor_frontend_utils.py) |
| Calibration-cache read/write | [`flash_rt/core/quant/calibrator.py`](../flash_rt/core/quant/calibrator.py) |
| `encoder_forward_calibrate` / `decoder_forward_calibrate` | [`flash_rt/hardware/thor/shared_primitives.py`](../flash_rt/hardware/thor/shared_primitives.py) |
| Pi0.5 `_calibrate` example | [`flash_rt/frontends/torch/pi05_thor.py`](../flash_rt/frontends/torch/pi05_thor.py) (L499-649) |
| `_recalibrate_with_real_data` | [`flash_rt/frontends/torch/pi05_thor.py`](../flash_rt/frontends/torch/pi05_thor.py) (L954) |
| GROOT split calibrate (Qwen3 + DiT separate) | [`flash_rt/frontends/torch/groot_thor.py`](../flash_rt/frontends/torch/groot_thor.py), [`flash_rt/frontends/torch/groot_rtx.py`](../flash_rt/frontends/torch/groot_rtx.py) |

---

**Related documentation**:
- [`docs/adding_new_model.md`](adding_new_model.md) — overall guide to adding a new model.
- [`docs/kernel_fusion.md`](kernel_fusion.md) — which kernels accept a d_scale argument and where the fusion boundaries are.
- [`docs/stable_api.md`](stable_api.md) — public contract for the `scale_into` parameter inside WEIGHT_SPEC.
- [`docs/precision_spec.md`](precision_spec.md) — declarative description of the quantization produced by calibration.

---

## 10. Multi-sample (dataset) calibration

> **TL;DR — empirically validated on Pi0.5 LIBERO-FT vs PyTorch FP32
> reference. Both platforms (RTX 5090 SM120 datacentre GPU, Jetson AGX
> Thor SM110 edge SoC) support the same ``calibrate(obs_list,
> percentile=...)`` API and stratified LIBERO sampling:**
>
> | Platform | Strategy | cos vs FP32 ref | maxdiff vs ref | calibrate (one-off) |
> |---|---|---|---|---|
> | RTX 5090  | single-frame            | 0.9996 | 0.031 | 0.21 s |
> | RTX 5090  | **stratified N = 8**    | **0.9998** | **0.020** | 0.69 s |
> | RTX 5090  | stratified N = 64       | 0.9998 | 0.020 | 2.21 s |
> | Thor SM110| single-frame            | 0.9989 | 0.046 | 0.35 s |
> | Thor SM110| stratified N = 8        | 0.9992 | 0.043 | 1.24 s |
> | Thor SM110| stratified N = 16       | 0.9994 | 0.036 | 2.20 s |
> | Thor SM110| **stratified N = 64**   | **0.9997** | **0.025** | 8.22 s |
> | Thor SM110| stratified N = 256      | 0.9997 | 0.023 | 32.10 s |
>
> - **Fine-tuned-for-domain models on RTX**: use **stratified N = 8**
>   — halves maxdiff, costs ~0.7 s one-off. ``N > 8`` plateaus.
> - **Fine-tuned-for-domain models on Thor**: use **stratified N = 64**
>   — cos improvement keeps scaling past N = 8 on SM110 (see §10.5
>   "Thor scaling is not the same as RTX"). Costs ~8 s one-off.
> - **Base / not-fine-tuned models**: keep single-frame at an
>   operating-point frame — multi-frame's wider scale can reduce
>   per-frame precision (see §10.8 "model type matters").
> - **Match the openpi-jax-mlir toolchain** on RTX for parity: N = 8,
>   stratified by episode × frame position, ``percentile = 99.9``.
> - **Graph-replay P50 is unchanged** on both platforms (RTX: Pi0.5
>   ~21 ms, Thor Pi0.5 ~45 ms 2v / ~55 ms 3v). Multi-sample
>   calibration only changes the scale *values* — the captured CUDA
>   Graph structure and replay cost are identical.
>
> Full data, methodology, and the precision-vs-coverage trade-off
> below.

---

The sections above describe single-frame calibration, which is what the
framework does by default. For real hardware deployment — where the
runtime distribution of activations spans lighting, occlusion, and
pose variation that a single frame does not cover — FlashRT also
supports **multi-frame dataset calibration with percentile clipping**.

### Motivation

A single frame produces an almost-correct amax. The failure modes are:

1. Runtime frames are *more extreme* than the calibration frame →
   activations saturate at ±448 (FP8 E4M3 max), accuracy cliff.
2. The calibration frame is an outlier → scales are inflated and normal
   runtime frames lose precision.

Adding more i.i.d. samples does not help against (1) — the only fix is
to *cover* the deployment distribution. Against (2), a percentile
reduction (not the true max) clips away outlier frames.

### API

```python
rt = load_model("pi05", "torch", "rtx_sm120")
rt.set_prompt(prompt)

# Default (back-compatible): single frame, percentile ignored
rt.calibrate(observations=[obs])

# Dataset mode
rt.calibrate(
    observations=my_frames,    # list or Iterable[dict]
    percentile=99.9,           # default; 100.0 == traditional max
    max_samples=256,
    verbose=True,              # print per-point dispersion summary
)
```

`rt.calibrate_with_real_data([obs])` is retained as a thin alias for
backward compatibility.

### Choosing `N` and `percentile`

| Scenario | N | percentile |
|---|---|---|
| Demo / single environment | 1 | n/a |
| Single-task real hardware | 16–64 | 99.9 |
| Multi-task / mixed scene | 128–256 | 99.9 |
| Outdoor / all-weather | 512–1024 | 99.9 |
| Training-set-derived samples with known outliers | 256+ | 99.0 or 95.0 |

### Pi0.5 (LIBERO-FT) measured against FP32 reference — stratified N≥8 wins

Earlier versions of this document compared multi-frame outputs only to
the single-frame output ("self"), which measures calibration drift but
not absolute quality. The proper measurement is **cosine vs. the
PyTorch FP32 reference model** running on the same inputs.

#### RTX 5090 (SM120)

Pi0.5 on RTX 5090, deterministic image pair (same input the official
`rtx_cosine_vs_official.py` uses), **all N ≥ 8 use stratified sampling
(episode × frame-position)** per the openpi-jax-mlir toolchain:

| Strategy | cos vs **FP32 ref** | cos vs self | maxdiff vs ref |
|---|---|---|---|
| self (N=1, target frame) | 0.999639 | 1.000000 | 0.0312 |
| other_N1 (N=1, different frame) | 0.999575 | 0.999437 | 0.0360 |
| **stratified N=8** | **0.999772** | 0.999865 | **0.0204** |
| stratified N=16 | 0.999583 | 0.999827 | 0.0344 |
| stratified N=32 | 0.999760 | 0.999836 | 0.0204 |
| **stratified N=64** ⭐ best | **0.999831** | 0.999738 | **0.0204** |
| stratified N=128 | 0.999731 | 0.999851 | 0.0379 |
| stratified N=256 | 0.999785 | 0.999544 | 0.0221 |

**Three things this table shows on RTX:**

1. **Single-frame is measurably worse.** Both single-frame rows
   (`self` 0.999639, `other_N1` 0.999575) trail every stratified
   configuration with N ≥ 8 by ~0.00015–0.00025 in absolute cosine,
   and ~50 % higher maxdiff. Single-frame's amax can under-estimate
   the distribution envelope seen at runtime.
2. **Stratified N = 8 is the sweet spot** (cos 0.9998, maxdiff 0.0204)
   — matching the openpi-jax-mlir toolchain's default and its comment
   "8 is sufficient".
3. **Going past N = 8 plateaus.** N = 64 is the numerical best (0.9998
   vs 0.9998) but the delta over N = 8 is within run-to-run noise.
   Stratification quality matters more than raw count.

The N = 16 row is an outlier because the stratified-sample picker's
exact frame positions happened to miss the high-amax regions that
N = 8 captured. This is a sampling artifact, not a trend.

#### Jetson AGX Thor (SM110)

Pi0.5 3-view on Thor SM110 against the Pi0.5 PyTorch FP32 reference
(`pytorch_reference.npz` shipped inside the the PyTorch deploy container
container). Measured by
[`tests/bench_thor_calibration_vs_ref.py`](../tests/bench_thor_calibration_vs_ref.py),
stratified samples drawn from LIBERO-10 (379 episodes / 101 k frames)
via [`flash_rt.datasets.libero`](../flash_rt/datasets/libero.py):

| Strategy | cos vs **FP32 ref** | maxdiff vs ref | calibrate time |
|---|---|---|---|
| N=1 (ref frame) | 0.998938 | 0.0459 | 0.35 s |
| stratified N=8 | 0.999165 | 0.0433 | 1.24 s |
| stratified N=16 | 0.999436 | 0.0355 | 2.20 s |
| stratified N=64 ⭐ | **0.999685** | **0.0250** | 8.22 s |
| stratified N=256 | 0.999742 | 0.0232 | 32.10 s |

Replay latency P50 is a flat 55.49 ms ± 0.02 ms across all rows — the
multi-sample path only changes scale *values*, not the captured CUDA
Graph structure. Graph replay is identical in cost to the single-frame
path.

**How Thor differs from RTX on the same test:**

1. **Thor's N=1 baseline sits ~0.0007 below RTX's.** SM110's FP8
   E4M3 kernels pick different CUTLASS tactics than SM120 — bit
   layouts and rounding edges differ slightly — so the single-frame
   starting point is a notch lower. Absolute quality is still well
   above the 0.998 deployment threshold.
2. **Thor keeps scaling past N = 8, unlike RTX.** Each doubling of
   N continues to deliver meaningful cos gain on Thor (N=8 → 16 =
   +0.00027; 16 → 64 = +0.00025; 64 → 256 = +0.00006). On RTX,
   past N = 8 is within run-to-run noise. This is because Thor's
   lower starting point leaves more room for the wider
   percentile-reduced envelope to help.
3. **Thor per-sample cost ~128 ms asymptotic** (≈ 4 × RTX's ~33 ms).
   Thor SM110 has fewer SMs and different memory bandwidth; each
   calibrate forward pass runs ~4× slower.

### Why "cos vs self" is not a substitute for "cos vs FP32 ref"

The previous measurement (before FP32 reference was wired up) reported
"cos > 0.9999 across all N" — that was measuring **the distance between
two FP8 outputs** (different calibrations of the same FlashRT pipeline)
which is a calibration-drift signal. It cannot identify which strategy
is *closer to the true model*, only how much strategies differ from
each other. The dual-cosine table above is the calibration-choice
signal that matters for deployment quality.

### Model type matters — Pi0.5 (LIBERO-FT) vs Pi0 (base)

Pi0 without LIBERO fine-tuning has a different trade-off because its
activations are broader. On Pi0 same-task dataset calibration measured
**against self-calibrated output** (not FP32 ref):

| Model | Task 8: cos(N=256, self) | Task 9: cos(N=256, self) |
|---|---|---|
| Pi0 (base) | 0.9961 | 0.9325 |
| Pi0.5 (LIBERO-FT) | 0.9999+ | 0.9999+ |

Pi0's wide activation distribution means stratified calibration
broadens the FP8 scale enough to reduce per-frame precision; Pi0.5's
tightly-distributed activations absorb the broader scale with no
measurable loss. **This is why Pi0.5 is the canonical deployment
target for measuring calibration quality on LIBERO** — and why the
recommendations below are keyed on "fine-tuned for domain" rather
than on model name.

### The precision vs. coverage trade-off — when **not** to use multi-frame

Multi-frame calibration is not unconditionally "better" on non-
fine-tuned models. It trades per-frame precision for distribution
coverage:

- **Single-frame** scale = `amax(this_frame) / 448`. Narrower scale
  means each FP8 value represents a tighter range, so typical
  activations are quantized with higher precision. If a runtime frame
  steps outside this range, its activations saturate at ±448 and
  precision collapses.
- **Multi-frame** scale = `percentile(per-frame amax) / 448`. Broader
  scale avoids saturation on any of the sampled frames, but every
  value — including typical ones — is represented with coarser steps.

Empirical demonstration on Pi0 RTX (LIBERO, percentile=99.9) measured
by cosine vs. self-calibration (the optimal FP8 output for each target):

| Target task | self | same-task N=16 | same-task N=64 | same-task N=256 |
|---|---|---|---|---|
| Task 8 (5047 frames, tight distribution) | 1.000 | 0.997 | 0.988 | 0.996 |
| Task 9 (3373 frames, wider variance)     | 1.000 | 0.965 | 0.931 | 0.932 |

Task 8's activations are tightly distributed, so multi-frame's wider
scale barely reduces precision. Task 9 has more episode-to-episode
variance in activation magnitudes, and the target frame's typical
activations do not fill the wider scale — multi-frame drops cosine
noticeably.

**Practical guidance**:

| Deployment scenario | Model is fine-tuned for domain? | Recommended |
|---|---|---|
| Fixed environment (demo, single lab setup) | yes | Single-frame or N=8 stratified — both fine |
| Fixed environment | no (base model) | Single-frame at an operating-point frame |
| Production deployment | yes | **N=8–32 stratified by episode / time-of-day**, percentile=99.9 |
| Production deployment | no | Keep N=1 at operating point; consider re-fine-tuning first |
| Runtime distribution drifts (lighting / outdoor / scene switches) | yes | N=64–256 sampling across the drift range |
| Runtime contains sensor outliers | either | N ≥ 256 with percentile ≤ 99.0 |

### Sampling strategy — stratification matters more than N

When assembling a multi-sample calibration set, diversity of
observations beats raw count. The openpi-jax-mlir production
toolchain's stratification pattern:

1. Select `min(num_episodes, max(N//2, 3))` distinct episodes uniformly
2. Within each selected episode, take `frame_step = len(ep)//N`
   equally-spaced frames
3. Cap total to `N`

This captures different task phases (reach / grasp / transport /
release) and different object positions within the same task, which
produces tighter activation-magnitude coverage than uniform-random
sampling over frames.

A naive `np.random.choice(all_frames, N)` can over-sample mid-episode
steady-state frames and under-sample the open/close-phase activations
where amax is higher.

The rule of thumb: multi-frame pays for itself when runtime frames
sometimes *exceed* the amax of any single representative calibration
frame. If all runtime frames are expected to stay within the amax
envelope of one well-chosen calibration frame, prefer that single
frame.

### Percentile vs. N — how clipping actually works

`percentile=99.9` is mathematically only meaningful when `N` is large
enough that "drop the top 0.1 %" can skip a whole sample. With
`N ≤ 1/(1-p) ≈ 10` for `p=0.9`, or `N ≤ 1000` for `p=0.999`, the
percentile interpolates between the last two ranked values and the
rejection effect is cosmetic. Rules of thumb:

| N | Effective clipping threshold for one outlier-per-set |
|---|---|
| 16  | percentile ≤ 93 (skip 1 of 16) |
| 64  | percentile ≤ 98.5 |
| 256 | percentile ≤ 99.6 |
| 1024 | percentile ≤ 99.9 |

Measured on a synthetic outlier-injection experiment with Pi0 RTX,
N=16, single adversarial frame: `percentile=99.9` gave essentially the
same cosine as `percentile=100` (0.985 vs 0.989 — within noise). The
clipping defaults are tuned for large-N deployments; if you are running
N ≤ 64 and expect outliers, drop the percentile accordingly (95 for
N=16, 98 for N=64).

### Measured calibration time (real LIBERO frames, percentile=99.9)

#### RTX 5090 (SM120)

| Model | N=1 | N=16 | N=64 | N=256 |
|---|---|---|---|---|
| Pi0 RTX (2v)   | 0.27 s | 0.77 s | 2.39 s | 8.84 s |
| Pi0.5 RTX (2v) | 0.21 s | 0.69 s | 2.21 s | 8.27 s |

Per-frame cost asymptotes to ~33 ms (~1.6× a single infer) as N grows —
the uncalibrated FP8 GEMM path is slightly slower than the captured-graph
replay path.

#### Jetson AGX Thor (SM110)

| Model | N=1 | N=8 | N=16 | N=64 | N=256 |
|---|---|---|---|---|---|
| Pi0.5 Thor (3v) | 0.35 s | 1.24 s | 2.20 s | 8.22 s | 32.10 s |

Per-frame cost asymptotes to ~128 ms (~2.3× a single 55 ms replay).
Measured with the same LIBERO dataset and `percentile=99.9` as the RTX
row, on Pi0.5 LIBERO-FT safetensors in the PyTorch deploy container.
See [`tests/bench_thor_calibration_vs_ref.py`](../tests/bench_thor_calibration_vs_ref.py).

Calibration runs once per deployment session on both platforms. A Thor
fleet running `N = 64` pays an 8-second start-up cost to gain 0.0008
absolute cosine and halve maxdiff vs single-frame — a trade worth
taking for production deployments, but skippable for a 50-shot demo on
a fixed scene where `N = 1` is already above the accuracy-sufficient
threshold.

#### GROOT N1.7 Thor aux-list validation

GROOT N1.7 Thor uses aux samples rather than LIBERO obs dicts for
multi-sample calibration. After `set_prompt(aux=...)` selects the
deployment prompt, `GrootN17TorchFrontendThor.calibrate(aux_list,
percentile=99.9)` percentile-reduces FP8 backbone activation scales
across pre-captured aux samples. The focused Thor regression is
[`tests/test_thor_groot_n17_calibrate.py`](../tests/test_thor_groot_n17_calibrate.py)
with `--ns 1,8`.

#### Which Thor frontends implement multi-sample calibration

Thor `Pi05TorchFrontendThor.calibrate(obs_list, percentile=99.9)`
supports N ≥ 2 today (same shape as the RTX API — see
[`flash_rt/frontends/torch/pi05_thor.py`](../flash_rt/frontends/torch/pi05_thor.py)
`_calibrate_multi_frame`). The torch Pi0 and Pi0-FAST Thor frontends,
plus the non-FP4 JAX Thor frontends, follow the same obs-list contract.
GROOT N1.7 Thor supports N ≥ 2 through its aux-list API described
above. On RTX the same multi-sample API is also available for
`groot_rtx` (see
[`flash_rt/frontends/torch/groot_rtx.py`](../flash_rt/frontends/torch/groot_rtx.py)
`_calibrate_multi_frame`), which percentile-reduces both the Qwen3 and
the DiT activation scales after running per-sample shadow forwards
through both stages. Motus RTX also supports
`MotusTorchFrontendRtx.calibrate(obs_list, percentile=99.9)` for
Stage3 bundles; it percentile-reduces Motus FP8 GEMM scales, AWQ-FP8
action/und scales, G7.24 action/und QKV scales, and VAE FP8 resample
scales before CUDA Graph capture. For the older GROOT N1.6 Thor path
(`groot_thor`), run `tests/test_thor_groot_calibrate.py --ns 1,8`
with N1.6 assets before using it as deployment evidence.

### What remains unchanged

The multi-sample path writes the same per-tensor amax buffer, produces
the same `calibrate_fp8()` scale computation, triggers the same
`autotune_gemms()` and `record_infer_graph()` steps. The captured
CUDA Graph is identical in structure to the single-frame path — only
the *values* of the scales differ. Infer-time latency is therefore
unchanged (validated in tests: replay p50 difference ≤ ±0.05 ms).

### Introspection

After calibration, `rt.precision_spec` exposes a `ModelPrecisionSpec`
with one `PrecisionSpec` per quantized tensor, recording the scale, the
method (`"single_frame"` / `"percentile"`), the N and percentile used.
See [`precision_spec.md`](precision_spec.md).
