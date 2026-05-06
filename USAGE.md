# FlashRT Usage Guide

Complete reference for installation, API parameters, mechanisms, and usage patterns.

---

## Installation

TL;DR — see [README.md § Build & install](README.md#build--install)
for the full walkthrough (Docker and non-Docker paths). The short
version:

```bash
git clone https://github.com/LiangSu8899/FlashRT.git
cd FlashRT
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

pip install -e ".[torch]"     # or "[jax]" / "[all]"

# Build — produces flash_rt_kernels.so AND (on RTX) flash_rt_fa2.so.
# Thor builds produce only the former (FA2 skipped, Thor uses fvk
# cuBLAS-decomposed attention).
mkdir build && cd build
cmake ..                       # auto-detects GPU arch from nvidia-smi
make -j$(nproc)
cp flash_rt_kernels*.so flash_rt_fa2*.so ../flash_rt/ 2>/dev/null || \
   cp flash_rt_kernels*.so ../flash_rt/     # Thor path
cd ..
```

**Crucially — no `pip install flash-attn` required.** FlashRT
vendors FA2 v2.7.4.post1 (fp16 + bf16) at source level and builds
it into `flash_rt/flash_rt_fa2.so`. Zero pip `flash-attn` wheel
dependency at runtime.

After installation, `import flash_rt` works from any directory.

---

## Pi0 / Pi0.5 prerequisite — PaliGemma tokenizer

> **TL;DR — required one-time download for Pi0 / Pi0.5 only.**
> Qwen3-8B and other text-only models do NOT need this.

Pi0 and Pi0.5 prefix prompts via the **PaliGemma SentencePiece
tokenizer** (`paligemma_tokenizer.model`, ~4.1 MiB). This file is
**not bundled inside the openpi pi0 / pi05 checkpoints** — it lives
in Google's public big_vision storage bucket, and you have to
fetch it once before running any Pi0 / Pi0.5 inference.

### One-shot download (recommended)

```bash
bash scripts/download_paligemma_tokenizer.sh
# → ~/.cache/flash_rt/paligemma_tokenizer.model
```

The script downloads from the public GCS HTTP endpoint
(`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`),
verifies the MD5, and is idempotent (re-runs are no-ops if the file
is already valid).

### Custom location

```bash
# Set this env var to point at any local copy:
export FLASH_RT_PALIGEMMA_TOKENIZER=/data/checkpoints/paligemma_tokenizer.model
```

The frontend's resolution order is:

1. `$FLASH_RT_PALIGEMMA_TOKENIZER` (explicit override)
2. `~/.cache/flash_rt/paligemma_tokenizer.model`
3. `~/.cache/openpi/big_vision/paligemma_tokenizer.model`
   (compatible with `openpi.shared.download.maybe_download`)
4. `/workspace/paligemma_tokenizer.model` (legacy container path)
5. openpi auto-download via `gs://big_vision/...`
   (only if `gcsfs` is installed)

If none resolves, the loader raises a `FileNotFoundError` whose
message contains the exact `curl` command above — no silent
failure modes, no segfaults.

### Q&A

**Q: I downloaded the openpi Pi0 / Pi0.5 checkpoint. Don't I already have this?**
A: No. openpi ships only `model.safetensors` + `assets/` + configs;
the tokenizer is a separate fetch by design (one tokenizer file,
many checkpoints).

**Q: Can I use the HuggingFace `google/paligemma-3b-pt-224` mirror instead?**
A: That repo contains the same `tokenizer.model` file, but PaliGemma
is a gated HF model (you must accept the license + supply an HF
token). The GCS public bucket is auth-free and a one-line `curl` —
strictly easier for fresh setups. The frontend accepts a tokenizer
loaded from either source as long as you point
`$FLASH_RT_PALIGEMMA_TOKENIZER` at it.

**Q: I'm getting `FileNotFoundError: paligemma_tokenizer.model not found`.**
A: Run the download script above (or `curl` directly into one of
the search paths). The error message is itself the fix — it spells
out the exact command.

**Q: Old setups had `/workspace/paligemma_tokenizer.model` —
do I need to migrate?**
A: No. The legacy path is still in the search list.

---

## Quick Start

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/checkpoint",
    framework="torch",
)

actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
)
# actions: numpy (10, 7) — 10 future steps, 7 DOF
```

---

## API Reference

### `flash_rt.load_model()`

```python
model = flash_rt.load_model(
    checkpoint,                # str: path to checkpoint directory
    framework="torch",         # "torch" or "jax"
    num_views=2,               # number of camera views (2 or 3)
    autotune=3,                # CUDA Graph autotune trials
    recalibrate=False,         # force fresh FP8 calibration
    config="pi05",             # model config
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint` | `str` | required | Path to checkpoint directory. Safetensors for `torch`, Orbax for `jax`. |
| `framework` | `str` | `"torch"` | `"torch"` (safetensors) or `"jax"` (Orbax). Both use the same C++/CUDA kernels. |
| `num_views` | `int` | `2` | Number of camera views. LIBERO uses 2 (base + wrist). |
| `autotune` | `int\|bool` | `3` | CUDA Graph autotune intensity. See [Autotune](#autotune). |
| `recalibrate` | `bool` | `False` | Force fresh FP8 calibration (and weight cache for JAX), ignoring cache. See [Calibration](#calibration). |
| `weight_cache` | `bool` | `True` | Cache FP8-quantized weights to disk. **JAX only** — reduces cold start from ~42s to ~6s. Torch loads in ~3s and ignores this. See [Weight Cache](#weight-cache-jax-only). |
| `config` | `str` | `"pi05"` | Model architecture config: `"pi05"`, `"pi0"`, `"groot"`, `"pi0fast"`. |
| `decode_cuda_graph` | `bool` | `False` | **Pi0-FAST only.** Capture action-phase decode as CUDA Graph. Trades startup time for per-token speed. See [Pi0-FAST](#pi0-fast). |
| `decode_graph_steps` | `int` | `80` | **Pi0-FAST only.** Number of action tokens to capture in the decode graph. Should cover your longest expected action sequence. |
| `use_fp4` | `bool` | `False` | **Pi0.5 torch + jax on Thor.** Enable NVFP4 quantization on the encoder FFN stack. When `True`, resolves to the production preset (`fp4_layers=tuple(range(18))` + `use_awq=True` + `use_p1_split_gu=True`). Requires SM100+ GPU. Other configs emit a warning and fall back to FP8. See [NVFP4](#nvfp4-pi05-only). |
| `fp4_layers` | `tuple[int]` \| `None` | `None` | Encoder layer indices to FP4-quantize. `None` → resolved by the `use_fp4` preset. Passing an explicit tuple overrides the preset. Only `(7,8,9)` and `range(18)`+AWQ are task-level validated. |
| `use_awq` | `bool` \| `None` | `None` | Activation-aware weight quant. Required for 18-layer FP4 scope (without it, cos collapses to ~0.33). `None` → resolved by the `use_fp4` preset. |
| `awq_alpha` | `float` | `0.5` | AWQ scaling exponent `s[k] = (a[k]/a.mean())^alpha`. |
| `use_p1_split_gu` | `bool` \| `None` | `None` | P1 split-GU 2-GEMM path (parity on Pi0.5, kernel reusable for Pi0.6). `None` → resolved by the `use_fp4` preset. |

### `model.predict()`

```python
actions = model.predict(
    images=[base_img, wrist_img],   # list of (224,224,3) uint8/float16 numpy
    prompt="pick up the red block", # text prompt (required on first call)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `list` or `dict` | required | Camera images. List: `[base, wrist]` or `[base, wrist, wrist_right]`. Dict: `{"image": ..., "wrist_image": ...}`. |
| `prompt` | `str\|None` | `None` | Task prompt. Required on first call. Omit to reuse previous prompt (no recalibration, no graph recapture). |

**Returns:** `np.ndarray` shape `(10, 7)` — 10 future action steps, 7 DOF.

**Prompt reuse:** When `prompt` is the same as the previous call, `predict()` skips `set_prompt()` entirely — only graph replay happens (~44ms). Changing the prompt triggers recalibration + graph recapture (~4s).

### `model.recalibrate()`

```python
model.recalibrate()
# Next predict() will force fresh calibration
actions = model.predict(images=..., prompt="...")
```

Clears the calibration cache for this checkpoint and forces recalibration on the next `predict()` call. Use after:
- Fine-tuning the model (activation distributions changed)
- Switching deployment domains (different image statistics)
- Debugging precision issues

---

## Pi0-FAST

Pi0-FAST is an **autoregressive** model — actions are generated as discrete
FAST tokens (Gemma 2B, 18 layers), not via diffusion. Total latency =
`prefill + N × per_token_decode` where N is typically 30–80 tokens.

### Performance Modes

| Mode | `set_prompt` (cold) | `set_prompt` (cached) | Per-token | 50-tok E2E |
|------|--------------------:|----------------------:|----------:|-----------:|
| **Default** | ~2.5 s | **~0.1 s** | 9.5 ms | ~480 ms |
| **Max-perf** (`decode_cuda_graph=True`) | ~4.0 s | **~1.5 s** | **8.3 ms** | **~447 ms** |

```python
# Default: good for interactive / multi-prompt scenarios
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_fast_base",
    config="pi0fast",
)
actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
    state=robot_state,
)

# Max-performance: best for fixed-prompt 24h continuous control
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_fast_base",
    config="pi0fast",
    decode_cuda_graph=True,       # capture decode loop as CUDA Graph
    decode_graph_steps=80,        # covers sequences up to 80 action tokens
)
```

**Default mode**: Each decode token runs through a Python loop with per-step
kernel launches. Lowest startup cost — `set_prompt` loads calibration from
cache in ~0.1s on subsequent runs.

**Max-performance mode**: The action-phase decode loop is captured as a single
CUDA Graph (same technique as Pi0's diffusion loop). All autoregressive token
dependencies (argmax → embedding lookup → next step) run entirely on GPU with
zero host-device synchronization per step.

- Adds ~1.5s to `set_prompt` for decode graph capture (on top of the cached
  calibration load)
- Saves ~1.2 ms/token (–13% per-token latency)
- Break-even at ~40 inferences per prompt
- For long sequences (80 tokens), saves ~96 ms per inference

### `set_prompt` Timing Breakdown

| Component | Cold (first ever) | Cached (same ckpt + Se) |
|-----------|------------------:|------------------------:|
| SigLIP graph capture | ~0.2 s | ~0.2 s |
| FP8 calibration (16 samples) | **~2.4 s** | **0 s** (loaded from cache) |
| Prefill graph capture | ~0.1 s | ~0.1 s |
| Decode graph capture (`decode_cuda_graph=True`) | ~1.5 s | ~1.5 s |
| **Total (default)** | **~2.5 s** | **~0.1 s** |
| **Total (max-perf)** | **~4.0 s** | **~1.5 s** |

The 2.4s calibration is the largest cold-start cost. It runs 16 forward
passes with noise perturbations to find robust FP8 activation scales, then
saves to `~/.flash_rt/calibration/`. On subsequent runs with the same
checkpoint and sequence length, calibration loads from cache instantly.

### Calibration Cache

Pi0-FAST follows the same calibration cache mechanism as Pi0/Pi0.5:

1. **`set_prompt()`**: Check cache → hit: load scales (0s) / miss: run 16-sample
   calibration (~2.4s) and save to cache
2. **First `predict()`**: Always recalibrates with real image data (regardless
   of cache) for optimal FP8 scales matching actual deployment images.
   The result is saved back to cache.
3. **Subsequent `predict()`**: Pure inference, no recalibration

**Same-scene optimization**: When running the same scene repeatedly (same
prompt, same checkpoint), the cached calibration from step 2 above is reused.
No need to force recalibration — the system handles it automatically.

**Force recalibration**:
```bash
rm -rf ~/.flash_rt/calibration/   # clear all cached scales
```

### Checkpoint Conversion

The Torch frontend requires safetensors format. Convert from Orbax:
```bash
python examples/convert_pi0fast_orbax_to_safetensors.py \
    --orbax_dir /path/to/pi0_fast_base \
    --output_dir /path/to/pi0_fast_base_converted
```
The JAX frontend reads Orbax directly.

---

## Autotune

CUDA Graph instantiation on Thor is non-deterministic — the same kernels can produce different execution schedules with ~2ms latency variance. Autotune recaptures the graph multiple times and keeps the fastest schedule.

### How it works

1. Capture CUDA Graph (SigLIP + Encoder + Decoder)
2. Benchmark the graph with CUDA events
3. If latency is within the fast regime (< 38.5ms for Enc+AE), accept
4. Otherwise, recapture and retry
5. After `autotune` trials, use whatever was captured last

### Parameter values

| `autotune=` | Behavior | Extra startup time | When to use |
|-------------|----------|-------------------|-------------|
| `0` or `False` | Single capture, no retry | 0 | Debugging, fastest startup |
| `3` (default) | Up to 3 retries | ~1s | Production (Torch almost always finds fast graph on trial 0) |
| `5` | Up to 5 retries | ~2.5s | JAX or when you need guaranteed best latency |
| `True` | Same as `3` | ~1s | Shorthand |

### Framework differences

- **Torch**: Almost always gets the fast graph on trial 0 (43-44ms total)
- **JAX**: XLA GPU memory state sometimes prevents fast graph. `jax.clear_caches()` is called before capture to help, but 5+ trials may be needed

### Autotune runs once per prompt

Autotune is part of `set_prompt()`. If you call `predict()` with the same prompt, the cached graph is replayed — no autotune overhead.

---

## Calibration

FP8 inference requires calibrated activation scales — per-layer maximum values that determine the FP8 quantization range. Incorrect scales cause precision loss.

### How it works

**Phase 1: Initial calibration** (during `set_prompt()`):
1. Check disk cache: `~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json`
2. **Cache hit**: Load scales from JSON (instant)
3. **Cache miss**: Run `encoder_forward_calibrate()` + `decoder_forward_calibrate()` with warmup data (~3-4s), save to cache

**Phase 2: Real-data recalibration** (on first `predict()` call):
1. After SigLIP processes the first real image, the encoder input (`enc_x`) contains realistic activation distributions
2. Rerun calibration with this real data for more accurate scales
3. Recapture CUDA Graph with updated scales
4. This happens only once — subsequent `predict()` calls skip this step

### Cache details

| Property | Value |
|----------|-------|
| Location | `~/.flash_rt/calibration/` |
| Key | `SHA256(checkpoint_first_64KB + file_size)[:16]` + `_Se{sequence_length}` |
| Format | JSON with `enc_scales`, `ae_scales`, `enc_alpha`, `enc_w_scales` |
| Invalidation | Automatic per-checkpoint hash. Different checkpoints or finetunes get different caches. |

### Cache isolation

- **Multi-model safe**: Each checkpoint produces a unique hash → separate cache files
- **Multi-framework safe**: Torch (safetensors) and JAX (Orbax) hash different files → separate caches
- **Multi-prompt safe**: Different prompt lengths produce different `Se` → separate caches
- **Fine-tune safe**: Modified checkpoint → different first 64KB → different hash → fresh calibration

### Forcing recalibration

Three ways to force fresh calibration:

```python
# Method 1: At load time
model = flash_rt.load_model(checkpoint, recalibrate=True)

# Method 2: At runtime
model.recalibrate()
model.predict(images=..., prompt="...")

# Method 3: CLI
python examples/quickstart.py --checkpoint /path/to/ckpt --recalibrate
```

Or manually delete the cache:
```bash
rm -rf ~/.flash_rt/calibration/
```

### Explicit multi-sample calibration — `model.calibrate()`

`model.calibrate()` lets you seed the FP8 activation scales with a list
of real observations instead of the default single-frame / zero-input
path. Use it when runtime images cover a wider distribution (different
lighting, object poses, camera angles) than any one calibration frame
could represent.

The API is the same everywhere — one call, any number of observations:

```python
model.calibrate(observations)
```

Everything else has a sensible default. You do **not** need to change
any other setting: the cache, graph capture, prompt, and `predict()`
all continue to behave as before.

#### I have a dataset — how do I call calibrate?

The common case: you have a robot-rollout dataset, and you want to
calibrate on N representative frames before deployment. Three lines:

```python
from flash_rt.datasets.libero import load_calibration_obs

# Pick 8 stratified frames (episode × frame-position) from a LIBERO-
# format dataset. Returns list[dict] — each dict has 'image',
# 'wrist_image', 'state' ready to hand straight to calibrate().
obs_list = load_calibration_obs("/path/to/libero_dataset", n=8)

model.calibrate(obs_list)
```

`load_calibration_obs` expects the LeRobot-v2 LIBERO layout
(`meta/info.json` + `data/chunk-NNN/episode_NNNNNN.parquet`). If your
dataset has a different layout, either:

1. Load your observations yourself and pass the list:

```python
obs_list = [
    {"image": ..., "wrist_image": ..., "state": ...},
    {"image": ..., "wrist_image": ..., "state": ...},
    # ... any N >= 1
]
model.calibrate(obs_list)
```

2. Or build a pandas DataFrame with
   `(task_index, episode_index, frame_index, index)` columns and use
   the lower-level helper:

```python
from flash_rt.core.calibration import stratified_sample

obs_list = stratified_sample(my_dataframe, my_load_fn, n=8)
model.calibrate(obs_list)
```

#### N = 1 — the default, keeps using the disk cache

If you pass a single observation (or skip `calibrate()` entirely — the
first `predict()` call triggers the same path), FlashRT uses the
legacy calibration pipeline:

```python
model.calibrate([obs])        # explicit N=1
# or simply:
model.predict(obs)            # first call auto-calibrates
```

The first run computes scales from one forward pass and writes them
to `~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json`. Every subsequent
process with the same checkpoint reads the cache — no forward pass
required.

#### N >= 2 — multi-sample dataset calibration

Same API, just pass more observations:

```python
model.calibrate(obs_list)                     # default percentile
model.calibrate(obs_list, percentile=99.9)    # explicit (same default)
```

Under the hood: each observation runs one calibration forward pass,
per-layer activation maxima are reduced by taking the 99.9 th
percentile across samples (so one outlier frame cannot inflate every
scale), and the reduced scales are written to the device before the
graph is recaptured.

Cache is still used: N >= 2 writes the same JSON format as N = 1, so
a second process run with the same checkpoint picks up the reduced
scales for free.

#### Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `observations` | `list[dict]` | required | Obs dicts matching the `predict()` contract (`image`, `wrist_image`, optional `state`). N = 1 → single-frame path; N >= 2 → multi-sample percentile path. |
| `percentile` | `float` | `99.9` | Percentile applied to per-sample per-tensor amax. `100.0` == traditional max. Lower (e.g. `95`) clips more aggressively — useful when your dataset has known outliers. |
| `max_samples` | `int \| None` | `None` | Upper bound, lets you pass an iterator without materializing the whole list. |
| `verbose` | `bool` | `False` | Log a one-line dispersion summary (median amax, outlier cutback, etc.). |

#### Which frontends support it

| Frontend | N = 1 | N >= 2 (dataset) |
|---|---|---|
| `pi05_thor` (torch) | ✅ | ✅ |
| `pi0_thor` (torch) | ✅ | ✅ |
| `pi0fast` (torch) | ✅ | ✅ |
| `pi05_thor_fp4` (torch, FP4 encoder active) | ✅ | ✅ (two-phase: FP8 + AWQ refit) |
| `pi05_thor_fp4` (jax, FP4 encoder active) | ✅ | ✅ (two-phase: FP8 + AWQ refit) |
| `pi0_rtx`, `pi05_rtx` (torch + jax) | ✅ | ✅ |
| `groot_rtx` (torch) | ✅ | ✅ |
| `groot_thor` (torch) | ✅ | ❌ (see note) |
| `pi05_thor` / `pi0_thor` / `pi0fast` (jax, non-FP4) | ✅ | ❌ (see note) |

Frontends marked ❌ raise `NotImplementedError` on N >= 2 — pass N = 1
there today. Reasons:

- **`groot_thor`**: the Thor port of the multi-sample path is staged
  for the next rollout; the N=1 calibrate path remains the default
  there. RTX (`groot_rtx`) ships the full N>=2 path today.
- Non-FP4 JAX Thor frontends (`pi05_thor`, `pi0_thor`, `pi0fast`): the
  FP8-only JAX path still uses the N=1 implicit-recalibrate shim; the
  JAX Pi0.5 FP4 frontend (`pi05_thor_fp4` with `framework="jax"`) does
  support N>=2, using the same two-phase flow as torch.

`pi05_thor_fp4` uses a two-phase multi-sample flow: Phase 1 reduces
FP8 activation scales across N samples (same loop the base class
uses), then Phase 2 collects AWQ per-channel activation amax with
the new FP8 scales active, percentile-reduces across samples, and
refits the NVFP4 packed weights + AWQ inv_s buffers in place — no
second graph capture. Roughly 2× the wall-clock of non-FP4 N samples
because both phases walk the dataset.

#### Choosing N and percentile

| Scenario | Recommended |
|---|---|
| Fixed environment, fine-tuned model | `N = 8`, `percentile = 99.9` |
| Fixed environment, base / non-fine-tuned model | `N = 1` at a representative frame |
| Runtime drifts (lighting / outdoor / scene switches) | `N = 64–256`, `percentile = 99.9` |
| Input contains sensor outliers | `N >= 256`, `percentile <= 99.0` |

Measured cosine and calibration-time numbers, plus the
precision-vs-coverage trade-off, live in
[docs/calibration.md §10](docs/calibration.md). The short version: a
LIBERO-fine-tuned Pi0.5 gains cos ≈ +0.0003 and halves max per-channel
deviation going from N = 1 to N = 8, at the cost of a few seconds of
one-off calibration.

#### Low-level helper

If your dataset cannot go through `load_calibration_obs`, the
lower-level helper takes a pandas DataFrame + a per-index loader:

```python
flash_rt.core.calibration.stratified_sample(
    metadata,        # pandas.DataFrame with {index, task_index,
                     #   episode_index, frame_index} columns
    load_fn,         # callable: index -> obs dict
    n=8,
    *,
    task_filter=None,        # narrow to one task_index
    exclude=None,            # skip specific global indices
) -> list[dict]
```

It picks N global frame indices stratified across episode × frame
position (not uniform-random, which would over-sample steady-state
frames) and applies `load_fn` to each.

#### Diagnostic: outlier-scale warning

After calibration, FlashRT scans the produced per-layer FP8 scales
and logs a `WARNING` via
`flash_rt.core.calibration.check_scale_ceiling` if any scale is more
than **20 ×** the median of the same calibration. This catches the
case where a single outlier frame in the calibration set stretched the
FP8 scale on one layer far beyond its peers — the typical sign that
the dataset contains a glitched / overexposed / occluded sample.
---

## Full Parameter Examples

```python
import flash_rt

# === Production deployment (recommended) ===
model = flash_rt.load_model(
    checkpoint="/path/to/pi05_libero_pytorch",
    framework="torch",
    autotune=3,          # stable 44ms
)
actions = model.predict(images=[img, wrist], prompt="pick up the red block")


# === JAX with Orbax checkpoint ===
model = flash_rt.load_model(
    checkpoint="/path/to/orbax_checkpoint",
    framework="jax",
    autotune=5,          # JAX may need more trials
)


# === After fine-tuning: force recalibration ===
model = flash_rt.load_model(
    checkpoint="/path/to/finetuned_checkpoint",
    framework="torch",
    recalibrate=True,    # ignore old cache
)


# === Fast iteration during development ===
model = flash_rt.load_model(
    checkpoint="/path/to/checkpoint",
    framework="torch",
    autotune=0,          # skip autotune for fastest startup
)


# === 3-camera setup ===
model = flash_rt.load_model(
    checkpoint="/path/to/checkpoint",
    framework="torch",
    num_views=3,
)
actions = model.predict(
    images=[base_img, wrist_left, wrist_right],
    prompt="pick up the cup",
)


# === Runtime recalibration (domain shift) ===
model = flash_rt.load_model(checkpoint="/path/to/checkpoint")
actions = model.predict(images=[img1, img2], prompt="task A")

# ... deployment domain changed ...
model.recalibrate()
actions = model.predict(images=[img3, img4], prompt="task B")  # fresh calibration
```

---

## Weight Cache (JAX only)

JAX (Orbax) checkpoint loading takes ~42s due to OCDBT deserialization + weight transform + FP8 quantization. The weight cache saves the final FP8-quantized engine weights to disk after first load, so subsequent loads skip all three steps.

### Why JAX only?

| Framework | Cold start | Bottleneck |
|-----------|-----------|------------|
| **Torch** (safetensors) | ~3s | mmap load — already fast |
| **JAX** (Orbax) | ~42s → **~6s with cache** | OCDBT deserialize + transform + FP8 quant |

Torch uses safetensors which is essentially a flat binary mmap — there's nothing to cache. JAX's Orbax format requires complex deserialization that the weight cache eliminates.

### How it works

1. **First load**: Orbax → transform → FP8 quantize → upload to GPU → **save binary cache** (`~/.flash_rt/weights/{hash}_nv{N}.bin`, ~4 GB)
2. **Subsequent loads**: Read binary cache → upload to GPU directly (~6s)

### Parameters

```python
# Default: cache enabled (recommended)
model = flash_rt.load_model(checkpoint, framework="jax")

# Disable cache (always re-quantize from Orbax)
model = flash_rt.load_model(checkpoint, framework="jax", weight_cache=False)

# Force re-quantize (clears both weight cache and calibration cache)
model = flash_rt.load_model(checkpoint, framework="jax", recalibrate=True)
```

### When to disable or clear weight cache

| Situation | Action |
|-----------|--------|
| First deploy | Automatic — cache miss triggers full load + save |
| Normal restart | Automatic — cache hit, ~6s load |
| **After fine-tuning** | `recalibrate=True` or `weight_cache=False` |
| **Checkpoint updated** | Automatic — different hash → new cache |
| Debugging precision | `weight_cache=False` to rule out cache issues |

### Cache isolation

Each cache file is keyed by `SHA256(checkpoint_manifest)[:16] + num_views`. Different checkpoints, different fine-tunes, different num_views all produce separate cache files. No cross-contamination.

---

## HTTP Server

FlashRT includes a FastAPI server for production deployment. The model loads once at startup; all subsequent requests are pure graph replay (~44ms).

### Quick start

```bash
pip install fastapi uvicorn

# Torch
python examples/server.py --checkpoint /path/to/ckpt

# JAX
python examples/server.py --checkpoint /path/to/ckpt --framework jax

# Custom port + thorough autotune
python examples/server.py --checkpoint /path/to/ckpt --port 9000 --autotune 5
```

### Endpoints

**`GET /health`** — Health check
```json
{"status": "ok", "framework": "torch", "version": "2.2.0", "prompt": "pick up the red block"}
```

**`POST /predict`** — Run inference
```json
{
    "prompt": "pick up the red block",
    "images": ["<base64_raw_uint8>", "<base64_raw_uint8>"],
    "image_shape": [224, 224, 3]
}
```

Response:
```json
{
    "actions": [[0.1, -0.2, ...], ...],
    "latency_ms": 44.3,
    "shape": [10, 7]
}
```

If `images` is omitted, dummy random images are used (for testing).

### Test with curl

```bash
# Simple test (dummy images)
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"prompt": "pick up the red block"}'

# With real images (Python)
python -c "
import requests, base64, numpy as np
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
b64 = base64.b64encode(img.tobytes()).decode()
resp = requests.post('http://localhost:8000/predict', json={
    'prompt': 'pick up the red block',
    'images': [b64, b64],
})
print(resp.json()['shape'], resp.json()['latency_ms'], 'ms')
"
```

### Thread safety

The server uses an asyncio lock to ensure only one inference runs at a time (single GPU). Concurrent requests are queued automatically.

---

## CLI Reference

```bash
# Basic inference
python examples/quickstart.py --checkpoint /path/to/ckpt

# JAX framework
python examples/quickstart.py --checkpoint /path/to/ckpt --framework jax

# Benchmark 20 iterations
python examples/quickstart.py --checkpoint /path/to/ckpt --benchmark 20

# Thorough autotune
python examples/quickstart.py --checkpoint /path/to/ckpt --autotune 5

# Force recalibration
python examples/quickstart.py --checkpoint /path/to/ckpt --recalibrate

# LIBERO evaluation
python examples/thor/eval_libero.py \
    --checkpoint /path/to/ckpt \
    --task_suite libero_spatial \
    --framework torch

# Quick LIBERO test (3 tasks × 3 episodes)
python examples/thor/eval_libero.py \
    --checkpoint /path/to/ckpt \
    --task_suite libero_spatial --quick

# HTTP server
python examples/server.py --checkpoint /path/to/ckpt --port 8000
```

---

## Startup Timeline

Typical `load_model()` + first `predict()` timing on Jetson AGX Thor:

### Pi0 / Pi0.5

| Phase | Torch | JAX (no cache) | JAX (cached) |
|-------|-------|---------------|-------------|
| Load checkpoint + FP8 quantize | ~3s | ~42s | **~6s** |
| `set_prompt()`: tokenize + RoPE + time conditioning | ~0.1s | ~0.1s | ~0.1s |
| `set_prompt()`: SigLIP graph capture | ~0.5s | ~0.5s | ~0.5s |
| `set_prompt()`: calibration (cache miss) | ~3s | ~3s | ~3s |
| `set_prompt()`: calibration (cache hit) | **0s** | **0s** | **0s** |
| `set_prompt()`: autotune=3 | ~1s | ~1.5s | ~1.5s |
| First `predict()`: real-data recalibration | ~1.5s | ~1.5s | ~1.5s |
| Subsequent `predict()`: graph replay | **~44ms** | **~44ms** | **~44ms** |

After the first `predict()`, all subsequent calls are pure CUDA Graph replay at ~44ms.
With weight cache + calibration cache, JAX warm start is **~6s** (vs ~42s cold start).

### Pi0-FAST

| Phase | Default | Max-perf (`decode_cuda_graph=True`) |
|-------|--------:|------------------------------------:|
| Load checkpoint + FP8 quantize | ~5s | ~5s |
| `set_prompt()`: calibration (cache miss) | ~2.4s | ~2.4s |
| `set_prompt()`: calibration (cache hit) | **0s** | **0s** |
| `set_prompt()`: SigLIP + prefill graph | ~0.3s | ~0.3s |
| `set_prompt()`: decode graph capture | — | ~1.5s |
| First `predict()`: real-data recalibration | ~2.8s | ~2.8s |
| Subsequent `predict()` (50 tokens) | **~480ms** | **~447ms** |

With calibration cache, default-mode `set_prompt()` drops from ~2.5s to **~0.1s**.
Max-perf mode is ~1.5s (decode graph capture dominates after cache hit).

---

## NVFP4 (Pi0.5 only)

Optional NVFP4 (Blackwell block-scaled FP4) quantization on the Pi0.5
encoder FFN stack, enabled via a single flag `use_fp4=True`. **Currently
only supported on Pi0.5 torch.** The gate applies in two directions:
- Other configs (`pi0` / `groot` / `pi0fast`) log a warning and fall back to FP8.
- `framework="jax"` with `use_fp4=True` also logs a warning and falls back to FP8, even with `config="pi05"` — JAX FP4 is not yet wired up (planned, see handoff prompt Task A).

```python
# Production-recommended — single flag, best-known config:
model = flash_rt.load_model(
    checkpoint,
    config="pi05",
    use_fp4=True,
)
# Equivalent to passing:
#   fp4_layers=tuple(range(18))   # full encoder FFN (18 layers)
#   use_awq=True                   # required for 18-layer scope
#   use_p1_split_gu=True           # production P1 path

# Advanced: override sub-flags for A/B or debug:
model = flash_rt.load_model(
    checkpoint, config="pi05",
    use_fp4=True,
    fp4_layers=(7, 8, 9),       # conservative subset
    use_awq=False,
    use_p1_split_gu=False,
)
```

### Preset resolution

When `use_fp4=True` and a sub-flag (`fp4_layers`, `use_awq`,
`use_p1_split_gu`) is left as `None`, it resolves to the production preset:

| Sub-flag | Preset value | Reason |
|---|---|---|
| `fp4_layers` | `tuple(range(18))` | Full encoder FFN coverage |
| `use_awq` | `True` | Required for 18-layer scope (without AWQ, full-scope FP4 cos collapses to ~0.33) |
| `use_p1_split_gu` | `True` | Split Gate+Up → 2× fp4out GEMM + combiner (Pi0.5 parity, Pi0.6 reusable) |

### What it does

- **GEMMs**: all Gate+Up / Down proj GEMMs across the 18 FFN layers run in
  NVFP4 (block-size 16, UE4M3 scales, Sm1xxBlockScaledConfig tile layout)
  instead of FP8. Attention (QKV, O) stays FP8 fp16-output.
- **P1 split-GU**: gate_proj and up_proj run as two separate
  `cutlass_fp4_gemm_fp4out` GEMMs (FP4-packed output + SFA via the
  `LinCombBlockScaleFactor` epilogue — proven CUTLASS pattern). A fused
  `geglu_two_mul_fp4_to_fp4` combiner reads both FP4 inputs, applies
  GELU + Down-AWQ inv_s + per-block FP4 quant, and writes the packed FP4 +
  SFA directly for the Down GEMM. Eliminates ~31 MB/layer of fp16 DRAM
  round-trip vs the merged-GU path.
- **AWQ** (activation-aware weight quant): per-input-channel pre-scale
  `s[k] = (a[k]/mean(a))^awq_alpha` on each NVFP4 weight, with the
  matching inverse scale fused into the pre-GEMM kernels
  (`residual_add_rms_norm_mul_fp4_sfa` for Gate+Up input,
  `geglu_two_mul_fp4_to_fp4` for Down input). Calibrated on first
  `predict()` with real images, requantized in-place so the captured CUDA
  Graph remains valid.
- **Residual stream**: stays fp16 through the FP4 region (NVIDIA
  `enable_llm_nvfp4` design — `output_quantizer` disabled).
- **Non-FP4 paths**: attention, decoder, SigLIP are unchanged (bit-identical
  to the FP8 baseline).

### Weight loading

When `use_fp4=True`, the FP4 layer weights are loaded directly as fp16 from
safetensors and NVFP4-quantized offline (no FP8 intermediate). This matches
the NVIDIA modelopt design and avoids a double-lossy FP8 → fp16 → FP4
round-trip. A fp8-dequant fallback path exists if direct fp16 load fails.

### Requirements

- SM100+ GPU with Blackwell Tensor Cores (validated on Thor SM110).
  Hardware without NVFP4 support silently falls back to FP8.
- `flash_rt_fp4.so` extension built alongside `flash_rt_kernels.so`
  (automatic in standard install).

### Validation

Pi0.5 on Jetson AGX Thor, LIBERO Spatial 10 tasks × 50 trials = 500 episodes:

| Config | Success | E2E P50 (normal regime) |
|---|---|---|
| FP8 baseline | 491 / 500 (98.2%) | ~43.5 ms |
| **NVFP4 full-18 + AWQ + P1 (`--use_fp4`)** | **491 / 500 (98.2%)** | **~43.5 ms** |

Task-level parity with the FP8 baseline. In-graph kernel-sum profile shows
-2.1 ms/infer theoretical saving, but CUDA Graph per-node scheduling
overhead on the full ~60-kernel pipeline absorbs most of it — net
wall-clock P50 is at parity in the production regime. The kernels are
designed to scale linearly on Pi0.6 (~2× compute) where per-node overhead
stays constant while per-kernel savings grow.

Multi-model precision regression (`tests/test_all_models_precision.py`):

| Model | Config | cos vs reference | P50 |
|---|---|---|---|
| Pi0.5 | FP8 baseline | vs_prod=0.9984, vs_old_torch=0.9999 | 44.5 ms |
| Pi0.5 | `use_fp4=True` preset | vs_pytorch_ref=0.9989, vs_prod=0.9974 | 43.3 ms |
| Pi0 | (unchanged) | vs_pytorch_ref=0.9972 | 46.7 ms |
| Pi0 JAX | (unchanged) | vs_pytorch_ref=0.9983 | 45.1 ms |
| GROOT N1.6 | (unchanged) | vs_pytorch_ref=0.9986 | 46.2 ms |

### Layer selection

`fp4_layers` accepts any subset of encoder layer indices 0-17. Two
configurations are task-level LIBERO-validated:
- `tuple(range(18))` + AWQ (production preset — `--use_fp4` default)
- `(7, 8, 9)` without AWQ (the conservative subset, from the first FP4 drop)

Other subsets are simulation-only (see the internal precision report).

### Known limits / roadmap

- **Decoder FP4** — precision simulation (S2: decoder all-proj) has cos
  0.9985 and passed LIBERO quick 9/9. Full kernel integration planned
  (est. -6 ms E2E).
- **SigLIP FFN FP4** — precision simulation favorable, integration
  requires fp16-native SigLIP weight loader.
- **FP4 on Pi0 / GROOT / Pi0-FAST** — architecture supports it; frontend
  subclasses not yet written. Kernels are reusable without change.
- **Full fp16 fallback** (`use_fp8=False`) — requires a complete fp16 GEMM /
  RMSNorm pipeline, not yet implemented.

## RL inference (classifier-free guidance)

Opt-in CFG inference path for advantage-conditioned VLA policies
trained with the RECAP recipe (π\*0.6,
[arXiv:2511.14759](https://arxiv.org/abs/2511.14759)). The default
`infer()` path is unchanged; RL mode is activated explicitly.

```python
rt = Pi05TorchFrontendRtx("/path/to/pi05_libero_pytorch", num_views=2)

# Opt-in to advantage-conditioned CFG (rebuilds pipeline as Pi05CFGPipeline)
rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
rt.set_prompt("fold the t-shirt")   # cond/uncond prompt pair embedded internally
rt.calibrate([obs])
actions = rt.infer(obs)["actions"]  # runs encoder x2 + decoder x2 per step

# Revert to the standard path any time
rt.set_rl_mode(cfg_enable=False)
```

Support matrix (v0.1.0):

| Frontend | CFG inference |
|---|---|
| `pi05_rtx` (torch) | ✅ |
| `pi0_rtx`, `pi0fast`, `groot_rtx`, all Thor / JAX frontends | ❌ (pi0-RTX pattern ports planned) |

Measured latency (RTX 5090, pi05_libero_pytorch, FP8):

| path | median (ms) | note |
|---|---|---|
| Baseline (no CFG) | 19.0 | single forward, graph replay |
| CFG β=1.5 serial (Phase 2) | 37.1 | 2× sequential forward, graph replay |
| **CFG β=1.5 fused batched (Phase 3b)** | **25.9** | cond+uncond in one B=2 forward |

For sustained 50 Hz real-robot control, combine RL mode with batched
mode to get the fused CFG path:

```python
rt.set_batched_mode(enable=True)
rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
rt.set_prompt("fold the t-shirt")
rt.calibrate([obs])
actions = rt.infer(obs)["actions"]   # 25.9 ms / call
```

Full latency table, numerical contract, and generic batched (RL
rollout) roadmap are in
[docs/rl_inference.md](docs/rl_inference.md).
