# Pi0.5 on AMD Instinct (MI300X / CDNA3 and MI350X / CDNA4)

Model-specific guide. For hardware support, the build, routing, shared
environment knobs and the test suite, see
[deployment_amd.md](deployment_amd.md) first.

## Performance

The two architectures use different FP8 formats and kernel routes, so their
results are reported separately. MI300X uses E4M3 FNUZ; MI350X uses OCP
E4M3. Both use architecture-specific packed MFMA decoder kernels where they
beat hipBLASLt.

Pi0.5, 2 camera views, 10-step denoise, median end-to-end latency on
real observation frames:

### MI350X (gfx950)

| Configuration | Median latency | vs torch.compile |
|---|---|---|
| PyTorch eager | 116.9 ms | — |
| torch.compile (max-autotune) | 40.5 ms | 1.00× |
| **FlashRT AMD, FP8 (default)** | **16.4 ms** | **2.47×** |
| FlashRT AMD, BF16 (`use_fp8=False`) | ~30 ms | 1.34× |

For context, the same protocol on an RTX 5090 (FlashRT FP8, CUDA)
measures 19.7 ms.

### MI300X (gfx942)

Measured on this RunPod with ROCm 7.2.4 and AITER, using real frames from
LIBERO Spatial episode 0, 50 warmup replays and 100 measured calls:

| Configuration | Min | Median | p95 | Max |
|---|---:|---:|---:|---:|
| FlashRT AMD, FP8 FNUZ, packed decoder | 41.42 ms | 43.99 ms | 44.20 ms | 52.65 ms |
| FlashRT AMD, BF16, packed decoder + fused boundaries | 34.97 ms | 35.14 ms | 35.43 ms | 50.15 ms |

The timed boundary is the public frontend call: real uint8 camera frames and
pinned denoising noise enter `infer`, and the host action array has returned.
Prompt processing, calibration and graph capture are outside the timed region.
For this gfx942 workload FP8 is an accuracy/footprint tier; it is not a latency
win over BF16. The BF16 result uses gfx942 packed MFMA for decoder QKV, output,
and merged gate/up projections, plus a BF16 boundary kernel that fuses the
gated residual update with the following adaptive RMS normalization. The down
projection remains on hipBLASLt because it is faster at `(M,N,K)=(10,1024,4096)`.
With the same pinned noise, FP8 versus BF16 action cosine was **0.999979** and
the maximum absolute action difference was **0.01562**.

The process was bound to CPUs 48–63, a compact subset of the measured GPU's
NUMA node. Without affinity, container scheduling introduced large gaps
between graph launches.

These numbers are measured with aiter available. Without it the
attention sites fall back to torch SDPA and end-to-end latency rises by
roughly 6 ms — see [deployment_amd.md](deployment_amd.md#build).

Accuracy: output cosine against the FP32 reference sits in a
**0.9992–0.9994** band across processes with the denoise noise pinned
(see [Judging accuracy](#judging-accuracy)). Graph replay inside a
process is bit-identical.

## Quick start

```bash
python examples/pi05_amd_quickstart.py --checkpoint <pi05_checkpoint_dir>
```

Flags: `--num-views {2,3}` (default 2), `--bf16` for the unquantized
tier, `--iters` for the timing loop length. On gfx950 expect roughly
16–17 ms median FP8 after warmup. On the measured gfx942 system FP8 processes
measured about 44 ms median and the optimized BF16 tier measured 35.14 ms.

Library use:

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch", num_views=2, use_fp8=True)
fe = model.pipeline
fe.set_prompt(prompt_text, state=state_vector)
fe.calibrate(observation)              # one-time, real data
result = fe.infer(observation)         # returns {"actions": ...}
```

## Precision tiers

- **FP8 (default)** — activations calibrated once from real data
  (`calibrate(...)`, percentile 99.9 by default, the same contract as
  [calibration.md](calibration.md)); weights quantized at load. The frontend
  selects FNUZ/240 on CDNA3 and OCP E4M3/448 on CDNA4 from extension metadata.
- **BF16** (`use_fp8=False`, or `FVK_PI05_AMD_FORCE_BF16=1`) — the
  unquantized baseline. Calling `calibrate` still prepares buffers and captures
  the runtime graph, but it collects no activation statistics in this tier.
- **FP4 (MXFP4, CDNA4 only)** — the GEMM kernels are unlocked and parity-verified
  (`GemmRunner.mxfp4_nt_dev`, hipBLASLt `HIP_R_4F_E2M1` with UE8M0 vec32
  block scales), but there is deliberately **no end-to-end FP4 tier**:
  on the current ROCm stack no available FP4 path beats FP8 at these
  shapes, so a user-facing knob would only select a slower path.
  `load_model(use_fp4=True)` logs a fallback to FP8;
  `use_fp4_decoder=True` remains Thor-only and raises.

Packed FNUZ is the default FP8 decoder route on gfx942. Packed BF16 is the
default for the gfx942 BF16 decoder. `FVK_AMD_CDNA3_BF16_SMALLM=0` restores
the BF16 hipBLASLt route; `FVK_AMD_DEC_GEMM=hipblaslt` restores the FP8
library route.

## Observation contract

The frontend validates observations strictly — a missing or wrong-shaped
view raises instead of leaving stale image buffers in the captured graph
inputs.

- `num_views` must be **2** (base + wrist camera, the LIBERO deployment)
  or **3** (+ right wrist camera).
- Provide either `observation["images"]` with exactly `num_views`
  entries, or the named keys `image`, `wrist_image`, `wrist_image_right`
  (the first `num_views` of them; supplying `wrist_image_right` with
  `num_views=2` is rejected as a view-count mismatch).
- Every image must be a `uint8` array of shape `(224, 224, 3)`. The
  normalization path is defined on the 0–255 range.

## Prompt-length strategies

Pi0.5 renders robot state into the prompt, so token length drifts with
the state values. Both strategies of the RTX frontend are supported with
the same semantics:

- `state_prompt_mode="exact"` (default) — one captured graph per exact
  prompt length, cached. Pair with `warm_state_prompt_buckets(...)` to
  front-load captures instead of paying them mid-episode.
- `state_prompt_mode="fixed"` — ONE padded graph serves every length
  (masked prefix plus runtime `devpos`/`seqused` K/V append), so no
  capture ever happens mid-loop. Latency follows the **padded** length,
  so right-size the capacity to the deployment's real prompt+state
  length with `state_prompt_fixed_max_len=<tokens>` (env
  `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN`; the default is the
  200-token ceiling, and a prompt exceeding the capacity raises rather
  than silently recapturing).

  In fixed mode the decoder runs the same hand-written split-KV kernel
  as exact mode (seqused pointer, fused FP8-out epilogue included) and
  the encoder runs the seqused variant of the MFMA flash kernel. At a
  right-sized capacity the premium over exact mode is well under 1 ms
  and accuracy holds (cosine 0.9993, on par with exact).

## Environment knobs

Shared knobs are in [deployment_amd.md](deployment_amd.md). Pi0.5-specific:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_DEC_ATTN` | `custom` | decoder cross-attention: `custom` (hand-written split-KV flash, fastest) or the backend default |
| `FVK_AMD_ATTN_FP8OUT` | `1` | fuse the decoder attention output's FP8 quantize into the attention epilogue (bit-identical to the standalone quantize) |
| `FVK_AMD_FIXED_ENC_ATTN` | `flash` | fixed-mode encoder attention: `flash` (MFMA flash kernel, seqused pointer) or `sdpa` (masked torch fallback) |
| `FVK_AMD_CALIB_DET_ATTN` | `flash` | route the encoder site through the deterministic MFMA flash kernel during FP8 calibration so collected scales are run-to-run stable; `off` calibrates on the library path |
| `FVK_AMD_DEC_GEMM` | `mfma` | decoder small-M GEMMs: packed native-format MFMA on measured shapes (FNUZ on gfx942, OCP on gfx950); `hipblaslt` disables it |
| `FVK_AMD_CDNA3_BF16_SMALLM` | `1` | use packed gfx942 BF16 for decoder QKV, O and merged gate/up; `0` restores hipBLASLt |
| `FVK_AMD_CDNA3_BF16_FUSION` | `1` | fuse decoder gated-residual + following AdaRMSNorm boundaries in the BF16 tier; `0` restores the two-kernel chain |
| `FVK_PI05_AMD_FORCE_BF16` | `0` | force the BF16 baseline regardless of `use_fp8` |
| `FLASHRT_PI05_STATE_PROMPT_MODE` | — | overrides the `state_prompt_mode` constructor argument |
| `FLASHRT_PI05_STATE_PROMPT_FIXED_MAX_LEN` | `200` | fixed-mode padded capacity in tokens |

`FRT_ATTN_NSPLIT` / `FRT_ATTN_FUSED` / `FRT_ATTN_REDUCE_ALT` are
attention micro-benchmark knobs for A/B sweeps; leave them unset in
production.

## Feature matrix versus the RTX frontend

| Surface | AMD |
|---|---|
| `set_prompt` / `warm_state_prompt_buckets` | ✅ |
| `calibrate` / `calibrate_with_real_data` (single and multi-frame percentile) | ✅ |
| `infer` / `precision_spec` / `get_latency_stats` | ✅ |
| `state_prompt_mode` `"exact"` and `"fixed"` (devpos/seqused) | ✅ |
| Temporal K/V caching (`cache_frames`), vision pooling/truncation knobs | ✅ |
| `infer(noise=...)` pinned-noise judging | ✅ |
| `set_rl_mode` (advantage-conditioned RL) | ❌ raises `NotImplementedError` |
| Batched serving (`set_batched_mode`, `infer_batch`) | ❌ raises `NotImplementedError` |
| End-to-end FP4 tier | ❌ deliberately not exposed (see Precision tiers) |

## Judging accuracy

The denoise trajectory is conditioned on its starting noise, so a fresh
random draw moves the output cosine by about 1e-3 — the same magnitude
as a real numerical regression.

```python
noise = numpy.random.default_rng(0).standard_normal((10, 32)).astype("float32")
actions = fe.infer(observation, noise=noise)["actions"]
```

Pin the noise to the exact array the reference was generated with and
record the seed (or a hash of the array) alongside the reference. With
pinned noise the FP8 band is 0.9992–0.9994 across processes; the
residual spread comes from library-attention nondeterminism and timed
algorithm picks. Serving should keep the default random draw — pinning
is a judging protocol, not a deployment setting.

## Reproducing the numbers

1. Build for the installed architecture with ROCm 7.x and a ROCm PyTorch
   build (`bash scripts/amd/build_amd.sh gfx942` for MI300X).
2. On a multi-NUMA host, find the selected GPU's `numa_node` through its PCI
   device in `/sys/class/drm`, then bind to a compact subset of that node's
   `local_cpulist`. For this pod GPU 0 is on node 1 and the command was:

   ```bash
   taskset -c 48-63 python benchmarks/pi05_amd_latency.py \
     --ckpt <pi05_checkpoint_dir> \
     --data <real_libero_episode.npz> \
     --warmup 50 --iters 100 --json-out pi05-fp8.json
   ```

   Add `--bf16` for the aligned BF16 control. Do not copy `48-63` to a
   different host without checking its topology.
3. For a judged comparison, run the identical loop on the CUDA build
   (RTX frontend). The protocols are the same: 50-iteration median after
   5 warmup replays, real image observations, `calibrate` before timing.
4. For the model-level test gates, point the suite at a checkpoint:

   ```bash
   export FLASH_RT_PI05_AMD_CKPT=<pi05_checkpoint_dir>
   # optional: a saved reference-actions .npy for the cosine gate
   export FLASH_RT_PI05_AMD_REF_ACTIONS=<reference_actions.npy>
   python -m pytest tests/test_amd_pi05_model.py -v
   ```

Cross-run medians can move with host scheduling and hipBLASLt algorithm
selection; compare arms under the same affinity and data protocol.

# Pi0.5 on AMD RDNA (3 / 3.5 / 4)

The shared BF16 Pi0.5 frontend supports W7900D (`gfx1100` / `amd_rdna3`),
Radeon 8060S (`gfx1151` / `amd_rdna35`) and R9700 (`gfx1201` / `amd_rdna4`).

The RDNA 3.5 target is the integrated Radeon 8060S GPU in
Ryzen AI Max+ 395 (the GPUs in AMD Ryzen™ AI Embedded X100 Series processors
share the same RDNA 3.5 architecture). Its public hardware key is
`amd_rdna35`; `gfx1151` is used only for build-time and runtime hardware
validation. General AMD build,
routing, and source-layout information is in
[deployment_amd.md](deployment_amd.md).

## Validation status

The RDNA bindings use the model-owned torch tensor pipeline. The native gfx1151
extension, independent OpenPI fixture, full and decoder-only HIP Graphs, temporal
schedule, native/fallback parity, output schema and routing have been validated on
Radeon 8060S. The default remains `cache_frames=1`; temporal reuse is explicit.

## Quick start

Create a Python 3.11 environment with a gfx1151 ROCm PyTorch build and the
ROCm development files, then build the architecture-selected extension:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  'torch[device-gfx1151]==2.12.0+rocm10.1.0a20260806' \
  'torchvision[device-gfx1151]==0.27.0+rocm10.1.0a20260806' \
  'torchaudio==2.11.0+rocm10.1.0a20260806' \
  'rocm[devel,device-gfx1151]==10.1.0a20260806' \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  --extra-index-url https://d183u042sr8tht.cloudfront.net/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-first-match \
  --prerelease allow
uv pip install --python .venv/bin/python -e '.[torch]' ml_dtypes pytest ninja build cmake pybind11
.venv/bin/rocm-sdk init

export ROCM_PATH="$(.venv/bin/rocm-sdk path --root)"
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1151
```

Library use:

```python
import flash_rt

model = flash_rt.load_model(
    "<checkpoint-dir>",
    config="pi05",
    framework="torch",
    hardware="amd_rdna35",  # optional on a gfx1151 machine
    action_dim=7,  # use your robot's actual output dimension
    num_views=2,
    cache_frames=1,  # set 2+ only after validating temporal quality
    use_fp8=False,
)
model.set_prompt("pick up the object", state=state_vector)
actions = model.predict(images=[base_image, wrist_image])
```

`hardware="auto"` maps the exact gfx1151 ISA to `amd_rdna35`. The
frontend also verifies that the extension was built from the RDNA source set,
uses wave32, and targets gfx1151 before loading the checkpoint.

### RDNA container quick start

Build the image for your target using
[`docker/Dockerfile.rocm.rdna`](../docker/Dockerfile.rocm.rdna), as described in the
[AMD build guide](deployment_amd.md#rdna-container). The same ROCm 10.1.0,
PyTorch 2.14.0, torchvision and Triton environment supports all three RDNA
targets; each image contains an extension for one exact ISA.

Keep checkpoints and the PaliGemma tokenizer outside the image. Select an
available render device and mount both resources read-only:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri/renderD128 --shm-size=8g \
  -v /path/to/pi05-checkpoint:/models/pi05:ro \
  -v /path/to/paligemma_tokenizer.model:/models/tokenizer.model:ro \
  -e FLASH_RT_PI05_RDNA_CKPT=/models/pi05 \
  -e FLASH_RT_PALIGEMMA_TOKENIZER=/models/tokenizer.model \
  -e FLASH_RT_PI05_ACTION_DIM=7 -e OMP_NUM_THREADS=8 \
  flashrt:rocm-gfx1100 \
  python -m pytest tests/test_amd_rdna_ops.py tests/test_amd_rdna_model.py -q -ra
```

Adjust the image tag, render device and action dimension for your deployment.
Omit the final Python command for an interactive shell. Check the summary
for skipped checkpoint/reference tests; a successful image build alone does
not validate GPU execution or model accuracy. The Python examples below run
inside the same container with checkpoint directory `/models/pi05`.

## Pi0.5 on AMD RDNA 3 (gfx1100 / W7900D)

W7900D uses the shared RDNA frontend and the gfx11 wave32 BF16 WMMA
fragment layout. The exact public hardware key is `amd_rdna3`:

```bash
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1100
```

```python
model = flash_rt.load_model(
    "<checkpoint-dir>", config="pi05", framework="torch",
    hardware="amd_rdna3",  # optional on a gfx1100 machine
    action_dim=7, num_views=2, cache_frames=1, use_fp8=False,
)
actions = model.predict(images=[base_image, wrist_image],
                        prompt="pick up the object", state=state_vector)
```

The [RDNA container](#rdna-container-quick-start) provides the ROCm 10.1.0 /
PyTorch 2.14.0 build and runtime environment for this target. W7900D passes
the native operator and model suites, including fixed-noise Graph/eager
parity and independent OpenPI comparisons of raw and public actions on
all eight bundled LIBERO observations.

The target keeps hipBLASLt for general GEMMs and SDPA for vision attention.
For `chunk_size <= 16`, the decoder QKV, output, merged gate/up and down
projections use streamed split-K WMMA with gfx1100-specific launch choices.
The standard decoder uses fixed (K, N) shapes: (1024, 2560), (2048, 1024),
(1024, 8192) and (4096, 1024). Other supported shapes with prepared weights
retain the generic WMMA path; unprepared projections use hipBLASLt.
Weights are transposed once before capture, adding 0.58 GiB for the standard
18-layer decoder. Larger horizons retain the existing library projection
path; the action projection retains its native M <= 48 path. Setting
`FLASHRT_RDNA35_HIP_SMALLM=0` disables these prepared projections.

Dense encoder prefixes use SDPA; padded prefixes retain the native kernel.
Native decoder GQA, normalization, RoPE, activation and residual fusions
remain enabled.
The gfx1151 and gfx1201 dispatch policies are preserved.

The performance validation workload uses `Fisher-Wang/pi05-libero-pytorch` revision
`ab107fbe6661e8abfec0f13dc27bfd3fafe24c69`, batch 1, two 224x224 RGB views,
10 denoising steps, horizon 10, action dimension 7, `cache_frames=1`,
seed 17, zero eight-dimensional state and prompt `pick up the object`
(143 tokens, encoder length 655).
The state is zero-padded to 32 dimensions, discretized and included in the
`Task / State / Action` prompt template. The 143 tokens are valid prompt
tokens, not token padding; the two images contribute another 512 tokens.

| W7900D path | Public `predict` P50 | Full Graph replay P50 |
|---|---:|---:|
| gfx1100 BF16 with streamed projections and dense-prefix SDPA | 80.51 ms | 78.12 ms |

The table reports the mean of per-process P50 values across 12 independent
processes. Use `OMP_NUM_THREADS=8`, default GEMM autotuning, 30 warmups and
100 synchronized measurements per process.
`predict` uses the public eager path, including input handling, Python
dispatch, transfers and CPU output. Replay uses a separately captured GPU
Graph; the difference between these metrics is not just input/output cost.
Setup, tuning and capture are excluded from both. A gfx1100 tuning comparison
must use an untuned gfx1100-enabled baseline; the revision without gfx1100
support cannot run this target. Run operator, fixed-noise Graph/eager,
library-reference and LIBERO parity tests alongside performance measurements.

To reproduce the public-API timing in a prepared environment, set
`CHECKPOINT` and `FLASH_RT_PALIGEMMA_TOKENIZER` to local resources, then run
this on each checkout. The existing `examples/quickstart.py` remains useful
for smoke tests; its state-free input is a different workload.

```bash
OMP_NUM_THREADS=8 python - <<'PYBENCH'
import os, time
import numpy as np
import torch
import flash_rt

torch.manual_seed(17)
model = flash_rt.load_model(
    os.environ["CHECKPOINT"], config="pi05", framework="torch",
    num_views=2, num_steps=10, action_dim=7, cache_frames=1, use_fp8=False)
rng = np.random.default_rng(17)
images = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
state = np.zeros(8, dtype=np.float32)
def predict():
    return model.predict(images=images, prompt="pick up the object", state=state)
predict()  # setup
for _ in range(30):
    predict()
samples = []
for _ in range(100):
    torch.cuda.synchronize()
    started = time.perf_counter()
    actions = predict()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - started) * 1000)
assert actions.shape == (10, 7) and np.isfinite(actions).all()
print("predict P50/P95 ms:", np.percentile(samples, [50, 95]))
PYBENCH
```

## Pi0.5 on AMD RDNA 4 (gfx1201 / R9700)

RDNA 4 reuses the same wave32 BF16 Pi0.5 frontend and HIP source set as
RDNA 3.5. The public hardware key is `amd_rdna4`. Build and load with the
matching ISA:

```bash
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1201
```

```python
model = flash_rt.load_model(
    "<checkpoint-dir>",
    config="pi05",
    framework="torch",
    hardware="amd_rdna4",  # optional on a gfx1201 machine
    action_dim=7,
    num_views=2,
    cache_frames=1,
    use_fp8=False,
)
```

`hardware="auto"` maps only the exact gfx1201 ISA to this backend. The
frontend requires a gfx1201-built RDNA extension and a visible gfx1201
device before loading the checkpoint.

The WMMA kernels share one source with gfx1151; `csrc/amd/kernels/wmma_rdna.h`
maps the gfx12 fragment layout, where each lane carries half of a 16-element
K block instead of a replicated full block. Routing differs from gfx1151 where
R9700 measurements differ:

- With `chunk_size <= 16`, the decoder QKV, output, merged gate/up, and down
  projections use a streamed split-K small-M kernel instead of hipBLASLt.
  These projections stream about 0.6 GB of weights per denoise step, so they
  are bandwidth bound. Measured with cold weights at M=10, the four
  projections take 14.4 ms per inference instead of 18.5 ms. The transposed
  weight copies add 0.58 GiB.
- Dense encoder prefixes stay on SDPA, which takes 66 us at 661 rows against
  263 us for the native kernel. The native kernel is used for padded prefixes
  (`FLASHRT_RDNA35_COMPACT_ENCODER=0`), where it takes 239 us against 602 us
  for masked SDPA.

The RDNA operator and model suites accept a visible gfx1201 device.
`test_optimized_path_matches_aten_fallback`
and `test_real_libero_frames_are_deterministic_and_match_aten_fallback`
compare the loaded path with the PyTorch fallback on fixed noise.
The LIBERO test reads the eight real observations in
`examples/thor/assets/pi05_libero.npz`.

End-to-end latency on R9700 follows the gfx1151 protocol: `infer` median
after warmup, BF16, 2 views, horizon 10, 10 denoise steps:

| Input | `cache_frames` | Median | p95 |
|---|---:|---:|---:|
| Synthetic 224² views, prompt length 149 | 1 | 54.8 ms | 54.9 ms |
| Synthetic 224² views, prompt length 149 | 2 | 38.3 ms (decoder-only frames ~21.8 ms) | 54.8 ms |
| Eight LIBERO observations, prompt length 150 | 1 | 54.9 ms | 55.2 ms |

Routing the decoder projections through hipBLASLt instead measures 60.0 ms
for the same `cache_frames=1` input.

## Precision tier

The RDNA 3 / 3.5 / 4 targets support BF16 weights and activations only. They
do not route through the CDNA4 FP8 kernels, packed MFMA layouts, aiter
provider, or wave64 helpers. `load_model()` converts its historical
`use_fp8=True` default to BF16 with a warning; direct frontend construction
rejects `use_fp8=True`.

## Observation contract

The public BF16 input range matches the CDNA Pi0.5 frontend:

- `num_views` is **2** (base + wrist) or **3** (+ right wrist).
- Images are `uint8` arrays with shape `(224, 224, 3)`, supplied either as an
  `images` list with exactly `num_views` entries or through the named
  `image`, `wrist_image`, and `wrist_image_right` keys.
- Prompt length is dynamic up to `max_prompt_len` (default 200).
- `action_dim` explicitly declares the robot output dimension (1..32). It may
  instead be supplied as `output_action_dim` in checkpoint `config.json`. The
  model's padded `action_dim` field and constant quantiles are not an output
  schema. A real constant final joint or gripper channel is preserved.
- The action horizon is any positive integer. By default it is read from
  `config.json`'s `action_horizon`, with 10 used when that field is absent.
- `num_steps` is any positive integer. The frontend regenerates the
  sinusoidal time schedule and applies the matching `-1 / num_steps` ODE
  projection scale.

Native decoder attention and the WMMA action projection are selected only
inside their validated small-shape profiles. Larger horizons and sequences
fall back to PyTorch SDPA or hipBLASLt, so changing the checkpoint profile
does not launch a shape-incompatible kernel.

`cache_frames=1` refreshes image/prompt K/V on every call. With
`cache_frames=2`, calls alternate between a full forward and decoder-only
execution using the preceding frame's K/V. Calling `set_prompt()` resets the
schedule and invalidates cached K/V, so the next inference is always a full refresh; consequently, a
serving loop that updates state through `set_prompt(..., state=...)` on every
frame intentionally receives no temporal-cache speedup.

Only successful inference calls advance the reuse period. An inference failure
invalidates the cached prefix and resets the schedule, so retrying with valid
inputs always performs a full refresh. The shared pipeline rejects decoder-only
calls without a valid prefix. Invalidating K/V contents retains captured graph
plans; a same-shape refresh updates the buffers used by their next replay.

## Environment knobs

The native path is enabled by default. These switches are intended for
same-process parity and performance comparisons:

| Env | Default | Meaning |
|---|---|---|
| `FLASHRT_RDNA35_HIP_ROPE` | `1` | BF16x2 QKV/RoPE kernel |
| `FLASHRT_RDNA35_HIP_DECODER` | `1` | decoder normalization, activation, and residual fusions |
| `FLASHRT_RDNA35_HIP_GQA` | `1` | wave32 decoder GQA for queries up to 16 rows; row-owned K/V up to 1024 |
| `FLASHRT_RDNA35_HIP_GQA_SPLIT_KEY` | `1` | split-key decoder GQA with K/V up to 2048 |
| `FLASHRT_RDNA35_HIP_ENCODER_ATTN` | `1` | native encoder GQA up to 4096 rows and 16 query heads; dense inputs above 1024 use SDPA when faster, and gfx1100/gfx1201 keep dense prefixes on SDPA |
| `FLASHRT_RDNA35_HIP_SMALLM` | `1` | WMMA action projection for M up to 48; on gfx1100/gfx1201 with `chunk_size <= 16`, also the decoder QKV, output, gate/up, and down projections |
| `FLASHRT_RDNA35_HIP_FFN_GATE_UP` | `1` | merged decoder gate/up dataflow |
| `FLASHRT_RDNA35_HIP_ENCODER_FFN` | `1` | merged encoder gate/up dataflow |
| `FLASHRT_RDNA35_HIP_LARGE_OPS` | `1` | shape-specialized vision/encoder normalization and residual kernels |
| `FLASHRT_RDNA35_PRECOMPUTE_MODULATION` | `1` | prepare timestep modulation before inference |
| `FLASHRT_RDNA35_HIP_GQA_KEYS` | `4` | split-key group size: 1, 2, 4, or 8 |
| `FLASHRT_RDNA35_COMPACT_ENCODER` | `1` | execute only the valid image + prompt prefix |
| `FLASHRT_RDNA35_GEMM_AUTOTUNE` | `1` | instance-local timed hipBLASLt algorithm selection |
| `FLASHRT_RDNA35_GEMM_ALGOS` | `16` | candidate count for the local GEMM selection |

No generated algorithm CSV is shipped and no process-global PyTorch
TunableOp state is modified. Triton is not a runtime dependency.

For performance comparisons, use the same environment and workload, warm up
each process, and interleave repeated baseline/candidate runs. Timed GEMM
selection can choose different hipBLASLt algorithms even for identical source;
record an unchanged-source control when investigating small latency differences.
`FLASHRT_RDNA35_GEMM_AUTOTUNE=0` uses the library heuristic and can help isolate
selection variance, but it changes the execution plan. Report that diagnostic
separately from default-autotune deployment timings.

## Feature matrix

| Surface | RDNA 3 / 3.5 / 4 BF16 |
|---|---|
| `set_prompt` / `infer` / `get_latency_stats` | ✅ |
| 2/3 views, dynamic prompt, horizon, and denoise steps | ✅ |
| Pinned-noise inference and optional full-model HIP Graph | ✅ |
| Native HIP attention, QKV/RoPE, normalization, activation, residual, small-M projection | ✅ |
| FP8 / FP4 | ❌ |
| Temporal K/V reuse (`cache_frames`) and decoder-only execution | ✅ |
| CDNA4 aiter, MFMA, and wave64 kernels | ❌ isolated by build and routing |

`cache_frames=1` is the lossless default and refreshes Vision/Encoder on every
frame. Values greater than one reuse the most recently encoded K/V prefix for
intermediate decoder-only frames. This improves effective throughput but uses
stale visual/prompt context, so deployments must validate task quality on real
trajectories. The RDNA backend does not enable an asynchronous serving pipeline.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/test_amd_rdna_routing.py \
  tests/test_amd_rdna_ops.py -q

FLASH_RT_PI05_ACTION_DIM=7 \
FLASH_RT_PI05_RDNA_CKPT="<checkpoint-dir>" \
  .venv/bin/python -m pytest tests/test_amd_rdna_model.py -q

.venv/bin/python -m pytest tests/test_amd*.py -q
```

`FLASH_RT_PI05_RDNA35_CKPT` remains an alias for the checkpoint variable.

Operator tests compare every native family with a PyTorch reference on
gfx1100, gfx1151 and gfx1201. The checkpoint-gated tests cover finite
outputs, fixed-noise determinism, HIP Graph parity, the
optimized-versus-library fallback, and the eight real LIBERO observations
shipped in `examples/thor/assets/pi05_libero.npz`. Profile any extra local
benchmark from an ignored build directory; benchmark harnesses are not
part of the source distribution.


## Model and target ownership

`flash_rt.models.pi05.torch_pipeline.Pi05TorchPipeline` owns the tensor model
semantics: vision layers, prefix encoding, decoder layers and denoising steps.
It accepts tensor-operation, GEMM and attention providers. The RDNA module is
only a constructor binding; it contains no model traversal. Native operation
selection lives in `flash_rt.amd.hardware.rdna35.ops`, `gemm` and `attention`.
`capabilities_rdna.py` validates the exact ISA and selects the `rdna3`, `rdna35`
or `rdna4` providers at setup. The shared frontend is `frontends/torch/pi05_rdna.py`
and its thin model binding is `models/pi05_rdna/pipeline.py`.
The shared providers remain in `hardware/rdna35/`, with generation-specific
bindings in `rdna3/` and `rdna4/`. Public hardware keys, extension
`build_info()` labels and `FLASHRT_RDNA35_*` controls remain supported.
The explicit portable torch provider runs the same model pipeline in CPU contract
tests. It is a refactor regression tool, not an independent model oracle.
RTX, Thor and CDNA use their own production pipelines.

Safetensors conversion and prompt embedding are shared by the CDNA and RDNA
frontends through `flash_rt.frontends.torch.pi05_checkpoint`. Importing them
loads neither HIP nor CUDA runtime libraries. The CDNA names remain re-exported
for existing callers.

## Independent reference fixtures

`test_independent_reference_fixture` accepts an NPZ produced by a pinned OpenPI
revision, without pickled objects. It compares **both raw and public actions**
for identical checkpoint, images, prompt, state and initial noise. It checks
checkpoint SHA-256 before inference and requires cosine >= 0.999 plus elementwise
`atol=0.1, rtol=0.03`; those BF16 bounds are acceptance criteria, not measured
results. A missing fixture skips this optional hardware test.

Required NPZ entries:

- `metadata`: scalar JSON string containing `producer: "openpi"`, the 40-character
  `producer_revision`, `checkpoint_sha256`, `num_steps`, `action_dim`, and
  `action_horizon`.
- `images`: uint8 `(num_views, 224, 224, 3)`, `prompt`: scalar string,
  optional `state`: the exact state passed to prompt tokenization, and `noise`: float32
  `(action_horizon, 32)` used by both producers (BF16-representable values avoid
  different initial rounding).
- `raw_actions`: float32 `(action_horizon, 32)` in normalized model space;
  `actions`: float32 `(action_horizon, action_dim)` after the independent
  reference's quantile unnormalization and explicit robot slicing.

The official `pi05_libero` configuration uses `discrete_state_input=False`.
Its fixtures omit `state` so FlashRT also uses a state-free prompt; the original
robot state is retained separately as `observation_state`. Fixtures for a
discrete-state prompt must include the exact tokenization state instead.

Use an independent OpenPI run to generate these arrays; never export expected
outputs from this pipeline or its fallback provider. Save arrays with
`numpy.savez` and keep fixtures out of the source distribution.

Generate reference data in a separate, pinned OpenPI PyTorch environment
using the official `pi05_libero` policy and the same checkpoint and inputs
used by FlashRT. Follow
[OpenPI's PyTorch setup](https://github.com/Physical-Intelligence/openpi#pytorch-support)
for its dependencies and `transformers_replace` files. Reuse the resulting
NPZ fixtures across RDNA targets; OpenPI is not required in the RDNA container.

The validated fixture producer uses OpenPI revision
`15a9616a00943ada6c20a0f158e3adb39df2ccac` with Transformers 4.53.2 and
safetensors 0.7.0. Reference inference on RTX 5090 retains OpenPI's mixed
BF16/FP32 parameter policy, disables compilation and TF32, and uses full
FP32 matmul precision. The same eight fixtures pass on gfx1100, gfx1151
and gfx1201 with the bounds above.

`FLASH_RT_PI05_REFERENCE` accepts one NPZ file or a directory of NPZ fixtures.
Run the numerical checks with:

```bash
FLASH_RT_PI05_ACTION_DIM=7 \
FLASH_RT_PI05_RDNA_CKPT="<checkpoint-dir>" \
FLASH_RT_PI05_REFERENCE="<openpi-fixtures-dir>" \
  python -m pytest tests/test_amd_rdna_model.py -k independent_reference -q
```

A skipped reference test does not establish parity with OpenPI.
