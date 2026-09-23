# Pi0.5 on AMD Instinct (MI350X / CDNA4)

Model-specific guide. For hardware support, the build, routing, shared
environment knobs and the test suite, see
[deployment_amd.md](deployment_amd.md) first.

## Performance

Pi0.5, 2 camera views, 10-step denoise, median end-to-end latency on
real observation frames:

| Configuration | Latency | vs torch.compile |
|---|---|---|
| PyTorch eager | 116.9 ms | — |
| torch.compile (max-autotune) | 40.5 ms | 1.00× |
| **FlashRT AMD, FP8 (default)** | **16.4 ms** | **2.47×** |
| FlashRT AMD, BF16 (`use_fp8=False`) | ~30 ms | 1.34× |

For context, the same protocol on an RTX 5090 (FlashRT FP8, CUDA)
measures 19.7 ms.

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
tier, `--iters` for the timing loop length. Expect roughly 16–17 ms
median FP8 after warmup, or roughly 30 ms with `--bf16`.

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
  [calibration.md](calibration.md)); weights quantized at load. MI350's
  FP8 is OCP `e4m3`, never the CDNA3 `fnuz` variant.
- **BF16** (`use_fp8=False`, or `FVK_PI05_AMD_FORCE_BF16=1`) — the
  unquantized baseline; no calibration required.
- **FP4 (MXFP4)** — the GEMM kernels are unlocked and parity-verified
  (`GemmRunner.mxfp4_nt_dev`, hipBLASLt `HIP_R_4F_E2M1` with UE8M0 vec32
  block scales), but there is deliberately **no end-to-end FP4 tier**:
  on the current ROCm stack no available FP4 path beats FP8 at these
  shapes, so a user-facing knob would only select a slower path.
  `load_model(use_fp4=True)` logs a fallback to FP8;
  `use_fp4_decoder=True` remains Thor-only and raises.

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
| `FVK_AMD_DEC_GEMM` | `mfma` | decoder small-M GEMMs: `mfma` (packed-weight MFMA kernels where measured faster) or `hipblaslt` |
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

1. Build on a gfx950 machine with ROCm 7.x and a ROCm PyTorch build.
2. `python examples/pi05_amd_quickstart.py --checkpoint <ckpt>` —
   expect roughly 16–17 ms median FP8 after warmup (`--bf16` for ~30 ms).
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

Cross-run medians move ±0.2–0.35 ms with the timed hipBLASLt algorithm
selection; compare arms inside one process where possible.

# Pi0.5 on AMD RDNA 3.5 (gfx1151)

This is the BF16 Pi0.5 backend for the integrated Radeon 8060S GPU in
Ryzen AI Max+ 395 (the GPUs in AMD Ryzen™ AI Embedded X100 Series processors
share the same RDNA 3.5 architecture). Its public hardware key is
`amd_rdna35`; `gfx1151` is used only for build-time and runtime hardware
validation. General AMD build,
routing, and source-layout information is in
[deployment_amd.md](deployment_amd.md).

## Validation status

The RDNA bindings now use the model-owned torch tensor pipeline. The contributor's
original hardware measurements predate this restructuring and are not a performance
claim for this revision. CPU checks cover import isolation, output schema, shared
execution and routing. Native gfx1151 compilation and independent whole-model
numerical parity have not been rerun for this revision.

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
    use_fp8=False,
)
model.set_prompt("pick up the object", state=state_vector)
actions = model.predict(images=[base_image, wrist_image])
```

`hardware="auto"` maps only the exact gfx1151 ISA to this backend. The
frontend also verifies that the extension was built from the RDNA source set,
uses wave32, and targets gfx1151 before loading the checkpoint.

## Precision tier

The RDNA 3.5 backend supports BF16 weights and activations only. It does not
route through the CDNA4 FP8 kernels, packed MFMA layouts, aiter provider, or
wave64 helpers. `load_model()` converts its historical `use_fp8=True` default
to BF16 with a warning; direct frontend construction rejects `use_fp8=True`.

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

## Environment knobs

The native path is enabled by default. These switches are intended for
same-process parity and performance comparisons:

| Env | Default | Meaning |
|---|---|---|
| `FLASHRT_RDNA35_HIP_ROPE` | `1` | BF16x2 QKV/RoPE kernel |
| `FLASHRT_RDNA35_HIP_DECODER` | `1` | decoder normalization, activation, and residual fusions |
| `FLASHRT_RDNA35_HIP_GQA` | `1` | wave32 decoder GQA for queries up to 16 rows; row-owned K/V up to 1024 |
| `FLASHRT_RDNA35_HIP_GQA_SPLIT_KEY` | `1` | split-key decoder GQA with K/V up to 2048 |
| `FLASHRT_RDNA35_HIP_ENCODER_ATTN` | `1` | native encoder GQA up to 4096 rows and 16 query heads; dense inputs above 1024 use SDPA when faster |
| `FLASHRT_RDNA35_HIP_SMALLM` | `1` | WMMA action projection for M up to 48 |
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

## Feature matrix

| Surface | RDNA 3.5 BF16 |
|---|---|
| `set_prompt` / `infer` / `get_latency_stats` | ✅ |
| 2/3 views, dynamic prompt, horizon, and denoise steps | ✅ |
| Pinned-noise inference and optional full-model HIP Graph | ✅ |
| Native HIP attention, QKV/RoPE, normalization, activation, residual, small-M projection | ✅ |
| FP8 / FP4 | ❌ |
| Temporal K/V reuse, decoder-only graph | ❌ |
| CDNA4 aiter, MFMA, and wave64 kernels | ❌ isolated by build and routing |

The first RDNA version is deliberately an operator-optimization backend. It
does not include temporal caching or an asynchronous serving pipeline.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/test_amd_rdna35_routing.py \
  tests/test_amd_rdna35_ops.py -q

FLASH_RT_PI05_ACTION_DIM=7 \
FLASH_RT_PI05_RDNA35_CKPT="<checkpoint-dir>" \
  .venv/bin/python -m pytest tests/test_amd_rdna35_model.py -q

.venv/bin/python -m pytest tests/test_amd*.py -q
```

Operator tests compare every native family with a PyTorch reference. The
checkpoint-gated tests cover finite outputs, fixed-noise determinism, HIP
Graph parity, and the optimized-versus-library fallback. Profile any extra
local benchmark from an ignored build directory; benchmark harnesses are not
part of the source distribution.


## Model and target ownership

`flash_rt.models.pi05.torch_pipeline.Pi05TorchPipeline` owns the tensor model
semantics: vision layers, prefix encoding, decoder layers and denoising steps.
It accepts tensor-operation, GEMM and attention providers. The RDNA module is
only a constructor binding; it contains no model traversal. Native operation
selection lives in `flash_rt.amd.hardware.rdna35.ops`, `gemm` and `attention`.
The explicit portable torch provider runs the same model pipeline in CPU contract
tests. It is a refactor regression tool, not an independent model oracle.
Existing legacy RTX/Thor/CDNA production pipelines retain their routes; this
change does not claim to migrate all legacy implementations.

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
  `state`: the exact state passed to prompt tokenization, and `noise`: float32
  `(action_horizon, 32)` used by both producers (BF16-representable values avoid
  different initial rounding).
- `raw_actions`: float32 `(action_horizon, 32)` in normalized model space;
  `actions`: float32 `(action_horizon, action_dim)` after the independent
  reference's quantile unnormalization and explicit robot slicing.

Use an independent OpenPI run to generate these arrays; never export expected
outputs from this pipeline or its fallback provider. Save arrays with
`numpy.savez` and keep fixtures out of the source distribution. Run with:

```bash
FLASH_RT_PI05_ACTION_DIM=7 \
FLASH_RT_PI05_RDNA35_CKPT="<checkpoint-dir>" \
FLASH_RT_PI05_REFERENCE="<openpi-fixture.npz>" \
  python -m pytest tests/test_amd_rdna35_model.py -k independent_reference -q
```

No independent fixture was executed as part of this source-only restructuring.
