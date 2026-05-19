# Motus FlashVLA Usage (Beta)

This document is the handoff path for Motus algorithm testing on the
FlashRT RTX backend inside FlashVLA. It covers the current E2E inference
contract, build steps, the supported precision profiles, and the
optional training-free TeaCache step-caching.
It also documents the RTC-lite execution wrapper, which converts the
chunked Motus output into a fixed-rate action stream without changing
model numerics.

The numbers in this guide are measured on:

- Model: **Stage3 Motus checkpoint** (the public `Motus_robotwin2`
  bundle is the validated reference)
- Dataset: **RoboTwin2 mini bundles** (4 samples; the public `sample_00`
  is the default reference)
- GPU: RTX 5090 (sm_120, 32 GB)
- Pipeline: 10 inference steps, the committed default
- Precision profile: `--fp4-profile fast` unless stated otherwise

If you swap the checkpoint, the dataset, or the GPU, re-record the
quickstart latency, cosine, and `peak_allocated` lines on your target
before treating any number here as a contract.

---

## Precision profiles at a glance

| Profile | Purpose | Wall (sample_00) | cos action | cos frames | VRAM peak |
|---|---|---|---|---|---|
| `fast` | **Validated Stage3 fast profile (default)** | ~170 ms | 0.99993 | 0.99911 | 28.2 GB |
| `fast` + TeaCache | Step caching on top of `fast` (env-gated) | **~100 ms** | 0.99992 | 0.99902 | 28.2 GB |
| `off` | Explicit FP8 trajectory baseline (FP4/NVFP4 disabled) | record per machine | — | — | record per machine |
| `fast-cache` | Latency-oriented FP4/NVFP4 without tiny-FP8 dispatch | record per bundle | — | — | record per bundle |
| `on` | Explicit FP4/NVFP4 experiment | record per bundle | — | — | record per bundle |

Cosine targets in this table are vs the upstream Motus E2E reference
(`outputs/predicted_*.pt` in the input bundle). They are the median
across 10 graph replays.

The validated red lines used during caching ablation:

- `cos(action) ≥ 0.999`
- `cos(frames) ≥ 0.99`

Low-level kernel A/B flags remain available for development, but they are
not part of the public algorithm-test interface. Re-validate trajectory
metrics before comparing any experimental flag combination.

Run each profile in a fresh Python process. Do not switch profiles
inside one long-lived process; swap modules read precision flags during
import/install.

---

## 1. Paths

Set these variables for your own checkout and checkpoint layout. Do not
rely on any developer-machine path.

```bash
export FLASHVLA_ROOT=/absolute/path/to/FlashVLA
export MOTUS_ROOT=/absolute/path/to/Motus
export MOTUS_CHECKPOINT=${MOTUS_ROOT}/pretrained_models/Motus_robotwin2
export MOTUS_WAN_PATH=${MOTUS_ROOT}/pretrained_models/Wan2.2-TI2V-5B
export MOTUS_VLM_PATH=${MOTUS_ROOT}/pretrained_models/Qwen3-VL-2B-Instruct
export MOTUS_INPUT_BUNDLE=/absolute/path/to/robotwin_mini_bundles/sample_00
```

`MOTUS_INPUT_BUNDLE` must contain:

```text
inputs/
  first_frame.pt
  state.pt
  instruction.txt
  t5_embed.pt
  vlm_inputs.pt
outputs/                    optional, used only for cosine check
  predicted_actions.pt
  predicted_frames.pt
```

FlashRT does not run Qwen/T5 preprocessing inside the hot path. Motus
algorithm tests should provide the same precomputed `t5_embed.pt` and
`vlm_inputs.pt` contract used by the upstream Motus E2E reference.

Motus FP4/VAE kernels are built into `flash_rt.flash_rt_kernels` by CMake.
There is no separate Motus kernel library directory in the public build.

---

## 2. Clone and container

Clone the FlashVLA repository and the upstream Motus repository into
the paths you exported above:

```bash
mkdir -p "$(dirname "${FLASHVLA_ROOT}")" "$(dirname "${MOTUS_ROOT}")"
git clone <flashvla-repo-url> "${FLASHVLA_ROOT}"
git clone <motus-repo-url> "${MOTUS_ROOT}"
```

If using Docker, mount your own workspace root and then set the path
variables inside the container. The container name, image, and mount
point below are examples; replace them with your environment values.

```bash
export HOST_WORKSPACE=/absolute/path/to/workspace
export CONTAINER_WORKSPACE=/workspace/project

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name motus-flashvla -it \
  -v "${HOST_WORKSPACE}:${CONTAINER_WORKSPACE}" \
  -v "${HOME}/.cache/modelscope:/workspace/modelscope" \
  -v "${HOME}/.cache/huggingface:/workspace/hfcache" \
  <cuda-pytorch-image> bash

export FLASHVLA_ROOT=${CONTAINER_WORKSPACE}/FlashVLA
export MOTUS_ROOT=${CONTAINER_WORKSPACE}/Motus
export MOTUS_CHECKPOINT=${MOTUS_ROOT}/pretrained_models/Motus_robotwin2
export MOTUS_WAN_PATH=${MOTUS_ROOT}/pretrained_models/Wan2.2-TI2V-5B
export MOTUS_VLM_PATH=${MOTUS_ROOT}/pretrained_models/Qwen3-VL-2B-Instruct
export MOTUS_INPUT_BUNDLE=${CONTAINER_WORKSPACE}/robotwin_mini_bundles/sample_00
```

---

## 3. Build FlashRT kernels

```bash
cd "${FLASHVLA_ROOT}"
export PYTHONPATH="${FLASHVLA_ROOT}"
export FLASH_RT_MOTUS_ROOT="${MOTUS_ROOT}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -e ".[torch]"

# The SM120 FP8/NVFP4 kernels use CUTLASS/CuTe headers. CUTLASS is not
# vendored in this repository, so set it up before configuring CMake.
if [ ! -d third_party/cutlass ]; then
  git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
fi

mkdir -p build
cd build
cmake .. -DGPU_ARCH=120 -DFLASHRT_ENABLE_MOTUS=ON
make -j"$(nproc)"
cd ..
```

Verify imports:

```bash
python - <<'PY'
import torch
import flash_rt
from flash_rt import flash_rt_kernels
print(torch.__version__, torch.cuda.get_device_name())
print("flash_rt ok", flash_rt.__version__)
print("kernels ok", hasattr(flash_rt_kernels, "GemmRunner"))
PY
```

---

## 4. Algorithm test contract

Use this contract when comparing FlashRT numbers with algorithm baselines:

- Run `off` and any FP4/NVFP4 profile in **separate** Python processes.
- The first `infer()` call performs FP8 calibration and CUDA Graph
  capture. Do not count it as steady-state latency; benchmark later
  graph replays only.
- The input bundle contract is fixed: `first_frame.pt`, `state.pt`,
  `instruction.txt`, `t5_embed.pt`, and `vlm_inputs.pt`.
- Qwen/T5 preprocessing is outside the FlashRT hot path. If a baseline
  includes Qwen/T5 preprocessing in its latency number, call that out
  separately.
- VAE decode is part of the Motus E2E standard. Do not skip decode or
  compare action-only latency against this full pipeline.
- Record `[motus.quickstart] cuda memory: peak_allocated=...` for each
  machine. The Stage3 `fast` profile reports ~28.2 GB peak on the
  reference RTX 5090.
- The default denoising loop uses 10 steps. Other step counts can be
  passed at frontend construction or through the quickstart flag and
  will rebuild the captured CUDA Graph.

---

## 5. Quickstart: validated Stage3 fast profile (`fast`)

```bash
cd "${FLASHVLA_ROOT}"
export PYTHONPATH="${FLASHVLA_ROOT}"
export FLASH_RT_MOTUS_ROOT="${MOTUS_ROOT}"
export FLASH_RT_MOTUS_FP4_PROFILE=fast
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/motus_quickstart.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile fast \
  --num-inference-steps 10 \
  --benchmark 10
```

Expected steady-state output on the RoboTwin2 mini `sample_00` bundle,
RTX 5090:

```text
graph P50          ~= 170 ms
peak_allocated     ~= 28.2 GB
cos action         ~= 0.99993
cos frames         ~= 0.99911
```

Cross-sample stability from the four-bundle validation run
(calibration done on each bundle's own `first_frame`, no recalibration
between). Wall time has normal run-to-run variance; use the quickstart
P50 on your machine as the latency contract and the cosine columns as
the stability check:

| Bundle    | validation P50 (ms) | cos action | cos frames |
|-----------|--------------:|-----------:|-----------:|
| sample_00 | 171.19        | 0.999929   | 0.999117   |
| sample_01 | 171.28        | 0.999935   | 0.998644   |
| sample_02 | 171.09        | 0.999921   | 0.999144   |
| sample_03 | 171.07        | 0.999913   | 0.998844   |

All four samples pass the red lines (`cos(action) ≥ 0.999`,
`cos(frames) ≥ 0.99`).

Use the saved action output with the trajectory evaluator when comparing
algorithm-facing changes. Cosine alone is not the acceptance metric.

---

## 6. TeaCache step caching (env-gated)

TeaCache is a training-free step-level cache shipped behind an env
flag. It caches `(video_velocity, action_velocity)` at the configured
compute steps and reuses them at skip steps, bypassing the 30-layer
transformer plus both output heads for skipped steps. The schedule is
fixed at install time and baked into the captured CUDA Graph.

### Activation

```bash
export FLASH_RT_MOTUS_USE_TEACACHE=1
# Optional: override the default skip schedule (default: 2,3,4,5,6,7,8
# for num_inference_steps=10)
# export FLASH_RT_MOTUS_TEACACHE_SKIP_STEPS=2,3,4,5,6,7,8
```

Then run the same quickstart command as §5 in a fresh Python process:

```bash
cd "${FLASHVLA_ROOT}"
export PYTHONPATH="${FLASHVLA_ROOT}"
export FLASH_RT_MOTUS_ROOT="${MOTUS_ROOT}"
export FLASH_RT_MOTUS_FP4_PROFILE=fast
export FLASH_RT_MOTUS_USE_TEACACHE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python examples/motus_quickstart.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile fast \
  --num-inference-steps 10 \
  --benchmark 10
```

### Schedules and trade-off

| Skip schedule           | #skip | Wall (ms) | cos action | cos frames | Δ vs `fast` |
|-------------------------|------:|----------:|-----------:|-----------:|-----------:|
| (off)                   | 0     | 169.7     | 0.99994    | 0.99912    | —          |
| `3,5,7`                 | 3     | 141.8     | 0.99993    | 0.99912    | -27.9 ms   |
| `2,3,5,6,8`             | 5     | 121.1     | 0.99994    | 0.99913    | -48.6 ms   |
| `2,3,4,5,6,7,8` (default) | 7   | **99.6** | 0.99992    | 0.99902    | **-70.1 ms** |
| `1,2,3,4,5,6,7,8`       | 8     |  90.1     | 0.99993    | 0.99894    | -79.6 ms   |

Cross-sample stability with the default skip schedule:

| Bundle    | wall P50 (ms) | cos action | cos frames |
|-----------|--------------:|-----------:|-----------:|
| sample_00 | 99.57         | 0.999923   | 0.999017   |
| sample_01 | 100.42        | 0.999947   | 0.998630   |
| sample_02 | 100.37        | 0.999945   | 0.999147   |
| sample_03 | 100.29        | 0.999918   | 0.998765   |

VRAM is unchanged at ~28.2 GB peak (the cached velocity buffers are
already part of the per-step working set).

### Other caching variants

Three additional training-free caching variants are bundled but
**default OFF**, kept as ablation infrastructure. None of them beats
TeaCache on wall on the Stage3 bundle, and TaylorSeer/MixCache leave a
smaller frames-cosine margin at the aggressive schedule.

```bash
export FLASH_RT_MOTUS_USE_EASYCACHE=1     # auto-pick schedule from time-embedding distance
export FLASH_RT_MOTUS_USE_TAYLORSEER=1    # order-0/1 final-velocity Taylor forecast
export FLASH_RT_MOTUS_USE_MIXCACHE=1      # hybrid per-skip-step order-0/1 by extrap coeff
```

The four cache methods are mutually exclusive; whichever env is
enabled first wins. Leave all four unset for the unmodified Stage3
fast profile.

---

## 7. RTC-lite action streaming

RTC-lite is an execution-layer wrapper for chunked action policies. It
does **not** make a single Motus model call faster. Instead, it pre-fills
an initial action chunk, then consumes actions at a fixed controller
rate while a background worker generates the next chunk.

Use RTC-lite when you care about action supply frequency / controller
continuity, not kernel latency.

Properties:

- No training.
- No denoiser VJP/backward guidance.
- No change to Motus CUDA Graph or model numerics.
- The foreground controller receives one action per tick.
- The background worker calls the same `pipe.infer()` path used by
  `motus_quickstart.py`.

Motus Stage3 `sample_00` currently returns:

```text
horizon=16
action_dim=14
profile fast latency ~= 170 ms
profile fast + TeaCache latency ~= 100 ms
```

### 50 Hz RTC-lite smoke test

Strict profile:

```bash
unset FLASH_RT_MOTUS_USE_TEACACHE
python examples/motus_rtc_lite.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile fast \
  --target-hz 50 \
  --ticks 64
```

Expected supply-layer output:

```text
horizon=16 action_dim=14 target_hz=50.00 latency_probe~=170 ms start_next_at~=6
served=64 elapsed=1.280s effective_hz=49.99
deadline_misses=0 held_actions=0
```

TeaCache profile:

```bash
export FLASH_RT_MOTUS_USE_TEACACHE=1
python examples/motus_rtc_lite.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile fast \
  --target-hz 50 \
  --ticks 64
```

Expected supply-layer output:

```text
horizon=16 action_dim=14 target_hz=50.00 latency_probe~=100 ms start_next_at~=10
served=64 elapsed=1.280s effective_hz=49.99
deadline_misses=0 held_actions=0
```

### Default execution strategy

The default strategy is:

```text
prefill initial chunk before the controller loop
start_next_at = derived from measured latency and target_hz
blend_steps = 0
miss_policy = hold_last
```

For the Stage3 50 Hz strict profile this derives `start_next_at≈6`.
For TeaCache this derives `start_next_at≈10`, which is later and leaves
more room for reacting to fresh observations.

We also swept `start_next_at in {4,6,8}` and `blend_steps in {0,1,2}`
at 50 Hz on `sample_00`:

| start_next_at | blend_steps | deadline misses | held actions | note |
|---:|---:|---:|---:|---|
| 4 | 0/1/2 | 0 | 0 | stable but starts next chunk earlier |
| 6 | 0/1/2 | 0 | 0 | preferred strict-profile default |
| 8 | 0/1/2 | 3 | 3 | too late for strict profile |

Keep `blend_steps=0` by default. Blending is an execution-layer action
edit; it should be enabled only after evaluating jerk / task success in
the target controller.

### What RTC-lite proves and does not prove

RTC-lite proves that the Motus Stage3 chunk output can supply a 50 Hz
foreground action stream under the measured latency, provided the first
chunk is prefetched before the loop starts.

It does not prove task success. The next validation step is a real
controller or simulator rollout measuring boundary jump, jerk,
deadline misses, and task metrics.

---

## 8. Quickstart: explicit FP8 baseline (`off`)

Use this in a separate Python process when you need a strict FP8
trajectory baseline with the Motus NVFP4 paths disabled:

```bash
export FLASH_RT_MOTUS_FP4_PROFILE=off
python examples/motus_quickstart.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile off \
  --num-inference-steps 10 \
  --benchmark 10
```

Record the resulting `graph P50`, `cos action`, `cos frames`, and
`peak_allocated` line for your machine and bundle. The `off` profile
turns these Motus NVFP4 paths off:

```text
video QKV
video O
video FFN (down)
cross Q
cross O
VAE FP4 kernels
```

It still uses FP8 for the rest of the Motus stack (the `off` setting
is about NVFP4, not about FP8 calibration). The first `infer()` call
runs FP8 calibration regardless of profile.

---

## 9. Calibration

Detailed FP8 calibration mechanics live in
[`docs/calibration.md`](calibration.md). The summary as it applies to
Motus:

- **Weights**: per-tensor FP8 scales are computed once at checkpoint
  load (`quant_fp8`) and stored alongside the weight tensors.
- **Activations**: per-GEMM-input FP8 scales are computed during the
  **first `infer()` call**, before CUDA Graph capture, by recording
  the per-tensor amax of the captured forward pass.
- The Motus frontend currently runs **single-sample calibration**: the
  one observation passed to the first `infer()` is the calibration
  sample. There is no public `calibrate(observations=[...], percentile=...)`
  API yet; the multi-sample dataset path documented in
  `docs/calibration.md` §10 is implemented for `pi05_thor`,
  `groot_thor`, and `groot_rtx`, not for Motus.
- The validated Stage3 bundle has shown stable cross-sample behaviour
  with single-sample calibration: cosine variance across `sample_00` to
  `sample_03` is at most 2e-5 on action and 5e-4 on frames (see §5/§6
  tables). Each sample's calibration stays inside the red lines for
  the other three samples too.
- If your downstream evaluation bundle distribution is wider than the
  Stage3 RoboTwin2 mini set (e.g. covers many lighting / occlusion
  conditions that single-sample calibration cannot represent),
  port the `pi05_thor._calibrate_multi_frame` percentile-reduced
  multi-sample path to the Motus frontend before relying on the
  current numbers.

---

## 10. Denoising step count

The quickstart exposes the denoising loop count as:

```bash
--num-inference-steps 10
```

The default and committed baseline is 10 steps. For algorithm
experiments, use a fresh process and pass the desired value before
frontend construction:

```bash
python examples/motus_quickstart.py \
  --checkpoint "${MOTUS_CHECKPOINT}" \
  --motus-root "${MOTUS_ROOT}" \
  --wan-path "${MOTUS_WAN_PATH}" \
  --vlm-path "${MOTUS_VLM_PATH}" \
  --input-bundle "${MOTUS_INPUT_BUNDLE}" \
  --fp4-profile fast \
  --num-inference-steps 6 \
  --no-compare \
  --benchmark 5
```

Changing the step count rebuilds the timestep schedule, AdaLN /
static modulation caches, Euler `dt`, the captured CUDA Graph, and the
default TeaCache skip schedule (the default `2..8` skip set assumes
`num_inference_steps=10`). It is not a runtime toggle inside an
already-constructed frontend.

---

## 11. Programmatic API

Use `set_prompt()` once per prompt/input-embedding bundle, then call
`infer()` for observations. The first `infer()` calibrates FP8 and
captures the CUDA Graph; later calls replay the graph.

```python
import os
import torch
from pathlib import Path

motus_root = Path(os.environ["MOTUS_ROOT"])
checkpoint = Path(os.environ["MOTUS_CHECKPOINT"])
wan_path = Path(os.environ["MOTUS_WAN_PATH"])
vlm_path = Path(os.environ["MOTUS_VLM_PATH"])
bundle = Path(os.environ["MOTUS_INPUT_BUNDLE"])

os.environ["FLASH_RT_MOTUS_ROOT"] = str(motus_root)
os.environ["FLASH_RT_MOTUS_FP4_PROFILE"] = "fast"
# Optional: turn TeaCache on for the ~100 ms operating point
# os.environ["FLASH_RT_MOTUS_USE_TEACACHE"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from examples.motus_quickstart import (
    _install_deepspeed_stub,
    _install_optional_import_stubs,
    _install_wan_config_filter,
    _patch_qwen3vl_image_features,
)

_install_deepspeed_stub()
_install_optional_import_stubs(motus_root)
_install_wan_config_filter()

from flash_rt.frontends.torch.motus_rtx import MotusTorchFrontendRtx

pipe = MotusTorchFrontendRtx(
    checkpoint_dir=str(checkpoint),
    wan_path=str(wan_path),
    vlm_path=str(vlm_path),
    num_inference_steps=10,
    autotune=0,
)
_patch_qwen3vl_image_features(pipe)

first_frame = torch.load(bundle / "inputs/first_frame.pt", map_location="cpu")
state = torch.load(bundle / "inputs/state.pt", map_location="cpu")
instruction = (bundle / "inputs/instruction.txt").read_text().strip()
t5 = torch.load(bundle / "inputs/t5_embed.pt", map_location="cpu")
vlm = torch.load(bundle / "inputs/vlm_inputs.pt", map_location="cpu")

pipe.set_prompt(
    instruction,
    t5_embeds=t5 if isinstance(t5, list) else [t5],
    vlm_inputs=vlm if isinstance(vlm, list) else [vlm],
)

with torch.no_grad():
    pipe.infer(first_frame, state=state)             # calibration + graph capture
    frames, actions = pipe.infer(first_frame, state=state)  # graph replay
```

---

## 12. Profile switch contract

Use only the top-level profile switch for algorithm A/B tests:

```bash
FLASH_RT_MOTUS_FP4_PROFILE=fast          # validated Stage3 fast profile (default)
FLASH_RT_MOTUS_FP4_PROFILE=off          # explicit FP8 trajectory baseline
FLASH_RT_MOTUS_FP4_PROFILE=fast-cache   # latency-oriented FP4/NVFP4 without tiny-FP8 dispatch
FLASH_RT_MOTUS_FP4_PROFILE=on           # explicit FP4/NVFP4 experiment
```

The `fast` profile enables action/und FFN multi-stream overlap
by default. Set this before Python starts to reproduce the older serial
FFN scheduling:

```bash
FLASH_RT_MOTUS_FFN_MULTI_STREAM=0
```

The TeaCache step cache is orthogonal to the profile switch:

```bash
FLASH_RT_MOTUS_USE_TEACACHE=1           # turn on TeaCache (default off)
FLASH_RT_MOTUS_TEACACHE_SKIP_STEPS=2,3,4,5,6,7,8   # default schedule for 10 steps
```

Avoid mixing the profile switch with low-level kernel flags such as
`FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO` in the same run. Low-level flags
remain available for kernel development, but they are not the
algorithm test interface. The top-level profile already sets the
precision and graph-capture defaults for that run.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named flash_rt_kernels` | Re-run the build and copy `flash_rt*.so` into `flash_rt/`. |
| `ModuleNotFoundError` for Motus/Wan modules | Set `FLASH_RT_MOTUS_ROOT` or pass `--motus-root`. |
| Quickstart reports missing paths | Set `MOTUS_ROOT`, `MOTUS_CHECKPOINT`, `MOTUS_WAN_PATH`, `MOTUS_VLM_PATH`, and `MOTUS_INPUT_BUNDLE`, or pass the matching CLI flags. |
| `--fp4-profile on` does not enable VAE FP4 | Rebuild with `cmake -B build -S . -DGPU_ARCH=120` and confirm the Motus VAE FP4 symbols are present in `flash_rt_kernels`. |
| FP4 on/off appears unchanged | Start a fresh Python process; do not switch profiles after importing the frontend. |
| First call is very slow | Expected: the first `infer()` calibrates and captures the CUDA Graph. Benchmark only later graph replays. |
| OOM during testing | The Stage3 `fast` profile fits in ~28.2 GB peak allocated on the reference 5090. Do not run two Motus full-graph processes in parallel on a 32 GB card, and do not feed inputs larger than the model's trained resolution without expanding the GPU. |
| Cosine changes after editing inputs | Confirm the input bundle follows the upstream Motus E2E contract and uses matching `instruction`, `t5_embed`, `vlm_inputs`, `first_frame`, and `state`. |
| TeaCache wall not dropping | Confirm `FLASH_RT_MOTUS_USE_TEACACHE=1` is set **before** `python` starts; the schedule is baked at install time and cannot be flipped per replay. |
| RTC-lite misses deadlines | Confirm the first chunk is prefetched before starting the controller loop. For the strict `fast` profile at 50 Hz, use the default latency-derived trigger or set `--start-next-at 6`; `--start-next-at 8` is too late on the reference 5090. |
