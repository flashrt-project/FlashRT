# GR00T N1.7 LIBERO on Jetson Thor

Run `GR00T-N1.7-LIBERO/libero_10` with FlashRT and compare decoded actions
against a fresh official Isaac GR00T run. This uses the single-camera,
four-step model configuration in the [JAL tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/).

| Configuration | Value |
|---|---|
| Camera | `image`, one current frame |
| Batch size | 1 |
| Denoising steps | 4 |
| Delivered actions | 16 × 7 |
| CPU threads | 2 |
| Timing | 20 warmups, 100 measured calls |

## Requirements

- Jetson AGX Thor with JetPack installed. Docker validation used L4T R39.2.1 / CUDA 13.2.
- For Docker: Docker and NVIDIA Container Toolkit; your account must be able to run Docker.
- For native installation: Python 3.12 and CUDA PyTorch on Thor.

Check the device before starting:

```bash
cat /etc/nv_tegra_release
nvidia-smi
nvpmodel -q
```

For China pip access, optionally run `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` before installation or weight preparation.

## Option A: Docker

### 1. Pull and check the runtime

```bash
export IMAGE=ghcr.io/flashrt-project/flashrt-thor:thor-v0.1.1
export grootS="$PWD/models"
export OUT="$PWD/results/groot"
mkdir -p "$grootS" "$OUT"
docker pull "$IMAGE"
docker run --rm --runtime=nvidia --gpus all "$IMAGE" doctor
```

Expected: `gpu` is `NVIDIA Thor`, capability is `[11, 0]`, and
`fa4_available` is `true`.

### 2. Download the model and backbone

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare groot
```

Both models download from ModelScope with checksum verification. Existing
matching files are reused. The required directories are:

```text
models/
├── GR00T-N1.7-LIBERO/libero_10/
└── Cosmos-Reason2-2B/
```

If both are already complete in `$MODELS`, skip this step.

### 3. Run accuracy checks and the benchmark

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate groot
```

The command selects the `image` camera, uses the local Cosmos backbone and
runs the official reference plus FlashRT FP8/FP4. Original weights remain
unchanged. Reports are saved in `$OUT`. Continue to **Check the results** below.

## Option B: Clone, install and compile

### 1. Install the runtime

Run these commands in a new working directory. If native setup is already
complete, activate its environment, load `runtime.env` and go to step 2.

```bash
git clone --branch docs/thor-release-and-community https://github.com/flashrt-project/FlashRT.git
cd FlashRT/repro/thor
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-dev build-essential cmake ninja-build git git-lfs ffmpeg libglib2.0-0 libgl1

python3.12 -m venv flashrt-venv
source flashrt-venv/bin/activate
python -m pip install 'torch==2.14.0+cu130' 'torchvision==0.29.0+cu130' --index-url https://download.pytorch.org/whl/cu130

export WORK="$PWD/runtime"
bash setup-native.sh
source "$WORK/runtime.env"
export grootS="$PWD/models"
export OUT="$PWD/results/groot"
mkdir -p "$grootS" "$OUT"
```

`setup-native.sh` compiles the tested Thor runtime and installs separate
official reference environments. Keep this shell open for the next steps.

### 2. Prepare the model and local configuration

```bash
bash prepare-models.sh groot
export GROOT="$MODELS/GR00T-local"
if [ ! -d "$GROOT" ]; then
  python local_groot_checkpoint.py \
    --checkpoint "$MODELS/GR00T-N1.7-LIBERO/libero_10" \
    --cosmos "$MODELS/Cosmos-Reason2-2B" --out "$GROOT"
fi
```

Skip the download if both model directories above are complete. The local
configuration selects one camera and links the weights without copying them.

### 3. Run accuracy checks and the benchmark

```bash
bash run-validation.sh groot
```

## Use the FlashRT Thor frontend in Python

This GR00T runtime exposes a lower-level frontend: processor inputs and
normalized state go into model inference, then `denormalize_action` produces
physical actions. The example below uses the bundled observation's prepared
processor inputs, exactly as the supplied validator does for model execution.
`validate groot` additionally runs fresh RGB/state/language preprocessing on
every measured call.

In native mode, keep `GROOT` and `REF` from the setup steps. For Docker,
enter the prepared image and create its local model configuration:

```bash
docker run --rm -it --runtime=nvidia --gpus all --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" bash
export REF=/opt/reference
export GROOT=/results/GR00T-api
if [ ! -d "$GROOT" ]; then
  python local_groot_checkpoint.py \
    --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
    --cosmos /models/Cosmos-Reason2-2B --out "$GROOT"
fi
```

Then run:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$REF/groot-venv/bin/python" - <<'PY'
import os
import numpy as np
import torch
from flash_rt.frontends.torch.groot_n17_thor_fp4 import (
    GrootN17TorchFrontendThorFP4,
)

fixture = torch.load("fixtures/groot-libero10-onecam.pt",
                     map_location="cpu", weights_only=False)
aux = fixture["aux"]
state = {"state." + k: np.asarray(v) for k, v in fixture["state"].items()}

frontend = GrootN17TorchFrontendThorFP4(
    os.environ["GROOT"], num_views=1, embodiment_tag="libero_sim",
)
frontend.set_prompt(aux=aux, prompt="fixture prompt")
torch.set_num_threads(2)

normalized_state = frontend.normalize_state(state)
noise = aux["initial_noise"].to("cuda").bfloat16().contiguous()
frontend._backbone_features = frontend.run_backbone_graph(aux)
normalized_actions = frontend.infer(
    normalized_state, initial_noise=noise,
    num_inference_timesteps=4, action_horizon=40,
)
decoded = frontend.denormalize_action(normalized_actions, state_dict=state)
keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
actions = np.concatenate([np.asarray(decoded[k]) for k in keys], axis=-1)
assert actions.shape == (1, 16, 7) and np.isfinite(actions).all()
print("Actions shape:", actions.shape)
print("First action:", actions[0, 0])
PY
```

- The constructor selects the NVFP4 action-head tier with the FP8 backbone.
  `embodiment_tag="libero_sim"` selects LIBERO's state/action normalization.
- `aux` contains the official processor's pixel patches, token IDs, image grid
  and calibration inputs. `set_prompt(aux=...)` prepares the fixed prompt/grid,
  quantization scales and execution graphs. The text is already encoded in
  `aux`; the string argument alone does not replace that encoded instruction.
  When the instruction or image grid changes, prepare new `aux` and call
  `set_prompt` again.
- `normalize_state` applies the checkpoint statistics. `run_backbone_graph`
  executes the vision/language backbone; assigning its result refreshes the
  features consumed by the four-step action head.
- `infer` returns padded normalized actions. `denormalize_action` applies the
  official physical-action decoding, yielding the **16 × 7** LIBERO chunk.
  The initial noise is fixed here so repeated runs can be compared.

Expected output starts with `Actions shape: (1, 16, 7)`. For a new observation,
prepare fresh processor inputs and state, refresh the backbone, then infer and
decode. The complete raw-input implementation is in
[`verify_groot_fixture.py`](../../repro/thor/verify_groot_fixture.py): its
`prepare()` calls the official processor and its `run()` executes these same
backbone, action-head and decode steps. Use that validator for the reported
model/preprocessing/complete-call timings.

## Check the results

Successful validation ends with `combined accuracy status: 0`. Run this
on the host, using the same `$OUT` as above:

```bash
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for tier in ('fp8', 'fp4'):
    report = json.loads((out / f'groot-{tier}.json').read_text())
    assert report['passed'], f'{tier}: numerical check failed'
    parts = report['component_medians_ms']
    print(f"{tier}: PASS, cosine={report['mean_sample_cosine']:.6f}, "
          f"model={parts['model_ms']:.2f} ms, "
          f"preprocessing={parts['data_processing_ms']:.2f} ms, "
          f"complete={report['p50_ms']:.2f} ms")
    print('Strict action-group diagnostic:', report['strict_group_diagnostic_passed'])
PY
```

Expected results from the public image:

| Tier | Mean cosine | Model inference | Preprocessing | Complete call |
|---|---:|---:|---:|---:|
| FP8 | 0.999467 | about 37.3 ms | about 3.2 ms | about 40.9 ms |
| FP4 | 0.999694 | about 24.3 ms | about 3.3 ms | about 27.9 ms |

Model inference includes input transfer, vision/backbone and the four-step
action head. Complete call also includes fresh CPU preprocessing and physical
action decoding. Loading, calibration and graph setup occur before timing.
Latency varies with power mode and clocks.

Passing requires mean physical-action cosine ≥0.999, worst-sample cosine
≥0.995, and repeated normalized outputs with cosine ≥0.9999 / maximum
absolute error ≤0.05. Per-group diagnostics are saved separately: the recorded
strict diagnostic is **false** for both tiers. FP4 rotation cosine is 0.96167,
with maximum absolute error 0.00760. Read these errors in `action_groups`;
an overall PASS does not mean every group passed. This is a fixed-observation
check, not a robot task-success test. [Recorded results](results.md).

If validation fails, check `$OUT/groot-capture.log` and
`$OUT/groot-fp8.log` / `groot-fp4.log`. If the backbone cannot be found,
check that `Cosmos-Reason2-2B` is complete and rerun local configuration setup
with a new `GROOT` directory.
