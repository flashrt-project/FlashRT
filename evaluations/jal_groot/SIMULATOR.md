# Two-camera LIBERO-10 task evaluation

Use the checkpoint's `image` **and** `wrist_image` cameras. The original one-camera pilot is historical; its latency and offline results stay unchanged. This follow-up runs all ten tasks, 50 official initial states each, over ten seeds. Episode wall time is not inference latency.

## Isolated simulator images

The dependencies follow [Isaac-GR00T's pinned LIBERO setup](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/gr00t/eval/sim/LIBERO/setup_libero.sh). The simulator layer retains each inference image's Torch, NumPy, preprocessing, TensorRT and ModelOpt versions. Installation uses `--no-deps` and checks protected package versions before and after. OpenCV is supplied by the existing headless wheel; training/notebook packages are unnecessary for this runner. The resulting versions are recorded inside each image at `/opt/libero/runtime-packages.json`.

This deliberately does not run the upstream full setup script: that script creates a separate Torch 2.9 environment and rewrites LIBERO configuration. Here sources, assets, caches and configuration remain within this project or the derived images.

From `evaluations/jal_groot`, after preparing the inference runtimes described in [REPRODUCE.md](REPRODUCE.md):

```bash
git -C vendor/Isaac-GR00T submodule update --init --depth 1 external_dependencies/LIBERO

docker build --network=host -f scripts/Dockerfile.libero \
  -t groot-comparison-jal-libero:local scripts

docker build --network=host -f scripts/Dockerfile.libero \
  --build-arg BASE_IMAGE=ghcr.io/flashrt-project/flashrt-thor@sha256:87ad7bce8635207425ac5e90a61d3d83c0ba1042fa28eef7d067474fee872c5c \
  --build-arg SIM_PYTHON=/opt/reference/groot-venv/bin/python \
  -t groot-comparison-flashrt-libero:local scripts

python3 scripts/simulator.py --cpu-only jal -- scripts/check_simulator_imports.py
python3 scripts/simulator.py --cpu-only flashrt -- scripts/check_simulator_imports.py
```

The CPU checks import the simulator and policy wrappers, enumerate the ten tasks, and assert CUDA was not initialized. They do not construct a simulator or load a model. The launcher disables GPU access with the `runc` runtime for these checks. Runtime network access is disabled in all launches. Simulator rendering defaults to OSMesa software rendering; this avoids renderer GPU competition but makes episode wall time unsuitable for inference performance claims.

## Preserve both checkpoint cameras

Reuse the model files, local backbone, dataset and inference runtimes prepared in [REPRODUCE.md](REPRODUCE.md); no second weight download is needed. From `evaluations/jal_groot`, create a separate metadata overlay that preserves both cameras:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
models = Path('artifacts/models')
source = models / 'GR00T-N1.7-LIBERO/libero_10'
overlay = models / 'groot-libero-twocam'
overlay.mkdir()  # Refuse to overwrite an existing experiment configuration.
(models / 'nvidia').mkdir(exist_ok=True)
backbone = models / 'nvidia/Cosmos-Reason2-2B'
if not backbone.exists():
    backbone.symlink_to('../Cosmos-Reason2-2B', target_is_directory=True)
for path in source.iterdir():
    if path.name not in {'config.json', 'processor_config.json', '.cache'}:
        (overlay / path.name).symlink_to(os.path.relpath(path, overlay), target_is_directory=path.is_dir())
local_backbone = '/comparison/artifacts/models/nvidia/Cosmos-Reason2-2B'
config = json.loads((source / 'config.json').read_text())
assert config['num_inference_timesteps'] == 4
config['model_name'] = local_backbone
(overlay / 'config.json').write_text(json.dumps(config, indent=2))
processor = json.loads((source / 'processor_config.json').read_text())
video = processor['processor_kwargs']['modality_configs']['libero_sim']['video']
assert video['modality_keys'] == ['image', 'wrist_image']
assert video['delta_indices'] == [0]
processor['processor_kwargs']['model_name'] = local_backbone
(overlay / 'processor_config.json').write_text(json.dumps(processor, indent=2))
PY
```

## Build two-camera JAL engines

Keep the two-camera overlay above. One-camera engines and calibration captures cannot be reused. To match the follow-up calibration recipe, use ten observations from the bundled official `libero_demo` (rather than the original report's IPEC calibration cohort):

```bash
python3 scripts/container.py jal -- \
  vendor/Isaac-GR00T/scripts/deployment/build_trt_pipeline.py \
  --model-path /comparison/artifacts/models/groot-libero-twocam \
  --dataset-path /comparison/vendor/Isaac-GR00T/demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --output-dir /comparison/artifacts/jal_trt_twocam \
  --execution-profile optimized --quantization mixed_nvfp4 \
  --calib-size 10 --batch-size 1 --steps export,build
```

## Full-suite run and check

Run from `evaluations/jal_groot`, with the simulator images and assets prepared above. The runner uses the pinned official policy/environment wrappers for images, state and gripper handling. It captures one separate, non-scored raw initial-state observation per task for FlashRT calibration, then reuses those ten captures for every round. JAL scales are already frozen in its engines; these captures only provide its warmup inputs.

```bash
checkpoint=/comparison/artifacts/models/groot-libero-twocam
source=/comparison/vendor/Isaac-GR00T
captures=/comparison/artifacts/captures/libero-suite
runs=/comparison/artifacts/libero-suite
engines=/comparison/artifacts/jal_trt_twocam/engines

python3 scripts/simulator.py flashrt -- -m groot_compare.libero_suite calibrate \
  --output "$runs/setup" --checkpoint "$checkpoint" --source-dir "$source" \
  --calibration-dir "$captures" --seed 20261026

for seed in 20261026 20261027 20261028 20261029 20261030 \
            20261102 20261103 20261104 20261105 20261106; do
  for task in {0..9}; do
    for backend in torch trt flashrt; do
      runtime=jal
      [ "$backend" = flashrt ] && runtime=flashrt
      python3 scripts/simulator.py "$runtime" -- -m groot_compare.libero_suite \
        "$backend" "$task" --output "$runs/seed-$seed" --seed "$seed" \
        --checkpoint "$checkpoint" --source-dir "$source" \
        --calibration-dir "$captures" --engine-dir "$engines" --render-backend egl
    done
  done
  python3 scripts/simulator.py --cpu-only flashrt -- -m groot_compare.verify_suite \
    "$runs/seed-$seed" "$seed"
done
```

Run on an idle GPU, or serialize these commands with your existing GPU lock. Keep all three backends for a given seed together. For two devices, split the seed list into disjoint groups; do not evaluate the same seed twice. Capture calibration once and reuse the same files and frozen engines on both devices.

Each backend/task performs **20 warmup calls**, then scores all 50 provided `.pruned_init` states. Settings are batch one, two cameras with one current frame each, **four denoising steps**, 16 returned actions with **eight executed per decision**, **five zero-action settling steps**, and **720 control steps** per episode. Settling is outside the control budget. State/decision-derived BF16-rounded noise is shared across backends; weights and calibration are fixed.

Each seed must produce `verified-summary.json` with `verified: true`, **1,500 episodes**, and **500 unique task/state keys per backend**. The verifier recomputes noise hashes, checks finite saved actions and paired initial states/processed inputs, and records bounded one-LSB EGL rendering differences. A task failure counts as a scored failure; a setup/runtime error prevents completion. An interrupted task can resume from saved episodes; completed tasks are not silently rerun. No videos are saved.

## Check the published ten-round counts without a GPU

```bash
python3 results/verify_libero_counts.py
```

This recomputes totals and sample standard deviations from [libero-ten-rounds.json](results/libero-ten-rounds.json). It checks the published counts, not the private raw rollout traces. The file also records device/runtime allocation and calibration provenance. Our measured follow-up used existing PyTorch 2.12/2.9 and TensorRT 10.16 environments, not the original NVIDIA PyTorch 2.10/TensorRT 10.15 environment; fresh-image equivalence is still a separate validation.
