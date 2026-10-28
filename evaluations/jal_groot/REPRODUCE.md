# Reproducing the matched comparison

The [results](README.md) were measured on a Thor T5000 with JetPack 7.2 / L4T 39.2.1, MAXN, fixed clocks and an otherwise idle GPU. This directory is the harness project root: run every command below from `evaluations/jal_groot`, not from the FlashRT repository root. Containers mount it at `/comparison`; that path is a container convention and does not depend on the host checkout location.

The inference adapters, runner, report code and helper scripts are copied unchanged from the completed experiment. The 45 CPU tests were rerun after packaging. No GPU experiment was rerun for this branch.

## Check the published results without a GPU

Only Python and NumPy are needed to check the supplied data. This does not download model assets, start containers or run inference:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . pytest
sha256sum --check results/SHA256SUMS
.venv/bin/python results/verify.py
```

The verifier checks original report and measured-code hashes, recomputes all offline error metrics from `results/action-predictions.csv`, and checks latency medians/p95 against the 1,800 individual samples in `results/latency-samples.json`. CSV values round-trip the original float32 predictions; there are 16 predicted actions per observation for every backend. Ground-truth columns are populated only for the 1,410 executed action steps. A blank ground-truth cell means the step was not scored, not a zero action. The CSV contains physical action outputs, not camera images or model parameters.

This verifies the published data's internal consistency. It does not independently re-execute the models or replay the simulation pilot. Continue below to prepare a new hardware experiment.

## What has and has not been reproduced

The measured JAL image was a local image derived from an existing official GR00T Thor environment with ModelOpt 0.39.0 installed. Its relevant package versions are recorded in [scripts/jal-runtime-constraints.txt](scripts/jal-runtime-constraints.txt). The exact measured image is not distributed here.

[Dockerfile.jal-repro](scripts/Dockerfile.jal-repro) is a **proposed reconstruction, inspected but never built or run**. It uses public pinned sources, a CUDA image digest, the frozen upstream dependency lock and measured Python versions. Availability of its pinned apt packages and equivalence to the measured runtime are unverified. A successful build and matching versions alone would not establish numerical or timing equivalence: repeat the matched evaluations before publishing new results. This is not a verified one-command fresh-machine reproduction.

The FlashRT image is public and pinned by digest. Its source revision is `e4bd9277521d137fcded8fa5ccb6091c05f5cd4b`; the launcher runs the FlashRT installed inside that image. Editing this fork's FlashRT source does **not** modify the pinned container runtime. A new image/source comparison is a separate experiment.

Weights, datasets, calibration captures, compiled engines and caches are intentionally excluded from Git. [configs/protocol.json](configs/protocol.json) freezes the original offline settings. Its `closed_loop.status` describes that offline protocol; the later pilot has a separate [recipe](SIMULATOR.md) and report.

## Prepare sources and model assets

Prerequisites are an aarch64 Thor host with Docker/NVIDIA runtime, Git LFS, Python, and the current `hf` CLI. Model access is subject to the upstream repositories' terms. Authenticate with `hf auth login` if required. Downloads happen on the host; workers use offline model access.

```bash
mkdir -p vendor artifacts/models artifacts/datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git vendor/Isaac-GR00T
git -C vendor/Isaac-GR00T checkout --detach 9c7e746b2cd37a810070a98ef41d290a07e806c2
git -C vendor/Isaac-GR00T lfs pull --include='demo_data/libero_demo/**' --exclude=''
curl -fL https://www.jetson-ai-lab.com/code-samples/groot_n17_on_thor/groot-n17-trt-optimization.patch -o artifacts/jal.patch
printf '%s\n' '6a818bfcbb149c0f0efa8567ecc0fd6909613bc53e00e08f66bb81c8c601d5a3  artifacts/jal.patch' | sha256sum --check -
git -C vendor/Isaac-GR00T apply --check ../../artifacts/jal.patch
git -C vendor/Isaac-GR00T apply ../../artifacts/jal.patch

hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include 'libero_10/*' --local-dir artifacts/models/GR00T-N1.7-LIBERO
hf download nvidia/Cosmos-Reason2-2B \
  --revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --local-dir artifacts/models/Cosmos-Reason2-2B
hf download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --type dataset \
  --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 \
  --local-dir artifacts/datasets/libero_10
```

Check every recorded model/config/tokenizer file before using the shared weight manifest. The runner binds to the manifest's hash; it does not independently rehash all weight payloads on every invocation.

```bash
python3 - <<'PY'
import hashlib, json, shutil
from pathlib import Path
manifest = Path('results/weight-hashes.json')
records = json.loads(manifest.read_text())
for record in records:
    path = Path(record['path'])
    assert path.stat().st_size == record['size'], path
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == record['sha256'], path
shutil.copyfile(manifest, 'artifacts/weight-hashes.json')
print(f'Verified {len(records)} model/config/tokenizer files')
PY
```

Create the same one-camera checkpoint overlay. The downloaded checkpoint normally uses `image` and `wrist_image`; the JAL comparison selects only `image` for **all three backends**, including the PyTorch reference. Weight files remain unchanged. The `nvidia/Cosmos-Reason2-2B` path spelling is significant to upstream backbone selection.

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
models = Path('artifacts/models')
source = models / 'GR00T-N1.7-LIBERO/libero_10'
overlay = models / 'groot-libero-onecam'
overlay.mkdir()  # Refuse to overwrite an existing experiment configuration.
(models / 'nvidia').mkdir(exist_ok=True)
(models / 'nvidia/Cosmos-Reason2-2B').symlink_to('../Cosmos-Reason2-2B', target_is_directory=True)
for path in source.iterdir():
    if path.name not in {'config.json', 'processor_config.json', '.cache'}:
        (overlay / path.name).symlink_to(os.path.relpath(path, overlay), target_is_directory=path.is_dir())
local_backbone = '/comparison/artifacts/models/nvidia/Cosmos-Reason2-2B'
config = json.loads((source / 'config.json').read_text())
assert config['num_inference_timesteps'] == 4
config['model_name'] = local_backbone
(overlay / 'config.json').write_text(json.dumps(config, indent=2))
processor = json.loads((source / 'processor_config.json').read_text())
processor['processor_kwargs']['modality_configs']['libero_sim']['video']['modality_keys'] = ['image']
processor['processor_kwargs']['model_name'] = local_backbone
(overlay / 'processor_config.json').write_text(json.dumps(processor, indent=2))
PY
```

Relative symlinks keep the overlay usable at any host checkout location; the explicit model path resolves inside `/comparison`.

## Prepare runtimes and run CPU checks

Run the harness checks in a host virtual environment. They use fake backends where appropriate, do not load the model and do not establish GPU correctness.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . pytest
.venv/bin/python -m pytest -q

docker pull ghcr.io/flashrt-project/flashrt-thor@sha256:87ad7bce8635207425ac5e90a61d3d83c0ba1042fa28eef7d067474fee872c5c

# UNVERIFIED reconstruction: this build was not performed for the published results.
docker build -f scripts/Dockerfile.jal-repro -t groot-comparison-jal:repro .
```

If that reconstruction builds and its runtime has been inspected, give it the local tag expected by the unchanged launchers. First ensure this will not replace an image you need to preserve. The tag is a launcher convention, not proof that this is the original measured image.

```bash
docker tag groot-comparison-jal:repro groot-comparison-jal:local
```

Record the actual image IDs, package versions, JetPack version, power mode and clock settings for a new run. The original and newly reconstructed image results must remain distinguishable. All remaining commands produce **new** artifacts in the ignored `artifacts/` directory; they never overwrite the published `results/` evidence. For another repetition, use a fresh evaluation workspace or archive its entire `artifacts/` directory before starting. Do not overwrite prior captures, runs or timing plans.

## Build JAL engines and export held-out observations

Engine compilation is a GPU workload. Arrange the intended idle GPU and power/clock conditions before it as well as before timing; on a shared host, preserve and later restore any applications and settings you change. These scripts do not manage host services or power modes.

```bash
python3 scripts/container.py jal -- \
  vendor/Isaac-GR00T/scripts/deployment/build_trt_pipeline.py \
  --model-path /comparison/artifacts/models/groot-libero-onecam \
  --dataset-path /comparison/vendor/Isaac-GR00T/demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --output-dir /comparison/artifacts/jal_trt_optimized_mixed_nvfp4 \
  --execution-profile optimized --quantization mixed_nvfp4 \
  --calib-dataset-path /comparison/artifacts/datasets/libero_10 \
  --calib-size 10 --batch-size 1 --steps export,build

COMPARE_DATA_ID='IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot@e1a223d30b896c1613f270a2bfc63d382b3de7e1'
python3 scripts/container.py --cpu-only jal -- -m groot_compare.dataset \
  --dataset-path /comparison/artifacts/datasets/libero_10 \
  --calibration-dataset-path /comparison/artifacts/datasets/libero_10 \
  --checkpoint-path /comparison/artifacts/models/groot-libero-onecam \
  --output-dir /comparison/artifacts/observations/heldout \
  --trajectory-ids 1 2 3 4 22 --calibration-trajectory-ids 18 5 34 \
  --execution-horizon 8 --steps 300 \
  --dataset-identity "$COMPARE_DATA_ID" --calibration-dataset-identity "$COMPARE_DATA_ID" \
  --reserved-calibration-episodes 0 \
  --reserved-calibration-dataset-identity "$COMPARE_DATA_ID"
```

JAL calibration remains episode 0, frames 0–9. FlashRT uses the first exported frame of episodes 18, 5 and 34, one per prompt. The primary cohort excludes all those episodes and contains 178 observations from episodes 1, 2, 3, 4 and 22. The exporter additionally rejects duplicate camera/state observations between calibration and evaluation. The original JAL demo episodes 0–4 overlap calibration and are not the primary held-out cohort.

## Capture calibration and run offline accuracy

Calibration captures are generated with the official model, before the held-out evaluation. FlashRT then processes each new image/state itself. Prompt-specific calibration and graph setup are outside the steady-state timer and must be repeated for unseen instructions.

```bash
python3 scripts/container.py flashrt -- -m groot_compare.run capture \
  --manifest /comparison/artifacts/observations/heldout/manifest.json \
  --checkpoint /comparison/artifacts/models/groot-libero-onecam \
  --capture-dir /comparison/artifacts/captures/heldout --cpu-threads 2

for backend in torch trt flashrt; do
  runtime=jal
  extra=()
  if [ "$backend" = trt ]; then
    extra=(--engine-dir /comparison/artifacts/jal_trt_optimized_mixed_nvfp4/engines)
  elif [ "$backend" = flashrt ]; then
    runtime=flashrt
    extra=(--capture-dir /comparison/artifacts/captures/heldout)
  fi
  python3 scripts/container.py "$runtime" -- -m groot_compare.run run \
    --backend "$backend" --phase accuracy \
    --manifest /comparison/artifacts/observations/heldout/manifest.json \
    --checkpoint /comparison/artifacts/models/groot-libero-onecam \
    --output "/comparison/artifacts/runs/heldout-${backend}-accuracy" \
    --cpu-threads 2 --warmup 20 --iters 100 --seed 20261026 "${extra[@]}"
done

.venv/bin/python -m groot_compare.report \
  --reference artifacts/runs/heldout-torch-accuracy \
  --candidates artifacts/runs/heldout-trt-accuracy artifacts/runs/heldout-flashrt-accuracy \
  --output artifacts/reports/heldout-accuracy.json
```

Those are fresh-run directory names. The original completed TRT accuracy run was named **`heldout-trt-accuracy-v2`** after fixing an inference-buffer update error in the adapter; only that completed run contributed to the published results. The packaged adapter already includes that fix. Do not substitute the original failed `heldout-trt-accuracy` debug run when auditing old artifacts.

## Measure three paired timing rounds

Arrange an idle GPU, MAXN and fixed clocks as in the recorded hardware configuration. `--exclusive-gpu` is an operator assertion, not automatic workload isolation. The helper rejects any running Docker containers but does not detect all non-container GPU workloads. Preserve and restore prior host settings and applications using the host's own management procedure.

```bash
python3 scripts/run_latency.py --exclusive-gpu

.venv/bin/python -m groot_compare.report \
  --reference artifacts/runs/heldout-torch-accuracy \
  --candidates artifacts/runs/heldout-trt-accuracy artifacts/runs/heldout-flashrt-accuracy \
  --timing-round artifacts/runs/heldout-trt-timing-r1 artifacts/runs/heldout-flashrt-timing-r1 \
  --timing-round artifacts/runs/heldout-trt-timing-r2 artifacts/runs/heldout-flashrt-timing-r2 \
  --timing-round artifacts/runs/heldout-trt-timing-r3 artifacts/runs/heldout-flashrt-timing-r3 \
  --output artifacts/reports/comparison.json
```

Round orders are TRT → FlashRT, FlashRT → TRT, TRT → FlashRT. Each round uses two CPU threads, 20 warmups and 100 measured calls per prompt, giving 900 samples per backend. The report validates paired input/noise/configuration fingerprints and binds every timing round to the accuracy evidence. The latency plan refuses to overwrite an existing plan.

Timing covers raw observations through decoded physical actions on the CPU, with GPU synchronization. Preprocessing, transfers and decoding are included; disk loading, noise generation, model loading, calibration, graph capture and warmup are excluded. The noise transfer is included. This boundary differs from JAL's sum of separately timed stages. The two optimized backends must be compared using this same runner.

For the separate pilot, continue with [SIMULATOR.md](SIMULATOR.md). Neither offline numerical closeness nor the small, non-discriminating 0/3 pilot establishes full-suite task-success equivalence. There is no agreed numerical acceptance margin; the reports state raw errors rather than accuracy pass/fail.
