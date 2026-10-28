# Fixed-state LIBERO pilot

`groot_compare.closed_loop` adds closed-loop task-success evaluation using the pinned Isaac-GR00T policy/environment wrappers and pinned LIBERO assets. It is separate from the offline accuracy and latency experiment. Its elapsed episode times are not inference latency measurements.

The default is the **three tasks whose prompts occur in the shared manifest**, one official fixed initial state per task, eight executed actions per policy call, and at most 720 environment steps per episode. The 720-step budget follows GR00T’s rollout default; native LIBERO’s default is 600. This is a pilot task subset, not a full LIBERO-10 score. `--full-suite` refuses to run until every task has a separate calibration record. `--episodes N --state-start K` selects distinct fixed state indices K through K+N-1; it never wraps indices.

The simulator images used for the original pilot were built and run. A new layer built on the proposed standalone JAL reconstruction remains unverified until that reconstruction and this pilot are rerun. The checked-in results remain separate from all new `artifacts/` output.

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

## Run the paired pilot after offline checks

Prepare the shared observations, matching calibration captures and JAL engines using [REPRODUCE.md](REPRODUCE.md) first. Run backends sequentially, with identical seed, state indices, step budget and rendering mode. Use new output directories for repeats.

```bash
python3 scripts/simulator.py jal -- -m groot_compare.closed_loop \
  --backend torch \
  --manifest /comparison/artifacts/observations/heldout/manifest.json \
  --checkpoint /comparison/artifacts/models/groot-libero-onecam \
  --episodes 1 --state-start 0 --seed 20261026 --cpu-threads 2 \
  --render-backend osmesa \
  --output /comparison/artifacts/closed_loop/torch-pilot

python3 scripts/simulator.py jal -- -m groot_compare.closed_loop \
  --backend trt \
  --engine-dir /comparison/artifacts/jal_trt_optimized_mixed_nvfp4/engines \
  --manifest /comparison/artifacts/observations/heldout/manifest.json \
  --checkpoint /comparison/artifacts/models/groot-libero-onecam \
  --episodes 1 --state-start 0 --seed 20261026 --cpu-threads 2 \
  --render-backend osmesa \
  --output /comparison/artifacts/closed_loop/trt-pilot

python3 scripts/simulator.py flashrt -- -m groot_compare.closed_loop \
  --backend flashrt --capture-dir /comparison/artifacts/captures/heldout \
  --manifest /comparison/artifacts/observations/heldout/manifest.json \
  --checkpoint /comparison/artifacts/models/groot-libero-onecam \
  --episodes 1 --state-start 0 --seed 20261026 --cpu-threads 2 \
  --render-backend osmesa \
  --output /comparison/artifacts/closed_loop/flashrt-pilot
```

`--task-ids` explicitly selects task indices if a smaller smoke test is needed. `--max-episode-steps` is a protocol change and is recorded. A shortened smoke test cannot establish standard task success.

## Pairing evidence

The runner uses LIBERO submodule `8f1084e3132a39270c3a13ebe37270a43ece2a01`, official `.pruned_init` state files, and the five raw zero-action settling steps from the pinned benchmark evaluator before the official history wrapper initializes. Image rotation, state representation, gripper normalization and gripper inversion remain in the official wrappers. The one-camera processor configuration selects the camera from those observations.

Each `closed_loop.json` records task prompts and IDs, source revisions and wrapper hashes, selected and settled initial-state hashes, BDDL/state-file hashes, initial processed-input fingerprints, exact noise and observation hashes per decision, actual executed steps and success. Physical action chunks are stored in adjacent NPZ files. Noise is deterministic by seed/task/state/decision and shared across backends; observations diverge naturally after different actions.

Before comparing success, require completed runs with the same selected tasks, seed, state indices, initialization/settling, render mode, step budget, configurations and initial fingerprints. A mismatch invalidates a paired comparison. Three episodes total per backend only establishes pilot behavior; it does not estimate full-suite accuracy reliably.

The comparison command enforces those initial-state, preprocessing, asset, protocol and shared-noise checks:

```bash
.venv/bin/python -m groot_compare.closed_loop_report \
  --runs artifacts/closed_loop/torch-pilot artifacts/closed_loop/trt-pilot artifacts/closed_loop/flashrt-pilot \
  --output artifacts/reports/closed-loop-pilot.json
```

The frozen offline input protocol remains unchanged so completed runs keep their original hashes. The pilot records its separate rollout settings, task subset, state indices and completion status in each `closed_loop.json`.
