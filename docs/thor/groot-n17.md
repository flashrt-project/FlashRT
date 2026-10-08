# GR00T N1.7 LIBERO on Jetson AGX Thor

Run FlashRT on `GR00T-N1.7-LIBERO/libero_10` with one camera, batch 1 and four
denoising steps. These are the model/input settings used by
[Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/).
The model's internal action horizon is 40; the LIBERO processor returns 16 actions,
each containing seven robot action dimensions.

## 1. Set up the runtime

Use JetPack 7.2 / CUDA 13. Check `nvidia-smi` and use MAXN/fixed clocks for timing:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Choose either route below.

### A. Pull the FlashRT image

On the Thor host:

```bash
export IMAGE=ghcr.io/flashrt-project/flashrt-thor:thor-v0.2.0
mkdir -p models results
docker pull "$IMAGE"
docker run --rm -it --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$PWD/models:/models" -v "$PWD/results:/results" "$IMAGE"
```

Inside the container:

```bash
cd /opt/FlashRT
export MODELS=/models OUT=/results
export REF=/opt/reference
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

The image contains the actual FlashRT source, compiled kernels, input samples
and separate official reference environments. Model weights stay in your mounted
`models` directory. No login is required for the public image.

### B. Clone, install and compile

Use a new checkout and environment. The host must already have CUDA PyTorch and its matching torchvision
for Jetson Thor and the CUDA 13 toolkit installed.

```bash
git clone https://github.com/flashrt-project/FlashRT.git
cd FlashRT
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export CUTE_DSL_ARCH=sm_101a
# Optional in China: export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
python -c "import torch, torchvision; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() == (11, 0)"
python -m pip install numpy==1.26.4 safetensors==0.8.0 sentencepiece pillow pybind11 ninja \
  nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 huggingface-hub modelscope==1.40.0 modelscope-hub==0.4.2 requests
python -m pip install --no-deps --no-build-isolation -e .
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -S . -B build -DGPU_ARCH=110 -DCMAKE_CUDA_ARCHITECTURES=OFF \
  -DCMAKE_BUILD_TYPE=Release -DFLASHRT_ENABLE_PI05_THOR=ON
cmake --build build -j2 --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided flash_rt_pi05_thor
python -c "from flash_rt import flash_rt_kernels, flash_rt_fp4, flash_rt_pi05_thor"
export PYTHONPATH="$PWD"
export LINGBOT_FA4_SRC="$PWD/csrc/attention/flash_attn_4_src"
```

The last import must succeed. The optional Pi0.5 kernel target is Thor-only;
it leaves the shared kernel module and other hardware builds unchanged.

For source installation, install the official GR00T reference code separately:

```bash
bash docker/install-reference.sh "$PWD/.reference" groot
export REF="$PWD/.reference" MODELS="$PWD/models" OUT="$PWD/results"
mkdir -p "$MODELS" "$OUT"
```

Use `"$REF/groot-venv/bin/python"` for both the reference and FlashRT commands:
the official processor is needed to turn camera images into model inputs and
decode physical robot actions. OpenPI and GR00T require different Transformers
versions, so they have separate reference environments.

## 2. Download weights and prepare a local config

If the checkpoints already exist, set the variables below and skip downloading.
The default source is NVIDIA's Hugging Face repositories. Accept the
[Cosmos license](https://huggingface.co/nvidia/Cosmos-Reason2-2B), then log in:

```bash
hf auth login
hf download nvidia/Cosmos-Reason2-2B --local-dir "$MODELS/Cosmos-Reason2-2B"
hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/*" \
  --local-dir "$MODELS/GR00T-N1.7-LIBERO"
```

Only if Hugging Face is unavailable, use the ModelScope backup:

```bash
python - <<'PY'
import os
from modelscope import snapshot_download
root = os.environ["MODELS"]
snapshot_download("nv-community/Cosmos-Reason2-2B", local_dir=root + "/Cosmos-Reason2-2B")
snapshot_download("nv-community/GR00T-N1.7-LIBERO", local_dir=root + "/GR00T-N1.7-LIBERO",
                  allow_file_pattern=["libero_10/config.json", "libero_10/processor_config.json",
                                      "libero_10/statistics.json", "libero_10/embodiment_id.json",
                                      "libero_10/model*.safetensors", "libero_10/model.safetensors.index.json"])
PY
```

Prepare the same one-camera config for both backends:

```bash
export CHECKPOINT="$MODELS/GR00T-N1.7-LIBERO/libero_10"
export COSMOS="$MODELS/Cosmos-Reason2-2B"
export GROOT="$MODELS/groot-libero-onecam"
"$REF/groot-venv/bin/python" examples/thor/groot.py --mode prepare \
  --checkpoint "$CHECKPOINT" --cosmos "$COSMOS" --output "$GROOT"
```

`prepare` creates a separate config pointing to the local Cosmos backbone and
selects the single `image` camera. Weight files are linked, not duplicated or
modified. Both the official policy and FlashRT use this same config. If you have
already prepared it, reuse `GROOT` instead of running `prepare` again.

## 3. Generate a fresh reference and verify FlashRT

```bash
"$REF/groot-venv/bin/python" examples/thor/groot.py --checkpoint "$GROOT" \
  --mode reference --cpu-threads 2 --output "$OUT/groot-reference.json"
"$REF/groot-venv/bin/python" examples/thor/groot.py --checkpoint "$GROOT" \
  --mode fp8 --cpu-threads 2 --reference "$OUT/groot-reference.pt" --output "$OUT/groot-fp8.json"
"$REF/groot-venv/bin/python" examples/thor/groot.py --checkpoint "$GROOT" \
  --mode fp4 --cpu-threads 2 --reference "$OUT/groot-reference.pt" --output "$OUT/groot-fp4.json"
```

The bundled `examples/thor/assets/groot_libero.npz` contains one raw LIBERO camera
frame, state and instruction. It contains no precomputed model features or
reference actions. The first command runs official GR00T and captures calibration
tensors and diffusion noise once into your result directory. The next commands
verify that the processor produces identical pixel patches, tokens and image grid,
then run FlashRT with that same noise.

Image preprocessing is performed again on every measured call. Only the fixed
instruction/grid setup is done once. Each report separates CPU preprocessing,
model inference, action decoding and the complete call. Twenty warmups precede
100 timed calls.

## 4. Check the result

The reference command must succeed. Both FlashRT commands must exit successfully
and report `passed: true`:

- Physical actions: `[1, 16, 7]`, all finite.
- Combined action cosine ≥ 0.999 and worst sample cosine ≥ 0.995.
- Repeated normalized outputs: cosine ≥ 0.9999 and maximum absolute error ≤ 0.05.
- Expected FP4 timing: approximately 23–25 ms model, 3–5 ms preprocessing,
  and 28–30 ms complete call. FP8 complete-call timing is approximately 41–45 ms.

The reports also include position, rotation and gripper errors. The rotation
cosine diagnostic can fall below 0.995 on this sample's near-zero rotations;
`strict_group_diagnostic_passed` reports that separately. It is not silently
counted as a pass. Fixed-input numerical checks do not measure robot task success.

`--cpu-threads 2` limits PyTorch's CPU worker overhead in the official image
processor. It preserves the processor's calculations and output values. The
controlled native check reduced preprocessing from 5.86 to 4.68 ms and the
complete call from 30.19 to 29.00 ms; this was measured on the same loaded model.

## How the FlashRT code runs

```python
import flash_rt

policy = flash_rt.load_model(
    checkpoint_dir, config="groot_n17", framework="torch", hardware="thor",
    num_views=1, embodiment_tag="libero_sim", use_fp4=True,
)
frontend = policy.pipeline
frontend.set_prompt(aux=reference_capture["aux"], prompt=instruction)
# For each frame: use the official processor to obtain pixels and normalized state.
frontend._backbone_features = frontend.run_backbone_graph(
    {"pixel_values": processed["pixel_values"].to("cuda")}
)
normalized = frontend.infer(
    processed["state"].to("cuda"), initial_noise=noise,
    num_inference_timesteps=4, action_horizon=40,
)
actions = frontend.denormalize_action(normalized, state_dict=robot_state)
```

`load_model` selects the FP8 backbone and NVFP4 action head. `set_prompt` uses
the captured official tensors for quantization setup and graph capture. Each
`run_backbone_graph` processes the current image; `infer` executes four denoising
steps; `denormalize_action` restores physical action units.

GR00T uses this documented frontend interface through `policy.pipeline`, rather
than the image-list `predict` interface used by Pi0.5. See `examples/thor/groot.py`
for the full processor setup and running loop. No official model forward occurs
inside the FlashRT timing loop.

JAL reports about 40 ms for TensorRT's complete pipeline. Compare complete calls
with complete calls, and model latency with model latency. The TensorRT engine
has not yet been run alongside FlashRT on the same device, so these instructions
do not claim a measured same-device TensorRT speedup.

To build this same image yourself, see [Docker](../../docker/README.md#thor).
