# Wan2.2 TI2V-5B Usage

This page documents the FlashRT official-pipeline route for
Wan2.2-TI2V-5B on RTX SM120.

FlashRT exposes a stable `set_prompt()` / `infer()` API around the official
Wan Python pipeline. ComfyUI support is handled outside the core repository
through custom nodes that call FlashRT public APIs.

## Requirements

Use the original Wan2.2-TI2V-5B checkpoint layout:

```text
Wan2.2-TI2V-5B/
  diffusion_pytorch_model-00001-of-00003.safetensors
  diffusion_pytorch_model-00002-of-00003.safetensors
  diffusion_pytorch_model-00003-of-00003.safetensors
  diffusion_pytorch_model.safetensors.index.json
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
  google/umt5-xxl/
```

The official Wan Python package must be importable. Either install the Wan
source package, add it to `PYTHONPATH`, or set one of:

```bash
export FLASH_RT_WAN22_ROOT=/path/to/Wan2.2/source
export MOTUS_ROOT=/path/to/Motus        # works when Wan is vendored in bak/wan
```

## API

```python
import flash_rt

model = flash_rt.load_model(
    "/path/to/Wan2.2-TI2V-5B",
    framework="torch",
    config="wan22_ti2v_5b",
    hardware="rtx_sm120",
)

model.set_prompt(
    "A cinematic shot of a blue sphere rolling across a wooden table",
)

out = model.infer(
    mode="t2v",
    width=832,
    height=480,
    frames=81,
    steps=20,
    shift=5.0,
    guide_scale=5.0,
    seed=1234,
    return_metadata=True,
)

video = out["video"]       # torch.Tensor [C, F, H, W]
metadata = out["metadata"]
```

For image-to-video:

```python
model.set_prompt("A handheld camera shot, smooth motion")
video = model.infer(
    mode="i2v",
    image=start_image,     # PIL.Image.Image
    width=832,
    height=480,
    frames=81,
    steps=20,
    shift=5.0,
    guide_scale=5.0,
)
```

`model.predict()` is not part of the Wan API because `predict()` is the VLA
action-output convenience wrapper.

## Recommended Baselines

Community 480p performance baseline:

```text
width=832
height=480
frames=81
steps=20
shift=5.0
guide_scale=5.0
sample_solver=unipc
```

Official quality baseline:

```text
width=1280
height=704
frames=121
steps=20 or 50
shift=5.0
guide_scale=5.0
sample_solver=unipc
```

Keep ComfyUI wall time separate from this official-pipeline timing. ComfyUI
adds graph-node scheduling, model-file repackaging, optional FP8/GGUF/Sage
attention paths, and video-output overhead.
