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

TeaCache is available as an explicit experimental acceleration option:

```python
out = model.infer(
    mode="t2v",
    width=1280,
    height=704,
    frames=121,
    steps=20,
    shift=5.0,
    guide_scale=5.0,
    teacache=True,
    teacache_threshold=0.3,
    teacache_start_step=1,
    teacache_end_step=-1,
    teacache_cache_device="cuda",
    return_metadata=True,
)

print(out["metadata"]["teacache"])
```

`teacache_threshold` controls the speed/quality trade-off. Larger values
skip more DiT steps and may change composition or motion. The Wan2.2 5B
route uses training-free relative-L1 TeaCache without model-specific
coefficients, so treat it as a validation tool until the threshold is
qualified for your prompt class.

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

## Benchmarks

RTX 5090, official Wan2.2-TI2V-5B checkpoint, T2V, `1280x704`,
`frames=121`, `steps=20`, `shift=5.0`, `guide_scale=5.0`,
`sample_solver=unipc`. Timings are `infer_seconds`, which includes text
encoding, DiT sampling, and VAE decode but excludes checkpoint load.

| Path | TeaCache threshold | DiT calls | Time | Peak VRAM | Note |
|---|---:|---:|---:|---:|---|
| FlashRT official pipeline | off | 20/20 | **178.6 s** | 24.37 GiB | baseline |
| FlashRT official pipeline | 0.3 | 8/20 | **114.2 s** | 24.37 GiB | 1.56x faster; visible quality drift on the test prompt |
| Upstream public reference | off | n/a | under 9 min | n/a | Wan2.2 TI2V-5B model-card 720p single consumer GPU reference |

Use `teacache_threshold=0.15`-`0.30` as a starting search range. In local
testing `0.03` did not skip steps on a small smoke run, while `0.3`
skipped aggressively and changed the result. Keep the no-TeaCache output
as the quality reference for each prompt class.

Upstream reference: [Wan2.2-TI2V-5B model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

Keep ComfyUI wall time separate from this official-pipeline timing. ComfyUI
adds graph-node scheduling, model-file repackaging, optional FP8/GGUF/Sage
attention paths, and video-output overhead.
