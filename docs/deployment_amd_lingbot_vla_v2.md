# LingBot-VLA-v2 on AMD RDNA

FlashRT runs LingBot-VLA-v2-6B with BF16 weights, native HIP operators and
HIP Graph replay. The batch-1 pipeline includes Qwen3-VL vision/language
prefill and the MoE action expert. For the build and hardware routing, see
[deployment_amd.md](deployment_amd.md).

## Hardware and precision

| Target | Precision | Validation |
|---|---|---|
| gfx1100 / W7900D | BF16 | Extension compiled; full model not yet validated |
| gfx1151 / Radeon 8060S, Strix Halo | BF16 | Operators and full model validated |
| gfx1201 / Radeon AI PRO R9700 | BF16 | Extension compiled; full model not yet validated |

Explicit hardware selection checks the device and extension before loading
weights. FP8 and other precision tiers are not supported by this frontend.

## Setup and usage

Use ROCm PyTorch, matching torchvision, transformers with Qwen3-VL support,
safetensors and PyYAML. Build the RDNA extension for the selected device:

```bash
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1151
```

Download the [LingBot-VLA-v2-6B checkpoint](https://huggingface.co/robbyant/lingbot-vla-v2-6b):

```bash
hf download robbyant/lingbot-vla-v2-6b --local-dir /path/to/lingbot-vla-v2-6b
```

The default RoboTwin inference settings are included in
[lingbot_vla_v2.yaml](../flash_rt/configs/lingbot_vla_v2.yaml), under `policy`.
The Qwen3-VL configuration and processor load from `Qwen/Qwen3-VL-4B-Instruct`
through the Hugging Face cache; no separate Qwen model weights are needed.
FP32 checkpoint tensors are converted directly to BF16 during loading.

```python
import torch
import flash_rt

torch.set_num_threads(4)
model = flash_rt.load_model(
    "/path/to/lingbot-vla-v2-6b",
    config="lingbot_vla_v2",
    hardware="amd_rdna35",
    precision="bf16",
    num_views=3,
)

# Three resized uint8 HWC images in the policy's camera order.
aux = model.pipeline.prepare_observation(images, "Place the cup on the table.")
model.set_prompt(aux=aux)

# Normalized and padded robot state, shape (1, 55) for RoboTwin.
actions = model.infer(state_normalized, aux=aux)
```

For offline use, download the processor assets:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --exclude '*.safetensors' '*.bin' --local-dir /path/to/Qwen3-VL-4B-Instruct
```

Set `tokenizer_path` in the checkpoint's `config.json`, preserving its other
fields. For the published checkpoint:

```json
{
  "vlm_family": "qwen3_vl",
  "tokenizer_path": "/path/to/Qwen3-VL-4B-Instruct"
}
```

For a fine-tuned policy, place its official YAML at
`<checkpoint>/lingbotvla_cli.yaml`. This file takes precedence over
`config.json` and overrides the bundled defaults; set `model.tokenizer_path`
there when using local processor assets. The frontend also accepts explicit
`policy_config` and `processor_path` when constructed directly.

The RoboTwin policy returns FP32 normalized actions with shape `(1, 50, 55)`.
Robot state mapping, normalization and action decoding use the policy's
training transforms. The generic `predict(images, prompt, state)` wrapper
does not apply those transforms for this model.

## Observations and Graph replay

Already processed observations can be passed as `aux` with fields `images`,
`image_grid_thw`, `img_masks`, `lang_tokens` and `lang_masks`. The images field
contains packed patches. Single-sample and batch-1 layouts are accepted.

Pass fresh `aux` to each full inference. Pixels, state and noise can change
without recapture; changes to tokens, masks or image grids require another
`set_prompt(aux=...)` call. The first inference calibrates and captures the
HIP Graph. `infer(..., use_graph=False)` selects eager execution.

For repeatable inference, supply BF16 `initial_noise` with shape `(1, 50, 55)`.
`num_steps` at load time controls denoising; `action_horizon` trims the
returned chunk while retaining the full policy suffix computation.

To sample several action chunks from the same observation, reuse its prefix:

```python
model.pipeline.prefill_observation(aux=aux)
actions = model.pipeline.infer_cached(state_normalized, initial_noise=noise)
```

Both methods accept `use_graph=False`. A prompt rebind or full `infer`
invalidates the cached prefix. Each frontend owns its buffers and Graphs;
serialize calls to an instance.

## Native execution

Dense and router projections use hipBLASLt. HIP kernels implement
normalization, RoPE, attention layout/softmax, SiLU, Top-k and routed MoE.
Patch Conv3d, GELU and some elementwise operations use PyTorch. Vision
attention retains BF16 scores and probabilities; language/action attention
uses FP32 post-RoPE Q/K, scores, probabilities and accumulation, then stores
BF16 output.

Routed experts use BF16 WMMA with FP32 accumulation and preserve the BF16
intermediate store. On gfx1151, gate/up and down weights use lossless compact
tiles with original BF16 fallback tiles. Packing adds about 1.98 GiB beside
the original weights and requires no inference-time decompression workspace.
Other RDNA targets use unpacked weights.

The gfx1151 pipeline compacts valid prefix queries, fuses norm/RoPE/layout
and residual operations, and overlaps the shared expert with routed experts
on a separate HIP stream. GEMM calibration preserves reference outputs
before Graph capture. Pi0.5 retains its existing operator routes.

## Performance and accuracy

Radeon 8060S / gfx1151, batch 1, three 256×256 views, 17 valid prompt tokens
padded to 72, 50 actions, ten denoising steps and four CPU intra-op threads.
The prompt is "Pick up the red block and place it in the bowl." Its valid
token count includes the Qwen chat template; padding follows the official
RoboTwin policy's 72-slot setting.

| Configuration | Median latency |
|---|---:|
| Official BF16 eager | 678.68 ms |
| Official BF16 torch.compile, max-autotune | 511.59 ms |
| FlashRT BF16, HIP Graph | 346.63 ms |

All three paths use the same resident GPU inputs and fixed noise, with BF16
outputs on the GPU. After warmup, 30 runs per path are timed with GPU events,
cycling through all six execution orders. Each call runs vision, language
prefill and all denoising steps. Preprocessing, transfers, loading, compilation,
calibration and capture
are excluded. FlashRT is 1.48× faster than torch.compile in this run. The
official paths use eager attention on ROCm.

Nine official BF16 controls with fixed routing-slot MoE summation matched
eager, Graph and replay outputs bitwise: cosine 1.0, RMSE and maximum error
zero. The official atomic MoE sum can vary between identical runs; reference
comparisons must use the same summation policy. In this latency run, direct
comparisons with official eager ranged from cosine 0.999821 to 0.999965, and
with the compiled policy from 0.998340 to 0.999845. The FlashRT output matched
the fixed-order BF16 control bitwise.

An earlier standalone inference run used 13.88 GiB allocated and
14.64 GiB reserved GPU memory.

## Tests

Routing, checkpoint layouts, HIP precision, MoE packing and Graph regressions
are in one file. Full-model cases skip when the assets below are not set:

```bash
python -m pytest -q tests/test_amd_rdna_lingbot_vla_v2.py

FLASH_RT_LINGBOT_VLA_V2_RDNA_CKPT=/path/to/checkpoint \
FLASH_RT_LINGBOT_VLA_V2_PROCESSOR=/path/to/Qwen3-VL-4B-Instruct \
FLASH_RT_LINGBOT_VLA_V2_REFERENCE=/path/to/official_reference.pt \
python -m pytest -q tests/test_amd_rdna_lingbot_vla_v2.py
```
