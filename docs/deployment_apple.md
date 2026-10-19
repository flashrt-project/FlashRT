# Experimental Apple Silicon backend

The initial Apple backend runs Pi0.5 in BF16 on PyTorch's `mps` device.
It reuses FlashRT's Pi0.5 safetensors conversion and shared tensor model
traversal, with torch GEMM, normalization, convolution and Metal-backed SDPA.
The attention provider explicitly expands grouped KV heads so it does not
depend on SDPA's CUDA-specific `enable_gqa` implementation.

This is a correctness-first starting point. Full checkpoint action accuracy,
LIBERO success rates and latency need qualification on each Mac. Existing
CUDA/HIP FP8/NVFP4 kernels and graph capture do not port directly to Metal.
There are no custom Metal kernels, MLX runtime, JAX frontend, temporal frame
caching or support for other model families in this initial backend.

## Python environment with uv

Use a native arm64 Python, macOS 14 or newer and an MPS-enabled PyTorch build
with BF16 support. The frontend probes BF16 before allocating model weights.
PyTorch's [MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
explains device availability and installation.

From the repository root:

```sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[apple]' pytest
python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_available())'
```

When Python is already installed, pass its executable to `uv venv --python`.
The development environment created for this checkout uses the bundled
arm64 Python 3.12 interpreter. No CMake or CUDA extension build is required.

## Run Pi0.5

Provide a torch-format checkpoint directory containing `model.safetensors`,
the usual action normalization statistics, and a PaliGemma tokenizer
(see [tokenizer setup](../USAGE.md#pi0--pi05-prerequisite--paligemma-tokenizer)). The complete
BF16 weights, vocabulary embedding table, activation buffers and temporary
tensors must fit in unified memory. Conversion also uses CPU temporaries.

```python
import numpy as np
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",
    config="pi05",
    framework="torch",
    hardware="apple_mps",   # "auto" also detects an available Apple MPS GPU
    precision="bf16",
    use_fp8=False,
    num_views=2,
    action_dim=7,           # set to your robot's actual output dimension
)
images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
actions = model.predict(images, prompt="pick up the red block")
print(actions.shape)
```

`action_dim` may instead be declared as `output_action_dim` in checkpoint
`config.json`. The action horizon is read from `action_horizon` in that file,
defaulting to 10. Set `num_steps` to change the denoising schedule. The time
embeddings and Euler step size are regenerated together. For direct control
of horizon and prompt capacity, construct `flash_rt.frontends.torch.pi05_mps.Pi05TorchFrontendMps`
with `chunk_size` and `max_prompt_len`.

Inputs are two or three 224×224 RGB uint8 NumPy images. Optional robot state
uses the existing normalized state-in-prompt convention. Actions are
unnormalized with the checkpoint's quantile statistics, then sliced to the
explicit robot dimension. Prompt embeddings are computed on CPU, while
vision, prefix encoding and all denoising steps execute on MPS.

`precision="auto"` also selects BF16. The historical `use_fp8=True` default
emits a warning and selects BF16; explicitly requesting `precision="fp8"`,
INT8, FP4, FP16 or FA4 fails. Other model/framework combinations fail during
dispatch. Eager execution refreshes vision and prefix encoding every frame;
`autotune` has no effect and `cache_frames` must be 1.

## Download to an external SSD and run real inputs

The [OpenPI Pi0.5 LIBERO conversion](https://huggingface.co/Fisher-Wang/pi05-libero-pytorch)
used for the local smoke run contains a 14,467,165,872-byte FP32 safetensors
file plus the original quantile normalization statistics. FlashRT converts
the model weights to BF16 when loading. Pin the revision for reproducibility:

```sh
uv pip install --python .venv/bin/python huggingface_hub
HF_HOME=/Volumes/SSD/FlashRT/.cache/huggingface .venv/bin/hf download \
  Fisher-Wang/pi05-libero-pytorch \
  --revision ab107fbe6661e8abfec0f13dc27bfd3fafe24c69 \
  --local-dir /Volumes/SSD/FlashRT/models/pi05-libero-pytorch
FLASH_RT_PALIGEMMA_TOKENIZER=/Volumes/SSD/FlashRT/tokenizers/paligemma_tokenizer.model \
  bash scripts/download_paligemma_tokenizer.sh
.venv/bin/python examples/apple/pi05.py \
  --checkpoint /Volumes/SSD/FlashRT/models/pi05-libero-pytorch \
  --tokenizer /Volumes/SSD/FlashRT/tokenizers/paligemma_tokenizer.model \
  --output /Volumes/SSD/FlashRT/runs/apple_mps_pi05_20261019 \
  --steps 10 --warmup 2 --iters 5 --samples 8 --cpu-reference
```

The example uses the repository's eight real LIBERO camera/state samples.
It normalizes state with the checkpoint's quantiles before tokenization,
holds action noise fixed, checks finite outputs and response to changed
observations, and saves `report.json`, `actions.npz`, and per-call timings.
`--cpu-reference` additionally compares one sample with the shared CPU BF16
pipeline after moving weights off MPS. CPU GEMM uses FP32 accumulation of
the exact BF16 operand values, rounded to BF16 before bias addition, to avoid
the arm64 scalar BF16 fallback. This checks device/provider numerics;
it does not compare against an independent OpenPI implementation or run the
LIBERO simulator. Download caches, tokenizer and original weights remain on
the SSD. The checkpoint directory also contains `download-manifest.json`
with the revision and expected model SHA-256 for this local download.

### Measured smoke run: M5 Pro, 24 GiB (2026-10-19)

The downloaded checkpoint passed `hf cache verify` for all five remote files;
the tokenizer passed its expected MD5. On macOS 27.0.1 / PyTorch 2.14.1:

| Setting or measurement | Result |
|---|---:|
| Views / denoising steps / action horizon | 2 / 10 / 10 |
| State-prompt length | 147–150 tokens |
| Checkpoint load and BF16 conversion | 37.23 s |
| Prewarmed inference, 2 warmups + 5 timed calls | p50 310.67 ms, p95 311.45 ms |
| Input preprocessing + state/prompt update + inference, 8 observations | p50 338.56 ms |
| MPS tensor allocations | 5.38 GiB |
| Fixed-noise repetition, maximum absolute difference | 0 |
| CPU comparison, seven raw robot channels, one observation | cosine 0.99999737, max abs 0.00390625 |
| CPU comparison, seven unnormalized robot channels, one observation | cosine 0.99999818, mean abs 0.00034081 |

All eight observations produced finite `(10, 7)` actions, and changing
observations changed the output. Tensor allocation is not total process or
system memory: the CPU embedding table and runtime overhead are additional.
These are short local smoke measurements, not production latency statistics
or LIBERO task-success results. The CPU comparison shares the same model
implementation and uses FP32 GEMM accumulation with BF16 inputs/outputs.

Local artifacts: `/Volumes/SSD/FlashRT/runs/apple_mps_pi05_20261019/report.json`,
`actions.npz` and `run.log`.

## Validate

```sh
.venv/bin/python -m pytest tests/test_apple_backend.py tests/test_pi05_torch_pipeline.py -q
```

Tests cover accelerator dispatch, native-extension isolation, unsupported
precision requests, CPU prompt placement, attention padding masks, and
reduced-size complete Pi0.5 traversal against CPU reference results. MPS
cases skip when the GPU is unavailable. These synthetic checks do not
establish full checkpoint parity or robot task success.

Local validation (2026-10-19): arm64 Python 3.12.14, PyTorch 2.14.1,
macOS 27.0.1, with Metal available. The Apple/shared-pipeline tests and
related AMD/NPU/prompt regression checks passed: 112 passed, 9 skipped
(missing optional Hugging Face dependencies/tokenizer assets). The passing Apple checks include BF16
attention at the actual SigLIP/Gemma head dimensions, reduced model
traversal, and public API inference with synthetic weights. A sandboxed
process may report MPS unavailable even when the host GPU is usable.

For checkpoint qualification, compare fixed-noise raw actions against the
same-origin reference using `model.pipeline.forward_with_fixed_noise(...)`.
GPU inputs must be BF16 MPS tensors; `capture_probes=True` records intermediate
stages. Benchmark only after warmup and report the Mac model, OS, PyTorch,
view count, prompt length, action horizon and denoising step count. Custom
Metal/MLX kernels should follow measured bottlenecks and a checkpoint parity
gate.
