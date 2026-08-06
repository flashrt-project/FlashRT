# FlashRT on Jetson AGX Thor (SM110)

End-to-end Pi0.5 evaluation on Jetson AGX Thor. For the full install
guide (Docker / native, dependencies, CMake build of the kernel
library) see [`docs/INSTALL.md`](../../docs/INSTALL.md). This page
covers only the Thor-specific run path.

## Prerequisites

- Jetson AGX Thor (SM110) with JetPack / L4T
- CUDA 13.0+ toolkit (matches the NGC PyTorch container default)
- FlashRT installed and verified per [`docs/INSTALL.md`](../../docs/INSTALL.md)
  (you should have `flash_rt/flash_rt_kernels*.so` in place and
  `python -c "import flash_rt; print(flash_rt.__version__)"` works)

## Run E2E LIBERO evaluation

```bash
python examples/thor/eval_libero.py \
    --checkpoint /path/to/pi05_libero_pytorch \
    --task_suite libero_spatial
```

Expected (default `--num_views 2`):

```
============================================================
FlashRT Thor — Pi0.5 LIBERO Spatial
============================================================
[1/3] Loading model + weights         (~10 s)
[2/3] Calibrate FP8 + capture graph   (~3 s, then cached)
[3/3] Running 50 episodes...
============================================================
P50 latency:    ~44 ms (23 Hz)
LIBERO Spatial: 491/500 (98.2%)
============================================================
```

The first invocation calibrates FP8 activation scales and saves them
to `~/.flash_rt/calibration/`. Subsequent runs against the same
checkpoint + prompt length skip calibration automatically (~0.1 s).

## NVFP4 (optional)

Pi0.5 also supports NVFP4 encoder FFN on Thor, with the same E2E
latency floor and identical task accuracy. Enable with:

```bash
python examples/thor/eval_libero.py \
    --checkpoint /path/to/pi05_libero_pytorch \
    --use_fp4
```

See [Thor VLA performance](#thor-vla-performance) below for the
latency/accuracy table across 1/2/3 views.

## Qwen3-VL (multimodal chat)

`qwen3_vl_quickstart.py` runs the official BF16 Qwen3-VL checkpoint as a chat
VLM (text-only, or a single image plus text). It needs the gated Qwen3-VL
kernel module in addition to `flash_rt_kernels`:

```bash
cmake -B build -S . -DGPU_ARCH=110 -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j$(nproc) \
    --target flash_rt_kernels flash_rt_qwen3_vl_kernels
```

```bash
python examples/thor/qwen3_vl_quickstart.py \
    --checkpoint /path/to/Qwen3-VL-2B-Instruct \
    --image FlashRT.png \
    --prompt "Describe this image in one sentence." \
    --max-new-tokens 32
```

Omit `--image` for a text-only prompt, `--no-graph` for the eager decode
reference, and `--benchmark N` to time prefill and warm graph decode
separately. `--weight-mode` picks the weight-only decode tier; measured on a
full-resolution image prompt (1581 tokens, 6256 vision patches):

| `--weight-mode` | decode | vs BF16 |
|---|---:|---:|
| `bf16` | 66.4 tok/s | 1.00× |
| `w8` | 106.6 tok/s | 1.61× |
| `w4` | 158.1 tok/s | 2.38× |

Prefill is eager on Thor (a prefill graph measured 0.3% — it is GPU-bound) and
measured ~230 ms P50 at full resolution with the cuDNN ViT patch attention.
Quality measurements, the `max_pixels` resolution knob, and the per-projection
`wq_overrides` sweep surface are in
[`docs/qwen3_vl_thor.md`](../../docs/qwen3_vl_thor.md).

## Chameleon-7B (multimodal chat)

`chameleon_quickstart.py` runs standalone Chameleon-7B (image + text) with
the dynamic per-tensor FP8 backbone and CUDA-graph prefill:

```bash
python examples/thor/chameleon_quickstart.py \
    --checkpoint /path/to/Chameleon_7B_mGPT \
    --image /path/to/image.jpg \
    --prompt "Describe the image." \
    --benchmark
```

Add `--use-trt-vqgan` when compatible TensorRT VQ-GAN engines exist
(`scripts/build_vqgan_trt.py`), and `FLASHRT_CHAMELEON_FA4_ATTN=1` for the
optional FA4 attention fast path. Measured ~190 ms E2E prefill (eager
VQGAN), ~120 ms with TRT VQGAN + FA4, and ~30 tok/s incremental decode.
Full details in [`docs/chameleon_usage.md`](../../docs/chameleon_usage.md).

## Thor VLA performance

### Precision (Pi0.5, 2-view LIBERO)

Cosine similarity measured with matched noise injection.

| Comparison | Cosine |
|-----------|--------|
| FlashRT Torch vs Production | **0.9996** |
| FlashRT JAX vs Production | **0.9999** |
| FlashRT Torch vs JAX | **0.9998** |

Module-level byte-exact verification on the same input:

- SigLIP (27 layers): byte-exact
- Encoder (18 layers): byte-exact
- Decoder (18 layers x 10 steps): byte-exact

### Latency (Thor)

Pi0.5:

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch** | **36.5 ms** (27 Hz) | **44.0 ms** (23 Hz) | **54.8 ms** (18 Hz) |
| **FlashRT JAX** (autotune=5) | **37.3 ms** (27 Hz) | **44.9 ms** (22 Hz) | **54.4 ms** (18 Hz) |
| NVIDIA TensorRT baseline | - | 91-95 ms | - |

Pi0:

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch** (autotune=5) | **37.6 ms** (27 Hz) | **45.8 ms** (22 Hz) | **56.7 ms** (18 Hz) |
| **FlashRT JAX** (autotune=5) | **37.8 ms** (26 Hz) | **45.8 ms** (22 Hz) | **55.9 ms** (18 Hz) |

Each additional camera view adds about 6 ms from 256 extra SigLIP
tokens and the corresponding encoder traffic. Pi0 E2E precision is
cosine **0.998** vs the FP16 PyTorch reference for both Torch and JAX
frontends.

GROOT N1.6:

| Stage | T=16 (LIBERO) | T=50 (padded max) | Method |
|-------|---------------|-------------------|--------|
| SigLIP (2 views, CUDA Graph) | 6.0 ms | 6.0 ms | Batched 2-view + Graph |
| Qwen3 16L (CUDA Graph) | 8.8 ms | 8.8 ms | FP8 GEMM + C kernel attention |
| DiT 32L x 4 steps (CUDA Graph) | 26 ms | 30 ms | FP8 + cuBLASLt epilogue fusion + cross-KV precompute |
| **Full E2E (image to action)** | **41 ms** (24 Hz) | **45 ms** (22 Hz) | All CUDA Graph |

T is the action horizon. T=50 is the padded production max across
embodiments; T=16 is LIBERO-specific. GROOT N1.6 E2E precision is
cosine **0.999** vs the FP32 PyTorch reference.

Pi0-FAST:

| Mode | Per-token | 50-token E2E | Method |
|------|-----------|-------------|--------|
| **Default** (`decode_cuda_graph=False`) | **8.7 ms** | **~464 ms** | CUTLASS FP8 wide GEMM, vocab pruning, prefill CUDA Graph |
| **Max-perf** (`decode_cuda_graph=True`) | **8.1 ms** | **~431 ms** | Decode loop captured as CUDA Graph |

### LIBERO benchmark (Thor, Pi0.5)

| Suite | Torch | JAX |
|-------|-------|-----|
| **LIBERO Spatial** (10 tasks x 50 ep) | **492/500 = 98.4%** | **490/500 = 98.0%** |
| **LIBERO 10** (10 tasks x 50 ep) | **465/500 = 93.0%** | **463/500 = 92.6%** |

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `No module named 'flash_rt_kernels'` | Build step skipped or non-editable install — see [`docs/INSTALL.md`](../../docs/INSTALL.md) §6 |
| First run slow (~30 s before benchmark) | Normal — FP8 calibration on first prompt length. Cached after. |
| `cuBLAS error code=13` when loading second model | Don't load multiple VLA checkpoints in one process; subprocess-isolate (Thor memory limit). |
| LIBERO score below 95% | Re-check the checkpoint format and `--task_suite` flag; report repro details if persistent. |

For deeper precision debugging, see [`docs/calibration.md`](../../docs/calibration.md) §4.
