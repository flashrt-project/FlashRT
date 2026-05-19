# FlashRT Jetson AGX Orin (SM87) Deployment Guide

> INT8 inference for Pi0.5 on Jetson AGX Orin 64GB.
> GPU: SM87 (Ampere), 16 SMs, LPDDR5X 204 GB/s, 4 MB L2, no native FP8.
> Uses the `pi05_rtx` pipeline with INT8 fast paths.

---

## Hardware

| Field | Value |
|---|---|
| Device | Jetson AGX Orin 64GB Developer Kit |
| GPU | SM87 (Ampere), 16 SMs |
| Compute | 60 TOPS INT8 (Tensor Core), 5.3 TFLOPS BF16 |
| Memory | 64 GB LPDDR5X unified, 204 GB/s |
| L2 cache | 4 MB |
| FP8 | not native (SM87 has no FP8 tensor cores) |
| CUDA / JetPack | tested with CUDA 12.6 / JetPack 6.x |

## Architecture mapping

`flash_rt/hardware/__init__.py` dispatches Orin to the RTX Pi0.5 pipeline:

```
("pi05", "torch", "rtx_sm87") → flash_rt.frontends.torch.pi05_rtx.Pi05TorchFrontendRtx
```

Orin lacks native FP8 tensor cores, so the RTX backend falls back to **INT8
W8A8 rowwise** for both encoder and decoder GEMMs when
`FVK_PI05_RTX_FORCE_INT8=1` is set. The default ThorU / RTX 4090 / RTX 5090
paths are unaffected by anything in this guide.

## Build

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -B build_orin_sm87 -S . \
  -DGPU_ARCH=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_HDIMS='96;128;256' \
  -DFA2_DTYPES='bf16' \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build_orin_sm87 -j4
```

CMake auto-detects SM87 and emits SASS for `compute_80,sm_87` (Ampere ISA
compatible). Build time ≈ 4-6 minutes on Orin.

## Usage

```python
import os
os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

pipe = Pi05TorchFrontendRtx(
    "/path/to/pi05_droid_pytorch",
    num_views=2,           # 1 or 2 cameras
    num_steps=10,          # flow-matching ODE steps
    vision_pool_factor=1,  # 1=no pool, 2=2×2, 4=4×4
    vision_num_layers=27,  # SigLIP layers (1-27)
    cache_frames=1,        # 1=bit-equal lossless, 2=K/V reuse (cos 0.991)
)
pipe.set_prompt("pick up the black envelope on the table")
pipe.calibrate_with_real_data([obs])  # once at startup, ~2 s

result = pipe.infer(obs)
actions = result["actions"]  # (chunk_size, action_dim) numpy
```

## INT8 fast path components

All INT8 paths activate via `FVK_PI05_RTX_FORCE_INT8=1`. Components added in
this guide:

| Component | File | Purpose |
|---|---|---|
| CUTLASS SM80 INT8 rowwise GEMM (128×128) | `csrc/gemm/cutlass_sm80_int8_rowwise.cu` | default INT8 W8A8 GEMM, BF16 output |
| CUTLASS SM80 INT8 rowwise GEMM (64×128 alt) | `csrc/gemm/cutlass_sm80_int8_rowwise_t64x128.cu` | alt tile for QKV-shape and decoder M=10 |
| Per-shape tile dispatcher | `cutlass_sm80_int8_rowwise.cu::prefer_t64x128_for_shape` | runtime selects 64×128 vs 128×128 by (M, N) |
| CUTLASS SM80 INT8 SiLU-gated EVT GEMM | `csrc/gemm/cutlass_sm80_int8_silu_gated.cu` | fused gate × silu(up) into one GEMM, eliminates `gate_geglu_merged` |
| `bias_gelu_bf16_strict` | `csrc/kernels/activation.{cu,cuh}` | fused (x+bias) → gelu, BF16-rounded mid-state matches the un-fused pair bit-for-bit |
| `bias_residual_layer_norm_bf16` | `csrc/kernels/norm.{cu,cuh}` | fused residual += x+bias, layer_norm; 3-pass strict form, bit-equal |
| `gate_residual_ada_norm_int8` | `csrc/kernels/fusion.{cu,cuh}` | decoder fused residual + ada_norm with INT8 output for downstream GEMM |
| INT8 dynamic per-row quant + dequant | `csrc/kernels/quantize.{cu,cuh}` | graph-compatible BF16 → INT8 / INT32 → BF16 |
| cublasLt INT8_NN runner | `csrc/gemm/gemm_runner.{cu,h}` | autotune wrapper for cublasLt INT8 GEMM (used by some shapes) |

### Tile dispatcher rationale

| Shape (M, N, K) | 128×128 | 64×128 | Winner |
|---|---|---|---|
| enc_qkv  (522, 2560, 2048) | 328 µs | **214 µs** | 64×128 (1.54×) |
| enc_o    (522, 2048, 2048) | **95 µs** | 113 µs | 128×128 |
| enc_gate (522, 8192, 2048) | **337 µs** | 398 µs | 128×128 |
| enc_up   (522, 8192, 2048) | **331 µs** | 399 µs | 128×128 |
| enc_down (522, 2048, 16384)| **1126 µs** | 1419 µs | 128×128 |
| decoder (M=10, N≤8192) | ~50 µs | **~46 µs** | 64×128 (1.08-1.12×) |

64×128 wins when 128×128 has bad wave packing on 16 SMs (qkv-shape M=522
with 100 blocks → 6.25 waves; 64×128 has 180 blocks → 11.25 waves, smaller
partial-wave fraction). Decoder M=10 always wins on 64×128 because the
smaller M-tile gives more total blocks.

## Performance (lossless, p50, 2-camera, AGX Orin 64GB)

| Config | Latency | Throughput | Cosine vs BF16 baseline |
|---|---|---|---|
| BF16 reference | 193 ms | 5.2 Hz | 1.000 (ref) |
| `cache_frames=1` | 124 ms | **8.04 Hz** | **1.000 (bit-equal)** |
| `cache_frames=2` | 127 / 39 ms* | **12.2 Hz** | 0.991 (1-frame K/V stale) |

`*` cache_frames=2 amortizes one full forward over two calls; effective
Hz = `2 / (t_full + t_decode_only)`.

Same checkpoint, same 27 SigLIP layers, 10 ODE steps, pool=1.

## Precision contract

- `cache_frames=1` (default) — bit-equal to the BF16 reference path. INT8
  rounding is fully fused into the same numerical sequence as the un-fused
  kernel pairs. Recommended for accuracy-critical evaluation.
- `cache_frames=2` — encoder K/V from frame N is reused as frame N+1's
  prefix, decoder runs on the new query only. 1-frame stale K/V; cosine
  similarity vs strict baseline measured at 0.991 over 20-frame fixed-seed
  sequences. Recommended for production deployment where 8 ms ODE-tail
  latency matters more than the last 0.9% of cosine.
- All other configs (`vision_pool_factor=2/4`, `vision_num_layers<27`)
  trade accuracy for speed and should be validated per task before use.

## Reproduction

```bash
export FVK_PI05_RTX_FORCE_INT8=1

# Lossless cache_frames=1 (8.04 Hz)
python examples/orin/bench_pi05.py \
  --checkpoint /path/to/pi05_droid_pytorch \
  --num_views 2 --num_steps 10 \
  --vision_pool_factor 1 --vision_num_layers 27 \
  --cache_frames 1

# Production cache_frames=2 (12.2 Hz, cos 0.991)
python examples/orin/bench_pi05.py \
  --checkpoint /path/to/pi05_droid_pytorch \
  --num_views 2 --num_steps 10 \
  --vision_pool_factor 1 --vision_num_layers 27 \
  --cache_frames 2
```

`bench_pi05.py` reports p50 / p99 latency, computed Hz, and cosine vs the
BF16 reference path.

## Known limitations on SM87

- No native FP8 tensor cores — INT8 W8A8 is the practical fast path on this
  generation.
- FA2 attention path uses the SM80 fp16 fwd kernels (`compute_80,sm_87`
  codegen). q-len ≤ 16 decoder shapes are not optimal here; the realistic
  Orin lossless ceiling sits around 8.0-8.2 Hz with cache_frames=1, vs
  ~21 Hz on Jetson AGX Thor (SM110, FP8 native) with the same checkpoint.
