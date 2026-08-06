# Benchmark Comparison

This page keeps baseline, TensorRT, and source-methodology tables out of
the README headline benchmark. Only compare rows with matching model,
hardware, view count, step count, and benchmark harness.

## Pi0.5

| Source | Hardware | Mode | Latency | Throughput | Link |
|---|---|---|---:|---:|---|
| OpenPI reference | Jetson AGX Thor | upstream reference, 3-view | **714 ms** | **1.4 Hz** | [OpenPI](https://github.com/Physical-Intelligence/openpi) |
| OpenPI reference | RTX 5090 | upstream reference | **244 ms** | **4.1 Hz** | [OpenPI](https://github.com/Physical-Intelligence/openpi) |
| NVIDIA Jetson AI Lab | Jetson AGX Thor | PyTorch BF16 | **163 ms** | **6.1 Hz** | [OpenPi Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/#performance) |
| NVIDIA Jetson AI Lab | Jetson AGX Thor | TensorRT FP8 | **95 ms** | **10.5 Hz** | [OpenPi Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/#performance) |
| NVIDIA Jetson AI Lab | Jetson AGX Thor | TensorRT FP8+NVFP4 | **94 ms** | **10.6 Hz** | [OpenPi Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/#performance) |

| FlashRT | Hardware | Baseline | Baseline latency | Speedup |
|---|---|---|---:|---:|
| NVFP4, 3-view, **51.51 ms** | Jetson AGX Thor | OpenPI reference, 3-view | 714 ms | **13.9x** |

## GROOT N1.6

NVIDIA Isaac GR00T reports **GR00T-N1.6-3B** with 4 denoising steps.

| Hardware | PyTorch eager | torch.compile | TensorRT | TensorRT Hz | Link |
|---|---:|---:|---:|---:|---|
| RTX 5090 | 58 ms | 37 ms | **31 ms** | **32.1 Hz** | [GR00T optimization](https://nvidia-isaac-gr00t.mintlify.app/deployment/optimization) |
| H100 | 77 ms | 38 ms | **36 ms** | **27.9 Hz** | [GR00T optimization](https://nvidia-isaac-gr00t.mintlify.app/deployment/optimization) |
| RTX 4090 | 82 ms | 44 ms | **43 ms** | **23.3 Hz** | [GR00T optimization](https://nvidia-isaac-gr00t.mintlify.app/deployment/optimization) |
| Jetson AGX Thor | 117 ms | 105 ms | **92 ms** | **10.9 Hz** | [GR00T optimization](https://nvidia-isaac-gr00t.mintlify.app/deployment/optimization) |
| Jetson AGX Orin | 300 ms | 199 ms | **173 ms** | **5.8 Hz** | [GR00T optimization](https://nvidia-isaac-gr00t.mintlify.app/deployment/optimization) |

| FlashRT | Hardware | Baseline | Baseline latency | Speedup |
|---|---|---|---:|---:|
| T=50, **45 ms** | Jetson AGX Thor | PyTorch eager | 117 ms | **2.60x** |
| T=50, **45 ms** | Jetson AGX Thor | torch.compile | 105 ms | **2.33x** |
| T=50, **45 ms** | Jetson AGX Thor | TensorRT | 92 ms | **2.04x** |
| T=50, 2-view, **13.08 ms** | RTX 5090 | PyTorch eager | 58 ms | **4.43x** |
| T=50, 2-view, **13.08 ms** | RTX 5090 | torch.compile | 37 ms | **2.83x** |
| T=50, 2-view, **13.08 ms** | RTX 5090 | TensorRT | 31 ms | **2.37x** |

### Local RTX 4090 Baselines

Local FlashRT measurements below were taken on **July 2-3, 2026** on one
idle **RTX 4090 (SM89)** using the in-repo RTX frontends, not TensorRT.
Unless noted otherwise, the latency number of interest is the
**steady-state** replay path (capture / graph-build cost excluded). Do not
compare these rows directly with the RTX 5090 / TensorRT table above
unless the model, harness, and warmup path match.

`groot` is the repo config name for the current **GR00T-N1.6-3B** path.
There is no separate third local checkpoint beyond `GR00T-N1.6-3B` and
`GR00T-N1.7-3B`.

| Config / Checkpoint | Hardware | Harness | Init | set_prompt | First infer | Steady-state | Notes | Output |
|---|---|---|---:|---:|---:|---:|---|---|
| `groot` | RTX 4090 (SM89) | repo config name for `GR00T-N1.6-3B`; 2-view, synthetic obs, `T=50`, FP8, `fp8_layout=nk` | 6182.18 ms | 2847.41 ms | 1070.17 ms | **18.39 ms p50** / 18.50 ms mean | July 2 local baseline; warm replay only | `(50, 128)` finite |
| `GR00T-N1.6-3B` | RTX 4090 (SM89) | same runtime path as `groot`; 2-view, synthetic obs, `T=50`, FP8, `fp8_layout=nk` | 6182.18 ms | 2847.41 ms | 1070.17 ms | **18.39 ms p50** / 18.50 ms mean | Alias row for the same local baseline | `(50, 128)` finite |
| `GR00T-N1.6-3B` | RTX 4090 (SM89) | local re-check; 2-view `FlashRT.png` duplicated to `image` + `wrist_image`, prompt=`pick up the red block`, `state=zeros(128)`, FP8, `fp8_layout=nk` | - | - | first call excluded | **18.60 ms mean** over 5 steady-state replays | July 3 local steady-state-only re-check; samples: 19.53 / 18.86 / 18.25 / 18.18 / 18.20 ms | `(50, 128)` finite |
| `groot_n17` / `GR00T-N1.7-3B` | RTX 4090 (SM89) | real 2-view fixture, `T=40`, FP8, `fp8_layout=nk`, `use_dit_graph=False` | 7689.45 ms | 910.33 ms | 31.92 ms | **32.60 ms p50** / 33.50 ms mean | July 2 local eager baseline | `(1, 40, 132)` finite |
| `groot_n17` / `GR00T-N1.7-3B` | RTX 4090 (SM89) | same real 2-view fixture, `T=40`, FP8, fixed SM89 DiT graph path, `use_dit_graph=True` | - | - | first graph capture excluded (`252.98 ms`) | **9.98 ms mean** over steady-state replays | July 3 local graph hot path before extra fusion; measured replays: 10.06 / 9.89 ms after capture | `(1, 40, 132)` finite |
| `groot_n17` / `GR00T-N1.7-3B` | RTX 4090 (SM89) | same real 2-view fixture, `T=40`, FP8, same graph path plus existing fused `bias_gelu_quantize_fp8_static_bf16` on the DiT FFN up->down handoff | - | - | first graph capture excluded | **9.69 ms mean** over steady-state replays | July 3 local fused re-check; measured replays: 9.74 / 9.68 / 9.67 / 9.67 ms | `(1, 40, 132)` finite |

### GROOT N1.7 on Jetson AGX Thor

Reference numbers published by NVIDIA for the same model family on the
same board, measured with their harness (`GR00T-N1.7` LIBERO fine-tune on
`libero_10`, 4 denoising steps, batch size 1, medians over 20 iterations
after 5 warmup iterations):

| Source | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| PyTorch eager | 47.7 ms | 68.2 ms | ~126 ms | 8.0 Hz |
| `torch.compile` | 48.6 ms | 46.8 ms | ~105 ms | 9.5 Hz |
| TensorRT BF16 (full pipeline) | 27.0 ms | 45.0 ms | ~81 ms | 12.3 Hz |
| TensorRT optimized + FP8 | 14.1 ms | 21.1 ms | ~44 ms | 22.6 Hz |
| TensorRT optimized + mixed NVFP4 | 13.6 ms | 17.2 ms | ~40 ms | 25.1 Hz |

FlashRT on the matched harness — the same `GR00T-N1.7` LIBERO
fine-tune on `libero_10`, one camera, 4 denoising steps, batch 1,
medians over 20 iterations after 5 warmup iterations, wall-clock
vision-backbone then action-head split:

| FlashRT tier | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| FP8 (default) | 11.0 ms | 25.8 ms | **36.8 ms** | 27.2 Hz |
| NVFP4 + FA4 (`use_fp4=True`) | 8.4 ms | 15.2 ms | **23.7 ms** | 42.3 Hz |

Measured as one same-session A/B/A on Jetson AGX Thor (JetPack 7.2,
MAXN): FP8 36.76 ms, NVFP4 23.65 ms, FP8 36.85 ms. The fixture is a
real `libero_10` observation captured through the official
preprocessing path; the NVFP4 tier's action cosine against the FP8
tier on it is 1.000000 and CUDA-graph replays are bit-identical on
both tiers.

The same tiers on the base `GR00T-N1.7-3B` checkpoint with a 2-view
fixture (`T=40`, same protocol), for reference:

| FlashRT tier | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| FP8 (default) | 23.5 ms | 26.5 ms | **50.2 ms** | 20 Hz |
| NVFP4 + FA4 (`use_fp4=True`) | 14.2 ms | 15.6 ms | **29.9 ms** | 33 Hz |

Also one same-session A/B/A (FP8 51.6 / NVFP4 29.9 / FP8 50.2 ms);
action cosine against the FP8 tier on that fixture is 0.99994.
Reproduce with:

```bash
# LIBERO fine-tune, one camera (matched harness)
python benchmarks/groot_n17_thor_latency.py \
    --ckpt <n17-libero-checkpoint-dir> --aux <1-camera-aux-fixture.pt> \
    --tier fp4 --views 1 --embodiment libero_sim --warmup 5 --iters 20

# base checkpoint, two cameras
python benchmarks/groot_n17_thor_latency.py \
    --ckpt <n17-checkpoint-dir> --aux <2-camera-aux-fixture.pt> \
    --tier fp4 --views 2 --warmup 5 --iters 20
```

Capture a camera-count-matched fixture with
`tests/_helpers/groot_n17/capture_aux_multi.py --views N`.

### N1.7 SM89 Steady-State Hot Replay Profile

Local Nsight Systems capture on **July 3, 2026** used the same real
2-view N1.7 fixture as the steady-state row above and captured exactly
one hot replay after graph build with
`--capture-range=cudaProfilerApi --cuda-graph-trace=node`. The summed GPU
kernel time inside that replay was **10.404 ms**, slightly above the
**9.98 ms** CUDA-event steady-state mean because of profiler overhead.

| Category | Share of summed kernel time | Notes |
|---|---:|---|
| FP8 GEMM (+ split-K reduce) | **44.05%** | dominant `sm89_xmma_gemm_e4m3...` cuBLASLt kernels plus split-K reduction |
| Elementwise / layout / norm | **21.86%** | `add_bias_bf16`, residual add, layout copy, layer norm, GELU, AdaLN |
| BF16 CUTLASS GEMM+ReLU | **19.06%** | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` |
| FA2 attention | **9.12%** | vendored `flash_fwd_kernel` |
| FP8 quantize | **3.00%** | `quantize_fp8_kernel_generic` |
| BF16 cuBLAS GEMM | **2.51%** | remaining non-FP8 GEMM work |

Top individual kernels from that hot replay:

| Kernel | Share | Instances | Avg |
|---|---:|---:|---:|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4...` | **31.63%** | 256 | 12.85 us |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **19.06%** | 212 | 9.35 us |
| `flash_fwd_kernel` | **9.12%** | 128 | 7.41 us |
| `add_bias_bf16_kernel` | **8.11%** | 570 | 1.48 us |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5...` | **6.48%** | 64 | 10.53 us |
| `quantize_fp8_kernel_generic` | **3.00%** | 256 | 1.22 us |

### GROOT N1.7 on Jetson AGX Thor

Reference numbers published by NVIDIA for the same model family on the
same board, measured with their harness (`GR00T-N1.7` LIBERO fine-tune on
`libero_10`, 4 denoising steps, batch size 1, medians over 20 iterations
after 5 warmup iterations):

| Source | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| PyTorch eager | 47.7 ms | 68.2 ms | ~126 ms | 8.0 Hz |
| `torch.compile` | 48.6 ms | 46.8 ms | ~105 ms | 9.5 Hz |
| TensorRT BF16 (full pipeline) | 27.0 ms | 45.0 ms | ~81 ms | 12.3 Hz |
| TensorRT optimized + FP8 | 14.1 ms | 21.1 ms | ~44 ms | 22.6 Hz |
| TensorRT optimized + mixed NVFP4 | 13.6 ms | 17.2 ms | ~40 ms | 25.1 Hz |

FlashRT on the matched harness — the same `GR00T-N1.7` LIBERO
fine-tune on `libero_10`, one camera, 4 denoising steps, batch 1,
medians over 20 iterations after 5 warmup iterations, wall-clock
vision-backbone then action-head split:

| FlashRT tier | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| FP8 (default) | 11.0 ms | 25.8 ms | **36.8 ms** | 27.2 Hz |
| NVFP4 + FA4 (`use_fp4=True`) | 8.4 ms | 15.2 ms | **23.7 ms** | 42.3 Hz |

Measured as one same-session A/B/A on Jetson AGX Thor (JetPack 7.2,
MAXN): FP8 36.76 ms, NVFP4 23.65 ms, FP8 36.85 ms. The fixture is a
real `libero_10` observation captured through the official
preprocessing path; the NVFP4 tier's action cosine against the FP8
tier on it is 1.000000 and CUDA-graph replays are bit-identical on
both tiers.

The same tiers on the base `GR00T-N1.7-3B` checkpoint with a 2-view
fixture (`T=40`, same protocol), for reference:

| FlashRT tier | Backbone | Action head | E2E | Frequency |
|---|---:|---:|---:|---:|
| FP8 (default) | 23.5 ms | 26.5 ms | **50.2 ms** | 20 Hz |
| NVFP4 + FA4 (`use_fp4=True`) | 14.2 ms | 15.6 ms | **29.9 ms** | 33 Hz |

Also one same-session A/B/A (FP8 51.6 / NVFP4 29.9 / FP8 50.2 ms);
action cosine against the FP8 tier on that fixture is 0.99994.
Reproduce with:

```bash
# LIBERO fine-tune, one camera (matched harness)
python benchmarks/groot_n17_thor_latency.py \
    --ckpt <n17-libero-checkpoint-dir> --aux <1-camera-aux-fixture.pt> \
    --tier fp4 --views 1 --embodiment libero_sim --warmup 5 --iters 20

# base checkpoint, two cameras
python benchmarks/groot_n17_thor_latency.py \
    --ckpt <n17-checkpoint-dir> --aux <2-camera-aux-fixture.pt> \
    --tier fp4 --views 2 --warmup 5 --iters 20
```

Capture a camera-count-matched fixture with
`tests/_helpers/groot_n17/capture_aux_multi.py --views N`.

### N1.7 SM89 Steady-State Hot Replay Profile After Reusing Existing Fused Kernel

After replacing the DiT FP8 FFN handoff chain
`add_bias_bf16 + gelu_inplace + quantize_fp8_static` with the existing
`bias_gelu_quantize_fp8_static_bf16` kernel, the same hot-replay-only
Nsight Systems capture reported **10.115 ms** summed GPU kernel time.
Local CUDA-event timing over the same graph steady state was
**9.69 ms mean**.

| Category | Share of summed kernel time | Notes |
|---|---:|---|
| FP8 GEMM (+ split-K reduce) | **45.32%** | essentially unchanged; still the main cost |
| Elementwise / layout / norm | **21.41%** | now includes the fused `bias_gelu_quantize_fp8_static_bf16` kernel |
| BF16 CUTLASS GEMM+ReLU | **19.56%** | unchanged FFN / projector GEMM family |
| FA2 attention | **9.31%** | unchanged attention share |
| BF16 cuBLAS GEMM | **2.59%** | unchanged |
| FP8 quantize | **1.40%** | reduced after fusing the FFN up-output quantize |

Top individual kernels after this reuse:

| Kernel | Share | Instances | Avg |
|---|---:|---:|---:|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4...` | **32.56%** | 256 | 12.86 us |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_nn_align8` | **19.56%** | 212 | 9.33 us |
| `flash_fwd_kernel` | **9.31%** | 128 | 7.36 us |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5...` | **6.63%** | 64 | 10.48 us |
| `add_bias_bf16_kernel` | **5.97%** | 442 | 1.37 us |
| `bias_gelu_quantize_fp8_static_bf16_kernel` | **3.15%** | 128 | 2.49 us |

## LingBot-VLA

LingBot model cleanup baseline:

| ns | Baseline latency |
|---:|---:|
| 5 | 1501 ms |
| 10 | 1741 ms |
| 50 | 2481 ms |

TRT-aligned FP4 loop comparison, using the same quantization scheme.

| Steps | TRT aligned FP4 loop | FlashRT full E2E | Speedup |
|---:|---:|---:|---:|
| 10 | ~122 ms | **64.1 ms** | **~1.9x** |
| 25 | ~304 ms | **97.5 ms** | **~3.1x** |
| 50 | ~608 ms | **155.8 ms** | **~3.9x** |

## Chameleon-7B

Baseline: HF `transformers` BF16 eager, transformer-only (input ids
built by FlashRT, model forward timed). FlashRT rows are the same harness
(real image `hand_1.jpg`, prompt "Describe the image.", target_size=512,
Se≈1053-1072, wall-clock P50). VQGAN backend and FA4 state are recorded per
row — generic default is eager VQGAN + CUTLASS FMHA; TRT VQGAN and FA4 are
explicit opt-ins.

| FlashRT FP8 | VQGAN | FA4 | Latency | Speedup vs HF |
|---|---:|---:|---:|---:|
| transformer-only | (n/a) | on | **104.2 ms** | **3.9×** |
| transformer-only | (n/a) | off | 111.2 ms | 3.6× |
| E2E | TRT opt-in | on | **120.2 ms** | **3.4×** |
| E2E | eager | on | 177.3 ms | 2.3× |
| E2E | eager | off | ~190 ms | ~2.1× |

| Baseline (HF BF16) | Latency |
|---|---:|
| transformer-only | 402.9 ms |

## Qwen3-8B

LLM rows list the baseline and FlashRT measurements without speedup.

| Metric | HF SDPA baseline | FlashRT |
|---|---:|---:|
| TTFT P=64 | 280 ms | **9.1 ms** |
| TTFT P=256 | 295 ms | **11.1 ms** |
| TTFT P=512 | 315 ms | **14.2 ms** |
| TTFT P=1024 | 366 ms | **24.8 ms** |
| Decode warm graph | 3.6 tok/s | **150 tok/s** |
| OAI server warm decode | - | **150 tok/s** |
| VRAM P=1024,N=256 | 5.99 GiB | 7.30 GiB |

## Higgs Audio v3

Higgs Audio v3 TTS-4B on RTX 5090. FlashRT numbers are the in-repo
single-stream AR decode path; SGLang is kept as reference data without
speedup.

| Metric | FlashRT FP8 | FlashRT BF16 | SGLang |
|---|---:|---:|---:|
| RTF | **~0.09** | **0.15** | 0.16-0.19 |
| TTFA | **~79 ms** | **~127 ms** | 0.36-0.63 s |
| Per-frame | **~3.6 ms** | **~6.0 ms** | ~6.4 ms |
| VRAM | **6.6 GB** | **9.6 GB** | 28.3 GB reserved |

| Mode | FlashRT | Unoptimized PyTorch reference | Speedup |
|---|---:|---:|---:|
| FP8 AR decode | **3.6 ms/frame** | 10.8 ms/frame | **3.0x** |
| BF16 AR decode | **6.0 ms/frame** | 10.8 ms/frame | **1.8x** |

BF16 prefill-only latency improves from **8.42 → 6.73 ms** at P=6 and
**11.74 → 6.82 ms** at P=13 in the same single-stream frontend benchmark.

## Video

| Path | Hardware | Mode | FlashRT | Baseline | Speedup |
|---|---|---|---:|---:|---:|
| Motus Stage3 | RTX 5090 | fast profile | **167 ms** | 1.3 s | **7.8x** |
| Motus Stage3 | RTX 5090 | TeaCache | **100 ms** | 1.3 s | **13.0x** |
| Wan2.2 TI2V-5B | RTX 5090 | 720p, 121f, 20 steps | **178.6 s** | 540 s | **3.0x** |
| Wan2.2 TI2V-5B | RTX 5090 | TeaCache 0.3 | **114.2 s** | 540 s | **4.7x** |
