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
| RTF | **0.095-0.11** | **0.15** | 0.16-0.19 |
| TTFA | **~94 ms** | **~138 ms** | 0.36-0.63 s |
| Per-frame | **~3.2 ms** | **~6.1 ms** | ~6.4 ms |
| VRAM | **6.6 GB** | **9.6 GB** | 28.3 GB reserved |

| Mode | FlashRT | Unoptimized PyTorch reference | Speedup |
|---|---:|---:|---:|
| FP8 AR decode | **3.2 ms/frame** | 10.8 ms/frame | **3.3x** |
| BF16 AR decode | **6.1 ms/frame** | 10.8 ms/frame | **1.8x** |

## Video

| Path | Hardware | Mode | FlashRT | Baseline | Speedup |
|---|---|---|---:|---:|---:|
| Motus Stage3 | RTX 5090 | fast profile | **167 ms** | 1.3 s | **7.8x** |
| Motus Stage3 | RTX 5090 | TeaCache | **100 ms** | 1.3 s | **13.0x** |
| Wan2.2 TI2V-5B | RTX 5090 | 720p, 121f, 20 steps | **178.6 s** | 540 s | **3.0x** |
| Wan2.2 TI2V-5B | RTX 5090 | TeaCache 0.3 | **114.2 s** | 540 s | **4.7x** |
