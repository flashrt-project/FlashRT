<p align="center">
  <img src="FlashRT.png" alt="FlashRT" width="640">
</p>

# FlashRT

**FlashRT is a high-performance realtime inference engine for small-batch, latency-sensitive AI workloads.**

A general kernel library composed into static graphs — no ONNX export, no engine compilation, no per-driver rebuild. Hand-written kernels (norm / activation / fusion / RoPE / FP8 / NVFP4 GEMM / attention) cover standard transformer, DiT, and SigLIP primitives. The composition pattern itself is hardware-agnostic; today the codebase ships with NVIDIA implementations spanning edge to server (Jetson AGX Thor through A100 / RTX 4090 / 5090).

The flagship integration today is **VLA control** — production frontends for Pi0, Pi0.5, GROOT N1.6, GROOT N1.7, and Pi0-FAST, validated on LIBERO where applicable. The same kernel set also powers BAGEL world-model research paths, Higgs Audio v3 TTS, Wan2.2 / Motus video-policy paths, and **single-stream LLM inference** with Qwen3.6-27B NVFP4 long-context serving. The pattern is workload-shaped (small-batch realtime), not model-class-shaped.

Existing inference tooling is shaped for different workloads — TensorRT for tactic-search compile to frozen engines, vLLM / SGLang for high-batch LLM serving. FlashRT targets the small-batch realtime cell with hand-tuned kernels and no compile step.

## FlashRT is fast with:

- **hand-written CUDA kernels**: norm, activation, residual+norm+quant fusion, RoPE / qkv-split, FP8 / NVFP4 GEMM, cuBLASLt FP8, CUTLASS SM100 FP8, vendored Flash-Attention 2, Thor CUTLASS FMHA
- **Static CUDA Graph capture** of the entire forward — zero Python overhead at replay
- **Production FP8 (E4M3) and NVFP4** with automatic per-tensor calibration, JSON-cached to disk
- **No compile, no export**: direct safetensors / Orbax loading, first call ~3 s, every call after is graph replay
- Survives CUDA driver upgrades, GPU swaps, and prompt changes without rebuild
- **Serving hosts** for OpenAI-compatible LLM/audio endpoints and robot execution-state scenarios

## FlashRT is easy to use with:

- **3-line API**: `flash_rt.load_model(...).predict(images, prompt)`
- **Auto-dispatched hardware**: same code path on Jetson Thor / RTX 5090 / RTX 4090
- **PyTorch and JAX frontends** share one kernel binary, equivalent results (cosine ≥ 0.999)
- **Plugin model registration** — add a new VLA via one frontend file + a declarative `WEIGHT_SPEC`, no fork required
- **LIBERO benchmark integration** out of the box; ~6 minutes from `git clone` to first inference

See [Supported Models](#supported-models), [Hardware Support](#hardware-support), and [Benchmark](#benchmark) for the current map.

## News

- [2026/06] **Higgs Audio v3 TTS-4B** lands on FlashRT with a kernelized FP8/BF16 decode path, streaming-friendly generation API, and a FastAPI serving host. See [Higgs usage](docs/higgs_audio_v3.md#3-quickstart), [Higgs performance](docs/higgs_audio_v3.md#performance), and [Higgs serving](serving/higgs_audio_agent/README.md).
- [2026/06] **FlashRT HF Kernels** are available as Hugging Face Kernel Hub packages under the `flashrt` namespace. See [LiangSu8899/FlashRT-HF-kernels](https://github.com/LiangSu8899/FlashRT-HF-kernels) and [huggingface.co/flashrt](https://huggingface.co/flashrt).
- [2026/06] The `serving/` layer is documented as the scenario-host layer for OpenAI-compatible LLM/audio serving and robot execution-state hosts. See [serving README](serving/README.md), [serving design](docs/serving_design.md), and [architecture](docs/architecture.md).
- [2026/05] **Qwen3.6-27B NVFP4** is supported with 256 K context on a single RTX 5090, OpenAI-compatible serving, FP8-KV long-context verify, and **145 tok/s warm decode at 256 K**. See [Qwen3.6 NVFP4](docs/qwen36_nvfp4.md) and [Benchmark](#benchmark).
- [2026/05] **Qwen3-8B NVFP4** text-only serving is supported on RTX 5090, with **9.1 ms TTFT at P=64** and **150 tok/s** warm decode. See [Qwen3-8B NVFP4](docs/qwen3_8b_nvfp4.md) and [Benchmark](#benchmark).
- [2026/05] **Wan2.2 TI2V-5B** official-pipeline baseline is available on RTX SM120, with opt-in TeaCache acceleration. See [Wan2.2 usage](docs/wan22_usage.md).
- [2026/05] **Lingbot-VLA** is supported. See [Lingbot usage](https://github.com/LiangSu8899/FlashRT/blob/main/docs/lingbot_usage.md).
- [2026/05] Community Pi0.5 hardware benchmarks: thanks to [@cuihengrui35](https://github.com/cuihengrui35) for **RTX 5060 Ti** results (**41.4 ms / ~24 Hz**, plus LIBERO Spatial **344/350 = 98.3%**) and [@wangerforcs](https://github.com/wangerforcs) for **NVIDIA L40** results (**26.6 ms / 38 Hz**) on 2-view FP8. See [community benchmarks](#community-benchmarks).
- [2026/05] Special thanks to [@gugudeshubao](https://github.com/gugudeshubao) for the **Pi0.5 Jetson AGX Orin (SM87) port**: INT8 W8A8 kernels, Orin tile dispatch, frame-cache inference, deployment docs, and benchmark results. Thanks also to [@strayberry](https://github.com/strayberry) for Orin BF16 Pi0.5 testing. See [Orin deployment](docs/deployment_orin.md) and [community benchmarks](#community-benchmarks).
- [2026/05] **Motus RTX beta** lands in FlashRT: Stage3 fast profile reaches **~167 ms** E2E on RTX 5090, **~100 ms** with TeaCache, and RTC-lite supports 50 Hz action streaming. See [Motus usage](docs/motus_usage_beta.md) and [Benchmark](#benchmark).

---

<a name="performance"></a>

## Benchmark

Baseline comparisons and source methodology live in [Benchmark Comparison](docs/benchmark_comparison.md).

#### Pi0.5

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | FP8, 2-view | **44.0 ms** | **23 Hz** | [Thor VLA](examples/thor/README.md#thor-vla-performance) |
| Jetson AGX Thor | NVFP4, 2-view | **39.78 ms** | **25 Hz** | [NVFP4](#nvfp4-encoder-ffn-pi05-only) |
| Jetson AGX Thor | NVFP4, 3-view | **51.51 ms** | **19 Hz** | [NVFP4](#nvfp4-encoder-ffn-pi05-only) |
| RTX 5090 | FP8, 2-view | **17.58 ms** | **57 Hz** | [Blackwell VLA](examples/blackwell/README.md#vla-latency-rtx-5090) |

#### Pi0

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | FP8, 2-view | **45.8 ms** | **22 Hz** | [Thor VLA](examples/thor/README.md#thor-vla-performance) |
| RTX 5090 | FP8, 1-view | **18.43 ms** | **54 Hz** | [API snippets](#api-snippets) |
| RTX 5090 | FP8, 2-view | **21.16 ms** | **47 Hz** | [API snippets](#api-snippets) |
| RTX 5090 | FP8, 3-view | **24.48 ms** | **41 Hz** | [API snippets](#api-snippets) |

#### GROOT N1.6

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | T=16 | **41 ms** | **24 Hz** | [Thor VLA](examples/thor/README.md#thor-vla-performance) |
| Jetson AGX Thor | T=50 | **45 ms** | **22 Hz** | [Thor VLA](examples/thor/README.md#thor-vla-performance) |
| RTX 5090 | T=16, 2-view | **12.53 ms** | **80 Hz** | [Blackwell VLA](examples/blackwell/README.md#vla-latency-rtx-5090) |
| RTX 5090 | T=50, 2-view | **13.08 ms** | **76 Hz** | [Blackwell VLA](examples/blackwell/README.md#vla-latency-rtx-5090) |

#### GROOT N1.7

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | DiT path | **49 ms** | **20 Hz** | [GROOT N1.7 API](#groot-n17-rtx) |
| RTX 5090 | DiT path | **22 ms** | **45 Hz** | [GROOT N1.7 API](#groot-n17-rtx) |

#### Pi0-FAST

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | max-perf | **8.1 ms/token** | **123 tok/s** | [Thor VLA](examples/thor/README.md#thor-vla-performance) |
| RTX 5090 | max-perf | **2.39 ms/token** | **418 tok/s** | [Blackwell VLA](examples/blackwell/README.md#vla-latency-rtx-5090) |

#### LingBot-VLA

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| Jetson AGX Thor | FA4, 10 steps | **64.1 ms** | **16 Hz** | [LingBot usage](docs/lingbot_usage.md#5-accuracy--latency-thor-sm_110-cuda-graph-replay) |
| Jetson AGX Thor | FA4, 25 steps | **97.5 ms** | **10 Hz** | [LingBot usage](docs/lingbot_usage.md#5-accuracy--latency-thor-sm_110-cuda-graph-replay) |
| Jetson AGX Thor | FA4, 50 steps | **155.8 ms** | **6 Hz** | [LingBot usage](docs/lingbot_usage.md#5-accuracy--latency-thor-sm_110-cuda-graph-replay) |

#### Qwen3.6-27B

RTX 5090:

| Mode | Prefill | Decode | Source |
|---|---:|---:|---|
| NVFP4, 128 | **31.4 ms** | **144.9 tok/s** | [Qwen3.6 NVFP4](docs/qwen36_nvfp4.md) |
| NVFP4, 4 K | **350.9 ms** | **129.6 tok/s** | [Qwen3.6 NVFP4](docs/qwen36_nvfp4.md) |
| NVFP4, 16 K | **1.570 s** | **175.4 tok/s** | [Qwen3.6 NVFP4](docs/qwen36_nvfp4.md) |
| NVFP4, 256 K | **87.976 s** | **144.6 tok/s** | [Qwen3.6 NVFP4](docs/qwen36_nvfp4.md) |

Jetson AGX Thor:

| Mode | Prefill | Decode | Source |
|---|---:|---:|---|
| NVFP4, 128 | **268 ms** | **42.8 tok/s** | [Qwen3.6 Thor](docs/qwen36_nvfp4.md#jetson-agx-thor-numbers) |
| NVFP4, 16 K | **19.23 s** | **52.9 tok/s** | [Qwen3.6 Thor](docs/qwen36_nvfp4.md#jetson-agx-thor-numbers) |

#### Qwen3-8B

| Hardware | Mode | Prefill | Decode | Source |
|---|---|---:|---:|---|
| RTX 5090 | P=64 | **9.1 ms** | **150 tok/s** | [Qwen3-8B NVFP4](docs/qwen3_8b_nvfp4.md) |
| RTX 5090 | P=1024 | **24.8 ms** | **150 tok/s** | [Qwen3-8B NVFP4](docs/qwen3_8b_nvfp4.md) |

#### Higgs Audio v3

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| RTX 5090 | FP8 AR decode | **3.2 ms/frame** | RTF **0.095-0.11** | [Higgs performance](docs/higgs_audio_v3.md#performance) |
| RTX 5090 | BF16 AR decode | **6.1 ms/frame** | RTF **0.15** | [Higgs performance](docs/higgs_audio_v3.md#performance) |

#### Motus Stage3

| Hardware | Mode | Latency | Throughput | Source |
|---|---|---:|---:|---|
| RTX 5090 | fast profile | **167 ms** | **6.0 Hz** | [Motus usage](docs/motus_usage_beta.md) |
| RTX 5090 | TeaCache | **100 ms** | **10 Hz** | [Motus usage](docs/motus_usage_beta.md) |

#### Wan2.2 TI2V-5B

| Hardware | Mode | Generation time | Source |
|---|---|---:|---|
| RTX 5090 | 720p, 121f, 20 steps | **178.6 s** | [Wan2.2 benchmarks](docs/wan22_usage.md#benchmarks) |
| RTX 5090 | TeaCache 0.3 | **114.2 s** | [Wan2.2 benchmarks](docs/wan22_usage.md#benchmarks) |

## Getting Started

- [Install FlashRT](#build--install)
- [Quick Start](#quick-start)
- [API snippets — Pi0 / Pi0.5 / GROOT / Pi0-FAST / Qwen3.6](#api-snippets)
- [Supported Models](#supported-models) · [Hardware Support](#hardware-support) · [Benchmark](#benchmark)
- [Serving](serving/README.md) · [Architecture](docs/architecture.md)
- [Qwen3.6-27B NVFP4 LLM path — quickstart, K selection, measured throughput](docs/qwen36_nvfp4.md) · [parameter reference](docs/qwen36_usage.md) · [OpenAI-compatible server example](serving/qwen36_agent/README.md)
- [Adding a new model](docs/adding_new_model.md)
- [Contributing](CONTRIBUTING.md)
- [Architecture](docs/architecture.md)

## Quick Start

> Already built? Run the snippet below. **Not yet built? See [Build & install](#build--install) first** — `cmake .. && make -j` produces the kernel `.so` files this snippet imports. About 6 minutes from `git clone` to first inference.

```python
import flash_rt   # Python module name; project is FlashRT (see About)

model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",
    config="pi05",          # or "pi0", "groot", "groot_n17", "pi0fast"
    framework="torch",      # or "jax"
)

actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
)
# Pi0.5: actions shape (10, 7) — 10 future steps, 7 DOF
```

First call: ~3 s (calibration + CUDA Graph capture). Every subsequent call: 44 ms graph replay on Thor. No `.engine` file, no rebuild after restart. Full snippets for Pi0 / GROOT / Pi0-FAST in [API snippets](#api-snippets).



## Start here

| If you want to … | Read |
|---|---|
| **Run your first inference** | [Build & install](#build--install) — Docker and native Linux paths |
| **See API examples for all 4 VLA models + the Qwen3.6 LLM** | [API snippets](#api-snippets) |
| **Run Qwen3.6-27B NVFP4 (LLM, 256 K on RTX 5090)** | [`docs/qwen36_nvfp4.md`](docs/qwen36_nvfp4.md) — quickstart, K selection, measured throughput · [`docs/qwen36_usage.md`](docs/qwen36_usage.md) — full parameter reference · [`serving/qwen36_agent/`](serving/qwen36_agent/README.md) — OpenAI-compatible HTTP server |
| **Run Qwen3-8B NVFP4 text serving** | [`docs/qwen3_8b_nvfp4.md`](docs/qwen3_8b_nvfp4.md) · [`examples/qwen3_openai_server.py`](examples/qwen3_openai_server.py) |
| **Run Higgs Audio v3 TTS** | [`docs/higgs_audio_v3.md`](docs/higgs_audio_v3.md) — usage + performance · [`serving/higgs_audio_agent/`](serving/higgs_audio_agent/README.md) — HTTP serving |
| **Run Motus RTX beta, TeaCache, or RTC-lite** | [`docs/motus_usage_beta.md`](docs/motus_usage_beta.md) · [`docs/rtc_lite_design.md`](docs/rtc_lite_design.md) |
| **Run Wan2.2 TI2V-5B official-pipeline baseline** | [`docs/wan22_usage.md`](docs/wan22_usage.md) |
| **Use FlashRT kernels through Hugging Face Kernel Hub** | [`LiangSu8899/FlashRT-HF-kernels`](https://github.com/LiangSu8899/FlashRT-HF-kernels) · [`huggingface.co/flashrt`](https://huggingface.co/flashrt) |
| **Run serving hosts** | [`serving/README.md`](serving/README.md) — scenario hosts · [`docs/serving_design.md`](docs/serving_design.md) — capsules and roadmap · [`docs/serving_production.md`](docs/serving_production.md) — production notes |
| **Look up the stable Python API surface** | [`docs/stable_api.md`](docs/stable_api.md) |
| **Integrate a new model into FlashRT** | [`docs/adding_new_model.md`](docs/adding_new_model.md) — end-to-end walkthrough; external plugin pattern in [`docs/plugin_model_template.md`](docs/plugin_model_template.md) |
| **Contribute a bug fix, benchmark, or model path** | [`CONTRIBUTING.md`](CONTRIBUTING.md) — development rules, validation expectations, and PR checklist |
| **Understand the architecture** | [`docs/architecture.md`](docs/architecture.md) — the 8 infrastructure components and how they compose |
| **Use a load-bearing API** (weight loading, attention, calibration) | [`docs/extension/weight_spec.md`](docs/extension/weight_spec.md) · [`docs/extension/attention_backend.md`](docs/extension/attention_backend.md) · [`docs/extension/calibration.md`](docs/extension/calibration.md) |
| **See supported model list** | [Supported Models](#supported-models) |
| **See measured performance** | [Benchmark](#benchmark) · [Benchmark comparison](docs/benchmark_comparison.md) |
| **Know which GPUs have been tested (and how to contribute a run)** | [Hardware Support](#hardware-support) · [Community benchmarks](#community-benchmarks) |
| **Know what kernels ship and whether they fit your model** | [`docs/kernel_catalog.md`](docs/kernel_catalog.md) — the "parts list" with a re-use decision tree |
| **See which fusion patterns exist and why some were rejected** | [`docs/kernel_fusion.md`](docs/kernel_fusion.md) |
| **Understand FP8 calibration mechanics** | [`docs/calibration.md`](docs/calibration.md) |
| **Train a Pi0.5 LoRA fine-tune (FP8 + LoRA, plain or RECAP/ACP-conditioned, PyTorch *or* JAX)** | [`training/README.md`](training/README.md). JAX companion at [`training/jax/README.md`](training/jax/README.md) |
| **Run advantage-conditioned (RECAP / π\*0.6) policies with classifier-free guidance** | [`docs/rl_inference.md`](docs/rl_inference.md) — PyTorch + JAX frontends both supported |
| **See how FlashRT differs from TensorRT / vLLM / SGLang** | [`docs/inference_engine_differences.md`](docs/inference_engine_differences.md) |

---

## Key techniques

The short version: kernel fusion + static FP8 + captured CUDA Graph
+ vendored in-SO Flash-Attention 2. Hand-written CUDA kernels cover
only the memory-bound ops (norm, activation, fusion, quant);
compute-bound GEMM / attention are delegated to cuBLASLt, CUTLASS,
and the vendored FA2.

Full details by topic:

- [`docs/kernel_catalog.md`](docs/kernel_catalog.md) — every kernel
  shipped, grouped by function, with a re-use decision tree for
  non-VLA models.
- [`docs/kernel_fusion.md`](docs/kernel_fusion.md) — production
  fusion patterns, the four historical dead-end optimizations, and
  why the current fusion set converged where it did.
- [`docs/calibration.md`](docs/calibration.md) — FP8 static
  calibration mechanics.
- [`docs/optimization-details.md`](docs/optimization-details.md) —
  line-by-line Pi0.5 latency breakdown (44 ms vs 70 ms baseline).

---

## API snippets

Already built? Jump to API examples below. Not yet built? See
[Build & install](#build--install) for the full Docker / native
Linux flow, then come back.

### 3 Lines of Code

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/checkpoint",
    framework="torch",    # or "jax"
    autotune=3,           # 0=off, 3=default, 5=thorough
)

actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
)
# Pi0.5: actions shape (10, 7) — 10 future steps, 7 DOF

# State is part of the VLA observation. Pi0/GROOT N1.6 consume it during
# inference; token-based variants encode it in the prompt prefix.

# Pi0 (continuous state input):
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_checkpoint",
    config="pi0",
)
actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
    state=state,
)
# Pi0: actions shape (10, 7)

# GROOT N1.6:
model = flash_rt.load_model(
    checkpoint="/path/to/groot_checkpoint",
    config="groot",
)
actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
    state=state,
)
# GROOT: actions shape (50, 128) — 50 steps, 128-dim padded

# Pi0-FAST (autoregressive — discrete token generation, not diffusion):
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_fast_base",  # Orbax (jax) or safetensors-converted (torch)
    config="pi0fast",
    framework="torch",  # or "jax"
)
actions = model.predict(images=[base_img, wrist_img], prompt="pick up the red block")
# Pi0-FAST: action sequence is generated as discrete FAST tokens then decoded
# to continuous actions via the FAST tokenizer (DCT inverse).

# Pi0-FAST max-performance mode (for fixed-prompt 24h deployment):
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_fast_base",
    config="pi0fast",
    decode_cuda_graph=True,       # capture decode loop as CUDA Graph
    decode_graph_steps=46,        # action tokens per inference (50 total with text prefix)
)
```

#### Qwen3.6-27B NVFP4 (LLM, RTX 5090)

The LLM path uses a dedicated frontend — same kernel binary, separate
generation API since chat completion has a different surface from VLA
control. See [`docs/qwen36_usage.md`](docs/qwen36_usage.md) for the
full parameter reference and [`docs/qwen36_nvfp4.md`](docs/qwen36_nvfp4.md)
for the K-curve / measured throughput / model-dependency notes.

```python
import os
import torch
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

# The NVFP4 ckpt has no MTP head; point this env var at a paired
# FP8 ckpt directory that contains mtp.safetensors. Without it,
# speculative decode is disabled (pure-decode still works at ~36 tok/s).
os.environ["FLASHRT_QWEN36_MTP_CKPT_DIR"] = "/path/to/qwen36_fp8_ckpt"

fe = Qwen36TorchFrontendRtx(
    "/path/to/qwen36_nvfp4",   # prithivMLmods/Qwen3.6-27B-NVFP4
    quant="nvfp4",
)

prompt = "Explain quantum entanglement in one short paragraph."
input_ids = fe._tokenizer(prompt, return_tensors="pt").input_ids.cuda()

out = fe.generate_own_speculative_KN_nvfp4(
    input_ids, max_new_tokens=256, K=6,   # K=6 peaks at NTOK<=128
)
text = fe._tokenizer.decode(out[0, input_ids.shape[1]:].tolist())
print(text)
```

For an OpenAI-API-compatible HTTP server (chat completions, drop-in
replacement for `OpenAI(base_url=...)`), use the production agent server
[`serving/qwen36_agent/`](serving/qwen36_agent/) (see its
[`README.md`](serving/qwen36_agent/README.md)):

```bash
pip install fastapi uvicorn
export FLASHRT_QWEN36_MTP_CKPT_DIR=/path/to/qwen36_mtp_ckpt
python -m serving.qwen36_agent.server \
    --checkpoint /path/to/qwen36_nvfp4 \
    --port 8000
# Then: curl http://localhost:8000/v1/chat/completions ...
```

### Framework Choice

| Checkpoint Format | `framework=` | Source |
|-------------------|:---:|--------|
| **safetensors** (HuggingFace/PyTorch) | `"torch"` | `model.safetensors` |
| **Orbax** (JAX/Physical Intelligence) | `"jax"` | `checkpoint/` dir |

Both frontends produce equivalent results (cosine > 0.999) and share the same `flash_rt_kernels.so`.

### Hardware Auto-Dispatch

User code does **not** need to know which GPU it's running on.
`load_model()` inspects `torch.cuda.get_device_capability()` at call
time and routes to the best-matching backend automatically:

| Compute capability | GPU | Backend |
|---|---|---|
| SM110 (11.0) | Jetson AGX Thor | `flash_rt.hardware.thor.*` |
| SM120 (12.0) | RTX 5090 Blackwell | `flash_rt.hardware.rtx.*`, falling back to Thor for models without a 5090-native class (Pi0-FAST uses Thor's in-file SM120 runtime fork) |
| SM89  (8.9)  | RTX 4090 Ada | `flash_rt.hardware.rtx.*` |

Override with `hardware="thor"` / `"rtx_sm120"` / `"rtx_sm89"` for
cross-hardware debugging — `"auto"` (default) is what you almost
always want. Unsupported SM levels raise a clear `RuntimeError` at
`load_model` time rather than falling back silently, because a wrong
backend at runtime is more expensive to debug than a clean crash.

```python
# Same code path on every supported GPU. On an RTX 5090 this resolves
# to RtxTorchGroot; on Jetson Thor it resolves to ThorPipelineTorchGroot.
model = flash_rt.load_model(
    "/path/to/groot_checkpoint",
    config="groot",
    embodiment_tag="gr1",     # see GROOT embodiment slots below
)
```

### GROOT N1.6 embodiment slots

GROOT's per-embodiment MLPs (state encoder / action encoder / action
decoder) live in 32 parallel slots inside a single checkpoint. In the
`GR00T-N1.6-3B` base checkpoint only a subset of those slots are
actually trained — the rest are at initialization std ~0.02 and emit
noise-like actions regardless of input. **Pick a trained slot for any
demo or deployment**:

| `embodiment_tag=` | Slot | Description |
|---|---|---|
| `gr1` | 20 | GR1 humanoid, 1 camera view. Good default for single-cam demos. |
| `robocasa_panda_omron` | 13 | Tabletop arm + mobile base, 3 camera views |
| `behavior_r1_pro` | 24 | BEHAVIOR humanoid, 3 camera views |
| `new_embodiment` | 10 | Placeholder for fine-tuning (UNTRAINED in base) |

Any other tag in the map (`libero_panda`, `oxe_google`, `oxe_widowx`,
`unitree_g1`, `oxe_droid`) is **untrained** in the base 3B checkpoint
and logs a warning at load time. Fine-tune one of those slots
yourself or pick a trained tag for immediate use.

### GROOT N1.7 RTX

```python
import flash_rt

model = flash_rt.load_model(
    "/path/to/GR00T-N1.7-3B",
    framework="torch",
    config="groot_n17",
    hardware="rtx_sm120",
    num_views=2,
    embodiment_tag="oxe_droid_relative_eef_relative_joint",
)

model.set_prompt(aux=aux, prompt="put the blue block in the green bowl")
actions_normalized = model.infer(
    state_normalized,
    initial_noise=initial_noise,
    use_dit_graph=True,
)
```

GROOT N1.7 is registered as `config="groot_n17"` for the RTX SM120
torch path. It uses the N1.7 `set_prompt(aux=...)` / normalized-state
`infer(...)` contract; see [USAGE.md](USAGE.md#groot-n17-rtx).

### Autotune

CUDA Graph instantiation is non-deterministic on Thor — the same kernels can produce different schedules with ~2ms variance. `autotune` recaptures until a fast schedule is found:

| `autotune=` | Behavior | Extra Startup |
|-------------|----------|---------------|
| `0` or `False` | Off — single capture, may be 2ms slower | 0 |
| `3` (default) | Retry up to 3× — usually finds fast graph on trial 0 | ~1s |
| `5` | Retry up to 5× — better chance for JAX | ~2.5s |
| `True` | Same as `3` | ~1s |

### Pi0-FAST Performance Modes

Pi0-FAST supports two decode modes, controlled by `decode_cuda_graph`:

| Parameter | `set_prompt` (cold) | `set_prompt` (cached) | 50-token E2E | Best for |
|-----------|--------------------:|----------------------:|-------------:|----------|
| `decode_cuda_graph=False` (default) | ~2.5 s | **~0.1 s** | **~464 ms** | Frequent prompt changes |
| `decode_cuda_graph=True` | ~4.0 s | **~1.5 s** | **~431 ms** | Fixed prompt, 24h deployment |

**How it works:**

- **Default mode** (`decode_cuda_graph=False`): Each decode token runs through a
  Python loop with per-step kernel launches. Lowest startup cost. FP8 calibration
  scales are cached to `~/.flash_rt/calibration/` after the first run — subsequent
  `set_prompt` calls with the same checkpoint skip the 2.4s calibration entirely.

- **Max-performance mode** (`decode_cuda_graph=True`): The action-phase decode loop
  is captured as a single CUDA Graph (same technique as Pi0's diffusion loop).
  Eliminates all Python dispatch overhead during decode. Adds ~1.5s to `set_prompt`
  for graph capture, but saves ~33 ms per 50-token inference.
  Break-even at ~45 inferences.

```python
# Default: good for interactive / multi-prompt scenarios
model = flash_rt.load_model(checkpoint, config="pi0fast")
model.set_prompt("pick up the red block", state=state)
# set_prompt: 0.1s (cached) / 2.5s (first run)
# infer: ~464 ms per 50-token sequence

# Max-performance: best for fixed-prompt continuous control
model = flash_rt.load_model(
    checkpoint, config="pi0fast",
    decode_cuda_graph=True,
    decode_graph_steps=46,    # covers sequences up to 46 action tokens (50 total)
)
model.set_prompt("pick up the red block", state=state)
# set_prompt: 1.5s (cached) / 4.0s (first run)
# infer: ~431 ms per 50-token sequence
```

**Calibration caching**: FP8 activation scales are automatically cached per
checkpoint and sequence length. Delete `~/.flash_rt/calibration/` to force
recalibration. The first `infer()` call always recalibrates with real image
data regardless of cache.

### NVFP4 encoder FFN (Pi0.5 only)

Optional NVFP4 (Blackwell block-scaled FP4) quantization on the Pi0.5 encoder
FFN stack. Currently implemented for **Pi0.5 torch only** — passing
`use_fp4=True` with any other config (pi0 / groot / pi0fast) emits a warning
and falls back to FP8.

```python
model = flash_rt.load_model(
    checkpoint,
    config="pi05",
    use_fp4=True,    # single flag → enables the production-validated preset
)
```

`use_fp4=True` resolves to the best-known production preset automatically:
- `fp4_layers` = full 18 encoder FFN layers
- `use_awq` = `True` — activation-aware weight quantization (AWQ)
- `use_p1_split_gu` = `True` — P1 split-GU 2-GEMM path

Advanced users can override any sub-flag explicitly at `load_model()` call
time (e.g. `fp4_layers=(7, 8, 9), use_awq=False` reverts to the conservative
L7-9 subset).

**What it does**:
- Gate+Up and Down GEMMs across all 18 encoder FFN layers run in NVFP4
  (block-size 16, UE4M3 block scales) instead of FP8.
- **AWQ** applies activation-aware per-input-channel pre-scaling to the
  quantized weights, with the inverse scale fused into pre-GEMM kernels
  (`residual_add_rms_norm_mul_fp4_sfa`, `geglu_two_mul_fp4_to_fp4`). This
  preserves precision under 18-layer FP4 (without AWQ, full-scope FP4 cos
  drops from ~0.998 to ~0.33 due to cumulative multi-layer drift).
- **P1 split-GU** splits the merged Gate+Up GEMM into separate gate_proj /
  up_proj NVFP4 GEMMs that emit packed FP4 + SFA directly (via
  `LinCombBlockScaleFactor` epilogue), combined by a dedicated
  `geglu_two_mul_fp4_to_fp4` kernel. Eliminates ~31 MB/layer of DRAM
  round-trips vs the merged-GU path.
- Residual stream stays fp16 through the FP4 region (NVIDIA
  `enable_llm_nvfp4` style — `output_quantizer` disabled).

**Requirements**:
- SM100+ GPU (validated on Thor SM110). Non-SM100 hardware silently falls
  back to FP8.
- `flash_rt_fp4.so` extension (built alongside `flash_rt_kernels.so`).

**Measured on Thor SM110, Pi0.5 / LIBERO Spatial 10 × 50 = 500 episodes**:

| Config | Task success | E2E P50 (normal) |
|---|---|---|
| FP8 baseline | 491 / 500 (98.2%) | ~43.5 ms |
| **NVFP4 full-18 + AWQ + P1 (`--use_fp4`)** | **491 / 500 (98.2%)** | **~43.5 ms** |

Task-level parity with the FP8 baseline (491/500 for both — P1 + AWQ
preserves FP4 precision across all 18 FFN layers).

**Replay-latency benchmark (1-view / 2-view / 3-view, N=8 LIBERO
stratified calibration, 50 graph replays, Thor SM110)**:

| Config | 1-view | 2-view | 3-view | cos vs PyTorch FP32 ref (3v) |
|---|---|---|---|---|
| FP8 baseline (torch) | 34.06 ms | 41.79 ms | 55.46 ms | 0.999236 |
| **NVFP4 encoder (torch)** | **31.91 ms** | **39.78 ms** | **51.51 ms** | **0.998932** |
| **NVFP4 encoder (jax, Orbax)** | **34.39 ms** | **43.65 ms** | **56.90 ms** | **0.999030** |

Encoder FP4 preserves cosine **≥ 0.9989** vs the PyTorch FP32 reference
across view counts, with no latency regression relative to the FP8
torch baseline. The JAX FP4 path derives NVFP4 weights directly from the
Orbax checkpoint (no torch dependency at runtime) and uses the same
two-phase multi-sample calibration flow as the torch FP4 path, producing
a slightly higher cos (0.99903 vs 0.99893 at 3v, same AWQ refit tuning).
Reproduce with
[`tests/bench_pi05_thor_views.py`](tests/bench_pi05_thor_views.py)
(defaults now include `jax_fp4`).

**What's next**:
- Decoder FP4 (S2 precision-validated set — 72 weight tensors, ~-6 ms estimated)
- `geglu_two_mul` SFA-prefetch optimization (O1, ~-0.5-1.1 ms)
- SigLIP FFN FP4 / AWQ auto-tune / Pi0.6 port

---

## Build & install

This is the hands-on "go from a fresh machine to a green benchmark"
section. For a single-page install reference (prerequisites,
troubleshooting table, JAX/transformers pin rationale) see
[`docs/INSTALL.md`](docs/INSTALL.md).

Docker and native Linux paths both produce the same two
extension modules:

| Artifact | Size | What it contains |
|---|---|---|
| `flash_rt/flash_rt_kernels.so` | ~3 MB | Hand-written memory-bound kernels (norm, activation, fusion, FP8 quant, cuBLASLt wrappers, Thor FMHA). **Always built.** |
| `flash_rt/flash_rt_fa2.so` | ~135 MB | Vendored Flash-Attention 2 v2.7.4.post1 fwd (fp16 + bf16, SM80/86/89/120). **Built only on RTX targets** — Thor skips it and uses `fvk.attention_qkv_fp16` (cuBLAS-decomposed) for attention instead. |

**Crucially — no `pip install flash-attn` required.** The FA2 kernel
is vendored at source level and built into `flash_rt_fa2.so` during
`cmake`/`make`; at runtime `import flash_rt` loads both .so files
directly, so you never hit the `flash-attn` wheel's
`torch × CUDA × driver × glibc` compatibility matrix. Setting
`FVK_RTX_FA2=0` is still supported as a fall-back to `pip flash-attn`
for debugging, but the default path has zero pip-wheel dependency.

### Option A — Prebuilt Docker image (fastest, recommended)

The published image already has CUDA 13.0, PyTorch 2.9, the
FlashRT kernels prebuilt, and CUTLASS vendored — pull and run, no
local compile, no `flash-attn` wheel hunting:

```bash
docker pull ghcr.io/liangsu8899/flashrt:latest
docker run --rm --gpus all -it ghcr.io/liangsu8899/flashrt:latest
# Drops you in a Python REPL with `flash_rt` already imported.
```

For Modal / RunPod / Vast and other cloud runners, point the image
config at the same registry — Modal cold-start drops from a 10-minute
kernel compile to a ~30-second pull:

```python
image = modal.Image.from_registry("ghcr.io/liangsu8899/flashrt:0.2.0")
```

Tags + advanced usage (build args, slim variants, mounting checkpoints):
see [`docker/README.md`](docker/README.md).

> **Thor (SM110)** is not covered by this image — Jetson is ARM64 and
> uses a different NVIDIA base. Thor users follow Option C below.

### Option B — Build the Docker image yourself

If you need a different GPU arch, want to pin a specific commit, or
prefer to vet the image source:

```bash
git clone https://github.com/LiangSu8899/FlashRT.git
cd FlashRT
docker build -t flashrt:dev -f docker/Dockerfile .
docker run --rm --gpus all -it flashrt:dev
```

Build args (`GPU_ARCH`, `FA2_HDIMS`, `BASE_IMAGE`, `CUTLASS_REF`)
documented in [`docker/README.md`](docker/README.md). Cold build on a
fresh host is ~25 min (NGC pull + FA2 codegen); warm rebuild ~12 min.

### Option C — Native Linux (no Docker)

System requirements:

| Component | Minimum | Notes |
|---|---|---|
| GPU | SM80+ (A100, 30xx+, Thor, 4090, 5090) | |
| NVIDIA driver | 545+ for CUDA 13, 525+ for CUDA 12.4 | 5090 needs 550+ |
| CUDA Toolkit | 12.4+ (Thor/Hopper) or 12.8+ (Blackwell) | CUDA 13 recommended on 5090 |
| Python | 3.10 / 3.11 / 3.12 | 3.12 on the default NGC image |
| GCC/G++ | 11+ with C++17 | |
| CMake | 3.24+ | |

**Create an isolated Python environment first.** The build step calls
`python3 -m pybind11 --cmakedir` to locate pybind11 headers, so the
Python that runs `cmake ..` MUST be the same interpreter the `.so`
files will be imported from. System-Python + conda-Python mix-ups are
the #1 native-install failure mode.

```bash
python3.12 -m venv .venv         # 3.10 / 3.11 / 3.12 all supported
source .venv/bin/activate
```

Minimum pip list (for the `torch` frontend; everything **must** be
installed *before* `cmake ..`):

```bash
# 1. PyTorch matching your CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 5090 / CUDA 12.8+
# or
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 4090 / A100 / Thor

# 2. Build helpers
pip install pybind11 cmake "numpy>=1.24" safetensors

# 3. Runtime / benchmarking
#    transformers is pinned <4.56 because the Pi0.5 PaliGemma tokenizer
#    path broke in 4.56+; drop the upper bound once we verify the new
#    tokenizer API.
pip install "transformers<4.56" pandas pillow pyarrow

# 4. JAX-side (optional — only if you will load Orbax checkpoints).
#    Versions are pinned because the Orbax/jaxlib/PJRT plugin ABI is
#    not stable across minor releases; upgrading any of the four
#    without matching the others is a reliable way to get cryptic
#    "PJRT device not registered" errors at import time. Pin bump is
#    tracked upstream — see docs/INSTALL.md §JAX for rationale.
pip install jax==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3 ml_dtypes==0.5.3
```

Then build:

```bash
git clone https://github.com/LiangSu8899/FlashRT.git
cd FlashRT
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

pip install -e ".[torch]"          # or "[jax]" / "[all]"
# NOTE: editable mode (-e) is required. The cmake build below drops
# compiled .so files into flash_rt/ in the source tree; editable
# install makes that directory importable directly. A non-editable
# `pip install .` would install a copy BEFORE the .so files exist and
# `import flash_rt` would fail at runtime with a missing-module error.

cmake -B build -S .                 # auto-detects GPU arch
cmake --build build -j$(nproc)
# CMake writes .so files directly into flash_rt/ — no `cp` /
# `make install` / `ninja install` step needed.
```

### GPU arch override

CMake reads `nvidia-smi --query-gpu=compute_cap` to pick the target
arch. Override for cross-compilation or when auto-detect fails:

```bash
cmake -B build -S . -DGPU_ARCH=110   # Jetson AGX Thor   (FA2 skipped, CUTLASS SM100 path ON)
cmake -B build -S . -DGPU_ARCH=120   # RTX 5090           (FA2 sm_80+sm_120 AOT, NVFP4 ON)
cmake -B build -S . -DGPU_ARCH=89    # RTX 4090           (FA2 sm_80 AOT natively runs on Ada)
cmake -B build -S . -DGPU_ARCH=86    # RTX 3090 / A10     (FA2 sm_80 AOT)
cmake -B build -S . -DGPU_ARCH=80    # A100               (FA2 sm_80 AOT)
```

FA2 is enabled by CMake when `GPU_ARCH ∈ {80, 86, 89, 120}`. Other
arches (notably Thor SM110 and SM90 Hopper) route attention through
the cuBLAS-decomposed `fvk.attention_qkv_fp16` path instead of FA2 —
`flash_rt_fa2.so` simply isn't built, and no runtime error results.

### Build timing (one-time)

On a 5090 with CUDA 13 in a warm container, `make -j$(nproc)`:

| Target | Time |
|---|---|
| `flash_rt_kernels` (main kernels) | ~2 min |
| `flash_rt_fa2` (FA2 vendor, default — 12 kernel .cu files × 3 arches) | **~4.5 min** (267 s) |
| Full `make -j$(nproc)` | ~6.5 min |

Subsequent rebuilds of only the hand-written kernels take ~2 min —
FA2 is a separate CMake target and is only re-linked, not recompiled,
unless the vendored source itself changes.

### Slim-build flags (developer iteration speed)

FA2's CUTLASS 3.x templates dominate cold-build cost. The default
matrix covers every RTX family card × fp16+bf16 × all 3 hdim
buckets, which is right for distribution but overkill when you're
iterating on a single 5090/4090 and a single model family. Three
opt-in CMake flags trade binary coverage for iteration speed:

| Flag | Default | What it does | `fa2` cold build on 5090 |
|---|---|---|---|
| — | (none) | 12 .cu × sm_80 + sm_120 + PTX fallback | **267 s (4.5 min)** |
| `-DFA2_ARCH_NATIVE_ONLY=ON` | OFF | Only emit SASS for the detected GPU; skip sm_80 + PTX passes | **110 s** (−59%) |
| `-DFA2_HDIMS="96;256"` | `"96;128;256"` | Drop `head_dim=128` (shipped models don't use it; reserved for future DiT variants) | **210 s** (−21%) |
| `-DFA2_DTYPES="fp16"` | `"fp16;bf16"` | Drop bf16 (Pi0 is fp16-only; Pi0.5 / GROOT need bf16) | **179 s** (−33%) |
| `-DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS="96;256" -DFA2_DTYPES="fp16"` | — | All three combined (single-card + pi0-only) | **87 s** (−67%) |

Shipped `flash_rt_fa2.so` size also shrinks — the all-three-slim
build produces **17.8 MB** (vs 135 MB default), a **87% reduction**
in binary size on the FA2 module.

Dropped entries still resolve at the Python layer — calling a
stubbed entry (e.g. `fa2.fwd_bf16` on a build with
`FA2_DTYPES="fp16"`) aborts the process with a clear
"rebuild with -DFA2_DTYPES=…" message instead of linker errors or
silent wrong output.

### ccache (iterative C++ rebuild speedup)

If `ccache` is on PATH at CMake-config time, it is enabled
automatically for both C++ and CUDA compiles. First build is
unchanged. Hit rate on the `.cpp` side (pybind bindings) is high,
so repeat edits to `csrc/bindings.cpp` / `csrc/fa2_bindings.cpp` get
fast rebuilds. CUDA .cu files — nvcc's invocation style makes
`ccache` hit rate unreliable, so treat CUDA speedup as a bonus
rather than a guarantee. Tip: set `CCACHE_DIR` to a host-mounted
path so the cache survives container rebuilds.

Install via `apt-get install ccache` (Ubuntu) or equivalent.

### Verify

```bash
python examples/quickstart.py \
    --checkpoint /path/to/pi05_checkpoint \
    --benchmark 20
```

Expected (default `--num_views 2`): `P50: ~44 ms (23 Hz)` on Thor.
On RTX 5090 pure replay is ~17.4 ms (57 Hz); `quickstart.py` reports
end-to-end wall clock (~19.5 ms / 51 Hz) because it wraps
`model.predict(...)` with `time.perf_counter` and therefore also
counts image normalization, upload, download, and un-normalization.
For the pure-replay number, time `model._pipe._enc_ae_graph.replay()`
between `cuda.Event` markers — see [Measurement protocol](#measurement-protocol).

### Verify

```bash
python examples/quickstart.py \
    --checkpoint /path/to/pi05_checkpoint \
    --benchmark 20
```

Expected (default `--num_views 2`): `P50: ~44 ms (23 Hz)` on Thor.
On RTX 5090 pure replay is ~17.6 ms (57 Hz); `quickstart.py` reports
the end-to-end wall clock (~19.5 ms / 51 Hz) because it wraps
`model.predict(...)` with `time.perf_counter` and therefore also
counts the graph-external image normalization, upload, download, and
un-normalization. For the pure-replay number, time
`model._pipe._enc_ae_graph.replay()` between `cuda.Event` markers —
see [Measurement protocol](#measurement-protocol).

**GROOT N1.6:**
```bash
python examples/quickstart.py \
    --checkpoint /path/to/groot_checkpoint \
    --config groot \
    --benchmark 20
```

Expected: `P50: ~44 ms (23 Hz)` on Thor.

---

## Architecture

FlashRT is layered so that **framework-specific IO** (safetensors / Orbax),
**declarative weight loading**, **framework-agnostic compute** (pointer-only
pipelines), and **hardware-dispatched attention kernels** each live in their
own module. Adding a new model touches at most one file per layer; adding a
new GPU target touches only `hardware/`.

```
flash_rt/
├── api.py                     ← Public API: load_model() + VLAModel.predict()
│
├── hardware/                  ← Hardware-dispatch + attention protocol
│   ├── __init__.py            ←   detect_arch() + _PIPELINE_MAP
│   ├── backend.py             ←   AttentionBackend protocol + SiteSpec
│   ├── thor/                  ←   Thor SM110 (Jetson AGX Thor)
│   │   ├── attn_backend.py        ← ThorFlashAttnBackend (Pi0.5/Pi0)
│   │   ├── attn_backend_groot.py  ← ThorGrootAttnBackend (GROOT Qwen3+DiT)
│   │   └── shared_primitives.py   ← SigLIP/Encoder/Decoder primitives + calibrate
│   └── rtx/                   ←   RTX SM120/SM89 (RTX 5090 / 4090)
│
├── executors/                 ← Declarative WEIGHT_SPEC framework (stage 7)
│   ├── weight_loader.py       ←   Item / LayerBlock / ModelWeightSpec + runner
│   ├── torch_weights.py       ←   SafetensorsSource + FusedQKV/FusedGateUp
│   └── jax_weights.py         ←   OrbaxDictSource + CudaBufferFlat
│
├── models/                    ← Framework-agnostic pipeline forwards
│   ├── pi05/pipeline.py       ←   Pi0.5 RTX pipeline class
│   ├── pi0/pipeline.py        ←   Pi0 decoder_forward (Thor+RTX)
│   ├── pi0fast/pipeline.py    ←   Pi0-FAST prefill + AR decode (runtime fork)
│   └── groot/                 ←   GROOT DiT + embodiments
│       ├── pipeline.py            ← RTX GROOT
│       ├── pipeline_thor.py       ← Thor GROOT (CKernelQwen3, CKernelDiTHead)
│       └── embodiments.py         ← per-embodiment state/action heads
│
├── frontends/                 ← Per-framework weight loading + CUDA Graph + infer
│   ├── torch/
│   │   ├── pi05_thor.py       ←   Pi0.5 Thor (PyTorch + safetensors)
│   │   ├── pi0_thor.py        ←   Pi0 Thor
│   │   ├── groot_thor.py      ←   GROOT Thor
│   │   ├── pi0fast.py         ←   Pi0-FAST (Thor+RTX runtime fork)
│   │   ├── pi05.py, groot.py  ←   RTX variants
│   │   └── _*_thor_spec.py    ←   Declarative WEIGHT_SPEC per model
│   └── jax/
│       ├── pi05_thor.py       ←   Pi0.5 Thor (JAX + Orbax)
│       ├── pi0_thor.py        ←   Pi0 Thor
│       ├── pi0fast.py         ←   Pi0-FAST
│       └── _*_thor_spec.py    ←   Declarative WEIGHT_SPEC per model
│
├── core/                      ← Shared infrastructure
│   ├── cuda_buffer.py         ←   CudaBuffer (cudaMalloc wrapper, JAX bridge)
│   ├── cuda_graph.py          ←   CUDA Graph capture helpers
│   ├── thor_frontend_utils.py ←   quant_fp8, interleave_qk, embed_prompt
│   ├── quant/calibrator.py    ←   FP8 calibration cache (save/load)
│   └── weights/               ←   loader.py, weight_cache, transformer
│
├── flash_rt/configs/         ← Per-model YAML configs (pi05.yaml, etc.)
└── flash_rt_kernels.*.so     ← 93 CUDA kernels (pybind11 — built from csrc/)

csrc/                       ← C++/CUDA source (compiled once, .so kept in repo)
├── kernels/                ← norm, activation, rope, quantize, fusion
├── gemm/                   ← cuBLASLt FP8 + CUTLASS FP8 helpers
├── attention/              ← CUTLASS FMHA (strided, per-view)
└── bindings.cpp            ← pybind11 → flash_rt_kernels.so

docs/                       ← Documentation
├── stable_api.md           ← Public API + naming convention
├── adding_new_model.md     ← End-to-end guide for adapting a new VLA model
├── calibration.md          ← FP8 weight/activation scale mechanics
├── kernel_fusion.md        ← 93 kernel reference + fusion patterns
├── optimization-details.md ← Pi0.5 44ms vs Myelin 70ms breakdown
└── plugin_model_template.md ← External-plugin model registration

tests/                      ← Precision + unit tests
├── test_all_models_precision.py   ← End-to-end cos + P50 sweep (4 models)
├── test_weight_loader.py           ← WEIGHT_SPEC protocols + composites
├── test_thor_attn_backend.py       ← Pi0.5/Pi0 AttentionBackend contract
├── test_thor_groot_attn_backend.py ← GROOT AttentionBackend contract
└── test_pi0fast_precision.py       ← Pi0-FAST AR decode precision

examples/
├── quickstart.py           ← 3-line usage demo
└── thor/eval_libero.py     ← LIBERO benchmark
```

### Key Design Principles

1. **Pipeline forward receives only int pointers** — no torch, no jax, no
   framework imports. Safe for CUDA Graph capture.
2. **Weight loading is declarative** — each model exports a
   `ModelWeightSpec` (composition of `LayerBlock`s + `Item`s). The
   `WeightLoader` runner executes it over a framework-specific source
   (safetensors for torch, Orbax `engine_w` dict for jax). Adding a new
   Paligemma-family model is a ~60-line spec file plus optional composites.
3. **Attention is protocolized** — `AttentionBackend.run(site=..., layer_idx=..., ...)`
   dispatches across `fmha_strided_full` (SigLIP),
   `attention_qkv_fp16` (GQA), `attention_qkv_fp16_state_masked`
   (Pi0-style), and `attention_mha_fp16` (GROOT) without model code
   knowing which kernel fires.
4. **Hardware-dispatched via `_PIPELINE_MAP`** — `(config, framework, arch)
   → (module, class)` is the single source of truth for which frontend
   loads on Thor SM110 vs RTX SM120 vs RTX SM89. External plugins can
   mutate the map at import time (see
   [`docs/plugin_model_template.md`](docs/plugin_model_template.md)).
5. **Calibration framework-agnostic + cached** — FP8 activation scales
   are computed once per `(checkpoint, seq_len)` pair, cached to
   `~/.flash_rt/calibration/`, then baked as host-scalar alphas
   (`act_scale × weight_scale`) into every CUDA Graph capture. See
   [`docs/calibration.md`](docs/calibration.md).
6. **CUDA Graph captures the entire forward** — Python loop unrolled at
   capture time, zero overhead at replay. All intermediate buffers must
   be pre-allocated in `_load_weights`; no dynamic allocation inside
   forward (see [`docs/kernel_fusion.md`](docs/kernel_fusion.md) §6).

---

## Supported Models

- **Pi0.5** (`config="pi05"`) — [quickstart](#quick-start), [API reference](USAGE.md#api-reference), [NVFP4 notes](USAGE.md#nvfp4-pi05-only), [Thor example](examples/thor/README.md), [RTX 5090 example](examples/blackwell/README.md)
- **Pi0** (`config="pi0"`) — [API snippets](#api-snippets), [usage guide](USAGE.md#api-reference)
- **GROOT N1.6** (`config="groot"`) — [API snippets](#api-snippets), [GROOT embodiment slots](#groot-n16-embodiment-slots)
- **GROOT N1.7** (`config="groot_n17"`) — 49 ms on Jetson AGX Thor, 22 ms on RTX 5090; [usage guide](USAGE.md#groot-n17-rtx), [API snippet](#groot-n17-rtx)
- **Pi0-FAST** (`config="pi0fast"`) — [usage guide](USAGE.md#pi0-fast), [performance modes](#pi0-fast-performance-modes)
- **LingBot-VLA** — [LingBot usage](docs/lingbot_usage.md), [Thor latency](docs/lingbot_usage.md#5-accuracy--latency-thor-sm_110-cuda-graph-replay)
- **Motus Stage3 RTX beta** (`config="motus"`) — [Motus usage](docs/motus_usage_beta.md), [RTC-lite](docs/rtc_lite_design.md)
- **Wan2.2 TI2V-5B** (`config="wan22_ti2v_5b"`) — [Wan2.2 usage](docs/wan22_usage.md)
- **Higgs Audio v3 TTS-4B** — [Higgs usage](docs/higgs_audio_v3.md#3-quickstart), [Higgs serving](serving/higgs_audio_agent/README.md)
- **Qwen3.6-27B NVFP4** — [Qwen3.6 usage](docs/qwen36_nvfp4.md), [parameter reference](docs/qwen36_usage.md), [serving](serving/qwen36_agent/README.md)
- **Qwen3-8B NVFP4** — [Qwen3-8B usage](docs/qwen3_8b_nvfp4.md), [OpenAI server example](examples/qwen3_openai_server.py)
- **BAGEL world-model path** — research preview; see [kernel catalog](docs/kernel_catalog.md) and [adding a model](docs/adding_new_model.md)
- **Reusable HF kernel packages** — [LiangSu8899/FlashRT-HF-kernels](https://github.com/LiangSu8899/FlashRT-HF-kernels), [huggingface.co/flashrt](https://huggingface.co/flashrt)

---

## Hardware Support

FlashRT's shipped implementations are NVIDIA CUDA today. The kernel
composition pattern is not NVIDIA-specific, but the current tested
artifacts and dispatch map are.

| Hardware | SM | Status | Validated paths / notes |
|---|---:|---|---|
| Jetson AGX Thor | SM110 | Production target | Pi0, Pi0.5, GROOT N1.6, Pi0-FAST, Qwen3.6 Thor path, Lingbot; CUTLASS FMHA / Thor attention paths; Pi0.5 FP8 and NVFP4 validation live in [examples/thor](examples/thor/README.md#thor-vla-performance). |
| RTX 5090 | SM120 | Production target | Pi0/Pi0.5/GROOT/Pi0-FAST RTX paths, Qwen3.6, Qwen3-8B, Higgs Audio v3 FP8, Motus, Wan2.2, HF Kernel Hub package validation; see [RTX 5090 latency](examples/blackwell/README.md#vla-latency-rtx-5090). |
| RTX 4090 | SM89 | Validated / supported target | RTX VLA build path and deployment recipe; Higgs BF16 path compiles/configures. See [deployment_rtx4090.md](docs/deployment_rtx4090.md). |
| RTX 5060 Ti | SM120 | Community validated | Pi0.5 FP8 and LIBERO Spatial submission; see [Community benchmarks](#community-benchmarks). |
| RTX 4060 Ti | SM89 | Validated build/run target | Included in current tested hardware list; run local benchmarks before making model-specific latency claims. |
| NVIDIA L40 | SM89 | Community validated | Pi0.5 FP8 submission; see [Community benchmarks](#community-benchmarks). |
| Jetson AGX Orin | SM87 | Community port | Pi0.5 INT8/BF16 paths, Orin tile dispatch, frame-cache inference; see [deployment_orin.md](docs/deployment_orin.md). |
| A100 / A10 / RTX 3090 / RTX 3080 / A5000 / A6000 and other SM80/86/89 GPUs | SM80/86/89 | Build target | CMake and FA2 gates cover Ampere/Ada shapes. Treat unlisted cards as expected to build until a benchmark or regression row is submitted. |

Feature notes:

- `flash_rt_kernels.so` is the always-built core extension.
- RTX targets build `flash_rt_fa2.so`; Thor routes attention through Thor-specific kernels.
- SM100+ targets can build the NVFP4/FP4 extension where the model frontend uses it.
- Unsupported SM levels fail at dispatch/build time instead of silently selecting an incorrect backend.

---


<a name="community-benchmarks"></a>

## Community Hardware Benchmarks

These runs are external hardware submissions using public quickstart or
deployment scripts. Exact latency depends on driver, CUDA, clock state,
warmup count, and checkpoint.

| Contributor | Hardware | Model | Cameras | Warmup | Cache | P50 | P95 / range | Throughput | Check |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [@cuihengrui35](https://github.com/cuihengrui35) | RTX 5060 Ti, SM120, 16 GB | Pi0.5 FP8 | 2 | 200 | - | **41.4 ms** | 40.9-43.2 ms | **24 Hz** | - |
| [@wangerforcs](https://github.com/wangerforcs) | NVIDIA L40, SM89 | Pi0.5 FP8 | 2 | 500 | - | **26.6 ms** | 26.2-27.3 ms | **38 Hz** | - |
| [@gugudeshubao](https://github.com/gugudeshubao) | Jetson AGX Orin 64 GB, SM87 | Pi0.5 DROID INT8 | 2 | - | 1 | **124 ms** | - | **8.04 Hz** | 1.000 |
| [@gugudeshubao](https://github.com/gugudeshubao) | Jetson AGX Orin 64 GB, SM87 | Pi0.5 DROID INT8 | 2 | - | 2 | **127 / 39 ms** | - | **12.2 Hz** | 0.991 |
| [@strayberry](https://github.com/strayberry) | Jetson AGX Orin 32 GB, SM87 | Pi0.5 BF16 | 2 | - | 1 | **215.9 ms** | 217.1 ms | **4.6 Hz** | - |
| [@strayberry](https://github.com/strayberry) | Jetson AGX Orin 32 GB, SM87 | Pi0.5 BF16 | 2 | - | 2 | **137 ms** | 218 ms | **7.3 Hz** | - |

Task-level submission:

| Contributor | Hardware | Task | Trials | Success | Rate |
|---|---|---|---:|---:|---:|
| [@cuihengrui35](https://github.com/cuihengrui35) | RTX 5060 Ti, SM120, 16 GB | Pi0.5 LIBERO Spatial | 350 | **344** | **98.3%** |

If you contribute a hardware benchmark, include the exact command, warmup count,
driver/CUDA/PyTorch versions, and `nvidia-smi` output. For new cards, start with
`python examples/quickstart.py --checkpoint <...> --benchmark 20`.

---



## Acknowledgments

- [CUTLASS](https://github.com/NVIDIA/cutlass) — GEMM templates and FMHA kernels
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Attention backend for SM89/SM120
- [Physical Intelligence](https://www.physicalintelligence.company/) — Pi0/Pi0.5 model architecture
- [OpenPI](https://github.com/Physical-Intelligence/openpi) — Reference PyTorch implementation
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — GROOT N1.6 model
