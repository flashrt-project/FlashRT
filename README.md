# FlashRT

**FlashRT is a high-performance realtime inference engine for small-batch, latency-sensitive AI workloads.**

A general kernel library composed into static graphs — no ONNX export, no engine compilation, no per-driver rebuild. Hand-written kernels (norm / activation / fusion / RoPE / FP8 / NVFP4 GEMM / attention) cover standard transformer, DiT, and SigLIP primitives. The composition pattern itself is hardware-agnostic; today the codebase ships with NVIDIA implementations spanning edge to server (Jetson AGX Thor through A100 / RTX 4090 / 5090).

The flagship integration today is **VLA control** — production frontends for Pi0, Pi0.5, GROOT N1.6, and Pi0-FAST, validated on LIBERO. The same kernel set also powers the BAGEL world-model image-generation pipeline (research preview) and audio / video generation (4× over PyTorch). FlashRT now also serves **single-stream LLM inference** — the v1 release ships **Qwen3.6-27B (NVFP4)** with **256 K context on a single RTX 5090**, an OpenAI-compatible HTTP server, and decode throughput of **~100 tok/s typical / 129 tok/s peak** (real warm-state range across mixed chat / reasoning / code prompts; see [Performance](#performance) for the breakdown). The pattern is workload-shaped (small-batch realtime), not model-class-shaped.

Existing inference tooling is shaped for different workloads — TensorRT for tactic-search compile to frozen engines, vLLM / SGLang for high-batch LLM serving. FlashRT targets the small-batch realtime cell with hand-tuned kernels and no compile step.

## FlashRT is fast with:

- **hand-written CUDA kernels**: norm, activation, residual+norm+quant fusion, RoPE / qkv-split, FP8 / NVFP4 GEMM, cuBLASLt FP8, CUTLASS SM100 FP8, vendored Flash-Attention 2, Thor CUTLASS FMHA
- **Static CUDA Graph capture** of the entire forward — zero Python overhead at replay
- **Production FP8 (E4M3) and NVFP4** with automatic per-tensor calibration, JSON-cached to disk
- **No compile, no export**: direct safetensors / Orbax loading, first call ~3 s, every call after is graph replay
- Survives CUDA driver upgrades, GPU swaps, and prompt changes without rebuild

## FlashRT is easy to use with:

- **3-line API**: `flash_rt.load_model(...).predict(images, prompt)`
- **Auto-dispatched hardware**: same code path on Jetson Thor / RTX 5090 / RTX 4090
- **PyTorch and JAX frontends** share one kernel binary, equivalent results (cosine ≥ 0.999)
- **Plugin model registration** — add a new VLA via one frontend file + a declarative `WEIGHT_SPEC`, no fork required
- **LIBERO benchmark integration** out of the box; ~6 minutes from `git clone` to first inference

## FlashRT supports:

- **VLA models**: Pi0, Pi0.5, GROOT N1.6, Pi0-FAST — all production-validated on LIBERO. BAGEL world-model (research preview) — image-gen pipeline at ~4× vs PyTorch.
- **LLM**: **Qwen3.6-27B NVFP4 — ~100 tok/s typical / 129 tok/s peak decode, 256 K context, single RTX 5090** — speculative decoding via the FP8 ckpt's MTP head, OpenAI-compatible HTTP server.
- **Hardware (today)**: NVIDIA Jetson AGX Thor (SM110), RTX 5090 (SM120), RTX 4090 (SM89), and SM80 / SM86 / SM89 cards (A100, RTX 3090, 4060 Ti, etc.). The kernel composition pattern is portable to other accelerators.
- **Frameworks**: PyTorch (safetensors) + JAX (Orbax) — same compiled kernels

Pi0.5: 44 ms / 23 Hz on Jetson AGX Thor (2v, FP8) · 39.78 ms / 25 Hz (2v, NVFP4) · 17.58 ms / 57 Hz on RTX 5090. Cosine ≥ 0.9996 vs the production reference. See [Performance](#performance) for the full sweep.

## Getting Started

- [Install FlashRT](#build--install)
- [Quick Start](#quick-start)
- [API snippets — Pi0 / Pi0.5 / GROOT / Pi0-FAST / Qwen3.6](#api-snippets)
- [Qwen3.6-27B NVFP4 LLM path — quickstart, K selection, measured throughput](docs/qwen36_nvfp4.md) · [parameter reference](docs/qwen36_usage.md) · [OpenAI-compatible server example](examples/qwen36_openai_server.py)
- [Adding a new model](docs/adding_new_model.md)
- [Contributing](CONTRIBUTING.md)
- [Architecture](docs/architecture.md)

## Quick Start

> Already built? Run the snippet below. **Not yet built? See [Build & install](#build--install) first** — `cmake .. && make -j` produces the kernel `.so` files this snippet imports. About 6 minutes from `git clone` to first inference.

```python
import flash_rt   # Python module name; project is FlashRT (see About)

model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",
    config="pi05",          # or "pi0", "groot", "pi0fast"
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
| **Run Qwen3.6-27B NVFP4 (LLM, ~100 tok/s typical / 129 tok/s peak on RTX 5090)** | [`docs/qwen36_nvfp4.md`](docs/qwen36_nvfp4.md) — quickstart, K selection, measured throughput · [`docs/qwen36_usage.md`](docs/qwen36_usage.md) — full parameter reference · [`examples/qwen36_openai_server.py`](examples/qwen36_openai_server.py) — OpenAI-compatible HTTP server |
| **Look up the stable Python API surface** | [`docs/stable_api.md`](docs/stable_api.md) |
| **Integrate a new model into FlashRT** | [`docs/adding_new_model.md`](docs/adding_new_model.md) — end-to-end walkthrough; external plugin pattern in [`docs/plugin_model_template.md`](docs/plugin_model_template.md) |
| **Contribute a bug fix, benchmark, or model path** | [`CONTRIBUTING.md`](CONTRIBUTING.md) — development rules, validation expectations, and PR checklist |
| **Understand the architecture** | [`docs/architecture.md`](docs/architecture.md) — the 8 infrastructure components and how they compose |
| **Use a load-bearing API** (weight loading, attention, calibration) | [`docs/extension/weight_spec.md`](docs/extension/weight_spec.md) · [`docs/extension/attention_backend.md`](docs/extension/attention_backend.md) · [`docs/extension/calibration.md`](docs/extension/calibration.md) |
| **See the supported models + measured performance** | [Performance](#performance) below |
| **Know which GPUs have been tested (and how to contribute a run)** | [Tested hardware + Help needed](#tested-hardware--whats-theoretically-supported) |
| **Know what kernels ship and whether they fit your model** | [`docs/kernel_catalog.md`](docs/kernel_catalog.md) — the "parts list" with a re-use decision tree |
| **See which fusion patterns exist and why some were rejected** | [`docs/kernel_fusion.md`](docs/kernel_fusion.md) |
| **Understand FP8 calibration mechanics** | [`docs/calibration.md`](docs/calibration.md) |
| **Train a Pi0.5 LoRA fine-tune (FP8 + LoRA, plain or RECAP/ACP-conditioned, PyTorch *or* JAX)** | [`training/README.md`](training/README.md). JAX companion at [`training/jax/README.md`](training/jax/README.md) |
| **Run advantage-conditioned (RECAP / π\*0.6) policies with classifier-free guidance** | [`docs/rl_inference.md`](docs/rl_inference.md) — PyTorch + JAX frontends both supported |
| **See how FlashRT differs from TensorRT / vLLM / SGLang** | [`docs/inference_engine_differences.md`](docs/inference_engine_differences.md) |

---

<a name="performance"></a>

## Performance

| Model | Hardware | Latency | Throughput |
|-------|----------|---------|------------|
| **Pi0.5** | **Jetson AGX Thor** (SM110) | **44 ms** | **23 Hz** |
| **Pi0** | **Jetson AGX Thor** (SM110) | **46 ms** | **22 Hz** |
| **Pi0.5** | **RTX 5090** (SM120) | **17.58 ms** (2v) | **57 Hz** |
| **Pi0** | **RTX 5090** (SM120) | **18.43 ms** (1v) / **21.16 ms** (2v) / **24.48 ms** (3v) | **54 / 47 / 41 Hz** |
| **GROOT N1.6** | **Jetson AGX Thor** (SM110) | **45 ms** (T=50) / **41 ms** (T=16) | **22 / 24 Hz** |
| **GROOT N1.6** | **RTX 5090** (SM120) | **13.08 ms** (T=50, 2v) / **12.53 ms** (T=16, 2v) | **76 / 80 Hz** |
| **Pi0-FAST** | **Jetson AGX Thor** (SM110) | **8.1 ms/token** (28 ms prefill + 8.1 × N decode) | **123 tok/s** |
| **Pi0-FAST** | **RTX 5090** (SM120) | **2.39 ms/token** (11 ms prefill + 2.39 × N decode) | **418 tok/s** |

### LLM — Qwen3.6-27B NVFP4 (RTX 5090)

Single-stream chat-completion latency. NVFP4 W4A16 main weights +
FP8→NVFP4-converted MTP head for K-step speculative decoding. All
numbers are decode-only tok/s (excluding prefill); same metric vLLM
and TensorRT-LLM report.

**Peak (single prompt, NTOK=128, no chat template)** — `"Explain
quantum entanglement in one short paragraph."`, 11 prompt tokens:

| Configuration | Decode latency | Throughput |
|---|---|---|
| **Qwen3.6-27B NVFP4** + spec K=3 | **8.49 ms/token** | **117.8 tok/s** |
| **Qwen3.6-27B NVFP4** + spec K=6 | **7.74 ms/token** | **128.9 tok/s** |

**Real-world warm-state (chat-template + NTOK=256, mixed-task workload)**
— measured across 6 prompts (EN chat / EN reasoning / CN chat / CN
factual / CN poetry / Code) at K=6:

| Stat | tok/s |
|---|---:|
| **mean** | **~93** |
| min (Code prompt) | 75 |
| max (CN factual prompt) | 101 |

So users can expect roughly **~100 tok/s typical** in production with
peak around **129 tok/s** on the easiest prompt class. The cliff is
content-dependent (drafter alignment with the input distribution):

* **Instruction-following / factual / chat** prompts hit the headline
  rate — drafter aligns well with the prompt distribution.
* **Code generation** drops to ~75 tok/s — drafter has lower acceptance
  on punctuation / indentation / bracket tokens. Same trade-off seen
  in vLLM and SGLang spec decode.
* **Long generations** (NTOK ≥ 256) shave ~5-10 tok/s vs short outputs
  — drafter quality decays past the prompt's local distribution.
* **First call** at a new (prompt_len, max_tokens) shape pays a
  ~5-25 s CUDA-Graph capture cost. The bundled OpenAI server
  ([`examples/qwen36_openai_server.py`](examples/qwen36_openai_server.py))
  pre-captures common shapes at startup via `--warmup`.

Long-context decode at fixed context length (TurboQuant packed KV
cache, single-token forward, AL=3.17 amortization):

| ctx | forward latency | est. tok/s with spec |
|---|---|---|
| 8 K | 26.6 ms | 119 |
| 32 K | 38.7 ms | 81 |
| 128 K | 87.7 ms | 36 |
| **256 K** | **153 ms** | **21** ← single-card |

CUDA Graph capture+replay at 32 K / 64 K / 128 K / 256 K passes the
cosine = 1.000000 gate (bit-identical token output across replays).
TTFT scales linearly at ~22 ms / prompt-token.

The full per-prompt variance breakdown (5 prompts × NTOK 128/256 ×
K=3/6) is in [`docs/qwen36_nvfp4.md`](docs/qwen36_nvfp4.md) §3.

### VRAM footprint (inference only, 2 views on RTX 5090)

Measured peak allocation during `model.predict()`:

| Model | dtype | Checkpoint on disk | Peak VRAM |
|---|---|---:|---:|
| **Pi0** | fp16 | 6.5 GB | **~10 GB** |
| **Pi0.5** | bf16 + FP8 | 13.5 GB | **~10 GB** |
| **GROOT N1.6** | fp16 | 6.1 GB | **~9 GB** |
| **Pi0-FAST** (jax) | bf16 | 11 GB | **~7 GB** |

Includes the CUDA context, cuBLASLt workspaces, FA2 scratch, and
the captured CUDA Graph. Thor (unified LPDDR 122 GiB) effectively
has no memory pressure. Practical card sizing:

| Card | Works for |
|---|---|
| **8 GB** (RTX 3060 Ti / 4060) | Pi0-FAST only; others will OOM at graph capture time |
| **12 GB** (RTX 3080 / 4070) | All four models with ~2 GB headroom |
| **16 GB+** (RTX 4080 / 4080 Super) | All four, comfortable |
| **24 GB+** (RTX 4090 / 3090 / 5090) | All four, plus room for larger views / longer prompts |

Measure locally by wrapping `model.predict(...)` in
`torch.cuda.max_memory_allocated()` after a warmup call.

> Pi0-FAST is autoregressive — total latency = `prefill + N × per-token decode`,
> where `N` is the number of action tokens generated per inference (variable,
> depends on the action sequence; typically 30–80 tokens). Throughput is reported
> per token, not per inference, since "1 inference" is not a fixed unit.

> Pi0 RTX 5090 latencies are steady-state p50 over 200 timed `infer` calls
> (CUDA-graph replay, 100 warmup) on real LIBERO frames. JAX (Orbax) and
> PyTorch (safetensors) frontends drive the same compiled pipeline, so
> their measured latencies are within 0.1 ms of each other at every view
> count.

### Comparison

| Solution | Hardware | Pi0 | Pi0.5 | GROOT N1.6 | Source |
|---|---|---|---|---|---|
| Original openpi (JAX, unoptimized) | Jetson Thor | — | **714 ms (1.4 Hz)** | — | [openpi](https://github.com/Physical-Intelligence/openpi) |
| PyTorch naive | RTX 4090 | — | ~200 ms | — | HuggingFace LeRobot |
| torch.compile | RTX 4090 | — | ~40 ms | — | HuggingFace LeRobot |
| Triton-based VLA | RTX 5090 | — | 26.6 ms (2v) | — | arXiv 2510.26742 |
| NVIDIA VLA-Perf | RTX 4090 | 31.06 ms (Pi0 3B) | — | — | arXiv 2602.18397 |
| NVIDIA Isaac GR00T (TensorRT) | Jetson Thor | — | 91–95 ms (3v) | ~95 ms | [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| **FlashRT** | **RTX 5090** | **21.16 ms** (2v) | **17.58 ms** (2v) | **13.08 ms** (T=50, 2v) | this work |
| **FlashRT** | **Jetson Thor** | **46 ms** (2v) | **39.78 ms** (2v) / **51.51 ms** (3v) (NVFP4) | **45 ms** (T=50, 2v) | this work |

On the same Jetson AGX Thor hardware, FlashRT goes from the original openpi JAX baseline (1.4 Hz) to 23 Hz (FP8) / 25 Hz (NVFP4) — a **~16-18× speedup at zero accuracy loss** (cosine ≥ 0.9996 vs the production reference).

FlashRT Pi0.5 Thor numbers above are the NVFP4 production preset (`use_fp4=True`); the FP8 baseline is 44.0 ms 2v / 54.8 ms 3v at the same task success (491/500). See [Latency (Thor)](#latency-thor) for the full sweep.

### Tested hardware + what's theoretically supported

**Verified working on: RTX 5090, RTX 4090, RTX 4060 Ti, Jetson AGX Thor.**

CMake's `ENABLE_FA2` gate accepts **any card in SM80 / 86 / 89 / 120**
(Ampere through Blackwell consumer). That means A100, A10, RTX 3090,
3080, A5000/A6000, 4090, 4080, 4070, 4060 Ti, 5090 — all *should*
build and run out of the box. "Theoretical" here just means the
other cards haven't gone through the regression suite yet; the
kernel set and dispatch paths are the same.

### Help needed — hardware, robots, models

This is a solo project. If you have access to any of the following
and are willing to kick the tires, please open an issue or PR with
your numbers / logs:

- **Other GPUs** (A100 / 3090 / 4080 / 4070 / 4060 Ti / AGX Orin / etc.) —
  run `python examples/quickstart.py --checkpoint <...> --benchmark 20`
  and paste the P50 number plus `nvidia-smi` output.
- **Real robot deployments** (LeRobot, custom arms, humanoid
  platforms) — smoothness, crash-safety, end-to-end latency
  including robot-side overhead.
- **New VLA / generative models** — Pi0.6, GR00T later versions,
  custom DiT / audio-gen / video-gen backbones. See
  [`docs/adding_new_model.md`](docs/adding_new_model.md) for the
  integration walkthrough; [`docs/kernel_catalog.md`](docs/kernel_catalog.md)
  has the parts list and a re-use decision tree for judging
  whether FlashRT fits before you start wiring anything up.

Drive-by benchmarks, bug reports, and "this crashed on my X" traces
are all welcome. The footprint is small — one author, one laptop,
two reference GPUs — so every independent data point genuinely
moves the project forward.

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

# Pi0 (continuous state input):
model = flash_rt.load_model(
    checkpoint="/path/to/pi0_checkpoint",
    config="pi0",
)
actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
)
# Pi0: actions shape (10, 7)
# Note: Pi0 also accepts 'state' in observation dict for continuous state input

# GROOT N1.6:
model = flash_rt.load_model(
    checkpoint="/path/to/groot_checkpoint",
    config="groot",
)
actions = model.predict(images=[base_img, wrist_img], prompt="pick up the red block")
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
replacement for `OpenAI(base_url=...)`), see
[`examples/qwen36_openai_server.py`](examples/qwen36_openai_server.py):

```bash
export FLASHRT_QWEN36_MTP_CKPT_DIR=/path/to/qwen36_fp8_ckpt
python examples/qwen36_openai_server.py \
    --checkpoint /path/to/qwen36_nvfp4 \
    --port 8000 --K 6
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

Latency columns below are **2-view**, pure CUDA Graph replay (p50, see
[Measurement protocol](#measurement-protocol)). All per-view
breakdowns live in the Latency sections further down.

| Model | Architecture | Latency (Thor, 2v) | Latency (RTX 5090, 2v) | Source |
|-------|-------------|:-:|:-:|--------|
| [**Pi0.5**](https://github.com/Physical-Intelligence/openpi) | PaliGemma 2B encoder + 300M decoder, 10-step diffusion | **44 ms** | **17.58 ms** | Physical Intelligence |
| [**Pi0**](https://github.com/Physical-Intelligence/openpi) | Same as Pi0.5, with continuous state input | **46 ms** | (Thor class w/ SM120 fork) | Physical Intelligence |
| [**GROOT N1.6**](https://github.com/NVIDIA/Isaac-GR00T) | Eagle3-VL + Qwen3 1.7B + AlternateVLDiT 32L, 4-step flow matching | **45 ms** (T=50) / **41 ms** (T=16) | **13.08 ms** (T=50) / **12.53 ms** (T=16) | NVIDIA |
| [**Pi0-FAST**](https://github.com/Physical-Intelligence/openpi) | Gemma 2B autoregressive, FAST tokenizer | **8.1 ms/token**, ~431 ms (50 tok) | **2.39 ms/token**, ~140 ms (50 tok, max-perf) | Physical Intelligence |

---

## Hardware Support

| Feature | Thor (SM110) | RTX 5090 (SM120) |
|---------|:----------:|:----------:|
| FP8 GEMM | CUTLASS | cuBLASLt |
| NVFP4 GEMM | — | CUTLASS |
| Attention | CUTLASS FMHA | FlashAttention-2 |
| CUDA Graph | Full E2E | Full E2E |
| Status | **Production** | **Production** |

---

## Precision (Thor, 2-view LIBERO)

Cosine similarity measured with matched noise injection.

| Comparison | Cosine |
|-----------|--------|
| FlashRT Torch vs Production | **0.9996** |
| FlashRT JAX vs Production | **0.9999** |
| FlashRT Torch vs JAX | **0.9998** |

**Module-level byte-exact verification** (same input → same output):
- SigLIP (27 layers): byte-exact
- Encoder (18 layers): byte-exact
- Decoder (18 layers × 10 steps): byte-exact

## Latency (Thor)

### Pi0.5

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch** | **36.5 ms** (27 Hz) | **44.0 ms** (23 Hz) | **54.8 ms** (18 Hz) |
| **FlashRT JAX** (autotune=5) | **37.3 ms** (27 Hz) | **44.9 ms** (22 Hz) | **54.4 ms** (18 Hz) |
| NVIDIA TensorRT baseline | — | 91–95 ms | — |

### Pi0

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch** (autotune=5) | **37.6 ms** (27 Hz) | **45.8 ms** (22 Hz) | **56.7 ms** (18 Hz) |
| **FlashRT JAX** (autotune=5) | **37.8 ms** (26 Hz) | **45.8 ms** (22 Hz) | **55.9 ms** (18 Hz) |

Each additional camera view adds ~6 ms (256 extra SigLIP tokens → more encoder DRAM traffic + SigLIP forward).

E2E precision: cosine **0.998** vs FP16 PyTorch reference (Torch and JAX both).

### GROOT N1.6

| Stage | T=16 (LIBERO) | T=50 (padded max) | Method |
|-------|---------------|-------------------|--------|
| SigLIP (2 views, CUDA Graph) | 6.0 ms | 6.0 ms | Batched 2-view + Graph |
| Qwen3 16L (CUDA Graph) | 8.8 ms | 8.8 ms | FP8 GEMM (calibrated act scales) + C kernel attention |
| DiT 32L x 4 steps (CUDA Graph) | 26 ms | 30 ms | FP8 + cuBLASLt epilogue fusion + cross-KV precompute |
| **Full E2E (image to action)** | **41 ms** (24 Hz) | **45 ms** (22 Hz) | All CUDA Graph |

T = action_horizon. T=50 is the padded max across all embodiments (used in production). T=16 is LIBERO-specific.

E2E precision: cosine **0.999** vs FP32 PyTorch reference. NVIDIA PyTorch baseline: ~95 ms.
FP8 activation scales calibrated per-layer for both Qwen3 and DiT, cached to `~/.flash_rt/calibration/`.

### Pi0-FAST

Pi0-FAST is a fundamentally different architecture from Pi0/Pi0.5 — actions are
generated as **discrete FAST tokens via autoregressive decoding** through a
single Gemma 2B model, not via diffusion. The FP8 inference path uses **BF16
residual stream** for both prefill and decode (Pi0-FAST hidden states reach
~569K, exceeding FP16's 65504 limit) with **FP8 GEMM** on weights.

**Jetson AGX Thor (SM110)**

| Mode | Per-token | 50-token E2E | Method |
|------|-----------|-------------|--------|
| **Default** (`decode_cuda_graph=False`) | **8.7 ms** | **~464 ms** | CUTLASS FP8 wide GEMM, vocab pruning, prefill CUDA Graph, text-phase logit skip |
| **Max-perf** (`decode_cuda_graph=True`) | **8.1 ms** | **~431 ms** | + decode loop captured as CUDA Graph |

**RTX 5090 (SM120)** — measured on Blackwell consumer silicon

| Mode | Prefill | Per-token | 50-token E2E | Throughput |
|------|---------|-----------|-------------|------------|
| **Default** (`decode_cuda_graph=False`) | **12.08 ms** | **2.87 ms** | **155.5 ms** | **348 tok/s** |
| **Max-perf** (`decode_cuda_graph=True`)  | **10.99 ms** | **2.39 ms** | **140.3 ms** | **418 tok/s** |

On RTX 5090 the entire SM100 CUTLASS FP8 kernel family is replaced by cuBLASLt
`fp8_gemm_descale_{fp16,bf16out}` (one runtime-gated fork in `pipeline_pi0fast.py`,
Thor path byte-for-byte unchanged). SigLIP attention uses `torch.nn.functional.
scaled_dot_product_attention` in place of the SM100-only strided FMHA.
The decode Down GEMM (`[M=1, N=2048, K=16384]`) hits a cuBLASLt heuristic gap
for K≥8192 at small M on SM120, so it's dispatched through a 4-way split-K
workaround — costs ~0.5 ms/token of the gap between the current 2.39 ms and
the 1.5–2.0 ms DRAM roofline. Numbers measured with 30-iteration median on a
fixed-image benchmark (min/max within 0.25 ms of median in graph mode).

**Speedup vs Thor SM110**: 3.07× in graph mode, 2.77× no-graph.

Total inference latency = `prefill + N × per_token_decode` where `N` is the
number of action tokens generated (variable per inference, typically 30–80).
Vocab pruning is automatic: once the model enters action token range, the
logit projection drops from 257K → 2K vocab (saves ~5 ms/token).

**Backend equivalence vs JAX bf16 reference** (per-segment cosine on identical prefix):

| Backend | Prefill xn | First logit | Decode xn | Decode logit | First token |
|---------|-----------|-------------|-----------|--------------|-------------|
| **FlashRT Torch** | 0.998 | 0.999 | 0.995 | 0.998 | MATCH (4022) |
| **FlashRT JAX**   | 0.995 | 0.997 | 0.987 | 0.993 | MATCH (4022) |

Both backends match JAX's first decoded token exactly, with all internal hidden
states ≥ 0.987 cosine vs the JAX bf16 reference (gemma_fast.Module.apply).
Run `python tests/test_pi0fast_precision.py` to verify on your hardware.

## Latency (RTX 5090)

### Measurement protocol

All RTX 5090 numbers in this section are **pure CUDA Graph replay p50**
(`cuda.Event` around `graph.replay()`), not the end-to-end
`quickstart.py` wall clock. The two differ by roughly 1–3 ms because
replay excludes graph-external work — image normalization, H2D upload,
D2H actions download, post-process un-normalization, Python wrapper —
which any real production caller has to pay, but which isn't part of
the engine itself.

| Metric | What it counts | Use it for |
|--------|----------------|-----------|
| **replay** (`cuda.Event` around `graph.replay()`) | GPU kernels only, captured graph(s) | Engine latency, comparisons between backends, apples-to-apples vs other kernel-level reports |
| **wall** (`time.perf_counter()` around `rtx.infer()`) | Everything inside `rtx.infer`: copies, graph, sync, decode, un-normalize | What a Python caller feels |

Replay is the canonical FlashRT benchmark column because:
- it's compiler/CPU independent (same kernels → same replay regardless of
  whether the Python wrapper is on Thor's Arm CPU or a 5090 host x86),
- it's what other framework benchmarks (NVIDIA Isaac, Triton-based VLA work)
  typically report,
- wall-clock picks up noise from image preprocessing and Python GC that
  shifts with host CPU, not GPU.

All replay runs below use `--warmup 50 --iters 500`. 500 warmup iters
on RTX 5090 actually lands in a slightly slower DVFS state than 50 for
small workloads (1v/2v Pi0.5 replay drifts by ~1 ms), so 50 is a better
"hot and honest" warmup than blindly cranking it higher.

### Reproducing

After the model is loaded and `set_prompt(...)` has been called once
(so the CUDA Graph is captured and the first inference is warm), time
graph replays directly:

```python
import torch, flash_rt, statistics

model = flash_rt.load_model("pi05", "/path/to/ckpt", framework="torch")
model.predict(images=[base, wrist], prompt="task")  # warm

graph = model._pipe._enc_ae_graph
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# 50 warmup, 500 measured
for _ in range(50): graph.replay()
torch.cuda.synchronize()

times_ms = []
for _ in range(500):
    start.record(); graph.replay(); end.record()
    torch.cuda.synchronize()
    times_ms.append(start.elapsed_time(end))

print(f"P50 replay: {statistics.median(times_ms):.2f} ms")
```

Swap `config="groot"` and `action_horizon=50/16` for the GROOT rows.

### Pi0.5

**RTX 5090 (SM120) — FP8 baseline, torch**:

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch (replay p50)** | **14.48 ms** (69 Hz) | **17.58 ms** (57 Hz) | **20.00 ms** (50 Hz) |
| (Wall p50 for reference) | 15.92 ms | 19.58 ms | 23.24 ms |

Replay std across 500 timed iterations is ~0.2 ms (1v) / 0.56 ms (2v) /
0.54 ms (3v). Per-view delta is ~3 ms — the cost is dominated by SigLIP
forward + patch embed, both linear in `num_views`.

E2E precision: cosine **0.998** vs FP16 PyTorch reference.

**Jetson AGX Thor (SM110) — torch, N=8 LIBERO stratified calibration**:

| Config | 1-view | 2-view | 3-view |
|--------|--------|--------|--------|
| **FP8 baseline (replay p50)** | **34.06 ms** (29 Hz) | **41.79 ms** (24 Hz) | **55.46 ms** (18 Hz) |
| **NVFP4 encoder (`use_fp4=True`)** | **31.91 ms** (31 Hz) | **39.78 ms** (25 Hz) | **51.51 ms** (19 Hz) |
| FP4 speedup vs FP8 | −2.15 ms (−6.3%) | −2.01 ms (−4.8%) | −3.95 ms (−7.1%) |

Measured via
[`tests/bench_pi05_thor_views.py`](tests/bench_pi05_thor_views.py);
50-iteration graph-replay P50. The NVFP4 row is the production preset
(full 18 encoder FFN layers + AWQ + P1 split-GU — see
[§NVFP4 encoder FFN](#nvfp4-encoder-ffn-pi05-only) below).

**Encoder FP4 holds cosine ≥ 0.9989 vs PyTorch FP32 reference at every
view count, with no latency regression (FP4 is actually faster than
FP8 on every row).** At 3v against the PyTorch FP32 reference:
`cos = 0.998932`, `maxdiff = 0.0372` — slightly tighter maxdiff than
the FP8 baseline (0.0414) thanks to the multi-sample AWQ refit.

### GROOT N1.6

`gr1` is the representative trained embodiment (see
[GROOT N1.6 embodiment slots](#groot-n16-embodiment-slots) above —
don't run the default `new_embodiment` against the base checkpoint
unless you've fine-tuned it, you'll get noise-like actions).

GROOT's rtx pipeline captures **three separate CUDA graphs** (SigLIP,
Qwen3, DiT) with non-graph torch work between them (pixel unshuffle +
mlp1, kv_text/kv_img split, state encode, cross-KV precompute). The
`replay` figure below sums the three captured-graph sections — each
one replayed on its own captured stream with a sync between stages —
to mirror the production ordering while excluding the interleaved
non-graph torch work. This is why it's noticeably below the wall
number: GROOT has more graph-external work than Pi0.5.

**T = 50** (padded max across all embodiments — production default)

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch (replay p50)** | **11.90 ms** (84 Hz) | **13.08 ms** (76 Hz) | **13.92 ms** (72 Hz) |
| (Wall p50 for reference) | 12.77 ms | 15.60 ms | 15.23 ms |

**T = 16** (LIBERO-style short horizon — skips ~34 rows in every DiT block)

| Frontend | 1-view | 2-view | 3-view |
|----------|--------|--------|--------|
| **FlashRT Torch (replay p50)** | **11.31 ms** (88 Hz) | **12.53 ms** (80 Hz) | **13.36 ms** (75 Hz) |
| (Wall p50 for reference) | 12.18 ms | 15.06 ms | 14.66 ms |

Replay std < 0.02 ms across all 6 cells — the graphs are deterministic
once captured. Per-view delta is only ~1 ms because GROOT's SigLIP
feeds through a 2×2 `pixel_unshuffle` that packs 256 patches per view
into 64 tokens, so the extra camera adds much less compute than
Pi0.5's 256-token-per-view path.

T=16 → T=50 costs only ~0.5 ms — most DiT-step cost is norm + the
shared state-row self-attn + cross-attn with the Qwen3 backbone
features, none of which grow with T. The linear-in-T pieces (action
encoder/decoder MLPs, a slice of DiT self-attn QKV) only account for
~1 ms of the total.

E2E precision: cosine **0.9992** vs Isaac-GR00T `Gr00tN1d6` reference
on `gr1`, matched noise + matched post-vlln backbone features. The
reference run requires the Isaac-GR00T stack (torch 2.7.1 /
transformers 4.51.3 / cp310 wheels), which cannot coexist with the
rtx kernel build environment in the same venv — drive both via
separate venvs and compare the saved tensors offline.

### Pi0-FAST

50-token end-to-end, Orbax/JAX frontend, RTX 5090:

| Mode | Quickstart P50 (50-token E2E) | Throughput |
|------|-------------------------------|------------|
| **Default** (`decode_cuda_graph=False`) | **147.4 ms** | **~340 tok/s** |
| **Max-perf** (`decode_cuda_graph=True`)  | **122.9 ms** | **~410 tok/s** |

```bash
# Default
python examples/quickstart.py \
    --checkpoint /path/to/pi0_fast_base \
    --config pi0fast --framework jax \
    --max_steps 50 --benchmark 20 --warmup 5

# Max-perf (decode loop captured as CUDA Graph; 50-token fixed horizon)
python examples/quickstart.py \
    --checkpoint /path/to/pi0_fast_base \
    --config pi0fast --framework jax \
    --decode_cuda_graph --decode_graph_steps 46 \
    --max_steps 50 --benchmark 20 --warmup 5
```

These are wall-clock end-to-end numbers (prefill + all decode tokens).
The per-token breakdown — 12 ms prefill / 2.87 ms per decode token
default, 11 ms / 2.39 ms max-perf — is measured with a 30-iteration
median benchmark in the Pi0-FAST detailed table above.

## LIBERO Benchmark (Thor, Pi0.5)

| Suite | Torch | JAX |
|-------|-------|-----|
| **LIBERO Spatial** (10 tasks × 50 ep) | **492/500 = 98.4%** | **490/500 = 98.0%** |
| **LIBERO 10** (10 tasks × 50 ep) | **465/500 = 93.0%** | **463/500 = 92.6%** |

---

## Acknowledgments

- [CUTLASS](https://github.com/NVIDIA/cutlass) — GEMM templates and FMHA kernels
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Attention backend for SM89/SM120
- [Physical Intelligence](https://www.physicalintelligence.company/) — Pi0/Pi0.5 model architecture
- [OpenPI](https://github.com/Physical-Intelligence/openpi) — Reference PyTorch implementation
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — GROOT N1.6 model
