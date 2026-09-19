# TensorRT Backend on TensorRT Edge-LLM (Pi0.5 on Jetson AGX Thor)

The FlashRT TensorRT backend is a TensorRT plugin library plus an ONNX
exporter. The same plugin library and the same engine run under plain
TensorRT ([tensorrt_usage.md](tensorrt_usage.md)) and under
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM). This
document covers the Edge-LLM side: how the example fits Edge-LLM, which
Edge-LLM interfaces it uses, how to reproduce the results and what is not
integrated yet. How the plugins are built is described in
[tensorrt_backend.md](tensorrt_backend.md).

## 1. What runs where

```
 TensorRT Edge-LLM build (experimental_models/pi05, overlay)
   pi05_policy_inference
     image loading, resize          runtime/imageUtils.h
     prompt tokens                  tokenizer/tokenizer.h (tokenizer.json)
     engine load, CUDA graph        common/trtUtils.h, CUDA runtime API
     openpi action unnormalization  (in the example)
                 │
                 ▼
 TensorRT engine  pi05.engine   (built by tools/build_pi05_engine.sh: ONNX → trtexec)
   images, lang_tokens, noise ─► Pi05Siglip ─► Pi05Encoder ─► Pi05Decoder ─► actions
                 │
                 ▼
 plugin library  libflashrt_trt_pi05.so   (IPluginV3, getCreators, domain "flashrt")
   FlashRT native stages and kernels: NVFP4 CUTLASS GEMMs, FP8 GEMMs, FlashAttention-4 (CuTe DSL AOT)
```

| | provided by |
|---|---|
| host process, image and tokenizer handling, CUDA graph capture | Edge-LLM |
| graph, memory planning, execution context | TensorRT |
| SigLIP, prefix encoder and 10-step action decoder arithmetic | FlashRT plugins |

Nothing in Edge-LLM is modified except one `add_subdirectory(pi05)` line in
`experimental_models/CMakeLists.txt`.

## 2. How it follows Edge-LLM's structure

| Edge-LLM convention | what the example does |
|---|---|
| experimental models live in `experimental_models/<model>/` with their own CMake target (as `experimental_models/cosmos3`) and build with `-DBUILD_EXPERIMENTAL_MODELS=ON` | `integrations/edgellm/pi05/` is linked there by `install_overlay.sh`; target `pi05_policy_inference` links `edgellmCore`, `exampleUtils`, `commonLibraryExt` and calls `add_cross_build_link_options` |
| plugins are `IPluginV3` (`cpp/plugins/`) | all FlashRT plugins are `IPluginV3` (`IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`, created by `IPluginCreatorV3One`), the same interfaces as Edge-LLM's plugins; the library exports them through `getCreators` |
| performance-critical attention kernels come from CuTe DSL, compiled ahead of time for the target | FlashAttention-4 is exported ahead of time from CuTe DSL (`tools/export_fa4_aot.py`) and linked into the plugin library as object files |
| action runners (Cosmos3 policy) replay one CUDA graph per engine call | the example captures the whole policy engine call with `captureTRTCudaGraph` and replays it with `cudaGraphLaunch` |
| tokenizers load `tokenizer.json` | `tools/export_paligemma_tokenizer.py` writes `tokenizer.json` for openpi's PaliGemma SentencePiece model and checks it tokenizes identically |

Edge-LLM interfaces used by `pi05_policy_inference.cpp`:

| header | used for |
|---|---|
| `common/trtUtils.h` | `deserializeCudaEngineFromFile`, `captureTRTCudaGraph` |
| `cuda_runtime.h` | `cudaMalloc`, `cudaMemcpy`, `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaGraphLaunch` |
| `runtime/imageUtils.h` | `loadImageFromFile`, `resizeImage` |
| `tokenizer/tokenizer.h` | `Tokenizer::loadFromHF`, `Tokenizer::encode` |
| `common/logger.h`, `common/checkMacros.h`, `common/tensor.h` | `gLogger`, `ELLM_CHECK`, `CUDA_CHECK`, `rt::Tensor` |
| TensorRT `NvInferRuntime.h` | `IRuntime::getPluginRegistry().loadLibrary` (FlashRT plugins, before deserializing), `IExecutionContext::setInputShape`, `setTensorAddress`, `enqueueV3` |

Engine I/O is the backend's standard policy engine
([tensorrt_usage.md](tensorrt_usage.md) §4): `images` fp16
`[cameras, 224, 224, 3]` in [-1, 1], `lang_tokens` int32 `[n]`, `noise` fp16
`[10, 32]`, output `actions` fp16 `[10, 32]`.

## 3. Reproduce on Jetson AGX Thor

Checked with JetPack 7.2 (CUDA 13.2, TensorRT 10.16.2 on the host), MAXN, and
TensorRT Edge-LLM v0.10.1. Everything below runs on the host; the engine must
be built with the same TensorRT that Edge-LLM links.

### 3.1 Plugin library and engine

Follow [tensorrt_usage.md](tensorrt_usage.md) §1–4 on the host: build FlashRT
and `build/tensorrt/libflashrt_trt_pi05.so`, create the calibration
observations and build the engine:

```bash
backends/tensorrt/tools/build_pi05_engine.sh <pi05_libero_pytorch> 2 libero_calib_8.npz out/pi05_libero
# ... ENGINE_PROMPT_DYNAMIC_PASS
```

`out/pi05_libero` then holds `pi05.engine` and the FlashRT references
(`siglip_all.safetensors`, `prompts.safetensors`) used by the check below.

### 3.2 Tokenizer

With `transformers` and `sentencepiece` installed (the openpi Thor container
has both):

```bash
python backends/tensorrt/tools/export_paligemma_tokenizer.py \
    ~/.cache/openpi/big_vision/paligemma_tokenizer.model out/tokenizer
# bos 2 2 | all texts match: True
```

### 3.3 Build Edge-LLM with the example

```bash
git clone --recurse-submodules --branch v0.10.1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
<FlashRT>/backends/tensorrt/integrations/edgellm/install_overlay.sh .
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DCUDA_CTK_VERSION=13.2 \
    -DENABLE_CUTE_DSL=ALL \
    -DBUILD_EXPERIMENTAL_MODELS=ON
cmake --build build --target pi05_policy_inference -j 2
# -> build/experimental_models/pi05/pi05_policy_inference
```

These are the Jetson Thor (JetPack 7.2) options of Edge-LLM's installation
guide (`docs/source/user_guide/getting_started/installation.md`, which also
lists its system packages) plus `BUILD_EXPERIMENTAL_MODELS`. Only the
`pi05_policy_inference` target and its dependencies are built.

### 3.4 Check against FlashRT

```bash
python backends/tensorrt/integrations/edgellm/check_pi05.py \
    <TensorRT-Edge-LLM>/build/experimental_models/pi05/pi05_policy_inference \
    out/pi05_libero out/tokenizer
```

The script runs the example for the four LIBERO prompts recorded by
`build_pi05_engine.sh`, on the recorded observation and noise, and compares
the prompt tokens and raw actions with FlashRT's own output:

```
prompt 0: 14 tokens | tokens match True | raw actions bitwise True (max diff 0) | median 23.85 ms, cuda graph True
...
EDGELLM_PI05_PASS
```

Run it with the Python environment FlashRT is installed in (it also needs
`pillow` and `safetensors`): the bitwise
comparison loads the cuBLAS build PyTorch ships (FlashRT's references use it)
and passes `--pixel_norm flashrt` (FlashRT's uint8 table). With the system
cuBLAS the actions differ by about 1e-5, which does not affect accuracy.

### 3.5 Run on your own inputs

```bash
pi05_policy_inference --engine out/pi05_libero/pi05.engine \
    --plugin build/tensorrt/libflashrt_trt_pi05.so \
    --tokenizer out/tokenizer --images base.png,wrist.png \
    --prompt "put the bowl on the plate" \
    --norm_stats <pi05_libero_pytorch>/assets/physical-intelligence/libero/norm_stats.json \
    --output actions.json --iters 200
```

The output has the prompt tokens, `raw_actions` `[10, 32]`, `actions`
`[10, action_dim]` after openpi's quantile unnormalization, and
`latency_ms`. Images are one file per unmasked camera in openpi order (LIBERO:
base, wrist), resized to 224×224. `--noise` takes a safetensors file with an
fp16 `noise_in` or `noise` tensor, otherwise noise comes from `--seed`.
`--no_cuda_graph` runs `enqueueV3` directly.

## 4. Results

Jetson AGX Thor, JetPack 7.2, MAXN, openpi `pi05_libero`, two cameras:

| check | result |
|---|---|
| prompt tokens, four LIBERO prompts | equal to openpi's tokenization |
| raw actions, four LIBERO prompts | bitwise equal to FlashRT's runtime |
| `actions` after unnormalization | equal to openpi's output transform to 1.2e-7 |
| latency of one policy call (CUDA graph replay, median of 200) | 23.9 ms; 25.7 ms when `10 + tokens + 256 × cameras` is not a multiple of 8 |

The latency covers the engine call only (SigLIP, prefix encoder and 10
denoising steps), not image decoding or tokenization. For reference, the
openpi Thor tutorial's FP8 + NVFP4 engine measures 48.12 ms model time under
the tutorial's `pi05_inference.py`, and this engine 26.02 ms under the same
script; see [tensorrt_usage.md](tensorrt_usage.md) for those measurements and
the accuracy comparison.

## 5. Integration depth

| Edge-LLM part | status |
|---|---|
| C++ runtime utilities, tokenizer, CUDA graphs | used by the example |
| plugin loading | the example loads the FlashRT library into TensorRT's plugin registry itself; Edge-LLM's `EDGELLM_PLUGIN_PATH` loading of `libNvInfer_edgellm_plugin.so` is not used by the example (loading both in one process is not tested yet) |
| engine build | `trtexec` on the backend's ONNX, not Edge-LLM's Python export or engine builder |
| LLM runtime (`LLMInferenceRuntime`, KV cache manager) | not used: Pi0.5 has no autoregressive decoding; the prefix KV cache lives inside the engine |
| action runner interface | not used: the example is a standalone program |

Directions that need agreement with the Edge-LLM maintainers:

1. **Action runner.** Wrap the policy engine in an Edge-LLM action runner
   (inputs: camera images, prompt, noise; output: actions) so a Pi0.5 engine
   plugs into the same serving and example code as other policies.
2. **Builder integration.** Map the Pi0.5 stages to the FlashRT plugin
   creators in Edge-LLM's engine builder, so `tensorrt_edgellm` can build the
   engine from a checkpoint instead of `build_pi05_engine.sh`.
3. **Plugin library loading.** Allow Edge-LLM to load additional plugin
   libraries next to `libNvInfer_edgellm_plugin.so`, so FlashRT plugins do not
   need a custom host.

## 6. Taking the kernels without taking the model

A whole pi0.5 stage is one way in; individual operators are another, and they
do not require Edge-LLM to know what pi0.5 is.
`backends/tensorrt/kernels/frt_ops.h` is a C ABI over raw device pointers with
no model semantics, so an Edge-LLM plugin can call it from its own shell. The
kernels behind it are CUDA C++ with CUTLASS and ahead-of-time compiled CuTe
DSL — the two forms Edge-LLM already builds with.

What is worth taking, measured on Thor against TensorRT at the shape the
openpi tutorial's own engine runs — three camera slots and a 208-token prompt
(method, both shapes and full tables in [tensorrt_ops.md](tensorrt_ops.md)):

| FlashRT operator | what a TensorRT graph does instead | measured |
|---|---|---|
| `FlashrtNvfp4Mlp` — gate/up epilogue emits NVFP4, down GEMM consumes it | dense NVFP4/FP8 linear layers lower to Q/DQ around two GEMMs, so the hidden activation lands in fp16 | **1.41×** (encoder FFN, NVFP4 both sides), **1.09×** (SigLIP FFN, FP8 there) |
| `FlashrtFa4Attention` — FlashAttention-4, head_dim 72 and head_dim 256 GQA | the `Attention` operator, or MatMul/Softmax/MatMul when the export does not use it | **1.72×** (head_dim 72) and **1.54×** (head_dim 256 GQA) against `Attention`; **3.48×** against the chain |
| `FlashrtNvfp4Linear` — quantize then block-scaled GEMM | one GEMM with the quantization fused into its prologue | **1.08×** here, **0.94×** at the smaller deployment shape — the one operator where the two are close, and the one where an isolated boundary costs about what it saves |

The last row is the useful one for deciding where a boundary belongs: an
isolated quantized GEMM is already well served, and the gain is in what a
plugin can keep *between* kernels. That is why FlashRT's plugins are cut at
fusion boundaries rather than at operator boundaries, and why the operator
layer exists anyway — a host that wants finer pieces can have them and see
what they cost ([tensorrt_ops.md](tensorrt_ops.md) §4).

## 7. Troubleshooting

| symptom | cause |
|---|---|
| `failed to load plugin library` | wrong `--plugin` path, or the library was built against a different TensorRT |
| engine fails to deserialize | the engine was built with a different TensorRT than Edge-LLM links (for example inside a container); rebuild it on the host |
| `lang_tokens length ... outside the engine profile` | prompt longer than the engine's `MAX_TOKENS` |
| `engine takes N images` | pass one image per unmasked camera the engine was built for |
| check reports max diff ~1e-5 instead of bitwise | system cuBLAS instead of PyTorch's: run the check with FlashRT's Python environment |
| `pi05_policy_inference` target not found | configure Edge-LLM with `-DBUILD_EXPERIMENTAL_MODELS=ON` after `install_overlay.sh` |
