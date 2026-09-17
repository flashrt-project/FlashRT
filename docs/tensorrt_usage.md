# TensorRT Backend Usage (Pi0.5 on Jetson AGX Thor)

Run FlashRT's Pi0.5 pipeline as a TensorRT engine. How the backend is built
is described in [tensorrt_backend.md](tensorrt_backend.md).

## Results

Jetson AGX Thor, JetPack 7.2 (TensorRT 10.16), MAXN, openpi `pi05_libero`
(two cameras + prompt, 10 denoising steps), openpi `policy.infer` inside the
openpi Thor container, medians of 100 calls:

| engine | prompt | model / total |
|---|---|---|
| openpi Thor tutorial (ModelOpt FP8 + NVFP4) | "put the bowl on the plate" | 47.96 / 48.71 ms |
| FlashRT TensorRT backend | "put the bowl on the plate" | **25.86 / 26.59 ms** |
| FlashRT TensorRT backend | 21-word LIBERO instruction | **24.08 / 24.82 ms** |

Action accuracy against openpi PyTorch (bf16) on 8 LIBERO observations with
pinned noise, cosine over the 7 action dimensions
(`integrations/openpi/compare_openpi_accuracy.py`):

| engine | mean | min |
|---|---|---|
| openpi Thor tutorial | 0.99962 | 0.99945 |
| FlashRT TensorRT backend | 0.99968 | 0.99946 |

The observations are the calibration observations of both engines.

The tutorial engine always computes three camera slots and a 208-token
prompt; the FlashRT engine computes the unmasked cameras and the actual
prompt. At the tutorial engine's shape, `trtexec --useCudaGraph` measures
32.69 ms for the FlashRT engine against 47.76 ms, and FlashRT's native
runtime runs 32.82 ms: the engine runs FlashRT's kernels at FlashRT's speed
and matches its outputs bit for bit.

## 1. Requirements

- Jetson AGX Thor, JetPack 7.2 (CUDA 13.2, TensorRT 10.16), MAXN power mode
  (`sudo nvpmodel -m 0 && sudo jetson_clocks`)
- FlashRT built for Thor with FA4 (see [pi05_thor.md](pi05_thor.md) §2 and
  [INSTALL.md](INSTALL.md)): `cmake -B build -S . -DGPU_ARCH=110`,
  `pip install -e ".[thor-fa4]"`, CUTLASS v4.4.2 at `third_party/cutlass`
  (`git submodule update --init third_party/cutlass`)
- Python packages `onnx` and `sentencepiece` for export
- The PaliGemma tokenizer (`paligemma_tokenizer.model`), found through
  `$FLASH_RT_PALIGEMMA_TOKENIZER`, `~/.cache/flash_rt/` or
  `~/.cache/openpi/big_vision/` (openpi downloads it there)
- A Pi0.5 PyTorch checkpoint (openpi `pi05_libero` converted to PyTorch, or a
  LeRobot Pi0.5 checkpoint)

## 2. Build the plugin library

```bash
cmake -S backends/tensorrt -B build/tensorrt
cmake --build build/tensorrt -j 2
# -> build/tensorrt/libflashrt_trt_pi05.so
```

The library contains FlashRT's kernels, the native stages and the FA4
modules; at run time it needs only CUDA, cuBLAS and TensorRT.

## 3. Calibration observations

FlashRT calibrates FP8 activation scales and AWQ weight scales on real
observations (8 is the production default). From the LIBERO LeRobot dataset,
wherever `openpi` and `lerobot` are installed (for example the openpi Thor
container):

```bash
python backends/tensorrt/tools/make_libero_fixture.py libero_calib_8.npz
```

The `.npz` holds `img_<i>`, `wrist_<i>`, `wrist_right_<i>` (uint8
`[224, 224, 3]`), `state_<i>` and `n`.

## 4. Build an engine

```bash
backends/tensorrt/tools/build_pi05_engine.sh \
    <checkpoint_dir> 2 libero_calib_8.npz out/pi05_libero
```

Arguments: checkpoint, number of unmasked cameras (2 for LIBERO: base and
wrist), calibration observations, output directory. The script calibrates
and records the stages, exports `out/pi05_libero/onnx/pi05.onnx`, builds
`out/pi05_libero/pi05.engine` with `trtexec` and checks the engine against
FlashRT for four LIBERO prompts. Set `PYTHON` if FlashRT is in a virtual
environment, `ONNX_PYTHON` for a separate export environment. The script
sets `CUTE_DSL_ARCH=sm_101a` (the Thor chip name FA4 compiles for with
nvidia-cutlass-dsl 4.5) unless it is already set.

Engine I/O:

| tensor | type | shape | meaning |
|---|---|---|---|
| `images` | fp16 | `[cameras, 224, 224, 3]` | HWC in [-1, 1], openpi camera order |
| `lang_tokens` | int32 | `[n]`, 2 ≤ n ≤ 256 | PaliGemma tokens of the prompt, unpadded |
| `noise` | fp16 | `[10, 32]` | initial flow-matching noise |
| `actions` | fp16 | `[10, 32]` | raw actions (before openpi's unnormalization) |

`n + 256 x cameras` must be even; repeat the last token when it is not.
Prompt tokens follow openpi: `<bos>`, the cleaned instruction, then `"\n"`.

To build elsewhere (for example inside a container with a different
TensorRT), copy the ONNX directory and the plugin library:

```bash
trtexec --onnx=pi05.onnx --dynamicPlugins=libflashrt_trt_pi05.so \
    --stronglyTyped --builderOptimizationLevel=0 --memPoolSize=workspace:2048 \
    --minShapes=lang_tokens:2 --optShapes=lang_tokens:14 --maxShapes=lang_tokens:256 \
    --saveEngine=pi05.engine
```

## 5. Run

### trtexec

```bash
trtexec --loadEngine=pi05.engine --dynamicPlugins=libflashrt_trt_pi05.so \
    --shapes=lang_tokens:14 --useCudaGraph
```

### TensorRT Python

```python
import tensorrt as trt, torch

trt.get_plugin_registry().load_library("libflashrt_trt_pi05.so")  # before deserializing
engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(open("pi05.engine", "rb").read())
ctx = engine.create_execution_context()

images = ...   # torch fp16 cuda [2, 224, 224, 3], uint8 / 127.5 - 1
tokens = ...   # torch int32 cuda [n]
noise = torch.randn(10, 32, dtype=torch.float16, device="cuda")
actions = torch.empty(10, 32, dtype=torch.float16, device="cuda")
ctx.set_input_shape("lang_tokens", tuple(tokens.shape))
for name, t in (("images", images), ("lang_tokens", tokens), ("noise", noise), ("actions", actions)):
    ctx.set_tensor_address(name, t.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
```

Capture `execute_async_v3` in a `torch.cuda.CUDAGraph` per prompt length for
steady-state latency.

### openpi

`backends/tensorrt/integrations/openpi/openpi_flashrt.py` replaces the
openpi Thor tutorial's `setup_pi0_tensorrt_engine` and keeps openpi's
transforms:

```python
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_flashrt import setup_pi0_flashrt_engine

policy = policy_config.create_trained_policy(_config.get_config("pi05_libero"), checkpoint_dir)
policy = setup_pi0_flashrt_engine(policy, "pi05.engine", "libflashrt_trt_pi05.so")
actions = policy.infer(example)["actions"]   # (10, 7)
```

It needs the tutorial's `deployment_scripts` on `PYTHONPATH`.
`compare_openpi_accuracy.py` compares engines with openpi PyTorch on a set of
observations; `benchmark_openpi.py` times `policy.infer` for either engine.

### TensorRT Edge-LLM

`backends/tensorrt/integrations/edgellm` adds a `pi05_policy_inference`
example to a TensorRT Edge-LLM checkout (image loading, `tokenizer.json`
tokenizer, CUDA graphs, openpi unnormalization); see its README.

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `Plugin not found` while parsing ONNX | use `--dynamicPlugins`, or load the library into the plugin registry first |
| engine fails to deserialize | engines are tied to the TensorRT version; rebuild from ONNX |
| `lang_tokens` shape rejected | prompt longer than `MAX_TOKENS`, or odd prefix length |
| engine check not bitwise, max difference ~1e-5 | a different cuBLAS build than FlashRT's references; accuracy is unaffected |
| latency varies with prompt length by ~1.8 ms | decoder attention cuBLAS kernels are slower when `10 + n + 256 x cameras` is not a multiple of 8 |
