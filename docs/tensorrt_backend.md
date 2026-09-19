# TensorRT Backend: How FlashRT Pipelines Become TensorRT Plugins

This document describes how the FlashRT TensorRT backend packages a FlashRT
inference pipeline as TensorRT plugins, and how to keep the plugins in step
with FlashRT. Usage is covered in [tensorrt_usage.md](tensorrt_usage.md).

The first packaged pipeline is Pi0.5 on Jetson AGX Thor (SM110), from the
production NVFP4 + FA4 configuration in [pi05_thor.md](pi05_thor.md).

## 1. Design

TensorRT supports custom layers through plugins (`IPluginV3`). A plugin that
wraps one operator splits the graph into many small TensorRT islands and
throws away FlashRT's fusions: in Pi0.5 the fused kernels cross the FFN,
residual, normalization and quantization boundaries. The backend therefore
places plugin boundaries where FlashRT's own fusions end:

| granularity | plugins | use |
|---|---|---|
| stage | `Pi05Siglip`, `Pi05Encoder`, `Pi05Decoder` | whole-policy engines (default) |
| layer / step | `Pi05SiglipLayer`, `Pi05EncoderLayer`, `Pi05DecoderStep` | custom graphs, per-layer placement |
| operator | `FlashrtNvfp4Linear`, `FlashrtNvfp4Mlp`, `FlashrtFa4Attention` | a host that wants FlashRT kernels in a graph of its own, with no pi0.5 in it |

All three are bitwise equal to FlashRT; the layer granularity costs 1.3% over
the stage on the encoder. The operator plugins are model-free and are
documented, with what each granularity costs, in
[tensorrt_ops.md](tensorrt_ops.md).

TensorRT owns the graph, memory, execution context and CUDA graph capture.
The plugins own the arithmetic, and that arithmetic is FlashRT's:

```
 host (trtexec / TensorRT Python or C++ / openpi / TensorRT Edge-LLM)
                            │
 TensorRT engine (ONNX)     ▼
   images ─► Pi05Siglip ─► Concat ◄─ Gather(embedding) ◄─ lang_tokens
                             │
             Slice(RoPE) ─► Pi05Encoder ─► K, V ─► Reshape
                                                     │
   noise ─────────────────────────────────► Pi05Decoder ─► actions
                            │
 plugin library             ▼
   IPluginV3 shims    backends/tensorrt/plugins/
   native stages      csrc/stages/pi05_thor/      (kernel call sequence)
   FlashRT kernels    csrc/{fused_fp4,gemm,kernels,quantize}/, csrc/attention/fa4_aot/
```

Everything outside the three plugins is plain ONNX (Concat, Reshape, Gather,
Cast, Mul, Shape, Slice) with no arithmetic of its own beyond the prompt
embedding scale.

## 2. Layers of the implementation

### 2.1 Native stages — `csrc/stages/pi05_thor/`

Framework-free C++ that runs one stage from raw device pointers:

| file | FlashRT production path it reproduces |
|---|---|
| `pi05_siglip.{h,cu}` | patch embedding, `siglip_forward_with_fp4_ffn`, post-LayerNorm projection |
| `pi05_encoder_layer.{h,cu}` | `encoder_forward_with_fp4_subset` (one layer) |
| `pi05_decoder_step.{h,cu}` | `decoder_forward_fp4` (one denoise step) |
| `fa4_attention.{h,cpp}` | FlashAttention-4 dispatch over the AOT modules |

Each stage calls the same kernels, in the same order, with the same variants
and flags as the FlashRT production branch. The stages take weights, scales
and scratch buffers as arguments and allocate nothing, so they can run inside
a TensorRT workspace and under CUDA graph capture.

### 2.2 FlashAttention-4 modules — `csrc/attention/fa4_aot/`

FlashRT's FA4 is a CuTe DSL JIT kernel. `backends/tensorrt/tools/export_fa4_aot.py`
compiles it through FlashRT's own entry point (`flash_rt.hardware.thor.fa4_backend`)
and exports C modules (header + object with the kernel binary embedded):

| module | attention | site |
|---|---|---|
| `fa4_hd256_fwd` | head dim 256, GQA 8/1, `Sq x 8 > 128` | encoder prefill |
| `fa4_hd256_q1_fwd` | same, `Sq x 8 <= 128` | decoder shapes |
| `fa4_hd72_fwd` | head dim 72, MHA, `Sq > 128` | SigLIP |

The FA4 compile key depends on sequence length only through the query stage
count, hence two hd256 modules. `fa4_default_scale()` computes `1/sqrt(hd)`
the way FA4's Python entry does (in double, then narrowed); the float-domain
quotient is one ulp off for head dim 72.

### 2.3 Plugins — `backends/tensorrt/plugins/`

Thin `IPluginV3` classes: shape inference, format checks, workspace sizing,
and binding TensorRT tensors to the stage's weight and scratch structs.
`pi05_plugins.cpp` exports `getCreators()` for TensorRT's plugin registry.

### 2.4 Operator layer — `backends/tensorrt/kernels/frt_ops.h`

A C ABI over the same kernels, taking raw device pointers and no model
semantics, with `backends/tensorrt/plugins/ops/` as its `IPluginV3` shells.
Stage plugins and operator plugins call the same kernels in the same order.
See [tensorrt_ops.md](tensorrt_ops.md).

### 2.5 Weights and calibration — `backends/tensorrt/tools/reference/`

The backend does not re-implement quantization. FlashRT's Python frontend
calibrates (static FP8 activation scales, AWQ, NVFP4 packing) on real
observations; the recording tools then capture the arguments of the
production forward calls, reproduce every layer in Python with the same
kernel sequence, check the reproduction is bitwise identical to the library
forward, and write weights, scales, inputs and outputs to safetensors:

| tool | records |
|---|---|
| `dump_siglip.py` | SigLIP weights, images, per-layer outputs, image tokens |
| `dump_encoder.py` | encoder weights and scales, prefix input, per-layer K/V, output |
| `dump_decoder.py` | decoder weights, styles, RoPE, prefix K/V, per-step actions |
| `dump_prompts.py` | embedding and RoPE tables; actions for several prompts from one calibration |
| `pi05_pipeline.py` | shared pipeline setup; one-layer encoder reference |

The same files are the ONNX weights (`export_onnx.py`) and the golden
references for the tests.

### 2.6 ONNX graph — `backends/tensorrt/tools/export_onnx.py`

Writes the policy graph with weights as external-data initializers:

- fp16 tensors stay float16;
- packed FP8/NVFP4 bytes become int32 blobs (TensorRT constants take no uint8);
- per-layer host scalars (weight descales) become plugin attributes.

With `--prompts` the prompt is an engine input: `lang_tokens` go through
`Gather(embedding)` and `fp16(fp32(x) * sqrt(D))`, and the encoder and
decoder RoPE rows are sliced from the RoPE table at the prefix length — the
same arithmetic as FlashRT's `set_prompt`. FlashRT keeps the prefix length
even by repeating the last prompt embedding; engine callers repeat the last
token.

## 3. Operator reference (domain `flashrt`, version 1)

The pi0.5 plugins are below; the model-free operator plugins are in
[tensorrt_ops.md](tensorrt_ops.md) §2.

Byte blobs are int32 tensors of `ceil(bytes / 4)` elements. `D`, `H`, `NH`,
`HD`, `L` are hidden size, FFN width, heads, head dim and layers.

### Pi05Siglip / Pi05SiglipLayer

| | Pi05Siglip | Pi05SiglipLayer |
|---|---|---|
| input 0 | `images` fp16 `[views, 224, 224, 3]` in [-1, 1] | `x` fp16 `[views*256, D]` |
| stage inputs | `pe_w [588, D]`, `pe_b`, `pos_emb [256, D]`, `postln_w`, `postln_b`, `proj_w [D, De]`, `proj_b` (fp16) | — |
| per layer (15) | `ln_attn_w`, `ln_attn_b`, `qkv_w` (blob, e4m3 `[D, 3D]`), `qkv_b`, `o_w` (blob), `o_b`, `ln_ffn_w`, `ln_ffn_b`, `awq_inv_s`, `up_packed`, `up_sfb`, `up_b`, `down_packed`, `down_sfb`, `down_b` | same, once |
| output | image tokens fp16 `[views*256, De]` | `x_out` fp16 `[views*256, D]` |
| attributes | `D`, `H_pad`, `NH`, `HD`, `spv`, `up_variant`, `De`, `L`, `alpha` (float[2L]: QKV, O) | `D`, `H_pad`, `NH`, `HD`, `spv`, `up_variant`, `qkv_alpha`, `o_alpha` |

### Pi05Encoder / Pi05EncoderLayer

| | Pi05Encoder | Pi05EncoderLayer |
|---|---|---|
| inputs | `x` fp16 `[Se, D]`, `rope` fp16 `[Se, HD]`, then `qkv_w` (blob), `qkv_scale` (float `[1]`) per layer, then per layer except the last: `o_packed`, `o_sfb`, `awq_inv_s` (fp16), `gu_il_packed`, `gu_il_sfb`, `down_packed`, `down_sfb` | `x`, `rope`, `qkv_w`, `qkv_scale`, the seven tail tensors |
| outputs | `x_out [Se, D]`, `k`, `v` fp16 `[L*Se, HD]` (layer-major) | `x_out`, `k`, `v` `[Se, HD]` |
| attributes | `D`, `H`, `NH`, `HD`, `L`, `attn_o_variant`, `down_variant`, `qkv_alpha` (float[L]) | same with `last`, scalar `qkv_alpha` |

### Pi05Decoder / Pi05DecoderStep

| | Pi05Decoder | Pi05DecoderStep |
|---|---|---|
| inputs | `noise` fp16 `[S, 32]`, `prefix_k`, `prefix_v` fp16 `[L*E*HD]`, `ain_w`, `ain_b`, `aow`, `aob`, `rope [S, 256]`, `sa`, `sf` `[steps*L*S*3D]`, `fs` `[steps*S*3D]`, NVFP4 blobs `qw`, `ow`, `gwil`, `dw` (packed + sfb, concatenated over layers) | same, with one step's `sa`, `sf`, `fs` |
| output | raw actions fp16 `[S, 32]` | noise after one step |
| attributes | `S`, `D`, `H`, `NH`, `HD`, `L`, `steps`, `v_qkv`, `v_o`, `v_gu`, `v_down`, `dt` | same without `steps` |

The K/V cache lives in the plugin workspace; the prefix rows are copied in on
each call (a plugin must not modify engine inputs).

## 4. Verification

Every layer of the stack is checked bit for bit against FlashRT, eagerly and
under an outer CUDA graph (`backends/tensorrt/tests/run_regression.sh`):

| level | tests |
|---|---|
| kernels | `fa4_aot_parity.py` (AOT FA4 vs JIT FA4) |
| native stages | `encoder_layer_parity`, `encoder_stage_parity`, `decoder_step_parity`, `siglip_parity` |
| single plugins and chains | `engine_encoder_layer.py`, `engine_encoder_layer_chain.py`, `engine_encoder_stage.py`, `engine_decoder_step_chain.py`, `engine_decoder_stage.py`, `engine_siglip.py` |
| whole policy | `engine_policy.py` (network API), `engine_file.py` (ONNX + trtexec), `engine_prompt_dynamic.py` (prompt input, several prompts) |

Bitwise gates need an otherwise idle GPU and PyTorch's cuBLAS build (the
decoder's small GEMMs select kernels by cuBLAS version).

## 5. Keeping the backend in sync with FlashRT

| change in FlashRT | what to do |
|---|---|
| kernel implementation, same interface and call order | rebuild the plugin library, re-record, run the regression |
| compile flags of a kernel group | mirror them in `backends/tensorrt/CMakeLists.txt` |
| new kernel source used by a stage | add it to the matching group in `CMakeLists.txt` |
| production call sequence (new fusion, variant, flag) | update the stage in `csrc/stages/pi05_thor/` and the reproduction in `tools/reference/`; the recording tools assert the production flags they reproduce, and the parity tests locate the first diverging layer |
| weight packing or calibration | usually nothing: the recording tools read FlashRT's frontend |
| FA4 source | rerun `export_fa4_aot.py`, then `fa4_aot_parity.py` |

The stage code is the single description of the kernel sequence for this
backend; moving the torch frontend onto the same stage functions would make
both paths share it.

## 6. Adding a pipeline

1. Pick plugin boundaries at the model's fusion boundaries (layer and stage).
2. Write the native stage in `csrc/stages/<model>/` against raw pointers.
3. Record references by reproducing the production forward in Python and
   checking it against the library before writing any C++.
4. Add native parity programs, then `IPluginV3` shims and engine tests.
5. Extend `export_onnx.py` with the graph and I/O.

## 7. TensorRT 10.16 notes

- Register plugins through `getCreators()`; build ONNX models with
  `trtexec --dynamicPlugins` (with `--staticPlugins` the ONNX parser does not
  find the creators). Python hosts call `trt.get_plugin_registry().load_library()`
  before deserializing.
- Constants reject uint8 and plugins accept no uint8 tensors: use int32 blobs
  and fp16 images.
- Dynamic 3D plugin tensors are laid out incorrectly at builder optimization
  level 0: flatten them to 2D or 1D.
- Concatenating plugin outputs crashed execution context creation; mark
  per-layer outputs separately.
- Builder optimization levels above 0 stall about 12.7 s per timing stream
  for these plugins with no benefit; build with `--builderOptimizationLevel=0`.
- ONNX `Slice` bounds need static shapes: take `Slice(Shape(x), [0], [1])`
  rather than `Unsqueeze(Gather(Shape(x), 0))`.
- Serialized engines are tied to the exact TensorRT version; build where you run.
