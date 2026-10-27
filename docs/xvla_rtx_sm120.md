# X-VLA (`lerobot/xvla-libero`) on RTX SM120

X-VLA is a soft-prompted flow-matching VLA: a Florence-2 DaViT-L vision
tower, a BART encoder, and a 24-layer action transformer that runs ten
denoising steps over a 262-token sequence to emit a 30x20 action chunk.
The RTX backend is a static-shape, fully captured CUDA Graph.

## Supported tuples

| Model | Framework | Hardware | Precision | Status |
|---|---|---|---|---|
| `xvla` | `torch` | `rtx_sm120` (SM120/SM121) | `nvfp4` (default), `fp8`, `bf16` | Validated |
| `xvla` | `jax` | — | — | Not implemented |
| `xvla` | `torch` | `thor` (SM110), `rtx_sm87`, `rtx_sm89`, AMD | — | Not implemented; fail-fast |

`XVLATorchFrontendRtx` checks the compute capability at construction and
raises for anything outside SM120/SM121. There is no silent fallback.

## Install

```bash
pip install -e ".[xvla]"
```

The `xvla` extra adds `transformers` for the BART tokenizer
(`facebook/bart-large`). If `transformers` is unavailable, `set_prompt`
raises a clear error instead of falling back to an untokenized prompt.

## Build

```bash
cmake -B build -S . -DGPU_ARCH=120 -DFLASHRT_BUILD_XVLA=ON -G Ninja \
      -DPython3_EXECUTABLE=<venv>/bin/python \
      -DCUDAToolkit_ROOT=<cuda-13-root>
cmake --build build -j$(nproc)
```

`FLASHRT_BUILD_XVLA` is OFF by default. The X-VLA sources
(`xvla_rtx.cu`, `xvla_vision_sm120.cu`, `xvla_fp4_gemm_sm120.cu`) and the
additive opt-in entries (`pdl_shared.cu`, `xvla_quantize_pdl.cu`,
`xvla_fp8_quant_pdl.cu`) plus their pybind entries are compiled only when the
option is ON and GPU_ARCH is 120/121. The existing shared symbols
(`add_bias_bf16`, `residual_add`, `layer_norm`,
`quantize_bf16_to_nvfp4_swizzled_v2`, `fp8_per_token_block128_quant_bf16`,
the CUTLASS wrappers and `common.cuh`) are unchanged; other models keep their
current launch/build behavior. The X-VLA pipeline calls the distinctly named
`*_pdl`/`*_v3`/`*_v2` siblings. The FA2 head_dim-64 tile set is built when
`FLASHRT_BUILD_XVLA=ON`, `ENABLE_FA2`, BF16 and head dimension 64 are enabled;
it is additive to the upstream dispatch.

## Usage

```python
import flash_rt

model = flash_rt.load_model(
    "lerobot/xvla-libero",          # or a local checkpoint directory
    config="xvla",
    framework="torch",
    hardware="rtx_sm120",
    num_views=2,
    xvla_precision="nvfp4",
)
actions = model.predict(
    [agentview, wrist],             # two raw 256x256 HxWxC camera frames
    prompt="pick up the black bowl",
    state=state,                    # 8-D proprio
)
```

`predict` tokenizes the prompt with the BART tokenizer (50 tokens,
right-padded), embeds it once, and captures the CUDA Graph on the first call.
Subsequent calls replay the graph; changing the prompt copies a new static
embedding and does not recapture. `set_prompt(...)` may be called explicitly
to bind the instruction before the first `predict`.

The public `infer(obs)` surface accepts the repository observation dict
(`images` or `image`/`wrist_image`, plus `state`); `predict` is the VLA
convenience wrapper. Raw camera frames are expected at 256x256; other sizes
are resized to 256 in the preprocessing path and then follow the tower's
256->224 resize-with-padding.

## Precision

The action transformer and encoder use NVFP4 W4A4 by default. `fp8` selects
block-128 FP8; `bf16` selects the unquantized reference path. The DaViT tower
runs NVFP4 W4A4; the slower block-128 FP8 tower tier is not part of this path.

- Weight quantization is static, from the checkpoint; activation scales are
  produced by the fused norm/quantize kernels at runtime for the static
  shapes. There is no calibration cache to invalidate.
- The reference is the LeRobot `XVLAPolicy` fp32 path on the same
  observations, per-frame action cosine. The acceptance gate is the minimum
  over the corpus, not the mean.
- Correctness must be re-checked after a precision change; the three tiers
  are not interchangeable at the bit level.

Measured on a real LIBERO corpus (96 frames, two camera views,
8-D state, 50-token instruction, 24 episodes x 4 phases; RTX 5060 Ti 16 GB,
CUDA 13.0, torch 2.11; wall-clock equal-scope `raw observation -> action
chunk`, n=192):

| precision | E2E mean (median / min) | delivery similarity mean / min | max abs error |
|---|---:|---|---:|
| `nvfp4` | 21.457 ms (21.441 / 21.339) | 0.999294 / 0.990952 | 0.440431 |

The reference baseline (LeRobot fp32, same corpus/harness) is 281.409 ms
(13.12x). This branch measures 21.457 ms mean, 21.441 ms median and
21.339 ms min (n=192). With `XVLA_PDL=0` the same build is 22.137 ms mean
(median 22.115, min 22.050); PDL-on/off outputs are bit-identical. The
provenance receipt reports `complete=true`,
`graph_external_model_math_ops=0`, 2222 CUDA-graph launches and 6
vendor-library model-math launches. These numbers are a benchmark
measurement, not a quickstart claim; reproduce with the same corpus and the
wall-clock boundary before comparing.

### Small-M GEMM vs the repository skinny NVFP4 family

The SM120 small-M entry (`csrc/kernels/xvla_fp4_gemm_sm120.cu`) replaces the
repository's 128x128 CUTLASS tile at M=262 (denoise), M=100 (encoder) and
M=30 (last action block). The repository's `nvfp4_skinny` family is scoped to
M<=48 rows and emits FP32 K-split partials for a fused consumer. Per-shape
coverage:

| Site | M | N | K | small-M tile | skinny NVFP4 (M<=48) |
|---|---:|---:|---:|---|---|
| denoise qkv | 262 | 3072 | 1024 | `<96,256,48,64,3,64,0>` | N/A: M>48 |
| denoise proj/fc2 | 262 | 1024 | 1024/4096 | `<80,128,80,32,3,128,0>` | N/A: M>48 |
| denoise fc1 (GELU+NVFP4) | 262 | 4096 | 1024 | `<96,128,48,32,2,128,1>` | N/A: M>48 |
| encoder qkv/fc1 | 100 | 3072/4096 | 1024 | `<112,128,112,32,4,64,0>` | N/A: M>48 |
| encoder proj/fc2 | 100 | 1024 | 1024/4096 | `<32,128,32,32,4,128,0>` | N/A: M>48 |
| last block qkv/proj/fc2 | 30 | 1024/3072 | 1024/4096 | `<32,128,32,32,4,128,0>` | in envelope; partials+consumer |
| last block fc1 (GELU+NVFP4) | 30 | 4096 | 1024 | `<32,128,32,32,4,128,1>` | in envelope; `sum_bias_gelu` consumer |

Only the M=30 last block is inside the skinny envelope. Even there the small-M
entry is not interchangeable with the skinny GEMM: the small-M consumes the
same cutlass-swizzled e2m1/SF operands as the rest of the pipeline and writes
bf16 D with a fused bias (or GELU+NVFP4 for fc1), while the skinny entry
requires fragment-major weights, row-major activation scales, and FP32
partials plus a separate fused consumer. The dominant denoise/encoder shapes
(M=262 and M=100) are outside the skinny contract, so the small-M kernel is
the only entry that covers the production matrix. `XVLA_SMALLM_GEMM=0`
restores the CUTLASS 128x128 path; the same-commit 96-frame A/B is
24.925/24.928 ms (CUTLASS) -> 22.753/22.804 ms (small-M), with delivery
similarity and max abs error bit-identical.

The M=30 overlap was measured in a graph replay on an RTX 5060 Ti (us,
min/mean; the skinny column is GEMM-only and excludes the sum+bias consumer
it requires):

| site | small-M | skinny |
|---|---:|---:|
| qkv | 3.18 / 3.18 | 4.85 / 4.89 |
| proj | 2.78 / 2.78 | 2.44 / 2.50 |
| fc2 | 6.71 / 6.74 | 4.82 / 4.84 |

The skinny qkv is slower even before its consumer; proj/fc2 are 0.3-1.9 us
faster as a GEMM-only lower bound, but a second path would need the skinny's
partials/consumer and layout conversion for one small shape (the last action
block), while the small-M entry already covers M=262/100/30 with fused bias
and GELU. The small-M entry is retained.

## Tuning flags

The defaults are the validated path. The flags below are available for
experiments and for isolating a kernel; they are read when the frontend /
pipeline is constructed.

| Flag | Default | Effect |
|---|---|---|
| `XVLA_SMALLM_GEMM` | `1` | Use the small-M NVFP4 GEMM; `0` falls back to CUTLASS |
| `XVLA_FA2_M32` | `1` | Use the two-warp 32-row FA2 query tile; `0` falls back to 64-row |
| `XVLA_PDL` | `1` | Programmatic dependent launch between graph nodes |
| `XVLA_PRUNE_LAST` | `1` | Run the last action block on the 30 action rows only |
| `XVLA_VISION_GELU_FUSED` | `1` | Fuse the tower's bias+GELU+NVFP4 epilogue |
| `XVLA_DECODER_GEMV` | `1` | Use the native decoder GEMV |
| `XVLA_VISION_GELU_FUSED_STAGES` | all | Restrict the fused tower GELU to a stage subset |

## Model integration notes

- The checkpoint adapter is the inline `_load_weights` mapping in
  `flash_rt/frontends/torch/xvla_rtx.py`; the checkpoint keys are those of
  `lerobot/xvla-libero` (`model.safetensors` + `config.json`). There is no
  separate `_xvla_rtx_spec.py` because the mapping also performs the
  vision-tower layout conversion and the FP8/NVFP4 packing; the docs and the
  frontend docstring are the adapter contract.
- The model does not use the generic `AttentionSpec` factory: the attention
  is full (non-causal), head_dim 64, with static sequence lengths, and the
  RTX pipeline binds the vendored FA2 tile entries directly. This is the
  explicit non-applicability of the generic attention-spec path.
- Precision is static weight quantization plus runtime activation quantize;
  there is no calibration cache. This is the explicit non-applicability of
  the calibration-cache contract.

## Tests

```bash
python -m compileall flash_rt/models/xvla flash_rt/frontends/torch/xvla_rtx.py
pytest -q tests/test_xvla_import.py tests/test_xvla_config.py \
          tests/test_xvla_frontend_protocol.py
pytest -q tests/test_xvla_shape_guards.py       # skips without CUDA/SM120
```

A checkpoint-backed first-light test is `tests/test_xvla_precision.py`; it is
opt-in and skips unless `XVLA_CKPT` (and optionally `XVLA_REF_ACTIONS`) is set.

## Unsupported

- No JAX/FLAX frontend.
- No Thor (SM110), Orin (SM87), Ada (SM89), or AMD backend.
- The third configured camera slot is an empty camera and is not a caller
  input; `num_views` is 2 (or 3 when the caller includes the empty slot).


## Build and isolation follow-up (2026-10-27)

The frontend checks required core symbols and the FA2 tile64 profile before
reading checkpoint files or allocating model weights. A missing/default build
reports `FLASHRT_BUILD_XVLA=ON`, BF16 and the required head-dimension selection
instead of failing later with an attribute error.

`device` must name the current CUDA device (or be `cuda`, which resolves to it).
Architecture and SM-count queries use that device. Requests for a different device
are rejected before loading; changing the current device after construction also
raises before prompt setup, capture or inference. Select the desired CUDA device
before constructing the frontend.

PDL is no longer a process-wide setting enabled at initialization. X-VLA core
launches use a thread-local context during `pipeline.run()` and restore it on
normal return or exceptions. The new FA2 tile entries accept `enable_pdl` per
call; existing FA2 entries always initialize it to false. Other models do not
inherit X-VLA's launch policy.

The native raw library exports the complete adapter surface. FP16-only and
disabled-tile builds keep all five new tile symbols as throwing stubs;
`has_xvla_tile64()` distinguishes those stubs from an enabled profile.
