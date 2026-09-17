# pi0.5 on TensorRT Edge-LLM

`pi05/pi05_policy_inference` runs a FlashRT pi0.5 engine inside a TensorRT
Edge-LLM build, using Edge-LLM's image loading, `tokenizer.json` tokenizer and
CUDA graph capture. It is an overlay: nothing in Edge-LLM is modified except
one `add_subdirectory` line.

## Build

```bash
backends/tensorrt/integrations/edgellm/install_overlay.sh <TensorRT-Edge-LLM>   # needs BUILD_EXPERIMENTAL_MODELS=ON
cmake -S <TensorRT-Edge-LLM> -B <TensorRT-Edge-LLM>/build
cmake --build <TensorRT-Edge-LLM>/build --target pi05_policy_inference -j 2
```

## Inputs

- Engine and plugin: `backends/tensorrt/tools/build_pi05_engine.sh <checkpoint> <views> <fixture> <out>`
  and `build/tensorrt/libflashrt_trt_pi05.so` ([usage](../../../../docs/tensorrt_usage.md)).
- Tokenizer: `backends/tensorrt/tools/export_paligemma_tokenizer.py paligemma_tokenizer.model <dir>`
  writes `tokenizer.json` (checked against SentencePiece).
- Images: one file per unmasked camera, in openpi order (for LIBERO: base,
  wrist).

## Run

```bash
pi05_policy_inference --engine pi05.engine --plugin libflashrt_trt_pi05.so \
    --tokenizer <tokenizer dir> --images base.png,wrist.png \
    --prompt "put the bowl on the plate" \
    --norm_stats <checkpoint>/assets/physical-intelligence/libero/norm_stats.json \
    --output actions.json --iters 200
```

The output has the prompt tokens, `raw_actions` [10, 32] and, with
`--norm_stats`, `actions` [10, action_dim] after openpi's quantile
unnormalization. `--noise` takes a safetensors file with an fp16 `noise_in`
tensor; otherwise noise is drawn from `--seed`. `--pixel_norm flashrt`
reproduces FlashRT's uint8 table for bitwise checks (openpi's `x / 255 * 2 - 1`
is the default).

## Checked on Jetson Thor (pi05_libero, 2 cameras)

- Four LIBERO prompts: tokens equal openpi's; raw actions equal the FlashRT
  library bit for bit (`--pixel_norm flashrt`, PyTorch-wheel cuBLAS).
- `actions` equal openpi's output transform to 1.2e-7.
- CUDA graph latency 23.9 ms (25.7 ms for prompt lengths whose decoder K/V
  length is not a multiple of 8).
