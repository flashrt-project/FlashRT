# LTX-2.5 (22B distilled, audio+video) on FlashRT — RTX SM120

FlashRT integration of the official [LTX-2](https://github.com/Lightricks/LTX-2)
`ltx-pipelines` two-stage distilled pipeline, with FlashRT compute swaps.

Three compute swaps ship here: the attention backend, the W4A4 NVFP4 FFN
chain, and — behind `compile_mode="capture"` — a resident transformer that
makes whole-loop CUDA-graph capture possible on one GPU. Measured warm
denoise at 1536×1024×121f: 23.9s upstream eager, 11.7s with all three.

## Requirements

- RTX 5090 (SM120), 32GB. Peak allocated ≈ 23.2GB at 1536×1024×121f.
- An environment with the official LTX-2 packages (`ltx-core`,
  `ltx-pipelines`, and `ltx-kernels` for the NVFP4 path), either installed or
  reachable through `FLASH_RT_LTX2_ROOT` (path to an LTX-2 monorepo checkout).
- The LTX-2.5 split checkpoint pack (one safetensors per component):

```
<pack>/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
<pack>/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
<pack>/vae/ltx-2.5-video-vae-bf16.safetensors
<pack>/vae/ltx-2.5-audio-vae-bf16.safetensors
<pack>/model_patches/ltx-2.5-duration-head-bf16.safetensors        (optional)
<pack>/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

The NVFP4 transformer ships prequantized block-16 weights with calibrated
static activation scales, so there is no calibration pass.

## Quickstart

```python
import flash_rt

pipe = flash_rt.load_model(
    checkpoint="/path/to/LTX-2.5",   # the split pack root
    config="ltx25",
    attention="sage2-fvk",           # optional; "auto" by default
    fuse=True,                       # the W4A4 FFN chain, on by default
    compile_mode="capture",          # eager by default; see below
)
pipe.set_prompt("A golden retriever running through a sunny meadow")
stats = pipe.infer(seed=42, output_path="out.mp4")
print(stats)
```

`attention`, `fuse` and `compile_mode` choose the execution assembly and are
forwarded to this frontend only; omitting one leaves the frontend's own
default. The same three are constructor arguments if you build
`Ltx25TorchFrontendRtx` directly.

`infer` accepts `height`, `width` (multiples of 32), `num_frames`
(`k*8+1`), `frame_rate`, `seed`, and `output_path` (omit to skip mp4 encode).

## Attention backends

Selected by the `attention=` constructor argument or `FLASH_RT_LTX25_ATTN`:

| value | path | notes |
|---|---|---|
| `auto` (default) | `sage2-fvk` when available | |
| `sage2-fvk` | FlashRT raw-pointer sage2 qk-int8/pv-fp8 kernels | graph-capture safe; video d128 sites |
| `sage2` | upstream `sageattention` package | needs the package built for SM120 |
| `sage3` | `sageattn3` FP4 Blackwell package | fastest, lower per-call accuracy; opt-in |
| `sdpa` | torch SDPA | baseline |

Audio-branch attention (head_dim 64, ~1k tokens) always runs SDPA: measured on
5090, the quantized paths lose to SDPA at that shape.

## FFN chain and compilation

`fuse=True` (default) replaces each transformer block's feed-forward with the
W4A4 NVFP4 chain: three launches (quantize, fused GEMM+bias+GELU emitting
FP4, GEMM) where upstream runs six. The chain accepts only 128-aligned row
counts — CUTLASS declines the rest and returns without writing an output, so
unaligned calls (the ~126-token audio branch) stay on the upstream module
rather than reading an unwritten buffer.

`compile_mode` selects the execution assembly:

| value | assembly |
|---|---|
| `None` (default) | eager |
| `"default"` | per-block `torch.compile`, sequence-length specialized |
| `"capture"` | per-block compile plus whole-loop CUDA-graph capture |

Capture requires the transformer to stay resident, which the swap builder
arranges; the memory contract that follows is below.

## Memory and lifecycle (capture mode)

The resident transformer holds ≈14GB. The text encoder loads ≈26GB for the
length of one prompt encode, and the two do not fit together on a 32GB part,
so residency is a lease rather than a permanent state:

- a prompt whose embeddings are already cached keeps the resident model and
  skips the encoder entirely;
- a prompt that is not cached ends the lease first, encodes, and lets the
  next stage call take a fresh lease. The cost is one transformer rebuild;
  nothing fails and nothing has to be released by hand.

Two explicit entry points, both idempotent and both on the object
`load_model` returns:

```python
pipe.release_resident()   # drop the resident transformer and its graphs
pipe.close()              # the above, plus prompt cache and pipeline
```

`release_resident` returns the device bytes it freed (0 outside capture mode,
which holds no lease; also 0 on models that keep nothing resident, so a
serving loop can call it without knowing which frontend it has). After either
call the frontend still works: the next `infer` rebuilds what it needs,
`close` reloading from the checkpoint.
VAE decode tiling is sized against the memory that remains once the resident
transformer is accounted for, so decode does not have to be given a manual
budget.

Measured on 5090 at 1536×1024×121f (median, video self-attention site,
S=24576): SDPA-cudnn 42.4ms, sage2 17.4ms, sage3 13.0ms. End-to-end stage-2
denoise per step: 5.11s (SDPA) → 3.84s (sage2), with output quality equivalent
under matched-input single-forward cosine and frame inspection.
