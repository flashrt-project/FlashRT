# LaWAM on RTX SM120

[LaWAM](https://huggingface.co/papers/2606.15768) (Latent World Action Model)
served from its LeRobot checkpoint
[`jialei02/lawam-pretrain-lerobot`](https://huggingface.co/jialei02/lawam-pretrain-lerobot)
on Blackwell consumer GPUs (`GPU_ARCH=120`).

## Build

```bash
cmake -B build -S . -DGPU_ARCH=120 -DFLASHRT_ENABLE_LAWAM=ON   # also builds FA2 FP16 causal (hdim 128)
cmake --build build -j$(( $(nproc) / 2 )) --target flash_rt_kernels flash_rt_fa2 flash_rt_lawam
```

`flash_rt_lawam` holds the kernels FlashRT had no equivalent for (camera
patchification for the two vision towers, the gated AdaLN rows of the LAM
decoder, the VLM-to-latent QFormer, FP16-accumulate GEMM plans, the skinny
weight-only INT8 DiT GEMM, and the positional-embedding consumers of the
skinny DiT). The tokenizer is
loaded with `transformers` (`AutoProcessor` of the checkpoint's `base_vlm`,
or `processor_path=`) at `set_prompt` time only.

## Usage

```python
import flash_rt
model = flash_rt.load_model("/path/to/lawam-pretrain-lerobot", config="lawam")
actions = model.predict([primary_rgb, wrist_rgb], prompt="put the bowl on the plate")
```

or directly:

```python
from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx
fe = LawamTorchFrontendRtx("/path/to/lawam-pretrain-lerobot")
fe.set_prompt("put the bowl on the plate")
fe.calibrate([(frames, task) for frames, task in real_observations])   # once
out = fe.infer(frames)["actions"]          # frames: uint8 [2, 256, 256, 3]
```

Inputs are two 256x256 RGB uint8 views (primary, wrist), as in the LeRobot
processor; the output is the unnormalized `action_horizon x 7` chunk
(defaults from the checkpoint config: 10 flow steps, chunk 50, horizon 24,
embodiment 25). `infer(noise=...)` takes the initial flow sample for
reproducible comparisons against the LeRobot policy.

## Pipeline

One CUDA graph per prompt holds the whole policy step:

| stage | kernels | precision |
|---|---|---|
| camera frames -> patch rows | `lawam.preprocess_u8` | FP16 |
| Qwen3-VL ViT (24 layers, DeepStack 5/11/17, mergers) | `lawam.gemm_f16acc`, `fp8_residual_norm_quant`, `fp8_qkv_split`, FA2 | FP16, FP16 accumulation |
| Qwen3-VL LLM (16 layers kept by LaWAM) | same, causal GQA FA2 | FP16, FP16 accumulation |
| VLM -> latent QFormer + LAM conditioning | `lawam.qformer_latent` | FP32 |
| DINOv3 ViT-B/16 (11 layers used), concurrent graph branch | `lawam.gemm_f16acc`, FA2 | FP16, FP16 accumulation |
| LAM decoder (12 AdaLN blocks) | `lawam.gemm_f16acc`, `lawam.gated_residual_norm`, `lawam.gemv_i8w` (modulation) | FP16, FP16 accumulation; INT8 weight-only GEMV |
| flow conditioning (enc_vlm, cross K/V of 8 blocks) | `lawam.gemm_f16acc`, BF16 GEMMs | FP16 / BF16 |
| flow DiT x 10 (16 blocks) | `lawam.w8a16_skinny_gemm`, `nvfp4_skinny_partials_attention`, `lawam.dit_*` consumers, FA2 small-query tile for the 512 image keys | INT8 weight-only, BF16 activations |
| Euler update, unnormalization | `lawam.dit_sum_bias_euler`, `lawam.unnormalize_actions` | FP32 |

Exact rewrites of the reference: tokens past the effective horizon never
influence the first 24 action rows (keys are masked, outputs discarded), so
the DiT runs 24 rows; the per-step timestep / AdaLN modulation and the
action positional embedding are precomputed; the cross-attention K/V are
computed once per call; DINOv3's last layer (unused hidden state) is
skipped and its LayerScale folded into the projections; proj_out_2 and the
first action-decoder layer are folded into one GEMM; the still frame's two
temporal patch copies are folded into the Qwen3-VL patch-embedding weight.

The LeRobot policy runs Qwen3-VL on `inputs_embeds` only, so Qwen3-VL uses
sequential (1-D) rotary positions for every token; the frontend matches it.

## Precision policy

`precision={"vit", "llm", "dino", "lam", "enc"}` selects `"f16acc"` (default),
`"fp16"` or `"fp8"` per stage; `precision["dit"]` selects `"w8a16"`
(default), `"bf16"` or `"nvfp4"`; `precision["dit_ffn"] = "nvfp4"` runs only
the DiT FFNs on NVFP4; `precision["lam_mod"]` is `"int8"` (default) or
`"fp16"` for the LAM AdaLN modulation GEMV; `precision["llm_attn"]` is `"fp16"`
(default, FP16 causal FA2, no cast before the o projection) or `"bf16"` (the
reference's attention rounding: aggregate 0.997 vs 0.992 on the 64 held-out
frames, 0.05 ms slower). `fp8_sites={"vit": [...],
"llm": [...]}` moves individual tower GEMMs to calibrated FP8.

LaWAM's flow lands in a different action mode on a few real frames under
small numeric changes: the reference policy itself moves 4 of 64 held-out
LIBERO frames when its VLM runs FP16 instead of BF16 (aggregate action
cosine 0.997). The default policy sits at that floor (0.997, same 64 frames);
per-tensor FP8 in any tower (0.87-0.99), even on single GEMM groups
(0.974-0.977), and NVFP4 DiT weights (all blocks, or only the FFNs: 0.995 on
one held-out set, 0.988 on another) fall measurably below it, so they are
opt-in tiers.

## Calibration

The default policy has no activation quantization (the DiT INT8 weights use
per-channel scales fixed at load). With an FP8 tier, `calibrate()` runs each
real observation once with dynamic per-tensor FP8 at every FP8 site, records
amax / 448 into the site's scale slot, and keeps the 99.9th percentile across
observations as the static scale (cached under
`~/.flash_rt/calibration/<ckpt>_lawam.json`, keyed by the precision policy).
