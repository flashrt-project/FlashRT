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
decoder, the VLM-to-latent QFormer, FP16-accumulate GEMMs with fused bias /
GELU / SwiGLU epilogues on the vendored CUTLASS, the skinny weight-only INT8
DiT GEMM, and the positional-embedding consumers of the skinny DiT). The tokenizer is
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
| Qwen3-VL ViT (24 layers, DeepStack 5/11/17, mergers) | `lawam.gemm_f16acc_act` (fc1 + bias + GELU in one GEMM), `fp8_residual_norm_quant`, `lawam.qkv_rope_split`, FA2 | FP16, FP16 accumulation |
| Qwen3-VL LLM (16 layers kept by LaWAM) | same, gate \| up + SwiGLU as one persistent `lawam.gemm_f16acc_persistent`, causal GQA FA2 | FP16, FP16 accumulation |
| VLM -> latent QFormer + LAM conditioning | `lawam.qformer_latent` | FP32 |
| DINOv3 ViT-B/16 (11 layers used), concurrent graph branch | `lawam.gemm_f16acc_act`, FA2 | FP16, FP16 accumulation |
| LAM decoder (12 AdaLN blocks) | `lawam.gemm_f16acc_act`, `lawam.gated_residual_norm`, `lawam.gemv_nvfp4w` (modulation) | FP16, FP16 accumulation; NVFP4 weight-only GEMV |
| flow conditioning (enc_vlm, cross K/V of 8 blocks) | `lawam.gemm_f16acc_act`, BF16 GEMMs | FP16 / BF16 |
| flow DiT x 10 (16 blocks), Euler updates, unnormalization | `lawam.dit_flow_persistent`: one cooperative kernel for the whole loop (below) | NVFP4 (FFN, self-attention QKV) / INT8 (other projections) weight-only, FP16 operands, FP16 accumulation per 256 / 128-wide k unit |

Exact rewrites of the reference: tokens past the effective horizon never
influence the first 24 action rows (keys are masked, outputs discarded), so
the DiT runs 24 rows; the per-step timestep / AdaLN modulation and the
action positional embedding are precomputed; the cross-attention K/V are
computed once per call; DINOv3's last layer (unused hidden state) is
skipped and its LayerScale folded into the projections; proj_out_2 and the
first action-decoder layer are folded into one GEMM; the still frame's two
temporal patch copies are folded into the Qwen3-VL patch-embedding weight.

The flow loop is latency bound (every GEMM has 24 rows), so
`lawam.dit_flow_persistent` runs it as one persistent cooperative kernel (one
CTA per SM) over an op table built by `pipeline_rtx.dit_persistent_ops`: grid
barriers separate dependent ops, the step's weights (packed at load time by
`pipeline_rtx.pack_dit_persistent` into the kernel's fragment order, per CTA in
consumption order) stream into a shared-memory ring that every MMA warp refills
behind itself, each GEMM's activation operand lands chunk by chunk (one
mbarrier per 256 columns), block-input LayerNorms run once per op distributed
over the CTAs, and the ffn down projection is split over 4 k groups whose
partials the next LayerNorm adds into the residual (written into the other of
two residual buffers: the phase reads whole old rows). `precision={"dit_impl": "chain"}`
selects the per-kernel chain (`pipeline_rtx.dit_step_forward`: skinny W4A16 /
W8A16 GEMMs with FP32 accumulation, `nvfp4_skinny_partials_attention`,
`lawam.dit_cross_attention`, the `lawam.dit_*` consumers).

Besides the DINOv3 branch, the graph moves work that is off the critical path
onto side branches: the GEMMs of the three DeepStack mergers (DRAM bound, read
first by the LLM) run next to the following ViT layers; the LAM input
projection and the flow image K / V of the DINOv3 tokens run on the DINOv3
branch; after the VLM, the AdaLN modulation GEMV of LAM blocks 1..11 (capped
grid, so the LAM GEMMs stay resident next to it; each block waits for its own
rows) and the flow's text conditioning run next to the LAM decoder. Every
split computes each output row the same way as the unsplit kernels.

The LeRobot policy runs Qwen3-VL on `inputs_embeds` only, so Qwen3-VL uses
sequential (1-D) rotary positions for every token; the frontend matches it.

The FP16-accumulate GEMMs run CUTLASS SM80 tensor-op tiles with the bias /
GELU / SwiGLU epilogues fused. The tile and serial split-K count of every
production shape come from `pipeline_rtx.F16ACC_PLANS` (measured on RTX 5060
Ti), so every process forms the same FP16 sums; other shapes are timed on their
first call. Tiles 8-10 are stream-K (the K iterations of the output tiles spread
over the 36 SMs, partial tiles reduced in fixed order); the LLM down projection
runs on them. Tiles 16.. select `lawam.gemm_f16acc_persistent`
(`lawam_gemm_persistent_sm120.cu`): one CTA per SM walks a static schedule of
data-parallel tiles and stream-K segments with one continuous cp.async pipeline,
split tiles are summed in k order by the last segment to arrive (no CTA waits on
another, so concurrent graph branches cannot deadlock), and the epilogue writes
16-byte rows through the shared-memory stage the last k-tile consumed. The LLM
gate | up GEMM runs it with a 224-row tile (all Se rows in one tile row) and a
SwiGLU epilogue; its weight is stored with gate and up interleaved in blocks of 8
columns (`pipeline_rtx.interleave_gate_up`).
`flash_rt.models.lawam.pipeline_rtx.F16ACC_SPLIT = "timed"` times
every shape, `"cublas"` (set before the graph is captured) uses the split of the cuBLASLt
`COMPUTE_16F` plan instead, so the GEMMs form the same FP16 sums as
`lawam.gemm_f16acc` (the erf-GELU and SwiGLU epilogues then reproduce the
separate passes bit for bit, the ViT's tanh-GELU epilogue differs from
cuBLASLt's by at most one FP16 ulp), and `F16ACC_KERNEL = "cublas"` the cuBLASLt
plans with stand-alone epilogue kernels.

## Precision policy

`precision={"vit", "llm", "dino", "lam", "enc"}` selects `"f16acc"` (default),
`"fp16"` or `"fp8"` per stage; `precision["dit"]` selects `"w8a16"`
(default), `"w4a16"` (NVFP4 weight-only), `"bf16"` or `"nvfp4"` (W4A4);
`precision["dit_ffn"]` overrides the DiT FFNs: `"w4a16"` (default, NVFP4
weight-only with BF16 activations) or `"nvfp4"` (W4A4), `precision["dit_qkv"]`
the self-attention QKV projections (`"w4a16"` default); `precision["lam_mod"]` is `"nvfp4"` (default,
weight-only E2M1 with per-16 UE4M3 block scales), `"int8"` (per-channel) or
`"fp16"` for the LAM AdaLN modulation GEMV; `precision["llm_attn"]` is `"fp16"`
(default, FP16 causal FA2, no cast before the o projection) or `"bf16"` (the
reference's attention rounding: aggregate 0.997 vs 0.992 on the 64 held-out
frames, 0.05 ms slower); `precision["dit_impl"]` is `"persistent"` (default,
the W8A16 / W4A16 tier only) or `"chain"`. `fp8_sites={"vit": [...],
"llm": [...]}` moves individual tower GEMMs to calibrated FP8.

LaWAM's flow lands in a different action mode on a few real frames under
small numeric changes: the reference policy itself moves 4 of 64 held-out
LIBERO frames when its VLM runs FP16 instead of BF16 (aggregate action
cosine 0.997). The default policy sits at that floor (0.997, same 64 frames);
per-tensor FP8 in any tower (0.87-0.99), even on single GEMM groups
(0.974-0.977), NVFP4 activations in the DiT FFNs (W4A4: 0.995 on one held-out
set, 0.988 on another) and NVFP4 weights on every DiT projection (0.990) or on
the output / cross-attention q projections (0.991-0.994) fall measurably below
it, so they are opt-in tiers. NVFP4 weights with BF16 activations on the DiT
FFNs and self-attention QKV (0.996 / 0.997) and on the LAM modulation stay at
the floor and are the default.

## Calibration

The default policy has no activation quantization (the DiT INT8 weights use
per-channel scales, the DiT FFN and LAM modulation NVFP4 weights per-16 block
scales, all fixed at load). With an FP8 tier, `calibrate()` runs each
real observation once with dynamic per-tensor FP8 at every FP8 site, records
amax / 448 into the site's scale slot, and keeps the 99.9th percentile across
observations as the static scale (cached under
`~/.flash_rt/calibration/<ckpt>_lawam.json`, keyed by the precision policy).
