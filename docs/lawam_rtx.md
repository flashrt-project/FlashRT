# LaWAM on RTX SM120

[LaWAM](https://huggingface.co/papers/2606.15768) (Latent World Action Model)
served from its LeRobot checkpoint
[`jialei02/lawam-pretrain-lerobot`](https://huggingface.co/jialei02/lawam-pretrain-lerobot)
on Blackwell consumer GPUs (`GPU_ARCH=120`).

## Support matrix

| | |
|---|---|
| Hardware | RTX SM120 (`GPU_ARCH=120`); measured on an RTX 5060 Ti 16 GB (36 SMs) |
| Framework | `torch` only. `jax`, `thor`, `rtx_sm89`, `rtx_sm87` and `rtx_sm110` fail at `load_model` before any extension is loaded |
| Checkpoint | LeRobot `jialei02/lawam-pretrain-lerobot` (`config.json`, `model.safetensors`, `policy_preprocessor.json`) |
| Build flag | `-DFLASHRT_ENABLE_LAWAM=ON` (default OFF) builds `flash_rt_lawam`; `flash_rt_kernels` and `flash_rt_fa2` are required too |
| Input | batch 1, two 256x256 RGB uint8 frames (primary, wrist) + a task string |
| Flow | 10 denoising steps, chunk 50, action horizon 24, action dim 7, embodiment 25 |
| GPU memory | 4.19 GB peak in the measured configuration (the checkpoint is 6.7 GB on disk) |
| Default precision | FP16-accumulate towers, W8A16 DiT, NVFP4 weight-only DiT FFN/QKV and LAM modulation; static FP8 sites are opt-in |
| Python deps | `torch` and `safetensors` for the weights, `transformers` (Qwen3-VL processor) and `Pillow` at `set_prompt` time |

The whole policy step — camera preprocessing, Qwen3-VL, QFormer, DINOv3, the
LAM decoder, flow conditioning, all 10 DiT steps and unnormalization — is
replayed from one CUDA graph per prompt.

## Build

CUTLASS v4.4.2 must be present at `third_party/cutlass`; the root CMake fails
fast otherwise (see [INSTALL.md](INSTALL.md)). From the repository root:

```bash
cmake -B build -S . -G Ninja -DGPU_ARCH=120 -DFLASHRT_ENABLE_LAWAM=ON
cmake --build build -j$(( $(nproc) / 2 )) --target flash_rt_kernels flash_rt_fa2 flash_rt_lawam
pip install -e ".[lawam]"
```

`-DFLASHRT_ENABLE_LAWAM=ON` also builds the FA2 FP16 causal head-dim-128 entry
(the same kernel Chameleon uses); this needs `ENABLE_FA2`, `128` in
`FA2_HDIMS` and `fp16` in `FA2_DTYPES`, which are the SM120 defaults and are
checked at configure time. `pip install -e ".[lawam]"` adds the `transformers`
and `Pillow` dependencies the Qwen3-VL processor needs.

Every entry is exported from the dedicated `flash_rt_lawam` module (C++
namespace `flash_rt::lawam`, sources `csrc/kernels/lawam/lawam_*`); none of
these names is added to the shared `flash_rt_kernels` surface.

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

### Constructor options

`LawamTorchFrontendRtx(checkpoint, *, processor_path=None, device="cuda:0",
num_views=2, dit_cfg=None, calibration_dir=None, precision=None, fp8_sites=None)`

| argument | meaning |
|---|---|
| `processor_path` | processor used for tokenization; defaults to the checkpoint's `base_vlm` |
| `device` | CUDA device the graph is captured on |
| `num_views` | must be 2 (primary + wrist) |
| `dit_cfg` | skinny DiT tile config per `(N, K)`; defaults to the measured `DIT_SKINNY_CFG` |
| `calibration_dir` | calibration cache directory (default `~/.flash_rt/calibration`) |
| `precision` | per-stage tier overrides (see Precision policy) |
| `fp8_sites` | per-layer FP8 site selection, e.g. `{"llm": ["gateup@8-15"]}` |

Methods: `set_prompt(task, capture=True)`, `calibrate(samples, percentile=99.9)`
/ `calibrate_with_real_data(obs_list)`, and `infer(obs, noise=None)`, which
returns `{"actions": np.ndarray}`.

## Pipeline

One CUDA graph per prompt holds the whole policy step:

| stage | kernels | precision |
|---|---|---|
| camera frames -> patch rows | `lawam.preprocess_u8` | FP16 |
| Qwen3-VL ViT (24 layers, DeepStack 5/11/17, mergers) | `lawam.gemm_f16acc_act` (fc1 + bias + GELU in one GEMM), `fp8_residual_norm_quant`, `lawam.qk_rope_inplace` (default) / `lawam.qkv_rope_split` (fp8 / fp16 tiers), FA2 | FP16, FP16 accumulation |
| Qwen3-VL LLM (16 layers kept by LaWAM) | `lawam.gemm_f16acc_act`, `fvk.fp8_qkv_split`, gate \| up + SwiGLU as one persistent `lawam.gemm_f16acc_persistent`, causal GQA FA2 | FP16, FP16 accumulation |
| VLM -> latent QFormer + LAM conditioning | `lawam.qformer_latent` | FP32 |
| DINOv3 ViT-B/16 (11 layers used), concurrent graph branch | `lawam.gemm_f16acc_act`, `lawam.qkv_rope_split`, FA2 | FP16, FP16 accumulation |
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
CTA per selected SM, 32–64 CTAs in multiples of four) over an op table built by `pipeline_rtx.dit_persistent_ops`: grid
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

The persistent grid uses `min(SM count, 64)` rounded down to a multiple of four.
The 36-SM configuration is unchanged; a 170-SM GPU uses 64 CTAs. Packing and
the native launch reject unsupported counts, and the launch checks cooperative
residency. Attention requires 32 CTAs to cover all 16 heads and both query tiles;
the LayerNorm owner reduction supports at most 64 owners.
LaWAM uses dedicated `lawam_fwd_*` FA2 entries for short-tile tuning; the
existing FA2 entries retain their original dispatch even in a LaWAM build.

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
`COMPUTE_16F` plan instead. Matching the split does not guarantee identical
FP16 sums across CUTLASS and cuBLASLt algorithms or GPU configurations. The
GEMM regression compares both results with an FP32 reference (relative L2
error below 0.003 and maximum absolute error below 0.04 for its normalized
random inputs). `F16ACC_KERNEL = "cublas"` uses cuBLASLt plans with stand-alone
epilogue kernels.

## Performance

Measured on one RTX 5060 Ti 16 GB (SM120, 36 SMs), driver 580.178.04, CUDA
13.0 (nvcc 13.0.88), PyTorch 2.14.1+cu130, CUTLASS v4.4.2, checkpoint
`jialei02/lawam-pretrain-lerobot`, default precision, 2 views, 10 flow steps,
chunk 50, action horizon 24, action dim 7.

| metric | stock LeRobot | FlashRT |
|---|---:|---:|
| E2E, host uint8 frames + task -> host `(24, 7)` actions, median | 152.81 ms | **18.85 ms** |
| E2E p90 | 153.83 ms | 18.89 ms |
| model-only (no pre/post-processing) | 146.72 ms | 18.84 ms |
| CUDA-graph replay, device-event time | - | 18.84 ms |
| peak GPU memory | 7.28 GB | 4.19 GB |
| speedup | 1.0x | **8.11x** |

E2E is wall-clock `time.perf_counter()` around one policy step, including the
host copies in and out. After 10 warmup calls a step is one `cudaGraphLaunch`
and one `cudaStreamSynchronize` (the flow noise is drawn on device inside the
graph); the replay row is the device-event time of that launch. The FlashRT
numbers are medians of 1000 iterations; the stock policy (LeRobot
`LaWAMPolicy`, TF32 off, same checkpoint) is the median of 60 iterations with
the same real frame, task and flow noise. The baseline includes the LeRobot
pre-/post-processors, so both sides cover the same scope.

Stage split of one FlashRT step (nsys; 532 kernels, 18.77 ms kernel time):
Qwen3-VL ViT 6.00 ms, LLM 5.39 ms, LAM decoder 1.49 ms, flow DiT 5.77 ms over
the 10 steps, QFormer + DINOv3 + flow conditioning 0.12 ms, GPU idle 0.14 ms.

Correctness on 64 held-out real LIBERO frames (aggregate cosine of the
normalized action chunks; gate 0.99): **0.9957**, with 4 samples changing
action mode (`eval32`: 0.9974, 1 flip). CUDA-graph replay and eager execution
agree exactly (max |delta| 0.0). These numbers are from one machine; a
different SM120 part changes the latency, not the selected precision tier.

Reproduce (needs the checkpoint):

```python
import statistics, time
import numpy as np
from flash_rt.frontends.torch.lawam_rtx import LawamTorchFrontendRtx

fe = LawamTorchFrontendRtx("<lawam-pretrain-lerobot>")
fe.set_prompt("put the bowl on the plate")
frames = np.zeros((2, 256, 256, 3), np.uint8)          # primary, wrist
for _ in range(10):
    fe.infer(frames)                                   # capture + warmup
ts = []
for _ in range(1000):
    t0 = time.perf_counter()
    fe.infer(frames)
    ts.append((time.perf_counter() - t0) * 1e3)
print(f"E2E median {statistics.median(ts):.2f} ms")
```

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
reference's attention rounding; slightly closer to the LeRobot outputs on real
frames and ~0.05 ms slower on the RTX 5060 Ti); `precision["dit_impl"]` is
`"persistent"` (default, the W8A16 / W4A16 tier only) or `"chain"`.
`fp8_sites={"vit": [...], "llm": [...]}` moves individual tower GEMMs to
calibrated FP8.

LaWAM's flow lands in a different action mode on a few real frames under
small numeric changes: the reference policy itself moves 4 of 64 held-out
LIBERO frames when its VLM runs FP16 instead of BF16 (the two LeRobot runs
have aggregate action cosine 0.997). Against the BF16 LeRobot reference the
default FlashRT policy scores 0.9957 on those 64 frames with the same 4 mode
flips, i.e. it sits at that precision floor; per-tensor FP8 in any tower (0.87-0.99), even on single GEMM groups
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
observations as the static scale. The result is cached under
`~/.flash_rt/calibration/<sha256>_lawam_v2.json`, keyed by the complete weight
file SHA-256, `config.json` / `policy_preprocessor.json` contents, the serialized
loaded processor (including vocabulary, special tokens, image settings and chat
template), the precision policy, `fp8_sites` and the device capability. Version 2
does not reuse the earlier partial-file cache. The weight digest is computed
once when FP8 weights are loaded, outside inference and graph capture (pass
`calibration_dir=` to move the cache).

## Limitations

- SM120 + `torch` only; the other `(framework, hardware)` tuples raise at
  `load_model` before the extension is loaded.
- One CUDA graph per prompt. `set_prompt()` (including `predict()` with a
  changed prompt) re-captures it, so do not switch instructions every step.
- Batch 1 and exactly two 256x256 RGB views; the LeRobot processor geometry is
  fixed.
- FP8 tiers need one `calibrate()` pass over real observations; the default
  tier needs none.
- The correctness check above is on held-out LIBERO frames. Validate the
  target embodiment and camera setup before deployment.
