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

## The same model through the structures layer

The runtime above drives the official pipeline. The transformer is also
reachable as an ordinary Diffusers host, where the structures layer attaches
to it without a model-specific path:

```python
from flash_rt import structures

plan = structures.attach(model, forward, scheme="nvfp4_balance")
print(plan.report())          # bound seams, gate results, ledger
plan.detach()                 # restores the host exactly
```

`attach` discovers the seams, calibrates on one real forward, gates accuracy
and latency per family, and keeps the host path wherever a gate declines.
Nothing here is LTX-specific: the attention seam is recognised by the
processor contract (separate query/key rotary boundaries, per-head gating),
not by a model or class name.

`scheme=` selects the precision profile, and two are relevant here:

| scheme | what it quantizes |
|---|---|
| `"nvfp4_balance"` | the projection GEMMs (W4A4); attention keeps the family's precision-first order |
| `"nvfp4_balance_sage"` | the same, and allows the quantized attention forms to be weighed first |

The split is deliberate. A quantized attention form trades a bounded error
for speed, so it is a precision decision like any other and arrives through
the profile a deployment selected — never through a device check or a host
binding. Naming a form does not force it: the family still qualifies the
shape, speed-gates the result, and falls through to the published order when
the installed package does not serve the site.

To tune one seam without registering a profile, name the forms directly:

```python
plan = structures.attach(model, forward, scheme="nvfp4_balance",
                         attention_forms=("sage2",))
```

Either way the answer comes back measured. `plan.report()` prints each
family's accuracy band and the paired latency it was judged on, and the
attachment can be reverted exactly, so the way to decide between these is to
run both and read the two reports rather than to take this table's word for
it on a different card.

### End to end

What a request costs, wall clock, same prompt and seed, distilled
single-pass recipe. The baseline is the unmodified host: a 44GB bf16
checkpoint that does not fit on a 32GB part, so it runs with weight
offloading, which is what a user of this model on this class of card
actually starts from.

| Request | Host (offload) | `"nvfp4_balance"` | `"nvfp4_balance_sage"` |
|---|---|---|---|
| 768×512×49f | 99.8 s | 6.0 s (16.6×), peak 29.9 GB | **5.7 s (17.5×)**, peak 26.8 GB |
| 1536×1024×121f | 181.6 s | **87.9 s (2.07×)**, peak 28.0 GB | does not fit, see below |

Medians of three warm runs, eager. Frames are inspection-equivalent to the
host's own output.

The full-size row is the honest one to read closely. Attaching all 48 blocks
succeeds and each block's gate measures 1.48×, but the assembled pipeline
sits close to the limit of a 32GB part: 23.9 GB resident before the request
begins, 28.0 GB at peak, and video-VAE tiling is needed for decode to have
room at all. Three things account for the distance between 2.07× here and
what the same hardware reaches with a hand-assembled configuration:

- this measurement is eager, with no compilation of the block stack;
- the quantized attention profile does not fit at this size yet — the
  per-shape staging pool costs about 3 GB on top, which the assembly step
  runs out of;
- the audio feed-forward stays at host precision throughout, 3.0 GB across
  the model, because its 126-row calls sit outside the fused chain's
  128-row alignment and the seam declines them rather than produce an
  unwritten output.

### Where the time goes, one transformer block

The table below is a diagnostic, not the result: it says which family earned
which part of the request time above. Real checkpoint weights, real captured
deployment inputs, paired alternating timing inside the gate.

| Site shape | `scheme=` | Block latency | Attention family | Peak memory |
|---|---|---|---|---|
| S=24576 (1536×1024×121f) | host, unattached | 134.3 ms | — | 12.2 GB |
| | `"nvfp4_balance"` | 117.1 ms (1.15×) | BF16 form bound, declined at 1.006× | 8.2 GB |
| | `"nvfp4_balance_sage"` | **89.8 ms (1.49×)** | activated, 1.259× | at the 32GB ceiling |
| S=2688 (768×512×49f) | host, unattached | 10.2 ms | — | 2.3 GB |
| | `"nvfp4_balance"` | 8.2 ms (1.25×) | declined | 1.7 GB |
| | `"nvfp4_balance_sage"` | 8.0 ms (1.28×) | activated, 1.022× | 4.7 GB |

Matched-forward cosine against the host's own output is 0.99999 in every row,
and `detach` restores it bit-exactly (max-abs 0.0). Three things in that table
are easy to misread, so they are worth stating:

- **A block ratio is not a kernel ratio.** The same attention that measures
  2.34× on its own (45.9 → 19.6 ms at S=24576) shows up as 1.259× for the
  attention unit, because the unit is judged against the whole block. The
  27 ms it saves is the same 27 ms in both numbers.
- **The projections-only profile leaves the BF16 attention form bound and
  declined.** That form measures 46.1 ms against the host's 45.9 ms here, so
  the gate is right to keep the host path; nothing about the quantized forms
  is being judged in that row.
- **Peak memory moves in both directions.** Quantizing the projections takes
  it from 12.2 to 8.2 GB. Preferring quantized attention gives some back,
  because each attention site owns its staging and quantization workspace:
  four sites at S=24576 reach the ceiling of a 32GB part. Pooling those
  workspaces is the open item before that profile is usable at full size.

### How the whole-model figures were produced

Blocks are attached one at a time, because a 44GB bf16 checkpoint is not
resident on a 32GB part: each block is materialized alone, attached on its
own real inputs, and its host weights released before the next. The
feed-forward seams are bound explicitly rather than by discovery, because
`vision_ffn` does not claim this host's shape — its projections carry no
bias and its norm sits outside the seam, both of which the structure's
boundary requires. Whether to widen that boundary is a catalog decision.

### Kernel availability is the package's own statement

The forms read their envelope from the installed artifact. The sage3 package
publishes head_dim 128 only in its CUDA 13 builds; on a CUDA 12.8 host it
advertises head_dim 64, so a 128-wide site is refused there and the ladder
falls through — visible on the refusal trail rather than as a silent
slowdown. Nothing in this repository keeps a second table of that.
