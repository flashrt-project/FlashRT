# GR00T N1.5 on RTX 4090 (SM89)

`config="groot_n15"`, `framework="torch"`, `hardware="rtx_sm89"` serves
**GR00T N1.5** (`GR00T_N1_5`: Eagle2.5 = SigLIP vision tower + 12-layer
Qwen3, VL self-attention, 16-layer flow-matching DiT), e.g. the RoboCasa
checkpoint `huangmz21/gr00t-n1.5-robocasa-atomic-seen-180k`, as FlashRT
kernels captured in one CUDA graph per prompt length.

| Tuple | Precision | Status |
|---|---|---|
| `groot_n15` / torch / `rtx_sm89` | FP8 (`use_fp8=True`, default) | validated on RTX 4090 |
| `groot_n15` / torch / `rtx_sm89` | BF16 (`use_fp8=False`) | validated on RTX 4090 |

Batch 1, one image tile per camera view (the Eagle processor's tiling at the
data config's resolution), the `new_embodiment` tag or any embodiment present
in the checkpoint metadata. Other hardware, JAX, batched requests and
multi-tile images are not supported; unsupported checkpoint configurations
(pixel shuffle, a projected `eagle_linear`, a non-interleaved DiT, RoPE
scaling, ...) are refused when the frontend is constructed.

## Build

The model-specific kernels are a separate module, off by default:

```bash
cmake -B build -S . -DGPU_ARCH=89 -DFLASHRT_ENABLE_GROOT_N15=ON
cmake --build build --target flash_rt_kernels flash_rt_fa2 flash_rt_groot_n15
```

`FLASHRT_ENABLE_GROOT_N15` requires `GPU_ARCH=89` and adds only the
`flash_rt_groot_n15` module (`csrc/groot_n15_bindings.cpp`,
`csrc/kernels/groot_n15/`); the default `flash_rt_kernels` target is
unchanged by it. `flash_rt_fa2` must carry the bf16 instantiations at
head_dim 64, 96 and 128 (the default `FA2_HDIMS` / `FA2_DTYPES` do; a build
narrowed to `FA2_HDIMS="96;256"` aborts in the FA2 wrapper when the frontend
probes its attention sites).

The observation and action transforms are the upstream Isaac-GR00T ones, so
the `gr00t` package must be importable: pass `groot_root=` (its source root)
or set `$GROOT_N15_ROOT`, and name the data config (`data_config=` or
`$GROOT_N15_DATA_CONFIG`, a `DATA_CONFIG_MAP` key). RoboCasa's `panda_omron`
is in the `robocasa-benchmark/Isaac-GR00T` fork. The upstream model is never
built, so `flash_attn` is not needed.

## Usage

```python
import flash_rt
# load_model forwards neither data_config nor groot_root; set the variables.
#   export GROOT_N15_ROOT=/path/to/Isaac-GR00T
#   export GROOT_N15_DATA_CONFIG=panda_omron
model = flash_rt.load_model(
    "/path/to/gr00t-n1.5-checkpoint", config="groot_n15",
    hardware="rtx_sm89", num_views=3,          # must equal the data config's views
    use_fp8=True)                              # False: the BF16 tier
actions = model.predict(
    images=[left, right, wrist],               # data-config video-key order, uint8 HxWx3
    prompt="open the microwave",
    state={"base_position": ..., ...})          # dict keyed by state name
# actions: {action_key: np.ndarray[horizon, dim]}, unnormalized
```

Direct construction takes the same arguments plus `data_config`,
`groot_root`, `num_steps` (flow-matching steps, default the checkpoint's),
`max_prompt_tokens` (default 1024), `cuda_graphs` and `graph_cache_size`
(default 8). `action_horizon`, when given, must equal the data config's,
which must equal the checkpoint's.

```python
from flash_rt.frontends.torch.groot_n15_rtx_sm89 import GrootN15TorchFrontendRtxSm89
fe = GrootN15TorchFrontendRtxSm89(ckpt, num_views=3, data_config="panda_omron",
                                  groot_root="/path/to/Isaac-GR00T", use_fp8=False)
fe.set_prompt("open the microwave")
out = fe.infer({"images": [left, right, wrist], "state": state})["actions"]
# normalized inputs + explicit initial noise, e.g. for A/B against the reference:
actions = fe.predict_normalized(normalized_input, noise)   # [1, 16, 32] fp32
```

The initial noise is drawn with `torch.randn` on the device unless passed.

## Computation path

```
observation --(upstream data-config transform: crop/resize, Eagle chat template,
              tokenizer, image processor, state normalization)--> normalized input
  vision   im2col (patch_im2col_nchw_bf16) -> patch projection -> + position
           27 x [LayerNorm, q/k/v, FA2 (head_dim 72), o, LayerNorm, fc1, GELU, fc2]
           post-LayerNorm -> mlp1 -> image rows of the token table
  language token lookup (image positions index those rows), 12 x Qwen3 decoder layer
           (HF-rounding RMSNorm, q/k/v, q/k norm + RoPE, FA2 causal GQA, o,
           gate/up, SiLU gate, down), final RMSNorm
  vl       vlln (fp32) + 4 x [LayerNorm, q/k/v, FA2 (head_dim 64), o, LayerNorm,
           ff, GELU, ff] on an fp32 residual stream
  cross    K/V of the 8 cross-attention DiT layers, projected once per call
  state    embodiment MLP (bmm, ReLU, bmm)
  denoise  4 x [action encoder (bmm, swish, bmm, + position),
                16 x DiT layer (AdaLayerNorm, q (+k/v), FA2 head_dim 48 cross or self,
                o, LayerNorm, ff, GELU, ff), output AdaLayerNorm, proj_out,
                action decoder, Euler update]
  --(upstream transform: unnormalize)--> per-key action dict
```

* Stage sequence: `flash_rt/models/groot_n15/forward.py`
  (`GrootN15Forward`); target binding: `models/groot_n15/pipeline_rtx.py`;
  attention bindings `_attention.py`; projection lowerings `_linear.py`.
* Frontend: `flash_rt/frontends/torch/groot_n15_rtx_sm89.py` (inputs, CUDA
  graphs, input-independent tables), `_groot_n15_weights.py` (checkpoint
  adapter), `_groot_n15_io.py` (upstream transforms).
* Kernels: `flash_rt_groot_n15` (below), the shared `flash_rt_kernels`
  entries `patch_im2col_nchw_bf16`, `embedding_lookup_bf16`,
  `bf16_matmul_cublaslt_ref_bf16`, `rms_norm_hf_bf16`,
  `residual_add_rms_norm_hf_bf16`, `qk_rms_norm_rope_hf_bf16`,
  `silu_mul_hf_bf16` and (FP8) `GemmRunner.fp8_nt_dev`, and
  `flash_rt_fa2.fwd_bf16` / `fwd_bf16_causal`.

The timestep-dependent AdaLayerNorm modulation, the action encoder's step
encoding, the position embeddings and the future tokens do not depend on the
input; the frontend builds them once from the checkpoint with the
reference's torch expressions. Nothing in the call path runs the upstream
model.

### `flash_rt_groot_n15`

| Entry | What it computes |
|---|---|
| `groot_n15_linear_bias_bf16` | `x @ W^T + b` on the cuBLASLt bias-epilogue algorithm torch's `addmm` runs |
| `groot_n15_bmm_bf16` | `x @ W` on the cuBLAS strided-batched call `torch.bmm` makes for one batch |
| `groot_n15_layer_norm_bf16` | optional bf16 residual add, LayerNorm (affine or not) to bf16 |
| `groot_n15_layer_norm_to_f32` | LayerNorm to fp32 (vlln) |
| `groot_n15_residual_layer_norm_f32` | fp32 residual add + LayerNorm to bf16 (VL self-attention) |
| `groot_n15_residual_add_f32_to_bf16` | final fp32 residual add, bf16 result |
| `groot_n15_ada_layer_norm_bf16` | optional residual add, `LayerNorm(x) * (1 + scale) + shift` with bf16 tables |
| `groot_n15_add_rows_bf16` | optional bias, then a per-row table (position embeddings) |
| `groot_n15_bias_act_bf16` | bias then nothing / ReLU / swish |
| `groot_n15_concat_bias_bf16` | action-encoder concat of `x + b` with the step encoding |
| `groot_n15_gelu_tanh_bf16` | GELU (tanh) in torch's expression order |
| `groot_n15_quant_rows_fp8` | FP8 tier: per-token E4M3 activation quantization |
| `groot_n15_rescale_rows_bf16` | FP8 tier: token x channel dequantization + bias |
| `groot_n15_actions_init`, `groot_n15_euler_step` | flow-matching state (fp32) and its bf16 copy |

Each entry takes raw device pointers and a stream, allocates nothing, and is
captured into the CUDA graph. The module is built without `--use_fast_math`;
the ops spell out every rounding.

## Numerical contract

**BF16 tier.** Every projection with a bias runs torch's addmm algorithm,
every projection without one (the Qwen3 decoder) torch's matmul algorithm,
and the embodiment MLPs torch's bmm, so each produces the reference's bits
-- against PyTorch 2.5, the release the GR00T N1.5 reference pins, whose
descriptors and workspace defaults these calls reproduce (they read
`CUBLASLT_WORKSPACE_SIZE` / `CUBLAS_WORKSPACE_CONFIG` as torch does); with
another PyTorch release the reference may pick other algorithms and the
projections are close rather than equal;
q/k/v and gate/up stay separate projections, as in the reference. The ops
reproduce the reference's dtypes under bf16 autocast: LayerNorm in fp32
rounded where the next Linear casts it, a bf16 residual stream in the vision
tower, language model and DiT and an fp32 one in the VL self-attention,
AdaLayerNorm's two fp32 roundings with bf16 `(1 + scale)` and `shift`, the
bias added in bf16 after each bmm. What differs from the reference: the
patch projection is an im2col GEMM rather than cuDNN's implicit-GEMM
convolution, LayerNorm reduces in another order (1-ulp differences on ~1e-5
of the elements), and attention is FlashRT's FA2 rather than flash-attn /
torch SDPA. The attention sites at head_dim <= 64 run without split-KV.

**FP8 tier.** The target's `fp8_projections` -- the vision tower, the VL
self-attention, the DiT's cross-attention K/V and the Qwen3 attention
projections -- run E4M3 with per-output-channel weight scales (set at load)
and per-token activation scales (computed on the device each call; no
calibration, nothing cached). The Qwen3 MLP projections stay BF16: their
inputs carry per-channel outliers a per-token scale does not absorb, and on
the fixture observations they moved the actions several times more than any
other group. The patch projection and the 49-row action head also stay
BF16. Each FP8 GEMM runs cuBLASLt's heuristic first choice: the candidate
algorithms are equally accurate (they differ by bf16 output rounding), but
the policy amplifies such ulp-level differences, so a run-dependent choice
would make the FP8 actions differ from one process to the next.

## Validation (RTX 4090)

```bash
python scripts/groot_n15_parity.py --checkpoint <checkpoint> --data-config panda_omron \
    --episodes <GR00T-X-Embodiment-Sim>/single_panda_gripper.{OpenDrawer,PnPCounterToSink,TurnOnMicrowave} \
    --noise-floor
```

Reference: the unmodified upstream `Gr00tPolicy` (Isaac-GR00T RoboCasa fork
at `9d7d7a9e`) with the RoboCasa checkpoint above, `panda_omron`, PyTorch
2.5.1+cu124, transformers 4.51.3, diffusers 0.30.2, flash-attn 2.7.4.post1.
Inputs: 9 real RoboCasa observations (three views, frames at 0, 1/3 and 2/3 of
episode 0 of `OpenDrawer`, `PnPCounterToSink` and `TurnOnMicrowave` from the
GR00T X-Embodiment sim dataset), two noise seeds each. FlashRT receives the
reference's normalized inputs and its initial noise.

| Metric (18 calls) | BF16 | FP8 | Reference noise floor |
|---|---:|---:|---:|
| normalized action cosine (32 dims), min | 0.999983 | 0.998783 | 0.999972 |
| real action dims (12 of 32): cosine, min | 0.999716 | 0.987941 | 0.999330 |
| real action dims: max abs error | 0.033 | 0.243 | 0.037 |
| discrete outputs (`gripper_close`, `control_mode`) equal | 18 / 18 | 18 / 18 | 18 / 18 |

"Reference noise floor" is the upstream policy against itself with its
attention switched from flash-attn to SDPA (same inputs and noise). For
scale: two noise seeds of the upstream policy on the same observation differ
by up to 0.98 on the real action dims (min cosine 0.73).

CUDA graph: replay is bitwise equal to eager execution; observations A -> B
-> A of one prompt length reuse one graph and reproduce the eager results; a
second prompt length beyond `graph_cache_size` evicts the first, which is
recaptured on its next call with the same results.

## Performance (RTX 4090)

```bash
python benchmarks/groot_n15_latency.py --checkpoint <checkpoint> --data-config panda_omron
```

Median of 50 calls after 5 warm-up calls, one observation for every row
(three synthetic 256x256 frames, a state inside the checkpoint's range, the
default prompt: 814 tokens, 768 of them image tokens). PyTorch 2.5.1+cu124,
CUDA 12.4, `GPU_ARCH=89`.

| Path | `get_action` / `infer` | `predict_normalized` | graph replay |
|---|---:|---:|---:|
| upstream `Gr00tPolicy` (eager, flash-attn) | 81.2 ms | -- | -- |
| FlashRT BF16 | 35.0 ms | 25.4 ms | 24.9 ms |
| FlashRT FP8 | 30.8 ms | 21.7 ms | 21.2 ms |

`infer` and `get_action` are wall clock and include the upstream transforms
(about 8 ms on the CPU) and the action unnormalization; `predict_normalized`
is input staging plus graph replay, wall clock; "graph replay" is the replay
alone, timed with CUDA events.

## Tests

```bash
PYTHONPATH=. pytest -q tests/test_groot_n15_kernels.py tests/test_groot_n15_rtx_sm89.py
# + parity against the upstream policy on an RTX 4090:
GROOT_N15_CHECKPOINT=... GROOT_N15_ROOT=... GROOT_N15_DATA_CONFIG=panda_omron \
  PYTHONPATH=. pytest -q tests/test_groot_n15_rtx_sm89.py
```
