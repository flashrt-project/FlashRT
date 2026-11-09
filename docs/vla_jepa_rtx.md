# VLA-JEPA (LeRobot) on RTX SM120/SM121

FlashRT native path for [LeRobot **VLA-JEPA**](https://huggingface.co/lerobot/VLA-JEPA-LIBERO):
a Qwen3-VL-2B vision-language backbone feeding a 16-block DiT-B flow-matching
action head, exported as a LeRobot policy.

The whole steady-state policy — raw uint8 frames, state normalization, the ViT
tower, the 28-layer Qwen3-VL decoder, the 4-step DiT flow head and the action
post-processing — is captured as **one CUDA Graph**. A decision is one graph
replay; the graph holds no ATen/framework or Triton kernels.

## Support matrix

| Framework | Hardware | Precision | Status |
|---|---|---|---|
| torch | `rtx_sm120` (SM120/SM121) | NVFP4 W4A4 + FP8 block-128 | **Validated** on RTX 5060 Ti 16 GB |
| torch | `rtx_sm120` (SM120/SM121) | FP8 / BF16 (`llm_precision`, `dit_precision`) | Structurally supported; accuracy and latency must be re-measured |
| torch | Thor SM110 | — | Not supported: this path uses SM120 block-scaled NVFP4/FP8 CUTLASS and the SM120 FP8 FMHA |
| torch | RTX 4090 SM89 / Orin SM87 | — | Not supported: no SM120 block-scaled kernels |
| JAX / AMD / NPU / Apple MPS | — | — | Not implemented |

The frontend fails fast when constructed on a non-SM120/SM121 device. Batch
size is 1, the prompt is padded to `seq_len=256`, and the action chunk is
`(7, 7)`.

## 1. Build

The model-specific preprocessing/post-processing and new K128 GEMM variants are opt-in.
It is compiled only with `FLASHRT_ENABLE_VLA_JEPA=ON` on `GPU_ARCH=120/121`;
the `vla_*` bindings are compiled under the same `FLASHRT_HAVE_VLA_JEPA`
definition. The K128 implementation, exports and instance launch-policy class
follow the same gate; existing shared GEMM interfaces remain available when OFF.

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -S . -B build \
  -DGPU_ARCH=120 \
  -DFLASHRT_ENABLE_VLA_JEPA=ON \
  -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j$(nproc)
```

`FA2_ARCH_NATIVE_ONLY=ON` is useful for a single-GPU developer build. The VLA
model also needs the Qwen3-VL module because its ViT/qkv producer lives in
`flash_rt_qwen3_vl_kernels`.

## 2. Install

```bash
pip install -e ".[torch,vla_jepa]"
```

The `vla_jepa` extra adds the Qwen3-VL `AutoProcessor` dependency
(`transformers>=4.57`). The `torch` extra already provides `torch` and
`safetensors`.

## 3. Checkpoint layout

The frontend reads a LeRobot VLA-JEPA checkpoint with this layout:

```
<checkpoint_path>/
  config.json
  model.safetensors
  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor_step_2_unnormalizer_processor.safetensors
  flashrt_qwen3vl/
    config.json
    tokenizer.json / tokenizer_config.json / processor files
    model.safetensors          # HF Qwen3-VL layout: model.visual.*,
                               # model.language_model.*, lm_head.weight
```

`flashrt_qwen3vl/` is a documented adapter, not a redistribution of the
checkpoint: extract the `model.qwen.model.*` tensors from the LeRobot
`model.safetensors` into the HF Qwen3-VL layout (`model.visual.*`,
`model.language_model.*`, `lm_head.weight`) and copy the Qwen3-VL-2B
`config.json` and tokenizer/processor files next to it. The model path loads
weights through `safetensors` and never downloads a checkpoint.

## 4. Quickstart

```python
import numpy as np
from flash_rt.frontends.torch.vla_jepa_rtx import VlaJepaTorchFrontendRtx

fe = VlaJepaTorchFrontendRtx("<checkpoint_path>", llm_precision="nvfp4",
                             dit_precision="nvfp4")
fe.set_prompt("put the white mug on the plate", observation_frames)  # [V,H,W,3]
# Optional: required only for static/clipped activation modes or a custom
# per-layer down selection. The default dynamic per-row scales need no
# calibration.
fe.calibrate([(frames, state) for frames, state in real_samples])
actions = fe.infer(observation_frames, state_raw)   # [7, 7] float32
```

`set_prompt` tokenizes the instruction, builds the image spans and MRoPE
tables, and stages the task-constant token embeddings. It can be called again
for a new task; the graph is recaptured only when the prompt geometry changes.

## 5. Constructor reference

| Argument | Default | Meaning |
|---|---|---|
| `checkpoint` | — | LeRobot checkpoint root (see layout above) |
| `qwen_dir` | `<checkpoint>/flashrt_qwen3vl` | HF Qwen3-VL-2B layout directory |
| `action_ckpt` | `<checkpoint>` | Directory holding the DiT/action-head weights |
| `device` | `"cuda:0"` | CUDA device |
| `num_views` | `2` | Camera views |
| `seq_len` | `256` | Padded prompt length (must cover the tokenized prompt) |
| `num_image_tokens` | `128` | Image tokens per view before merge |
| `chunk_size` / `action_dim` | `7` / `7` | Action chunk shape |
| `state_dim` | `8` | Raw robot-state width |
| `num_cond_tokens` | `32` | DiT conditioning tokens |
| `num_inference_timesteps` | `4` | Euler steps |
| `timestep_buckets` | `1000` | Timestep embedding table size |
| `resize_images_to` | `(224, 224)` | Area-resize target before the Qwen3-VL processor resize |
| `use_cuda_graph` | `True` | Capture/replay the whole steady-state pipeline |
| `gemm` | `"cublaslt"` | BF16 projection backend (`"cublaslt"` or `"w16a16"`) |
| `llm_precision` | `"nvfp4"` | Decoder GEMM tier: `bf16`, `fp8`, `fp8_residual_safe`, `nvfp4` |
| `nvfp4_sites` | `"all"` | Decoder NVFP4 sites (`mlp`, `gate_up`, `down`, `qkv`, `attn`, `all`, or a comma list) |
| `nvfp4_act_mode` | `"dynamic"` | Activation quantizer: `dynamic`, `mse`, `secondmax`, `static`, `clipped` (`static`/`clipped`/`mse:<sites>` need `calibrate`) |
| `llm_attention` | `"fp8_fp4out"` | Decoder prefill attention route served by the RTX Qwen3 attention backend: `fp8_fp4out`, `fp8`, `sage2`, `fa2` |
| `dit_precision` | `"nvfp4"` | DiT tier: `bf16` or `nvfp4` |
| `verbose` | `False` | Print setup/calibration diagnostics |

`fe.calibrate(samples, warm=2)` accepts an iterable of `(frames_u8, state_raw)`
pairs using the same shapes as `infer`. It accumulates the real-observation
activation statistics used by the `static`/`clipped` activation modes and by
per-layer down selection; it invalidates a captured graph.

## 6. Performance

Validated on **RTX 5060 Ti 16 GB (SM120), driver 595.84, CUDA 13.0,
torch 2.11.0+cu130**, batch 1, 2 views, `seq_len=256`, 4 Euler steps,
chunk `(7,7)`, NVFP4 defaults. Inputs are 16 real LIBERO observations
(4 suites) × 3 repetitions after warmup.

| Metric | Value |
|---|---|
| E2E wall p50 (raw frames → post-processed chunk) | **8.114 ms** |
| CUDA Graph replay median (5 reps) | **8.000 ms** |
| Stock LeRobot 0.6.1 eager/SDPA/bf16 baseline, same 16 frames | **69.631 ms** p50 |
| Speedup | **8.58×** |
| Action-chunk cosine vs the LeRobot chunk | min **0.996595**, mean **0.998798** (gate 0.99) |
| Disjoint 8-frame held-out set | min **0.998756** |
| Graph coverage | `complete=true`, `graph_external_model_math_ops=0`, `cuda_graph_launches=1` |
| Operator provenance | 978.6 device events/replay, 53 distinct kernels, **0 ATen / 0 Triton** |

The wall-clock metric includes host staging, the boundary H2D/D2H memcpy
nodes, graph replay, readback and post-processing. The replay metric excludes
the host-side staging/readback. Do not compare this table's replay number to
a wall-clock baseline from another harness.

### Reproduce

The validation harness used the same real frames and boundary for baseline
and candidate; it is not shipped as a public example. The minimal timing loop
for the frontend is:

```python
import time, torch
from flash_rt.frontends.torch.vla_jepa_rtx import VlaJepaTorchFrontendRtx

fe = VlaJepaTorchFrontendRtx("<checkpoint_path>")
fe.set_prompt(task, frames[0])
for frames, state in frames_and_states[:10]:       # warmup
    fe.infer(frames, state)
torch.cuda.synchronize()

ts = []
for frames, state in frames_and_states:
    t0 = time.perf_counter()
    fe.infer(frames, state)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()
print("p50 wall ms:", ts[len(ts) // 2])
```

For a correctness comparison, run the same raw observations and flow start
state through the canonical LeRobot 0.6.1 policy and compare the delivered
`(7,7)` chunks with a per-frame cosine; the gate is 0.99.

## 7. Precision and calibration

The delivered NVFP4 default is a measured choice, not an assumption:

* **ViT tower**: all 24 blocks and all four linear sites run SM120 block-scaled
  NVFP4 W4A4. The fc1 GEMM folds bias+GELU into its epilogue on all blocks by
  default; merger fc1 is NVFP4 and merger fc2 stays FP8 block-128 (the
  GELU-output tails cost accuracy in 4 bits).
* **Qwen3-VL decoder**: q/k/v, o and gate_up run NVFP4 W4A4; the MLP uses the
  repository dual-GEMM SwiGLU fold; down runs NVFP4 on all 28 layers. Layer 2
  carries an extreme SiLU tail (calibrated amax ~4.4e4) and is pre-scaled by
  exactly 2^-4 in the fold, restored in the down GEMM's alpha. Attention is the
  all-FP8 causal GQA FMHA with the fused NVFP4 o-operand epilogue.
* **DiT-B flow head**: FFN on the skinny NVFP4 family; q/k/v/o W4A4 on all 16
  blocks with attention executing; conditioning K/V NVFP4 with a bias epilogue.
* **Calibration**: weights are packed once at load; activations use per-row,
  per-16-element dynamic scales. The per-layer/per-block exceptions and the
  `static`/`clipped` modes are calibrated from real observations; the default
  dynamic mode does not require a calibration pass.

Correctness is measured against the canonical LeRobot chunk on the same real
inputs and flow start state. The gate is min cosine ≥ 0.99 on the 16-frame
delivery set, with a disjoint 8-frame set as a generalization check.

## 8. Precision policy and process-wide knobs

The delivered precision policy (ViT/DiT/decoder site and block selection,
NVFP4 tiles and fusion folds) is a static calibrated description in
`flash_rt/frontends/torch/_vla_jepa_rtx_spec.py`; it is not configured through
environment variables. Model-level choices are constructor arguments (§5).

The frontend and pipeline own a `VlaJepaKernelPolicy`. It enables the validated
producers and PDL only while executing that instance's setup/forward call.
The native scope is thread-local and restores the previous policy on return or
exception; other models keep their defaults. Captured graph nodes retain their
launch attributes without leaving a policy enabled between calls.
The `FLASHRT_*` variables below remain explicit process-level overrides:

| Variable | Default | Effect |
|---|---|---|
| `FLASHRT_PDL` | off (VLA opts in) | Programmatic dependent launch; `0` serializes it, `1` forces it on |
| `FLASHRT_QUANT_V3` | old path (VLA opts in) | Register-resident BF16→NVFP4 quantizer; `0` forces the staged kernel |
| `FLASHRT_FP8_QUANT_WARP` | old path (VLA opts in) | Warp-per-block BF16→FP8 block-128 quantizer; `0` forces the 128-thread kernel |
| `FLASHRT_LN_QUANT_V3` | old path (VLA opts in) | Register-resident LayerNorm→NVFP4 producer; `0` forces the staged chain |
| `FLASHRT_RMS_NVFP4_ROWS` | old path (VLA opts in) | Register-resident RMSNorm→NVFP4 rows producer |
| `FLASHRT_SILU_FP8_WARP` | old path (VLA opts in) | Warp-per-block SiLU→FP8 producer |

## 9. Tests

```bash
PYTHONPATH=. python -m pytest tests/test_vla_jepa_smoke.py -v
PYTHONPATH=. python -m pytest tests/test_vla_jepa_kernels.py -q
PYTHONPATH=. python -m pytest tests/test_kernel_policy.py -q
PYTHONPATH=. python -m pytest tests/test_fmha_fp8_causal_gqa.py -q
```

`test_vla_jepa_smoke.py` needs neither a checkpoint nor a GPU: it covers
registry routing, the `load_model` redirect, the gated-kernel fail-fast and
the constructor guard. `test_vla_jepa_kernels.py` checks the
preprocessing/post-processing kernels against float references and skips when
the VLA-JEPA object is not built. `test_kernel_policy.py` pins the shared
producer/PDL defaults to the pre-existing path and checks the environment
override precedence. The FMHA test covers the shared SM120 FP8 attention
route used by the decoder.

## 10. Limitations

* RTX SM120/SM121 only; no SM89/Thor/AMD/NPU/MPS fallback.
* Batch size 1, a single prompt padded to `seq_len=256`, and the LIBERO
  `(7,7)` action schema. Other VLA-JEPA checkpoints require revalidation.
* The checkpoint adapter expects the LeRobot VLA-JEPA layout described in §3;
  it does not download or convert arbitrary checkpoints.
* The default dynamic activation scales need no calibration; the `static` and
  `clipped` modes and the per-layer down selection do, and their accuracy is
  checkpoint/data dependent.
* The VLA-JEPA preprocessing kernels are model-specific and are compiled only
  when `FLASHRT_ENABLE_VLA_JEPA=ON`.

## Files

| Layer | Path |
|---|---|
| Model pipeline + static buffers | `flash_rt/models/vla_jepa/pipeline_rtx.py` |
| Frontend: prompt, preprocessing, graph capture | `flash_rt/frontends/torch/vla_jepa_rtx.py` |
| Checkpoint layout, weight loader, precision policy | `flash_rt/frontends/torch/_vla_jepa_rtx_spec.py` |
| Decoder attention route | `flash_rt/hardware/rtx/attn_backend_qwen3.py` |
| Boundary preprocessing/post-processing kernels | `csrc/kernels/vla_preprocess.cu` |
| Routing | `flash_rt/hardware/__init__.py` (`_PIPELINE_MAP`) |


Default inference noise is initialized before graph warmup/capture using a
local NumPy generator with seed 0, rounded to BF16. It is reused across calls
until `set_noise(noise)`, `infer(noise=...)` or `actions_norm` updates it;
`set_noise()` restores that fixed Gaussian default. Every eager decision and
graph replay starts from the staged noise rather than the preceding Euler
loop's output. No global NumPy or torch RNG state is modified.
