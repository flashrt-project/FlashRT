# MolmoAct2 — RTX SM120 usage

MolmoAct2 (allenai) is a single-stream **VLA** built from three towers:

* a SigLIP-style **vision tower** — 25 blocks, 16 heads of width 72, 14×14
  patches at a 378×378 resize (729 tokens per view, 3 views);
* a **pooling adapter** — the two checkpoint-selected ViT feature layers
  (24 and 18) are concatenated (2×1152), mean-pooled over 2×2 windows, and
  projected to the text width;
* a Qwen2.5-style **VLM text tower** — 36 layers, GQA 32 Q / 8 KV heads of
  width 128, producing the per-layer K/V context;
* a 36-layer **flow-matching action expert** — 8 heads of width 96, cross
  attention over the text K/V, 10 Euler steps, 30×14 action chunk.

The RTX path runs the complete pipeline — preprocessing, all three towers, the
flow loop, the pinned host↔device copies, and the action readback — inside one
CUDA Graph, using repository kernels only (cuBLASLt BF16/FP8, CUTLASS SM120
block-scaled NVFP4 GEMMs, the vendored FA2, the SM120 FP8 causal GQA attention,
and the MolmoAct2 fused producers).

MolmoAct2's observation includes the 14-dim robot state and its graph is
calibrated on real observations, so it does not fit `load_model`'s
`predict(images, prompt)` wrapper. `flash_rt.load_model(config="molmoact2")`
raises with this redirect; construct the frontend directly.

## Support matrix

| | |
|---|---|
| Hardware | RTX 5090 / RTX 5060 Ti / Blackwell **SM120/SM121** |
| Unsupported | SM89, SM87, SM80/86, SM110; JAX; `precision` other than `nvfp4`/`fp8`/`bf16` |
| GPU memory | ~3.6 GiB peak at the delivered configuration |
| Framework | PyTorch (`torch`) |
| Precision | `nvfp4` (delivered), `fp8`, `bf16` (reference) |
| Build flags | `-DGPU_ARCH=120 -DFLASHRT_ENABLE_MOLMOACT2=ON` |
| Checkpoint | `allenai/MolmoAct2` (safetensors + `config.json` + `norm_stats.json` + `tokenizer.json`) |
| Observation | 3 × 360×640 uint8 frames + 14-dim state + fixed task string |
| Output | 30×14 float32 action chunk on the CPU |
| Flow schedule | 10 Euler steps (the checkpoint's `flow_matching_num_steps`) |
| Latency | **51.0 ms** graph replay, **7.87×** over the bf16 HF reference |
| Graph coverage | one replay; 0 ATen/Triton ops, 0 allocations, 0 syncs inside |

## 1. Build

The model-specific kernels are gated behind an explicit option. It requires
`GPU_ARCH=120` or `121`: the action-expert kernel, the small-M NVFP4 GEMM and
the 72-deep vision attention use block-scaled NVFP4 MMA, TMA/bulk async copies
and the SM120 shared-memory budget. Unrelated builds never compile them.

```bash
cmake -S . -B build -DGPU_ARCH=120 -DFLASHRT_ENABLE_MOLMOACT2=ON
cmake --build build -j$(nproc)
```

This produces `flash_rt/flash_rt_kernels*.so` (with the `molmo_*` producers,
the persistent action kernel, the small-M NVFP4 GEMM and the vision attention)
and, on the same arch, `flash_rt/flash_rt_fa2*.so` (cross attention and the
optional FP8-output epilogue). Omitting the flag still builds
`flash_rt_kernels`, but the MolmoAct2 symbols are absent and
`MolmoAct2TorchFrontendRtx(...)` raises a `RuntimeError` naming the flag before
any weight is loaded.

Configure-time errors are explicit: `FLASHRT_ENABLE_MOLMOACT2=ON` on a non-
SM120/121 target fails at configure time instead of producing kernels that
would trap at run time.

## 2. Install

```bash
pip install -e ".[torch]"
```

The frontend needs `tokenizers` (shipped with `transformers`) to build the
prompt, and `ml_dtypes` only when a caller supplies explicit flow noise.

## 3. Quickstart

The checkpoint is the official `allenai/MolmoAct2` safetensors directory plus
its `config.json`, `tokenizer.json` and `norm_stats.json`. `norm_tag` selects
the robot's statistics entry in `norm_stats.json` (the validated entry is
`yam_dual_molmoact2` for the BimanualYAM setup; another robot needs its own
tag).

```python
import numpy as np
from flash_rt.frontends.torch.molmoact2_rtx import MolmoAct2TorchFrontendRtx

fe = MolmoAct2TorchFrontendRtx(
    "<checkpoint_path>",        # allenai/MolmoAct2 directory
    precision="nvfp4",          # production tier
    num_steps=10,               # checkpoint flow schedule
    norm_tag="yam_dual_molmoact2",
)

task = "pick up the block and place it in the tray"
fe.set_prompt(task)             # builds the static 692-token prompt

# Static FP8/NVFP4 scales must come from real observations, once, before
# capture. `samples` is a list of (images, state) pairs; images are
# (3, 360, 640, 3) uint8 (top, left, right), state is (14,) float32.
fe.calibrate(samples)
fe.record_infer_graph()

out = fe.infer(images, state)   # one graph replay
actions = out["actions"]        # (30, 14) float32, CPU
```

`fe.run_eager(images, state)` executes the same pipeline without the captured
graph (debug / parity only); it is slower and must not be used for latency
claims.

## 4. Constructor reference

```python
MolmoAct2TorchFrontendRtx(
    checkpoint_dir,
    *,
    device: str = "cuda",
    precision: str = "bf16",        # "bf16" | "fp8" | "nvfp4"
    num_steps: int = 10,            # flow-matching Euler steps
    norm_tag: str = "yam_dual_molmoact2",
    use_cuda_graph: bool = True,
    fp8_layout: str | None = None,  # only the "kn" weight layout ships
    hardware: str | None = None,
)
```

| Parameter | Values | Meaning |
|---|---|---|
| `checkpoint_dir` | path | MolmoAct2 directory (`config.json`, `norm_stats.json`, `tokenizer.json`, safetensors). |
| `device` | `"cuda"`, `"cuda:N"` | Target GPU. SM120/SM121 required; the kernels are compiled for `sm_120a`. |
| `precision` | `"bf16"` | Full-precision reference path (no calibration needed). |
| | `"fp8"` | E4M3 block-128 weights/activations, static real-data scales. |
| | `"nvfp4"` | Delivered tier: NVFP4 scope `tvp` over the text GEMMs, vision `w1`/`qkv` and the pooling adapter; FP8 elsewhere; BF16 at the alignment-excluded sites (`vis_patch_embed`, `act_embed`, `act_final`). |
| `num_steps` | int | Flow-matching Euler steps. 10 is the checkpoint's trained schedule; 2 passes the 0.99 gate on the calibration set but changes the trajectory and is not the delivered configuration. |
| `norm_tag` | str | `metadata_by_tag` entry in `norm_stats.json` (robot/setup and action mask). |
| `use_cuda_graph` | bool | Capture the full steady-state pipeline (default). |
| `fp8_layout` | `"kn"` | Weight layout; only `kn` (`bf16_nn`/`fp8_nn_dev`) is implemented. |
| `hardware` | str | Optional layout hint; the default resolves the same `kn` layout. |

Methods:

* `set_prompt(task, *, state=None, norm_tag=None)` — tokenize the fixed task,
  discretize the state, count the 3×196 image tokens, and allocate every
  static buffer.
* `calibrate(samples)` — run the real observations through the pipeline to
  collect the static FP8/NVFP4 scales, then freeze them. Required before
  capture for `fp8`/`nvfp4`.
* `record_infer_graph()` — capture the whole steady-state pipeline on a
  dedicated stream.
* `infer(images, state, noise=None) -> {"actions": (30, 14) ndarray}` — fill
  the pinned inputs, replay the graph once, synchronize, and return the action
  chunk.
* `run_eager(images, state, noise=None)` — same math without the graph.
* `precision_spec`, `state_dict_ptrs()` — introspection.

## 5. Performance

Measured on an RTX 5060 Ti (SM120, 16 GB), CUDA 13.0, PyTorch 2.14.1+cu130,
checkpoint `allenai/MolmoAct2`, batch 1, 3 × 360×640 frames, 692-token prompt,
30×14 chunk, at the delivered configuration: `precision="nvfp4"`, NVFP4 scope
`tvp`, 10 Euler steps, one captured graph, 20 warmup replays before the timed
loop. The metric is **CUDA Graph replay latency of the complete equal-scope
pipeline** — the pinned host→device input copies, all model math, and the
device→pinned-host action copy are inside the replay; the only graph-external
work is one `cudaStreamSynchronize` after the replay.

| Metric | Value |
|---|---|
| E2E graph replay, median (n=100) | **51.0 ms** (min 50.8, p90 51.0) |
| Sustained 1000-replay validation | median 50.96 ms, drift +0.27 ms |
| HF bf16 reference, same observations | 401.2 ms (n=20) |
| Speedup | **7.87×** |
| Peak GPU memory | 3.56 GiB (reference 11.20 GiB) |
| Graph launches per inference | 1 |
| Hot-path ATen / Triton / Python / allocs / syncs | 0 / 0 / 0 / 0 / 1 readback sync |

Where the 51 ms goes (current profile): NVFP4 text GEMMs ~24.7 ms, the
persistent action-expert kernel ~15.8 ms (~44 µs/layer against a ~25 µs DRAM
floor), vision attention ~4.2 ms, text attention ~2.8 ms, fused producers
~2.0 ms, graph gaps ~0.5 ms.

The reference number is the official HF processor + bf16 model with the
official action CUDA graph on the same raw frames, so the comparison includes
preprocessing on both sides. Do not compare these numbers to wall-clock
`quickstart.py --benchmark` latencies from other models — this table reports
graph replay, not wall-clock `predict()`.

## 6. Precision

Correctness is measured as the cosine between the candidate and the bf16 HF
action chunk on real BimanualYAM observations, with the static scales frozen
on a disjoint calibration set:

* calibration set: 8/8 samples ≥ 0.99, minimum **0.9915**, mean 0.9984;
* held-out set: 12/12 samples ≥ 0.99, minimum **0.9963**;
* graph vs eager: **bit-equal** on all delivery samples (max abs difference
  0.0);
* post-run cosines after 1000 replays are unchanged (min 0.9915), i.e. the
  captured graph does not drift.

The NVFP4 scope is a measured choice, not an assumption: quantizing every
vision projection (`tv`), or `vis_qkv+vis_o`, or using one Euler step, all fall
below the 0.99 gate on the calibration set. The delivered scope is the largest
one that passes both sets.

## 7. Environment knobs

The route switches below were measured; the defaults are the delivered
configuration. The `0` variants are kept for reproducing the measured
non-winners, not because they are expected to be faster.

| Variable | Default | Meaning |
|---|---|---|
| `FLASHRT_MOLMO_NVFP4_SCOPE` | `tvp` | NVFP4 weight scope: `tvp` (delivered), `tvm`, `tv`, `tvmqp`, `full`. Scopes above `tvp` fail the 0.99 gate. |
| `FLASHRT_MOLMO_NVFP4_EXTRA` | – | Comma-separated extra NVFP4 weight sites (debug / scope sweeps). |
| `FLASHRT_MOLMO_BF16_SITES` | – | Comma-separated prefixes forced to BF16. |
| `FLASHRT_MOLMO_SKINNY_SCOPE` | `outd` | Action-expert skinny NVFP4 sites: `outd`, `out`, `attn`, `ffn`, `all`, `off`. Broader scopes fall below the gate. |
| `FLASHRT_MOLMO_ACTION_MK` | `1` | Persistent action-expert layer kernel (all 36 layers, grid barriers, L2 prefetch). |
| `FLASHRT_MOLMO_VIT_ATTN` | `f16pv` | Vision attention: `f16pv` (delivered, FP16-accumulated P·V), `native` (all-bf16, needed for the fused FP8 epilogue), `persist` (persistent-CTA experiment), `fa2`. |
| `FLASHRT_MOLMO_FP8_ATTN` | `1` | SM120 FP8 causal GQA kernel for the text prefill; `0` uses vendored bf16 FA2. |
| `FLASHRT_MOLMO_ATTN_Q64` | `1` | 64×64 text-attention tile with two CTAs per SM; `0` keeps 128×128. |
| `FLASHRT_MOLMO_XATTN_SMS` | `18` | Cross-attention split budget (18 → 4 KV splits at 30×692×8×96). |
| `FLASHRT_MOLMO_SWIGLU_FOLD` / `_CEIL_SF` | `1` / `1` | Text-FFN SwiGLU epilogue fold and its ceil block-scale rounding. |
| `FLASHRT_MOLMO_ACT_FUSED` | `1` | One fused up+gate+SwiGLU producer instead of three launches. |
| `FLASHRT_MOLMO_CTX_BATCHED` / `_CTX_GROUP` | `1` / `2` | Batch the per-layer action-context projections. |
| `FLASHRT_MOLMO_QKV_VEC`, `_SILU_VEC` | `1` | Vectorized FP8 q/k norm+RoPE and SwiGLU→NVFP4 producers. |
| `FLASHRT_MOLMO_FA2_FP8OUT` | `0` | Emit the vision o-projection FP8 operand from the FA2 epilogue (opt-in). |
| `FLASHRT_MOLMO_SMALLM_GEMM`, `_ATTN_WSP`, `_FUSED_RED` | `0` | Measured non-winners kept as opt-in experiments (bit-identical output, slower). |

## 8. Reproduce

The full equal-scope harness needs the checkpoint and real observations. The
minimal timing loop is:

```python
import time, torch
from flash_rt.frontends.torch.molmoact2_rtx import MolmoAct2TorchFrontendRtx

fe = MolmoAct2TorchFrontendRtx("<checkpoint_path>", precision="nvfp4")
fe.set_prompt("<task>")
fe.calibrate(samples)              # real (images, state) pairs
fe.record_infer_graph()

for _ in range(20):
    fe.infer(images, state)        # warm the clocks

times = []
torch.cuda.synchronize()
for _ in range(100):
    t0 = time.perf_counter()
    fe.infer(images, state)
    times.append((time.perf_counter() - t0) * 1e3)
times.sort()
print(f"median {times[50]:.2f} ms  min {times[0]:.2f} ms")
```

`fe.infer` returns only after the graph-external readback sync, so the
per-call wall time is replay + one sync. For a replay-only figure, time the
`torch.cuda.Event` pair around the same call; the difference is the single
readback synchronization.

## 9. Tests

```bash
PYTHONPATH=. python -m pytest tests/test_molmoact2_smoke.py -v
```

The smoke test covers registry routing, the `load_model` redirect, the gated
kernel surface, the static tower dimensions, and the pure prompt/state
helpers. It needs neither a checkpoint nor a GPU; on a build without
`-DFLASHRT_ENABLE_MOLMOACT2=ON` the kernel-surface test skips and the
fail-fast test asserts the actionable error.

## 10. Limitations

* SM120/SM121 only. The action-expert kernel and the small-M NVFP4 GEMM use
  block-scaled NVFP4 MMA and 96 KB+ shared memory.
* Fixed production shapes: 3 views at 360×640, a 692-token prompt, a 30-step
  chunk and a 14-dim state. A different task changes the prompt length and
  requires a new `set_prompt` + capture; a different camera resolution or
  action dimension changes the static buffers.
* `precision="fp8"`/`"nvfp4"` require real-observation calibration before
  capture; there is no random-input calibration path.
* One prompt/graph per frontend instance, batch size 1. The flow noise is
  fixed by the pipeline (pass `noise=` to `infer` to reproduce a specific
  trajectory).
* The `norm_tag` must exist in the checkpoint's `norm_stats.json`; the
  validated tag is the BimanualYAM one.
* Not wired into `load_model` / the `predict()` VLA API (see the redirect at
  the top).
