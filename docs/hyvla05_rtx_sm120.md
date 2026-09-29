# Hy-Embodied-0.5-VLA on RTX consumer Blackwell (SM120)

> RTX (SM120) adaptation of the existing HyVLA Thor path. SM120 has
> tcgen05 FP8 (block-128) and NVFP4 tensor cores but neither the SM110 (Thor)
> FP8 megakernel path nor the SM80-family INT8 W8A8 kernels that Orin SM87
> uses. The RTX frontend keeps the validated HyVLA IO / prefix / CUDA-graph
> orchestration and binds the lower-precision GEMM slots to the SM120
> block-128 FP8 and NVFP4 kernels.

## Platform

| Field | Value |
|---|---|
| Device | NVIDIA GeForce RTX 5060 Ti (Blackwell) — the validation target |
| GPU family | SM120 only (validated); built as `sm_120a`. SM121 / GB10 is **out of scope** for this build (see Limitations) |
| Native FP8 / FP4 | FP8 e4m3 block-128 (tcgen05), NVFP4 W4A4 |
| CUDA toolkit | 13.0 |
| Driver | 595.84 |
| PyTorch | 2.14.0+cu130 |
| Build target | `-DGPU_ARCH=120` |
| Attention | repository-native SM120 FA2 tile (`flash_rt_fa2.fwd_bf16_tile`) for denoise + ViT; memory-efficient SDPA available as the path probe |

## Dispatch

`flash_rt/hardware/__init__.py` registers:

```python
("hyvla", "torch", "rtx_sm120") -> (
    "flash_rt.frontends.torch.hyvla_rtx",
    "HyVLATorchFrontendRtx",
)
```

`detect_arch()` already returns `"rtx_sm120"` for cc 12.0 and 12.1, so
`flash_rt.load_model(ckpt, config="hyvla", framework="torch")` resolves on the
target without a hardware argument.

## Files

| File | Purpose |
|---|---|
| `flash_rt/frontends/torch/hyvla_rtx.py` | SM120 frontend; inherits the Orin/Thor tokenizer/preprocess/prefix/graph orchestration, binds `_PIPE_CLS = HyVLARTXBF16Pipeline`, selects the precision tier from the named kwargs, and quantizes the expert/VLM/ViT weights. |
| `flash_rt/models/hyvla/pipeline_rtx.py` | SM120 lowered execution plan (`HyVLARTXBF16Pipeline`): block-128 FP8 GEMMs, NVFP4 producers, sm120 FA2 attention, head-dim-72 ViT path, im2col patch embed, plus the SM120 scheduling overrides (`_block`/`prefill`/`merger_forward` fused-residual path). Subclasses `HyVLAOrinBF16Pipeline`; `pipeline_thor.py` / `pipeline_orin.py` stay at upstream (no hardware forks in the shared files). |
| `csrc/kernels/hyvla_prefill_attn.cu`, `hyvla_euler.cu` | fused attention-prep / Euler kernels used by the SM120 plan (gated by `FLASHRT_ENABLE_HYVLA`). |
| `csrc/kernels/act_bf16.cu` | shared `silu_bf16` / `gelu_erf_bf16` elementwise activations (model-agnostic, compiled unconditionally). |
| `csrc/gemm/fp4_w4a4_mma_cksplit_sm120.cu` | NVFP4 W4A4 split-K MMA for the expert o/dn GEMMs. |
| `csrc/gemm/cutlass_sm120_block128_fp8_gemm_bias_sm120.cu` | **additive** sibling of the shared block-128 FP8 GEMM with a per-column bias epilogue (own TU; the upstream GEMM is unmodified). |
| `csrc/gemm/fp8_smallM_splitk_block128_sm120.cu` | **additive** block-128-scaled split-K FP8 GEMM (own TU; the upstream split-K is unmodified). |
| `csrc/kernels/hyvla_fused_thor.cu` | upstream base kernel unchanged; **appends** `hyvla_rope_qknorm_kvwrite_parallel_bf16` (bit-exact parallel-head grid) and `..._qb_bf16` (FA2 denoise Q packing) used by the SM120 plan. |
| `csrc/attention/fa2_tile_inst/flash_fwd_smallq_bf16_sm80.cu` | short-query FA2 tile (`fwd_bf16_tile`, head_dim ≤ 128): `<96,64,32,4>` / `<128,64,64,4>` instead of the vendored `<128,64,4>`; handles a runtime head_dim 72 via the `is_even_K=false` path. |
| `tests/test_rtx_hyvla05_e2e_check.py` | BF16 baseline vs default-tier fixed-noise smoke, eager reproducibility, input boundaries, and the optional real-frame 0.999 gate. |
| `tests/test_rtx_hyvla05_graphsafe.py` | graph-vs-eager and replay-stability gates. |
| `tests/test_rtx_hyvla05_arch_gate.py` | SM120 fail-fast gate (rejects SM121) and named-tier signature contract (mocked CUDA, no device). |
| `tests/test_rtx_hyvla05_dispatch.py` | dispatch resolution and pipeline binding. |
| `tests/test_rtx_hyvla05_routing.py` | `load_model` named-tier forwarding. |

## Precision policy

| Component | `use_fp4=False` (pure FP8) | `use_fp4=True` (default) |
|---|---|---|
| Embeddings / token assembly | BF16 | BF16 |
| HYViT2 patch embed | im2col bf16 matmul | im2col bf16 matmul |
| HYViT2 qkv / proj | FP8 block-128 | **NVFP4** |
| HYViT2 MLP fc1 / fc2 | FP8 block-128 | **NVFP4** (fused bias+GELU+quant) |
| MoT VLM prefill QKV/O | FP8 block-128 | **NVFP4** |
| MoT VLM prefill FFN | FP8 block-128 | **NVFP4** |
| Expert denoise QKV/O/FFN | FP8 block-128 (split-K o/dn) | FP8 block-128 if `use_fp4_expert=False`, else **NVFP4 W4A4** |
| RMSNorm / residual / RoPE / QK-Norm | fused BF16 kernels | fused BF16 kernels |
| Attention | sm120 FA2 (denoise), sm120 FA2 / efficient SDPA (ViT) | same |
| Action head / state / time | BF16, FP32 Euler update | same |

`use_fp4` is the master switch for the NVFP4 **ViT + prefill** tower;
`use_fp4_expert` independently promotes the expert denoise tower (default
`False`). With `use_fp4=False` no NVFP4 runs anywhere — the whole model is
block-128 FP8. The `load_model` default on sm120 is `use_fp8=True,
use_fp4=True, use_fp4_expert=False`.

Named constructor kwargs (mirroring the Thor/Orin tiers so `load_model`
forwards them):

```python
HyVLATorchFrontendRtx(ckpt, use_fp8=True, use_fp4=True,
                      use_fp4_expert=False)                  # default tier
HyVLATorchFrontendRtx(ckpt, use_fp8=True, use_fp4=True,
                      use_fp4_expert=True)                   # all NVFP4 (fastest)
HyVLATorchFrontendRtx(ckpt, use_fp8=True)                    # pure FP8
HyVLATorchFrontendRtx(ckpt, use_fp8=False, use_int8=False)   # BF16 reference
```

`use_fp8_block128` is a legacy alias for `use_fp8`. Orin SM87 INT8 is never
selected on sm120 (`use_int8` is forced off).

### Optimization levers (`HYVLA_*` opt-out environment variables)

These are internal, default-on performance levers. They are **not**
precision-tier selection (that is `use_fp4` / `use_fp4_expert`); the ViT/prefill
NVFP4 sub-levers listed here only apply inside `use_fp4=True`. Each can be
disabled with `=0` for A/B or debugging; most are read in one place
(`hyvla_rtx._flags()`).

| Variable | Default | Effect when disabled |
|---|---|---|
| `HYVLA_NATIVE_TEMPORAL` | 1 | bf16x2 vectorised ViT temporal-mix kernel -> torch composite |
| `HYVLA_VIT_SPACETIME_FUSED` | 1 | fused spacetime residual-add+LN -> torch path |
| `HYVLA_MERGER_NATIVE` | 1 | native merger pool/gate kernels -> torch path |
| `HYVLA_PREFIX_NATIVE` | 1 | native prefix assembly -> per-call tensor build |
| `HYVLA_FA2_PREPARE_Q` | 1 | fuse FA2 prepare_q into the RoPE kernel |
| `HYVLA_FA2_GATHER_QUANT_O` | 1 | fuse FA2 gather + FP8 block-128 quant on the o path |
| `HYVLA_EXPERT_OD_SPLITK_B128` | 1 | block-128 split-K for the M=41 expert o/dn GEMMs |
| `HYVLA_EXPERT_OD_FP8` | 0 (opt-in) | keep expert o/dn on FP8 inside the `use_fp4` tier |
| `HYVLA_VIT_HD72` | 1 | native head-dim 72 (no 72->96 pad) -> padded 96 path |
| `HYVLA_VIT_PROJ_NVFP4` / `HYVLA_VIT_QKV_NVFP4` | 1 | ViT proj / qkv NVFP4 (only under `use_fp4`) |
| `HYVLA_VIT_PATCH_GEMM` | 1 | im2col bf16 patch embed -> cudnn conv |
| `HYVLA_VIT_FC2_NVFP4`, `HYVLA_VIT_FC1_GELU_FUSE` | 1 | fused NVFP4 fc2 producer / fc1 bias+GELU+quant epilogue (only under `use_fp4`) |
| `HYVLA_PREFILL_QKVO_NVFP4` | 1 | prefill QKV/O NVFP4 (only under `use_fp4`) |
| `HYVLA_VIT_FC2_AWQ` | unset | path to a SmoothQuant calibration JSON for the ViT fc2 NVFP4 weights |
| `FLASHRT_HYVLA_FORCE_ARCH` | unset | documented dev override of the SM120 probe |

## Measured gates

Measured on the downloaded checkpoint, six real RoboTwin history frames,
prompt `pick up the bottle`; cosine vs the **official eager** raw action
(`dims[:20]`). Latency is wall-clock around `predict_actions`, warmup 5 +
median of 20.

| Config (kwargs) | Action cosine vs official eager | E2E graph |
|---|---:|---:|
| BF16 baseline | 0.999990 | 134.5 ms |
| **`use_fp8=True, use_fp4=True, use_fp4_expert=False` (default)** | **0.999768** | **55.6 ms** |
| `use_fp8=True, use_fp4=True, use_fp4_expert=True` (all NVFP4) | 0.998620 | 52.3 ms |
| `use_fp8=True` (pure FP8) | 0.999942 | 92.4 ms |

The default NVFP4 ViT+prefill tier clears the repository **0.999** gate
(0.999768) at ~40% lower latency than pure FP8; the pure-FP8 tier clears it
with more margin (0.999942); the all-NVFP4 tier is fastest but below the gate
(opt-in). The official eager reference anchor is ~600 ms (single 595.6;
three-run 603.3 / 621.6 / 578.1), so the tiers are ~10.8x / ~11.5x / ~6.5x
faster end-to-end.

Fixed-noise synthetic gate (`seed-0` images/state, `RandomState(0)` noise):

| Quantity | Pure FP8 block-128 |
|---|---:|
| graph vs eager cosine | 0.999987 |
| Replay cosine (two replays, static inputs) | >= 0.9999 |
| BF16 vs tier cosine (synthetic smoke) | >= 0.95 (asserted) |

The synthetic BF16-vs-tier number is a smoke signal only: uniform-random
inputs do not lie on the action manifold, so the quantized tier's cosine is not
distribution-representative (see `VERIFIER.md` and `docs/calibration.md`). The
distribution-level gate is the real-frame number above (0.999768 for the
default tier; 0.999942 for pure FP8). The e2e
test asserts >= 0.999 on a recorded real-frame fixture when
`HYVLA_RTX_PARITY_FIXTURE` points at one, and a >= 0.95 smoke otherwise.

### Reproducing the measured gates

The table above was produced with the following caliber. No fixture is shipped
(the real-frame `.npz` is captured locally; `*.npz` is git-ignored), and the
snippet below is self-contained:

- **Input**: six real RoboTwin history frames `(3, 6, 3, 240, 320)`, a fixed
  `state` `(1, 20)`, and one fixed shared flow noise `(1, 40, 32)`.
- **Cosine**: vs the official eager raw action chunk, over `dims[:20]`.
- **Latency**: wall-clock around `predict_actions` with `torch.cuda.synchronize()`
  before/after, warmup 5 + median of 20. FlashRT is timed on the delivered
  CUDA-graph path; the official model is eager only (it cannot be captured).

```python
# Per-tier cosine + E2E graph latency (load one ~9 GB frontend at a time).
import time
import numpy as np
import torch
from flash_rt.frontends.torch.hyvla_rtx import HyVLATorchFrontendRtx

CKPT = "/path/to/Hy-Embodied-0.5-VLA-RoboTwin"
d = np.load("real_hist.npz")                       # images/state/noise/raw
images = torch.as_tensor(d["images"], dtype=torch.float32)
state = torch.as_tensor(d["state"], dtype=torch.float32)
noise = torch.as_tensor(d["noise"], dtype=torch.float32)
raw = d["raw"].reshape(d["raw"].shape[0], -1)


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def measure(label, **tier):
    fe = HyVLATorchFrontendRtx(CKPT, **tier)
    fe.set_prompt("pick up the bottle")
    with torch.no_grad():
        out = fe.predict_actions(images, state=state, noise=noise, use_graph=True)
    b = np.asarray(out).reshape(out.shape[0], out.shape[1], -1)[0]
    n = min(raw.shape[1], b.shape[1])
    with torch.no_grad():
        for _ in range(5):                         # warmup
            fe.predict_actions(images, state=state, noise=noise, use_graph=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fe.predict_actions(images, state=state, noise=noise, use_graph=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"{label:16s} cos={cos(raw[:, :n], b[:, :n]):.6f} "
          f"graph median={np.median(ts):.2f} ms")


measure("BF16", use_fp8=False, use_int8=False, use_fused=True)
measure("FP8", use_fp8=True)
measure("default(V4NV)", use_fp8=True, use_fp4=True, use_fp4_expert=False)
measure("all-NVFP4", use_fp8=True, use_fp4=True, use_fp4_expert=True)
```

**Ground truth (official eager).** Feed ~30 sequential real RoboTwin frames to
`HyVLAPolicyWrapper` (`img_history_size=6`, `img_history_interval=5`); at a step
where the 6-frame history is saturated, save `observation.images.*` as the
`(3, 6, 3, 240, 320)` stack, plus `state` and the raw action chunk. Pass an
explicit `noise=noise.clone()` on every official forward: the official eager
path mutates the passed noise tensor in place, so reusing one tensor corrupts
the reference (observed as a spurious ~0.95 cosine). Time it the same way
(eager, warmup 5 + median of 20).

## Graph safety and determinism

The default tier is graph-safe (graph-vs-eager cosine >= 0.9999) but **not
bit-reproducible**: the expert o/dn GEMMs use split-K with atomic accumulation,
so two replays on identical static inputs differ at the ULP level. The graph
gate is therefore cosine-based, consistent with the fused-vs-unfused tolerance
used for the Orin path. Bit-exact replay is not an invariant of this target.

## Build

```bash
cmake -S . -B build_sm120 -G Ninja \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUTLASS_DIR="$CUTLASS_DIR" \
  -DPython3_EXECUTABLE=/usr/bin/python3.12 \
  -DGPU_ARCH=120 \
  -DFLASHRT_ENABLE_HYVLA=ON \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_DTYPES='fp16;bf16' \
  -DFA2_HDIMS='64;96;128;256' \
  -DFLASHRT_BUILD_FA2_PYTHON_ADAPTER=ON
cmake --build build_sm120 -j1        # the CUDA link needs ~15 GB RAM per job
```

## Verification

Hardware fail-fast, dispatch and routing gates run without a device or
checkpoint:

```bash
PYTHONPATH=. python -m pytest \
  tests/test_rtx_hyvla05_arch_gate.py \
  tests/test_rtx_hyvla05_dispatch.py \
  tests/test_rtx_hyvla05_routing.py -v
```

Checkpoint-gated precision and graph-safety gates (skipped automatically when
`FLASHRT_HYVLA_CHECKPOINT` is unset): run each module in a separate process so
only one ~9 GB weight copy is resident.

```bash
FLASHRT_HYVLA_CHECKPOINT=/path/to/Hy-Embodied-0.5-VLA-RoboTwin \
  PYTHONPATH=. python -m pytest tests/test_rtx_hyvla05_graphsafe.py -v
FLASHRT_HYVLA_CHECKPOINT=/path/to/Hy-Embodied-0.5-VLA-RoboTwin \
  PYTHONPATH=. python -m pytest tests/test_rtx_hyvla05_e2e_check.py -v
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and ensure the GPU is
otherwise idle (the default tier needs ~9 GB of a 16 GB card).

## Assets

Official model repository: see the Hy-Embodied-0.5-VLA project page on
Hugging Face. Checkpoint target:

```bash
/path/to/checkpoint/Hy-Embodied-0.5-VLA-RoboTwin
```

Direct Hugging Face access may be reset; use `HF_ENDPOINT=https://hf-mirror.com`
when needed.

## Current caveats

- The sm120 default tier keeps the expert denoise tower on FP8 block-128,
  which is what clears the 0.999 gate (0.999768). `use_fp4_expert=True`
  promotes the expert to NVFP4 for the fastest tier (~0.9986, below the gate)
  and is opt-in; the two tiers are otherwise identical.
- The SM120 expert o/dn split-K GEMMs are not bit-reproducible (atomic
  accumulation); precision is bounded by cosine, not exactness.
- The prefix / mask / RoPE tables are cached per `(prompt, num_cam)` in the
  shared frontend, so the first call per prompt pays the build cost.
- Only **SM120** (RTX 5060 Ti) has been validated end-to-end. SM121 / GB10
  (capability 12.1) is **out of scope** for this backend: the shipped build is
  `sm_120a` (architecture-specific), which does not run on SM121, and there is
  no base PTX fallback. The dispatcher rejects capability 12.1 up front.
  Supporting SM121 requires a separate `-DGPU_ARCH=121` build (`sm_121a`) plus
  its own validation.
- The `load_model` default on sm120 is `use_fp8=True, use_fp4=True,
  use_fp4_expert=False`; pass `use_fp8=False` for the BF16 reference path or
  `use_fp4_expert=True` (via the frontend constructor) for the all-NVFP4 tier.
