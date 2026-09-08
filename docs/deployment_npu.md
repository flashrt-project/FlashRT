# FlashRT on Ascend NPU (910/A2 via torch_npu / CANN)

Model-independent guide to the Ascend NPU backend: supported hardware,
routing, layout, execution model and how to run the tests.

Per-model guides:

| Model | Guide | Status |
|---|---|---|
| Pi0.5 | this file | BF16 captured-graph E2E, ~0.086 s median on 910B-class (optimized path) |

## Supported hardware

**Ascend 910 / A2 generation (Atlas 800I A2 and friends) with CANN 7/8
and torch_npu.** Detected by `flash_rt.hardware.detect_arch()` returning
`"npu"` whenever `torch.npu.is_available()` — the check runs before the
CUDA/ROCm branches because on a CANN box `torch.cuda.is_available()` is
False.

- **No FP8 tier.** 910/A2 parts have no FP8 tensor hardware; `load_model`
  coerces the default `use_fp8=True` to the **BF16** tier with a warning
  (see `flash_rt/api.py`, the `arch == "npu"` branch).
- No C++ extension is built or required: the backend is plain torch ops
  on torch_npu (aclnn) + `torch.npu.graph` capture.

## Why capture is the point (910B4, 2 views, 10 steps, BF16)

| Path | Median | Note |
|---|---|---|
| torch_npu eager (python → aclnn per op) | ~0.45–0.56 s | host launch/alloc bound (13k allocs, ~2.7k ops) |
| NPU graph capture, reference op sequence | ~0.15–0.20 s | graph replay; bit-identical to eager |
| **NPU graph capture, optimized serving path** | **~0.086 s** | fused norms / rope-table / prompt-flash attention |
| device GEMM roofline at these shapes | ~0.18 ms/big GEMM (≈204 TFLOPS bf16) | compute is not the bottleneck in eager |

The optimized serving path is `flash_rt/npu/models/pi05/fast.py` (reference
math in `pipeline.py` stays untouched — it is the fp32 golden source).
`fast.py` reuses CANN/torch_npu fused ops (`npu_rms_norm`,
`npu_add_rms_norm`, `npu_add_layer_norm`, `npu_prompt_flash_attention`),
precomputes per-step AdaRMS styles and RoPE cos/sin tables outside the
graph, and keeps K/V in preallocated buffers. Every op was admitted by the
isolated capture probe first (a failed capture aborts the process —
verified for `F.scaled_dot_product_attention`; `npu_layer_norm_eval` and
`conv2d` are aclop ops and cannot be captured at all).

Ascend's single-op Python/aclnn launch (~0.1–0.6 ms) is ~5–10× a CUDA
launch; graph capture removes that host churn and the fused-op reuse cuts
per-graph node scheduling further. Reaching the AMD-native ~30 ms BF16
tier additionally requires custom Ascend C glue kernels for the fusions
the official library does not ship — see `docs/pi05_npu_l3_plan.md`.

## Layout

```
flash_rt/npu/
  __init__.py             package doc
  core/device.py          ensure_npu / streams / synchronize
  core/npu_graph.py       NpuGraph over torch.npu.NPUGraph
  hardware/__init__.py    attention-backend notes (part-specific via filenames)
  models/pi05/pipeline.py Pi05 pipeline: fp32 eager reference + torch_npu
                          ops; conv patch embed is reshape/im2col (conv2d is
                          an aclop op and cannot be captured)
  models/pi05/fast.py     Optimized serving path (fused norms, rope tables,
                          prompt-flash attention; reference math untouched)
  frontends/torch/pi05.py Pi05TorchFrontendNpu (per-prompt-length captured
                          runner, public VLA API)
```

Routing touchpoints: `flash_rt/hardware/__init__.py` (`detect_arch`, the
`("pi05", "torch", "npu")` entry), `flash_rt/api.py` (extension-less NPU
gate + FP8→BF16 coercion).

## Capture rules the backend lives by

- All device tensors are fixed-address buffers; per-frame inputs are
  `copy_`d in, never reallocated.
- Anything that copies host→device during capture is forbidden: RoPE
  `pos`/`inv` are built on-device, time conditioning (`conds`) is
  precomputed per step outside the graph, prompt embeddings are computed
  into a static buffer at `set_prompt`.
- `torch.npu.graph` cannot capture aclop operators (e.g. `conv2d`) —
  avoid them on the hot path.
- Capture is per prompt length (AMD `"exact"` semantics). Rebuild when
  the length changes; reuse the cached runner otherwise.

## Environment knobs

| Env | Meaning |
|---|---|
| `FLASH_RT_NPU_PI05_CKPT` | pi05 checkpoint dir for `tests/test_npu_pi05_model.py` |

## Run

```bash
python -m pytest tests/test_npu_arch_gate.py tests/test_npu_pi05_model.py -q

python examples/pi05_amd_quickstart.py --checkpoint <ckpt>  # NPU variant:
python - <<'PY'
import flash_rt, numpy as np
m = flash_rt.load_model("<ckpt>", config="pi05", hardware="npu", num_views=2)
img = np.random.randint(0,255,(224,224,3),dtype=np.uint8)
fe = m.pipeline
fe.set_prompt("pick up the black bowl")
a = fe.infer({"image": img, "wrist_image": img})["actions"]   # (10,7)
print(a)
PY
```

## Verification mechanism (gates every fusion change must hold)

Reference spaces (both from the same canonical recipe: image seed 0,
noise seed 1, prompt `"pick up the cup"`, 2 views, 10 steps):

- **raw (10, 32)** — the model's *normalized* action output; most sensitive
  space for numerical gates.
- robot (10, 7) — unnormalized real actions behind `norm_stats`.

Ladder (each rung must pass before the next is meaningful):

1. **Capture validity**: `torch.npu.graph` capture succeeds and replay of
   the same buffers is bit-identical (no per-op host dispatch creeps back).
2. **Eager parity**: captured replay equals eager execution of the *same*
   op sequence (capture must not change numerics).
3. **fp32 reference gate**: normalized raw actions cosine ≥ **0.9999**
   against the golden file (fp32 CPU eager, pinned noise). A real regression
   lands at ~1e-3 cosine, far below the bar.
4. **Latency tripwire**: captured full-frame median < 350 ms (optimized
   path runs ~86 ms on 910B4; 350 ms catches anything that re-enables eager
   dispatch or drops back to a reference op sequence); real A/B numbers come
   from the paired-median protocol, never a single arm.

Artifacts / commands:

```bash
# one-off, slow (fp32 CPU full pass): produce the golden reference
python scripts/npu/make_pi05_reference.py <checkpoint_dir> ref.npz

# model gates (checkpoint + reference env-driven)
FLASH_RT_NPU_PI05_REF=ref.npz python -m pytest \
    tests/test_npu_pi05_model.py tests/test_npu_arch_gate.py -q
```

Helpers live in `flash_rt/npu/verify.py` (`canonical_images`,
`pinned_noise`, `cosine`, reference save/load). For fusion development the
protocol is: change op sequence → gate 1–2 (unchanged semantics, must be
bit-identical) or gate 3 (any reduction-order change) → paired-median
latency A/B vs the previous build.

## Status / not yet ported

- `state_prompt_mode="fixed"`, temporal K/V caching (`cache_frames>1`),
  RL/batched serving: raise `NotImplementedError`.
- FP8/FP4 tiers: not applicable (no FP8 hardware on 910/A2).
- Optimized path reaches ~86 ms BF16 (reuse of CANN fused ops). Custom
  Ascend C glue kernels for the fusions the official library lacks are the
  next milestone (see `docs/pi05_npu_l3_plan.md`); 910B-native INT8
  calibration would be the ~2× quantisation step beyond that.
