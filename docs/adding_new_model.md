# Adding a New Model: The Complete Guide

> **Target audience**: Engineers who need to integrate a new VLA model into FlashRT (e.g. Pi0.6, a fresh open-source VLA).
>
> **Time estimate**: A single `(framework, hardware)` combination runs around 800-1200 lines of code, or 1-2 weeks of work, assuming the model's structure is close to Pi0.5 / Pi0. All four combinations (torch/jax × thor/rtx) take roughly 3-4 weeks.
>
> **Read in this order** (don't skip ahead — each doc assumes the previous):
> 1. **This doc §0–§1** (you are here) — the repository contract and which files you'll touch
> 2. [`flash_rt/frontends/torch/_template/README.md`](../flash_rt/frontends/torch/_template/) — the **starter package**. Open in a separate window before reading further; the rest of this doc references it. The template has 4 stub files (frontend / pipeline / weights_spec / attention) you copy and fill in.
> 3. [`docs/stable_api.md`](stable_api.md) — public API surface and naming conventions you must respect
> 4. [`docs/calibration.md`](calibration.md) — how FP8 calibration works (read **§2 + §10** before writing your `_calibrate` twin)
> 5. [`docs/kernel_fusion.md`](kernel_fusion.md) — kernel naming + decision tree for which `fvk.*` to call where (skim §1 + §2; reference the rest as you write `pipeline.py`)
> 6. [`docs/plugin_model_template.md`](plugin_model_template.md) — only if you're shipping a closed-source model as an external plugin instead of a PR into this repo. Also contains the **first-light cosine routing table** (Q&A section) — the most useful debugging reference once your model first runs.

**Suggested first-week schedule** for an ML-infra engineer with the source model already running in PyTorch:
- Day 1 — read items 1–3, copy the template, list every weight tensor in your checkpoint
- Day 2–3 — fill in `weights_spec.py` (declarative, mostly mechanical) and `attention.py` (~60 lines)
- Day 4–5 — write `pipeline.py` (the bulk of the work; `encoder_forward` first, leave `decoder_forward` for day 6)
- Day 6–7 — wire up `frontend.py`, run first inference, debug cosine using the routing table

---

## 0. Repository Contract (hard rules every new model must follow)

```
Mandatory rules:

1. Every (model, hardware) compute path lives in its own file:
     models/<model>/pipeline_<hw>.py
   — The suffix is required (_thor / _rtx). There is no default
     pipeline.py entry point.
   — No runtime hardware forks such as `if self._has_sm100` or
     `if arch == 'thor'`.
   — Even if two hardware implementations are 90% identical, they must
     still be split. Shared logic is reused through function calls or
     imports, not through in-file branching.

2. Every (model, framework, hardware) IO path = its own frontend:
     frontends/<framework>/<model>_<hw>.py
     class name: <Model><Fw>Frontend<Hw>
   — Example: frontends/torch/pi05_rtx.py contains Pi05TorchFrontendRtx
   — Same rule: split thor and rtx frontends even when they duplicate
     most of their code.

3. hardware/<hw>/shared_primitives.py is a closed set:
   — Only model-agnostic helpers belong here:
       _gpu_* helpers, _measure_scale_gpu,
       siglip_forward        (usable by any model with a SigLIP tower),
       encoder_forward       (usable by any Paligemma-encoder model),
       encoder_forward_calibrate
   — Model-specific forward / decoder functions are not allowed in this
     file. They go into models/<m>/pipeline_<hw>.py.

4. _PIPELINE_MAP is strictly one-to-one:
     ("model", "framework", "hw") -> ("flash_rt.frontends.<fw>.<m>_<hw>",
                                      "ClassName")
   Each tuple points to exactly one file and one class. Multiple tuples
   sharing a class (i.e. runtime forking) is forbidden.

Known historical exception (do NOT copy this pattern):
  pi0fast: frontends/torch/pi0fast.py is a single file with 14 runtime
           `_has_sm100` branches. Deprecated layout — pending split into
           per-hardware files. New models must follow rules 1-4 above.
```

---

## 1. Overview: which files you will touch

Walking through a hypothetical `pi06` model (Paligemma backbone) that needs to support both Thor and RTX, under both torch and jax:

```
flash_rt/
├── hardware/__init__.py                  # 4 new lines in _PIPELINE_MAP
├── hardware/thor/attn_backend.py         # add make_pi06_attention_spec (if shapes change)
├── hardware/rtx/attn_backend.py          # (RTX) same
├── models/pi06/
│   ├── pipeline_thor.py                  # NEW — Thor forward functions
│   └── pipeline_rtx.py                   # NEW — RTX Pi06Pipeline class
├── frontends/torch/
│   ├── _pi06_thor_spec.py                # NEW — Thor torch WEIGHT_SPEC
│   ├── _pi06_rtx_spec.py                 # NEW — RTX torch WEIGHT_SPEC (if dims / iface differ)
│   ├── pi06_thor.py                      # NEW — Thor torch frontend
│   └── pi06_rtx.py                       # NEW — RTX torch frontend
├── frontends/jax/
│   ├── _pi06_thor_spec.py                # NEW
│   ├── _pi06_rtx_spec.py                 # NEW
│   ├── pi06_thor.py                      # NEW
│   └── pi06_rtx.py                       # NEW
├── configs/pi06.yaml                     # metadata
└── tests/test_all_models_precision.py    # add one segment
```

All four combinations together: **~3500-4500 lines**.
A single `(framework, hardware)` combination: **~800-1200 lines** (of which ~120 lines are declarative spec).

Reference implementations:
- **pi05** — all four combinations complete: `models/pi05/{pipeline_thor.py, pipeline_rtx.py}` plus four frontends
- **pi0** — Thor is done, RTX is being refactored in stage 8
- **groot** — Thor and RTX are done (jax only on Thor)

---

## 1.5. Working from the template

Before reading §2, **copy the starter template**:

```bash
# For a new model called "mymodel" on Thor:
cp -r flash_rt/frontends/torch/_template /tmp/mymodel_thor_work
cd /tmp/mymodel_thor_work
$EDITOR README.md   # 5-min read; explains the file split
```

Then work file-by-file in this order (each file's docstring tells you exactly what to translate from your source model):

1. **`weights_spec.py`** → `flash_rt/frontends/torch/_<mymodel>_thor_spec.py`
   The declarative weight loader. Map each `state_dict[...]` key from your reference checkpoint to a `WEIGHT_SPEC` row. Pure mechanical work; ~60-120 lines for a Pi0.5-shape model.

2. **`attention.py`** → append `make_<mymodel>_attention_spec()` to `flash_rt/hardware/thor/attn_backend.py`
   ~60 lines. Declares one `add_site()` call per distinct attention shape in your model (vision, encoder, decoder, etc.).

3. **`pipeline.py`** → `flash_rt/models/<mymodel>/pipeline_thor.py`
   **The hard part.** Translate your reference model's `forward()` into a sequence of `fvk.*` kernel calls. The template's `# WHAT YOU TRANSLATE` block at the top shows the line-by-line mapping pattern. You'll write two functions per stage: a production forward (FP8, captured into CUDA Graph) and a calibration twin (BF16 + measures activation amax). 200-400 lines per hardware target.

4. **`frontend.py`** → `flash_rt/frontends/torch/<mymodel>_thor.py`
   Wires it all together. Owns weight upload, buffer allocation, calibration cache, and CUDA Graph capture. Class name must be `<Model>TorchFrontendThor` per §0 rule 2. ~400-700 lines.

After all four files compile and your first `infer()` produces non-NaN output, run cosine vs your PyTorch FP32 reference. Use the **first-light cosine routing table** in [`plugin_model_template.md`](plugin_model_template.md) to narrow down where to look — that table maps the cos number you see directly to the most likely root cause.

For RTX, repeat with `pipeline_rtx.py` / `<mymodel>_rtx.py`. For JAX, the template covers torch only — copy from `frontends/jax/pi05_thor.py` for the JAX patterns (Orbax loading, weight cache, etc.).

---

## 2. Step-by-step walkthrough

§2 below provides the **detailed reference** for each step the template guides you through. Use it as a lookup, not a tutorial — you should already have copied the template before reading further.

### (1) AttentionSpec — 15-35 lines

File: [`flash_rt/hardware/thor/attn_backend.py`](../flash_rt/hardware/thor/attn_backend.py) (Thor) or [`flash_rt/hardware/rtx/attn_backend.py`](../flash_rt/hardware/rtx/attn_backend.py) (RTX).

Copy [`make_pi05_attention_spec`](../flash_rt/hardware/thor/attn_backend.py) and adjust:

```python
def make_pi06_attention_spec(num_views: int, *,
                              enc_total_keys: int, dec_total_keys: int) -> AttentionSpec:
    """Pi0.6: 24L encoder / 24L decoder / H_dim=256 / GQA 8:1."""
    S_sig = num_views * 256
    NH_sig, HD_sig = 16, 72              # SigLIP 1152/16
    NH_enc, HD_enc = 8, 256              # Paligemma 2048/8
    NH_dec, HD_dec = 8, 256              # Action expert

    return AttentionSpec(
        sites=[
            SiteSpec(
                name="siglip", layer_count=27, q_seq_len=S_sig, kv_seq_len=S_sig,
                num_heads=NH_sig, head_dim=HD_sig,
                extra={"kernel": None},  # fmha_strided_full dispatcher
            ),
            SiteSpec(
                name="encoder", layer_count=24, q_seq_len=...,  # Se filled at runtime
                kv_seq_len=enc_total_keys,
                num_heads=NH_enc, head_dim=HD_enc, num_kv_heads=1,
                extra={"kernel": "standard"},
            ),
            SiteSpec(
                name="decoder", layer_count=24, q_seq_len=10,
                kv_seq_len=dec_total_keys,
                num_heads=NH_dec, head_dim=HD_dec, num_kv_heads=1,
                extra={"kernel": "standard"},
            ),
        ],
    )
```

**Allowed values for `extra["kernel"]`** (see [`backend.py:AttentionBackend`](../flash_rt/hardware/backend.py) for the full table):

| kernel value | underlying fvk primitive | used for |
|----------|--------------|------|
| `None` (siglip only) | `fmha_strided_full` | SigLIP 2D vision attention |
| `"standard"` | `attention_qkv_fp16` | GQA encoder/decoder, no state padding |
| `"state_masked"` | `attention_qkv_fp16_state_masked` | Pi0 decoder (the first token is state) |
| `"mha"` | `attention_mha_fp16` | GROOT Qwen3 full-MHA plus DiT self/cross |

Do not invent new kernel values. If you need a new variant, extend the dispatch branches in [`hardware/thor/attn_backend.py:ThorFlashAttnBackend.run`](../flash_rt/hardware/thor/attn_backend.py).

---

### (2) Pipeline forward — 200-400 lines per hardware; **the bulk of the hand-written code**

Files:
- `flash_rt/models/pi06/pipeline_thor.py` (Thor path)
- `flash_rt/models/pi06/pipeline_rtx.py` (RTX path)

**Each hardware gets its own file, even if the two paths end up looking similar.** Genuinely shared code lives in `hardware/<hw>/shared_primitives.py` (reserved for truly model-agnostic helpers) or is imported between sibling functions.

Recent references to copy from:
- Thor: [`models/pi0/pipeline_thor.py`](../flash_rt/models/pi0/pipeline_thor.py) — Pi0 decoder forward
- Thor: [`models/pi05/pipeline_thor.py`](../flash_rt/models/pi05/pipeline_thor.py) — Pi0.5 `postln_project` / `decoder_forward` / `decoder_forward_calibrate`
- RTX: [`models/pi05/pipeline_rtx.py`](../flash_rt/models/pi05/pipeline_rtx.py) — the `Pi05Pipeline` class (framework-agnostic, takes AttnBackend via injection)
- RTX: [`models/groot/pipeline_rtx.py`](../flash_rt/models/groot/pipeline_rtx.py) — GROOT's three-graph flow

Every forward function must obey the **pointer-interface contract**:

```python
def decoder_forward_pi06(
    gemm: fvk.GemmRunner,
    fvk_module,                    # flash_rt_kernels
    bufs: dict,                    # {'x': ptr, 'xn': ptr, ...}
    weights: dict,                 # {'qw': ptr, 'ow': ptr, 'gu': ptr, 'd': ptr, ...}
    dims: dict,                    # {'S': 10, 'Da': 1024, 'Ha': 4096, 'La': 24, ...}
    scales_dev: int,               # device pointer to fp32 scale array
    *,
    attn=None,                     # AttentionBackend; None = legacy fallback
    stream: int = 0,
):
    """Every argument is a raw pointer or a Python primitive that ctypes can pass
    through — this is what makes the function safe to call repeatedly during
    CUDA Graph capture."""
    ...
```

**Why this interface**: CUDA Graph capture requires the same Python function, calling the same sequence of kernels, with the same pointers, on every replay. Tensor objects can be garbage-collected or reallocated between replays; raw `.data_ptr()` values cannot.

**Catalog of kernels available for building forwards**: [`docs/kernel_fusion.md`](kernel_fusion.md) lists all 93 public fvk functions and the legal fusion patterns.

**Key rules**:
- All intermediate buffers must be **pre-allocated** in `frontend._load_weights`. The forward only reads pointers — **no dynamic allocation.**
- Never call `.cpu()`, `.numpy()`, `torch.empty()`, or `sync()` inside a forward.
- Attention goes through `attn.run(site=..., layer_idx=i, ...)`. Do not call `fvk.attention_qkv_fp16(...)` directly.
- Full rule set: [`docs/kernel_fusion.md` §5 known-failed optimizations](kernel_fusion.md#failed-optimizations)

---

### (3) Torch WEIGHT_SPEC — 60-120 lines per `(framework, hardware)` combo, **declarative**

Files:
- `flash_rt/frontends/torch/_pi06_thor_spec.py`
- `flash_rt/frontends/torch/_pi06_rtx_spec.py` (only if dims or weight interface differ)

When the two hardwares share the exact same weight interface (common — both sides read the same safetensors checkpoint), a single spec file can back both frontends via `from ._pi06_thor_spec import build_spec`. **The default is still one spec file per hardware**, to avoid a future dim change on one side accidentally regressing the other.

If the backbone is in the Paligemma / Gemma family (very likely):

```python
from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    FlatCat, FusedGateUp, FusedQKV, Quant, TensorList, ToFp16, tT,
)
from flash_rt.frontends.torch._thor_spec_common import (
    paligemma_encoder_block, paligemma_siglip_block,
)


def _decoder_block():
    dp = "paligemma_with_expert.gemma_expert.model.layers.{i}"
    return LayerBlock(
        prefix_fmt="", num_layers=24, name="decoder",
        items=[
            Item("qkv_w",
                 FusedQKV(q=f"{dp}.self_attn.q_proj.weight",
                          k=f"{dp}.self_attn.k_proj.weight",
                          v=f"{dp}.self_attn.v_proj.weight",
                          norm_fuse=f"{dp}.input_layernorm.weight",
                          interleave_q_heads=8,
                          interleave_k_heads=1),
                 [tT(), Quant()],
                 FlatCat("_dec_qkv_flat"), scale_into="_ae_w_scales"),
            # ... follow the pattern in _pi0_thor_spec.py
        ],
    )


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="torch",
        blocks=[
            paligemma_siglip_block(),
            paligemma_encoder_block(num_layers=24),
            _decoder_block(),
        ],
    )
```

If the backbone is a **novel architecture** (Qwen3, etc.): look at [`frontends/torch/groot_thor.py::_load_qwen3_weights`](../flash_rt/frontends/torch/groot_thor.py), which is still a hand-written loop rather than a declarative spec. You will likely need to either:
- add a new shared block builder to `_thor_spec_common.py`, or
- define a new composite source (something like `FusedQKV`) — see `flash_rt/executors/torch_weights.py`.

**Op order must be byte-identical**: compare your spec, op by op, against the legacy hand-written loader — `.T.contiguous()` vs `.t().contiguous()`, `ToFp16` before or after `Quant`, exactly when `norm_fuse` is applied. A single character wrong causes precision regressions.

---

### (4) Frontend — 700-1000 lines per frontend; ~2800-4000 lines across all four

Files:
- `flash_rt/frontends/torch/pi06_thor.py`  (class: `Pi06TorchFrontendThor`)
- `flash_rt/frontends/torch/pi06_rtx.py`   (class: `Pi06TorchFrontendRtx`)
- `flash_rt/frontends/jax/pi06_thor.py`    (class: `Pi06JaxFrontendThor`)
- `flash_rt/frontends/jax/pi06_rtx.py`     (class: `Pi06JaxFrontendRtx`)

**Class-name rule**: `<Model><Framework>Frontend<HW>` in CamelCase — e.g. `Pi05TorchFrontendThor`, `Pi05TorchFrontendRtx`, `GrootJaxFrontendThor`.

Skeleton: copy the nearest sibling (same framework, same hardware) and edit:

| What to change | Where | Lines |
|---|---|---|
| `__init__` | `num_views`, checkpoint path | a few |
| `_load_norm_stats` | new norm_stats path (if it moved) | 20 |
| `_load_weights` | call `_pi06_<hw>_spec.build_spec()`; update dim constants; update `_sig_weights` keys | 120 |
| `set_prompt` | tokenizer; time_mlp precompute; call `_calibrate` and `_capture_*_graph` | 100 |
| `_calibrate` | call `encoder_forward_calibrate` / `decoder_forward_calibrate` | 150 |
| `_capture_*_graph` | update dims; call `models/pi06/pipeline_<hw>.py::{encoder,decoder}_forward_pi06` | 135 |
| `_autotune_enc_ae` | copy unchanged | 50 |
| `infer` | input preprocessing / noise / action decode | 80 |
| `get_latency_stats` | copy unchanged | 15 |

**Things you must never do**:
- Allocate new tensors inside `infer` (violates the CUDA Graph contract).
- Change graph topology inside `_calibrate` (triggers Myelin tactic drift).
- Skip `.contiguous()` (column-major vs row-major layout bugs).
- **Detect hardware at runtime inside a frontend (`hasattr(fvk, ...)`) and branch on it** — this is the pi0fast anti-pattern. New models must ship two independent thor/rtx frontends.

---

### (5) JAX-side differences worth calling out

**Where the JAX side diverges from torch**:
- The source is `OrbaxDictSource(engine_w)`, where `engine_w` is the `dict[str, ndarray]` exported by openpi. The key names are **not** HF safetensors paths; they follow openpi's internal schema (e.g. `vision.layer.{i}.qkv.weight`). See [`_thor_spec_common.py`](../flash_rt/frontends/jax/_thor_spec_common.py).
- `engine_w` typically has QKV and gate_up **already fused** (openpi does this during export). So the spec does not need `FusedQKV` / `FusedGateUp` — plain `JaxQuant()` is enough.
- The sink is `CudaBufferFlat` / `CudaBufferAttr` plus an explicit `cache=...` argument (weight caching relies on it).
- The frontend must set `self._cache_blobs = {}` before calling `WeightLoader.run(...)`.

**Weight cache behavior**: the default is `weight_cache=True`. The first load takes ~30-40s; results are cached to `~/.flash_rt/weights/<hash>_nv<N>.bin`, and subsequent loads take ~5s. When modifying a spec you **must preserve the cache key schema** (`sig_wt_fp8.{0..11}`, etc.); otherwise the cache format changes and users have to wipe it manually.

---

### (6) `_PIPELINE_MAP` registration — 4 lines (per hardware × per framework)

File: [`flash_rt/hardware/__init__.py`](../flash_rt/hardware/__init__.py)

```python
_PIPELINE_MAP: dict[...] = {
    ...  # existing entries
    # ── Pi0.6 ──
    ("pi06", "torch", "thor"):
        ("flash_rt.frontends.torch.pi06_thor", "Pi06TorchFrontendThor"),
    ("pi06", "torch", "rtx_sm120"):
        ("flash_rt.frontends.torch.pi06_rtx",  "Pi06TorchFrontendRtx"),
    ("pi06", "jax", "thor"):
        ("flash_rt.frontends.jax.pi06_thor",   "Pi06JaxFrontendThor"),
    ("pi06", "jax", "rtx_sm120"):
        ("flash_rt.frontends.jax.pi06_rtx",    "Pi06JaxFrontendRtx"),
}
```

One entry, one class. Two entries pointing at the same class is the pi0fast anti-pattern.

Then confirm that `config="pi06"` is accepted in [`api.py::load_model`](../flash_rt/api.py) — the function validates configs near the top.

---

### (7) Config YAML — 30 lines

File: `flash_rt/configs/pi06.yaml`

Copy [`pi05.yaml`](../flash_rt/configs/pi05.yaml) as a starting point. Fields: `num_layers`, `hidden_dim`, `num_heads`, `head_dim`, `ffn_hidden_dim`, `num_views`, `action_horizon`, `vocab_size`, and so on.

This YAML is used only for logging and metadata. Runtime dimensions still come from the constants hard-coded inside the frontend.

---

### (8) Precision test — 80 lines

File: [`tests/test_all_models_precision.py`](../tests/test_all_models_precision.py)

1. Near the top add `PI06_SCRIPT = """..."""`: load the pipe, set a prompt, run 5 warmup iterations, patch the RNG, record 20 latency samples, compute cosine similarity against the pytorch reference.
2. Add `'pi06': ('Pi0.6', PI06_SCRIPT)` to the `_configs` dict.
3. Save the pytorch reference to `/tmp/pi06_pytorch_ref.npy`.

On the 5090 side, additionally add a pi06 segment to your local
smoke / cosine / latency benchmark scripts (you'll likely have your
own; the public test suite covers smoke and unit tests, latency
benchmarks are typically per-team).

---

## 3. Validation protocol — run on every commit

```bash
# CPU unit tests (seconds)
python tests/test_weight_loader.py            # 16/16
python tests/test_thor_attn_backend.py        # 12/12
python tests/test_thor_groot_attn_backend.py  # 11/11

# 5090 GPU validation (if you added an RTX path)
python examples/quickstart.py --checkpoint <ckpt> --config pi06 \
    --benchmark 200                            # smoke + latency
# Cosine: load the model, run predict() with matched_noise, compare
# against your PyTorch FP32 reference run on the same observation.

# Thor GPU precision sweep (~3-5 minutes)
free -h | head -2   # always check free memory before heavy Thor runs
python tests/test_all_models_precision.py --model pi06
```

**Thresholds**:
- First-time bring-up of a new model: cos ≥ 0.995 (vs pytorch ref), and P50 inside the target latency budget.
- Models structurally derived from Pi0.5 / Pi0: the "bit-identical" band (~0.9986) indicates the FP8 bytes match exactly.

**If cosine falls out of the window**:
1. Don't guess. First check the spec's op order byte-for-byte against the legacy loader.
2. Use an A/B comparison to rule out Myelin tactic jitter — run 2-3 times back-to-back.
3. If it really is a regression, revert the commit immediately. **Don't** try to patch it in a follow-up.

---

## 4. Thor-specific pitfalls (must read)

### 4.1 CUDA Graph non-determinism

Recompiling the same MLIR → Myelin picks a different tactic → ±2ms P50 drift and ~0.001 cos jitter. This is specific to Thor.

**Don't**:
- Draw conclusions from a single measurement (always A/B).
- "Fix" a ±0.001 jitter in a new commit (it's almost certainly tactic drift, not code).
- Compare latency numbers taken at different times.

**Do**:
- Use a timing cache to pin the tactic (though you cannot choose the optimal one directly).
- Keep a reference timing cache around (see `deployment/scripts/l2v2_timing_cache.bin`).

### 4.2 Don't run heavy tasks in parallel

Thor has 122Gi of unified memory. Loading two models concurrently will OOM. Tests must run serially.

### 4.3 Kernel binding and CMake ownership

Adding a new model should usually reuse existing `fvk.*` entries. If you do
need a new kernel or a new pybind entry, keep the CMake target ownership in
lockstep with the binding:

- An unconditional `m.def(...)` in `csrc/bindings.cpp` must link against an
  implementation that exists in every supported `GPU_ARCH` build, or it must
  call an unconditional stub that raises a clear "not built / not supported"
  error.
- If the `.cu` implementation is hardware- or feature-gated, the binding must
  be gated the same way. Do not let a public `flash_rt_kernels` symbol depend
  on a model-specific object target that only builds for one architecture.
- Shared quantize, layout, RoPE, activation, and utility kernels belong in
  the main `flash_rt_kernels` target unless every binding and every caller is
  gated with the same condition.
- Model-specific object libraries should contain only model- and
  hardware-specific kernels.

For any CMake / binding ownership change, validate every affected hardware
family:

```bash
cmake -B build_<arch> -S . -DGPU_ARCH=<arch>
cmake --build build_<arch> -j$(nproc) --target flash_rt_kernels
PYTHONPATH=. python - <<'PY'
from flash_rt import flash_rt_kernels
print(flash_rt_kernels.__file__)
PY
nm -D -u flash_rt/flash_rt_kernels*.so | c++filt | grep 'flash_rt::' || true
```

When moving a source out of a model-specific object target into
`flash_rt_kernels`, test both sides of the change: the architecture that was
missing the symbol must import cleanly, and the architecture that already had
the source must not fail with duplicate definitions.

### 4.4 Don't rebuild the kernel .so

`flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so` (3.6MB) is a production binary. Adding a new model should not trigger a kernel rebuild — every fvk function you need is already in this .so. If you genuinely need a new kernel, that's a separate CUDA development flow, with explicit version backups.

---

## 5. Time estimate (realistic)

Assuming the new model is structurally similar to Pi0.5 / Pi0 (Paligemma backbone, flow-matching decoder), for a **single `(framework, hardware)` combination**:

| Phase | Estimate |
|------|------|
| (1)(6)(7) Skeleton and registration | half a day |
| (2) Pipeline forward — forked from Pi0 with dim-constant edits | 1-2 days |
| (3) WEIGHT_SPEC authoring + byte-diff validation | 1 day |
| (4) Frontend — fork Pi0, edit dims / calibration / graph capture | 3-4 days |
| (8) Tests and debugging | 2-3 days |
| **Total per combination** | **~1-2 weeks** |

**All four combinations** (torch/jax × thor/rtx): roughly **3-4 weeks** — subsequent frontends reuse a lot of code.

If the backbone is a new architecture (Qwen3-like), add **1-2 more weeks** for shared-block extensions, kernel compatibility evaluation, and possibly a new attention variant.

---

## 6. Quick checklist

- [ ] (1) New AttentionSpec added to the correct hardware's `attn_backend.py`; unit tests pass.
- [ ] (2) Pipeline forward functions use the pointer-only interface, do no dynamic allocation, and **each hardware has its own `pipeline_<hw>.py` file**.
- [ ] (3) `_<model>_<hw>_spec.py` smoke-builds via `build_spec()`.
- [ ] (4) Frontend is fully implemented, **each `(framework, hardware)` has its own `<m>_<hw>.py` file**, and all buffers are pre-allocated in `_load_weights`.
- [ ] **No file uses `if self._has_sm100` or `hasattr(fvk, '...')` to branch on hardware.**
- [ ] **`shared_primitives.py` has not gained any model-specific functions.**
- [ ] Any new `csrc/bindings.cpp` entry and its `.cu` implementation have matching CMake guards / target ownership.
- [ ] If a kernel source moved between object targets and `flash_rt_kernels`, every affected `GPU_ARCH` builds `flash_rt_kernels`, imports it from Python, and has no missing or duplicate symbols.
- [ ] (6) The four `_PIPELINE_MAP` entries are one-to-one, with no two rows pointing at the same class.
- [ ] (7) YAML dims match the constants in the code.
- [ ] (8) `test_all_models_precision.py` passes three consecutive A/B runs.
- [ ] Weight-cache keys remain compatible with legacy (if the JAX spec changed).
- [ ] Commit format: `feat(<model>-<framework>-<hw>): ...`

---

## 7. FAQ

**Q: Why are runtime hardware forks like `if hasattr(fvk, 'cutlass_fp8_sq')` disallowed?**
A: Because of the lesson learned from pi0fast. A single file with many branches grows maintenance cost explosively: adding a new hardware means touching 14 spots; adding a new optimization means redoing it on every branch; stack traces no longer tell you which hardware path you were on; and CUDA Graphs capture different kernel sequences per hardware anyway, so `if` branching can't actually unify them. Splitting per hardware lets each file focus on exactly one execution path.

**Q: The thor and rtx frontends are 90% identical — wouldn't merging them save a lot of code?**
A: Short-term, yes. But "shared between two ends" means adding a third hardware requires splitting again, every change risks breaking the other side, and the test matrix becomes N×M. With per-hardware files, adding a new hardware is just adding a new file while the existing files stay stable. The total line count is slightly higher, but maintenance entropy is dramatically lower.

**Q: `KeyError: ...` at load time?**
A: Some key in your WEIGHT_SPEC doesn't exist in the checkpoint. Inspect the actual safetensors keys:
```bash
python -c "from safetensors import safe_open; sf=safe_open('/path/to/model.safetensors', 'pt'); [print(k) for k in list(sf.keys())[:50]]"
```

**Q: After loading, cosine is around 0.5?**
A: Likely causes: wrong QKV interleave (bad GQA head count), mixing `.T.contiguous()` with `.t().contiguous()`, or applying `norm_fuse` at the wrong point. Start with [`docs/calibration.md §4 precision journey`](calibration.md#precision-history).

**Q: CUDA Graph capture fails?**
A: Your forward contains a dynamic allocation or a conditional branch that causes capture to take a different kernel path. Details in [`kernel_fusion.md §6`](kernel_fusion.md#cuda-graph-rules).

**Q: JAX loading takes ~40s — too slow?**
A: That's the expected first-load cost. Keep `weight_cache=True` (the default); subsequent loads are ~5s. If you changed the WEIGHT_SPEC's cache key, you need `rm -rf ~/.flash_rt/weights/` so the cache can be rebuilt.

**Q: New model OOMs on Thor?**
A: Thor has 122Gi of unified memory. Check: (1) `free -h` shows free memory greater than model size × 1.5; (2) no other pipeline is running concurrently; (3) the previous `weight_cache` version has been cleaned up.
