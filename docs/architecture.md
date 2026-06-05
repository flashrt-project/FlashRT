# FlashRT Architecture

> **Target audience**: anyone trying to understand what FlashRT *is* as a system — not how to add a model (see [`adding_new_model.md`](adding_new_model.md)) or how a specific kernel works (see [`kernel_catalog.md`](kernel_catalog.md)). This doc names the eight infrastructure components that make up FlashRT, says where each one lives in the repo, and explains how they compose into a working inference engine.
>
> **TL;DR**: FlashRT core is **not** a compiler or graph rewriter. It is a **kernel library plus eight infrastructure components** wired together by a thin per-model pipeline. The repository also ships a `serving/` scenario-host layer on top of the execution contract for OpenAI-compatible LLM/audio endpoints and robot rollout hosts. Core mechanisms stay in `flash_rt/`, `exec/`, and `csrc/`; scenario policy stays in `serving/`.

---

## 1. Runtime layers

```
serving/   scenario hosts: sessions, schedulers, protocols, robot loops,
           OpenAI-compatible HTTP/SSE, streaming audio, rollout policy
             │
             ▼
flash_rt/   Python frontends: weights, calibration, CUDA Graph capture,
           model APIs and per-model pipeline composition
             │
             ▼
exec/       C ABI execution contract: Buffer / Graph / Plan, replay-time
           mechanism for native hosts
             │
             ▼
csrc/       CUDA/C++ kernels and vendored attention/GEMM building blocks
```

The serving layer is deliberately above the model runtime. It decides
session policy, streaming protocol, episode lifecycle, interrupt/reset
behavior, and host language. The core runtime owns the graph and kernel
mechanism. See [`../serving/README.md`](../serving/README.md),
[`serving_design.md`](serving_design.md), and
[`serving_production.md`](serving_production.md).

## 2. The eight core components

```
┌──────────────────────────────────────────────────────────────┐
│  1. Public API                                               │
│     flash_rt/api.py — load_model() + VLAModel.predict()     │
└──────────────────────────────────────────────────────────────┘
        │ resolves (config, framework, arch) → frontend class
        ↓
┌──────────────────────────────────────────────────────────────┐
│  2. Hardware Dispatch Map                                    │
│     flash_rt/hardware/__init__.py::_PIPELINE_MAP            │
│     (config, framework, arch) → (module, class)              │
└──────────────────────────────────────────────────────────────┘
        │ instantiates per-(framework × hardware) frontend
        ↓
┌──────────────────────────────────────────────────────────────┐
│  3. Frontend (per (model, framework, hardware) triple)       │
│     flash_rt/frontends/{torch,jax}/<model>_<arch>.py        │
│     - Loads weights via WEIGHT_SPEC                          │
│     - Builds AttentionSpec, gets AttentionBackend            │
│     - Drives Calibration                                     │
│     - Captures CUDA Graph                                    │
└──────────────────────────────────────────────────────────────┘
        │ uses
        ├─→ ┌────────────────────────────────────────────────┐
        │   │  4. Weight Loading (declarative)               │
        │   │     flash_rt/executors/weight_loader.py       │
        │   │     ModelWeightSpec → WeightLoader.run()       │
        │   └────────────────────────────────────────────────┘
        ├─→ ┌────────────────────────────────────────────────┐
        │   │  5. Attention Backend (protocol)               │
        │   │     flash_rt/hardware/backend.py              │
        │   │     AttentionSpec → AttentionBackend impl      │
        │   └────────────────────────────────────────────────┘
        ├─→ ┌────────────────────────────────────────────────┐
        │   │  6. Calibration Framework                      │
        │   │     flash_rt/core/quant/calibrator.py         │
        │   │     ckpt_hash + Se → ~/.flash_rt/calibration  │
        │   └────────────────────────────────────────────────┘
        └─→ ┌────────────────────────────────────────────────┐
            │  7. CUDA Graph Capture                         │
            │     flash_rt/core/cuda_graph.py               │
            │     forward(int_ptrs) → captured graph         │
            └────────────────────────────────────────────────┘
                            │ calls
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  8. Kernel Library (the .so files)                           │
│     flash_rt/flash_rt_kernels.so   (~98 entries)           │
│     flash_rt/flash_rt_fp4.so       (~23 entries, SM120)    │
│     flash_rt/flash_rt_fa2.so       (vendored FA2, RTX)     │
│     flash_rt/libfmha_fp16_strided.so (Thor / SM100+)        │
│     csrc/ — source                                           │
└──────────────────────────────────────────────────────────────┘
```

The frontend is the only file you write per model. Everything below it is shared infrastructure. Everything above it is dispatch.

---

## 3. Component-by-component

### 3.1 Public API (`flash_rt/api.py`)

The single entry point. Two functions:

```python
model = flash_rt.load_model(checkpoint, config="pi05", framework="torch")
actions = model.predict(images=..., prompt=...)
```

`load_model` does three things in order:
1. Detects GPU compute capability (`torch.cuda.get_device_capability()`) and resolves `arch` ∈ `{thor, rtx_sm120, rtx_sm89}`.
2. Looks up `(config, framework, arch)` in the Hardware Dispatch Map (component 2).
3. Imports the resolved module, instantiates the frontend class, returns it wrapped in a thin `VLAModel` adapter.

The API is stable and deliberately small. It does *not* expose attention backends, calibration internals, or graph capture state. Those are accessible by reaching into `model._pipe`, but only the four documented frontends do that.

See [`stable_api.md`](stable_api.md) for the full surface.

### 3.2 Hardware Dispatch Map (`flash_rt/hardware/__init__.py`)

A single dict — the **only** place in the codebase that knows which frontend handles which hardware:

```python
_PIPELINE_MAP: dict[tuple[str, str, str], tuple[str, str]] = {
    ("pi05",   "torch", "thor"):       ("flash_rt.frontends.torch.pi05_thor",       "Pi05TorchFrontendThor"),
    ("pi05",   "torch", "rtx_sm120"):  ("flash_rt.frontends.torch.pi05",            "Pi05TorchFrontendRtx"),
    ("pi05",   "jax",   "thor"):       ("flash_rt.frontends.jax.pi05_thor",         "Pi05JaxFrontendThor"),
    ("groot",  "torch", "thor"):       ("flash_rt.frontends.torch.groot_thor",      "GrootTorchFrontendThor"),
    # … one row per (model, framework, arch) triple
}
```

External plugins extend the map at import time without forking the repo. See [`plugin_model_template.md`](plugin_model_template.md).

This is the simplest possible dispatcher. There is no plugin manifest, no entry-points scanning, no manifest YAML. The map is python and explicit.

### 3.3 Frontend (`flash_rt/frontends/{torch,jax}/<model>_<arch>.py`)

The frontend is the per-model file. It is the *only* piece a new model adds.

A frontend's responsibilities:

| Responsibility | Component used |
|---|---|
| Read checkpoint → fill in module attributes | Weight Loader (4) |
| Declare attention shapes → get a backend | Attention Backend (5) |
| Compute / load FP8 activation scales | Calibration Framework (6) |
| Capture forward as a static graph | CUDA Graph (7) |
| Call kernels in `forward()` | Kernel Library (8) |

A frontend is typically 800–1500 LOC, all linear. There is no inheritance hierarchy you fight — just duck-typed methods (`set_prompt`, `infer`) that the public API calls.

The four shipped models each have between two and four frontends (one per `(framework, arch)` combination). They do not share code in the forward path; they share code through components 4–8.

### 3.4 Weight Loading (`flash_rt/executors/weight_loader.py`)

A declarative description of how every weight tensor in a checkpoint maps to module attributes, including transforms (transpose, fuse-norm, FP8 quant) and quantization scales.

```python
WEIGHT_SPEC = ModelWeightSpec(
    framework="torch",
    blocks=[
        LayerBlock(prefix_fmt="encoder.layers.{i}", num_layers=18, items=[
            Item(name="q_proj.weight", key="{prefix}.q_proj.weight",
                 transforms=[Transpose(), QuantFP8()],
                 sink=TensorList("q_weights"),
                 scale_into="q_w_scales"),
            # … one Item per logical weight
        ]),
    ],
    singletons=[ Item(name="embed", key="model.embed.weight", sink=Attr("embed")) ],
)
```

The runner is framework-agnostic. Concrete `WeightSource` implementations exist for safetensors (torch) and Orbax dict (JAX); the same `WEIGHT_SPEC` works against either by swapping the source.

Full doc: [`extension/weight_spec.md`](extension/weight_spec.md).

### 3.5 Attention Backend (`flash_rt/hardware/backend.py`)

A protocol with two key methods:

```python
class AttentionBackend(Protocol):
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]: ...
    def run(self, site: str, layer_idx: int, q_seq: int, *, kv_seq=None, ...) -> int: ...
```

A *site* is a distinct attention shape (e.g. SigLIP vision, PaliGemma encoder, Pi0.5 decoder). Each site has a `SiteSpec` declaring head counts, head dim, max sequence lengths, optional sliding-window. A model declares an `AttentionSpec` (a dict of sites) and gets back a backend instance — the same pipeline source code runs on Thor (CUTLASS FMHA / decomposed cuBLAS) and RTX (vendored FlashAttention-2) by swapping which backend implementation is wired in.

Full doc: [`extension/attention_backend.md`](extension/attention_backend.md).

### 3.6 Calibration Framework (`flash_rt/core/quant/calibrator.py`)

FP8 calibration is a *framework*, not a per-model script. It owns:

- Cache key derivation (`{ckpt_hash}_Se{N}.json`) — survives checkpoint swaps and seq-length changes.
- Cache location (`~/.flash_rt/calibration/`) — same path across models, configs, frameworks.
- Save / load JSON shape per model — known fields plus a free-form `extra` map for model-specific scales.
- A two-phase flow: (a) measurement pass that records amax per FP8 GEMM input, (b) refit pass that compounds `alpha = act_scale × weight_scale` and bakes them as host scalars into the captured graph.

A frontend writes `_calibrate(self, sample_obs)` that follows the protocol; everything around it (cache lookup, cache save, file naming) is shared.

Full doc: [`extension/calibration.md`](extension/calibration.md). Mechanics doc: [`calibration.md`](calibration.md).

### 3.7 CUDA Graph Capture (`flash_rt/core/cuda_graph.py`)

A small helper that wraps `torch.cuda.CUDAGraph` (and a JAX equivalent) with two FlashRT-specific extensions:

- **Pointer-only forward**: the captured forward must take only `int` device pointers. No torch tensors, no JAX arrays. This is enforced at capture time and prevents accidental allocations during replay.
- **Tactic-stable autotune**: optionally re-captures up to N times and keeps the fastest schedule. Works around Myelin / cuBLASLt tactic non-determinism on Thor (~2ms variance per capture).

The captured graph is stored as `self._enc_ae_graph` on the frontend. Replay is `graph.replay()` plus a sync. No `.engine` file. No serialization. The graph lives in memory for the process lifetime; restart = re-capture (~50–500 ms on warm cache).

### 3.8 Kernel Library (`flash_rt/*.so`, source under `csrc/`)

The bottom of the stack. Hand-written CUDA kernels for the memory-bound ops (norm, activation, residual + norm + quant fusions, qkv split + RoPE, patch embed, etc.) plus thin wrappers around cuBLASLt FP8 GEMM, CUTLASS SM100 FP8 GEMM, vendored FlashAttention-2, and Thor's CUTLASS FMHA.

Three modules, all loaded by `import flash_rt`:

| Module | Always built | Contents |
|---|---|---|
| `flash_rt_kernels.so` | yes | ~98 pybind entries: norm / activation / fusion / quant / GEMM / attention / RoPE / utils |
| `flash_rt_fp4.so` | SM100+ only | ~23 entries: NVFP4 weight prep + SM120 block-scaled GEMM |
| `flash_rt_fa2.so` | RTX only | 2 entries: FA2 fp16 / bf16 forward |

Full inventory: [`kernel_catalog.md`](kernel_catalog.md). Fusion patterns and naming conventions: [`kernel_fusion.md`](kernel_fusion.md).

The kernel library is **stable** — every shipped kernel preserves its signature across releases. New kernels are added freely; existing ones are not renamed without a deprecation cycle.

---

## 4. The composition story

What happens when you call `model.predict(images, prompt)`:

```
predict(images, prompt)
  │
  │ first call only:
  │   set_prompt(prompt)
  │     ├─ tokenize + run language tower if needed
  │     ├─ frontend._calibrate(sample_obs)         ← Calibration (6)
  │     │     ├─ if cache hit: load JSON, bake alpha into module attrs
  │     │     └─ if cache miss: run 1 fwd, record amax, save JSON
  │     ├─ build AttentionSpec, hardware.make_attention_backend(arch, spec) ← Attention (5)
  │     ├─ WeightLoader(source, target=self, spec=WEIGHT_SPEC).run()        ← Weights (4)
  │     │     (already done at load_model; relisted for completeness)
  │     └─ cuda_graph.capture(self._forward_ptrs_only)                      ← Graph (7)
  │
  │ every call:
  └─ self._enc_ae_graph.replay()                   ← runs Kernel Library (8)
     post-process action
     return
```

The first call is slow (calibration + capture). Every subsequent call is `graph.replay()` plus three pointer copies. **No compilation. No `.engine` file. No re-export when the prompt changes**. The captured graph is reused across prompts because everything that varies (prompt tokens, observation pixels, action noise seed) is *input*, not graph topology.

---

## 5. Today's runtime characteristics

The current architecture is shaped by the small-batch realtime workload:

1. **Direct kernel composition** — the runtime executes the kernel sequence the frontend wrote, in the order written. The author of the frontend is the optimizer; there is no graph compiler / tactic search in the runtime today. Adding compiler-driven optimization passes is a possible direction; the current shape works because the target shape space is small enough for hand-tuned choices to be competitive.
2. **Static KV cache layout** — KV caches are per-layer device buffers, sized at construction. Pi0-FAST does decode-time KV writes into these pre-sized slots. Paging / eviction / cross-request sharing is not what the current design does, but the pointer-stable backend protocol leaves room for richer KV management when a workload needs it.
3. **Batching captured into the graph** — small batches (CFG, multi-policy, multi-frame) are supported within a captured graph. Cross-request continuous batching across users is the workload vLLM / SGLang are shaped for; FlashRT can be deployed as one process per captured graph today, and a serving layer that fans out to FlashRT workers is an explicit extension path. See [`inference_engine_differences.md`](inference_engine_differences.md) for the workload framing.

---

## 6. File map (one-line locator)

| Component | Source | Line count (approx) |
|---|---|---|
| Public API | `flash_rt/api.py` | 200 |
| Hardware dispatch | `flash_rt/hardware/__init__.py` | 100 |
| Frontends | `flash_rt/frontends/{torch,jax}/*.py` | 800–1500 each |
| Weight Loader | `flash_rt/executors/weight_loader.py` | 330 |
| Weight sources | `flash_rt/executors/{torch,jax}_weights.py` | 400 each |
| Attention protocol | `flash_rt/hardware/backend.py` | 405 |
| Attention backends | `flash_rt/hardware/{thor,rtx}/attn_backend*.py` | 300–600 each |
| Calibration cache | `flash_rt/core/quant/calibrator.py` | 167 |
| Calibration runner | per-frontend `_calibrate()` | 100–200 |
| CUDA Graph helper | `flash_rt/core/cuda_graph.py` | 150 |
| Kernel library | `csrc/`, built into `flash_rt/*.so` | (see catalog) |

The whole infrastructure side (excluding kernels and frontends) is **~3000 lines**. Eight files. Adding a new model is a frontend; everything else is reused unchanged.

---

## 7. Where to read next

- **You want to add a model** → [`adding_new_model.md`](adding_new_model.md)
- **You want to understand a single component deeply** →
  [`extension/weight_spec.md`](extension/weight_spec.md) ·
  [`extension/attention_backend.md`](extension/attention_backend.md) ·
  [`extension/calibration.md`](extension/calibration.md)
- **You want the kernel inventory** → [`kernel_catalog.md`](kernel_catalog.md)
- **You want serving hosts / capsules / HTTP endpoints** →
  [`../serving/README.md`](../serving/README.md) ·
  [`serving_design.md`](serving_design.md) ·
  [`serving_production.md`](serving_production.md)
- **You want to know how this differs from TensorRT / vLLM / SGLang** → [`inference_engine_differences.md`](inference_engine_differences.md)
