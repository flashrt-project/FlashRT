# Attention Backend: the protocol

> **Target audience**: anyone writing or reading the attention call site in a pipeline, adding support for a new attention shape (sliding window, MLA, cross-attn variants), or porting a frontend to a new GPU family.
>
> **What this doc is for**: the contract — `SiteSpec`, `AttentionSpec`, the `AttentionBackend` protocol, the slot / pointer model, and the lifecycle.
> **What this doc is not**: the kernel inventory. For that see [`../kernel_catalog.md`](../kernel_catalog.md).
>
> **Source**: [`flash_rt/hardware/backend.py`](../../flash_rt/hardware/backend.py) — protocol + base class + `SiteSpec` / `AttentionSpec`. Concrete backends live in `flash_rt/hardware/{thor,rtx}/attn_backend*.py`.

---

## 1. The problem this solves

Every model in FlashRT has 1–4 distinct attention shapes (for example, GROOT has SigLIP vision, Qwen3 backbone, DiT self-attention, DiT cross-attention). Each shape has its own preferred kernel, and each hardware target has its own preferred kernel for the same shape:

| Shape | Thor SM110 | RTX SM120 / SM89 |
|---|---|---|
| SigLIP vision (per view, MHA, HD=72) | `fmha_strided_full` (CUTLASS) | `fa2.fwd_fp16` |
| Paligemma encoder (GQA 8Q/1KV, HD=256) | `attention_qkv_fp16` (cuBLAS-decomposed) | `fa2.fwd_fp16` |
| Pi0.5 decoder (state-masked cross) | `attention_qkv_fp16_state_masked` | `fa2.fwd_fp16` + manual mask |
| GROOT DiT MHA (HD=128) | `attention_mha_fp16` (cuBLAS) | `fa2.fwd_fp16` |

If the pipeline calls these kernels directly, the pipeline source is hardware-forked. With `AttentionBackend` it is not — pipelines call `backend.run("encoder", layer_idx, q_seq=...)` and the backend instance decides which kernel fires.

---

## 2. `SiteSpec` — one attention shape

```python
@dataclass
class SiteSpec:
    num_layers:      int                 # how many layers reuse this shape
    num_q_heads:     int                 # MHA: == num_kv_heads; GQA: > num_kv_heads
    num_kv_heads:    int
    head_dim:        int
    max_q_seq:       int                 # slot allocation upper bound
    max_kv_seq:      int | None = None   # None == self-attn (mirrors max_q_seq)
    batch_axis:      int = 1             # >1 for batched SigLIP across views
    sliding_window:  int | None = None   # SWA window (Gemma 3 / Pi0.6)
    causal:          bool = False        # reserved for future decoder-only paths
    extra:           dict = {}           # backend-specific hints, escape hatch
```

A *site* is "all the layers in this model that share this attention shape". Layer count, head count, head dim, sequence-length budgets, and optional features (SWA, causal) live here. The pipeline declares one `SiteSpec` per shape; the backend pre-allocates Q/K/V slots sized per `(num_layers, max_q_seq, max_kv_seq, num_q_heads, head_dim)`.

The `extra` dict is an escape hatch for backend-specific keys — for instance, Pi0's decoder uses `extra={"kernel": "state_masked"}` to opt into the row-0 state-token mask.

---

## 3. `AttentionSpec` — the model's attention surface

```python
spec = (AttentionSpec()
        .add_site("siglip",  num_layers=27, num_q_heads=16, num_kv_heads=16,
                  head_dim=72, max_q_seq=256, batch_axis=2)
        .add_site("encoder", num_layers=18, num_q_heads=8,  num_kv_heads=1,
                  head_dim=256, max_q_seq=968)
        .add_site("decoder", num_layers=18, num_q_heads=8,  num_kv_heads=1,
                  head_dim=256, max_q_seq=10, max_kv_seq=978,
                  extra={"kernel": "state_masked"}))
```

The frontend builds the spec in a `make_attention_spec(...)` static method on the pipeline class — this keeps shape knowledge with the model and decouples it from the backend's allocation logic.

The frontend then asks the hardware module for a backend instance:

```python
from flash_rt.hardware import make_attention_backend

backend = make_attention_backend(arch="thor", spec=spec)  # → ThorFlashAttnBackend
# or
backend = make_attention_backend(arch="rtx_sm120", spec=spec)  # → RtxFlashAttnBackend
```

`make_attention_backend` is an internal helper that resolves `(arch, model_kind)` to a concrete class and returns a fully-allocated instance. The pipeline then injects the backend as a constructor argument.

---

## 4. The `AttentionBackend` protocol

```python
class AttentionBackend(Protocol):
    def sites(self) -> tuple[str, ...]: ...
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]: ...
    def run(self, site: str, layer_idx: int, q_seq: int, *,
            kv_seq: int | None = None, stream: int = 0,
            state_nk: int | None = None) -> int: ...
    def head_dim(self, site: str) -> int: ...
    def num_q_heads(self, site: str) -> int: ...
    def num_kv_heads(self, site: str) -> int: ...
```

Three things matter, the rest are accessors.

### 4.1 `get_slot_ptrs(site, layer_idx) → {role: int}`

Returns raw device pointers for the per-layer Q / K / V (and possibly O) buffers. The pipeline calls this once per `(site, layer_idx)` at construction and caches the dict. **Pointers are stable across CUDA Graph capture and replay** — that is the whole reason this protocol exists rather than passing torch tensors around.

The dict always contains keys `"Q"`, `"K"`, `"V"`. Some backends include `"O"` (Thor aliases O with Q for memory savings; RTX returns the FA2 output pointer at `run()` time and does not pre-publish `"O"`).

### 4.2 `run(site, layer_idx, q_seq, ...) → int`

Executes attention for one (site, layer) and returns a device pointer to the output. The contract:

- `q_seq` is the **active** Q rows for this call. May be smaller than `SiteSpec.max_q_seq`. Rows past `q_seq` are ignored. This is how variable-length prompts work without re-capture.
- `kv_seq=None` means self-attention; otherwise cross-attention with explicit KV length.
- `stream` is the CUDA stream pointer (0 = default). Backends launch on this stream.
- `state_nk` is honored only when `SiteSpec.extra["kernel"] == "state_masked"` (Pi0 decoder). The first Q row attends only to the first `state_nk` KV positions; remaining rows attend over the full `kv_seq`.
- The returned pointer is stable across CUDA Graph capture + replay for the same `(site, layer_idx)`.

The pipeline sees the same call sequence on every hardware target. Only the kernel fired by `run()` changes.

### 4.3 Two storage models

Backends are allowed to differ on **who allocates the buffers**:

- **Backend-owned** (RTX): the backend allocates Q / K / V as torch tensors in `__init__`. The pipeline writes Q / K / V into the published pointers. Modern vendored FA2 backends also own stable output tensors; legacy pip `flash_attn_func` fallback paths keep a torch output reference so the caching allocator doesn't reassign across capture / replay.
- **Pipeline-owned** (Thor): the pipeline allocates its own Q (which doubles as the output buffer post-attn), plus per-layer K / V cache as part of the weights dict. The backend is a thin wrapper around `fvk.attention_qkv_fp16` and takes pointers at `run()` time.

A pipeline written to the protocol does not care which storage model fires — it always reads pointers via `get_slot_ptrs` and lets `run()` return the output pointer. This is the key portability property.

---

## 5. Lifecycle and threading

- **One backend per pipeline.** The backend is constructed once (typically before the pipeline, then injected), lives as long as the pipeline does, and is destroyed with it. Sharing a backend across pipelines is not supported.
- **Single-threaded w.r.t. CUDA.** Cross-Python-thread calls into the same backend are undefined.
- **No reentry during graph capture.** During `cuda.graph.capture`, `run()` may be called many times (once per `(site, layer)`), but all calls run on the captured stream. After capture, the runtime path is `graph.replay()` — the backend's `run()` is *not* called per inference.

---

## 6. The four shipped backends

| Backend | Arch | Used by | Backing kernel(s) |
|---|---|---|---|
| `ThorFlashAttnBackend` | thor (SM110) | Pi0.5, Pi0 | `fmha_strided_full` (SigLIP) + `attention_qkv_fp16` / `attention_qkv_fp16_state_masked` (encoder / decoder) |
| `ThorGrootAttnBackend` | thor (SM110) | GROOT N1.6 | `fmha_strided_full` (SigLIP) + `attention_qkv_fp16` (Qwen3) + `attention_mha_fp16` (DiT) |
| `RtxFlashAttnBackend` | rtx_sm120 / rtx_sm89 | Pi0.5, Pi0 | `fa2.fwd_fp16` / `fa2.fwd_bf16` for every site |
| `RtxGrootAttnBackend` | rtx_sm120 / rtx_sm89 | GROOT N1.6 | `fa2.fwd_*` for SigLIP / Qwen3 / DiT self / DiT cross |

Pi0-FAST is the explicit exception — it does not use this protocol, because its hardware differences are in GEMM dispatch rather than attention shape. See `frontends/torch/pi0fast.py` for the single-file SM-fork pattern it uses instead.

---

## 7. Adding a new backend (new GPU family)

When porting to a new arch (e.g. RDNA, AMD MI300):

1. **Implement the protocol.** Subclass `AttentionBackendBase` (gives you the accessor defaults) and implement `get_slot_ptrs` and `run`.
2. **Decide your storage model.** Backend-owned is easier to reason about. Pipeline-owned saves memory if you can alias buffers, but requires the pipeline to allocate.
3. **Wire it into `make_attention_backend`.** Add a row mapping `(arch, model_kind) → YourBackend`.
4. **Run the conformance test.** `tests/test_thor_attn_backend.py` and `tests/test_thor_groot_attn_backend.py` exercise the protocol end-to-end against captured-graph replay. Same tests run against the new backend by parameterizing the `arch` fixture.

The protocol does not assume CUDA. A backend that wraps `hipblaslt` or a CPU reference path is legal.

---

## 8. Adding a new attention shape

When adding a new model with an attention shape no existing backend handles (e.g. sliding-window for Gemma 4, MLA for DeepSeek V3+):

1. **Extend `SiteSpec` if necessary.** Sliding window is already supported (`sliding_window: int | None`). Genuinely new features (latent attention, CSA, HCA) need a new field — keep it optional.
2. **Implement the kernel.** Add a `.cu` to `csrc/attention/`, expose via pybind, document in `kernel_catalog.md`.
3. **Update the backend.** The backend's `run()` becomes a `match site` dispatch — add a branch for the new shape and call your new kernel.
4. **Conformance test.** Extend `tests/test_*_attn_backend.py` with a new site and assert pointer stability + output match against a reference.

The model pipeline does not change. It always calls `backend.run(...)`. The new kernel only changes what happens *inside* the backend.

---

## 9. Pointer-stability contract (the load-bearing rule)

Every guarantee below must hold for the captured-graph fast path to work:

1. `get_slot_ptrs(site, layer_idx)` returns the **same dict-of-ints** every call within a backend's lifetime, for the same `(site, layer_idx)`.
2. The pointer values themselves are stable across `graph.capture()` and `graph.replay()`.
3. The output pointer returned by `run(site, layer_idx, ...)` is valid until the next `run(same_site, same_layer_idx)` call, or end-of-backend lifetime, whichever comes first. (Backends that re-use slots — Thor — return the same pointer every call; backends that allocate per call — RTX FA2 — hold a reference so the caching allocator doesn't reassign.)
4. **Pipelines must not cache output pointers across `run()` calls** at the same site / layer. Re-read after every `run()`. (In practice, both shipped backends return the same pointer for the same `(site, layer)`, but this is implementation, not contract.)

If any of these break, captured graphs will silently read / write bogus memory after replay. The conformance tests check 1 and 2; 3 and 4 are pipeline discipline.

---

## 10. Protocol scope

Three concerns sit outside this protocol — handled at other layers today, available as extension surfaces if a workload needs them:

1. **KV cache management.** Backends allocate *static* per-layer K / V slots sized at construction; Pi0-FAST writes into these slots during AR decode. Richer KV management (paging, eviction, cross-request sharing) is a separate concern, addressable by adding a KV-policy layer on top of the slot pointers — the protocol's pointer-stability guarantee is the foundation it would build on.
2. **Quantization of attention.** Today's backends consume fp16 / bf16 Q / K / V; FP8 stops at the GEMM input boundary on either side of attention. FP8 attention (FlashInfer / SGLang-style) is a possible future direction — adding it would extend, not break, the protocol.
3. **Cross-layer fusion.** Each `run()` is one layer; cross-layer fusion (e.g. attention output + post-attn norm) lives in the pipeline via the `residual_add_rms_norm_fp8_*` kernel family, outside the protocol surface.

---

## 11. Where to look in the source

| What | File |
|---|---|
| Protocol, `SiteSpec`, `AttentionSpec`, base class | [`flash_rt/hardware/backend.py`](../../flash_rt/hardware/backend.py) |
| Thor Pi0/Pi0.5 backend | `flash_rt/hardware/thor/attn_backend.py` |
| Thor GROOT backend | `flash_rt/hardware/thor/attn_backend_groot.py` |
| RTX Pi0/Pi0.5 backend | `flash_rt/hardware/rtx/attn_backend.py` |
| RTX GROOT backend | `flash_rt/hardware/rtx/attn_backend_groot.py` |
| Pipeline call site (Pi0.5, all hardware) | `flash_rt/models/pi05/pipeline.py` |
| `make_attention_backend` resolver | `flash_rt/hardware/__init__.py` |
| Conformance tests | `tests/test_thor_attn_backend.py`, `tests/test_thor_groot_attn_backend.py` |
