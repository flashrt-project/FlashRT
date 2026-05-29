# FlashRT Execution Contract (common-layer spec)

> Status: design draft · 2026-05-28 · branch `spec/exec-contract`
> This layer defines FlashRT's **common execution contract**: a clean, minimal C ABI that
> takes the already-proven "kernelize + CUDA Graph replay" seam — today an implicit convention
> scattered across the Python frontends — and turns it into an explicit, embeddable common layer.
> It fixes **mechanism only, never scenario policy**, and has **zero dependency on the csrc kernel layer**.

---

## TL;DR — mechanism · structure · example

### Mechanism
FlashRT runs a model by capturing its kernel sequence into a **CUDA Graph once** (setup), then
**replaying that graph forever** (hot path). This contract is a thin C ABI that takes over only the
*replay side*. A model frontend captures its graph however it likes — torch (`torch.cuda.graph`) or
the framework's ctypes `CUDAGraph` — and **adopts** the instantiated graph-exec into the contract;
the contract then owns replay, shape-variant selection, zero-copy buffer hand-off, streams, events,
and a Plan DAG. It adds **zero replay overhead** (the very same `cudaGraphLaunch`) and the default
path is **byte-identical**. All capture-time work — autotune, calibration, warmup — stays in the
frontend and is fully under the caller's explicit control; the contract never sees it.

### Structure — 3 objects + 1 key (mechanism only)
```
Buffer   named device memory; the ONLY "state" (KV cache, scales, noise, subgoal embedding...).
         Graphs are wired together by SHARING one Buffer (zero-copy hand-off).
Graph    a ShapeKey -> graph-exec variant table (exact key + LRU). capture (own) or adopt
         (external exec); replay(key, stream); bind named I/O ports.
Plan     a dumb DAG of (Graph, ShapeKey) replays across streams with explicit cross-stream deps.
         Data dependencies only — no priority/deadline/preemption (that's upper-layer policy).
ShapeKey opaque u64 encoding (B, S, ...). Batch is just a FIELD of the key, not a new axis.
```
Plus mechanism helpers: prioritized + wrappable streams (incl. the default stream), cross-stream
events, and device-to-device `buffer_copy`. Lives in a top-level `exec/` layer, **sibling to `csrc/`
with zero dependency on it** (§5). Capture/autotune/calibration/sessions/schedulers/protocols are
**out of scope** — the upper layer (frontend / serving) owns them; the contract only *adapts* (§8.5).

### Simple example (C ABI; the torch/ctypes frontend keeps capturing as today)
```c
frt_ctx c = frt_ctx_create();

/* (1) LLM/VLA: drive a graph the frontend already captured, bit-identical */
frt_graph g = frt_graph_create(c, "decode", /*max_variants=*/256);  /* exact-key LRU table */
frt_graph_adopt(g, /*key=*/cur_pos, external_exec);  /* e.g. torch raw_cuda_graph_exec() */
int s = frt_ctx_wrap_stream(c, host_stream);          /* reuse the frontend's own stream */
frt_graph_replay(g, /*key=*/cur_pos, s);              /* == cudaGraphLaunch, zero overhead */

/* (2) Multi-subgraph (VLA): vision -> encoder -> action, zero-copy + cross-stream */
frt_buffer x = frt_buffer_alloc(c, "enc_x", nbytes);
frt_graph_bind(vision, "out", x);  frt_graph_bind(encoder, "in", x);  /* same buffer => no copy */
frt_plan p = frt_plan_create(c);
int a = frt_plan_add(p, vision,  0, s0);
int b = frt_plan_add(p, encoder, 0, s1);
frt_plan_after(p, b, a);              /* encoder (s1) waits vision (s0) via an event */
frt_plan_execute(p, 0);
```

**Proven (v1):** drives a real **LLM** (Qwen3.6 decode+spec — bit-identical, 133 tok/s, no regression)
and a real **VLA** (Pi0.5 FP16 + FP8 — cosine 1.0); every primitive validated. Acceptance: §8.

---

## Layering & serving hosts

### Setup (cold, once) vs hot path (replay) — and where each language sits
Everything model-specific and iterative is **setup** (cold, runs once): weight load, **autotune**,
**calibration** (single / multi-frame N>1 dataset / int8-static / AWQ), graph **capture**, and `adopt`.
It stays in the **Python frontend** — fastest to develop, inspect, debug; adding a model = a script, no
recompile. The contract sees only the *result*: an adopted graph (+ variants) + buffers (scales/state).

```
  SETUP (cold, once, model-specific)            HOT PATH (replay, hot, model-agnostic)
  ───────────────────────────────────          ──────────────────────────────────────
  Python frontend:                              host loop (Python today; C++/Rust at deploy):
    load weights                                  write inputs -> Buffer
    calibrate (N>1 dataset) / autotune    ──▶     frt_graph_replay / frt_plan_execute  ──▶ GPU
    capture graph (torch or ctypes)               read outputs                              kernel
    frt_graph.adopt(exec) / buffer.wrap()         (one cudaGraphLaunch per replay = pure GPU)  graph
```
Why this split is optimal: the rigid, debug-heavy "engine-as-code" problem (TensorRT-LLM-style
*one model = one compiled engine*) comes from compiling **model logic**. Here the kernel sequence is
captured at runtime into a graph (**data**), so the C++ layer holds **zero model-specific code — one
binary drives all models**. Flexibility lives in Python; only the stable, mechanical replay is native.

### Who drives the loop — today (A) vs deployment (B)
```
  (A) today — Python frontend drives; exec is the C++ library it calls
      Python frontend ──pybind──▶ C++ exec (libflashrt_exec) ──cudaGraphLaunch──▶ GPU kernel graph
      (pybind crosses the SAME C ABI a native host would; Python is the loop driver for now)

  (B) deployment — a native host drives; Python only at setup (same process) or absent
      C++/Rust host ──C ABI──▶ C++ exec (libflashrt_exec.so) ──cudaGraphLaunch──▶ GPU kernel graph
      (no Python in the hot loop; capture/adopt done once — in-process Python, or ported to C++)
```
A captured CUDA graph is **not serializable across processes** (baked pointers are process-local), so
capture must run in the **same process** that replays — which is why "Python setup once + native hot
loop in one process" is the pragmatic embedded path (Python out of the hot loop without porting capture).

### Two reference hosts (live in `serving/` — scenario/policy, not the contract)
Same `libflashrt_exec`; the host language is chosen per scenario.
```
  serving/robot_host/  (C++)  — real-time VLA            serving/llm_agent/  (Rust) — LLM server
  ───────────────────────────────────────────            ─────────────────────────────────────────
  C++ host loop:                                          Rust host (axum/SSE + session registry):
    vision ─enc_x─▶ action   (Plan, 1 DAG)                  per token: write tok -> Buffer
    ASR on a concurrent stream (overlap/interrupt)          frt_graph_replay(decode, key=cur_pos, s)
    subgoal change = overwrite Buffer (no recapture)        read logits -> argmax -> append
    interrupt granularity = one short replay                spec/MTP = more adopted graphs, same way
  Why C++: real-time, ROS2-adjacent, one toolchain         Why Rust: async/safety shell; FFI seam
  with exec+kernel, no FFI seam, easiest debug.            crossed once per token (thin).
```

---

## 0. Positioning: what this layer is / is not

What FlashRT already does (see `flash_rt/core/cuda_graph.py`, `flash_rt/frontends/`):

> **setup-once** (`set_prompt`: tokenize / calibrate / autotune / capture)
> → **replay-forever** (`infer`: copy inputs into static buffers → `graph.replay(stream)` → clone outputs)

Multi-subgraph wiring (vision→encoder→action) already exists, and it is dead simple: **the vision
graph writes `_enc_x`, the encoder graph reads `_enc_x` from the same device pointer — no HBM
round-trip, no copy.** Multi-batch already exists too: a separate graph is captured for B=2.

**This layer's whole job is to make that already-validated seam an explicit contract. It invents no
new abstraction.**

### IS (mechanism)
- A **replayable graph node** + its bound, named I/O buffers
- Zero-copy hand-off between graphs via a **shared buffer**
- `select` among **shape variants** (batch 1-8, seqlen buckets) and replay
- **Multi-stream + event sync + stream priority** (the hardware mechanisms behind parallel
  scheduling, interruption, gap-fill)
- **Imperative driving**: the host may fire any graph/Plan on demand

### IS NOT (policy — lives in serving / user hosts)
- ❌ session registry, prefix/radix cache, KV append/fork/evict semantics
- ❌ scheduler: priority decisions, deadlines, preemption policy
- ❌ protocols: OpenAI `/v1/chat`, SSE, MCP, agent protocols
- ❌ robotics: sensor triggers, action cadence, multi-rate orchestration
- ❌ scenario tags on a graph like `family: llm|vla`, `latency_hint`, `bottleneck_hint`

> Red line: **if any scenario forces you to add a field to the common layer, that field is policy
> and belongs back in an example.** The only things allowed to grow are `ShapeKey` semantics and the
> number of buffers/graphs.

---

## 1. Object model: 3 concepts + 1 key

```
Buffer    A named device memory region. Graphs are wired together by SHARING one Buffer
          (zero-copy). Every "mutable state" (KV, vision cache, subgoal embedding, scales)
          is a Buffer. The framework owns its lifetime + device pointer only; append/fork/
          evict are caller logic written on top of the pointer.

Graph     A captured graph-exec + its bound named I/O Buffers + replay(stream).
          Internally a ShapeKey -> graph-exec variant table (batch 1-8 / seqlen buckets go here).

Plan      An ordered replay of (Graph, ShapeKey) across streams, with explicit event deps.
          A dumb DAG: it expresses DATA dependencies only — never priority/deadline/preemption.

ShapeKey  An opaque u64 encoding (B, S, ...). Batch is NOT a new axis — just one field of the key.
```

### Key design decisions
1. **Batch is not a new axis; it is a field of `ShapeKey`.** And the ShapeKey is an **exact key, not
   a bucket**: Qwen3.6 keys a decode graph per exact `cur_pos`, a verify graph per `(cur_pos, K)`,
   evicted by an LRU table (cap 256). Bucketing (e.g. seqlen {512,1024}) is just **one caller
   strategy** layered on top of an exact key — not a framework concept. **Which keys to capture and
   how to evict is caller/example policy**; the framework only provides the keyed variant table +
   capacity LRU + "does this key have a variant".
2. **No `State` object.** State is uniformly a `Buffer`. The framework defines no append/fork verbs.
3. **`Plan` is dumb.** It expresses data deps only. Multi-model co-host and VLA multi-rate are
   composed by the upper layer using Plans as building blocks.
4. **Capture "intelligence" stays out of the common layer.** Capture is cold and model-specific
   (autotune/calibrate) and stays in the Python frontend. The common layer provides only
   framework-agnostic `begin/end capture` (stream level) + buffer registration, owning the
   **replay-time contract + buffer registry**. This is the core red line that keeps the layer thin.

---

## 2. C ABI (authoritative form: `exec/include/flashrt/exec.h`)

```c
typedef struct frt_ctx_s*    frt_ctx;     /* owns arena + stream/event pool */
typedef struct frt_buffer_s* frt_buffer;
typedef struct frt_graph_s*  frt_graph;
typedef struct frt_plan_s*   frt_plan;
typedef struct frt_event_s*  frt_event;   /* cross-stream sync point */
typedef uint64_t             frt_shape_key;

/* --- ctx / stream / event --- */
frt_ctx   frt_ctx_create(void);
int       frt_ctx_stream(frt_ctx, int priority);   /* prioritized stream; 0 = normal */
frt_event frt_ctx_event (frt_ctx);                 /* imperative cross-stream sync (interrupt / spec snapshot) */
int       frt_event_record(frt_event, int stream_id);
int       frt_stream_wait (frt_ctx, int stream_id, frt_event);

/* --- Buffer: named device memory; graphs wire via sharing it --- */
frt_buffer frt_buffer_alloc(frt_ctx, const char* name, size_t bytes);
frt_buffer frt_buffer_wrap (frt_ctx, const char* name, void* dptr, size_t bytes);
void*      frt_buffer_dptr (frt_buffer);   /* caller does copy_/append/zero; framework owns no semantics */
int        frt_buffer_copy (frt_ctx, frt_buffer dst, size_t dst_off,
                            frt_buffer src, size_t src_off, size_t bytes, int stream_id);

/* --- Graph: a ShapeKey -> graph-exec variant table (exact key + LRU, cap = max_variants) --- */
frt_graph  frt_graph_create (frt_ctx, const char* name, size_t max_variants);
int        frt_graph_capture(frt_graph, frt_shape_key key,
                             void (*record)(void* user, void* stream), void* user);
int        frt_graph_bind   (frt_graph, const char* port, frt_buffer);  /* named I/O */
int        frt_graph_replay (frt_graph, frt_shape_key key, int stream_id); /* missing key -> error */
int        frt_graph_has_variant(frt_graph, frt_shape_key key);

/* --- Plan: dumb DAG, data dependencies only --- */
frt_plan   frt_plan_create (frt_ctx);
int        frt_plan_add    (frt_plan, frt_graph, frt_shape_key key, int stream_id);
int        frt_plan_after  (frt_plan, int node_idx, int dep_node_idx); /* event sync */
int        frt_plan_execute(frt_plan, frt_shape_key key);
```

That's all. No session, scheduler, OpenAI, KV, or family.

---

## 3. Design principles (splitting / concurrency / interruption)

### 3.1 Split criterion: split at data/cadence seams, not to expose idle bubbles
A graph is a closed capsule. **Every split point = one HBM materialization** (intermediate values
stay in registers/smem within a capsule; across graphs they must land in HBM). So split a boundary
only when **any** of these holds, otherwise fuse into one graph:
1. The output is consumed at a **different cadence** (vision 30Hz → action 50Hz)
2. Inputs **must be mutated** between stages (copy new obs / new token)
3. Control flow **must return to host** (spec accept, sampling, early exit, **interrupt point**)
4. The stage is **reused** (vision encoder across calls / across models)

Without one of these, don't split — you would forfeit the megakernel fusion you worked for.
**Don't split where you just finished fusing.**

### 3.2 Parallel scheduling / gap-fill = multi-stream, and it's policy
To reclaim small-batch GPU headroom, put the second model entirely on another stream and let the
hardware overlap — **not** by finely splitting the first model and hand-inserting work into bubbles.
The mechanism (multi-stream + event) is free within the 3 atoms; **whether to overlap, or to keep
the GPU idle to protect p99, is upper-layer policy** and stays out of the common layer.

### 3.3 Interruption: short graphs make it free at graph boundaries
A CUDA graph cannot be preempted mid-replay — but that is exactly the dividend of kernelization:
**the graphs are short** (VLA inference ~17ms, LLM decode sub-ms), so **interrupt granularity = one
replay duration**, and the host re-decides between replays. All three real-robot interrupt actions
are expressible with the 3 atoms + multi-stream:

| Robot action | Implementation | New concept? |
|---|---|---|
| Voice concurrency (ASR ‖ VLA) | ASR on a separate stream, hardware overlaps | No |
| Change subgoal / goal | subgoal embedding is a `Buffer`; host overwrites it; next replay uses it — **no recapture** | No |
| Hard interrupt (abort current action) | host loop stops issuing the next replay, issues a high-priority graph; granularity = one short graph | No |

Two accompanying disciplines (usage, not new concepts):
- **Imperative driving is first-class.** `Plan` covers only the static DAG inside one inference
  (vision→encoder→action). The **interruptible outer loop** (read sensor/voice → decide which Plan
  to fire → swap a buffer → abort/switch) **belongs to the robot host, not the framework**. The
  framework only guarantees graphs/Plans can be fired imperatively on demand.
- **Anything mutable at "interrupt cadence" must be a bound `Buffer`, never baked into the graph.**
  subgoal/prompt embeddings, target poses, mode flags — all become overwritable Buffers, so changing
  the goal is a µs-scale copy, not a seconds-scale recapture. (Proven pattern: FP8 scales are updated
  in-place today, graph pointers stay valid, no recapture.)

---

## 4. Scenario mapping (the framework need not understand scenarios)

| Scenario | How the atoms compose | Framework knows the "scenario"? |
|---|---|---|
| Multi-subgraph VLA (vision→llm→action) | 3 `Graph`s; vision out-port and llm in-port `bind` the same `Buffer`; a `Plan` chains them | No — just 3 graphs sharing a buffer |
| Multi-model co-host (Pi05 + Qwen) | Two graph sets + two `Plan`s on different `stream_id`s; ordering is the host's call | No — just executes DAGs |
| LLM decode + KV | KV is a `Buffer`; the decode graph binds it; each step the host copies into the KV offset, then `replay(key=seqlen_bucket)` | No — unaware of KV/session |
| Batch 1-8 | `ShapeKey` carries B; capture 4 variants; host picks the key by the packed B | No — just a key |
| Spec decode / MTP | draft/verify are `Graph`s; **accept length and rollback live in host code**; rollback = rewrite the KV buffer's logical length | No — unaware of spec |
| Voice interrupt → change subgoal | the ASR `Graph` runs on another stream; the subgoal-embedding `Buffer` is overwritten by the host | No — just a buffer overwrite |

> This table is the acceptance test: **if any scenario forces a new common-layer field, the design
> failed — go back to the §0 red line.**

---

## 5. Directory layout

**Core point: the exec layer has zero dependency on the csrc kernel layer.** It captures whatever
the `record` callback enqueues onto a stream (FlashRT kernels / torch ops / raw CUDA), and only ever
sees streams / graphs / events. It is a kernel-agnostic + framework-agnostic **orchestration layer**,
orthogonal to csrc's compute kernels, with its own independent cross-hardware backend axis. So it sits
**as a sibling to csrc**, not inside it.

```
exec/                              # NEW top-level: execution-contract layer (orchestration, not kernels; zero csrc dep)
  include/flashrt/exec.h           #   public C ABI — the authoritative form of this spec
  src/
    context.cpp                    #   frt_ctx: arena + stream/event pool
    buffer.cpp                     #   frt_buffer
    graph.cpp                      #   frt_graph: ShapeKey -> graph-exec variant table + capture/replay
    plan.cpp                       #   frt_plan: dumb DAG executor
  backend/                         #   exec's own cross-hardware backend axis (orthogonal to csrc's kernel backends)
    backend.h                      #     graph/stream/event abstract interface
    cuda/cuda_backend.cpp          #     CUDA impl (cudaGraph/Stream/Event)
    # future: hip/ , level_zero/ ...
  bindings/
    exec_pybind.cpp                #   pybind -> flash_rt.runtime.exec (dev/migration only)
  CMakeLists.txt                   #   builds libflashrt_exec (.so) as an independent target

flash_rt/runtime/exec.py           # thin Python wrapper (sibling to rtc.py), dev only
serving/robot_host/               # scenario: multi-model + interrupt (VLA + ASR), policy-layer demo
serving/llm_agent/                # scenario: session/KV/OpenAI shell (folds in existing *_openai_server.py)
docs/exec_contract.md              # this document

csrc/                              # unchanged: pure compute kernels (attention/gemm/conv/...), own cross-hw backends
```

Capture "intelligence" (autotune/calibrate) stays in the Python frontend; this layer owns only the
replay-time contract.

---

## 5b. Paper validation: Qwen3.6 + Pi05 (step 1, done)

Map both **real** frontends' graph/buffer/key structures onto `frt_*` to test sufficiency.
Conclusion: **the 3-atom core holds, with no scenario field forced out**; validation did force out 3
**mechanism** primitives (now added to `exec.h`), none of which touch the policy red line.

### Qwen3.6 (LLM + linear attention + MTP spec)
- **Graphs (all exact-key, LRU table cap 256)**: decode by `cur_pos`, verify by `(cur_pos,K)`,
  MTP-draft by `mtp_pos`, prefill-chunk by `(start,len)`. → one `frt_graph` per type, ShapeKey
  encodes those exact values, `max_variants=256` LRU. **No batch axis (B=1)**.
- **All state is `Buffer`**: `K_cache/V_cache` (16 layers), linear-attn `_lin_state/_lin_conv_state`,
  MTP KV, `_static_token_id`, `_logits_buf`, RoPE tables, snapshot bufs. → all `frt_buffer_alloc` + `bind`.
- **KV append** = write into `K_cache[layer, cur_pos:cur_pos+1]` (host writes the offset before/within
  the replay). **No append verb.**
- **Spec flow**: host imperatively runs K draft graphs → concurrent snapshot on a side stream
  (`_snap_stream` + wait_stream) → replay the `(cur_pos,K)` verify graph → host argmax-compares to
  pick accept length N → on mismatch, `Buffer`-to-`Buffer` copy rollback + re-advance with the
  `K=N+1` graph. **Accept/rollback are entirely host control flow**; the framework is unaware of spec.
- **Forced-out mechanisms**: ① multi-stream + **standalone event** (the snapshot side-stream's
  imperative wait_stream, outside any Plan) → `frt_ctx_event/record/stream_wait`; ② `frt_buffer_copy`
  for snapshot/restore; ③ Graph variant table `max_variants` + LRU.

### Pi05 (VLA multi-subgraph + diffusion)
- **Graphs**: SigLIP (vision) one; encoder+decoder fused into one (**all 10 diffusion steps inside
  the graph, one replay runs them all, no host loop**); CFG B=2 captures a separate
  `_enc_ae_graph_b2` (there is even an outer graph fusing lang-swap + SigLIP×2 + enc + dec into one).
  → vision/enc-dec are two `frt_graph`s; B=2 is the B-field variant of a ShapeKey.
- **Zero-copy hand-off (the core)**: SigLIP writes `_enc_x` → encoder reads the same
  `_enc_x.data_ptr()`, **no copy**; encoder writes `_Kc/_Vc` → decoder reads the same pointers.
  → both graphs `bind` the same `frt_buffer` — exactly the wiring mechanism the design intends.
- **Per-step time embeds** are precomputed into `_sa_all/_sf_all/_fs_all` at set_prompt time, read-only
  inside the graph. → `Buffer`, bind.
- **Batch/CFG**: `set_batched_mode(B=2)` triggers a separate capture = the B-variant of the ShapeKey.
  **B is not a new concept.**
- **"Plan vs one big graph" is the author's choice**: Pi05 can either chain vision + enc-dec via a
  `Plan`, or capture them as one graph like the CFG outer graph. The contract supports both — which
  validates the flexibility.
- Today Pi05 is single-stream; multi-stream/multi-rate is forward-looking for `serving/robot_host/`,
  and the mechanism is already in place.

### Verdict
- ✅ KV append / rollback / accept / batch / multi-subgraph are **all expressed with only Buffer +
  exact-key Graph + host control flow + shared bind**, with **no session/scheduler/family/KV-verb
  policy field added**. Red line held.
- ➕ Added 3 mechanisms: `event` (imperative cross-stream), `buffer_copy` (torch-free rollback), graph
  `max_variants` + LRU. This is "minimal complete", not bloat — without them, imperative interrupt
  sync and spec rollback would have to drop to raw CUDA, defeating the embeddability goal.

---

## 6. Rollout (3 steps, not 22 weeks)

1. **Spec (~1 week)**: this document + `exec/include/flashrt/exec.h`. Acceptance: use the §4 table to
   wire Qwen decode and Pi05 vision→action **on paper** without adding any field. (Done — see §5b.)
2. **C++ impl + pybind + one real model (2-3 weeks)**: implement Buffer/Graph/Plan + the CUDA backend;
   route the **existing Qwen frontend's replay hot path** through it (capture stays in Python).
   Acceptance: E2E tok/s and cos **do not regress** (the floor); multi-session B≤8 runs through the
   same `select(key)`.
3. **One multi-node + interrupt example (1-2 weeks)**: `serving/robot_host/` chains Pi05
   vision→llm→action across streams via a Plan, and demonstrates concurrent ASR + subgoal-buffer
   overwrite interruption. Acceptance: Pi05 latency does not regress; **no scenario field was added to
   the common layer throughout** (if one was, go back to step 1).

The Rust shell, OpenAI, and scheduler all live in serving / the upper layer, off this main line.

---

## 7. Honest boundaries
- **Single-model B=1 decode loop**: `replay` is already a ~10µs ctypes call; Python is fine, and a C++
  rewrite buys ≈0.
- **Where C++ actually earns its keep**: (a) the multi-node / multi-stream / event Plan execution loop
  (escaping GIL/GC jitter); (b) embeddability — a robot host or Rust agent server can link a pure
  C-ABI `.so` directly.
- Therefore the common layer = replay-time execution of Buffer/Graph/Plan + a C ABI, **and not one bit
  more.**

---

## 8. v1 acceptance checklist

Status of the first version (branch `spec/exec-contract`). Everything below is implemented and
verified in-container on RTX 5090 (sm_120).

### 8.1 Primitives (mechanism)
- [x] `Buffer` alloc / wrap / dptr / bytes / name — owned + external pointers
- [x] `frt_buffer_copy` device-to-device async on a stream
- [x] `Graph` capture (own, via record callback) — stream-level RELAXED capture
- [x] `Graph` **adopt** an external instantiated exec (torch `raw_cuda_graph_exec()` or ctypes
      `CUDAGraph._graph_exec`); non-owned, never freed by the layer
- [x] `Graph` `bind` named ports (zero-copy hand-off bookkeeping + lifetime)
- [x] `Graph` ShapeKey **variant table + LRU** (exact key, `max_variants`); missing key →
      `FRT_ERR_NO_VARIANT` (never a silent no-op)
- [x] `Plan` dumb DAG: `add` / `after` (cross-stream event dep) / `execute` / `sync`
- [x] Events: `frt_ctx_event` / `frt_event_record` / `frt_stream_wait` (imperative cross-stream)
- [x] Streams: `frt_ctx_stream(priority)` + `frt_ctx_wrap_stream` (external); **default stream
      (handle 0) is a first-class wrappable stream** (id validated by range, not handle nullness)
- [x] Lifetime: ctx owns streams/events/buffers/graphs/plans; clean teardown, no leaks of owned memory

Test: `exec/tests/test_exec.py` (5/5) + `exec/tests/test_plan_vla.py` (VLA-shaped Plan chain).

### 8.2 Real-model integration (correctness — bit-identical / cosine)
- [x] **Qwen3.6-27B NVFP4 (LLM)** decode-S=1 graph driven via exec — bit-identical to torch replay
      (`test_adopt_qwen36.py`, max abs diff 0.0)
- [x] **Qwen3.6** spec-decode path (MTP-chain + verify graphs) via exec — output sha identical,
      spec stats identical (commit `0b988d8`)
- [x] **Qwen3.6** spec rollback (lin/conv restore) via `frt_buffer_copy` — sha identical (`c3c66f4`)
- [x] **Pi0.5 (VLA) FP16** full infer graph via exec — `cos = 1.000000`, bit-identical
      (`test_adopt_pi05.py`)
- [x] **Pi0.5 (VLA) FP8** full infer graph via exec — `cos = 1.000000`, bit-identical
      (`test_adopt_pi05.py --fp8`)

### 8.3 Performance (no regression — opt-in flag off vs on)
- [x] Qwen3.6 decode tok/s: 133.7 (off) vs 133.9 (on), K=6 representative prompt — within noise
- [x] Qwen3.6 prefill TTFT: 217.1 ms off vs on — unchanged
- [x] Pi0.5 FP16 p50: 34.5 ms off vs on — unchanged
- [x] Pi0.5 FP8 p50: 24.55 ms off vs on — unchanged
- Rationale: frt replay is the same `cudaGraphLaunch` as the baseline; zero added replay overhead by
  design.
- Measurement notes: `wall ≈ internal` (the predict() Python pre/post is ~0.1 ms — negligible).
  Absolute latency is **warmup/clock sensitive**: the RTX 5090 SM clock ramps toward its 3105 MHz peak
  over a run, so a short warmup reads ~1–2 ms high (e.g. Pi0.5 FP16 internal_p50 34.0 ms at warmup=30
  vs 33.1 ms / min 32.2 ms at warmup=200). Warmup, calibration, and clock pinning stay under the
  caller's explicit control (the contract does not touch capture/warmup); the **flag on-vs-off A/B is
  parity at any warmup level** — that is the acceptance signal, not the absolute number.

### 8.4 Opt-in integration (default path byte-identical, additive-only)
- [x] Qwen3.6: `FLASHRT_QWEN36_USE_EXEC=1` (decode + spec + rollback)
- [x] Pi0.5 FP16: `FLASHRT_PI05_USE_EXEC=1` (`Pi05PipelineFP16`)
- [x] Pi0.5 FP8/default: `FLASHRT_PI05_USE_EXEC=1` (`Pi05Pipeline`)
- [x] Flag unset → original path runs unchanged (verified: identical output + latency)

### 8.5 Deliberately OUT of scope (upper layer; the spec only adapts)
Validated against the real VLA feature surface — all are capture/setup-time and stay in the frontend;
each resolves to "a captured graph (+ batch/CFG/decoder variants) + buffers (scales/state)", which the
spec already expresses:
- autotune (`autotune_gemms`), FP8 / int8-static / AWQ calibration, multi-frame (N>1) dataset
  calibration → run before `record_infer_graph`; the layer only adopts the resulting graph
- in-place recalibration without recapture → scales are `Buffer`s; adopted exec stays valid
- `set_prompt_batch` / `set_batched_mode` (B=2 CFG) → a ShapeKey B-variant (another adopted graph)
- CFG/RL pipelines, temporal-KV decoder-only graph → additional adopted graphs
- sessions, schedulers, OpenAI/MCP, sensor/cadence orchestration, interruption policy

### 8.6 Build / reproduce (in-container)
```
# exec layer (seconds)
cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release && cmake --build exec/build -j
# flash_rt fa2 with fp16 (needed by Pi0.5 FP16): default FA2_DTYPES already fp16;bf16
cmake -S . -B build -DGPU_ARCH=120 && cmake --build build -j --target flash_rt_fa2
# tests
python exec/tests/test_exec.py
python exec/tests/test_plan_vla.py
python exec/tests/test_adopt_qwen36.py          # needs FLASHRT_QWEN36_{NVFP4,MTP}_CKPT_DIR
python exec/tests/test_adopt_pi05.py  [--fp8]    # needs a pi05 ckpt
```

### 8.7 Known remaining (not blocking v1 acceptance)
- Long-context Qwen3.6 spec paths (≥512-token FP8-KV / TQ routes) not yet routed through exec
  (short BF16/spec path is)
- `serving/robot_host/` interrupt demo, `serving/llm_agent/` OpenAI shell, Rust shell — upper layer
- Full 64K–256K Qwen TTFT/decode re-alignment with the doc (512–32K done)
- Merge `spec/exec-contract` back to main

---

## 9. Mechanism-not-policy: the red line and PR review checklist

This section is the maintenance contract for the execution layer and everything built on top of it
(`serving/`). It is the rule every contract-touching or serving PR is reviewed against. Keep it short
on purpose — if a change needs a longer justification than this, it is probably policy.

### 9.1 The red line

1. **The contract fixes MECHANISM, never scenario policy.** Three primitives plus one key:
   `Buffer` / `Graph` / `Plan` / `ShapeKey`. The **only** allowed extensions are ShapeKey semantics
   and the *number* of buffers/graphs. Anything else that wants a field is policy.
2. **If a scenario pushes you to add a field or verb to `exec/`, that field is policy.** Push it back
   to `serving/` or user code. There is no session, KV, cache, schedule, batch-policy, protocol, agent,
   or robot concept in the contract, and there must never be one.
3. **Policy lives above the contract** (in `serving/` or the user host), and includes: sessions and
   prefix/KV reuse, eviction/scheduling, batching policy, OpenAI/MCP protocol and tool-call parsing,
   agent/ReAct loops, robot episode/cadence/interruption orchestration, and prompt routing.
4. **Capture-time intelligence stays in the Python frontend.** Autotune, FP8/NVFP4/AWQ calibration,
   multi-frame calibration, CFG-batched capture, and warmup all run before graph capture; the contract
   only **adopts** the resulting instantiated exec and owns replay-time. The contract never captures on
   behalf of a scenario.
5. **`exec/` is a top-level sibling of `csrc/` with zero `csrc` dependency.** It is kernel-agnostic: it
   only sees streams/graphs/events and replays whatever was captured (kernel, torch op, or raw CUDA).
   It must not `#include` or link csrc kernels.
6. **Additive-only at the integration seam.** Existing kernels, bindings, loaders, and graph-capture
   paths are not modified; new behavior is new methods / new files / new ShapeKey variants, gated by an
   opt-in flag, with the default path byte-identical. A shared-method edit that changes behavior on a
   path other than the new one is the one thing to flag, justify, and back with before/after numbers.
7. **Cut graphs only at real seams** — data/cadence boundaries, host control-flow returns, or reuse
   points — never to fill scheduling bubbles. Each cut materializes one HBM round-trip and can erase a
   fusion win.

### 9.2 PR review checklist (contract-touching or serving example)

- [ ] `git diff main -- exec/` is empty, **or** the change only adds a Buffer/Graph/Plan/Event/ShapeKey
      mechanism (no session/KV/schedule/protocol/scenario field, verb, or struct member).
- [ ] The serving/example layer does not import or call into the scenario's behavior from inside the
      contract; orchestration (sessions, loops, interruption, tool calls) is in `serving/` only.
- [ ] GPU/model state stays owned by the frontend; the serving layer holds metadata only (token
      journal, cache plan, episode bookkeeping) — never device buffers it allocated behind the contract.
- [ ] Capture/calibration/warmup stays in the frontend; the contract only adopts. No new capture path
      was added to `exec/`.
- [ ] The change is additive: new methods/files/flags, default path unchanged. If a shared method was
      edited, the PR lists the affected non-new paths and shows output is preserved (bit-identical /
      token-exact) and reports before/after latency.
- [ ] Correctness gate: the new path produces output identical to the pre-existing path it replaces
      (bit-identical for deterministic LLM decode, token-exact for spec decode, cosine ≥ 0.999 for VLA
      diffusion), with a runnable command in the PR.
- [ ] Optional dependencies in tests use `pytest.importorskip(...)` so they skip (not fail) when a
      serving-only dependency (e.g. `fastapi`) is absent from a kernel-only environment.
- [ ] Public repo hygiene: English docs/code/comments, no local paths, no machine/container/env names,
      no generated build outputs or checkpoints committed.

### 9.3 Worked example — `serving/qwen36_agent` (passes the checklist)

The Qwen3.6 agent-serving host is the reference application of this red line, and is what a reviewer
should compare a new serving example against:

- **`exec/` untouched** — `git diff main -- exec/` is empty; the agent adds zero contract verbs.
- **Pure policy layer** — `server.py` is a thin FastAPI shell; `service.py`/`session.py`/`prefix.py`
  own session cache, exact token-prefix reuse, and request planning; `tool_stream.py` parses tool
  calls without leaking partial JSON. All of this is policy and lives only in `serving/qwen36_agent/`.
- **State ownership respected** — `SessionRecord` explicitly tracks only the token journal and cache
  metadata; the GPU KV / linear-attention / MTP state stays owned by the Qwen frontend.
- **Drives the model through the frontend, not the contract** — `qwen36_engine.py` is a thin adapter to
  the `AgentEngine` protocol; the contract is reached (optionally) inside the frontend via
  `FLASHRT_QWEN36_USE_EXEC`, not from the serving layer.
- **Additive frontend split** — new `*_agent` / `*_committed_stream` methods; the one shared-method edit
  (`_long_tq_effective_k`, a speculative-K tuning heuristic) is verify-corrected, so decode output is
  unchanged and only speed is affected.
- **Verified** — policy + OpenAI HTTP surface: `tests/test_qwen36_agent_serving_policy.py` and
  `tests/test_qwen36_server_warmup.py`; real-model equivalence:
  `tests/test_qwen36_agent_gpu_split.py` asserts the split prefill + committed-stream path is
  **token-identical** to the old full-generate path (short, long, long-append, long-text).
