# FlashRT Execution Contract (common-layer spec)

> Status: design draft · 2026-05-28 · branch `spec/exec-contract`
> This layer defines FlashRT's **common execution contract**: a clean, minimal C ABI that
> takes the already-proven "kernelize + CUDA Graph replay" seam — today an implicit convention
> scattered across the Python frontends — and turns it into an explicit, embeddable common layer.
> It fixes **mechanism only, never scenario policy**, and has **zero dependency on the csrc kernel layer**.

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

### IS NOT (policy — lives in examples and user hosts)
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
examples/robot_host/               # scenario: multi-model + interrupt (VLA + ASR), policy-layer demo
examples/llm_agent/                # scenario: session/KV/OpenAI shell (folds in existing *_openai_server.py)
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
- Today Pi05 is single-stream; multi-stream/multi-rate is forward-looking for `examples/robot_host/`,
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
3. **One multi-node + interrupt example (1-2 weeks)**: `examples/robot_host/` chains Pi05
   vision→llm→action across streams via a Plan, and demonstrates concurrent ASR + subgoal-buffer
   overwrite interruption. Acceptance: Pi05 latency does not regress; **no scenario field was added to
   the common layer throughout** (if one was, go back to step 1).

The Rust shell, OpenAI, and scheduler all live in examples / the upper layer, off this main line.

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
verified in-container on RTX 5090 (sm_120) / pi0-stablehlo-test (torch 2.9, CUDA 13, py3.12).

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
- `examples/robot_host/` interrupt demo, `examples/llm_agent/` OpenAI shell, Rust shell — upper layer
- Full 64K–256K Qwen TTFT/decode re-alignment with the doc (512–32K done)
- Merge `spec/exec-contract` back to main
