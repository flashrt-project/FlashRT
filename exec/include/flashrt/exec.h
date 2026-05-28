/* FlashRT Execution Contract — public C ABI (the "spec").
 *
 * The common execution layer over FlashRT's "kernelize + CUDA Graph replay"
 * core. It fixes MECHANISM only, never scenario POLICY.
 *
 *   IS  : a replayable graph node + named I/O buffers + zero-copy buffer
 *         hand-off + shape-variant select + multi-stream/event/priority +
 *         imperative host-driven replay.
 *   NOT : sessions, KV append/fork/evict semantics, schedulers, OpenAI/MCP
 *         protocols, sensor/cadence orchestration, family/latency tags.
 *
 * Layering: this layer has ZERO dependency on csrc/ kernels. It captures
 * whatever the `record` callback enqueues onto a stream (FlashRT kernels,
 * torch ops, or raw CUDA) and only ever sees streams / graphs / events. That
 * independence is exactly why it lives in its own top-level `exec/` layer,
 * sibling to (not inside) the kernel layer csrc/. Its hardware backend axis
 * (cuda today, hip/... later, under exec/backend/) is orthogonal to the
 * per-kernel hardware backends in csrc/.
 *
 * Pure C ABI so Python (dev), Rust (server shell), C++ and on-robot hosts can
 * all link the same `libflashrt_exec`. Capture intelligence (autotune /
 * calibrate) stays in the Python frontend; this layer owns only the
 * replay-time contract + buffer registry.
 *
 * Design rationale: docs/exec_contract.md
 */
#ifndef FLASHRT_EXEC_H
#define FLASHRT_EXEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Opaque handles                                                     */
/* ------------------------------------------------------------------ */
typedef struct frt_ctx_s*    frt_ctx;    /* owns arena + stream/event pool */
typedef struct frt_buffer_s* frt_buffer; /* named device memory region     */
typedef struct frt_graph_s*  frt_graph;  /* ShapeKey -> graph-exec table    */
typedef struct frt_plan_s*   frt_plan;   /* dumb DAG of (graph, key) nodes  */
typedef struct frt_event_s*  frt_event;  /* cross-stream sync point         */

/* ShapeKey: opaque u64 encoding (B, S, ...). Batch is NOT a new axis — it is
 * just one field of the key. Encoding scheme is owned by the caller; the
 * framework only uses it for variant lookup equality. */
typedef uint64_t frt_shape_key;

/* All int-returning functions: 0 = ok, negative = error code. Variant or
 * binding lookups that miss return an error — never silently no-op. */
typedef enum {
    FRT_OK              =  0,
    FRT_ERR_INVALID     = -1,  /* bad handle / null arg                     */
    FRT_ERR_NO_VARIANT  = -2,  /* replay/execute for an un-captured key     */
    FRT_ERR_UNBOUND     = -3,  /* graph port has no buffer bound            */
    FRT_ERR_CAPTURE     = -4,  /* graph capture/instantiate failed          */
    FRT_ERR_BACKEND     = -5,  /* underlying hardware backend error         */
    FRT_ERR_OOM         = -6,  /* arena allocation failed                   */
} frt_status;

/* ------------------------------------------------------------------ */
/* Context + streams                                                  */
/* ------------------------------------------------------------------ */
frt_ctx frt_ctx_create(void);
void    frt_ctx_destroy(frt_ctx);

/* Get/create a stream with the given hardware priority.
 *   priority: 0 = normal; more-negative = higher priority (CUDA convention).
 * Returns a stream_id (>=0) used by replay/plan, or negative frt_status.
 * Mechanism only: the ABILITY to prioritize is here; WHICH model gets which
 * priority is host policy. */
int frt_ctx_stream(frt_ctx, int priority);

/* Wrap an externally-owned stream (e.g. a torch stream's cuda handle) as a
 * stream_id usable by replay/plan. Non-owned: frt never destroys it. Lets the
 * exec layer replay onto the SAME stream the host framework already uses, so
 * existing stream choreography (wait_stream, etc.) is preserved. */
int frt_ctx_wrap_stream(frt_ctx, void* external_stream);

/* Events: the hardware sync primitive for IMPERATIVE cross-stream ordering,
 * used directly by the host (not only inside a Plan). This is what lets a
 * robot host express "make the action stream wait for the ASR stream's last
 * replay" or fan-in a snapshot stream — the real pattern Qwen3.6 already uses
 * (snapshot on a side stream, wait_stream both ways) around spec-decode.
 * Mechanism only. */
frt_event frt_ctx_event(frt_ctx);
void      frt_event_destroy(frt_event);
int       frt_event_record(frt_event, int stream_id);  /* record on stream  */
int       frt_stream_wait (frt_ctx, int stream_id, frt_event); /* stream waits */

/* ------------------------------------------------------------------ */
/* Buffer — the only "state" primitive.                               */
/*   KV cache, vision cache, subgoal/prompt embedding, scales: all are        */
/*   Buffers. append/fork/evict are caller logic on top of dptr; the          */
/*   framework owns lifetime + pointer only, never the verbs.                 */
/* ------------------------------------------------------------------ */
frt_buffer frt_buffer_alloc(frt_ctx, const char* name, size_t bytes);
/* Wrap an externally-owned device pointer (e.g. a torch tensor's data_ptr). */
frt_buffer frt_buffer_wrap (frt_ctx, const char* name, void* dptr, size_t bytes);
void*      frt_buffer_dptr (frt_buffer);   /* stable device pointer          */
size_t     frt_buffer_bytes(frt_buffer);
const char* frt_buffer_name(frt_buffer);

/* Async device-to-device copy on a stream. Convenience so a torch-free host
 * (robot / Rust) can do snapshot+restore (spec-decode KV rollback) and
 * buffer-swap without dropping to raw CUDA. Mechanism only; the host owns
 * offsets and when to call it. */
int frt_buffer_copy(frt_ctx, frt_buffer dst, size_t dst_off,
                    frt_buffer src, size_t src_off, size_t bytes, int stream_id);

/* ------------------------------------------------------------------ */
/* Graph — a table of ShapeKey -> captured graph-exec.                */
/*   The key is EXACT, not bucketed: Qwen3.6 keys a decode graph per exact    */
/*   cur_pos (and verify per (cur_pos,K)). Bucketing (e.g. seqlen {512,1024}) */
/*   is just ONE caller strategy on top of an exact key — not a framework     */
/*   concept. `max_variants`>0 bounds the table with LRU eviction (Qwen uses  */
/*   256); 0 = unbounded. Eviction frees only the evicted graph-exec, never   */
/*   the bound buffers (those are independent frt_buffer). */
/* ------------------------------------------------------------------ */
frt_graph frt_graph_create(frt_ctx, const char* name, size_t max_variants);
void      frt_graph_destroy(frt_graph);

/* Capture one variant. The framework wraps stream-level begin/end-capture
 * (RELAXED mode) around `record`, which must enqueue the kernel launches for
 * this variant onto the passed stream and nothing else. `stream` is an opaque
 * backend stream the framework supplies. Re-capturing an existing key
 * replaces that variant. */
int frt_graph_capture(frt_graph, frt_shape_key key,
                      void (*record)(void* user, void* stream), void* user);

/* Adopt an externally-owned, already-instantiated graph-exec (e.g. from
 * torch.cuda.CUDAGraph.raw_cuda_graph_exec()). frt registers it under `key`
 * and drives replay, but does NOT own it: LRU eviction and frt_graph_destroy
 * never free an adopted exec — the external owner does. This is the
 * torch-friendly path: capture + allocator handling stay in the framework
 * that owns the graph, while the exec layer owns the variant table + replay.
 * Re-adopting/re-capturing an existing key replaces that variant. */
int frt_graph_adopt(frt_graph, frt_shape_key key, void* external_graph_exec);

/* Bind a named I/O port to a buffer. Two graphs sharing one buffer on
 * matching ports = zero-copy hand-off (this is the entire multi-subgraph /
 * multi-model wiring mechanism). Bindings are graph-wide (shared by all
 * variants). */
int frt_graph_bind(frt_graph, const char* port, frt_buffer);

/* Select the variant for `key` and replay it on `stream_id`. Missing key ->
 * FRT_ERR_NO_VARIANT (never a silent no-op). Imperative: the host may call
 * this on demand — this is what makes interruption possible at graph
 * boundaries. */
int frt_graph_replay(frt_graph, frt_shape_key key, int stream_id);

/* Introspection (for warmup / host policy deciding what to capture). */
int frt_graph_has_variant(frt_graph, frt_shape_key key); /* 1 yes, 0 no */

/* ------------------------------------------------------------------ */
/* Plan — dumb DAG. Data dependencies only: NO priority/deadline/preempt.  */
/*   For the static inner DAG of one inference (vision->encoder->action).     */
/*   The interruptible OUTER loop lives in the host, not here.                */
/* ------------------------------------------------------------------ */
frt_plan frt_plan_create(frt_ctx);
void     frt_plan_destroy(frt_plan);

/* Append a node: replay `graph`@`key` on `stream_id`. Returns node index
 * (>=0) or negative frt_status. */
int frt_plan_add(frt_plan, frt_graph graph, frt_shape_key key, int stream_id);

/* Declare a cross-stream dependency: node_idx waits (via event) until
 * dep_node_idx has completed. Within one stream, add-order already implies
 * sequencing; use this only across streams. */
int frt_plan_after(frt_plan, int node_idx, int dep_node_idx);

/* Execute the whole DAG once. `key` is the default for nodes added with the
 * sentinel FRT_KEY_INHERIT; nodes with an explicit key ignore it. */
int frt_plan_execute(frt_plan, frt_shape_key key);

#define FRT_KEY_INHERIT ((frt_shape_key)~0ull)

/* Block until all streams touched by this plan are idle. Optional: hosts that
 * pipeline across replays may prefer their own event-based waits. */
int frt_plan_sync(frt_plan);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FLASHRT_EXEC_H */
