# FlashRT Qwen3.6 agent serving — production architecture (seam-locking design)

> **Status: APPROVED — decisions in §7 resolved (thinnest, coding-agent-first).**
> This document locks the *interfaces* (seams) of the
> production agent server so the core is designed once and never rewritten, then
> stages the *implementation* correctness-first. It builds on
> [`serving_design.md`](serving_design.md) (the capsule rationale) and is governed
> by the mechanism-not-policy rule in [`exec_contract.md`](exec_contract.md) §9.
> Nothing here adds a session/cache/schedule verb to the execution contract; all
> of it lives in `serving/`. The one possible contract addition (host-backed
> Buffer + cross-space copy) is already sanctioned by serving_design §3 and is
> needed only by stage D.

---

## 0. Scope and non-goals

**In scope (this document locks the interfaces for):** a single-node,
latency-first OpenAI-compatible server for *agent* workloads — coding agents,
multi-agent setups, and long-horizon tasks — on one consumer/edge GPU, around the
existing single hot stateful Qwen3.6 frontend.

**Explicitly NOT built (and the interfaces must not assume them):**

- a **radix tree / paged-block KV** — impossible on hybrid recurrent state and
  fatal to full-graph replay (serving_design §1, §3). Multi-session is
  capsule swap-in/out, not paged concurrency.
- a **heavy scheduler / continuous batching** — the engine is a single serial
  stateful consumer; the worker is a thin serial loop, not a vLLM-style scheduler.
- **multi-GPU routing now** — the interface must leave room for it (one worker =
  one GPU/context, router by session affinity later) but we do not build it.

The discipline: lock the seams comprehensively *now*; implement the thin correct
version behind them; defer the heavy implementations without blocking them in the
interface. "Design once" means the seams, not every feature.

---

## 1. What the workloads demand, and the three invariants

Target workloads and the pressure each puts on the server:

- **Coding agent / multi-agent.** Large reused prefix (system + tool schemas +
  repo index, 10k–50k tokens); every turn a small user/tool/diff/log suffix; tool
  calls always present; concurrent requests from parallel sub-agents; try/revert
  branches.
- **Long-horizon task.** Minutes-to-hours sessions; sustained decode near
  `max_seq`; must survive client disconnect / server restart; KV memory pressure;
  the user must be able to cancel a long generation cleanly.

Three invariants the design must never violate (they come from the engine, not
taste):

1. **One stateful engine.** The Qwen3.6 frontend is a single hot GPU state
   (KV + recurrent + conv + MTP). Exactly one request mutates it at a time.
2. **KV must never lead the visible transcript.** If generation aborts (disconnect,
   exception, cancel, stop-token lookahead) the GPU state may be ahead of the
   committed journal; that session is then unsafe to hot-append and must be
   invalidated (rebuild/restore next turn). The current `completed=False →
   hot_session_id=None` guard in `service.py` is this invariant; every new path
   must preserve it.
3. **Full-graph replay is positional.** Decode graphs are keyed by exact
   `(cur_pos, draft_k, mtp_cache_base)`; a request can only be cancelled at an
   accept boundary, and prefix reuse on hybrid state is snapshot/restore, never
   block gather.

### Measured finding (2026-05-31): EOS turns can stay hot if the boundary is committed

The production stream now stops at the visible stop-token boundary and does not
leave speculative lookahead ahead of the client-visible transcript. A second
issue was Qwen3.6's hidden non-thinking generation prompt
(`<think>\n\n</think>\n\n`): OpenAI clients replay only visible assistant text,
so the full rendered prompt can differ from the internal hot KV journal by a few
tokens even when the message prefix is semantically identical. The serving layer
now appends the newly added message suffix after a known-equivalent visible
message prefix, instead of forcing a full render byte-prefix match.

This is the R1 continuation contract: the hot journal may retain internal
non-thinking control tokens, and continuation happens from that committed
boundary. A naive cold canonical render that strips those internal tokens is not
the reference for hot reuse.

Verified end-to-end on realistic EOS-terminated long-sequence turns:
`append(cold) → message_append → message_append`, with `new_prefill_tokens`
dropping from thousands to tens and TTFT staying ~70-150 ms. Capsule restore is
still the right primitive for shared-prefix reuse across fresh sessions,
branches, restarts, or non-hot workers; it is no longer required for the normal
single hot coding-agent loop.

The frontend-level correctness gate is token-exact: decode with a stop token
inside a speculative chunk, rollback to the stop boundary, append a suffix, and
decode again must match prefill to the exact stopped boundary plus the same
suffix. This is covered for short and long routes in
`tests/test_qwen36_agent_gpu_split.py`.

### Measured finding (2026-05-31): session id is not an ecosystem contract

OpenAI-compatible clients generally resend the full message/tool history and do
not carry a backend-specific session id. A FlashRT session id is therefore only a
native affinity hint. The production path must be automatic and content-addressed:
tokenize the incoming OpenAI request, try to attach it to the current hot
token/message prefix, then fall back to capsule restore or cold prefill. This
matches the ecosystem expectation set by OpenAI prompt caching, vLLM Automatic
Prefix Caching, and SGLang prefix caching: clients may provide namespace hints
(`prompt_cache_key`, `cache_salt`), but they should not need FlashRT-specific
fields for normal prefix reuse.

The implemented v1 remains capsule/hot-state granular, not block-radix. It does
not add a KV-block table or scheduler to the execution contract.

---

## 2. The five seams (the heart of this design)

Each seam is an interface to lock now. Sketches are design-level Python, not final
code.

### Seam 1 — engine ⟷ worker boundary

One worker thread owns the engine exclusively. The async HTTP layer only parses,
enqueues, and forwards; it never touches the engine. This moves blocking GPU work
off the event loop (fixes `/health` starvation) and gives one place for admission,
ordering, and cancellation.

```python
@dataclass
class WorkItem:
    request: AgentRequest
    submitted_at: float
    cancel: "CancelToken"          # cooperative; checked at accept boundaries
    sink: "ResultSink"             # future (non-stream) or bounded channel (stream)

class EngineWorker:
    """Single thread; owns AgentEngine + SessionRegistry. Serial consumer."""
    def submit(self, item: WorkItem) -> None: ...      # called from event loop, non-blocking
    def _run(self) -> None: ...                        # pop queue → prefill → decode → emit
    def depth(self) -> int: ...                        # queued count, for /health + admission

class CancelToken:
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...
```

- Non-stream request: handler `await`s a future the worker resolves.
- Stream request: worker pushes `DecodeChunk`s into a **bounded** channel; the SSE
  handler drains it. A slow client fills the channel → the worker applies
  backpressure or (policy) drops the stream and invalidates the session — GPU
  progress is decoupled from client read.
- Cancellation: `cancel.cancelled` is checked at each accept boundary inside
  `generate_stream`; on cancel the worker stops, then runs the existing
  invalidation guard (invariant 2).

The worker is intentionally a serial loop, **not** a scheduler. Multi-GPU later =
N workers + an affinity router in front of `submit`; `AgentService`/engine
unchanged.

### Seam 2 — request lifecycle FSM

Every request carries a small state record with a timestamp per transition. This
*is* the observability schema (Seam 5) and the cancel/reject semantics in one.

```
  ENQUEUED ──(admission ok)──▶ QUEUED ──(worker picks up)──▶ PREFILL ──▶ DECODE ──▶ DONE
      │                          │                              │           │
      └──(admission reject)──▶ REJECTED                         │           ├──(cancel/disconnect)──▶ CANCELLED
                                 (429 / 413 / 503)              └───────────┴──(engine error)──────▶ ERROR

  Invariant: any exit that is not DONE-at-a-clean-boundary  ⇒  invalidate hot session.
```

Transitions stamp: `enqueued_at, started_at, first_token_at, finished_at`, plus
`terminal_state`. CANCELLED and ERROR both trip invariant 2.

### Seam 3 — automatic prefix / capsule policy interface

Extend the existing `SessionRegistry` + `PrefixPlan` (do not replace), but do
not make a client session id the compatibility contract. The worker first asks
an automatic prefix policy to match the incoming tokenized OpenAI request against
the current hot state; explicit session ids are affinity hints. Add the `restore`
and `fork` actions and a capsule store; this is serving_design §6 steps 2–3 made
into a locked interface.

```python
# PrefixPlan.action ∈ {exact, append, message_append, restore, rebuild, fork, truncate}
#   restore  : incoming extends a PINNED capsule (not the hot session) → restore + suffix prefill
#   fork     : restore one capsule into an independent branch (tree-of-thought / retry)

class CapsuleStore:
    def pin(self, key: str, capsule, *, budget_bytes: int) -> None: ...
    def get(self, key: str): ...
    def evict_lru(self) -> None: ...
    def footprint(self) -> int: ...        # for the budget (Seam 4)

class AutoPrefixPolicy:                    # what the worker asks before each request
    def plan(self, incoming_tokens, *, session_hint, tools, salt) -> PrefixPlan: ...
    def on_commit(self, session, tokens, *, lookahead: int) -> None: ...   # invariant 2
```

Namespace source is `prompt_cache_key > cache_salt > native salt/default`.
Pin source is an OpenAI-side field (`flashrt_pin_prefix`) or a `/v1/sessions`
capsule option. Restore-vs-rebuild and which boundary to pin stay here (policy),
never in the contract.

### Seam 4 — resource budget / admission

One limits object consulted at enqueue (admission) and at pin (capsule budget).
Reject before OOM; never crash.

```python
@dataclass(frozen=True)
class Limits:
    max_prompt_tokens: int
    max_output_tokens: int
    max_active_sessions: int
    max_queue_depth: int
    session_idle_ttl_s: float
    capsule_budget_bytes: int          # GPU + host tiers
# admission → REJECTED(429 queue full / 413 too large / 503 over budget) at Seam 2.
```

### Seam 5 — metrics schema

One structured record per request, derived from Seam 2 timestamps + engine stats.
Emitted as the existing one-line log and an aggregate on `/health` (optionally a
`/metrics` endpoint later).

```
  queued_ms, tokenize_ms, prefill_ms, first_delta_ms, decode_ms, decode_tok_per_s,
  spec_attempts, spec_accepts, accept_length, graph_capture_count,
  prefix_action, cached_tokens, new_prefill_tokens, terminal_state
```

(`GenerationStats.graph_misses` already exists as a placeholder; wire it here.)

---

## 3. How the gaps map onto the seams

| gap | what it is | seam(s) that make it safe | stage |
| --- | --- | --- | --- |
| **B** | capsule not wired into the server (pin/restore policy, VRAM budget). The frontend capsule API is shipped + bit-exact (`test_qwen36_agent_capsule.py`). This is the shared-prefix reuse lever for fresh sessions, branches, restarts, and non-hot workers; the single hot EOS loop now uses `message_append`. | Seam 3 + Seam 4 | **1 (first feature)** |
| C | clean cancellation + KV-never-leads-transcript on abort | Seam 1 + Seam 2 | 1 (correctness) |
| F | resource limits / reject-before-OOM (capsule budget needed by B) | Seam 4 | 1–2 |
| G | observability (queued/tokenize/capture/spec) | Seam 2 + Seam 5 | 2 |
| (thin worker) | move GPU work off the event loop; admission | Seam 1 + Seam 2 | 2 |
| A | tool-call / text turn contiguous append. The hot EOS loop is now viable: stop-aware committed stream + message-boundary suffix append keeps visible OpenAI history connected to the internal KV journal, with safe rebuild fallback on divergence. | Seam 3 | shipped, keep hardening |
| D | long-horizon resume: capsule → host RAM / disk | Seam 3 + the one exec addition (§6) | deferred, interface reserved |
| E | branch / undo as agent ops (fork / time-travel) | Seam 3 | deferred, interface reserved |

---

## 4. Correctness gates (non-negotiable)

- **A:** tool-call multi-turn (assistant `content=null` + `tool_calls`, then `tool`
  result) resent as full history is **token-exact** vs a cold full prefill of the
  same rendered transcript; the suffix tokenizer reproduces the committed token
  prefix or honestly reports `rebuild`. New test alongside
  `test_qwen36_agent_gpu_split.py`.
- **C:** a cancelled / disconnected request leaves no session marked hot
  (invariant 2); the next turn rebuilds/restores and is token-exact.
- **B/E:** capsule restore / fork stays bit-exact (already gated by
  `test_qwen36_agent_capsule.py`); the *server* path inherits the same assertion.
- **no-regression:** default path byte-identical; existing policy + warmup +
  gpu-split suites stay green. Additive, opt-in.

---

## 5. Staged implementation (behind the locked seams)

Correctness first; the user-visible levers next; heavy work deferred but its
interface reserved. Each stage has its own acceptance gate.

Reordered after the EOS finding: capsule (B) is the coding-agent lever and goes
first; the tool-call contiguous-append (A) is demoted to optional.

- **Stage 1 — capsule in the server + the correctness it needs (gaps B, C, and the
  F budget B depends on).** Wire the shipped frontend capsule API into the policy
  layer: a `CapsuleStore` (pin + small LRU + a byte budget so an over-budget pin is
  rejected, not OOM), a `restore` `PrefixPlan` action, and the `flashrt_pin_prefix`
  request field; snapshot at a chunk-aligned boundary (`capsule_aligned_len`) for
  the long route. Make cancel/abort a first-class transition reusing the existing
  invalidation guard (C). Gate: pinned-prefix `restore + append(suffix) + decode`
  is **token-exact vs a cold full prefill** of the same prompt (the
  `test_qwen36_agent_capsule.py` contract, now at the server level); over-budget
  pin rejected; abort leaves no hot session; default path byte-identical.
- **Stage 2 — thin worker + admission + metrics (Seams 1, 2, 4, 5; gaps F, G).**
  Move GPU work onto one worker thread; HTTP enqueues; bounded queue + admission
  (reject, not crash); lifecycle FSM + metrics. Gate: `/health` responsive during a
  long decode; queue-full → 429; metric record complete; single-stream latency
  unchanged.
- **Optional — A (tool-call contiguous append).** Only helps `max_tokens`-capped
  turns; revisit if a non-EOS streaming pattern needs it.
- **Deferred, interface reserved — D (capsule→host/disk resume), E (fork/undo as
  agent ops), multi-GPU router.** No code now; Seam 1/3 and the §6 exec addition
  leave room. Built when a workload needs them.

---

## 6. The single allowed execution-contract addition

Only stage D needs it: **host-backed `Buffer` + device↔host async copy** (D2H/H2D),
so a capsule can be parked off-GPU and a long-horizon session can resume after a
restart. This is still "named memory + copy" — mechanism, not policy
(serving_design §3, exec §9). No `session` / `cache` / `schedule` verb enters the
contract. Until D, capsules stay GPU-resident and the contract is untouched.

---

## 7. Decisions (resolved)

Resolved thinnest-first, coding-agent-first; every "later" option keeps its
interface hook so it can land without a core rewrite.

1. **Concurrency contract for multi-agent:** **(a) fair serial queue now.** N
   concurrent requests (even on different sessions) serialize through the one
   worker. Capsule-swap per request (b) is a stage-3+ policy once `CapsuleStore`
   exists; Seam 1 leaves the hook.
2. **Cancel granularity:** **accept-boundary cancel only** now. A hard decode
   deadline (`max_decode_ms`) is a later worker policy; the `CancelToken` /
   `Limits` interfaces reserve it.
3. **Pin API surface:** **`flashrt_pin_prefix` request field** now (no new
   endpoint — thinnest for a coding agent that pins its system+repo prefix once).
   An explicit `/v1/sessions` capsule endpoint is a later addition.
4. **Metrics:** **fold into the `/health` aggregate** now (plus the existing
   per-completion log line). A separate `/metrics` (Prometheus-style) is later.
5. **Relationship to serving_design:** this **extends** serving_design §6/§10 —
   serving_design stays the capsule rationale; this is the production engineering
   layer. No supersession.

Guiding principle for all of the above: usability and experience for one coding
agent first; thinnest core that is correct; hooks (not implementations) for the
rest.
