# FlashRT serving design — graph-replay-native execution-state capsules

This document explains the **signature serving idea** of FlashRT and why it is
different from the prefix-caching designs in vLLM and SGLang. It is the design
rationale for everything under [`serving/`](../serving/); the mechanism it builds
on is the execution contract in [`docs/exec_contract.md`](exec_contract.md), and
the mechanism-not-policy rule in that document's §9 governs every host here.

---

## TL;DR — the one idea

> Everyone else gives up full-graph capture to get prefix flexibility (they run
> eager/piecewise so the attention kernel can gather KV from arbitrary paged
> blocks). **FlashRT keeps full-graph capture and buys prefix reuse back with
> execution-state *capsules*.**

A **capsule** is the full, restorable execution state at a committed token
boundary — not a KV block. Because we capture the whole forward as a CUDA graph
over contiguous static buffers, the entire state at a boundary is a fixed set of
named buffers. Freeze them and you can `restore`, `fork`, and `time-travel` a
session — none of which a block/paged engine can do, because it never captured
the whole state in the first place.

```
  cold prefill ONCE            snapshot                  restore (one copy)         fork (share the prefix)
  ┌──────────────────┐        ┌──────────┐              ┌────────────┐             ┌──────────┐
  │ system + tools   │  ─────▶ │ CAPSULE  │── restore ──▶│ session A   │   ┌───────▶│ branch 1 │
  │ + repo index/    │        │ (frozen  │              │ warm start  │   │        └──────────┘
  │   project memory │        │  state)  │── restore ──▶│ session B   │ ──┤
  │  (10k–50k tokens)│        └──────────┘              └────────────┘   └───────▶│ branch 2 │
  └──────────────────┘             │                                              └──────────┘
        ~seconds              snapshot once,            ~milliseconds (bandwidth-bound copy),
       (compute-bound)        restore many times        independent of prefix length
```

The same capsule mechanism is the spine of all three serving examples:

| scenario | capsule verb it uses |
| --- | --- |
| coding agent (`qwen36_agent/`) | **restore** a pinned shared-prefix capsule on every fresh session/turn |
| RL rollout (`robot_recap/`) | **restore-to-initial** on each episode reset (no recapture) |
| reasoning branches / retries | **fork** one prefilled prefix into N continuations |

---

## 1. Why we cannot copy vLLM / SGLang (and why that is good)

vLLM Automatic Prefix Caching is a paged **KV block pool** (hash blocks by parent
+ tokens + salt, ref-count, evict). SGLang RadixAttention is a **radix tree of KV
prefixes** (find longest cached prefix, compute only the suffix). Both are
excellent — for their home turf: many tenants, high concurrency, high throughput,
paged KV. References:

- vLLM Prefix Caching: <https://docs.vllm.ai/en/stable/design/prefix_caching/>
- SGLang RadixAttention: <https://docs.sglang.ai/> (RadixAttention / Prefix Caching / HiCache)

Two hard constraints make a block/radix design the **wrong** mechanism for
FlashRT — not by taste, but by construction:

**(a) Hybrid state is not prefix-addressable.** Qwen3.6 is a hybrid
linear-attention / full-attention model. Full-attention KV is positional and
*could* be sliced per prefix. But the linear-attention **recurrent state** and
**conv state** are a fold over the whole prefix: the state at position N is a
function of all N tokens, with no "first 1000 tokens" sub-slice. The only way to
reuse a prefix is to **snapshot the recurrent/conv/MTP state at the boundary and
restore it**. A radix tree of KV blocks cannot represent recurrent-state reuse at
all.

**(b) CUDA Graph replay forbids arbitrary block gather.** Paged KV works because
vLLM/SGLang run eager/piecewise, so the attention kernel reads a block table and
gathers KV from arbitrary physical blocks every forward. FlashRT captures the
**whole forward as a graph over absolute device pointers**; replay reuses the
exact same addresses. Our hand-tuned contiguous kernels have **no block-table
indirection** — that is precisely why they are fast. Pointing attention at a
different KV region at replay time would mean recapturing.

So block-based prefix caching is both impossible (hybrid state) and pointless
(it would force us to abandon graph capture). Snapshot/restore is the only
correct mechanism — and it happens to be strictly more capable.

---

## 2. The capsule — one mechanism, four verbs

A capsule freezes a **committed execution boundary**. We already enumerated its
contents in [`serving/qwen36_agent/frontend_split.md`](../serving/qwen36_agent/frontend_split.md)
("State that defines a boundary"); the capsule is simply that state made into a
storable, restorable object:

```
  ┌──────────────────────── CAPSULE @ committed boundary (pos = P) ───────────────────────┐
  │  metadata (tiny)                 small fixed-size state (cheap to snapshot)            │
  │  • token position / cur_pos      • linear-attention recurrent state                   │
  │  • current/next token            • conv state                                         │
  │  • token-prefix digest + salt    • MTP tail / compact cache + valid range             │
  │  • graph-bucket coverage         • last hidden (seeds MTP)                             │
  │                                                                                        │
  │  KV region (the big, position-growing part)                                            │
  │  • full-attention persistent KV, valid range [0, P)                                    │
  │  • long-context FP8 / TQ dequant-stage valid end                                       │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

Once a boundary is a storable object, four verbs fall out for free:

- **snapshot** — freeze the boundary into a capsule (park to GPU, host RAM, or
  disk).
- **restore** — copy a capsule back into the live frontend buffers, then prefill
  only the suffix after the boundary. Reuses the *same captured graphs* — no
  recapture.
- **fork** — restore one capsule into several live sessions and continue them
  divergently. One prefill of the shared prefix, N branches.
- **time-travel** — restore an *earlier* boundary of the same session: undo the
  last tool call / retry from a checkpoint.

The cost asymmetry is the whole point: **snapshot the small state once is free;
restore is a bandwidth-bound copy that is roughly flat in prefix length; cold
prefill is compute-bound and grows with prefix length.** Restore wins by orders
of magnitude exactly when the prefix is large and shared.

---

## 3. Contract (mechanism) vs serving (policy)

The capsule is a serving-layer concept. It needs almost nothing new from the
contract, which keeps §9's red line intact.

```
  serving/  (policy)   capsule registry: digest match, pin, LRU/evict, when-to-snapshot,
                       which boundary, restore-vs-rebuild decision     ← all here
  ───────────────────────────────────────────────────────────────────────────────────
  flash_rt/ (frontend) snapshot_capsule() / restore_capsule(): copy the boundary buffers
                       (capture/calibration already live here)
  ───────────────────────────────────────────────────────────────────────────────────
  exec/     (contract) Buffer (named memory) + buffer_copy.
                       ONE addition: host-backed Buffer + cross-space async copy (D2H/H2D)
                       so capsules can park off-GPU. A capsule = a set of Buffers + a copy.
```

The only mechanism the contract gains is **host-backed buffers and cross-space
(device↔host) async copy**. That is still "named memory + copy" — mechanism, not
policy. Everything that decides *which* capsule to keep, when to snapshot, and
whether a request restores or rebuilds stays in `serving/` (it extends the
existing `SessionRegistry`). No `session` / `cache` / `schedule` verb enters the
contract.

We deliberately do **not** build:

- a **radix tree** — it solves automatic longest-prefix discovery across *many
  concurrent* sessions; our target is one interactive session plus a few
  explicitly pinned shared prefixes, where explicit pin + linear longest-prefix +
  a small LRU is simpler and more debuggable;
- **paged / block KV** — it would force block-table indirection into the
  attention kernels and break graph replay (see §1);
- **dense per-1K-token checkpoints** — the small state is cheap to snapshot, but
  KV grows with position; snapshotting full KV every 1K tokens does not fit in
  memory. Capsules are taken at *meaningful* boundaries (a pinned shared prefix,
  an episode start, a turn boundary), not on a fixed token grid.

---

## 4. The scenarios — same capsule, different faces

### 4.1 `qwen36_agent/` — coding agent: the pinned shared-prefix capsule

A local coding agent resends the same large prefix every turn — system prompt,
tool schemas, repo index/summary, project memory — then a small new
user/tool/diff/log suffix. Cold-prefilling 10k–50k shared tokens on every fresh
session or branch is the dominant latency. The capsule kills it:

```
  startup (once):  cold prefill [ system + tool schemas + repo index ]  ──▶  PIN capsule  ●
                          (compute-bound, ~seconds)

  every turn / fresh session / retry:
     incoming tokens ── longest-prefix vs pinned capsule ──┐
                                                           │ extends pin?
                   ┌──────────── yes ─────────────────────┴──── no ─────────┐
                   ▼                                                          ▼
        restore ● (one copy, ~ms)                                  rebuild (cold prefill)
        + prefill ONLY the suffix                                  + (optionally) pin a new capsule
                   │
                   ▼
           decode (committed SSE stream, spec-decode accept boundaries)
```

This is the retrofit of the cases the current host falls back on: today a
non-hot session or a divergent/truncated prompt **rebuilds** (cold prefill);
with capsules those become **restore + suffix prefill**. The hot contiguous
append path (already shipped) is unchanged and remains the lowest-latency path
for one continuous session.

### 4.2 `robot_recap/` — RL rollout: episode reset *is* restore-to-initial

The RECAP rollout host already resets model state between episodes "with no
recapture". That reset is exactly a capsule **restore** to the episode-initial
boundary:

```
  episode start ──▶ RUNNING ──(keyboard END / value<thr / timeout)──▶ STOP_INFER
       ▲   ● restore initial capsule          one CHUNK per replay        │
       │                                                                   ▼
   next ep ◀──────── restore-to-initial (●, no recapture) ◀── RECORD(.npz) ◀── robot_reset_to_initial()

   concurrently via ONE exec ctx:  policy(stream P) ‖ value critic(stream C)
```

`reset_state()` in `rollout_host.py` restores the captured policy boundary with no
recapture — the **same capsule mechanism** as the agent, in a different scenario.
The two sides express it at the level that fits their state: the LLM agent has a
model-specific `snapshot_capsule`/`restore_capsule` frontend API (its boundary is a
large hybrid state — KV + recurrent + conv + MTP), while the robot boundary is a
small set of buffers (the diffusion seed; in production also the observation), so
it uses the **execution contract's Buffer snapshot directly** — a capsule is a
`frt` Buffer, snapshot/restore is `frt` device-to-device copy, no new mechanism.
`verify_capsule.py` validates the robot side **bit-identical** (cosine 1.0, exact),
including after the live boundary was overwritten — so episode reset *is* a capsule
restore, evidenced on both sides.

### 4.3 `robot_pi07/` — hierarchy: buffer hand-off (the contrast case)

Not every multi-model pattern needs a capsule. The π0.7 hierarchy is a
**zero-copy buffer hand-off**, not a state snapshot:

```
  PLANNER (low rate) ──subtask (shared Buffer)──▶ ACTOR (high rate) ──▶ actions
                              ▲
        interrupt / verbal coaching: overwrite the subtask buffer (no recapture)
```

This is here to mark the boundary: capsules are for *restoring/forking a whole
session state*; a shared `Buffer` is for *passing a value between live models*.
Both are mechanism the contract already provides; the host picks the right one.

### The unification

```
                       ┌──────────────── ONE capsule mechanism ────────────────┐
                       │   snapshot · restore · fork · time-travel             │
                       └───────────────────────────────────────────────────────┘
                                │                         │
                  LLM agent: restore a pinned    Robot rollout: restore-to-
                  shared-prefix on warm start    initial on each episode
                  / fork branches / undo a turn  / interruptible per-chunk replay
```

The flag is not "we also have prefix caching". It is: **FlashRT sessions are
checkpointable, forkable, and restorable — because we capture full execution
state — and the same capsule serves both long-running LLM agents and robot RL
rollout.**

Evidence status: snapshot + restore are shipped and **bit-exact on both sides** —
the LLM agent (short and long FP8-KV routes, benched, §5/§7) via a frontend capsule
API, and the robot rollout (`verify_capsule.py`, cosine 1.0) via the execution
contract's Buffer snapshot. Fork is covered by the capsule tests; time-travel is
the same verb (restore an earlier boundary).

---

## 5. Efficiency — what each scenario actually gains

Be precise about where the speedup is, so the benchmark measures the right thing:

- **Decode throughput (tok/s): unchanged by design.** Decode is the same captured
  graph replay with or without capsules. Capsules touch *prefill / time-to-first-
  token*, never steady-state decode.
- **One continuous hot session: ~0 gain.** The shipped contiguous append already
  reuses that session's own prefix. A single-session benchmark will (correctly)
  show no change — that is not a regression, it is the append path doing its job.
- **Fresh session / multi-session / shared prefix: large TTFT win.** This is the
  coding-agent reality (many turns, many sessions sharing the system+repo prefix,
  retries and branches). Restore replaces a compute-bound cold prefill of the
  shared prefix (grows with length, seconds at 10k–50k tokens) with a
  bandwidth-bound copy (~flat in length, milliseconds). The win scales with the
  shared-prefix length and the number of sessions/branches reusing it.
- **Fork: N branches for one prefill.** Tree-of-thought / multi-sample / parallel
  tool-call hypotheses pay one prefill of the shared prefix instead of N.

Costs to keep honest:

- **KV footprint dominates a capsule.** Small state (recurrent/conv/MTP/metadata)
  is tiny and fixed-size; the KV region grows with the boundary position. The
  number of resident capsules is bounded by host RAM (FP8-KV roughly halves it).
- **Restore is bandwidth-bound.** D2D restore is cheap; parking/restoring to host
  RAM costs a D2H/H2D copy over PCIe — still far below a multi-second cold
  prefill, but not free. Disk (L3) is for persistent project capsules, not the
  hot path.

---

## 6. Retrofit plan — `qwen36_agent/` → capsules

Additive, opt-in, and staged so each step has its own acceptance gate.

1. **Frontend snapshot/restore (flash_rt/) — DONE, in-GPU (D2D).**
   `snapshot_capsule()` / `restore_capsule(capsule)` on the Qwen3.6 frontend copy
   the boundary buffers (§2) for both the short and the long FP8-KV routes;
   `long_prefill_chunk_size()` / `capsule_aligned_len()` give the chunk-aligned
   boundary for cold-identical long-route append. Additive, default path untouched.
2. **Serving capsule registry (serving/qwen36_agent/).** Extend `SessionRegistry`
   with a capsule store (pin + small LRU) and a `"restore"` `PrefixPlan` action, so
   today's `rebuild` / non-hot / `activate_rebuild` cases that match a pinned
   capsule become restore + suffix-prefill.
3. **Pin API.** Let the host pin a capsule for the shared prefix
   (system + tool schemas + repo index) once at startup or on first use; an
   OpenAI-side field (`flashrt_pin_prefix`) or a `/v1/sessions` capsule option.
4. **Off-GPU capsule store (exec/).** When parking capsules off-GPU is needed, add
   host-backed `Buffer` + device↔host async copy to the contract (pure mechanism).
   Not required while capsules stay resident on the GPU.
5. **Later: fork / time-travel.** Restore one capsule into N sessions (fork) and
   restore an earlier same-session boundary (undo a turn) — same verbs, no new
   mechanism.

Out of scope for v1: radix tree, paged KV, dense token-grid checkpoints (§3),
cross-deployment capsule portability (§8).

---

## 7. Acceptance — correctness gates and measured results

### 7.1 Correctness (non-negotiable) — `tests/test_qwen36_agent_capsule.py`

A capsule restore is **bit-identical to the path it replaces**, asserted
token-exact:

- **restore-equivalence:** `restore + decode` == cold `prefill + decode` of the
  same prefix, on the short and long FP8-KV routes (real text), including restore
  after the live buffers were dirtied by another prompt, and fork (two branches
  from one capsule).
- **chunk-aligned long append == cold full prefill:** snapshotting at a
  chunk-aligned boundary (`capsule_aligned_len`) makes `restore + append(suffix) +
  decode` token-identical to a cold full prefill — the chunked-GDN recurrent state
  is chunk-aligned, so an unaligned boundary would diverge under FP8 rounding.
- **hybrid-state surface:** the recurrent/conv/MTP/FP8-KV snapshot round-trips
  bit-exact; the long route re-dequantizes the BF16 stage from the restored FP8
  cache (mirrors `reset_state`).
- **no-regression:** full suite (capsule + agent policy + gpu split) is green.

### 7.2 Measured performance (RTX 5090, in-container)

The single-session decode benchmark (133 tok/s) does **not** move — capsules touch
prefill / TTFT only. The win is on a large shared prefix reused across turns:
`cold` = re-prefill prefix+suffix every turn; `capsule` = restore + append(suffix).

Short committed-stream route (185-token shared prefix, real coding-agent tasks,
median of 7, < 1% across runs):

| task | full / suffix | cold TTFT | capsule TTFT | speedup | token-exact |
| --- | --- | --- | --- | --- | --- |
| fill-doc | 258 / 73 | ~5.47 s | ~1.85 s | 2.96x | yes |
| write-code | 223 / 38 | ~4.77 s | ~1.10 s | 4.33x | yes |
| algorithm | 225 / 40 | ~4.82 s | ~1.13 s | 4.26x | yes |

Long FP8-KV route (production agent path, chunk-aligned 2k/4k/8k prefix, median of
5, < 0.5% across runs):

| shared prefix | cold TTFT | capsule TTFT | speedup | capsule MB | capsule == cold |
| --- | --- | --- | --- | --- | --- |
| 2048 tok | ~288 ms | ~138 ms | 2.08x | 168 MB | yes |
| 4096 tok | ~388 ms | ~73 ms | 5.28x | 211 MB | yes |
| 8192 tok | ~816 ms | ~142 ms | 5.72x | 360 MB | yes |

Cold TTFT grows with prefix length; capsule TTFT stays roughly flat (restore is a
~0.1 ms device-to-device copy, so only the suffix is recomputed), so the **speedup
widens with prefix length** — and keeps widening toward the 10k–50k shared
prefixes a real coding agent resends each turn. Decode throughput is unchanged.

---

## 8. Honest boundaries

- A capsule is a **binary state blob** bound to exact model weights + quant +
  kernel version + graph bucketing. Persisting it (L3 / disk) is a *same-
  deployment warm-start* or a *within-team shared capsule for an identical
  deployment* — not a portable, cross-version text cache like a token-level APC.
- The target is individual / small-team / edge — one to a few interactive
  sessions, latency-first. The capsule registry tiers naturally (GPU → host RAM →
  disk) but stays single-node; large-cluster distributed KV (e.g. SGLang HiCache)
  is deliberately out of scope.
- Capsules are an opt-in serving feature. They do not change the contract beyond
  the §3 mechanism addition, and they do not change steady-state decode.

---

## 9. What this design is good at (and what it complements)

Capsules are not a replacement for paged or radix prefix caching — they are a
different point in the design space, chosen for FlashRT's scenario: latency-first,
small-batch, consumer/edge, individuals and small teams. It is worth being precise
about where each approach shines, because the goal is to **cover cases that are
awkward for block/radix KV caches**, not to compete on their home turf.

**Where mainstream prefix caching is the right tool.** vLLM Automatic Prefix
Caching and SGLang RadixAttention are excellent for high-concurrency, multi-tenant,
throughput-first serving of dense-attention LLMs, with automatic cross-request
prefix discovery and paged memory. If that is the workload, use them — FlashRT does
not target it.

**Where the capsule approach fits naturally, and which other approaches don't
cover as easily:**

- **Hybrid models (linear-attention / recurrent / conv state).** A block or radix
  KV cache reuses a prefix by addressing KV blocks; but a recurrent/conv state is a
  *fold* over the entire prefix and has no block to address. Reusing it requires
  snapshotting the state itself — which is exactly what a capsule is. We
  demonstrated this bit-exact for Qwen3.6's gated-delta-net recurrent state and conv
  state, including the chunk-alignment subtlety that exact reuse of a chunked linear
  scan requires. This is the gap capsules fill most cleanly.
- **Keeping full-graph capture while still reusing work.** Paged caching is what
  lets eager/piecewise engines gather KV from arbitrary blocks; the cost is giving
  up a single captured graph over the whole forward. FlashRT keeps the captured
  graph (its latency advantage on small batches) and recovers prefix reuse through
  a state snapshot instead. Different trade, suited to a different scenario.
- **One mechanism across LLM, VLA, and robotics.** Because a capsule is just the
  committed execution state as a set of buffers, the same snapshot/restore idea
  serves an LLM agent's warm-start, a VLA diffusion policy, and a robot RL rollout's
  episode reset. General LLM-serving stacks are not built to span these domains; a
  framework that does can offer one consistent serving story across them.
- **Deterministic, bit-exact reproducibility.** Snapshotting exact state makes a
  session or rollout reproducible to the token / action — useful for debugging, RL
  data integrity, and regression testing. Throughput-first stochastic batching does
  not aim to provide this.

The honest summary: pick the tool by scenario. For high-concurrency multi-tenant
LLM serving, the paged/radix engines lead. For latency-first single/few-session
work on consumer and edge hardware — especially hybrid models, VLAs, and robot
rollouts — capsules cover ground those engines were not designed to reach. A fair,
measured comparison on this scenario is planned once development settles (§10).

---

## 10. Roadmap — further along this philosophy

The capsule primitive opens more than prefix reuse. Each item below is the same
snapshot / restore / fork / time-travel mechanism applied to a new need; none
requires a new contract verb (mechanism-not-policy holds, see exec_contract §9).

1. **Cross-process / persistent capsules (L3 warm-start).** Park a capsule in host
   RAM, or serialize it to disk, so a server restart or an edge device can resume a
   pinned prefix (system + tool schema + repo index) instantly instead of
   cold-prefilling. Bounded to a same-deployment binary blob (§8), but high value
   for edge cold-start.
2. **Single-GPU few-session time-sharing.** Keep several users' session capsules in
   host RAM and swap the hot one onto the GPU on demand — latency-first
   instant-switch, distinct from continuous batching, matched to the small-team
   target.
3. **Fork & time-travel as first-class agent/robot operations.** Fork one prefilled
   prefix into N branches (tree-of-thought, best-of-N, parallel tool-call
   hypotheses); restore an earlier committed boundary to undo a turn (agent retry)
   or an action chunk (robot exploration / safe rollback).
4. **Robot capsule ergonomics.** The robot side is already bit-exact via the
   contract Buffer snapshot (`verify_capsule.py`); a thin convenience wrapper that
   snapshots the full rollout boundary (seed + observation + any cached frames) in
   one call, and wiring it into `rollout_host.py`'s reset, would make the robot
   capsule as ergonomic as the LLM one.
5. **Deterministic replay / reproducibility tooling.** Capsule + recorded inputs →
   bit-exact session/rollout replay for debugging and RL data integrity.
6. **Scenario comparison study (post-development).** A fair, measured comparison on
   FlashRT's turf — consumer GPU (5090) and edge (Orin/Thor), single/few-session
   latency-first — reporting warm-start TTFT, decode tok/s, VLA time-to-first-action,
   restore latency, capsule footprint, and cold-start time, with conditions stated
   explicitly and no claims outside the measured scenario.
