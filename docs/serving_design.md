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

`reset_state()` in `rollout_host.py` restores the captured policy boundary — the
same snapshot/restore verb the coding agent uses, in a different scenario.

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

Additive, opt-in, and staged so each step has its own acceptance gate. Nothing
below modifies the contract beyond the §3 mechanism addition.

1. **Contract mechanism (exec/).** Add host-backed `Buffer` + `buffer_copy`
   device↔host async variants. Pure mechanism; covered by the exec toy tests.
2. **Frontend snapshot/restore (flash_rt/).** Add `snapshot_capsule()` /
   `restore_capsule(capsule)` to the Qwen3.6 frontend that copy the boundary
   buffers in §2 (additive methods, alongside the existing `*_agent` split
   methods; default path untouched).
3. **Serving capsule registry (serving/qwen36_agent/).** Extend `SessionRegistry`
   with a capsule store (pin + small LRU) and a new `PrefixPlan` action
   `"restore"`. Today's `rebuild` / non-hot / `activate_rebuild` cases that match
   a pinned or stored capsule prefix become restore + suffix-prefill.
4. **Pin API.** Let the host pin a capsule for the shared prefix
   (system + tool schemas + repo index) once at startup or on first use; an
   OpenAI-side field (`flashrt_pin_prefix`) or a `/v1/sessions` capsule option.
5. **Later: fork / time-travel.** Restore one capsule into N sessions (fork) and
   restore an earlier same-session boundary (undo a turn) — same verbs, no new
   mechanism.

Out of scope for v1: radix tree, paged KV, dense token-grid checkpoints (§3),
cross-deployment capsule portability (§7).

---

## 7. Acceptance design — correctness gates and expected performance

### 7.1 Correctness (non-negotiable, same yardstick as the shipped append test)

`tests/test_qwen36_agent_gpu_split.py` already asserts `append == full-generate`
token-exact. Capsules add the symmetric gate:

- **restore-equivalence:** `restore(capsule) + prefill(suffix) + decode` produces
  **token-identical** output to a cold `full prefill(prefix + suffix) + decode`,
  for short, long (FP8-KV), and long-append prompt buckets, greedy, fixed K.
- **fork-equivalence:** two sessions forked from one capsule each match their own
  independent cold run, token-exact.
- **hybrid-state bit-exactness:** the recurrent/conv/MTP snapshot must round-trip
  bit-exact; a silent mismatch here corrupts output without crashing. This is the
  highest-risk surface and gets a dedicated buffer-level round-trip assert before
  any end-to-end claim.
- **no-regression:** capsules default OFF; with the flag off the host is
  byte-identical and same-latency to the shipped path.

### 7.2 Performance — the benchmark and what we expect

The single-session decode benchmark (133 tok/s, ~217 ms TTFT) will **not** move —
and should not. The capsule benchmark must model the real workload: a **large
shared prefix reused across many turns/sessions**.

Benchmark: fix a shared prefix at several lengths (e.g. 4k / 16k / 64k tokens),
then for each length measure, for a fresh session that reuses the prefix:

| metric | cold (no capsule) | restore (capsule) | expectation |
| --- | --- | --- | --- |
| TTFT | full prefill of (prefix + suffix) | copy(KV+state) + prefill(suffix) | restore ≪ cold; gap grows with prefix length |
| prefill tokens computed | prefix + suffix | suffix only | prefix tokens saved per reuse |
| decode tok/s | baseline | baseline | unchanged (gate: within noise) |
| capsule bytes | — | KV(prefix) + small state | report per length; FP8-KV ≈ half |
| restore latency | — | D2D or D2H/H2D copy time | bandwidth-bound, report ms |

**Expected result:** for a continuous single session, ~0 change (append already
optimal). For shared-prefix / multi-session / fork workloads, **TTFT for a warm
session drops from a length-growing cold prefill (seconds at tens of thousands of
tokens) to a roughly length-flat restore (milliseconds)**, with the suffix being
the only recomputed work. Decode throughput is unchanged. The headline metric is
**warm-session TTFT and prefill-tokens-saved per reuse**, reported against the
shared-prefix length and the number of reuses — not single-shot decode tok/s.

Report the cold-vs-restore crossover (the prefix length beyond which restore
wins) and the capsule memory footprint per length, in-container on RTX 5090, with
the runnable commands — same evidence bar as `docs/exec_contract.md` §8.

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
