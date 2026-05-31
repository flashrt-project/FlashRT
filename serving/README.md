# serving/ — scenario hosts on the FlashRT execution contract

The **serving layer**: native hosts that drive the execution contract
(`libflashrt_exec`, see [`docs/exec_contract.md`](../docs/exec_contract.md)) for a
concrete scenario. This is the **scenario / policy** layer — sessions,
schedulers, protocols, sensor loops, episode/rollout control, interrupt policy
live here, deliberately *out of* the contract.

The signature serving idea — **graph-replay-native execution-state capsules**
(snapshot / restore / fork / time-travel), and why it differs from vLLM/SGLang
prefix caching — is documented in
[`docs/serving_design.md`](../docs/serving_design.md). It is the unifying design
behind the hosts below. That document also covers **what this design is good at
and what it complements** (§9 — capsules fill cases awkward for block/radix KV
caches: hybrid recurrent state, full-graph capture, LLM+VLA+robot under one
mechanism) and the **capsule roadmap** (§10 — persistent/L3 warm-start,
few-session time-sharing, fork & time-travel, robot-side parity, a measured
scenario comparison).

## Layering

```
  flash_rt/  (Python frontend)  setup: weights · calibration · autotune · capture · adopt   [cold, once]
  serving/   (this dir)         hosts: drive replay / Plan; define the scenario              [hot path]
  exec/      (C ABI contract)   replay-time mechanism: Buffer / Graph / Plan                 [hot path]
  csrc/      (kernels)          the captured compute (fvk / cutlass / ...)                    [hot path]
```

The contract gives the **mechanism** (per-chunk interruptible replay, zero-copy
buffer hand-off, multi-stream/event, buffer reset). Each host here adds the
**policy** (state machine, keyboard, recording, reset, scheduling). All hosts
link the *same* `libflashrt_exec`; the host language is chosen per scenario.

## Two runnable Python hosts (real Pi05; community-playable)

### `robot_recap/` — RL rollout host (π*0.6 / RECAP-style)
Advantage-conditioned policy + a value-function critic, with a host-side
**episode state machine**. Built directly to solve the real rollout pain
*"inference keeps running between episodes — I can't stop to reset / record."*

```
  START ──▶ RUNNING ──(keyboard END / value<thr / timeout)──▶ STOP_INFER
    ▲           │  one action CHUNK per replay; host decides between chunks        │
    │           ▼                                                                   ▼
  next ep ◀── RESET(model-state buffers, no recapture) ◀── RECORD(.npz) ◀── robot_reset_to_initial()
                                                                          ◀── AWAIT human reset

  concurrently, via ONE exec ctx:   policy(stream P)  ‖  value critic(stream C)  → auto-termination
```
Maps every part of the community question to a host hook (not the model, not the
contract):
- **keyboard start/end** → a pluggable event source (scripted default; swap in
  pynput/termios) checked *between* chunks;
- **record episodes** → per-chunk trajectory buffer serialized to `episode_*.npz`
  on episode end (`--record-dir`);
- **reset robot to initial** → a `robot_reset_to_initial()` hook (call your
  driver's home pose);
- **stop inference until next episode** → chunk-level interrupt: the host simply
  stops issuing replays at the boundary (granularity = one short replay), then
  resets model state buffers — **no recapture**.

Files: `verify_recap.py` (RL/CFG inference driven by the contract, cosine 1.0) ·
`rollout_host.py` (the full host above).

### `robot_handoff/` — hierarchical planner→actor buffer hand-off
A multi-model hierarchy (shape inspired by π0.7; does not run π0.7 — co-hosts
models we have):
```
  PLANNER (low rate) ──subtask (shared Buffer)──▶ ACTOR (high rate) ──▶ actions
                              ▲
        interrupt / verbal coaching: overwrite the subtask buffer (no recapture)
```
Two Pi05 co-hosted via ONE exec ctx; planner→actor hand-off through a shared
buffer (verified byte-equal); multi-rate (1:N); mid-run subtask interrupt. An
optional world-model stage can use Wan2.2 in place of π0.7's BAGEL.
File: `verify_handoff.py`.

> Together these cover the two multi-model shapes the contract is built for:
> **concurrent** (RECAP policy‖critic + interruptible rollout) and **sequential**
> (planner→actor hand-off + multi-rate).

## Native deployment host examples

- **`robot_host/`** (C++) — real-time VLA deployment host pattern (Plan,
  concurrent stream, buffer-overwrite interrupt). One toolchain with
  exec+kernel, no FFI seam; the target form for on-robot / ROS2 deployment.
- **`qwen36_agent/`** (Python first) — Qwen3.6 long-session agent serving
  policy: exact token-prefix reuse, contiguous session cache, OpenAI-compatible
  tool calls, and true SSE streaming at spec-decode accept boundaries. This is
  the production path for local coding-agent style workloads; the hot compute
  still comes from the Qwen3.6 kernel/CUDA-graph frontend.
- **`llm_agent/`** (Rust) — LLM session server (per-token replay over the C ABI;
  async/safety shell). This remains a reference shell for future high-concurrency
  control-plane experiments; it is not the near-term production path.

`qwen36_agent/` is a runnable Python-first serving example. The C++/Rust hosts
remain reference skeletons and will mirror the verified Python hosts above as
their production paths mature.

## Honest scope (Python hosts)
They drive **real Pi05 graphs** through the contract and verify the hot-path
mechanism (multi-model co-host, hand-off, interrupt, reset, recording). For the
mechanism demo they reuse the captured chunk and use stand-in critic / subtask
wiring (random-init value function; plumbing hand-off, not a semantic
planner→language mapping). Setup (capture) is done once by the in-process Python
frontend; the host then drives replay via the contract. A captured CUDA graph is
not serializable across processes, so capture runs in the same process that
replays.
