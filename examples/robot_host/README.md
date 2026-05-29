# examples/robot_host — C++ real-time VLA host (skeleton)

Illustrative **deployment hot-path host** for a VLA (Pi0.5-style) built directly
on the FlashRT execution contract (`libflashrt_exec`, pure C ABI). It is the
**scenario / policy** layer — it lives here in examples, *not* in the contract.

## Why C++ here
Real-time robotics: tight loop, low jitter, interrupts, ROS2-adjacent drivers.
The host, the exec layer, and the CUDA kernels are then **one toolchain, no FFI
seam** — the easiest cross-layer debug, and the natural language for on-robot
real-time code. (For a networked LLM server, prefer Rust — see
`examples/llm_agent/`.)

## What it demonstrates (`realtime_host.cpp`)
- Drive adopted graphs from a native loop: `frt_graph_replay` / `frt_plan_execute`.
- Multi-subgraph **vision → action** as one DAG with zero-copy buffer hand-off.
- **Voice interrupt** on a concurrent lower-priority stream; **subgoal change**
  by overwriting a bound `Buffer` — *no recapture* (µs, not seconds).
- Interrupt granularity = one short replay (decide between ticks).

## Setup (cold) is NOT shown here
Capture + calibration + autotune + `adopt` happen **once** before this loop. Two
ways, both keep Python out of the hot loop:
- **Pragmatic (recommended first):** the Python frontend runs in the **same
  process**, does setup+capture+adopt once, then this C++ loop drives replay.
  Python is present but idle during the loop.
- **Pure no-Python:** port that one model's setup/capture to C++ (the heavy,
  rigid option — only when the robot truly cannot host a Python runtime).

> A captured CUDA graph cannot be serialized across processes (its baked
> pointers are process-local), so capture must run in the **same process** that
> replays — which is exactly why "Python setup once, native hot loop" is the
> pragmatic path. See `docs/exec_contract.md`.

This skeleton shows only the hot loop + C-ABI usage; the `frt_*` handles are
produced by setup. It is a reference structure, not a runnable binary on its own.
