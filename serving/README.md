# serving/ — scenario hosts on the FlashRT execution contract

The **serving layer**: native hosts that drive the execution contract
(`libflashrt_exec`, see `docs/exec_contract.md`) for a concrete scenario. This
is the **scenario / policy** layer — sessions, schedulers, protocols, sensor
loops, interrupt policy live here, deliberately *out of* the contract.

FlashRT layering:
```
flash_rt/  (Python frontend)   setup: weights, calibration, autotune, capture, adopt  [cold]
serving/   (this dir)          hosts: drive replay / Plan, define the scenario          [hot path]
exec/      (C ABI contract)    replay-time mechanism: Buffer / Graph / Plan             [hot path]
csrc/      (kernels)           the captured compute (fvk / cutlass / ...)               [hot path]
```

Hosts (host language chosen per scenario; all link the same `libflashrt_exec`):
- **`robot_host/`** (C++) — real-time VLA loop: vision→action Plan, concurrent
  ASR stream, subgoal-buffer interrupt. C++ = one toolchain with exec+kernel, no
  FFI seam, real-time/ROS2-adjacent.
- **`llm_agent/`** (Rust) — LLM session server: per-token replay over the C ABI.
  Rust = async/safety shell; the per-token FFI seam is crossed once per token.

Both are **reference skeletons**: they show the hot loop + C-ABI usage. Setup
(capture + adopt) builds their handles once — via the Python frontend in the
same process (Python out of the hot loop), or ported to native for a no-Python
build. A captured CUDA graph is not serializable across processes, so capture
runs in the same process that replays.
