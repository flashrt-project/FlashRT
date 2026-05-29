# examples/llm_agent — Rust LLM session host (skeleton)

Illustrative **deployment hot-path host** for an LLM (Qwen3.6-style) built on the
FlashRT execution contract (`libflashrt_exec`, pure C ABI), driven from Rust.
Scenario / policy layer — lives in examples, *not* in the contract.

## Why Rust here
A networked agent/LLM server is a **control plane**: many clients, async
HTTP/SSE, session registry, graceful errors. Rust's async + safety shine in that
shell, and the per-token hot path crosses the FFI seam only **once per token**
(a single `frt_graph_replay`), so the boundary is thin. (For an on-robot
real-time VLA loop, prefer C++ — see `examples/robot_host/`.)

## What it demonstrates (`src/main.rs`)
- A minimal FFI binding to the C-ABI subset (`frt_ctx_*`, `frt_graph_*`).
- A session/decode loop: per token → write token into the input `Buffer` →
  `frt_graph_replay(decode, key = cur_pos, stream)` → read logits → argmax →
  append. Spec-decode/MTP would add more adopted graphs (verify/draft) the same way.
- ShapeKey = `cur_pos` (exact key, LRU); batch (B=1..8 packing) is just another
  field of the key — no new concept.

## Setup (cold) is NOT shown here
Weight load + calibration + autotune + capture + `adopt` happen once before
serving — by the Python frontend in the same process (Python out of the hot
loop), or ported to native for a no-Python build. The captured graph must be
adopted in the **same process** that replays it (CUDA graphs are not
serializable across processes). This skeleton shows only the Rust hot loop +
C-ABI usage; it is a reference structure, not a runnable binary on its own.

## Build sketch
`build.rs` links `libflashrt_exec`; `cargo build` produces the host. Wire an
HTTP/SSE layer (axum/tonic) around `decode_step` for the OpenAI-compatible
surface.
