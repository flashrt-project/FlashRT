//! Illustrative Rust LLM session host on the FlashRT execution contract.
//!
//! Shows only the DEPLOYMENT hot path (the decode loop) driving an adopted
//! decode graph through the C ABI. Setup (weight load / calibrate / autotune /
//! capture / adopt) happens once before this — by the Python frontend in the
//! same process, or ported to native. The `frt_*` handles below are produced
//! by that setup. Reference structure, not a standalone runnable binary.

use std::os::raw::{c_char, c_int};

// ── Minimal FFI to the C-ABI subset (see exec/include/flashrt/exec.h) ──
#[repr(C)] pub struct FrtCtx { _p: [u8; 0] }
#[repr(C)] pub struct FrtGraph { _p: [u8; 0] }
#[repr(C)] pub struct FrtBuffer { _p: [u8; 0] }

extern "C" {
    fn frt_ctx_create() -> *mut FrtCtx;
    fn frt_ctx_wrap_stream(ctx: *mut FrtCtx, external_stream: *mut std::ffi::c_void) -> c_int;
    fn frt_buffer_dptr(b: *mut FrtBuffer) -> *mut std::ffi::c_void;
    fn frt_graph_create(ctx: *mut FrtCtx, name: *const c_char, max_variants: usize) -> *mut FrtGraph;
    fn frt_graph_adopt(g: *mut FrtGraph, key: u64, external_graph_exec: *mut std::ffi::c_void) -> c_int;
    fn frt_graph_replay(g: *mut FrtGraph, key: u64, stream_id: c_int) -> c_int;
}

const FRT_OK: c_int = 0;

/// Handles produced by setup (capture + adopt), handed to the hot loop.
pub struct LlmHandles {
    pub ctx: *mut FrtCtx,
    pub decode: *mut FrtGraph,   // ShapeKey = cur_pos (exact key, LRU)
    pub token_in: *mut FrtBuffer,
    pub logits: *mut FrtBuffer,
    pub stream_id: c_int,        // wraps the frontend's capture/replay stream
}

/// One decode step: write `tok` into the input buffer, replay the per-pos graph,
/// return the logits device pointer for argmax. (Spec-decode/MTP add more
/// adopted graphs — verify/draft — driven exactly the same way.)
pub unsafe fn decode_step(h: &LlmHandles, cur_pos: u64, tok: i64) -> Result<*mut std::ffi::c_void, c_int> {
    // host writes `tok` into h.token_in's dptr (omitted: a tiny H2D copy)
    let _dst = frt_buffer_dptr(h.token_in);
    let _ = tok;
    let rc = frt_graph_replay(h.decode, cur_pos, h.stream_id); // == cudaGraphLaunch
    if rc != FRT_OK { return Err(rc); }
    Ok(frt_buffer_dptr(h.logits)) // host reads logits, argmax -> next token
}

/// Generate `max_new` tokens for one session. A real server wraps this with
/// axum/SSE + a session registry; batch (B=1..8) packing is just another
/// ShapeKey field, not a new concept here.
pub unsafe fn generate(h: &LlmHandles, mut cur_pos: u64, first_tok: i64, max_new: usize) {
    let mut tok = first_tok;
    for _ in 0..max_new {
        let _logits = match decode_step(h, cur_pos, tok) {
            Ok(p) => p,
            Err(rc) => { eprintln!("frt replay failed rc={rc}"); break; }
        };
        // tok = argmax(_logits); append to the session's output (omitted)
        cur_pos += 1;
    }
}

fn main() {
    eprintln!("flashrt llm_agent: skeleton — setup (capture+adopt) must build LlmHandles first.");
    let _ = (frt_ctx_create as usize, frt_ctx_wrap_stream as usize, frt_graph_create as usize,
             frt_graph_adopt as usize); // reference the FFI so the linker keeps it
}
