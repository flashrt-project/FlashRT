// ================================================================
// FlashRT — Path C megakernel: encoder MLP fused (G7+GeGLU+G8)
// on a single SM100 kernel without materializing the hidden tensor.
//
// Status: SCAFFOLD ONLY.  This first commit lands the extern C
// signature, the build wiring, and a *fallback* implementation that
// calls the existing 3-step path (sq_gelu + sq + mul_fp16 + 2sm21).
// The fallback exists so that integration tests and bindings can be
// exercised end-to-end while the real kernel is still being written.
//
// Replacing the fallback is the main goal of subsequent sessions.
// See megakernel_dev/notes/PATH_C_DESIGN.md for the architectural plan.
//
// Why Path C is the next step:
// - Tile-sweep + epilogue-activation fusion already pushed FP16 3v
//   from 98.93 ms (Phase 1.A) to 91 ms in the Thor SM110 hot regime
//   (stable; FP8 2v = 44 ms anchors the regime).
// - Empirically tested and REJECTED in hot regime (all regress):
//     LinCombDeEltAct mul-aux (k64 +0.3ms, sq +9ms)
//     LinearCombination beta!=0 residual-fused epilogue (+11ms)
//   Root cause: any external GMEM tensor read inside the SM100
//   epilogue (Aux or C) adds enough TMA descriptor pressure to keep
//   Thor in the slower power state.
// - The only remaining fusion path is keeping intermediates entirely
//   in SMEM/tmem, which requires a hand-composed multi-stage kernel
//   (Path C).
//
// ================================================================
// Reference math (what the megakernel must compute):
//     gate_buf[m, h] = GELU_tanh(sum_k X[m, k] * W_gate[h, k])
//     up_buf[m, h]   =          sum_k X[m, k] * W_up  [h, k]
//     hid[m, h]      = gate_buf[m, h] * up_buf[m, h]
//     out[m, n]      = sum_h hid[m, h] * W_down[n, h]
//   where M=Se, K=D, H=ffn_hidden, N=D.
// ================================================================
// Resource budget for the eventual real kernel (Thor SM100):
//     SMEM per CTA <= 228 KB.  Planned slices:
//       X stage:       2 * TILE_M * K_CHUNK * 2 B
//       W_gate stage:  2 * H_CHUNK * K_CHUNK * 2 B
//       W_up stage:    2 * H_CHUNK * K_CHUNK * 2 B
//       W_down stage:  2 * TILE_N_OUT * H_CHUNK * 2 B
//       hid_smem:      TILE_M * H_CHUNK * 2 B   (geglu intermediate)
//       out_smem:      TILE_M * TILE_N_OUT * 2 B (epilogue staging)
//     Target: TILE_M=64, TILE_N_OUT=64, K_CHUNK=64, H_CHUNK=128
// ================================================================

#include "../gemm_types_sm100_fp16.h"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Declared in csrc/kernels/activation.cuh; forward-declared here so the
// header chain doesn't pull in unrelated dependencies.
void mul_fp16(const __half* a, const __half* b, __half* out, int n,
              cudaStream_t stream);

// Fallback runner.  Calls the three existing CUTLASS GEMMs + the
// mul_fp16 element-wise kernel that the production spec uses today.
// The signature matches what the real megakernel will eventually
// expose, so callers wired to encoder_mlp_sm100 can transparently
// upgrade.
extern "C" int cutlass_fp16_k64_gelu(void* A, void* B, void* D, int M, int N, int K,
                                      float alpha, float beta, cudaStream_t stream);
extern "C" int cutlass_fp16_sq_gelu(void* A, void* B, void* D, int M, int N, int K,
                                     float alpha, float beta, cudaStream_t stream);
extern "C" int cutlass_fp16_sq(void* A, void* B, void* D, int M, int N, int K,
                                float alpha, float beta, cudaStream_t stream);
extern "C" int cutlass_fp16_2sm21(void* A, void* B, void* D, int M, int N, int K,
                                   float alpha, float beta, cudaStream_t stream);

// Encoder MLP fused: out[M, N_out] = G8(GeGLU(G7_gate(X, W_gate), G7_up(X, W_up)), W_down).
//
//   X        : [M,    K]       row-major, FP16
//   W_gate   : [H,    K]       row-major (N-major view = ColumnMajor on K), FP16
//   W_up     : [H,    K]       row-major,                                    FP16
//   W_down   : [N_out, H]      row-major,                                    FP16
//   gate_buf : [M, H]   scratch, FP16  (used by fallback only)
//   up_buf   : [M, H]   scratch, FP16  (used by fallback only)
//   hid_buf  : [M, H]   scratch, FP16  (used by fallback only)
//   out      : [M, N_out]      row-major, FP16
//
// Sizes are taken as runtime ints to keep the signature stable across
// FFN dimensions.  Eventually the real kernel will template on shape.
extern "C" int encoder_mlp_fused_fp16(
    void* X, void* W_gate, void* W_up, void* W_down,
    void* gate_buf, void* up_buf, void* hid_buf, void* out,
    int M, int N_out, int K, int H,
    cudaStream_t stream)
{
    // TODO(path-c): replace with the single SMEM-staged kernel.
    // Reference fallback below preserves correctness.
    int rc;
    rc = cutlass_fp16_sq_gelu(X, W_gate, gate_buf, M, H, K, 1.0f, 0.0f, stream);
    if (rc != 0) return rc;
    rc = cutlass_fp16_sq(X, W_up, up_buf, M, H, K, 1.0f, 0.0f, stream);
    if (rc != 0) return rc;
    mul_fp16(reinterpret_cast<const __half*>(gate_buf),
             reinterpret_cast<const __half*>(up_buf),
             reinterpret_cast<__half*>(hid_buf),
             M * H, stream);
    rc = cutlass_fp16_2sm21(hid_buf, W_down, out, M, N_out, H, 1.0f, 0.0f, stream);
    return rc;
}

// ================================================================
// Implementation roadmap (must follow this approach — see
// megakernel_dev/notes/PATH_C_DESIGN.md for full rationale)
// ================================================================
//
// CORRECT approach: reuse production CUTLASS CollectiveMma by calling
// its public methods (load_init, load, load_tail, init_tmem_tensors,
// mma_init, mma, slice_accumulator) from a custom megakernel struct.
// This inherits all of production CUTLASS's multi-stage SMEM pipelining,
// warp specialization, TMA pipelining, and cluster optimizations.
//
// RED LINE: do NOT hand-build a kernel from cute primitives via the
// `cute/tutorial/blackwell/` examples.  Tutorials are correctness toys
// that omit the multi-stage / warp-spec optimizations production
// `CollectiveMma` already implements.  Sub-mainloop perf MUST match
// `cutlass_fp16_sq` / `cutlass_fp16_2sm21` isolation numbers before
// fusion can buy anything on top.
//
// Step 1 — single-mainloop wrapper kernel:
//   Use cutlass::gemm::collective::CollectiveBuilder<...> to obtain a
//   production CollectiveMma at our split-G7 tile.  Write a custom
//   kernel struct whose operator() calls collective_mainloop.load() in
//   producer warps and collective_mainloop.mma() in consumer warps —
//   structurally identical to sm100_gemm_tma_warpspecialized.hpp's
//   operator() but in our codebase so we can extend it.  Validate:
//     cosine 1.000000 vs cutlass_fp16_sq
//     isolation perf within 5% of cutlass_fp16_sq (~750 us at n=500)
//
// Step 2 — dual mainloop sharing X:
//   Construct two CollectiveMma instances (one per W_gate, W_up).
//   Producer warps interleave TMA loads for the two B operands; A
//   (= X) is loaded once and shared.  Two TMEM accumulators (gate /
//   up), no GMEM materialization.  Validate cosine 1.0 vs reference.
//
// Step 3 — in-SMEM GeGLU + third mainloop:
//   Apply gelu_tanh(gate_acc) * up_acc in TMEM→RMEM path; stage the
//   FP16 result `hid` into SMEM.  Build a third CollectiveMma whose
//   A operand sources from SMEM (`hid`) and B = W_down.  Producer
//   warps load W_down via TMA; consumer warps execute the final mma()
//   to produce out_acc.  Production CollectiveEpilogue stores out_acc
//   to GMEM.  Cosine 1.0 vs the 3-step fallback at the encoder shape.
//
// Step 4 — tile sweep per sub-mainloop:
//   Each of the three CollectiveMma instances can have an independent
//   (TILE_M, TILE_N, TILE_K, ClusterShape) selected via the CollectiveBuilder
//   template parameters.  Hot-regime A/B-sweep them, gated on FP8 2v ≈ 44 ms.
//
// Phase C (production wiring) — only if hot-regime win >= 2 ms and no
// bimodal behavior.  Squash WIP commits per feedback_commit_discipline.
//
// Reading materials:
//   - cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp
//     (the production kernel orchestrator — copy its operator() body
//     as the starting point for our custom megakernel)
//   - cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp
//     (the production CollectiveMma whose load/mma we will call)
//   - cutlass/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp
//     (the production CollectiveEpilogue that will store the final
//     out_acc; reused as-is)
