// Persistent dependent-GEMM sequence (see sm100_gemm_seq_persistent_kernel.hpp).
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

constexpr int kSeqMaxProblems = 6;

// One NVFP4 GEMM of the sequence in the (activation A, weight B) convention of
// cutlass_fp4_gemm_variant: D (M x N fp16, row-major) = A (M x K) * B (N x K)^T.
struct SeqGemmDesc {
  const void* A; const void* SFA;   // activations + SFA
  const void* B; const void* SFB;   // weights + SFB
  void* D;
  int M, N, K;
  float alpha, beta;
  // Elementwise phase run after this problem's outputs are complete (before the
  // next problem loads its activations). phase 0 = none; 1 = gate_res_adarms:
  // residual += D * prev_gate; xn = quant(adarms(residual, style)); gate = style gate.
  // GeGLU gate_up problem: B holds the interleaved gate/up weights (N = N_il); the epilogue
  // writes the compact NVFP4 hidden (M x N/2) + SFA instead of D (D must still point at a
  // (M x N) fp16 scratch the descriptor can describe).
  int geglu = 0;                      // epilogue mode: 0 pass-through, 1 GeGLU compact store, 2 gated residual + ssq partials
  void* compact_packed = nullptr; void* compact_sfa = nullptr;
  // mode 2: D must be the residual buffer (M x N fp16 row-major); gate rows come from the previous style block.
  const void* res_in = nullptr; const void* gate_in = nullptr; long res_pitch = 0; long gate_pitch = 0; void* partials = nullptr;
  // B / SFB point at the base of per-cluster activation slots (batch index = cluster id).
  int b_private = 0; int b_slots = 0;
  // phase after this problem: 0 none, 1 gate_res_adarms (rows over CTAs, two sync points), 2 gate_res, 3 private adarms
  int phase = 0;
  const void* ph_prev_gate = nullptr; void* ph_residual = nullptr; const void* ph_style = nullptr;
  void* ph_packed = nullptr; void* ph_sfa = nullptr; void* ph_gate = nullptr;
  int ph_S = 0; int ph_D = 0;
  const void* ph_partials = nullptr; void* ph_slot_packed = nullptr; void* ph_slot_sfa = nullptr;
  long ph_slot_packed_pitch = 0; long ph_slot_sfa_pitch = 0;
};

// Runs the n problems (n <= kSeqMaxProblems) in one persistent launch. `counter`
// is a zero-initialised int in device memory (the kernel resets it on exit).
// grid_out (optional, 3 ints) receives the launched grid.
// variant = (weight k-tiles issued before the wait on a problem's first tile, on its other tiles):
// 0 = (3,3), 1 = (8,0), 2 = (5,0), 3 = (8,3), 4 = (3,0). flags bit0 skips the barriers (probe; independent problems only).
int cutlass_fp4_gemm_seq_run(int n, const SeqGemmDesc* d, int* counter, cudaStream_t stream, int* grid_out = nullptr, int variant = 0, int flags = 0);

}  // namespace fp4
}  // namespace flash_rt
