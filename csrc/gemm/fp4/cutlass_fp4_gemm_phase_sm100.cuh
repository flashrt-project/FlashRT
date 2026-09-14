#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {

// Host-side description of the phase carried by a decoder GEMM launch
// (plain data so the binding translation unit needs no CUTLASS headers).
// kind 1: gated residual + AdaRMS + NVFP4 quantize (pi05_gate_res_adarms_fp4_sfa_native_fp16 order);
// kind 2: gated residual only (gate_res_fp16).
struct PhaseHostArgs {
  int kind = 0;
  int num_phase_ctas = 0;        // even (two per cluster)
  void* counter = nullptr;       // 2 x uint32, zero-initialised
  const void* x = nullptr;       // GEMM output (fg)
  const void* prev_gate = nullptr;
  void* residual = nullptr;
  const void* style = nullptr;   // (S, 3D) rows: scale, shift, gate
  void* packed = nullptr;
  void* sfa = nullptr;
  void* gate_out = nullptr;
  int S = 0;
  int D = 0;
  int dbg = 0;
};

// Same operand order as cutlass_fp4_gemm_variant (A = activations, B = weights).
int cutlass_fp4_gemm_phase(void const* A, void const* SFA, void const* B, void const* SFB, void* D,
                           int M, int N, int K, float alpha, float beta, cudaStream_t stream,
                           PhaseHostArgs const& phase);

}  // namespace fp4
}  // namespace flash_rt
