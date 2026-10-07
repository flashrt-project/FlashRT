// ============================================================================
//  FlashRT — Pi0.5 decoder step edges (fp16, action dim 32).
//
//  action_in:  x[S, D] = fp16(fp16(noise[S, 32] @ w[32, D]) + b[D])  -- one launch
//              instead of the cuBLAS fp16 GEMM + add_bias_fp16 pair.
//  adarms_action_out: per row, the final AdaRMS (xn = fp16(x * rstd * (1 + scale)
//              + shift), gate copied) fused with the action head
//              delta = xn @ aow[D, 32] (fp32), v = fp16(delta + aob), step =
//              fp16(v * fp16(dt)), noise += step (fp16) -- one launch instead
//              of adarms_fp16 + split-K GEMM + reduce + action_update_from_fp32.
//  Same per-element rounding chain as the originals; dot products accumulate
//  in fp32 in a fixed but different order.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

int pi05_action_in_fp16(const void* noise, const void* w, const void* b, void* x, int S, int D, int A,
                        cudaStream_t stream);

int pi05_adarms_action_out_fp16(const void* x, const void* style, void* xn, void* gate, const void* aow,
                                const void* aob, void* noise, int S, int D, int A, float dt, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
