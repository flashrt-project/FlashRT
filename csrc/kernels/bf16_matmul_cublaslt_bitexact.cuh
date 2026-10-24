// ================================================================
// FlashRT — cuBLASLt BF16 GEMM that keeps the heuristic algorithm's bits.
//
//   out[M, N] = x[M, K] · W[N, K]^T   (bf16 in / out, fp32 accumulate)
//
// The result is always the one cuBLASLt's heuristic first choice produces
// -- the algorithm torch's matmul reaches for at these shapes -- but the
// launch uses the fastest candidate that reproduces it bit for bit. On the
// first eager call for a shape, the heuristic's candidates are run on random
// probe activations against the real weight; a candidate is kept only if
// every output element equals the first choice's, then the kept ones are
// timed. Candidates that reduce K in another order (split-K, stream-K)
// disagree on some element and drop out, so the timing can only pick between
// algorithms that compute the same thing.
//
// A first call under stream capture, a shape whose probe would not fit the
// probe budget, or a device without room for the probe takes the heuristic
// first choice untimed. Plans are per device; each stream gets its own
// workspace on its first call, which must therefore not be under capture.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt::kernels {

void bf16_matmul_cublaslt_bitexact_bf16(const __nv_bfloat16* x,
                                        const __nv_bfloat16* W,
                                        __nv_bfloat16* out,
                                        int M, int N, int K,
                                        cudaStream_t stream);

}  // namespace flash_rt::kernels
