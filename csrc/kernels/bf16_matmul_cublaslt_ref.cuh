// ================================================================
// FlashRT — cuBLASLt BF16 GEMM on the algorithm torch's matmul runs.
//
//   out[M, N] = x[M, K] · W[N, K]^T   (bf16 in / out, fp32 accumulate)
//
// tuned = false -- the strict path. Always cuBLASLt's heuristic first choice
// for the problem, asked with the descriptors and workspace limit torch's
// matmul reaches the same answer with, so the result is torch's bit for bit
// by construction (with the process's one libcublasLt, which torch loads).
//
// tuned = true -- may run a faster algorithm. On the first eager call for a
// shape, the heuristic's other candidates run on probe operands drawn here --
// activations *and* weights, independent of the caller's tensors -- and a
// candidate is kept only if every output equals the first choice's on every
// draw; the fastest kept one becomes the shape's algorithm after it also
// matches the first choice on the caller's own operands. Candidates that
// reduce K in another order (split-K, stream-K) disagree on the probes and
// drop out. This is an empirical check over the probe draws, not a proof
// that the kept algorithm equals the first choice for every input; callers
// that need the guarantee use the strict path.
//
// The tuned path takes the first choice untimed for a first call under
// stream capture, for shapes whose probes exceed the probe budget, and when
// the device has no room for the probes. Plans are per device; each stream
// gets its own workspace on its first call, which must therefore not be
// under capture.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt::kernels {

void bf16_matmul_cublaslt_ref_bf16(const __nv_bfloat16* x,
                                   const __nv_bfloat16* W,
                                   __nv_bfloat16* out,
                                   int M, int N, int K,
                                   bool tuned, cudaStream_t stream);

}  // namespace flash_rt::kernels
