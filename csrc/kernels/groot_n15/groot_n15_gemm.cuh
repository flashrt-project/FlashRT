// ================================================================
// FlashRT — GR00T N1.5 BF16 GEMMs on the algorithms torch runs.
//
// The GR00T N1.5 reference runs its projections under bf16 autocast, so a
// Linear with a bias is torch's addmm (cuBLASLt with the bias in the fp32
// epilogue, rounded to bf16 once) and the embodiment-specific projections are
// torch.bmm over one batch (cuBLAS strided-batched GEMM, the bias added by a
// separate bf16 op). These two entries issue the same library calls with the
// same descriptors, so the library returns the algorithm torch's call gets:
//
//   linear_bias_bf16: out[M, N] = x[M, K] * W[N, K]^T + bias[N]
//       torch.nn.functional.linear / addmm on CUDA >= 12.1: cuBLASLt,
//       CUBLASLT_EPILOGUE_BIAS, the CUBLASLT_WORKSPACE_SIZE limit (1 MiB
//       unless set), alignment preferences taken from the four pointers,
//       heuristic first choice.
//   bmm_bf16:         out[M, N] = x[M, K] * W[K, N]   (W row-major [K, N])
//       torch.bmm with one batch: cublasGemmStridedBatchedEx, fp32 compute,
//       CUBLAS_GEMM_DEFAULT_TENSOR_OP, default math mode, and the
//       CUBLAS_WORKSPACE_CONFIG workspace (torch's non-Hopper default unless
//       set).
//
// These are the descriptors and defaults of PyTorch 2.5, the release the
// GR00T N1.5 reference pins; the equality holds against a reference process
// running that release with the same two environment variables.
//
// Handles and plans are per device; each stream gets its own workspace on its
// first call, which must therefore not be under stream capture.
// ================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt::groot_n15 {

void linear_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* W,
                      const __nv_bfloat16* bias, __nv_bfloat16* out,
                      int M, int N, int K, cudaStream_t stream);

void bmm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* W,
              __nv_bfloat16* out, int M, int N, int K, cudaStream_t stream);

}  // namespace flash_rt::groot_n15
