// SPDX-License-Identifier: Apache-2.0
//
// Dense products of the prompt-chunk and drafter-context programs: C = A B^T with row-major A [m][k] and
// B [n][k] in BF16, C [m][n] in BF16 or FP32, FP32 accumulation, on cuBLAS through a handle owned by this module.
// The precision is fixed here (CUBLAS_COMPUTE_32F, reduced-precision reductions disallowed), so the products do not
// depend on PyTorch's process-wide matmul settings, and the caller's workspace makes the calls capturable into CUDA
// graphs (each runtime passes its own, so graphs of different runtimes never share one).

#include <cublas_v2.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

void check_cublas(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(s)));
}

std::mutex g_mutex;                                  // a handle is used by one call at a time (stream + workspace)
std::unordered_map<int, cublasHandle_t> g_handles;   // per device

cublasHandle_t handle_locked() {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  cublasHandle_t& h = g_handles[dev];
  if (!h) {
    check_cublas(cublasCreate(&h), "cublasCreate");
    check_cublas(cublasSetMathMode(h, cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)),
                 "cublasSetMathMode");
  }
  return h;
}

}  // namespace

void gemm_bf16(const GemmParams& p, cudaStream_t stream) {
  if (p.m < 1 || p.n < 1 || p.k < 1 || !p.a || !p.b || !p.c || p.lda < p.k || p.ldb < p.k || p.ldc < p.n)
    throw std::invalid_argument("gemm: m, n, k >= 1, a, b, c, lda >= k, ldb >= k, ldc >= n");
  if (!p.workspace || p.workspace_bytes < kGemmWorkspaceBytes)
    throw std::invalid_argument("gemm: a workspace of at least GEMM_WORKSPACE_BYTES");
  const float alpha = 1.0f, beta = 0.0f;
  std::lock_guard<std::mutex> lock(g_mutex);
  cublasHandle_t h = handle_locked();
  check_cublas(cublasSetStream(h, stream), "cublasSetStream");
  // cublasSetStream resets the workspace to cuBLAS's own pool: set the caller's after it
  check_cublas(cublasSetWorkspace(h, p.workspace, p.workspace_bytes), "cublasSetWorkspace");
  // row-major C [m][n] = A B^T is column-major C^T [n][m] = B^T' A^T': op(B) = T over B's [k][n] column-major view
  check_cublas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k, &alpha, p.b, CUDA_R_16BF, p.ldb, p.a,
                            CUDA_R_16BF, p.lda, &beta, p.c, p.out_f32 ? CUDA_R_32F : CUDA_R_16BF, p.ldc,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

}  // namespace inkling2
}  // namespace flash_rt
