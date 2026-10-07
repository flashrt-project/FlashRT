// See attn_lt.cuh.
#include "kernels/pi05/thor/attn_lt.cuh"
#include "kernels/pi05/thor/pdl.cuh"
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace flash_rt {
namespace fp4 {
namespace {

// Same math as kernels/softmax.cu softmax_fp16_kernel (one warp per row, cols <= 1024).
constexpr int SM_WARP = 32, SM_MAX_COLS = 1024, SM_ITERS = SM_MAX_COLS / SM_WARP;
__global__ void softmax_rows_fp16_kernel(__half* data, int rows, int cols) {
  flashrt_pdl_wait_and_trigger();
  const int lane = threadIdx.x % SM_WARP, row = blockIdx.x;
  if (row >= rows) return;
  __half* src = data + static_cast<size_t>(row) * cols;
  const int cols2 = cols / 2;
  __half2* src2 = reinterpret_cast<__half2*>(src);
  float reg[SM_ITERS];
  float mx = -1e30f;
  #pragma unroll
  for (int it = 0; it < SM_ITERS / 2; it++) {
    const int c2 = it * SM_WARP + lane;
    if (c2 < cols2) {
      const __half2 v2 = src2[c2];
      reg[it * 2] = __half2float(v2.x); reg[it * 2 + 1] = __half2float(v2.y);
      mx = fmaxf(mx, fmaxf(reg[it * 2], reg[it * 2 + 1]));
    } else { reg[it * 2] = -1e30f; reg[it * 2 + 1] = -1e30f; }
  }
  if ((cols & 1) && lane == 0) { const float v = __half2float(src[cols - 1]); reg[SM_ITERS - 1] = v; mx = fmaxf(mx, v); }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sm = 0.f;
  #pragma unroll
  for (int it = 0; it < SM_ITERS; it++) { reg[it] = __expf(reg[it] - mx); sm += reg[it]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
  const float inv = 1.f / (sm + 1e-8f);
  #pragma unroll
  for (int it = 0; it < SM_ITERS / 2; it++) {
    const int c2 = it * SM_WARP + lane;
    if (c2 < cols2) { __half2 v2; v2.x = __float2half(reg[it * 2] * inv); v2.y = __float2half(reg[it * 2 + 1] * inv); src2[c2] = v2; }
  }
  if ((cols & 1) && lane == 0) src[cols - 1] = __float2half(reg[SM_ITERS - 1] * inv);
}

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool ready = false;
};
struct Entry { int M = 0, T = 0, HD = 0; Plan qk, pv; bool valid = false; };
cublasLtHandle_t g_lt = nullptr;
std::mutex g_mu;
Entry g_cache[8];
int g_next = 0;

static float time_algo(cublasLtHandle_t lt, Plan& p, const void* alpha, const void* A, const void* B, const void* beta, void* C,
                       cudaStream_t st, int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; ++i) cublasLtMatmul(lt, p.desc, alpha, A, p.la, B, p.lb, beta, C, p.lc, C, p.lc, &p.algo, nullptr, 0, st);
  cudaEventRecord(a, st);
  for (int i = 0; i < reps; ++i) cublasLtMatmul(lt, p.desc, alpha, A, p.la, B, p.lb, beta, C, p.lc, C, p.lc, &p.algo, nullptr, 0, st);
  cudaEventRecord(b, st); cudaEventSynchronize(b);
  float ms = 0.f; cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
  return ms;
}

// Build the plan for a col-major GEMM (opA, opB, m, n, k, lda, ldb, ldc) and pick the fastest heuristic
// (workspace-free) by timing on this stream; falls back to heuristic 0 while a graph is being captured.
static bool make_plan(Plan& p, cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, int lda, int ldb, int ldc,
                      const void* alpha, const void* A, const void* B, const void* beta, void* C, cudaStream_t st) {
  if (cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof(oa));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof(ob));
  const int ra = oa == CUBLAS_OP_N ? m : k, ca = oa == CUBLAS_OP_N ? k : m;
  const int rb = ob == CUBLAS_OP_N ? k : n, cb = ob == CUBLAS_OP_N ? n : k;
  cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, ra, ca, lda);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, rb, cb, ldb);
  cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16F, m, n, ldc);
  cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = 0; cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  cublasLtMatmulHeuristicResult_t res[16]; int nres = 0;
  const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_lt, p.desc, p.la, p.lb, p.lc, p.lc, pref, 16, res, &nres);
  cublasLtMatmulPreferenceDestroy(pref);
  if (hs != CUBLAS_STATUS_SUCCESS || nres == 0) return false;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(st, &cs);
  int best = 0;
  if (cs == cudaStreamCaptureStatusNone) {
    float best_ms = 1e30f;
    for (int i = 0; i < nres; ++i) {
      p.algo = res[i].algo;
      if (cublasLtMatmul(g_lt, p.desc, alpha, A, p.la, B, p.lb, beta, C, p.lc, C, p.lc, &p.algo, nullptr, 0, st) != CUBLAS_STATUS_SUCCESS) continue;
      const float ms = time_algo(g_lt, p, alpha, A, B, beta, C, st, 20);
      if (ms < best_ms) { best_ms = ms; best = i; }
    }
  }
  p.algo = res[best].algo;
  p.ready = true;
  return true;
}

}  // namespace

int attention_qkv_fp16_lt(const void* Q, const void* K, const void* V, void* logits, void* out,
                          int S, int S_kv, int NH, int HD, float attn_scale, cudaStream_t stream) {
  if (S < 1 || S_kv < 1 || S_kv > SM_MAX_COLS || NH < 1 || HD < 1) return -1;
  const int M = S * NH;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -2;
  Entry* e = nullptr;
  for (auto& c : g_cache) if (c.valid && c.M == M && c.T == S_kv && c.HD == HD) e = &c;
  const float zero = 0.f, one = 1.f;
  if (!e) {
    e = &g_cache[g_next]; g_next = (g_next + 1) % 8;
    *e = Entry{}; e->M = M; e->T = S_kv; e->HD = HD;
    // QK^T: logits(S_kv, M) = K^T(S_kv, HD) * Q(HD, M)   (col-major, as attention_qkv_fp16)
    if (!make_plan(e->qk, CUBLAS_OP_T, CUBLAS_OP_N, S_kv, M, HD, HD, HD, S_kv, &attn_scale, K, Q, &zero, logits, stream)) return -3;
    // PV: out(HD, M) = V(HD, S_kv) * logits(S_kv, M)
    if (!make_plan(e->pv, CUBLAS_OP_N, CUBLAS_OP_N, HD, M, S_kv, HD, S_kv, HD, &one, V, logits, &zero, out, stream)) return -3;
    e->valid = true;
  }
  if (cublasLtMatmul(g_lt, e->qk.desc, &attn_scale, K, e->qk.la, Q, e->qk.lb, &zero, logits, e->qk.lc, logits, e->qk.lc, &e->qk.algo, nullptr, 0, stream) != CUBLAS_STATUS_SUCCESS) return -4;
  launch_maybe_pdl(softmax_rows_fp16_kernel, dim3(M), dim3(SM_WARP), 0, stream, static_cast<__half*>(logits), M, S_kv);
  if (cublasLtMatmul(g_lt, e->pv.desc, &one, V, e->pv.la, logits, e->pv.lb, &zero, out, e->pv.lc, out, e->pv.lc, &e->pv.algo, nullptr, 0, stream) != CUBLAS_STATUS_SUCCESS) return -5;
  const cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -static_cast<int>(err);
}

}  // namespace fp4
}  // namespace flash_rt
