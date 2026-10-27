// ================================================================
// FlashRT — GR00T N1.5 BF16 GEMMs on the algorithms torch runs.
// See groot_n15_gemm.cuh for the contract.
//
// Both entries describe the product the way torch describes it: the row-major
// result [M, N] is computed as its column-major transpose [N, M], with the
// weight as the left operand. cuBLAS(Lt) picks its algorithm from that
// description (shapes, transposes, leading dimensions, workspace limit and,
// for cuBLASLt, the pointer alignments), so describing it identically is what
// makes the choice -- and the bits -- the same.
// ================================================================

#include "groot_n15_gemm.cuh"

#include <cublasLt.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace flash_rt::groot_n15 {

namespace {

// The workspace limits torch 2.5 gives the two calls, read the way torch
// reads them: CUBLASLT_WORKSPACE_SIZE in KiB (default 1024) for addmm's
// epilogue path, CUBLAS_WORKSPACE_CONFIG as ":SIZE_KiB:COUNT" pairs (default
// ":4096:2:16:8" on non-Hopper parts) for the cuBLAS handle.
size_t lt_workspace_bytes() {
  static const size_t bytes = [] {
    size_t kib = 1024;
    if (const char* v = std::getenv("CUBLASLT_WORKSPACE_SIZE")) {
      try {
        kib = static_cast<size_t>(std::stoul(v));
      } catch (...) {
      }
    }
    return kib * 1024;
  }();
  return bytes;
}

size_t blas_workspace_bytes() {
  static const size_t bytes = [] {
    const size_t fallback = 4096ull * 1024 * 2 + 16ull * 1024 * 8;
    const char* v = std::getenv("CUBLAS_WORKSPACE_CONFIG");
    if (!v) return fallback;
    const std::string config(v);
    const std::regex pair(":([0-9]+):([0-9]+)");
    size_t total = 0;
    for (std::sregex_iterator it(config.begin(), config.end(), pair), end; it != end; ++it) {
      total += static_cast<size_t>(std::stoul((*it)[1].str())) * 1024 *
               static_cast<size_t>(std::stoul((*it)[2].str()));
    }
    return total ? total : fallback;
  }();
  return bytes;
}

#define GROOT_N15_LT_CHECK(expr)                                               \
  do {                                                                         \
    cublasStatus_t _st = (expr);                                               \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(std::string("groot_n15 cuBLAS error ") +        \
                               std::to_string(static_cast<int>(_st)) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                          \
  } while (0)

#define GROOT_N15_CUDA_CHECK(expr)                                             \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("groot_n15 CUDA error: ") +         \
                               cudaGetErrorString(_e) + " at " + __FILE__ +    \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

// torch's _getAlignment: the largest power of two up to 256 dividing the
// address.
uint32_t alignment_of(const void* p) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  uint32_t al = 256;
  while (a % al) al /= 2;
  return al;
}

struct LtPlan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

struct LtKey {
  int M, N, K;
  uint32_t al_a, al_b, al_c, al_d;
  bool operator==(const LtKey& o) const {
    return M == o.M && N == o.N && K == o.K && al_a == o.al_a && al_b == o.al_b &&
           al_c == o.al_c && al_d == o.al_d;
  }
};

struct LtKeyHash {
  size_t operator()(const LtKey& k) const {
    size_t h = static_cast<size_t>(k.M);
    h = h * 1315423911u + static_cast<size_t>(k.N);
    h = h * 1315423911u + static_cast<size_t>(k.K);
    h = h * 31u + (k.al_a ^ (k.al_b << 9) ^ (k.al_c << 18) ^ (k.al_d << 27));
    return h;
  }
};

// Per device: one cuBLASLt handle, one cuBLAS handle, the epilogue plans.
// Per (device, stream): the two workspaces.
struct DeviceState {
  cublasLtHandle_t lt = nullptr;
  cublasHandle_t blas = nullptr;
  std::unordered_map<LtKey, LtPlan, LtKeyHash> plans;
  std::unordered_map<cudaStream_t, void*> lt_ws, blas_ws;
};

std::mutex g_mu;
std::unordered_map<int, DeviceState> g_devices;

DeviceState& device_state() {
  int dev = 0;
  GROOT_N15_CUDA_CHECK(cudaGetDevice(&dev));
  DeviceState& st = g_devices[dev];
  if (!st.lt) GROOT_N15_LT_CHECK(cublasLtCreate(&st.lt));
  if (!st.blas) GROOT_N15_LT_CHECK(cublasCreate(&st.blas));
  return st;
}

void* stream_workspace(std::unordered_map<cudaStream_t, void*>& pool,
                       cudaStream_t stream, size_t bytes, const char* who) {
  auto it = pool.find(stream);
  if (it != pool.end()) return it->second;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  GROOT_N15_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
  if (cap != cudaStreamCaptureStatusNone) {
    throw std::runtime_error(std::string(who) +
                             ": the first call on a stream must run eagerly (it "
                             "allocates the stream's workspace), not under capture");
  }
  void* ws = nullptr;
  GROOT_N15_CUDA_CHECK(cudaMalloc(&ws, bytes));
  pool.emplace(stream, ws);
  return ws;
}

const LtPlan& lt_plan(DeviceState& st, const LtKey& key) {
  auto it = st.plans.find(key);
  if (it != st.plans.end()) return it->second;

  // torch addmm with a row-major result: C^T[N, M] = op(W)[N, K] * x^T[K, M].
  LtPlan p;
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  GROOT_N15_LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  GROOT_N15_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  GROOT_N15_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));
  GROOT_N15_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  GROOT_N15_LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, key.K, key.N, key.K));
  GROOT_N15_LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, key.K, key.M, key.K));
  GROOT_N15_LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, key.N, key.M, key.N));

  cublasLtMatmulPreference_t pref;
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws = lt_workspace_bytes();
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &key.al_a, sizeof(key.al_a)));
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &key.al_b, sizeof(key.al_b)));
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &key.al_c, sizeof(key.al_c)));
  GROOT_N15_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &key.al_d, sizeof(key.al_d)));
  cublasLtMatmulHeuristicResult_t found{};
  int n = 0;
  const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
      st.lt, p.desc, p.a, p.b, p.c, p.c, pref, 1, &found, &n);
  cublasLtMatmulPreferenceDestroy(pref);
  GROOT_N15_LT_CHECK(hs);
  if (n == 0) throw std::runtime_error("groot_n15 linear_bias_bf16: no cuBLASLt algorithm");
  p.algo = found.algo;
  return st.plans.emplace(key, p).first->second;
}

}  // namespace

void linear_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* W,
                      const __nv_bfloat16* bias, __nv_bfloat16* out,
                      int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (!x || !W || !bias || !out) {
    throw std::runtime_error("groot_n15 linear_bias_bf16: null pointer argument");
  }
  // torch only takes the epilogue path for a weight with more than one row
  // and column; the model has no such projection.
  if (N < 2 || K < 2) {
    throw std::runtime_error("groot_n15 linear_bias_bf16: N and K must be at least 2");
  }
  std::lock_guard<std::mutex> lock(g_mu);
  DeviceState& st = device_state();
  void* ws = stream_workspace(st.lt_ws, stream, lt_workspace_bytes(), "groot_n15 linear_bias_bf16");
  // torch passes the bias pointer's alignment as the D alignment.
  const LtKey key{M, N, K, alignment_of(W), alignment_of(x), alignment_of(out),
                  alignment_of(bias)};
  const LtPlan& p = lt_plan(st, key);
  GROOT_N15_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  const float alpha = 1.0f, beta = 0.0f;
  GROOT_N15_LT_CHECK(cublasLtMatmul(st.lt, p.desc, &alpha, W, p.a, x, p.b, &beta, out, p.c,
                                    out, p.c, &p.algo, ws, lt_workspace_bytes(), stream));
}

void bmm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* W,
              __nv_bfloat16* out, int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (!x || !W || !out) {
    throw std::runtime_error("groot_n15 bmm_bf16: null pointer argument");
  }
  std::lock_guard<std::mutex> lock(g_mu);
  DeviceState& st = device_state();
  const size_t ws_bytes = blas_workspace_bytes();
  void* ws = stream_workspace(st.blas_ws, stream, ws_bytes, "groot_n15 bmm_bf16");
  GROOT_N15_LT_CHECK(cublasSetStream(st.blas, stream));
  GROOT_N15_LT_CHECK(cublasSetWorkspace(st.blas, ws, ws_bytes));
  GROOT_N15_LT_CHECK(cublasSetMathMode(st.blas, CUBLAS_DEFAULT_MATH));
  // torch.bmm with a row-major result: C^T[N, M] = W^T[N, K] * x^T[K, M], where
  // the row-major W [K, N] is W^T column-major (ld N) and x [M, K] is x^T (ld K).
  const float alpha = 1.0f, beta = 0.0f;
  GROOT_N15_LT_CHECK(cublasGemmStridedBatchedEx(
      st.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
      W, CUDA_R_16BF, N, static_cast<long long>(K) * N,
      x, CUDA_R_16BF, K, static_cast<long long>(M) * K, &beta,
      out, CUDA_R_16BF, N, static_cast<long long>(M) * N,
      1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace flash_rt::groot_n15
