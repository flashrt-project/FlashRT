// ================================================================
// FlashRT — cuBLASLt BF16 GEMM restricted to bit-identical algorithms.
// See bf16_matmul_cublaslt_bitexact.cuh for the contract.
//
// Probe size: two accumulation orders give the same bf16 output unless an
// fp32 sum lands within a few fp32 ulps of a bf16 rounding boundary, so a
// different order agrees on any single element with probability close to
// one. The probe therefore compares at least kProbeOutputs elements -- over
// several random activation draws when M * N is small -- which a different
// order cannot pass by chance.
//
// The matmul descriptors and the 32 MiB workspace preference are the ones
// bf16_matmul_cublaslt_bf16 uses, so the heuristic sees the same problem
// and returns the same first choice.
// ================================================================

#include "bf16_matmul_cublaslt_bitexact.cuh"

#include <cublasLt.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace flash_rt::kernels {

namespace {

constexpr int kCandidates = 16;
constexpr long long kProbeOutputs = 1LL << 18;
constexpr int kMaxProbeRounds = 256;
constexpr size_t kMaxProbeBytes = 256ull << 20;   // per output-sized probe buffer
constexpr size_t kWorkspaceBytes = 32ull << 20;

#define FLASHRT_LT_CHECK(expr)                                                 \
  do {                                                                         \
    cublasStatus_t _st = (expr);                                               \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(std::string("cuBLASLt error ") +                \
                               std::to_string(static_cast<int>(_st)) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                                          \
  } while (0)

#define FLASHRT_CUDA_CHECK(expr)                                               \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(_e) + " at " + __FILE__ +    \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  std::vector<cublasLtMatmulAlgo_t> candidates;   // [0] = heuristic first choice
  cublasLtMatmulAlgo_t algo{};
  bool tuned = false;
};

struct Key {
  int M, N, K;
  bool operator==(const Key& o) const { return M == o.M && N == o.N && K == o.K; }
};

struct KeyHash {
  size_t operator()(const Key& k) const {
    size_t h = static_cast<size_t>(k.M);
    h = h * 1315423911u + static_cast<size_t>(k.N);
    return h * 1315423911u + static_cast<size_t>(k.K);
  }
};

// Per device: the handle and the plans (the heuristic answers for the device
// it is asked on). Per (device, stream): the workspace, so GEMMs in flight on
// different streams never share scratch.
struct DeviceState {
  cublasLtHandle_t lt = nullptr;
  std::unordered_map<Key, Plan, KeyHash> plans;
  std::unordered_map<cudaStream_t, void*> workspaces;
};

std::mutex g_mu;
std::unordered_map<int, DeviceState> g_devices;

DeviceState& device_state() {
  int dev = 0;
  FLASHRT_CUDA_CHECK(cudaGetDevice(&dev));
  DeviceState& st = g_devices[dev];
  if (!st.lt) FLASHRT_LT_CHECK(cublasLtCreate(&st.lt));
  return st;
}

void* stream_workspace(DeviceState& st, cudaStream_t stream) {
  auto it = st.workspaces.find(stream);
  if (it != st.workspaces.end()) return it->second;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  FLASHRT_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
  if (cap != cudaStreamCaptureStatusNone) {
    throw std::runtime_error(
        "bf16_matmul_cublaslt_bitexact_bf16: the first call on a stream must "
        "run eagerly (it allocates the stream's workspace), not under capture");
  }
  void* ws = nullptr;
  FLASHRT_CUDA_CHECK(cudaMalloc(&ws, kWorkspaceBytes));
  st.workspaces.emplace(stream, ws);
  return ws;
}

Plan& get_plan(DeviceState& st, int M, int N, int K) {
  auto it = st.plans.find({M, N, K});
  if (it != st.plans.end()) return it->second;

  Plan p;
  const cublasOperation_t op_n = CUBLAS_OP_N, op_t = CUBLAS_OP_T;
  const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
  FLASHRT_LT_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_LT_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, M, K, K));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
      p.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, N, K, K));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
      p.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, M, N, N));
  FLASHRT_LT_CHECK(cublasLtMatrixLayoutSetAttribute(
      p.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws = kWorkspaceBytes;
  FLASHRT_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  cublasLtMatmulHeuristicResult_t found[kCandidates];
  int n = 0;
  FLASHRT_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      st.lt, p.desc, p.a, p.b, p.c, p.c, pref, kCandidates, found, &n));
  cublasLtMatmulPreferenceDestroy(pref);
  if (n == 0) throw std::runtime_error("cuBLASLt: no BF16 GEMM algorithm");
  for (int i = 0; i < n; ++i) p.candidates.push_back(found[i].algo);
  p.algo = p.candidates[0];
  return st.plans.emplace(Key{M, N, K}, p).first->second;
}

struct Launch {
  cublasLtHandle_t lt;
  void* workspace;
  cudaStream_t stream;
};

cublasStatus_t run(const Launch& l, const Plan& p, const cublasLtMatmulAlgo_t& algo,
                   const void* x, const void* W, void* out) {
  const float alpha = 1.0f, beta = 0.0f;
  return cublasLtMatmul(l.lt, p.desc, &alpha, x, p.a, W, p.b, &beta, out, p.c,
                        out, p.c, &algo, l.workspace, kWorkspaceBytes, l.stream);
}

__device__ __forceinline__ uint32_t mix(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
  return v;
}

// Uniform-ish values in [-2, 2), distinct per (round, element).
__global__ void probe_fill_kernel(__nv_bfloat16* x, long long n, uint32_t round) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const uint32_t h = mix(static_cast<uint32_t>(i) * 0x9e3779b9U ^ mix(round + 1));
    x[i] = __float2bfloat16_rn((static_cast<float>(h >> 8) * (1.0f / 16777216.0f)) * 4.0f - 2.0f);
  }
}

__global__ void probe_compare_kernel(const uint16_t* a, const uint16_t* b,
                                     long long n, int* mismatch) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    if (a[i] != b[i]) {
      *mismatch = 1;
      return;
    }
  }
}

// Probe scratch. An allocation failure leaves ptr null and clears the error:
// tuning only changes speed, so a full device keeps the first choice.
struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) {
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
      ptr = nullptr;
      cudaGetLastError();
    }
  }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

void tune(const Launch& l, Plan& p, const __nv_bfloat16* x, const __nv_bfloat16* W,
          __nv_bfloat16* out, int M, int N, int K) {
  const cudaStream_t s = l.stream;
  p.tuned = true;
  const int nc = static_cast<int>(p.candidates.size());
  const size_t out_bytes = static_cast<size_t>(M) * N * sizeof(__nv_bfloat16);
  if (nc <= 1 || out_bytes > kMaxProbeBytes) return;

  // 1. Which candidates reproduce the first choice on random activations.
  const long long outputs = static_cast<long long>(M) * N;
  const int rounds = static_cast<int>(std::min<long long>(
      kMaxProbeRounds, (kProbeOutputs + outputs - 1) / outputs));
  const long long in_elems = static_cast<long long>(M) * K;
  DeviceBuffer px(static_cast<size_t>(in_elems) * sizeof(__nv_bfloat16));
  DeviceBuffer ref(out_bytes), cand(out_bytes);
  DeviceBuffer flags(sizeof(int) * nc);
  if (!px.ptr || !ref.ptr || !cand.ptr || !flags.ptr) return;
  FLASHRT_CUDA_CHECK(cudaMemsetAsync(flags.ptr, 0, sizeof(int) * nc, s));
  int* flag = static_cast<int*>(flags.ptr);
  std::vector<bool> usable(nc, true);
  const int grid = 1024, block = 256;
  for (int r = 0; r < rounds; ++r) {
    probe_fill_kernel<<<grid, block, 0, s>>>(static_cast<__nv_bfloat16*>(px.ptr), in_elems,
                                             static_cast<uint32_t>(r));
    FLASHRT_LT_CHECK(run(l, p, p.candidates[0], px.ptr, W, ref.ptr));
    for (int i = 1; i < nc; ++i) {
      if (!usable[i]) continue;
      if (run(l, p, p.candidates[i], px.ptr, W, cand.ptr) != CUBLAS_STATUS_SUCCESS) {
        usable[i] = false;
        continue;
      }
      probe_compare_kernel<<<grid, block, 0, s>>>(
          static_cast<const uint16_t*>(ref.ptr), static_cast<const uint16_t*>(cand.ptr),
          outputs, flag + i);
    }
  }
  std::vector<int> host(nc, 0);
  FLASHRT_CUDA_CHECK(cudaMemcpyAsync(host.data(), flag, sizeof(int) * nc,
                                     cudaMemcpyDeviceToHost, s));
  FLASHRT_CUDA_CHECK(cudaStreamSynchronize(s));
  for (int i = 1; i < nc; ++i) usable[i] = usable[i] && host[i] == 0;

  // 2. The fastest of them on the real operands.
  cudaEvent_t e0, e1;
  FLASHRT_CUDA_CHECK(cudaEventCreate(&e0));
  FLASHRT_CUDA_CHECK(cudaEventCreate(&e1));
  float best = 0.f;
  int best_i = 0;
  for (int i = 0; i < nc; ++i) {
    if (!usable[i]) continue;
    FLASHRT_LT_CHECK(run(l, p, p.candidates[i], x, W, out));   // warm-up
    FLASHRT_CUDA_CHECK(cudaEventRecord(e0, s));
    for (int rep = 0; rep < 3; ++rep) FLASHRT_LT_CHECK(run(l, p, p.candidates[i], x, W, out));
    FLASHRT_CUDA_CHECK(cudaEventRecord(e1, s));
    FLASHRT_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    FLASHRT_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i == 0 || ms < best) {
      best = ms;
      best_i = i;
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  p.algo = p.candidates[best_i];
}

}  // namespace

void bf16_matmul_cublaslt_bitexact_bf16(const __nv_bfloat16* x,
                                        const __nv_bfloat16* W,
                                        __nv_bfloat16* out,
                                        int M, int N, int K,
                                        cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) return;
  if (!x || !W || !out) {
    throw std::runtime_error("bf16_matmul_cublaslt_bitexact_bf16: null pointer argument");
  }
  std::lock_guard<std::mutex> lock(g_mu);
  DeviceState& st = device_state();
  const Launch l{st.lt, stream_workspace(st, stream), stream};
  Plan& p = get_plan(st, M, N, K);
  if (!p.tuned) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    FLASHRT_CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
    if (cap == cudaStreamCaptureStatusNone) tune(l, p, x, W, out, M, N, K);
  }
  FLASHRT_LT_CHECK(run(l, p, p.algo, x, W, out));
}

}  // namespace flash_rt::kernels
