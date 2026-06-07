#include "gemm/hipblaslt_matmul.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipblaslt/hipblaslt.h>

namespace {

const char* hipblas_status_name(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
      return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:
      return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:
      return "HIPBLAS_STATUS_UNKNOWN";
    default:
      return "HIPBLAS_STATUS_UNRECOGNIZED";
  }
}

void check_hipblas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << what << " failed: " << hipblas_status_name(status)
        << " (" << static_cast<int>(status) << ")";
    throw std::runtime_error(oss.str());
  }
}

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << hipGetErrorString(status);
    throw std::runtime_error(oss.str());
  }
}

struct HipblasLtHandleGuard {
  hipblasLtHandle_t handle = nullptr;

  HipblasLtHandleGuard() {
    check_hipblas(hipblasLtCreate(&handle), "hipblasLtCreate");
  }

  ~HipblasLtHandleGuard() {
    if (handle != nullptr) {
      hipblasLtDestroy(handle);
    }
  }
};

struct LtContext {
  hipblasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 32ull * 1024ull * 1024ull;

  LtContext() {
    check_hipblas(hipblasLtCreate(&handle), "hipblasLtCreate");
    check_hip(hipMalloc(&workspace, workspace_bytes), "hipMalloc(global workspace)");
  }

  ~LtContext() {
    if (workspace != nullptr) {
      (void)hipFree(workspace);
    }
    if (handle != nullptr) {
      hipblasLtDestroy(handle);
    }
  }
};

LtContext& lt_context() {
  static LtContext context;
  return context;
}

struct CachedAlgo {
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_size = 0;
};

std::unordered_map<std::string, CachedAlgo>& algo_cache() {
  static std::unordered_map<std::string, CachedAlgo> cache;
  return cache;
}

std::mutex& algo_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  try {
    return std::max(0, std::stoi(value));
  } catch (...) {
    return fallback;
  }
}

bool env_bool(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

CachedAlgo cached_algo(const std::string& key,
                       const std::function<CachedAlgo()>& create) {
  {
    std::lock_guard<std::mutex> lock(algo_cache_mutex());
    auto it = algo_cache().find(key);
    if (it != algo_cache().end()) {
      return it->second;
    }
  }

  CachedAlgo value = create();
  {
    std::lock_guard<std::mutex> lock(algo_cache_mutex());
    auto [it, inserted] = algo_cache().emplace(key, value);
    return it->second;
  }
}

struct MatmulDescGuard {
  hipblasLtMatmulDesc_t desc = nullptr;

  explicit MatmulDescGuard(hipblasComputeType_t compute_type) {
    check_hipblas(
        hipblasLtMatmulDescCreate(&desc, compute_type, HIP_R_32F),
        "hipblasLtMatmulDescCreate");
  }

  ~MatmulDescGuard() {
    if (desc != nullptr) {
      hipblasLtMatmulDescDestroy(desc);
    }
  }
};

struct MatrixLayoutGuard {
  hipblasLtMatrixLayout_t layout = nullptr;

  MatrixLayoutGuard(hipDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
    check_hipblas(hipblasLtMatrixLayoutCreate(&layout, type, rows, cols, ld),
                  "hipblasLtMatrixLayoutCreate");
  }

  ~MatrixLayoutGuard() {
    if (layout != nullptr) {
      hipblasLtMatrixLayoutDestroy(layout);
    }
  }
};

struct PreferenceGuard {
  hipblasLtMatmulPreference_t pref = nullptr;

  explicit PreferenceGuard(uint64_t max_workspace_bytes) {
    check_hipblas(hipblasLtMatmulPreferenceCreate(&pref),
                  "hipblasLtMatmulPreferenceCreate");
    check_hipblas(
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_bytes, sizeof(max_workspace_bytes)),
        "hipblasLtMatmulPreferenceSetAttribute(workspace)");
  }

  ~PreferenceGuard() {
    if (pref != nullptr) {
      hipblasLtMatmulPreferenceDestroy(pref);
    }
  }
};

struct HipWorkspaceGuard {
  void* ptr = nullptr;

  explicit HipWorkspaceGuard(size_t bytes) {
    if (bytes > 0) {
      check_hip(hipMalloc(&ptr, bytes), "hipMalloc(workspace)");
    }
  }

  ~HipWorkspaceGuard() {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
    }
  }
};

struct LinearBf16Plan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t weight_layout = nullptr;
  hipblasLtMatrixLayout_t x_layout = nullptr;
  hipblasLtMatrixLayout_t out_layout = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_size = 0;
  bool has_bias = false;

  LinearBf16Plan(int64_t m, int64_t n, int64_t k, bool bias, const void* bias_ptr)
      : has_bias(bias) {
    LtContext& lt = lt_context();
    check_hipblas(
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate(linear plan)");

    const hipblasOperation_t trans_lhs = HIPBLAS_OP_T;
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                      &trans_lhs, sizeof(trans_lhs)),
                  "hipblasLtMatmulDescSetAttribute(plan transA)");

    if (has_bias) {
      const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
      check_hipblas(hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                        &epilogue, sizeof(epilogue)),
                    "hipblasLtMatmulDescSetAttribute(plan epilogue)");
      check_hipblas(hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                        &bias_ptr, sizeof(bias_ptr)),
                    "hipblasLtMatmulDescSetAttribute(plan bias)");
      const hipDataType bias_type = HIP_R_16BF;
      check_hipblas(hipblasLtMatmulDescSetAttribute(
                        desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                        &bias_type, sizeof(bias_type)),
                    "hipblasLtMatmulDescSetAttribute(plan bias type)");
    }

    check_hipblas(
        hipblasLtMatrixLayoutCreate(&weight_layout, HIP_R_16BF,
                                    static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(n), k),
        "hipblasLtMatrixLayoutCreate(plan weight)");
    check_hipblas(
        hipblasLtMatrixLayoutCreate(&x_layout, HIP_R_16BF,
                                    static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(m), k),
        "hipblasLtMatrixLayoutCreate(plan x)");
    check_hipblas(
        hipblasLtMatrixLayoutCreate(&out_layout, HIP_R_16BF,
                                    static_cast<uint64_t>(n),
                                    static_cast<uint64_t>(m), n),
        "hipblasLtMatrixLayoutCreate(plan out)");

    PreferenceGuard pref(static_cast<uint64_t>(lt.workspace_bytes));
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    check_hipblas(hipblasLtMatmulAlgoGetHeuristic(
                      lt.handle, desc, weight_layout, x_layout,
                      out_layout, out_layout, pref.pref, 1,
                      &heuristic, &returned),
                  "hipblasLtMatmulAlgoGetHeuristic(linear plan)");
    if (returned <= 0 || heuristic.state != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipBLASLt did not return a usable BF16 Linear plan");
    }
    if (heuristic.workspaceSize > lt.workspace_bytes) {
      throw std::runtime_error("hipBLASLt plan requested too much workspace");
    }
    algo = heuristic.algo;
    workspace_size = heuristic.workspaceSize;
  }

  ~LinearBf16Plan() {
    if (out_layout != nullptr) {
      hipblasLtMatrixLayoutDestroy(out_layout);
    }
    if (x_layout != nullptr) {
      hipblasLtMatrixLayoutDestroy(x_layout);
    }
    if (weight_layout != nullptr) {
      hipblasLtMatrixLayoutDestroy(weight_layout);
    }
    if (desc != nullptr) {
      hipblasLtMatmulDescDestroy(desc);
    }
  }

  LinearBf16Plan(const LinearBf16Plan&) = delete;
  LinearBf16Plan& operator=(const LinearBf16Plan&) = delete;
};

std::unordered_map<std::string, std::shared_ptr<LinearBf16Plan>>& linear_plan_cache() {
  static std::unordered_map<std::string, std::shared_ptr<LinearBf16Plan>> cache;
  return cache;
}

std::mutex& linear_plan_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string linear_bf16_key(int64_t m, int64_t n, int64_t k, bool has_bias) {
  std::ostringstream key;
  key << "linear_bf16:"
      << m << "x" << n << "x" << k << ":bias=" << has_bias;
  return key.str();
}

std::shared_ptr<LinearBf16Plan> get_linear_bf16_plan(
    int64_t m, int64_t n, int64_t k, bool has_bias, const void* bias_ptr) {
  const std::string key = linear_bf16_key(m, n, k, has_bias);
  {
    std::lock_guard<std::mutex> lock(linear_plan_cache_mutex());
    auto it = linear_plan_cache().find(key);
    if (it != linear_plan_cache().end()) {
      return it->second;
    }
  }

  auto plan = std::make_shared<LinearBf16Plan>(m, n, k, has_bias, bias_ptr);
  {
    std::lock_guard<std::mutex> lock(linear_plan_cache_mutex());
    auto [it, inserted] = linear_plan_cache().emplace(key, plan);
    return it->second;
  }
}

}  // namespace

void hipblaslt_matmul_impl(const void* a, const void* b,
                           const float* a_scale, const float* b_scale,
                           void* out,
                           int64_t m, int64_t n, int64_t k,
                           hipDataType input_type,
                           hipblasComputeType_t compute_type,
                           hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("matmul dimensions must be positive");
  }

  LtContext& lt = lt_context();
  MatmulDescGuard op(compute_type);

  // Torch tensors are row-major. hipBLASLt's column-major path is the most
  // stable baseline, so compute D^T = B^T @ A^T directly in the same memory.
  MatrixLayoutGuard b_as_lhs(input_type, static_cast<uint64_t>(n),
                             static_cast<uint64_t>(k), n);
  MatrixLayoutGuard a_as_rhs(input_type, static_cast<uint64_t>(k),
                             static_cast<uint64_t>(m), k);
  MatrixLayoutGuard out_as_transposed(HIP_R_16BF, static_cast<uint64_t>(n),
                                      static_cast<uint64_t>(m), n);

  if (a_scale != nullptr || b_scale != nullptr) {
    if (a_scale == nullptr || b_scale == nullptr) {
      throw std::invalid_argument("FP8 matmul requires both a_scale and b_scale");
    }
    const void* b_scale_ptr = b_scale;
    const void* a_scale_ptr = a_scale;
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                      &b_scale_ptr, sizeof(b_scale_ptr)),
                  "hipblasLtMatmulDescSetAttribute(A scale)");
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                      &a_scale_ptr, sizeof(a_scale_ptr)),
                  "hipblasLtMatmulDescSetAttribute(B scale)");

    const hipblasLtMatmulMatrixScale_t scale_mode =
        HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                      &scale_mode, sizeof(scale_mode)),
                  "hipblasLtMatmulDescSetAttribute(A scale mode)");
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                      &scale_mode, sizeof(scale_mode)),
                  "hipblasLtMatmulDescSetAttribute(B scale mode)");
  }

  PreferenceGuard pref(static_cast<uint64_t>(lt.workspace_bytes));

  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  check_hipblas(hipblasLtMatmulAlgoGetHeuristic(
                    lt.handle, op.desc, b_as_lhs.layout, a_as_rhs.layout,
                    out_as_transposed.layout, out_as_transposed.layout, pref.pref, 1,
                    &heuristic, &returned),
                "hipblasLtMatmulAlgoGetHeuristic");
  if (returned <= 0 || heuristic.state != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("hipBLASLt did not return a usable GEMM algorithm");
  }

  if (heuristic.workspaceSize > lt.workspace_bytes) {
    throw std::runtime_error("hipBLASLt requested more workspace than the ROCm context owns");
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(hipblasLtMatmul(lt.handle, op.desc, &alpha,
                                b, b_as_lhs.layout,
                                a, a_as_rhs.layout,
                                &beta,
                                out, out_as_transposed.layout,
                                out, out_as_transposed.layout,
                                &heuristic.algo,
                                lt.workspace, heuristic.workspaceSize,
                                stream),
                "hipblasLtMatmul");
}

void hipblaslt_matmul_bf16(const void* a, const void* b, void* out,
                           int64_t m, int64_t n, int64_t k,
                           hipStream_t stream) {
  hipblaslt_matmul_impl(a, b, nullptr, nullptr, out, m, n, k,
                        HIP_R_16BF, HIPBLAS_COMPUTE_32F, stream);
}

void hipblaslt_matmul_fp8_e4m3fnuz_bf16(const void* a, const void* b,
                                        const float* a_scale,
                                        const float* b_scale,
                                        void* out,
                                        int64_t m, int64_t n, int64_t k,
                                        hipStream_t stream) {
  hipblaslt_matmul_impl(a, b, a_scale, b_scale, out, m, n, k,
                        HIP_R_8F_E4M3_FNUZ,
                        HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ,
                        stream);
}

void hipblaslt_linear_bf16(const void* x, const void* weight, const void* bias,
                           void* out,
                           int64_t m, int64_t n, int64_t k,
                           hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("linear dimensions must be positive");
  }

  LtContext& lt = lt_context();
  auto plan = get_linear_bf16_plan(m, n, k, bias != nullptr, bias);

  if (bias != nullptr) {
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      plan->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                      &bias, sizeof(bias)),
                  "hipblasLtMatmulDescSetAttribute(plan bias update)");
  }

  if (plan->workspace_size > lt.workspace_bytes) {
    throw std::runtime_error("hipBLASLt plan requested more workspace than the ROCm context owns");
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(hipblasLtMatmul(lt.handle, plan->desc, &alpha,
                                weight, plan->weight_layout,
                                x, plan->x_layout,
                                &beta,
                                out, plan->out_layout,
                                out, plan->out_layout,
                                &plan->algo,
                                lt.workspace, plan->workspace_size,
                                stream),
                "hipblasLtMatmul(linear)");
}

void hipblaslt_linear_fp8_e4m3fnuz_bf16(
    const void* x, const void* weight,
    const float* x_scale, const float* weight_scale,
    const void* bias, void* out,
    int64_t m, int64_t n, int64_t k,
    hipStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("linear dimensions must be positive");
  }
  if (x_scale == nullptr || weight_scale == nullptr) {
    throw std::invalid_argument("FP8 linear requires x_scale and weight_scale");
  }

  LtContext& lt = lt_context();
  MatmulDescGuard op(HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ);

  const hipblasOperation_t trans_lhs = HIPBLAS_OP_T;
  check_hipblas(hipblasLtMatmulDescSetAttribute(
                    op.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                    &trans_lhs, sizeof(trans_lhs)),
                "hipblasLtMatmulDescSetAttribute(transA)");

  const void* weight_scale_ptr = weight_scale;
  const void* x_scale_ptr = x_scale;
  check_hipblas(hipblasLtMatmulDescSetAttribute(
                    op.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    &weight_scale_ptr, sizeof(weight_scale_ptr)),
                "hipblasLtMatmulDescSetAttribute(weight scale)");
  check_hipblas(hipblasLtMatmulDescSetAttribute(
                    op.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    &x_scale_ptr, sizeof(x_scale_ptr)),
                "hipblasLtMatmulDescSetAttribute(x scale)");

  const hipblasLtMatmulMatrixScale_t scale_mode =
      HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  check_hipblas(hipblasLtMatmulDescSetAttribute(
                    op.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    &scale_mode, sizeof(scale_mode)),
                "hipblasLtMatmulDescSetAttribute(A scale mode)");
  check_hipblas(hipblasLtMatmulDescSetAttribute(
                    op.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    &scale_mode, sizeof(scale_mode)),
                "hipblasLtMatmulDescSetAttribute(B scale mode)");

  if (bias != nullptr) {
    const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                      &epilogue, sizeof(epilogue)),
                  "hipblasLtMatmulDescSetAttribute(epilogue)");
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                      &bias, sizeof(bias)),
                  "hipblasLtMatmulDescSetAttribute(bias)");
    const hipDataType bias_type = HIP_R_16BF;
    check_hipblas(hipblasLtMatmulDescSetAttribute(
                      op.desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                      &bias_type, sizeof(bias_type)),
                  "hipblasLtMatmulDescSetAttribute(bias type)");
  }

  MatrixLayoutGuard weight_layout(HIP_R_8F_E4M3_FNUZ, static_cast<uint64_t>(k),
                                  static_cast<uint64_t>(n), k);
  MatrixLayoutGuard x_as_rhs(HIP_R_8F_E4M3_FNUZ, static_cast<uint64_t>(k),
                             static_cast<uint64_t>(m), k);
  MatrixLayoutGuard out_as_transposed(HIP_R_16BF, static_cast<uint64_t>(n),
                                      static_cast<uint64_t>(m), n);

  std::ostringstream key;
  key << "linear_fp8_e4m3fnuz_bf16:"
      << m << "x" << n << "x" << k << ":bias=" << (bias != nullptr);
  CachedAlgo algo = cached_algo(key.str(), [&]() {
    PreferenceGuard pref(static_cast<uint64_t>(lt.workspace_bytes));
    const bool autotune = env_bool("FLASHRT_ROCM_HIPBLASLT_AUTOTUNE");
    const int requested_algos = std::max(
        1, std::min(env_int("FLASHRT_ROCM_HIPBLASLT_MAX_ALGOS",
                            autotune ? 32 : 1),
                    128));
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics(
        static_cast<std::size_t>(requested_algos));
    int returned = 0;
    check_hipblas(hipblasLtMatmulAlgoGetHeuristic(
                      lt.handle, op.desc, weight_layout.layout, x_as_rhs.layout,
                      out_as_transposed.layout, out_as_transposed.layout, pref.pref,
                      requested_algos, heuristics.data(), &returned),
                  "hipblasLtMatmulAlgoGetHeuristic(fp8 linear)");
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    for (int i = 0; i < returned; ++i) {
      if (heuristics[static_cast<std::size_t>(i)].state == HIPBLAS_STATUS_SUCCESS &&
          heuristics[static_cast<std::size_t>(i)].workspaceSize <= lt.workspace_bytes) {
        candidates.push_back(heuristics[static_cast<std::size_t>(i)]);
      }
    }
    if (candidates.empty()) {
      throw std::runtime_error("hipBLASLt did not return a usable FP8 Linear algorithm");
    }
    int selected = std::min(
        env_int("FLASHRT_ROCM_HIPBLASLT_ALGO_INDEX", 0),
        static_cast<int>(candidates.size()) - 1);

    if (autotune && candidates.size() > 1) {
      const int warmup = std::max(
          0, env_int("FLASHRT_ROCM_HIPBLASLT_AUTOTUNE_WARMUP", 2));
      const int repeat = std::max(
          1, env_int("FLASHRT_ROCM_HIPBLASLT_AUTOTUNE_REPEAT", 5));
      hipEvent_t start = nullptr;
      hipEvent_t stop = nullptr;
      check_hip(hipEventCreate(&start), "hipEventCreate(start)");
      check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

      const float alpha = 1.0f;
      const float beta = 0.0f;
      float best_ms = std::numeric_limits<float>::infinity();
      int best_idx = selected;
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        bool ok = true;
        for (int r = 0; r < warmup; ++r) {
          hipblasStatus_t status = hipblasLtMatmul(
              lt.handle, op.desc, &alpha,
              weight, weight_layout.layout,
              x, x_as_rhs.layout,
              &beta,
              out, out_as_transposed.layout,
              out, out_as_transposed.layout,
              &candidates[i].algo,
              lt.workspace, candidates[i].workspaceSize,
              stream);
          if (status != HIPBLAS_STATUS_SUCCESS) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          continue;
        }
        check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(autotune warmup)");
        float total_ms = 0.0f;
        for (int r = 0; r < repeat; ++r) {
          check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
          hipblasStatus_t status = hipblasLtMatmul(
              lt.handle, op.desc, &alpha,
              weight, weight_layout.layout,
              x, x_as_rhs.layout,
              &beta,
              out, out_as_transposed.layout,
              out, out_as_transposed.layout,
              &candidates[i].algo,
              lt.workspace, candidates[i].workspaceSize,
              stream);
          if (status != HIPBLAS_STATUS_SUCCESS) {
            ok = false;
            break;
          }
          check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
          check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
          float elapsed_ms = 0.0f;
          check_hip(hipEventElapsedTime(&elapsed_ms, start, stop),
                    "hipEventElapsedTime");
          total_ms += elapsed_ms;
        }
        if (ok) {
          const float mean_ms = total_ms / static_cast<float>(repeat);
          if (mean_ms < best_ms) {
            best_ms = mean_ms;
            best_idx = static_cast<int>(i);
          }
        }
      }

      if (start != nullptr) {
        (void)hipEventDestroy(start);
      }
      if (stop != nullptr) {
        (void)hipEventDestroy(stop);
      }
      selected = best_idx;
    }

    CachedAlgo value;
    value.algo = candidates[static_cast<std::size_t>(selected)].algo;
    value.workspace_size = candidates[static_cast<std::size_t>(selected)].workspaceSize;
    return value;
  });

  if (algo.workspace_size > lt.workspace_bytes) {
    throw std::runtime_error("hipBLASLt requested more workspace than the ROCm context owns");
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(hipblasLtMatmul(lt.handle, op.desc, &alpha,
                                weight, weight_layout.layout,
                                x, x_as_rhs.layout,
                                &beta,
                                out, out_as_transposed.layout,
                                out, out_as_transposed.layout,
                                &algo.algo,
                                lt.workspace, algo.workspace_size,
                                stream),
                "hipblasLtMatmul(fp8 linear)");
}

std::size_t hipblaslt_algo_cache_size() {
  std::lock_guard<std::mutex> lock(algo_cache_mutex());
  return algo_cache().size();
}

std::vector<std::string> hipblaslt_algo_cache_keys() {
  std::lock_guard<std::mutex> lock(algo_cache_mutex());
  std::vector<std::string> keys;
  keys.reserve(algo_cache().size());
  for (const auto& item : algo_cache()) {
    keys.push_back(item.first);
  }
  return keys;
}

void hipblaslt_algo_cache_clear() {
  std::lock_guard<std::mutex> lock(algo_cache_mutex());
  algo_cache().clear();
}

std::size_t hipblaslt_linear_plan_cache_size() {
  std::lock_guard<std::mutex> lock(linear_plan_cache_mutex());
  return linear_plan_cache().size();
}

std::vector<std::string> hipblaslt_linear_plan_cache_keys() {
  std::lock_guard<std::mutex> lock(linear_plan_cache_mutex());
  std::vector<std::string> keys;
  keys.reserve(linear_plan_cache().size());
  for (const auto& item : linear_plan_cache()) {
    keys.push_back(item.first);
  }
  return keys;
}

void hipblaslt_linear_plan_cache_clear() {
  std::lock_guard<std::mutex> lock(linear_plan_cache_mutex());
  linear_plan_cache().clear();
}
