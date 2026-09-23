// ================================================================
// FlashRT AMD -- hipBLASLt BF16 runner implementation (RDNA 3.5)
//
// See hipblaslt_runner_rdna.h for the row-major contract. This translation
// unit owns resource creation, descriptor caching, heuristic selection, and
// optional first-use timing. Steady-state execution performs one cached
// hipblasLtMatmul call on the caller's stream and allocates nothing.
// ================================================================

#include "hipblaslt_runner_rdna.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Convert HIP/hipBLASLt status codes to exceptions at the C++ boundary; the
// pybind layer translates these into Python RuntimeError instances.
void check_hip(hipError_t status, const char *operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void check_hipblaslt(hipblasStatus_t status, const char *operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void validate_shape(int rows, int columns, int inner) {
  if (rows < 1 || columns < 1 || inner < 1) {
    throw std::runtime_error("RDNA hipBLASLt GEMM dimensions must be positive");
  }
}

} // namespace

// ── Handle and workspace lifetime ──

RdnaGemmRunner::RdnaGemmRunner(std::size_t workspace_size)
    : workspace_size_(workspace_size) {
  check_hipblaslt(hipblasLtCreate(&handle_), "hipblasLtCreate");
  try {
    if (workspace_size_ != 0) {
      check_hip(hipMalloc(&workspace_, workspace_size_), "hipMalloc workspace");
    }
  } catch (...) {
    hipblasLtDestroy(handle_);
    handle_ = nullptr;
    throw;
  }
}

RdnaGemmRunner::~RdnaGemmRunner() {
  for (auto &[key, entry] : cache_) {
    (void)key;
    if (entry.weight != nullptr) {
      hipblasLtMatrixLayoutDestroy(entry.weight);
    }
    if (entry.input != nullptr) {
      hipblasLtMatrixLayoutDestroy(entry.input);
    }
    if (entry.output != nullptr) {
      hipblasLtMatrixLayoutDestroy(entry.output);
    }
    if (entry.operation != nullptr) {
      hipblasLtMatmulDescDestroy(entry.operation);
    }
  }
  if (workspace_ != nullptr) {
    (void)hipFree(workspace_);
  }
  if (handle_ != nullptr) {
    hipblasLtDestroy(handle_);
  }
}

void RdnaGemmRunner::enable_lazy_autotune(int num_algorithms) {
  if (num_algorithms < 1) {
    throw std::runtime_error("RDNA hipBLASLt autotune pool must be positive");
  }
  lazy_autotune_ = true;
  lazy_pool_ = num_algorithms;
}

// Bias is a runtime pointer rather than cache-key state. Updating only this
// descriptor field lets one shape entry serve every layer with that shape.
void RdnaGemmRunner::set_bias(Entry &entry, void *bias) {
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(entry.operation,
                                     HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                     sizeof(bias)),
      "hipblasLtMatmulDescSetAttribute(BIAS_POINTER)");
}

// ── Descriptor and default-algorithm cache ──

RdnaGemmRunner::Entry &RdnaGemmRunner::get_or_create(
    Operation operation, int rows, int columns, int inner, int weight_stride,
    int output_stride, void *bias) {
  validate_shape(rows, columns, inner);
  if (weight_stride < columns) {
    throw std::runtime_error(
        "RDNA hipBLASLt weight stride must cover all columns");
  }
  if (output_stride < columns) {
    throw std::runtime_error(
        "RDNA hipBLASLt output stride must cover all columns");
  }
  const Key key{
      operation, rows, columns, inner, weight_stride, output_stride};
  const auto found = cache_.find(key);
  if (found != cache_.end()) {
    if (operation == Operation::Bf16NnBias) {
      set_bias(found->second, bias);
    }
    return found->second;
  }

  Entry entry;
  hipblasOperation_t no_transpose = HIPBLAS_OP_N;
  check_hipblaslt(
      hipblasLtMatmulDescCreate(&entry.operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F),
      "hipblasLtMatmulDescCreate");
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(entry.operation,
                                     HIPBLASLT_MATMUL_DESC_TRANSA,
                                     &no_transpose, sizeof(no_transpose)),
      "hipblasLtMatmulDescSetAttribute(TRANSA)");
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(entry.operation,
                                     HIPBLASLT_MATMUL_DESC_TRANSB,
                                     &no_transpose, sizeof(no_transpose)),
      "hipblasLtMatmulDescSetAttribute(TRANSB)");

  if (operation == Operation::Bf16NnBias) {
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    hipDataType bias_type = HIP_R_16BF;
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(entry.operation,
                                       HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epilogue, sizeof(epilogue)),
        "hipblasLtMatmulDescSetAttribute(EPILOGUE)");
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(entry.operation,
                                       HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                       &bias_type, sizeof(bias_type)),
        "hipblasLtMatmulDescSetAttribute(BIAS_DATA_TYPE)");
    set_bias(entry, bias);
  }

  // hipBLASLt uses column-major descriptors. Reinterpreting the row-major
  // tensors and swapping operands computes D_row = A_row @ W_row.
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&entry.weight, HIP_R_16BF, columns, inner,
                                  weight_stride),
      "hipblasLtMatrixLayoutCreate(weight)");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&entry.input, HIP_R_16BF, inner, rows,
                                  inner),
      "hipblasLtMatrixLayoutCreate(input)");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&entry.output, HIP_R_16BF, columns, rows,
                                  output_stride),
      "hipblasLtMatrixLayoutCreate(output)");

  hipblasLtMatmulPreference_t preference = nullptr;
  check_hipblaslt(hipblasLtMatmulPreferenceCreate(&preference),
                  "hipblasLtMatmulPreferenceCreate");
  check_hipblaslt(
      hipblasLtMatmulPreferenceSetAttribute(
          preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size_, sizeof(workspace_size_)),
      "hipblasLtMatmulPreferenceSetAttribute");
  hipblasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  const hipblasStatus_t heuristic_status = hipblasLtMatmulAlgoGetHeuristic(
      handle_, entry.operation, entry.weight, entry.input, entry.output,
      entry.output, preference, 1, &heuristic, &returned);
  hipblasLtMatmulPreferenceDestroy(preference);
  check_hipblaslt(heuristic_status, "hipblasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    throw std::runtime_error("RDNA hipBLASLt found no BF16 GEMM algorithm");
  }
  entry.algorithm = heuristic.algo;

  auto [inserted, unused] = cache_.emplace(key, entry);
  (void)unused;
  return inserted->second;
}

// ── Steady-state launch ──

void RdnaGemmRunner::run(Entry &entry, void *input, void *weight, void *output,
                         hipStream_t stream) {
  float alpha = 1.0f;
  float beta = 0.0f;
  check_hipblaslt(
      hipblasLtMatmul(handle_, entry.operation, &alpha, weight, entry.weight,
                     input, entry.input, &beta, output, entry.output, output,
                     entry.output, &entry.algorithm, workspace_,
                     workspace_size_, stream),
      "hipblasLtMatmul");
}

// ── Optional first-use autotuning ──
// Warm each candidate, reject runtime failures, then time ten launches with
// HIP events on the caller's stream. The resulting algorithm is process-local
// and valid only for the current runtime/device plus the descriptor key.
void RdnaGemmRunner::tune(Entry &entry, void *input, void *weight,
                          void *output, int num_algorithms,
                          hipStream_t stream) {
  if (entry.tuned) {
    return;
  }
  if (num_algorithms < 1) {
    throw std::runtime_error("RDNA hipBLASLt autotune pool must be positive");
  }

  hipblasLtMatmulPreference_t preference = nullptr;
  check_hipblaslt(hipblasLtMatmulPreferenceCreate(&preference),
                  "hipblasLtMatmulPreferenceCreate");
  check_hipblaslt(
      hipblasLtMatmulPreferenceSetAttribute(
          preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_size_, sizeof(workspace_size_)),
      "hipblasLtMatmulPreferenceSetAttribute");
  std::vector<hipblasLtMatmulHeuristicResult_t> candidates(num_algorithms);
  int returned = 0;
  const hipblasStatus_t heuristic_status = hipblasLtMatmulAlgoGetHeuristic(
      handle_, entry.operation, entry.weight, entry.input, entry.output,
      entry.output, preference, num_algorithms, candidates.data(), &returned);
  hipblasLtMatmulPreferenceDestroy(preference);
  check_hipblaslt(heuristic_status, "hipblasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    entry.tuned = true;
    return;
  }

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check_hip(hipEventCreate(&start), "hipEventCreate(start)");
  check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

  constexpr int warmup_iterations = 3;
  constexpr int benchmark_iterations = 10;
  float best_milliseconds = 0.0f;
  int best_index = -1;
  float alpha = 1.0f;
  float beta = 0.0f;

  for (int index = 0; index < returned; ++index) {
    bool usable = true;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      const hipblasStatus_t status = hipblasLtMatmul(
          handle_, entry.operation, &alpha, weight, entry.weight, input,
          entry.input, &beta, output, entry.output, output, entry.output,
          &candidates[index].algo, workspace_, workspace_size_, stream);
      if (status != HIPBLAS_STATUS_SUCCESS) {
        usable = false;
        break;
      }
    }
    if (!usable || hipStreamSynchronize(stream) != hipSuccess) {
      continue;
    }

    check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
      const hipblasStatus_t status = hipblasLtMatmul(
          handle_, entry.operation, &alpha, weight, entry.weight, input,
          entry.input, &beta, output, entry.output, output, entry.output,
          &candidates[index].algo, workspace_, workspace_size_, stream);
      if (status != HIPBLAS_STATUS_SUCCESS) {
        usable = false;
        break;
      }
    }
    if (!usable) {
      continue;
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float milliseconds = 0.0f;
    check_hip(hipEventElapsedTime(&milliseconds, start, stop),
              "hipEventElapsedTime");
    milliseconds /= benchmark_iterations;
    if (best_index < 0 || milliseconds < best_milliseconds) {
      best_index = index;
      best_milliseconds = milliseconds;
    }
  }

  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  if (best_index >= 0) {
    entry.algorithm = candidates[best_index].algo;
  }
  entry.tuned = true;
}

// ── Public operation entry points ──

void RdnaGemmRunner::bf16_nn(void *input, void *weight, void *output, int rows,
                             int columns, int inner, int weight_stride,
                             int output_stride, hipStream_t stream) {
  Entry &entry = get_or_create(Operation::Bf16Nn, rows, columns, inner,
                               weight_stride, output_stride, nullptr);
  if (lazy_autotune_ && !entry.tuned) {
    tune(entry, input, weight, output, lazy_pool_, stream);
  }
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::bf16_nn_bias(void *input, void *weight, void *output,
                                  void *bias, int rows, int columns, int inner,
                                  int weight_stride, int output_stride,
                                  hipStream_t stream) {
  if (bias == nullptr) {
    throw std::runtime_error("RDNA hipBLASLt bias pointer must not be null");
  }
  Entry &entry = get_or_create(Operation::Bf16NnBias, rows, columns, inner,
                               weight_stride, output_stride, bias);
  if (lazy_autotune_ && !entry.tuned) {
    tune(entry, input, weight, output, lazy_pool_, stream);
  }
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::autotune_bf16_nn(void *input, void *weight, void *output,
                                      int rows, int columns, int inner,
                                      int weight_stride, int output_stride,
                                      int num_algorithms, hipStream_t stream) {
  Entry &entry = get_or_create(Operation::Bf16Nn, rows, columns, inner,
                               weight_stride, output_stride, nullptr);
  tune(entry, input, weight, output, num_algorithms, stream);
}

void RdnaGemmRunner::autotune_bf16_nn_bias(
    void *input, void *weight, void *output, void *bias, int rows, int columns,
    int inner, int weight_stride, int output_stride, int num_algorithms,
    hipStream_t stream) {
  if (bias == nullptr) {
    throw std::runtime_error("RDNA hipBLASLt bias pointer must not be null");
  }
  Entry &entry = get_or_create(Operation::Bf16NnBias, rows, columns, inner,
                               weight_stride, output_stride, bias);
  tune(entry, input, weight, output, num_algorithms, stream);
}
