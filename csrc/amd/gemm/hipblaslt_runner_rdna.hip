// ================================================================
// FlashRT AMD -- hipBLASLt BF16/FP32 runner implementation (RDNA 3.5)
//
// See hipblaslt_runner_rdna.h for the row-major contract. This translation
// unit owns resource creation, descriptor caching, heuristic selection, and
// optional first-use timing. Steady-state execution performs one cached
// hipblasLtMatmul call on the caller's stream and allocates nothing.
// ================================================================

#include "hipblaslt_runner_rdna.h"

#include <algorithm>
#include <cstring>
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
    int output_stride, void *bias, int input_stride, int batches,
    int input_batch_stride, int weight_batch_stride, int output_batch_stride) {
  validate_shape(rows, columns, inner);
  const bool transposed =
      operation == Operation::Bf16Nt || operation == Operation::Bf16NtBias ||
      operation == Operation::Fp32Nt || operation == Operation::Fp32BmmNt ||
      operation == Operation::Bf16BmmNt || operation == Operation::Fp32GqaBmmNt;
  const bool with_bias =
      operation == Operation::Bf16NnBias || operation == Operation::Bf16NtBias;
  const bool fp32 = operation == Operation::Fp32Nt ||
                    operation == Operation::Fp32BmmNn ||
                    operation == Operation::Fp32BmmNt ||
                    operation == Operation::Fp32GqaBmmNn ||
                    operation == Operation::Fp32GqaBmmNt;
  const hipDataType dtype = fp32 ? HIP_R_32F : HIP_R_16BF;
  if (weight_stride < (transposed ? inner : columns)) {
    throw std::runtime_error(
        "RDNA hipBLASLt weight stride must cover all columns");
  }
  if (output_stride < columns) {
    throw std::runtime_error(
        "RDNA hipBLASLt output stride must cover all columns");
  }
  const Key key{operation,
                rows,
                columns,
                inner,
                weight_stride,
                output_stride,
                input_stride,
                batches,
                input_batch_stride,
                weight_batch_stride,
                output_batch_stride};
  const auto found = cache_.find(key);
  if (found != cache_.end()) {
    if (with_bias) {
      set_bias(found->second, bias);
    }
    return found->second;
  }

  Entry entry;
  hipblasOperation_t no_transpose = HIPBLAS_OP_N;
  hipblasOperation_t weight_operation =
      transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  check_hipblaslt(
      hipblasLtMatmulDescCreate(&entry.operation, HIPBLAS_COMPUTE_32F,
                                HIP_R_32F),
      "hipblasLtMatmulDescCreate");
  check_hipblaslt(hipblasLtMatmulDescSetAttribute(
                      entry.operation, HIPBLASLT_MATMUL_DESC_TRANSA,
                      &weight_operation, sizeof(weight_operation)),
                  "hipblasLtMatmulDescSetAttribute(TRANSA)");
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(entry.operation,
                                     HIPBLASLT_MATMUL_DESC_TRANSB,
                                     &no_transpose, sizeof(no_transpose)),
      "hipblasLtMatmulDescSetAttribute(TRANSB)");

  if (with_bias) {
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
  check_hipblaslt(hipblasLtMatrixLayoutCreate(
                      &entry.weight, dtype, transposed ? inner : columns,
                      transposed ? columns : inner, weight_stride),
                  "hipblasLtMatrixLayoutCreate(weight)");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&entry.input, dtype, inner, rows,
                                  input_stride ? input_stride : inner),
      "hipblasLtMatrixLayoutCreate(input)");
  check_hipblaslt(hipblasLtMatrixLayoutCreate(&entry.output, dtype, columns,
                                              rows, output_stride),
                  "hipblasLtMatrixLayoutCreate(output)");

  if (batches > 1) {
    auto set_batch = [batches](hipblasLtMatrixLayout_t layout, int stride) {
      const int64_t offset = stride;
      check_hipblaslt(hipblasLtMatrixLayoutSetAttribute(
                          layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches,
                          sizeof(batches)),
                      "hipblasLtMatrixLayoutSetAttribute(BATCH_COUNT)");
      check_hipblaslt(
          hipblasLtMatrixLayoutSetAttribute(
              layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &offset,
              sizeof(offset)),
          "hipblasLtMatrixLayoutSetAttribute(STRIDED_BATCH_OFFSET)");
    };
    set_batch(entry.input, input_batch_stride);
    set_batch(entry.weight, weight_batch_stride);
    set_batch(entry.output, output_batch_stride);
  }

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
    throw std::runtime_error("RDNA hipBLASLt found no GEMM algorithm");
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

// ── Reference-output calibration before graph capture ──
// Keep the cached algorithm's output while screening every calibration input.
// Select the fastest surviving candidate only after all layers have run.
void RdnaGemmRunner::begin_bf16_nt_calibration(int rows, int columns, int inner,
                                               bool biased,
                                               int num_algorithms) {
  validate_shape(rows, columns, inner);
  if (num_algorithms < 1)
    throw std::runtime_error(
        "RDNA hipBLASLt calibration pool must be positive");
  calibration_rows_ = rows;
  calibration_columns_ = columns;
  calibration_inner_ = inner;
  calibration_biased_ = biased;
  calibration_pool_ = num_algorithms;
  calibrating_ = true;
}

void RdnaGemmRunner::calibrate(const Key &key, Entry &entry, void *input,
                               void *weight, void *output, hipStream_t stream) {
  auto &calibration = calibrations_[key];
  const size_t element_bytes = key.operation == Operation::Fp32GqaBmmNn
                                   ? sizeof(float)
                                   : sizeof(std::uint16_t);
  const size_t row_bytes = static_cast<size_t>(key.columns) * element_bytes;
  const size_t output_bytes = row_bytes * key.rows * key.batches;
  auto read_output = [&](std::vector<std::uint8_t> &target) {
    for (int batch = 0; batch < key.batches; ++batch) {
      check_hip(hipMemcpy2DAsync(
                    target.data() + batch * key.rows * row_bytes, row_bytes,
                    static_cast<char *>(output) +
                        batch * key.output_batch_stride * element_bytes,
                    static_cast<size_t>(key.output_stride) * element_bytes,
                    row_bytes, key.rows, hipMemcpyDeviceToHost, stream),
                "calibration read output");
    }
    check_hip(hipStreamSynchronize(stream), "calibration synchronize");
  };
  const bool first = calibration.samples == 0;
  if (first) {
    calibration.reference.resize(output_bytes);
    calibration.actual.resize(calibration.reference.size());
    hipblasLtMatmulPreference_t preference = nullptr;
    check_hipblaslt(hipblasLtMatmulPreferenceCreate(&preference),
                    "calibration preference");
    check_hipblaslt(hipblasLtMatmulPreferenceSetAttribute(
                        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &workspace_size_, sizeof(workspace_size_)),
                    "calibration workspace limit");
    std::vector<hipblasLtMatmulHeuristicResult_t> pool(calibration_pool_);
    int returned = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle_, entry.operation, entry.weight, entry.input, entry.output,
        entry.output, preference, calibration_pool_, pool.data(), &returned);
    hipblasLtMatmulPreferenceDestroy(preference);
    check_hipblaslt(status, "calibration heuristic pool");
    for (int i = 0; i < returned; ++i)
      calibration.candidates.push_back({pool[i].algo});
  }
  run(entry, input, weight, output, stream);
  read_output(calibration.reference);
  hipEvent_t start = nullptr, stop = nullptr;
  if (first) {
    check_hip(hipEventCreate(&start), "calibration start event");
    check_hip(hipEventCreate(&stop), "calibration stop event");
  }
  const float alpha = 1.0f, beta = 0.0f;
  for (auto &candidate : calibration.candidates) {
    if (!candidate.valid)
      continue;
    auto launch = [&]() {
      return hipblasLtMatmul(
          handle_, entry.operation, &alpha, weight, entry.weight, input,
          entry.input, &beta, output, entry.output, output, entry.output,
          &candidate.algorithm, workspace_, workspace_size_, stream);
    };
    candidate.valid = launch() == HIPBLAS_STATUS_SUCCESS;
    if (!candidate.valid)
      continue;
    read_output(calibration.actual);
    candidate.valid = std::memcmp(calibration.reference.data(),
                                  calibration.actual.data(), output_bytes) == 0;
    if (!candidate.valid || !first)
      continue;
    for (int i = 0; i < 3; ++i)
      candidate.valid &= launch() == HIPBLAS_STATUS_SUCCESS;
    if (!candidate.valid)
      continue;
    check_hip(hipEventRecord(start, stream), "calibration time start");
    for (int i = 0; i < 10; ++i)
      candidate.valid &= launch() == HIPBLAS_STATUS_SUCCESS;
    check_hip(hipEventRecord(stop, stream), "calibration time stop");
    check_hip(hipEventSynchronize(stop), "calibration time synchronize");
    check_hip(hipEventElapsedTime(&candidate.milliseconds, start, stop),
              "calibration elapsed time");
    candidate.milliseconds /= 10;
  }
  if (first) {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
  }
  ++calibration.samples;
  run(entry, input, weight, output, stream);
}

int RdnaGemmRunner::end_calibration() {
  int accepted = 0;
  for (auto &[key, calibration] : calibrations_) {
    auto &entry = cache_.at(key);
    const Calibration::Candidate *best = nullptr;
    for (const auto &candidate : calibration.candidates) {
      if (!candidate.valid)
        continue;
      ++accepted;
      if (best == nullptr || candidate.milliseconds < best->milliseconds)
        best = &candidate;
    }
    if (best != nullptr)
      entry.algorithm = best->algorithm;
    entry.tuned = true;
  }
  calibrations_.clear();
  return accepted;
}

int RdnaGemmRunner::end_bf16_nt_calibration() {
  calibrating_ = false;
  return end_calibration();
}

void RdnaGemmRunner::begin_fp32_gqa_pv_calibration(int num_algorithms) {
  if (num_algorithms < 1)
    throw std::runtime_error(
        "RDNA hipBLASLt calibration pool must be positive");
  calibration_pool_ = num_algorithms;
  fp32_gqa_pv_calibrating_ = true;
}

int RdnaGemmRunner::end_fp32_gqa_pv_calibration() {
  fp32_gqa_pv_calibrating_ = false;
  return end_calibration();
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

void RdnaGemmRunner::bf16_bmm(void *input, void *weight, void *output, int rows,
                              int columns, int inner, int input_stride,
                              int weight_stride, int output_stride,
                              int input_batch_stride, int weight_batch_stride,
                              int output_batch_stride, int batches,
                              bool transposed, hipStream_t stream) {
  Entry &entry = get_or_create(
      transposed ? Operation::Bf16BmmNt : Operation::Bf16BmmNn, rows, columns,
      inner, weight_stride, output_stride, nullptr, input_stride, batches,
      input_batch_stride, weight_batch_stride, output_batch_stride);
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::fp32_bmm(void *input, void *weight, void *output, int rows,
                              int columns, int inner, int input_stride,
                              int weight_stride, int output_stride,
                              int input_batch_stride, int weight_batch_stride,
                              int output_batch_stride, int batches,
                              bool transposed, hipStream_t stream) {
  Entry &entry = get_or_create(
      transposed ? Operation::Fp32BmmNt : Operation::Fp32BmmNn, rows, columns,
      inner, weight_stride, output_stride, nullptr, input_stride, batches,
      input_batch_stride, weight_batch_stride, output_batch_stride);
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::fp32_gqa_bmm(void *input, void *weight, void *output,
                                  int queries, int keys, int dim, int heads,
                                  int kv_heads, int query_stride,
                                  bool transposed, hipStream_t stream) {
  fp32_gqa_bmm_rows(input, weight, output, queries, queries, keys, dim, heads,
                    kv_heads, query_stride, transposed, stream);
}

void RdnaGemmRunner::fp32_gqa_bmm_rows(void *input, void *weight, void *output,
                                       int queries, int reference_queries,
                                       int keys, int dim, int heads,
                                       int kv_heads, int query_stride,
                                       bool transposed, hipStream_t stream) {
  const int groups = heads / kv_heads;
  const auto operation =
      transposed ? Operation::Fp32BmmNt : Operation::Fp32BmmNn;
  // Keep the original head-wise algorithm and its FP32 accumulation order.
  Entry &source = get_or_create(
      operation, reference_queries, transposed ? keys : dim,
      transposed ? dim : keys, heads * dim, transposed ? keys : dim, nullptr,
      transposed ? query_stride : keys, heads,
      transposed ? dim : reference_queries * keys, dim,
      transposed ? reference_queries * keys : reference_queries * dim);
  const auto grouped_operation =
      transposed ? Operation::Fp32GqaBmmNt : Operation::Fp32GqaBmmNn;
  Entry &entry = get_or_create(
      grouped_operation, queries * groups, transposed ? keys : dim,
      transposed ? dim : keys, kv_heads * dim, transposed ? keys : dim, nullptr,
      transposed ? dim : keys, kv_heads,
      transposed ? queries * groups * dim : queries * groups * keys, dim,
      transposed ? queries * groups * keys : queries * groups * dim);
  if (!entry.tuned)
    entry.algorithm = source.algorithm;
  if (fp32_gqa_pv_calibrating_ && !transposed && heads == 32 && !entry.tuned) {
    const Key key{grouped_operation,
                  queries * groups,
                  dim,
                  keys,
                  kv_heads * dim,
                  dim,
                  keys,
                  kv_heads,
                  queries * groups * keys,
                  dim,
                  queries * groups * dim};
    calibrate(key, entry, input, weight, output, stream);
    return;
  }
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::fp32_nt(void *input, void *weight, void *output, int rows,
                             int columns, int inner, int weight_stride,
                             int output_stride, hipStream_t stream) {
  Entry &entry = get_or_create(Operation::Fp32Nt, rows, columns, inner,
                               weight_stride, output_stride, nullptr);
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::bf16_nt(void *input, void *weight, void *output,
                             void *bias, int rows, int columns, int inner,
                             int weight_stride, int output_stride,
                             hipStream_t stream) {
  const auto operation = bias ? Operation::Bf16NtBias : Operation::Bf16Nt;
  Entry &entry = get_or_create(operation, rows, columns, inner, weight_stride,
                               output_stride, bias);
  if (calibrating_ && !entry.tuned && rows == calibration_rows_ &&
      columns == calibration_columns_ && inner == calibration_inner_ &&
      (bias != nullptr) == calibration_biased_) {
    const Key key{
        operation, rows, columns, inner, weight_stride, output_stride, 0,
        1,         0,    0,       0};
    calibrate(key, entry, input, weight, output, stream);
    return;
  }
  if (lazy_autotune_ && !entry.tuned) {
    tune(entry, input, weight, output, lazy_pool_, stream);
  }
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::bf16_nt_rows(void *input, void *weight, void *output,
                                  void *bias, int rows, int reference_rows,
                                  int columns, int inner, int weight_stride,
                                  int output_stride, hipStream_t stream) {
  const auto operation = bias ? Operation::Bf16NtBias : Operation::Bf16Nt;
  Entry &source = get_or_create(operation, reference_rows, columns, inner,
                                weight_stride, output_stride, bias);
  // Preserve the padded-shape accumulation order without changing either cache.
  Entry entry = get_or_create(operation, rows, columns, inner, weight_stride,
                              output_stride, bias);
  entry.algorithm = source.algorithm;
  run(entry, input, weight, output, stream);
}

void RdnaGemmRunner::bf16_nt_pair(void *input, void *weight, void *output,
                                  int rows, int reference_rows, int columns,
                                  int inner, hipStream_t stream) {
  Entry &source = get_or_create(Operation::Bf16Nt, reference_rows, columns,
                                inner, inner, columns, nullptr);
  // Batch the two projections without changing either matrix's reduction tiles.
  Entry entry =
      get_or_create(Operation::Bf16BmmNt, rows, columns, inner, inner, columns,
                    nullptr, inner, 2, 0, columns * inner, rows * columns);
  entry.algorithm = source.algorithm;
  run(entry, input, weight, output, stream);
}

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
