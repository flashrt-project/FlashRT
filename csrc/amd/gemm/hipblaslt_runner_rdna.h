#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <functional>
#include <unordered_map>

// ================================================================
// FlashRT AMD -- instance-local hipBLASLt runner (RDNA 3.5)
//
// Provides the general BF16 GEMM path used when a shape is not handled by a
// hand-written wave32 kernel. Public matrix semantics are row-major:
//
//   output[rows, columns] = input[rows, inner] @ weight[inner, columns]
//
// hipBLASLt descriptors are column-major, so the implementation swaps the
// logical operands and describes their existing row-major storage without a
// transpose or temporary copy. `weight_stride` and `output_stride` preserve
// support for packed gate/up weights and output views with padded rows.
//
// Descriptor/algorithm entries are cached by operation and complete shape.
// Optional lazy autotuning benchmarks a bounded heuristic candidate set once
// per entry, then keeps the selected algorithm in this runner instance. No
// process-global PyTorch TunableOp state, CSV database, or persisted opaque
// algorithm ID is used. The owning Python model creates one runner and warms
// it before HIP graph capture.
// ================================================================
class RdnaGemmRunner {
public:
  explicit RdnaGemmRunner(std::size_t workspace_size = 128ULL << 20);
  ~RdnaGemmRunner();

  RdnaGemmRunner(const RdnaGemmRunner &) = delete;
  RdnaGemmRunner &operator=(const RdnaGemmRunner &) = delete;

  void enable_lazy_autotune(int num_algorithms = 16);

  void bf16_nn(void *input, void *weight, void *output, int rows, int columns,
               int inner, int weight_stride, int output_stride,
               hipStream_t stream = nullptr);
  void bf16_nn_bias(void *input, void *weight, void *output, void *bias,
                    int rows, int columns, int inner, int weight_stride,
                    int output_stride, hipStream_t stream = nullptr);

  void autotune_bf16_nn(void *input, void *weight, void *output, int rows,
                        int columns, int inner, int weight_stride,
                        int output_stride, int num_algorithms = 16,
                        hipStream_t stream = nullptr);
  void autotune_bf16_nn_bias(void *input, void *weight, void *output,
                             void *bias, int rows, int columns, int inner,
                             int weight_stride, int output_stride,
                             int num_algorithms = 16,
                             hipStream_t stream = nullptr);

private:
  enum class Operation : int { Bf16Nn = 0, Bf16NnBias = 1 };

  struct Key {
    Operation operation;
    int rows;
    int columns;
    int inner;
    int weight_stride;
    int output_stride;

    bool operator==(const Key &other) const {
      return operation == other.operation && rows == other.rows &&
             columns == other.columns && inner == other.inner &&
             weight_stride == other.weight_stride &&
             output_stride == other.output_stride;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key &key) const {
      std::size_t value = std::hash<int>()(static_cast<int>(key.operation));
      value ^= std::hash<int>()(key.rows) + 0x9e3779b9 + (value << 6) +
               (value >> 2);
      value ^= std::hash<int>()(key.columns) + 0x9e3779b9 + (value << 6) +
               (value >> 2);
      value ^= std::hash<int>()(key.inner) + 0x9e3779b9 + (value << 6) +
               (value >> 2);
      value ^= std::hash<int>()(key.weight_stride) + 0x9e3779b9 +
               (value << 6) + (value >> 2);
      value ^= std::hash<int>()(key.output_stride) + 0x9e3779b9 +
               (value << 6) + (value >> 2);
      return value;
    }
  };

  struct Entry {
    // Descriptor lifetimes match the runner. The bias pointer is mutable and
    // is refreshed on every biased call because it is not part of the key.
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight = nullptr;
    hipblasLtMatrixLayout_t input = nullptr;
    hipblasLtMatrixLayout_t output = nullptr;
    hipblasLtMatmulAlgo_t algorithm{};
    bool tuned = false;
  };

  hipblasLtHandle_t handle_ = nullptr;
  void *workspace_ = nullptr;
  std::size_t workspace_size_ = 0;
  bool lazy_autotune_ = false;
  int lazy_pool_ = 16;
  std::unordered_map<Key, Entry, KeyHash> cache_;

  Entry &get_or_create(Operation operation, int rows, int columns, int inner,
                       int weight_stride, int output_stride, void *bias);
  void set_bias(Entry &entry, void *bias);
  void run(Entry &entry, void *input, void *weight, void *output,
           hipStream_t stream);
  void tune(Entry &entry, void *input, void *weight, void *output,
            int num_algorithms, hipStream_t stream);
};
