// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_wy_bf16.cuh"

#include <cuda_bf16.h>
#include <cublasLt.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace flash_rt {
namespace kernels {
namespace linear_attention {

namespace {

constexpr int kChunk = 64;
constexpr int kThreads = 256;

#define FLASHRT_CUBLASLT_CHECK(expr)                                      \
  do {                                                                    \
    cublasStatus_t _st = (expr);                                          \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                   \
      throw std::runtime_error(std::string("cuBLASLt error ") +           \
                               std::to_string(static_cast<int>(_st)) +    \
                               " at " + __FILE__ + ":" +                \
                               std::to_string(__LINE__));                 \
    }                                                                     \
  } while (0)

struct LtKey {
  int kind;
  int batches;
  int head_dim;
  bool operator==(const LtKey& o) const {
    return kind == o.kind && batches == o.batches && head_dim == o.head_dim;
  }
};

struct LtKeyHash {
  size_t operator()(const LtKey& k) const {
    return (static_cast<size_t>(k.kind) << 40) ^
           (static_cast<size_t>(k.batches) << 20) ^
           static_cast<size_t>(k.head_dim);
  }
};

struct LtPlan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

static cublasLtHandle_t g_lt = nullptr;
static void* g_workspace = nullptr;
static size_t g_workspace_size = 64 * 1024 * 1024;
static std::mutex g_mu;
static std::unordered_map<LtKey, LtPlan, LtKeyHash> g_plans;

static void ensure_lt() {
  if (g_lt) return;
  FLASHRT_CUBLASLT_CHECK(cublasLtCreate(&g_lt));
  cudaError_t err = cudaMalloc(&g_workspace, g_workspace_size);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed for WY workspace: ") +
                             cudaGetErrorString(err));
  }
}

static LtPlan& get_kkt_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{0, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasOperation_t op_t = CUBLAS_OP_T;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t a_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t b_stride = a_stride;
  int64_t c_stride = static_cast<int64_t>(kChunk) * kChunk;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_32F, kChunk, kChunk, kChunk));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY KKT batched BF16 algorithm");
  }
  plan.algo = heuristic.algo;

  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

static LtPlan& get_mm64d_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{1, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t a_stride = static_cast<int64_t>(kChunk) * kChunk;
  int64_t b_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t c_stride = static_cast<int64_t>(kChunk) * head_dim;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_16BF, kChunk, kChunk, kChunk));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY 64xD batched BF16 algorithm");
  }
  plan.algo = heuristic.algo;
  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

static LtPlan& get_qh_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{2, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t a_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t b_stride = static_cast<int64_t>(head_dim) * head_dim;
  int64_t c_stride = static_cast<int64_t>(kChunk) * head_dim;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_16BF, head_dim, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY QH batched BF16 algorithm");
  }
  plan.algo = heuristic.algo;
  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

static LtPlan& get_ktv_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{3, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_t = CUBLAS_OP_T;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t ab_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t c_stride = static_cast<int64_t>(head_dim) * head_dim;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ab_stride,
      sizeof(ab_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_16BF, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ab_stride,
      sizeof(ab_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_16BF, head_dim, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY KtV batched BF16 algorithm");
  }
  plan.algo = heuristic.algo;
  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

static LtPlan& get_f32_wstate_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{4, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t a_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t b_stride = static_cast<int64_t>(head_dim) * head_dim;
  int64_t c_stride = static_cast<int64_t>(kChunk) * head_dim;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_32F, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_32F, head_dim, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_32F, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY f32 W-state algorithm");
  }
  plan.algo = heuristic.algo;
  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

static LtPlan& get_f32_ktv_plan(int batches, int head_dim) {
  std::lock_guard<std::mutex> lock(g_mu);
  ensure_lt();
  LtKey key{5, batches, head_dim};
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;

  LtPlan plan;
  cublasOperation_t op_t = CUBLAS_OP_T;
  cublasOperation_t op_n = CUBLAS_OP_N;
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  int32_t batch_count = batches;
  int64_t ab_stride = static_cast<int64_t>(kChunk) * head_dim;
  int64_t c_stride = static_cast<int64_t>(head_dim) * head_dim;

  FLASHRT_CUBLASLT_CHECK(
      cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.a_desc, CUDA_R_32F, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ab_stride,
      sizeof(ab_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.b_desc, CUDA_R_32F, kChunk, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ab_stride,
      sizeof(ab_stride)));

  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
      &plan.c_desc, CUDA_R_32F, head_dim, head_dim, head_dim));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
      sizeof(row_order)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
      sizeof(batch_count)));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
      plan.c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  cublasLtMatmulPreference_t pref;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_workspace_size,
      sizeof(g_workspace_size)));
  cublasLtMatmulHeuristicResult_t heuristic;
  int returned = 0;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
      pref, 1, &heuristic, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned == 0) {
    throw std::runtime_error("cuBLASLt: no WY f32 KtV algorithm");
  }
  plan.algo = heuristic.algo;
  auto [inserted, _] = g_plans.emplace(key, plan);
  return inserted->second;
}

__global__ void pack_k_chunks_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    __nv_bfloat16* __restrict__ k_pack,
    int S,
    int num_k_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_k_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int kh = (idx / (head_dim * kChunk)) % num_k_heads;
  const int chunk = idx / (head_dim * kChunk * num_k_heads);
  const int s = chunk * kChunk + t;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  k_pack[idx] = (s < S)
      ? k_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d]
      : zero;
}

__global__ void apply_kkt_gating_kernel(
    const float* __restrict__ kkt_base,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    float* __restrict__ A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group)
{
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= kChunk * kChunk) return;
  const int i = pair / kChunk;
  const int j = pair - i * kChunk;
  const int vh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int kh = vh / qk_group;
  const int si = chunk * kChunk + i;
  const int sj = chunk * kChunk + j;
  const size_t a_off =
      (((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + i)
       * kChunk + j);
  if (kh >= num_k_heads || i <= j || si >= S || sj >= S) {
    A[a_off] = 0.0f;
    return;
  }

  const size_t base_off =
      (((static_cast<size_t>(chunk) * num_k_heads + kh) * kChunk + i)
       * kChunk + j);
  const float dot = kkt_base[base_off];
  const float beta_i =
      static_cast<float>(beta[static_cast<size_t>(si) * num_v_heads + vh]);
  const float gi =
      static_cast<float>(g_cumsum[static_cast<size_t>(si) * num_v_heads + vh]);
  const float gj =
      static_cast<float>(g_cumsum[static_cast<size_t>(sj) * num_v_heads + vh]);
  A[a_off] = beta_i * dot * __expf(gi - gj);
}

__global__ void apply_kkt_beta_kernel(
    const float* __restrict__ kkt_base,
    const __nv_bfloat16* __restrict__ beta,
    float* __restrict__ A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group)
{
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= kChunk * kChunk) return;
  const int i = pair / kChunk;
  const int j = pair - i * kChunk;
  const int vh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int kh = vh / qk_group;
  const int si = chunk * kChunk + i;
  const int sj = chunk * kChunk + j;
  const size_t a_off =
      (((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + i)
       * kChunk + j);
  if (kh >= num_k_heads || i <= j || si >= S || sj >= S) {
    A[a_off] = 0.0f;
    return;
  }

  const size_t base_off =
      (((static_cast<size_t>(chunk) * num_k_heads + kh) * kChunk + i)
       * kChunk + j);
  const float dot = kkt_base[base_off];
  const float beta_i =
      static_cast<float>(beta[static_cast<size_t>(si) * num_v_heads + vh]);
  A[a_off] = beta_i * dot;
}

__global__ void pack_recompute_wu_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    const float* __restrict__ Ai,
    __nv_bfloat16* __restrict__ Ai_pack,
    __nv_bfloat16* __restrict__ rhs_w,
    __nv_bfloat16* __restrict__ rhs_u,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int ai_total = chunks * num_v_heads * kChunk * kChunk;
  const int rhs_total = chunks * num_v_heads * kChunk * head_dim;
  const int total = ai_total + rhs_total;
  if (idx >= total) return;

  if (idx < ai_total) {
    const int j = idx % kChunk;
    const int i = (idx / kChunk) % kChunk;
    const int vh = (idx / (kChunk * kChunk)) % num_v_heads;
    const int chunk = idx / (kChunk * kChunk * num_v_heads);
    const size_t src =
        (((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + i)
         * kChunk + j);
    Ai_pack[idx] = __float2bfloat16(Ai[src]);
    return;
  }

  const int ridx = idx - ai_total;
  const int d = ridx % head_dim;
  const int t = (ridx / head_dim) % kChunk;
  const int vh = (ridx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = ridx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (s >= S || kh >= num_k_heads) {
    rhs_w[ridx] = zero;
    rhs_u[ridx] = zero;
    return;
  }
  const float b =
      static_cast<float>(beta[static_cast<size_t>(s) * num_v_heads + vh]);
  const float g =
      static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
  const float vv =
      static_cast<float>(v[(static_cast<size_t>(s) * num_v_heads + vh)
                           * head_dim + d]);
  const float kk =
      static_cast<float>(k_l2[(static_cast<size_t>(s) * num_k_heads + kh)
                              * head_dim + d]);
  rhs_u[ridx] = __float2bfloat16(vv * b);
  rhs_w[ridx] = __float2bfloat16(kk * b * __expf(g));
}

__global__ void pack_recompute_rhs_row_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ rhs_w,
    __nv_bfloat16* __restrict__ rhs_u,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  __shared__ float b_s;
  __shared__ float eg_s;

  const int row = blockIdx.x;
  const int t = row % kChunk;
  const int vh = (row / kChunk) % num_v_heads;
  const int chunk = row / (kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  const size_t out_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * head_dim
      + static_cast<size_t>(t) * head_dim;

  if (threadIdx.x == 0) {
    if (s < S && kh < num_k_heads) {
      b_s = static_cast<float>(
          beta[static_cast<size_t>(s) * num_v_heads + vh]);
      const float g = static_cast<float>(
          g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
      eg_s = __expf(g);
    } else {
      b_s = 0.0f;
      eg_s = 0.0f;
    }
  }
  __syncthreads();

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    if (s >= S || kh >= num_k_heads) {
      rhs_w[out_base + d] = zero;
      rhs_u[out_base + d] = zero;
      continue;
    }
    const float vv =
        static_cast<float>(v[(static_cast<size_t>(s) * num_v_heads + vh)
                             * head_dim + d]);
    const float kk =
        static_cast<float>(k_l2[(static_cast<size_t>(s) * num_k_heads + kh)
                                * head_dim + d]);
    const float b = b_s;
    rhs_u[out_base + d] = __float2bfloat16(vv * b);
    rhs_w[out_base + d] = __float2bfloat16((kk * b) * eg_s);
  }
}

__global__ void pack_recompute_rhs_nogate_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ rhs_w,
    __nv_bfloat16* __restrict__ rhs_u,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (s >= S || kh >= num_k_heads) {
    rhs_w[idx] = zero;
    rhs_u[idx] = zero;
    return;
  }
  const float b =
      static_cast<float>(beta[static_cast<size_t>(s) * num_v_heads + vh]);
  const float g =
      static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
  const float vv =
      static_cast<float>(v[(static_cast<size_t>(s) * num_v_heads + vh)
                           * head_dim + d]);
  const float kk =
      static_cast<float>(k_l2[(static_cast<size_t>(s) * num_k_heads + kh)
                              * head_dim + d]);
  // For Ai_no_gate:
  //   u_gated = exp(g_i) * Ai_no @ (v * beta * exp(-g))
  //   w_gated = exp(g_i) * Ai_no @ (k * beta)
  rhs_u[idx] = __float2bfloat16(vv * b * __expf(-g));
  rhs_w[idx] = __float2bfloat16(kk * b);
}

__global__ void scale_recompute_wu_nogate_kernel(
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ w_pack,
    __nv_bfloat16* __restrict__ u_pack,
    int S,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  if (s >= S) {
    w_pack[idx] = __float2bfloat16(0.0f);
    u_pack[idx] = __float2bfloat16(0.0f);
    return;
  }
  (void)d;
  const float eg = __expf(static_cast<float>(
      g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]));
  w_pack[idx] = __float2bfloat16(static_cast<float>(w_pack[idx]) * eg);
  u_pack[idx] = __float2bfloat16(static_cast<float>(u_pack[idx]) * eg);
}

__global__ void unpack_recompute_wu_kernel(
    const __nv_bfloat16* __restrict__ w_pack,
    const __nv_bfloat16* __restrict__ u_pack,
    __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ u,
    int S,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = S * num_v_heads * head_dim;
  if (idx >= total) return;
  const int d = idx % head_dim;
  const int vh = (idx / head_dim) % num_v_heads;
  const int s = idx / (num_v_heads * head_dim);
  const int chunk = s / kChunk;
  const int t = s - chunk * kChunk;
  const size_t pidx =
      ((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + t)
      * head_dim + d;
  w[idx] = w_pack[pidx];
  u[idx] = u_pack[pidx];
}

__global__ void solve_tril_b16_diag_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    int S,
    int num_v_heads)
{
  const int c = threadIdx.x;
  const int b16 = blockIdx.x;
  const int vh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int chunk_start = chunk * kChunk;
  const int block_start = b16 * 16;
  const int T = min(kChunk, S - chunk_start);
  const int Tb = max(0, min(16, T - block_start));
  const size_t base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * kChunk;

  for (int idx = c; idx < 16 * 16; idx += 16) {
    const int r = idx / 16;
    const int col = idx - r * 16;
    const int gr = block_start + r;
    const int gc = block_start + col;
    Ai[base + gr * kChunk + gc] =
        (r == col && r < Tb) ? 1.0f : 0.0f;
  }
  __syncthreads();

  if (c >= Tb) return;
  for (int r = c + 1; r < Tb; ++r) {
    const int gr = block_start + r;
    const int gc = block_start + c;
    float val = -A[base + gr * kChunk + gc];
    for (int m = c + 1; m < r; ++m) {
      const int gm = block_start + m;
      val -= A[base + gr * kChunk + gm] * Ai[base + gm * kChunk + gc];
    }
    Ai[base + gr * kChunk + gc] = val;
  }
}

__device__ __forceinline__ float b16_mul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    int a_block_r,
    int a_block_c,
    int b_block_r,
    int b_block_c,
    int i,
    int j)
{
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    acc = fmaf(A[(a_block_r * 16 + i) * kChunk + (a_block_c * 16 + k)],
               B[(b_block_r * 16 + k) * kChunk + (b_block_c * 16 + j)],
               acc);
  }
  return acc;
}

__global__ void solve_tril_b64_merge16_shared_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    __nv_bfloat16* __restrict__ Ai_pack,
    int S,
    int num_v_heads)
{
  extern __shared__ float smem[];
  float* A_s = smem;
  float* Ai_s = smem + kChunk * kChunk;

  const int vh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * kChunk;
  const int T = min(kChunk, S - chunk_start);
  const size_t base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * kChunk;
  const float* A_blk = A + base;
  float* Ai_blk = Ai + base;

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    const int r = idx / kChunk;
    const int c = idx - r * kChunk;
    A_s[idx] = A_blk[idx];
    Ai_s[idx] = (r / 16 == c / 16) ? Ai_blk[idx] : 0.0f;
  }
  __syncthreads();

  auto compute_immediate = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 1) * 16 + j;
      if (r < T && c < T) {
        float tmp[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          tmp[k] = b16_mul(Ai_s, A_s, br, br, br, br - 1, i, k);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(tmp[k],
                     Ai_s[((br - 1) * 16 + k) * kChunk + c],
                     acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_immediate(1);
  compute_immediate(2);
  compute_immediate(3);
  __syncthreads();

  auto compute_second = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 2) * 16 + j;
      if (r < T && c < T) {
        float sum[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          sum[k] = b16_mul(A_s, Ai_s, br, br - 2,
                           br - 2, br - 2, k, j);
          sum[k] += b16_mul(A_s, Ai_s, br, br - 1,
                            br - 1, br - 2, k, j);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(Ai_s[(br * 16 + i) * kChunk + (br * 16 + k)],
                     sum[k], acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_second(2);
  compute_second(3);
  __syncthreads();

  for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
    const int i = idx / 16;
    const int j = idx - i * 16;
    const int r = 3 * 16 + i;
    const int c = j;
    if (r < T && c < T) {
      float sum[16];
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        sum[k] = b16_mul(A_s, Ai_s, 3, 0, 0, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 1, 1, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 2, 2, 0, k, j);
      }
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        acc = fmaf(Ai_s[(3 * 16 + i) * kChunk + (3 * 16 + k)],
                   sum[k], acc);
      }
      Ai_s[r * kChunk + c] = -acc;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    const float val = Ai_s[idx];
    if (Ai_blk != nullptr) {
      Ai_blk[idx] = val;
    }
    if (Ai_pack != nullptr) {
      Ai_pack[base + idx] = __float2bfloat16(val);
    }
  }
}

__global__ void solve_tril_b64_fused_shared_kernel(
    const float* __restrict__ A,
    float* __restrict__ Ai,
    __nv_bfloat16* __restrict__ Ai_pack,
    int S,
    int num_v_heads)
{
  extern __shared__ float smem[];
  float* A_s = smem;
  float* Ai_s = smem + kChunk * kChunk;

  const int vh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * kChunk;
  const int T = min(kChunk, S - chunk_start);
  const size_t base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * kChunk;
  const float* A_blk = A + base;
  float* Ai_blk = (Ai != nullptr) ? (Ai + base) : nullptr;

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    const int r = idx / kChunk;
    const int c = idx - r * kChunk;
    A_s[idx] = A_blk[idx];
    Ai_s[idx] = (r == c && r < T) ? 1.0f : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    const int br = threadIdx.x >> 4;
    const int c = threadIdx.x & 15;
    const int block_start = br * 16;
    const int Tb = max(0, min(16, T - block_start));
    if (c < Tb) {
      for (int r = c + 1; r < Tb; ++r) {
        const int gr = block_start + r;
        const int gc = block_start + c;
        float val = -A_s[gr * kChunk + gc];
        for (int m = c + 1; m < r; ++m) {
          const int gm = block_start + m;
          val -= A_s[gr * kChunk + gm] * Ai_s[gm * kChunk + gc];
        }
        Ai_s[gr * kChunk + gc] = val;
      }
    }
  }
  __syncthreads();

  auto compute_immediate = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 1) * 16 + j;
      if (r < T && c < T) {
        float tmp[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          tmp[k] = b16_mul(Ai_s, A_s, br, br, br, br - 1, i, k);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(tmp[k],
                     Ai_s[((br - 1) * 16 + k) * kChunk + c],
                     acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_immediate(1);
  compute_immediate(2);
  compute_immediate(3);
  __syncthreads();

  auto compute_second = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 2) * 16 + j;
      if (r < T && c < T) {
        float sum[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          sum[k] = b16_mul(A_s, Ai_s, br, br - 2,
                           br - 2, br - 2, k, j);
          sum[k] += b16_mul(A_s, Ai_s, br, br - 1,
                            br - 1, br - 2, k, j);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(Ai_s[(br * 16 + i) * kChunk + (br * 16 + k)],
                     sum[k], acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_second(2);
  compute_second(3);
  __syncthreads();

  for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
    const int i = idx / 16;
    const int j = idx - i * 16;
    const int r = 3 * 16 + i;
    const int c = j;
    if (r < T && c < T) {
      float sum[16];
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        sum[k] = b16_mul(A_s, Ai_s, 3, 0, 0, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 1, 1, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 2, 2, 0, k, j);
      }
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        acc = fmaf(Ai_s[(3 * 16 + i) * kChunk + (3 * 16 + k)],
                   sum[k], acc);
      }
      Ai_s[r * kChunk + c] = -acc;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    const float val = Ai_s[idx];
    if (Ai_blk != nullptr) {
      Ai_blk[idx] = val;
    }
    if (Ai_pack != nullptr) {
      Ai_pack[base + idx] = __float2bfloat16(val);
    }
  }
}

__global__ void solve_tril_b64_from_kkt_gated_pack_kernel(
    const float* __restrict__ kkt_base,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ Ai_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group)
{
  extern __shared__ float smem[];
  float* A_s = smem;
  float* Ai_s = smem + kChunk * kChunk;
  float* beta_s = Ai_s + kChunk * kChunk;
  float* g_s = beta_s + kChunk;

  const int vh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kh = vh / qk_group;
  const int chunk_start = chunk * kChunk;
  const int T = min(kChunk, S - chunk_start);
  const size_t out_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * kChunk;
  const size_t kkt_head_base =
      (static_cast<size_t>(chunk) * num_k_heads + kh) * kChunk * kChunk;

  for (int idx = threadIdx.x; idx < kChunk; idx += blockDim.x) {
    const int s = chunk_start + idx;
    if (idx < T && s < S) {
      beta_s[idx] =
          static_cast<float>(beta[static_cast<size_t>(s) * num_v_heads + vh]);
      g_s[idx] = static_cast<float>(
          g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
    } else {
      beta_s[idx] = 0.0f;
      g_s[idx] = 0.0f;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    const int r = idx / kChunk;
    const int c = idx - r * kChunk;
    float a = 0.0f;
    if (kh < num_k_heads && c < r && r < T && c < T) {
      const float dot = kkt_base[kkt_head_base + idx];
      a = beta_s[r] * dot * __expf(g_s[r] - g_s[c]);
    }
    A_s[idx] = a;
    Ai_s[idx] = (r == c && r < T) ? 1.0f : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    const int br = threadIdx.x >> 4;
    const int c = threadIdx.x & 15;
    const int block_start = br * 16;
    const int Tb = max(0, min(16, T - block_start));
    if (c < Tb) {
      for (int r = c + 1; r < Tb; ++r) {
        const int gr = block_start + r;
        const int gc = block_start + c;
        float val = -A_s[gr * kChunk + gc];
        for (int m = c + 1; m < r; ++m) {
          const int gm = block_start + m;
          val -= A_s[gr * kChunk + gm] * Ai_s[gm * kChunk + gc];
        }
        Ai_s[gr * kChunk + gc] = val;
      }
    }
  }
  __syncthreads();

  auto compute_immediate = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 1) * 16 + j;
      if (r < T && c < T) {
        float tmp[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          tmp[k] = b16_mul(Ai_s, A_s, br, br, br, br - 1, i, k);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(tmp[k],
                     Ai_s[((br - 1) * 16 + k) * kChunk + c],
                     acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_immediate(1);
  compute_immediate(2);
  compute_immediate(3);
  __syncthreads();

  auto compute_second = [&](int br) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
      const int i = idx / 16;
      const int j = idx - i * 16;
      const int r = br * 16 + i;
      const int c = (br - 2) * 16 + j;
      if (r < T && c < T) {
        float sum[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          sum[k] = b16_mul(A_s, Ai_s, br, br - 2,
                           br - 2, br - 2, k, j);
          sum[k] += b16_mul(A_s, Ai_s, br, br - 1,
                            br - 1, br - 2, k, j);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
          acc = fmaf(Ai_s[(br * 16 + i) * kChunk + (br * 16 + k)],
                     sum[k], acc);
        }
        Ai_s[r * kChunk + c] = -acc;
      }
    }
  };

  compute_second(2);
  compute_second(3);
  __syncthreads();

  for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
    const int i = idx / 16;
    const int j = idx - i * 16;
    const int r = 3 * 16 + i;
    const int c = j;
    if (r < T && c < T) {
      float sum[16];
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        sum[k] = b16_mul(A_s, Ai_s, 3, 0, 0, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 1, 1, 0, k, j);
        sum[k] += b16_mul(A_s, Ai_s, 3, 2, 2, 0, k, j);
      }
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 16; ++k) {
        acc = fmaf(Ai_s[(3 * 16 + i) * kChunk + (3 * 16 + k)],
                   sum[k], acc);
      }
      Ai_s[r * kChunk + c] = -acc;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kChunk * kChunk; idx += blockDim.x) {
    Ai_pack[out_base + idx] = __float2bfloat16(Ai_s[idx]);
  }
}

__global__ void pack_output_qkv_kernel(
    const __nv_bfloat16* __restrict__ q_l2,
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ v_new,
    __nv_bfloat16* __restrict__ q_pack,
    __nv_bfloat16* __restrict__ k_pack_hv,
    __nv_bfloat16* __restrict__ v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (s >= S || kh >= num_k_heads) {
    q_pack[idx] = zero;
    k_pack_hv[idx] = zero;
    v_pack[idx] = zero;
    return;
  }

  q_pack[idx] =
      q_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
  k_pack_hv[idx] =
      k_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
  v_pack[idx] =
      v_new[(static_cast<size_t>(s) * num_v_heads + vh) * head_dim + d];
}

__global__ void pack_output_qv_kernel(
    const __nv_bfloat16* __restrict__ q_l2,
    const __nv_bfloat16* __restrict__ v_new,
    __nv_bfloat16* __restrict__ q_pack,
    __nv_bfloat16* __restrict__ v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (s >= S || kh >= num_k_heads) {
    q_pack[idx] = zero;
    v_pack[idx] = zero;
    return;
  }

  q_pack[idx] =
      q_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
  v_pack[idx] =
      v_new[(static_cast<size_t>(s) * num_v_heads + vh) * head_dim + d];
}

__global__ void pack_output_q_kernel(
    const __nv_bfloat16* __restrict__ q_l2,
    __nv_bfloat16* __restrict__ q_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  if (s >= S || kh >= num_k_heads) {
    q_pack[idx] = __float2bfloat16(0.0f);
    return;
  }

  q_pack[idx] =
      q_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
}

__global__ void pack_chunk_h_inputs_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ k_pack_hv,
    __nv_bfloat16* __restrict__ w_pack,
    __nv_bfloat16* __restrict__ u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  if (s >= S || kh >= num_k_heads) {
    k_pack_hv[idx] = zero;
    w_pack[idx] = zero;
    u_pack[idx] = zero;
    return;
  }
  k_pack_hv[idx] =
      k_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
  const size_t uv_off =
      (static_cast<size_t>(s) * num_v_heads + vh) * head_dim + d;
  w_pack[idx] = w[uv_off];
  u_pack[idx] = u[uv_off];
}

__global__ void pack_k_hv_kernel(
    const __nv_bfloat16* __restrict__ k_l2,
    __nv_bfloat16* __restrict__ k_pack_hv,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int chunks = (S + kChunk - 1) / kChunk;
  const int total = chunks * num_v_heads * kChunk * head_dim;
  if (idx >= total) return;

  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = (idx / (head_dim * kChunk)) % num_v_heads;
  const int chunk = idx / (head_dim * kChunk * num_v_heads);
  const int s = chunk * kChunk + t;
  const int kh = vh / qk_group;
  if (s >= S || kh >= num_k_heads) {
    k_pack_hv[idx] = __float2bfloat16(0.0f);
    return;
  }
  k_pack_hv[idx] =
      k_l2[(static_cast<size_t>(s) * num_k_heads + kh) * head_dim + d];
}

__global__ void make_vnew_decay_and_scale_state_kernel(
    const __nv_bfloat16* __restrict__ u_pack,
    const __nv_bfloat16* __restrict__ wh_pack,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ h0_chunk,
    __nv_bfloat16* __restrict__ v_new,
    __nv_bfloat16* __restrict__ v_pack_out,
    __nv_bfloat16* __restrict__ decayed_v_pack,
    int S,
    int chunk,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pack_total = num_v_heads * kChunk * head_dim;
  const int state_total = num_v_heads * head_dim * head_dim;
  const int total = pack_total + state_total;
  if (idx >= total) return;
  const int start = chunk * kChunk;

  if (idx < pack_total) {
    const int d = idx % head_dim;
    const int t = (idx / head_dim) % kChunk;
    const int vh = idx / (head_dim * kChunk);
    const int s = start + t;
    const size_t pidx =
        ((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + t)
        * head_dim + d;
    const int T = min(kChunk, S - start);
    if (t >= T || s >= S) {
      if (v_pack_out != nullptr) {
        v_pack_out[pidx] = __float2bfloat16(0.0f);
      }
      decayed_v_pack[pidx] = __float2bfloat16(0.0f);
      return;
    }
    const float val =
        static_cast<float>(u_pack[pidx]) - static_cast<float>(wh_pack[pidx]);
    const size_t out_off =
        (static_cast<size_t>(s) * num_v_heads + vh) * head_dim + d;
    const __nv_bfloat16 val_bf16 = __float2bfloat16(val);
    if (v_new != nullptr) {
      v_new[out_off] = val_bf16;
    }
    if (v_pack_out != nullptr) {
      v_pack_out[pidx] = val_bf16;
    }
    const float g_last =
        static_cast<float>(g_cumsum[
            static_cast<size_t>(start + T - 1) * num_v_heads + vh]);
    const float gt =
        static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
    decayed_v_pack[pidx] = __float2bfloat16(val * __expf(g_last - gt));
    return;
  }

  const int sidx = idx - pack_total;
  const int col = sidx % head_dim;
  const int row = (sidx / head_dim) % head_dim;
  const int vh = sidx / (head_dim * head_dim);
  const int T = min(kChunk, S - start);
  if (T <= 0) return;
  const float g_last =
      static_cast<float>(g_cumsum[
          static_cast<size_t>(start + T - 1) * num_v_heads + vh]);
  const size_t off =
      (static_cast<size_t>(vh) * head_dim + row) * head_dim + col;
  const __nv_bfloat16 old_state = state[off];
  if (h0_chunk != nullptr) {
    h0_chunk[off] = old_state;
  }
  state[off] = __float2bfloat16(static_cast<float>(old_state) *
                                __expf(g_last));
}

__global__ void init_state_f32_kernel(
    const __nv_bfloat16* __restrict__ state,
    float* __restrict__ state_f32,
    int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  state_f32[idx] = static_cast<float>(state[idx]);
}

__global__ void downcast_state_f32_kernel(
    const float* __restrict__ state_f32,
    __nv_bfloat16* __restrict__ dst,
    int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  dst[idx] = __float2bfloat16(state_f32[idx]);
}

__global__ void compute_wh_f32_state_kernel(
    const __nv_bfloat16* __restrict__ w_pack,
    const float* __restrict__ state_f32,
    float* __restrict__ wh_f32,
    int S,
    int chunk,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_v_heads * kChunk * head_dim;
  if (idx >= total) return;
  const int d = idx % head_dim;
  const int t = (idx / head_dim) % kChunk;
  const int vh = idx / (head_dim * kChunk);
  const int start = chunk * kChunk;
  const int s = start + t;
  if (s >= S) {
    wh_f32[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  const size_t state_base = static_cast<size_t>(vh) * head_dim * head_dim;
  const size_t w_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * head_dim
      + static_cast<size_t>(t) * head_dim;
  for (int r = 0; r < head_dim; ++r) {
    acc = fmaf(static_cast<float>(w_pack[w_base + r]),
               state_f32[state_base + static_cast<size_t>(r) * head_dim + d],
               acc);
  }
  wh_f32[idx] = acc;
}

__global__ void make_vnew_decay_and_scale_state_f32_wh_kernel(
    const __nv_bfloat16* __restrict__ u_pack,
    float* __restrict__ wh_decayed_f32,
    const __nv_bfloat16* __restrict__ g_cumsum,
    float* __restrict__ state_f32,
    __nv_bfloat16* __restrict__ v_new,
    __nv_bfloat16* __restrict__ decayed_v_pack,
    int S,
    int chunk,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int pack_total = num_v_heads * kChunk * head_dim;
  const int state_total = num_v_heads * head_dim * head_dim;
  const int total = pack_total + state_total;
  if (idx >= total) return;
  const int start = chunk * kChunk;

  if (idx < pack_total) {
    const int d = idx % head_dim;
    const int t = (idx / head_dim) % kChunk;
    const int vh = idx / (head_dim * kChunk);
    const int s = start + t;
    const size_t pidx =
        ((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + t)
        * head_dim + d;
    const int T = min(kChunk, S - start);
    if (t >= T || s >= S) {
      decayed_v_pack[pidx] = __float2bfloat16(0.0f);
      return;
    }
    const float val = static_cast<float>(u_pack[pidx]) - wh_decayed_f32[idx];
    const size_t out_off =
        (static_cast<size_t>(s) * num_v_heads + vh) * head_dim + d;
    v_new[out_off] = __float2bfloat16(val);
    const float g_last =
        static_cast<float>(g_cumsum[
            static_cast<size_t>(start + T - 1) * num_v_heads + vh]);
    const float gt =
        static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
    const float decayed = val * __expf(g_last - gt);
    wh_decayed_f32[idx] = decayed;
    decayed_v_pack[pidx] = __float2bfloat16(decayed);
    return;
  }

  const int sidx = idx - pack_total;
  const int col = sidx % head_dim;
  const int row = (sidx / head_dim) % head_dim;
  const int vh = sidx / (head_dim * head_dim);
  const int T = min(kChunk, S - start);
  if (T <= 0) return;
  const float g_last =
      static_cast<float>(g_cumsum[
          static_cast<size_t>(start + T - 1) * num_v_heads + vh]);
  const size_t off =
      (static_cast<size_t>(vh) * head_dim + row) * head_dim + col;
  state_f32[off] *= __expf(g_last);
}

__global__ void update_state_f32_direct_kernel(
    const __nv_bfloat16* __restrict__ k_pack_hv,
    const float* __restrict__ decayed_f32,
    float* __restrict__ state_f32,
    int S,
    int chunk,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_v_heads * head_dim * head_dim;
  if (idx >= total) return;
  const int col = idx % head_dim;
  const int row = (idx / head_dim) % head_dim;
  const int vh = idx / (head_dim * head_dim);
  const int start = chunk * kChunk;
  const int T = min(kChunk, S - start);
  if (T <= 0) return;

  float acc = state_f32[idx];
  const size_t k_base =
      (static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk * head_dim;
  const size_t v_base = static_cast<size_t>(vh) * kChunk * head_dim;
  for (int t = 0; t < T; ++t) {
    acc = fmaf(static_cast<float>(k_pack_hv[k_base + t * head_dim + row]),
               decayed_f32[v_base + t * head_dim + col],
               acc);
  }
  state_f32[idx] = acc;
}

__global__ void cast_pack_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  dst[idx] = static_cast<float>(src[idx]);
}

__global__ void apply_output_local_a_kernel(
    const float* __restrict__ qk_base,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ local_a_pack,
    int S,
    int num_v_heads)
{
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= kChunk * kChunk) return;
  const int i = pair / kChunk;
  const int j = pair - i * kChunk;
  const int vh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int si = chunk * kChunk + i;
  const int sj = chunk * kChunk + j;
  const size_t off =
      (((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + i)
       * kChunk + j);
  if (j > i || si >= S || sj >= S) {
    local_a_pack[off] = __float2bfloat16(0.0f);
    return;
  }
  const float gi =
      static_cast<float>(g_cumsum[static_cast<size_t>(si) * num_v_heads + vh]);
  const float gj =
      static_cast<float>(g_cumsum[static_cast<size_t>(sj) * num_v_heads + vh]);
  local_a_pack[off] = __float2bfloat16(qk_base[off] * __expf(gi - gj));
}

__global__ void combine_output_kernel(
    const __nv_bfloat16* __restrict__ qh_pack,
    const __nv_bfloat16* __restrict__ local_pack,
    const __nv_bfloat16* __restrict__ g_cumsum,
    __nv_bfloat16* __restrict__ out,
    int S,
    int num_v_heads,
    int head_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = S * num_v_heads * head_dim;
  if (idx >= total) return;
  const int d = idx % head_dim;
  const int vh = (idx / head_dim) % num_v_heads;
  const int s = idx / (num_v_heads * head_dim);
  const int chunk = s / kChunk;
  const int t = s - chunk * kChunk;
  const size_t pidx =
      ((static_cast<size_t>(chunk) * num_v_heads + vh) * kChunk + t)
      * head_dim + d;
  const float gi =
      static_cast<float>(g_cumsum[static_cast<size_t>(s) * num_v_heads + vh]);
  const float qh = static_cast<float>(qh_pack[pidx]) * __expf(gi);
  const float local = static_cast<float>(local_pack[pidx]);
  constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128)
  out[idx] = __float2bfloat16((qh + local) * kScale);
}

}  // namespace

void gdn_wy_kkt_b64_bf16_cublaslt(
    const void* k_l2,
    const void* beta,
    const void* g_cumsum,
    void*       k_pack,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_k_heads;
  const int pack_total = batches * kChunk * head_dim;
  pack_k_chunks_kernel<<<(pack_total + kThreads - 1) / kThreads,
                         kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<__nv_bfloat16*>(k_pack),
      S, num_k_heads, head_dim);

  LtPlan& plan = get_kkt_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      k_pack, plan.a_desc,
      k_pack, plan.b_desc,
      &beta0,
      kkt_base, plan.c_desc,
      kkt_base, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));

  apply_kkt_gating_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(kkt_base),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<float*>(A),
      S, num_k_heads, num_v_heads, qk_group);
}

void gdn_wy_kkt_b64_bf16_cublaslt_packed_k(
    const void* k_pack,
    const void* beta,
    const void* g_cumsum,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_k_heads;

  LtPlan& plan = get_kkt_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      k_pack, plan.a_desc,
      k_pack, plan.b_desc,
      &beta0,
      kkt_base, plan.c_desc,
      kkt_base, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));

  apply_kkt_gating_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(kkt_base),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<float*>(A),
      S, num_k_heads, num_v_heads, qk_group);
}

void gdn_wy_kkt_b64_bf16_cublaslt_packed_k_only(
    const void* k_pack,
    void*       kkt_base,
    int S,
    int num_k_heads,
    int head_dim,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_k_heads;

  LtPlan& plan = get_kkt_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      k_pack, plan.a_desc,
      k_pack, plan.b_desc,
      &beta0,
      kkt_base, plan.c_desc,
      kkt_base, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
}

void gdn_wy_kkt_b64_bf16_cublaslt_nogate(
    const void* k_l2,
    const void* beta,
    void*       k_pack,
    void*       kkt_base,
    void*       A,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_k_heads;
  const int pack_total = batches * kChunk * head_dim;
  pack_k_chunks_kernel<<<(pack_total + kThreads - 1) / kThreads,
                         kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<__nv_bfloat16*>(k_pack),
      S, num_k_heads, head_dim);

  LtPlan& plan = get_kkt_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      k_pack, plan.a_desc,
      k_pack, plan.b_desc,
      &beta0,
      kkt_base, plan.c_desc,
      kkt_base, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));

  apply_kkt_beta_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(kkt_base),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<float*>(A),
      S, num_k_heads, num_v_heads, qk_group);
}

void gdn_wy_recompute_wu_b64_bf16_cublaslt(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    void*       w,
    void*       u,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int ai_total = batches * kChunk * kChunk;
  const int rhs_total = batches * kChunk * head_dim;
  pack_recompute_wu_kernel<<<
      (ai_total + rhs_total + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      reinterpret_cast<__nv_bfloat16*>(rhs_w),
      reinterpret_cast<__nv_bfloat16*>(rhs_u),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& plan = get_mm64d_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_u, plan.b_desc,
      &beta0,
      u_pack, plan.c_desc,
      u_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_w, plan.b_desc,
      &beta0,
      w_pack, plan.c_desc,
      w_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
  unpack_recompute_wu_kernel<<<
      (S * num_v_heads * head_dim + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(w_pack),
      reinterpret_cast<const __nv_bfloat16*>(u_pack),
      reinterpret_cast<__nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(u),
      S, num_v_heads, head_dim);
}

void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai,
    void*       Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int ai_total = batches * kChunk * kChunk;
  const int rhs_total = batches * kChunk * head_dim;
  pack_recompute_wu_kernel<<<
      (ai_total + rhs_total + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<const float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      reinterpret_cast<__nv_bfloat16*>(rhs_w),
      reinterpret_cast<__nv_bfloat16*>(rhs_u),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& plan = get_mm64d_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_u, plan.b_desc,
      &beta0,
      u_pack, plan.c_desc,
      u_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_w, plan.b_desc,
      &beta0,
      w_pack, plan.c_desc,
      w_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
}

void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  pack_recompute_rhs_row_kernel<<<
      batches * kChunk, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(rhs_w),
      reinterpret_cast<__nv_bfloat16*>(rhs_u),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& plan = get_mm64d_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_u, plan.b_desc,
      &beta0,
      u_pack, plan.c_desc,
      u_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_w, plan.b_desc,
      &beta0,
      w_pack, plan.c_desc,
      w_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
}

void gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs_nogate(
    const void* k_l2,
    const void* v,
    const void* beta,
    const void* g_cumsum,
    const void* Ai_pack,
    void*       rhs_w,
    void*       rhs_u,
    void*       w_pack,
    void*       u_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int rhs_total = batches * kChunk * head_dim;
  pack_recompute_rhs_nogate_kernel<<<
      (rhs_total + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(rhs_w),
      reinterpret_cast<__nv_bfloat16*>(rhs_u),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& plan = get_mm64d_plan(batches, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_u, plan.b_desc,
      &beta0,
      u_pack, plan.c_desc,
      u_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, plan.desc, &alpha,
      Ai_pack, plan.a_desc,
      rhs_w, plan.b_desc,
      &beta0,
      w_pack, plan.c_desc,
      w_pack, plan.c_desc,
      &plan.algo, g_workspace, g_workspace_size, stream));

  scale_recompute_wu_nogate_kernel<<<
      (rhs_total + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(w_pack),
      reinterpret_cast<__nv_bfloat16*>(u_pack),
      S, num_v_heads, head_dim);
}

void gdn_wy_solve_tril_b64_f32_parallel(
    const void* A,
    void*       Ai,
    int S,
    int num_v_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  const int chunks = (S + kChunk - 1) / kChunk;
  solve_tril_b16_diag_kernel<<<dim3(4, num_v_heads, chunks), 16,
                                0, stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      S, num_v_heads);
  solve_tril_b64_merge16_shared_kernel<<<dim3(num_v_heads, chunks), 256,
                                          2 * kChunk * kChunk * sizeof(float),
                                          stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      nullptr,
      S, num_v_heads);
}

void gdn_wy_solve_tril_b64_f32_parallel_pack(
    const void* A,
    void*       Ai,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  const int chunks = (S + kChunk - 1) / kChunk;
  solve_tril_b16_diag_kernel<<<dim3(4, num_v_heads, chunks), 16,
                                0, stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      S, num_v_heads);
  solve_tril_b64_merge16_shared_kernel<<<dim3(num_v_heads, chunks), 256,
                                          2 * kChunk * kChunk * sizeof(float),
                                          stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      S, num_v_heads);
}

void gdn_wy_solve_tril_b64_f32_fused_pack(
    const void* A,
    void*       Ai,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  const int chunks = (S + kChunk - 1) / kChunk;
  solve_tril_b64_fused_shared_kernel<<<dim3(num_v_heads, chunks), 256,
                                       2 * kChunk * kChunk * sizeof(float),
                                       stream>>>(
      reinterpret_cast<const float*>(A),
      reinterpret_cast<float*>(Ai),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      S, num_v_heads);
}

void gdn_wy_solve_tril_b64_f32_fused_pack_only(
    const void* A,
    void*       Ai_pack,
    int S,
    int num_v_heads,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  const int chunks = (S + kChunk - 1) / kChunk;
  solve_tril_b64_fused_shared_kernel<<<dim3(num_v_heads, chunks), 256,
                                       2 * kChunk * kChunk * sizeof(float),
                                       stream>>>(
      reinterpret_cast<const float*>(A),
      nullptr,
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      S, num_v_heads);
}

void gdn_wy_solve_tril_b64_from_kkt_pack_only(
    const void* kkt_base,
    const void* beta,
    const void* g_cumsum,
    void*       Ai_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  solve_tril_b64_from_kkt_gated_pack_kernel<<<
      dim3(num_v_heads, chunks), 256,
      (2 * kChunk * kChunk + 2 * kChunk) * sizeof(float), stream>>>(
      reinterpret_cast<const float*>(kkt_base),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(Ai_pack),
      S, num_k_heads, num_v_heads, qk_group);
}

void gdn_wy_chunk_h_b64_bf16_cublaslt(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int all_batches = chunks * num_v_heads;
  const int pack_total = all_batches * kChunk * head_dim;
  pack_chunk_h_inputs_kernel<<<(pack_total + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(u),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(w_pack),
      reinterpret_cast<__nv_bfloat16*>(u_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& wh_plan = get_qh_plan(num_v_heads, head_dim);
  LtPlan& ktv_plan = get_ktv_plan(num_v_heads, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;
  const size_t pack_chunk_elems =
      static_cast<size_t>(num_v_heads) * kChunk * head_dim;
  const size_t state_elems =
      static_cast<size_t>(num_v_heads) * head_dim * head_dim;

  auto* state_bf16 = reinterpret_cast<__nv_bfloat16*>(state);
  auto* h0_bf16 = reinterpret_cast<__nv_bfloat16*>(h0);
  auto* k_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(k_pack_hv);
  auto* w_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(w_pack);
  auto* u_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(u_pack);
  auto* wh_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(wh_pack);
  auto* decayed_bf16 = reinterpret_cast<__nv_bfloat16*>(decayed_v_pack);

  for (int ci = 0; ci < chunks; ++ci) {
    const size_t off = static_cast<size_t>(ci) * pack_chunk_elems;
    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, wh_plan.desc, &alpha,
        w_pack_bf16 + off, wh_plan.a_desc,
        state_bf16, wh_plan.b_desc,
        &beta0,
        wh_pack_bf16 + off, wh_plan.c_desc,
        wh_pack_bf16 + off, wh_plan.c_desc,
        &wh_plan.algo, g_workspace, g_workspace_size, stream));

    const int work = static_cast<int>(pack_chunk_elems + state_elems);
    make_vnew_decay_and_scale_state_kernel<<<
        (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        u_pack_bf16, wh_pack_bf16,
        reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
        state_bf16, h0_bf16 + static_cast<size_t>(ci) * state_elems,
        reinterpret_cast<__nv_bfloat16*>(v_new), nullptr, decayed_bf16,
        S, ci, num_v_heads, head_dim);

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, ktv_plan.desc, &alpha,
        k_pack_bf16 + off, ktv_plan.a_desc,
        decayed_bf16 + off, ktv_plan.b_desc,
        &beta1,
        state_bf16, ktv_plan.c_desc,
        state_bf16, ktv_plan.c_desc,
        &ktv_plan.algo, g_workspace, g_workspace_size, stream));
  }
}

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32state(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       delta_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int all_batches = chunks * num_v_heads;
  const int pack_total = all_batches * kChunk * head_dim;
  pack_chunk_h_inputs_kernel<<<(pack_total + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(u),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(w_pack),
      reinterpret_cast<__nv_bfloat16*>(u_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  const size_t pack_chunk_elems =
      static_cast<size_t>(num_v_heads) * kChunk * head_dim;
  const int state_elems = num_v_heads * head_dim * head_dim;

  auto* state_bf16 = reinterpret_cast<__nv_bfloat16*>(state);
  auto* h0_bf16 = reinterpret_cast<__nv_bfloat16*>(h0);
  auto* k_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(k_pack_hv);
  auto* w_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(w_pack);
  auto* u_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(u_pack);
  auto* decayed_bf16 = reinterpret_cast<__nv_bfloat16*>(decayed_v_pack);
  auto* state_fp32 = reinterpret_cast<float*>(state_f32);
  auto* delta_fp32 = reinterpret_cast<float*>(delta_f32);

  init_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                           kThreads, 0, stream>>>(
      state_bf16, state_fp32, state_elems);

  for (int ci = 0; ci < chunks; ++ci) {
    __nv_bfloat16* h0_chunk =
        h0_bf16 + static_cast<size_t>(ci) * state_elems;
    downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                                 kThreads, 0, stream>>>(
        state_fp32, h0_chunk, state_elems);

    compute_wh_f32_state_kernel<<<
        (static_cast<int>(pack_chunk_elems) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        w_pack_bf16, state_fp32, delta_fp32,
        S, ci, num_v_heads, head_dim);

    const int work = static_cast<int>(pack_chunk_elems) + state_elems;
    make_vnew_decay_and_scale_state_f32_wh_kernel<<<
        (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        u_pack_bf16, delta_fp32,
        reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
        state_fp32, reinterpret_cast<__nv_bfloat16*>(v_new),
        decayed_bf16, S, ci, num_v_heads, head_dim);

    update_state_f32_direct_kernel<<<(state_elems + kThreads - 1) / kThreads,
                                      kThreads, 0, stream>>>(
        k_pack_bf16, delta_fp32, state_fp32,
        S, ci, num_v_heads, head_dim);
  }

  downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      state_fp32, state_bf16, state_elems);
}

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm(
    const void* k_l2,
    const void* u,
    const void* w,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       w_pack,
    void*       u_pack,
    void*       wh_pack,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       chunk_f32,
    void*       acc_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int all_batches = chunks * num_v_heads;
  const int pack_total = all_batches * kChunk * head_dim;
  pack_chunk_h_inputs_kernel<<<(pack_total + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(u),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(w_pack),
      reinterpret_cast<__nv_bfloat16*>(u_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& wh_plan = get_f32_wstate_plan(num_v_heads, head_dim);
  LtPlan& ktv_plan = get_f32_ktv_plan(num_v_heads, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;
  const size_t pack_chunk_elems =
      static_cast<size_t>(num_v_heads) * kChunk * head_dim;
  const int state_elems = num_v_heads * head_dim * head_dim;

  auto* state_bf16 = reinterpret_cast<__nv_bfloat16*>(state);
  auto* h0_bf16 = reinterpret_cast<__nv_bfloat16*>(h0);
  auto* k_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(k_pack_hv);
  auto* w_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(w_pack);
  auto* u_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(u_pack);
  auto* decayed_bf16 = reinterpret_cast<__nv_bfloat16*>(decayed_v_pack);
  auto* state_fp32 = reinterpret_cast<float*>(state_f32);
  auto* chunk_fp32 = reinterpret_cast<float*>(chunk_f32);
  auto* acc_fp32 = reinterpret_cast<float*>(acc_f32);

  init_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                           kThreads, 0, stream>>>(
      state_bf16, state_fp32, state_elems);

  for (int ci = 0; ci < chunks; ++ci) {
    __nv_bfloat16* h0_chunk =
        h0_bf16 + static_cast<size_t>(ci) * state_elems;
    downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                                 kThreads, 0, stream>>>(
        state_fp32, h0_chunk, state_elems);

    const size_t off = static_cast<size_t>(ci) * pack_chunk_elems;
    cast_pack_bf16_to_f32_kernel<<<
        (static_cast<int>(pack_chunk_elems) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        w_pack_bf16 + off, chunk_fp32, static_cast<int>(pack_chunk_elems));

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, wh_plan.desc, &alpha,
        chunk_fp32, wh_plan.a_desc,
        state_fp32, wh_plan.b_desc,
        &beta0,
        acc_fp32, wh_plan.c_desc,
        acc_fp32, wh_plan.c_desc,
        &wh_plan.algo, g_workspace, g_workspace_size, stream));

    const int work = static_cast<int>(pack_chunk_elems) + state_elems;
    make_vnew_decay_and_scale_state_f32_wh_kernel<<<
        (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        u_pack_bf16, acc_fp32,
        reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
        state_fp32, reinterpret_cast<__nv_bfloat16*>(v_new),
        decayed_bf16, S, ci, num_v_heads, head_dim);

    cast_pack_bf16_to_f32_kernel<<<
        (static_cast<int>(pack_chunk_elems) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        k_pack_bf16 + off, chunk_fp32, static_cast<int>(pack_chunk_elems));

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, ktv_plan.desc, &alpha,
        chunk_fp32, ktv_plan.a_desc,
        acc_fp32, ktv_plan.b_desc,
        &beta1,
        state_fp32, ktv_plan.c_desc,
        state_fp32, ktv_plan.c_desc,
        &ktv_plan.algo, g_workspace, g_workspace_size, stream));
  }

  downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      state_fp32, state_bf16, state_elems);
}

void gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm_packed_wu(
    const void* k_l2,
    const void* w_pack,
    const void* u_pack,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       decayed_v_pack,
    void*       state_f32,
    void*       chunk_f32,
    void*       acc_f32,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int all_batches = chunks * num_v_heads;
  const int pack_total = all_batches * kChunk * head_dim;
  pack_k_hv_kernel<<<(pack_total + kThreads - 1) / kThreads,
                      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& wh_plan = get_f32_wstate_plan(num_v_heads, head_dim);
  LtPlan& ktv_plan = get_f32_ktv_plan(num_v_heads, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;
  const size_t pack_chunk_elems =
      static_cast<size_t>(num_v_heads) * kChunk * head_dim;
  const int state_elems = num_v_heads * head_dim * head_dim;

  auto* state_bf16 = reinterpret_cast<__nv_bfloat16*>(state);
  auto* h0_bf16 = reinterpret_cast<__nv_bfloat16*>(h0);
  auto* k_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(k_pack_hv);
  auto* w_pack_bf16 = reinterpret_cast<const __nv_bfloat16*>(w_pack);
  auto* u_pack_bf16 = reinterpret_cast<const __nv_bfloat16*>(u_pack);
  auto* decayed_bf16 = reinterpret_cast<__nv_bfloat16*>(decayed_v_pack);
  auto* state_fp32 = reinterpret_cast<float*>(state_f32);
  auto* chunk_fp32 = reinterpret_cast<float*>(chunk_f32);
  auto* acc_fp32 = reinterpret_cast<float*>(acc_f32);

  init_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                           kThreads, 0, stream>>>(
      state_bf16, state_fp32, state_elems);

  for (int ci = 0; ci < chunks; ++ci) {
    __nv_bfloat16* h0_chunk =
        h0_bf16 + static_cast<size_t>(ci) * state_elems;
    downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                                 kThreads, 0, stream>>>(
        state_fp32, h0_chunk, state_elems);

    const size_t off = static_cast<size_t>(ci) * pack_chunk_elems;
    cast_pack_bf16_to_f32_kernel<<<
        (static_cast<int>(pack_chunk_elems) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        w_pack_bf16 + off, chunk_fp32, static_cast<int>(pack_chunk_elems));

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, wh_plan.desc, &alpha,
        chunk_fp32, wh_plan.a_desc,
        state_fp32, wh_plan.b_desc,
        &beta0,
        acc_fp32, wh_plan.c_desc,
        acc_fp32, wh_plan.c_desc,
        &wh_plan.algo, g_workspace, g_workspace_size, stream));

    const int work = static_cast<int>(pack_chunk_elems) + state_elems;
    make_vnew_decay_and_scale_state_f32_wh_kernel<<<
        (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        u_pack_bf16, acc_fp32,
        reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
        state_fp32, reinterpret_cast<__nv_bfloat16*>(v_new),
        decayed_bf16, S, ci, num_v_heads, head_dim);

    cast_pack_bf16_to_f32_kernel<<<
        (static_cast<int>(pack_chunk_elems) + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        k_pack_bf16 + off, chunk_fp32, static_cast<int>(pack_chunk_elems));

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, ktv_plan.desc, &alpha,
        chunk_fp32, ktv_plan.a_desc,
        acc_fp32, ktv_plan.b_desc,
        &beta1,
        state_fp32, ktv_plan.c_desc,
        state_fp32, ktv_plan.c_desc,
        &ktv_plan.algo, g_workspace, g_workspace_size, stream));
  }

  downcast_state_f32_kernel<<<(state_elems + kThreads - 1) / kThreads,
                               kThreads, 0, stream>>>(
      state_fp32, state_bf16, state_elems);
}

void gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
    const void* k_l2,
    const void* w_pack,
    const void* u_pack,
    const void* g_cumsum,
    void*       state,
    void*       h0,
    void*       v_new,
    void*       k_pack_hv,
    void*       wh_pack,
    void*       decayed_v_pack,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int all_batches = chunks * num_v_heads;
  const int pack_total = all_batches * kChunk * head_dim;
  pack_k_hv_kernel<<<(pack_total + kThreads - 1) / kThreads,
                      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  LtPlan& wh_plan = get_qh_plan(num_v_heads, head_dim);
  LtPlan& ktv_plan = get_ktv_plan(num_v_heads, head_dim);
  const float alpha = 1.0f;
  const float beta0 = 0.0f;
  const float beta1 = 1.0f;
  const size_t pack_chunk_elems =
      static_cast<size_t>(num_v_heads) * kChunk * head_dim;
  const size_t state_elems =
      static_cast<size_t>(num_v_heads) * head_dim * head_dim;

  auto* state_bf16 = reinterpret_cast<__nv_bfloat16*>(state);
  auto* h0_bf16 = reinterpret_cast<__nv_bfloat16*>(h0);
  auto* k_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(k_pack_hv);
  auto* w_pack_bf16 = reinterpret_cast<const __nv_bfloat16*>(w_pack);
  auto* u_pack_bf16 = reinterpret_cast<const __nv_bfloat16*>(u_pack);
  auto* wh_pack_bf16 = reinterpret_cast<__nv_bfloat16*>(wh_pack);
  auto* decayed_bf16 = reinterpret_cast<__nv_bfloat16*>(decayed_v_pack);

  for (int ci = 0; ci < chunks; ++ci) {
    const size_t off = static_cast<size_t>(ci) * pack_chunk_elems;
    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, wh_plan.desc, &alpha,
        w_pack_bf16 + off, wh_plan.a_desc,
        state_bf16, wh_plan.b_desc,
        &beta0,
        wh_pack_bf16 + off, wh_plan.c_desc,
        wh_pack_bf16 + off, wh_plan.c_desc,
        &wh_plan.algo, g_workspace, g_workspace_size, stream));

    const int work = static_cast<int>(pack_chunk_elems + state_elems);
    make_vnew_decay_and_scale_state_kernel<<<
        (work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        u_pack_bf16, wh_pack_bf16,
        reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
        state_bf16, h0_bf16 + static_cast<size_t>(ci) * state_elems,
        reinterpret_cast<__nv_bfloat16*>(v_new), wh_pack_bf16, decayed_bf16,
        S, ci, num_v_heads, head_dim);

    FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
        g_lt, ktv_plan.desc, &alpha,
        k_pack_bf16 + off, ktv_plan.a_desc,
        decayed_bf16 + off, ktv_plan.b_desc,
        &beta1,
        state_bf16, ktv_plan.c_desc,
        state_bf16, ktv_plan.c_desc,
        &ktv_plan.algo, g_workspace, g_workspace_size, stream));
  }
}

void gdn_wy_output_o_b64_bf16_cublaslt(
    const void* q_l2,
    const void* k_l2,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       k_pack_hv,
    void*       v_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int pack_total = batches * kChunk * head_dim;
  pack_output_qkv_kernel<<<(pack_total + kThreads - 1) / kThreads,
                           kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_l2),
      reinterpret_cast<const __nv_bfloat16*>(k_l2),
      reinterpret_cast<const __nv_bfloat16*>(v_new),
      reinterpret_cast<__nv_bfloat16*>(q_pack),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(v_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  const float alpha = 1.0f;
  const float beta0 = 0.0f;

  LtPlan& qh_plan = get_qh_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qh_plan.desc, &alpha,
      q_pack, qh_plan.a_desc,
      h0, qh_plan.b_desc,
      &beta0,
      qh_pack, qh_plan.c_desc,
      qh_pack, qh_plan.c_desc,
      &qh_plan.algo, g_workspace, g_workspace_size, stream));

  LtPlan& qk_plan = get_kkt_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qk_plan.desc, &alpha,
      q_pack, qk_plan.a_desc,
      k_pack_hv, qk_plan.b_desc,
      &beta0,
      qk_base, qk_plan.c_desc,
      qk_base, qk_plan.c_desc,
      &qk_plan.algo, g_workspace, g_workspace_size, stream));

  apply_output_local_a_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(qk_base),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(local_a_pack),
      S, num_v_heads);

  LtPlan& mm_plan = get_mm64d_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, mm_plan.desc, &alpha,
      local_a_pack, mm_plan.a_desc,
      v_pack, mm_plan.b_desc,
      &beta0,
      local_pack, mm_plan.c_desc,
      local_pack, mm_plan.c_desc,
      &mm_plan.algo, g_workspace, g_workspace_size, stream));

  combine_output_kernel<<<
      (S * num_v_heads * head_dim + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qh_pack),
      reinterpret_cast<const __nv_bfloat16*>(local_pack),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, head_dim);
}

void gdn_wy_output_o_b64_bf16_cublaslt_packed_k(
    const void* q_l2,
    const void* k_pack_hv,
    const void* v_new,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       v_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int pack_total = batches * kChunk * head_dim;
  pack_output_qv_kernel<<<(pack_total + kThreads - 1) / kThreads,
                           kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_l2),
      reinterpret_cast<const __nv_bfloat16*>(v_new),
      reinterpret_cast<__nv_bfloat16*>(q_pack),
      reinterpret_cast<__nv_bfloat16*>(v_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  const float alpha = 1.0f;
  const float beta0 = 0.0f;

  LtPlan& qh_plan = get_qh_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qh_plan.desc, &alpha,
      q_pack, qh_plan.a_desc,
      h0, qh_plan.b_desc,
      &beta0,
      qh_pack, qh_plan.c_desc,
      qh_pack, qh_plan.c_desc,
      &qh_plan.algo, g_workspace, g_workspace_size, stream));

  LtPlan& qk_plan = get_kkt_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qk_plan.desc, &alpha,
      q_pack, qk_plan.a_desc,
      k_pack_hv, qk_plan.b_desc,
      &beta0,
      qk_base, qk_plan.c_desc,
      qk_base, qk_plan.c_desc,
      &qk_plan.algo, g_workspace, g_workspace_size, stream));

  apply_output_local_a_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(qk_base),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(local_a_pack),
      S, num_v_heads);

  LtPlan& mm_plan = get_mm64d_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, mm_plan.desc, &alpha,
      local_a_pack, mm_plan.a_desc,
      v_pack, mm_plan.b_desc,
      &beta0,
      local_pack, mm_plan.c_desc,
      local_pack, mm_plan.c_desc,
      &mm_plan.algo, g_workspace, g_workspace_size, stream));

  combine_output_kernel<<<
      (S * num_v_heads * head_dim + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qh_pack),
      reinterpret_cast<const __nv_bfloat16*>(local_pack),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, head_dim);
}

void gdn_wy_output_o_b64_bf16_cublaslt_packed_kv(
    const void* q_l2,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h0,
    const void* g_cumsum,
    void*       q_pack,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const int pack_total = batches * kChunk * head_dim;
  pack_output_q_kernel<<<(pack_total + kThreads - 1) / kThreads,
                          kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_l2),
      reinterpret_cast<__nv_bfloat16*>(q_pack),
      S, num_k_heads, num_v_heads, head_dim, qk_group);

  const float alpha = 1.0f;
  const float beta0 = 0.0f;

  LtPlan& qh_plan = get_qh_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qh_plan.desc, &alpha,
      q_pack, qh_plan.a_desc,
      h0, qh_plan.b_desc,
      &beta0,
      qh_pack, qh_plan.c_desc,
      qh_pack, qh_plan.c_desc,
      &qh_plan.algo, g_workspace, g_workspace_size, stream));

  LtPlan& qk_plan = get_kkt_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qk_plan.desc, &alpha,
      q_pack, qk_plan.a_desc,
      k_pack_hv, qk_plan.b_desc,
      &beta0,
      qk_base, qk_plan.c_desc,
      qk_base, qk_plan.c_desc,
      &qk_plan.algo, g_workspace, g_workspace_size, stream));

  apply_output_local_a_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(qk_base),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(local_a_pack),
      S, num_v_heads);

  LtPlan& mm_plan = get_mm64d_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, mm_plan.desc, &alpha,
      local_a_pack, mm_plan.a_desc,
      v_pack, mm_plan.b_desc,
      &beta0,
      local_pack, mm_plan.c_desc,
      local_pack, mm_plan.c_desc,
      &mm_plan.algo, g_workspace, g_workspace_size, stream));

  combine_output_kernel<<<
      (S * num_v_heads * head_dim + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qh_pack),
      reinterpret_cast<const __nv_bfloat16*>(local_pack),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, head_dim);
}

void gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv(
    const void* q_pack,
    const void* k_pack_hv,
    const void* v_pack,
    const void* h0,
    const void* g_cumsum,
    void*       qk_base,
    void*       local_a_pack,
    void*       qh_pack,
    void*       local_pack,
    void*       out,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream)
{
  if (S <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_dim <= 0 || qk_group <= 0) {
    return;
  }
  const int chunks = (S + kChunk - 1) / kChunk;
  const int batches = chunks * num_v_heads;
  const float alpha = 1.0f;
  const float beta0 = 0.0f;

  LtPlan& qh_plan = get_qh_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qh_plan.desc, &alpha,
      q_pack, qh_plan.a_desc,
      h0, qh_plan.b_desc,
      &beta0,
      qh_pack, qh_plan.c_desc,
      qh_pack, qh_plan.c_desc,
      &qh_plan.algo, g_workspace, g_workspace_size, stream));

  LtPlan& qk_plan = get_kkt_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, qk_plan.desc, &alpha,
      q_pack, qk_plan.a_desc,
      k_pack_hv, qk_plan.b_desc,
      &beta0,
      qk_base, qk_plan.c_desc,
      qk_base, qk_plan.c_desc,
      &qk_plan.algo, g_workspace, g_workspace_size, stream));

  apply_output_local_a_kernel<<<
      dim3((kChunk * kChunk + kThreads - 1) / kThreads,
           num_v_heads, chunks),
      kThreads, 0, stream>>>(
      reinterpret_cast<const float*>(qk_base),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(local_a_pack),
      S, num_v_heads);

  LtPlan& mm_plan = get_mm64d_plan(batches, head_dim);
  FLASHRT_CUBLASLT_CHECK(cublasLtMatmul(
      g_lt, mm_plan.desc, &alpha,
      local_a_pack, mm_plan.a_desc,
      v_pack, mm_plan.b_desc,
      &beta0,
      local_pack, mm_plan.c_desc,
      local_pack, mm_plan.c_desc,
      &mm_plan.algo, g_workspace, g_workspace_size, stream));

  combine_output_kernel<<<
      (S * num_v_heads * head_dim + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qh_pack),
      reinterpret_cast<const __nv_bfloat16*>(local_pack),
      reinterpret_cast<const __nv_bfloat16*>(g_cumsum),
      reinterpret_cast<__nv_bfloat16*>(out),
      S, num_v_heads, head_dim);
}

}  // namespace linear_attention
}  // namespace kernels
}  // namespace flash_rt
