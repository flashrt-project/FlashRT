#include <array>
#include <stdexcept>
#include <vector>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_query_attention_forward_wmma.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32 = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AElementOp = PassThrough;
using B0ElementOp = PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp = PassThrough;
using CElementOp = PassThrough;
using EmptyBias = ck::Tuple<>;

constexpr ck::index_t NumDimG = 2;
constexpr ck::index_t NumDimM = 1;
constexpr ck::index_t NumDimN = 1;
constexpr ck::index_t NumDimK = 1;
constexpr ck::index_t NumDimO = 1;
constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
constexpr auto MaskingSpec = ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
constexpr auto TensorSpec = ck::tensor_operation::device::TensorSpecialization::Default;

template <ck::index_t QueryGroupNumber>
using Pi05GqaBf16 = ck::tensor_operation::device::DeviceGroupedQueryAttentionForward_Wmma<
    NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,
    BF16, BF16, BF16, BF16, EmptyBias, F32, EmptyBias, F32, F32,
    AElementOp, B0ElementOp, Acc0ElementOp, B1ElementOp, CElementOp,
    GemmSpec, TensorSpec, TensorSpec, TensorSpec, TensorSpec, 1,
    QueryGroupNumber,
    256,
    128, 128, 64, 8, 8,
    64, 64, 8,
    16, 16, 16,
    1, 8, 4,
    S<2, 128, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, true,
    S<2, 16, 8>, S<0, 2, 1>, S<0, 2, 1>, 1, 1, 1, false,
    1, 1, S<1, 128, 1, 2>, 8,
    MaskingSpec>;

template <ck::index_t QueryGroupNumber>
void run_pi05_gqa_bf16(const void* q, const void* k, const void* v, void* out,
                       int batch, int q_heads, int kv_heads,
                       int q_seq, int kv_seq, int head_dim,
                       float softmax_scale, hipStream_t stream) {
  if (batch <= 0 || q_heads <= 0 || kv_heads <= 0 || q_seq <= 0 ||
      kv_seq <= 0 || head_dim <= 0) {
    throw std::runtime_error("invalid Pi0.5 attention shape");
  }
  if (q_heads != kv_heads * QueryGroupNumber) {
    throw std::runtime_error("invalid Pi0.5 attention head grouping");
  }

  using DeviceOp = Pi05GqaBf16<QueryGroupNumber>;
  DeviceOp op;
  auto invoker = op.MakeInvoker();

  auto argument = DeviceOp::MakeArgument(
      static_cast<const BF16*>(q), static_cast<const BF16*>(k),
      static_cast<const BF16*>(v), static_cast<BF16*>(out),
      q_seq, kv_seq, head_dim, head_dim, batch, q_heads,
      softmax_scale, true, true);

  if (!op.IsSupportedArgument(argument)) {
    throw std::runtime_error("unsupported Pi0.5 CK attention shape");
  }
  invoker.Run(argument, StreamConfig{stream, false});
}

}  // namespace

void launch_pi05_siglip_attention_bf16(const void* q, const void* k,
                                       const void* v, void* out,
                                       int batch, int q_seq,
                                       hipStream_t stream) {
  run_pi05_gqa_bf16<16>(q, k, v, out, batch, 16, 1, q_seq, q_seq, 72,
                        0.11785113019775793f, stream);
}

void launch_pi05_gqa8_attention_bf16(const void* q, const void* k,
                                     const void* v, void* out,
                                     int batch, int q_seq, int kv_seq,
                                     hipStream_t stream) {
  run_pi05_gqa_bf16<8>(q, k, v, out, batch, 8, 1, q_seq, kv_seq, 256,
                       0.0625f, stream);
}
