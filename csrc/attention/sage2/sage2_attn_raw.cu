/*
 * FlashRT raw-pointer wrapper for the SageAttention2 SM120 QK-int8/PV-fp8
 * attention core.
 *
 * Source basis:
 *   SageAttention qk_int_sv_f8_cuda_sm89.cu, Apache-2.0.
 *
 * This file is additive and intentionally exposes only the Motus prior-test
 * shape contract: NHD contiguous Q/K/O, V in Sage per-channel fp8 transposed
 * layout [B, D, H, padded_K], BF16 output, non-causal, D=128.
 */

#include "sage2_attn_raw.cuh"

#include <algorithm>
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <mutex>

#include "qattn/qk_int_sv_f8_core.cuh"
#undef PACK_SIZE_QK
#undef PACK_SIZE_V
#undef PACK_SIZE_O
#undef MMA_QK_M
#undef MMA_QK_N
#undef MMA_QK_K
#undef MMA_SV_M
#undef MMA_SV_N
#undef MMA_SV_K
#include "qattn/qk_int_sv_f16_core.cuh"

namespace flash_rt::attention::sage2 {
namespace {

constexpr int kHeadDim = 128;
constexpr int kCtaQ = 128;
constexpr int kCtaK = 64;
constexpr int kWarpQ = 32;
constexpr int kWarpK = 64;

inline int div_up_int(int x, int y) {
  return (x + y - 1) / y;
}

}  // namespace

int qk_int8_sv_f8_bf16_nhd_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    cudaStream_t stream) {
  if (!q_int8 || !k_int8 || !v_fp8 || !out_bf16 || !q_scale || !k_scale || !v_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0) {
    return -2;
  }

  const int padded_k = div_up_int(seqlen_k, kCtaK) * kCtaK;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);

  // V layout from SageAttention2 per_channel_fp8(NHD):
  //   [B, D, H, padded_K], contiguous.
  const uint32_t stride_bz_v = static_cast<uint32_t>(kHeadDim * num_heads * padded_k);
  const uint32_t stride_h_v = static_cast<uint32_t>(padded_k);
  const uint32_t stride_d_v = static_cast<uint32_t>(num_heads * padded_k);

  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kCudaCore,
      MaskMode::kNone,
      false,
      true,
      false>);

  Kernel kernel = qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kCudaCore,
      MaskMode::kNone,
      false,
      true,
      false>;

  const size_t smem_qk =
      static_cast<size_t>(kCtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim);
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qk, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));

  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(v_fp8)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      nullptr,
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(v_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      1,
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale);

  return static_cast<int>(cudaGetLastError());
}

int qk_int8_sv_f16_bf16_nhd_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    cudaStream_t stream) {
  if (!q_int8 || !k_int8 || !v_half || !out_bf16 || !q_scale || !k_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0) {
    return -2;
  }

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_v = static_cast<uint32_t>(seqlen_k * num_heads * kHeadDim);
  const uint32_t stride_seq_v = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_v = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kTensorCore,
      MaskMode::kNone,
      false,
      false>);

  Kernel kernel = qk_int_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kTensorCore,
      MaskMode::kNone,
      false,
      false>;

  const size_t smem_qkv =
      static_cast<size_t>(kCtaQ * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(half));
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));

  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<half*>(reinterpret_cast<const half*>(v_half)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      nullptr,
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      1,
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale);

  return static_cast<int>(cudaGetLastError());
}

}  // namespace flash_rt::attention::sage2
