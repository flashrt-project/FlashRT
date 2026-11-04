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
      softmax_scale, 0.0f, false, nullptr, nullptr, false, nullptr, 0u, 0u);

  return static_cast<int>(cudaGetLastError());
}

namespace {

// Shared launcher for the QK-int8 / PV-fp8 core. CtaQ/WarpQ select the query
// tile geometry: the 128/32 pair is the delivered full-query tile, 32/16 is
// the short-query tile (S_q <= 32) and 128/16 trades per-warp registers for
// eight warps so a 128-row query spreads its softmax over twice the lanes.
template <int CtaQ, int WarpQ, bool kLse = false>
int launch_f8_strided(const void* q_int8,
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
                      int q_batch_stride,
                      int k_batch_stride,
                      int v_padded_k,
                      int o_batch_stride,
                      cudaStream_t stream,
                      float pad_extra,
                      bool pad_exact,
                      void* out_fp8,
                      float* out_fp8_scale,
                      bool scale_per_head,
                      float* lse = nullptr,
                      const int* ring_origin = nullptr,
                      int ring_R = 0,
                      int ring_start = 0)
{
  if (!q_int8 || !k_int8 || !v_fp8 || !out_bf16 || !q_scale || !k_scale || !v_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0) {
    return -2;
  }

  // The caller fixes the V padded-K row length; the live window is a prefix
  // of it, so the layout does not change as the cache grows.
  const uint32_t stride_bz_q = static_cast<uint32_t>(q_batch_stride);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_k = static_cast<uint32_t>(k_batch_stride);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_v = static_cast<uint32_t>(kHeadDim * num_heads * v_padded_k);
  const uint32_t stride_h_v = static_cast<uint32_t>(v_padded_k);
  const uint32_t stride_d_v = static_cast<uint32_t>(num_heads * v_padded_k);

  const uint32_t stride_bz_o = static_cast<uint32_t>(o_batch_stride);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f8_attn_kernel<
      CtaQ, kCtaK, WarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kCudaCore,
      MaskMode::kNone,
      kLse,
      true,
      false>);

  Kernel kernel = qk_int_sv_f8_attn_kernel<
      CtaQ, kCtaK, WarpQ, kWarpK, kHeadDim,
      DataType::kInt8,
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,
      false,
      nv_bfloat16,
      ComputeUnit::kCudaCore,
      MaskMode::kNone,
      kLse,
      true,
      false>;

  const size_t smem_qk =
      static_cast<size_t>(CtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim);
  const size_t smem_o = static_cast<size_t>(CtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qk, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, CtaQ), num_heads, batch);
  dim3 block(32, (CtaQ / WarpQ) * (kCtaK / kWarpK));

  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(v_fp8)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      lse,
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
      softmax_scale, pad_extra, pad_exact,
      reinterpret_cast<int8_t*>(out_fp8), out_fp8_scale, scale_per_head,
      ring_origin, static_cast<uint32_t>(ring_R), static_cast<uint32_t>(ring_start));

  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int qk_int8_sv_f8_bf16_nhd_d128_strided_ring(
    const void* q_int8, const void* k_int8, const void* v_fp8, void* out_bf16,
    float* lse, const void* q_scale, const void* k_scale, const void* v_scale,
    int batch, int seqlen_q, int seqlen_k, int num_heads, float softmax_scale,
    int q_batch_stride, int k_batch_stride, int v_padded_k, int o_batch_stride,
    cudaStream_t stream, void* out_fp8, float* out_fp8_scale, bool scale_per_head,
    int tile, const int* ring_origin, int ring_R, int ring_start) {
  if (ring_origin == nullptr || ring_R <= 0 || seqlen_k > ring_R) {
    return -4;
  }
#define SAGE_RING_LAUNCH(CQ, WQ, LSE)                                                      \
  return launch_f8_strided<CQ, WQ, LSE>(q_int8, k_int8, v_fp8, out_bf16, q_scale, k_scale,  \
                                        v_scale, batch, seqlen_q, seqlen_k, num_heads,     \
                                        softmax_scale, q_batch_stride, k_batch_stride,     \
                                        v_padded_k, o_batch_stride, stream, 0.0f, false,   \
                                        out_fp8, out_fp8_scale, scale_per_head, lse,       \
                                        ring_origin, ring_R, ring_start)
  if (lse != nullptr) {
    if (tile == 1) SAGE_RING_LAUNCH(32, 16, true);
    if (tile == 2) SAGE_RING_LAUNCH(128, 16, true);
    SAGE_RING_LAUNCH(kCtaQ, kWarpQ, true);
  }
  if (tile == 1) {
    if (seqlen_q > 32) return -3;
    SAGE_RING_LAUNCH(32, 16, false);
  }
  if (tile == 2) SAGE_RING_LAUNCH(128, 16, false);
  if (tile == 3) SAGE_RING_LAUNCH(64, 16, false);
  SAGE_RING_LAUNCH(kCtaQ, kWarpQ, false);
#undef SAGE_RING_LAUNCH
}

int qk_int8_sv_f8_bf16_nhd_d128_strided(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra,
    bool pad_exact,
    void* out_fp8,
    float* out_fp8_scale,
    bool scale_per_head) {
  return launch_f8_strided<kCtaQ, kWarpQ>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                  k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                  num_heads, softmax_scale, q_batch_stride,
                                  k_batch_stride, v_padded_k, o_batch_stride,
                                  stream, pad_extra, pad_exact, out_fp8, out_fp8_scale,
                                  scale_per_head);
}

int qk_int8_sv_f8_bf16_nhd_d128_strided_sq(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra,
    bool pad_exact,
    void* out_fp8,
    float* out_fp8_scale,
    bool scale_per_head) {
  if (seqlen_q > 32) {
    return -3;
  }
  return launch_f8_strided<32, 16>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                  k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                  num_heads, softmax_scale, q_batch_stride,
                                  k_batch_stride, v_padded_k, o_batch_stride,
                                  stream, pad_extra, pad_exact, out_fp8, out_fp8_scale,
                                  scale_per_head);
}

int qk_int8_sv_f8_bf16_nhd_d128_strided_w16(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra,
    bool pad_exact,
    void* out_fp8,
    float* out_fp8_scale,
    bool scale_per_head) {
  return launch_f8_strided<128, 16>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                  k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                  num_heads, softmax_scale, q_batch_stride,
                                  k_batch_stride, v_padded_k, o_batch_stride,
                                  stream, pad_extra, pad_exact, out_fp8, out_fp8_scale,
                                  scale_per_head);
}

int qk_int8_sv_f8_bf16_nhd_d128_strided_q64(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    float pad_extra,
    bool pad_exact,
    void* out_fp8,
    float* out_fp8_scale,
    bool scale_per_head) {
  return launch_f8_strided<64, 16>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                  k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                  num_heads, softmax_scale, q_batch_stride,
                                  k_batch_stride, v_padded_k, o_batch_stride,
                                  stream, pad_extra, pad_exact, out_fp8, out_fp8_scale,
                                  scale_per_head);
}

// Same 128x32 core with the per-(head, row) log-sum-exp also written (log2
// domain, [B, H, S_q]). A caller that attends one query set to several key
// segments merges the segments' outputs with these. The core writes the LSE
// minus S_FP8_OFFSET (see sage2_tail_merge.cu, which adds it back).
int qk_int8_sv_f8_bf16_nhd_d128_strided_lse(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    float* lse,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    float softmax_scale,
    int q_batch_stride,
    int k_batch_stride,
    int v_padded_k,
    int o_batch_stride,
    cudaStream_t stream,
    bool scale_per_head,
    int tile) {
  if (lse == nullptr) {
    return -1;
  }
  if (tile == 1) {
    return launch_f8_strided<32, 16, true>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                    k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                    num_heads, softmax_scale, q_batch_stride,
                                    k_batch_stride, v_padded_k, o_batch_stride,
                                    stream, 0.0f, false, nullptr, nullptr,
                                    scale_per_head, lse);
  }
  if (tile == 2) {
    return launch_f8_strided<128, 16, true>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                    k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                    num_heads, softmax_scale, q_batch_stride,
                                    k_batch_stride, v_padded_k, o_batch_stride,
                                    stream, 0.0f, false, nullptr, nullptr,
                                    scale_per_head, lse);
  }
  return launch_f8_strided<kCtaQ, kWarpQ, true>(q_int8, k_int8, v_fp8, out_bf16, q_scale,
                                  k_scale, v_scale, batch, seqlen_q, seqlen_k,
                                  num_heads, softmax_scale, q_batch_stride,
                                  k_batch_stride, v_padded_k, o_batch_stride,
                                  stream, 0.0f, false, nullptr, nullptr,
                                  scale_per_head, lse);
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

// Strided-batch entry: identical kernel, but the batch strides come from the
// caller. Used when K/V live in a fixed-capacity cache whose live window is a
// prefix (the dense stride the plain entry assumes would be seqlen_k*H*D).
int qk_int8_sv_f16_bf16_nhd_d128_strided(
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
    int q_batch_stride,
    int k_batch_stride,
    int v_batch_stride,
    int o_batch_stride,
    cudaStream_t stream) {
  if (!q_int8 || !k_int8 || !v_half || !out_bf16 || !q_scale || !k_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0) {
    return -2;
  }
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_seq_v = static_cast<uint32_t>(num_heads * kHeadDim);
  const uint32_t stride_h_v = static_cast<uint32_t>(kHeadDim);
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
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
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
      static_cast<uint32_t>(q_batch_stride), stride_seq_q, stride_h_q,
      static_cast<uint32_t>(k_batch_stride), stride_seq_k, stride_h_k,
      static_cast<uint32_t>(v_batch_stride), stride_seq_v, stride_h_v,
      static_cast<uint32_t>(o_batch_stride), stride_seq_o, stride_h_o,
      softmax_scale);
  return static_cast<int>(cudaGetLastError());
}

int qk_int8_sv_f16_bf16_gqa_nhd_d256(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream) {
  constexpr int kHeadDim256 = 256;
  if (!q_int8 || !k_int8 || !v_half || !out_bf16 || !q_scale || !k_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }

  const int num_kv_groups = num_q_heads / num_kv_heads;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim256);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim256);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim256);

  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim256);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim256);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim256);

  const uint32_t stride_bz_v = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim256);
  const uint32_t stride_seq_v = static_cast<uint32_t>(num_kv_heads * kHeadDim256);
  const uint32_t stride_h_v = static_cast<uint32_t>(kHeadDim256);

  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim256);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim256);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim256);

  using Kernel = decltype(&qk_int_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim256,
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
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim256,
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
      static_cast<size_t>(kCtaQ * kHeadDim256 * sizeof(int8_t) +
                          kCtaK * kHeadDim256 * sizeof(int8_t) +
                          kCtaK * kHeadDim256 * sizeof(half));
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim256 * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
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
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale);

  return static_cast<int>(cudaGetLastError());
}

int qk_int8_sv_f8_bf16_gqa_nhd_d256(
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
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream) {
  constexpr int kHeadDim256 = 256;
  if (!q_int8 || !k_int8 || !v_fp8 || !out_bf16 ||
      !q_scale || !k_scale || !v_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }

  const int num_kv_groups = num_q_heads / num_kv_heads;
  const int padded_k = div_up_int(seqlen_k, kCtaK) * kCtaK;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim256);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim256);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim256);

  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim256);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim256);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim256);

  // Sage per-channel FP8 V layout: [B, D, Hkv, padded_K].
  const uint32_t stride_bz_v = static_cast<uint32_t>(kHeadDim256 * num_kv_heads * padded_k);
  const uint32_t stride_h_v = static_cast<uint32_t>(padded_k);
  const uint32_t stride_d_v = static_cast<uint32_t>(num_kv_heads * padded_k);

  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim256);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim256);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim256);

  using Kernel = decltype(&qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim256,
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
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim256,
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

  const size_t smem_qkv =
      static_cast<size_t>(kCtaQ * kHeadDim256 +
                          kCtaK * kHeadDim256 +
                          kCtaK * kHeadDim256);
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim256 * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
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
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale, 0.0f, false, nullptr, nullptr, false, nullptr, 0u, 0u);

  return static_cast<int>(cudaGetLastError());
}

}  // namespace flash_rt::attention::sage2
