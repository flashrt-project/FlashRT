// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini (qwen3_5_moe) layout-glue kernels. See nexn2_misc.cuh for the
// head-count differences vs the qwen36 variants.

#include "kernels/nexn2_misc.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;        // GDN head_k_dim == head_v_dim
constexpr int kVHeads = 32;     // Nex-N2 linear-attn V-heads
constexpr int kConvRow = 8192;  // conv_out row: Q2048 + K2048 + V4096
constexpr int kQHeads = 16;     // Nex-N2 full-attn Q-heads
constexpr int kFullHD = 256;    // full-attn head_dim
constexpr int kSplitThreads = 256;

// 16 K-heads -> 32 V-heads broadcast (ratio 2). Mirror of the qwen36 16->48
// kernel with floor(h/2) and an 8192-wide conv row.
__global__ void nexn2_lin_split_qkv_broadcast_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q32,
    __nv_bfloat16* __restrict__ k32,
    __nv_bfloat16* __restrict__ v32,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * kVHeads * kHD;
  if (idx >= total) return;

  const int t = idx % kHD;
  const int h = (idx / kHD) % kVHeads;
  const int s = idx / (kVHeads * kHD);
  const int src_h = h / 2;
  const size_t row = static_cast<size_t>(s) * kConvRow;
  q32[idx] = conv_out[row + src_h * kHD + t];
  k32[idx] = conv_out[row + 2048 + src_h * kHD + t];
  v32[idx] = conv_out[row + 4096 + h * kHD + t];
}

// Split q_proj (S, 16, 512) -> q_pre (S, 16, 256) + gate (S, 16, 256).
__global__ void nexn2_split_q_gate_kernel(
    const __nv_bfloat16* __restrict__ q_proj,
    __nv_bfloat16* __restrict__ q_pre,
    __nv_bfloat16* __restrict__ gate,
    int S)
{
  const int idx = blockIdx.x * kSplitThreads + threadIdx.x;
  const int total = S * kQHeads * kFullHD;
  if (idx >= total) return;

  const int t = idx % kFullHD;
  const int h = (idx / kFullHD) % kQHeads;
  const int s = idx / (kQHeads * kFullHD);
  const size_t src = (static_cast<size_t>(s) * kQHeads + h) * (2 * kFullHD) + t;
  q_pre[idx] = q_proj[src];
  gate[idx] = q_proj[src + kFullHD];
}

}  // namespace

void nexn2_lin_split_qkv_broadcast_bf16(
    const void* conv_out,
    void*       q32,
    void*       k32,
    void*       v32,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * kVHeads * kHD;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  nexn2_lin_split_qkv_broadcast_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<__nv_bfloat16*>(q32),
      reinterpret_cast<__nv_bfloat16*>(k32),
      reinterpret_cast<__nv_bfloat16*>(v32),
      S);
}

void nexn2_split_q_gate_bf16(
    const void* q_proj,
    void*       q_pre,
    void*       gate,
    int S,
    cudaStream_t stream)
{
  if (S <= 0) return;
  const int total = S * kQHeads * kFullHD;
  dim3 grid((total + kSplitThreads - 1) / kSplitThreads);
  dim3 block(kSplitThreads);
  nexn2_split_q_gate_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_proj),
      reinterpret_cast<__nv_bfloat16*>(q_pre),
      reinterpret_cast<__nv_bfloat16*>(gate),
      S);
}

}  // namespace kernels
}  // namespace flash_rt
