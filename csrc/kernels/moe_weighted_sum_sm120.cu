// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE unpermute for Nex-N2-mini / qwen3_5_moe (sm120). See the header.
//
// One CTA per token gathers that token's TOPK expert-output rows from d_dn and
// accumulates the router-weighted sum into out[t] in fp32. The TOPK (row,
// weight) pairs are staged in shared memory once per CTA; threads then stride
// the HID dimension with coalesced reads of each gathered row. The k-loop runs
// in a fixed order so the fp32 accumulation is bit-reproducible (the torch
// path's atomic / library reductions were not).

#include "kernels/moe_weighted_sum_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int WS_THREADS = 256;

__global__ __launch_bounds__(WS_THREADS) void moe_weighted_sum_kernel(
    const __nv_bfloat16* __restrict__ d_dn,
    const int* __restrict__ rows,
    const float* __restrict__ tw,
    float* __restrict__ out,
    int S, int TOPK, int HID, int dn_stride) {
  const int t = blockIdx.x;
  if (t >= S) return;

  extern __shared__ char smem[];
  int* s_rows = reinterpret_cast<int*>(smem);
  float* s_w = reinterpret_cast<float*>(s_rows + TOPK);
  for (int k = threadIdx.x; k < TOPK; k += blockDim.x) {
    s_rows[k] = rows[(size_t)t * TOPK + k];
    s_w[k] = tw[(size_t)t * TOPK + k];
  }
  __syncthreads();

  for (int hid = threadIdx.x; hid < HID; hid += blockDim.x) {
    float acc = 0.f;
#pragma unroll 4
    for (int k = 0; k < TOPK; ++k) {
      acc += __bfloat162float(
                 d_dn[(size_t)s_rows[k] * dn_stride + hid]) * s_w[k];
    }
    out[(size_t)t * HID + hid] = acc;
  }
}

}  // namespace

int moe_weighted_sum_sm120_bf16(
    const void* d_dn, const void* rows, const void* tw, void* out,
    int S, int TOPK, int HID, int dn_stride, cudaStream_t stream) {
  if (!d_dn || !rows || !tw || !out) return 1;
  if (S <= 0 || TOPK <= 0 || HID <= 0 || dn_stride < HID) return 2;
  const size_t shmem = (size_t)TOPK * (sizeof(int) + sizeof(float));
  moe_weighted_sum_kernel<<<S, WS_THREADS, shmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(d_dn),
      reinterpret_cast<const int*>(rows),
      reinterpret_cast<const float*>(tw),
      reinterpret_cast<float*>(out),
      S, TOPK, HID, dn_stride);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
