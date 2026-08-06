#include "causal_conv1d_rows_edge.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kMaxK = 4;
constexpr int kThreadsX = 256;      // matches the existing prefill entry

// Written the same way as the existing entry's, not merely equivalent to it:
// the two are meant to agree to the bit.
__device__ __forceinline__ float rows_silu(float v) {
  return v / (1.0f + __expf(-v));
}

// One thread, one channel, `kRows` consecutive tokens. The k-1 inputs a token
// shares with the next are kept in registers and shifted along, so the reads
// are one element per output rather than k.
template <int kRows>
__global__ void causal_conv1d_rows_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ hist,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int conv_dim, int k,
    bool apply_silu)
{
  const int c = blockIdx.x * kThreadsX + threadIdx.x;
  if (c >= conv_dim) return;
  const int s0 = blockIdx.y * kRows;
  if (s0 >= S) return;
  const int b = blockIdx.z;

  float wv[kMaxK];
  #pragma unroll
  for (int i = 0; i < kMaxK; ++i) {
    wv[i] = (i < k) ? static_cast<float>(w[c * k + i]) : 0.0f;
  }
  const float b0 = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.0f;

  const size_t base = static_cast<size_t>(b) * S * conv_dim + c;

  // win[j] holds x[s0 - (k-1) + j], the window the first output needs. Before
  // the start of this block that is the previous block's trailing inputs when
  // there are any, and zero when the sequence itself starts here.
  float win[kMaxK];
  #pragma unroll
  for (int j = 0; j < kMaxK; ++j) {
    const int t = s0 - (k - 1) + j;
    if (j >= k) { win[j] = 0.0f; continue; }
    if (t >= 0 && t < S) {
      win[j] = static_cast<float>(x[base + static_cast<size_t>(t) * conv_dim]);
    } else if (t < 0 && hist != nullptr) {
      // hist is (B, conv_dim, k-1), newest last: t == -1 is the final column.
      const int hj = t + (k - 1);
      win[j] = static_cast<float>(
          hist[(static_cast<size_t>(b) * conv_dim + c) * (k - 1) + hj]);
    } else {
      win[j] = 0.0f;
    }
  }

  #pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int s = s0 + r;
    if (s >= S) break;
    if (r > 0) {
      // Shift by one and pull in the token that just became current.
      #pragma unroll
      for (int j = 0; j < kMaxK - 1; ++j) win[j] = win[j + 1];
      win[k - 1] = static_cast<float>(
          x[base + static_cast<size_t>(s) * conv_dim]);
    }
    float acc = b0;
    #pragma unroll
    for (int i = 0; i < kMaxK; ++i) {
      if (i < k) acc = fmaf(win[i], wv[i], acc);
    }
    if (apply_silu) acc = rows_silu(acc);
    out[base + static_cast<size_t>(s) * conv_dim] = __float2bfloat16(acc);
  }
}

}  // namespace

void causal_conv1d_qwen36_rows_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       out,
    int B,
    int S,
    int conv_dim,
    int k,
    bool apply_silu,
    cudaStream_t stream)
{
  if (B <= 0 || S <= 0 || conv_dim <= 0 || k <= 0 || k > kMaxK) return;

  causal_conv1d_qwen36_rows_hist_bf16(x, w, bias, nullptr, out, B, S,
                                      conv_dim, k, apply_silu, stream);
}

void causal_conv1d_qwen36_rows_hist_bf16(
    const void* x,
    const void* w,
    const void* bias,
    const void* hist,
    void*       out,
    int B,
    int S,
    int conv_dim,
    int k,
    bool apply_silu,
    cudaStream_t stream)
{
  if (B <= 0 || S <= 0 || conv_dim <= 0 || k <= 0 || k > kMaxK) return;

  constexpr int kRows = 8;
  const dim3 block(kThreadsX);
  const dim3 grid((conv_dim + kThreadsX - 1) / kThreadsX,
                  (S + kRows - 1) / kRows,
                  B);
  causal_conv1d_rows_kernel<kRows><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(hist),
      reinterpret_cast<__nv_bfloat16*>(out),
      B, S, conv_dim, k, apply_silu);
}

}  // namespace kernels
}  // namespace flash_rt
