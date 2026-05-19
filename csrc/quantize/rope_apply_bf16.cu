// SPDX-License-Identifier: Apache-2.0
// G7.15 — Fused 3D RoPE apply replacing 5-6 Python-dispatched
// launches with a single CUDA kernel. See header.
//
// G7.16 — Templated on output dtype. bf16 output enables the
// cat(q_video_rope, action_q, und_q) downstream to stay in bf16,
// letting FA2 dispatch its bf16 tensor-core fast path instead of
// fp32 fallback.

#include "rope_apply_bf16.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

template <typename TOut>
__device__ __forceinline__ TOut cvt_out(float v);

template <>
__device__ __forceinline__ float cvt_out<float>(float v) { return v; }

template <>
__device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename TOut>
__global__ void rope_apply_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float*          __restrict__ freqs_re,
    const float*          __restrict__ freqs_im,
    TOut*                __restrict__ out,
    int B, int T, int N, int head_dim, int seq_len)
{
  const int c_complex = head_dim >> 1;
  const long long total = (long long)B * T * N * c_complex;
  const long long idx =
      (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  long long rem = idx;
  const int c = (int)(rem % (long long)c_complex); rem /= c_complex;
  const int n = (int)(rem % (long long)N);          rem /= N;
  const int t = (int)(rem % (long long)T);          rem /= T;
  const int b = (int)rem;

  const long long base =
      ((((long long)b * T + t) * N + n) * head_dim) + (long long)(2 * c);
  const float re_x = __bfloat162float(in[base]);
  const float im_x = __bfloat162float(in[base + 1]);

  float re_y, im_y;
  if (t < seq_len) {
    const long long f_off = (long long)t * c_complex + c;
    const float re_f = freqs_re[f_off];
    const float im_f = freqs_im[f_off];
    re_y = re_x * re_f - im_x * im_f;
    im_y = re_x * im_f + im_x * re_f;
  } else {
    re_y = re_x;
    im_y = im_x;
  }

  out[base]     = cvt_out<TOut>(re_y);
  out[base + 1] = cvt_out<TOut>(im_y);
}

}  // namespace

void rope_apply_bf16_to_fp32(
    const void*  in_bf16,
    const float* freqs_re,
    const float* freqs_im,
    void*        out_fp32,
    int B, int T, int N, int head_dim, int seq_len,
    cudaStream_t stream)
{
  const int c_complex = head_dim >> 1;
  const long long total = (long long)B * T * N * c_complex;
  if (total <= 0) return;
  const int block_sz = 256;
  const unsigned grid =
      (unsigned)((total + block_sz - 1) / block_sz);
  rope_apply_bf16_kernel<float><<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      freqs_re, freqs_im,
      reinterpret_cast<float*>(out_fp32),
      B, T, N, head_dim, seq_len);
}

void rope_apply_bf16_to_bf16(
    const void*  in_bf16,
    const float* freqs_re,
    const float* freqs_im,
    void*        out_bf16,
    int B, int T, int N, int head_dim, int seq_len,
    cudaStream_t stream)
{
  const int c_complex = head_dim >> 1;
  const long long total = (long long)B * T * N * c_complex;
  if (total <= 0) return;
  const int block_sz = 256;
  const unsigned grid =
      (unsigned)((total + block_sz - 1) / block_sz);
  rope_apply_bf16_kernel<__nv_bfloat16><<<grid, block_sz, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(in_bf16),
      freqs_re, freqs_im,
      reinterpret_cast<__nv_bfloat16*>(out_bf16),
      B, T, N, head_dim, seq_len);
}

}  // namespace quantize
}  // namespace flash_rt
