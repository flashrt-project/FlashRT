// SPDX-License-Identifier: Apache-2.0
// The elementwise tail of `diffusers.WanRMS_norm`, reproduced exactly.
//
// The reference is
//
//     normalized = F.normalize(x.float(), dim=1).to(x.dtype)
//     return normalized * self.scale * self.gamma + self.bias
//
// which is eight ATen passes over the activation. Five of them -- the division,
// the narrowing cast, and the three parameter applications -- are pure
// elementwise work with no reduction, so they can become one launch, provided
// every intermediate rounding is reproduced.
//
// It has to be every one of them. The plain `bf16_rms_norm_ncdhw` in this
// directory computes `x * rsqrt(mean(x^2) + eps) * gamma` in fp32 and rounds
// once; against this reference that is off by one bf16 ulp on 45% of elements
// (relF 3.3e-3). This model's closed loop is sensitive at that level -- a 1-ulp
// difference on 16 of 786 432 elements is enough to flip a denoising step -- so
// a cosine-similar approximation is not available here.
//
// The normaliser itself is deliberately *not* computed here. ATen's
// `norm(2, dim=1)` over a strided channel axis is not a plain sequential sum --
// a hand-written accumulation, strided or blocked, with 1 to 16 accumulators,
// disagrees with it on 30-60% of outputs by up to one fp32 ulp -- and a
// one-ulp difference in the normaliser moves ~1.5e-5 of the elements across a
// bf16 rounding boundary, which is ~1000 elements per chunk. So the callers
// pass the normaliser in, and this kernel owns only the part whose rounding is
// fully determined.
//
// Measured against the reference on real encoder activations at every encoder
// shape, this decomposition is bit-identical element for element.

#include "bf16_norm_tail_ncdhw.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {
namespace {

__device__ __forceinline__ __nv_bfloat16 norm_tail_elem_(
    __nv_bfloat16 xv, float nv, __nv_bfloat16 gv,
    const __nv_bfloat16* __restrict__ bias, float scale) {
  const float d = fmaxf(nv, 1e-12f);

  // Each step rounds to bf16, and the scalar is used at fp32 precision (not
  // pre-rounded to bf16). Both were established by bit-comparison on real
  // activations, not read off the source: `x.float() * scale` is computed in
  // opmath and rounded, and rounding `scale` to bf16 first differs on 34% of
  // elements.
  const float q = __fdiv_rn(__bfloat162float(xv), d);
  const __nv_bfloat16 nb = __float2bfloat16_rn(q);
  const float t = __fmul_rn(__bfloat162float(nb), scale);
  const __nv_bfloat16 tb = __float2bfloat16_rn(t);
  const float u = __fmul_rn(__bfloat162float(tb), __bfloat162float(gv));
  const __nv_bfloat16 ub = __float2bfloat16_rn(u);
  // the reference adds a Python float 0.0 here; that is exact except at -0.0,
  // and `__fadd_rn` reproduces it rather than assuming
  const float w = bias ? __fadd_rn(__bfloat162float(ub), __bfloat162float(*bias))
                       : __fadd_rn(__bfloat162float(ub), 0.0f);
  return __float2bfloat16_rn(w);
}

__global__ void norm_tail_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ nrm,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    long long n,
    int C,
    long long thw,
    float scale)
{
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // x is [B,C,T,H,W] and nrm is [B,1,T,H,W], so the normaliser index drops the
  // channel term. `i / C` is not that term's inverse -- C does not divide thw --
  // hence the explicit split.
  const long long row = i / thw;          // b*C + c
  const long long spatial = i - row * thw;
  const long long b = row / C;
  const int c = (int)(row - b * C);

  y[i] = norm_tail_elem_(x[i], nrm[b * thw + spatial], gamma[c],
                         bias ? bias + c : nullptr, scale);
}

// Same tail on a channels-last ([B,T,H,W,C] in memory) activation: the
// channel is the fastest index and the normaliser ([B,1,T,H,W], one value per
// position) is indexed by the position alone.
__global__ void norm_tail_ndhwc_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ nrm,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    long long n,
    int C,
    float scale)
{
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long pos = i / C;
  const int c = (int)(i - pos * C);
  y[i] = norm_tail_elem_(x[i], nrm[pos], gamma[c], bias ? bias + c : nullptr, scale);
}

// SiLU exactly as ATen's bf16 `silu` evaluates it: x / (1 + exp(-x)) in fp32
// opmath on the bf16 value, rounded once. The division is spelled __fdiv_rn
// because this target builds with --prec-div=false.
__device__ __forceinline__ __nv_bfloat16 silu_bf16_(__nv_bfloat16 v) {
  const float x = __bfloat162float(v);
  return __float2bfloat16_rn(__fdiv_rn(x, __fadd_rn(1.0f, expf(-x))));
}

// Eight channels per thread (C % 8 == 0, so all eight share one position and
// one normaliser value); same per-element expression sequence as the scalar
// kernel. SILU additionally applies the activation that follows every encoder
// norm except the attention one, removing that elementwise pass.
template <bool SILU>
__global__ void norm_tail_ndhwc_v8_kernel(
    const uint4* __restrict__ x,
    const float* __restrict__ nrm,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ bias,
    uint4* __restrict__ y,
    long long n8,
    int C8,
    float scale)
{
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const long long pos = i / C8;
  const int c0 = (int)(i - pos * C8) * 8;
  const float nv = nrm[pos];
  const uint4 xv = x[i];
  const uint4 gv = *reinterpret_cast<const uint4*>(gamma + c0);
  const __nv_bfloat16* xe = reinterpret_cast<const __nv_bfloat16*>(&xv);
  const __nv_bfloat16* ge = reinterpret_cast<const __nv_bfloat16*>(&gv);
  uint4 out;
  __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const __nv_bfloat16 r = norm_tail_elem_(xe[j], nv, ge[j],
                                            bias ? bias + c0 + j : nullptr, scale);
    oe[j] = SILU ? silu_bf16_(r) : r;
  }
  y[i] = out;
}

}  // namespace

int bf16_norm_tail_ndhwc_v8(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale, int silu,
    cudaStream_t stream)
{
  if (!x_bf16 || !nrm_fp32 || !gamma_bf16 || !y_bf16) return -1;
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0 || (C % 8) != 0) return -2;
  if ((reinterpret_cast<uintptr_t>(x_bf16) | reinterpret_cast<uintptr_t>(y_bf16) |
       reinterpret_cast<uintptr_t>(gamma_bf16)) & 15) return -5;
  const long long n8 = (long long)B * C * T * H * W / 8;
  const int block = 256;
  const long long grid_ll = (n8 + block - 1) / block;
  if (grid_ll > (long long)INT32_MAX) return -4;
  auto kfn = silu ? norm_tail_ndhwc_v8_kernel<true> : norm_tail_ndhwc_v8_kernel<false>;
  kfn<<<static_cast<unsigned>(grid_ll), block, 0, stream>>>(
      reinterpret_cast<const uint4*>(x_bf16),
      reinterpret_cast<const float*>(nrm_fp32),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<uint4*>(y_bf16),
      n8, C / 8, scale);
  return static_cast<int>(cudaGetLastError());
}

int bf16_norm_tail_ncdhw(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale,
    cudaStream_t stream)
{
  if (!x_bf16 || !nrm_fp32 || !gamma_bf16 || !y_bf16) return -1;
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -2;
  const long long n = (long long)B * C * T * H * W;
  if (n <= 0) return -3;

  const int block = 256;
  const long long grid_ll = (n + block - 1) / block;
  if (grid_ll > (long long)INT32_MAX) return -4;

  norm_tail_kernel<<<static_cast<unsigned>(grid_ll), block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const float*>(nrm_fp32),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      n, C, (long long)T * H * W, scale);
  return static_cast<int>(cudaGetLastError());
}

int bf16_norm_tail_ndhwc(
    const void* x_bf16,
    const void* nrm_fp32,
    const void* gamma_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int B, int C, int T, int H, int W,
    float scale,
    cudaStream_t stream)
{
  if (!x_bf16 || !nrm_fp32 || !gamma_bf16 || !y_bf16) return -1;
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -2;
  const long long n = (long long)B * C * T * H * W;
  const int block = 256;
  const long long grid_ll = (n + block - 1) / block;
  if (grid_ll > (long long)INT32_MAX) return -4;
  norm_tail_ndhwc_kernel<<<static_cast<unsigned>(grid_ll), block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const float*>(nrm_fp32),
      reinterpret_cast<const __nv_bfloat16*>(gamma_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      n, C, scale);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace quantize
}  // namespace flash_rt
