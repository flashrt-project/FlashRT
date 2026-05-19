// SPDX-License-Identifier: Apache-2.0
// G7.17 — Fused AdaLayerNorm + per-tensor FP8 quantize. See header.
// Math mirrors csrc/kernels/dit_bf16.cu::ada_layer_norm_bf16 with the
// final store path replaced by fp8 e4m3 quantize+pack.

#include "ada_layer_norm_fp8.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace quantize {

namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ uint8_t f32_to_fp4_e2m1(float x) {
  if (x == 0.0f) return 0;
  uint8_t sign = x < 0 ? 0x8 : 0;
  float ax = fabsf(x);
  if (ax < 0.25f) return sign;
  if (ax < 0.75f) return sign | 0x1;  // 0.5
  if (ax < 1.5f) return sign | 0x2;   // 1
  if (ax < 3.0f) return sign | 0x3;   // 2
  if (ax < 5.0f) return sign | 0x6;   // 4
  return sign | 0x7;                  // 6
}

__device__ __forceinline__ uint8_t f32_to_ue4m3_ceil_local(float x) {
  if (x <= 0.0f) return 0;
  if (x < 0.0009765625f) return 1;
  int exp;
  float mant = frexpf(x, &exp);
  exp -= 1;
  float frac = mant * 2.0f - 1.0f;
  int mantissa = (int)ceilf(frac * 8.0f);
  if (mantissa >= 8) {
    mantissa = 0;
    exp += 1;
  }
  int biased_exp = exp + 7;
  if (biased_exp <= 0) return 1;
  if (biased_exp >= 15) return 0x7F;
  return (uint8_t)((biased_exp << 3) | mantissa);
}

__device__ __forceinline__ float ue4m3_to_f32_local(uint8_t v) {
  if (v == 0) return 0.0f;
  int exp = ((v >> 3) & 0xF) - 7;
  int mant = v & 0x7;
  float m = 1.0f + mant / 8.0f;
  return ldexpf(m, exp);
}

__global__ void ada_layer_norm_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    int dim, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  const __nv_bfloat162* sc2 =
      reinterpret_cast<const __nv_bfloat162*>(scale);
  const __nv_bfloat162* sh2 =
      reinterpret_cast<const __nv_bfloat162*>(shift);
  __nv_fp8_e4m3* out_row = out + (long long)row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];

  // ── Pass 1: mean ──
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  // ── Pass 2: variance ──
  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);

  // ── Pass 3: ada_modulate + fp8 quantize ──
  // Register-level round-through bf16 BEFORE fp8 conversion to match
  // the reference (ada_layer_norm_bf16 → bf16 store → quantize_fp8)
  // bit-exactly while skipping the actual bf16 global-memory round-trip.
  // This is the precision/memory tradeoff sweet spot: zero memory cost
  // for the bf16 intermediate, exact cos match with the 2-launch chain.
  const float inv_a = 1.0f / *act_scale_ptr;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0_f32 = n0 * (1.0f + __bfloat162float(sv.x))
                          + __bfloat162float(hv.x);
    const float v1_f32 = n1 * (1.0f + __bfloat162float(sv.y))
                          + __bfloat162float(hv.y);
    // Round through bf16 in registers (no memory write).
    const float v0 = __bfloat162float(__float2bfloat16(v0_f32));
    const float v1 = __bfloat162float(__float2bfloat16(v1_f32));
    float q0 = fminf(fmaxf(v0 * inv_a, -kFp8Max), kFp8Max);
    float q1 = fminf(fmaxf(v1 * inv_a, -kFp8Max), kFp8Max);
    out_row[2 * i]     = __nv_fp8_e4m3(q0);
    out_row[2 * i + 1] = __nv_fp8_e4m3(q1);
  }
}

__global__ void awq_ada_layer_norm_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    const __nv_bfloat16* __restrict__ inv_s,
    __nv_fp8_e4m3*       __restrict__ out,
    const float*          __restrict__ act_scale_ptr,
    int dim, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  const __nv_bfloat162* sc2 =
      reinterpret_cast<const __nv_bfloat162*>(scale);
  const __nv_bfloat162* sh2 =
      reinterpret_cast<const __nv_bfloat162*>(shift);
  const __nv_bfloat162* inv2 =
      reinterpret_cast<const __nv_bfloat162*>(inv_s);
  __nv_fp8_e4m3* out_row = out + (long long)row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);

  const float inv_a = 1.0f / *act_scale_ptr;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i], iv = inv2[i];
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0 = __bfloat162float(__float2bfloat16(
        n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x)));
    const float v1 = __bfloat162float(__float2bfloat16(
        n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y)));
    float q0 = fminf(fmaxf(v0 * __bfloat162float(iv.x) * inv_a,
                           -kFp8Max), kFp8Max);
    float q1 = fminf(fmaxf(v1 * __bfloat162float(iv.y) * inv_a,
                           -kFp8Max), kFp8Max);
    out_row[2 * i]     = __nv_fp8_e4m3(q0);
    out_row[2 * i + 1] = __nv_fp8_e4m3(q1);
  }
}

__global__ void ada_layer_norm_fp8_modfp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ scale,
    const __nv_fp8_e4m3* __restrict__ shift,
    const float* __restrict__ scale_deq,
    const float* __restrict__ shift_deq,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ act_scale_ptr,
    int dim, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  __nv_fp8_e4m3* out_row = out + (long long)row * dim;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float mean = shared[0] / static_cast<float>(dim);

  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) shared[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) shared[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(shared[0] / static_cast<float>(dim) + eps);

  const float s_deq = *scale_deq;
  const float h_deq = *shift_deq;
  const float inv_a = 1.0f / *act_scale_ptr;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 xv = x2[i];
    const int j = i << 1;
    const float s0 = static_cast<float>(scale[j]) * s_deq;
    const float s1 = static_cast<float>(scale[j + 1]) * s_deq;
    const float h0 = static_cast<float>(shift[j]) * h_deq;
    const float h1 = static_cast<float>(shift[j + 1]) * h_deq;
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0 = __bfloat162float(__float2bfloat16(n0 * (1.0f + s0) + h0));
    const float v1 = __bfloat162float(__float2bfloat16(n1 * (1.0f + s1) + h1));
    out_row[j] = __nv_fp8_e4m3(fminf(fmaxf(v0 * inv_a, -kFp8Max), kFp8Max));
    out_row[j + 1] = __nv_fp8_e4m3(fminf(fmaxf(v1 * inv_a, -kFp8Max), kFp8Max));
  }
}

__global__ void ada_layer_norm_nvfp4_swizzled_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sf_swizzled,
    int dim, int num_blocks, int n_col_blocks, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  const __nv_bfloat162* sc2 =
      reinterpret_cast<const __nv_bfloat162*>(scale);
  const __nv_bfloat162* sh2 =
      reinterpret_cast<const __nv_bfloat162*>(shift);
  uint8_t* row_packed = packed + (long long)row * dim / 2;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float* red = shared;
  float* blk_scale = shared + 256;

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) red[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) red[0] = val;
  __syncthreads();
  const float mean = red[0] / static_cast<float>(dim);

  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) red[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) red[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(red[0] / static_cast<float>(dim) + eps);

  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
    blk_scale[b] = 0.0f;
  __syncthreads();

  for (int i2 = threadIdx.x; i2 < dim2; i2 += blockDim.x) {
    __nv_bfloat162 xv = x2[i2], sv = sc2[i2], hv = sh2[i2];
    const int i = i2 << 1;
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0 = __bfloat162float(__float2bfloat16(
        n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x)));
    const float v1 = __bfloat162float(__float2bfloat16(
        n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y)));
    atomicMax((int*)&blk_scale[i >> 4], __float_as_int(fabsf(v0)));
    atomicMax((int*)&blk_scale[(i + 1) >> 4], __float_as_int(fabsf(v1)));
  }
  __syncthreads();

  const int rb = row / 128;
  const int ri = row % 128;
  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
    float amax = __int_as_float(*(int*)&blk_scale[b]);
    uint8_t ue = f32_to_ue4m3_ceil_local(amax / 6.0f);
    int cb = b / 4;
    int ci = b % 4;
    int out_idx = (rb * n_col_blocks + cb) * 512
        + (ri % 32) * 16 + (ri / 32) * 4 + ci;
    sf_swizzled[out_idx] = ue;
    blk_scale[b] = ue4m3_to_f32_local(ue);
  }
  __syncthreads();

  for (int p = threadIdx.x; p < dim / 2; p += blockDim.x) {
    const int i = p << 1;
    const int i2 = p;
    __nv_bfloat162 xv = x2[i2], sv = sc2[i2], hv = sh2[i2];
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float m0 = __bfloat162float(__float2bfloat16(
        n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x)));
    const float m1 = __bfloat162float(__float2bfloat16(
        n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y)));
    const int b0 = i >> 4;
    const int b1 = (i + 1) >> 4;
    const float inv0 = blk_scale[b0] > 0.0f ? 1.0f / blk_scale[b0] : 0.0f;
    const float inv1 = blk_scale[b1] > 0.0f ? 1.0f / blk_scale[b1] : 0.0f;
    uint8_t lo = f32_to_fp4_e2m1(m0 * inv0);
    uint8_t hi = f32_to_fp4_e2m1(m1 * inv1);
    row_packed[p] = (hi << 4) | (lo & 0x0F);
  }
}

__global__ void ada_layer_norm_nvfp4_swizzled_modfp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ scale,
    const __nv_fp8_e4m3* __restrict__ shift,
    const float* __restrict__ scale_deq,
    const float* __restrict__ shift_deq,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sf_swizzled,
    int dim, int num_blocks, int n_col_blocks, float eps)
{
  const int row = blockIdx.x;
  const __nv_bfloat162* x2 =
      reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
  uint8_t* row_packed = packed + (long long)row * dim / 2;
  const int dim2 = dim >> 1;

  extern __shared__ float shared[];
  float* red = shared;
  float* blk_scale = shared + 256;

  float local_sum = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
  }
  float val = local_sum;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (!lane) red[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) red[0] = val;
  __syncthreads();
  const float mean = red[0] / static_cast<float>(dim);

  float local_var = 0.f;
  for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
    __nv_bfloat162 v = x2[i];
    const float d0 = __bfloat162float(v.x) - mean;
    const float d1 = __bfloat162float(v.y) - mean;
    local_var += d0 * d0 + d1 * d1;
  }
  val = local_var;
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, o);
  if (!lane) red[wid] = val;
  __syncthreads();
  if (!wid) {
    val = (lane < (blockDim.x >> 5)) ? red[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, o);
  }
  __syncthreads();
  if (!threadIdx.x) red[0] = val;
  __syncthreads();
  const float inv_std = rsqrtf(red[0] / static_cast<float>(dim) + eps);

  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
    blk_scale[b] = 0.0f;
  __syncthreads();

  const float s_deq = *scale_deq;
  const float h_deq = *shift_deq;
  for (int i2 = threadIdx.x; i2 < dim2; i2 += blockDim.x) {
    __nv_bfloat162 xv = x2[i2];
    const int i = i2 << 1;
    const float s0 = static_cast<float>(scale[i]) * s_deq;
    const float s1 = static_cast<float>(scale[i + 1]) * s_deq;
    const float h0 = static_cast<float>(shift[i]) * h_deq;
    const float h1 = static_cast<float>(shift[i + 1]) * h_deq;
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float v0 = __bfloat162float(__float2bfloat16(n0 * (1.0f + s0) + h0));
    const float v1 = __bfloat162float(__float2bfloat16(n1 * (1.0f + s1) + h1));
    atomicMax((int*)&blk_scale[i >> 4], __float_as_int(fabsf(v0)));
    atomicMax((int*)&blk_scale[(i + 1) >> 4], __float_as_int(fabsf(v1)));
  }
  __syncthreads();

  const int rb = row / 128;
  const int ri = row % 128;
  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
    float amax = __int_as_float(*(int*)&blk_scale[b]);
    uint8_t ue = f32_to_ue4m3_ceil_local(amax / 6.0f);
    int cb = b / 4;
    int ci = b % 4;
    int out_idx = (rb * n_col_blocks + cb) * 512
        + (ri % 32) * 16 + (ri / 32) * 4 + ci;
    sf_swizzled[out_idx] = ue;
    blk_scale[b] = ue4m3_to_f32_local(ue);
  }
  __syncthreads();

  for (int p = threadIdx.x; p < dim / 2; p += blockDim.x) {
    const int i = p << 1;
    __nv_bfloat162 xv = x2[p];
    const float s0 = static_cast<float>(scale[i]) * s_deq;
    const float s1 = static_cast<float>(scale[i + 1]) * s_deq;
    const float h0 = static_cast<float>(shift[i]) * h_deq;
    const float h1 = static_cast<float>(shift[i + 1]) * h_deq;
    const float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
    const float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
    const float m0 = __bfloat162float(__float2bfloat16(n0 * (1.0f + s0) + h0));
    const float m1 = __bfloat162float(__float2bfloat16(n1 * (1.0f + s1) + h1));
    const int b0 = i >> 4;
    const int b1 = (i + 1) >> 4;
    const float inv0 = blk_scale[b0] > 0.0f ? 1.0f / blk_scale[b0] : 0.0f;
    const float inv1 = blk_scale[b1] > 0.0f ? 1.0f / blk_scale[b1] : 0.0f;
    uint8_t lo = f32_to_fp4_e2m1(m0 * inv0);
    uint8_t hi = f32_to_fp4_e2m1(m1 * inv1);
    row_packed[p] = (hi << 4) | (lo & 0x0F);
  }
}

}  // namespace

void ada_layer_norm_fp8(
    const void*  x_bf16,
    const void*  scale_bf16,
    const void*  shift_bf16,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  // dim must be even for bf162 vectorization (always true for our shapes).
  ada_layer_norm_fp8_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(scale_bf16),
      reinterpret_cast<const __nv_bfloat16*>(shift_bf16),
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      dim, eps);
}

void ada_layer_norm_fp8_modfp8(
    const void*  x_bf16,
    const void*  scale_fp8,
    const void*  shift_fp8,
    const float* scale_deq,
    const float* shift_deq,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  ada_layer_norm_fp8_modfp8_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_fp8_e4m3*>(scale_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(shift_fp8),
      scale_deq, shift_deq,
      reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
      act_scale,
      dim, eps);
}

void awq_ada_layer_norm_fp8(
    const void*  x_bf16,
    const void*  scale_bf16,
    const void*  shift_bf16,
    const void*  inv_s_bf16,
    void*        out_fp8,
    const float* act_scale,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  awq_ada_layer_norm_fp8_kernel
      <<<seq_len, 256, 256 * sizeof(float), stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x_bf16),
          reinterpret_cast<const __nv_bfloat16*>(scale_bf16),
          reinterpret_cast<const __nv_bfloat16*>(shift_bf16),
          reinterpret_cast<const __nv_bfloat16*>(inv_s_bf16),
          reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
          act_scale, dim, eps);
}

void ada_layer_norm_nvfp4_swizzled(
    const void* x_bf16,
    const void* scale_bf16,
    const void* shift_bf16,
    void* packed_u8,
    void* sf_swizzled_u8,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  const int num_blocks = (dim + 15) / 16;
  const int n_col_blocks = (num_blocks + 3) / 4;
  const int smem = (256 + num_blocks) * sizeof(float);
  ada_layer_norm_nvfp4_swizzled_kernel<<<seq_len, 256, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(scale_bf16),
      reinterpret_cast<const __nv_bfloat16*>(shift_bf16),
      reinterpret_cast<uint8_t*>(packed_u8),
      reinterpret_cast<uint8_t*>(sf_swizzled_u8),
      dim, num_blocks, n_col_blocks, eps);
}

void ada_layer_norm_nvfp4_swizzled_modfp8(
    const void* x_bf16,
    const void* scale_fp8,
    const void* shift_fp8,
    const float* scale_deq,
    const float* shift_deq,
    void* packed_u8,
    void* sf_swizzled_u8,
    int seq_len, int dim, float eps,
    cudaStream_t stream)
{
  if (seq_len <= 0 || dim <= 0) return;
  const int num_blocks = (dim + 15) / 16;
  const int n_col_blocks = (num_blocks + 3) / 4;
  const int smem = (256 + num_blocks) * sizeof(float);
  ada_layer_norm_nvfp4_swizzled_modfp8_kernel<<<seq_len, 256, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<const __nv_fp8_e4m3*>(scale_fp8),
      reinterpret_cast<const __nv_fp8_e4m3*>(shift_fp8),
      scale_deq, shift_deq,
      reinterpret_cast<uint8_t*>(packed_u8),
      reinterpret_cast<uint8_t*>(sf_swizzled_u8),
      dim, num_blocks, n_col_blocks, eps);
}

}  // namespace quantize
}  // namespace flash_rt
