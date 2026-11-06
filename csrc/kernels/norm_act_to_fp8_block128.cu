// ================================================================
// FlashRT — fused {LayerNorm, GELU-tanh} -> FP8 block-128 quantization
//
// Produces the exact activation layout consumed by the per-token /
// per-128-K-block FP8 GEMM (``fp8_block128_gemm_cutlass_sm120_bf16out``):
// FP8 e4m3 values plus an (M, K/128) f32 descale, where each 128-wide
// K-block carries its own scale = amax / 448.
//
// Fusing the activation quantization into the producing elementwise op
// removes the intermediate bf16 round-trip (one global write + one read
// of the M x K activation) and a kernel launch per GEMM input. The math
// matches the standalone ``layer_norm`` / ``gelu_inplace`` +
// ``fp8_per_token_block128_quant_bf16`` chain up to fp32 summation order
// in the row reductions.
//
//   layer_norm_to_fp8_block128_bf16:
//     out[m,i]   = ((x[m,i]-mean_m)*rstd_m*gamma[i] + beta[i]) / scale
//     scale[m,kb] = amax_{i in block kb} |normed| / 448
//   gelu_tanh_to_fp8_block128_bf16:
//     out[m,i]   = gelu_tanh(x[m,i]) / scale     (same per-block scale)
//
// Scheduling: one CUDA block per row, one *warp* per 128-column K-block and
// four consecutive columns per lane (one 8-byte load), so a K-block's amax is
// a pure warp reduction -- no __syncthreads per block, eight times fewer
// memory transactions than the scalar one-element-per-lane form. The K % 128
// == 0 contract of the layout is what makes the vector form exact: a 128-wide
// block is always 32 lanes x 4 columns, and the FP8 output of one K-block is
// 4-byte aligned.
//
// General (model-agnostic): any transformer whose LayerNorm / GELU feeds
// a block-128 FP8 GEMM can reuse these. K must be a multiple of 128.
// ================================================================

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kBlock = 128;          // K-block
constexpr int kThreads = 128;        // threads per CUDA block (4 warps)
constexpr float kFp8Max = 448.0f;
constexpr float kGeluC = 0.7978845608028654f;   // sqrt(2/pi)

// Reduce across the 128-thread (4-warp) CUDA block; the result is
// broadcast to every thread via ``sh[0]``. ``is_max`` selects max vs sum.
__device__ __forceinline__ float block_reduce128(
    float v, float* sh, bool is_max) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    const float t = __shfl_xor_sync(0xffffffff, v, o);
    v = is_max ? fmaxf(v, t) : v + t;
  }
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < 4) ? sh[lane] : (is_max ? 0.0f : 0.0f);
    for (int o = 2; o > 0; o >>= 1) {
      const float t = __shfl_xor_sync(0xffffffff, v, o);
      v = is_max ? fmaxf(v, t) : v + t;
    }
    if (lane == 0) sh[0] = v;
  }
  __syncthreads();
  const float r = sh[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float gelu_tanh(float v) {
  const float t = tanhf(kGeluC * (v + 0.044715f * v * v * v));
  return v * 0.5f * (1.0f + t);
}

// Warp-level max (all lanes see the result), used for the per-K-block amax.
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float4 load4(const __nv_bfloat16* p) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
  return make_float4(__bfloat162float(h[0]), __bfloat162float(h[1]),
                     __bfloat162float(h[2]), __bfloat162float(h[3]));
}

__device__ __forceinline__ uint32_t quant4(const float* v, float inv_scale) {
  __nv_fp8_e4m3 q[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float t = v[j] * inv_scale;
    t = fminf(fmaxf(t, -kFp8Max), kFp8Max);
    q[j] = __nv_fp8_e4m3(t);
  }
  return *reinterpret_cast<const uint32_t*>(q);
}

// One CUDA block (128 threads / 4 warps) per row. LayerNorm over ``dim`` then a
// per-128-K-block FP8 quantization of the normalized row. The normalized value
// is recomputed in the quant pass (cheap bf16 reload) so no per-row scratch is
// needed.
__global__ void layer_norm_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale,
    int dim, float eps) {
  __shared__ float sh[4];
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  const __nv_bfloat16* xr = x + static_cast<long>(m) * dim;
  __nv_fp8_e4m3* orow = out + static_cast<long>(m) * dim;

  float s = 0.0f;
  for (int i = t * 4; i < dim; i += kThreads * 4) {
    const float4 v = load4(xr + i);
    s += (v.x + v.y) + (v.z + v.w);
  }
  const float mean = block_reduce128(s, sh, false) / dim;

  float vsum = 0.0f;
  for (int i = t * 4; i < dim; i += kThreads * 4) {
    const float4 v = load4(xr + i);
    const float d0 = v.x - mean, d1 = v.y - mean, d2 = v.z - mean, d3 = v.w - mean;
    vsum += (d0 * d0 + d1 * d1) + (d2 * d2 + d3 * d3);
  }
  const float rstd = rsqrtf(block_reduce128(vsum, sh, false) / dim + eps);

  const int n_kb = dim / kBlock;
  for (int kb = warp; kb < n_kb; kb += kThreads / 32) {
    const int i = kb * kBlock + lane * 4;
    const float4 xv = load4(xr + i);
    const float4 g = load4(gamma + i);
    const float4 b = load4(beta + i);
    float n[4] = {
        (xv.x - mean) * rstd * g.x + b.x,
        (xv.y - mean) * rstd * g.y + b.y,
        (xv.z - mean) * rstd * g.z + b.z,
        (xv.w - mean) * rstd * g.w + b.w,
    };
    const float amax = warp_max(fmaxf(fmaxf(fabsf(n[0]), fabsf(n[1])),
                                      fmaxf(fabsf(n[2]), fabsf(n[3]))));
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (lane == 0) scale[m * n_kb + kb] = sc;
    *reinterpret_cast<uint32_t*>(orow + i) = quant4(n, 1.0f / sc);
  }
}

// Plus optional per-column bias: gelu(x + bias), then the block-128 quant.
__global__ void gelu_tanh_bias_to_fp8_block128_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale,
    int dim) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const __nv_bfloat16* xr = x + static_cast<long>(m) * dim;
  __nv_fp8_e4m3* orow = out + static_cast<long>(m) * dim;

  const int n_kb = dim / kBlock;
  for (int kb = warp; kb < n_kb; kb += kThreads / 32) {
    const int i = kb * kBlock + lane * 4;
    const float4 xv = load4(xr + i);
    float g[4];
    if (bias != nullptr) {
      const float4 b = load4(bias + i);
      g[0] = gelu_tanh(xv.x + b.x);
      g[1] = gelu_tanh(xv.y + b.y);
      g[2] = gelu_tanh(xv.z + b.z);
      g[3] = gelu_tanh(xv.w + b.w);
    } else {
      g[0] = gelu_tanh(xv.x);
      g[1] = gelu_tanh(xv.y);
      g[2] = gelu_tanh(xv.z);
      g[3] = gelu_tanh(xv.w);
    }
    const float amax = warp_max(fmaxf(fmaxf(fabsf(g[0]), fabsf(g[1])),
                                      fmaxf(fabsf(g[2]), fabsf(g[3]))));
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (lane == 0) scale[m * n_kb + kb] = sc;
    *reinterpret_cast<uint32_t*>(orow + i) = quant4(g, 1.0f / sc);
  }
}


// ── Legacy (pre-vectorization) schedule ────────────────────────────────
// Kept for every caller that has not opted into the model fast-path policy:
// one 128-thread block per row, scalar loads. This is the upstream
// implementation, byte-for-byte; FLASHRT_* policy off (the default for
// non-VLA models) selects it.
__global__ void layer_norm_to_fp8_block128_legacy_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale,
    int dim, float eps) {
  __shared__ float sh[4];
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const __nv_bfloat16* xr = x + static_cast<long>(m) * dim;
  __nv_fp8_e4m3* orow = out + static_cast<long>(m) * dim;

  float s = 0.0f;
  for (int i = t; i < dim; i += kBlock) s += __bfloat162float(xr[i]);
  const float mean = block_reduce128(s, sh, false) / dim;

  float vsum = 0.0f;
  for (int i = t; i < dim; i += kBlock) {
    const float d = __bfloat162float(xr[i]) - mean;
    vsum += d * d;
  }
  const float rstd = rsqrtf(block_reduce128(vsum, sh, false) / dim + eps);

  const int n_kb = dim / kBlock;
  for (int kb = 0; kb < n_kb; ++kb) {
    const int i = kb * kBlock + t;
    const float normed =
        (__bfloat162float(xr[i]) - mean) * rstd *
            __bfloat162float(gamma[i]) +
        __bfloat162float(beta[i]);
    const float amax = block_reduce128(fabsf(normed), sh, true);
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (t == 0) scale[m * n_kb + kb] = sc;
    float q = normed / sc;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    orow[i] = __nv_fp8_e4m3(q);
  }
}

__global__ void gelu_tanh_to_fp8_block128_legacy_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale,
    int dim) {
  __shared__ float sh[4];
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const __nv_bfloat16* xr = x + static_cast<long>(m) * dim;
  __nv_fp8_e4m3* orow = out + static_cast<long>(m) * dim;

  const int n_kb = dim / kBlock;
  for (int kb = 0; kb < n_kb; ++kb) {
    const int i = kb * kBlock + t;
    const float g = gelu_tanh(__bfloat162float(xr[i]));
    const float amax = block_reduce128(fabsf(g), sh, true);
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (t == 0) scale[m * n_kb + kb] = sc;
    float q = g / sc;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    orow[i] = __nv_fp8_e4m3(q);
  }
}

__global__ void gelu_tanh_bias_to_fp8_block128_legacy_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale,
    int dim) {
  __shared__ float sh[4];
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const __nv_bfloat16* xr = x + static_cast<long>(m) * dim;
  __nv_fp8_e4m3* orow = out + static_cast<long>(m) * dim;

  const int n_kb = dim / kBlock;
  for (int kb = 0; kb < n_kb; ++kb) {
    const int i = kb * kBlock + t;
    const float g = gelu_tanh(
        __bfloat162float(xr[i]) + __bfloat162float(bias[i]));
    const float amax = block_reduce128(fabsf(g), sh, true);
    const float sc = fmaxf(amax / kFp8Max, 1.0e-12f);
    if (t == 0) scale[m * n_kb + kb] = sc;
    float q = g / sc;
    q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
    orow[i] = __nv_fp8_e4m3(q);
  }
}

}  // namespace

// Original entry points: the upstream scalar schedule. Every existing model
// keeps calling these, so their behaviour is unchanged.
void layer_norm_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_to_fp8_block128_legacy_kernel<<<rows, kBlock, 0, stream>>>(
      x, gamma, beta, out, scale, dim, eps);
}

void gelu_tanh_to_fp8_block128_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, cudaStream_t stream) {
  gelu_tanh_to_fp8_block128_legacy_kernel<<<rows, kBlock, 0, stream>>>(
      x, out, scale, dim);
}

void gelu_tanh_bias_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* out, float* scale, int rows, int dim,
    cudaStream_t stream) {
  gelu_tanh_bias_to_fp8_block128_legacy_kernel<<<rows, kBlock, 0, stream>>>(
      x, bias, out, scale, dim);
}

// Vectorized schedule, opt-in by model-local call site (VLA-JEPA's tower).
// Additive symbols: no existing binding or caller changes.
void layer_norm_to_fp8_block128_bf16_v2(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_to_fp8_block128_kernel<<<rows, kThreads, 0, stream>>>(
      x, gamma, beta, out, scale, dim, eps);
}

void gelu_tanh_to_fp8_block128_bf16_v2(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, cudaStream_t stream) {
  gelu_tanh_bias_to_fp8_block128_kernel<<<rows, kThreads, 0, stream>>>(
      x, nullptr, out, scale, dim);
}

void gelu_tanh_bias_to_fp8_block128_bf16_v2(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* out, float* scale, int rows, int dim,
    cudaStream_t stream) {
  gelu_tanh_bias_to_fp8_block128_kernel<<<rows, kThreads, 0, stream>>>(
      x, bias, out, scale, dim);
}

}  // namespace kernels
}  // namespace flash_rt
