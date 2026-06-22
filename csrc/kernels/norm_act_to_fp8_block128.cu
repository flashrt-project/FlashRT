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
// ``fp8_per_token_block128_quant_bf16`` chain bit-for-bit.
//
//   layer_norm_to_fp8_block128_bf16:
//     out[m,i]   = ((x[m,i]-mean_m)*rstd_m*gamma[i] + beta[i]) / scale
//     scale[m,kb] = amax_{i in block kb} |normed| / 448
//   gelu_tanh_to_fp8_block128_bf16:
//     out[m,i]   = gelu_tanh(x[m,i]) / scale     (same per-block scale)
//
// General (model-agnostic): any transformer whose LayerNorm / GELU feeds
// a block-128 FP8 GEMM can reuse these. K must be a multiple of 128.
// ================================================================

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kBlock = 128;          // K-block (and threads per CUDA block)
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

// One CUDA block (128 threads) per row. LayerNorm over ``dim`` then a
// per-128-K-block FP8 quantization of the normalized row. The normalized
// value is recomputed in the quant pass (cheap bf16 reload) so no
// per-row scratch is needed.
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

// One CUDA block (128 threads) per row. GELU-tanh then a per-128-K-block
// FP8 quantization. No cross-row reduction is needed; only the per-block
// amax.
__global__ void gelu_tanh_to_fp8_block128_kernel(
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

}  // namespace

void layer_norm_to_fp8_block128_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, float eps, cudaStream_t stream) {
  layer_norm_to_fp8_block128_kernel<<<rows, kBlock, 0, stream>>>(
      x, gamma, beta, out, scale, dim, eps);
}

void gelu_tanh_to_fp8_block128_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, float* scale,
    int rows, int dim, cudaStream_t stream) {
  gelu_tanh_to_fp8_block128_kernel<<<rows, kBlock, 0, stream>>>(
      x, out, scale, dim);
}

}  // namespace kernels
}  // namespace flash_rt
