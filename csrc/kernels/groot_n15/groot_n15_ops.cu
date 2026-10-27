// ================================================================
// FlashRT — GR00T N1.5 BF16 norms and elementwise ops.
// See groot_n15_ops.cuh for the contract. Every rounding is spelled out
// with an _rn intrinsic so the compiler cannot contract or reorder it.
// ================================================================

#include "groot_n15_ops.cuh"

#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace flash_rt::groot_n15 {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxPerThread = 8;   // dim <= 2048
constexpr int kMaxDim = kThreads * kMaxPerThread;

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 to_bf(const float v) { return __float2bfloat16_rn(v); }

// Sum over the block; every thread receives the total.
__device__ __forceinline__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float t = (lane < (kThreads >> 5)) ? smem[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  __syncthreads();
  return t;
}

// Mean and reciprocal standard deviation of the kept row values.
__device__ __forceinline__ void row_stats(const float* v, int dim, float eps,
                                          float* smem, float& mean, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    if (i < dim) s = __fadd_rn(s, v[k]);
  }
  mean = __fdiv_rn(block_sum(s, smem), static_cast<float>(dim));
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    if (i < dim) {
      const float d = __fsub_rn(v[k], mean);
      q = __fmaf_rn(d, d, q);
    }
  }
  const float var = __fdiv_rn(block_sum(q, smem), static_cast<float>(dim));
  rstd = rsqrtf(__fadd_rn(var, eps));
}

// torch's LayerNorm element: gamma * (rstd * (x - mean)) + beta.
__device__ __forceinline__ float ln_affine(float x, float mean, float rstd, float g, float b) {
  return __fmaf_rn(g, __fmul_rn(rstd, __fsub_rn(x, mean)), b);
}
__device__ __forceinline__ float ln_plain(float x, float mean, float rstd) {
  return __fmul_rn(rstd, __fsub_rn(x, mean));
}

// mode: 0 = affine (gamma/beta), 1 = plain, 2 = ada (scale1/shift after plain)
template <int kMode>
__global__ void layer_norm_bf16_kernel(const __nv_bfloat16* __restrict__ res_in,
                                       const __nv_bfloat16* __restrict__ delta,
                                       __nv_bfloat16* __restrict__ res_out,
                                       const __nv_bfloat16* __restrict__ p0,
                                       const __nv_bfloat16* __restrict__ p1,
                                       __nv_bfloat16* __restrict__ out,
                                       int dim, float eps) {
  __shared__ float smem[kThreads / 32];
  const long long base = static_cast<long long>(blockIdx.x) * dim;
  float v[kMaxPerThread];
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    v[k] = 0.f;
    if (i < dim) {
      float r = bf(res_in[base + i]);
      if (delta) {
        const __nv_bfloat16 rb = to_bf(__fadd_rn(r, bf(delta[base + i])));
        if (res_out) res_out[base + i] = rb;
        r = bf(rb);
      } else if (res_out) {
        res_out[base + i] = res_in[base + i];
      }
      v[k] = r;
    }
  }
  float mean, rstd;
  row_stats(v, dim, eps, smem, mean, rstd);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    if (i >= dim) continue;
    float y;
    if constexpr (kMode == 0) {
      y = ln_affine(v[k], mean, rstd, bf(p0[i]), bf(p1[i]));
    } else if constexpr (kMode == 1) {
      y = ln_plain(v[k], mean, rstd);
    } else {
      y = __fadd_rn(__fmul_rn(ln_plain(v[k], mean, rstd), bf(p0[i])), bf(p1[i]));
    }
    out[base + i] = to_bf(y);
  }
}

__global__ void layer_norm_to_f32_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ gamma,
                                         const __nv_bfloat16* __restrict__ beta,
                                         float* __restrict__ out32, int dim, float eps) {
  __shared__ float smem[kThreads / 32];
  const long long base = static_cast<long long>(blockIdx.x) * dim;
  float v[kMaxPerThread];
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    v[k] = (i < dim) ? bf(x[base + i]) : 0.f;
  }
  float mean, rstd;
  row_stats(v, dim, eps, smem, mean, rstd);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    if (i < dim) out32[base + i] = ln_affine(v[k], mean, rstd, bf(gamma[i]), bf(beta[i]));
  }
}

__global__ void residual_layer_norm_f32_kernel(float* __restrict__ res32,
                                               const __nv_bfloat16* __restrict__ delta,
                                               const __nv_bfloat16* __restrict__ gamma,
                                               const __nv_bfloat16* __restrict__ beta,
                                               __nv_bfloat16* __restrict__ out,
                                               int dim, float eps) {
  __shared__ float smem[kThreads / 32];
  const long long base = static_cast<long long>(blockIdx.x) * dim;
  float v[kMaxPerThread];
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    v[k] = 0.f;
    if (i < dim) {
      float r = res32[base + i];
      if (delta) {
        r = __fadd_rn(bf(delta[base + i]), r);
        res32[base + i] = r;
      }
      v[k] = r;
    }
  }
  float mean, rstd;
  row_stats(v, dim, eps, smem, mean, rstd);
#pragma unroll
  for (int k = 0; k < kMaxPerThread; ++k) {
    const int i = threadIdx.x + k * kThreads;
    if (i < dim) out[base + i] = to_bf(ln_affine(v[k], mean, rstd, bf(gamma[i]), bf(beta[i])));
  }
}

__global__ void residual_add_f32_to_bf16_kernel(const float* __restrict__ res32,
                                                const __nv_bfloat16* __restrict__ delta,
                                                __nv_bfloat16* __restrict__ out, long long n) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    out[i] = to_bf(__fadd_rn(bf(delta[i]), res32[i]));
  }
}

__global__ void add_rows_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ bias,
                                     const __nv_bfloat16* __restrict__ table,
                                     __nv_bfloat16* __restrict__ out,
                                     int rows, int dim, int period) {
  const long long n = static_cast<long long>(rows) * dim;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i % dim);
    const int row = static_cast<int>(i / dim);
    float t = bf(x[i]);
    if (bias) t = bf(to_bf(__fadd_rn(t, bf(bias[col]))));
    out[i] = to_bf(__fadd_rn(t, bf(table[static_cast<long long>(row % period) * dim + col])));
  }
}

template <int kAct>
__global__ void bias_act_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ bias,
                                     __nv_bfloat16* __restrict__ out, int dim, long long n) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const __nv_bfloat16 t = to_bf(__fadd_rn(bf(x[i]), bf(bias[i % dim])));
    if constexpr (kAct == kBiasOnly) {
      out[i] = t;
    } else if constexpr (kAct == kBiasRelu) {
      out[i] = bf(t) > 0.f ? t : to_bf(0.f);
    } else {
      // torch sigmoid in fp32 opmath: 1 / (1 + exp(-x)), rounded to bf16.
      const float tf = bf(t);
      const __nv_bfloat16 s = to_bf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-tf))));
      out[i] = to_bf(__fmul_rn(tf, bf(s)));
    }
  }
}

__global__ void concat_bias_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ bias,
                                        const __nv_bfloat16* __restrict__ tail,
                                        __nv_bfloat16* __restrict__ out,
                                        int rows, int dim, int tail_dim) {
  const int width = dim + tail_dim;
  const long long n = static_cast<long long>(rows) * width;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i % width);
    const long long row = i / width;
    out[i] = col < dim ? to_bf(__fadd_rn(bf(x[row * dim + col]), bf(bias[col])))
                       : tail[row * tail_dim + (col - dim)];
  }
}

__global__ void gelu_tanh_bf16_kernel(__nv_bfloat16* __restrict__ x, long long n) {
  // torch GeluCUDAKernelImpl (approximate="tanh"), opmath float.
  constexpr float kBeta = static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
  constexpr float kKappa = 0.044715f;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const float v = bf(x[i]);
    const float cube = __fmul_rn(__fmul_rn(v, v), v);
    const float inner = __fmul_rn(kBeta, __fmaf_rn(kKappa, cube, v));
    x[i] = to_bf(__fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.0f, tanhf(inner))));
  }
}

__global__ void quant_rows_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                      uint8_t* __restrict__ out, float* __restrict__ scales,
                                      int K) {
  __shared__ float smem[kThreads / 32];
  const long long base = static_cast<long long>(blockIdx.x) * K;
  float m = 0.f;
  for (int k = threadIdx.x; k < K; k += kThreads) m = fmaxf(m, fabsf(bf(x[base + k])));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = m;
  __syncthreads();
  float t = (lane < (kThreads >> 5)) ? smem[lane] : 0.f;
  for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
  const float amax = fmaxf(t, 1e-12f);
  const float scale = amax / 448.0f;
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
  for (int k = threadIdx.x; k < K; k += kThreads) {
    out[base + k] = static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(__fdiv_rn(bf(x[base + k]), scale), __NV_SATFINITE, __NV_E4M3));
  }
}

__global__ void rescale_rows_bf16_kernel(__nv_bfloat16* __restrict__ d,
                                         const float* __restrict__ row_scale,
                                         const float* __restrict__ col_scale,
                                         const __nv_bfloat16* __restrict__ bias,
                                         int N, long long n) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i % N);
    float v = __fmul_rn(bf(d[i]), __fmul_rn(row_scale[i / N], col_scale[col]));
    if (bias) v = __fadd_rn(v, bf(bias[col]));
    d[i] = to_bf(v);
  }
}

__global__ void actions_init_kernel(const float* __restrict__ noise32, float* __restrict__ act32,
                                    __nv_bfloat16* __restrict__ act_bf16, long long n) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const float v = noise32[i];
    act32[i] = v;
    act_bf16[i] = to_bf(v);
  }
}

__global__ void euler_step_kernel(const __nv_bfloat16* __restrict__ pred, float* __restrict__ act32,
                                  __nv_bfloat16* __restrict__ act_bf16, float dt, long long n) {
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const float step = bf(to_bf(__fmul_rn(bf(pred[i]), dt)));
    const float v = __fadd_rn(act32[i], step);
    act32[i] = v;
    act_bf16[i] = to_bf(v);
  }
}

void check_launch(const char* who) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(who) + ": launch failed: " + cudaGetErrorString(e));
  }
}

int grid_for(long long n) {
  long long blocks = (n + kThreads - 1) / kThreads;
  if (blocks > 4096) blocks = 4096;
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

void check_rows(const char* who, int rows, int dim) {
  if (dim <= 0 || dim > kMaxDim) {
    throw std::runtime_error(std::string(who) + ": dim must be in [1, " +
                             std::to_string(kMaxDim) + "], got " + std::to_string(dim));
  }
  if (rows < 0) throw std::runtime_error(std::string(who) + ": rows must be >= 0");
}

}  // namespace

void layer_norm_bf16(const __nv_bfloat16* res_in, const __nv_bfloat16* delta,
                     __nv_bfloat16* res_out, const __nv_bfloat16* gamma,
                     const __nv_bfloat16* beta, __nv_bfloat16* out,
                     int rows, int dim, float eps, cudaStream_t stream) {
  check_rows("groot_n15 layer_norm_bf16", rows, dim);
  if (rows == 0) return;
  if (!res_in || !out) throw std::runtime_error("groot_n15 layer_norm_bf16: null input or output");
  if ((gamma == nullptr) != (beta == nullptr)) {
    throw std::runtime_error("groot_n15 layer_norm_bf16: pass both gamma and beta, or neither");
  }
  if (gamma) {
    layer_norm_bf16_kernel<0><<<rows, kThreads, 0, stream>>>(res_in, delta, res_out, gamma, beta,
                                                             out, dim, eps);
    check_launch("groot_n15 layer_norm_bf16");
  } else {
    layer_norm_bf16_kernel<1><<<rows, kThreads, 0, stream>>>(res_in, delta, res_out, nullptr,
                                                             nullptr, out, dim, eps);
    check_launch("groot_n15 layer_norm_bf16");
  }
}

void layer_norm_to_f32(const __nv_bfloat16* x, const __nv_bfloat16* gamma,
                       const __nv_bfloat16* beta, float* out32,
                       int rows, int dim, float eps, cudaStream_t stream) {
  check_rows("groot_n15 layer_norm_to_f32", rows, dim);
  if (rows == 0) return;
  if (!x || !gamma || !beta || !out32) {
    throw std::runtime_error("groot_n15 layer_norm_to_f32: null pointer argument");
  }
  layer_norm_to_f32_kernel<<<rows, kThreads, 0, stream>>>(x, gamma, beta, out32, dim, eps);
  check_launch("groot_n15 layer_norm_to_f32");
}

void residual_layer_norm_f32(float* res32, const __nv_bfloat16* delta,
                             const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
                             __nv_bfloat16* out, int rows, int dim, float eps,
                             cudaStream_t stream) {
  check_rows("groot_n15 residual_layer_norm_f32", rows, dim);
  if (rows == 0) return;
  if (!res32 || !gamma || !beta || !out) {
    throw std::runtime_error("groot_n15 residual_layer_norm_f32: null pointer argument");
  }
  residual_layer_norm_f32_kernel<<<rows, kThreads, 0, stream>>>(res32, delta, gamma, beta, out,
                                                                dim, eps);
  check_launch("groot_n15 residual_layer_norm_f32");
}

void residual_add_f32_to_bf16(const float* res32, const __nv_bfloat16* delta,
                              __nv_bfloat16* out, long long n, cudaStream_t stream) {
  if (n <= 0) return;
  if (!res32 || !delta || !out) {
    throw std::runtime_error("groot_n15 residual_add_f32_to_bf16: null pointer argument");
  }
  residual_add_f32_to_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(res32, delta, out, n);
  check_launch("groot_n15 residual_add_f32_to_bf16");
}

void ada_layer_norm_bf16(const __nv_bfloat16* res_in, const __nv_bfloat16* delta,
                         __nv_bfloat16* res_out, const __nv_bfloat16* scale1,
                         const __nv_bfloat16* shift, __nv_bfloat16* out,
                         int rows, int dim, float eps, cudaStream_t stream) {
  check_rows("groot_n15 ada_layer_norm_bf16", rows, dim);
  if (rows == 0) return;
  if (!res_in || !scale1 || !shift || !out) {
    throw std::runtime_error("groot_n15 ada_layer_norm_bf16: null pointer argument");
  }
  layer_norm_bf16_kernel<2><<<rows, kThreads, 0, stream>>>(res_in, delta, res_out, scale1, shift,
                                                           out, dim, eps);
  check_launch("groot_n15 ada_layer_norm_bf16");
}

void add_rows_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                   const __nv_bfloat16* table, __nv_bfloat16* out,
                   int rows, int dim, int period, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if (!x || !table || !out || period <= 0) {
    throw std::runtime_error("groot_n15 add_rows_bf16: null pointer or period <= 0");
  }
  add_rows_bf16_kernel<<<grid_for(static_cast<long long>(rows) * dim), kThreads, 0, stream>>>(
      x, bias, table, out, rows, dim, period);
  check_launch("groot_n15 add_rows_bf16");
}

void bias_act_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                   __nv_bfloat16* out, int rows, int dim, int act,
                   cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if (!x || !bias || !out) throw std::runtime_error("groot_n15 bias_act_bf16: null pointer argument");
  const long long n = static_cast<long long>(rows) * dim;
  switch (act) {
    case kBiasOnly:
      bias_act_bf16_kernel<kBiasOnly><<<grid_for(n), kThreads, 0, stream>>>(x, bias, out, dim, n);
      check_launch("groot_n15 bias_act_bf16");
      break;
    case kBiasRelu:
      bias_act_bf16_kernel<kBiasRelu><<<grid_for(n), kThreads, 0, stream>>>(x, bias, out, dim, n);
      check_launch("groot_n15 bias_act_bf16");
      break;
    case kBiasSwish:
      bias_act_bf16_kernel<kBiasSwish><<<grid_for(n), kThreads, 0, stream>>>(x, bias, out, dim, n);
      check_launch("groot_n15 bias_act_bf16");
      break;
    default:
      throw std::runtime_error("groot_n15 bias_act_bf16: act must be 0 (none), 1 (relu) or 2 (swish)");
  }
}

void concat_bias_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                      const __nv_bfloat16* tail, __nv_bfloat16* out,
                      int rows, int dim, int tail_dim, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0 || tail_dim < 0) return;
  if (!x || !bias || !out || (tail_dim > 0 && !tail)) {
    throw std::runtime_error("groot_n15 concat_bias_bf16: null pointer argument");
  }
  const long long n = static_cast<long long>(rows) * (dim + tail_dim);
  concat_bias_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, bias, tail, out, rows, dim,
                                                               tail_dim);
  check_launch("groot_n15 concat_bias_bf16");
}

void gelu_tanh_bf16(__nv_bfloat16* x, long long n, cudaStream_t stream) {
  if (n <= 0) return;
  if (!x) throw std::runtime_error("groot_n15 gelu_tanh_bf16: null pointer argument");
  gelu_tanh_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, n);
  check_launch("groot_n15 gelu_tanh_bf16");
}

void quant_rows_fp8(const __nv_bfloat16* x, uint8_t* out, float* scales,
                    int M, int K, cudaStream_t stream) {
  if (M <= 0 || K <= 0) return;
  if (!x || !out || !scales) throw std::runtime_error("groot_n15 quant_rows_fp8: null pointer argument");
  quant_rows_fp8_kernel<<<M, kThreads, 0, stream>>>(x, out, scales, K);
  check_launch("groot_n15 quant_rows_fp8");
}

void rescale_rows_bf16(__nv_bfloat16* d, const float* row_scale, const float* col_scale,
                       const __nv_bfloat16* bias, int M, int N, cudaStream_t stream) {
  if (M <= 0 || N <= 0) return;
  if (!d || !row_scale || !col_scale) {
    throw std::runtime_error("groot_n15 rescale_rows_bf16: null pointer argument");
  }
  const long long n = static_cast<long long>(M) * N;
  rescale_rows_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(d, row_scale, col_scale, bias,
                                                                 N, n);
  check_launch("groot_n15 rescale_rows_bf16");
}

void actions_init(const float* noise32, float* act32, __nv_bfloat16* act_bf16,
                  long long n, cudaStream_t stream) {
  if (n <= 0) return;
  if (!noise32 || !act32 || !act_bf16) {
    throw std::runtime_error("groot_n15 actions_init: null pointer argument");
  }
  actions_init_kernel<<<grid_for(n), kThreads, 0, stream>>>(noise32, act32, act_bf16, n);
  check_launch("groot_n15 actions_init");
}

void euler_step(const __nv_bfloat16* pred, float* act32, __nv_bfloat16* act_bf16,
                float dt, long long n, cudaStream_t stream) {
  if (n <= 0) return;
  if (!pred || !act32 || !act_bf16) {
    throw std::runtime_error("groot_n15 euler_step: null pointer argument");
  }
  euler_step_kernel<<<grid_for(n), kThreads, 0, stream>>>(pred, act32, act_bf16, dt, n);
  check_launch("groot_n15 euler_step");
}

}  // namespace flash_rt::groot_n15
