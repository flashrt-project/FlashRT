// See pi05_action_edges.cuh.
#include "kernels/pi05/thor/pi05_action_edges.cuh"
#include "kernels/pi05/thor/pi05_pdl.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr int ACT = 32;   // action dim

// x[s, d] = fp16( fp16( sum_k noise[s, k] * w[k, d] ) + b[d] ): the fp16 GEMM output rounding, then the bias add.
__global__ void __launch_bounds__(256)
action_in_kernel(const __half* __restrict__ noise, const __half* __restrict__ w, const __half* __restrict__ b,
                 __half* __restrict__ x, int S, int D) {
  flashrt_pdl_wait_and_trigger();
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= S * D) return;
  const int s = idx / D, d = idx - s * D;
  const __half* nr = noise + s * ACT;
  float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < ACT; ++k) acc = fmaf(__half2float(nr[k]), __half2float(w[k * D + d]), acc);
  const float g = __half2float(__float2half(acc));
  x[idx] = __float2half(g + __half2float(b[d]));
}

// Per row s (one CTA): AdaRMS exactly as adarms_fp16_kernel (rstd over fp32 squares, block reduction),
// xn = fp16(x * rstd * (1 + scale) + shift), gate = style gate; then the action head on the fp16 xn:
// delta[a] = sum_i xn[i] * aow[i, a] (fp32), v = fp16(delta + aob[a]), step = fp16(v * fp16(dt)),
// noise[s, a] = fp16(noise[s, a] + step)  -- the same rounding chain as gmm_fp16_out_fp32 + action_update_from_fp32.
__global__ void __launch_bounds__(256)
adarms_action_out_kernel(const __half* __restrict__ x, const __half* __restrict__ style,
                         __half* __restrict__ xn_out, __half* __restrict__ gate_out,
                         const __half* __restrict__ aow, const __half* __restrict__ aob,
                         __half* __restrict__ noise, int S, int D, float dt) {
  flashrt_pdl_wait_and_trigger();
  const int s = blockIdx.x;
  if (s >= S) return;
  const __half* row = x + static_cast<size_t>(s) * D;
  const __half* sc = style + static_cast<size_t>(s) * 3 * D;
  const __half* sh = sc + D;
  const __half* gt = sh + D;
  __shared__ float shv[8];
  __shared__ float red[8][ACT];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  float ssq = 0.f;
  for (int i = threadIdx.x; i < D; i += 256) { const float v = __half2float(row[i]); ssq += v * v; }
  for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  if (!lane) shv[wid] = ssq;
  __syncthreads();
  if (!wid) {
    ssq = (lane < 8) ? shv[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  }
  __syncthreads();
  if (!threadIdx.x) shv[0] = ssq;
  __syncthreads();
  const float rstd = rsqrtf(shv[0] / D + 1e-6f);

  float part[ACT];
  #pragma unroll
  for (int a = 0; a < ACT; ++a) part[a] = 0.f;
  for (int i = threadIdx.x; i < D; i += 256) {
    const float v = __half2float(row[i]) * rstd;
    const __half xh = __float2half(v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]));
    xn_out[static_cast<size_t>(s) * D + i] = xh;
    gate_out[static_cast<size_t>(s) * D + i] = gt[i];
    const float xf = __half2float(xh);
    const uint4* wr = reinterpret_cast<const uint4*>(aow + static_cast<size_t>(i) * ACT);
    #pragma unroll
    for (int q = 0; q < ACT / 8; ++q) {
      const uint4 w4 = wr[q];
      const __half* wh = reinterpret_cast<const __half*>(&w4);
      #pragma unroll
      for (int e = 0; e < 8; ++e) part[q * 8 + e] = fmaf(xf, __half2float(wh[e]), part[q * 8 + e]);
    }
  }
  #pragma unroll
  for (int a = 0; a < ACT; ++a) {
    float p = part[a];
    for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
    part[a] = p;
  }
  if (!lane) {
    #pragma unroll
    for (int a = 0; a < ACT; ++a) red[wid][a] = part[a];
  }
  __syncthreads();
  if (threadIdx.x < ACT) {
    const int a = threadIdx.x;
    float delta = 0.f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) delta += red[w][a];
    const __half dt_h = __float2half(dt);
    const __half v = __float2half(delta + __half2float(aob[a]));
    const __half step = __float2half(__half2float(v) * __half2float(dt_h));
    __half* n = noise + static_cast<size_t>(s) * ACT + a;
    *n = __float2half(__half2float(*n) + __half2float(step));
  }
}

}  // namespace

int pi05_action_in_fp16(const void* noise, const void* w, const void* b, void* x, int S, int D, int A, cudaStream_t stream) {
  if (A != ACT || S <= 0 || D <= 0) return -1;
  const int total = S * D;
  launch_maybe_pdl(action_in_kernel, dim3((total + 255) / 256), dim3(256), 0, stream,
                   static_cast<const __half*>(noise), static_cast<const __half*>(w), static_cast<const __half*>(b),
                   static_cast<__half*>(x), S, D);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int pi05_adarms_action_out_fp16(const void* x, const void* style, void* xn, void* gate, const void* aow, const void* aob,
                                void* noise, int S, int D, int A, float dt, cudaStream_t stream) {
  if (A != ACT || S <= 0 || D <= 0 || (D & 7) || (reinterpret_cast<uintptr_t>(aow) & 15)) return -1;
  launch_maybe_pdl(adarms_action_out_kernel, dim3(S), dim3(256), 0, stream,
                   static_cast<const __half*>(x), static_cast<const __half*>(style), static_cast<__half*>(xn),
                   static_cast<__half*>(gate), static_cast<const __half*>(aow), static_cast<const __half*>(aob),
                   static_cast<__half*>(noise), S, D, dt);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
