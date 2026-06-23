// ================================================================
// FlashRT — fused bias epilogues (bf16)
//
// When a GEMM's bias cannot ride its own epilogue (e.g. a shared
// block-scaled FP8/FP4 GEMM with a fixed epilogue), the per-column bias
// add would otherwise be a standalone kernel that re-reads and re-writes
// the whole GEMM output. These kernels fold that bias add into the op
// that consumes the GEMM output, so the bias never round-trips HBM:
//
//   residual_add_bias_bf16 : residual += (x + bias)   (proj / fc2 -> res)
//   qkv_split_bias_bf16    : split (x + bias) into q/k/v   (qkv -> attn)
//
// ``bias`` is broadcast over rows. General (model-agnostic).
// ================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

__global__ void residual_add_bias_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int dim, long total) {
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int i = idx % dim;
    residual[idx] = __float2bfloat16(
        __bfloat162float(residual[idx]) + __bfloat162float(x[idx]) +
        __bfloat162float(bias[i]));
  }
}

// qkv = (rows, Hq+Hk+Hv); split column-wise into q/k/v with the matching
// bias slice added. q/k/v are (rows, H*) row-major.
__global__ void qkv_split_bias_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int Hq, int Hk, int Hv, long total) {
  const int W = Hq + Hk + Hv;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int m = idx / W;
    const int col = idx - static_cast<long>(m) * W;
    const float val =
        __bfloat162float(qkv[idx]) + __bfloat162float(bias[col]);
    if (col < Hq) {
      q[static_cast<long>(m) * Hq + col] = __float2bfloat16(val);
    } else if (col < Hq + Hk) {
      k[static_cast<long>(m) * Hk + (col - Hq)] = __float2bfloat16(val);
    } else {
      v[static_cast<long>(m) * Hv + (col - Hq - Hk)] = __float2bfloat16(val);
    }
  }
}

int grid_for(long total) {
  const long b = (total + 255) / 256;
  return static_cast<int>(b < 65535 ? (b < 1 ? 1 : b) : 65535);
}

}  // namespace

void residual_add_bias_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x,
    const __nv_bfloat16* bias, int rows, int dim, cudaStream_t stream) {
  const long total = static_cast<long>(rows) * dim;
  residual_add_bias_kernel<<<grid_for(total), 256, 0, stream>>>(
      residual, x, bias, dim, total);
}

void qkv_split_bias_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int rows, int hq, int hk, int hv, cudaStream_t stream) {
  const long total = static_cast<long>(rows) * (hq + hk + hv);
  qkv_split_bias_kernel<<<grid_for(total), 256, 0, stream>>>(
      qkv, bias, q, k, v, hq, hk, hv, total);
}

}  // namespace kernels
}  // namespace flash_rt
