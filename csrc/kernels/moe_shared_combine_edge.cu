#include "moe_shared_combine_edge.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

__global__ void moe_shared_gate_combine_kernel(
    const float* __restrict__ routed,
    const __nv_bfloat16* __restrict__ shared,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int S,
    int dim)
{
  const int row = blockIdx.x;
  if (row >= S) return;
  // expf, not the fast intrinsic: routing downstream is discrete, and a gate
  // that lands a few ulp away flips ties in later layers.
  const float g = 1.0f / (1.0f + expf(-static_cast<float>(gate[row])));
  const size_t base = static_cast<size_t>(row) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    // Multiply and add as two rounded operations, not one contracted fma.
    // Written as `routed + shared * g` the compiler contracts it, which is one
    // rounding instead of two and therefore a different number -- measured, one
    // element in 16384 by one ulp, where the routed sum is small against the
    // gated shared term. That is a fine trade in isolation and the wrong one
    // here: the decode step computes this as separate tensor ops, and this
    // kernel is only allowed to stand in for it if it lands on the same bits.
    out[base + i] = __float2bfloat16(__fadd_rn(
        routed[base + i],
        __fmul_rn(static_cast<float>(shared[base + i]), g)));
  }
}

}  // namespace

void moe_shared_gate_combine_edge_bf16(
    const void* routed,
    const void* shared,
    const void* gate,
    void*       out,
    int S,
    int dim,
    cudaStream_t stream)
{
  if (S <= 0 || dim <= 0) return;
  const int threads = dim < 256 ? 128 : 256;
  moe_shared_gate_combine_kernel<<<S, threads, 0, stream>>>(
      reinterpret_cast<const float*>(routed),
      reinterpret_cast<const __nv_bfloat16*>(shared),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<__nv_bfloat16*>(out), S, dim);
}

}  // namespace kernels
}  // namespace flash_rt
