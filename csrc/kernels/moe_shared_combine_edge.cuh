#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// The tail of a fine-grained MoE layer: gate the shared expert and add it to
// the routed sum.
//
//   out = routed + shared * sigmoid(gate_logit[row])
//
// Replaces a sigmoid, a broadcast multiply, an add and a cast -- four tensor
// ops and two full (S, hidden) fp32 intermediates per layer.
//
// The existing bf16 gate-mul-residual kernel does not fit here: the routed sum
// arrives in fp32 from the weighted reduction, and taking it through bf16 to
// reach that kernel would change the accumulation rather than merely fuse it.
// This keeps the arithmetic in fp32 and rounds once, at the store.
//
//   routed  (S, dim) fp32
//   shared  (S, dim) bf16
//   gate    (S,) bf16, the raw gate logit -- the sigmoid is applied here
//   out     (S, dim) bf16
void moe_shared_gate_combine_edge_bf16(
    const void* routed,
    const void* shared,
    const void* gate,
    void*       out,
    int S,
    int dim,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
