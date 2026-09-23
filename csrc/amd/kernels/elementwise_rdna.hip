// ================================================================
// FlashRT AMD -- BF16 residual kernels (RDNA 3.5, wave32)
//
// Implements the Pi0.5 residual update in either form:
//   out[i] = residual[i] + update[i]
//   out[i] = residual[i] + update[i] * gate[i % width]
//
// The production path uses packed BF16x2 loads/stores. Arithmetic is promoted
// to FP32 and rounded once to BF16 at the output, matching the PyTorch tensor
// expression. The HasGate template removes the modulo, gate load, and branch
// completely for ordinary residual additions. No LDS or synchronization is
// required; each thread owns one independent pair.
//
// A scalar fallback handles odd tensor sizes or widths so the public raw-
// pointer entry keeps a general contract even though Pi0.5 shapes are even.
// ================================================================

#include "common_hip_rdna.h"

namespace {

// Tail-safe fallback for shapes that cannot be traversed as BF16 pairs.
__global__ void residual_scalar_rdna_kernel(
    const __hip_bfloat16 *__restrict__ update,
    const __hip_bfloat16 *__restrict__ residual,
    const __hip_bfloat16 *__restrict__ gate,
    __hip_bfloat16 *__restrict__ output, int size, int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  float value = __bfloat162float(update[index]);
  if (gate != nullptr) {
    value *= __bfloat162float(gate[index % width]);
  }
  output[index] = __float2bfloat16(
      value + __bfloat162float(residual[index]));
}

// Production vector path. Gate is only dereferenced in the HasGate=true
// specialization, so callers may pass nullptr for an ungated residual.
template <bool HasGate>
__global__ void residual_rdna_kernel(
    const __hip_bfloat162 *__restrict__ update,
    const __hip_bfloat162 *__restrict__ residual,
    const __hip_bfloat162 *__restrict__ gate,
    __hip_bfloat162 *__restrict__ output, int pairs, int width_pairs) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pairs) {
    return;
  }
  const __hip_bfloat162 update_value = update[pair];
  const __hip_bfloat162 residual_value = residual[pair];
  float first = __bfloat162float(update_value.x);
  float second = __bfloat162float(update_value.y);
  if constexpr (HasGate) {
    const __hip_bfloat162 gate_value = gate[pair % width_pairs];
    first *= __bfloat162float(gate_value.x);
    second *= __bfloat162float(gate_value.y);
  }
  first += __bfloat162float(residual_value.x);
  second += __bfloat162float(residual_value.y);
  output[pair] = __hip_bfloat162(
      __float2bfloat16(first), __float2bfloat16(second));
}

} // namespace

// ── Host dispatch ──

void residual_rdna(const __hip_bfloat16 *update, const __hip_bfloat16 *residual,
                   const __hip_bfloat16 *gate, __hip_bfloat16 *output, int size,
                   int width, hipStream_t stream) {
  constexpr int threads = 256;
  if ((size & 1) != 0 || (width & 1) != 0) {
    residual_scalar_rdna_kernel
        <<<(size + threads - 1) / threads, threads, 0, stream>>>(
            update, residual, gate, output, size, width);
    return;
  }
  const int pairs = size / 2;
  const int width_pairs = width / 2;
  const auto *update2 = reinterpret_cast<const __hip_bfloat162 *>(update);
  const auto *residual2 = reinterpret_cast<const __hip_bfloat162 *>(residual);
  const auto *gate2 = reinterpret_cast<const __hip_bfloat162 *>(gate);
  auto *output2 = reinterpret_cast<__hip_bfloat162 *>(output);
  if (gate == nullptr) {
    residual_rdna_kernel<false>
        <<<(pairs + threads - 1) / threads, threads, 0, stream>>>(
            update2, residual2, gate2, output2, pairs, width_pairs);
  } else {
    residual_rdna_kernel<true>
        <<<(pairs + threads - 1) / threads, threads, 0, stream>>>(
            update2, residual2, gate2, output2, pairs, width_pairs);
  }
}
