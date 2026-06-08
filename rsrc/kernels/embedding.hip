#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

__global__ void embedding_lookup_bf16_kernel(
    const int64_t* input_ids,
    const hip_bfloat16* embedding,
    hip_bfloat16* out,
    int rows,
    int hidden) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const int row = idx / hidden;
  const int col = idx - row * hidden;
  const int64_t token = input_ids[row];
  out[idx] = embedding[token * hidden + col];
}

}  // namespace

void launch_embedding_lookup_bf16(
    const int64_t* input_ids,
    const void* embedding,
    void* out,
    int rows,
    int hidden,
    hipStream_t stream) {
  if (rows <= 0 || hidden <= 0) {
    return;
  }
  const int total = rows * hidden;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(embedding_lookup_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     input_ids,
                     static_cast<const hip_bfloat16*>(embedding),
                     static_cast<hip_bfloat16*>(out),
                     rows,
                     hidden);
}
