// Generic bf16 embedding lookup — see header for design notes.
// Moved here from qwen36_misc.cu as part of the generic-helper ownership
// cleanup (#112). The kernel is unchanged; only the file and the public symbol
// name (embedding_lookup_bf16) are neutral now.

#include "embedding_lookup_bf16.cuh"

namespace flash_rt::kernels {

namespace {

constexpr int kThreads = 256;

__global__ void embedding_lookup_bf16_kernel(
    const int64_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embed,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int hidden)
{
  const int row = blockIdx.y;
  if (row >= rows) return;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= hidden) return;
  const int64_t tok = token_ids[row];
  out[row * hidden + col] = embed[tok * hidden + col];
}

}  // namespace

void embedding_lookup_bf16(
    const int64_t* token_ids,
    const __nv_bfloat16* embed,
    __nv_bfloat16* out,
    int rows,
    int hidden,
    cudaStream_t stream)
{
  if (rows <= 0 || hidden <= 0) return;
  const dim3 block(kThreads);
  const dim3 grid((hidden + kThreads - 1) / kThreads, rows);
  embedding_lookup_bf16_kernel<<<grid, block, 0, stream>>>(
      token_ids, embed, out, rows, hidden);
}

}  // namespace flash_rt::kernels
