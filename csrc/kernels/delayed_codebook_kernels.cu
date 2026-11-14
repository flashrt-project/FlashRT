#include "delayed_codebook_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {
namespace {

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform_open01(
    uint64_t seed, uint64_t step, int codebook) {
  const uint64_t key = seed ^ (step * 0xd2b74407b1ce6e93ULL) ^
                       (static_cast<uint64_t>(codebook) *
                        0xca5a826395121157ULL);
  const uint32_t bits = static_cast<uint32_t>(splitmix64(key) >> 40);
  return (static_cast<float>(bits) + 0.5f) * 0x1.0p-24f;
}

__global__ void delayed_codebook_argmax_kernel(
    const __nv_bfloat16* logits,
    int64_t* codes,
    int num_codebooks,
    int codebook_vocab,
    int delay,
    int boc) {
  const int cb = blockIdx.x;
  if (cb >= num_codebooks) return;
  if (delay < num_codebooks && cb > delay) {
    if (threadIdx.x == 0) codes[cb] = static_cast<int64_t>(boc);
    return;
  }

  extern __shared__ unsigned char smem[];
  float* vals = reinterpret_cast<float*>(smem);
  int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

  const int tid = threadIdx.x;
  float best = -3.402823466e38f;
  int best_i = 0;
  const __nv_bfloat16* row = logits + cb * codebook_vocab;
  for (int i = tid; i < codebook_vocab; i += blockDim.x) {
    const float v = __bfloat162float(row[i]);
    if (v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }
  vals[tid] = best;
  idxs[tid] = best_i;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float ov = vals[tid + stride];
      const int oi = idxs[tid + stride];
      if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
        vals[tid] = ov;
        idxs[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) codes[cb] = static_cast<int64_t>(idxs[0]);
}

__global__ void delayed_codebook_sample_kernel(
    const __nv_bfloat16* logits,
    int64_t* codes,
    int num_codebooks,
    int codebook_vocab,
    int delay,
    int boc,
    float temperature,
    uint64_t seed,
    uint64_t step) {
  const int cb = blockIdx.x;
  if (cb >= num_codebooks) return;
  if (delay < num_codebooks && cb > delay) {
    if (threadIdx.x == 0) codes[cb] = static_cast<int64_t>(boc);
    return;
  }

  extern __shared__ unsigned char smem[];
  float* reduce = reinterpret_cast<float*>(smem);
  float* weights = reduce + blockDim.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* row = logits + cb * codebook_vocab;

  float local_max = -3.402823466e38f;
  for (int i = tid; i < codebook_vocab; i += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(row[i]));
  }
  reduce[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  for (int i = tid; i < codebook_vocab; i += blockDim.x) {
    const float w = expf((__bfloat162float(row[i]) - row_max) / temperature);
    weights[i] = w;
    local_sum += w;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }

  if (tid == 0) {
    const float target = uniform_open01(seed, step, cb) * reduce[0];
    float cumulative = 0.0f;
    int selected = codebook_vocab - 1;
    for (int i = 0; i < codebook_vocab; ++i) {
      // FP32 rounding can put the target at the CDF endpoint. Fall back
      // to its last positive weight, never an underflowed zero-mass token.
      if (weights[i] > 0.0f) selected = i;
      cumulative += weights[i];
      if (target < cumulative) {
        selected = i;
        break;
      }
    }
    codes[cb] = static_cast<int64_t>(selected);
  }
}

__global__ void delayed_codebook_embed_sum_kernel(
    const int64_t* codes,
    const __nv_bfloat16* codebook,
    __nv_bfloat16* embed,
    int num_codebooks,
    int codebook_vocab,
    int hidden) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  float acc = 0.0f;
  for (int cb = 0; cb < num_codebooks; ++cb) {
    const int code = static_cast<int>(codes[cb]);
    const int row = cb * codebook_vocab + code;
    acc += __bfloat162float(codebook[row * hidden + h]);
  }
  embed[h] = __float2bfloat16(acc);
}

}  // namespace

void delayed_codebook_argmax_embed_bf16(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* codebook,
    int64_t* codes_out,
    __nv_bfloat16* embed_out,
    int num_codebooks,
    int codebook_vocab,
    int hidden,
    int delay,
    int boc,
    cudaStream_t stream) {
  if (num_codebooks <= 0 || codebook_vocab <= 0 || hidden <= 0) return;
  const int arg_threads = 1024;
  const size_t smem = arg_threads * (sizeof(float) + sizeof(int));
  delayed_codebook_argmax_kernel<<<num_codebooks, arg_threads, smem, stream>>>(
      logits, codes_out, num_codebooks, codebook_vocab, delay, boc);
  const int emb_threads = 256;
  const int emb_blocks = (hidden + emb_threads - 1) / emb_threads;
  delayed_codebook_embed_sum_kernel<<<emb_blocks, emb_threads, 0, stream>>>(
      codes_out, codebook, embed_out, num_codebooks, codebook_vocab, hidden);
}

void delayed_codebook_sample_embed_bf16(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* codebook,
    int64_t* codes_out,
    __nv_bfloat16* embed_out,
    int num_codebooks,
    int codebook_vocab,
    int hidden,
    int delay,
    int boc,
    float temperature,
    uint64_t seed,
    uint64_t step,
    cudaStream_t stream) {
  if (num_codebooks <= 0 || codebook_vocab <= 0 || hidden <= 0 ||
      temperature <= 0.0f) {
    return;
  }
  const int sample_threads = 256;
  const size_t smem = (sample_threads + codebook_vocab) * sizeof(float);
  delayed_codebook_sample_kernel<<<num_codebooks, sample_threads, smem, stream>>>(
      logits, codes_out, num_codebooks, codebook_vocab, delay, boc,
      temperature, seed, step);
  const int emb_threads = 256;
  const int emb_blocks = (hidden + emb_threads - 1) / emb_threads;
  delayed_codebook_embed_sum_kernel<<<emb_blocks, emb_threads, 0, stream>>>(
      codes_out, codebook, embed_out, num_codebooks, codebook_vocab, hidden);
}

}  // namespace flash_rt::kernels
