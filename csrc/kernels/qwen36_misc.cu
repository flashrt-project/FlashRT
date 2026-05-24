#include "qwen36_misc.cuh"

#include <limits>

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

__device__ __forceinline__ __nv_bfloat16 partial_rope_value(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int row,
    int head,
    int col,
    int heads,
    int head_dim,
    int rope_dim)
{
  const size_t base = (static_cast<size_t>(row) * heads + head) * head_dim;
  if (col >= rope_dim) return x[base + col];

  const int half = rope_dim >> 1;
  const int rot_col = (col < half) ? (col + half) : (col - half);
  float rot = static_cast<float>(x[base + rot_col]);
  if (col < half) rot = -rot;
  const float xv = static_cast<float>(x[base + col]);
  const float cv = static_cast<float>(cos[row * rope_dim + col]);
  const float sv = static_cast<float>(sin[row * rope_dim + col]);
  const float rot_sin_bf = static_cast<float>(__float2bfloat16(rot * sv));
  return __float2bfloat16(rot_sin_bf + xv * cv);
}

__global__ void partial_rope_qk_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_total = rows * q_heads * head_dim;
  const int k_total = rows * k_heads * head_dim;
  const int total = q_total + k_total;
  if (idx >= total) return;

  if (idx < q_total) {
    const int col = idx % head_dim;
    const int head = (idx / head_dim) % q_heads;
    const int row = idx / (head_dim * q_heads);
    q_out[idx] = partial_rope_value(
        q_in, cos, sin, row, head, col, q_heads, head_dim, rope_dim);
  } else {
    const int k_idx = idx - q_total;
    const int col = k_idx % head_dim;
    const int head = (k_idx / head_dim) % k_heads;
    const int row = k_idx / (head_dim * k_heads);
    k_out[k_idx] = partial_rope_value(
        k_in, cos, sin, row, head, col, k_heads, head_dim, rope_dim);
  }
}

__global__ void tq_prepare_scalars_kernel(
    const __half* __restrict__ k_norm,
    const __half* __restrict__ k_rnorm,
    const __half* __restrict__ v_norm,
    float* __restrict__ norm_k,
    float* __restrict__ coef_rnorm,
    float* __restrict__ norm_v,
    int n,
    float coef)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  norm_k[idx] = __half2float(k_norm[idx]);
  coef_rnorm[idx] = __half2float(k_rnorm[idx]) * coef;
  norm_v[idx] = __half2float(v_norm[idx]);
}

__global__ void spec_argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int64_t* __restrict__ argmax_out,
    int rows,
    int vocab)
{
  extern __shared__ unsigned char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  const int row = blockIdx.x;
  if (row >= rows) return;
  const __nv_bfloat16* row_logits =
      logits + static_cast<size_t>(row) * vocab;

  float best_val = -std::numeric_limits<float>::infinity();
  int best_idx = 0;
  for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
    const float v = static_cast<float>(row_logits[col]);
    if (v > best_val || (v == best_val && col < best_idx)) {
      best_val = v;
      best_idx = col;
    }
  }
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_val = s_val[threadIdx.x + stride];
      const int other_idx = s_idx[threadIdx.x + stride];
      const float cur_val = s_val[threadIdx.x];
      const int cur_idx = s_idx[threadIdx.x];
      if (other_val > cur_val
          || (other_val == cur_val && other_idx < cur_idx)) {
        s_val[threadIdx.x] = other_val;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    argmax_out[row] = static_cast<int64_t>(s_idx[0]);
  }
}

__global__ void spec_argmax_bf16_partition_kernel(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idx,
    int rows,
    int vocab,
    int parts)
{
  extern __shared__ unsigned char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  const int row = blockIdx.x;
  const int part = blockIdx.y;
  if (row >= rows || part >= parts) return;

  const int cols_per_part = (vocab + parts - 1) / parts;
  const int begin = part * cols_per_part;
  const int end = min(vocab, begin + cols_per_part);
  const __nv_bfloat16* row_logits =
      logits + static_cast<size_t>(row) * vocab;

  float best_val = -std::numeric_limits<float>::infinity();
  int best_idx = 0;
  for (int col = begin + threadIdx.x; col < end; col += blockDim.x) {
    const float v = static_cast<float>(row_logits[col]);
    if (v > best_val || (v == best_val && col < best_idx)) {
      best_val = v;
      best_idx = col;
    }
  }
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_val = s_val[threadIdx.x + stride];
      const int other_idx = s_idx[threadIdx.x + stride];
      const float cur_val = s_val[threadIdx.x];
      const int cur_idx = s_idx[threadIdx.x];
      if (other_val > cur_val
          || (other_val == cur_val && other_idx < cur_idx)) {
        s_val[threadIdx.x] = other_val;
        s_idx[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int out = row * parts + part;
    partial_vals[out] = s_val[0];
    partial_idx[out] = s_idx[0];
  }
}

__global__ void spec_reduce_accept_partition_kernel(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idx,
    const int64_t* __restrict__ drafts,
    int64_t* __restrict__ argmax_out,
    int* __restrict__ accept_n,
    int rows,
    int parts,
    int spec_k)
{
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    float best_val = -std::numeric_limits<float>::infinity();
    int best_idx = 0;
    for (int part = 0; part < parts; ++part) {
      const int off = row * parts + part;
      const float v = partial_vals[off];
      const int idx = partial_idx[off];
      if (v > best_val || (v == best_val && idx < best_idx)) {
        best_val = v;
        best_idx = idx;
      }
    }
    argmax_out[row] = static_cast<int64_t>(best_idx);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int n = 0;
    for (; n < spec_k; ++n) {
      if (argmax_out[n] != drafts[n]) break;
    }
    accept_n[0] = n;
  }
}

__global__ void spec_accept_kernel(
    const int64_t* __restrict__ argmax_out,
    const int64_t* __restrict__ drafts,
    int* __restrict__ accept_n,
    int spec_k)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int n = 0;
  for (; n < spec_k; ++n) {
    if (argmax_out[n] != drafts[n]) break;
  }
  accept_n[0] = n;
}

}  // namespace

void qwen36_embedding_lookup_bf16(
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

void qwen36_partial_rope_qk_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    cudaStream_t stream)
{
  if (rows <= 0 || q_heads <= 0 || k_heads <= 0 || head_dim <= 0
      || rope_dim <= 0) {
    return;
  }
  const int total = rows * (q_heads + k_heads) * head_dim;
  const dim3 block(kThreads);
  const dim3 grid((total + kThreads - 1) / kThreads);
  partial_rope_qk_bf16_kernel<<<grid, block, 0, stream>>>(
      q_in, k_in, cos, sin, q_out, k_out,
      rows, q_heads, k_heads, head_dim, rope_dim);
}

void qwen36_tq_prepare_scalars(
    const __half* k_norm,
    const __half* k_rnorm,
    const __half* v_norm,
    float* norm_k,
    float* coef_rnorm,
    float* norm_v,
    int n,
    float coef,
    cudaStream_t stream)
{
  if (n <= 0) return;
  const dim3 block(kThreads);
  const dim3 grid((n + kThreads - 1) / kThreads);
  tq_prepare_scalars_kernel<<<grid, block, 0, stream>>>(
      k_norm, k_rnorm, v_norm, norm_k, coef_rnorm, norm_v, n, coef);
}

void qwen36_argmax_bf16(
    const __nv_bfloat16* logits,
    int64_t* argmax_out,
    int rows,
    int vocab,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0) return;
  const int threads = 1024;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  spec_argmax_bf16_kernel<<<rows, threads, smem, stream>>>(
      logits, argmax_out, rows, vocab);
}

void qwen36_spec_accept_greedy_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    int rows,
    int vocab,
    int spec_k,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0) return;
  const int threads = 1024;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  spec_argmax_bf16_kernel<<<rows, threads, smem, stream>>>(
      logits, argmax_out, rows, vocab);
  spec_accept_kernel<<<1, 1, 0, stream>>>(
      argmax_out, drafts, accept_n, spec_k);
}

void qwen36_spec_accept_partitioned_bf16(
    const __nv_bfloat16* logits,
    const int64_t* drafts,
    int64_t* argmax_out,
    int* accept_n,
    float* partial_vals,
    int* partial_idx,
    int rows,
    int vocab,
    int spec_k,
    int parts,
    cudaStream_t stream)
{
  if (rows <= 0 || vocab <= 0 || parts <= 0) return;
  parts = min(parts, 128);
  const int threads = 256;
  const size_t smem = threads * (sizeof(float) + sizeof(int));
  const dim3 grid(rows, parts);
  spec_argmax_bf16_partition_kernel<<<grid, threads, smem, stream>>>(
      logits, partial_vals, partial_idx, rows, vocab, parts);
  spec_reduce_accept_partition_kernel<<<1, 128, 0, stream>>>(
      partial_vals, partial_idx, drafts, argmax_out, accept_n,
      rows, parts, spec_k);
}

}  // namespace flash_rt::kernels
