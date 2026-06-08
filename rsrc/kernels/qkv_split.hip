#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

__global__ void qkv_split_bf16_kernel(const hip_bfloat16* qkv,
                                      hip_bfloat16* q,
                                      hip_bfloat16* k,
                                      hip_bfloat16* v,
                                      int seq,
                                      int q_dim,
                                      int k_dim,
                                      int v_dim) {
  const int qkv_dim = q_dim + k_dim + v_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * qkv_dim;
  if (idx >= total) {
    return;
  }

  const int row = idx / qkv_dim;
  const int col = idx % qkv_dim;
  const hip_bfloat16 val = qkv[idx];
  if (col < q_dim) {
    q[row * q_dim + col] = val;
  } else if (col < q_dim + k_dim) {
    k[row * k_dim + (col - q_dim)] = val;
  } else {
    v[row * v_dim + (col - q_dim - k_dim)] = val;
  }
}

__global__ void qkv_split_rope_bf16_kernel(const hip_bfloat16* qkv,
                                           const hip_bfloat16* rope,
                                           hip_bfloat16* q,
                                           hip_bfloat16* k,
                                           hip_bfloat16* v,
                                           int seq,
                                           int q_dim,
                                           int k_dim,
                                           int v_dim,
                                           int head_dim) {
  const int qkv_dim = q_dim + k_dim + v_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * qkv_dim;
  if (idx >= total) {
    return;
  }

  const int row = idx / qkv_dim;
  const int col = idx % qkv_dim;
  if (col < q_dim) {
    const int q_col = col;
    const int head = q_col / head_dim;
    const int d = q_col % head_dim;
    const int pair = d / 2;
    const int pair_base = row * qkv_dim + head * head_dim + pair * 2;
    const float x0 = static_cast<float>(qkv[pair_base]);
    const float x1 = static_cast<float>(qkv[pair_base + 1]);
    const int rope_base = row * head_dim + pair * 2;
    const float c = static_cast<float>(rope[rope_base]);
    const float s = static_cast<float>(rope[rope_base + 1]);
    q[row * q_dim + q_col] = hip_bfloat16(
        (d & 1) == 0 ? x0 * c - x1 * s : x1 * c + x0 * s);
  } else if (col < q_dim + k_dim) {
    const int k_col = col - q_dim;
    const int d = k_col % head_dim;
    const int pair = d / 2;
    const int pair_base = row * qkv_dim + q_dim + pair * 2;
    const float x0 = static_cast<float>(qkv[pair_base]);
    const float x1 = static_cast<float>(qkv[pair_base + 1]);
    const int rope_base = row * head_dim + pair * 2;
    const float c = static_cast<float>(rope[rope_base]);
    const float s = static_cast<float>(rope[rope_base + 1]);
    k[row * k_dim + k_col] = hip_bfloat16(
        (d & 1) == 0 ? x0 * c - x1 * s : x1 * c + x0 * s);
  } else {
    const int v_col = col - q_dim - k_dim;
    v[row * v_dim + v_col] = qkv[idx];
  }
}

__global__ void qwen3_qk_norm_rope_bf16_kernel(
    const hip_bfloat16* qkv,
    const float* cos,
    const float* sin,
    const hip_bfloat16* q_norm_w,
    const hip_bfloat16* k_norm_w,
    hip_bfloat16* q,
    hip_bfloat16* k,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim) {
  const int total_heads = q_heads + kv_heads;
  const int block = blockIdx.x;
  const int row = block / total_heads;
  const int logical_head = block % total_heads;
  if (row >= seq) {
    return;
  }

  extern __shared__ float smem[];
  const int q_dim = q_heads * head_dim;
  const int k_dim = kv_heads * head_dim;
  const int qkv_dim = q_dim + 2 * k_dim;
  const bool is_q = logical_head < q_heads;
  const int head = is_q ? logical_head : logical_head - q_heads;
  const int base = row * qkv_dim + (is_q ? head * head_dim : q_dim + head * head_dim);
  const hip_bfloat16* weight = is_q ? q_norm_w : k_norm_w;

  float sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = static_cast<float>(qkv[base + d]);
    sum += v * v;
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + 1.0e-6f);
  const int half = head_dim / 2;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const int other_d = (d < half) ? d + half : d - half;
    const float x = static_cast<float>(qkv[base + d]) * inv *
                    static_cast<float>(weight[d]);
    const float other = static_cast<float>(qkv[base + other_d]) * inv *
                        static_cast<float>(weight[other_d]);
    const float rotated = (d < half) ? -other : other;
    const float c = cos[row * head_dim + d];
    const float s = sin[row * head_dim + d];
    const hip_bfloat16 out = hip_bfloat16(x * c + rotated * s);
    if (is_q) {
      q[row * q_dim + head * head_dim + d] = out;
    } else {
      k[row * k_dim + head * head_dim + d] = out;
    }
  }
}

__global__ void qwen3_v_split_bf16_kernel(
    const hip_bfloat16* qkv,
    hip_bfloat16* v,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim) {
  const int k_dim = kv_heads * head_dim;
  const int q_dim = q_heads * head_dim;
  const int qkv_dim = q_dim + 2 * k_dim;
  const int total = seq * k_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = idx / k_dim;
  const int col = idx % k_dim;
  v[idx] = qkv[row * qkv_dim + q_dim + k_dim + col];
}

__global__ void qwen3_qk_norm_rope_cache_bf16_kernel(
    const hip_bfloat16* qkv,
    const float* cos,
    const float* sin,
    const hip_bfloat16* q_norm_w,
    const hip_bfloat16* k_norm_w,
    hip_bfloat16* q,
    hip_bfloat16* k_cache,
    int layer_idx,
    int max_seq,
    int kv_start,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim) {
  const int total_heads = q_heads + kv_heads;
  const int block = blockIdx.x;
  const int row = block / total_heads;
  const int logical_head = block % total_heads;
  if (row >= seq) {
    return;
  }

  extern __shared__ float smem[];
  const int q_dim = q_heads * head_dim;
  const int k_dim = kv_heads * head_dim;
  const int qkv_dim = q_dim + 2 * k_dim;
  const bool is_q = logical_head < q_heads;
  const int head = is_q ? logical_head : logical_head - q_heads;
  const int base = row * qkv_dim + (is_q ? head * head_dim : q_dim + head * head_dim);
  const hip_bfloat16* weight = is_q ? q_norm_w : k_norm_w;

  float sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = static_cast<float>(qkv[base + d]);
    sum += v * v;
  }
  smem[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + 1.0e-6f);
  const int half = head_dim / 2;
  const int pos = kv_start + row;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const int other_d = (d < half) ? d + half : d - half;
    const float x = static_cast<float>(qkv[base + d]) * inv *
                    static_cast<float>(weight[d]);
    const float other = static_cast<float>(qkv[base + other_d]) * inv *
                        static_cast<float>(weight[other_d]);
    const float rotated = (d < half) ? -other : other;
    const float c = cos[row * head_dim + d];
    const float s = sin[row * head_dim + d];
    const hip_bfloat16 out = hip_bfloat16(x * c + rotated * s);
    if (is_q) {
      q[(row * q_heads + head) * head_dim + d] = out;
    } else {
      k_cache[((layer_idx * max_seq + pos) * kv_heads + head) * head_dim + d] = out;
    }
  }
}

__global__ void qwen3_v_cache_bf16_kernel(
    const hip_bfloat16* qkv,
    hip_bfloat16* v_cache,
    int layer_idx,
    int max_seq,
    int kv_start,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim) {
  const int k_dim = kv_heads * head_dim;
  const int q_dim = q_heads * head_dim;
  const int qkv_dim = q_dim + 2 * k_dim;
  const int total = seq * k_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = idx / k_dim;
  const int col = idx % k_dim;
  const int pos = kv_start + row;
  v_cache[(layer_idx * max_seq + pos) * k_dim + col] =
      qkv[row * qkv_dim + q_dim + k_dim + col];
}

}  // namespace

void launch_qkv_split_bf16(const void* qkv, void* q, void* k, void* v,
                           int seq, int q_dim, int k_dim, int v_dim,
                           hipStream_t stream) {
  if (seq <= 0 || q_dim <= 0 || k_dim <= 0 || v_dim <= 0) {
    return;
  }
  const int total = seq * (q_dim + k_dim + v_dim);
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qkv_split_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     static_cast<hip_bfloat16*>(v),
                     seq, q_dim, k_dim, v_dim);
}

void launch_qkv_split_rope_bf16(const void* qkv, const void* rope,
                                void* q, void* k, void* v,
                                int seq, int q_dim, int k_dim, int v_dim,
                                int head_dim, hipStream_t stream) {
  if (seq <= 0 || q_dim <= 0 || k_dim <= 0 || v_dim <= 0 || head_dim <= 0) {
    return;
  }
  const int total = seq * (q_dim + k_dim + v_dim);
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qkv_split_rope_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     static_cast<const hip_bfloat16*>(rope),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     static_cast<hip_bfloat16*>(v),
                     seq, q_dim, k_dim, v_dim, head_dim);
}

void launch_qwen3_qkv_norm_rope_bf16(
    const void* qkv,
    const float* cos,
    const float* sin,
    const void* q_norm_w,
    const void* k_norm_w,
    void* q,
    void* k,
    void* v,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    hipStream_t stream) {
  if (seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int total_heads = seq * (q_heads + kv_heads);
  hipLaunchKernelGGL(qwen3_qk_norm_rope_bf16_kernel,
                     dim3(total_heads), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     cos,
                     sin,
                     static_cast<const hip_bfloat16*>(q_norm_w),
                     static_cast<const hip_bfloat16*>(k_norm_w),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     seq, q_heads, kv_heads, head_dim);
  const int k_dim = kv_heads * head_dim;
  const int total_v = seq * k_dim;
  const int blocks = (total_v + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen3_v_split_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     static_cast<hip_bfloat16*>(v),
                     seq, q_heads, kv_heads, head_dim);
}

void launch_qwen3_qkv_norm_rope_cache_bf16(
    const void* qkv,
    const float* cos,
    const float* sin,
    const void* q_norm_w,
    const void* k_norm_w,
    void* q,
    void* k_cache,
    void* v_cache,
    int layer_idx,
    int max_seq,
    int kv_start,
    int seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    hipStream_t stream) {
  if (seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int total_heads = seq * (q_heads + kv_heads);
  hipLaunchKernelGGL(qwen3_qk_norm_rope_cache_bf16_kernel,
                     dim3(total_heads), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     cos,
                     sin,
                     static_cast<const hip_bfloat16*>(q_norm_w),
                     static_cast<const hip_bfloat16*>(k_norm_w),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k_cache),
                     layer_idx, max_seq, kv_start, seq,
                     q_heads, kv_heads, head_dim);
  const int k_dim = kv_heads * head_dim;
  const int total_v = seq * k_dim;
  const int blocks = (total_v + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen3_v_cache_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     static_cast<hip_bfloat16*>(v_cache),
                     layer_idx, max_seq, kv_start, seq,
                     q_heads, kv_heads, head_dim);
}
