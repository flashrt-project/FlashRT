#include <cfloat>
#include <cmath>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

__device__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void qwen3_decode_attention_bf16_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k_cache,
    const hip_bfloat16* v_cache,
    hip_bfloat16* out,
    int layer_idx,
    int max_seq,
    int kv_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    float softmax_scale) {
  const int q_head = blockIdx.x;
  if (q_head >= q_heads || kv_seq <= 0) {
    return;
  }
  const int group = q_heads / kv_heads;
  const int kv_head = q_head / group;

  extern __shared__ float smem[];
  float* scores = smem;
  float* reduce = smem + kv_seq;

  const hip_bfloat16* qh = q + q_head * head_dim;
  for (int pos = 0; pos < kv_seq; ++pos) {
    const hip_bfloat16* kh =
        k_cache + ((layer_idx * max_seq + pos) * kv_heads + kv_head) * head_dim;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      sum += static_cast<float>(qh[d]) * static_cast<float>(kh[d]);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[pos] = reduce[0] * softmax_scale;
    }
    __syncthreads();
  }

  float local_max = -FLT_MAX;
  for (int pos = threadIdx.x; pos < kv_seq; pos += blockDim.x) {
    local_max = fmaxf(local_max, scores[pos]);
  }
  reduce[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x],
                                  reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_den = 0.0f;
  for (int pos = threadIdx.x; pos < kv_seq; pos += blockDim.x) {
    local_den += expf(scores[pos] - max_score);
  }
  reduce[threadIdx.x] = local_den;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_den = 1.0f / reduce[0];

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int pos = 0; pos < kv_seq; ++pos) {
      const float prob = expf(scores[pos] - max_score) * inv_den;
      const hip_bfloat16* vh =
          v_cache + ((layer_idx * max_seq + pos) * kv_heads + kv_head) * head_dim;
      acc += prob * static_cast<float>(vh[d]);
    }
    out[q_head * head_dim + d] = hip_bfloat16(acc);
  }
}

__global__ void qwen3_decode_attention_gate_quant_fp8_fnuz_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k_cache,
    const hip_bfloat16* v_cache,
    const hip_bfloat16* gate,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int layer_idx,
    int max_seq,
    int kv_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    float softmax_scale) {
  const int q_head = blockIdx.x;
  if (q_head >= q_heads || kv_seq <= 0 || head_dim != 256) {
    return;
  }
  const int group = q_heads / kv_heads;
  const int kv_head = q_head / group;

  extern __shared__ float smem[];
  float* scores = smem;
  float* reduce = smem + kv_seq;
  float* values = reduce + kBlock;

  const hip_bfloat16* qh = q + q_head * head_dim;
  for (int pos = 0; pos < kv_seq; ++pos) {
    const hip_bfloat16* kh =
        k_cache + ((layer_idx * max_seq + pos) * kv_heads + kv_head) * head_dim;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      sum += static_cast<float>(qh[d]) * static_cast<float>(kh[d]);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[pos] = reduce[0] * softmax_scale;
    }
    __syncthreads();
  }

  float local_max = -FLT_MAX;
  for (int pos = threadIdx.x; pos < kv_seq; pos += blockDim.x) {
    local_max = fmaxf(local_max, scores[pos]);
  }
  reduce[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x],
                                  reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce[0];

  float local_den = 0.0f;
  for (int pos = threadIdx.x; pos < kv_seq; pos += blockDim.x) {
    local_den += expf(scores[pos] - max_score);
  }
  reduce[threadIdx.x] = local_den;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_den = 1.0f / reduce[0];

  if (threadIdx.x < head_dim) {
    const int d = threadIdx.x;
    float acc = 0.0f;
    for (int pos = 0; pos < kv_seq; ++pos) {
      const float prob = expf(scores[pos] - max_score) * inv_den;
      const hip_bfloat16* vh =
          v_cache + ((layer_idx * max_seq + pos) * kv_heads + kv_head) * head_dim;
      acc += prob * static_cast<float>(vh[d]);
    }
    const int idx = q_head * head_dim + d;
    const hip_bfloat16 attn_bf16 = hip_bfloat16(acc);
    const hip_bfloat16 y_bf16 =
        hip_bfloat16(static_cast<float>(attn_bf16) *
                     sigmoidf_fast(static_cast<float>(gate[idx])));
    values[d] = static_cast<float>(y_bf16);
  }
  __syncthreads();

  constexpr int kQuantBlock = 128;
  for (int block_id = 0; block_id < 2; ++block_id) {
    const int lane = threadIdx.x;
    const int d = block_id * kQuantBlock + lane;
    reduce[lane] = lane < kQuantBlock ? fabsf(values[d]) : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
      }
      __syncthreads();
    }
    const float s = fmaxf(reduce[0] / 240.0f, 1.0e-8f);
    if (lane == 0) {
      scale[q_head * 2 + block_id] = s;
    }
    __syncthreads();
    if (lane < kQuantBlock) {
      const int idx = q_head * head_dim + d;
      out[idx] = __hip_fp8_e4m3_fnuz(values[d] / s);
    }
    __syncthreads();
  }
}

}  // namespace

void launch_qwen3_decode_attention_bf16(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int layer_idx,
    int max_seq,
    int kv_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    hipStream_t stream) {
  if (kv_seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const std::size_t shared_bytes =
      static_cast<std::size_t>(kv_seq + kBlock) * sizeof(float);
  hipLaunchKernelGGL(qwen3_decode_attention_bf16_kernel,
                     dim3(q_heads), dim3(kBlock), shared_bytes, stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k_cache),
                     static_cast<const hip_bfloat16*>(v_cache),
                     static_cast<hip_bfloat16*>(out),
                     layer_idx, max_seq, kv_seq, q_heads, kv_heads, head_dim,
                     scale);
}

void launch_qwen3_decode_attention_gate_quant_fp8_fnuz(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const void* gate,
    void* out,
    float* scale,
    int layer_idx,
    int max_seq,
    int kv_seq,
    int q_heads,
    int kv_heads,
    int head_dim,
    hipStream_t stream) {
  if (kv_seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }
  const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const std::size_t shared_bytes =
      static_cast<std::size_t>(kv_seq + kBlock + head_dim) * sizeof(float);
  hipLaunchKernelGGL(qwen3_decode_attention_gate_quant_fp8_fnuz_kernel,
                     dim3(q_heads), dim3(kBlock), shared_bytes, stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k_cache),
                     static_cast<const hip_bfloat16*>(v_cache),
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     layer_idx, max_seq, kv_seq, q_heads, kv_heads, head_dim,
                     attn_scale);
}
