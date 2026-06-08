#include <cmath>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;

__device__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ float siluf_fast(float x) {
  return x * sigmoidf_fast(x);
}

__device__ float softplusf_stable(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

__global__ void qwen36_causal_conv1d_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* out,
    int rows,
    int dim,
    int kernel_size,
    bool apply_silu) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * dim;
  if (idx >= total) {
    return;
  }
  const int row = idx / dim;
  const int col = idx % dim;
  float acc = 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    const int src_row = row - (kernel_size - 1) + k;
    if (src_row >= 0) {
      acc += static_cast<float>(x[src_row * dim + col]) *
             static_cast<float>(weight[col * kernel_size + k]);
    }
  }
  out[idx] = hip_bfloat16(apply_silu ? siluf_fast(acc) : acc);
}

__global__ void qwen36_causal_conv1d_state_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    const hip_bfloat16* state,
    hip_bfloat16* new_state,
    hip_bfloat16* out,
    int rows,
    int dim,
    int kernel_size,
    bool apply_silu) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int state_rows = kernel_size - 1;
  const int out_total = rows * dim;
  const int state_total = state_rows * dim;
  const int total = out_total + state_total;
  if (idx >= total) {
    return;
  }
  if (idx < out_total) {
    const int row = idx / dim;
    const int col = idx % dim;
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
      const int src_row = row - state_rows + k;
      const hip_bfloat16* src = nullptr;
      int src_offset = 0;
      if (src_row >= 0) {
        src = x;
        src_offset = src_row * dim + col;
      } else {
        src = state;
        src_offset = (state_rows + src_row) * dim + col;
      }
      acc += static_cast<float>(src[src_offset]) *
             static_cast<float>(weight[col * kernel_size + k]);
    }
    out[idx] = hip_bfloat16(apply_silu ? siluf_fast(acc) : acc);
    return;
  }

  const int sidx = idx - out_total;
  const int state_row = sidx / dim;
  const int col = sidx % dim;
  const int combined_row = rows + state_row - state_rows;
  if (combined_row >= 0) {
    new_state[sidx] = x[combined_row * dim + col];
  } else {
    new_state[sidx] = state[(state_rows + combined_row) * dim + col];
  }
}

__global__ void qwen36_causal_conv1d_state_split_qkv_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    const hip_bfloat16* state,
    hip_bfloat16* new_state,
    hip_bfloat16* q,
    hip_bfloat16* k_out,
    hip_bfloat16* v,
    int rows,
    int dim,
    int kernel_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int state_rows = kernel_size - 1;
  const int conv_total = rows * dim;
  const int state_total = state_rows * dim;
  const int total = conv_total + state_total;
  if (idx >= total) {
    return;
  }
  if (idx < conv_total) {
    const int row = idx / dim;
    const int col = idx % dim;
    float acc = 0.0f;
    for (int kk = 0; kk < kernel_size; ++kk) {
      const int src_row = row - state_rows + kk;
      const hip_bfloat16* src = nullptr;
      int src_offset = 0;
      if (src_row >= 0) {
        src = x;
        src_offset = src_row * dim + col;
      } else {
        src = state;
        src_offset = (state_rows + src_row) * dim + col;
      }
      acc += static_cast<float>(src[src_offset]) *
             static_cast<float>(weight[col * kernel_size + kk]);
    }
    const hip_bfloat16 y = hip_bfloat16(siluf_fast(acc));
    if (col < 2048) {
      const int head16 = col / 128;
      const int d = col % 128;
      const int head48_base = head16 * 3;
      for (int rep = 0; rep < 3; ++rep) {
        q[(row * 48 + head48_base + rep) * 128 + d] = y;
      }
    } else if (col < 4096) {
      const int kcol = col - 2048;
      const int head16 = kcol / 128;
      const int d = kcol % 128;
      const int head48_base = head16 * 3;
      for (int rep = 0; rep < 3; ++rep) {
        k_out[(row * 48 + head48_base + rep) * 128 + d] = y;
      }
    } else {
      const int vcol = col - 4096;
      v[row * 48 * 128 + vcol] = y;
    }
    return;
  }

  const int sidx = idx - conv_total;
  const int state_row = sidx / dim;
  const int col = sidx % dim;
  const int combined_row = rows + state_row - state_rows;
  if (combined_row >= 0) {
    new_state[sidx] = x[combined_row * dim + col];
  } else {
    new_state[sidx] = state[(state_rows + combined_row) * dim + col];
  }
}

__global__ void qwen36_causal_conv1d_state_split_qkv_bf16_inplace_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* state,
    hip_bfloat16* q,
    hip_bfloat16* k_out,
    hip_bfloat16* v,
    int dim,
    int kernel_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= dim || kernel_size != 4) {
    return;
  }
  const hip_bfloat16 s0 = state[col];
  const hip_bfloat16 s1 = state[dim + col];
  const hip_bfloat16 s2 = state[2 * dim + col];
  const hip_bfloat16 x0 = x[col];
  float acc = 0.0f;
  acc += static_cast<float>(s0) * static_cast<float>(weight[col * kernel_size + 0]);
  acc += static_cast<float>(s1) * static_cast<float>(weight[col * kernel_size + 1]);
  acc += static_cast<float>(s2) * static_cast<float>(weight[col * kernel_size + 2]);
  acc += static_cast<float>(x0) * static_cast<float>(weight[col * kernel_size + 3]);
  const hip_bfloat16 y = hip_bfloat16(siluf_fast(acc));

  if (col < 2048) {
    const int head16 = col / 128;
    const int d = col % 128;
    const int head48_base = head16 * 3;
    for (int rep = 0; rep < 3; ++rep) {
      q[(head48_base + rep) * 128 + d] = y;
    }
  } else if (col < 4096) {
    const int kcol = col - 2048;
    const int head16 = kcol / 128;
    const int d = kcol % 128;
    const int head48_base = head16 * 3;
    for (int rep = 0; rep < 3; ++rep) {
      k_out[(head48_base + rep) * 128 + d] = y;
    }
  } else {
    const int vcol = col - 4096;
    v[vcol] = y;
  }

  state[col] = s1;
  state[dim + col] = s2;
  state[2 * dim + col] = x0;
}

__global__ void qwen36_lin_split_qkv_broadcast_bf16_kernel(
    const hip_bfloat16* qkv,
    hip_bfloat16* q,
    hip_bfloat16* k,
    hip_bfloat16* v,
    int rows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * 48 * 128;
  if (idx >= total) {
    return;
  }
  const int d = idx % 128;
  const int head48 = (idx / 128) % 48;
  const int row = idx / (48 * 128);
  const int head16 = head48 / 3;
  const int q_base = row * 10240;
  q[idx] = qkv[q_base + head16 * 128 + d];
  k[idx] = qkv[q_base + 2048 + head16 * 128 + d];
  v[idx] = qkv[q_base + 4096 + head48 * 128 + d];
}

__global__ void qwen36_gdn_gating_bf16_kernel(
    const hip_bfloat16* a,
    const hip_bfloat16* b,
    const hip_bfloat16* A_log,
    const hip_bfloat16* dt_bias,
    hip_bfloat16* g,
    hip_bfloat16* beta,
    int rows,
    int heads) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * heads;
  if (idx >= total) {
    return;
  }
  const int h = idx % heads;
  const float av = static_cast<float>(a[idx]);
  const float bv = static_cast<float>(b[idx]);
  const float alog = static_cast<float>(A_log[h]);
  const float dt = static_cast<float>(dt_bias[h]);
  beta[idx] = hip_bfloat16(sigmoidf_fast(bv));
  g[idx] = hip_bfloat16(-expf(alog) * softplusf_stable(av + dt));
}

__global__ void qwen36_silu_mul_bf16_kernel(
    const hip_bfloat16* gate,
    const hip_bfloat16* up,
    hip_bfloat16* out,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float gv = static_cast<float>(gate[idx]);
  const float uv = static_cast<float>(up[idx]);
  out[idx] = hip_bfloat16(siluf_fast(gv) * uv);
}

__global__ void qwen36_sigmoid_mul_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* gate,
    hip_bfloat16* out,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float xv = static_cast<float>(x[idx]);
  const float gv = static_cast<float>(gate[idx]);
  out[idx] = hip_bfloat16(xv * sigmoidf_fast(gv));
}

__global__ void qwen36_add_bf16_kernel(
    const hip_bfloat16* a,
    const hip_bfloat16* b,
    hip_bfloat16* out,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  out[idx] = hip_bfloat16(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
}

__global__ void qwen36_copy_bf16_kernel(
    const hip_bfloat16* x,
    hip_bfloat16* out,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  out[idx] = x[idx];
}

__global__ void qwen36_full_v_broadcast_bf16_kernel(
    const hip_bfloat16* v,
    hip_bfloat16* out,
    int rows,
    int kv_heads,
    int q_heads,
    int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * q_heads * head_dim;
  if (idx >= total) {
    return;
  }
  const int d = idx % head_dim;
  const int qh = (idx / head_dim) % q_heads;
  const int row = idx / (q_heads * head_dim);
  const int kvh = qh / (q_heads / kv_heads);
  out[idx] = v[(row * kv_heads + kvh) * head_dim + d];
}

__global__ void qwen36_full_qk_norm_partial_rope_cache_bf16_kernel(
    const hip_bfloat16* q_gate,
    const hip_bfloat16* k_proj,
    const hip_bfloat16* v_proj,
    const float* cos,
    const float* sin,
    const hip_bfloat16* q_norm_w,
    const hip_bfloat16* k_norm_w,
    hip_bfloat16* q,
    hip_bfloat16* k_cache,
    hip_bfloat16* v_cache,
    int layer_idx,
    int max_seq,
    int kv_start,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int rotary_dim,
    float eps) {
  const int total_heads = rows * (q_heads + kv_heads);
  const int block = blockIdx.x;
  if (block >= total_heads) {
    return;
  }
  const int row = block / (q_heads + kv_heads);
  const int logical_head = block % (q_heads + kv_heads);
  const bool is_q = logical_head < q_heads;
  const int head = is_q ? logical_head : logical_head - q_heads;
  const hip_bfloat16* src =
      is_q ? q_gate + (row * q_heads + head) * head_dim
           : k_proj + (row * kv_heads + head) * head_dim;
  const hip_bfloat16* weight = is_q ? q_norm_w : k_norm_w;

  extern __shared__ float smem[];
  float sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float v = static_cast<float>(src[d]);
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

  const float inv = rsqrtf(smem[0] / static_cast<float>(head_dim) + eps);
  const int pos = kv_start + row;
  const int half_rotary = rotary_dim / 2;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float x = static_cast<float>(src[d]) * inv *
              static_cast<float>(weight[d]);
    if (d < rotary_dim) {
      const int other_d = (d < half_rotary) ? d + half_rotary : d - half_rotary;
      const float other = static_cast<float>(src[other_d]) * inv *
                          static_cast<float>(weight[other_d]);
      const float rotated = (d < half_rotary) ? -other : other;
      const float c = cos[row * rotary_dim + d];
      const float s = sin[row * rotary_dim + d];
      x = x * c + rotated * s;
    }
    const hip_bfloat16 out = hip_bfloat16(x);
    if (is_q) {
      q[(row * q_heads + head) * head_dim + d] = out;
    } else {
      k_cache[((layer_idx * max_seq + pos) * kv_heads + head) * head_dim + d] = out;
    }
  }

  if (!is_q) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      v_cache[((layer_idx * max_seq + pos) * kv_heads + head) * head_dim + d] =
          v_proj[(row * kv_heads + head) * head_dim + d];
    }
  }
}

__global__ void qwen36_small_linear_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* out,
    int rows,
    int out_features,
    int hidden) {
  const int row = blockIdx.x;
  const int out_col = blockIdx.y;
  if (row >= rows || out_col >= out_features) {
    return;
  }
  extern __shared__ float smem[];
  float acc = 0.0f;
  const int x_base = row * hidden;
  const int w_base = out_col * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    acc += static_cast<float>(x[x_base + col]) *
           static_cast<float>(weight[w_base + col]);
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row * out_features + out_col] = hip_bfloat16(smem[0]);
  }
}

__global__ void qwen36_in_proj_ab_gating_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* a_weight,
    const hip_bfloat16* b_weight,
    const hip_bfloat16* A_log,
    const hip_bfloat16* dt_bias,
    hip_bfloat16* g,
    hip_bfloat16* beta,
    int rows,
    int heads,
    int hidden) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows || head >= heads) {
    return;
  }
  extern __shared__ float smem[];
  float* a_sums = smem;
  float* b_sums = smem + blockDim.x;
  float acc_a = 0.0f;
  float acc_b = 0.0f;
  const int x_base = row * hidden;
  const int w_base = head * hidden;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float xv = static_cast<float>(x[x_base + col]);
    acc_a += xv * static_cast<float>(a_weight[w_base + col]);
    acc_b += xv * static_cast<float>(b_weight[w_base + col]);
  }
  a_sums[threadIdx.x] = acc_a;
  b_sums[threadIdx.x] = acc_b;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      a_sums[threadIdx.x] += a_sums[threadIdx.x + stride];
      b_sums[threadIdx.x] += b_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int idx = row * heads + head;
    const float av = static_cast<float>(hip_bfloat16(a_sums[0]));
    const float bv = static_cast<float>(hip_bfloat16(b_sums[0]));
    const float alog = static_cast<float>(A_log[head]);
    const float dt = static_cast<float>(dt_bias[head]);
    beta[idx] = hip_bfloat16(sigmoidf_fast(bv));
    g[idx] = hip_bfloat16(-expf(alog) * softplusf_stable(av + dt));
  }
}

__global__ void qwen36_rms_norm_gated_silu_bf16_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* z,
    const hip_bfloat16* weight,
    hip_bfloat16* out,
    int rows,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float smem[];
  float sum = 0.0f;
  const int base = row * dim;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
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
  const float inv = rsqrtf(smem[0] / static_cast<float>(dim) + eps);
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    const int idx = base + col;
    const float xv = static_cast<float>(x[idx]);
    const float zv = static_cast<float>(z[idx]);
    const float wv = static_cast<float>(weight[col]);
    out[idx] = hip_bfloat16((xv * inv * wv) * siluf_fast(zv));
  }
}

__global__ void qwen36_rms_norm_gated_silu_quant_fp8_fnuz_1x128_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* z,
    const hip_bfloat16* weight,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int heads,
    int dim,
    float eps) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  if (row >= rows || head >= heads || dim != 128) {
    return;
  }
  extern __shared__ float smem[];
  const int base = (row * heads + head) * dim;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    const float v = static_cast<float>(x[base + col]);
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
  const float inv = rsqrtf(smem[0] / static_cast<float>(dim) + eps);

  float local_amax = 0.0f;
  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    const int idx = base + col;
    const float xv = static_cast<float>(x[idx]);
    const float zv = static_cast<float>(z[idx]);
    const float wv = static_cast<float>(weight[col]);
    const hip_bfloat16 y_bf16 = hip_bfloat16((xv * inv * wv) * siluf_fast(zv));
    local_amax = fmaxf(local_amax, fabsf(static_cast<float>(y_bf16)));
  }
  smem[threadIdx.x] = local_amax;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(smem[0] / 240.0f, 1.0e-8f);
  if (threadIdx.x == 0) {
    scale[row * heads + head] = s;
  }
  __syncthreads();

  for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    const int idx = base + col;
    const float xv = static_cast<float>(x[idx]);
    const float zv = static_cast<float>(z[idx]);
    const float wv = static_cast<float>(weight[col]);
    const hip_bfloat16 y_bf16 = hip_bfloat16((xv * inv * wv) * siluf_fast(zv));
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  }
}

__global__ void qwen36_rms_norm_quant_fp8_fnuz_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float smem[];
  float* sums = smem;
  float* block_amax = smem + blockDim.x;
  const int block_count = hidden / 128;
  const int row_base = row * hidden;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[row_base + col]);
    sum += v * v;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(hidden) + eps);

  for (int b = threadIdx.x; b < block_count; b += blockDim.x) {
    block_amax[b] = 0.0f;
  }
  __syncthreads();

  for (int b = 0; b < block_count; ++b) {
    float local = 0.0f;
    const int block_base = row_base + b * 128;
    for (int j = threadIdx.x; j < 128; j += blockDim.x) {
      const int col = b * 128 + j;
      const float xv = static_cast<float>(x[block_base + j]);
      const float wv = static_cast<float>(weight[col]);
      local = fmaxf(local, fabsf(xv * inv * wv));
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] = fmaxf(sums[threadIdx.x], sums[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      block_amax[b] = sums[0];
    }
    __syncthreads();
  }

  for (int b = threadIdx.x; b < block_count; b += blockDim.x) {
    scale[row * block_count + b] = fmaxf(block_amax[b] / 240.0f, 1.0e-8f);
  }
  __syncthreads();

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = row_base + col;
    const int b = col / 128;
    const float xv = static_cast<float>(x[idx]);
    const float wv = static_cast<float>(weight[col]);
    const float s = scale[row * block_count + b];
    out[idx] = __hip_fp8_e4m3_fnuz((xv * inv * wv) / s);
  }
}

__global__ void qwen36_rms_norm_bf16_quant_fp8_fnuz_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* weight,
    hip_bfloat16* out_bf16,
    __hip_fp8_e4m3_fnuz* out_fp8,
    float* scale,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float smem[];
  float* sums = smem;
  float* block_amax = smem + blockDim.x;
  const int block_count = hidden / 128;
  const int row_base = row * hidden;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const float v = static_cast<float>(x[row_base + col]);
    sum += v * v;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(hidden) + eps);

  for (int b = threadIdx.x; b < block_count; b += blockDim.x) {
    block_amax[b] = 0.0f;
  }
  __syncthreads();

  for (int b = 0; b < block_count; ++b) {
    float local = 0.0f;
    const int block_base = row_base + b * 128;
    for (int j = threadIdx.x; j < 128; j += blockDim.x) {
      const int col = b * 128 + j;
      const float xv = static_cast<float>(x[block_base + j]);
      const float wv = static_cast<float>(weight[col]);
      local = fmaxf(local, fabsf(xv * inv * wv));
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] = fmaxf(sums[threadIdx.x], sums[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      block_amax[b] = sums[0];
    }
    __syncthreads();
  }

  for (int b = threadIdx.x; b < block_count; b += blockDim.x) {
    scale[row * block_count + b] = fmaxf(block_amax[b] / 240.0f, 1.0e-8f);
  }
  __syncthreads();

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = row_base + col;
    const int b = col / 128;
    const float xv = static_cast<float>(x[idx]);
    const float wv = static_cast<float>(weight[col]);
    const float y = xv * inv * wv;
    const float s = scale[row * block_count + b];
    out_bf16[idx] = hip_bfloat16(y);
    out_fp8[idx] = __hip_fp8_e4m3_fnuz(y / s);
  }
}

__global__ void qwen36_add_rms_norm_bf16_quant_fp8_fnuz_kernel(
    const hip_bfloat16* a,
    const hip_bfloat16* b,
    const hip_bfloat16* weight,
    hip_bfloat16* residual_out,
    hip_bfloat16* out_bf16,
    __hip_fp8_e4m3_fnuz* out_fp8,
    float* scale,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float smem[];
  float* sums = smem;
  float* block_amax = smem + blockDim.x;
  const int block_count = hidden / 128;
  const int row_base = row * hidden;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = row_base + col;
    const hip_bfloat16 residual =
        hip_bfloat16(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
    residual_out[idx] = residual;
    const float v = static_cast<float>(residual);
    sum += v * v;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(hidden) + eps);

  for (int block = threadIdx.x; block < block_count; block += blockDim.x) {
    block_amax[block] = 0.0f;
  }
  __syncthreads();

  for (int block = 0; block < block_count; ++block) {
    float local = 0.0f;
    const int block_base = row_base + block * 128;
    for (int j = threadIdx.x; j < 128; j += blockDim.x) {
      const int col = block * 128 + j;
      const float xv = static_cast<float>(residual_out[block_base + j]);
      const float wv = static_cast<float>(weight[col]);
      local = fmaxf(local, fabsf(xv * inv * wv));
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        sums[threadIdx.x] = fmaxf(sums[threadIdx.x], sums[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      block_amax[block] = sums[0];
    }
    __syncthreads();
  }

  for (int block = threadIdx.x; block < block_count; block += blockDim.x) {
    scale[row * block_count + block] =
        fmaxf(block_amax[block] / 240.0f, 1.0e-8f);
  }
  __syncthreads();

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int idx = row_base + col;
    const int block = col / 128;
    const float xv = static_cast<float>(residual_out[idx]);
    const float wv = static_cast<float>(weight[col]);
    const float y = xv * inv * wv;
    const float s = scale[row * block_count + block];
    out_bf16[idx] = hip_bfloat16(y);
    out_fp8[idx] = __hip_fp8_e4m3_fnuz(y / s);
  }
}

__global__ void qwen36_quant_fp8_fnuz_1x128_kernel(
    const hip_bfloat16* x,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  const int block_id = blockIdx.y;
  if (row >= rows || block_id >= hidden / 128) {
    return;
  }
  extern __shared__ float smem[];
  const int base = row * hidden + block_id * 128;
  float local = 0.0f;
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    local = fmaxf(local, fabsf(static_cast<float>(x[base + j])));
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(smem[0] / 240.0f, 1.0e-8f);
  if (threadIdx.x == 0) {
    scale[row * (hidden / 128) + block_id] = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    out[base + j] = __hip_fp8_e4m3_fnuz(static_cast<float>(x[base + j]) / s);
  }
}

__global__ void qwen36_fp8_blockscale_matvec_bf16_kernel(
    const __hip_fp8_e4m3_fnuz* x,
    const __hip_fp8_e4m3_fnuz* weight,
    const float* x_scale,
    const float* weight_scale,
    hip_bfloat16* out,
    int out_features,
    int hidden) {
  const int row = blockIdx.x;
  if (row >= out_features) {
    return;
  }
  extern __shared__ float smem[];
  const int k_blocks = hidden / 128;
  const int row_block = row / 128;
  float acc = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    const int kb = col / 128;
    const float xs = x_scale[kb];
    const float ws = weight_scale[row_block * k_blocks + kb];
    const float xv = static_cast<float>(x[col]) * xs;
    const float wv = static_cast<float>(weight[row * hidden + col]) * ws;
    acc += xv * wv;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[row] = hip_bfloat16(smem[0]);
  }
}

__global__ void qwen36_silu_mul_quant_fp8_fnuz_1x128_kernel(
    const hip_bfloat16* gate,
    const hip_bfloat16* up,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  const int block_id = blockIdx.y;
  if (row >= rows || block_id >= hidden / 128) {
    return;
  }
  extern __shared__ float smem[];
  const int base = row * hidden + block_id * 128;
  float local = 0.0f;
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    const int idx = base + j;
    const float gv = static_cast<float>(gate[idx]);
    const float uv = static_cast<float>(up[idx]);
    const hip_bfloat16 y_bf16 = hip_bfloat16(siluf_fast(gv) * uv);
    local = fmaxf(local, fabsf(static_cast<float>(y_bf16)));
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(smem[0] / 240.0f, 1.0e-8f);
  if (threadIdx.x == 0) {
    scale[row * (hidden / 128) + block_id] = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    const int idx = base + j;
    const float gv = static_cast<float>(gate[idx]);
    const float uv = static_cast<float>(up[idx]);
    const hip_bfloat16 y_bf16 = hip_bfloat16(siluf_fast(gv) * uv);
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  }
}

__global__ void qwen36_sigmoid_mul_quant_fp8_fnuz_1x128_kernel(
    const hip_bfloat16* x,
    const hip_bfloat16* gate,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int hidden) {
  const int row = blockIdx.x;
  const int block_id = blockIdx.y;
  if (row >= rows || block_id >= hidden / 128) {
    return;
  }
  extern __shared__ float smem[];
  const int base = row * hidden + block_id * 128;
  float local = 0.0f;
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    const int idx = base + j;
    const float xv = static_cast<float>(x[idx]);
    const float gv = static_cast<float>(gate[idx]);
    const hip_bfloat16 y_bf16 = hip_bfloat16(xv * sigmoidf_fast(gv));
    local = fmaxf(local, fabsf(static_cast<float>(y_bf16)));
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(smem[0] / 240.0f, 1.0e-8f);
  if (threadIdx.x == 0) {
    scale[row * (hidden / 128) + block_id] = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < 128; j += blockDim.x) {
    const int idx = base + j;
    const float xv = static_cast<float>(x[idx]);
    const float gv = static_cast<float>(gate[idx]);
    const hip_bfloat16 y_bf16 = hip_bfloat16(xv * sigmoidf_fast(gv));
    out[idx] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  }
}

__global__ void qwen36_gated_deltanet_recurrent_bf16_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    float* state,
    hip_bfloat16* out,
    int rows,
    int heads,
    int dim) {
  const int head = blockIdx.x;
  if (head >= heads || threadIdx.x >= dim) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + blockDim.x;
  const int e = threadIdx.x;
  float* head_state = state + head * dim * dim;

  for (int row = 0; row < rows; ++row) {
    const int base = (row * heads + head) * dim;
    const float q_e = static_cast<float>(q[base + e]);
    const float k_e = static_cast<float>(k[base + e]);
    q_sums[threadIdx.x] = q_e * q_e;
    k_sums[threadIdx.x] = k_e * k_e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        q_sums[threadIdx.x] += q_sums[threadIdx.x + stride];
        k_sums[threadIdx.x] += k_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    const float beta_h = static_cast<float>(beta[row * heads + head]);
    const float decay = expf(fminf(fmaxf(static_cast<float>(g[row * heads + head]),
                                         -30.0f),
                                  0.0f));

    float pred_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      const float kd = static_cast<float>(k[base + d]) * k_inv;
      pred_e += kd * head_state[d * dim + e];
    }
    const float delta_e =
        (static_cast<float>(v[base + e]) - pred_e) * beta_h;
    for (int d = 0; d < dim; ++d) {
      const float kd = static_cast<float>(k[base + d]) * k_inv;
      head_state[d * dim + e] = head_state[d * dim + e] * decay + kd * delta_e;
    }
    __syncthreads();

    float out_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      const float qd = static_cast<float>(q[base + d]) * q_inv;
      out_e += qd * head_state[d * dim + e];
    }
    out[base + e] = hip_bfloat16(out_e);
    __syncthreads();
  }
}

__global__ void qwen36_gated_deltanet_recurrent_bf16_v2_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    float* state,
    hip_bfloat16* out,
    int rows,
    int heads,
    int dim) {
  const int head = blockIdx.x;
  const int e = threadIdx.x;
  if (head >= heads || e >= dim) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + blockDim.x;
  float* q_norm = smem + 2 * blockDim.x;
  float* k_norm = smem + 3 * blockDim.x;
  float* head_state = state + head * dim * dim;

  for (int row = 0; row < rows; ++row) {
    const int base = (row * heads + head) * dim;
    const float q_e = static_cast<float>(q[base + e]);
    const float k_e = static_cast<float>(k[base + e]);
    q_sums[e] = q_e * q_e;
    k_sums[e] = k_e * k_e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (e < stride) {
        q_sums[e] += q_sums[e + stride];
        k_sums[e] += k_sums[e + stride];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    q_norm[e] = q_e * q_inv;
    k_norm[e] = k_e * k_inv;
    __syncthreads();

    const float beta_h = static_cast<float>(beta[row * heads + head]);
    const float decay = expf(fminf(fmaxf(static_cast<float>(g[row * heads + head]),
                                         -30.0f),
                                  0.0f));

    float pred_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      pred_e += k_norm[d] * head_state[d * dim + e];
    }
    const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
    for (int d = 0; d < dim; ++d) {
      head_state[d * dim + e] = head_state[d * dim + e] * decay + k_norm[d] * delta_e;
    }

    float out_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      out_e += q_norm[d] * head_state[d * dim + e];
    }
    out[base + e] = hip_bfloat16(out_e);
    __syncthreads();
  }
}

__global__ void qwen36_gated_deltanet_recurrent_broadcast3_bf16_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    float* state,
    hip_bfloat16* out,
    int groups,
    int dim) {
  const int group = blockIdx.x;
  const int e = threadIdx.x;
  const int rep = threadIdx.y;
  if (group >= groups || e >= dim || rep >= 3) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + dim;
  float* q_norm = smem + 2 * dim;
  float* k_norm = smem + 3 * dim;
  const int head_base = group * 3;
  const int qk_base = head_base * dim;

  if (rep == 0) {
    const float q_e = static_cast<float>(q[qk_base + e]);
    const float k_e = static_cast<float>(k[qk_base + e]);
    q_sums[e] = q_e * q_e;
    k_sums[e] = k_e * k_e;
  }
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (rep == 0 && e < stride) {
      q_sums[e] += q_sums[e + stride];
      k_sums[e] += k_sums[e + stride];
    }
    __syncthreads();
  }
  if (rep == 0) {
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    q_norm[e] = static_cast<float>(q[qk_base + e]) * q_inv;
    k_norm[e] = static_cast<float>(k[qk_base + e]) * k_inv;
  }
  __syncthreads();

  const int head = head_base + rep;
  const int base = head * dim;
  float* head_state = state + head * dim * dim;
  const float beta_h = static_cast<float>(beta[head]);
  const float decay = expf(fminf(fmaxf(static_cast<float>(g[head]), -30.0f), 0.0f));

  float pred_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    pred_e += k_norm[d] * head_state[d * dim + e];
  }
  const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
  for (int d = 0; d < dim; ++d) {
    head_state[d * dim + e] = head_state[d * dim + e] * decay + k_norm[d] * delta_e;
  }
  __syncthreads();

  float out_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    out_e += q_norm[d] * head_state[d * dim + e];
  }
  out[base + e] = hip_bfloat16(out_e);
  __syncthreads();
}

__global__ void qwen36_gated_deltanet_recurrent_norm_quant_fp8_fnuz_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    const hip_bfloat16* z,
    const hip_bfloat16* norm_weight,
    float* state,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int heads,
    int dim,
    float eps) {
  const int head = blockIdx.x;
  const int e = threadIdx.x;
  if (head >= heads || e >= dim) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + blockDim.x;
  float* q_norm = smem + 2 * blockDim.x;
  float* k_norm = smem + 3 * blockDim.x;
  float* head_state = state + head * dim * dim;

  const int base = head * dim;
  const float q_e = static_cast<float>(q[base + e]);
  const float k_e = static_cast<float>(k[base + e]);
  q_sums[e] = q_e * q_e;
  k_sums[e] = k_e * k_e;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      q_sums[e] += q_sums[e + stride];
      k_sums[e] += k_sums[e + stride];
    }
    __syncthreads();
  }
  const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
  const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
  q_norm[e] = q_e * q_inv;
  k_norm[e] = k_e * k_inv;
  __syncthreads();

  const float decay = expf(fminf(fmaxf(static_cast<float>(g[head]), -30.0f), 0.0f));
  const float beta_h = static_cast<float>(beta[head]);

  float pred_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    pred_e += k_norm[d] * head_state[d * dim + e];
  }
  const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
  for (int d = 0; d < dim; ++d) {
    head_state[d * dim + e] = head_state[d * dim + e] * decay + k_norm[d] * delta_e;
  }
  __syncthreads();

  float out_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    out_e += q_norm[d] * head_state[d * dim + e];
  }
  const float attn_bf16 = static_cast<float>(hip_bfloat16(out_e));
  q_norm[e] = attn_bf16;
  q_sums[e] = attn_bf16 * attn_bf16;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      q_sums[e] += q_sums[e + stride];
    }
    __syncthreads();
  }
  const float norm_inv = rsqrtf(q_sums[0] / static_cast<float>(dim) + eps);
  const float zv = static_cast<float>(z[base + e]);
  const float wv = static_cast<float>(norm_weight[e]);
  const hip_bfloat16 y_bf16 = hip_bfloat16((attn_bf16 * norm_inv * wv) * siluf_fast(zv));
  k_sums[e] = fabsf(static_cast<float>(y_bf16));
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      k_sums[e] = fmaxf(k_sums[e], k_sums[e + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(k_sums[0] / 240.0f, 1.0e-8f);
  if (e == 0) {
    scale[head] = s;
  }
  out[base + e] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  __syncthreads();
}

__global__ void qwen36_gated_deltanet_recurrent_norm_quant_seq_fp8_fnuz_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    const hip_bfloat16* z,
    const hip_bfloat16* norm_weight,
    float* state,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int rows,
    int heads,
    int dim,
    float eps) {
  const int head = blockIdx.x;
  const int e = threadIdx.x;
  if (head >= heads || e >= dim) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + blockDim.x;
  float* q_norm = smem + 2 * blockDim.x;
  float* k_norm = smem + 3 * blockDim.x;
  float* head_state = state + head * dim * dim;

  for (int row = 0; row < rows; ++row) {
    const int base = (row * heads + head) * dim;
    const int gate_base = row * heads + head;
    const float q_e = static_cast<float>(q[base + e]);
    const float k_e = static_cast<float>(k[base + e]);
    q_sums[e] = q_e * q_e;
    k_sums[e] = k_e * k_e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (e < stride) {
        q_sums[e] += q_sums[e + stride];
        k_sums[e] += k_sums[e + stride];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    q_norm[e] = q_e * q_inv;
    k_norm[e] = k_e * k_inv;
    __syncthreads();

    const float decay =
        expf(fminf(fmaxf(static_cast<float>(g[gate_base]), -30.0f), 0.0f));
    const float beta_h = static_cast<float>(beta[gate_base]);

    float pred_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      pred_e += k_norm[d] * head_state[d * dim + e];
    }
    const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
    for (int d = 0; d < dim; ++d) {
      head_state[d * dim + e] =
          head_state[d * dim + e] * decay + k_norm[d] * delta_e;
    }
    __syncthreads();

    float out_e = 0.0f;
    for (int d = 0; d < dim; ++d) {
      out_e += q_norm[d] * head_state[d * dim + e];
    }
    const float attn_bf16 = static_cast<float>(hip_bfloat16(out_e));
    q_sums[e] = attn_bf16 * attn_bf16;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (e < stride) {
        q_sums[e] += q_sums[e + stride];
      }
      __syncthreads();
    }
    const float norm_inv =
        rsqrtf(q_sums[0] / static_cast<float>(dim) + eps);
    const float zv = static_cast<float>(z[base + e]);
    const float wv = static_cast<float>(norm_weight[e]);
    const hip_bfloat16 y_bf16 =
        hip_bfloat16((attn_bf16 * norm_inv * wv) * siluf_fast(zv));
    k_sums[e] = fabsf(static_cast<float>(y_bf16));
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (e < stride) {
        k_sums[e] = fmaxf(k_sums[e], k_sums[e + stride]);
      }
      __syncthreads();
    }
    const float s = fmaxf(k_sums[0] / 240.0f, 1.0e-8f);
    if (e == 0) {
      scale[row * heads + head] = s;
    }
    out[base + e] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
    __syncthreads();
  }
}

__global__ void qwen36_gated_deltanet_recurrent_broadcast3_norm_quant_fp8_fnuz_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    const hip_bfloat16* z,
    const hip_bfloat16* norm_weight,
    float* state,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int groups,
    int dim,
    float eps) {
  const int group = blockIdx.x;
  const int e = threadIdx.x;
  const int rep = threadIdx.y;
  if (group >= groups || e >= dim || rep >= 3) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + dim;
  float* q_norm = smem + 2 * dim;
  float* k_norm = smem + 3 * dim;
  float* norm_sums = smem + 4 * dim;
  float* amax_sums = smem + 7 * dim;
  const int head_base = group * 3;
  const int qk_base = head_base * dim;

  if (rep == 0) {
    const float q_e = static_cast<float>(q[qk_base + e]);
    const float k_e = static_cast<float>(k[qk_base + e]);
    q_sums[e] = q_e * q_e;
    k_sums[e] = k_e * k_e;
  }
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (rep == 0 && e < stride) {
      q_sums[e] += q_sums[e + stride];
      k_sums[e] += k_sums[e + stride];
    }
    __syncthreads();
  }
  if (rep == 0) {
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    q_norm[e] = static_cast<float>(q[qk_base + e]) * q_inv;
    k_norm[e] = static_cast<float>(k[qk_base + e]) * k_inv;
  }
  __syncthreads();

  const int head = head_base + rep;
  const int base = head * dim;
  float* head_state = state + head * dim * dim;
  const float beta_h = static_cast<float>(beta[head]);
  const float decay = expf(fminf(fmaxf(static_cast<float>(g[head]), -30.0f), 0.0f));

  float pred_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    pred_e += k_norm[d] * head_state[d * dim + e];
  }
  const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
  for (int d = 0; d < dim; ++d) {
    head_state[d * dim + e] = head_state[d * dim + e] * decay + k_norm[d] * delta_e;
  }
  __syncthreads();

  float out_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    out_e += q_norm[d] * head_state[d * dim + e];
  }
  const float attn_bf16 = static_cast<float>(hip_bfloat16(out_e));
  norm_sums[rep * dim + e] = attn_bf16 * attn_bf16;
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      norm_sums[rep * dim + e] += norm_sums[rep * dim + e + stride];
    }
    __syncthreads();
  }

  const float norm_inv = rsqrtf(norm_sums[rep * dim] / static_cast<float>(dim) + eps);
  const float zv = static_cast<float>(z[base + e]);
  const float wv = static_cast<float>(norm_weight[e]);
  const hip_bfloat16 y_bf16 = hip_bfloat16((attn_bf16 * norm_inv * wv) * siluf_fast(zv));
  amax_sums[rep * dim + e] = fabsf(static_cast<float>(y_bf16));
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      amax_sums[rep * dim + e] =
          fmaxf(amax_sums[rep * dim + e], amax_sums[rep * dim + e + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(amax_sums[rep * dim] / 240.0f, 1.0e-8f);
  if (e == 0) {
    scale[head] = s;
  }
  out[base + e] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  __syncthreads();
}

__global__ void qwen36_gated_deltanet_recurrent_broadcast3_fastout_norm_quant_fp8_fnuz_kernel(
    const hip_bfloat16* q,
    const hip_bfloat16* k,
    const hip_bfloat16* v,
    const hip_bfloat16* g,
    const hip_bfloat16* beta,
    const hip_bfloat16* z,
    const hip_bfloat16* norm_weight,
    float* state,
    __hip_fp8_e4m3_fnuz* out,
    float* scale,
    int groups,
    int dim,
    float eps) {
  const int group = blockIdx.x;
  const int e = threadIdx.x;
  const int rep = threadIdx.y;
  if (group >= groups || e >= dim || rep >= 3) {
    return;
  }
  extern __shared__ float smem[];
  float* q_sums = smem;
  float* k_sums = smem + dim;
  float* q_norm = smem + 2 * dim;
  float* k_norm = smem + 3 * dim;
  float* norm_sums = smem + 4 * dim;
  float* amax_sums = smem + 7 * dim;
  const int head_base = group * 3;
  const int qk_base = head_base * dim;

  if (rep == 0) {
    const float q_e = static_cast<float>(q[qk_base + e]);
    const float k_e = static_cast<float>(k[qk_base + e]);
    q_sums[e] = q_e * q_e;
    k_sums[e] = k_e * k_e;
  }
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (rep == 0 && e < stride) {
      q_sums[e] += q_sums[e + stride];
      k_sums[e] += k_sums[e + stride];
    }
    __syncthreads();
  }
  if (rep == 0) {
    const float q_inv = rsqrtf(q_sums[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_sums[0] + 1.0e-6f);
    q_norm[e] = static_cast<float>(q[qk_base + e]) * q_inv;
    k_norm[e] = static_cast<float>(k[qk_base + e]) * k_inv;
    q_sums[e] = q_norm[e] * k_norm[e];
  }
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (rep == 0 && e < stride) {
      q_sums[e] += q_sums[e + stride];
    }
    __syncthreads();
  }
  const float qk_dot = q_sums[0];

  const int head = head_base + rep;
  const int base = head * dim;
  float* head_state = state + head * dim * dim;
  const float beta_h = static_cast<float>(beta[head]);
  const float decay = expf(fminf(fmaxf(static_cast<float>(g[head]), -30.0f), 0.0f));

  float pred_e = 0.0f;
  float q_prev_e = 0.0f;
  for (int d = 0; d < dim; ++d) {
    const float sde = head_state[d * dim + e];
    pred_e += k_norm[d] * sde;
    q_prev_e += q_norm[d] * sde;
  }
  const float delta_e = (static_cast<float>(v[base + e]) - pred_e) * beta_h;
  for (int d = 0; d < dim; ++d) {
    head_state[d * dim + e] = head_state[d * dim + e] * decay + k_norm[d] * delta_e;
  }

  const float attn_bf16 = static_cast<float>(hip_bfloat16(decay * q_prev_e + qk_dot * delta_e));
  norm_sums[rep * dim + e] = attn_bf16 * attn_bf16;
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      norm_sums[rep * dim + e] += norm_sums[rep * dim + e + stride];
    }
    __syncthreads();
  }

  const float norm_inv = rsqrtf(norm_sums[rep * dim] / static_cast<float>(dim) + eps);
  const float zv = static_cast<float>(z[base + e]);
  const float wv = static_cast<float>(norm_weight[e]);
  const hip_bfloat16 y_bf16 = hip_bfloat16((attn_bf16 * norm_inv * wv) * siluf_fast(zv));
  amax_sums[rep * dim + e] = fabsf(static_cast<float>(y_bf16));
  __syncthreads();
  for (int stride = dim / 2; stride > 0; stride >>= 1) {
    if (e < stride) {
      amax_sums[rep * dim + e] =
          fmaxf(amax_sums[rep * dim + e], amax_sums[rep * dim + e + stride]);
    }
    __syncthreads();
  }
  const float s = fmaxf(amax_sums[rep * dim] / 240.0f, 1.0e-8f);
  if (e == 0) {
    scale[head] = s;
  }
  out[base + e] = __hip_fp8_e4m3_fnuz(static_cast<float>(y_bf16) / s);
  __syncthreads();
}

}  // namespace

void launch_qwen36_causal_conv1d_bf16(
    const void* x, const void* weight, void* out,
    int rows, int dim, int kernel_size, bool apply_silu, hipStream_t stream) {
  if (rows <= 0 || dim <= 0 || kernel_size <= 0) {
    return;
  }
  const int total = rows * dim;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_causal_conv1d_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, dim, kernel_size, apply_silu);
}

void launch_qwen36_causal_conv1d_state_bf16(
    const void* x, const void* weight, const void* state, void* new_state, void* out,
    int rows, int dim, int kernel_size, bool apply_silu, hipStream_t stream) {
  if (rows <= 0 || dim <= 0 || kernel_size <= 0) {
    return;
  }
  const int total = rows * dim + (kernel_size - 1) * dim;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_causal_conv1d_state_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(state),
                     static_cast<hip_bfloat16*>(new_state),
                     static_cast<hip_bfloat16*>(out),
                     rows, dim, kernel_size, apply_silu);
}

void launch_qwen36_causal_conv1d_state_split_qkv_bf16(
    const void* x, const void* weight, const void* state, void* new_state,
    void* q, void* k, void* v, int rows, int dim, int kernel_size,
    hipStream_t stream) {
  if (rows <= 0 || dim <= 0 || kernel_size <= 0) {
    return;
  }
  const int total = rows * dim + (kernel_size - 1) * dim;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_causal_conv1d_state_split_qkv_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<const hip_bfloat16*>(state),
                     static_cast<hip_bfloat16*>(new_state),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     static_cast<hip_bfloat16*>(v),
                     rows, dim, kernel_size);
}

void launch_qwen36_causal_conv1d_state_split_qkv_bf16_inplace(
    const void* x, const void* weight, void* state,
    void* q, void* k, void* v, int dim, int kernel_size,
    hipStream_t stream) {
  if (dim <= 0 || kernel_size != 4) {
    return;
  }
  const int blocks = (dim + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_causal_conv1d_state_split_qkv_bf16_inplace_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(state),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     static_cast<hip_bfloat16*>(v),
                     dim, kernel_size);
}

void launch_qwen36_lin_split_qkv_broadcast_bf16(
    const void* qkv, void* q, void* k, void* v, int rows,
    hipStream_t stream) {
  if (rows <= 0) {
    return;
  }
  const int total = rows * 48 * 128;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_lin_split_qkv_broadcast_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(qkv),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k),
                     static_cast<hip_bfloat16*>(v),
                     rows);
}

void launch_qwen36_gdn_gating_bf16(
    const void* a, const void* b, const void* A_log, const void* dt_bias,
    void* g, void* beta, int rows, int heads, hipStream_t stream) {
  if (rows <= 0 || heads <= 0) {
    return;
  }
  const int total = rows * heads;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_gdn_gating_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(a),
                     static_cast<const hip_bfloat16*>(b),
                     static_cast<const hip_bfloat16*>(A_log),
                     static_cast<const hip_bfloat16*>(dt_bias),
                     static_cast<hip_bfloat16*>(g),
                     static_cast<hip_bfloat16*>(beta),
                     rows, heads);
}

void launch_qwen36_silu_mul_bf16(
    const void* gate, const void* up, void* out,
    int total, hipStream_t stream) {
  if (total <= 0) {
    return;
  }
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_silu_mul_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(up),
                     static_cast<hip_bfloat16*>(out),
                     total);
}

void launch_qwen36_sigmoid_mul_bf16(
    const void* x, const void* gate, void* out,
    int total, hipStream_t stream) {
  if (total <= 0) {
    return;
  }
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_sigmoid_mul_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<hip_bfloat16*>(out),
                     total);
}

void launch_qwen36_add_bf16(
    const void* a, const void* b, void* out,
    int total, hipStream_t stream) {
  if (total <= 0) {
    return;
  }
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_add_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(a),
                     static_cast<const hip_bfloat16*>(b),
                     static_cast<hip_bfloat16*>(out),
                     total);
}

void launch_qwen36_copy_bf16(
    const void* x, void* out, int total, hipStream_t stream) {
  if (total <= 0) {
    return;
  }
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_copy_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<hip_bfloat16*>(out),
                     total);
}

void launch_qwen36_full_v_broadcast_bf16(
    const void* v, void* out,
    int rows, int kv_heads, int q_heads, int head_dim,
    hipStream_t stream) {
  if (rows <= 0 || kv_heads <= 0 || q_heads <= 0 || head_dim <= 0) {
    return;
  }
  const int total = rows * q_heads * head_dim;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(qwen36_full_v_broadcast_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<hip_bfloat16*>(out),
                     rows, kv_heads, q_heads, head_dim);
}

void launch_qwen36_full_qk_norm_partial_rope_cache_bf16(
    const void* q_gate, const void* k_proj, const void* v_proj,
    const float* cos, const float* sin,
    const void* q_norm_w, const void* k_norm_w,
    void* q, void* k_cache, void* v_cache,
    int layer_idx, int max_seq, int kv_start, int rows,
    int q_heads, int kv_heads, int head_dim, int rotary_dim,
    float eps, hipStream_t stream) {
  if (rows <= 0 || q_heads <= 0 || kv_heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0) {
    return;
  }
  const int total_heads = rows * (q_heads + kv_heads);
  hipLaunchKernelGGL(qwen36_full_qk_norm_partial_rope_cache_bf16_kernel,
                     dim3(total_heads), dim3(kBlock), kBlock * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(q_gate),
                     static_cast<const hip_bfloat16*>(k_proj),
                     static_cast<const hip_bfloat16*>(v_proj),
                     cos, sin,
                     static_cast<const hip_bfloat16*>(q_norm_w),
                     static_cast<const hip_bfloat16*>(k_norm_w),
                     static_cast<hip_bfloat16*>(q),
                     static_cast<hip_bfloat16*>(k_cache),
                     static_cast<hip_bfloat16*>(v_cache),
                     layer_idx, max_seq, kv_start, rows,
                     q_heads, kv_heads, head_dim, rotary_dim, eps);
}

void launch_qwen36_small_linear_bf16(
    const void* x, const void* weight, void* out,
    int rows, int out_features, int hidden, hipStream_t stream) {
  if (rows <= 0 || out_features <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_small_linear_bf16_kernel,
                     dim3(rows, out_features), dim3(kBlock),
                     kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, out_features, hidden);
}

void launch_qwen36_in_proj_ab_gating_bf16(
    const void* x, const void* a_weight, const void* b_weight,
    const void* A_log, const void* dt_bias, void* g, void* beta,
    int rows, int heads, int hidden, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || hidden <= 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_in_proj_ab_gating_bf16_kernel,
                     dim3(rows, heads), dim3(kBlock),
                     2 * kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(a_weight),
                     static_cast<const hip_bfloat16*>(b_weight),
                     static_cast<const hip_bfloat16*>(A_log),
                     static_cast<const hip_bfloat16*>(dt_bias),
                     static_cast<hip_bfloat16*>(g),
                     static_cast<hip_bfloat16*>(beta),
                     rows, heads, hidden);
}

void launch_qwen36_rms_norm_gated_silu_bf16(
    const void* x, const void* z, const void* weight, void* out,
    int rows, int dim, float eps, hipStream_t stream) {
  if (rows <= 0 || dim <= 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_rms_norm_gated_silu_bf16_kernel,
                     dim3(rows), dim3(kBlock), kBlock * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out),
                     rows, dim, eps);
}

void launch_qwen36_rms_norm_gated_silu_quant_fp8_fnuz_1x128(
    const void* x, const void* z, const void* weight, void* out,
    float* scale, int rows, int heads, int dim, float eps,
    hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim != 128) {
    return;
  }
  hipLaunchKernelGGL(qwen36_rms_norm_gated_silu_quant_fp8_fnuz_1x128_kernel,
                     dim3(rows, heads), dim3(128),
                     128 * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, heads, dim, eps);
}

void launch_qwen36_rms_norm_quant_fp8_fnuz(
    const void* x, const void* weight, void* out, float* scale,
    int rows, int hidden, float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  const int block_count = hidden / 128;
  const std::size_t shared = (kBlock + block_count) * sizeof(float);
  hipLaunchKernelGGL(qwen36_rms_norm_quant_fp8_fnuz_kernel,
                     dim3(rows), dim3(kBlock), shared, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden, eps);
}

void launch_qwen36_rms_norm_bf16_quant_fp8_fnuz(
    const void* x, const void* weight, void* out_bf16,
    void* out_fp8, float* scale, int rows, int hidden,
    float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  const int block_count = hidden / 128;
  const std::size_t shared = (kBlock + block_count) * sizeof(float);
  hipLaunchKernelGGL(qwen36_rms_norm_bf16_quant_fp8_fnuz_kernel,
                     dim3(rows), dim3(kBlock), shared, stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(out_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     scale,
                     rows, hidden, eps);
}

void launch_qwen36_add_rms_norm_bf16_quant_fp8_fnuz(
    const void* a, const void* b, const void* weight, void* residual_out,
    void* out_bf16, void* out_fp8, float* scale, int rows, int hidden,
    float eps, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  const int block_count = hidden / 128;
  const std::size_t shared = (kBlock + block_count) * sizeof(float);
  hipLaunchKernelGGL(qwen36_add_rms_norm_bf16_quant_fp8_fnuz_kernel,
                     dim3(rows), dim3(kBlock), shared, stream,
                     static_cast<const hip_bfloat16*>(a),
                     static_cast<const hip_bfloat16*>(b),
                     static_cast<const hip_bfloat16*>(weight),
                     static_cast<hip_bfloat16*>(residual_out),
                     static_cast<hip_bfloat16*>(out_bf16),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out_fp8),
                     scale,
                     rows, hidden, eps);
}

void launch_qwen36_quant_fp8_fnuz_1x128(
    const void* x, void* out, float* scale,
    int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_quant_fp8_fnuz_1x128_kernel,
                     dim3(rows, hidden / 128), dim3(128),
                     128 * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden);
}

void launch_qwen36_fp8_blockscale_matvec_bf16(
    const void* x, const void* weight, const float* x_scale,
    const float* weight_scale, void* out, int out_features, int hidden,
    hipStream_t stream) {
  if (out_features <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_fp8_blockscale_matvec_bf16_kernel,
                     dim3(out_features), dim3(kBlock),
                     kBlock * sizeof(float), stream,
                     static_cast<const __hip_fp8_e4m3_fnuz*>(x),
                     static_cast<const __hip_fp8_e4m3_fnuz*>(weight),
                     x_scale,
                     weight_scale,
                     static_cast<hip_bfloat16*>(out),
                     out_features, hidden);
}

void launch_qwen36_silu_mul_quant_fp8_fnuz_1x128(
    const void* gate, const void* up, void* out, float* scale,
    int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_silu_mul_quant_fp8_fnuz_1x128_kernel,
                     dim3(rows, hidden / 128), dim3(128),
                     128 * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<const hip_bfloat16*>(up),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden);
}

void launch_qwen36_sigmoid_mul_quant_fp8_fnuz_1x128(
    const void* x, const void* gate, void* out, float* scale,
    int rows, int hidden, hipStream_t stream) {
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_sigmoid_mul_quant_fp8_fnuz_1x128_kernel,
                     dim3(rows, hidden / 128), dim3(128),
                     128 * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(x),
                     static_cast<const hip_bfloat16*>(gate),
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     rows, hidden);
}

void launch_qwen36_gated_deltanet_recurrent_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, float* state, void* out,
    int rows, int heads, int dim, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim <= 0) {
    return;
  }
  const int threads = 128;
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_bf16_v2_kernel,
                     dim3(heads), dim3(threads), 4 * threads * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     state,
                     static_cast<hip_bfloat16*>(out),
                     rows, heads, dim);
}

void launch_qwen36_gated_deltanet_recurrent_broadcast3_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, float* state, void* out,
    int groups, int dim, hipStream_t stream) {
  if (groups <= 0 || dim != 128) {
    return;
  }
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_broadcast3_bf16_kernel,
                     dim3(groups), dim3(dim, 3),
                     4 * dim * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     state,
                     static_cast<hip_bfloat16*>(out),
                     groups, dim);
}

void launch_qwen36_gated_deltanet_recurrent_norm_quant_fp8_fnuz(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* z, const void* norm_weight,
    float* state, void* out, float* scale,
    int heads, int dim, float eps, hipStream_t stream) {
  if (heads <= 0 || dim != 128) {
    return;
  }
  const int threads = 128;
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_norm_quant_fp8_fnuz_kernel,
                     dim3(heads), dim3(threads), 4 * threads * sizeof(float),
                     stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     state,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     heads, dim, eps);
}

void launch_qwen36_gated_deltanet_recurrent_norm_quant_seq_fp8_fnuz(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* z, const void* norm_weight,
    float* state, void* out, float* scale,
    int rows, int heads, int dim, float eps, hipStream_t stream) {
  if (rows <= 0 || heads <= 0 || dim <= 0) {
    return;
  }
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_norm_quant_seq_fp8_fnuz_kernel,
                     dim3(heads), dim3(dim),
                     4 * dim * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     state,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale, rows, heads, dim, eps);
}

void launch_qwen36_gated_deltanet_recurrent_broadcast3_norm_quant_fp8_fnuz(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* z, const void* norm_weight,
    float* state, void* out, float* scale,
    int groups, int dim, float eps, hipStream_t stream) {
  if (groups <= 0 || dim != 128) {
    return;
  }
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_broadcast3_norm_quant_fp8_fnuz_kernel,
                     dim3(groups), dim3(dim, 3),
                     10 * dim * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     state,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     groups, dim, eps);
}

void launch_qwen36_gated_deltanet_recurrent_broadcast3_fastout_norm_quant_fp8_fnuz(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* z, const void* norm_weight,
    float* state, void* out, float* scale,
    int groups, int dim, float eps, hipStream_t stream) {
  if (groups <= 0 || dim != 128) {
    return;
  }
  hipLaunchKernelGGL(qwen36_gated_deltanet_recurrent_broadcast3_fastout_norm_quant_fp8_fnuz_kernel,
                     dim3(groups), dim3(dim, 3),
                     10 * dim * sizeof(float), stream,
                     static_cast<const hip_bfloat16*>(q),
                     static_cast<const hip_bfloat16*>(k),
                     static_cast<const hip_bfloat16*>(v),
                     static_cast<const hip_bfloat16*>(g),
                     static_cast<const hip_bfloat16*>(beta),
                     static_cast<const hip_bfloat16*>(z),
                     static_cast<const hip_bfloat16*>(norm_weight),
                     state,
                     static_cast<__hip_fp8_e4m3_fnuz*>(out),
                     scale,
                     groups, dim, eps);
}
