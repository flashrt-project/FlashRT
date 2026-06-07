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
