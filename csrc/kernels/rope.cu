// ================================================================
// FlashRT — RoPE kernels
// Standard RoPE, QKV split, fused QKV split + RoPE
// ================================================================

#include "rope.cuh"
#include "common.cuh"

// ── RoPE ──
__global__ void rope_kernel(const __nv_bfloat16* __restrict__ rope_weights,
                            __nv_bfloat16* __restrict__ Q,
                            __nv_bfloat16* __restrict__ K,
                            int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q = seq_len * num_heads * half_dim;
    int total_k = seq_len * half_dim;

    if (idx < total_q) {
        int seq_pos = idx / (num_heads * half_dim);
        int rem = idx % (num_heads * half_dim);
        int head = rem / half_dim;
        int d = rem % half_dim;

        int q_base = (seq_pos * num_heads + head) * head_dim;
        float q0 = bf16_to_f32(Q[q_base + 2 * d]);
        float q1 = bf16_to_f32(Q[q_base + 2 * d + 1]);
        int rope_base = seq_pos * head_dim;
        float c = bf16_to_f32(rope_weights[rope_base + 2 * d]);
        float s = bf16_to_f32(rope_weights[rope_base + 2 * d + 1]);
        Q[q_base + 2 * d]     = f32_to_bf16(q0 * c - q1 * s);
        Q[q_base + 2 * d + 1] = f32_to_bf16(q1 * c + q0 * s);
    }

    if (idx < total_k) {
        int seq_pos = idx / half_dim;
        int d = idx % half_dim;
        int k_base = seq_pos * head_dim;
        float k0 = bf16_to_f32(K[k_base + 2 * d]);
        float k1 = bf16_to_f32(K[k_base + 2 * d + 1]);
        int rope_base = seq_pos * head_dim;
        float c = bf16_to_f32(rope_weights[rope_base + 2 * d]);
        float s = bf16_to_f32(rope_weights[rope_base + 2 * d + 1]);
        K[k_base + 2 * d]     = f32_to_bf16(k0 * c - k1 * s);
        K[k_base + 2 * d + 1] = f32_to_bf16(k1 * c + k0 * s);
    }
}

void rope_apply(const __nv_bfloat16* rope_weights,
                __nv_bfloat16* Q, __nv_bfloat16* K,
                int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
    int half_dim = head_dim / 2;
    int total = seq_len * num_heads * half_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        rope_weights, Q, K, seq_len, num_heads, head_dim);
}

// ── QKV Split ──
// Generic template (dtype-agnostic: pure memcpy by column region).
template<typename T>
__global__ void qkv_split_kernel_t(const T* __restrict__ qkv,
                                    T* __restrict__ Q,
                                    T* __restrict__ K,
                                    T* __restrict__ V,
                                    int seq, int q_dim, int k_dim, int v_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx < total) {
        int row = idx / qkv_dim;
        int col = idx % qkv_dim;
        T val = qkv[idx];
        if (col < q_dim) {
            Q[row * q_dim + col] = val;
        } else if (col < q_dim + k_dim) {
            K[row * k_dim + (col - q_dim)] = val;
        } else {
            V[row * v_dim + (col - q_dim - k_dim)] = val;
        }
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void qkv_split_kernel_t<__half>(
    const __half*, __half*, __half*, __half*, int, int, int, int))
FVK_KERNEL_INSTANTIATE(__global__ void qkv_split_kernel_t<__nv_bfloat16>(
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int))
// Legacy BF16 kernel name — kept as thin alias for ABI compatibility.
__global__ void qkv_split_kernel(const __nv_bfloat16* __restrict__ qkv,
                                  __nv_bfloat16* __restrict__ Q,
                                  __nv_bfloat16* __restrict__ K,
                                  __nv_bfloat16* __restrict__ V,
                                  int seq, int q_dim, int k_dim, int v_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx < total) {
        int row = idx / qkv_dim;
        int col = idx % qkv_dim;
        __nv_bfloat16 val = qkv[idx];
        if (col < q_dim) {
            Q[row * q_dim + col] = val;
        } else if (col < q_dim + k_dim) {
            K[row * k_dim + (col - q_dim)] = val;
        } else {
            V[row * v_dim + (col - q_dim - k_dim)] = val;
        }
    }
}

void qkv_split(const __nv_bfloat16* qkv,
               __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
               int seq, int q_dim, int k_dim, int v_dim,
               cudaStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_kernel<<<blocks, threads, 0, stream>>>(qkv, Q, K, V, seq, q_dim, k_dim, v_dim);
}

void qkv_split_fp16(const __half* qkv,
                    __half* Q, __half* K, __half* V,
                    int seq, int q_dim, int k_dim, int v_dim,
                    cudaStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_kernel_t<__half><<<blocks, threads, 0, stream>>>(
        qkv, Q, K, V, seq, q_dim, k_dim, v_dim);
}

// ── Fused QKV Split + RoPE ──
__global__ void qkv_split_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ rope_weights,
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ V,
    int seq, int q_dim, int k_dim, int v_dim, int head_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx >= total) return;

    int row = idx / qkv_dim;
    int col = idx % qkv_dim;

    if (col < q_dim) {
        int q_col = col;
        int head = q_col / head_dim;
        int d_in_head = q_col % head_dim;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;

        int pair_base_qkv = row * qkv_dim + head * head_dim + pair * 2;
        float x0 = bf16_to_f32(qkv[pair_base_qkv]);
        float x1 = bf16_to_f32(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = bf16_to_f32(rope_weights[rope_base]);
        float s = bf16_to_f32(rope_weights[rope_base + 1]);

        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) {
            Q[out_idx] = f32_to_bf16(x0 * c - x1 * s);
        } else {
            Q[out_idx] = f32_to_bf16(x1 * c + x0 * s);
        }
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;

        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = bf16_to_f32(qkv[pair_base_qkv]);
        float x1 = bf16_to_f32(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = bf16_to_f32(rope_weights[rope_base]);
        float s = bf16_to_f32(rope_weights[rope_base + 1]);

        int out_idx = row * k_dim + k_col;
        if (is_odd == 0) {
            K[out_idx] = f32_to_bf16(x0 * c - x1 * s);
        } else {
            K[out_idx] = f32_to_bf16(x1 * c + x0 * s);
        }
    } else {
        int v_col = col - q_dim - k_dim;
        V[row * v_dim + v_col] = qkv[idx];
    }
}

void qkv_split_rope(const __nv_bfloat16* qkv,
                     const __nv_bfloat16* rope_weights,
                     __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                     int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                     cudaStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, seq, q_dim, k_dim, v_dim, head_dim);
}

// ── Fused QKV Split + RoPE with a RUNTIME (device) K/V-cache row offset ──
// Identical math to qkv_split_rope, but the K/V destination row is shifted by
// ``devpos[0]`` read at launch time. Q is still written contiguous (rows 0..seq).
// This lets a single fixed-shape CUDA graph append the decoder's K/V right
// after a variable-length valid prefix (devpos = valid prefix length), so the
// cross-attention can be one contiguous (seqused) call instead of recapturing
// per prompt length. K/V are the cache BASE pointers (row 0), not pre-offset.
__global__ void qkv_split_rope_devpos_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ rope_weights,
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ V,
    const int* __restrict__ devpos,
    int seq, int q_dim, int k_dim, int v_dim, int head_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx >= total) return;

    int row = idx / qkv_dim;
    int col = idx % qkv_dim;
    int pos = devpos[0];  // runtime K/V-cache row offset (valid prefix length)

    if (col < q_dim) {
        int q_col = col;
        int head = q_col / head_dim;
        int d_in_head = q_col % head_dim;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;
        int pair_base_qkv = row * qkv_dim + head * head_dim + pair * 2;
        float x0 = bf16_to_f32(qkv[pair_base_qkv]);
        float x1 = bf16_to_f32(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = bf16_to_f32(rope_weights[rope_base]);
        float s = bf16_to_f32(rope_weights[rope_base + 1]);
        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) Q[out_idx] = f32_to_bf16(x0 * c - x1 * s);
        else             Q[out_idx] = f32_to_bf16(x1 * c + x0 * s);
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = bf16_to_f32(qkv[pair_base_qkv]);
        float x1 = bf16_to_f32(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = bf16_to_f32(rope_weights[rope_base]);
        float s = bf16_to_f32(rope_weights[rope_base + 1]);
        int out_idx = (pos + row) * k_dim + k_col;
        if (is_odd == 0) K[out_idx] = f32_to_bf16(x0 * c - x1 * s);
        else             K[out_idx] = f32_to_bf16(x1 * c + x0 * s);
    } else {
        int v_col = col - q_dim - k_dim;
        V[(pos + row) * v_dim + v_col] = qkv[idx];
    }
}

void qkv_split_rope_devpos(const __nv_bfloat16* qkv,
                           const __nv_bfloat16* rope_weights,
                           __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                           const int* devpos,
                           int seq, int q_dim, int k_dim, int v_dim,
                           int head_dim, cudaStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_devpos_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, devpos, seq, q_dim, k_dim, v_dim, head_dim);
}

// ── Fused QKV Split + RoPE + KV Cache Write (FP16) ──
// Direct port of pi05 qkv_split_rope_kvcache_k for FP16 data.
// Q → contiguous (S, Q_dim)
// K → Kc[kc_offset + s*kc_stride + k_col] with RoPE applied
// V → Vc[kc_offset + s*kc_stride + v_col] direct copy
__global__ void qkv_split_rope_kvcache_fp16_kernel(
    const __half* __restrict__ qkv, const __half* __restrict__ rope,
    __half* __restrict__ Q, __half* __restrict__ Kc, __half* __restrict__ Vc,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * qkv_stride;
    if (idx >= total) return;
    int s = idx / qkv_stride;
    int c = idx % qkv_stride;

    if (c < Q_dim) {
        // Q region: apply RoPE
        int d_in_head = c % HD;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;
        int pair_base = s * qkv_stride + (c / HD) * HD + pair * 2;
        float x0 = __half2float(qkv[pair_base]);
        float x1 = __half2float(qkv[pair_base + 1]);
        int rope_base = s * HD + pair * 2;
        float cos_v = __half2float(rope[rope_base]);
        float sin_v = __half2float(rope[rope_base + 1]);
        if (is_odd == 0)
            Q[s * Q_dim + c] = __float2half(x0 * cos_v - x1 * sin_v);
        else
            Q[s * Q_dim + c] = __float2half(x1 * cos_v + x0 * sin_v);
    } else if (c < Q_dim + K_dim) {
        // K region: apply RoPE + write to KV cache
        int k_col = c - Q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base = s * qkv_stride + Q_dim + pair * 2;
        float x0 = __half2float(qkv[pair_base]);
        float x1 = __half2float(qkv[pair_base + 1]);
        int rope_base = s * HD + pair * 2;
        float cos_v = __half2float(rope[rope_base]);
        float sin_v = __half2float(rope[rope_base + 1]);
        __half val;
        if (is_odd == 0)
            val = __float2half(x0 * cos_v - x1 * sin_v);
        else
            val = __float2half(x1 * cos_v + x0 * sin_v);
        Kc[kc_offset + s * kc_stride + k_col] = val;
    } else {
        // V region: copy to KV cache directly
        int v_col = c - Q_dim - K_dim;
        Vc[kc_offset + s * kc_stride + v_col] = qkv[idx];
    }
}

void qkv_split_rope_kvcache_fp16(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride,
    cudaStream_t stream) {
    int total = S * qkv_stride;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kvcache_fp16_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope, Q, Kc, Vc, S, Q_dim, K_dim, HD, qkv_stride,
        kc_offset, kc_stride);
}

__global__ void qkv_split_rope_kvcache_fp16_devpos_kernel(
    const __half* __restrict__ qkv, const __half* __restrict__ rope,
    __half* __restrict__ Q, __half* __restrict__ Kc, __half* __restrict__ Vc,
    const int* __restrict__ devpos,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * qkv_stride;
    if (idx >= total) return;
    int s = idx / qkv_stride;
    int c = idx % qkv_stride;
    int pos = devpos[0];

    if (c < Q_dim) {
        int d_in_head = c % HD;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;
        int pair_base = s * qkv_stride + (c / HD) * HD + pair * 2;
        float x0 = __half2float(qkv[pair_base]);
        float x1 = __half2float(qkv[pair_base + 1]);
        int rope_base = s * HD + pair * 2;
        float cos_v = __half2float(rope[rope_base]);
        float sin_v = __half2float(rope[rope_base + 1]);
        if (is_odd == 0)
            Q[s * Q_dim + c] = __float2half(x0 * cos_v - x1 * sin_v);
        else
            Q[s * Q_dim + c] = __float2half(x1 * cos_v + x0 * sin_v);
    } else if (c < Q_dim + K_dim) {
        int k_col = c - Q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base = s * qkv_stride + Q_dim + pair * 2;
        float x0 = __half2float(qkv[pair_base]);
        float x1 = __half2float(qkv[pair_base + 1]);
        int rope_base = s * HD + pair * 2;
        float cos_v = __half2float(rope[rope_base]);
        float sin_v = __half2float(rope[rope_base + 1]);
        __half val;
        if (is_odd == 0)
            val = __float2half(x0 * cos_v - x1 * sin_v);
        else
            val = __float2half(x1 * cos_v + x0 * sin_v);
        Kc[kc_offset + (pos + s) * kc_stride + k_col] = val;
    } else {
        int v_col = c - Q_dim - K_dim;
        Vc[kc_offset + (pos + s) * kc_stride + v_col] = qkv[idx];
    }
}

void qkv_split_rope_kvcache_fp16_devpos(
    const __half* qkv, const __half* rope,
    __half* Q, __half* Kc, __half* Vc,
    const int* devpos,
    int S, int Q_dim, int K_dim, int HD, int qkv_stride,
    int kc_offset, int kc_stride,
    cudaStream_t stream) {
    int total = S * qkv_stride;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kvcache_fp16_devpos_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope, Q, Kc, Vc, devpos, S, Q_dim, K_dim, HD, qkv_stride,
        kc_offset, kc_stride);
}
