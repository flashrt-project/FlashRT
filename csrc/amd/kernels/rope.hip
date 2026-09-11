// ================================================================
// FlashRT AMD — QKV split kernels (CDNA wave64)
// QKV split, fused QKV split + RoPE, and the devpos variant
// (runtime device K/V-cache row offset for fixed-shape prompt mode).
// Mirrors csrc/kernels/rope.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
// ================================================================

#include "common_hip.h"

// ── QKV Split ──
// Pure memcpy by column region: qkv (seq, q_dim+k_dim+v_dim) row-major
// scatters into Q (seq, q_dim), K (seq, k_dim), V (seq, v_dim).
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

template __global__ void qkv_split_kernel_t<__half>(
    const __half*, __half*, __half*, __half*, int, int, int, int);
template __global__ void qkv_split_kernel_t<__hip_bfloat16>(
    const __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*,
    int, int, int, int);

void qkv_split(const __hip_bfloat16* qkv,
               __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
               int seq, int q_dim, int k_dim, int v_dim,
               hipStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_kernel_t<__hip_bfloat16><<<blocks, threads, 0, stream>>>(
        qkv, Q, K, V, seq, q_dim, k_dim, v_dim);
}

// ── Fused QKV Split + RoPE ──
// Interleaved-pair rotation: within each head, elements (2d, 2d+1) form a
// rotation pair. rope_weights layout: (seq, head_dim) with interleaved
// (cos, sin) per pair — rope_weights[row, 2p] = cos, [row, 2p+1] = sin.
// Q has q_dim/head_dim heads (all rotated); K is single-head (k_dim wide,
// rotated with the same per-row weights); V is copied through.
__global__ void qkv_split_rope_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
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
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);

        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) {
            Q[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        } else {
            Q[out_idx] = __float2bfloat16(x1 * c + x0 * s);
        }
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;

        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);

        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);

        int out_idx = row * k_dim + k_col;
        if (is_odd == 0) {
            K[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        } else {
            K[out_idx] = __float2bfloat16(x1 * c + x0 * s);
        }
    } else {
        int v_col = col - q_dim - k_dim;
        V[row * v_dim + v_col] = qkv[idx];
    }
}

// ── Pair-per-thread 2D variant (CDNA retune) ──
// One rotation pair per thread, blockIdx.y = row: no div/mod for the
// row, one dword load + one dword store per thread; each thread
// computes BOTH rotated outputs of its pair — identical float math to
// the two elements of the flat kernel, bit-exact (verified by the
// ew_tune probe). ~1.5x on encoder-seq shapes.
__global__ void qkv_split_rope_pair_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
    int q_dim, int k_dim, int v_dim, int head_dim, int qkv_dim) {
    int row = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int npairs = qkv_dim >> 1;
    if (p >= npairs) return;
    int col = p << 1;

    const __hip_bfloat162* src2 =
        reinterpret_cast<const __hip_bfloat162*>(qkv + (size_t)row * qkv_dim);
    __hip_bfloat162 xv = src2[p];

    if (col < q_dim) {
        int d_in_head = col % head_dim;   // even by construction
        __hip_bfloat162 cs = reinterpret_cast<const __hip_bfloat162*>(
            rope_weights)[((size_t)row * head_dim + d_in_head) >> 1];
        float x0 = __bfloat162float(xv.x), x1 = __bfloat162float(xv.y);
        float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
        reinterpret_cast<__hip_bfloat162*>(Q)[((size_t)row * q_dim + col) >> 1] =
            __halves2bfloat162(__float2bfloat16(x0 * c - x1 * s),
                               __float2bfloat16(x1 * c + x0 * s));
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        __hip_bfloat162 cs = reinterpret_cast<const __hip_bfloat162*>(
            rope_weights)[((size_t)row * head_dim + k_col) >> 1];
        float x0 = __bfloat162float(xv.x), x1 = __bfloat162float(xv.y);
        float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
        reinterpret_cast<__hip_bfloat162*>(K)[((size_t)row * k_dim + k_col) >> 1] =
            __halves2bfloat162(__float2bfloat16(x0 * c - x1 * s),
                               __float2bfloat16(x1 * c + x0 * s));
    } else {
        int v_col = col - q_dim - k_dim;
        reinterpret_cast<__hip_bfloat162*>(V)[((size_t)row * v_dim + v_col) >> 1] = xv;
    }
}

void qkv_split_rope(const __hip_bfloat16* qkv,
                    const __hip_bfloat16* rope_weights,
                    __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                    int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                    hipStream_t stream) {
    int qkv_dim = q_dim + k_dim + v_dim;
    if (((q_dim | k_dim | v_dim | head_dim) & 1) == 0) {
        int npairs = qkv_dim >> 1;
        dim3 grid((npairs + 255) / 256, seq);
        qkv_split_rope_pair_kernel<<<grid, 256, 0, stream>>>(
            qkv, rope_weights, Q, K, V, q_dim, k_dim, v_dim, head_dim, qkv_dim);
        return;
    }
    int total = seq * qkv_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, seq, q_dim, k_dim, v_dim, head_dim);
}

// ── Fused QKV Split + RoPE with a RUNTIME (device) K/V-cache row offset ──
// Identical math to qkv_split_rope, but the K/V destination row is shifted by
// ``devpos[0]`` read at launch time. Q is still written contiguous (rows 0..seq).
// This lets a single fixed-shape HIP graph append the decoder's K/V right
// after a variable-length valid prefix (devpos = valid prefix length), so the
// cross-attention can be one contiguous (seqused) call instead of recapturing
// per prompt length. K/V are the cache BASE pointers (row 0), not pre-offset.
__global__ void qkv_split_rope_devpos_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
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
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) Q[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             Q[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = (pos + row) * k_dim + k_col;
        if (is_odd == 0) K[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             K[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else {
        int v_col = col - q_dim - k_dim;
        V[(pos + row) * v_dim + v_col] = qkv[idx];
    }
}

void qkv_split_rope_devpos(const __hip_bfloat16* qkv,
                           const __hip_bfloat16* rope_weights,
                           __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                           const int* devpos,
                           int seq, int q_dim, int k_dim, int v_dim,
                           int head_dim, hipStream_t stream) {
    int total = seq * (q_dim + k_dim + v_dim);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    qkv_split_rope_devpos_kernel<<<blocks, threads, 0, stream>>>(
        qkv, rope_weights, Q, K, V, devpos, seq, q_dim, k_dim, v_dim, head_dim);
}
