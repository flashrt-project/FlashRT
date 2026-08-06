#include "qwen3_int8_kv.cuh"

#include <math_constants.h>

namespace flash_rt::kernels {

namespace {

constexpr int HD = 128;          // head_dim
constexpr int NKV = 8;           // kv heads
constexpr int GQA = 2;           // q heads per kv head (16/8)
constexpr int CHUNK = 128;       // kv positions per flash-decoding chunk

// ── row quantize: one warp per 128-elem row ──
__global__ void kv_rows_quant_int8_kernel(
    const __nv_bfloat16* __restrict__ src, int8_t* __restrict__ dst,
    __nv_bfloat16* __restrict__ scales, int n_rows) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const __nv_bfloat16* s = src + (size_t)row * HD;
    float v[4];
    float amax = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = __bfloat162float(s[lane * 4 + k]);
        amax = fmaxf(amax, fabsf(v[k]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    const float scale = fmaxf(amax / 127.0f, 1e-8f);
    const float inv = 1.0f / scale;
    int8_t* d = dst + (size_t)row * HD;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        d[lane * 4 + k] = static_cast<int8_t>(lrintf(v[k] * inv));
    }
    if (lane == 0) scales[row] = __float2bfloat16(scale);
}

// ── flash-decoding partial: grid (NKV, n_chunks), 128 threads ──
// Phase A: 4 warps compute chunk scores (both GQA q-heads) into smem.
// Phase B: chunk softmax + P·V accumulation, thread per output dim.
__global__ __launch_bounds__(128) void attn_decode_int8kv_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int8_t* __restrict__ k8, const int8_t* __restrict__ v8,
    const __nv_bfloat16* __restrict__ ks, const __nv_bfloat16* __restrict__ vs,
    float* __restrict__ part_o, float* __restrict__ part_m,
    float* __restrict__ part_l, int kv_len, float softmax_scale) {
    const int h = blockIdx.x;               // kv head
    const int c = blockIdx.y;               // chunk
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j0 = c * CHUNK;
    const int jn = min(kv_len - j0, CHUNK); // valid positions in this chunk

    __shared__ float s0[CHUNK], s1[CHUNK];
    __shared__ float red0[CHUNK], red1[CHUNK];   // reduce scratch / p values
    __shared__ float m01[2], l01[2];

    // q rows for the two q-heads of this kv head; 4 dims per lane.
    const __nv_bfloat16* q0 = q + (size_t)(h * GQA) * HD;
    const __nv_bfloat16* q1 = q0 + HD;
    float q0r[4], q1r[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        q0r[k] = __bfloat162float(q0[lane * 4 + k]);
        q1r[k] = __bfloat162float(q1[lane * 4 + k]);
    }

    // Phase A: scores. Warp w handles positions j = w + 4*t.
    for (int t = warp; t < CHUNK; t += 4) {
        if (t < jn) {
            const int j = j0 + t;
            const int8_t* kr = k8 + ((size_t)j * NKV + h) * HD;
            const int kv4 = *reinterpret_cast<const int*>(kr + lane * 4);
            const int8_t* kb = reinterpret_cast<const int8_t*>(&kv4);
            float d0 = 0.0f, d1 = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float kf = static_cast<float>(kb[k]);
                d0 = fmaf(q0r[k], kf, d0);
                d1 = fmaf(q1r[k], kf, d1);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                d0 += __shfl_xor_sync(0xffffffff, d0, off);
                d1 += __shfl_xor_sync(0xffffffff, d1, off);
            }
            if (lane == 0) {
                const float kscale =
                    __bfloat162float(ks[(size_t)j * NKV + h]) * softmax_scale;
                s0[t] = d0 * kscale;
                s1[t] = d1 * kscale;
            }
        } else if (lane == 0) {
            s0[t] = -CUDART_INF_F;
            s1[t] = -CUDART_INF_F;
        }
    }
    __syncthreads();

    // Chunk softmax stats (block reduce over CHUNK slots).
    red0[tid] = s0[tid];
    red1[tid] = s1[tid];
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (tid < off) {
            red0[tid] = fmaxf(red0[tid], red0[tid + off]);
            red1[tid] = fmaxf(red1[tid], red1[tid + off]);
        }
        __syncthreads();
    }
    const float m0 = red0[0], m1 = red1[0];
    __syncthreads();
    // p = exp(s - m) * v_scale (fold V row scale into p once per row).
    {
        const int t = tid;
        float p0 = 0.0f, p1 = 0.0f;
        if (t < jn) {
            const float vscale = __bfloat162float(vs[(size_t)(j0 + t) * NKV + h]);
            p0 = __expf(s0[t] - m0);
            p1 = __expf(s1[t] - m1);
            s0[t] = p0 * vscale;
            s1[t] = p1 * vscale;
        } else {
            s0[t] = 0.0f;
            s1[t] = 0.0f;
        }
        red0[t] = p0;
        red1[t] = p1;
    }
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (tid < off) {
            red0[tid] += red0[tid + off];
            red1[tid] += red1[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) { l01[0] = red0[0]; m01[0] = m0; }
    if (tid == 1) { l01[1] = red1[0]; m01[1] = m1; }

    // Phase B: acc_d = sum_j p'_j * v8[j][d]; thread per dim.
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int t = 0; t < jn; ++t) {
        const int8_t* vr = v8 + ((size_t)(j0 + t) * NKV + h) * HD;
        const float vf = static_cast<float>(vr[tid]);
        acc0 = fmaf(s0[t], vf, acc0);
        acc1 = fmaf(s1[t], vf, acc1);
    }
    __syncthreads();
    const int qh0 = h * GQA, qh1 = qh0 + 1;
    float* po = part_o + ((size_t)c * (NKV * GQA)) * HD;
    po[(size_t)qh0 * HD + tid] = acc0;
    po[(size_t)qh1 * HD + tid] = acc1;
    if (tid == 0) {
        part_m[c * (NKV * GQA) + qh0] = m01[0];
        part_l[c * (NKV * GQA) + qh0] = l01[0];
    }
    if (tid == 1) {
        part_m[c * (NKV * GQA) + qh1] = m01[1];
        part_l[c * (NKV * GQA) + qh1] = l01[1];
    }
}

// ── combine: grid (16 q-heads), 128 threads (one per dim) ──
__global__ __launch_bounds__(128) void attn_decode_int8kv_combine_kernel(
    const float* __restrict__ part_o, const float* __restrict__ part_m,
    const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
    int n_chunks) {
    const int qh = blockIdx.x;
    const int d = threadIdx.x;
    constexpr int NQ = NKV * GQA;
    float M = -CUDART_INF_F;
    for (int c = 0; c < n_chunks; ++c) {
        M = fmaxf(M, part_m[c * NQ + qh]);
    }
    float o = 0.0f, l = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float w = __expf(part_m[c * NQ + qh] - M);
        o = fmaf(w, part_o[((size_t)c * NQ + qh) * HD + d], o);
        l = fmaf(w, part_l[c * NQ + qh], l);
    }
    out[(size_t)qh * HD + d] = __float2bfloat16(o / l);
}

}  // namespace

void qwen3_kv_rows_quant_int8(
    const __nv_bfloat16* src, int8_t* dst, __nv_bfloat16* scales,
    int n_rows, cudaStream_t stream) {
    constexpr int kThreads = 256;                      // 8 rows per block
    const int rows_per_block = kThreads / 32;
    dim3 grid((n_rows + rows_per_block - 1) / rows_per_block);
    kv_rows_quant_int8_kernel<<<grid, kThreads, 0, stream>>>(
        src, dst, scales, n_rows);
}

void qwen3_attn_decode_int8kv_partial(
    const __nv_bfloat16* q, const int8_t* k8, const int8_t* v8,
    const __nv_bfloat16* ks, const __nv_bfloat16* vs,
    float* part_o, float* part_m, float* part_l,
    int kv_len, int n_chunks, float softmax_scale, cudaStream_t stream) {
    dim3 grid(NKV, n_chunks);
    attn_decode_int8kv_partial_kernel<<<grid, 128, 0, stream>>>(
        q, k8, v8, ks, vs, part_o, part_m, part_l, kv_len, softmax_scale);
}

void qwen3_attn_decode_int8kv_combine(
    const float* part_o, const float* part_m, const float* part_l,
    __nv_bfloat16* out, int n_chunks, cudaStream_t stream) {
    attn_decode_int8kv_combine_kernel<<<dim3(NKV * GQA), 128, 0, stream>>>(
        part_o, part_m, part_l, out, n_chunks);
}

}  // namespace flash_rt::kernels
