// ================================================================
// FlashRT — Hy-VLA fused attention-prep megakernel (Thor SM110)
//
// One launch replaces ~11 tiny torch ops per attention block:
//   split(qkv) → RoPE(q) → RoPE(k) → QK-Norm(q) → QK-Norm(k)
//   → write K,V into the layer KV-cache at row `off`.
//
// Hy-VLA order is RoPE-FIRST then QK-Norm (RMSNorm over head_dim),
// the reverse of the existing qwen3 fused kernel — hence a bespoke
// kernel. rotate_half (NeoX) convention; cos/sin are per-position
// (shared across heads). GQA: nq query heads, nkv KV heads.
//
// Layouts (all bf16, contiguous):
//   qkv   : (S, (nq+2*nkv)*hd)
//   cos/sin: (S, hd)   qn_w/kn_w: (hd)
//   q_out : (nq, S, hd)                 == (1,nq,S,hd) for SDPA
//   kbuf/vbuf: (nkv, S_tot, hd)         (one layer slice); write row off+s
// ================================================================

#include "common.cuh"
#include <cuda_bf16.h>

__global__ void hyvla_rope_qknorm_kvwrite_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ qn_w,
    const __nv_bfloat16* __restrict__ kn_w,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kbuf,
    __nv_bfloat16* __restrict__ vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps, int kv_rep)
{
    const int s = blockIdx.x;
    const int tid = threadIdx.x;               // 0..hd-1
    if (tid >= hd) return;
    const int Dqkv = (nq + 2 * nkv) * hd;
    const int half = hd >> 1;
    const int partner = (tid < half) ? (tid + half) : (tid - half);
    const float sgn = (tid < half) ? -1.0f : 1.0f;   // rotate_half sign
    __shared__ float shared[64];

    const float cs = to_f32<__nv_bfloat16>(cos[s * hd + tid]);
    const float sn = to_f32<__nv_bfloat16>(sin[s * hd + tid]);
    const float qnw = to_f32<__nv_bfloat16>(qn_w[tid]);
    const float knw = to_f32<__nv_bfloat16>(kn_w[tid]);

    // Q heads: RoPE then QK-Norm -> q_out
    for (int h = 0; h < nq; ++h) {
        __syncthreads();
        const __nv_bfloat16* base = qkv + s * Dqkv + h * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        q_out[h * S * hd + s * hd + tid] = from_f32<__nv_bfloat16>(roped * rms * qnw);
    }
    // K heads: RoPE then QK-Norm -> kbuf[off+s]
    for (int kh = 0; kh < nkv; ++kh) {
        __syncthreads();
        const __nv_bfloat16* base = qkv + s * Dqkv + nq * hd + kh * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        __nv_bfloat16 kval = from_f32<__nv_bfloat16>(roped * rms * knw);
        for (int rr = 0; rr < kv_rep; ++rr)
            kbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = kval;
    }
    // V heads: raw copy -> vbuf[off+s] (replicated kv_rep times for GQA)
    for (int kh = 0; kh < nkv; ++kh) {
        const __nv_bfloat16* base = qkv + s * Dqkv + (nq + nkv) * hd + kh * hd;
        __nv_bfloat16 vval = base[tid];
        for (int rr = 0; rr < kv_rep; ++rr)
            vbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = vval;
    }
}

extern "C" void hyvla_rope_qknorm_kvwrite_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(hd);
    hyvla_rope_qknorm_kvwrite_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)kbuf, (__nv_bfloat16*)vbuf,
        S, nq, nkv, hd, S_tot, off, eps, kv_rep < 1 ? 1 : kv_rep);
}

// ================================================================
// Additive variants (new symbols; the base kernel above is unchanged).
// ================================================================

// Parallel variant: identical math to hyvla_rope_qknorm_kvwrite_bf16 (each
// block reduces exactly one head through the same 128-thread
// block_reduce_sum tree, so the result is bit-exact), but the grid is
// (S, nq + 2*nkv) so the heads of a position run in independent blocks
// instead of one block serialising all of them. Removes the occupancy bound
// at the small denoise suffix (S=41).
__global__ void hyvla_rope_qknorm_kvwrite_parallel_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ qn_w,
    const __nv_bfloat16* __restrict__ kn_w,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kbuf,
    __nv_bfloat16* __restrict__ vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps, int kv_rep)
{
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;               // 0..hd-1
    if (tid >= hd) return;
    const int Dqkv = (nq + 2 * nkv) * hd;
    const int half = hd >> 1;
    const int partner = (tid < half) ? (tid + half) : (tid - half);
    const float sgn = (tid < half) ? -1.0f : 1.0f;   // rotate_half sign
    __shared__ float shared[64];

    const float cs = to_f32<__nv_bfloat16>(cos[s * hd + tid]);
    const float sn = to_f32<__nv_bfloat16>(sin[s * hd + tid]);

    if (h < nq) {
        // Q head: RoPE then QK-Norm -> q_out
        const float qnw = to_f32<__nv_bfloat16>(qn_w[tid]);
        const __nv_bfloat16* base = qkv + s * Dqkv + h * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        q_out[h * S * hd + s * hd + tid] = from_f32<__nv_bfloat16>(roped * rms * qnw);
    } else if (h < nq + nkv) {
        // K head: RoPE then QK-Norm -> kbuf[off+s] (replicated kv_rep times)
        const int kh = h - nq;
        const float knw = to_f32<__nv_bfloat16>(kn_w[tid]);
        const __nv_bfloat16* base = qkv + s * Dqkv + (nq + kh) * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        __nv_bfloat16 kval = from_f32<__nv_bfloat16>(roped * rms * knw);
        for (int rr = 0; rr < kv_rep; ++rr)
            kbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = kval;
    } else {
        // V head: raw copy -> vbuf[off+s] (replicated kv_rep times for GQA)
        const int kh = h - nq - nkv;
        const __nv_bfloat16* base = qkv + s * Dqkv + (nq + nkv + kh) * hd;
        __nv_bfloat16 vval = base[tid];
        for (int rr = 0; rr < kv_rep; ++rr)
            vbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = vval;
    }
}

extern "C" void hyvla_rope_qknorm_kvwrite_parallel_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    dim3 grid(S, nq + 2 * nkv);
    dim3 block(hd);
    hyvla_rope_qknorm_kvwrite_parallel_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)kbuf, (__nv_bfloat16*)vbuf,
        S, nq, nkv, hd, S_tot, off, eps, kv_rep < 1 ? 1 : kv_rep);
}

// Q-only layout variant: same RoPE/QK-Norm math as the parallel kernel, but
// the Q head result is written in the FA2 denoise `qb` packing (2, S, nq, hd)
// directly, so the standalone prepare_q transpose is no longer needed. K/V
// writes are identical to the base kernel (bit-exact). New symbol.
__global__ void hyvla_rope_qknorm_kvwrite_qb_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const __nv_bfloat16* __restrict__ qn_w,
    const __nv_bfloat16* __restrict__ kn_w,
    __nv_bfloat16* __restrict__ qb,
    __nv_bfloat16* __restrict__ kbuf,
    __nv_bfloat16* __restrict__ vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps, int kv_rep)
{
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;               // 0..hd-1
    if (tid >= hd) return;
    const int Dqkv = (nq + 2 * nkv) * hd;
    const int half = hd >> 1;
    const int partner = (tid < half) ? (tid + half) : (tid - half);
    const float sgn = (tid < half) ? -1.0f : 1.0f;   // rotate_half sign
    __shared__ float shared[64];

    const float cs = to_f32<__nv_bfloat16>(cos[s * hd + tid]);
    const float sn = to_f32<__nv_bfloat16>(sin[s * hd + tid]);

    if (h < nq) {
        // Q head: RoPE then QK-Norm -> qb (2,S,nq,hd) denoise packing.
        const float qnw = to_f32<__nv_bfloat16>(qn_w[tid]);
        const __nv_bfloat16* base = qkv + s * Dqkv + h * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        const long long row_stride = (long long)nq * hd;
        long long off_qb = (s == 0)
            ? (long long)h * hd + tid
            : row_stride * S + (long long)(s - 1) * row_stride + (long long)h * hd + tid;
        qb[off_qb] = from_f32<__nv_bfloat16>(roped * rms * qnw);
    } else if (h < nq + nkv) {
        // K head: RoPE then QK-Norm -> kbuf[off+s] (replicated kv_rep times)
        const int kh = h - nq;
        const float knw = to_f32<__nv_bfloat16>(kn_w[tid]);
        const __nv_bfloat16* base = qkv + s * Dqkv + (nq + kh) * hd;
        float x  = to_f32<__nv_bfloat16>(base[tid]);
        float xp = to_f32<__nv_bfloat16>(base[partner]);
        float roped = x * cs + sgn * xp * sn;
        float ss = block_reduce_sum(roped * roped, shared);
        float rms = rsqrtf(ss / hd + eps);
        __nv_bfloat16 kval = from_f32<__nv_bfloat16>(roped * rms * knw);
        for (int rr = 0; rr < kv_rep; ++rr)
            kbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = kval;
    } else {
        // V head: raw copy -> vbuf[off+s] (replicated kv_rep times for GQA)
        const int kh = h - nq - nkv;
        const __nv_bfloat16* base = qkv + s * Dqkv + (nq + nkv + kh) * hd;
        __nv_bfloat16 vval = base[tid];
        for (int rr = 0; rr < kv_rep; ++rr)
            vbuf[(kh * kv_rep + rr) * S_tot * hd + (off + s) * hd + tid] = vval;
    }
}

extern "C" void hyvla_rope_qknorm_kvwrite_qb_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* qb, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    dim3 grid(S, nq + 2 * nkv);
    dim3 block(hd);
    hyvla_rope_qknorm_kvwrite_qb_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        (const __nv_bfloat16*)qn_w, (const __nv_bfloat16*)kn_w,
        (__nv_bfloat16*)qb, (__nv_bfloat16*)kbuf, (__nv_bfloat16*)vbuf,
        S, nq, nkv, hd, S_tot, off, eps, kv_rep < 1 ? 1 : kv_rep);
}
