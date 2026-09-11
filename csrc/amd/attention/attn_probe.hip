// ================================================================
// FlashRT AMD — decoder-attention phase-ablation probe (CDNA4)
//
// Companion to kernels/stream_probe.h and kernels/ew_tune.h. The
// production split-KV partial kernel measures ~17us in an in-graph
// chain while a work model predicts ~4-5: this probe replicates the
// kernel with a runtime phase mask so each phase's cost is measured,
// not guessed. Mask bits: 1=stage Q+K, 2=phase A scores, 4=stage V,
// 8=phase B softmax, 16=phase C PV, 32=workspace store. Phases build
// on garbage inputs when earlier bits are off — the probe measures
// TIME only; outputs are never checked (never route production here).
// ================================================================

#include "../kernels/common_hip.h"
#include <cfloat>
#include <hip/hip_fp8.h>
#include <stdexcept>

#define APROBE_NTHREADS  256
#define APROBE_SQ_MAX    16
#define APROBE_D_MAX     256
#define APROBE_CHUNK_MAX 64
#define APROBE_KV_PITCH  (APROBE_D_MAX / 2 + 1)
#define APROBE_Q_PITCH   (APROBE_D_MAX + 2)
#define APROBE_S_PITCH   (APROBE_CHUNK_MAX + 1)

namespace {

__device__ __forceinline__ void aprobe_stage_ilp4(
    __hip_bfloat162* __restrict__ dst,
    const __hip_bfloat162* __restrict__ src,
    int cn, int dh, int tid) {
    const int total = cn * dh;
    int idx = tid;
    for (; idx + 3 * APROBE_NTHREADS < total; idx += 4 * APROBE_NTHREADS) {
        const int i0 = idx, i1 = idx + APROBE_NTHREADS;
        const int i2 = idx + 2 * APROBE_NTHREADS, i3 = idx + 3 * APROBE_NTHREADS;
        const __hip_bfloat162 v0 = src[i0], v1 = src[i1];
        const __hip_bfloat162 v2 = src[i2], v3 = src[i3];
        const int r0 = i0 / dh, e0 = i0 - r0 * dh;
        const int r1 = i1 / dh, e1 = i1 - r1 * dh;
        const int r2 = i2 / dh, e2 = i2 - r2 * dh;
        const int r3 = i3 / dh, e3 = i3 - r3 * dh;
        dst[r0 * APROBE_KV_PITCH + e0] = v0;
        dst[r1 * APROBE_KV_PITCH + e1] = v1;
        dst[r2 * APROBE_KV_PITCH + e2] = v2;
        dst[r3 * APROBE_KV_PITCH + e3] = v3;
    }
    for (; idx < total; idx += APROBE_NTHREADS) {
        const int r = idx / dh, e = idx - r * dh;
        dst[r * APROBE_KV_PITCH + e] = src[idx];
    }
}

template <int NSPLIT>
__global__ __launch_bounds__(APROBE_NTHREADS)
void attn_partial_probe_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    float* __restrict__ ws,
    int Sq, int Skv, int Hq, int D, float scale, int mask) {
    const int split = blockIdx.x;
    const int h     = blockIdx.y;
    const int tid   = threadIdx.x;

    alignas(16) __shared__ __hip_bfloat162 kv_lds[APROBE_CHUNK_MAX * APROBE_KV_PITCH];
    alignas(16) __shared__ float q_lds[APROBE_SQ_MAX * APROBE_Q_PITCH];
    alignas(16) __shared__ float s_lds[APROBE_SQ_MAX * APROBE_S_PITCH];

    int skv_eff = Skv;
    if (skv_eff > NSPLIT * APROBE_CHUNK_MAX) skv_eff = NSPLIT * APROBE_CHUNK_MAX;
    const int chunk = (skv_eff + NSPLIT - 1) / NSPLIT;
    const int start = split * chunk;
    int end = start + chunk;
    if (end > skv_eff) end = skv_eff;
    const int cn = end - start;
    const int dh = D >> 1;

    const size_t base = (size_t)(split * Hq + h) * (size_t)Sq * (size_t)(D + 2);
    float* acc_ws = ws + base;
    float* m_ws   = ws + base + (size_t)Sq * D;
    float* l_ws   = m_ws + Sq;
    if (cn <= 0) {
        if (tid < Sq) { m_ws[tid] = -FLT_MAX; l_ws[tid] = 0.0f; }
        return;
    }

    if (mask & 1) {
        for (int idx = tid; idx < Sq * D; idx += APROBE_NTHREADS) {
            const int q = idx / D;
            const int d = idx - q * D;
            q_lds[q * APROBE_Q_PITCH + d] =
                __bfloat162float(Q[((size_t)q * Hq + h) * D + d]);
        }
        const __hip_bfloat162* K2 = reinterpret_cast<const __hip_bfloat162*>(K);
        aprobe_stage_ilp4(kv_lds, K2 + (size_t)start * dh, cn, dh, tid);
    }
    __syncthreads();

    if (mask & 2) {
        const int npairs = Sq * cn;
        for (int p = tid; p < npairs; p += APROBE_NTHREADS) {
            const int q = p / cn;
            const int r = p - q * cn;
            const __hip_bfloat162* kr = &kv_lds[r * APROBE_KV_PITCH];
            const float2* qr =
                reinterpret_cast<const float2*>(&q_lds[q * APROBE_Q_PITCH]);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            if (mask & 256) {
                // 8-deep explicit load batching (dh % 8: D % 16 == 0
                // at the site). Per-chain accumulation order is
                // identical to the baseline loop — bit-exact.
                for (int i = 0; i < dh; i += 8) {
                    __hip_bfloat162 kk[8];
                    float2 qq[8];
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) { kk[t] = kr[i + t]; qq[t] = qr[i + t]; }
                    #pragma unroll
                    for (int t = 0; t < 8; t += 2) {
                        a0 = fmaf(__bfloat162float(kk[t].x),     qq[t].x,     a0);
                        a1 = fmaf(__bfloat162float(kk[t].y),     qq[t].y,     a1);
                        a2 = fmaf(__bfloat162float(kk[t + 1].x), qq[t + 1].x, a2);
                        a3 = fmaf(__bfloat162float(kk[t + 1].y), qq[t + 1].y, a3);
                    }
                }
            } else {
                #pragma unroll 4
                for (int i = 0; i < dh; i += 2) {
                    const __hip_bfloat162 k0 = kr[i];
                    const __hip_bfloat162 k1 = kr[i + 1];
                    const float2 q0 = qr[i];
                    const float2 q1 = qr[i + 1];
                    a0 = fmaf(__bfloat162float(k0.x), q0.x, a0);
                    a1 = fmaf(__bfloat162float(k0.y), q0.y, a1);
                    a2 = fmaf(__bfloat162float(k1.x), q1.x, a2);
                    a3 = fmaf(__bfloat162float(k1.y), q1.y, a3);
                }
            }
            s_lds[q * APROBE_S_PITCH + r] = ((a0 + a1) + (a2 + a3)) * scale;
        }
    }
    __syncthreads();

    if (mask & 4) {
        const __hip_bfloat162* V2 = reinterpret_cast<const __hip_bfloat162*>(V);
        aprobe_stage_ilp4(kv_lds, V2 + (size_t)start * dh, cn, dh, tid);
    }
    if (mask & 8) {
        if (tid < Sq) {
            float* sp = &s_lds[tid * APROBE_S_PITCH];
            if (mask & 64) {
                // ILP4 softmax: 4 independent chains break the serial
                // LDS-latency dependency (max is order-free; the exp
                // sum combines chains in fixed order).
                float m0 = -FLT_MAX, m1 = -FLT_MAX, m2 = -FLT_MAX, m3 = -FLT_MAX;
                int j = 0;
                for (; j + 3 < cn; j += 4) {
                    m0 = fmaxf(m0, sp[j]);
                    m1 = fmaxf(m1, sp[j + 1]);
                    m2 = fmaxf(m2, sp[j + 2]);
                    m3 = fmaxf(m3, sp[j + 3]);
                }
                for (; j < cn; ++j) m0 = fmaxf(m0, sp[j]);
                const float m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
                float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
                j = 0;
                for (; j + 3 < cn; j += 4) {
                    const float e0 = __expf(sp[j] - m);
                    const float e1 = __expf(sp[j + 1] - m);
                    const float e2 = __expf(sp[j + 2] - m);
                    const float e3 = __expf(sp[j + 3] - m);
                    sp[j] = e0; sp[j + 1] = e1; sp[j + 2] = e2; sp[j + 3] = e3;
                    l0 += e0; l1 += e1; l2 += e2; l3 += e3;
                }
                for (; j < cn; ++j) {
                    const float e = __expf(sp[j] - m);
                    sp[j] = e;
                    l0 += e;
                }
                m_ws[tid] = m;
                l_ws[tid] = (l0 + l1) + (l2 + l3);
            } else {
                float m = -FLT_MAX;
                for (int j = 0; j < cn; ++j) m = fmaxf(m, sp[j]);
                float l = 0.0f;
                for (int j = 0; j < cn; ++j) {
                    const float e = __expf(sp[j] - m);
                    sp[j] = e;
                    l += e;
                }
                m_ws[tid] = m;
                l_ws[tid] = l;
            }
        }
    }
    __syncthreads();

    if (mask & 16) {
        const int nunits = Sq * dh;
        for (int u = tid; u < nunits; u += APROBE_NTHREADS) {
            const int q  = u / dh;
            const int dp = u - q * dh;
            const float* sp = &s_lds[q * APROBE_S_PITCH];
            const __hip_bfloat162* vp = &kv_lds[dp];
            float ax, ay;
            if (mask & 512) {
                // ILP8 PV: 8 independent chains.
                float axs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                float ays[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                int j = 0;
                for (; j + 7 < cn; j += 8) {
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const __hip_bfloat162 v = vp[(j + t) * APROBE_KV_PITCH];
                        const float pj = sp[j + t];
                        axs[t] = fmaf(pj, __bfloat162float(v.x), axs[t]);
                        ays[t] = fmaf(pj, __bfloat162float(v.y), ays[t]);
                    }
                }
                for (; j < cn; ++j) {
                    const __hip_bfloat162 v = vp[j * APROBE_KV_PITCH];
                    const float pj = sp[j];
                    axs[0] = fmaf(pj, __bfloat162float(v.x), axs[0]);
                    ays[0] = fmaf(pj, __bfloat162float(v.y), ays[0]);
                }
                ax = ((axs[0] + axs[1]) + (axs[2] + axs[3]))
                   + ((axs[4] + axs[5]) + (axs[6] + axs[7]));
                ay = ((ays[0] + ays[1]) + (ays[2] + ays[3]))
                   + ((ays[4] + ays[5]) + (ays[6] + ays[7]));
            } else if (mask & 128) {
                // ILP4 PV: 4 independent accumulator chains, fixed
                // combine order.
                float ax0 = 0.0f, ay0 = 0.0f, ax1 = 0.0f, ay1 = 0.0f;
                float ax2 = 0.0f, ay2 = 0.0f, ax3 = 0.0f, ay3 = 0.0f;
                int j = 0;
                for (; j + 3 < cn; j += 4) {
                    const __hip_bfloat162 v0 = vp[(j)     * APROBE_KV_PITCH];
                    const __hip_bfloat162 v1 = vp[(j + 1) * APROBE_KV_PITCH];
                    const __hip_bfloat162 v2 = vp[(j + 2) * APROBE_KV_PITCH];
                    const __hip_bfloat162 v3 = vp[(j + 3) * APROBE_KV_PITCH];
                    const float p0 = sp[j], p1 = sp[j + 1];
                    const float p2 = sp[j + 2], p3 = sp[j + 3];
                    ax0 = fmaf(p0, __bfloat162float(v0.x), ax0);
                    ay0 = fmaf(p0, __bfloat162float(v0.y), ay0);
                    ax1 = fmaf(p1, __bfloat162float(v1.x), ax1);
                    ay1 = fmaf(p1, __bfloat162float(v1.y), ay1);
                    ax2 = fmaf(p2, __bfloat162float(v2.x), ax2);
                    ay2 = fmaf(p2, __bfloat162float(v2.y), ay2);
                    ax3 = fmaf(p3, __bfloat162float(v3.x), ax3);
                    ay3 = fmaf(p3, __bfloat162float(v3.y), ay3);
                }
                for (; j < cn; ++j) {
                    const __hip_bfloat162 v = vp[j * APROBE_KV_PITCH];
                    const float pj = sp[j];
                    ax0 = fmaf(pj, __bfloat162float(v.x), ax0);
                    ay0 = fmaf(pj, __bfloat162float(v.y), ay0);
                }
                ax = (ax0 + ax1) + (ax2 + ax3);
                ay = (ay0 + ay1) + (ay2 + ay3);
            } else {
                ax = 0.0f; ay = 0.0f;
                for (int j = 0; j < cn; ++j) {
                    const __hip_bfloat162 v = vp[j * APROBE_KV_PITCH];
                    const float pj = sp[j];
                    ax = fmaf(pj, __bfloat162float(v.x), ax);
                    ay = fmaf(pj, __bfloat162float(v.y), ay);
                }
            }
            if (mask & 32) {
                float* out = acc_ws + (size_t)q * D + 2 * dp;
                out[0] = ax;
                out[1] = ay;
            } else if (ax == 1234.5f && ay == 6789.0f) {
                acc_ws[0] = ax;   // dead guard: keep the loop live
            }
        }
    }
}

} // namespace

void attn_partial_probe(const __hip_bfloat16* Q, const __hip_bfloat16* K,
                        const __hip_bfloat16* V, float* ws,
                        int Sq, int Skv, int Hq, int D, float scale,
                        int nsplit, int mask, hipStream_t stream) {
    dim3 grid(nsplit, Hq);
    switch (nsplit) {
    case 8:  attn_partial_probe_kernel<8><<<grid, APROBE_NTHREADS, 0, stream>>>(Q, K, V, ws, Sq, Skv, Hq, D, scale, mask); break;
    case 16: attn_partial_probe_kernel<16><<<grid, APROBE_NTHREADS, 0, stream>>>(Q, K, V, ws, Sq, Skv, Hq, D, scale, mask); break;
    case 32: attn_partial_probe_kernel<32><<<grid, APROBE_NTHREADS, 0, stream>>>(Q, K, V, ws, Sq, Skv, Hq, D, scale, mask); break;
    default: throw std::runtime_error("attn_partial_probe: nsplit must be 8/16/32");
    }
}
