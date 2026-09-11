// ================================================================
// FlashRT AMD — encoder/prefill self-attention, MFMA flash (gfx950)
//
// One fixed production site: bidirectional GQA self-attention
//     Q (S, Hq, D) bf16 — S ~ encoder tokens (vision + prompt)
//     K (S, D)     bf16 — single KV head (GQA Hq:1)
//     V (S, D)     bf16
//     O (S, Hq, D) bf16
//
// STATUS: EXPERIMENTAL — not routed by any production path yet. The
// library engine (CK-tile FMHA, ~47us in-situ) reaches only a few
// percent of the matrix-core rate at this shape (hd 256, GQA 8:1,
// batch 1); this kernel currently measures ~71us (parity-green,
// deterministic) and ships as the research vehicle for the remaining
// iterations (softmax micro-arch, staging write cost, block sizing).
// It is a from-scratch FA2-style kernel on V_MFMA_F32_16X16X32_BF16:
//
//   grid (ceil(S/64), Hq), block 256 (4 waves). Each wave owns one
//   16-row q-tile; the workgroup's 4 waves share each staged 32-row
//   K/V block (K row-major, V TRANSPOSED to d-major so both MFMA
//   operand fragments are contiguous 16B LDS reads).
//
//   Per KV block: scores 16x32 = 2 MFMA chains over D (8 steps of
//   k=32 each), online-softmax update in registers (row stats via
//   16-lane xor shuffle trees), P rounded to bf16 and bounced through
//   LDS into a-fragment layout, then PV accumulates the full 16x256
//   O tile held in 16 f32x4 accumulator fragments per lane.
//
//   Determinism: fixed block order, fixed shuffle trees, fp32
//   accumulate, no atomics. Numerics: P is rounded to bf16 before PV
//   (standard flash practice) — parity is cos-gated vs an fp32
//   reference.
//
// Constraints (checked): D == 256, S <= 4096, Hq <= 16. Tail q rows
// are computed but not stored; tail kv columns are masked to -inf
// before the softmax.
// ================================================================

#include "../kernels/common_hip.h"
#include <cfloat>
#include <stdexcept>

namespace {

typedef float floatx4 __attribute__((ext_vector_type(4)));
typedef short shortx8 __attribute__((ext_vector_type(8)));

#define EFA_D        256
#define EFA_DSTEPS   (EFA_D / 32)    // MFMA k-steps over D
#define EFA_KVBLK    32              // KV rows staged per block
#define EFA_NTHREADS 256

// LDS: K block row-major (32 x 256 bf16), V block d-major with the
// row padded to 40 elements (80 B: 16-byte aligned for the b-frag
// reads, and the 20-banks-per-row advance kills the transpose-write
// conflicts a tight 64 B stride would have), plus a 16x32 bf16 P
// bounce region per wave.
#define EFA_VPITCH (EFA_KVBLK + 8)
#define EFA_KPITCH (EFA_D + 8)
// Double-buffered K/V tiles (~79 KB total): block b+1's staging writes
// need no barrier against block b's reads, so the loop runs on ONE
// workgroup barrier per block. Requires the gfx950 dynamic-LDS opt-in
// (hipFuncSetAttribute, > 64 KB static limit).
struct EfaLds {
    __hip_bfloat16 kt[2][EFA_KVBLK][EFA_KPITCH];
    __hip_bfloat16 vt[2][EFA_D][EFA_VPITCH];
    __hip_bfloat16 p[4][16][EFA_KVBLK];
};

__global__ __launch_bounds__(EFA_NTHREADS)
void encoder_attention_flash_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    __hip_bfloat16* __restrict__ O,
    int S, int Hq, float scale, int mask,
    const int* __restrict__ seqused) {
    // Runtime valid length (FA2 seqused contract): S stays the padded
    // capacity that sized the grid; the device value masks compute.
    // Uniform across the workgroup, so the derived loop bounds and
    // barriers stay convergent.
    const int Scap = S;   // padded capacity (grid/store bound)
    if (seqused != nullptr) {
        const int se = seqused[0];
        if (se > 0 && se < S) S = se;
    }
    extern __shared__ unsigned char lds_raw[];
    EfaLds& lds = *reinterpret_cast<EfaLds*>(lds_raw);

    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int h    = blockIdx.y;
    const int q0   = (blockIdx.x * 4 + wave) * 16;   // this wave's q-tile
    const int fcol = lane & 15;
    const int kgrp = lane >> 4;

    // ── Stage the wave's Q fragments (pre-scaled) in registers ──
    // a-frag step s: lane supplies Q[q0 + fcol][32*s + 8*kgrp .. +8].
    // Pad q rows read row (S-1) (their outputs are never stored, but
    // their softmax must stay finite).
    shortx8 qf[EFA_DSTEPS];
    {
        const int qrow = (q0 + fcol < S) ? (q0 + fcol) : (S - 1);
        const __hip_bfloat16* qp = Q + ((size_t)qrow * Hq + h) * EFA_D + kgrp * 8;
        #pragma unroll
        for (int s = 0; s < EFA_DSTEPS; ++s) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = __bfloat162float(qp[s * 32 + e]) * scale;
                reinterpret_cast<__hip_bfloat16*>(&qf[s])[e] = __float2bfloat16(v);
            }
        }
    }

    // ── Online-softmax state and O accumulators ──
    float m_run[4], l_run[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { m_run[i] = -FLT_MAX; l_run[i] = 0.0f; }
    floatx4 oacc[EFA_D / 16];
    #pragma unroll
    for (int t = 0; t < EFA_D / 16; ++t) oacc[t] = (floatx4){0, 0, 0, 0};

    const int nblk = (S + EFA_KVBLK - 1) / EFA_KVBLK;

    // ── Software-pipelined K/V loads: the global reads for block b+1
    // issue during block b's compute, so the staging phase is pure LDS
    // writes. r-major mapping for the V transpose scatter (consecutive
    // 2 B columns across lanes), slot-major 16 B vector writes for K.
    // V: each thread owns a PAIR of adjacent rows (2r, 2r+1) of one
    // 8-element window, so the transpose scatter becomes 8 x 4 B pair
    // writes instead of 16 x 2 B — half the LDS write instructions,
    // consecutive-bank pattern preserved.
    uint4 vv[4], vv2[4], kk[4];
    auto load_block = [&](int blk) {
        const int kv0l = blk * EFA_KVBLK;
        const int kvnl = (blk < nblk) ? min(EFA_KVBLK, S - kv0l) : 0;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idxv = tid + j * EFA_NTHREADS;   // pair-slot id
            const int rv = (idxv & (EFA_KVBLK / 2 - 1)) * 2;
            const int ov = (idxv >> 4) * 8;
            vv[j] = (rv < kvnl)
                ? *reinterpret_cast<const uint4*>(V + (size_t)(kv0l + rv) * EFA_D + ov)
                : (uint4){0, 0, 0, 0};
            vv2[j] = (rv + 1 < kvnl)
                ? *reinterpret_cast<const uint4*>(V + (size_t)(kv0l + rv + 1) * EFA_D + ov)
                : (uint4){0, 0, 0, 0};
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idxk = tid + j * EFA_NTHREADS;
            const int rk = idxk >> 5;
            const int ok = (idxk & 31) * 8;
            kk[j] = (rk < kvnl)
                ? *reinterpret_cast<const uint4*>(K + (size_t)(kv0l + rk) * EFA_D + ok)
                : (uint4){0, 0, 0, 0};
        }
    };
    if (mask & 1) load_block(0);

    for (int b = 0; b < nblk; ++b) {
        const int kv0 = b * EFA_KVBLK;
        const int kvn = min(EFA_KVBLK, S - kv0);

        const int buf = b & 1;
        if (mask & 1) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = tid + j * EFA_NTHREADS;
                const int rk = idx >> 5;
                const int ok = (idx & 31) * 8;
                *reinterpret_cast<uint4*>(&lds.kt[buf][rk][ok]) = kk[j];
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int idx = tid + j * EFA_NTHREADS;
                const int rv = (idx & (EFA_KVBLK / 2 - 1)) * 2;
                const int ov = (idx >> 4) * 8;
                const __hip_bfloat16* v0 =
                    reinterpret_cast<const __hip_bfloat16*>(&vv[j]);
                const __hip_bfloat16* v1 =
                    reinterpret_cast<const __hip_bfloat16*>(&vv2[j]);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    __hip_bfloat162 pr = __halves2bfloat162(v0[e], v1[e]);
                    *reinterpret_cast<__hip_bfloat162*>(&lds.vt[buf][ov + e][rv]) = pr;
                }
            }
        }
        __syncthreads();   // this block's kt/vt published (single barrier)
        if (mask & 1) load_block(b + 1);   // overlaps this block's compute

        // ── Scores: two 16x16 tiles (kv halves) over D ──
        floatx4 sc[2];
        sc[0] = (floatx4){0, 0, 0, 0};
        sc[1] = (floatx4){0, 0, 0, 0};
        if (mask & 2)
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            floatx4 acc = {0, 0, 0, 0};
            // b-frag from the shared LDS K tile; all 8 step fragments
            // prefetched before the (serial) MFMA chain consumes them.
            const __hip_bfloat16* kp = &lds.kt[buf][half * 16 + fcol][kgrp * 8];
            shortx8 kf[EFA_DSTEPS];
            #pragma unroll
            for (int s = 0; s < EFA_DSTEPS; ++s)
                kf[s] = *reinterpret_cast<const shortx8*>(kp + s * 32);
            #pragma unroll
            for (int s = 0; s < EFA_DSTEPS; ++s)
                acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    qf[s], kf[s], acc, 0, 0, 0);
            sc[half] = acc;
        }

        // ── Online softmax update (rows m = 4*kgrp + i) ──
        float pvals[2][4];
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int i = 0; i < 4; ++i) pvals[half][i] = 0.5f;
        if (mask & 4)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            // Row max over the 32 kv columns of this block: each lane
            // holds 2 candidates (one per half); mask the kv tail.
            const int n = fcol;
            float c0 = (n < kvn)      ? sc[0][i] : -FLT_MAX;
            float c1 = (16 + n < kvn) ? sc[1][i] : -FLT_MAX;
            float mx = fmaxf(c0, c1);
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor(mx, off));
            // mx is now the block row max across the 16-lane group
            const float m_new = fmaxf(m_run[i], mx);
            const float rescale = __expf(m_run[i] - m_new);
            const float p0 = (n < kvn)      ? __expf(c0 - m_new) : 0.0f;
            const float p1 = (16 + n < kvn) ? __expf(c1 - m_new) : 0.0f;
            // l stays a PER-LANE partial (this lane's two columns);
            // the cross-lane tree runs once in the epilogue.
            l_run[i] = l_run[i] * rescale + (p0 + p1);
            m_run[i] = m_new;
            pvals[0][i] = p0;
            pvals[1][i] = p1;
            #pragma unroll
            for (int t = 0; t < EFA_D / 16; ++t) oacc[t][i] *= rescale;
        }

        // ── Bounce P (bf16) into a-frag layout via LDS ──
        // C layout: lane holds P[m = 4*kgrp + i][n = fcol].
        if (mask & 8)
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                lds.p[wave][4 * kgrp + i][half * 16 + fcol] =
                    __float2bfloat16(pvals[half][i]);
        // p is written and read by the SAME wave (its private region);
        // per-wave LDS ordering makes a WG barrier unnecessary — a
        // scheduling fence stops the compiler from hoisting the reads.
        __builtin_amdgcn_wave_barrier();

        // ── PV: O_tile(16 x 256) += P(16 x 32) x V(32 x 256) ──
        // a-frag: lane supplies P[m = fcol][kv = 8*kgrp .. +8]
        const shortx8 pf = *reinterpret_cast<const shortx8*>(
            &lds.p[wave][fcol][kgrp * 8]);
        if (mask & 16)
        #pragma unroll
        for (int t = 0; t < EFA_D / 16; ++t) {
            // b-frag: lane supplies V[kv = 8*kgrp ..+8][d = 16t + fcol]
            // from the transposed tile: vt[d][kv] -> 8 x 2B strided by
            // EFA_KVBLK... transposed layout makes the D-index the row,
            // so the fragment is vt[16t + fcol][8*kgrp ..+8]: contiguous.
            const shortx8 vf = *reinterpret_cast<const shortx8*>(
                &lds.vt[buf][16 * t + fcol][kgrp * 8]);
            oacc[t] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                pf, vf, oacc[t], 0, 0, 0);
        }
    }

    // ── Epilogue: O = acc / l for valid q rows; pad rows in
    // [S, Scap) are ZEROED (never skipped) so downstream consumers of
    // the padded buffer — including FP8 calibration's absmax — see
    // well-defined finite values instead of stale/NaN memory. ──
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = 4 * kgrp + i;
        const int qrow = q0 + m;
        if (qrow >= Scap) continue;
        float lt = l_run[i];
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            lt += __shfl_xor(lt, off);
        const float inv_l =
            (qrow < S && lt > 0.0f) ? 1.0f / lt : 0.0f;
        #pragma unroll
        for (int t = 0; t < EFA_D / 16; ++t) {
            const int d = 16 * t + fcol;
            O[((size_t)qrow * Hq + h) * EFA_D + d] =
                __float2bfloat16(qrow < S ? oacc[t][i] * inv_l : 0.0f);
        }
    }
}

} // namespace

void encoder_attention_flash(const __hip_bfloat16* Q, const __hip_bfloat16* K,
                             const __hip_bfloat16* V, __hip_bfloat16* O,
                             int S, int Hq, int D, float scale,
                             hipStream_t stream, int mask,
                             const int* seqused) {
    if (D != EFA_D)
        throw std::runtime_error("encoder_attention_flash: D must be 256");
    if (S < 1 || S > 4096 || Hq < 1 || Hq > 16)
        throw std::runtime_error("encoder_attention_flash: bad S/Hq");
    static bool lds_configured = false;
    if (!lds_configured) {
        hipFuncSetAttribute(
            reinterpret_cast<const void*>(&encoder_attention_flash_kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(EfaLds));
        lds_configured = true;
    }
    dim3 grid((S + 63) / 64, Hq);
    encoder_attention_flash_kernel<<<grid, EFA_NTHREADS, sizeof(EfaLds), stream>>>(
        Q, K, V, O, S, Hq, scale, mask, seqused);
}
