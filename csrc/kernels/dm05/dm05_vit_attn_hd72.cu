// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 SigLIP self-attention at head_dim 72.
//
// One CTA is 64 query rows (4 warps x 16) of one (view, head) pair; 64-key
// blocks stream through a two-stage cp.async pipeline. The reduction order is
// FA2's on purpose — key blocks walked from last to first, raw scores carried
// in the running max, p = exp2(s * c - m * c) with c = scale * log2 e,
// per-thread fp32 row sums reduced at the end, P rounded to bf16 — because the
// caller calibrated its static FP8 scales against a build that used FA2's
// hdim-96 instantiation for these heads. Reproducing the order is what keeps
// the change from moving that calibration.
//
// See dm05_vit_attn_hd72.cuh for the interface and the shape constraints.

#include "dm05_vit_attn_hd72.cuh"

#include "dm05_common.cuh"

#include <stdexcept>

namespace flash_rt {
namespace dm05 {
namespace {

// head_dim 72, padded to KP = 88 so a row of the K/V tile holds nine 8-wide
// 16-byte groups and stays 16-byte aligned for the ldmatrix loads below.
constexpr int HD = 72;
constexpr int KP = 88;
constexpr int BN = 64;

template <int MT>
__global__ void __launch_bounds__(128)
vit_attn_hd72_kernel(const __nv_bfloat16* __restrict__ qkv, long long ld, int D,
                     __nv_bfloat16* __restrict__ out, long long o_ld, int S, float scale_log2,
                     __nv_fp8_e4m3* __restrict__ q8, const float* __restrict__ q8_scale) {
    constexpr int BM = 64 * MT;
    static_assert(BM <= 2 * BN, "Q aliases one K|V stage");
    extern __shared__ __align__(16) uint8_t smem[];
    // two pipeline stages, each [K: BN rows][V: BN rows]; Q (BM rows) aliases
    // stage 1 until it has been moved to registers
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem);
    auto kst = [&](int b) { return stg + b * 2 * BN * KP; };
    auto vst = [&](int b) { return stg + b * 2 * BN * KP + BN * KP; };
    __nv_bfloat16* sq = kst(1);
    const int qb = blockIdx.x, h = blockIdx.y, view = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const long long row0 = (long long)view * S;
    const __nv_bfloat16* qbase = qkv + (row0 + (long long)qb * BM) * ld + h * HD;
    const __nv_bfloat16* kbase = qkv + row0 * ld + D + h * HD;
    const __nv_bfloat16* vbase = qkv + row0 * ld + 2 * D + h * HD;
    // dims 72..87 of every tile row are zero: the QK^T tail step reduces over
    // them, and the PV tail reads them as zero keys.
    for (int r = tid; r < 4 * BN; r += 128)
        *reinterpret_cast<uint4*>(stg + r * KP + HD) = make_uint4(0, 0, 0, 0);
    __syncthreads();
    auto load_rows = [&](__nv_bfloat16* dst, const __nv_bfloat16* src, int rows) {
        for (int i = tid; i < rows * 9; i += 128) {
            const int r = i / 9, c = i % 9;
            dm05_cp16(dst + r * KP + c * 8, src + (long long)r * ld + c * 8, true);
        }
    };
    const int nblk = S / BN;
    // FA2 walks the key blocks from the last to the first; the same order keeps
    // the online softmax accumulation identical
    load_rows(sq, qbase, BM);
    load_rows(kst(0), kbase + (long long)(nblk - 1) * BN * ld, BN);
    load_rows(vst(0), vbase + (long long)(nblk - 1) * BN * ld, BN);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    uint32_t qa[MT][5][4];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const __nv_bfloat16* qr = sq + (warp * 16 * MT + m * 16 + g) * KP + 2 * t;
#pragma unroll
        for (int kk = 0; kk < 5; ++kk) {
            qa[m][kk][0] = *reinterpret_cast<const uint32_t*>(qr + kk * 16);
            qa[m][kk][1] = *reinterpret_cast<const uint32_t*>(qr + 8 * KP + kk * 16);
            qa[m][kk][2] = *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 8);
            qa[m][kk][3] = *reinterpret_cast<const uint32_t*>(qr + 8 * KP + kk * 16 + 8);
        }
    }
    __syncthreads();
    float o[MT][9][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < 9; ++n) o[m][n][0] = o[m][n][1] = o[m][n][2] = o[m][n][3] = 0.f;
    float mrow[MT][2], lrow[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) { mrow[m][0] = mrow[m][1] = -INFINITY; lrow[m][0] = lrow[m][1] = 0.f; }
    for (int j = 0; j < nblk; ++j) {
        const int buf = j & 1;
        if (j + 1 < nblk) {
            load_rows(kst(buf ^ 1), kbase + (long long)(nblk - 2 - j) * BN * ld, BN);
            load_rows(vst(buf ^ 1), vbase + (long long)(nblk - 2 - j) * BN * ld, BN);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const __nv_bfloat16* kt = kst(buf);
        const __nv_bfloat16* vt = vst(buf);
        float sc[MT][8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int m = 0; m < MT; ++m) sc[m][nt][0] = sc[m][nt][1] = sc[m][nt][2] = sc[m][nt][3] = 0.f;
            // K fragments by ldmatrix: x4 = dims [32p, 32p + 32) of keys nt*8..+7 (two k-steps)
            const __nv_bfloat16* kr = kt + (nt * 8 + (lane & 7)) * KP + (lane >> 3) * 8;
#pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                uint32_t kb[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(kb[0]), "=r"(kb[1]), "=r"(kb[2]), "=r"(kb[3])
                             : "r"((unsigned)__cvta_generic_to_shared(kr + pp * 32)));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    dm05_mma_bf16_16816(sc[m][nt], qa[m][2 * pp], kb[0], kb[1]);
                    dm05_mma_bf16_16816(sc[m][nt], qa[m][2 * pp + 1], kb[2], kb[3]);
                }
            }
            {
                // Tail k-step: head_dim is 72, so dims 72..80 of this 16-wide step
                // are the zero pad. qa[m][4][0..1] and the first ldmatrix register
                // are exactly dims 64..72, so an m16n8k8 reduces over the real
                // dimensions only — bit-identical to the padded m16n8k16, since the
                // pad contributes exactly zero, and it halves this step's ldmatrix.
                uint32_t kb0;
                const __nv_bfloat16* kr2 = kt + (nt * 8 + (lane & 7)) * KP + 64 + ((lane >> 3) & 1) * 8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                             : "=r"(kb0) : "r"((unsigned)__cvta_generic_to_shared(kr2)));
#pragma unroll
                for (int m = 0; m < MT; ++m) dm05_mma_bf16_1688(sc[m][nt], qa[m][4][0], qa[m][4][1], kb0);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(sc[m][nt][0], sc[m][nt][1]));
                mx1 = fmaxf(mx1, fmaxf(sc[m][nt][2], sc[m][nt][3]));
            }
#pragma unroll
            for (int o2 = 2; o2 >= 1; o2 >>= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
            }
            const float mn0 = fmaxf(mrow[m][0], mx0), mn1 = fmaxf(mrow[m][1], mx1);
            const float a0 = dm05_exp2((mrow[m][0] - mn0) * scale_log2);
            const float a1 = dm05_exp2((mrow[m][1] - mn1) * scale_log2);
            const float ms0 = mn0 * scale_log2, ms1 = mn1 * scale_log2;
            mrow[m][0] = mn0; mrow[m][1] = mn1;
            // FA2 order: rescale the running sum, then add each p of the thread in
            // fragment order
            float rs0 = lrow[m][0] * a0, rs1 = lrow[m][1] * a1;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                sc[m][nt][0] = dm05_exp2(fmaf(sc[m][nt][0], scale_log2, -ms0));
                sc[m][nt][1] = dm05_exp2(fmaf(sc[m][nt][1], scale_log2, -ms0));
                sc[m][nt][2] = dm05_exp2(fmaf(sc[m][nt][2], scale_log2, -ms1));
                sc[m][nt][3] = dm05_exp2(fmaf(sc[m][nt][3], scale_log2, -ms1));
                rs0 += sc[m][nt][0]; rs0 += sc[m][nt][1];
                rs1 += sc[m][nt][2]; rs1 += sc[m][nt][3];
            }
            lrow[m][0] = rs0; lrow[m][1] = rs1;
#pragma unroll
            for (int n = 0; n < 9; ++n) { o[m][n][0] *= a0; o[m][n][1] *= a0; o[m][n][2] *= a1; o[m][n][3] *= a1; }
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t a[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                __nv_bfloat162 p0 = __floats2bfloat162_rn(sc[m][2 * kk][0], sc[m][2 * kk][1]);
                __nv_bfloat162 p1 = __floats2bfloat162_rn(sc[m][2 * kk][2], sc[m][2 * kk][3]);
                __nv_bfloat162 p2 = __floats2bfloat162_rn(sc[m][2 * kk + 1][0], sc[m][2 * kk + 1][1]);
                __nv_bfloat162 p3 = __floats2bfloat162_rn(sc[m][2 * kk + 1][2], sc[m][2 * kk + 1][3]);
                a[m][0] = *reinterpret_cast<uint32_t*>(&p0); a[m][1] = *reinterpret_cast<uint32_t*>(&p1);
                a[m][2] = *reinterpret_cast<uint32_t*>(&p2); a[m][3] = *reinterpret_cast<uint32_t*>(&p3);
            }
            const __nv_bfloat16* vrow = vt + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * KP;
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"((unsigned)__cvta_generic_to_shared(vrow + np * 16 + (lane >> 4) * 8)));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    dm05_mma_bf16_16816(o[m][2 * np], a[m], b[0], b[1]);
                    dm05_mma_bf16_16816(o[m][2 * np + 1], a[m], b[2], b[3]);
                }
            }
            uint32_t b0, b1;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b0), "=r"(b1)
                         : "r"((unsigned)__cvta_generic_to_shared(vrow + 64)));
#pragma unroll
            for (int m = 0; m < MT; ++m) dm05_mma_bf16_16816(o[m][8], a[m], b0, b1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        float l0 = lrow[m][0], l1 = lrow[m][1];
#pragma unroll
        for (int o2 = 2; o2 >= 1; o2 >>= 1) {     // FA2 Allreduce<4>: offsets 2 then 1
            l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
            l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
        }
        const float i0 = dm05_rcp(l0), i1 = dm05_rcp(l1);
        const long long r0 = row0 + (long long)qb * BM + warp * 16 * MT + m * 16 + g;
        __nv_bfloat16* op0 = out + r0 * o_ld + h * HD + 2 * t;
        __nv_bfloat16* op1 = op0 + 8 * o_ld;
        if (q8) {
            // The o_projection's static FP8 input written directly: round the
            // attention output to bf16, then quantize_fp8_static's
            // fp8(clamp(x * rcp(s))), matching what the standalone pass would
            // have produced over the same buffer.
            const float qi = dm05_rcp(q8_scale[0]);
            __nv_fp8_e4m3* qp0 = q8 + (op0 - out);
            __nv_fp8_e4m3* qp1 = q8 + (op1 - out);
#pragma unroll
            for (int n = 0; n < 9; ++n) {
                const __nv_bfloat162 b0 = __floats2bfloat162_rn(o[m][n][0] * i0, o[m][n][1] * i0);
                const __nv_bfloat162 b1 = __floats2bfloat162_rn(o[m][n][2] * i1, o[m][n][3] * i1);
                __align__(2) __nv_fp8_e4m3 f0[2], f1[2];
                f0[0] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(__low2float(b0), qi), -448.f), 448.f));
                f0[1] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(__high2float(b0), qi), -448.f), 448.f));
                f1[0] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(__low2float(b1), qi), -448.f), 448.f));
                f1[1] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(__high2float(b1), qi), -448.f), 448.f));
                *reinterpret_cast<uint16_t*>(qp0 + n * 8) = *reinterpret_cast<uint16_t*>(f0);
                *reinterpret_cast<uint16_t*>(qp1 + n * 8) = *reinterpret_cast<uint16_t*>(f1);
            }
        } else {
#pragma unroll
            for (int n = 0; n < 9; ++n) {
                *reinterpret_cast<__nv_bfloat162*>(op0 + n * 8) = __floats2bfloat162_rn(o[m][n][0] * i0, o[m][n][1] * i0);
                *reinterpret_cast<__nv_bfloat162*>(op1 + n * 8) = __floats2bfloat162_rn(o[m][n][2] * i1, o[m][n][3] * i1);
            }
        }
    }
}

}  // namespace

void vit_attn_hd72(const __nv_bfloat16* qkv, long long ld, int D,
                   __nv_bfloat16* out, long long o_ld,
                   int S, int NH, int nv, float scale, int mt,
                   __nv_fp8_e4m3* q8, const float* q8_scale,
                   cudaStream_t stream) {
    // Shape constraints of the kernel, checked here rather than left to produce
    // a result with a different reduction order. D == NH * 72 is the fused
    // buffer's layout; S % 128 == 0 covers the 64-row Q blocks with MT <= 2;
    // D % 8 == 0 makes the 16-byte qkv loads in bounds.
    if (S <= 0 || NH <= 0 || nv <= 0)
        throw std::runtime_error("dm05_vit_attn_hd72: S, NH and nv must be positive");
    if (D != NH * HD)
        throw std::runtime_error("dm05_vit_attn_hd72: D must equal NH * 72");
    if (S % 128 != 0)
        throw std::runtime_error("dm05_vit_attn_hd72: S must be a multiple of 128");
    if (D % 8 != 0)
        throw std::runtime_error("dm05_vit_attn_hd72: D must be a multiple of 8");
    if (mt != 1 && mt != 2)
        throw std::runtime_error("dm05_vit_attn_hd72: mt must be 1 or 2");
    if (ld < 3 * D || o_ld < D)
        throw std::runtime_error("dm05_vit_attn_hd72: row strides are smaller than the row");

    constexpr size_t sm = (size_t)4 * BN * KP * 2;
    // as fa2_wrapper: softmax_scale * float(M_LOG2E)
    const float scale_log2 = scale * float(1.4426950408889634);

    if (mt == 1) {
        static bool attr1 = false;
        if (!attr1) {
            cudaFuncSetAttribute(vit_attn_hd72_kernel<1>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            attr1 = true;
        }
        vit_attn_hd72_kernel<1><<<dim3(S / 64, NH, nv), 128, sm, stream>>>(
            qkv, ld, D, out, o_ld, S, scale_log2, q8, q8_scale);
    } else {
        static bool attr2 = false;
        if (!attr2) {
            cudaFuncSetAttribute(vit_attn_hd72_kernel<2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            attr2 = true;
        }
        vit_attn_hd72_kernel<2><<<dim3(S / 128, NH, nv), 128, sm, stream>>>(
            qkv, ld, D, out, o_ld, S, scale_log2, q8, q8_scale);
    }
}

}  // namespace dm05
}  // namespace flash_rt
