// SPDX-License-Identifier: Apache-2.0
//
// M=16 NVFP4-weight GEMM as a register-direct weight stream; see
// fp4_stream_m16_sm89.cuh for the contract and the numerics argument.
//
// One warp owns one n8 tile (8 output columns) and walks its whole K range:
// a DEPTH-deep register ring keeps the next 16-byte fragment loads (plus the
// SF byte and the three block scales they need) in flight while the current
// 128-K block is decoded and multiplied. Measured at the Wan2.2 action-stream
// shapes (M=16, real checkpoint weights, cold L2, RTX 4060 Ti): 3072x3072
// 27.2 -> 23.2 us, 9216x3072 75.0 -> 64.4 us, 14336x3072 115.3 -> 90.9 us,
// 3072x14336 100.8 -> 91.0 us against the shared-memory kernels, all
// bit-identical.

#include "fp4_stream_m16_sm89.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

__device__ __forceinline__ void mma_e4m3_(float& d0, float& d1, float& d2, float& d3,
                                          uint32_t a0, uint32_t a1, uint32_t a2,
                                          uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4_(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                         uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// Same decode as ue4m3_sf_to_fp32 in the shared-memory kernels.
__device__ __forceinline__ float ue4m3_sf_(uint32_t v) {
    const int e = (v >> 3) & 0xF;
    const int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// Streaming 16-byte load: read once, never reused, so keep it out of L1.
__device__ __forceinline__ uint4 ldg_stream_(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t prmt_(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

__device__ __forceinline__ uint32_t cvt2_(float a, float b) {
    return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE,
                                              __NV_E4M3);
}

// Four e2m1 nibbles (low 16 bits of w) -> four e4m3 bytes through the LUT:
// P0/P1 hold codes 0-7, N0/N1 codes 8-15; bit 3 of each nibble picks the half.
__device__ __forceinline__ uint32_t dec4_(uint32_t w, uint32_t P0, uint32_t P1,
                                          uint32_t N0, uint32_t N1) {
    const uint32_t sel = w & 0x7777u;
    const uint32_t p = prmt_(P0, P1, sel);
    const uint32_t n = prmt_(N0, N1, sel);
    return prmt_(p, n, ((w >> 1) & 0x4444u) | 0x3210u);
}

struct Slot { uint32_t sf; float ws, as0, as1; };

template <int DEPTH, int WARPS, bool A_SMEM>
__global__ __launch_bounds__(WARPS * 32)
void fp4_stream_m16_kernel(const uint8_t* __restrict__ A,
                           const uint4* __restrict__ Wf,
                           const uint8_t* __restrict__ Bsf,
                           const float* __restrict__ gscale,
                           const float* __restrict__ act_scale,
                           const float* __restrict__ w_scale,
                           __nv_bfloat16* __restrict__ D,
                           __nv_bfloat16* __restrict__ residual,
                           const __nv_bfloat16* __restrict__ bias,
                           int N, int K, int per_split) {
    extern __shared__ __align__(128) uint8_t As[];
    const int K128 = K >> 7;
    const int NT = N >> 3;
    const int ngroups = (NT + WARPS - 1) / WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const float gs = gscale[0];
    // e2m1 codebook, exact in fp32 (the shared-memory kernels hold it as fp16)
    const float cb[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    // ldmatrix lane geometry of the shared-memory kernels
    const int arow = ((lane >> 4) << 3) + (lane & 7);
    const int acol = (lane >> 3) & 1;
    const uint32_t As_base = (uint32_t)__cvta_generic_to_shared(As);
    const uint8_t* a0p = A + (size_t)g * K + 4 * t;
    const uint8_t* a1p = A + (size_t)(g + 8) * K + 4 * t;

    if (A_SMEM) {
        // A is identical for every unit: stage it once, 128B-swizzled per row.
        const int chunks = 16 * (K >> 4);
        for (int c = threadIdx.x; c < chunks; c += WARPS * 32) {
            const int row = c / (K >> 4);
            const int ch = c - row * (K >> 4);
            const uint4 v = *reinterpret_cast<const uint4*>(A + (size_t)row * K + ch * 16);
            *reinterpret_cast<uint4*>(As + row * K + (ch >> 3) * 128
                                      + (((ch & 7) ^ (row & 7)) << 4)) = v;
        }
        __syncthreads();
    }

    for (int grp = blockIdx.x; grp < ngroups; grp += gridDim.x) {
        const int tile = grp * WARPS + warp;
        if (tile >= NT) continue;
        const int n = tile * 8 + g;
        const uint4* wp = Wf + (size_t)tile * K128 * 32 + lane;
        const uint8_t* sfp = Bsf + (size_t)(n >> 7) * K128 * 128 + (n & 127);
        const float* wsp = w_scale + (size_t)(n >> 7) * K128;
        const float* as0p = act_scale + (size_t)g * K128;
        const float* as1p = act_scale + (size_t)(g + 8) * K128;

        uint4 ring[DEPTH];
        Slot rs[DEPTH];
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            ring[d] = ldg_stream_(wp + d * 32);
            rs[d].sf = __ldg(sfp + d * 128);
            rs[d].ws = __ldg(wsp + d);
            rs[d].as0 = __ldg(as0p + d);
            rs[d].as1 = __ldg(as1p + d);
        }
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        float tot0 = 0.f, tot1 = 0.f, tot2 = 0.f, tot3 = 0.f;
        for (int kb0 = 0; kb0 < K128; kb0 += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int kb = kb0 + d;
                const uint4 wv = ring[d];
                const Slot sc = rs[d];
                const int nk = kb + DEPTH;
                if (nk < K128) {
                    ring[d] = ldg_stream_(wp + (size_t)nk * 32);
                    rs[d].sf = __ldg(sfp + (size_t)nk * 128);
                    rs[d].ws = __ldg(wsp + nk);
                    rs[d].as0 = __ldg(as0p + nk);
                    rs[d].as1 = __ldg(as1p + nk);
                }
                // L1 path: issue all sixteen A-fragment loads before the LUT
                // work so their latency overlaps it (one wait per block, not four).
                uint32_t af[4][4];
                if (!A_SMEM) {
#pragma unroll
                    for (int ka = 0; ka < 4; ++ka) {
                        const int ko = kb * 128 + ka * 32;
                        af[ka][0] = __ldg(reinterpret_cast<const uint32_t*>(a0p + ko));
                        af[ka][1] = __ldg(reinterpret_cast<const uint32_t*>(a1p + ko));
                        af[ka][2] = __ldg(reinterpret_cast<const uint32_t*>(a0p + ko + 16));
                        af[ka][3] = __ldg(reinterpret_cast<const uint32_t*>(a1p + ko + 16));
                    }
                }
                // The 16 e4m3 bytes this row's K-block can take, computed with
                // the expansion's exact expression `code * (sf * gs * 1/ws)`.
                const float ws_inv = (sc.ws > 0.0f) ? (1.0f / sc.ws) : 0.0f;
                const float s = ue4m3_sf_(sc.sf) * gs * ws_inv;
                const uint32_t P0 = cvt2_(cb[0] * s, cb[1] * s) | (cvt2_(cb[2] * s, cb[3] * s) << 16);
                const uint32_t P1 = cvt2_(cb[4] * s, cb[5] * s) | (cvt2_(cb[6] * s, cb[7] * s) << 16);
                const uint32_t N0 = cvt2_(-cb[0] * s, -cb[1] * s) | (cvt2_(-cb[2] * s, -cb[3] * s) << 16);
                const uint32_t N1 = cvt2_(-cb[4] * s, -cb[5] * s) | (cvt2_(-cb[6] * s, -cb[7] * s) << 16);
                float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
                const uint32_t wa[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                for (int ka = 0; ka < 4; ++ka) {
                    uint32_t a0, a1, a2, a3;
                    if (A_SMEM) {
                        const int chunk = 2 * ka + acol;
                        ldsm_x4_(a0, a2, a1, a3,
                                 As_base + arow * K + kb * 128 + ((chunk ^ (arow & 7)) << 4));
                    } else {
                        a0 = af[ka][0]; a1 = af[ka][1]; a2 = af[ka][2]; a3 = af[ka][3];
                    }
                    const uint32_t b0 = dec4_(wa[ka], P0, P1, N0, N1);
                    const uint32_t b1 = dec4_(wa[ka] >> 16, P0, P1, N0, N1);
                    mma_e4m3_(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                }
                acc0 += c0 * (sc.as0 * sc.ws);
                acc1 += c1 * (sc.as0 * sc.ws);
                acc2 += c2 * (sc.as1 * sc.ws);
                acc3 += c3 * (sc.as1 * sc.ws);
                if ((kb + 1) % per_split == 0) {
                    // split boundary: the split-K reduce's `s += part[sp]`
                    tot0 += acc0; tot1 += acc1; tot2 += acc2; tot3 += acc3;
                    acc0 = acc1 = acc2 = acc3 = 0.f;
                }
            }
        }
        const int col = tile * 8 + 2 * t;
        if (residual == nullptr) {
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)g * N + col]) =
                __floats2bfloat162_rn(tot0, tot1);
            *reinterpret_cast<__nv_bfloat162*>(&D[(size_t)(g + 8) * N + col]) =
                __floats2bfloat162_rn(tot2, tot3);
        } else {
            // the split-K reduce's residual fold, operand for operand
            const float tv[4] = {tot0, tot1, tot2, tot3};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = g + ((j >> 1) << 3);
                const int c = col + (j & 1);
                const float p = __bfloat162float(__float2bfloat16_rn(tv[j]));
                const float v = __bfloat162float(__float2bfloat16_rn(
                    __fadd_rn(p, __bfloat162float(bias[c]))));
                __nv_bfloat16* r = residual + (size_t)row * N + c;
                *r = __float2bfloat16_rn(__bfloat162float(*r) + v);
            }
        }
    }
}

__global__ void nvfp4_pack_frag_m16_kernel(const uint8_t* __restrict__ src,
                                           uint32_t* __restrict__ dst, int N, int K) {
    const int K128 = K >> 7;
    const size_t total = (size_t)(N >> 3) * K128 * 128;   // 32-bit words
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const int ka = i & 3;
        const int lane = (i >> 2) & 31;
        const size_t tk = i >> 7;
        const int kb = (int)(tk % K128);
        const int tile = (int)(tk / K128);
        const int n = tile * 8 + (lane >> 2);
        // word = this lane's m16n8k32 B fragment: nibbles 0-3 are k..k+3,
        // nibbles 4-7 are k+16..k+19
        const int kbase = kb * 128 + ka * 32 + 4 * (lane & 3);
        uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int k = kbase + (j & 3) + ((j >> 2) << 4);
            const uint8_t byte = src[(size_t)n * (K >> 1) + (k >> 1)];
            w |= (uint32_t)((k & 1) ? (byte >> 4) : (byte & 0xF)) << (4 * j);
        }
        dst[i] = w;
    }
}

template <int DEPTH, int WARPS, bool A_SMEM>
int launch_(const void* A, const void* Wf, const void* Bsf, const float* gscale,
            const float* act_scale, const float* w_scale, void* D, void* residual,
            const void* bias, int N, int K, int per_split, cudaStream_t s) {
    auto kfn = fp4_stream_m16_kernel<DEPTH, WARPS, A_SMEM>;
    const int smem = A_SMEM ? 16 * K : 0;
    const int groups = (N / 8 + WARPS - 1) / WARPS;
    int grid = groups;
    if (A_SMEM) {
        // persistent: A is staged once per CTA, so launch one resident wave
        static int occ = 0;
        if (occ == 0) {
            cudaFuncSetAttribute((const void*)kfn,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            int dev = 0, sms = 0, per = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kfn, WARPS * 32, 16 * 3072);
            occ = per * sms;
        }
        if (grid > occ) grid = occ;
    }
    kfn<<<grid, WARPS * 32, smem, s>>>(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint4*>(Wf),
        reinterpret_cast<const uint8_t*>(Bsf), gscale, act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D), reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(bias), N, K, per_split);
    return (int)cudaGetLastError();
}

}  // namespace

int fp4_stream_m16_sm89(const void* A, const void* Wf, const void* Bsf,
                        const float* gscale, const float* act_scale,
                        const float* w_scale, void* D, void* residual,
                        const void* bias, int M, int N, int K, int splits,
                        int a_smem, int depth, int warps, cudaStream_t s) {
    if (M != 16 || (N % 8) != 0 || (K % 128) != 0) return -1;
    const int K128 = K / 128;
    if (splits <= 0 || (K128 % splits) != 0) return -2;
    if ((K128 % depth) != 0) return -3;
    if (a_smem && K > 3072) return -4;            // 16*K bytes of shared memory
    if ((residual == nullptr) != (bias == nullptr)) return -5;
    const int per = K128 / splits;
#define FP4S_CASE(DP, WP, AS)                                                     \
    if (depth == DP && warps == WP && (a_smem != 0) == AS)                        \
        return launch_<DP, WP, AS>(A, Wf, Bsf, gscale, act_scale, w_scale, D,     \
                                   residual, bias, N, K, per, s);
    FP4S_CASE(2, 4, false) FP4S_CASE(4, 4, false) FP4S_CASE(2, 8, false)
    FP4S_CASE(4, 8, false) FP4S_CASE(2, 4, true) FP4S_CASE(4, 4, true)
    FP4S_CASE(2, 8, true) FP4S_CASE(4, 8, true)
#undef FP4S_CASE
    return -6;
}

int nvfp4_pack_frag_m16(const void* packed, void* frag, int N, int K,
                        cudaStream_t s) {
    if ((N % 8) != 0 || (K % 128) != 0) return -1;
    nvfp4_pack_frag_m16_kernel<<<1024, 256, 0, s>>>(
        reinterpret_cast<const uint8_t*>(packed), reinterpret_cast<uint32_t*>(frag),
        N, K);
    return (int)cudaGetLastError();
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
