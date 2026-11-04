// SPDX-License-Identifier: Apache-2.0
#pragma once

// SM89 GEMM whose weight operand is stored NVFP4 instead of FP8 e4m3.
//
// Why this exists: NVFP4's *MMA* (`...kind::f8f6f4`) is sm_120a-only, but that
// op is not what a narrow-M site needs. Every site that uses this kernel runs
// at M <= 32, where the GEMM is pure weight bandwidth and the tensor core is
// idle -- the action stream of the Wan2.2 block stack re-reads the whole
// weight set 51 times a chunk for 16-token forwards. There, halving the weight
// bytes is the entire win, and the plain Ada FP8 MMA is three orders of
// magnitude faster than it needs to be.
//
// MEASURED SCOPE, so nobody re-derives it: this only pays at narrow M. At M=16
// it beats the FP8 kernel on half the weight bytes where the launch is wide
// enough to hide the expansion (1.11x at N=9216, 1.45x at N=14336, 0.81-0.90x
// at the 48-CTA shapes). At M=256 it loses outright -- 0.51x at BM=64 and 0.36x
// at BM=64/BN=128 -- because a wide-M GEMM is already tensor-pipe bound, so the
// expansion is pure addition instead of overlapped work even before it is re-run
// once per m-tile. The video stream therefore stays on FP8.
//
// So the kernel keeps fp8_bs_gemm_kernel's structure verbatim -- same cp.async
// pipeline, same 128B swizzle, same ldmatrix + m16n8k32 e4m3 MMA, same
// block-128 activation/weight scale fold -- and changes only how the B stage is
// filled: the packed NVFP4 tile is cp.async'd into a raw staging buffer and
// expanded to FP8 in shared memory before ldmatrix.
//
// The expansion reproduces `fp8_block128_to_nvfp4_swizzled_bf16`'s
// reconstruction exactly (code * ue4m3_sf * global_scale), which is defined on
// the *normalized* FP8 weight the pipeline already stores, so the existing
// per-128-block `w_scale` fold stays correct and unchanged. The only added
// rounding is that reconstruction landing in e4m3 rather than fp32: 2.6%
// relative on the real checkpoint, against the 10.0% the FP4 codebook itself
// costs (both measured, and the E2E cosine gate passes on the combination).

#include "fp8_bs_gemm_device.cuh"

namespace flash_rt {
namespace gemm {
namespace block128_sm89_splitk {

using namespace flash_rt::gemm::block128_sm89;

// E2M1 magnitude for codes 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6}. Bit 2 is the
// low exponent bit, bit 0 the mantissa, so the value is mant * 2^(e) with
// mant in {1, 1.5}; codes 0 and 1 are the two subnormals. Built from the
// exponent field rather than ldexpf so it is exact and branch-free.
__device__ __forceinline__ float e2m1_mag(uint32_t m) {
    const float mant = ((m & 1u) != 0u) ? 1.5f : 1.0f;
    const float two_e = __uint_as_float(((m >> 1) + 126u) << 23);  // 2^((m>>1)-1)
    return (m < 2u) ? (0.5f * (float)m) : (mant * two_e);
}

// ue4m3_to_fp32, the same decode csrc/quantize/fp8_block128_to_nvfp4_swizzled.cu
// uses when it writes the SF byte.
__device__ __forceinline__ float ue4m3_sf_to_fp32(uint8_t v) {
    const int e = (v >> 3) & 0xF;
    const int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// BLOCK_K is pinned to 128, as in the FP8 kernel: one k-iteration is exactly
// one DeepSeek block-128 scale block, and the packed NVFP4 row segment for one
// such block is 64 bytes.
//
//  - A   : [M, K] row-major FP8 e4m3, act_scale [M, K/128] fp32
//  - Bq  : [N, K/2] packed e2m1 nibbles (low nibble = even k)
//  - Bsf : UE4M3 scale factors for Bq in CUTLASS Sm1xxBlockScaledConfig<16>
//          order -- the layout fp8_block128_to_nvfp4_swizzled_bf16 writes
//  - gscale: [1] fp32 per-tensor global scale from the same quantizer
//  - w_scale: [N/128, K/128] fp32 from the *FP8* block-128 quantization
//  - D   : [M, N] row-major BF16
//
// EXACT says M == BLOCK_M, N % BLOCK_N == 0 and K % BLOCK_K == 0, which is true
// of every production site. It removes the per-element bounds checks from the
// staging, the expansion and the epilogue: those guards were not free, they
// forced the whole expansion body behind a branch and kept the address
// arithmetic from being hoisted (SASS: 52 BSYNC/BSSY plus ~580 integer ops per
// K-iteration, against 32 fp8 converts of real work).
template <int BLOCK_M, int BLOCK_N, int NUM_WARPS, int STAGES,
          int MIN_BLOCKS_PER_SM, bool EXACT = false, bool TILE_MAJOR = false,
          bool SF128 = false>
__global__ __launch_bounds__(NUM_WARPS * 32, MIN_BLOCKS_PER_SM)
void fp4_bs_gemm_splitk_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const uint8_t* __restrict__ Bq,
    const uint8_t* __restrict__ Bsf,
    const float* __restrict__ gscale,
    const float* __restrict__ act_scale,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ D,
    float* __restrict__ Dpart,
    int M, int N, int K, int splits)
{
    constexpr int BLOCK_K    = 128;
    constexpr int THREADS    = NUM_WARPS * 32;
    constexpr int M_ATOMS    = BLOCK_M / 16;
    constexpr int N_ATOMS    = BLOCK_N / 8;
    constexpr int N_ATOMS_PW = N_ATOMS / NUM_WARPS;
    constexpr int N_PAIRS_PW = N_ATOMS_PW / 2;
    constexpr int K_ATOMS    = BLOCK_K / 32;         // = 4
    constexpr int NUM_CHUNKS_PER_ROW = BLOCK_K / 16; // 8 fp8 chunks of 16B
    constexpr int Q_CHUNKS_PER_ROW   = (BLOCK_K / 2) / 16;  // 4 fp4 chunks of 16B
    constexpr int SWIZZLE_MASK   = NUM_CHUNKS_PER_ROW - 1;  // = 7
    constexpr int Q_SWIZZLE_MASK = Q_CHUNKS_PER_ROW - 1;    // = 3

    static_assert(BLOCK_M % 16 == 0, "BLOCK_M multiple of 16");
    static_assert(BLOCK_N % 8 == 0,  "BLOCK_N multiple of 8");
    static_assert(BLOCK_N <= 128, "one CTA must fit one N scale block");
    static_assert((BLOCK_N / 8) % NUM_WARPS == 0, "N-atoms split across warps");
    static_assert(N_ATOMS_PW >= 2 && N_ATOMS_PW % 2 == 0,
                  "ldmatrix pairs 2 N-atoms: N_ATOMS_PW must be even >= 2");
    static_assert(BLOCK_K % 32 == 0, "BLOCK_K multiple of the MMA k-atom");
    // The per-thread expansion plan is built without an in-loop bounds check
    // (the check had to go: it kept the address arithmetic from being hoisted).
    // That is only sound because EXP_CHUNKS divides evenly by THREADS for every
    // instantiation, so the last iteration's idx is still in range.
    static_assert(((BLOCK_N * (BLOCK_K / 16)) % (NUM_WARPS * 32)) == 0,
                  "expansion chunks must divide evenly across the block");

    constexpr int SCALE_KTILE = 8;
    constexpr int A_TILE  = BLOCK_M * BLOCK_K;
    constexpr int BQ_TILE = BLOCK_N * (BLOCK_K / 2);
    constexpr int B_TILE  = BLOCK_N * BLOCK_K;

    extern __shared__ uint8_t smem_raw[];
    uint8_t* A_smem  = smem_raw;                        // STAGES * A_TILE
    uint8_t* Bq_smem = A_smem + STAGES * A_TILE;        // STAGES * BQ_TILE
    // The expanded FP8 tile is produced by this CTA's own ALU, not by a
    // latency-bound cp.async, so it needs no stage depth: one buffer, guarded
    // by the barriers that already bracket the ldmatrix.
    uint8_t* B_smem  = Bq_smem + STAGES * BQ_TILE;      // B_TILE
    float* as_smem = reinterpret_cast<float*>(B_smem + B_TILE);
    float* ws_smem = as_smem + BLOCK_M * SCALE_KTILE;
    uint8_t* maglut = reinterpret_cast<uint8_t*>(ws_smem + SCALE_KTILE);
    float* sf_lut = reinterpret_cast<float*>(maglut + 64);   // 256 UE4M3 bytes

    const int cta_m = blockIdx.x;
    const int cta_n = blockIdx.y;
    const int m_base = cta_m * BLOCK_M;
    const int n_base = cta_n * BLOCK_N;

    // Split-K: this CTA owns k-iterations [k_begin, k_end) and writes its
    // partial sum (fp32) to its own slice of Dpart. The reduction kernel sums
    // the slices and performs the single bf16 rounding.
    const int split = blockIdx.z;
    const int K_ITERS = (K + BLOCK_K - 1) / BLOCK_K;
    const int per_split = (K_ITERS + splits - 1) / splits;
    const int k_begin = split * per_split;
    const int k_end = (k_begin + per_split < K_ITERS) ? (k_begin + per_split) : K_ITERS;
    if (k_begin >= k_end) return;

    const int t = threadIdx.x;
    const int warp_id = t / 32;
    const int lane = t % 32;
    const int l = lane % 4;
    const int h = lane / 4;
    const int frag_group = lane / 8;
    const int row_in_frag = lane % 8;
    const int row_block = frag_group / 2;
    const int col_block = frag_group % 2;

    // FP16 codebook for the 16 e2m1 nibbles, one entry per 4-byte slot so the
    // 16 entries land in 16 distinct banks. Decoding the magnitude
    // arithmetically costs ~9 instructions per element and was the single
    // largest cost in the expansion; a lookup is one LDS.
    if (t < 16) {
        const float v = (t & 8) ? -e2m1_mag((uint32_t)(t & 7))
                                :  e2m1_mag((uint32_t)(t & 7));
        reinterpret_cast<float*>(maglut)[t] = v;
    }
    // Decode the 256 possible UE4M3 scale-factor bytes once per CTA. The
    // per-chunk decode (branch + ldexpf) was ~9% of the expansion ALU, and the
    // fp32 codebook removes two half->float converts per element pair. Both
    // changes are exact: the codebook values are binary-exact and the LUT
    // entries are the same decode.
    for (int i = t; i < 256; i += THREADS) sf_lut[i] = ue4m3_sf_to_fp32((uint8_t)i);
    __syncthreads();

    const int K128 = K >> 7;
    const int n_col_super = ((K >> 4) + 3) >> 2;   // ceil(K/16 / 4)
    const float gs = gscale[0];
    const uint8_t* Bsf_cta = Bsf;

    auto stage_scales = [&](int kb0) {
        const int as_total = BLOCK_M * SCALE_KTILE;
        for (int idx = t; idx < as_total; idx += THREADS) {
            int r = idx / SCALE_KTILE;
            int kc = idx - r * SCALE_KTILE;
            int row = m_base + r;
            int kb = kb0 + kc;
            as_smem[idx] = (row < M && kb < K128)
                ? act_scale[(size_t)row * K128 + kb] : 0.0f;
        }
        for (int kc = t; kc < SCALE_KTILE; kc += THREADS) {
            int kb = kb0 + kc;
            ws_smem[kc] = (kb < K128)
                ? w_scale[(size_t)(n_base >> 7) * K128 + kb] : 0.0f;
        }
        __syncthreads();
    };

    auto issue_load = [&](int stage, int k_base) {
        constexpr int A_CHUNKS = BLOCK_M * NUM_CHUNKS_PER_ROW;
        constexpr int A_ITERS = (A_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= A_CHUNKS) break;
            int row_a = idx / NUM_CHUNKS_PER_ROW;
            int chunk_a = idx % NUM_CHUNKS_PER_ROW;
            int m_glob = m_base + row_a;
            int k_glob = k_base + chunk_a * 16;
            const uint8_t* a_src = nullptr;
            if (EXACT || (m_glob < M && k_glob < K)) {
                a_src = reinterpret_cast<const uint8_t*>(&A[(size_t)m_glob * K + k_glob]);
            }
            int csw = chunk_a ^ (row_a & SWIZZLE_MASK);
            cp_async_16(
                to_smem(&A_smem[stage * A_TILE + row_a * BLOCK_K + csw * 16]),
                a_src);
        }
        // B arrives as packed NVFP4: half the bytes, same 16-byte chunking.
        constexpr int BQ_CHUNKS = BLOCK_N * Q_CHUNKS_PER_ROW;
        constexpr int BQ_ITERS = (BQ_CHUNKS + THREADS - 1) / THREADS;
        #pragma unroll
        for (int it = 0; it < BQ_ITERS; ++it) {
            int idx = it * THREADS + t;
            if (idx >= BQ_CHUNKS) break;
            int row_b = idx / Q_CHUNKS_PER_ROW;
            int chunk_b = idx % Q_CHUNKS_PER_ROW;
            int n_glob = n_base + row_b;
            int kq = (k_base >> 1) + chunk_b * 16;
            const uint8_t* b_src = nullptr;
            if (EXACT || (n_glob < N && (k_base + chunk_b * 32) < K)) {
                if (TILE_MAJOR) {
                    // Tile-major packed weights: see the unsplit kernel and
                    // nvfp4_pack_tile_major.cuh. Pure address change.
                    b_src = Bq + (((size_t)(n_glob >> 6) * (K >> 7)
                                   + (k_base >> 7)) << 12)
                           + (size_t)(n_glob & 63) * 64 + chunk_b * 16;
                } else {
                    b_src = &Bq[(size_t)n_glob * (K >> 1) + kq];
                }
            }
            int csq = chunk_b ^ (row_b & Q_SWIZZLE_MASK);
            cp_async_16(
                to_smem(&Bq_smem[stage * BQ_TILE
                                 + row_b * (BLOCK_K >> 1) + csq * 16]),
                b_src);
        }
    };

    // Per-thread expansion plan. None of this depends on K, so it is computed
    // once instead of inside every K-iteration: as written inline it was ~10
    // integer instructions per weight element in SASS, i.e. more than the
    // conversion it was feeding.
    constexpr int EXP_CHUNKS = BLOCK_N * NUM_CHUNKS_PER_ROW;
    constexpr int EXP_ITERS = (EXP_CHUNKS + THREADS - 1) / THREADS;
    int e_src[EXP_ITERS], e_dst[EXP_ITERS], e_sfrow[EXP_ITERS], e_chunk[EXP_ITERS];
    int e_row[EXP_ITERS];
    #pragma unroll
    for (int it = 0; it < EXP_ITERS; ++it) {
        const int idx = it * THREADS + t;
        const int row_b = idx / NUM_CHUNKS_PER_ROW;
        const int chunk = idx % NUM_CHUNKS_PER_ROW;
        const int n_glob = n_base + row_b;
        const int csq = (chunk >> 1) ^ (row_b & Q_SWIZZLE_MASK);
        e_src[it] = row_b * (BLOCK_K >> 1) + csq * 16;
        e_dst[it] = row_b * BLOCK_K
                  + (chunk ^ (row_b & SWIZZLE_MASK)) * 16;
        e_sfrow[it] = (n_glob >> 7) * n_col_super * 512
                    + ((n_glob & 127) & 31) * 16
                    + ((n_glob & 127) >> 5) * 4;
        e_row[it] = n_glob;
        e_chunk[it] = chunk;
    }

    // Expand one staged NVFP4 K-block into the FP8 B tile the MMA consumes.
    // One thread per 16-value FP8 chunk, which is exactly one NVFP4 scale
    // block: the SF byte, the 8 packed bytes and the 16 outputs line up.
    //
    // `ws_inv` is 1/w_scale for this K-block. The NVFP4 was quantized from the
    // *dequantized* FP8 weight, so `code * sf * gscale` reconstructs the
    // absolute weight; dividing by w_scale re-normalizes it to the operand the
    // FP8 GEMM's own scale fold expects, and that fold stays untouched.
    auto expand_b = [&](int stage, int k_base, float ws_inv) {
        const uint8_t* bq = Bq_smem + stage * BQ_TILE;
        #pragma unroll
        for (int it = 0; it < EXP_ITERS; ++it) {
            const int chunk = e_chunk[it];
            const int b = (k_base >> 4) + chunk;          // 16-wide SF block
            // SF128: one UE4M3 byte per (row, 128-K block), layout
            // [N/128, K/128, 128]. Same bytes as the duplicated per-16
            // packing with 8x fewer reads.
            const uint8_t sf_byte = SF128
                ? Bsf[(((size_t)(e_row[it] >> 7) * (K >> 7) + (k_base >> 7)) << 7)
                      + (e_row[it] & 127)]
                : Bsf_cta[e_sfrow[it] + (b >> 2) * 512 + (b & 3)];
            const float s = sf_lut[sf_byte] * gs * ws_inv;

            // 16 fp4 values live in 8 bytes = the high/low halves of the two
            // 16-byte staging chunks this FP8 chunk spans.
            const uint4 ev = *reinterpret_cast<const uint4*>(bq + e_src[it]);
            const uint32_t word[2] = { (chunk & 1) ? ev.z : ev.x,
                                       (chunk & 1) ? ev.w : ev.y };

            // Nibble -> fp16 magnitude (sign is in the codebook entry), scaled,
            // then two values per fp8 convert. 8 nibbles per 32-bit word.
            uint16_t wy[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t byte = (word[j >> 2] >> (8 * (j & 3))) & 0xFFu;
                const float* lut = reinterpret_cast<const float*>(maglut);
                const float lo = lut[byte & 0x0Fu];
                const float hi = lut[byte >> 4];
                wy[j] = __nv_cvt_float2_to_fp8x2(
                    make_float2(lo * s, hi * s),
                    __NV_SATFINITE, __NV_E4M3);
            }
            uint4 pack;
            pack.x = (uint32_t)wy[0] | ((uint32_t)wy[1] << 16);
            pack.y = (uint32_t)wy[2] | ((uint32_t)wy[3] << 16);
            pack.z = (uint32_t)wy[4] | ((uint32_t)wy[5] << 16);
            pack.w = (uint32_t)wy[6] | ((uint32_t)wy[7] << 16);
            *reinterpret_cast<uint4*>(B_smem + e_dst[it]) = pack;
        }
    };

    float acc[M_ATOMS][N_ATOMS_PW][4];
    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[mi][ni][j] = 0.0f;

    // The split's first k-iteration may not start on a SCALE_KTILE boundary
    // (splits=4 starts at kb=6 for K=3072), and the scales for that block are
    // read before the in-loop restage would fill them. Stage the containing
    // block up front; the loop restages at every multiple of SCALE_KTILE.
    stage_scales((k_begin / SCALE_KTILE) * SCALE_KTILE);

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        int it = k_begin + s;
        if (it < k_end) issue_load(it % STAGES, it * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    int compute_stage = k_begin % STAGES;
    for (int k_iter = k_begin; k_iter < k_end; ++k_iter) {
        int issue_iter = k_iter + (STAGES - 1);
        int issue_stage = issue_iter % STAGES;
        if (issue_iter < k_end) issue_load(issue_stage, issue_iter * BLOCK_K);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 1));
        __syncthreads();

        // The raw NVFP4 stage is now resident; expand it into the FP8 tile.
        // The scales stage first: the expansion needs this K-block's 1/w_scale.
        const int kb = k_iter;
        if ((kb % SCALE_KTILE) == 0) stage_scales(kb);
        const float ws_here = ws_smem[kb % SCALE_KTILE];
        expand_b(compute_stage, kb * BLOCK_K,
                 (ws_here > 0.0f) ? (1.0f / ws_here) : 0.0f);
        __syncthreads();

        float tacc[M_ATOMS][N_ATOMS_PW][4];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi)
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni)
                #pragma unroll
                for (int j = 0; j < 4; ++j) tacc[mi][ni][j] = 0.0f;

        uint8_t* A_stage = A_smem + compute_stage * A_TILE;
        #pragma unroll
        for (int ka = 0; ka < K_ATOMS; ++ka) {
            uint32_t A_regs[M_ATOMS][4];
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                int row = mi * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (row & SWIZZLE_MASK);
                ldmatrix_x4_b16(A_regs[mi][0], A_regs[mi][1], A_regs[mi][2], A_regs[mi][3],
                                to_smem(&A_stage[row * BLOCK_K + csw * 16]));
            }
            uint32_t B_regs[N_PAIRS_PW][4];
            #pragma unroll
            for (int np = 0; np < N_PAIRS_PW; ++np) {
                int nrow = warp_id * N_ATOMS_PW * 8 + np * 16 + row_block * 8 + row_in_frag;
                int chunk = 2 * ka + col_block;
                int csw = chunk ^ (nrow & SWIZZLE_MASK);
                ldmatrix_x4_b16(B_regs[np][0], B_regs[np][1], B_regs[np][2], B_regs[np][3],
                                to_smem(&B_smem[nrow * BLOCK_K + csw * 16]));
            }
            #pragma unroll
            for (int mi = 0; mi < M_ATOMS; ++mi) {
                #pragma unroll
                for (int np = 0; np < N_PAIRS_PW; ++np) {
                    int ni0 = np * 2, ni1 = np * 2 + 1;
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni0][0], tacc[mi][ni0][1], tacc[mi][ni0][2], tacc[mi][ni0][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][0], B_regs[np][1]);
                    mma_m16n8k32_e4m3(
                        tacc[mi][ni1][0], tacc[mi][ni1][1], tacc[mi][ni1][2], tacc[mi][ni1][3],
                        A_regs[mi][0], A_regs[mi][2], A_regs[mi][1], A_regs[mi][3],
                        B_regs[np][2], B_regs[np][3]);
                }
            }
        }

        int kbt = kb % SCALE_KTILE;
        float ws_cta = ws_smem[kbt];
        #pragma unroll
        for (int mi = 0; mi < M_ATOMS; ++mi) {
            float as0 = as_smem[(mi * 16 + h) * SCALE_KTILE + kbt];
            float as1 = as_smem[(mi * 16 + h + 8) * SCALE_KTILE + kbt];
            #pragma unroll
            for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
                acc[mi][ni][0] += tacc[mi][ni][0] * (as0 * ws_cta);
                acc[mi][ni][1] += tacc[mi][ni][1] * (as0 * ws_cta);
                acc[mi][ni][2] += tacc[mi][ni][2] * (as1 * ws_cta);
                acc[mi][ni][3] += tacc[mi][ni][3] * (as1 * ws_cta);
            }
        }
        __syncthreads();
        compute_stage = (compute_stage + 1) % STAGES;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    #pragma unroll
    for (int mi = 0; mi < M_ATOMS; ++mi) {
        int row0 = m_base + mi * 16 + h;
        int row1 = row0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_ATOMS_PW; ++ni) {
            int n_pair_base = n_base + warp_id * N_ATOMS_PW * 8 + ni * 8 + 2 * l;
            float* p0 = Dpart + (size_t)split * M * N + (size_t)row0 * N + n_pair_base;
            if (EXACT || (row0 < M && col_pair_ok(n_pair_base, N))) {
                p0[0] = acc[mi][ni][0];
                p0[1] = acc[mi][ni][1];
            } else if (!EXACT && row0 < M) {
                if (n_pair_base < N) p0[0] = acc[mi][ni][0];
                if (n_pair_base + 1 < N) p0[1] = acc[mi][ni][1];
            }
            float* p1 = p0 + 8 * (size_t)N;
            if (EXACT || (row1 < M && col_pair_ok(n_pair_base, N))) {
                p1[0] = acc[mi][ni][2];
                p1[1] = acc[mi][ni][3];
            } else if (!EXACT && row1 < M) {
                if (n_pair_base < N) p1[0] = acc[mi][ni][2];
                if (n_pair_base + 1 < N) p1[1] = acc[mi][ni][3];
            }
        }
    }
}

// Sum the per-split fp32 partials and round once to bf16, matching the
// unsplit kernel's single conversion of its fp32 accumulator.
__global__ void fp4_splitk_reduce_kernel(const float* __restrict__ part,
                                         __nv_bfloat16* __restrict__ D,
                                         int M, int N, int splits,
                                         __nv_bfloat16* __restrict__ residual,
                                         const __nv_bfloat16* __restrict__ bias) {
    const long total = (long)M * N;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int sp = 0; sp < splits; ++sp) s += part[(size_t)sp * total + i];
        if (residual != nullptr) {
            // Exact fold of the delivered chain: reduce rounds once to bf16,
            // gated_residual adds the bias and rounds, then adds the residual
            // and rounds. Same operands, same order, same three roundings.
            const int j = (int)(i % N);
            const float p = __bfloat162float(__float2bfloat16_rn(s));
            const float v = __bfloat162float(__float2bfloat16_rn(
                __fadd_rn(p, __bfloat162float(bias[j]))));
            residual[i] = __float2bfloat16_rn(
                __bfloat162float(residual[i]) + v);
        } else {
            D[i] = __float2bfloat16_rn(s);
        }
    }
}

}  // namespace block128_sm89_splitk
}  // namespace gemm
}  // namespace flash_rt
