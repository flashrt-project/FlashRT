// SPDX-License-Identifier: Apache-2.0
//
// FP8 (block-128) -> NVFP4 (swizzled SF + per-tensor global_scale) — see
// fp8_block128_to_nvfp4_swizzled.cuh for design notes.

#include "fp8_block128_to_nvfp4_swizzled.cuh"

#include <cuda_fp8.h>

namespace flash_rt {
namespace quantize {

namespace {

// ---------------- helpers (mirrors of csrc/kernels/quantize.cu) ----------------
// Reproduced as static inlines so we don't link to internal helpers across
// translation units. Behavior is bit-identical to the originals.

__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;
    else if (a < 0.75f)  mag = 1;
    else if (a < 1.25f)  mag = 2;
    else if (a < 1.75f)  mag = 3;
    else if (a < 2.5f)   mag = 4;
    else if (a < 3.5f)   mag = 5;
    else if (a < 5.0f)   mag = 6;
    else                 mag = 7;
    return sign | mag;
}

__device__ __forceinline__ uint8_t fp32_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;
    uint32_t bits = __float_as_uint(v);
    int float_exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;
    int ue_exp = float_exp + 7;
    if (ue_exp <= 0) {
        float scaled = v * 512.0f;
        int m = (int)ceilf(scaled);
        if (m > 7) return (1 << 3) | 0;
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;
    int m = (int)(frac >> 20);
    if (frac & 0xFFFFF) m++;
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;
    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_fp32(uint8_t v) {
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

__device__ __forceinline__ float fp8_e4m3_byte_to_fp32(uint8_t b) {
    __nv_fp8_e4m3 v = *reinterpret_cast<const __nv_fp8_e4m3*>(&b);
    return float(v);
}

// ---------------- kernels ----------------

// Pass 1: per-row reduction of |w_fp32| via atomicMax to one device scalar.
// One block per row; threads stride across K. blockDim chosen at launch.
__global__ void weight_global_amax_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ block_scales,
    float* __restrict__ global_amax,
    int N, int K) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int blk_row_idx = row / 128;
    const int sblock_row_stride = K / 128;
    const size_t row_off = (size_t)row * K;

    float thread_max = 0.f;
    for (int col = threadIdx.x; col < K; col += blockDim.x) {
        const uint8_t b8 = w_fp8[row_off + col];
        const int sblock_col_idx = col / 128;
        const float bs = block_scales[blk_row_idx * sblock_row_stride
                                      + sblock_col_idx];
        const float w = fp8_e4m3_byte_to_fp32(b8) * bs;
        const float aw = fabsf(w);
        if (aw > thread_max) thread_max = aw;
    }

    // Block-wide warp-shuffle reduction.
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(
            thread_max,
            __shfl_xor_sync(0xffffffff, thread_max, off));
    }
    if (lane == 0) smem[wid] = thread_max;
    __syncthreads();
    if (wid == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        thread_max = (lane < n_warps) ? smem[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            thread_max = fmaxf(
                thread_max,
                __shfl_xor_sync(0xffffffff, thread_max, off));
        }
        if (lane == 0) {
            atomicMax(reinterpret_cast<int*>(global_amax),
                      __float_as_int(thread_max));
        }
    }
}

// Tiny finalizer: out_global_scale = global_amax / (FP4_MAX * SF_MAX) =
// global_amax / 2688. Falls back to 1.0 if amax is zero (avoid div-by-zero).
__global__ void compute_global_scale_kernel(
    const float* global_amax,
    float* out_global_scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float a = *global_amax;
        *out_global_scale = (a > 0.f) ? (a / 2688.f) : 1.f;
    }
}

// Pass 2: per-NVFP4-block (16 elements) quantize. One block per row; threads
// loop across blocks within the row.
__global__ void weight_quantize_pass2_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ block_scales,
    const float* __restrict__ global_scale_ptr,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sf_swz,
    int N, int K,
    int num_blocks_per_row, int n_col_super) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const float gscale = *global_scale_ptr;
    const float inv_g = (gscale > 0.f) ? (1.f / gscale) : 0.f;

    const int blk_row_idx = row / 128;
    const int sblock_row_stride = K / 128;
    const size_t row_in_off = (size_t)row * K;
    const size_t row_out_off = (size_t)row * (K / 2);

    const int rb = row / 128;
    const int ri = row % 128;

    for (int b = threadIdx.x; b < num_blocks_per_row; b += blockDim.x) {
        const int col0 = b * 16;
        // 16 < 128 and col0 % 128 + 16 <= 128 by construction (b*16 mod 128
        // is always a multiple of 16 in [0, 112]), so all 16 elements share
        // the same FP8 block scale.
        const int sblock_col_idx = col0 / 128;
        const float bs = block_scales[blk_row_idx * sblock_row_stride
                                      + sblock_col_idx];

        // Dequant 16 fp8 -> fp32, track block amax.
        float w_f32[16];
        float block_amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t b8 = w_fp8[row_in_off + col0 + i];
            const float w = fp8_e4m3_byte_to_fp32(b8) * bs;
            w_f32[i] = w;
            const float aw = fabsf(w);
            if (aw > block_amax) block_amax = aw;
        }

        // Per-block scale in absolute units = block_amax / 6. Store as UE4M3
        // SF byte = scale_abs / global_scale (so byte * global_scale = scale_abs).
        const float ue4m3_value = (block_amax / 6.f) * inv_g;
        const uint8_t sf_byte = fp32_to_ue4m3_ceil(ue4m3_value);

        // Decoded-back per-block scale in absolute units (for inv multiply).
        const float sf_decoded = ue4m3_to_fp32(sf_byte);
        const float block_scale_abs = sf_decoded * gscale;
        const float inv_block_scale = (block_scale_abs > 0.f)
            ? (1.f / block_scale_abs) : 0.f;

        // Quantize 16 -> packed e2m1.
        uint8_t* packed_row = packed + row_out_off;
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const float v0 = w_f32[i] * inv_block_scale;
            const float v1 = w_f32[i + 1] * inv_block_scale;
            const uint8_t lo = fp32_to_e2m1(v0);
            const uint8_t hi = fp32_to_e2m1(v1);
            packed_row[(col0 + i) >> 1] = (hi << 4) | (lo & 0x0F);
        }

        // Swizzled SF write — same permutation as
        // csrc/kernels/quantize.cu :: quantize_bf16_to_nvfp4_swizzled_kernel
        // (lines 408-422), i.e. matches the SM120 GEMM expectation.
        const int cb = b / 4;
        const int ci = b % 4;
        const int out_idx = (rb * n_col_super + cb) * 512
                          + (ri % 32) * 16
                          + (ri / 32) * 4
                          + ci;
        sf_swz[out_idx] = sf_byte;
    }
}

#ifdef FLASHRT_ENABLE_LINGBOT_VA_SM89
// Pass 2, per-128 variant: one thread per 128-wide K block (one FP8 block).
// The scale is the block amax/6, written as the same UE4M3 byte to all eight
// 16-wide sub-block SF slots, so the existing expansion reconstructs a weight
// quantized with the per-128 scale. Two reads of the row, no register array.
__global__ void weight_quantize_pass2_sf128_kernel(
    const uint8_t* __restrict__ w_fp8,
    const float* __restrict__ block_scales,
    const float* __restrict__ global_scale_ptr,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ sf_swz,
    int N, int K,
    int n_col_super,
    int plain) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const float gscale = *global_scale_ptr;
    const float inv_g = (gscale > 0.f) ? (1.f / gscale) : 0.f;
    const int blk_row_idx = row / 128;
    const int sblock_row_stride = K / 128;
    const size_t row_in_off = (size_t)row * K;
    const size_t row_out_off = (size_t)row * (K / 2);
    const int rb = row / 128;
    const int ri = row % 128;
    const int num_kb = K / 128;

    for (int kb = threadIdx.x; kb < num_kb; kb += blockDim.x) {
        const int col0 = kb * 128;
        const float bs = block_scales[blk_row_idx * sblock_row_stride + kb];

        float amax = 0.f;
        #pragma unroll 8
        for (int i = 0; i < 128; ++i) {
            const uint8_t b8 = w_fp8[row_in_off + col0 + i];
            const float w = fp8_e4m3_byte_to_fp32(b8) * bs;
            const float aw = fabsf(w);
            if (aw > amax) amax = aw;
        }

        const uint8_t sf_byte = fp32_to_ue4m3_ceil((amax / 6.f) * inv_g);
        const float sf_decoded = ue4m3_to_fp32(sf_byte) * gscale;
        const float inv_block_scale = (sf_decoded > 0.f) ? (1.f / sf_decoded) : 0.f;

        uint8_t* packed_row = packed + row_out_off;
        #pragma unroll 8
        for (int i = 0; i < 128; i += 2) {
            const float v0 = fp8_e4m3_byte_to_fp32(w_fp8[row_in_off + col0 + i]) * bs
                           * inv_block_scale;
            const float v1 = fp8_e4m3_byte_to_fp32(w_fp8[row_in_off + col0 + i + 1]) * bs
                           * inv_block_scale;
            const uint8_t lo = fp32_to_e2m1(v0);
            const uint8_t hi = fp32_to_e2m1(v1);
            packed_row[(col0 + i) >> 1] = (hi << 4) | (lo & 0x0F);
        }

        if (plain) {
            // [N/128, K/128, 128]: one byte per (row, 128-K block), 128
            // contiguous bytes per K-block, 8x fewer bytes in the image.
            sf_swz[(((size_t)(row >> 7) * (K >> 7) + kb) << 7) + (row & 127)] =
                sf_byte;
        } else {
            // The eight sub-block slots of this 128 block all carry the same
            // byte (precision-test layout; the delivered kernel reads the
            // per-16 image).
            #pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
                const int b = kb * 8 + sb;
                const int cb = b / 4;
                const int ci = b % 4;
                const int out_idx = (rb * n_col_super + cb) * 512
                                  + (ri % 32) * 16
                                  + (ri / 32) * 4
                                  + ci;
                sf_swz[out_idx] = sf_byte;
            }
        }
    }
}

#endif  // FLASHRT_ENABLE_LINGBOT_VA_SM89
}  // namespace

void fp8_block128_to_nvfp4_swizzled_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream) {
    // 1. Reset scratch global amax.
    cudaMemsetAsync(scratch_global_amax, 0, sizeof(float), stream);

    // 2. Pass 1: row-block reduction → atomicMax.
    {
        dim3 grid(N);
        dim3 block(256);
        weight_global_amax_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            scratch_global_amax,
            N, K);
    }

    // 3. Compute global_scale = amax / 2688.
    compute_global_scale_kernel<<<1, 1, 0, stream>>>(
        scratch_global_amax, out_global_scale);

    // 4. Pass 2: per-NVFP4-block quantize + swizzled SF write.
    const int num_blocks_per_row = K / 16;
    const int n_col_super = (num_blocks_per_row + 3) / 4;
    {
        dim3 grid(N);
        dim3 block(256);
        weight_quantize_pass2_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            out_global_scale,
            nvfp4_packed,
            nvfp4_sf_swizzled,
            N, K, num_blocks_per_row, n_col_super);
    }
}

#ifdef FLASHRT_ENABLE_LINGBOT_VA_SM89
void fp8_block128_to_nvfp4_sf128_swizzled_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream) {
    // Same global-amax pass and global scale as the per-16 variant; only the
    // micro-block scale granularity changes (128 instead of 16).
    cudaMemsetAsync(scratch_global_amax, 0, sizeof(float), stream);
    {
        dim3 grid(N);
        dim3 block(256);
        weight_global_amax_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            scratch_global_amax,
            N, K);
    }
    compute_global_scale_kernel<<<1, 1, 0, stream>>>(
        scratch_global_amax, out_global_scale);

    const int n_col_super = (((K / 16) + 3) >> 2);
    {
        dim3 grid(N);
        dim3 block(64);
        weight_quantize_pass2_sf128_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            out_global_scale,
            nvfp4_packed,
            nvfp4_sf_swizzled,
            N, K, n_col_super, 0);
    }
}

void fp8_block128_to_nvfp4_sf128_plain_bf16(
    const void* w_fp8,
    const float* w_block_scale_fp32,
    uint8_t* nvfp4_packed,
    uint8_t* sf128_plain,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream) {
    cudaMemsetAsync(scratch_global_amax, 0, sizeof(float), stream);
    {
        dim3 grid(N);
        dim3 block(256);
        weight_global_amax_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            scratch_global_amax,
            N, K);
    }
    compute_global_scale_kernel<<<1, 1, 0, stream>>>(
        scratch_global_amax, out_global_scale);
    const int n_col_super = (((K / 16) + 3) >> 2);
    {
        dim3 grid(N);
        dim3 block(64);
        weight_quantize_pass2_sf128_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(w_fp8),
            w_block_scale_fp32,
            out_global_scale,
            nvfp4_packed,
            sf128_plain,
            N, K, n_col_super, 1);
    }
}

#endif  // FLASHRT_ENABLE_LINGBOT_VA_SM89
}  // namespace quantize
}  // namespace flash_rt
