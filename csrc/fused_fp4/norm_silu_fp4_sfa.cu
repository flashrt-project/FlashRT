// ============================================================================
//  Fused FP4 pre-GEMM kernels (F2/F3/F4).
//
//  Numerical correctness spec: bit-exact with the corresponding sequential
//  2-kernel path, modulo fp16 intermediate rounding that both paths share:
//
//      F2 ≡ rms_norm_noweight_fp16 → quantize_fp4_dynamic_sfa_fp16
//      F3 ≡ residual_add_rms_norm_noweight_fp16 → quantize_fp4_dynamic_sfa_fp16
//      F4 ≡ gate_silu_mul_merged_fp16 → quantize_fp4_dynamic_sfa_fp16
//
//  Shared-memory staging layout: normalized (or silu-muled) fp16 values are
//  written to shared memory in row-major order, then per-16-element NVFP4
//  block workers read from shared mem, compute amax, UE4M3-quant the block
//  scale, e2m1-pack elements, and write scale byte to CUTLASS SFA layout.
// ============================================================================
#include "fused_fp4/norm_silu_fp4_sfa.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <string>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fused_fp4 {

#if FV_HAVE_CUTLASS

using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

inline void check_fp4_kernel_launch(const char* kernel_name) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(kernel_name) + " launch failed: "
            + cudaGetErrorString(error));
    }
}

// ── Device helpers (local copies to keep this TU standalone) ──
__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3(float x) {
    return __nv_fp8_e4m3(fmaxf(x, 0.f));
}

// Stage 4 (shared): read fp16 from shared mem, amax → UE4M3 scale → e2m1 pack
// + SFA write. Each thread processes 1 NVFP4 block (16 consecutive fp16).
template <class LayoutSF>
__device__ __forceinline__ void quant_block_from_smem(
    const __half* smem_normed,
    uint8_t* packed_row,
    uint8_t* dst_sfa,
    LayoutSF layout,
    int row, int block_idx) {
    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __half2float(smem_normed[block_idx * 16 + i]);
        float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = quantize_ue4m3(desired);
    float bs_dq = static_cast<float>(bs_q);
    int sfa_off = layout(row, block_idx * 16, 0);
    dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1(vals[2 * p    ] * inv_bs);
        uint8_t hi = fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
        packed_row[block_idx * 8 + p] = lo | (hi << 4);
    }
}

// Register-only Pi0.5 kernels keep the original 256-thread reduction order,
// then quantize four strided 256-element segments with 16-lane warp groups.
// This removes the shared-memory round trip without changing fp16 rounding.
template <bool UseNativeFp4, class LayoutSF>
__device__ __forceinline__ void pi05_quantize_register_value(
    float value,
    uint8_t* packed_row,
    uint8_t* dst_sfa,
    LayoutSF layout,
    int row,
    int block_idx,
    int lane_in_block) {
    float amax = fabsf(value);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffff, amax, offset, 16));
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = quantize_ue4m3(desired);
    const float inv_bs = 1.f / static_cast<float>(bs_q);
    if (lane_in_block == 0) {
        const int sfa_off = layout(row, block_idx * 16, 0);
        dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);
    }
    if constexpr (UseNativeFp4) {
        const float next = __shfl_down_sync(0xffffffff, value, 1, 16);
        if ((lane_in_block & 1) == 0) {
            packed_row[block_idx * 8 + lane_in_block / 2] =
                static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
                    make_float2(value * inv_bs, next * inv_bs),
                    __NV_E2M1, cudaRoundNearest));
        }
    } else {
        const uint8_t code = fp32_to_e2m1(value * inv_bs);
        const uint8_t next = static_cast<uint8_t>(__shfl_down_sync(
            0xffffffff, static_cast<unsigned>(code), 1, 16));
        if ((lane_in_block & 1) == 0) {
            packed_row[block_idx * 8 + lane_in_block / 2] =
                code | (next << 4);
        }
    }
}

template <bool UseNativeFp4, class LayoutSF>
__global__ void pi05_adarms_fp4_sfa_register_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ style,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    __half* __restrict__ gate,
    LayoutSF layout,
    int D) {
    const int row_idx = blockIdx.x;
    const __half* row = x + row_idx * D;
    const __half* sc = style + row_idx * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    uint8_t* packed_row = packed + row_idx * (D / 2);

    float values[4];
    float sum_sq = 0.f;
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const int i = threadIdx.x + segment * blockDim.x;
        const float value = __half2float(row[i]);
        values[segment] = value;
        sum_sq += value * value;
    }

    __shared__ float reduction[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) reduction[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = lane < 8 ? reduction[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) reduction[0] = sum_sq;
    __syncthreads();

    const float rstd = rsqrtf(reduction[0] / D + 1e-6f);
    const int lane_in_block = threadIdx.x & 15;
    const int block_group = threadIdx.x >> 4;
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const int i = threadIdx.x + segment * blockDim.x;
        const float normed = values[segment] * rstd *
            (1.f + __half2float(sc[i])) + __half2float(sh[i]);
        const __half rounded = __float2half(normed);
        gate[row_idx * D + i] = gt[i];
        pi05_quantize_register_value<UseNativeFp4>(
            __half2float(rounded), packed_row, dst_sfa, layout, row_idx,
            segment * 16 + block_group, lane_in_block);
    }
}

template <bool UseNativeFp4, class LayoutSF>
__global__ void pi05_gate_res_adarms_fp4_sfa_register_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ prev_gate,
    __half* __restrict__ residual,
    const __half* __restrict__ style,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    __half* __restrict__ gate,
    LayoutSF layout,
    int D) {
    const int row_idx = blockIdx.x;
    const __half* sc = style + row_idx * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    uint8_t* packed_row = packed + row_idx * (D / 2);

    float values[4];
    float sum_sq = 0.f;
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const int i = threadIdx.x + segment * blockDim.x;
        const int elem = row_idx * D + i;
        const float value = __half2float(residual[elem]) +
            __half2float(x[elem]) * __half2float(prev_gate[elem]);
        const __half rounded = __float2half(value);
        residual[elem] = rounded;
        values[segment] = __half2float(rounded);
        sum_sq += value * value;
    }

    __shared__ float reduction[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (lane == 0) reduction[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = lane < 8 ? reduction[lane] : 0.f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) reduction[0] = sum_sq;
    __syncthreads();

    const float rstd = rsqrtf(reduction[0] / D + 1e-6f);
    const int lane_in_block = threadIdx.x & 15;
    const int block_group = threadIdx.x >> 4;
    #pragma unroll
    for (int segment = 0; segment < 4; ++segment) {
        const int i = threadIdx.x + segment * blockDim.x;
        const int elem = row_idx * D + i;
        const float normed = values[segment] * rstd *
            (1.f + __half2float(sc[i])) + __half2float(sh[i]);
        const __half rounded = __float2half(normed);
        gate[elem] = gt[i];
        pi05_quantize_register_value<UseNativeFp4>(
            __half2float(rounded), packed_row, dst_sfa, layout, row_idx,
            segment * 16 + block_group, lane_in_block);
    }
}

// ──────────────────────────────────────────────────────────────────
// F2: rms_norm(x) → fp4 + SFA
// Grid: (S,)  Block: 256  Shared: D * sizeof(__half)
// Layout matches existing rms_norm_noweight_fp16: strided coalesced load,
// per-thread cache 8 floats; after reduction we write normalized fp16 to
// shared memory in row-major order, then block-workers do stage-4.
// ──────────────────────────────────────────────────────────────────
template <class LayoutSF>
__global__ void f2_rms_norm_fp4_sfa_kernel(
    const __half* __restrict__ x,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D) {
    const int r = blockIdx.x;
    const __half* row_ptr = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);

    const __half2* row2 = reinterpret_cast<const __half2*>(row_ptr);
    const int D2 = D / 2;

    extern __shared__ __half smem_normed[];

    constexpr int ELEMS_PER_THREAD = 8;
    float cache[ELEMS_PER_THREAD];
    float ssq = 0;
    int c2_base[ELEMS_PER_THREAD / 2];

    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        c2_base[it] = c2;
        if (c2 < D2) {
            __half2 v2 = row2[c2];
            cache[it*2]   = __half2float(v2.x);
            cache[it*2+1] = __half2float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float rms = __frsqrt_rn(sh[0] / D + 1e-6f);

    // Write normalized fp16 values to shared mem (row-major in contiguous K).
    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = c2_base[it];
        if (c2 < D2) {
            float v0 = cache[it*2]   * rms;
            float v1 = cache[it*2+1] * rms;
            __half2 h2 = __halves2half2(__float2half(v0), __float2half(v1));
            reinterpret_cast<__half2*>(smem_normed)[c2] = h2;
        }
    }
    __syncthreads();

    // Stage 4: per-16-block workers.
    const int n_blocks = D / 16;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        quant_block_from_smem(smem_normed, packed_row, dst_sfa, layout, r, b);
    }
}

// ──────────────────────────────────────────────────────────────────
// F2 + AWQ: rms_norm(x) * inv_s → fp4 + SFA (per-channel inverse scale)
// Same structure as F2; the normalized value is multiplied by inv_s[c]
// before the fp16 staging round.
// ──────────────────────────────────────────────────────────────────
template <class LayoutSF>
__global__ void f2_rms_norm_mul_fp4_sfa_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ inv_s,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D) {
    const int r = blockIdx.x;
    const __half* row_ptr = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);

    const __half2* row2 = reinterpret_cast<const __half2*>(row_ptr);
    const __half2* inv2 = reinterpret_cast<const __half2*>(inv_s);
    const int D2 = D / 2;

    extern __shared__ __half smem_normed[];

    constexpr int ELEMS_PER_THREAD = 8;
    float cache[ELEMS_PER_THREAD];
    float ssq = 0;
    int c2_base[ELEMS_PER_THREAD / 2];

    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        c2_base[it] = c2;
        if (c2 < D2) {
            __half2 v2 = row2[c2];
            cache[it*2]   = __half2float(v2.x);
            cache[it*2+1] = __half2float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float rms = __frsqrt_rn(sh[0] / D + 1e-6f);

    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = c2_base[it];
        if (c2 < D2) {
            const float2 iv = __half22float2(inv2[c2]);
            float v0 = cache[it*2]   * rms * iv.x;
            float v1 = cache[it*2+1] * rms * iv.y;
            __half2 h2 = __halves2half2(__float2half(v0), __float2half(v1));
            reinterpret_cast<__half2*>(smem_normed)[c2] = h2;
        }
    }
    __syncthreads();

    const int n_blocks = D / 16;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        quant_block_from_smem(smem_normed, packed_row, dst_sfa, layout, r, b);
    }
}

// ──────────────────────────────────────────────────────────────────
// F3: residual += x; rms_norm(residual) → fp4 + SFA
// ──────────────────────────────────────────────────────────────────
template <class LayoutSF>
__global__ void f3_res_rms_fp4_sfa_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int D) {
    const int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    uint8_t* packed_row = packed + r * (D / 2);
    const int D2 = D / 2;

    __half2* res2w = reinterpret_cast<__half2*>(res_row);
    const __half2* res2 = reinterpret_cast<const __half2*>(res_row);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row);

    extern __shared__ __half smem_normed[];

    constexpr int ELEMS_PER_THREAD = 8;
    float cache[ELEMS_PER_THREAD];
    float ssq = 0;
    int c2_base[ELEMS_PER_THREAD / 2];

    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        c2_base[it] = c2;
        if (c2 < D2) {
            __half2 rv2 = res2[c2], xv2 = x2[c2];
            float a = __half2float(rv2.x) + __half2float(xv2.x);
            float b = __half2float(rv2.y) + __half2float(xv2.y);
            cache[it*2]   = a;
            cache[it*2+1] = b;
            // Residual update in place (match reference kernel).
            res2w[c2] = __halves2half2(__float2half(a), __float2half(b));
            ssq += a*a + b*b;
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float rms = __frsqrt_rn(sh[0] / D + 1e-6f);

    #pragma unroll
    for (int it = 0; it < ELEMS_PER_THREAD / 2; it++) {
        int c2 = c2_base[it];
        if (c2 < D2) {
            float v0 = cache[it*2]   * rms;
            float v1 = cache[it*2+1] * rms;
            __half2 h2 = __halves2half2(__float2half(v0), __float2half(v1));
            reinterpret_cast<__half2*>(smem_normed)[c2] = h2;
        }
    }
    __syncthreads();

    const int n_blocks = D / 16;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        quant_block_from_smem(smem_normed, packed_row, dst_sfa, layout, r, b);
    }
}

// ──────────────────────────────────────────────────────────────────
// F4: gate_silu_mul(merged[S,2H]) → hid[S,H] → fp4 + SFA
// Grid: (S,)  Block: 256  Shared: H * sizeof(__half)
// gate = merged[:, :H], up = merged[:, H:], out[i] = GELU(gate[i]) * up[i].
// GELU approximation matches csrc/kernels/activation.cu (tanh-approx constant
// 1.5957691216057308f = sqrt(2/pi) * 1 — actually the factor here is for the
// approximation used in Pi0.5 / Gemma).
// ──────────────────────────────────────────────────────────────────
template <class LayoutSF>
__global__ void f4_silu_mul_fp4_sfa_kernel(
    const __half* __restrict__ merged,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    const int r = blockIdx.x;
    const __half* merged_row = merged + r * 2 * H;
    uint8_t* packed_row = packed + r * (H / 2);

    extern __shared__ __half smem_hid[];

    // Stage 1: compute silu_mul per element (same formula as
    // gate_silu_mul_merged_kernel<__half>).
    for (int col = threadIdx.x; col < H; col += blockDim.x) {
        float g = __half2float(merged_row[col]);
        float u = __half2float(merged_row[H + col]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        smem_hid[col] = __float2half(gelu * u);
    }
    __syncthreads();

    // Stage 2: per-16-block workers.
    const int n_blocks = H / 16;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        quant_block_from_smem(smem_hid, packed_row, dst_sfa, layout, r, b);
    }
}

#endif  // FV_HAVE_CUTLASS

// ── Host entry points ──
void rms_norm_fp4_sfa_fp16(
    const __half* x, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    const size_t smem_bytes = dim * sizeof(__half);
    f2_rms_norm_fp4_sfa_kernel<<<seq_len, 256, smem_bytes, stream>>>(
        x, packed, sfa, layout, dim);
#else
    (void)x; (void)packed; (void)sfa; (void)seq_len; (void)dim; (void)stream;
#endif
}

void rms_norm_mul_fp4_sfa_fp16(
    const __half* x, const __half* inv_s, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    const size_t smem_bytes = dim * sizeof(__half);
    f2_rms_norm_mul_fp4_sfa_kernel<<<seq_len, 256, smem_bytes, stream>>>(
        x, inv_s, packed, sfa, layout, dim);
#else
    (void)x; (void)inv_s; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void residual_add_rms_norm_fp4_sfa_fp16(
    __half* residual, const __half* x,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    const size_t smem_bytes = dim * sizeof(__half);
    f3_res_rms_fp4_sfa_kernel<<<seq_len, 256, smem_bytes, stream>>>(
        residual, x, packed, sfa, layout, dim);
#else
    (void)residual; (void)x; (void)packed; (void)sfa;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void pi05_adarms_fp4_sfa_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    pi05_adarms_fp4_sfa_register_kernel<false><<<seq_len, 256, 0, stream>>>(
        x, style, packed, sfa, gate, layout, dim);
    check_fp4_kernel_launch("pi05_adarms_fp4_sfa_fp16");
#else
    (void)x; (void)style; (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void pi05_gate_res_adarms_fp4_sfa_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style, uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    pi05_gate_res_adarms_fp4_sfa_register_kernel<false><<<seq_len, 256, 0, stream>>>(
            x, prev_gate, residual, style, packed, sfa, gate, layout, dim);
    check_fp4_kernel_launch("pi05_gate_res_adarms_fp4_sfa_fp16");
#else
    (void)x; (void)prev_gate; (void)residual; (void)style;
    (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void pi05_adarms_fp4_sfa_native_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    pi05_adarms_fp4_sfa_register_kernel<true><<<seq_len, 256, 0, stream>>>(
        x, style, packed, sfa, gate, layout, dim);
    check_fp4_kernel_launch("pi05_adarms_fp4_sfa_native_fp16");
#else
    (void)x; (void)style; (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void pi05_gate_res_adarms_fp4_sfa_native_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style, uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    pi05_gate_res_adarms_fp4_sfa_register_kernel<true><<<
        seq_len, 256, 0, stream>>>(
            x, prev_gate, residual, style, packed, sfa, gate, layout, dim);
    check_fp4_kernel_launch("pi05_gate_res_adarms_fp4_sfa_native_fp16");
#else
    (void)x; (void)prev_gate; (void)residual; (void)style;
    (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)stream;
#endif
}

void gate_silu_mul_fp4_sfa_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    // For F4 the "K" dimension of the FP4 GEMM that consumes this is half_dim.
    auto shape = cute::make_shape(seq_len, 1, half_dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    const size_t smem_bytes = half_dim * sizeof(__half);
    f4_silu_mul_fp4_sfa_kernel<<<seq_len, 256, smem_bytes, stream>>>(
        merged, packed, sfa, layout, half_dim);
#else
    (void)merged; (void)packed; (void)sfa;
    (void)seq_len; (void)half_dim; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
