// ============================================================================
//  FlashRT — silu_mul_two_fp4_to_fp4: P1 split-GU FFN combiner.
//
//  Reads two FP4 inputs (gate, up — each packed [S, H/2] + SFA) produced
//  by separate NVFP4 GEMMs, dequantizes each, computes silu(gate) * up in
//  fp32 registers, requantizes to FP4 + SFA for the next-stage Down GEMM.
//
//  Replaces the merged-GU + F4 v2+mul pair in the AWQ baseline with a
//  third leg that reads HALF the activation DRAM (8.6 MB combined fp4 vs
//  31.6 MB fp16 today).
//
//  Layout:
//    gate_packed[s, b*8+p] = byte holding gate elements (s, b*16 + 2p),
//                            (s, b*16 + 2p + 1)
//    gate_SFA[layout(s, b*16, 0)] = UE4M3 fp8 scale for that 16-block
//    Same for up. Output packed[s, b*8+p] + SFA same shape.
//
//  Design: 1 thread = 1 NVFP4 block (16 elements). Mirrors F4 v2 v2_kernel
//  (silu_mul_fp4_sfa_v2.cu) but reads two fp4 inputs instead of one
//  fp16 [S, 2H] merged buffer.
//
//  Additive: does NOT modify existing kernels.
// ============================================================================
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mutex>
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

using CfgF4P1 = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ float e2m1_to_fp32_p1(uint8_t v) {
    // magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6} indexed by v & 0x7
    static constexpr float mags[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mags[v & 0x7];
    return (v & 0x8) ? -m : m;
}

__device__ __forceinline__ uint8_t fp32_to_e2m1_p1(float x) {
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

__device__ __forceinline__ float silu_mul_p1(float g, float u) {
    // Same formula as csrc/kernels/activation.cu and silu_mul_fp4_sfa_v2.cu.
    float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
    return gelu * u;
}

__device__ float geglu_gate_lut_p1[256 * 16];

std::once_flag g_geglu_gate_lut_once;

inline void check_fp4_kernel_launch(const char* kernel_name) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(kernel_name) + " launch failed: "
            + cudaGetErrorString(error));
    }
}

__global__ void init_geglu_gate_lut_p1_kernel() {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 256 * 16) return;
    const uint8_t scale_byte = static_cast<uint8_t>(index >> 4);
    const uint8_t fp4_code = static_cast<uint8_t>(index & 0xF);
    __nv_fp8_e4m3 scale_q;
    *reinterpret_cast<uint8_t*>(&scale_q) = scale_byte;
    const float g = e2m1_to_fp32_p1(fp4_code) * static_cast<float>(scale_q);
    geglu_gate_lut_p1[index] =
        g / (1.0f + expf(
            -1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

inline void ensure_geglu_gate_lut(cudaStream_t stream) {
    std::call_once(g_geglu_gate_lut_once, [stream] {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cudaError_t error = cudaStreamIsCapturing(stream, &capture_status);
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamIsCapturing failed: ")
                + cudaGetErrorString(error));
        }
        if (capture_status != cudaStreamCaptureStatusNone) {
            throw std::runtime_error(
                "geglu gate LUT must be initialized before CUDA graph capture");
        }

        init_geglu_gate_lut_p1_kernel<<<16, 256, 0, stream>>>();
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("init_geglu_gate_lut_p1_kernel launch failed: ")
                + cudaGetErrorString(error));
        }
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("init_geglu_gate_lut_p1_kernel execution failed: ")
                + cudaGetErrorString(error));
        }
    });
}

template <class LayoutSF>
__global__ void silu_mul_two_fp4_to_fp4_kernel(
    const uint8_t* __restrict__ gate_packed,   // [S, H/2]
    const uint8_t* __restrict__ gate_sfa,
    const uint8_t* __restrict__ up_packed,     // [S, H/2]
    const uint8_t* __restrict__ up_sfa,
    uint8_t* __restrict__ out_packed,          // [S, H/2]
    uint8_t* __restrict__ out_sfa,
    LayoutSF layout_in,                        // SFA layout for inputs (S, H)
    LayoutSF layout_out,                       // SFA layout for output  (S, H)
    int H) {
    // 1 thread = 1 NVFP4 block (16 elements).
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;

    // Read SFA scales (UE4M3 = positive fp8_e4m3 bit pattern).
    int sfa_off = layout_in(row, col_base, 0);
    uint8_t gate_sf_byte = gate_sfa[sfa_off];
    uint8_t up_sf_byte   = up_sfa[sfa_off];
    // Reinterpret as fp8 e4m3 → float.
    __nv_fp8_e4m3 gate_bs_q, up_bs_q;
    *reinterpret_cast<uint8_t*>(&gate_bs_q) = gate_sf_byte;
    *reinterpret_cast<uint8_t*>(&up_bs_q)   = up_sf_byte;
    float gate_scale = static_cast<float>(gate_bs_q);
    float up_scale   = static_cast<float>(up_bs_q);

    // Read 8 packed bytes per input (16 elements each).
    const uint8_t* gp = gate_packed + row * (H / 2) + block_idx * 8;
    const uint8_t* up = up_packed   + row * (H / 2) + block_idx * 8;

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t gb = gp[p];
        uint8_t ub = up[p];
        float g_lo = e2m1_to_fp32_p1(gb & 0xF) * gate_scale;
        float g_hi = e2m1_to_fp32_p1(gb >> 4)  * gate_scale;
        float u_lo = e2m1_to_fp32_p1(ub & 0xF) * up_scale;
        float u_hi = e2m1_to_fp32_p1(ub >> 4)  * up_scale;
        float v0 = silu_mul_p1(g_lo, u_lo);
        float v1 = silu_mul_p1(g_hi, u_hi);
        vals[2*p]   = v0;
        vals[2*p+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    // Per-block scale + quantize + pack + SFA write
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);

    int out_sfa_off = layout_out(row, col_base, 0);
    out_sfa[out_sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* op = out_packed + row * (H / 2) + block_idx * 8;
    const float inv_bs = 1.f / bs_dq;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t lo = fp32_to_e2m1_p1(vals[2*p]   * inv_bs);
        uint8_t hi = fp32_to_e2m1_p1(vals[2*p+1] * inv_bs);
        op[p] = lo | (hi << 4);
    }
}

// AWQ-Down variant: fuse per-input-channel inv_s multiply between
// silu_mul and FP4 quant. Used by P1 split-GU path with use_awq=True.
template <bool UseGateLut, bool UseVectorIo, bool UseNativeFp4, class LayoutSF>
__global__ void silu_mul_two_mul_fp4_to_fp4_kernel(
    const uint8_t* __restrict__ gate_packed,
    const uint8_t* __restrict__ gate_sfa,
    const uint8_t* __restrict__ up_packed,
    const uint8_t* __restrict__ up_sfa,
    const __half*  __restrict__ inv_s,           // [H], shared across rows
    uint8_t* __restrict__ out_packed,
    uint8_t* __restrict__ out_sfa,
    LayoutSF layout_in, LayoutSF layout_out,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;
    const int col_base = block_idx * 16;

    int sfa_off = layout_in(row, col_base, 0);
    uint8_t gsf = gate_sfa[sfa_off];
    uint8_t usf = up_sfa[sfa_off];
    __nv_fp8_e4m3 gq, uq;
    *reinterpret_cast<uint8_t*>(&gq) = gsf;
    *reinterpret_cast<uint8_t*>(&uq) = usf;
    float gs = static_cast<float>(gq);
    float us = static_cast<float>(uq);

    const uint8_t* gp = gate_packed + row * (H/2) + block_idx * 8;
    const uint8_t* up = up_packed   + row * (H/2) + block_idx * 8;
    const __half* inv = inv_s + col_base;

    uint64_t gate_codes = 0;
    uint64_t up_codes = 0;
    if constexpr (UseVectorIo) {
        const uint2 gate_vec = *reinterpret_cast<const uint2*>(gp);
        const uint2 up_vec = *reinterpret_cast<const uint2*>(up);
        gate_codes = static_cast<uint64_t>(gate_vec.x) |
                     (static_cast<uint64_t>(gate_vec.y) << 32);
        up_codes = static_cast<uint64_t>(up_vec.x) |
                   (static_cast<uint64_t>(up_vec.y) << 32);
    }

    float vals[16];
    float amax = 0.f;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t gb;
        uint8_t ub;
        if constexpr (UseVectorIo) {
            gb = static_cast<uint8_t>(gate_codes >> (p * 8));
            ub = static_cast<uint8_t>(up_codes >> (p * 8));
        } else {
            gb = gp[p];
            ub = up[p];
        }
        float u_lo;
        float u_hi;
        if constexpr (UseNativeFp4) {
            const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
                static_cast<__nv_fp4x2_storage_t>(ub), __NV_E2M1);
            const float2 pair = __half22float2(
                *reinterpret_cast<const __half2*>(&raw));
            u_lo = pair.x * us;
            u_hi = pair.y * us;
        } else {
            u_lo = e2m1_to_fp32_p1(ub & 0xF) * us;
            u_hi = e2m1_to_fp32_p1(ub >> 4)  * us;
        }
        float inv_lo;
        float inv_hi;
        if constexpr (UseVectorIo) {
            const float2 inv_pair = __half22float2(
                reinterpret_cast<const __half2*>(inv)[p]);
            inv_lo = inv_pair.x;
            inv_hi = inv_pair.y;
        } else {
            inv_lo = __half2float(inv[2*p]);
            inv_hi = __half2float(inv[2*p+1]);
        }
        float v0, v1;
        if constexpr (UseGateLut) {
            v0 = geglu_gate_lut_p1[
                     (static_cast<int>(gsf) << 4) | (gb & 0xF)] *
                 u_lo * inv_lo;
            v1 = geglu_gate_lut_p1[
                     (static_cast<int>(gsf) << 4) | (gb >> 4)] *
                 u_hi * inv_hi;
        } else {
            float g_lo = e2m1_to_fp32_p1(gb & 0xF) * gs;
            float g_hi = e2m1_to_fp32_p1(gb >> 4)  * gs;
            v0 = silu_mul_p1(g_lo, u_lo) * inv_lo;
            v1 = silu_mul_p1(g_hi, u_hi) * inv_hi;
        }
        vals[2*p]   = v0;
        vals[2*p+1] = v1;
        float a0 = fabsf(v0), a1 = fabsf(v1);
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    float bs_dq = static_cast<float>(bs_q);
    int out_sfa_off = layout_out(row, col_base, 0);
    out_sfa[out_sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);

    uint8_t* op = out_packed + row * (H/2) + block_idx * 8;
    const float inv_bs = 1.f / bs_dq;
    uint64_t packed_result = 0;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        uint8_t packed_byte;
        if constexpr (UseNativeFp4) {
            packed_byte = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
                make_float2(vals[2*p] * inv_bs, vals[2*p+1] * inv_bs),
                __NV_E2M1, cudaRoundNearest));
        } else {
            uint8_t lo = fp32_to_e2m1_p1(vals[2*p]   * inv_bs);
            uint8_t hi = fp32_to_e2m1_p1(vals[2*p+1] * inv_bs);
            packed_byte = lo | (hi << 4);
        }
        if constexpr (UseVectorIo) {
            packed_result |= static_cast<uint64_t>(packed_byte) << (p * 8);
        } else {
            op[p] = packed_byte;
        }
    }
    if constexpr (UseVectorIo) {
        *reinterpret_cast<uint2*>(op) = make_uint2(
            static_cast<uint32_t>(packed_result),
            static_cast<uint32_t>(packed_result >> 32));
    }
}

#endif

void silu_mul_two_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);

    const int n_blocks = H / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_fp4_to_fp4_kernel<<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa,
        out_packed, out_sfa, layout, layout, H);
    check_fp4_kernel_launch("silu_mul_two_fp4_to_fp4");
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)out_packed; (void)out_sfa; (void)seq_len; (void)H; (void)stream;
#endif
}

void silu_mul_two_mul_fp4_to_fp4(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);
    const int n_blocks = H / 16;
    const int threads = 256;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_mul_fp4_to_fp4_kernel<false, false, false><<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa, inv_s,
        out_packed, out_sfa, layout, layout, H);
    check_fp4_kernel_launch("silu_mul_two_mul_fp4_to_fp4");
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)inv_s; (void)out_packed; (void)out_sfa;
    (void)seq_len; (void)H; (void)stream;
#endif
}

void silu_mul_two_mul_fp4_to_fp4_lut(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    ensure_geglu_gate_lut(stream);
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);
    const int n_blocks = H / 16;
    const int threads = 512;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_mul_fp4_to_fp4_kernel<true, true, false><<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa, inv_s,
        out_packed, out_sfa, layout, layout, H);
    check_fp4_kernel_launch("silu_mul_two_mul_fp4_to_fp4_lut");
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)inv_s; (void)out_packed; (void)out_sfa;
    (void)seq_len; (void)H; (void)stream;
#endif
}

void silu_mul_two_mul_fp4_to_fp4_lut_native(
    const uint8_t* gate_packed, const uint8_t* gate_sfa,
    const uint8_t* up_packed,   const uint8_t* up_sfa,
    const __half*  inv_s,
    uint8_t* out_packed, uint8_t* out_sfa,
    int seq_len, int H, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    ensure_geglu_gate_lut(stream);
    auto shape = cute::make_shape(seq_len, 1, H, 1);
    auto layout = CfgF4P1::tile_atom_to_shape_SFA(shape);
    const int n_blocks = H / 16;
    const int threads = 512;
    const int y_groups = (n_blocks + threads - 1) / threads;
    dim3 grid(seq_len, y_groups);
    dim3 block(threads);
    silu_mul_two_mul_fp4_to_fp4_kernel<true, true, true><<<grid, block, 0, stream>>>(
        gate_packed, gate_sfa, up_packed, up_sfa, inv_s,
        out_packed, out_sfa, layout, layout, H);
    check_fp4_kernel_launch("silu_mul_two_mul_fp4_to_fp4_lut_native");
#else
    (void)gate_packed; (void)gate_sfa; (void)up_packed; (void)up_sfa;
    (void)inv_s; (void)out_packed; (void)out_sfa;
    (void)seq_len; (void)H; (void)stream;
#endif
}

}  // namespace fused_fp4
}  // namespace flash_rt
