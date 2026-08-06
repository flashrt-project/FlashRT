// ============================================================================
//  E0M3 decoder activation quantizers with optional per-16 Hadamard rotation.
//
//  Copies of the production kernels with the element encoder swapped for the
//  sign-magnitude uniform INT4 grid and an optional in-register FWHT-16
//  (device helpers duplicated locally to stay additive). Quantizer division
//  and reciprocal use IEEE intrinsics so --use_fast_math cannot perturb fp8
//  rounding ties.
// ============================================================================
#include "fused_fp4/pi05_e0m3_act.cuh"

#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {

#if FV_HAVE_CUTLASS

namespace e0m3_act_detail {

using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t fp32_to_e0m3_act(float x) {
    int mag = __float2int_rn(fabsf(x));
    if (mag > 7) mag = 7;
    uint8_t sign = (x < 0.f && mag > 0) ? 0x8u : 0x0u;
    return sign | static_cast<uint8_t>(mag);
}

// In-register FWHT over 16 values held by one thread; multiply by 1/4 to
// make the transform orthonormal (H16 entries are +-1).
__device__ __forceinline__ void fwht16_regs(float v[16]) {
    #pragma unroll
    for (int step = 1; step < 16; step <<= 1) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if ((i & step) == 0) {
                const float a = v[i];
                const float b = v[i + step];
                v[i] = a + b;
                v[i + step] = a - b;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) v[i] *= 0.25f;
}

// FWHT across a 16-lane group (one value per lane), butterfly via shfl_xor.
__device__ __forceinline__ float fwht16_lane(float v, int lane_in_block) {
    #pragma unroll
    for (int step = 1; step < 16; step <<= 1) {
        const float other = __shfl_xor_sync(0xffffffff, v, step, 16);
        v = (lane_in_block & step) ? (other - v) : (other + v);
    }
    return v * 0.25f;
}

// 16-lane-group E0M3 quantize (software packing; the native fp4 cvt only
// emits E2M1 so E0M3 always encodes in software).
template <bool UseRht, class LayoutSF>
__device__ __forceinline__ void quantize_register_value_e0m3(
    float value,
    uint8_t* packed_row,
    uint8_t* dst_sfa,
    LayoutSF layout,
    int row,
    int block_idx,
    int lane_in_block) {
    if constexpr (UseRht) {
        value = fwht16_lane(value, lane_in_block);
    }
    float amax = fabsf(value);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffff, amax, offset, 16));
    }

    float desired = __fdiv_rn(amax, 7.f);
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float inv_bs = __frcp_rn(static_cast<float>(bs_q));
    if (lane_in_block == 0) {
        const int sfa_off = layout(row, block_idx * 16, 0);
        dst_sfa[sfa_off] = *reinterpret_cast<uint8_t*>(&bs_q);
    }
    const uint8_t code = fp32_to_e0m3_act(value * inv_bs);
    const uint8_t next = static_cast<uint8_t>(__shfl_down_sync(
        0xffffffff, static_cast<unsigned>(code), 1, 16));
    if ((lane_in_block & 1) == 0) {
        packed_row[block_idx * 8 + lane_in_block / 2] =
            code | (next << 4);
    }
}

// Whole-block (16 values in registers) E0M3 quantize for the vec kernels.
template <bool UseRht, class LayoutSF>
__device__ __forceinline__ void quantize_block_e0m3(
    float vals[16],
    uint2* dst_packed,
    uint8_t* dst_sfa,
    LayoutSF layout,
    int row, int block_idx, int n_blocks) {
    if constexpr (UseRht) {
        fwht16_regs(vals);
    }
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float a = fabsf(vals[i]);
        if (a > amax) amax = a;
    }
    float desired = __fdiv_rn(amax, 7.f);
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float inv_bs = __frcp_rn(static_cast<float>(bs_q));

    dst_sfa[layout(row, block_idx * 16, 0)] =
        *reinterpret_cast<uint8_t*>(&bs_q);

    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const uint8_t lo = fp32_to_e0m3_act(vals[2 * p] * inv_bs);
        const uint8_t hi = fp32_to_e0m3_act(vals[2 * p + 1] * inv_bs);
        ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    dst_packed[row * n_blocks + block_idx] = out;
}

__device__ __forceinline__ float gelu_mul_e0m3(float g, float u) {
    // Same constants as the production GeGLU kernels.
    float gelu = g / (1.0f + expf(-1.5957691216057308f * g *
                                  (1.0f + 0.044715f * g * g)));
    return gelu * u;
}

// ── Kernels ────────────────────────────────────────────────────────

template <bool UseRht, class LayoutSF>
__global__ void adarms_e0m3_sfa_kernel(
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
        quantize_register_value_e0m3<UseRht>(
            __half2float(rounded), packed_row, dst_sfa, layout, row_idx,
            segment * 16 + block_group, lane_in_block);
    }
}

template <bool UseRht, class LayoutSF>
__global__ void gate_res_adarms_e0m3_sfa_kernel(
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
        quantize_register_value_e0m3<UseRht>(
            __half2float(rounded), packed_row, dst_sfa, layout, row_idx,
            segment * 16 + block_group, lane_in_block);
    }
}

template <bool UseRht, class LayoutSF>
__global__ void quantize_e0m3_sfa_vec_kernel(
    const int4* __restrict__ src,
    uint2* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int N, int D8) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int n_blocks = D8 >> 1;
  if (row >= N || block_idx >= n_blocks) return;

  const int4 raw0 = src[row * D8 + 2 * block_idx];
  const int4 raw1 = src[row * D8 + 2 * block_idx + 1];
  const __half* h0 = reinterpret_cast<const __half*>(&raw0);
  const __half* h1 = reinterpret_cast<const __half*>(&raw1);

  float vals[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    vals[i] = __half2float(h0[i]);
    vals[8 + i] = __half2float(h1[i]);
  }
  quantize_block_e0m3<UseRht>(vals, dst_packed, dst_sfa, layout,
                              row, block_idx, n_blocks);
}

template <bool UseRht, class LayoutSF>
__global__ void gate_geglu_e0m3_sfa_vec_kernel(
    const __half* __restrict__ merged,
    uint2* __restrict__ packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout,
    int H) {
    const int block_idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int row       = blockIdx.x;
    const int n_blocks  = H / 16;
    if (block_idx >= n_blocks) return;

    const int col_base = block_idx * 16;
    const __half* merged_row = merged + row * 2 * H;
    const int4* gate4 = reinterpret_cast<const int4*>(merged_row + col_base);
    const int4* up4 =
        reinterpret_cast<const int4*>(merged_row + H + col_base);

    const int4 graw[2] = {gate4[0], gate4[1]};
    const int4 uraw[2] = {up4[0], up4[1]};
    const __half2* g2 = reinterpret_cast<const __half2*>(graw);
    const __half2* u2 = reinterpret_cast<const __half2*>(uraw);

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        vals[2 * i] = gelu_mul_e0m3(__half2float(g2[i].x),
                                    __half2float(u2[i].x));
        vals[2 * i + 1] = gelu_mul_e0m3(__half2float(g2[i].y),
                                        __half2float(u2[i].y));
    }
    quantize_block_e0m3<UseRht>(vals, packed, dst_sfa, layout,
                                row, block_idx, n_blocks);
}

}  // namespace e0m3_act_detail

#endif  // FV_HAVE_CUTLASS

namespace fused_fp4 {

void pi05_adarms_e0m3_sfa_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, int use_rht, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    using namespace e0m3_act_detail;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    if (use_rht) {
        adarms_e0m3_sfa_kernel<true><<<seq_len, 256, 0, stream>>>(
            x, style, packed, sfa, gate, layout, dim);
    } else {
        adarms_e0m3_sfa_kernel<false><<<seq_len, 256, 0, stream>>>(
            x, style, packed, sfa, gate, layout, dim);
    }
#else
    (void)x; (void)style; (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)use_rht; (void)stream;
#endif
}

void pi05_gate_res_adarms_e0m3_sfa_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, int use_rht, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    using namespace e0m3_act_detail;
    auto shape = cute::make_shape(seq_len, 1, dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    if (use_rht) {
        gate_res_adarms_e0m3_sfa_kernel<true><<<seq_len, 256, 0, stream>>>(
            x, prev_gate, residual, style, packed, sfa, gate, layout, dim);
    } else {
        gate_res_adarms_e0m3_sfa_kernel<false><<<seq_len, 256, 0, stream>>>(
            x, prev_gate, residual, style, packed, sfa, gate, layout, dim);
    }
#else
    (void)x; (void)prev_gate; (void)residual; (void)style;
    (void)packed; (void)sfa; (void)gate;
    (void)seq_len; (void)dim; (void)use_rht; (void)stream;
#endif
}

int gate_geglu_e0m3_sfa_vec_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, int use_rht, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
    using namespace e0m3_act_detail;
    if (half_dim % 16 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(merged) & 15) ||
        (reinterpret_cast<uintptr_t>(packed) & 7)) return -1;
    const int n_blocks = half_dim / 16;
    const int threads = 128;
    dim3 grid(seq_len, (n_blocks + threads - 1) / threads);

    auto shape = cute::make_shape(seq_len, 1, half_dim, 1);
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    if (use_rht) {
        gate_geglu_e0m3_sfa_vec_kernel<true><<<grid, threads, 0, stream>>>(
            merged, reinterpret_cast<uint2*>(packed), sfa, layout, half_dim);
    } else {
        gate_geglu_e0m3_sfa_vec_kernel<false><<<grid, threads, 0, stream>>>(
            merged, reinterpret_cast<uint2*>(packed), sfa, layout, half_dim);
    }
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
    (void)merged; (void)packed; (void)sfa;
    (void)seq_len; (void)half_dim; (void)use_rht; (void)stream;
    return -2;
#endif
}

}  // namespace fused_fp4

namespace fp4 {

int quantize_e0m3_dynamic_sfa_fp16_vec(
    const void* src_fp16, void* dst_packed, void* dst_sfa,
    int N, int D, bool is_sfb, int use_rht, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  using namespace e0m3_act_detail;
  if (D % 16 != 0) return -1;
  if ((reinterpret_cast<uintptr_t>(src_fp16) & 15) ||
      (reinterpret_cast<uintptr_t>(dst_packed) & 7)) return -1;
  const int n_blocks = D / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);

  auto shape = cute::make_shape(
      is_sfb ? 1 : N,
      is_sfb ? N : 1,
      D, 1);

  if (is_sfb) {
    auto layout = Cfg::tile_atom_to_shape_SFB(shape);
    if (use_rht) {
      quantize_e0m3_sfa_vec_kernel<true><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const int4*>(src_fp16),
          reinterpret_cast<uint2*>(dst_packed),
          reinterpret_cast<uint8_t*>(dst_sfa), layout, N, D >> 3);
    } else {
      quantize_e0m3_sfa_vec_kernel<false><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const int4*>(src_fp16),
          reinterpret_cast<uint2*>(dst_packed),
          reinterpret_cast<uint8_t*>(dst_sfa), layout, N, D >> 3);
    }
  } else {
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    if (use_rht) {
      quantize_e0m3_sfa_vec_kernel<true><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const int4*>(src_fp16),
          reinterpret_cast<uint2*>(dst_packed),
          reinterpret_cast<uint8_t*>(dst_sfa), layout, N, D >> 3);
    } else {
      quantize_e0m3_sfa_vec_kernel<false><<<grid, threads, 0, stream>>>(
          reinterpret_cast<const int4*>(src_fp16),
          reinterpret_cast<uint2*>(dst_packed),
          reinterpret_cast<uint8_t*>(dst_sfa), layout, N, D >> 3);
    }
  }
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_fp16; (void)dst_packed; (void)dst_sfa;
  (void)N; (void)D; (void)is_sfb; (void)use_rht; (void)stream;
  return -2;
#endif
}

}  // namespace fp4
}  // namespace flash_rt
