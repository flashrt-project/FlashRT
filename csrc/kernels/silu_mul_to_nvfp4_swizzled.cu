// SPDX-License-Identifier: Apache-2.0
//
// Fused silu(gate) * up + nvfp4 swizzled quantize.
// See header for design notes.
//
// Implementation pattern: row-per-block, two-pass (amax then quantize)
// — same structure as quantize_bf16_to_nvfp4_swizzled but reads
// gate/up and applies silu + multiply inline. Intermediate bf16 is
// stored in smem so we never round-trip through HBM.

#include "silu_mul_to_nvfp4_swizzled.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

// ── Helpers reproduced from quantize.cu (anonymous-namespace inlines).
// Bit-identical so the fused path produces the same SF / packed bytes
// as the unfused (silu_mul → quantize_bf16_to_nvfp4_swizzled) chain.

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
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

__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
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

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// One block per row. Smem holds:
//   * num_blocks floats for per-block amax / scale (reused across passes)
//   * cols bf16 for the silu*up intermediate (so pass 2 doesn't re-read
//     gate/up from HBM)
//
// Block layout:
//   blockIdx.x = row index (0..rows-1)
//   blockDim.x = 256 (matches the unfused quantize kernel)
__global__ void silu_mul_to_nvfp4_swizzled_kernel(
    const __nv_bfloat16* __restrict__ gate,    // (rows, cols) bf16
    const __nv_bfloat16* __restrict__ up,      // (rows, cols) bf16
    uint8_t* __restrict__ packed,              // (rows, cols/2) u8
    uint8_t* __restrict__ sf_swz,              // swizzled SFs
    int cols,
    int num_blocks,
    int n_col_blocks) {
  int row = blockIdx.x;
  const __nv_bfloat16* gate_row = gate + (size_t)row * cols;
  const __nv_bfloat16* up_row   = up   + (size_t)row * cols;
  uint8_t*             packed_row = packed + (size_t)row * (cols / 2);

  extern __shared__ __align__(16) uint8_t smem_raw[];
  // First num_blocks floats: per-block amax → scale → cached scale.
  float* smem_scales = reinterpret_cast<float*>(smem_raw);
  // Then cols bf16: the silu*up intermediate.
  __nv_bfloat16* smem_silu_mul = reinterpret_cast<__nv_bfloat16*>(
      smem_raw + num_blocks * sizeof(float));

  // Pass 1: zero amax, compute silu*up + per-block atomicMax of |silu*up|.
  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
    smem_scales[b] = 0.0f;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    // Match silu_mul_qwen36 rounding: silu(g) → bf16 round-trip → * u → bf16.
    float g = __bfloat162float(gate_row[i]);
    float u = __bfloat162float(up_row[i]);
    float silu_g = silu_f32(g);
    float silu_g_bf = static_cast<float>(__float2bfloat16(silu_g));
    __nv_bfloat16 silu_mul_bf = __float2bfloat16(silu_g_bf * u);
    smem_silu_mul[i] = silu_mul_bf;

    float val = fabsf(static_cast<float>(silu_mul_bf));
    int blk = i >> 4;
    atomicMax((int*)&smem_scales[blk], __float_as_int(val));
  }
  __syncthreads();

  // Pass 1.5: amax → UE4M3 SF, write swizzled, cache decoded scale.
  int rb = row / 128;
  int ri = row % 128;
  for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
    float amax = __int_as_float(*(int*)&smem_scales[b]);
    float scale = amax / 6.0f;
    uint8_t ue_scale = float_to_ue4m3_ceil(scale);
    int cb = b / 4;
    int ci = b % 4;
    int out_idx = (rb * n_col_blocks + cb) * 512
                + (ri % 32) * 16 + (ri / 32) * 4 + ci;
    sf_swz[out_idx] = ue_scale;
    smem_scales[b] = ue4m3_to_float(ue_scale);
  }
  __syncthreads();

  // Pass 2: re-read smem silu*up, quantize 16-element blocks, pack.
  int half_cols = cols >> 1;
  for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
    int i = p * 2;
    int blk = i >> 4;
    float scale = smem_scales[blk];
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    float v0 = __bfloat162float(smem_silu_mul[i])     * inv_scale;
    float v1 = __bfloat162float(smem_silu_mul[i + 1]) * inv_scale;

    // Boundary case: if i+1 falls in a different block (only happens
    // when block size doesn't align with i — for cols multiple of 16
    // and 16-elt blocks, i and i+1 always share blk; defensive code
    // mirrors the unfused kernel for parity).
    int blk1 = (i + 1) >> 4;
    if (blk1 != blk) {
      float scale1 = smem_scales[blk1];
      float inv1 = (scale1 > 0.0f) ? (1.0f / scale1) : 0.0f;
      v1 = __bfloat162float(smem_silu_mul[i + 1]) * inv1;
    }

    uint8_t fp4_lo = float_to_fp4_e2m1(v0);
    uint8_t fp4_hi = float_to_fp4_e2m1(v1);
    packed_row[p] = (fp4_hi << 4) | (fp4_lo & 0x0F);
  }
}

}  // namespace

int silu_mul_to_nvfp4_swizzled_bf16(
    const void* gate,
    const void* up,
    void*       packed,
    void*       sf_swz,
    int         rows,
    int         cols,
    cudaStream_t stream) {
  if (!gate || !up || !packed || !sf_swz) return 1;
  if (rows <= 0 || cols <= 0 || (cols & 0xF) != 0) return 2;

  int num_blocks = (cols + 15) / 16;
  int n_col_blocks = (num_blocks + 3) / 4;
  int threads = 256;
  size_t smem_size = num_blocks * sizeof(float)
                   + cols * sizeof(__nv_bfloat16);
  // RTX 5090 default-attribute smem is 48 KiB/block. For cols=12288:
  // smem = 768*4 + 12288*2 = 3072 + 24576 = 27648 bytes < 48 KiB OK.
  // For larger cols (or rows>1), opt into the dynamic upper bound.
  if (smem_size > 48 * 1024) {
    cudaFuncSetAttribute(
        silu_mul_to_nvfp4_swizzled_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100 * 1024);
  }
  silu_mul_to_nvfp4_swizzled_kernel<<<rows, threads, smem_size, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(up),
      reinterpret_cast<uint8_t*>(packed),
      reinterpret_cast<uint8_t*>(sf_swz),
      cols, num_blocks, n_col_blocks);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
