// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 swizzled -> BF16 weight dequant. See header for the contract.
// SF swizzle offset + FP4 codebook mirror csrc/kernels/fp4_w4a4_matvec_sm120.cu
// exactly so the dequantized weight matches the W4A4 GEMM's view of B.

#include "nvfp4_dequant_swizzled_to_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

namespace {

// FP4 e2m1 codebook: signed 4-bit nibble -> fp32 magnitude+sign.
__device__ __constant__ float c_fp4_cb[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// UE4M3 (FP8 e4m3, non-negative scale) decode, computed arithmetically
// (no LUT dependency on the matvec TU). Bias 7; E=0 subnormal; E=15,M=7 NaN->0.
__device__ __forceinline__ float ue4m3_decode(uint8_t b) {
  const int e = (b >> 3) & 0xF;
  const int m = b & 0x7;
  if (e == 0) {
    return static_cast<float>(m) * 0.001953125f;  // m * 2^-9
  }
  if (e == 0xF && m == 7) {
    return 0.0f;
  }
  return (1.0f + static_cast<float>(m) * 0.125f) * exp2f(static_cast<float>(e - 7));
}

// Identical to the matvec kernel's swizzle map (Sm1xx blockscaled atom).
__device__ __forceinline__ int sf_swz_offset(int row, int k_block,
                                              int n_col_super) {
  const int rb = row >> 7;
  const int ri = row & 127;
  const int cb = k_block >> 2;
  const int ci = k_block & 3;
  const int super_idx = rb * n_col_super + cb;
  const int inner_off = (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + ci;
  return super_idx * 512 + inner_off;
}

// One thread per packed byte: decodes two consecutive k-elements (both in
// the same 16-element scale block, since pairs (2i, 2i+1) never straddle a
// 16 boundary). Grid-strided over N*(K/2) bytes.
__global__ void dequant_kernel(const uint8_t* __restrict__ B,
                               const uint8_t* __restrict__ SFB,
                               __nv_bfloat16* __restrict__ D,
                               long total_bytes, int K, int n_col_super,
                               float alpha) {
  const int kbytes = K >> 1;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total_bytes; idx += (long)gridDim.x * blockDim.x) {
    const int row = idx / kbytes;
    const int byte_in_row = idx - (long)row * kbytes;
    const int k0 = byte_in_row << 1;          // even k-element
    const int k_block = k0 >> 4;              // 16 elements per block
    const float sf = ue4m3_decode(__ldg(SFB + sf_swz_offset(row, k_block,
                                                            n_col_super)));
    const float s = sf * alpha;
    const uint8_t byte = __ldg(B + idx);
    const float lo = c_fp4_cb[byte & 0xF] * s;
    const float hi = c_fp4_cb[(byte >> 4) & 0xF] * s;
    const long base = (long)row * K + k0;
    D[base] = __float2bfloat16(lo);
    D[base + 1] = __float2bfloat16(hi);
  }
}

}  // namespace

void nvfp4_dequant_swizzled_to_bf16(const uint8_t* B_packed,
                                    const uint8_t* SFB,
                                    void* D_bf16,
                                    int N, int K,
                                    float alpha,
                                    void* stream) {
  const long total_bytes = (long)N * (K >> 1);
  const int n_col_super = ((K >> 4) + 3) / 4;
  const int threads = 256;
  long blocks = (total_bytes + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;  // grid-stride handles the rest
  dequant_kernel<<<(int)blocks, threads, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
      B_packed, SFB, reinterpret_cast<__nv_bfloat16*>(D_bf16),
      total_bytes, K, n_col_super, alpha);
}

}  // namespace quantize
}  // namespace flash_rt
