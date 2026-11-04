// SPDX-License-Identifier: Apache-2.0
#pragma once
// M=16 NVFP4-weight / FP8-activation block-128 scaled GEMM -> BF16, as a
// register-direct weight stream. Device body: fp4_stream_m16_sm89.cu.
//
// At M=16 the GEMM is a weight stream: the tensor core is idle and the kernel
// is as fast as it keeps DRAM busy. The shared-memory kernels
// (fp4_bs_gemm{,_splitk}_sm89) stage the packed tile, expand it to FP8 in
// shared memory and ldmatrix it back, which caps them at 176-222 GB/s on an
// RTX 4060 Ti. Here the payload is pre-shuffled ("fragment-major") so that one
// 16-byte load per lane per 128-K block is exactly that lane's four
// m16n8k32 B fragments; the nibble -> e4m3 decode is a per-(row, K-block)
// 16-entry byte LUT applied with PRMT in registers.
//
// Numerics are bit-identical to fp4_bs_gemm{,_splitk}_kernel with the SF128
// scale layout: the LUT holds the same cvt.e4m3(code * sf * gscale / w_scale)
// bytes the shared-memory expansion writes, each 128-K block runs the same
// four MMAs from a zero accumulator, and the block scale fold is the same
// expression. `splits` reproduces the split-K kernel's fp32 summation order
// in-register (partials over K128/splits blocks, summed in split order), so no
// partial buffer and no reduction launch are needed.
//
// A    : [16, K] FP8 e4m3 row-major, act_scale [16, K/128] fp32
// Wf   : fragment-major payload from nvfp4_pack_frag_m16 ([N/8, K/128, 32, 16B])
// Bsf  : UE4M3 SF128 scale factors, [N/128, K/128, 128] u8
// gscale : [1] fp32 per-tensor global scale
// w_scale: [N/128, K/128] fp32 from the FP8 block-128 quantization
// D    : [16, N] BF16 (unused when `residual` is given)
// residual/bias: optional exact fold of residual += bf16(bf16(out) + bias)

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

// a_smem = 0: A fragments read through L1 (best for the narrow/long-K sites);
// a_smem = 1: A staged once per CTA in shared memory and read with ldmatrix
// (best for the wide-N sites; K <= 3072). depth in {2, 4}, warps in {4, 8}.
int fp4_stream_m16_sm89(const void* A, const void* Wf, const void* Bsf,
                        const float* gscale, const float* act_scale,
                        const float* w_scale, void* D, void* residual,
                        const void* bias, int M, int N, int K, int splits,
                        int a_smem, int depth, int warps, cudaStream_t s);

// Row-major packed e2m1 [N, K/2] (low nibble = even k) -> fragment-major.
int nvfp4_pack_frag_m16(const void* packed, void* frag, int N, int K,
                        cudaStream_t s);

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt
