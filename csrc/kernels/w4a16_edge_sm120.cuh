// SPDX-License-Identifier: Apache-2.0
//
// W4A16 GEMV variants for a bandwidth-poor part (Jetson-class Blackwell).
//
// Same math, same weight layout, same results as w4a16_matvec_sm120 and
// moe_grouped_w4a16_sm120 -- bit for bit. What differs is two things the
// profiler found on a 20-SM part with ~244 GB/s of memory, where the original
// pair sits at 51% of that while the BF16 GEMV of the same shape reaches 100%:
//
//   1. The staged activation is read from shared memory at a 32-byte stride
//      across lanes, which puts eight lanes of a 128-bit load phase on four
//      banks. Measured: 432,685 bank conflicts over 98,816 shared loads, 2.41x
//      the wavefronts the traffic needs. Padding each block's footprint to 48
//      bytes lands the phase on eight distinct banks.
//
//   2. The UE4M3 block scale is decoded through a 256-entry __constant__ LUT
//      indexed by a per-lane byte. Constant memory serves one address per
//      cycle, so a divergent index serialises. The decode is four integer ops,
//      so it does not need a table at all.
//
// Neither changes an arithmetic result, which is the point: the variant is
// accepted only if it is bitwise identical to the kernel it replaces.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// y(1,N) = (x(1,K) bf16) . (W(N,K) NVFP4)^T, fp32 accumulate, bf16 out.
// Arguments and semantics are those of w4a16_matvec_sm120_bf16.
int w4a16_matvec_edge_sm120_bf16(
    const void*  x_bf16,
    const void*  W_packed,
    const void*  SFB,
    void*        out,
    int          N,
    int          K,
    float        alpha,
    cudaStream_t stream);

// Grouped per-slot GEMV: D[s,:] = A[s,:] . W[eidx[s]]^T * alpha[eidx[s]].
// Arguments and semantics are those of moe_grouped_w4a16_sm120_bf16.
int moe_grouped_w4a16_edge_sm120_bf16(
    const void*  A,
    const void*  W,
    const void*  SFB,
    const void*  alpha,
    const void*  eidx,
    void*        D,
    int          slots,
    int          N,
    int          K,
    long         a_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
