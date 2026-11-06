// SPDX-License-Identifier: Apache-2.0
//
// Even-column FP4 compaction for the SwiGLU epilogue-fold output.
//
// The fused SwiGLU GEMM (cutlass_nvfp4_dual_gemm_silu_fp4out) runs on the
// interleaved gate|up weight at N=2*intermediate and writes silu(gate)*up
// DUPLICATED into both columns of each pair: packed byte j holds the same FP4
// value in both nibbles (= silu_mul[j]). With OutputSFVectorSize=32 the SFD is
// already per-16-OUTPUT-col and byte-identical to the down GEMM's per-16 SFA, so
// only the FP4 DATA needs compacting to [M, intermediate]:
//
//   out_byte[m, k] = (in_byte[m, 2k] & 0xF) | ((in_byte[m, 2k+1] & 0xF) << 4)
//
// i.e. gather the low nibble (the silu_mul value) of two adjacent input bytes.
// Memory-bound, trivial; ~3 us at the MLP shape. The SFD tensor is passed
// straight through to the down GEMM unchanged.

#include "fp4_swiglu_compact_sm120.cuh"
#include "pdl_launch.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {
// One thread per output byte. in: [M, inter] bytes (dup'd nibbles).
// out: [M, inter/2] bytes (2 silu values packed).
__global__ void swiglu_even_col_compact_kernel(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
    int M, int inter)
{
    const int out_cols = inter >> 1;                 // bytes per output row
    const long total = (long)M * out_cols;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        int row = idx / out_cols;
        int k   = idx - (long)row * out_cols;
        const uint8_t* irow = in + (long)row * inter;
        uint8_t a = irow[2 * k];
        uint8_t b = irow[2 * k + 1];
        out[(long)row * out_cols + k] = (a & 0x0F) | ((b & 0x0F) << 4);
    }
}
}  // namespace

void fp4_swiglu_even_col_compact(
    const void* in_packed, void* out_packed, int M, int inter,
    cudaStream_t stream)
{
    const long total = (long)M * (inter >> 1);
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    swiglu_even_col_compact_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(in_packed),
        reinterpret_cast<uint8_t*>(out_packed), M, inter);
}

namespace {
// VECTORIZED variant: one thread per 16 input bytes (= 8 output bytes). Reads a
// coalesced 16B (uint4) chunk, packs the low nibbles of the 8 adjacent byte pairs
// into 8 output bytes, writes a coalesced 8B (uint2). The original kernel above is
// 1-thread-per-output-byte (byte-granular loads, ~45% BW); this lifts it toward the
// 9.4MB-traffic roofline. Requires inter % 16 == 0 (qwen3: inter=12288 → 768/row).
__global__ void swiglu_even_col_compact_v2_kernel(
    const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
    int M, int inter)
{
    const int in_cols  = inter;          // input bytes per row (dup'd nibbles)
    const int out_cols = inter >> 1;     // output bytes per row (packed)
    const long row_chunks = in_cols >> 4;          // 16 input bytes / chunk
    const long total = (long)M * row_chunks;
    // PDL: the down GEMM may stage its prologue while this pass runs; the
    // fused GEMM's output is read only after the grid-dependency wait.
    flash_rt::pdl::pdl_trigger();
    flash_rt::pdl::pdl_wait();
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        long row = idx / row_chunks;
        long c   = idx - row * row_chunks;
        uint4 v = *reinterpret_cast<const uint4*>(in + row * in_cols + (c << 4));
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        uint8_t ob[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            ob[j] = (b[2 * j] & 0x0F) | ((b[2 * j + 1] & 0x0F) << 4);
        *reinterpret_cast<uint2*>(out + row * out_cols + (c << 3)) =
            *reinterpret_cast<const uint2*>(ob);
    }
}
}  // namespace

void fp4_swiglu_even_col_compact_v2(
    const void* in_packed, void* out_packed, int M, int inter,
    cudaStream_t stream)
{
    // Fall back to the scalar kernel if the row width isn't a multiple of 16 bytes.
    if ((inter & 0x0F) != 0) {
        fp4_swiglu_even_col_compact(in_packed, out_packed, M, inter, stream);
        return;
    }
    const long total = (long)M * (inter >> 4);
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    flash_rt::pdl::launch(swiglu_even_col_compact_v2_kernel, dim3(blocks), dim3(threads), 0,
                          stream, flash_rt::pdl::enabled(),
                          reinterpret_cast<const uint8_t*>(in_packed),
                          reinterpret_cast<uint8_t*>(out_packed), M, inter);
}

}  // namespace kernels
}  // namespace flash_rt
