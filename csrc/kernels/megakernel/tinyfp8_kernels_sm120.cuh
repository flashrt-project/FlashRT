// SPDX-License-Identifier: Apache-2.0
//
// tiny_fp8 small-shape FP8 W·X GEMM kernels for sm_120a (RTX 5090).
//
// 2-stage tile variants used by the motus tiny_fp8 dispatcher to
// route hot action-expert / und-module sites where cuBLASLt heuristic
// underperforms hand-tuned tiles at small M:
//
//   gemm_M8_N32_K128  : (8,  9216, 1024)        action QKV joint
//   gemm_M8_N32_K256  : (8,  1024, 4096)        action FFN_dn
//   gemm_M8_N32_K512  : (8,  4096, 1024)        action FFN_up
//                       (8,  1024, 3072)        action wan_o
//   gemm_M16_N32_K64  : (138, 2048, 512)        und FFN_up / und O (M=138 → pad 144)
//   gemm_M16_N64_K64  : (138, 512,  2048)       und FFN_dn (M=138 → pad 144)
//   gemm_M32_N32_K128 : (21, 9216, 1024)        Stage3 action QKV
//   gemm_M32_N32_K512 : (21, 1024, 3072)        Stage3 action O
//
// All take A in (M, K) row-major FP8e4m3 and B in (N, K) row-major FP8e4m3,
// write D in (M, N) bf16. D = alpha * (A @ B^T).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace megakernel {

int tinyfp8_gemm_M8_N32_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M8_N32_K256_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M8_N32_K512_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M16_N32_K64_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M16_N64_K64_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M32_N32_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm_M32_N32_K512_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);
int tinyfp8_gemm3_M16_N64_K128_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K, float alpha, cudaStream_t s);

}  // namespace megakernel
}  // namespace flash_rt
