#pragma once

#include <hip/hip_runtime.h>

// ================================================================
// FlashRT AMD — fused decoder-FFN kernel pair (CDNA4, wave64)
//
// Replaces the pi05 decoder's per-(layer,step) FFN sub-chain
//
//   gate|up fp8 GEMM (M=10, N=2*4096, K=1024)   [fp8_nt_dev]
//     -> gate_geglu_merged_fp8 (geglu + fp8 quantize)
//     -> down fp8 GEMM (M=10, N=1024, K=4096)   [fp8_nt_dev]
//     -> gate_mul_residual (residual += x * gate)
//
// with TWO weight-streaming GEMM kernels whose epilogues absorb the
// elementwise pieces. The chain is bandwidth-bound: weight bytes per
// chain are 8.39 MB (gate|up) + 4.19 MB (down) fp8 = 12.6 MB, a
// ~2.5 us floor at 5 TB/s, while the 4-kernel chain measures
// ~25-30 us (hipBLASLt small-M tiles + 3 intermediate round trips +
// launch gaps). Both kernels stream their weight matrix exactly once
// with 16-byte lane loads (dwordx4), keep the tiny A operand in
// registers, accumulate FP32, and fold with wave shuffles — the same
// proven streaming skeleton as smallm_fp8_nt_dev (smallm_fp8.hip):
// identical lane-K partition, chunk order, and shuffle tree, so the
// GEMM accumulation order (and thus the fp32 result) is bit-identical
// to smallm_fp8_nt_dev on the same inputs.
//
// Layout: weights are the frontend's "nk" layout, row-major (N, K)
// with K contiguous — each output column owns one contiguous K-row of
// the weight, the stream-friendliest layout (no split-K, no LDS, no
// workspace, no atomics; replays inside captured graphs are
// bit-identical).
//
// ── Kernel 1: smallm_fp8_gateup_geglu ──
//   OutFp8(M, N_half) = quant_e4m3( geglu( A_fp8(M,K) @ W(2*N_half,K)^T
//                                          * sa * sw ) / s_out )
//   W row j       = gate column j   (j <  N_half)
//   W row N_half+j = up  column j   (merged buffer convention of
//   gate_geglu_merged_fp8: gate is the FIRST half, up the SECOND).
//   Access pattern: geglu output column j needs BOTH W rows j and
//   N_half + j, which sit N_half*K bytes apart in the nk layout. Each
//   wave therefore streams two row groups per owned column — its
//   gate-row (offset j*K) and its up-row (offset (N_half+j)*K) — off
//   the same lane K-window, reusing the converted A chunk for both.
//   Epilogue (per output element, replicating the unfused chain's
//   rounding points EXACTLY):
//     g = bf16_round(acc_gate * sa * sw)        // fp8_nt_dev's BF16 D
//     u = bf16_round(acc_up   * sa * sw)
//     gelu = g / (1 + exp(-1.5957691216057308*g*(1 + 0.044715*g*g)))
//     v = clamp(gelu*u / s_out, ±448) -> __hip_fp8_e4m3 (RNE, sat)
//
//   Grid: 256-thread WGs (4 waves), NPW geglu-columns per wave.
//   Default NPW=2 -> 8 columns (16 weight rows, 16 KB @ K=1024) per
//   WG -> N_half/8 = 512 WGs at N_half=4096: 2 per CU on MI350X's 256
//   CUs, and the same register footprint as the proven smallm nt
//   NPW=4 config (4 B-row streams per converted A chunk). The _alt
//   config uses NPW=4 (256 WGs, more A reuse, higher VGPR pressure).
//
// ── Kernel 2: smallm_fp8_down_gateres ──
//   x(M, N)   = bf16_round( H_fp8(M,K) @ W_down(N,K)^T * sh * sw )
//   residual(M, N) += x * gate        (in place, BF16)
//   Epilogue replicates gate_mul_residual exactly: the GEMM result is
//   rounded to BF16 FIRST (that is what the unfused kernel reads from
//   the GEMM's BF16 output buffer), then
//     r = f32(residual) + f32(x_bf16) * f32(gate_bf16) -> bf16_round.
//   The intermediate x buffer is never materialized in HBM.
//
//   Grid: default NPW=1 -> 4 columns (4 KB @ K=4096) per WG ->
//   N/4 = 256 WGs at N=1024: exactly one per CU with a 4-chunk serial
//   K loop per wave. The _alt config uses NPW=2 (128 WGs, halves the
//   per-byte A-convert ALU cost by reusing the converted A chunk
//   across 2 columns — bench both on device).
//
// Bandwidth math (pi05 shapes, per chain):
//   mega1: 8.39 MB weights + 10 KB A + 40 KB OutFp8  -> 1.69 us @ 5 TB/s
//   mega2: 4.19 MB weights + 40 KB H + 60 KB res/gate -> 0.86 us @ 5 TB/s
//   vs the unfused chain's ~13.0 MB + 460 KB of intermediates + 4
//   launch/gap overheads.
//
// FP8 is OCP e4m3 (__hip_fp8_e4m3, HIP_R_8F_E4M3 — never fnuz).
// Scales are DEVICE float pointers dereferenced in-kernel (HIP Graph
// safe), matching the hipBLASLt A/B_SCALE_POINTER contract.
//
// Constraints (checked, throw std::runtime_error on violation):
//   1 <= M <= 16, K % 16 == 0 (whole 16B lane loads); N / N_half is
//   guarded per column. A/H/W must be 16-byte aligned.
// ================================================================

// Kernel 1: gate|up GEMM + GeGLU + static-scale FP8 quantize.
//   A_fp8:        (M, K) fp8 e4m3, row-major
//   W_gateup:     (2*N_half, K) fp8 e4m3, row-major "nk" (gate rows
//                 [0, N_half), up rows [N_half, 2*N_half))
//   OutFp8:       (M, N_half) fp8 e4m3, row-major
//   d_scale_a/w:  device fp32 descales for A and W
//   d_scale_out:  device fp32 quantization scale of the down-proj
//                 activation (divides, exactly as gate_geglu_merged_fp8)
void smallm_fp8_gateup_geglu(const void* A_fp8, const void* W_gateup,
                             void* OutFp8,
                             const float* d_scale_a, const float* d_scale_w,
                             const float* d_scale_out,
                             int M, int N_half, int K, hipStream_t stream);

// Alternate config for A/B benching (NPW=4: 2x A reuse, half the WGs).
void smallm_fp8_gateup_geglu_alt(const void* A_fp8, const void* W_gateup,
                                 void* OutFp8,
                                 const float* d_scale_a, const float* d_scale_w,
                                 const float* d_scale_out,
                                 int M, int N_half, int K, hipStream_t stream);

// Kernel 2: down GEMM + gate*x + residual (in place).
//   H_fp8:        (M, K) fp8 e4m3, row-major (kernel 1's output)
//   W_down:       (N, K) fp8 e4m3, row-major "nk"
//   residual:     (M, N) bf16, updated in place: residual += x * gate
//   gate:         (M, N) bf16 (the ada-norm gate buffer)
//   d_scale_h/w:  device fp32 descales for H and W
void smallm_fp8_down_gateres(const void* H_fp8, const void* W_down,
                             void* residual, const void* gate,
                             const float* d_scale_h, const float* d_scale_w,
                             int M, int N, int K, hipStream_t stream);

// Alternate config for A/B benching (NPW=2: 2x A reuse, half the WGs).
void smallm_fp8_down_gateres_alt(const void* H_fp8, const void* W_down,
                                 void* residual, const void* gate,
                                 const float* d_scale_h, const float* d_scale_w,
                                 int M, int N, int K, hipStream_t stream);
