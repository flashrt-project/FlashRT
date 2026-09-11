#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

// ================================================================
// FlashRT AMD — hand-tuned small-M FP8 GEMM (CDNA4, wave64)
//
// Weight-streaming kernels for the decoder's tiny-M GEMMs
// (M ≈ action-chunk rows, N/K in the 1-4K range). At these shapes
// the GEMM is a GEMV-with-few-rows: the cost is streaming the FP8
// weight matrix once at DRAM rate, not matrix-core math. hipBLASLt
// tiles (MT16x16 etc.) leave most of that bandwidth on the table.
//
// Semantics: exact drop-in for GemmRunner::fp8_nn_dev / fp8_nt_dev
// (see hipblaslt_runner.h):
//
//   nn:  D_bf16(M,N) = (A_fp8(M,K) @ B_fp8(K,N)) * d_scale_a[0] * d_scale_b[0]
//        A row-major (M,K), B row-major (K,N)  — the frontend's "kn" layout.
//   nt:  D_bf16(M,N) = (A_fp8(M,K) @ B_fp8(N,K)^T) * d_scale_a[0] * d_scale_b[0]
//        B row-major (N,K)                     — the frontend's "nk" layout.
//
// FP8 is OCP e4m3 (__hip_fp8_e4m3 byte format, HIP_R_8F_E4M3 — never
// fnuz). Accumulation is FP32; the per-tensor descales are DEVICE
// float pointers dereferenced inside the kernel (HIP Graph safe) and
// applied to the FP32 accumulator before the BF16 store, matching the
// hipBLASLt A/B_SCALE_POINTER contract.
//
// Determinism: every reduction runs in a fixed order — sequential K
// loop per lane, fixed-shape shuffle tree per wave, and (nn split-K
// only) a second reduce kernel that sums slice partials in ascending
// slice order from a caller-provided FP32 workspace. No atomics
// anywhere, so replays inside a captured graph are bit-identical.
//
// Constraints (checked, throw std::runtime_error on violation):
//   nn: 1 <= M <= 16, N % 64 == 0, K % 64 == 0
//   nt: 1 <= M <= 16, K % 16 == 0   (N is guarded per-column)
//   A and B must be 16-byte aligned (any torch/hipMalloc allocation).
// ================================================================

// Maximum split-K factor the nn path will ever choose. Sizes the
// caller-provided workspace: pass a device buffer of at least
// smallm_fp8_nn_dev_ws_bytes(M, N) bytes as split_ws. The workspace
// holds unscaled FP32 partials laid out ws[s][m][n] (s = k-slice).
#define SMALLM_FP8_MAX_SPLIT 32

// Required split_ws size in bytes for smallm_fp8_nn_dev(_alt).
size_t smallm_fp8_nn_dev_ws_bytes(int M, int N);

// nn ("kn" weights, K-major rows of N): drop-in for GemmRunner::fp8_nn_dev.
// split_ws: device FP32 workspace (>= smallm_fp8_nn_dev_ws_bytes(M, N))
// used for the deterministic split-K reduction. May be nullptr, in which
// case split-K is disabled (fewer workgroups in flight — slower on wide
// GPUs for small N, but no workspace needed).
void smallm_fp8_nn_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream);

// Alternate nn config for A/B benching: targets 2x the workgroup count
// (deeper split-K — more memory in flight, more reduce traffic).
void smallm_fp8_nn_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream);

// nt ("nk" weights, N-major rows of K): drop-in for GemmRunner::fp8_nt_dev.
// This is the stream-friendliest layout — every output column owns a
// contiguous K row, so no split-K and no workspace are ever needed.
// split_ws is accepted for signature symmetry and ignored (may be nullptr).
//
// Default path: the double-buffered LDS pipeline (smallm_fp8_nt_lds_dev
// below) with async staging; env overrides, read on the host at
// launch/capture time (a captured graph keeps the capture-time choice):
//   FRT_SMALLM_REG=1  -> legacy register-only ILP4 kernel
//   FRT_LDS_ASYNC=0   -> LDS pipeline with sync (register-hoisted) staging
void smallm_fp8_nt_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream);

// nt LDS pipeline, explicit entry for A/B benching: double-buffered
// global->LDS staging of the WG's weight columns (2-16KB LDS per WG,
// w4 only), consumer fed from LDS off the load critical path. Both
// staging variants are compiled into the binary; use_async != 0 picks
// gfx9 LDS-DMA (__builtin_amdgcn_global_load_lds, 4B/lane per issue),
// use_async == 0 picks the register-hoisted sync fallback with the
// identical LDS layout. Deterministic fixed-order accumulation (NOT
// bit-identical to the register nt kernel — different lane<->k split).
void smallm_fp8_nt_lds_dev(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, int use_async,
                           hipStream_t stream);

// 1 if the binary was built with the async LDS-DMA staging variant
// (FRT_LDS_ASYNC=1, the default); 0 if use_async degenerates to sync.
int smallm_fp8_nt_lds_async_available();

// Alternate nt config for A/B benching: 8-wave (512-thread) WGs
// instead of 4-wave, same total wave count (halved grid). Values are
// bit-identical to smallm_fp8_nt_dev for every shape.
void smallm_fp8_nt_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream);
