#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ================================================================
// FlashRT AMD — MFMA small-M FP8 GEMM (gfx950, wave64)
//
// Second-generation hand kernel for the decoder's tiny-M GEMMs,
// built on two measured facts from the M4/M5 campaigns:
//   1. the VALU convert+FMA inner product of smallm_fp8.hip is the
//      "unresolved 4MB-regime limiter" — hipBLASLt wins with MI16x16
//      matrix cores even at M=10 (the documented reactivation
//      condition for this line), and
//   2. the stream_probe winning recipe (independent load chains,
//      ~64B in flight per lane) applies to the MFMA operand stream:
//      B fragments are per-lane contiguous 8B in the nt (N,K) weight
//      layout, 16 rows in parallel — the rs_x4_i4 pattern that
//      measured ~5 TB/s.
//
// Shape contract (decoder pi05 sites: M=10, N∈{1024,2560,8192},
// K∈{1024,2048,4096}):
//   nt: D_bf16(M,N) = (A_fp8(M,K) @ W_fp8(N,K)^T) * dsa[0] * dsb[0]
//   1 <= M <= 16, N % 16 == 0, K % 1024 == 0, 16B-aligned pointers.
//
// Structure: grid = N/16 workgroups x 256 threads (4 waves). Each
// workgroup owns a 16-column tile; the 4 waves split K in 4 fixed
// segments, accumulate with V_MFMA_F32_16X16X32_FP8_FP8 (FP32
// accumulators, OCP e4m3 operands — gfx950 MFMA fp8 is OCP, not
// fnuz), then reduce the 4 partials in ascending-wave order through
// LDS and apply dsa*dsb before one BF16 store. No atomics; replay
// inside a captured graph is bit-identical.
//
// A (<= 16x4096 fp8 = 64KB max, real rows only) is staged once into
// LDS by the whole workgroup; fragment reads for pad rows m >= M
// return zero (their C rows are never stored).
//
// `variant` selects the operand orientation (0 = A-as-a/W-as-b,
// 1 = swapped) so the parity bench discovers the ISA fragment
// mapping empirically instead of trusting documentation; the host
// wrapper's production entry pins the verified one.
// ================================================================

int smallm_mfma_variant_count();
const char* smallm_mfma_variant_name(int id);

void smallm_mfma_nt(int variant,
                    const void* A_fp8, const void* W_fp8,
                    __hip_bfloat16* D, int M, int N, int K,
                    const float* d_scale_a, const float* d_scale_b,
                    hipStream_t stream);

// Split-K partial form for the WG-starved narrow-N shapes (N=1024 ->
// only 64 workgroups, per-WG streaming caps ~10GB/s): grid becomes
// (N/16, splits), each split streams K/splits and writes UNSCALED
// FP32 partials to ws[(split*M + m)*N + n]. The consumer kernel
// (gate_residual_ada_norm_fp8_ksum) sums the splits in ascending
// order, applies dsa*dsb and rounds to bf16 — preserving the
// unfused chain's dataflow with no extra graph node. Deterministic:
// no atomics. Constraints: as smallm_mfma_nt, plus splits in {2,4}
// and K % (splits*4096) friendly (K/(splits*4*32) in {2,4,8,16,32}).
void smallm_mfma_nt_partial(const void* A_fp8, const void* W_fp8,
                            float* ws, int M, int N, int K, int splits,
                            hipStream_t stream);

// Packed-weight form — the bare-metal lever: weights are static, so
// they are repacked ONCE at setup into the exact per-lane consumption
// order, making every workgroup's stream perfectly linear (the
// stream_probe co_x4_i4 pattern) instead of 16 rows strided K apart.
//
// Packed layout (per 16-row n-tile t, tile block = 16*K bytes):
//   pos(t, sp, lane) = t*16*K + (sp*64 + lane)*16, holding
//   W[n0+(lane&15)][64*sp + (lane>>4)*8 .. +8] ++ (same, +32)
// i.e. torch: Wq.view(N//16, 16, K//64, 2, 4, 8)
//                .permute(0, 2, 4, 1, 3, 5).contiguous()
// Constraints: as smallm_mfma_nt, plus K % 2048 == 0 for the wave
// split (K/(64*WAVES) chunk pairs per lane, WAVES=4).
void smallm_mfma_nt_packed(const void* A_fp8, const void* Wp_fp8,
                           __hip_bfloat16* D, int M, int N, int K,
                           const float* d_scale_a, const float* d_scale_b,
                           hipStream_t stream);
