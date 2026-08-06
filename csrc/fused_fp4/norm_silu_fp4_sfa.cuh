// ============================================================================
//  FlashRT — F2/F3/F4: fused pre-GEMM kernels that produce NVFP4 + SFA in one
//  launch, matching FP8 path's single-kernel pre-GEMM pattern.
//
//  F2: rms_norm_noweight + fp4_quant + SFA            (layer-entry / no residual)
//  F3: residual_add + rms_norm_noweight + fp4_quant + SFA   (per-layer transition)
//  F4: gate_silu_mul_merged + fp4_quant + SFA         (post Gate+Up, pre Down)
//
//  All additive. No existing kernel modified.
// ============================================================================
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace fused_fp4 {

// F2: rms_norm → fp4 packed + SFA tile-interleaved.
// x [S, D] fp16 → packed [S, D/2] uint8 + sfa (CUTLASS SFA layout, is_sfb=false)
// F2 + AWQ: rms_norm(x) * inv_s[c] → fp4 + SFA.
void rms_norm_mul_fp4_sfa_fp16(
    const __half* x, const __half* inv_s, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

void rms_norm_fp4_sfa_fp16(
    const __half* x, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

// F3: residual += x; rms_norm(residual) → fp4 packed + SFA. In-place update of
// residual buffer (both paths, FP8 and FP4, already do this).
void residual_add_rms_norm_fp4_sfa_fp16(
    __half* residual, const __half* x,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

// F3 v2: register-only layout (1 thread = 1 NVFP4 block of 16 elements).
// Requires blockDim = D/16. Designed for Pi0.5 D=2048 (128 threads).
void residual_add_rms_norm_fp4_sfa_v2_fp16(
    __half* residual, const __half* x,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

// Pi0.5 decoder C1: AdaRMSNorm(x, style) -> FP4 + SFA and fp16 gate.
// style is [S, 3D] with scale, shift, gate contiguous along the last axis.
void pi05_adarms_fp4_sfa_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream);

// Pi0.5 decoder C4->C5 / C7->C1: residual += x * prev_gate, then
// AdaRMSNorm(residual, style) -> FP4 + SFA and the next fp16 gate.
void pi05_gate_res_adarms_fp4_sfa_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style, uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream);

// Same Pi0.5 fused preprocessing with native SM110 round-to-nearest-even
// E2M1x2 output conversion. RMS, fp16 rounding, SFA, residual, and gate
// semantics are unchanged.
void pi05_adarms_fp4_sfa_native_fp16(
    const __half* x, const __half* style,
    uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream);

void pi05_gate_res_adarms_fp4_sfa_native_fp16(
    const __half* x, const __half* prev_gate, __half* residual,
    const __half* style, uint8_t* packed, uint8_t* sfa, __half* gate,
    int seq_len, int dim, cudaStream_t stream);

// F4: gate_silu_mul_merged(merged [S, 2H]) → hid [S, H] → fp4 packed + SFA.
//     GELU(gate) * up, where merged = [gate || up] along dim 1.
void gate_silu_mul_fp4_sfa_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream);

// F4 v2: same semantics as F4, streamlined 1-thread-per-NVFP4-block layout
// (no shared memory, no syncthreads). Typically ~1.3-1.5x faster at H=8192.
void gate_silu_mul_fp4_sfa_v2_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream);

// Vectorized bit-exact variant of gate_silu_mul_fp4_sfa_v2_fp16 (16B loads,
// 8B packed stores). Returns nonzero without launching on unaligned
// buffers; callers fall back to the v2 kernel. Implemented in
// silu_mul_fp4_sfa_vec.cu.
int gate_silu_mul_fp4_sfa_vec_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream);

// F3 + AWQ multiply: residual_add + rms_norm + per-channel inv_s multiply +
// fp4 quant + SFA write — all in one kernel. Used by the AWQ frontend path
// to avoid the extra per_channel_mul + unfused quant launches.
void residual_add_rms_norm_mul_fp4_sfa_fp16(
    __half* residual, const __half* x, const __half* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

// F4 v2 + AWQ multiply: gate_silu_mul + per-channel inv_s multiply +
// fp4 quant + SFA write — register-only, single-launch.
void gate_silu_mul_mul_fp4_sfa_v2_fp16(
    const __half* merged, const __half* inv_s,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
