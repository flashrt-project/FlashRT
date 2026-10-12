// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 SigLIP self-attention at head_dim 72.
//
// The SigLIP tower this serves runs 16 heads of width 72 over a sequence that
// is a multiple of 128. FlashRT reaches head_dim 72 through FA2's hdim-96
// instantiation, which pads every head to 96 and spends a quarter of the MMA
// work on the pad; FA2 has no 72-wide tile and cannot express one. This kernel
// reduces QK^T over 80 padded dims (the pad is exactly zero in shared memory)
// and the PV product over the exact 72 output dims, so no lane is wasted on
// the tail.
//
// It is a full (non-causal) forward attention over one sequence per (view,
// head). Q, K and V are read in place from a fused qkv buffer laid out as
// three D-wide sections per row: Q at +0, K at +D, V at +2D.
//
// Real kernel constraints, checked before launch:
//   D == NH * 72      (the fused buffer's row is exactly the head count)
//   S % 128 == 0      (Q is loaded in 64-row blocks with MT up to 2)
//   D % 8 == 0        (16-byte vector loads over the qkv row)
// Unsupported shapes fail fast in the launcher rather than computing a result
// with a different reduction order.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace dm05 {

// Full attention over a fused qkv buffer, head_dim fixed at 72.
//
//   qkv/ld   — [nv * S, 3 * D] bf16, row stride ld elements; Q at column h*72,
//              K at D + h*72, V at 2*D + h*72 for head h
//   out/o_ld — [nv * S, D] bf16 attention output
//   S        — sequence length per view (keys and queries are the same length)
//   NH       — head count, D == NH * 72
//   nv       — number of views; the grid's z dimension
//   scale    — softmax scale (head_dim ** -0.5 as the caller computes it)
//   mt       — query rows per CTA, 1 or 2 blocks of 64 (2 is the default)
//   q8 / q8_scale — optional. When q8 is non-null the epilogue writes the
//              static-FP8 (e4m3) copy of the output that the o_projection
//              reads, and the bf16 store is skipped: the consumer is the FP8
//              GEMM, so a bf16 copy would be written and never read. The FP8
//              values are byte-identical to running FlashRT's
//              quantize_fp8_static over the bf16 output of the same call, and
//              q8_scale is the device scalar that pass would read, so the
//              caller's separate quantize pass disappears. `out` is left
//              untouched in this mode. Pass q8 = nullptr for a bf16-only write.
//
// The kernel is safe to capture in a CUDA graph: it takes only raw pointers,
// allocates nothing, and every address it reads is the caller's buffer.
void vit_attn_hd72(const __nv_bfloat16* qkv, long long ld, int D,
                   __nv_bfloat16* out, long long o_ld,
                   int S, int NH, int nv, float scale, int mt,
                   __nv_fp8_e4m3* q8, const float* q8_scale,
                   cudaStream_t stream);

// Largest dynamic shared memory the kernel asks for, in bytes. Exposed so a
// caller can check it against the device's per-block opt-in limit.
inline int vit_attn_hd72_smem_bytes() { return 4 * 64 * 88 * 2; }

}  // namespace dm05
}  // namespace flash_rt
