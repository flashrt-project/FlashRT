// SPDX-License-Identifier: Apache-2.0
//
// Grouped NVFP4 block-scaled GEMM for sm_100-class Blackwell (datacenter
// SM100 / Jetson AGX Thor SM110): every routed expert of a MoE layer in one
// launch.
//
// Why this exists. Prefill routes S*8 tokens across 256 experts. Serving that
// as one GEMV per (token, expert) slot re-reads an expert's weight once per
// token that chose it -- 9.7 GB a layer at S=1024 -- and serving it as one
// GEMM per expert costs 256 launches and 256 Python iterations a layer, whose
// host time exceeded the device time. Neither scales: the first is bounded by
// L2 bandwidth, the second by the host.
//
// A grouped GEMM is bounded by neither. One launch covers every expert, each
// weight is read once, and -- because CUTLASS accepts the per-group problem
// shapes from device memory (the host-side array is optional) -- the launch
// geometry is host-known and the routing is not, which is what a CUDA-graph
// capture requires. That is the property this is really for: capture a prefill
// chunk once and replay it for any context length, rather than tuning a
// threshold per prompt length.
//
// Wire format matches cutlass_nvfp4_w4a16_gemm_sm100: e2m1 nibbles, UE4M3
// block scales of 16, Sm1xx block-scaled atom layout, BF16 out. The per-expert
// global scale enters as the epilogue's per-group alpha.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace gemm {

// Scratch the entry point needs, in bytes, for `groups` groups. Holds the
// per-group pointer/stride/layout arrays it fills on device, plus the CUTLASS
// workspace. Allocate once and reuse; it does not depend on the token counts.
size_t moe_grouped_gemm_nvfp4_sm100_scratch_bytes(int groups);

// D[off_e : off_e + cnt_e, :] = A[off_e : off_e + cnt_e, :] @ W[e].T * alpha[e]
//
//   A_packed    (slots, K/2)  u8   rows sorted by expert
//   SFA         per-group block-scaled atom layouts, group e at byte offset
//               sfa_offsets[e]
//   W_stack     (E, N, K/2)   u8
//   SFB_stack   (E, sfb_bytes) u8
//   alpha_dev   (E,) f32      device
//   D           (slots, N)    bf16
//   group_off   (E + 1,) i32  device, prefix sums of the per-expert counts
//   sfa_off     (E,) i32      device, byte offset of group e's SFA block
//
// Nothing is read to the host: the group shapes are derived on device from
// group_off. Returns 0 on success, nonzero on argument or CUTLASS error.
int moe_grouped_gemm_nvfp4_sm100_bf16out(
    const void*  A_packed,
    const void*  SFA,
    const void*  W_stack,
    const void*  SFB_stack,
    const void*  alpha_dev,
    void*        D,
    const void*  group_off,
    const void*  sfa_off,
    int          groups,
    int          N,
    int          K,
    long         w_stride,
    long         sfb_stride,
    void*        scratch,
    size_t       scratch_bytes,
    cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
