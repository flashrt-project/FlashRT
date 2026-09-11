#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Everything a grouped MoE prefill needs from its router logits, as kernels.
//
// The chain this replaces was softmax, top-k, a renormalising divide, a stable
// argsort, two gathers, a bincount, a cumulative sum and a scatter -- ten
// tensor ops per layer, of which the top-k alone cost 25 ms of a 2048-token
// prefill.
//
// The permutation is built as a counting sort with per-block offsets, not an
// atomic scatter: prefill seeds a decode that has to reproduce, so slot order
// within an expert is fixed (ascending slot index, matching a stable argsort)
// rather than left to the order blocks happen to arrive in.
//
//   logits    (S, n_experts) bf16
//   ti        (S, topk) int32    out, expert per (token, rank)
//   tw        (S, topk) fp32     out, weights renormalised over the top-k
//   se        (S * topk,) int32  out, expert per sorted slot
//   stok      (S * topk,) int64  out, token per sorted slot -- 64-bit, alone
//             among these, because it is handed to the grouped activation
//             quantiser as its gather index and that kernel reads a long.
//             Emitting int32 here reads as garbage row indices there, which
//             surfaces as an illegal access three kernels later.
//   inv       (S * topk,) int32  out, sorted row holding slot i
//   group_off (n_experts + 1,) int32  out, prefix sums over experts
//   ws        workspace, moe_route_prefill_workspace_bytes(S, topk, n_experts)
//
// n_experts must be 32 times a power of two, at most 1024, since the top-k
// holds a row across one warp; topk at most 32. Returns 0 on success.
int moe_route_prefill_bf16(
    const void* logits,
    void*       ti,
    void*       tw,
    void*       se,
    void*       stok,
    void*       inv,
    void*       group_off,
    void*       ws,
    int         ws_bytes,
    int         S,
    int         n_experts,
    int         topk,
    cudaStream_t stream);

int moe_route_prefill_workspace_bytes(int S, int topk, int n_experts);

// Per-expert scale-factor byte offsets for the block-scaled activation layout,
// derived from the group boundaries the routing kernel already produced. The
// layout blocks rows by 128, so a group of c rows takes ceil(c / 128) super
// blocks of n_col * 512 bytes.
//   group_off (n_experts + 1,) int32
//   sfa_off   (n_experts,) int32  out
void moe_route_sfa_offsets(
    const void* group_off,
    void*       sfa_off,
    int         n_experts,
    int         n_col,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
