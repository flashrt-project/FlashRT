#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Causal depthwise conv1d over a whole prompt, several output tokens per
// thread.
//
// The existing prefill entry gives one thread one (channel, token), so each
// input element is fetched once for every output that needs it -- k times --
// and the channel's weight row is fetched once per token. At the Qwen3.6
// prefill shape that is 134 MB of reads for 67 MB of data, and the kernel
// measures about four times off what its traffic implies.
//
// Here a thread walks `rows` consecutive tokens of one channel, holding the
// last k inputs in registers, so each input is read once and the weight row
// once per thread rather than once per token.
//
//   x    (B, S, conv_dim) bf16
//   w    (conv_dim, k) bf16
//   bias (conv_dim,) bf16 or null
//   out  (B, S, conv_dim) bf16
//   hist (B, conv_dim, k-1) bf16 or null -- the previous block's last k-1
//        inputs, channel-major with the newest last, which is the layout the
//        decode conv state already carries. Null means the sequence starts
//        here and the reads before it are zero.
//
// `hist` is what lets a chunked prefill stop concatenating. Prepending the
// history to the activations and slicing the result back off copies the whole
// block twice per layer -- 691 ms of a 32768-token prefill, in `cat` and its
// batched copy -- to supply three tokens of context.
//
// k must be at most 4. Layout, causality and the optional silu match the
// existing entry exactly; this is the same function computed with less
// traffic.
void causal_conv1d_qwen36_rows_bf16(
    const void* x,
    const void* w,
    const void* bias,
    void*       out,
    int B,
    int S,
    int conv_dim,
    int k,
    bool apply_silu,
    cudaStream_t stream);

// Same, continuing from a previous block's trailing inputs.
void causal_conv1d_qwen36_rows_hist_bf16(
    const void* x,
    const void* w,
    const void* bias,
    const void* hist,
    void*       out,
    int B,
    int S,
    int conv_dim,
    int k,
    bool apply_silu,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
