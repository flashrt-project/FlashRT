// SPDX-License-Identifier: Apache-2.0
//
// Host-side entry points for the Spark-X2.5 model kernels.
//
// Built into its own module (flash_rt_sparkx25) rather than into
// flash_rt_kernels: the pipeline is trained, validated and shipped as a unit, so
// a change here must not re-trigger the shared kernel build. See the
// pybind11_add_module block for this module in the root CMakeLists.txt.
//
// Declared with void*/uint8_t* and a raw stream handle so this header can be
// included from host-only translation units (the pybind bindings) without
// pulling in CUDA device types. The definitions in spark_kernels.cu must use
// exactly these types, or the two translation units disagree on the mangled
// name and the extension fails to link.

#pragma once

#include <cstdint>

struct CUstream_st;
using cudaStream_t_ = CUstream_st*;

namespace flash_rt::spark_x25 {

// `gate` and `up` may be column slices of one wider buffer (the fused gate+up
// GEMM writes [gate ; up] along N), so each takes its own leading-dimension
// stride in elements rather than assuming contiguity.
void gelu_mul_to_nvfp4_swizzled_bf16(
    const void* gate, const void* up,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, int gate_ld, int up_ld, CUstream_st* stream);

// `k8`/`v8` and their per-(row, KV head) scales are an optional E4M3 mirror of
// the bf16 cache that decode attention reads; pass null to skip it (sliding
// layers, whose decode range is the window and whose bytes are already small).
void qkv_post_rope_kvwrite_bf16(
    const void* qkv,
    const float* cos_tab, const float* sin_tab,
    void* q_buf, void* k_cache, void* v_cache,
    void* k_ring, void* v_ring,
    void* k8, void* v8, void* k8_scale, void* v8_scale,
    int rows, int q_heads, int kv_heads, int head_dim, int rope_dim,
    const int* pos_dev, int ring_w, int lin_w, CUstream_st* stream);

// Bit-identical to FlashRT's generic bf16 matmul on this shape, but with the
// inner loop unrolled so the loads pipeline instead of serialising.
void gproj_bf16(const void* x, const void* W, void* out,
                int rows, int g_dim, int K, CUstream_st* stream);

// `lin_w` is the wrap width of the linear cache: 0 keeps absolute indexing
// (a cache that spans the context), `prefill_chunk + W` makes it a mirrored
// ring whose one period holds any single prefill window.
//
// Copy the surviving W-row window from a sliding layer's linear cache into its
// mirrored ring. Rows are written one per block so no two blocks alias a slot.
void seed_ring_bf16(const void* k_lin, const void* v_lin,
                    void* k_ring, void* v_ring,
                    int base_pos, int count, int kv_dim, int W, int lin_w,
                    CUstream_st* stream);

// Write a scalar into device memory (no framework op in the pipeline).
void set_int32(int* dst, int value, CUstream_st* stream);

// Device-side advance of the decode position and the two attention lengths.
void step_positions_bf16(int* pos, int* full_klen, int* slide_klen,
                         int64_t* tokens_out, int64_t* token_in,
                         int W, int first, CUstream_st* stream);

// Per-head sigmoid gate on the attention output (bf16 result only).
void attn_out_gate_bf16(const void* attn, const void* gate, void* out,
                        int rows, int heads, int head_dim, CUstream_st* stream);

// Fused per-head sigmoid gate + NVFP4 pack. `out` may be null when the bf16
// result is not needed.
void attn_out_gate_to_nvfp4_bf16(const void* attn, const void* gate, void* out,
                                 uint8_t* packed, uint8_t* sf_swz,
                                 int rows, int heads, int head_dim,
                                 CUstream_st* stream);

// Fused residual add + RMSNorm + NVFP4 (swizzled SF) for a single decode row,
// byte-compatible with FlashRT's v2 kernel. Decode only: prefill keeps the
// FlashRT entry. Requires cols % 8 == 0.
void residual_add_rms_norm_to_nvfp4_bf16(
    const void* h_in, const void* attn_proj, void* h_post,
    const void* rms_weight, void* packed, void* sf_swz,
    int cols, float eps, CUstream_st* stream);

// Greedy sample over one row of logits, lower index winning ties. Same result
// as FlashRT's qwen36_argmax_bf16 but with 16-byte loads behind an unroll, so
// the scan has more than one load in flight per warp.
void argmax_bf16(const void* logits, void* argmax_out, int vocab,
                 CUstream_st* stream);


void attn_state_init_bf16(void* row_max, void* row_sum, int q_heads,
                          CUstream_st* stream);
// `kv8 != 0` reads an E4M3 K cache with `k8_scale` (slots, kv_heads), which is
// what halves the KV bytes; 0 keeps bf16.
void attn_scores_bf16(const void* q, const void* k_cache, const void* k8_scale,
                      void* s_out, void* row_max, const int* klen_dev, int cap,
                      int s_stride, int q_heads, int kv_heads, int head_dim,
                      int group, float scale, int kv8, CUstream_st* stream);
void attn_softmax_bf16(const void* s_in, void* p_out, const void* row_max,
                       void* row_sum, const int* klen_dev, int cap, int stride,
                       int q_heads, int nchunk, CUstream_st* stream);
void attn_pv_bf16(const void* v_cache, const void* v8_scale, const void* p_in,
                  void* o_part, const int* klen_dev, int stride, int q_heads,
                  int kv_heads, int head_dim, int group, int nsplit, int kv8,
                  CUstream_st* stream);
void attn_pv_combine_bf16(const void* o_part, const void* row_sum, void* o_out,
                          int nsplit, int q_heads, int head_dim,
                          CUstream_st* stream);


// ── E4M3 KV -> bf16 staging (long-window prefill) ────────────────────────
void kv_dequant_bf16(const void* src, const void* scale, void* dst, int count,
                     int kv_heads, int head_dim, CUstream_st* stream);
}  // namespace flash_rt::spark_x25
