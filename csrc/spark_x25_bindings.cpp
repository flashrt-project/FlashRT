// SPDX-License-Identifier: Apache-2.0
//
// pybind surface for the Spark-X2.5 model kernels. Pointers are passed as
// integers and streams as integers, matching the FlashRT kernel library's
// convention, so the runtime can hand out stable raw pointers captured into a
// CUDA Graph.

#include "kernels/spark_x25/spark_x25_kernels.cuh"

#include <cstdint>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// The header deliberately avoids CUDA types, so the stream is spelled
// with its underlying tag type here rather than cudaStream_t (same type).
static inline CUstream_st* to_stream(uintptr_t s) {
  return reinterpret_cast<CUstream_st*>(s);
}

PYBIND11_MODULE(flash_rt_sparkx25, m) {
  m.doc() = "Spark-X2.5 decode kernels (partial RoPE + KV write, GELU-mul NVFP4 "
            "quantize, per-head attention output gate)";

  m.def("gelu_mul_to_nvfp4_swizzled_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t packed, uintptr_t sf_swz,
           int rows, int cols, int gate_ld, int up_ld, uintptr_t stream) {
          flash_rt::spark_x25::gelu_mul_to_nvfp4_swizzled_bf16(
              reinterpret_cast<const void*>(gate),
              reinterpret_cast<const void*>(up),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sf_swz),
              rows, cols, gate_ld, up_ld, to_stream(stream));
        });

  m.def("qkv_post_rope_kvwrite_bf16",
        [](uintptr_t qkv, uintptr_t cos_tab, uintptr_t sin_tab,
           uintptr_t q_buf, uintptr_t k_cache, uintptr_t v_cache,
           uintptr_t k_ring, uintptr_t v_ring,
           uintptr_t k8, uintptr_t v8, uintptr_t k8_scale, uintptr_t v8_scale,
           int rows, int q_heads, int kv_heads, int head_dim, int rope_dim,
           uintptr_t pos_dev, int ring_w, int lin_w, uintptr_t stream) {
          flash_rt::spark_x25::qkv_post_rope_kvwrite_bf16(
              reinterpret_cast<const void*>(qkv),
              reinterpret_cast<const float*>(cos_tab),
              reinterpret_cast<const float*>(sin_tab),
              reinterpret_cast<void*>(q_buf),
              reinterpret_cast<void*>(k_cache),
              reinterpret_cast<void*>(v_cache),
              reinterpret_cast<void*>(k_ring),
              reinterpret_cast<void*>(v_ring),
              reinterpret_cast<void*>(k8), reinterpret_cast<void*>(v8),
              reinterpret_cast<void*>(k8_scale),
              reinterpret_cast<void*>(v8_scale),
              rows, q_heads, kv_heads, head_dim, rope_dim,
              reinterpret_cast<const int*>(pos_dev), ring_w, lin_w,
              to_stream(stream));
        });

  m.def("seed_ring_bf16",
        [](uintptr_t k_lin, uintptr_t v_lin, uintptr_t k_ring, uintptr_t v_ring,
           int base_pos, int count, int kv_dim, int W, int lin_w,
           uintptr_t stream) {
          flash_rt::spark_x25::seed_ring_bf16(
              reinterpret_cast<const void*>(k_lin), reinterpret_cast<const void*>(v_lin),
              reinterpret_cast<void*>(k_ring), reinterpret_cast<void*>(v_ring),
              base_pos, count, kv_dim, W, lin_w, to_stream(stream));
        });

  m.def("step_positions_bf16",
        [](uintptr_t pos, uintptr_t full_klen, uintptr_t slide_klen,
           uintptr_t tokens_out, uintptr_t token_in,
           int W, int first, uintptr_t stream) {
          flash_rt::spark_x25::step_positions_bf16(
              reinterpret_cast<int*>(pos), reinterpret_cast<int*>(full_klen),
              reinterpret_cast<int*>(slide_klen),
              reinterpret_cast<int64_t*>(tokens_out),
              reinterpret_cast<int64_t*>(token_in), W, first, to_stream(stream));
        });

  m.def("attn_out_gate_to_nvfp4_bf16",
        [](uintptr_t attn, uintptr_t gate, uintptr_t out,
           uintptr_t packed, uintptr_t sf_swz,
           int rows, int heads, int head_dim, uintptr_t stream) {
          flash_rt::spark_x25::attn_out_gate_to_nvfp4_bf16(
              reinterpret_cast<const void*>(attn),
              reinterpret_cast<const void*>(gate),
              reinterpret_cast<void*>(out),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sf_swz),
              rows, heads, head_dim, to_stream(stream));
        });

  m.def("set_int32",
        [](uintptr_t dst, int value, uintptr_t stream) {
          flash_rt::spark_x25::set_int32(reinterpret_cast<int*>(dst), value, to_stream(stream));
        });

  m.def("gproj_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out,
           int rows, int g_dim, int K, uintptr_t stream) {
          flash_rt::spark_x25::gproj_bf16(
              reinterpret_cast<const void*>(x),
              reinterpret_cast<const void*>(W),
              reinterpret_cast<void*>(out),
              rows, g_dim, K, to_stream(stream));
        });

  m.def("attn_out_gate_bf16",
        [](uintptr_t attn, uintptr_t gate, uintptr_t out,
           int rows, int heads, int head_dim, uintptr_t stream) {
          flash_rt::spark_x25::attn_out_gate_bf16(
              reinterpret_cast<const void*>(attn),
              reinterpret_cast<const void*>(gate),
              reinterpret_cast<void*>(out),
              rows, heads, head_dim, to_stream(stream));
        });

  m.def("residual_add_rms_norm_to_nvfp4_bf16",
        [](uintptr_t h_in, uintptr_t attn_proj, uintptr_t h_post,
           uintptr_t rms_weight, uintptr_t packed, uintptr_t sf_swz,
           int cols, float eps, uintptr_t stream) {
          flash_rt::spark_x25::residual_add_rms_norm_to_nvfp4_bf16(
              reinterpret_cast<const void*>(h_in),
              reinterpret_cast<const void*>(attn_proj),
              reinterpret_cast<void*>(h_post),
              reinterpret_cast<const void*>(rms_weight),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sf_swz),
              cols, eps, to_stream(stream));
        });

  m.def("argmax_bf16",
        [](uintptr_t logits, uintptr_t argmax_out, int vocab, uintptr_t stream) {
          flash_rt::spark_x25::argmax_bf16(
              reinterpret_cast<const void*>(logits),
              reinterpret_cast<void*>(argmax_out),
              vocab, to_stream(stream));
        });


  m.def("attn_state_init_bf16",
        [](uintptr_t row_max, uintptr_t row_sum, int q_heads, uintptr_t stream) {
          flash_rt::spark_x25::attn_state_init_bf16(reinterpret_cast<void*>(row_max),
                                          reinterpret_cast<void*>(row_sum),
                                          q_heads, to_stream(stream));
        });
  m.def("attn_scores_bf16",
        [](uintptr_t q, uintptr_t k_cache, uintptr_t k8_scale, uintptr_t s_out,
           uintptr_t row_max, uintptr_t klen_dev, int cap, int s_stride,
           int q_heads, int kv_heads, int head_dim, int group, double scale,
           int kv8, uintptr_t stream) {
          flash_rt::spark_x25::attn_scores_bf16(
              reinterpret_cast<const void*>(q),
              reinterpret_cast<const void*>(k_cache),
              reinterpret_cast<const void*>(k8_scale),
              reinterpret_cast<float*>(s_out), reinterpret_cast<void*>(row_max),
              reinterpret_cast<const int*>(klen_dev), cap, s_stride, q_heads,
              kv_heads, head_dim, group, (float)scale, kv8,
              to_stream(stream));
        });
  m.def("attn_softmax_bf16",
        [](uintptr_t s_in, uintptr_t p_out, uintptr_t row_max, uintptr_t row_sum,
           uintptr_t klen_dev, int cap, int stride, int q_heads, int nchunk,
           uintptr_t stream) {
          flash_rt::spark_x25::attn_softmax_bf16(
              reinterpret_cast<const void*>(s_in),
              reinterpret_cast<float*>(p_out),
              reinterpret_cast<const void*>(row_max),
              reinterpret_cast<void*>(row_sum),
              reinterpret_cast<const int*>(klen_dev), cap, stride, q_heads,
              nchunk, to_stream(stream));
        });
  m.def("attn_pv_bf16",
        [](uintptr_t v_cache, uintptr_t v8_scale, uintptr_t p_in,
           uintptr_t o_part, uintptr_t klen_dev, int stride, int q_heads,
           int kv_heads, int head_dim, int group, int nsplit, int kv8,
           uintptr_t stream) {
          flash_rt::spark_x25::attn_pv_bf16(
              reinterpret_cast<const void*>(v_cache),
              reinterpret_cast<const void*>(v8_scale),
              reinterpret_cast<const float*>(p_in),
              reinterpret_cast<float*>(o_part),
              reinterpret_cast<const int*>(klen_dev), stride, q_heads, kv_heads,
              head_dim, group, nsplit, kv8, to_stream(stream));
        });
  m.def("attn_pv_combine_bf16",
        [](uintptr_t o_part, uintptr_t row_sum, uintptr_t o_out, int nsplit,
           int q_heads, int head_dim, uintptr_t stream) {
          flash_rt::spark_x25::attn_pv_combine_bf16(
              reinterpret_cast<const float*>(o_part),
              reinterpret_cast<const float*>(row_sum),
              reinterpret_cast<void*>(o_out), nsplit, q_heads,
              head_dim, to_stream(stream));
        });

  m.def("kv_dequant_bf16",
        [](uintptr_t src, uintptr_t scale, uintptr_t dst, int count,
           int kv_heads, int head_dim, uintptr_t stream) {
          flash_rt::spark_x25::kv_dequant_bf16(
              reinterpret_cast<const void*>(src),
              reinterpret_cast<const void*>(scale),
              reinterpret_cast<void*>(dst), count, kv_heads, head_dim,
              to_stream(stream));
        });
}
