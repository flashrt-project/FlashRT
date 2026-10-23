// SPDX-License-Identifier: Apache-2.0
//
// pybind surface for the Flint model kernels (module flash_rt_flint). Pointers
// and streams are passed as integers, matching the FlashRT kernel library's
// convention, so the runtime can hand stable raw pointers to captured graphs.

#include "kernels/flint/flint_kernels.cuh"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace fl = flash_rt::flint;

namespace {

inline cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
template <typename T>
inline T* ptr(uintptr_t p) { return reinterpret_cast<T*>(p); }

void check_launch(const char* op) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}

}  // namespace

PYBIND11_MODULE(flash_rt_flint, m) {
  m.doc() = "Flint (SM120) W4A16 NVFP4 kernels: repack, decode GEMV/MoE/attention, routing";

  m.attr("ARGMAX_BLOCKS") = fl::kArgmaxBlocks;

  m.def("repack_nvfp4",
        [](uintptr_t src, long long rows, int K, uintptr_t codes, uintptr_t scales, long long group_rows,
           long long group_stride, long long row_offset, uintptr_t stream) {
          if (K % 64 != 0) throw std::invalid_argument("repack_nvfp4: K must be a multiple of 64");
          fl::repack_nvfp4(ptr<const void>(src), rows, K, ptr<void>(codes), ptr<void>(scales), group_rows,
                           group_stride, row_offset, to_stream(stream));
          check_launch("repack_nvfp4");
        });

  m.def("embed_at_pos_bf16", [](uintptr_t tokens, uintptr_t pos, uintptr_t table, uintptr_t out, uintptr_t stream) {
    fl::embed_at_pos_bf16(ptr<const int>(tokens), ptr<const int>(pos), ptr<const void>(table), ptr<void>(out),
                          to_stream(stream));
          check_launch("embed_at_pos_bf16");
        });

  m.def("add_rmsnorm_bf16",
        [](uintptr_t x, uintptr_t residual, uintptr_t w, uintptr_t out, int rows, float eps, int first,
           uintptr_t stream, int write_res) {
          fl::add_rmsnorm_bf16(ptr<const void>(x), ptr<void>(residual), ptr<const float>(w), ptr<void>(out),
                               rows, eps, first, to_stream(stream), write_res);
          check_launch("add_rmsnorm_bf16");
        },
        py::arg("x"), py::arg("residual"), py::arg("w"), py::arg("out"), py::arg("rows"), py::arg("eps"),
        py::arg("first"), py::arg("stream"), py::arg("write_res") = 1);

  m.def("gemv_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t out, int N, int K, float gs,
           int epilogue, uintptr_t stream) {
          if (fl::gemv_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                             ptr<void>(out), N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_w4a16: unsupported K=" + std::to_string(K) +
                                        " epilogue=" + std::to_string(epilogue));
          check_launch("gemv_w4a16");
        });

  m.def("qk_norm_rope_kv_bf16",
        [](uintptr_t qkv, uintptr_t qw, uintptr_t kw, uintptr_t cos_t, uintptr_t sin_t, uintptr_t pos,
           uintptr_t q_out, uintptr_t cache, int ring, int rope, float eps, uintptr_t stream) {
          if (ring & (ring - 1)) throw std::invalid_argument("qk_norm_rope_kv_bf16: ring must be a power of two");
          fl::qk_norm_rope_kv_bf16(ptr<const void>(qkv), ptr<const float>(qw), ptr<const float>(kw),
                                   ptr<const float>(cos_t), ptr<const float>(sin_t), ptr<const int>(pos),
                                   ptr<void>(q_out), ptr<void>(cache), ring, rope, eps, to_stream(stream));
          check_launch("qk_norm_rope_kv_bf16");
        });

  m.def("decode_attention_bf16",
        [](uintptr_t q, uintptr_t cache, uintptr_t pos, int ring, int window, uintptr_t sinks, float scale,
           int n_split, uintptr_t o_part, uintptr_t ml_part, uintptr_t out, uintptr_t stream) {
          if (ring & (ring - 1)) throw std::invalid_argument("decode_attention_bf16: ring must be a power of two");
          if (n_split < 1 || n_split > 1024) throw std::invalid_argument("decode_attention_bf16: bad n_split");
          fl::decode_attention_bf16(ptr<const void>(q), ptr<const void>(cache), ptr<const int>(pos), ring,
                                    window, ptr<const float>(sinks), scale, n_split, ptr<float>(o_part),
                                    ptr<float>(ml_part), ptr<void>(out), to_stream(stream));
          check_launch("decode_attention_bf16");
        });

  m.def("router_topk_bf16",
        [](uintptr_t xn, uintptr_t wr, uintptr_t logits, uintptr_t counter, uintptr_t ids, uintptr_t tw,
           uintptr_t stream) {
          fl::router_topk_bf16(ptr<const void>(xn), ptr<const void>(wr), ptr<float>(logits),
                               ptr<unsigned>(counter), ptr<int>(ids), ptr<float>(tw), to_stream(stream));
          check_launch("router_topk_bf16");
        });

  m.def("moe_gate_up_bf16",
        [](uintptr_t xn, uintptr_t ids, uintptr_t ecodes, uintptr_t escales, uintptr_t egs, uintptr_t scodes,
           uintptr_t sscales, float sgs, uintptr_t act, uintptr_t stream) {
          fl::moe_gate_up_bf16(ptr<const void>(xn), ptr<const int>(ids), ptr<const void>(ecodes),
                               ptr<const void>(escales), ptr<const float>(egs), ptr<const void>(scodes),
                               ptr<const void>(sscales), sgs, ptr<void>(act), to_stream(stream));
          check_launch("moe_gate_up_bf16");
        });

  m.def("moe_down_combine_bf16",
        [](uintptr_t act, uintptr_t ids, uintptr_t tw, uintptr_t ecodes, uintptr_t escales, uintptr_t egs,
           uintptr_t scodes, uintptr_t sscales, float sgs, uintptr_t out, uintptr_t stream) {
          fl::moe_down_combine_bf16(ptr<const void>(act), ptr<const int>(ids), ptr<const float>(tw),
                                    ptr<const void>(ecodes), ptr<const void>(escales), ptr<const float>(egs),
                                    ptr<const void>(scodes), ptr<const void>(sscales), sgs, ptr<void>(out),
                                    to_stream(stream));
          check_launch("moe_down_combine_bf16");
        });

  m.def("argmax_next_f32",
        [](uintptr_t logits, int n, uintptr_t cand_v, uintptr_t cand_i, uintptr_t counter, uintptr_t tokens,
           uintptr_t pos, uintptr_t n_prompt, uintptr_t stream) {
          fl::argmax_next_f32(ptr<const float>(logits), n, ptr<float>(cand_v), ptr<int>(cand_i),
                              ptr<unsigned>(counter), ptr<int>(tokens), ptr<int>(pos), ptr<const int>(n_prompt),
                              to_stream(stream));
          check_launch("argmax_next_f32");
        });

  m.attr("EPI_BF16") = fl::kEpiBf16;
  m.attr("EPI_F32") = fl::kEpiF32;
  m.attr("EPI_SWIGLU_DENSE") = fl::kEpiSwigluDense;
  m.attr("EPI_SWIGLU_ROUTED") = fl::kEpiSwigluRouted;

  m.def("embed_rows_bf16", [](uintptr_t tokens, uintptr_t pos, int n, uintptr_t table, uintptr_t out, uintptr_t stream) {
    fl::embed_rows_bf16(ptr<const int>(tokens), ptr<const int>(pos), n, ptr<const void>(table), ptr<void>(out),
                        to_stream(stream));
          check_launch("embed_rows_bf16");
        });

  m.def("qk_norm_rope_kv_rows_bf16",
        [](uintptr_t qkv, int n, uintptr_t qw, uintptr_t kw, uintptr_t cos_t, uintptr_t sin_t, uintptr_t pos,
           uintptr_t q_out, uintptr_t cache, int ring, int rope, float eps, uintptr_t stream) {
          if (ring & (ring - 1)) throw std::invalid_argument("qk_norm_rope_kv_rows_bf16: ring must be a power of two");
          fl::qk_norm_rope_kv_rows_bf16(ptr<const void>(qkv), n, ptr<const float>(qw), ptr<const float>(kw),
                                        ptr<const float>(cos_t), ptr<const float>(sin_t), ptr<const int>(pos),
                                        ptr<void>(q_out), ptr<void>(cache), ring, rope, eps, to_stream(stream));
          check_launch("qk_norm_rope_kv_rows_bf16");
        });

  m.def("prefill_attention_bf16",
        [](uintptr_t q, int n, uintptr_t cache, uintptr_t pos, int ring, int window, uintptr_t sinks, float scale,
           uintptr_t out, int n_split, uintptr_t o_part, uintptr_t ml_part, uintptr_t counters, int variant,
           uintptr_t stream) {
          if (ring & (ring - 1)) throw std::invalid_argument("prefill_attention_bf16: ring must be a power of two");
          if (fl::prefill_attention_bf16(ptr<const void>(q), n, ptr<const void>(cache), ptr<const int>(pos), ring,
                                          window, ptr<const float>(sinks), scale, ptr<void>(out), n_split,
                                          ptr<float>(o_part), ptr<float>(ml_part), ptr<unsigned>(counters), variant,
                                          to_stream(stream)) != 0)
            throw std::invalid_argument("prefill_attention_bf16: bad variant/split arguments");
          check_launch("prefill_attention_bf16");
        });

  m.def("gemm_w4a16_qkrope",
        [](uintptr_t A, int M, int lda, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t q_norm,
           uintptr_t k_norm, uintptr_t cos_t, uintptr_t sin_t, uintptr_t pos, uintptr_t q_out, uintptr_t cache, int ring,
           int rope, float eps, uintptr_t stream) {
          if (fl::gemm_w4a16_qkrope(ptr<const void>(A), M, lda, ptr<const void>(codes), ptr<const void>(scales), N, K,
                                    gs, ptr<const float>(q_norm), ptr<const float>(k_norm), ptr<const float>(cos_t),
                                    ptr<const float>(sin_t), ptr<const int>(pos), ptr<void>(q_out), ptr<void>(cache),
                                    ring, rope, eps, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_qkrope: unsupported N=" + std::to_string(N) + " K=" +
                                        std::to_string(K));
          check_launch("gemm_w4a16_qkrope");
        });
  m.def("gemm_w4a16",
        [](uintptr_t A, int M, int lda, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C, int ldc,
           int epilogue, uintptr_t stream) {
          if (fl::gemm_w4a16(ptr<const void>(A), M, lda, ptr<const void>(codes), ptr<const void>(scales), N, K, gs,
                             ptr<void>(C), ldc, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16: unsupported N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                        " epilogue=" + std::to_string(epilogue));
          check_launch("gemm_w4a16");
        });

  m.def("gemm_w4a16_grouped",
        [](uintptr_t A, uintptr_t gather, int lda, uintptr_t tiles, uintptr_t n_tiles, int max_tiles, uintptr_t codes,
           uintptr_t scales, int N, int K, uintptr_t gs_e, uintptr_t C, int ldc, int epilogue, int tile_rows,
           int mean_rows, uintptr_t stream) {
          if (fl::gemm_w4a16_grouped(ptr<const void>(A), ptr<const int>(gather), lda, ptr<const int>(tiles),
                                     ptr<const int>(n_tiles), max_tiles, ptr<const void>(codes), ptr<const void>(scales),
                                     N, K, ptr<const float>(gs_e), ptr<void>(C), ldc, epilogue, tile_rows,
                                     mean_rows, to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_grouped: unsupported N=" + std::to_string(N) + " K=" +
                                        std::to_string(K) + " epilogue=" + std::to_string(epilogue) +
                                        " tile_rows=" + std::to_string(tile_rows));
          check_launch("gemm_w4a16_grouped");
        });

  m.def("gemm_w4a16_cfg",
        [](uintptr_t A, int M, int lda, uintptr_t codes, uintptr_t scales, int N, int K, float gs, uintptr_t C, int ldc,
           int epilogue, int bm, int stages, int split, int kb, uintptr_t stream) {
          if (fl::gemm_w4a16_cfg(ptr<const void>(A), M, lda, ptr<const void>(codes), ptr<const void>(scales), N, K, gs,
                                 ptr<void>(C), ldc, epilogue, fl::GemmConfig{bm, stages, split, kb},
                                 to_stream(stream)) != 0)
            throw std::invalid_argument("gemm_w4a16_cfg: unsupported configuration");
          check_launch("gemm_w4a16_cfg");
        });

  m.def("set_gemm_override", [](int bm, int stages, int split, int kb, int gbm, int gstages, int gsplit, int gkb) {
    fl::set_gemm_override(fl::GemmConfig{bm, stages, split, kb}, fl::GemmConfig{gbm, gstages, gsplit, gkb});
  });

  m.def("router_topk_rows_bf16",
        [](uintptr_t xn, int n, uintptr_t wr, uintptr_t ids, uintptr_t tw, uintptr_t counts, uintptr_t pos,
           uintptr_t n_end, uintptr_t stream) {
          fl::router_topk_rows_bf16(ptr<const void>(xn), n, ptr<const void>(wr), ptr<int>(ids), ptr<float>(tw),
                                    ptr<int>(counts), ptr<const int>(pos), ptr<const int>(n_end), to_stream(stream));
          check_launch("router_topk_rows_bf16");
        });

  m.def("moe_scan_place",
        [](uintptr_t ids, int n, int tile_rows, uintptr_t counts, uintptr_t sorted_tok, uintptr_t inv, uintptr_t tiles,
           uintptr_t n_tiles, uintptr_t pos, uintptr_t n_end, uintptr_t stream) {
          fl::moe_scan_place(ptr<const int>(ids), n, tile_rows, ptr<int>(counts), ptr<int>(sorted_tok), ptr<int>(inv),
                             ptr<int>(tiles), ptr<int>(n_tiles), ptr<const int>(pos), ptr<const int>(n_end),
                             to_stream(stream));
          check_launch("moe_scan_place");
        });

  m.def("moe_combine_rows_bf16",
        [](uintptr_t y, uintptr_t inv, uintptr_t tw, uintptr_t shared, uintptr_t out, int n, uintptr_t stream) {
          fl::moe_combine_rows_bf16(ptr<const void>(y), ptr<const int>(inv), ptr<const float>(tw),
                                    ptr<const void>(shared), ptr<void>(out), n, to_stream(stream));
          check_launch("moe_combine_rows_bf16");
        });

  m.def("moe_combine_norm_rows_bf16",
        [](uintptr_t y, uintptr_t inv, uintptr_t tw, uintptr_t shared, uintptr_t residual, uintptr_t w, uintptr_t out,
           int n, float eps, uintptr_t stream) {
          fl::moe_combine_norm_rows_bf16(ptr<const void>(y), ptr<const int>(inv), ptr<const float>(tw),
                                         ptr<const void>(shared), ptr<void>(residual), ptr<const float>(w),
                                         ptr<void>(out), n, eps, to_stream(stream));
          check_launch("moe_combine_norm_rows_bf16");
        });

  m.def("advance_pos", [](uintptr_t pos, int n, uintptr_t stream) { fl::advance_pos(ptr<int>(pos), n, to_stream(stream));
          check_launch("advance_pos");
        });

  // ---- fused decode step (PDL) ----
  m.def("gemv_fused_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t res_in, uintptr_t res_out, uintptr_t norm_w,
           int first, float eps, uintptr_t out, int N, int K, float gs, int epilogue, uintptr_t stream) {
          if (fl::gemv_fused_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                              ptr<const void>(res_in), ptr<void>(res_out), ptr<const float>(norm_w), first, eps,
                              ptr<void>(out), N, K, gs, epilogue, to_stream(stream)) != 0)
            throw std::invalid_argument("gemv_fused_w4a16: unsupported K=" + std::to_string(K) + " epilogue=" +
                                        std::to_string(epilogue) + (norm_w ? " (fused norm)" : ""));
          check_launch("gemv_fused_w4a16");
        });
  m.def("add_rmsnorm_pdl_bf16",
        [](uintptr_t x, uintptr_t res_in, uintptr_t w, float eps, uintptr_t res_out, uintptr_t out, uintptr_t stream) {
          fl::add_rmsnorm_pdl_bf16(ptr<const void>(x), ptr<const void>(res_in), ptr<const float>(w), eps, ptr<void>(res_out),
                                ptr<void>(out), to_stream(stream));
          check_launch("add_rmsnorm_pdl_bf16");
        });
  m.def("lm_head_greedy_w4a16",
        [](uintptr_t codes, uintptr_t scales, uintptr_t x, uintptr_t res_in, uintptr_t res_out, uintptr_t norm_w,
           float eps, uintptr_t logits, int N, float gs, uintptr_t cand_v, uintptr_t cand_i, int cand_cap,
           uintptr_t counter, uintptr_t tokens, uintptr_t pos, uintptr_t n_prompt, uintptr_t stream) {
          if (fl::lm_head_greedy_w4a16(ptr<const void>(codes), ptr<const void>(scales), ptr<const void>(x),
                                       ptr<const void>(res_in), ptr<void>(res_out), ptr<const float>(norm_w), eps,
                                       ptr<float>(logits), N, gs, ptr<float>(cand_v), ptr<int>(cand_i), cand_cap,
                                       ptr<unsigned>(counter), ptr<int>(tokens), ptr<int>(pos),
                                       ptr<const int>(n_prompt), to_stream(stream)) != 0)
            throw std::invalid_argument("lm_head_greedy_w4a16: unsupported rows per warp or candidate buffer too small");
          check_launch("lm_head_greedy_w4a16");
        });
  m.def("router_norm_bf16",
        [](uintptr_t x, uintptr_t res_in, uintptr_t nw, float eps, uintptr_t res_out, uintptr_t xn_out, uintptr_t wr,
           uintptr_t logits, uintptr_t counter, uintptr_t ids, uintptr_t tw, uintptr_t stream) {
          fl::router_norm_bf16(ptr<const void>(x), ptr<const void>(res_in), ptr<const float>(nw), eps, ptr<void>(res_out),
                               ptr<void>(xn_out), ptr<const void>(wr), ptr<float>(logits), ptr<unsigned>(counter),
                               ptr<int>(ids), ptr<float>(tw), to_stream(stream));
          check_launch("router_norm_bf16");
        });
  m.def("moe_gate_up_pdl_bf16",
        [](uintptr_t xn, uintptr_t ids, uintptr_t ecodes, uintptr_t escales, uintptr_t egs, uintptr_t scodes,
           uintptr_t sscales, float sgs, uintptr_t act, uintptr_t stream) {
          fl::moe_gate_up_pdl_bf16(ptr<const void>(xn), ptr<const int>(ids), ptr<const void>(ecodes), ptr<const void>(escales),
                                ptr<const float>(egs), ptr<const void>(scodes), ptr<const void>(sscales), sgs,
                                ptr<void>(act), to_stream(stream));
          check_launch("moe_gate_up_pdl_bf16");
        });
  m.def("moe_down_combine_pdl_bf16",
        [](uintptr_t act, uintptr_t ids, uintptr_t tw, uintptr_t ecodes, uintptr_t escales, uintptr_t egs,
           uintptr_t scodes, uintptr_t sscales, float sgs, uintptr_t out, uintptr_t stream) {
          fl::moe_down_combine_pdl_bf16(ptr<const void>(act), ptr<const int>(ids), ptr<const float>(tw),
                                     ptr<const void>(ecodes), ptr<const void>(escales), ptr<const float>(egs),
                                     ptr<const void>(scodes), ptr<const void>(sscales), sgs, ptr<void>(out),
                                     to_stream(stream));
          check_launch("moe_down_combine_pdl_bf16");
        });
  m.def("embed_pdl_bf16", [](uintptr_t tokens, uintptr_t pos, uintptr_t table, uintptr_t out, uintptr_t stream) {
    fl::embed_pdl_bf16(ptr<const int>(tokens), ptr<const int>(pos), ptr<const void>(table), ptr<void>(out), to_stream(stream));
          check_launch("embed_pdl_bf16");
        });
  m.def("decode_attention_fused_bf16",
        [](uintptr_t qkv, uintptr_t qw, uintptr_t kw, uintptr_t cos_t, uintptr_t sin_t, uintptr_t pos, uintptr_t cache,
           int ring, int window, int rope, float eps, uintptr_t sinks, float scale, int n_split, uintptr_t o_part,
           uintptr_t ml_part, uintptr_t counters, uintptr_t out, uintptr_t stream, int warps, uintptr_t pf_ranges,
           int n_pf, int kv_slots) {
          if (ring & (ring - 1)) throw std::invalid_argument("decode_attention_fused_bf16: ring must be a power of two");
          if (n_split < 1 || n_split > 1024) throw std::invalid_argument("decode_attention_fused_bf16: bad n_split");
          if (fl::decode_attention_fused_bf16(ptr<const void>(qkv), ptr<const float>(qw), ptr<const float>(kw),
                                              ptr<const float>(cos_t), ptr<const float>(sin_t), ptr<const int>(pos),
                                              ptr<void>(cache), ring, window, rope, eps, ptr<const float>(sinks), scale,
                                              n_split, warps, ptr<float>(o_part), ptr<float>(ml_part),
                                              ptr<unsigned>(counters), ptr<void>(out), to_stream(stream),
                                              ptr<const unsigned long long>(pf_ranges), n_pf, kv_slots) != 0)
            throw std::invalid_argument("decode_attention_fused_bf16: warps must be 4, 8 or 16");
          check_launch("decode_attention_fused_bf16");
        },
        py::arg("qkv"), py::arg("qw"), py::arg("kw"), py::arg("cos_t"), py::arg("sin_t"), py::arg("pos"), py::arg("cache"),
        py::arg("ring"), py::arg("window"), py::arg("rope"), py::arg("eps"), py::arg("sinks"), py::arg("scale"),
        py::arg("n_split"), py::arg("o_part"), py::arg("ml_part"), py::arg("counters"), py::arg("out"), py::arg("stream"),
        py::arg("warps") = 4, py::arg("pf_ranges") = 0, py::arg("n_pf") = 0, py::arg("kv_slots") = 0);

  m.def("set_moe_launch_config", [](int rw, int rr, int gw, int gr, int dw, int dr) {
    fl::set_moe_launch_config(fl::MoeLaunchConfig{rw, rr, gw, gr, dw, dr});
  });
  m.def("get_moe_launch_config", []() {
    const auto c = fl::get_moe_launch_config();
    return py::make_tuple(c.router_warps, c.router_rows, c.gu_warps, c.gu_rows, c.dn_warps, c.dn_rows);
  });

  m.def("set_lm_head_rows", [](int rows) { fl::set_lm_head_rows(rows); });
  m.def("set_gemv_launch_config", [](int qr, int qw, int orr, int ow) {
    fl::set_gemv_launch_config(fl::GemvLaunchConfig{qr, qw, orr, ow});
  });
}
