// ============================================================================
//  FlashRT — slim pybind module for the LingBot FA2 entries.
//
//  The generic flash_rt_fa2 module exposes fp16, causal, window, seqused,
//  split-KV and every head-dimension/tile variant. LingBot's native
//  transformer calls only fa2.fwd_bf16 (S_q > 64) and fa2.fwd_bf16_tile
//  (S_q <= 64). This module binds exactly those two C entries from
//  fa2_wrapper_lingbot.cu; the unused FA2 kernels do not enter the fatbin.
//
//  The stage-2 decoder stack (and the rest of FlashRT) keeps using the
//  generic csrc/fa2_bindings.cpp target.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "attention/fa2_wrapper.h"

namespace py = pybind11;

static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

static const char* kDocstring = R"(FlashAttention-2 fwd (LingBot slim build).

Args:
    Q, K, V, O: int device pointers. Q is (batch, seqlen_q, num_heads_q, head_dim);
      K/V are (batch, seqlen_k, num_heads_kv, head_dim); O has Q's shape.
    softmax_lse: int device pointer, fp32, shape (batch, num_heads_q, seqlen_q).
    softmax_lse_accum, o_accum: splitkv scratch. The slim build supports only
      num_splits=1; pass 0 for both.
    *_strides: 3-tuple (batch, row, head) in elements (matches .stride()).
    softmax_scale: typically 1.0 / sqrt(head_dim).
    num_sms: current device's SM count (kept for signature compatibility).
    stream: CUDA stream (int handle; 0 = default stream).
)";

template <typename Fn>
static auto make_fwd(Fn fn) {
    return [fn](uintptr_t Q, uintptr_t K, uintptr_t V,
                uintptr_t O, uintptr_t softmax_lse,
                uintptr_t softmax_lse_accum, uintptr_t o_accum,
                int batch, int seqlen_q, int seqlen_k,
                int num_heads_q, int num_heads_kv, int head_dim,
                py::tuple q_strides, py::tuple k_strides,
                py::tuple v_strides, py::tuple o_strides,
                float softmax_scale, int num_sms, uintptr_t stream) {
        fn(reinterpret_cast<const void*>(Q),
           reinterpret_cast<const void*>(K),
           reinterpret_cast<const void*>(V),
           reinterpret_cast<void*>(O),
           reinterpret_cast<void*>(softmax_lse),
           reinterpret_cast<void*>(softmax_lse_accum),
           reinterpret_cast<void*>(o_accum),
           batch, seqlen_q, seqlen_k,
           num_heads_q, num_heads_kv, head_dim,
           py::cast<int>(q_strides[0]), py::cast<int>(q_strides[1]), py::cast<int>(q_strides[2]),
           py::cast<int>(k_strides[0]), py::cast<int>(k_strides[1]), py::cast<int>(k_strides[2]),
           py::cast<int>(v_strides[0]), py::cast<int>(v_strides[1]), py::cast<int>(v_strides[2]),
           py::cast<int>(o_strides[0]), py::cast<int>(o_strides[1]), py::cast<int>(o_strides[2]),
           softmax_scale, num_sms, to_stream(stream));
    };
}

static auto make_fwd_seqused(
    void (*fn)(const void*, const void*, const void*, void*, void*, const void*,
               int, int, int, int, int, int, int, int, int, int, int, int,
               int, int, int, int, int, int, float, int, cudaStream_t)) {
    return [fn](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                uintptr_t softmax_lse, uintptr_t seqused_k,
                int batch, int seqlen_q, int seqlen_k,
                int num_heads_q, int num_heads_kv, int head_dim,
                py::tuple q_strides, py::tuple k_strides,
                py::tuple v_strides, py::tuple o_strides,
                float softmax_scale, int num_sms, uintptr_t stream) {
        fn(reinterpret_cast<const void*>(Q), reinterpret_cast<const void*>(K),
           reinterpret_cast<const void*>(V), reinterpret_cast<void*>(O),
           reinterpret_cast<void*>(softmax_lse),
           reinterpret_cast<const void*>(seqused_k),
           batch, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim,
           py::cast<int>(q_strides[0]), py::cast<int>(q_strides[1]), py::cast<int>(q_strides[2]),
           py::cast<int>(k_strides[0]), py::cast<int>(k_strides[1]), py::cast<int>(k_strides[2]),
           py::cast<int>(v_strides[0]), py::cast<int>(v_strides[1]), py::cast<int>(v_strides[2]),
           py::cast<int>(o_strides[0]), py::cast<int>(o_strides[1]), py::cast<int>(o_strides[2]),
           softmax_scale, num_sms, to_stream(stream));
    };
}

PYBIND11_MODULE(flash_rt_fa2, m) {
    m.doc() = "FlashRT — LingBot slim Flash-Attention 2 forward (bf16 hdim128).";

    m.def("fwd_bf16", make_fwd(&fvk_attention_fa2_fwd_bf16),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("softmax_lse_accum") = 0, py::arg("o_accum") = 0,
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale") = 1.0f,
        py::arg("num_sms") = 0,
        py::arg("stream") = 0,
        kDocstring);

    m.def("fwd_bf16_tile", make_fwd_seqused(&fvk_attention_fa2_fwd_bf16_tile),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("seqused_k"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale") = 1.0f, py::arg("num_sms") = 0,
        py::arg("stream") = 0,
        "LingBot short-query FA2 tile (bf16, head_dim=128). "
        "seqused_k may be 0 for full attention.");
}
