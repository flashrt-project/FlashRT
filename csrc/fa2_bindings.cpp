// ============================================================================
//  FlashRT — pybind module for vendored Flash-Attention 2.
//
//  Built as a SEPARATE .so from flash_rt_kernels.so to keep the main kernel
//  binary small (~3.6 MB) and to avoid FA2's heavy CUTLASS 3.x template
//  compile time from gating every rebuild of our own kernels. Follows the
//  same pattern as flash_rt_fp4.so.
//
//  Python-side usage:
//
//      import flash_rt.flash_rt_kernels as fvk        # unchanged
//      import flash_rt.flash_rt_fa2     as fa2        # new, additive
//      fa2.fwd_fp16(Q, K, V, O, ...)
//      fa2.fwd_bf16(Q, K, V, O, ...)    # added in the bf16-vendor step
//
//  Only built when ENABLE_FA2 is ON at CMake time (SM80/86/89/120). Thor
//  builds skip this module entirely.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace py = pybind11;

static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

// Forward declarations (definitions in csrc/attention/fa2_wrapper.cu).
extern "C" void fvk_attention_fa2_fwd_fp16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

extern "C" void fvk_attention_fa2_fwd_bf16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

// Causal variant — definition in csrc/attention/fa2_wrapper_causal.cu.
// Currently only (bf16, head_dim=128) is built. Used by Qwen3-8B
// prefill (S=N causal self-attention).
extern "C" void fvk_attention_fa2_fwd_bf16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);


// Shared docstring. pybind::def's doc arg takes a single string; we want the
// same text for both fwd_fp16 and fwd_bf16 so deduplicate via static const.
static const char* kDocstring = R"(FlashAttention-2 fwd (vendored). GQA-capable cross-attention.

Args:
    Q, K, V, O: int device pointers. Q is (batch, seqlen_q, num_heads_q, head_dim);
      K/V are (batch, seqlen_k, num_heads_kv, head_dim); O has Q's shape.
    softmax_lse: int device pointer, fp32, shape (batch, num_heads_q, seqlen_q).
    softmax_lse_accum, o_accum: splitkv scratch buffers, fp32. When both non-zero
      AND num_sms > 0, wrapper enables the num_splits heuristic and may dispatch
      to the splitkv kernel. Sizes must fit worst-case:
        softmax_lse_accum: (max_splits, batch, num_heads_q, seqlen_q) fp32
        o_accum:           (max_splits, batch, num_heads_q, seqlen_q, head_dim_rounded) fp32
      Pass 0 to force num_splits=1 (no splitkv, lower SM occupancy on small shapes).
    *_strides: 3-tuple (batch, row, head) in elements (matches .stride()).
    softmax_scale: typically 1.0 / sqrt(head_dim).
    num_sms: current device's SM count (from torch.cuda.get_device_properties(...)
             .multi_processor_count). 0 disables splitkv.
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


PYBIND11_MODULE(flash_rt_fa2, m) {
    m.doc() = "FlashRT — vendored Flash-Attention 2 forward (fp16 + bf16).";

    m.def("fwd_fp16", make_fwd(&fvk_attention_fa2_fwd_fp16),
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

    // Causal sibling. Same signature as fwd_bf16 but applies a causal
    // mask inside FA2 (template Is_causal=true). Currently only
    // head_dim=128 is built; calls with other head_dim abort with a
    // clear message. Used by Qwen3-8B prefill (S=N causal self-attn).
    m.def("fwd_bf16_causal", make_fwd(&fvk_attention_fa2_fwd_bf16_causal),
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
}
