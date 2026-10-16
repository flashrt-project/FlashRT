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

#include "attention/fa2_wrapper.h"

namespace py = pybind11;

static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

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

// FlashRT local patch 4: make_fwd sibling for the *_fp8out entries. Adds the FP8
// destination, its device scale pointer, its strides, and the bf16-skip flag.
template <typename Fn>
static auto make_fwd_fp8out(Fn fn) {
    return [fn](uintptr_t Q, uintptr_t K, uintptr_t V,
                uintptr_t O, uintptr_t softmax_lse,
                uintptr_t softmax_lse_accum, uintptr_t o_accum,
                int batch, int seqlen_q, int seqlen_k,
                int num_heads_q, int num_heads_kv, int head_dim,
                py::tuple q_strides, py::tuple k_strides,
                py::tuple v_strides, py::tuple o_strides,
                uintptr_t O_fp8, uintptr_t o_fp8_scale, py::tuple o_fp8_strides,
                bool skip_bf16_output,
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
           softmax_scale, num_sms, to_stream(stream),
           reinterpret_cast<void*>(O_fp8),
           reinterpret_cast<const void*>(o_fp8_scale),
           py::cast<int>(o_fp8_strides[0]), py::cast<int>(o_fp8_strides[1]),
           py::cast<int>(o_fp8_strides[2]),
           skip_bf16_output ? 1 : 0);
    };
}

template <typename Fn>
static auto make_fwd_window(Fn fn) {
    return [fn](uintptr_t Q, uintptr_t K, uintptr_t V,
                uintptr_t O, uintptr_t softmax_lse,
                uintptr_t softmax_lse_accum, uintptr_t o_accum,
                int batch, int seqlen_q, int seqlen_k,
                int num_heads_q, int num_heads_kv, int head_dim,
                py::tuple q_strides, py::tuple k_strides,
                py::tuple v_strides, py::tuple o_strides,
                float softmax_scale, int window_left, int window_right,
                int num_sms, uintptr_t stream) {
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
           softmax_scale, window_left, window_right, num_sms, to_stream(stream));
    };
}


// seqused variant: same as make_fwd but with a device seqused_k pointer and no
// splitkv accum args (num_splits is forced to 1 inside the entry).
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


static auto make_fwd_seqused_splitkv(
    void (*fn)(const void*, const void*, const void*, void*, void*, const void*,
               void*, void*,
               int, int, int, int, int, int, int, int, int, int, int, int,
               int, int, int, int, int, int, float, int, cudaStream_t)) {
    return [fn](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                uintptr_t softmax_lse, uintptr_t seqused_k,
                uintptr_t softmax_lse_accum, uintptr_t o_accum,
                int batch, int seqlen_q, int seqlen_k,
                int num_heads_q, int num_heads_kv, int head_dim,
                py::tuple q_strides, py::tuple k_strides,
                py::tuple v_strides, py::tuple o_strides,
                float softmax_scale, int num_sms, uintptr_t stream) {
        fn(reinterpret_cast<const void*>(Q), reinterpret_cast<const void*>(K),
           reinterpret_cast<const void*>(V), reinterpret_cast<void*>(O),
           reinterpret_cast<void*>(softmax_lse),
           reinterpret_cast<const void*>(seqused_k),
           reinterpret_cast<void*>(softmax_lse_accum),
           reinterpret_cast<void*>(o_accum),
           batch, seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim,
           py::cast<int>(q_strides[0]), py::cast<int>(q_strides[1]), py::cast<int>(q_strides[2]),
           py::cast<int>(k_strides[0]), py::cast<int>(k_strides[1]), py::cast<int>(k_strides[2]),
           py::cast<int>(v_strides[0]), py::cast<int>(v_strides[1]), py::cast<int>(v_strides[2]),
           py::cast<int>(o_strides[0]), py::cast<int>(o_strides[1]), py::cast<int>(o_strides[2]),
           softmax_scale, num_sms, to_stream(stream));
    };
}


PYBIND11_MODULE(flash_rt_fa2, m) {
    m.doc() = "FlashRT — vendored Flash-Attention 2 forward (fp16 + bf16).";

    m.def("fwd_bf16_seqused", make_fwd_seqused(&fvk_attention_fa2_fwd_bf16_seqused),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("seqused_k"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale") = 1.0f, py::arg("num_sms") = 0,
        py::arg("stream") = 0);

    m.def("fwd_bf16_tile", make_fwd_seqused(&fvk_attention_fa2_fwd_bf16_tile),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("seqused_k"),
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale") = 1.0f, py::arg("num_sms") = 0,
        py::arg("stream") = 0,
        "Hy-VLA small-N FA2 query-tile specialisation (bf16, head_dim<=128). "
        "seqused_k may be 0 for full attention. Same semantics as fwd_bf16 / "
        "fwd_bf16_seqused but with a narrow query tile tuned for short seqlen_q.");

    // FlashRT local patch 4: quantized-output siblings. O_fp8 = 0 makes each identical
    // to the entry without the suffix.
    m.def("fwd_bf16_fp8out", make_fwd_fp8out(&fvk_attention_fa2_fwd_bf16_fp8out),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("softmax_lse_accum") = 0, py::arg("o_accum") = 0,
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("O_fp8") = 0, py::arg("o_fp8_scale") = 0,
        py::arg("o_fp8_strides") = py::make_tuple(0, 0, 0),
        py::arg("skip_bf16_output") = false,
        py::arg("softmax_scale") = 1.0f, py::arg("num_sms") = 0, py::arg("stream") = 0,
        "Full-attention bf16 FA2 that also writes a static-FP8 (e4m3) copy of O. "
        "o_fp8_scale is the device scalar quantize_fp8_static reads; the FP8 values "
        "match that kernel bit for bit. o_fp8_strides is (batch, row, head) in "
        "elements. skip_bf16_output drops the bf16 store when the FP8 copy is the "
        "only consumer. With O_fp8 = 0 this is exactly fwd_bf16.");

    m.def("fwd_bf16_causal_fp8out", make_fwd_fp8out(&fvk_attention_fa2_fwd_bf16_causal_fp8out),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("softmax_lse_accum") = 0, py::arg("o_accum") = 0,
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("O_fp8") = 0, py::arg("o_fp8_scale") = 0,
        py::arg("o_fp8_strides") = py::make_tuple(0, 0, 0),
        py::arg("skip_bf16_output") = false,
        py::arg("softmax_scale") = 1.0f, py::arg("num_sms") = 0, py::arg("stream") = 0,
        "Causal sibling of fwd_bf16_fp8out (bf16, head_dim=256). With O_fp8 = 0 this "
        "is exactly fwd_bf16_causal.");

    m.def("fwd_bf16_seqused_splitkv",
        make_fwd_seqused_splitkv(&fvk_attention_fa2_fwd_bf16_seqused_splitkv),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("seqused_k"), py::arg("softmax_lse_accum")=0, py::arg("o_accum")=0,
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"), py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale")=1.0f, py::arg("num_sms")=0, py::arg("stream")=0);

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

    // Non-causal entry carrying FA2's window fields. A bounded past is a
    // local mask: Is_local switches on whenever window_size_* >= 0 and
    // Is_causal is false, which is exactly the [q-W+1, q] band a sliding
    // layer needs. Same signature as fwd_bf16 plus window_left/window_right.
    m.def("fwd_bf16_window", make_fwd_window(&fvk_attention_fa2_fwd_bf16_window),
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"), py::arg("softmax_lse"),
        py::arg("softmax_lse_accum") = 0, py::arg("o_accum") = 0,
        py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
        py::arg("num_heads_q"), py::arg("num_heads_kv"), py::arg("head_dim"),
        py::arg("q_strides"), py::arg("k_strides"),
        py::arg("v_strides"), py::arg("o_strides"),
        py::arg("softmax_scale") = 1.0f,
        py::arg("window_left") = -1,
        py::arg("window_right") = -1,
        py::arg("num_sms") = 0,
        py::arg("stream") = 0,
        kDocstring);

    // Causal sibling. Same signature as fwd_bf16 but applies a causal
    // mask inside FA2 (template Is_causal=true). Currently only
    // head_dim=128 is built; calls with other head_dim raise a
    // RuntimeError with a clear message. Used by Qwen3-8B prefill
    // (S=N causal self-attn).
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

#ifdef FLASHRT_ENABLE_CHAMELEON
    // FP16 causal sibling — head_dim=128 only. Used by Chameleon-7B
    // causal self-attention (32 layers MHA 32x128) on Orin SM87.
    m.def("fwd_fp16_causal", make_fwd(&fvk_attention_fa2_fwd_fp16_causal),
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
#endif
}
