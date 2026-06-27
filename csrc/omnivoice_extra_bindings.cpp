// ================================================================
// flash_rt_omnivoice — standalone pybind module for OmniVoice-specific
// fused kernels (kept separate from flash_rt_kernels so they can be
// added/rebuilt independently without touching the main bindings).
//
// Kernels: omnivoice_cfg_logsoftmax_bf16, omnivoice_qk_norm_rope_bf16
// ================================================================
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/omnivoice/omnivoice_cfg_combine.cuh"
#include "kernels/omnivoice/omnivoice_qk_norm_rope.cuh"

namespace py = pybind11;

static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_omnivoice, m) {
    m.doc() = "OmniVoice fused kernels";

    m.def("omnivoice_cfg_logsoftmax_bf16",
        [](uintptr_t c_logits, uintptr_t u_logits, uintptr_t out,
           int rows, int cols, int mask_col, double guidance_scale,
           uintptr_t stream) {
            omnivoice_cfg_logsoftmax_bf16(
                reinterpret_cast<const __nv_bfloat16*>(c_logits),
                reinterpret_cast<const __nv_bfloat16*>(u_logits),
                reinterpret_cast<__nv_bfloat16*>(out),
                rows, cols, mask_col, (float)guidance_scale, to_stream(stream));
        },
        py::arg("c_logits"), py::arg("u_logits"), py::arg("out"),
        py::arg("rows"), py::arg("cols"), py::arg("mask_col"),
        py::arg("guidance_scale"), py::arg("stream") = 0,
        "Fused CFG combine (BF16). Replaces 3x torch log_softmax.");

    m.def("omnivoice_qk_norm_rope_bf16",
        [](uintptr_t dq, uintptr_t q_weight, uintptr_t k_weight,
           uintptr_t cos, uintptr_t sin,
           uintptr_t q_temp, uintptr_t k_temp,
           int BS, int NH, int NKV, int HD, int QKVD, float eps,
           uintptr_t stream) {
            flash_rt::kernels::omnivoice_qk_norm_rope_bf16(
                reinterpret_cast<const __nv_bfloat16*>(dq),
                reinterpret_cast<const __nv_bfloat16*>(q_weight),
                reinterpret_cast<const __nv_bfloat16*>(k_weight),
                reinterpret_cast<const __nv_bfloat16*>(cos),
                reinterpret_cast<const __nv_bfloat16*>(sin),
                reinterpret_cast<__nv_bfloat16*>(q_temp),
                reinterpret_cast<__nv_bfloat16*>(k_temp),
                BS, NH, NKV, HD, QKVD, eps,
                reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("dq"), py::arg("q_weight"), py::arg("k_weight"),
        py::arg("cos"), py::arg("sin"),
        py::arg("q_temp"), py::arg("k_temp"),
        py::arg("BS"), py::arg("NH"), py::arg("NKV"), py::arg("HD"),
        py::arg("QKVD"), py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused Q/K RMSNorm + RoPE (warp-per-head). "
        "Replaces qkv_split + norm + rope (3 launches → 1).");
}
