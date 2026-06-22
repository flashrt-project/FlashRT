// ============================================================================
//  FlashRT — pybind module for Qwen3-VL kernels.
//
//  Built as a SEPARATE .so (flash_rt_qwen3_vl_kernels) from
//  flash_rt_kernels.so, gated by the FLASHRT_BUILD_QWEN3_VL CMake option, so
//  the shared production kernel binary is never rebuilt for this model. Same
//  pattern as flash_rt_fa2 / fmha_fp16_strided.
//
//  Python-side usage:
//
//      import flash_rt.flash_rt_kernels        as fvk     # unchanged
//      import flash_rt.flash_rt_qwen3_vl_kernels as vlk   # additive
//      vlk.rope_neox_qk_bf16(...)
//
//  Kernels here are general (rotate_half RoPE, etc.); they live in this
//  module only to keep the shared binary stable, and may be promoted to
//  flash_rt_kernels if another model needs them.
// ============================================================================

#include <pybind11/pybind11.h>

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace py = pybind11;

namespace flash_rt {
namespace kernels {
void rope_neox_qk_bf16(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    int rows, int q_heads, int k_heads, int head_dim, cudaStream_t stream);
}  // namespace kernels
}  // namespace flash_rt

static cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

template <typename T>
static T* as_ptr(uintptr_t p) {
    return reinterpret_cast<T*>(p);
}

PYBIND11_MODULE(flash_rt_qwen3_vl_kernels, m) {
    m.doc() = "FlashRT Qwen3-VL kernels (separate module; additive).";

    m.def(
        "rope_neox_qk_bf16",
        [](uintptr_t q_in, uintptr_t k_in, uintptr_t cos, uintptr_t sin,
           uintptr_t q_out, uintptr_t k_out, int rows, int q_heads,
           int k_heads, int head_dim, uintptr_t stream) {
            flash_rt::kernels::rope_neox_qk_bf16(
                as_ptr<const __nv_bfloat16>(q_in),
                as_ptr<const __nv_bfloat16>(k_in),
                as_ptr<const __nv_bfloat16>(cos),
                as_ptr<const __nv_bfloat16>(sin),
                as_ptr<__nv_bfloat16>(q_out), as_ptr<__nv_bfloat16>(k_out),
                rows, q_heads, k_heads, head_dim, to_stream(stream));
        },
        py::arg("q_in"), py::arg("k_in"), py::arg("cos"), py::arg("sin"),
        py::arg("q_out"), py::arg("k_out"), py::arg("rows"),
        py::arg("q_heads"), py::arg("k_heads"), py::arg("head_dim"),
        py::arg("stream") = 0);
}
