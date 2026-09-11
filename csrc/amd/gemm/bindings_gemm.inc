// ================================================================
// FlashRT AMD — GEMM bindings fragment (hipBLASLt GemmRunner + FvkContext)
//
// This file is #included INSIDE the PYBIND11_MODULE body of
// csrc/amd/bindings.cpp. It reuses the pointer helpers defined there
// (to_ptr / typed_ptr / to_stream) and keeps the exact method and
// py::arg names of the CUDA module (csrc/bindings.cpp:611-828).
//
// Integration steps (owner of bindings.cpp / CMakeLists.txt):
//   1. At file scope in csrc/amd/bindings.cpp, add:
//        #include "context.h"
//        #include "gemm/hipblaslt_runner.h"
//   2. Inside PYBIND11_MODULE(flash_rt_amd_kernels, m) { ... }, add:
//        #include "gemm/bindings_gemm.inc"
//   3. In csrc/amd/CMakeLists.txt:
//        find_package(hipblaslt REQUIRED)
//        target_link_libraries(<module_target> PRIVATE roc::hipblaslt)
//        add gemm/hipblaslt_runner.hip to the module's source list
//      (hipblaslt ships its CMake config with ROCm; if find_package
//       needs help, add ${ROCM_PATH}/lib/cmake/hipblaslt to
//       CMAKE_PREFIX_PATH.)
//
// Notes vs the CUDA bindings:
//   - Only the ported subset is exposed: FvkContext, bf16_run,
//     bf16_nn, bf16_nn_res, bf16_nn_bias, bf16_nn_bias_gelu,
//     bf16_nn_bias_res, fp8_nn_dev,
//     fp8_nt_dev, autotune_bf16_nn, autotune_fp8_nn_dev,
//     autotune_fp8_nt_dev.
//   - AMD-only additions (no CUDA counterpart): mxfp4_nt_dev and
//     autotune_mxfp4_nt_dev (gfx950 native MXFP4 a4w4 via hipBLASLt
//     VEC32_UE8M0 block scales).
//   - bf16_run exists on the CUDA C++ class but is not exposed by the
//     CUDA bindings; it is exposed here (same name/signature as the
//     C++ method) because the AMD line needs a Python-reachable
//     BF16 NT path.
//   - FvkContext.handle_ptr returns the hipblasLtHandle_t address
//     (the CUDA property returns the cublasHandle_t address).
// ================================================================

    // ── FvkContext: per-instance hipBLASLt handle ──
    py::class_<FvkContext>(m, "FvkContext")
        .def(py::init<>())
        .def_property_readonly("handle_ptr", [](const FvkContext& ctx) {
            return reinterpret_cast<uintptr_t>(ctx.hipblaslt_handle);
        });

    // ── GemmRunner ──
    py::class_<GemmRunner>(m, "GemmRunner")
        .def(py::init<>())
        // Inference methods (stream-based, HIP Graph compatible)
        .def("bf16_run", [](GemmRunner& self,
                             uintptr_t A, uintptr_t B, uintptr_t D,
                             int M, int N, int K, uintptr_t stream) {
            self.bf16_run(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_res", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D,
                                int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_res(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias", [](GemmRunner& self,
                                 uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                 int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                               M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_gelu", [](GemmRunner& self,
                                      uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                      int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_gelu(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                    M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_res", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_res(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("fp8_nn_dev", [](GemmRunner& self,
                               uintptr_t A, uintptr_t B, uintptr_t D,
                               int M, int N, int K,
                               uintptr_t d_scale_a, uintptr_t d_scale_b,
                               uintptr_t stream) {
            self.fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                             typed_ptr<float>(d_scale_a),
                             typed_ptr<float>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        .def("fp8_nt_dev", [](GemmRunner& self,
                               uintptr_t A, uintptr_t B, uintptr_t D,
                               int M, int N, int K,
                               uintptr_t d_scale_a, uintptr_t d_scale_b,
                               uintptr_t stream) {
            self.fp8_nt_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                             typed_ptr<float>(d_scale_a),
                             typed_ptr<float>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        // MXFP4 a4w4 (E2M1 packed 2-per-byte + per-1x32-block ue8m0 uint8
        // scale tensors; see hipblaslt_runner.h for the exact layouts)
        .def("mxfp4_nt_dev", [](GemmRunner& self,
                                 uintptr_t A, uintptr_t A_scales,
                                 uintptr_t B, uintptr_t B_scales, uintptr_t D,
                                 int M, int N, int K, uintptr_t stream) {
            self.mxfp4_nt_dev(to_ptr(A), to_ptr(A_scales),
                               to_ptr(B), to_ptr(B_scales), to_ptr(D),
                               M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("A_scales"),
           py::arg("B"), py::arg("B_scales"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        // ── GROOT N1.7 surface: FP16 GEMM + FP8 epilogue variants ──
        // (same names/args as the CUDA bindings)
        .def("fp16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.fp16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("fp8_nn_bias", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                              M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),
           py::arg("stream") = 0)
        .def("fp8_nn_gelu_bias", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_gelu_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha"),
           py::arg("stream") = 0)
        .def("fp8_descale_fp16", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K,
                                     uintptr_t act_descale, uintptr_t w_descale,
                                     uintptr_t stream) {
            self.fp8_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                   typed_ptr<float>(act_descale),
                                   typed_ptr<float>(w_descale), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0)
        // Lazy autotune: timed selection on first use of each shape
        // (pre-capture eager calls only; see hipblaslt_runner.h note)
        .def("enable_lazy_autotune", [](GemmRunner& self, int num_algos) {
            self.enable_lazy_autotune(num_algos);
        }, py::arg("num_algos") = 16)
        // Autotune
        .def("autotune_bf16_nn", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, int num_algos) {
            self.autotune_bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
        .def("autotune_fp8_nn_dev", [](GemmRunner& self,
                                        uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_b,
                                        int num_algos) {
            self.autotune_fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                      typed_ptr<float>(d_scale_a),
                                      typed_ptr<float>(d_scale_b), num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("num_algos") = 16)
        .def("autotune_fp8_nt_dev", [](GemmRunner& self,
                                        uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_b,
                                        int num_algos) {
            self.autotune_fp8_nt_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                      typed_ptr<float>(d_scale_a),
                                      typed_ptr<float>(d_scale_b), num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("num_algos") = 16)
        .def("autotune_mxfp4_nt_dev", [](GemmRunner& self,
                                          uintptr_t A, uintptr_t A_scales,
                                          uintptr_t B, uintptr_t B_scales, uintptr_t D,
                                          int M, int N, int K, int num_algos) {
            self.autotune_mxfp4_nt_dev(to_ptr(A), to_ptr(A_scales),
                                        to_ptr(B), to_ptr(B_scales), to_ptr(D),
                                        M, N, K, num_algos);
        }, py::arg("A"), py::arg("A_scales"),
           py::arg("B"), py::arg("B_scales"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
        .def("autotune_fp16_nn", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, int num_algos) {
            self.autotune_fp16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
        .def("autotune_fp8_descale_fp16", [](GemmRunner& self,
                                              uintptr_t A, uintptr_t B, uintptr_t D,
                                              int M, int N, int K,
                                              uintptr_t act_descale, uintptr_t w_descale,
                                              int num_algos) {
            self.autotune_fp8_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                            typed_ptr<float>(act_descale),
                                            typed_ptr<float>(w_descale), num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("act_descale"), py::arg("w_descale"), py::arg("num_algos") = 16)
    ;
