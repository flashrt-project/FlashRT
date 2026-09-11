// ================================================================
// FlashRT AMD — MFMA small-M BF16 packed GEMM bindings fragment
//
// #included INSIDE the PYBIND11_MODULE body of csrc/amd/bindings.cpp,
// after the FP8 MFMA block. Reuses the pointer helpers defined there
// (to_ptr / typed_ptr / to_stream). Raw-pointer ABI.
//
// Semantics: see gemm/smallm_mfma_bf16.h. Wp is the PREPACKED weight
// (torch one-liner in the header); bias is a bf16 vector of length N.
// The three entries pin the epilogue; `variant` selects the launch
// geometry (0 = auto, see header) so the bench can drive every
// configuration through one calling convention.
// ================================================================

    m.def("smallm_mfma_bf16_nn_bias", [](uintptr_t A, uintptr_t Wp, uintptr_t bias,
                                         uintptr_t D, int M, int N, int K,
                                         int variant, uintptr_t stream) {
        smallm_mfma_bf16_nn_packed(variant, 0, to_ptr(A), to_ptr(Wp), to_ptr(bias),
                                   typed_ptr<__hip_bfloat16>(D), M, N, K,
                                   to_stream(stream));
    }, py::arg("A"), py::arg("Wp"), py::arg("bias"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("variant") = 0, py::arg("stream") = 0);

    m.def("smallm_mfma_bf16_nn_bias_gelu", [](uintptr_t A, uintptr_t Wp, uintptr_t bias,
                                              uintptr_t D, int M, int N, int K,
                                              int variant, uintptr_t stream) {
        smallm_mfma_bf16_nn_packed(variant, 1, to_ptr(A), to_ptr(Wp), to_ptr(bias),
                                   typed_ptr<__hip_bfloat16>(D), M, N, K,
                                   to_stream(stream));
    }, py::arg("A"), py::arg("Wp"), py::arg("bias"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("variant") = 0, py::arg("stream") = 0);

    m.def("smallm_mfma_bf16_nn_bias_res", [](uintptr_t A, uintptr_t Wp, uintptr_t bias,
                                             uintptr_t D, int M, int N, int K,
                                             int variant, uintptr_t stream) {
        smallm_mfma_bf16_nn_packed(variant, 2, to_ptr(A), to_ptr(Wp), to_ptr(bias),
                                   typed_ptr<__hip_bfloat16>(D), M, N, K,
                                   to_stream(stream));
    }, py::arg("A"), py::arg("Wp"), py::arg("bias"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("variant") = 0, py::arg("stream") = 0);

    m.def("smallm_mfma_bf16_variants", []() {
        py::list v;
        for (int i = 0; i < smallm_mfma_bf16_variant_count(); ++i)
            v.append(smallm_mfma_bf16_variant_name(i));
        return v;
    });
