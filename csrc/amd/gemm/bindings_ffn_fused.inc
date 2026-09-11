// ================================================================
// FlashRT AMD — fused decoder-FFN bindings fragment
//
// #included INSIDE the PYBIND11_MODULE body of csrc/amd/bindings.cpp,
// after bindings_smallm.inc. Reuses the pointer helpers defined there
// (to_ptr / typed_ptr / to_stream). Raw-pointer ABI.
//
// Semantics: gemm/decoder_ffn_fused.h — the two megakernels that
// replace the pi05 decoder FFN sub-chain (gate|up GEMM -> geglu ->
// fp8 quant, and down GEMM -> gate*x + residual).
// ================================================================

    m.def("smallm_fp8_gateup_geglu", [](uintptr_t A, uintptr_t W, uintptr_t OutFp8,
                                        int M, int N_half, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_w,
                                        uintptr_t d_scale_out, uintptr_t stream) {
        smallm_fp8_gateup_geglu(to_ptr(A), to_ptr(W), to_ptr(OutFp8),
                                typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_w),
                                typed_ptr<float>(d_scale_out),
                                M, N_half, K, to_stream(stream));
    }, py::arg("A"), py::arg("W"), py::arg("OutFp8"),
       py::arg("M"), py::arg("N_half"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_w"),
       py::arg("d_scale_out"), py::arg("stream") = 0);

    m.def("smallm_fp8_gateup_geglu_alt", [](uintptr_t A, uintptr_t W, uintptr_t OutFp8,
                                            int M, int N_half, int K,
                                            uintptr_t d_scale_a, uintptr_t d_scale_w,
                                            uintptr_t d_scale_out, uintptr_t stream) {
        smallm_fp8_gateup_geglu_alt(to_ptr(A), to_ptr(W), to_ptr(OutFp8),
                                    typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_w),
                                    typed_ptr<float>(d_scale_out),
                                    M, N_half, K, to_stream(stream));
    }, py::arg("A"), py::arg("W"), py::arg("OutFp8"),
       py::arg("M"), py::arg("N_half"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_w"),
       py::arg("d_scale_out"), py::arg("stream") = 0);

    m.def("smallm_fp8_down_gateres", [](uintptr_t H, uintptr_t W,
                                        uintptr_t residual, uintptr_t gate,
                                        int M, int N, int K,
                                        uintptr_t d_scale_h, uintptr_t d_scale_w,
                                        uintptr_t stream) {
        smallm_fp8_down_gateres(to_ptr(H), to_ptr(W),
                                to_ptr(residual), to_ptr(gate),
                                typed_ptr<float>(d_scale_h), typed_ptr<float>(d_scale_w),
                                M, N, K, to_stream(stream));
    }, py::arg("H"), py::arg("W"), py::arg("residual"), py::arg("gate"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_h"), py::arg("d_scale_w"), py::arg("stream") = 0);

    m.def("smallm_fp8_down_gateres_alt", [](uintptr_t H, uintptr_t W,
                                            uintptr_t residual, uintptr_t gate,
                                            int M, int N, int K,
                                            uintptr_t d_scale_h, uintptr_t d_scale_w,
                                            uintptr_t stream) {
        smallm_fp8_down_gateres_alt(to_ptr(H), to_ptr(W),
                                    to_ptr(residual), to_ptr(gate),
                                    typed_ptr<float>(d_scale_h), typed_ptr<float>(d_scale_w),
                                    M, N, K, to_stream(stream));
    }, py::arg("H"), py::arg("W"), py::arg("residual"), py::arg("gate"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_h"), py::arg("d_scale_w"), py::arg("stream") = 0);
