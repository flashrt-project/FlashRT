// ================================================================
// FlashRT AMD — small-M FP8 GEMM bindings fragment
//
// #included INSIDE the PYBIND11_MODULE body of csrc/amd/bindings.cpp,
// after bindings_gemm.inc. Reuses the pointer helpers defined there
// (to_ptr / typed_ptr / to_stream). Raw-pointer ABI, same arg order
// as GemmRunner.fp8_nn_dev / fp8_nt_dev plus a nullable split_ws.
//
// Semantics: exact drop-ins for the GemmRunner FP8 paths — see
// gemm/smallm_fp8.h. split_ws (nn only; ignored by nt) is a device
// FP32 buffer of >= smallm_fp8_nn_ws_bytes(M, N) bytes; pass 0 to
// disable split-K.
// ================================================================

    m.def("smallm_fp8_nn_ws_bytes", [](int M, int N) {
        return (size_t)smallm_fp8_nn_dev_ws_bytes(M, N);
    }, py::arg("M"), py::arg("N"));

    m.def("smallm_fp8_nn_dev", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                  int M, int N, int K,
                                  uintptr_t d_scale_a, uintptr_t d_scale_b,
                                  uintptr_t split_ws, uintptr_t stream) {
        smallm_fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D),
                          typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_b),
                          M, N, K, typed_ptr<float>(split_ws), to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_b"),
       py::arg("split_ws") = 0, py::arg("stream") = 0);

    m.def("smallm_fp8_nn_dev_alt", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                      int M, int N, int K,
                                      uintptr_t d_scale_a, uintptr_t d_scale_b,
                                      uintptr_t split_ws, uintptr_t stream) {
        smallm_fp8_nn_dev_alt(to_ptr(A), to_ptr(B), to_ptr(D),
                              typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_b),
                              M, N, K, typed_ptr<float>(split_ws), to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_b"),
       py::arg("split_ws") = 0, py::arg("stream") = 0);

    m.def("smallm_fp8_nt_dev", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                  int M, int N, int K,
                                  uintptr_t d_scale_a, uintptr_t d_scale_b,
                                  uintptr_t split_ws, uintptr_t stream) {
        smallm_fp8_nt_dev(to_ptr(A), to_ptr(B), to_ptr(D),
                          typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_b),
                          M, N, K, typed_ptr<float>(split_ws), to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_b"),
       py::arg("split_ws") = 0, py::arg("stream") = 0);

    // LDS double-buffered nt pipeline, explicit entry for A/B rows.
    // Arg order mirrors smallm_fp8_nt_dev with use_async in the
    // split_ws position, so bench/parity harnesses can drive all nt
    // variants through one calling convention (use_async: 0 = sync
    // register-hoisted staging, nonzero = gfx9 LDS-DMA staging).
    m.def("smallm_fp8_nt_lds_dev", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                      int M, int N, int K,
                                      uintptr_t d_scale_a, uintptr_t d_scale_b,
                                      int use_async, uintptr_t stream) {
        smallm_fp8_nt_lds_dev(to_ptr(A), to_ptr(B), to_ptr(D),
                              typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_b),
                              M, N, K, use_async, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_b"),
       py::arg("use_async") = 1, py::arg("stream") = 0);

    m.def("smallm_fp8_nt_lds_async_available", []() {
        return smallm_fp8_nt_lds_async_available() != 0;
    });

    m.def("smallm_fp8_nt_dev_alt", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                      int M, int N, int K,
                                      uintptr_t d_scale_a, uintptr_t d_scale_b,
                                      uintptr_t split_ws, uintptr_t stream) {
        smallm_fp8_nt_dev_alt(to_ptr(A), to_ptr(B), to_ptr(D),
                              typed_ptr<float>(d_scale_a), typed_ptr<float>(d_scale_b),
                              M, N, K, typed_ptr<float>(split_ws), to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("d_scale_a"), py::arg("d_scale_b"),
       py::arg("split_ws") = 0, py::arg("stream") = 0);
