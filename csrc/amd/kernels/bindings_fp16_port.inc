// ================================================================
// FlashRT AMD — bindings for the FP16 backbone kernel port
// (included inside PYBIND11_MODULE in ../bindings.cpp; additive).
// Python-visible names and py::arg names mirror csrc/bindings.cpp
// for the same symbols EXACTLY, so the pipeline layer stays
// portable text across platforms. Launchers are declared in
// kernels/fp16_ops.h.
// ================================================================

// ── FP16 elementwise / activation ──
m.def("add_bias_fp16", [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
    add_bias_fp16(typed_ptr<__half>(x), typed_ptr<__half>(b),
                  S, D, to_stream(stream));
}, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"), py::arg("stream") = 0);

m.def("gelu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
    gelu_inplace_fp16(typed_ptr<__half>(x), n, to_stream(stream));
}, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

m.def("silu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
    silu_inplace_fp16(typed_ptr<__half>(x), n, to_stream(stream));
}, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

m.def("relu_inplace_bf16", [](uintptr_t x, int n, uintptr_t stream) {
    relu_inplace_bf16(typed_ptr<__hip_bfloat16>(x), n, to_stream(stream));
}, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

m.def("mul_fp16", [](uintptr_t a, uintptr_t b, uintptr_t out,
                     int n, uintptr_t stream) {
    mul_fp16(typed_ptr<__half>(a), typed_ptr<__half>(b),
             typed_ptr<__half>(out), n, to_stream(stream));
}, py::arg("a"), py::arg("b"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

m.def("residual_add_fp16", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
    residual_add_fp16(typed_ptr<__half>(residual), typed_ptr<__half>(x),
                      n, to_stream(stream));
}, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

m.def("gpu_fill_neginf_fp16", [](uintptr_t dst, int n, uintptr_t stream) {
    gpu_fill_neginf_fp16(typed_ptr<__half>(dst), n, to_stream(stream));
}, py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

m.def("gpu_strided_copy_fp16", [](uintptr_t src, uintptr_t dst,
                                  int rows, int dst_cols, int src_stride, int col_offset,
                                  uintptr_t stream) {
    gpu_strided_copy_fp16(typed_ptr<__half>(src), typed_ptr<__half>(dst),
                          rows, dst_cols, src_stride, col_offset, to_stream(stream));
}, py::arg("src"), py::arg("dst"),
   py::arg("rows"), py::arg("dst_cols"), py::arg("src_stride"), py::arg("col_offset"),
   py::arg("stream") = 0);

m.def("gpu_repeat_interleave_heads", [](uintptr_t src, uintptr_t dst,
                                        int S, int NH_src, int HD, int repeat, uintptr_t stream) {
    gpu_repeat_interleave_heads(typed_ptr<__half>(src), typed_ptr<__half>(dst),
                                S, NH_src, HD, repeat, to_stream(stream));
}, py::arg("src"), py::arg("dst"),
   py::arg("S"), py::arg("NH_src"), py::arg("HD"), py::arg("repeat"),
   py::arg("stream") = 0);

m.def("cast_fp16_to_bf16", [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
    cast_fp16_to_bf16(typed_ptr<__half>(in), typed_ptr<__hip_bfloat16>(out),
                      n, to_stream(stream));
}, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

m.def("cast_bf16_to_fp16", [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
    cast_bf16_to_fp16(typed_ptr<__hip_bfloat16>(in), typed_ptr<__half>(out),
                      n, to_stream(stream));
}, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

m.def("concat2_bf16", [](uintptr_t a, uintptr_t b, uintptr_t out,
                         int rows, int cols_a, int cols_b,
                         uintptr_t stream) {
    concat2_bf16(typed_ptr<__hip_bfloat16>(a), typed_ptr<__hip_bfloat16>(b),
                 typed_ptr<__hip_bfloat16>(out),
                 rows, cols_a, cols_b, to_stream(stream));
}, py::arg("a"), py::arg("b"), py::arg("out"),
   py::arg("rows"), py::arg("cols_a"), py::arg("cols_b"),
   py::arg("stream") = 0);

// ── FP16 quantize / FP8 fusion ──
m.def("quantize_fp8_static_fp16", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
    quantize_fp8_static_fp16(typed_ptr<__half>(input),
                             typed_ptr<__hip_fp8_e4m3>(output),
                             reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
}, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

m.def("silu_mul_split_fp8_fp16", [](uintptr_t gate, uintptr_t up, uintptr_t out,
                                    int n, uintptr_t d_scale, uintptr_t stream) {
    silu_mul_split_fp8_fp16(typed_ptr<__half>(gate), typed_ptr<__half>(up),
                            typed_ptr<__hip_fp8_e4m3>(out), n,
                            reinterpret_cast<const float*>(d_scale), to_stream(stream));
}, py::arg("gate"), py::arg("up"), py::arg("out"),
   py::arg("n"), py::arg("d_scale"), py::arg("stream") = 0);

// ── FP16 norm / RoPE / norm→FP8 ──
m.def("layer_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                            uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
    layer_norm_fp16(typed_ptr<__half>(x), typed_ptr<__half>(weight),
                    typed_ptr<__half>(bias), typed_ptr<__half>(out),
                    seq_len, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
   py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

m.def("rope_rotate_half_fp16", [](uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
                                  int S, int NH, int HD, uintptr_t stream) {
    rope_rotate_half_fp16(typed_ptr<__half>(x),
                          typed_ptr<__half>(cos_table),
                          typed_ptr<__half>(sin_table),
                          S, NH, HD, to_stream(stream));
}, py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
   py::arg("S"), py::arg("NH"), py::arg("HD"), py::arg("stream") = 0);

m.def("rms_norm_fp8_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps,
                              uintptr_t d_scale, uintptr_t stream) {
    rms_norm_fp8_fp16(typed_ptr<__half>(x), typed_ptr<__half>(weight),
                      typed_ptr<__hip_fp8_e4m3>(out), seq_len, dim, eps,
                      reinterpret_cast<const float*>(d_scale), to_stream(stream));
}, py::arg("x"), py::arg("weight"), py::arg("out"),
   py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
   py::arg("d_scale") = 0, py::arg("stream") = 0);

m.def("residual_add_rms_norm_fp8_fp16", [](uintptr_t residual, uintptr_t x,
                                           uintptr_t weight, uintptr_t out,
                                           int seq_len, int dim, float eps,
                                           uintptr_t d_scale, uintptr_t stream) {
    residual_add_rms_norm_fp8_fp16(typed_ptr<__half>(residual),
                                   typed_ptr<__half>(x),
                                   typed_ptr<__half>(weight),
                                   typed_ptr<__hip_fp8_e4m3>(out), seq_len, dim, eps,
                                   reinterpret_cast<const float*>(d_scale), to_stream(stream));
}, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
   py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
   py::arg("d_scale") = 0, py::arg("stream") = 0);

// ── Vectorized fp16 backbone entries (int status returns; -1 when the
//    dim%8 / n%16 / 16-byte-alignment preconditions do not hold) ──
m.def("rms_norm_fp16_vec", [](uintptr_t x, uintptr_t w, uintptr_t out,
                              int rows, int dim, float eps,
                              uintptr_t stream) -> int {
    return rms_norm_fp16_vec(typed_ptr<__half>(x), typed_ptr<__half>(w),
                             typed_ptr<__half>(out),
                             rows, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("w"), py::arg("out"), py::arg("rows"),
   py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

m.def("layer_norm_fp16_vec", [](uintptr_t x, uintptr_t w, uintptr_t b,
                                uintptr_t out, int rows, int dim,
                                float eps, uintptr_t stream) -> int {
    return layer_norm_fp16_vec(typed_ptr<__half>(x), typed_ptr<__half>(w),
                               typed_ptr<__half>(b), typed_ptr<__half>(out),
                               rows, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("out"),
   py::arg("rows"), py::arg("dim"), py::arg("eps") = 1e-6f,
   py::arg("stream") = 0);

m.def("layer_norm_fp8_static_fp16_vec",
      [](uintptr_t x, uintptr_t w, uintptr_t b, uintptr_t out,
         uintptr_t d_scale, int rows, int dim, float eps,
         uintptr_t stream) -> int {
    return layer_norm_fp8_static_fp16_vec(
        typed_ptr<__half>(x), typed_ptr<__half>(w), typed_ptr<__half>(b),
        typed_ptr<__hip_fp8_e4m3>(out),
        reinterpret_cast<const float*>(d_scale),
        rows, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("out"),
   py::arg("d_scale"), py::arg("rows"), py::arg("dim"),
   py::arg("eps") = 1e-6f, py::arg("stream") = 0);

m.def("rope_rotate_half_fp16_vec", [](uintptr_t x, uintptr_t cos_table,
                                      uintptr_t sin_table, int S, int NH,
                                      int HD, uintptr_t stream) -> int {
    return rope_rotate_half_fp16_vec(
        typed_ptr<__half>(x),
        typed_ptr<__half>(cos_table),
        typed_ptr<__half>(sin_table),
        S, NH, HD, to_stream(stream));
}, py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
   py::arg("S"), py::arg("NH"), py::arg("HD"), py::arg("stream") = 0);

m.def("quantize_fp8_static_fp16_vec", [](uintptr_t in, uintptr_t out,
                                         uintptr_t d_scale, int n,
                                         uintptr_t stream) -> int {
    return quantize_fp8_static_fp16_vec(
        typed_ptr<__half>(in),
        typed_ptr<__hip_fp8_e4m3>(out),
        reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
}, py::arg("in"), py::arg("out"), py::arg("d_scale"), py::arg("n"),
   py::arg("stream") = 0);

m.def("residual_add_fp16_vec", [](uintptr_t residual, uintptr_t x, int n,
                                  uintptr_t stream) -> int {
    return residual_add_fp16_vec(typed_ptr<__half>(residual),
                                 typed_ptr<__half>(x),
                                 n, to_stream(stream));
}, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

m.def("gpu_repeat_interleave_heads_vec", [](uintptr_t src, uintptr_t dst,
                                            int S, int NH_src, int HD,
                                            int repeat,
                                            uintptr_t stream) -> int {
    return gpu_repeat_interleave_heads_vec(
        typed_ptr<__half>(src), typed_ptr<__half>(dst),
        S, NH_src, HD, repeat, to_stream(stream));
}, py::arg("src"), py::arg("dst"), py::arg("S"), py::arg("NH_src"),
   py::arg("HD"), py::arg("repeat"), py::arg("stream") = 0);

// ── LayerNorm-style AdaLN (BF16 + FP8 out) ──
m.def("layer_norm_no_affine_bf16",
      [](uintptr_t x, uintptr_t out, int seq_len, int dim, float eps,
         uintptr_t stream) {
    layer_norm_no_affine_bf16(typed_ptr<__hip_bfloat16>(x),
                              typed_ptr<__hip_bfloat16>(out),
                              seq_len, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
   py::arg("eps") = 1e-5f, py::arg("stream") = 0);

m.def("ada_layer_norm_bf16",
      [](uintptr_t x, uintptr_t scale, uintptr_t shift, uintptr_t out,
         int seq_len, int dim, float eps, uintptr_t stream) {
    ada_layer_norm_bf16(typed_ptr<__hip_bfloat16>(x),
                        typed_ptr<__hip_bfloat16>(scale),
                        typed_ptr<__hip_bfloat16>(shift),
                        typed_ptr<__hip_bfloat16>(out),
                        seq_len, dim, eps, to_stream(stream));
}, py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("out"),
   py::arg("seq_len"), py::arg("dim"),
   py::arg("eps") = 1e-5f, py::arg("stream") = 0);

m.def("ada_layer_norm_fp8",
      [](uintptr_t x_bf16, uintptr_t scale_bf16, uintptr_t shift_bf16,
         uintptr_t out_fp8, uintptr_t act_scale,
         int seq_len, int dim, float eps, uintptr_t stream) {
    ada_layer_norm_fp8(typed_ptr<__hip_bfloat16>(x_bf16),
                       typed_ptr<__hip_bfloat16>(scale_bf16),
                       typed_ptr<__hip_bfloat16>(shift_bf16),
                       typed_ptr<__hip_fp8_e4m3>(out_fp8),
                       reinterpret_cast<const float*>(act_scale),
                       seq_len, dim, eps, to_stream(stream));
}, py::arg("x_bf16"), py::arg("scale_bf16"), py::arg("shift_bf16"),
   py::arg("out_fp8"), py::arg("act_scale"),
   py::arg("seq_len"), py::arg("dim"),
   py::arg("eps") = 1e-6f, py::arg("stream") = 0);

m.def("bias_gelu_quantize_fp8_static_bf16",
      [](uintptr_t in_bf16, uintptr_t bias_bf16, uintptr_t out_fp8,
         uintptr_t act_scale, long long M, int N, uintptr_t stream) {
    bias_gelu_quantize_fp8_static_bf16(
        typed_ptr<__hip_bfloat16>(in_bf16),
        bias_bf16 ? typed_ptr<__hip_bfloat16>(bias_bf16) : nullptr,
        typed_ptr<__hip_fp8_e4m3>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        M, N, to_stream(stream));
},
   py::arg("in_bf16"), py::arg("bias_bf16"),
   py::arg("out_fp8"), py::arg("act_scale"),
   py::arg("M"), py::arg("N"),
   py::arg("stream") = 0);
