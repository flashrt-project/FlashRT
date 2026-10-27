// SPDX-License-Identifier: Apache-2.0
//
// flash_rt_groot_n15 — standalone pybind module for GR00T N1.5 kernels.
//
// Kept separate from flash_rt_kernels: FLASHRT_ENABLE_GROOT_N15 gates the
// whole module and nothing in the default target references it. The GR00T
// N1.5 pipeline also uses shared flash_rt_kernels entries and flash_rt_fa2;
// this module holds only what those do not provide.
//
// Every entry takes raw device pointers and a stream so it can be captured in
// a CUDA graph and replayed with stable addresses. A null optional pointer is
// passed as 0.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/groot_n15/groot_n15_gemm.cuh"
#include "kernels/groot_n15/groot_n15_ops.cuh"

namespace py = pybind11;

namespace {

inline cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }
inline const __nv_bfloat16* cbf(uintptr_t p) { return reinterpret_cast<const __nv_bfloat16*>(p); }
inline __nv_bfloat16* bf(uintptr_t p) { return reinterpret_cast<__nv_bfloat16*>(p); }

}  // namespace

PYBIND11_MODULE(flash_rt_groot_n15, m) {
    namespace g = flash_rt::groot_n15;
    m.doc() =
        "GR00T N1.5 BF16 kernels (raw-pointer, CUDA-graph capturable). Built with "
        "-DFLASHRT_ENABLE_GROOT_N15=ON on GPU_ARCH=89.";

    m.def("groot_n15_linear_bias_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t bias, uintptr_t out, int M, int N, int K,
           uintptr_t stream) {
            g::linear_bias_bf16(cbf(x), cbf(W), cbf(bias), bf(out), M, N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("bias"), py::arg("out"), py::arg("M"), py::arg("N"),
        py::arg("K"), py::arg("stream") = 0,
        "out[M, N] = x[M, K] @ W[N, K]^T + bias[N], bf16, on the cuBLASLt bias-epilogue "
        "algorithm torch's addmm runs. The first call on a stream must not be captured.");

    m.def("groot_n15_bmm_bf16",
        [](uintptr_t x, uintptr_t W, uintptr_t out, int M, int N, int K, uintptr_t stream) {
            g::bmm_bf16(cbf(x), cbf(W), bf(out), M, N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("W"), py::arg("out"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        "out[M, N] = x[M, K] @ W[K, N] (W row-major [K, N]), bf16, on the cuBLAS "
        "strided-batched call torch.bmm makes for one batch. The first call on a stream "
        "must not be captured.");

    m.def("groot_n15_layer_norm_bf16",
        [](uintptr_t res_in, uintptr_t delta, uintptr_t res_out, uintptr_t gamma, uintptr_t beta,
           uintptr_t out, int rows, int dim, float eps, uintptr_t stream) {
            g::layer_norm_bf16(cbf(res_in), cbf(delta), bf(res_out), cbf(gamma), cbf(beta),
                               bf(out), rows, dim, eps, to_stream(stream));
        },
        py::arg("res_in"), py::arg("delta"), py::arg("res_out"), py::arg("gamma"),
        py::arg("beta"), py::arg("out"), py::arg("rows"), py::arg("dim"), py::arg("eps"),
        py::arg("stream") = 0,
        "r = res_in (+ delta, rounded to bf16) -> res_out; out = bf16(LayerNorm(r)), "
        "affine when gamma/beta are given (both or neither). dim <= 2048.");

    m.def("groot_n15_layer_norm_to_f32",
        [](uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out32, int rows, int dim,
           float eps, uintptr_t stream) {
            g::layer_norm_to_f32(cbf(x), cbf(gamma), cbf(beta), reinterpret_cast<float*>(out32),
                                 rows, dim, eps, to_stream(stream));
        },
        py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("out32"), py::arg("rows"),
        py::arg("dim"), py::arg("eps"), py::arg("stream") = 0,
        "out32 = LayerNorm(x) in fp32 (bf16 input, affine). dim <= 2048.");

    m.def("groot_n15_residual_layer_norm_f32",
        [](uintptr_t res32, uintptr_t delta, uintptr_t gamma, uintptr_t beta, uintptr_t out,
           int rows, int dim, float eps, uintptr_t stream) {
            g::residual_layer_norm_f32(reinterpret_cast<float*>(res32), cbf(delta), cbf(gamma),
                                       cbf(beta), bf(out), rows, dim, eps, to_stream(stream));
        },
        py::arg("res32"), py::arg("delta"), py::arg("gamma"), py::arg("beta"), py::arg("out"),
        py::arg("rows"), py::arg("dim"), py::arg("eps"), py::arg("stream") = 0,
        "res32 += delta (fp32 add) when delta is given; out = bf16(LayerNorm(res32)). "
        "dim <= 2048.");

    m.def("groot_n15_residual_add_f32_to_bf16",
        [](uintptr_t res32, uintptr_t delta, uintptr_t out, long long n, uintptr_t stream) {
            g::residual_add_f32_to_bf16(reinterpret_cast<const float*>(res32), cbf(delta), bf(out),
                                        n, to_stream(stream));
        },
        py::arg("res32"), py::arg("delta"), py::arg("out"), py::arg("n"), py::arg("stream") = 0,
        "out = bf16(res32 + delta).");

    m.def("groot_n15_ada_layer_norm_bf16",
        [](uintptr_t res_in, uintptr_t delta, uintptr_t res_out, uintptr_t scale1,
           uintptr_t shift, uintptr_t out, int rows, int dim, float eps, uintptr_t stream) {
            g::ada_layer_norm_bf16(cbf(res_in), cbf(delta), bf(res_out), cbf(scale1), cbf(shift),
                                   bf(out), rows, dim, eps, to_stream(stream));
        },
        py::arg("res_in"), py::arg("delta"), py::arg("res_out"), py::arg("scale1"),
        py::arg("shift"), py::arg("out"), py::arg("rows"), py::arg("dim"), py::arg("eps"),
        py::arg("stream") = 0,
        "r = res_in (+ delta, rounded to bf16) -> res_out; out = bf16(LayerNorm(r) * scale1 "
        "+ shift) with scale1 = bf16(1 + scale) and shift bf16 vectors. dim <= 2048.");

    m.def("groot_n15_add_rows_bf16",
        [](uintptr_t x, uintptr_t bias, uintptr_t table, uintptr_t out, int rows, int dim,
           int period, uintptr_t stream) {
            g::add_rows_bf16(cbf(x), cbf(bias), cbf(table), bf(out), rows, dim, period,
                             to_stream(stream));
        },
        py::arg("x"), py::arg("bias"), py::arg("table"), py::arg("out"), py::arg("rows"),
        py::arg("dim"), py::arg("period"), py::arg("stream") = 0,
        "t = x (+ bias[col], rounded to bf16); out = bf16(t + table[row % period, col]).");

    m.def("groot_n15_bias_act_bf16",
        [](uintptr_t x, uintptr_t bias, uintptr_t out, int rows, int dim, int act,
           uintptr_t stream) {
            g::bias_act_bf16(cbf(x), cbf(bias), bf(out), rows, dim, act, to_stream(stream));
        },
        py::arg("x"), py::arg("bias"), py::arg("out"), py::arg("rows"), py::arg("dim"),
        py::arg("act"), py::arg("stream") = 0,
        "t = bf16(x + bias[col]); out = t (act 0), relu(t) (act 1) or "
        "bf16(t * bf16(sigmoid(t))) (act 2).");

    m.def("groot_n15_concat_bias_bf16",
        [](uintptr_t x, uintptr_t bias, uintptr_t tail, uintptr_t out, int rows, int dim,
           int tail_dim, uintptr_t stream) {
            g::concat_bias_bf16(cbf(x), cbf(bias), cbf(tail), bf(out), rows, dim, tail_dim,
                                to_stream(stream));
        },
        py::arg("x"), py::arg("bias"), py::arg("tail"), py::arg("out"), py::arg("rows"),
        py::arg("dim"), py::arg("tail_dim"), py::arg("stream") = 0,
        "out[r, :dim] = bf16(x[r] + bias); out[r, dim:] = tail[r].");

    m.def("groot_n15_gelu_tanh_bf16",
        [](uintptr_t x, long long n, uintptr_t stream) {
            g::gelu_tanh_bf16(bf(x), n, to_stream(stream));
        },
        py::arg("x"), py::arg("n"), py::arg("stream") = 0,
        "In-place GELU (tanh approximation) in torch's fp32 expression order.");

    m.def("groot_n15_quant_rows_fp8",
        [](uintptr_t x, uintptr_t out, uintptr_t scales, int M, int K, uintptr_t stream) {
            g::quant_rows_fp8(cbf(x), reinterpret_cast<uint8_t*>(out),
                              reinterpret_cast<float*>(scales), M, K, to_stream(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("scales"), py::arg("M"), py::arg("K"),
        py::arg("stream") = 0,
        "Per-row E4M3 quantization: scales[m] = amax(|x[m]|) / 448, out = e4m3(x / scale) "
        "(saturating).");

    m.def("groot_n15_rescale_rows_bf16",
        [](uintptr_t d, uintptr_t row_scale, uintptr_t col_scale, uintptr_t bias, int M, int N,
           uintptr_t stream) {
            g::rescale_rows_bf16(bf(d), reinterpret_cast<const float*>(row_scale),
                                 reinterpret_cast<const float*>(col_scale), cbf(bias), M, N,
                                 to_stream(stream));
        },
        py::arg("d"), py::arg("row_scale"), py::arg("col_scale"), py::arg("bias"), py::arg("M"),
        py::arg("N"), py::arg("stream") = 0,
        "In place: d = bf16(d * row_scale[m] * col_scale[n] (+ bias[n])) -- the dequantization "
        "of an FP8 GEMM run with unit scales; bias may be 0.");

    m.def("groot_n15_actions_init",
        [](uintptr_t noise32, uintptr_t act32, uintptr_t act_bf16, long long n, uintptr_t stream) {
            g::actions_init(reinterpret_cast<const float*>(noise32), reinterpret_cast<float*>(act32),
                            bf(act_bf16), n, to_stream(stream));
        },
        py::arg("noise32"), py::arg("act32"), py::arg("act_bf16"), py::arg("n"),
        py::arg("stream") = 0,
        "act32 = noise32; act_bf16 = bf16(noise32).");

    m.def("groot_n15_euler_step",
        [](uintptr_t pred, uintptr_t act32, uintptr_t act_bf16, float dt, long long n,
           uintptr_t stream) {
            g::euler_step(cbf(pred), reinterpret_cast<float*>(act32), bf(act_bf16), dt, n,
                          to_stream(stream));
        },
        py::arg("pred"), py::arg("act32"), py::arg("act_bf16"), py::arg("dt"), py::arg("n"),
        py::arg("stream") = 0,
        "act32 += bf16(dt * pred); act_bf16 = bf16(act32).");
}
