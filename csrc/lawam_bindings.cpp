// SPDX-License-Identifier: Apache-2.0
//
// flash_rt_lawam — standalone pybind module for LaWAM-specific kernels.
//
// Kept separate from flash_rt_kernels so the module can be built, installed and
// rebuilt on its own; FLASHRT_ENABLE_LAWAM gates the whole thing and nothing in
// the default target references it.
//
// Every entry takes raw device pointers and a stream so it can be captured in a
// CUDA graph and replayed with stable addresses. Entries return cudaError_t as
// int (0 on success).

#include <pybind11/pybind11.h>

#include <cstdint>
#include <utility>
#include <cuda_runtime.h>

#include "kernels/lawam/lawam_kernels.cuh"

namespace py = pybind11;

namespace {

cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

template <typename T>
T* ptr(uintptr_t p) {
    return reinterpret_cast<T*>(p);
}

}  // namespace

PYBIND11_MODULE(flash_rt_lawam, m) {
    namespace L = flash_rt::lawam;
    m.doc() =
        "LaWAM fused kernels (raw-pointer, CUDA-graph capturable). Built with "
        "-DFLASHRT_ENABLE_LAWAM=ON on GPU_ARCH=120.";

    m.def("memcpy_async",
        [](uintptr_t dst, uintptr_t src, size_t nbytes, uintptr_t stream) {
            return static_cast<int>(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                                                    nbytes, cudaMemcpyDefault, to_stream(stream)));
        },
        py::arg("dst"), py::arg("src"), py::arg("nbytes"), py::arg("stream") = 0,
        "cudaMemcpyAsync with cudaMemcpyDefault (device or pinned host pointers); graph-capturable.");

    m.def("preprocess_u8",
        [](uintptr_t frames, int num_views, uintptr_t qwen, uintptr_t dino, uintptr_t stream) {
            return L::preprocess_u8(ptr<const uint8_t>(frames), num_views, ptr<__half>(qwen), ptr<__half>(dino),
                                    to_stream(stream));
        },
        py::arg("frames"), py::arg("num_views"), py::arg("qwen"), py::arg("dino"), py::arg("stream") = 0,
        "uint8 [views, 256, 256, 3] frames -> Qwen3-VL merge-order FP16 patch rows [views*256, 768] "
        "(temporal copies folded) and DINOv3 FP16 patch rows of view 0 [256, 768].");

    m.def("gated_residual_norm",
        [](uintptr_t x, uintptr_t delta, uintptr_t delta_bias, uintptr_t gate, uintptr_t delta2, int mode,
           float eps, uintptr_t weight, uintptr_t bias, uintptr_t scale, uintptr_t shift, uintptr_t out_fp8,
           uintptr_t out_scale, uintptr_t out_f16, uintptr_t out_bf16, int rows, int dim, uintptr_t stream) {
            return L::gated_residual_norm(ptr<__half>(x), ptr<const __half>(delta), ptr<const __half>(delta_bias),
                                          ptr<const __half>(gate), ptr<const __half>(delta2), mode, eps,
                                          ptr<const __half>(weight), ptr<const __half>(bias),
                                          ptr<const __half>(scale), ptr<const __half>(shift),
                                          ptr<__nv_fp8_e4m3>(out_fp8), ptr<const float>(out_scale),
                                          ptr<__half>(out_f16), ptr<__nv_bfloat16>(out_bf16), rows, dim,
                                          to_stream(stream));
        },
        py::arg("x"), py::arg("delta"), py::arg("delta_bias"), py::arg("gate"), py::arg("delta2"), py::arg("mode"),
        py::arg("eps"), py::arg("weight"), py::arg("bias"), py::arg("scale"), py::arg("shift"),
        py::arg("out_fp8"), py::arg("out_scale"), py::arg("out_f16"), py::arg("out_bf16"), py::arg("rows"),
        py::arg("dim"), py::arg("stream") = 0,
        "x += (delta + delta_bias) * gate + delta2; y = norm(x) (mode 0 LN, 1 RMS, 2 identity) * weight + bias, "
        "* (1 + scale) + shift; FP8 / FP16 / BF16 outputs. Pointers 0 = absent.");

    for (const auto& [name, tanh_approx] : {std::pair<const char*, bool>{"bias_gelu_erf", false},
                                             std::pair<const char*, bool>{"bias_gelu_tanh", true}}) {
        const bool t = tanh_approx;
        m.def(name,
            [t](uintptr_t x, int ld, uintptr_t bias, uintptr_t out_fp8, uintptr_t scale, uintptr_t out_f16,
                int rows, int n, uintptr_t stream) {
                return L::bias_gelu(ptr<const __half>(x), ld, ptr<const __half>(bias), t,
                                    ptr<__nv_fp8_e4m3>(out_fp8), ptr<const float>(scale), ptr<__half>(out_f16),
                                    rows, n, to_stream(stream));
            },
            py::arg("x"), py::arg("ld"), py::arg("bias"), py::arg("out_fp8"), py::arg("scale"), py::arg("out_f16"),
            py::arg("rows"), py::arg("n"), py::arg("stream") = 0,
            t ? "gelu_tanh(x + bias) -> e4m3(y / *scale) and / or FP16."
              : "gelu_erf(x + bias) -> e4m3(y / *scale) and / or FP16.");
    }

    m.def("silu_mul",
        [](uintptr_t gate, uintptr_t up, int ld, uintptr_t out, int rows, int n, uintptr_t stream) {
            return L::silu_mul(ptr<const __half>(gate), ptr<const __half>(up), ld, ptr<__half>(out), rows, n,
                               to_stream(stream));
        },
        py::arg("gate"), py::arg("up"), py::arg("ld"), py::arg("out"), py::arg("rows"), py::arg("n"),
        py::arg("stream") = 0, "FP16 silu(gate) * up.");

    m.def("scatter_rows",
        [](uintptr_t src, uintptr_t bias, uintptr_t idx, uintptr_t dst, int n, int dim, uintptr_t stream) {
            return L::scatter_rows(ptr<const __half>(src), ptr<const __half>(bias), ptr<const int>(idx),
                                   ptr<__half>(dst), n, dim, to_stream(stream));
        },
        py::arg("src"), py::arg("bias"), py::arg("idx"), py::arg("dst"), py::arg("n"), py::arg("dim"),
        py::arg("stream") = 0, "dst[idx[i]] = src[i] + bias (FP16 rows, int32 idx).");

    m.def("cast_f16_bf16",
        [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
            return L::cast_f16_bf16(ptr<const __half>(in), ptr<__nv_bfloat16>(out), n, to_stream(stream));
        },
        py::arg("in_"), py::arg("out"), py::arg("n"), py::arg("stream") = 0, "FP16 -> BF16 (n % 8 == 0).");

    m.def("qformer_latent",
        [](uintptr_t vlm, uintptr_t idx, int nq, int vdim, int code, int heads, int ctx, py::dict w,
           uintptr_t scratch, uintptr_t latent, uintptr_t cond_silu, uintptr_t stream) {
            auto f = [&](const char* k) { return ptr<const float>(w[k].cast<uintptr_t>()); };
            L::QFormerWeights q{f("query"), f("norm_q_w"), f("norm_q_b"), f("norm_kv_w"), f("norm_kv_b"),
                                f("wq"), f("wk"), f("wv"), f("in_bias"), f("wo"), f("bo"), f("ffn_ln_w"),
                                f("ffn_ln_b"), f("w1"), f("b1"), f("w2"), f("b2"), f("final_w"), f("final_b"),
                                f("act_w"), f("act_b")};
            return L::qformer_latent(ptr<const __half>(vlm), ptr<const int>(idx), nq, vdim, code, heads, ctx, q,
                                     ptr<float>(scratch), ptr<float>(latent), ptr<__half>(cond_silu),
                                     to_stream(stream));
        },
        py::arg("vlm"), py::arg("idx"), py::arg("nq"), py::arg("vdim"), py::arg("code"), py::arg("heads"),
        py::arg("ctx"), py::arg("weights"), py::arg("scratch"), py::arg("latent"), py::arg("cond_silu"),
        py::arg("stream") = 0,
        "VLMToLAMQFormer over the act-query rows idx of the FP16 VLM states; writes the FP32 latent and "
        "silu(action_in_proj(latent)) FP16. `weights` maps names to FP32 device pointers.");

    m.def("action_encode",
        [](uintptr_t x, uintptr_t w1, uintptr_t b1, uintptr_t out, int rows, int n, int h, uintptr_t stream) {
            return L::action_encode(ptr<const float>(x), ptr<const float>(w1), ptr<const float>(b1),
                                    ptr<__nv_bfloat16>(out), rows, n, h, to_stream(stream));
        },
        py::arg("x"), py::arg("w1"), py::arg("b1"), py::arg("out"), py::arg("rows"), py::arg("n"), py::arg("h"),
        py::arg("stream") = 0, "silu(x @ W1 + b1) -> BF16; x FP32 [rows, n], W1 FP32 [n, h].");

    m.def("gaussian_noise",
        [](uintptr_t x, uintptr_t seed, int count, uintptr_t stream) {
            return L::gaussian_noise(ptr<float>(x), ptr<unsigned long long>(seed), count, to_stream(stream));
        },
        py::arg("x"), py::arg("seed"), py::arg("count"), py::arg("stream") = 0,
        "x ~ N(0, 1) keyed by the device uint64 *seed, which is advanced on device.");

    m.def("unnormalize_actions",
        [](uintptr_t x, int ld, uintptr_t lo, uintptr_t hi, uintptr_t out, int rows, int out_dim,
           uintptr_t stream) {
            return L::unnormalize_actions(ptr<const float>(x), ld, ptr<const float>(lo), ptr<const float>(hi),
                                          ptr<float>(out), rows, out_dim, to_stream(stream));
        },
        py::arg("x"), py::arg("ld"), py::arg("lo"), py::arg("hi"), py::arg("out"), py::arg("rows"),
        py::arg("out_dim"), py::arg("stream") = 0, "MIN_MAX unnormalization of the first out_dim columns.");

    m.def("gemm_f16acc",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, uintptr_t stream) {
            return L::gemm_f16acc(ptr<const __half>(A), ptr<const __half>(B), ptr<__half>(D), ptr<const __half>(bias),
                                  M, N, K, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        "FP16 GEMM with FP16 accumulation: D[M,N] = A[M,K] @ B[K,N] (+ bias). Plan cached per shape.");

    m.def("dit_layer_norm",
        [](uintptr_t x, uintptr_t scale, uintptr_t shift, uintptr_t pos, uintptr_t out, uintptr_t out_sf,
           uintptr_t out_gs, uintptr_t out_bf16, int rows, int dim, float eps, bool pdl, uintptr_t stream) {
            return L::dit_layer_norm(ptr<const __nv_bfloat16>(x), ptr<const __nv_bfloat16>(scale),
                                     ptr<const __nv_bfloat16>(shift), ptr<const __nv_bfloat16>(pos),
                                     ptr<uint8_t>(out), ptr<uint8_t>(out_sf), ptr<const float>(out_gs),
                                     ptr<__nv_bfloat16>(out_bf16), rows, dim, eps, pdl, to_stream(stream));
        },
        py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("pos"), py::arg("out"), py::arg("out_sf"),
        py::arg("out_gs"), py::arg("out_bf16"), py::arg("rows"), py::arg("dim"), py::arg("eps"),
        py::arg("pdl") = true, py::arg("stream") = 0,
        "LayerNorm(x) * (1 + scale) + shift + pos[row] -> NVFP4 skinny operand or BF16.");

    m.def("dit_sum_bias_residual_norm",
        [](uintptr_t partials, int splits, uintptr_t a_gs, uintptr_t w_gs, uintptr_t bias, uintptr_t residual,
           bool accumulate, uintptr_t scale, uintptr_t shift, uintptr_t pos, uintptr_t out, uintptr_t out_sf,
           uintptr_t out_gs, uintptr_t out_bf16, int rows, int dim, float eps, bool pdl, uintptr_t stream) {
            return L::dit_sum_bias_residual_norm(
                ptr<const float>(partials), splits, ptr<const float>(a_gs), ptr<const float>(w_gs),
                ptr<const __nv_bfloat16>(bias), ptr<__nv_bfloat16>(residual), accumulate,
                ptr<const __nv_bfloat16>(scale), ptr<const __nv_bfloat16>(shift), ptr<const __nv_bfloat16>(pos),
                ptr<uint8_t>(out), ptr<uint8_t>(out_sf), ptr<const float>(out_gs), ptr<__nv_bfloat16>(out_bf16),
                rows, dim, eps, pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"),
        py::arg("residual"), py::arg("accumulate"), py::arg("scale"), py::arg("shift"), py::arg("pos"),
        py::arg("out"), py::arg("out_sf"), py::arg("out_gs"), py::arg("out_bf16"), py::arg("rows"),
        py::arg("dim"), py::arg("eps"), py::arg("pdl") = true, py::arg("stream") = 0,
        "residual (+)= skinny GEMM sum + bias, then the modulated LayerNorm + pos into NVFP4 / BF16.");

    m.def("dit_sum_bias_act",
        [](uintptr_t partials, int splits, uintptr_t a_gs, uintptr_t w_gs, uintptr_t bias, int act, uintptr_t out,
           uintptr_t out_sf, uintptr_t out_gs, int rows, int n, bool pdl, uintptr_t stream) {
            return L::dit_sum_bias_act(ptr<const float>(partials), splits, ptr<const float>(a_gs),
                                       ptr<const float>(w_gs), ptr<const __nv_bfloat16>(bias), act,
                                       ptr<uint8_t>(out), ptr<uint8_t>(out_sf), ptr<const float>(out_gs), rows, n,
                                       pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"), py::arg("act"),
        py::arg("out"), py::arg("out_sf"), py::arg("out_gs"), py::arg("rows"), py::arg("n"),
        py::arg("pdl") = true, py::arg("stream") = 0, "NVFP4 act(sum + bias); act 0 identity, 1 ReLU.");

    m.def("dit_sum_bias_bf16",
        [](uintptr_t partials, int splits, uintptr_t a_gs, uintptr_t w_gs, uintptr_t bias, uintptr_t out, int rows,
           int n, bool pdl, uintptr_t stream) {
            return L::dit_sum_bias_bf16(ptr<const float>(partials), splits, ptr<const float>(a_gs),
                                        ptr<const float>(w_gs), ptr<const __nv_bfloat16>(bias),
                                        ptr<__nv_bfloat16>(out), rows, n, pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"), py::arg("out"),
        py::arg("rows"), py::arg("n"), py::arg("pdl") = true, py::arg("stream") = 0,
        "BF16 skinny GEMM sum + bias.");

    m.def("dit_sum_bias_euler",
        [](uintptr_t partials, int splits, uintptr_t a_gs, uintptr_t w_gs, uintptr_t bias, uintptr_t x, float dt,
           int rows, int n, bool pdl, uintptr_t stream) {
            return L::dit_sum_bias_euler(ptr<const float>(partials), splits, ptr<const float>(a_gs),
                                         ptr<const float>(w_gs), ptr<const float>(bias), ptr<float>(x), dt, rows,
                                         n, pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"), py::arg("x"),
        py::arg("dt"), py::arg("rows"), py::arg("n"), py::arg("pdl") = true, py::arg("stream") = 0,
        "FP32 x += dt * (skinny GEMM sum + bias).");
}
