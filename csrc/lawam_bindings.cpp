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

#include <cstring>

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

    m.def("graph_step",
        [](uintptr_t graph_exec, uintptr_t stream, uintptr_t in0, uintptr_t in0_pinned, size_t in0_bytes,
           uintptr_t in1, uintptr_t in1_pinned, size_t in1_bytes, uintptr_t out_pinned, uintptr_t out,
           size_t out_bytes) {
            // one policy step from host buffers: stage the inputs into the pinned buffers the graph
            // copies from, replay the graph, wait for it and hand the pinned outputs back
            py::gil_scoped_release nogil;
            if (in0 != 0 && in0 != in0_pinned)
                std::memcpy(reinterpret_cast<void*>(in0_pinned), reinterpret_cast<const void*>(in0), in0_bytes);
            if (in1 != 0 && in1 != in1_pinned)
                std::memcpy(reinterpret_cast<void*>(in1_pinned), reinterpret_cast<const void*>(in1), in1_bytes);
            cudaError_t e = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec), to_stream(stream));
            if (e == cudaSuccess) e = cudaStreamSynchronize(to_stream(stream));
            if (e == cudaSuccess && out != 0 && out != out_pinned)
                std::memcpy(reinterpret_cast<void*>(out), reinterpret_cast<const void*>(out_pinned), out_bytes);
            return static_cast<int>(e);
        },
        py::arg("graph_exec"), py::arg("stream"), py::arg("in0"), py::arg("in0_pinned"), py::arg("in0_bytes"),
        py::arg("in1"), py::arg("in1_pinned"), py::arg("in1_bytes"), py::arg("out_pinned"), py::arg("out"),
        py::arg("out_bytes"),
        "Host inputs -> pinned staging, cudaGraphLaunch, wait, pinned outputs -> host (GIL released).");

    m.def("memset_async",
        [](uintptr_t dst, int value, size_t nbytes, uintptr_t stream) {
            return static_cast<int>(cudaMemsetAsync(reinterpret_cast<void*>(dst), value, nbytes, to_stream(stream)));
        },
        py::arg("dst"), py::arg("value"), py::arg("nbytes"), py::arg("stream") = 0, "cudaMemsetAsync; graph-capturable.");

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

    m.def("qformer_latent",
        [](uintptr_t vlm, uintptr_t idx, int nq, int vdim, int code, int heads, int ctx, py::dict w,
           uintptr_t scratch, uintptr_t latent, uintptr_t cond_silu, uintptr_t stream) {
            auto f = [&](const char* k) { return ptr<const float>(w[k].cast<uintptr_t>()); };
            L::QFormerWeights q{f("query"), f("norm_q_w"), f("norm_q_b"), f("norm_kv_w"), f("norm_kv_b"),
                                f("wq_t"), f("wk"), f("wv"), f("in_bias"), f("wo_t"), f("bo"), f("ffn_ln_w"),
                                f("ffn_ln_b"), f("w1_t"), f("b1"), f("w2_t"), f("b2"), f("final_w"), f("final_b"),
                                f("act_w_t"), f("act_b")};
            return L::qformer_latent(ptr<const __half>(vlm), ptr<const int>(idx), nq, vdim, code, heads, ctx, q,
                                     ptr<float>(scratch), ptr<float>(latent), ptr<__half>(cond_silu),
                                     to_stream(stream));
        },
        py::arg("vlm"), py::arg("idx"), py::arg("nq"), py::arg("vdim"), py::arg("code"), py::arg("heads"),
        py::arg("ctx"), py::arg("weights"), py::arg("scratch"), py::arg("latent"), py::arg("cond_silu"),
        py::arg("stream") = 0,
        "VLMToLAMQFormer over the act-query rows idx of the FP16 VLM states; writes the FP32 latent and "
        "silu(action_in_proj(latent)) FP16. `weights` maps names to FP32 device pointers (the head "
        "projections wq_t / wo_t / w1_t / w2_t / act_w_t transposed to [in, out]).");

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

    m.def("gemm_f16acc_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, int tile, uintptr_t stream) {
            return L::gemm_f16acc_bf16out(ptr<const __half>(A), ptr<const __half>(B), ptr<__nv_bfloat16>(D),
                                          ptr<const __nv_bfloat16>(bias), M, N, K, tile, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("tile") = 0, py::arg("stream") = 0,
        "D[M,N] (BF16) = A[M,K] @ B[K,N] (FP16, FP16 accumulation) + bias[N] (BF16).");
    m.def("gemm_f16acc_set_autotune", &L::gemm_f16acc_set_autotune, py::arg("enable"),
        "Time the heuristic's non-split-K candidates for plans created afterwards (default off).");

    m.def("gemv_i8w",
        [](uintptr_t x, uintptr_t w, uintptr_t w_scale, uintptr_t bias, uintptr_t out, int N, int K, uintptr_t stream) {
            return L::gemv_i8w(ptr<const __half>(x), ptr<const int8_t>(w), ptr<const float>(w_scale),
                               ptr<const __half>(bias), ptr<__half>(out), N, K, to_stream(stream));
        },
        py::arg("x"), py::arg("w"), py::arg("w_scale"), py::arg("bias"), py::arg("out"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        "out[n] = w_scale[n] * sum_k int8 w[n,k] * x[k] + bias[n] (FP16 x / out, per-channel scales).");

    m.def("gemm_f16acc",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, bool gelu, uintptr_t stream) {
            return L::gemm_f16acc(ptr<const __half>(A), ptr<const __half>(B), ptr<__half>(D), ptr<const __half>(bias),
                                  M, N, K, gelu, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("gelu") = false, py::arg("stream") = 0,
        "FP16 GEMM with FP16 accumulation: D[M,N] = A[M,K] @ B[K,N] (+ bias, optionally tanh-GELU). "
        "Plan cached per shape.");

    m.def("gemm_bf16_f32out",
        [](uintptr_t A, uintptr_t W, uintptr_t D, int M, int N, int K, uintptr_t stream) {
            return L::gemm_bf16_f32out(ptr<const __nv_bfloat16>(A), ptr<const __nv_bfloat16>(W), ptr<float>(D), M, N,
                                       K, to_stream(stream));
        },
        py::arg("A"), py::arg("W"), py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0,
        "D[M,N] FP32 = A[M,K] @ W[N,K]^T (BF16 operands, nn.Linear weight layout). Plan cached per shape.");

    m.def("w8a16_skinny_config_count", &L::w8a16_skinny_config_count,
          "Number of weight-only W8A16 tile configs (M <= 32).");
    m.def("w8a16_skinny_config_k_chunk", &L::w8a16_skinny_config_k_chunk, py::arg("cfg"),
          "K columns per FP32 partial of config cfg: partials [K / k_chunk][M][N].");
    m.def("w8a16_skinny_gemm",
        [](uintptr_t A, uintptr_t W, uintptr_t w_scale, uintptr_t partials, int M, int N, int K, int cfg, bool pdl,
           uintptr_t stream) {
            return L::w8a16_skinny_gemm(ptr<const __nv_bfloat16>(A), ptr<const int8_t>(W), ptr<const float>(w_scale),
                                        ptr<float>(partials), M, N, K, cfg, pdl, to_stream(stream));
        },
        py::arg("A"), py::arg("W"), py::arg("w_scale"), py::arg("partials"), py::arg("M"), py::arg("N"),
        py::arg("K"), py::arg("cfg"), py::arg("pdl") = true, py::arg("stream") = 0,
        "Skinny W8A16 GEMM (M <= 32): FP32 K-split partials of A (BF16 [M,K]) @ W (INT8 [N,K], per-channel scales)^T.");

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
           uintptr_t out_sf, uintptr_t out_gs, uintptr_t out_bf16, int rows, int n, bool pdl, uintptr_t stream) {
            return L::dit_sum_bias_act(ptr<const float>(partials), splits, ptr<const float>(a_gs),
                                       ptr<const float>(w_gs), ptr<const __nv_bfloat16>(bias), act,
                                       ptr<uint8_t>(out), ptr<uint8_t>(out_sf), ptr<const float>(out_gs),
                                       ptr<__nv_bfloat16>(out_bf16), rows, n, pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"), py::arg("act"),
        py::arg("out"), py::arg("out_sf"), py::arg("out_gs"), py::arg("out_bf16"), py::arg("rows"), py::arg("n"),
        py::arg("pdl") = true, py::arg("stream") = 0,
        "act(sum + bias) -> NVFP4 or BF16 (out_bf16); act 0 identity, 1 ReLU, 2 tanh-GELU.");

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

    m.def("gemm_f16acc_act",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, int act, int split,
           uintptr_t stream) {
            return L::gemm_f16acc_act(ptr<const __half>(A), ptr<const __half>(B), ptr<__half>(D),
                                      ptr<const __half>(bias), M, N, K, act, split, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act"), py::arg("split") = 0, py::arg("stream") = 0,
        "D[M,N] = act(A[M,K] @ B[K,N] + bias) FP16 accumulate, fused epilogue (act 0 identity, bias optional; 1 tanh-GELU; 2 erf-GELU).");
    m.def("gemm_f16acc_split", &L::gemm_f16acc_split, py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("bias") = false, py::arg("gelu") = false,
          "Serial split-K count of the cuBLASLt plan gemm_f16acc runs for this shape (-1: no plan).");
    m.def("gemm_f16acc_act_cfg",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, int act, int tile, int split,
           uintptr_t stream) {
            return L::gemm_f16acc_act_cfg(ptr<const __half>(A), ptr<const __half>(B), ptr<__half>(D),
                                          ptr<const __half>(bias), M, N, K, act, tile, split, to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("act"), py::arg("tile"), py::arg("split") = 1, py::arg("stream") = 0,
        "gemm_f16acc_act with a fixed tile configuration and serial split-K count.");

    m.def("gemm_f16acc_persistent",
        [](uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias, int M, int N, int K, int ldd, int act, int cfg,
           int sched, uintptr_t stream) {
            return L::gemm_f16acc_persistent(ptr<const __half>(A), ptr<const __half>(B), ptr<__half>(D),
                                             ptr<const __half>(bias), M, N, K, ldd, act, cfg, sched,
                                             to_stream(stream));
        },
        py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("ldd"), py::arg("act"), py::arg("cfg"), py::arg("sched") = 0, py::arg("stream") = 0,
        "Persistent FP16-accumulate GEMM (data-parallel + stream-K schedule); act 3 = SwiGLU over 8-column "
        "interleaved gate | up weights.");

    m.def("gemm_f16acc_swiglu",
        [](uintptr_t A, uintptr_t Bg, uintptr_t Bu, int ldb, uintptr_t Dg, uintptr_t Du, int ldd, uintptr_t D, int M,
           int N, int K, int split, uintptr_t stream) {
            return L::gemm_f16acc_swiglu(ptr<const __half>(A), ptr<const __half>(Bg), ptr<const __half>(Bu), ldb,
                                         ptr<__half>(Dg), ptr<__half>(Du), ldd, ptr<__half>(D), M, N, K, split,
                                         to_stream(stream));
        },
        py::arg("A"), py::arg("Bg"), py::arg("Bu"), py::arg("ldb"), py::arg("Dg"), py::arg("Du"), py::arg("ldd"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("split") = 0, py::arg("stream") = 0,
        "D = silu(A @ Bg) * (A @ Bu): one FP16-accumulate dual GEMM (serial split-K), gate / up sums in Dg / Du.");
    m.def("gemm_f16acc_swiglu_cfg",
        [](uintptr_t A, uintptr_t Bg, uintptr_t Bu, int ldb, uintptr_t Dg, uintptr_t Du, int ldd, uintptr_t D, int M,
           int N, int K, int tile, int split, uintptr_t stream) {
            return L::gemm_f16acc_swiglu_cfg(ptr<const __half>(A), ptr<const __half>(Bg), ptr<const __half>(Bu), ldb,
                                             ptr<__half>(Dg), ptr<__half>(Du), ldd, ptr<__half>(D), M, N, K, tile,
                                             split, to_stream(stream));
        },
        py::arg("A"), py::arg("Bg"), py::arg("Bu"), py::arg("ldb"), py::arg("Dg"), py::arg("Du"), py::arg("ldd"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("tile"), py::arg("split") = 1,
        py::arg("stream") = 0, "gemm_f16acc_swiglu with a fixed tile configuration.");

    m.def("dit_cross_attention_state_floats", &L::dit_cross_attention_state_floats, py::arg("rows"),
          py::arg("heads"), py::arg("kv_slices"),
          "Floats of the cross-attention combine workspace for rows, heads and kv_slices.");
    m.def("dit_cross_attention",
        [](uintptr_t partials, int splits, uintptr_t a_gs, uintptr_t w_gs, uintptr_t bias, uintptr_t k, uintptr_t v,
           int kv_ld, int kv_seq, int kv_slices, uintptr_t out, int rows, int heads, int hd, float scale,
           uintptr_t state, uintptr_t counters, bool pdl, uintptr_t stream) {
            return L::dit_cross_attention(ptr<const float>(partials), splits, ptr<const float>(a_gs),
                                          ptr<const float>(w_gs), ptr<const __nv_bfloat16>(bias),
                                          ptr<const __nv_bfloat16>(k), ptr<const __nv_bfloat16>(v), kv_ld, kv_seq,
                                          kv_slices, ptr<__nv_bfloat16>(out), rows, heads, hd, scale,
                                          ptr<float>(state), ptr<unsigned>(counters), pdl, to_stream(stream));
        },
        py::arg("partials"), py::arg("splits"), py::arg("a_gs"), py::arg("w_gs"), py::arg("bias"), py::arg("k"),
        py::arg("v"), py::arg("kv_ld"), py::arg("kv_seq"), py::arg("kv_slices"), py::arg("out"), py::arg("rows"),
        py::arg("heads"), py::arg("hd"), py::arg("scale"), py::arg("state"), py::arg("counters"),
        py::arg("pdl") = true, py::arg("stream") = 0,
        "Action-row cross-attention over a long static K / V, Q from skinny-GEMM partials, keys split over CTAs.");

    m.def("gemv_nvfp4w",
        [](uintptr_t x, uintptr_t w, uintptr_t sf, uintptr_t gs, uintptr_t bias, uintptr_t out, int N, int K,
           uintptr_t stream, int max_ctas) {
            return L::gemv_nvfp4w(ptr<const __half>(x), ptr<const uint8_t>(w), ptr<const uint8_t>(sf),
                                  ptr<const float>(gs), ptr<const __half>(bias), ptr<__half>(out), N, K,
                                  to_stream(stream), max_ctas);
        },
        py::arg("x"), py::arg("w"), py::arg("sf"), py::arg("gs"), py::arg("bias"), py::arg("out"), py::arg("N"),
        py::arg("K"), py::arg("stream") = 0, py::arg("max_ctas") = 0,
        "out[N] = NVFP4 weights (E2M1 pairs, UE4M3 per-16 block scales, FP32 global scale) @ x[K] + bias; "
        "max_ctas > 0 caps the grid.");

    m.def("w4a16_skinny_config_count", &L::w4a16_skinny_config_count,
          "Number of NVFP4 weight-only W4A16 tile configs (M <= 32).");
    m.def("w4a16_skinny_config_k_chunk", &L::w4a16_skinny_config_k_chunk, py::arg("cfg"),
          "K columns per FP32 partial of config cfg: partials [K / k_chunk][M][N].");
    m.def("w4a16_skinny_gemm",
        [](uintptr_t A, uintptr_t W, uintptr_t sf, uintptr_t partials, int M, int N, int K, int cfg, bool pdl,
           uintptr_t stream) {
            return L::w4a16_skinny_gemm(ptr<const __nv_bfloat16>(A), ptr<const uint8_t>(W), ptr<const uint8_t>(sf),
                                        ptr<float>(partials), M, N, K, cfg, pdl, to_stream(stream));
        },
        py::arg("A"), py::arg("W"), py::arg("sf"), py::arg("partials"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("cfg"), py::arg("pdl") = true, py::arg("stream") = 0,
        "Skinny W4A16 GEMM (M <= 32): FP32 K-split partials of A (BF16 [M,K]) @ NVFP4 W (E2M1 + UE4M3 block scales)^T.");

    m.def("qk_rope_inplace",
        [](uintptr_t qkv, int ld, uintptr_t cos, uintptr_t sin, int S, int n_qk, int hd, uintptr_t stream) {
            return L::qk_rope_inplace(ptr<__half>(qkv), ld, ptr<const __half>(cos), ptr<const __half>(sin), S, n_qk,
                                      hd, to_stream(stream));
        },
        py::arg("qkv"), py::arg("ld"), py::arg("cos"), py::arg("sin"), py::arg("S"), py::arg("n_qk"), py::arg("hd"),
        py::arg("stream") = 0, "Rotate-half RoPE (head dim 64) in place on the Q | K heads of a fused QKV buffer.");
    m.def("qkv_rope_split",
        [](uintptr_t qkv, int ld, uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias, float k_mul, float v_mul,
           uintptr_t cos, uintptr_t sin, uintptr_t q_out, uintptr_t k_out, uintptr_t v_out, bool out_bf16, int S,
           int n_q, int n_k, int hd, uintptr_t stream) {
            return L::qkv_rope_split(ptr<const __half>(qkv), ld, ptr<const __half>(q_bias), ptr<const __half>(k_bias),
                                     ptr<const __half>(v_bias), k_mul, v_mul, ptr<const __half>(cos),
                                     ptr<const __half>(sin), ptr<void>(q_out), ptr<void>(k_out), ptr<void>(v_out),
                                     out_bf16, S, n_q, n_k, hd, to_stream(stream));
        },
        py::arg("qkv"), py::arg("ld"), py::arg("q_bias"), py::arg("k_bias"), py::arg("v_bias"), py::arg("k_mul"),
        py::arg("v_mul"), py::arg("cos"), py::arg("sin"), py::arg("q_out"), py::arg("k_out"), py::arg("v_out"),
        py::arg("out_bf16"), py::arg("S"), py::arg("n_q"), py::arg("n_k"), py::arg("hd"), py::arg("stream") = 0,
        "Fused-qkv split + bias + K/V scale + rotate-half RoPE (head dim 64), FP16 / BF16 out.");

    m.def("dit_flow_smem_bytes", &L::dit_flow_smem_bytes,
          "Dynamic shared-memory bytes the persistent flow kernel requests.");
    m.def("dit_flow_persistent",
        [](uintptr_t ops, int n_ops, uintptr_t site_tab, int n_sites, uintptr_t cta_units, int n_steps,
           uintptr_t stream_units, long long cta_stride, uintptr_t counter, int num_ctas, uintptr_t stream,
           bool reset_counter) {
            return L::dit_flow_persistent(ptr<const L::DitOp>(ops), n_ops, ptr<const int>(site_tab), n_sites,
                                          ptr<const int>(cta_units), n_steps, ptr<const uint8_t>(stream_units),
                                          cta_stride, ptr<unsigned>(counter), num_ctas, to_stream(stream),
                                          reset_counter);
        },
        py::arg("ops"), py::arg("n_ops"), py::arg("site_tab"), py::arg("n_sites"), py::arg("cta_units"),
        py::arg("n_steps"), py::arg("stream_units"), py::arg("cta_stride"),
        py::arg("counter"), py::arg("num_ctas"), py::arg("stream") = 0, py::arg("reset_counter") = true,
        "The flow loop (DiT steps, action encoder / decoder, Euler, unnormalize) as one persistent cooperative "
        "kernel over a host-built op table and per-CTA packed weight units.");
}
