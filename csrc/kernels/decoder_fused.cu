// ================================================================
// FlashRT — Decoder fused kernels (FP16)
// Direct port of pi05 engine ae_forward_static kernels.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cmath>
#include <stdexcept>
#include <string>

// ── C1: Fused AdaRMSNorm → FP8 with static scale ──
// Combines: RMSNorm(x) * (1+scale) + shift → FP8, gate output
__global__ void fused_adarms_fp8_static_fp16_kernel(
    const __half* __restrict__ x, const __half* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out, __half* __restrict__ gate_out,
    int S, int D, const float* __restrict__ descale_ptr) {
    int r = blockIdx.x; if (r >= S) return;
    const __half* row = x + r * D;
    const __half* sc = style + r * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]); sum_sq += v * v;
    }
    __shared__ float shv[8];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq; __syncthreads();
    if (!wid) { sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o); }
    __syncthreads(); if (!threadIdx.x) shv[0] = sum_sq; __syncthreads();
    float rstd = rsqrtf(shv[0] / D + 1e-6f);
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]) * rstd;
        float normed = v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]);
        out[r*D+i] = __nv_fp8_e4m3(fminf(fmaxf(normed * inv_scale, -448.0f), 448.0f));
        gate_out[r*D+i] = __float2half(__half2float(gt[i]));
    }
}

// ── C4→C5: Fused gate×residual + AdaRMSNorm → FP8 ──
// Combines: residual += gemm_out * gate, RMSNorm(residual) * (1+scale) + shift → FP8
__global__ void gate_res_adarms_fp8_static_fp16_kernel(
    const __half* __restrict__ gemm_out, const __half* __restrict__ prev_gate,
    __half* __restrict__ residual, const __half* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ fp8_out, __half* __restrict__ gate_out,
    int S, int D, const float* __restrict__ descale_ptr) {
    int r = blockIdx.x; if (r >= S) return;
    const __half* sc = style + r * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    extern __shared__ float shv[];
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float res = __half2float(residual[r*D+i]) + __half2float(gemm_out[r*D+i]) * __half2float(prev_gate[r*D+i]);
        residual[r*D+i] = __float2half(res);
        sum_sq += res * res;
    }
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq; __syncthreads();
    if (!wid) { sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o); }
    __syncthreads(); if (!threadIdx.x) shv[0] = sum_sq; __syncthreads();
    float rstd = rsqrtf(shv[0] / D + 1e-6f);
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(residual[r*D+i]) * rstd;
        float normed = v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]);
        fp8_out[r*D+i] = __nv_fp8_e4m3(fminf(fmaxf(normed * inv_scale, -448.0f), 448.0f));
        gate_out[r*D+i] = __float2half(__half2float(gt[i]));
    }
}

// ── C6: Merged GeGLU → FP8 with static scale ──
// Reads from merged [S, 2H] buffer: first H = gate, second H = up
// Applies GELU (tanh approx) to gate, multiply by up, quantize to FP8
// NOTE: pi05 names this "silu" but it's actually GELU tanh approximation
__global__ void geglu_fp8_static_fp16_kernel(
    const __half* __restrict__ merged, __nv_fp8_e4m3* __restrict__ out,
    int S, int H, const float* __restrict__ descale_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * H) return;
    int s = i / H, h = i % H;
    float gv = __half2float(merged[s * 2 * H + h]);
    float uv = __half2float(merged[s * 2 * H + H + h]);
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // pi05 uses: x / (1 + exp(-1.5957691216057308 * x * (1 + 0.044715 * x^2)))
    float gelu = gv / (1.0f + expf(-1.5957691216057308f * gv * (1.0f + 0.044715f * gv * gv)));
    float val = gelu * uv;
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    out[i] = __nv_fp8_e4m3(fminf(fmaxf(val * inv_scale, -448.0f), 448.0f));
}

// ── Simple gate × residual (last layer, no norm) ──
__global__ void gate_res_fp16_kernel(const __half* __restrict__ gemm_out,
                                      const __half* __restrict__ gate,
                                      __half* __restrict__ residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    residual[i] = __float2half(__half2float(residual[i]) + __half2float(gemm_out[i]) * __half2float(gate[i]));
}

// ── AdaRMSNorm (BF16/FP16 output, for final step) ──
__global__ void adarms_fp16_kernel(const __half* __restrict__ x, const __half* __restrict__ style,
                                    __half* __restrict__ out, __half* __restrict__ gate_out,
                                    int S, int D) {
    int r = blockIdx.x; if (r >= S) return;
    const __half* row = x + r * D;
    const __half* sc = style + r * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]); sum_sq += v * v;
    }
    __shared__ float shv[8];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq; __syncthreads();
    if (!wid) { sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o); }
    __syncthreads(); if (!threadIdx.x) shv[0] = sum_sq; __syncthreads();
    float rstd = rsqrtf(shv[0] / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]) * rstd;
        out[r*D+i] = __float2half(v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]));
        gate_out[r*D+i] = __float2half(__half2float(gt[i]));
    }
}

// ── Host wrappers ──

void fused_adarms_fp8_static_fp16(const __half* x, const __half* style,
                                    __nv_fp8_e4m3* out, __half* gate_out,
                                    int S, int D, const float* descale_ptr,
                                    cudaStream_t stream) {
    fused_adarms_fp8_static_fp16_kernel<<<S, 256, 0, stream>>>(x, style, out, gate_out, S, D, descale_ptr);
}

void gate_res_adarms_fp8_static_fp16(const __half* gemm_out, const __half* prev_gate,
                                       __half* residual, const __half* style,
                                       __nv_fp8_e4m3* fp8_out, __half* gate_out,
                                       int S, int D, const float* descale_ptr,
                                       cudaStream_t stream) {
    gate_res_adarms_fp8_static_fp16_kernel<<<S, 256, 8*sizeof(float), stream>>>(
        gemm_out, prev_gate, residual, style, fp8_out, gate_out, S, D, descale_ptr);
}

void geglu_fp8_static_fp16(const __half* merged, __nv_fp8_e4m3* out,
                             int S, int H, const float* descale_ptr,
                             cudaStream_t stream) {
    geglu_fp8_static_fp16_kernel<<<(S*H + 255)/256, 256, 0, stream>>>(merged, out, S, H, descale_ptr);
}

void gate_res_fp16(const __half* gemm_out, const __half* gate,
                    __half* residual, int n, cudaStream_t stream) {
    gate_res_fp16_kernel<<<(n + 255)/256, 256, 0, stream>>>(gemm_out, gate, residual, n);
}

void adarms_fp16(const __half* x, const __half* style,
                  __half* out, __half* gate_out, int S, int D,
                  cudaStream_t stream) {
    adarms_fp16_kernel<<<S, 256, 0, stream>>>(x, style, out, gate_out, S, D);
}

// ── Simple bias add: x[i] += b[i % D] (pi05 bias_k) ──
__global__ void add_bias_fp16_kernel(__half* x, const __half* b, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * D) x[i] = __float2half(__half2float(x[i]) + __half2float(b[i % D]));
}

void add_bias_fp16(__half* x, const __half* b, int S, int D, cudaStream_t stream) {
    add_bias_fp16_kernel<<<(S*D + 255)/256, 256, 0, stream>>>(x, b, S, D);
}

// ── gmm: cuBLAS NN GEMM with beta parameter (pi05 gmm) ──
// C = alpha * A @ B + beta * C  (FP16)
// A: (M, K), B: (K, N) row-major, C: (M, N)
// Stateless: receives cuBLAS handle from caller (FvkContext).
void gmm_fp16(cublasHandle_t handle, const __half* A, const __half* B, __half* C,
               int M, int N, int K, float beta, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    float alpha = 1.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
        &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ── FP8 GEMM with device descale pointers → FP16 output ──
// Exact port of pi05 gmm_fp8_kn_descale: cached descriptor + algo per (M,N,K).
// Scale pointers updated per-call. Layout: col-major matching pi05.
#include <cublasLt.h>
#include <unordered_map>

static cublasLtHandle_t g_fp8_lt = nullptr;
static void* g_fp8_ws = nullptr;
static size_t g_fp8_ws_sz = 32 * 1024 * 1024;  // Match production (32MB)

struct LtGemmKey {
    int M, N, K;
    bool operator==(const LtGemmKey& o) const { return M==o.M && N==o.N && K==o.K; }
};
struct LtGemmKeyHash {
    size_t operator()(const LtGemmKey& k) const {
        size_t h = std::hash<int>()(k.M);
        h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};
struct CachedLtGemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
};
static std::unordered_map<LtGemmKey, CachedLtGemm, LtGemmKeyHash> g_lt_cache;

static std::string fp8_gemm_shape(const char* name, int M, int N, int K) {
    return std::string(name) + " [" + std::to_string(M) + "," +
           std::to_string(N) + "," + std::to_string(K) + "]";
}

static void check_cublaslt(cublasStatus_t status, const char* name,
                           int M, int N, int K, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            fp8_gemm_shape(name, M, N, K) + ": " + op +
            " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

static void check_cuda(cudaError_t status, const char* name,
                       int M, int N, int K, const char* op) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            fp8_gemm_shape(name, M, N, K) + ": " + op +
            " failed with CUDA error " + cudaGetErrorString(status));
    }
}

static void ensure_fp8_lt(const char* name, int M, int N, int K) {
    if (!g_fp8_lt) {
        check_cublaslt(cublasLtCreate(&g_fp8_lt), name, M, N, K,
                       "cublasLtCreate");
        check_cuda(cudaMalloc(&g_fp8_ws, g_fp8_ws_sz), name, M, N, K,
                   "cudaMalloc workspace");
    }
}

static void check_heuristic(cublasStatus_t status, int returned_results,
                            const char* name, int M, int N, int K) {
    check_cublaslt(status, name, M, N, K, "cublasLtMatmulAlgoGetHeuristic");
    if (returned_results == 0) {
        throw std::runtime_error(
            fp8_gemm_shape(name, M, N, K) +
            ": cuBLASLt returned no FP8 GEMM algorithm");
    }
}

void fp8_gemm_descale_fp16(const void* A_fp8, const void* B_fp8, void* C_fp16,
                             int M, int N, int K,
                             const float* act_descale, const float* w_descale,
                             cudaStream_t stream) {
    const char* name = "fp8_gemm_descale_fp16";
    ensure_fp8_lt(name, M, N, K);

    LtGemmKey key{M, N, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg{};
        check_cublaslt(cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                       name, M, N, K, "cublasLtMatmulDescCreate");
        cublasOperation_t opN = CUBLAS_OP_N;
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSA");
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSB");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N),
                       name, M, N, K, "create A layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K),
                       name, M, N, K, "create B layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N),
                       name, M, N, K, "create C layout");
        cublasLtMatmulPreference_t pref;
        check_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                       name, M, N, K, "cublasLtMatmulPreferenceCreate");
        check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz)),
                       name, M, N, K, "set workspace preference");
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
            g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
            pref, 1, &result, &ret);
        cublasLtMatmulPreferenceDestroy(pref);
        check_heuristic(heuristic_status, ret, name, M, N, K);
        cg.algo = result.algo;
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale)),
                   name, M, N, K, "set A scale pointer");
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale)),
                   name, M, N, K, "set B scale pointer");
    float alpha = 1.0f, beta = 0.0f;
    check_cublaslt(cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_fp16, cg.Cdesc, C_fp16, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream),
                   name, M, N, K, "cublasLtMatmul");
}

// FP32 output variant
void fp8_gemm_descale_f32out(const void* A_fp8, const void* B_fp8, void* C_fp32,
                              int M, int N, int K,
                              const float* act_descale, const float* w_descale,
                              cudaStream_t stream) {
    const char* name = "fp8_gemm_descale_f32out";
    ensure_fp8_lt(name, M, N, K);

    LtGemmKey key{M, N + 9200000, K};  // unique key for f32out
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg{};
        check_cublaslt(cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                       name, M, N, K, "cublasLtMatmulDescCreate");
        cublasOperation_t opN = CUBLAS_OP_N;
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSA");
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSB");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N),
                       name, M, N, K, "create A layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K),
                       name, M, N, K, "create B layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_32F, N, M, N),
                       name, M, N, K, "create C layout");
        cublasLtMatmulPreference_t pref;
        check_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                       name, M, N, K, "cublasLtMatmulPreferenceCreate");
        check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz)),
                       name, M, N, K, "set workspace preference");
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
            g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
            pref, 1, &result, &ret);
        cublasLtMatmulPreferenceDestroy(pref);
        check_heuristic(heuristic_status, ret, name, M, N, K);
        cg.algo = result.algo;
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale)),
                   name, M, N, K, "set A scale pointer");
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale)),
                   name, M, N, K, "set B scale pointer");
    float alpha = 1.0f, beta = 0.0f;
    check_cublaslt(cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_fp32, cg.Cdesc, C_fp32, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream),
                   name, M, N, K, "cublasLtMatmul");
}

// BF16 output variant — for models trained in BF16 with activations exceeding
// FP16 range (Pi0-FAST decode_step). Same FP8 inputs and per-tensor descales as
// the FP16 variant; only the C matrix dtype is BF16.
void fp8_gemm_descale_bf16out(const void* A_fp8, const void* B_fp8, void* C_bf16,
                               int M, int N, int K,
                               const float* act_descale, const float* w_descale,
                               cudaStream_t stream) {
    const char* name = "fp8_gemm_descale_bf16out";
    ensure_fp8_lt(name, M, N, K);

    LtGemmKey key{M, N + 9100000, K};  // unique key for bf16out (avoid clash with fp16/f32out)
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg{};
        check_cublaslt(cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                       name, M, N, K, "cublasLtMatmulDescCreate");
        cublasOperation_t opN = CUBLAS_OP_N;
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSA");
        check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
                       name, M, N, K, "set TRANSB");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N),
                       name, M, N, K, "create A layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K),
                       name, M, N, K, "create B layout");
        check_cublaslt(cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16BF, N, M, N),
                       name, M, N, K, "create C layout");
        cublasLtMatmulPreference_t pref;
        check_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                       name, M, N, K, "cublasLtMatmulPreferenceCreate");
        check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz)),
                       name, M, N, K, "set workspace preference");
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
            g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
            pref, 1, &result, &ret);
        cublasLtMatmulPreferenceDestroy(pref);
        check_heuristic(heuristic_status, ret, name, M, N, K);
        cg.algo = result.algo;
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale)),
                   name, M, N, K, "set A scale pointer");
    check_cublaslt(cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale)),
                   name, M, N, K, "set B scale pointer");
    float alpha = 1.0f, beta = 0.0f;
    check_cublaslt(cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_bf16, cg.Cdesc, C_bf16, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream),
                   name, M, N, K, "cublasLtMatmul");
}
