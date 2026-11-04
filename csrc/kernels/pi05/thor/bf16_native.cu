// Project Thor pi05 BF16/FP32 native arithmetic. No ATen compute operators.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
namespace py = pybind11;
using bf = __nv_bfloat16;
static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void blas(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(e));
}
static void dnn(cudnnStatus_t e) {
    if (e != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(e));
}
template <class T>
__device__ float rd(const T* x, int i) {
    return (float)x[i];
}
template <class T>
__device__ T cv(float x) {
    return (T)x;
}
template <>
__device__ bf cv<bf>(float x) {
    return __float2bfloat16_rn(x);
}
template <class A, class O>
__global__ void cast_k(const A* a, O* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = cv<O>(rd(a, i));
}
template <class T>
__global__ void binary_k(const T* a, const T* b, T* o, int n, int width, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = rd(a, i), y = rd(b, width ? i % width : i);
        o[i] = cv<T>(op == 0 ? __fadd_rn(x, y) : __fmul_rn(x, y));
    }
}
template <class T>
__global__ void unary_k(const T* a, T* o, int n, int kind, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = rd(a, i), y = x;
    if (kind == 0) y = x / (1.f + expf(-x));
    if (kind == 1) y = .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
    if (kind == 2) y = .5f * x * (1.f + erff(x * .7071067811865475f));
    if (kind == 3) y = x * scale;
    o[i] = cv<T>(y);
}
template <class T>
__global__ void bias_fill_k(const T* b, T* o, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = b[i % d];
}
template <class T>
__global__ void rms_k(const T* x, const float* gamma, const float* style, T* o, T* gate, int rows,
                      int d, float eps) {
    int row = blockIdx.x;
    float sum = 0;
    for (int j = threadIdx.x; j < d; j += 256) {
        float a = rd(x, row * d + j);
        sum = __fadd_rn(sum, __fmul_rn(a, a));
    }
    __shared__ typename cub::BlockReduce<float, 256>::TempStorage temp;
    __shared__ float inv;
    float total = cub::BlockReduce<float, 256>(temp).Sum(sum);
    if (threadIdx.x == 0) inv = rsqrtf(total / (float)d + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < d; j += 256) {
        float y = __fmul_rn(rd(x, row * d + j), inv);
        if (style) {
            y = __fadd_rn(__fmul_rn(y, 1.f + style[j]), style[d + j]);
            if (gate) gate[row * d + j] = cv<T>(style[2 * d + j]);
        } else
            y = __fmul_rn(y, 1.f + gamma[j]);
        o[row * d + j] = cv<T>(y);
    }
}
// Match the frozen PyTorch Reduce.cuh vector-of-four accumulation tree.
// One logical row uses 32 lanes for prefix rows or 64 lanes for 10 expert rows.
template <int BW>
__global__ void rms_reference_k(const bf* x, const float* gamma, const float* style, bf* o,
                                bf* gate, int d, float eps) {
    int row = blockIdx.x, tx = threadIdx.x;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int idx = tx; idx * 4 + 3 < d; idx += BW) {
        float v0 = rd(x, row * d + idx * 4), v1 = rd(x, row * d + idx * 4 + 1),
              v2 = rd(x, row * d + idx * 4 + 2), v3 = rd(x, row * d + idx * 4 + 3);
        a0 = __fadd_rn(a0, __fmul_rn(v0, v0));
        a1 = __fadd_rn(a1, __fmul_rn(v1, v1));
        a2 = __fadd_rn(a2, __fmul_rn(v2, v2));
        a3 = __fadd_rn(a3, __fmul_rn(v3, v3));
    }
    float total = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    __shared__ float scratch[BW];
    __shared__ float inv;
    if constexpr (BW > 32) {
        scratch[tx] = total;
        __syncthreads();
        if (tx < 32) total = __fadd_rn(total, scratch[tx + 32]);
    }
    if (tx < 32) {
        for (int delta = 16; delta; delta >>= 1)
            total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, delta));
        if (tx == 0) inv = rsqrtf(__fadd_rn(__fmul_rn(total, 1.f / d), eps));
    }
    __syncthreads();
    for (int j = tx; j < d; j += BW) {
        float y = __fmul_rn(rd(x, row * d + j), inv);
        if (style) {
            y = __fadd_rn(__fmul_rn(y, __fadd_rn(1.f, style[j])), style[d + j]);
            if (gate) gate[row * d + j] = cv<bf>(style[2 * d + j]);
        } else
            y = __fmul_rn(y, __fadd_rn(1.f, gamma[j]));
        o[row * d + j] = cv<bf>(y);
    }
}
struct Welford {
    float mean, m2, count;
};
struct FloatMax {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct Combine {
    __device__ Welford operator()(const Welford& a, const Welford& b) const {
        float count = a.count + b.count;
        if (count == 0) return {0, 0, 0};
        float delta = b.mean - a.mean;
        return {a.mean + delta * (b.count / count),
                a.m2 + b.m2 + delta * delta * (a.count * b.count / count), count};
    }
};
template <class T>
__global__ void norm_k(const T* x, const float* w, const float* b, T* o, int d, float eps) {
    Welford a = {0, 0, 0};
    for (int j = threadIdx.x; j < d; j += 256) {
        float v = rd(x, blockIdx.x * d + j);
        float count = a.count + 1;
        float delta = v - a.mean;
        float mean = a.mean + delta / count;
        a.m2 += delta * (v - mean);
        a.mean = mean;
        a.count = count;
    }
    __shared__ typename cub::BlockReduce<Welford, 256>::TempStorage temp;
    __shared__ float mean, inv;
    Welford t = cub::BlockReduce<Welford, 256>(temp).Reduce(a, Combine{});
    if (threadIdx.x == 0) {
        mean = t.mean;
        inv = rsqrtf(t.m2 / t.count + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < d; j += 256) {
        float v = (rd(x, blockIdx.x * d + j) - mean) * inv;
        v = v * w[j] + b[j];
        o[blockIdx.x * d + j] = cv<T>(v);
    }
}
// Match the vectorized-layernorm lane/vector ordering of the pinned reference.
// Derived from PyTorch layer_norm_kernel.cu (BSD-3-Clause), commit 5aff3928d8.
__device__ Welford ln_combine(Welford B, Welford A) {
    float count = A.count + B.count;
    if (count == 0) return {0, 0, 0};
    float delta = B.mean - A.mean, inv = 1.f / count, na = A.count * inv, nb = B.count * inv;
    float mean = __fmaf_rn(na, A.mean, nb * B.mean);
    float m2 = __fmaf_rn((delta * delta) * A.count, nb, A.m2 + B.m2);
    return {mean, m2, count};
}
__global__ void norm_reference_k(const bf* x, const float* w, const float* b, bf* o, int d,
                                 float eps) {
    int lane = threadIdx.x, warp = threadIdx.y, tx = lane + warp * 32, row = blockIdx.x;
    Welford a = {0, 0, 0};
    for (int vi = tx; vi < d / 4; vi += 128) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float v = rd(x, row * d + vi * 4 + j), delta = v - a.mean, count = a.count + 1.f,
                  mean = __fmaf_rn(delta, 1.f / count, a.mean);
            a.m2 = __fmaf_rn(delta, v - mean, a.m2);
            a.mean = mean;
            a.count = count;
        }
    }
    for (int offset = 16; offset; offset >>= 1) {
        Welford z = {__shfl_down_sync(0xffffffff, a.mean, offset),
                     __shfl_down_sync(0xffffffff, a.m2, offset),
                     __shfl_down_sync(0xffffffff, a.count, offset)};
        a = ln_combine(a, z);
    }
    __shared__ Welford warps[4];
    __shared__ float mean, inv;
    if (lane == 0) warps[warp] = a;
    __syncthreads();
    if (tx == 0) {
        Welford x0 = ln_combine(warps[0], warps[2]);
        Welford x1 = ln_combine(warps[1], warps[3]);
        Welford t = ln_combine(x0, x1);
        mean = t.mean;
        inv = rsqrtf(t.m2 / (float)d + eps);
    }
    __syncthreads();
    for (int vi = tx; vi < d / 4; vi += 128) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = vi * 4 + j;
            float y = inv * (rd(x, row * d + c) - mean);
            o[row * d + c] = cv<bf>(__fmaf_rn(w[c], y, b[c]));
        }
    }
}
__global__ void rope_k(const bf* q, const bf* k, const bf* cosine, const bf* sine, bf* oq, bf* ok,
                       int s, int hq, int hk, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nq = s * hq * d;
    if (i >= s * (hq + hk) * d) return;
    bool isq = i < nq;
    int j = isq ? i : i - nq;
    int h = isq ? hq : hk;
    int pos = j / (h * d), c = j % d;
    auto x = isq ? q : k;
    auto o = isq ? oq : ok;
    float r = rd(x, j + (c < d / 2 ? d / 2 : -d / 2));
    if (c < d / 2) r = -r;
    bf a = cv<bf>(__fmul_rn(rd(x, j), rd(cosine, pos * d + c)));
    bf b = cv<bf>(__fmul_rn(r, rd(sine, pos * d + c)));
    o[j] = cv<bf>(__fadd_rn((float)a, (float)b));
}
__global__ void softmax_k(const bf* score, const float* mask, bf* prob, int sq, int sk,
                          float scale) {
    int row = blockIdx.x;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < sk; j += 256) {
        float v = (float)cv<bf>(rd(score, row * sk + j) * scale);
        if (mask) v += mask[(row % sq) * sk + j];
        m = fmaxf(m, v);
    }
    __shared__ typename cub::BlockReduce<float, 256>::TempStorage temp;
    __shared__ float maxv, sumv;
    float mx = cub::BlockReduce<float, 256>(temp).Reduce(m, FloatMax{});
    if (threadIdx.x == 0) maxv = mx;
    __syncthreads();
    float sum = 0;
    for (int j = threadIdx.x; j < sk; j += 256) {
        float v = (float)cv<bf>(rd(score, row * sk + j) * scale);
        if (mask) v += mask[(row % sq) * sk + j];
        sum += expf(v - maxv);
    }
    __syncthreads();
    float sm = cub::BlockReduce<float, 256>(temp).Sum(sum);
    if (threadIdx.x == 0) sumv = sm;
    __syncthreads();
    for (int j = threadIdx.x; j < sk; j += 256) {
        float v = (float)cv<bf>(rd(score, row * sk + j) * scale);
        if (mask) v += mask[(row % sq) * sk + j];
        prob[row * sk + j] = cv<bf>(expf(v - maxv) / sumv);
    }
}
// Persistent-softmax row order: 16 registers/lane, XOR warp reductions.
// Frozen PI05 text rows have 456/466 keys, both padded to 512 internally.
__global__ void softmax_reference_k(const bf* score, const float* mask, bf* prob, int sq, int sk,
                                    float scale) {
    int row = blockIdx.x, lane = threadIdx.x;
    float e[16], mx = -INFINITY;
#pragma unroll
    for (int it = 0; it < 16; it++) {
        int j = lane + it * 32;
        float v = -INFINITY;
        if (j < sk) {
            v = (float)cv<bf>(__fmul_rn(rd(score, row * sk + j), scale));
            if (mask) v = __fadd_rn(v, mask[(row % sq) * sk + j]);
        }
        e[it] = v;
        mx = mx > v ? mx : v;
    }
    for (int delta = 16; delta; delta >>= 1) {
        float b = __shfl_xor_sync(0xffffffff, mx, delta);
        mx = mx < b ? b : mx;
    }
    float sum = 0;
#pragma unroll
    for (int it = 0; it < 16; it++) {
        e[it] = expf(e[it] - mx);
        sum = __fadd_rn(sum, e[it]);
    }
    for (int delta = 16; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta));
#pragma unroll
    for (int it = 0; it < 16; it++) {
        int j = lane + it * 32;
        if (j < sk) prob[row * sk + j] = cv<bf>(e[it] / sum);
    }
}
__global__ void patches_k(const float* im, float* out, int s, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s * k) return;
    int t = i / k, p = i % k;
    int c = p / 196, dy = (p % 196) / 14, dx = p % 14;
    int y = (t / 16) * 14 + dy, x = (t % 16) * 14 + dx;
    out[i] = im[c * 224 * 224 + y * 224 + x];
}
__global__ void euler_k(const float* x, const float* v, float* out, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fadd_rn(x[i], __fmul_rn(dt, v[i]));
}
__global__ void nchw_to_nhwc_k(const float* x, float* y, int c, int hw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c * hw) y[i] = x[(i % c) * hw + i / c];
}
class NativeConv {
    cudnnHandle_t h;
    cudnnTensorDescriptor_t xd, yd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;

   public:
    cudnnConvolutionFwdAlgo_t algorithm;
    cudnnMathType_t math_type;
    NativeConv() {
        dnn(cudnnCreate(&h));
        dnn(cudnnCreateTensorDescriptor(&xd));
        dnn(cudnnCreateTensorDescriptor(&yd));
        dnn(cudnnCreateFilterDescriptor(&wd));
        dnn(cudnnCreateConvolutionDescriptor(&cd));
        dnn(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, 224, 224));
        dnn(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 1152, 16, 16));
        dnn(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 1152, 3, 14, 14));
        dnn(cudnnSetConvolution2dDescriptor(cd, 0, 0, 14, 14, 1, 1, CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT));
        dnn(cudnnSetConvolutionMathType(cd, CUDNN_DEFAULT_MATH));
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        int count = 0;
        dnn(cudnnGetConvolutionForwardAlgorithm_v7(h, xd, wd, cd, yd, 8, &count, perf));
        bool found = false;
        for (int i = 0; i < count; i++) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                algorithm = perf[i].algo;
                math_type = perf[i].mathType;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("No cuDNN convolution algorithm");
        dnn(cudnnSetConvolutionMathType(cd, math_type));
        dnn(cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, algorithm,
                                                    &workspace_bytes));
        if (workspace_bytes) check(cudaMalloc(&workspace, workspace_bytes));
    }
    ~NativeConv() {
        if (workspace) cudaFree(workspace);
        cudnnDestroyTensorDescriptor(xd);
        cudnnDestroyTensorDescriptor(yd);
        cudnnDestroyFilterDescriptor(wd);
        cudnnDestroyConvolutionDescriptor(cd);
        cudnnDestroy(h);
    }
    void run(uintptr_t x, uintptr_t w, uintptr_t out, uintptr_t s) {
        dnn(cudnnSetStream(h, (cudaStream_t)s));
        float one = 1, zero = 0;
        dnn(cudnnConvolutionForward(h, &one, xd, (void*)x, wd, (void*)w, cd, algorithm, workspace,
                                    workspace_bytes, &zero, yd, (void*)out));
    }
    int algo() const { return (int)algorithm; }
    int math() const { return (int)math_type; }
};
class NativeRunner {
    struct Lt {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t a, b, c;
        cublasLtMatmulAlgo_t algo;
    };
    cublasHandle_t h;
    cublasLtHandle_t lt;
    void* workspace = nullptr;
    std::map<std::tuple<int, int, int, int>, Lt> cache;

   public:
    NativeRunner() {
        blas(cublasCreate(&h));
        blas(cublasLtCreate(&lt));
        check(cudaMalloc(&workspace, 8519680));
        blas(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    }
    ~NativeRunner() {
        for (auto& [key, v] : cache) {
            cublasLtMatmulDescDestroy(v.op);
            cublasLtMatrixLayoutDestroy(v.a);
            cublasLtMatrixLayoutDestroy(v.b);
            cublasLtMatrixLayoutDestroy(v.c);
        }
        if (workspace) cudaFree(workspace);
        cublasDestroy(h);
        cublasLtDestroy(lt);
    }
    void stream(uintptr_t s) {
        blas(cublasSetStream(h, (cudaStream_t)s));
        blas(cublasSetWorkspace(h, workspace, 8519680));
    }
    void gemm(uintptr_t a, uintptr_t w, uintptr_t bias, uintptr_t out, int m, int n, int k,
              int dtype, uintptr_t s) {
        stream(s);
        float alpha = 1, beta = 0;
        auto st = (cudaStream_t)s;
        cudaDataType_t t = dtype == 0 ? CUDA_R_16BF : CUDA_R_32F;
        if (bias) {
            auto key = std::make_tuple(m, n, k, dtype);
            auto it = cache.find(key);
            if (it == cache.end()) {
                Lt e;
                blas(cublasLtMatmulDescCreate(
                    &e.op, dtype == 0 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUDA_R_32F));
                cublasOperation_t tr = CUBLAS_OP_T;
                blas(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &tr,
                                                    sizeof(tr)));
                cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
                blas(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi,
                                                    sizeof(epi)));
                blas(cublasLtMatrixLayoutCreate(&e.a, t, k, n, k));
                blas(cublasLtMatrixLayoutCreate(&e.b, t, k, m, k));
                blas(cublasLtMatrixLayoutCreate(&e.c, t, n, m, n));
                cublasLtMatmulPreference_t pref;
                blas(cublasLtMatmulPreferenceCreate(&pref));
                size_t ws = 8519680;
                blas(cublasLtMatmulPreferenceSetAttribute(
                    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
                cublasLtMatmulHeuristicResult_t result;
                int count = 0;
                blas(cublasLtMatmulAlgoGetHeuristic(lt, e.op, e.a, e.b, e.c, e.c, pref, 1, &result,
                                                    &count));
                cublasLtMatmulPreferenceDestroy(pref);
                if (!count) throw std::runtime_error("No biased Lt algorithm");
                e.algo = result.algo;
                it = cache.emplace(key, e).first;
            }
            auto& e = it->second;
            void* bp = (void*)bias;
            blas(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp,
                                                sizeof(bp)));
            blas(cublasLtMatmul(lt, e.op, &alpha, (void*)w, e.a, (void*)a, e.b, &beta, (void*)out,
                                e.c, (void*)out, e.c, &e.algo, workspace, 8519680, st));
        } else
            blas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, (void*)w, t, k,
                              (void*)a, t, k, &beta, (void*)out, t, n,
                              dtype == 0 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void attention(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t scores, uintptr_t prob,
                   uintptr_t out, uintptr_t mask, int sq, int sk, int hq, int hk, int d,
                   float scale, uintptr_t s) {
        if (hk != 1 && hk != hq) throw std::invalid_argument("GQA group shape unsupported");
        stream(s);
        float one = 1, zero = 0;
        blas(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, sk, sq, d, &one, (void*)k, CUDA_R_16BF, hk * d,
            hk == 1 ? 0 : d, (void*)q, CUDA_R_16BF, hq * d, d, &zero, (void*)scores, CUDA_R_16BF,
            sk, (long long)sq * sk, hq, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (sk > 512)
            throw std::invalid_argument("Frozen native softmax supports at most 512 keys");
        softmax_reference_k<<<hq * sq, 32, 0, (cudaStream_t)s>>>((bf*)scores, (float*)mask,
                                                                 (bf*)prob, sq, sk, scale);
        blas(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N, d, sq, sk, &one, (void*)v, CUDA_R_16BF, hk * d,
            hk == 1 ? 0 : d, (void*)prob, CUDA_R_16BF, sk, (long long)sq * sk, &zero, (void*)out,
            CUDA_R_16BF, hq * d, d, hq, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
};
PYBIND11_MODULE(pi05_bf16_native_ext, m) {
    py::class_<NativeConv>(m, "NativeConv")
        .def(py::init<>())
        .def("run", &NativeConv::run)
        .def_property_readonly("algorithm", &NativeConv::algo)
        .def_property_readonly("math_type", &NativeConv::math);
    py::class_<NativeRunner>(m, "NativeRunner")
        .def(py::init<>())
        .def("gemm", &NativeRunner::gemm)
        .def("attention", &NativeRunner::attention);
    m.def("binary", [](uintptr_t a, uintptr_t b, uintptr_t o, int n, int width, int op, int dtype,
                       uintptr_t s) {
        if (dtype == 0)
            binary_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((bf*)a, (bf*)b, (bf*)o, n, width,
                                                                   op);
        else
            binary_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)a, (float*)b, (float*)o,
                                                                   n, width, op);
        check(cudaGetLastError());
    });
    m.def("unary", [](uintptr_t a, uintptr_t o, int n, int kind, float scale, int dtype,
                      uintptr_t s) {
        if (dtype == 0)
            unary_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((bf*)a, (bf*)o, n, kind, scale);
        else
            unary_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)a, (float*)o, n, kind,
                                                                  scale);
        check(cudaGetLastError());
    });
    m.def("cast", [](uintptr_t a, uintptr_t o, int n, bool to_bf16, uintptr_t s) {
        if (to_bf16)
            cast_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)a, (bf*)o, n);
        else
            cast_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((bf*)a, (float*)o, n);
        check(cudaGetLastError());
    });
    m.def("norm", [](uintptr_t x, uintptr_t w, uintptr_t b, uintptr_t o, int rows, int d, float eps,
                     uintptr_t s) {
        norm_reference_k<<<rows, dim3(32, 4), 0, (cudaStream_t)s>>>((bf*)x, (float*)w, (float*)b,
                                                                    (bf*)o, d, eps);
        check(cudaGetLastError());
    });
    m.def("rms", [](uintptr_t x, uintptr_t w, uintptr_t style, uintptr_t o, uintptr_t gate,
                    int rows, int d, float eps, uintptr_t s) {
        if (rows == 10)
            rms_reference_k<64><<<rows, 64, 0, (cudaStream_t)s>>>((bf*)x, (float*)w, (float*)style,
                                                                  (bf*)o, (bf*)gate, d, eps);
        else
            rms_reference_k<32><<<rows, 32, 0, (cudaStream_t)s>>>((bf*)x, (float*)w, (float*)style,
                                                                  (bf*)o, (bf*)gate, d, eps);
        check(cudaGetLastError());
    });
    m.def("rope", [](uintptr_t q, uintptr_t k, uintptr_t c, uintptr_t sn, uintptr_t oq,
                     uintptr_t ok, int seq, int hq, int hk, int d, uintptr_t s) {
        int n = seq * (hq + hk) * d;
        rope_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((bf*)q, (bf*)k, (bf*)c, (bf*)sn,
                                                             (bf*)oq, (bf*)ok, seq, hq, hk, d);
        check(cudaGetLastError());
    });
    m.def("patches", [](uintptr_t x, uintptr_t o, uintptr_t s) {
        patches_k<<<(256 * 588 + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)x, (float*)o, 256,
                                                                        588);
        check(cudaGetLastError());
    });
    m.def("euler", [](uintptr_t x, uintptr_t v, uintptr_t o, int n, float dt, uintptr_t s) {
        euler_k<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)x, (float*)v, (float*)o, n,
                                                              dt);
        check(cudaGetLastError());
    });
    m.def("copy", [](uintptr_t x, uintptr_t o, size_t bytes, uintptr_t s) {
        check(
            cudaMemcpyAsync((void*)o, (void*)x, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)s));
    });
    m.def("nchw_to_nhwc", [](uintptr_t x, uintptr_t o, int c, int hw, uintptr_t s) {
        nchw_to_nhwc_k<<<(c * hw + 255) / 256, 256, 0, (cudaStream_t)s>>>((float*)x, (float*)o, c,
                                                                          hw);
        check(cudaGetLastError());
    });
}
