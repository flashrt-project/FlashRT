// FlashRT · TurboQuant K/V dequant — unpack + (cuBLAS GEMM) + combine
// =====================================================================
// Phase 3A B9 step S3.  Add-only (new file).  Existing TQ Python helpers
// and FA2 wrapper untouched.
//
// Strategy decision:
//
//   The naive "fully-fused single CUDA kernel" approach (v1, deleted)
//   runs the 256×256 dequant GEMM portion on FP32 ALU and clocks 5.5 ms
//   — slower than Python (3.0 ms) and 32× off the bf16 tensor-core
//   roof (0.3 ms for one bf16 GEMM at M=128K).
//
//   Right design: do the GEMM portion via cuBLAS bf16 (tensor cores,
//   measured 0.10 ms / GEMM at M=128K) and only own the BW-bound
//   pre/post:
//
//     1) tq_unpack_packed_bf16 : packed bytes → (M,256) bf16 ×3 tensors
//                                (y_k, qjl±1 in bf16, y_v).  BW-bound.
//     2) (cuBLAS in caller)    : (2M,256)·(256,256) + (M,256)·(256,256)
//     3) tq_combine_kv_bf16    : norm·(K_mse + coef·rnorm·K_qjl), V_unit·norm_v
//                                element-wise on bf16.  BW-bound.
//
//   Caller orchestrates: 2 cuBLAS GEMMs + the 2 BW kernels here.
//
// Per-call BW accounting (kv_seq=32K):
//   read packed   : 32K·4·(128+32+128) = 38.5 MB
//   write y_k+qjl+y_v bf16 (3 tensors): 32K·4·256·2·3 = 192.0 MB
//   read for GEMM (cuBLAS):              read y_k|y_v (256 MB), qjl 64 MB
//   write GEMM out:                       192 MB
//   read for combine: 192 MB; write K+V: 128 MB
//   total ≈ 1.0 GB → 0.56 ms BW roof @ 1.79 TB/s.
//
//   Theoretical compute roof (3 bf16 GEMMs @ M=128K, N=K=256):
//     50.3 G fma → 0.48 ms @ 105 TFLOPS bf16.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int D = 256;             // head_dim
constexpr int NUM_KV = 4;          // GQA KV head count
constexpr int IDX_PACKED_BYTES = D / 2;     // 128
constexpr int QJL_PACKED_BYTES = D / 8;     // 32
constexpr int CB_K_MAX = 16;       // up to 4-bit codebook
constexpr int CB_V_MAX = 16;

struct TqLtFp32Plan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool valid = false;
};

static inline uint64_t tq_lt_plan_key(
    int M, int N, int K, int algo_idx, int trans_b)
{
    return ((uint64_t)(uint32_t)M << 32)
        ^ ((uint64_t)(uint32_t)N << 22)
        ^ ((uint64_t)(uint32_t)K << 12)
        ^ ((uint64_t)(uint32_t)(algo_idx & 0xff) << 4)
        ^ (uint64_t)(trans_b & 0xf);
}

static TqLtFp32Plan* tq_get_lt_fp32_plan(
    cublasLtHandle_t lt,
    int M, int N, int K, int algo_idx, bool trans_b,
    size_t workspace_size)
{
    static std::mutex mu;
    static std::unordered_map<uint64_t, TqLtFp32Plan> cache;

    if (algo_idx < 0) algo_idx = 0;
    const uint64_t key = tq_lt_plan_key(M, N, K, algo_idx, trans_b ? 1 : 0);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return &it->second;

    TqLtFp32Plan plan;
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;

    cublasLtMatmulDescCreate(&plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(
        plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(
        plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    cublasLtMatrixLayoutCreate(&plan.a_desc, CUDA_R_32F, M, K, K);
    cublasLtMatrixLayoutSetAttribute(
        plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order));
    cublasLtMatrixLayoutCreate(
        &plan.b_desc, CUDA_R_32F, trans_b ? N : K, trans_b ? K : N,
        trans_b ? K : N);
    cublasLtMatrixLayoutSetAttribute(
        plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order));
    cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, M, N, N);
    cublasLtMatrixLayoutSetAttribute(
        plan.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order));

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size));
    constexpr int kMaxAlgos = 32;
    cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos]{};
    int returned = 0;
    cublasLtMatmulAlgoGetHeuristic(
        lt, plan.desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.c_desc,
        pref, kMaxAlgos, heuristics, &returned);
    cublasLtMatmulPreferenceDestroy(pref);
    if (returned > 0) {
        if (algo_idx >= returned) algo_idx = returned - 1;
        plan.algo = heuristics[algo_idx].algo;
        plan.valid = true;
    }

    auto [inserted_it, _] = cache.emplace(key, plan);
    return &inserted_it->second;
}

// ── Unpack kernel ───────────────────────────────────────────────────
// Block: D threads, one per output coord.  Grid: M blocks (M = S*NUM_KV).
// One row per block.  All three outputs written in one launch.
//
// Templated on bit-widths so the nibble→codebook-mask is a compile-time
// no-op when the slot fully uses its bits (b=4 → mask 0xF == nib).

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_unpack_packed_bf16_kernel(
    const uint8_t*       __restrict__ k_idx_packed,
    const uint8_t*       __restrict__ k_qjl_packed,
    const uint8_t*       __restrict__ v_idx_packed,
    const float*         __restrict__ cb_k_mse,
    const float*         __restrict__ cb_v,
    __nv_bfloat16*       __restrict__ y_k,
    __nv_bfloat16*       __restrict__ qjl_bf,
    __nv_bfloat16*       __restrict__ y_v,
    int M)
{
    constexpr int K_MASK = (1 << B_MSE_K) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_MSE_K;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
    __syncthreads();

    const int byte_idx = tid >> 1;
    const int nib_hi   = tid & 1;

    // ── y_k from k_idx_packed ──
    const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    knib = nib_hi ? ((kb >> 4) & 0xF) : (kb & 0xF);
    y_k[row * D + tid] = __float2bfloat16(scb_k[knib & K_MASK]);

    // ── y_v from v_idx_packed ──
    const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    vnib = nib_hi ? ((vb >> 4) & 0xF) : (vb & 0xF);
    y_v[row * D + tid] = __float2bfloat16(scb_v[vnib & V_MASK]);

    // ── qjl±1 from 1-bit packed ──
    const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
    const bool    qb1 = (qb >> (tid & 7)) & 1u;
    qjl_bf[row * D + tid] = qb1
        ? __float2bfloat16(1.0f) : __float2bfloat16(-1.0f);
}

// ── Mixed-output unpack: y_k bf16, qjl FP32, y_v bf16 ─────────────
// Phase 3B-α 3.5b: tq_dequant_cutlass needs y_k/y_v as bf16 (CUTLASS
// EVT input) and qjl as fp32 (precision-preserving Sr GEMM input).
// Original tq_unpack_packed_bf16 writes qjl as bf16, requiring a
// separate ~192 MB BW cast at 32K.  This variant fuses the cast into
// the unpack (qjl values are 1.0/-1.0 — bit-exact in any IEEE format).
template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_unpack_packed_mixed_kernel(
    const uint8_t*       __restrict__ k_idx_packed,
    const uint8_t*       __restrict__ k_qjl_packed,
    const uint8_t*       __restrict__ v_idx_packed,
    const float*         __restrict__ cb_k_mse,
    const float*         __restrict__ cb_v,
    __nv_bfloat16*       __restrict__ y_k,
    float*               __restrict__ qjl_f,
    __nv_bfloat16*       __restrict__ y_v,
    int M)
{
    constexpr int K_MASK = (1 << B_MSE_K) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_MSE_K;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
    __syncthreads();

    const int byte_idx = tid >> 1;
    const int nib_hi   = tid & 1;
    const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    knib = nib_hi ? ((kb >> 4) & 0xF) : (kb & 0xF);
    y_k[row * D + tid] = __float2bfloat16(scb_k[knib & K_MASK]);

    const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    vnib = nib_hi ? ((vb >> 4) & 0xF) : (vb & 0xF);
    y_v[row * D + tid] = __float2bfloat16(scb_v[vnib & V_MASK]);

    const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
    const bool    qb1 = (qb >> (tid & 7)) & 1u;
    qjl_f[row * D + tid] = qb1 ? 1.0f : -1.0f;
}

// ── Combine kernel ──────────────────────────────────────────────────
// Element-wise: K = norm·(K_mse + coef·rnorm·K_qjl); V = v_norm·V_unit
// All three K_mse/K_qjl/V_unit are (M, 256) bf16 from cuBLAS GEMMs.
// Output K, V (M, 256) bf16.
//
// Block: D threads, Grid: M blocks (M=S*NUM_KV).  BW-bound: read 4×bf16
// per coord + write 2×bf16 per coord; per-row norms scalar.

__global__ __launch_bounds__(D, 4)
void tq_combine_kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ k_mse,    // (M, 256)
    const __nv_bfloat16* __restrict__ k_qjl,    // (M, 256)
    const __nv_bfloat16* __restrict__ v_unit,   // (M, 256)
    const __half*        __restrict__ k_norm,   // (M,)
    const __half*        __restrict__ k_rnorm,  // (M,)
    const __half*        __restrict__ v_norm,   // (M,)
    __nv_bfloat16*       __restrict__ k_out,
    __nv_bfloat16*       __restrict__ v_out,
    int M, float coef)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const float kn  = __half2float(k_norm[row]);
    const float krn = __half2float(k_rnorm[row]);
    const float vn  = __half2float(v_norm[row]);
    const float coef_krn = coef * krn;

    const int off = row * D + tid;
    const float kmse = __bfloat162float(k_mse [off]);
    const float kqj  = __bfloat162float(k_qjl [off]);
    const float vun  = __bfloat162float(v_unit[off]);

    const float k_val = kn * (kmse + coef_krn * kqj);
    const float v_val = vn * vun;

    k_out[off] = __float2bfloat16(k_val);
    v_out[off] = __float2bfloat16(v_val);
}

// Variant taking fp32 GEMM outputs (better precision when rotation/jl
// are kept in fp32 and torch.matmul runs in TF32 mode).  Same shape,
// same layout, same launch bounds.
__global__ __launch_bounds__(D, 4)
void tq_combine_kv_fp32_in_kernel(
    const float*         __restrict__ k_mse,    // (M, 256) fp32
    const float*         __restrict__ k_qjl,    // (M, 256) fp32
    const float*         __restrict__ v_unit,   // (M, 256) fp32
    const __half*        __restrict__ k_norm,
    const __half*        __restrict__ k_rnorm,
    const __half*        __restrict__ v_norm,
    __nv_bfloat16*       __restrict__ k_out,
    __nv_bfloat16*       __restrict__ v_out,
    int M, float coef)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const float kn  = __half2float(k_norm[row]);
    const float krn = __half2float(k_rnorm[row]);
    const float vn  = __half2float(v_norm[row]);
    const float coef_krn = coef * krn;

    const int off = row * D + tid;
    const float k_val = kn * (k_mse[off] + coef_krn * k_qjl[off]);
    const float v_val = vn * v_unit[off];

    k_out[off] = __float2bfloat16(k_val);
    v_out[off] = __float2bfloat16(v_val);
}

// ── Write-side: BF16 K/V → packed B8 cache (Q_prod for K, Q_mse for V) ──
// Mirror of TurboQuantKVCache.write_kv but pure CUDA.  Capture-safe:
// no torch tensor allocations, all I/O via raw pointers + integers.
//
// Math (per (s, kv) row):
//   K (Q_prod, b_k_total bits, b_k_mse=b_k_total-1):
//     norm_k   = ||k||_2  (fp32)
//     k_unit   = k / norm_k                          (fp32)
//     y_k[j]   = sum_c k_unit[c] · rotation[j, c]    (GEMV: rotation^T)
//     idx_k[j] = argmin_b |y_k[j] - cb_k_mse[b]|     (b_mse-bit slot)
//     dq[c]    = sum_j cb_k_mse[idx_k[j]] · rotation[j, c]
//     r[c]     = k_unit[c] - dq[c]
//     rnorm_k  = ||r||_2
//     Sr[j]    = sum_c r[c] · jl[j, c]
//     qjl[j]   = (Sr[j] >= 0)
//     Pack idx_k 4-bit (256 nibbles → 128 B);  qjl 1-bit (256 → 32 B)
//   V (Q_mse, b_v bits): same minus the qjl correction.
//
// Decode hot path: S=1, num_kv=4 → 4 blocks total per layer × 16 layers
// = 64 blocks/forward.  Per-block cost ~30-50 μs.  Compute negligible
// vs the read+attention path; this kernel exists for capture safety,
// not raw speed.

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 1)
void tq_write_kv_packed_kernel(
    const __nv_bfloat16* __restrict__ k_in,        // (S, NUM_KV, D)
    const __nv_bfloat16* __restrict__ v_in,        // (S, NUM_KV, D)
    int s_start, int S,
    const float*  __restrict__ rotation,           // (D, D) fp32
    const float*  __restrict__ jl,                 // (D, D) fp32
    const float*  __restrict__ cb_k_mse,           // (2^B_MSE_K,) fp32
    const float*  __restrict__ cb_v,               // (2^B_V,) fp32
    uint8_t* __restrict__ k_idx_packed_layer,      // (max_seq, NUM_KV, 128)
    uint8_t* __restrict__ k_qjl_packed_layer,      // (max_seq, NUM_KV, 32)
    __half*  __restrict__ k_norm_layer,            // (max_seq, NUM_KV)
    __half*  __restrict__ k_rnorm_layer,           // (max_seq, NUM_KV)
    uint8_t* __restrict__ v_idx_packed_layer,      // (max_seq, NUM_KV, 128)
    __half*  __restrict__ v_norm_layer)
{
    constexpr int K_MASK = (1 << B_MSE_K) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_MSE_K;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;     // 0 .. S*NUM_KV-1
    const int s_local = row / NUM_KV;
    const int kv  = row - s_local * NUM_KV;
    const int s_global = s_start + s_local;
    const int tid = threadIdx.x;    // 0..D-1

    // ── smem ──
    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    __shared__ float skun[D];   // k_unit
    __shared__ float svun[D];   // v_unit
    __shared__ float syk [D];   // y_k = k_unit @ rot^T
    __shared__ float syv [D];   // y_v = v_unit @ rot^T
    __shared__ float sxk_dq[D]; // K dequant_unit (scratch for residual)
    __shared__ float sresid[D]; // residual
    __shared__ float sred[8];   // reduction scratch (one slot per warp)

    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];

    // ── load k, v as fp32 ──
    const int row_in = (s_local * NUM_KV + kv) * D + tid;
    const float kx = __bfloat162float(k_in[row_in]);
    const float vx = __bfloat162float(v_in[row_in]);

    // ── reductions: ||k||² and ||v||² ──
    auto warp_reduce = [](float v) {
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, o);
        return v;
    };
    float kx2 = kx * kx, vx2 = vx * vx;
    float kw = warp_reduce(kx2);
    float vw = warp_reduce(vx2);
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) { sred[wid] = kw; }
    __syncthreads();
    float kn2 = (tid < 8) ? sred[tid] : 0.f;
    kn2 = warp_reduce(kn2);
    if (tid == 0) sred[0] = kn2;
    __syncthreads();
    const float norm_k = sqrtf(sred[0]) + 1e-12f;
    __syncthreads();
    if (lane == 0) sred[wid] = vw;
    __syncthreads();
    float vn2 = (tid < 8) ? sred[tid] : 0.f;
    vn2 = warp_reduce(vn2);
    if (tid == 0) sred[0] = vn2;
    __syncthreads();
    const float norm_v = sqrtf(sred[0]) + 1e-12f;
    __syncthreads();

    // ── unit vectors (smem) ──
    skun[tid] = kx / norm_k;
    svun[tid] = vx / norm_v;
    __syncthreads();

    // ── GEMV1: y_k[j] = Σ_c k_unit[c] · rotation[j, c] ──
    //          y_v[j] = Σ_c v_unit[c] · rotation[j, c]
    // Each thread = one j.  Loads rotation[j, c] for c=0..D-1 (gmem,
    // L2-cached across blocks since rotation is shared).
    {
        float ack = 0.f, acv = 0.f;
        #pragma unroll 4
        for (int c = 0; c < D; ++c) {
            const float r = rotation[tid * D + c];
            ack += skun[c] * r;
            acv += svun[c] * r;
        }
        syk[tid] = ack;
        syv[tid] = acv;
    }
    __syncthreads();

    // ── argmin + V output write (V is done — pack idx_v) ──
    // K idx: 8-bucket search; V idx: 16-bucket search.
    int idx_k = 0, idx_v = 0;
    {
        const float yj = syk[tid];
        float best = fabsf(yj - scb_k[0]);
        for (int b = 1; b < CB_K_LEN; ++b) {
            const float d = fabsf(yj - scb_k[b]);
            if (d < best) { best = d; idx_k = b; }
        }
    }
    {
        const float yj = syv[tid];
        float best = fabsf(yj - scb_v[0]);
        for (int b = 1; b < CB_V_LEN; ++b) {
            const float d = fabsf(yj - scb_v[b]);
            if (d < best) { best = d; idx_v = b; }
        }
    }

    // ── store idx for use by GEMV2 (dequant) ──
    // Reuse syk / syv as integer index buffers via reinterpret? Simpler:
    // compute dequant_k via fresh GEMV reading cb_k[idx_k] per thread.
    // Each thread c needs sum_j cb_k[idx_k_for_j] · rot[j, c].
    // Scatter idx_k into smem first.
    __shared__ int sidx_k[D];
    sidx_k[tid] = idx_k & K_MASK;
    __syncthreads();

    // ── GEMV2: dequant_k_unit[c] = Σ_j cb_k[idx_k[j]] · rot[j, c] ──
    {
        float ac = 0.f;
        #pragma unroll 4
        for (int j = 0; j < D; ++j) {
            ac += scb_k[sidx_k[j]] * rotation[j * D + tid];
        }
        sxk_dq[tid] = ac;
    }
    __syncthreads();

    // ── residual + rnorm ──
    const float rc = skun[tid] - sxk_dq[tid];
    sresid[tid] = rc;
    __syncthreads();
    float r2 = rc * rc;
    float rw = warp_reduce(r2);
    if (lane == 0) sred[wid] = rw;
    __syncthreads();
    float rn2 = (tid < 8) ? sred[tid] : 0.f;
    rn2 = warp_reduce(rn2);
    if (tid == 0) sred[0] = rn2;
    __syncthreads();
    const float rnorm_k = sqrtf(sred[0]);

    // ── GEMV3: qjl[j] = sign(Σ_c r[c] · jl[j, c]) ──
    int qjl_bit = 0;
    {
        float ac = 0.f;
        #pragma unroll 4
        for (int c = 0; c < D; ++c) {
            ac += sresid[c] * jl[tid * D + c];
        }
        qjl_bit = (ac >= 0.f) ? 1 : 0;
    }

    // ── pack & write outputs ──
    // k_idx_packed: 4-bit packed (256 → 128 bytes per row).
    // Two threads (tid even, tid odd) cooperate: even = lo nib, odd = hi.
    {
        const int byte_idx = tid >> 1;
        const bool is_hi = tid & 1;
        // each thread holds one nibble; pair via warp shfl_xor
        unsigned packed_byte;
        unsigned my_nib = idx_k & K_MASK;
        unsigned other_nib =
            __shfl_xor_sync(0xffffffff, my_nib, 1);
        if (is_hi) packed_byte = (other_nib & 0xF) | ((my_nib & 0xF) << 4);
        else       packed_byte = (my_nib & 0xF) | ((other_nib & 0xF) << 4);
        if (!is_hi) {  // only even threads write
            k_idx_packed_layer[
                (s_global * NUM_KV + kv) * IDX_PACKED_BYTES + byte_idx]
                = (uint8_t)packed_byte;
        }
    }
    {
        const int byte_idx = tid >> 1;
        const bool is_hi = tid & 1;
        unsigned my_nib = idx_v & V_MASK;
        unsigned other_nib =
            __shfl_xor_sync(0xffffffff, my_nib, 1);
        unsigned packed_byte;
        if (is_hi) packed_byte = (other_nib & 0xF) | ((my_nib & 0xF) << 4);
        else       packed_byte = (my_nib & 0xF) | ((other_nib & 0xF) << 4);
        if (!is_hi) {
            v_idx_packed_layer[
                (s_global * NUM_KV + kv) * IDX_PACKED_BYTES + byte_idx]
                = (uint8_t)packed_byte;
        }
    }
    // qjl: 8 threads' bits → 1 byte (lsb-first).
    {
        const int byte_idx = tid >> 3;
        const int bit_idx  = tid & 7;
        unsigned bit = qjl_bit & 1u;
        // Collect 8 bits via warp shfl
        unsigned mask = bit << bit_idx;
        for (int o = 1; o < 8; o <<= 1)
            mask |= __shfl_xor_sync(0xffffffff, mask, o);
        if (bit_idx == 0) {
            k_qjl_packed_layer[
                (s_global * NUM_KV + kv) * QJL_PACKED_BYTES + byte_idx]
                = (uint8_t)(mask & 0xFF);
        }
    }
    // norms (one writer per row)
    if (tid == 0) {
        k_norm_layer [s_global * NUM_KV + kv] = __float2half(norm_k);
        k_rnorm_layer[s_global * NUM_KV + kv] = __float2half(rnorm_k);
        v_norm_layer [s_global * NUM_KV + kv] = __float2half(norm_v);
    }
}

// FP32-output variant: writes y_k, qjl±1, y_v as fp32 directly.
// Avoids the downstream `.float()` cast that allocated a 256 MB temp
// at 32K (768 MB BW: read bf16 + write fp32).  This kernel does the
// same total BW (384 MB write fp32 vs 192 MB bf16 → 2× write) but
// eliminates the bf16 read + the extra GPU-side cast.  Net BW saved:
// ~384 MB / call × 16 layers ≈ 6 GB / forward at 32K.
template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_unpack_packed_fp32_kernel(
    const uint8_t*       __restrict__ k_idx_packed,
    const uint8_t*       __restrict__ k_qjl_packed,
    const uint8_t*       __restrict__ v_idx_packed,
    const float*         __restrict__ cb_k_mse,
    const float*         __restrict__ cb_v,
    float*               __restrict__ y_k,
    float*               __restrict__ qjl_f,
    float*               __restrict__ y_v,
    int M)
{
    constexpr int K_MASK = (1 << B_MSE_K) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_MSE_K;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
    __syncthreads();

    const int byte_idx = tid >> 1;
    const int nib_hi   = tid & 1;
    const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    knib = nib_hi ? ((kb >> 4) & 0xF) : (kb & 0xF);
    y_k[row * D + tid] = scb_k[knib & K_MASK];

    const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    vnib = nib_hi ? ((vb >> 4) & 0xF) : (vb & 0xF);
    y_v[row * D + tid] = scb_v[vnib & V_MASK];

    const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
    const bool    qb1 = (qb >> (tid & 7)) & 1u;
    qjl_f[row * D + tid] = qb1 ? 1.0f : -1.0f;
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3A B9-S10: cuBLAS-bit-exact write path (graph-capture safe).
//
// The single-kernel write (tq_write_kv_packed) had ~14% qjl byte
// mismatch with the slow Python ref because its hand-rolled fp32 GEMV
// did not match cuBLAS's TF32 reduction order on borderline Sr[j]≈0
// cases.  Fix: split write into (4 CUDA kernels for the
// non-GEMM math) + (3 cuBLAS GEMMs via torch.matmul, which are
// bit-identical to the Python reference's torch.matmul).
//
//   K1  bf16 K,V → fp32 k_unit, v_unit + fp32 norm_k, norm_v
//   GEMM A (cuBLAS): [k_unit; v_unit] @ rotation^T → y_k, y_v  fp32
//   K2  argmin → idx_k, idx_v ; pack 4-bit ; gather cb_k[idx_k] for dq
//   GEMM B (cuBLAS): cb_k[idx_k] @ rotation → dq_k             fp32
//   K3  residual = k_unit - dq_k ; rnorm_k = ‖residual‖
//   GEMM C (cuBLAS): residual @ jl^T → Sr                       fp32
//   K4  pack qjl from sign(Sr) ; write norm_k, rnorm_k, v_norm to cache
//
// All four kernels are block-per-row, 256 threads; per-block work is
// tiny (256-element reductions / scatters / packs).  No torch tensor
// allocations; capture-safe.

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_write_k1_unit_norm_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    float* __restrict__ k_unit_out,
    float* __restrict__ v_unit_out,
    float* __restrict__ norm_k_out,
    float* __restrict__ norm_v_out,
    int M)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int off = row * D + tid;
    const float kx = __bfloat162float(k_in[off]);
    const float vx = __bfloat162float(v_in[off]);

    __shared__ float reduce[8];
    auto warp_red = [](float v) {
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, o);
        return v;
    };
    const int wid  = tid >> 5;
    const int lane = tid & 31;

    float ks = warp_red(kx * kx);
    if (lane == 0) reduce[wid] = ks;
    __syncthreads();
    float kn2 = (tid < 8) ? reduce[tid] : 0.f;
    kn2 = warp_red(kn2);
    if (tid == 0) reduce[0] = kn2;
    __syncthreads();
    const float norm_k = sqrtf(reduce[0]);
    __syncthreads();
    float vs = warp_red(vx * vx);
    if (lane == 0) reduce[wid] = vs;
    __syncthreads();
    float vn2 = (tid < 8) ? reduce[tid] : 0.f;
    vn2 = warp_red(vn2);
    if (tid == 0) reduce[0] = vn2;
    __syncthreads();
    const float norm_v = sqrtf(reduce[0]);

    // Match Python: x_unit = x / (norm + eps), but stored norm = ||x||
    const float inv_kn = 1.0f / (norm_k + 1e-12f);
    const float inv_vn = 1.0f / (norm_v + 1e-12f);
    k_unit_out[off] = kx * inv_kn;
    v_unit_out[off] = vx * inv_vn;
    if (tid == 0) {
        norm_k_out[row] = norm_k;
        norm_v_out[row] = norm_v;
    }
}

template <int B_MSE_K, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_write_k2_argmin_pack_kernel(
    const float*   __restrict__ y_k,
    const float*   __restrict__ y_v,
    const float*   __restrict__ cb_k_mse,
    const float*   __restrict__ cb_v,
    uint8_t*       __restrict__ k_idx_packed_layer,
    uint8_t*       __restrict__ v_idx_packed_layer,
    float*         __restrict__ dq_in,            // (M, D) — cb_k[idx_k]
    int s_start, int num_kv, int M)
{
    constexpr int K_MASK = (1 << B_MSE_K) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_MSE_K;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;          // 0 .. M-1; row = s*num_kv + kv
    if (row >= M) return;
    const int s_local = row / num_kv;
    const int kv = row - s_local * num_kv;
    const int s_global = s_start + s_local;
    const int tid = threadIdx.x;
    const int off = row * D + tid;

    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];
    __syncthreads();

    // K argmin (8 buckets)
    int idx_k = 0;
    {
        const float yj = y_k[off];
        float best = fabsf(yj - scb_k[0]);
        for (int b = 1; b < CB_K_LEN; ++b) {
            const float dst = fabsf(yj - scb_k[b]);
            if (dst < best) { best = dst; idx_k = b; }
        }
        idx_k &= K_MASK;
    }
    // V argmin (16 buckets)
    int idx_v = 0;
    {
        const float yj = y_v[off];
        float best = fabsf(yj - scb_v[0]);
        for (int b = 1; b < CB_V_LEN; ++b) {
            const float dst = fabsf(yj - scb_v[b]);
            if (dst < best) { best = dst; idx_v = b; }
        }
        idx_v &= V_MASK;
    }
    // Gather cb_k[idx_k] for the next GEMM (dequant).
    dq_in[off] = scb_k[idx_k];

    // 4-bit pack: pair (tid even = lo, tid odd = hi) into one byte.
    {
        const int byte_idx = tid >> 1;
        const bool is_hi = tid & 1;
        unsigned other_k = __shfl_xor_sync(0xffffffff, idx_k, 1);
        unsigned packed_k =
            is_hi ? ((other_k & 0xF) | ((idx_k & 0xF) << 4))
                  : ((idx_k   & 0xF) | ((other_k & 0xF) << 4));
        if (!is_hi) {
            k_idx_packed_layer[
                (s_global * num_kv + kv) * IDX_PACKED_BYTES + byte_idx]
                = (uint8_t)packed_k;
        }
        unsigned other_v = __shfl_xor_sync(0xffffffff, idx_v, 1);
        unsigned packed_v =
            is_hi ? ((other_v & 0xF) | ((idx_v & 0xF) << 4))
                  : ((idx_v   & 0xF) | ((other_v & 0xF) << 4));
        if (!is_hi) {
            v_idx_packed_layer[
                (s_global * num_kv + kv) * IDX_PACKED_BYTES + byte_idx]
                = (uint8_t)packed_v;
        }
    }
}

__global__ __launch_bounds__(D, 4)
void tq_write_k3_residual_rnorm_kernel(
    const float* __restrict__ k_unit,
    const float* __restrict__ dq_k,
    float*       __restrict__ residual,
    float*       __restrict__ rnorm_k,
    int M)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int off = row * D + tid;
    const float r = k_unit[off] - dq_k[off];
    residual[off] = r;

    __shared__ float reduce[8];
    auto warp_red = [](float v) {
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, o);
        return v;
    };
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    float rs = warp_red(r * r);
    if (lane == 0) reduce[wid] = rs;
    __syncthreads();
    float rn2 = (tid < 8) ? reduce[tid] : 0.f;
    rn2 = warp_red(rn2);
    if (tid == 0) rnorm_k[row] = sqrtf(rn2);
}

__global__ __launch_bounds__(D, 4)
void tq_write_k4_qjl_norms_kernel(
    const float* __restrict__ Sr,
    const float* __restrict__ norm_k,
    const float* __restrict__ rnorm_k,
    const float* __restrict__ norm_v,
    uint8_t*     __restrict__ k_qjl_packed_layer,
    __half*      __restrict__ k_norm_layer,
    __half*      __restrict__ k_rnorm_layer,
    __half*      __restrict__ v_norm_layer,
    int s_start, int num_kv, int M)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int s_local = row / num_kv;
    const int kv = row - s_local * num_kv;
    const int s_global = s_start + s_local;
    const int tid = threadIdx.x;
    const int off = row * D + tid;

    // 1-bit pack qjl from sign(Sr)
    const unsigned bit = (Sr[off] >= 0.f) ? 1u : 0u;
    const int byte_idx = tid >> 3;
    const int bit_idx  = tid & 7;
    unsigned mask = bit << bit_idx;
    for (int o = 1; o < 8; o <<= 1)
        mask |= __shfl_xor_sync(0xffffffff, mask, o);
    if (bit_idx == 0) {
        k_qjl_packed_layer[
            (s_global * num_kv + kv) * QJL_PACKED_BYTES + byte_idx]
            = (uint8_t)(mask & 0xFF);
    }
    if (tid == 0) {
        const int dst = s_global * num_kv + kv;
        k_norm_layer [dst] = __float2half(norm_k[row]);
        k_rnorm_layer[dst] = __float2half(rnorm_k[row]);
        v_norm_layer [dst] = __float2half(norm_v[row]);
    }
}

}  // namespace

// ── C ABI ─────────────────────────────────────────────────────────────
extern "C"
void tq_write_k1_unit_norm_launch(
    const void* k_in, const void* v_in,
    void* k_unit_out, void* v_unit_out,
    void* norm_k_out, void* norm_v_out,
    int M, int b_k_mse, int b_v, cudaStream_t stream)
{
    dim3 grid(M); dim3 block(256);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_write_k1_unit_norm_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)k_in,
            (const __nv_bfloat16*)v_in,
            (float*)k_unit_out, (float*)v_unit_out,
            (float*)norm_k_out, (float*)norm_v_out, M);
    };
    launch(std::integral_constant<int,3>{}, std::integral_constant<int,4>{});
}

extern "C"
void tq_write_k2_argmin_pack_launch(
    const void* y_k, const void* y_v,
    const void* cb_k_mse, const void* cb_v,
    void* k_idx_packed_layer, void* v_idx_packed_layer,
    void* dq_in,
    int s_start, int num_kv, int M,
    int b_k_mse, int b_v, cudaStream_t stream)
{
    dim3 grid(M); dim3 block(256);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_write_k2_argmin_pack_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const float*)y_k, (const float*)y_v,
            (const float*)cb_k_mse, (const float*)cb_v,
            (uint8_t*)k_idx_packed_layer,
            (uint8_t*)v_idx_packed_layer,
            (float*)dq_in,
            s_start, num_kv, M);
    };
    if      (b_k_mse == 3 && b_v == 4) launch(std::integral_constant<int,3>{}, std::integral_constant<int,4>{});
    else if (b_k_mse == 2 && b_v == 3) launch(std::integral_constant<int,2>{}, std::integral_constant<int,3>{});
    else                                launch(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
}

extern "C"
void tq_write_k3_residual_rnorm_launch(
    const void* k_unit, const void* dq_k,
    void* residual, void* rnorm_k,
    int M, cudaStream_t stream)
{
    dim3 grid(M); dim3 block(256);
    tq_write_k3_residual_rnorm_kernel<<<grid, block, 0, stream>>>(
        (const float*)k_unit, (const float*)dq_k,
        (float*)residual, (float*)rnorm_k, M);
}

extern "C"
void tq_write_k4_qjl_norms_launch(
    const void* Sr,
    const void* norm_k, const void* rnorm_k, const void* norm_v,
    void* k_qjl_packed_layer,
    void* k_norm_layer, void* k_rnorm_layer, void* v_norm_layer,
    int s_start, int num_kv, int M, cudaStream_t stream)
{
    dim3 grid(M); dim3 block(256);
    tq_write_k4_qjl_norms_kernel<<<grid, block, 0, stream>>>(
        (const float*)Sr,
        (const float*)norm_k, (const float*)rnorm_k, (const float*)norm_v,
        (uint8_t*)k_qjl_packed_layer,
        (__half*)k_norm_layer, (__half*)k_rnorm_layer, (__half*)v_norm_layer,
        s_start, num_kv, M);
}

extern "C"
void tq_write_kv_packed_launch(
    const void* k_in, const void* v_in,
    int s_start, int S,
    const void* rotation, const void* jl,
    const void* cb_k_mse, const void* cb_v,
    void* k_idx_packed_layer, void* k_qjl_packed_layer,
    void* k_norm_layer, void* k_rnorm_layer,
    void* v_idx_packed_layer, void* v_norm_layer,
    int b_k_mse, int b_v,
    cudaStream_t stream)
{
    dim3 grid(S * 4);   // NUM_KV=4
    dim3 block(256);    // D=256
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_write_kv_packed_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)k_in,
            (const __nv_bfloat16*)v_in,
            s_start, S,
            (const float*)rotation,
            (const float*)jl,
            (const float*)cb_k_mse,
            (const float*)cb_v,
            (uint8_t*)k_idx_packed_layer,
            (uint8_t*)k_qjl_packed_layer,
            (__half*)k_norm_layer,
            (__half*)k_rnorm_layer,
            (uint8_t*)v_idx_packed_layer,
            (__half*)v_norm_layer);
    };
    if (b_k_mse == 3 && b_v == 4) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,4>{});
    } else if (b_k_mse == 2 && b_v == 3) {
        launch(std::integral_constant<int,2>{},
                std::integral_constant<int,3>{});
    } else if (b_k_mse == 3 && b_v == 3) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,3>{});
    } else {
        launch(std::integral_constant<int,4>{},
                std::integral_constant<int,4>{});
    }
}

extern "C"
void tq_unpack_packed_bf16_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k, void* qjl_bf, void* y_v,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_unpack_packed_bf16_kernel<K_BITS, V_BITS><<<grid, block, 0, stream>>>(
            (const uint8_t*)k_idx_packed,
            (const uint8_t*)k_qjl_packed,
            (const uint8_t*)v_idx_packed,
            (const float*)cb_k_mse,
            (const float*)cb_v,
            (__nv_bfloat16*)y_k, (__nv_bfloat16*)qjl_bf, (__nv_bfloat16*)y_v,
            M);
    };
    if (b_k_mse == 3 && b_v == 4) {
        launch(std::integral_constant<int,3>{}, std::integral_constant<int,4>{});
    } else if (b_k_mse == 2 && b_v == 3) {
        launch(std::integral_constant<int,2>{}, std::integral_constant<int,3>{});
    } else if (b_k_mse == 3 && b_v == 3) {
        launch(std::integral_constant<int,3>{}, std::integral_constant<int,3>{});
    } else {
        launch(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
    }
}

extern "C"
void tq_unpack_packed_mixed_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k_bf16, void* qjl_fp32, void* y_v_bf16,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_unpack_packed_mixed_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const uint8_t*)k_idx_packed,
            (const uint8_t*)k_qjl_packed,
            (const uint8_t*)v_idx_packed,
            (const float*)cb_k_mse,
            (const float*)cb_v,
            (__nv_bfloat16*)y_k_bf16,
            (float*)qjl_fp32,
            (__nv_bfloat16*)y_v_bf16,
            M);
    };
    if (b_k_mse == 3 && b_v == 4) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,4>{});
    } else if (b_k_mse == 2 && b_v == 3) {
        launch(std::integral_constant<int,2>{},
                std::integral_constant<int,3>{});
    } else if (b_k_mse == 3 && b_v == 3) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,3>{});
    } else {
        launch(std::integral_constant<int,4>{},
                std::integral_constant<int,4>{});
    }
}

extern "C"
void tq_unpack_packed_fp32_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* v_idx_packed,
    const void* cb_k_mse, const void* cb_v,
    void* y_k, void* qjl_f, void* y_v,
    int M, int b_k_mse, int b_v,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_unpack_packed_fp32_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const uint8_t*)k_idx_packed,
            (const uint8_t*)k_qjl_packed,
            (const uint8_t*)v_idx_packed,
            (const float*)cb_k_mse,
            (const float*)cb_v,
            (float*)y_k, (float*)qjl_f, (float*)y_v,
            M);
    };
    if (b_k_mse == 3 && b_v == 4) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,4>{});
    } else if (b_k_mse == 2 && b_v == 3) {
        launch(std::integral_constant<int,2>{},
                std::integral_constant<int,3>{});
    } else if (b_k_mse == 3 && b_v == 3) {
        launch(std::integral_constant<int,3>{},
                std::integral_constant<int,3>{});
    } else {
        launch(std::integral_constant<int,4>{},
                std::integral_constant<int,4>{});
    }
}

extern "C"
void tq_combine_kv_bf16_launch(
    const void* k_mse, const void* k_qjl, const void* v_unit,
    const void* k_norm, const void* k_rnorm, const void* v_norm,
    void* k_out, void* v_out,
    int M, float coef,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    tq_combine_kv_bf16_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)k_mse,
        (const __nv_bfloat16*)k_qjl,
        (const __nv_bfloat16*)v_unit,
        (const __half*)k_norm,
        (const __half*)k_rnorm,
        (const __half*)v_norm,
        (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        M, coef);
}

// ── FP32 act × FP32 weight → FP32 out, TF32 tensor cores (cuBLAS) ──
// torch.matmul(fp32, fp32) with allow_tf32=False (PyTorch default)
// runs on FP32 ALU (~91 TFLOPS); we need tensor cores for the long-ctx
// hot path.  Setting allow_tf32 globally would affect the rest of the
// model.  This wrapper enables TF32 only for our dequant GEMMs.
//
// Precision: TF32 multiplicands have 10-bit mantissa (vs FP32 23-bit);
// FP32 accumulator preserved.  For the random-orthogonal rotation
// matrix's small magnitudes (~N(0, 1/d)), TF32 rounding accumulates to
// ~1e-2 relative — comparable to BF16's 7-bit-mantissa quantum but
// without rounding the products before accumulation.  Validated to
// keep token argmax stable in the long-context validation suite.
extern "C"
void tq_fp32_gemm_tf32_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b_fp32, CUDA_R_32F, N,
        a_fp32, CUDA_R_32F, K,
        &beta,
        c_fp32, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

extern "C"
void tq_fp32_gemm_fp32_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b_fp32, CUDA_R_32F, N,
        a_fp32, CUDA_R_32F, K,
        &beta,
        c_fp32, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}

extern "C"
void tq_fp32_gemm_fp32_bt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b_fp32, CUDA_R_32F, K,
        a_fp32, CUDA_R_32F, K,
        &beta,
        c_fp32, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}

extern "C"
void tq_fp32_gemm_lt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
    cudaStream_t stream);

extern "C"
void tq_fp32_gemm_lt_bt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
    cudaStream_t stream);

extern "C"
void tq_fp32_gemm_lt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    tq_fp32_gemm_lt_algo_launch(
        a_fp32, b_fp32, c_fp32, M, N, K, 0, stream);
}

extern "C"
void tq_fp32_gemm_lt_bt_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    tq_fp32_gemm_lt_bt_algo_launch(
        a_fp32, b_fp32, c_fp32, M, N, K, 0, stream);
}

extern "C"
void tq_fp32_gemm_lt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
    cudaStream_t stream)
{
    static cublasLtHandle_t lt = nullptr;
    static void* workspace = nullptr;
    static size_t workspace_size = 32 * 1024 * 1024;
    if (lt == nullptr) {
        cublasLtCreate(&lt);
        cudaMalloc(&workspace, workspace_size);
    }

    TqLtFp32Plan* plan = tq_get_lt_fp32_plan(
        lt, M, N, K, algo_idx, false, workspace_size);
    const float alpha = 1.0f, beta = 0.0f;
    if (plan->valid) {
        cublasLtMatmul(
            lt, plan->desc, &alpha,
            a_fp32, plan->a_desc,
            b_fp32, plan->b_desc,
            &beta,
            c_fp32, plan->c_desc,
            c_fp32, plan->c_desc,
            &plan->algo, workspace, workspace_size, stream);
    }
}

extern "C"
void tq_fp32_gemm_lt_bt_algo_launch(
    const void* a_fp32, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    int algo_idx,
    cudaStream_t stream)
{
    static cublasLtHandle_t lt = nullptr;
    static void* workspace = nullptr;
    static size_t workspace_size = 32 * 1024 * 1024;
    if (lt == nullptr) {
        cublasLtCreate(&lt);
        cudaMalloc(&workspace, workspace_size);
    }

    TqLtFp32Plan* plan = tq_get_lt_fp32_plan(
        lt, M, N, K, algo_idx, true, workspace_size);
    const float alpha = 1.0f, beta = 0.0f;
    if (plan->valid) {
        cublasLtMatmul(
            lt, plan->desc, &alpha,
            a_fp32, plan->a_desc,
            b_fp32, plan->b_desc,
            &beta,
            c_fp32, plan->c_desc,
            c_fp32, plan->c_desc,
            &plan->algo, workspace, workspace_size, stream);
    }
}

// ── BF16 act × FP32 weight → FP32 out  (cuBLAS gemmEx mixed-type) ──
// Replaces the `yk.float() @ rotation_fp32` path: avoids the 256 MB
// .float() input temp + cast launch.  Activation stays bf16 (carries
// the codebook quantization, no precision lost vs upcasting), weight
// stays fp32 (preserves the random-orthogonal rotation precision —
// quantizing it to bf16 was what broke B8 token-argmax 2/16 in the
// earlier bf16-bf16 attempt).  Accumulator + output: FP32.
//
// Layout: A row-major (M,K), B row-major (K,N), C row-major (M,N).
// cuBLAS is col-major; to compute C = A·B in row-major we issue
// gemm(N, M, K) with B^T·A^T = C^T using leading dims = row strides.
extern "C"
void tq_bf16_fp32_gemm_launch(
    const void* a_bf16, const void* b_fp32,
    void* c_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b_fp32, CUDA_R_32F, N,
        a_bf16, CUDA_R_16BF, K,
        &beta,
        c_fp32, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
}

extern "C"
void tq_combine_kv_fp32_in_launch(
    const void* k_mse, const void* k_qjl, const void* v_unit,
    const void* k_norm, const void* k_rnorm, const void* v_norm,
    void* k_out, void* v_out,
    int M, float coef,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    tq_combine_kv_fp32_in_kernel<<<grid, block, 0, stream>>>(
        (const float*)k_mse,
        (const float*)k_qjl,
        (const float*)v_unit,
        (const __half*)k_norm,
        (const __half*)k_rnorm,
        (const __half*)v_norm,
        (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        M, coef);
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3B-α S3: single-launch fused dequant.
//
// Replaces the [unpack → cuBLAS GEMM × 2 → combine] read_kv_fast pipe
// with one kernel.  Per row r, per output coord c (1 thread):
//
//   y_k[c]    = cb_k[ idx_k_packed[r, c] ]      (codebook lookup)
//   y_v[c]    = cb_v[ idx_v_packed[r, c] ]
//   q[c]      = (qjl_packed[r, c] ? +1 : -1)
//   K_pre[c]  = sum_j y_k[j]    * rotation[j, c]
//   V_pre[c]  = sum_j y_v[j]    * rotation[j, c]
//   Sr[c]     = sum_j q[j]      * jl[j, c]
//   K_out[c]  = norm_k * (K_pre[c] + coef * rnorm_k * Sr[c])    bf16
//   V_out[c]  = norm_v * V_pre[c]                                bf16
//
// Block: 256 threads = D output coords.  Grid: M = (kv_seq * NUM_KV).
// Per-block staging in smem:
//   scb_k:  ≤ 16 fp32  (codebook)
//   scb_v:  ≤ 16 fp32
//   syk:    256 fp32   (1 KB)  — y_k for this row
//   syv:    256 fp32   (1 KB)  — y_v
//   sqjl:   256 fp32   (1 KB)  — qjl ±1
//   total:  ≈ 3.1 KB   (fits comfortably; no Π/S in smem — read from L2)
//
// Per-CTA work:  3 × 256 × 256 = 196608 FMAs.  Per launch:  M × 65K FMAs.
// At kv_seq=32K, M=128K → 8.4 GFMA / layer.  fp32 ALU @ ~30 TFLOPS →
// ~280 μs / layer.  Π and S read from gmem, ~256 KB each, L2-hot after
// first CTA — effectively free per layer.
//
// Compared to the multi-kernel B9 path:
//   B9 32K, per layer:  unpack 0.10 + 2× GEMM 0.50 + combine 0.08 = 0.68 ms
//   Plus fp32 intermediate BW: ~1 GB write + 1 GB read = 1.1 ms / layer
//   α-3 fused, per layer: ~0.3 ms FMA + ~0.07 ms BW (only packed in,
//                          bf16 out) = ~0.37 ms / layer.
//   Saves ~0.3 ms / layer × 16 = ~5 ms at 32K decode forward.
//
// Bit-stable across M: every row uses identical FMA reduction order
// (compile-time-fixed inner loop), so subsequent windowed/incremental
// callers can use the same kernel without TF32-cuBLAS schedule drift.

template <int B_K_MSE, int B_V>
__global__ __launch_bounds__(D, 4)
void tq_dequant_kv_fused_kernel(
    const uint8_t*       __restrict__ k_idx_packed,   // (M, 128)
    const uint8_t*       __restrict__ k_qjl_packed,   // (M, 32)
    const __half*        __restrict__ k_norm_in,      // (M,)
    const __half*        __restrict__ k_rnorm_in,     // (M,)
    const uint8_t*       __restrict__ v_idx_packed,   // (M, 128)
    const __half*        __restrict__ v_norm_in,      // (M,)
    const float*         __restrict__ rotation,       // (256, 256) fp32
    const float*         __restrict__ jl,             // (256, 256) fp32
    const float*         __restrict__ cb_k_mse,       // (1<<B_K_MSE,)
    const float*         __restrict__ cb_v,           // (1<<B_V,)
    __nv_bfloat16*       __restrict__ k_out,          // (M, 256) bf16
    __nv_bfloat16*       __restrict__ v_out,          // (M, 256) bf16
    int M, float coef)
{
    constexpr int K_MASK = (1 << B_K_MSE) - 1;
    constexpr int V_MASK = (1 << B_V) - 1;
    constexpr int CB_K_LEN = 1 << B_K_MSE;
    constexpr int CB_V_LEN = 1 << B_V;

    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    __shared__ float scb_k[CB_K_MAX];
    __shared__ float scb_v[CB_V_MAX];
    __shared__ float syk[D];
    __shared__ float syv[D];
    __shared__ float sqjl[D];

    if (tid < CB_K_LEN) scb_k[tid] = cb_k_mse[tid];
    if (tid < CB_V_LEN) scb_v[tid] = cb_v[tid];

    // Decode one element of y_k / y_v / qjl per thread.
    const int byte_idx = tid >> 1;
    const int nib_hi   = tid & 1;
    const uint8_t kb = k_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    knib = nib_hi ? ((kb >> 4) & 0xF) : (kb & 0xF);
    const uint8_t vb = v_idx_packed[row * IDX_PACKED_BYTES + byte_idx];
    const int    vnib = nib_hi ? ((vb >> 4) & 0xF) : (vb & 0xF);
    const uint8_t qb = k_qjl_packed[row * QJL_PACKED_BYTES + (tid >> 3)];
    const bool    qb1 = (qb >> (tid & 7)) & 1u;

    __syncthreads();
    syk[tid]  = scb_k[knib & K_MASK];
    syv[tid]  = scb_v[vnib & V_MASK];
    sqjl[tid] = qb1 ? 1.0f : -1.0f;
    __syncthreads();

    // Inner reduction: K_pre[c=tid] = sum_j syk[j] * rotation[j, tid]
    // Same for V_pre and Sr (qjl @ jl).  Fp32 ALU; deterministic FMA
    // order across all rows / all M values (compile-time loop bounds).
    // Bottleneck: fp32 ALU throughput on consumer Blackwell (~50 TF)
    // vs cuBLAS TF32 tensor cores (~660 TF) used by the B9 path.
    // This kernel is ~4× slower than B9 read_kv_fast at every ctx
    // (gate Gα-3a verifies cos=1.0 vs B9 stage; perf is in §α-3
    // close).  A wmma/tensor-core variant is the next iteration.
    float k_acc = 0.f, v_acc = 0.f, q_acc = 0.f;
    #pragma unroll 8
    for (int j = 0; j < D; ++j) {
        const float r = rotation[j * D + tid];
        const float l = jl[j * D + tid];
        k_acc = fmaf(syk[j],  r, k_acc);
        v_acc = fmaf(syv[j],  r, v_acc);
        q_acc = fmaf(sqjl[j], l, q_acc);
    }

    const float kn  = __half2float(k_norm_in [row]);
    const float krn = __half2float(k_rnorm_in[row]);
    const float vn  = __half2float(v_norm_in [row]);

    const float k_val = kn * (k_acc + coef * krn * q_acc);
    const float v_val = vn * v_acc;

    const int off = row * D + tid;
    k_out[off] = __float2bfloat16(k_val);
    v_out[off] = __float2bfloat16(v_val);
}

extern "C"
void tq_dequant_kv_fused_launch(
    const void* k_idx_packed, const void* k_qjl_packed,
    const void* k_norm, const void* k_rnorm,
    const void* v_idx_packed, const void* v_norm,
    const void* rotation, const void* jl,
    const void* cb_k_mse, const void* cb_v,
    void* k_out, void* v_out,
    int M, float coef,
    int b_k_mse, int b_v,
    cudaStream_t stream)
{
    dim3 grid(M);
    dim3 block(D);
    auto launch = [&](auto BMK, auto BV) {
        constexpr int K_BITS = decltype(BMK)::value;
        constexpr int V_BITS = decltype(BV)::value;
        tq_dequant_kv_fused_kernel<K_BITS, V_BITS>
            <<<grid, block, 0, stream>>>(
            (const uint8_t*)k_idx_packed,
            (const uint8_t*)k_qjl_packed,
            (const __half*)k_norm,
            (const __half*)k_rnorm,
            (const uint8_t*)v_idx_packed,
            (const __half*)v_norm,
            (const float*)rotation,
            (const float*)jl,
            (const float*)cb_k_mse,
            (const float*)cb_v,
            (__nv_bfloat16*)k_out,
            (__nv_bfloat16*)v_out,
            M, coef);
    };
    if      (b_k_mse == 3 && b_v == 4) launch(std::integral_constant<int,3>{}, std::integral_constant<int,4>{});
    else if (b_k_mse == 2 && b_v == 3) launch(std::integral_constant<int,2>{}, std::integral_constant<int,3>{});
    else if (b_k_mse == 3 && b_v == 3) launch(std::integral_constant<int,3>{}, std::integral_constant<int,3>{});
    else                                launch(std::integral_constant<int,4>{}, std::integral_constant<int,4>{});
}
