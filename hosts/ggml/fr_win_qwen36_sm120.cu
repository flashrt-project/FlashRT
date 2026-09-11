// SPDX-License-Identifier: Apache-2.0
//
// FlashRT ggml adapter — second target: RTX 5090 (SM120) + Qwen3.6-35B-A3B
// window set. Pattern-matched subgraph windows over ggml-cuda's fuse hook:
//
//   - fused-region GEMVs (GDN in_proj / attn qkv) through the NVFP4 W4A4
//     warp-split-K blockscale MMA GEMV (f32 act -> NVFP4 quant + swizzled
//     SFA -> GEMV -> staging served to all region members)
//   - GDN cell span (conv + gated delta net + epilogue, M<=4 with
//     per-token state/conv snapshots and checkpoint replay)
//   - MoE expert span K0/K1/K2 consuming ggml's native K-quant blocks via
//     its own vec_dot device functions (bit-exact q8_1 activation clone),
//     shared-expert folded, M<=4
//   - out-proj / lm-head / router windows, spec-draft head serving
//
// All launches join the host's PDL chain (ggml_cuda_kernel_launch). Weights
// for the FP4 windows come from side-band packs (FRT_REGIONS_PACK /
// FRT_HEAD_PACK) until the in-process repack cache lands (see DEVELOPMENT).
// Runtime switches: FRT_* per window (see the target section in USAGE.md).
// Kernels ported from FlashRT csrc (Apache-2.0, same authorship).

#include "common.cuh"
#include "vecdotq.cuh"
#include "convert.cuh"

// Heavy math comes from csrc (single source; the adapter only translates):
// the M-rows activation quantizer and the warp-split-K W4A4 GEMV.
#include "../../csrc/quantize/f32_act_to_nvfp4_swizzled_mrows_sm120.cuh"
#include "../../csrc/kernels/fp4_w4a4_mma_warpsplit_mrows_f32out_sm120.cuh"

// Model-specific constants come from the binding (single source:
// flash_rt/catalog/bindings/llamacpp_qwen36_35b_sm120.yaml); regenerate
// the header with tools/gen_binding_header.py after editing the binding.
#include "fr_binding_qwen36_35b_sm120.h"

// ---- switch semantics -----------------------------------------------------
// Compiled-in windows default ON. Three layers of control, most specific wins:
//   FRT_<X>_SWAP=0/1              per-window A/B override (historic names)
//   GGML_FLASHRT_NO_<X>=1         per-window disable
//   GGML_CUDA_FLASHRT_DISABLE=1   whole layer off (stock llama.cpp)
// Archive windows (judged off on this target) stay opt-in via their FRT_*
// switches only. The full-tier head swap changes output quality, so it too
// stays opt-in (FRT_HEAD_SWAP=1).
static bool frt_layer_enabled(void) {
    static int on = -1;
    if (on < 0) { const char * s = getenv("GGML_CUDA_FLASHRT_DISABLE"); on = (s && s[0] == '1') ? 0 : 1; }
    return on == 1;
}
static bool frt_window_on(const char * frt_env, const char * no_env) {
    if (!frt_layer_enabled()) return false;
    const char * f = getenv(frt_env);
    if (f) return f[0] == '1';
    const char * n = getenv(no_env);
    return !(n && n[0] == '1');
}
// The spec draft model's MTP layer reuses the target's tensor-name scheme at
// layer indices >= n_layer, and its attn q/k/v happen to match the kind-1
// pack shapes. Serving the draft's projections from FP4 only moves
// acceptance, never output — but it is a separate judgment call, so it is
// gated off by default until judged (FRT_DRAFT_REGIONS=1 to enable).
static bool frt_layer_in_scope(int layer) {
    static int draft_on = -1;
    if (draft_on < 0) { const char * e = getenv("FRT_DRAFT_REGIONS"); draft_on = (e && e[0] == '1') ? 1 : 0; }
    return layer < frt_binding::n_layer || draft_on == 1;
}

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace frt {

// ---------------- activation quantize (f32 row -> NVFP4 + swizzled SFA) ---

// device vocabulary of this window file, backed by the csrc single source
__device__ __forceinline__ int sfa_offset_128x64(int row, int k, int dim) {
    return flash_rt::quantize::nvfp4_sfa_offset_128x64(row, k, dim);
}
__device__ __forceinline__ uint8_t fp32_to_e2m1(float x) {
    return flash_rt::quantize::nvfp4_act_f32_to_e2m1(x);
}
template <int THREADS>
__device__ __forceinline__ void quant_act_fp4_f32_body(
        const float * __restrict__ x, uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sfa, int D, int row = 0) {
    flash_rt::quantize::f32_act_to_nvfp4_row<THREADS>(x, dst_packed, dst_sfa, D, row);
}

#ifdef GGML_CUDA_USE_PDL
constexpr bool frt_launch_pdl = true;
#else
constexpr bool frt_launch_pdl = false;
#endif

// standalone M-rows activation quantize through the csrc entry
static void frt_quant_act_launch(const float * x, void * dst_packed, void * dst_sfa,
        int D, int M, int64_t x_srow, cudaStream_t stream) {
    const int rc = flash_rt::quantize::f32_act_to_nvfp4_swizzled_mrows(
        x, dst_packed, dst_sfa, D, M, (long long) x_srow, frt_launch_pdl, stream);
    if (rc != 0) fprintf(stderr, "frt: quant_act launch failed (%d)\n", rc);
}

#define FRT_M_DISPATCH(M, ...) do { switch (M) { \
    case 1: { constexpr int MT = 1; __VA_ARGS__; } break; \
    case 2: { constexpr int MT = 2; __VA_ARGS__; } break; \
    case 3: { constexpr int MT = 3; __VA_ARGS__; } break; \
    default:{ constexpr int MT = 4; __VA_ARGS__; } break; } } while (0)

// ---------------- warp-split-K NVFP4 W4A4 GEMV (csrc single source) --------

// runtime (STAGES, WARPS) selection for the region GEMVs: FRT_WS_CFG=s<S>w<W>
// (default s4w2). K/64 must be divisible by W.
static void frt_ws_launch(const uint8_t * A, const uint8_t * B,
        const uint8_t * SFA, const uint8_t * SFB, float * D,
        float alpha, int N, int K, int M, cudaStream_t stream, int def_cfg = 0) {
    static int env_cfg = -1;
    if (env_cfg < 0) {
        const char * e = getenv("FRT_WS_CFG");
        env_cfg = 0;
        if (e) {
            int sc = 0, w = 0;
            if (sscanf(e, "s%dw%d", &sc, &w) == 2) env_cfg = sc * 10 + w;
        }
    }
    const int cfg = env_cfg ? env_cfg : (def_cfg ? def_cfg : 42);
    const int rc = flash_rt::gemm::fp4_w4a4_mma_sm120_warpsplit_mrows_f32out(
        A, B, D, M, N, K, SFA, SFB, alpha, /*warps=*/cfg % 10, /*stages=*/cfg / 10,
        frt_launch_pdl, stream);
    if (rc != 0) fprintf(stderr, "frt: warpsplit launch failed (%d, cfg=%d)\n", rc, cfg);
}

// ---------------- W4A16 matvec (NVFP4 weight x f32 act, f32 out) ----------
// Ported from FlashRT w4a16_matvec_sm120 (same swizzled NVFP4 weight layout),
// modified: f32 activation staged in smem, f32 output. No activation quant.

__device__ __constant__ float c_ue4m3[256];       // CUTLASS UE4M3 (FlashRT packs)
__device__ __constant__ float c_e4m3_half[256];   // ggml NVFP4 scale: E4M3 / 2, NaN->0

static void frt_init_ue4m3_lut(void) {
    static bool inited = false;
    if (inited) return;
    inited = true;
    float lut[256];
    for (int i = 0; i < 256; ++i) {
        const int e = (i >> 3) & 0xF;
        const int m = i & 0x7;
        lut[i] = (e == 0) ? (float) m * ldexpf(1.0f, -9)
                          : (1.0f + (float) m / 8.0f) * ldexpf(1.0f, e - 7);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_ue4m3, lut, sizeof(lut)));
    float lut2[256];
    for (int i = 0; i < 256; ++i) {
        const int lo = i & 0x7F;
        if (lo == 0x7F) { lut2[i] = 0.0f; continue; }   // E4M3 NaN -> 0 (ggml CPU semantics)
        const int e = (lo >> 3) & 0xF;
        const int m = lo & 0x7;
        // their stored scale = e4m3; their nibble table = 2x e2m1; we use true
        // e2m1 via the cvt intrinsic, so the plain e4m3 value pairs correctly.
        float v = (e == 0) ? (float) m / 8.0f * ldexpf(1.0f, -6)
                           : (1.0f + (float) m / 8.0f) * ldexpf(1.0f, e - 7);
        lut2[i] = (i & 0x80) ? -v : v;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_e4m3_half, lut2, sizeof(lut2)));
}

__device__ __forceinline__ int frt_sf_off(int rb_ncs, int row_inner, int k_block) {
    return (rb_ncs + (k_block >> 2)) * 512 + row_inner + (k_block & 3);
}

__device__ __forceinline__ float frt_blockdot_f32(uint64_t b_pack, const float2 * xb2) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_fp4x2_storage_t bb = static_cast<__nv_fp4x2_storage_t>(b_pack >> (j * 8));
        const __half2_raw wr = __nv_cvt_fp4x2_to_halfraw2(bb, __NV_E2M1);
        const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wr));
        const float2 xf = xb2[j];
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
    }
    return acc;
}

// 8 rows / block, 1 warp / row; x (f32) staged in smem shared by the warps.
__global__ void w4a16_matvec_f32(
        const float * __restrict__ x,
        const uint8_t * __restrict__ W,
        const uint8_t * __restrict__ SFB,
        float * __restrict__ out,
        float alpha, int N, int K, int n_col_super) {
    extern __shared__ float x_shf[];
    const int K_int4 = K >> 2;              // 4 f32 per int4
    const int4 * x_i4 = reinterpret_cast<const int4 *>(x);
    int4 * x_sh_i4 = reinterpret_cast<int4 *>(x_shf);
    for (int j = threadIdx.x; j < K_int4; j += 256) x_sh_i4[j] = x_i4[j];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= N) return;

    const int K_BLOCKS = K >> 4;
    const uint64_t * w_blk = reinterpret_cast<const uint64_t *>(W + (size_t) row * (K >> 1));
    const float2 * x_blk = reinterpret_cast<const float2 *>(x_shf);

    const int rb = row >> 7;
    const int ri = row & 127;
    const int rb_ncs = rb * n_col_super;
    const int row_inner = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;

    float acc = 0.0f;
    int kb = lane;
    constexpr int UNROLL = 4;
    const int step = 32 * UNROLL;
    for (; kb + 32 * (UNROLL - 1) < K_BLOCKS; kb += step) {
        uint64_t wv[UNROLL];
        float sf[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) wv[u] = w_blk[kb + 32 * u];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            sf[u] = c_ue4m3[__ldg(SFB + frt_sf_off(rb_ncs, row_inner, kb + 32 * u))];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            acc += frt_blockdot_f32(wv[u], x_blk + (size_t)(kb + 32 * u) * 8) * sf[u];
    }
    for (; kb < K_BLOCKS; kb += 32) {
        const float s = c_ue4m3[__ldg(SFB + frt_sf_off(rb_ncs, row_inner, kb))];
        acc += frt_blockdot_f32(w_blk[kb], x_blk + (size_t) kb * 8) * s;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0) out[row] = acc * alpha;
}

// ---------------- MoE expert GEMV on ggml-native NVFP4 blocks --------------
// Reads llama.cpp's own block_nvfp4 layout in place (no repack, no extra VRAM):
//   64 elems = d[4] UE4M3 sub-scales (16 elems each) + qs[32]
//   qs[sub*8 + j] holds elem (sub*16 + j) in low nibble, (sub*16 + j + 8) high.
// f32 activations read directly (no q8_1 activation quant at all).

// one warp computes one output row; lane l handles K-block (64 elems) l, l+32, ...
__device__ __forceinline__ float frt_nvfp4_rowdot(
        const uint8_t * __restrict__ row,   // K/64 blocks * 36 B
        const float * __restrict__ x, int K) {
    const int lane = threadIdx.x & 31;
    const int kb_n = K >> 6;
    float acc = 0.0f;
    for (int kb = lane; kb < kb_n; kb += 32) {
        const uint8_t * blk = row + (size_t) kb * 36;
        const float * xb = x + (size_t) kb * 64;
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const float d = c_e4m3_half[blk[sub]];
            // block stride is 36 B: qs is only 4-byte aligned, build the u64 from two u32 loads
            const uint32_t q_lo = *reinterpret_cast<const uint32_t *>(blk + 4 + sub * 8);
            const uint32_t q_hi = *reinterpret_cast<const uint32_t *>(blk + 8 + sub * 8);
            const uint64_t q = ((uint64_t) q_hi << 32) | q_lo;
            const float * xs = xb + sub * 16;
            float sacc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const __nv_fp4x2_storage_t bb = static_cast<__nv_fp4x2_storage_t>(q >> (j * 8));
                const __half2_raw wr = __nv_cvt_fp4x2_to_halfraw2(bb, __NV_E2M1);
                const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wr));
                sacc = fmaf(wf.x, xs[j], sacc);        // low nibble -> elem j
                sacc = fmaf(wf.y, xs[j + 8], sacc);    // high nibble -> elem j+8
            }
            acc = fmaf(d, sacc, acc);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    return acc;
}

// 8 warps / block, one row each. rows_total = n_used * n_per_expert.
// BROADCAST: all experts share x (gate/up). Else x per expert slot (down).
template <bool BROADCAST>
__global__ void frt_moe_mmid_f32(
        const float * __restrict__ x,       // (K) or (K, n_used)
        const uint8_t * __restrict__ w,     // expert-major NVFP4
        const int32_t * __restrict__ ids,   // (n_used)
        float * __restrict__ out,           // (n_per, n_used)
        int K, int n_per, int n_used,
        int64_t expert_stride, int64_t x_stride) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= n_used * n_per) return;
    const int e = r / n_per;
    const int n = r % n_per;
    const uint8_t * row = w + (size_t) ids[e] * expert_stride + (size_t) n * ((K >> 6) * 36);
    const float * xe = BROADCAST ? x : x + (size_t) e * x_stride;
    const float acc = frt_nvfp4_rowdot(row, xe, K);
    if ((threadIdx.x & 31) == 0) out[(size_t) e * n_per + n] = acc;
}

// ---------------- fused-region packs (GDN in_proj / attn qkv) -------------

struct frt_region {
    int64_t N = 0, K = 0;
    float alpha = 0.f;
    uint8_t * d_packed = nullptr;
    uint8_t * d_sf     = nullptr;
};

struct frt_region_state {
    bool tried = false;
    bool ok    = false;
    bool inproj_on = false;   // FRT_INPROJ_SWAP  (GDN kind 0)
    bool attn_on   = false;   // FRT_ATTNQKV_SWAP (attn kind 1)
    bool shexp_on  = false;   // FRT_SHEXP_SWAP (shared expert span, kinds 2+3)
    bool outproj_on = false;  // FRT_OUTPROJ_SWAP (ssm_out / attn_output, kind 4)
    frt_region regions[5][64]; // [kind][layer]: 0=gdn in_proj 1=attn qkv 2=shexp gate|up 3=shexp down 4=out_proj
    float * d_staging = nullptr;        // 12352 f32
    float * d_conv_out = nullptr;       // 8192 f32 (GDN cell)
    float * d_attn_buf = nullptr;       // 4096 f32 (GDN cell)
    float * d_scalar   = nullptr;       // 1 f32 (shexp sigmoid gate)
    block_q8_1 * d_outq8 = nullptr;     // gdn epilogue q8 output (128 blocks)
    const void * outq8_node = nullptr;  // graph node whose q8 is staged in d_outq8
    uint8_t * d_apack = nullptr;        // K/2
    uint8_t * d_sfa   = nullptr;        // 128 * K/16
    // capture-time leader tracking
    const void * leader_src = nullptr;
    int64_t leader_key = -1;
};

static frt_region_state g_reg;

static bool frt_online_on(void);   // defined with the in-process repack section

static bool frt_regions_load(void) {
    if (g_reg.tried) return g_reg.ok;
    g_reg.tried = true;
    const char * c = getenv("FRT_SHEXP_SWAP");
    const char * d = getenv("FRT_OUTPROJ_SWAP");
    g_reg.inproj_on = frt_window_on("FRT_INPROJ_SWAP", "GGML_FLASHRT_NO_INPROJ");
    g_reg.attn_on   = frt_window_on("FRT_ATTNQKV_SWAP", "GGML_FLASHRT_NO_ATTNQKV");
    g_reg.shexp_on  = frt_layer_enabled() && c && c[0] == '1';    // archive: opt-in
    g_reg.outproj_on = frt_layer_enabled() && d && d[0] == '1';   // archive: opt-in
    if (!g_reg.inproj_on && !g_reg.attn_on && !g_reg.shexp_on && !g_reg.outproj_on) return false;
    const char * path = getenv("FRT_REGIONS_PACK");
    if (!path && frt_online_on()) {
        // online repack: per-region weight buffers arrive from the pre-capture
        // hook; only the shared serve buffers are sized here, from the binding.
        int64_t maxN = 0, maxK = 0;
        for (int kind = 0; kind < frt_binding::n_region_kinds; ++kind) {
            maxN = std::max<int64_t>(maxN, frt_binding::region_n[kind]);
            maxK = std::max<int64_t>(maxK, frt_binding::region_k[kind]);
        }
        CUDA_CHECK(cudaMalloc(&g_reg.d_staging, 4 * maxN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_reg.d_conv_out, 4 * 8192 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_reg.d_attn_buf, 4 * 4096 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_reg.d_scalar, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_reg.d_outq8, 4 * 128 * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&g_reg.d_apack, 2 * maxK));
        CUDA_CHECK(cudaMalloc(&g_reg.d_sfa, 128 * (maxK / 16)));
        CUDA_CHECK(cudaMemset(g_reg.d_sfa, 0, 128 * (maxK / 16)));
        fprintf(stderr, "frt-regions: online repack mode (inproj=%d attnqkv=%d)\n",
                (int) g_reg.inproj_on, (int) g_reg.attn_on);
        g_reg.ok = true;
        return true;
    }
    if (!path) { fprintf(stderr, "frt-regions: FRT_REGIONS_PACK missing\n"); return false; }
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "frt-regions: cannot open %s\n", path); return false; }
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x46525452) { fclose(f); return false; }
    const int64_t count = hdr[1];
    int64_t maxN = 0, maxK = 0;
    for (int64_t e = 0; e < count; ++e) {
        int64_t layer, kind, N, K, pkb, sfb; double alpha;
        if (fread(&layer, 8, 1, f) != 1) break;
        fread(&kind, 8, 1, f); fread(&N, 8, 1, f); fread(&K, 8, 1, f);
        fread(&alpha, 8, 1, f); fread(&pkb, 8, 1, f); fread(&sfb, 8, 1, f);
        if (kind < 0 || kind > 4 || layer < 0 || layer >= 64) { fclose(f); return false; }
        frt_region & r = g_reg.regions[kind][layer];
        r.N = N; r.K = K; r.alpha = (float) alpha;
        uint8_t * h = (uint8_t *) malloc((size_t)(pkb > sfb ? pkb : sfb));
        CUDA_CHECK(cudaMalloc(&r.d_packed, pkb));
        fread(h, 1, pkb, f);
        CUDA_CHECK(cudaMemcpy(r.d_packed, h, pkb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&r.d_sf, sfb));
        fread(h, 1, sfb, f);
        CUDA_CHECK(cudaMemcpy(r.d_sf, h, sfb, cudaMemcpyHostToDevice));
        free(h);
        if (N > maxN) maxN = N;
        if (K > maxK) maxK = K;
    }
    fclose(f);
    CUDA_CHECK(cudaMalloc(&g_reg.d_staging, 4 * maxN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_reg.d_conv_out, 4 * 8192 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_reg.d_attn_buf, 4 * 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_reg.d_scalar, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_reg.d_outq8, 4 * 128 * sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc(&g_reg.d_apack, 2 * maxK));
    CUDA_CHECK(cudaMalloc(&g_reg.d_sfa, 128 * (maxK / 16)));
    CUDA_CHECK(cudaMemset(g_reg.d_sfa, 0, 128 * (maxK / 16)));
    fprintf(stderr, "frt-regions: loaded %lld regions (inproj=%d attnqkv=%d)\n",
            (long long) count, (int) g_reg.inproj_on, (int) g_reg.attn_on);
    g_reg.ok = true;
    return true;
}

// serve one member of a fused region. leader==true runs quant+GEMV into staging.
static bool frt_region_serve(ggml_backend_cuda_context & ctx, int kind, int layer,
        bool leader, int64_t row_off, int64_t rows,
        const ggml_tensor * src1, ggml_tensor * dst) {
    frt_region & r = g_reg.regions[kind][layer];
    if (r.N == 0) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    const int M = (int) src1->ne[1];   // token-batch width (spec verify runs M = 1 + n_draft)
    if (M < 1 || M > 4 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (dst->ne[1] != M || !ggml_is_contiguous(dst)) return false;
    if (!ggml_is_contiguous(src1)) return false;
    const int64_t key = ((int64_t) kind << 32) | layer;
    cudaStream_t stream = ctx.stream();
    if (leader) {
        frt_quant_act_launch((const float *) src1->data, g_reg.d_apack, g_reg.d_sfa, (int) r.K, M, (int64_t) r.K, stream);
        frt::frt_ws_launch(g_reg.d_apack, r.d_packed, g_reg.d_sfa, r.d_sf,
            g_reg.d_staging, r.alpha, (int) r.N, (int) r.K, M, stream);
        g_reg.leader_src = src1->data;
        g_reg.leader_key = (key << 2) | M;
    } else {
        // follower: only valid if the leader ran with the same activation
        if (g_reg.leader_key != ((key << 2) | M) || g_reg.leader_src != src1->data) return false;
    }
    if (M == 1) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, g_reg.d_staging + row_off,
            rows * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else {   // one strided copy for all token rows
        CUDA_CHECK(cudaMemcpy2DAsync(dst->data, rows * sizeof(float),
            g_reg.d_staging + row_off, r.N * sizeof(float),
            rows * sizeof(float), M, cudaMemcpyDeviceToDevice, stream));
    }
    return true;
}

// dims: GDN in_proj = [qkv 8192 | z 4096 | a 32 | b 32]; attn = [q 8192 | k 512 | v 512]
static bool frt_regions_mul_mat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!frt_regions_load()) return false;
    int layer = -1; char rest[64] = {0};
    if (sscanf(src0->name, "blk.%d.%63s", &layer, rest) != 2 || layer < 0 || layer >= frt_binding::layer_scan_max) return false;
    if (!frt_layer_in_scope(layer)) return false;
    for (int kind = 0; kind < frt_binding::n_region_kinds; ++kind) {
        if (kind == 0 && !g_reg.inproj_on) continue;
        if (kind == 1 && !g_reg.attn_on) continue;
        for (int m = 0; m < frt_binding::region_n_members[kind]; ++m) {
            const auto & mem = frt_binding::region_members[kind][m];
            if (strcmp(rest, mem.name) == 0)
                return frt_region_serve(ctx, kind, layer, mem.leader, mem.off, mem.rows, src1, dst);
        }
    }
    if (g_reg.shexp_on) {
        if (strcmp(rest, "ffn_gate_shexp.weight") == 0) return frt_region_serve(ctx, 2, layer, true,   0, 512, src1, dst);
        if (strcmp(rest, "ffn_up_shexp.weight") == 0)   return frt_region_serve(ctx, 2, layer, false, 512, 512, src1, dst);
    }
    if (g_reg.outproj_on) {
        if (strcmp(rest, frt_binding::out_proj_names[0]) == 0 || strcmp(rest, frt_binding::out_proj_names[1]) == 0)
            return frt_region_serve(ctx, 4, layer, true, 0, 2048, src1, dst);
    }
    return false;
}

// ---------------- side-band pack + hook ------------------------------------

struct frt_head_state {
    bool     tried = false;
    bool     ok    = false;
    bool     draft_only = false;   // FRT_HEAD_DRAFT without FRT_HEAD_SWAP
    int64_t  N     = 0;
    int64_t  K     = 0;
    float    alpha = 0.f;
    uint8_t * d_packed = nullptr;
    uint8_t * d_sf     = nullptr;
    uint8_t * d_apack  = nullptr;   // K/2 bytes
    uint8_t * d_sfa    = nullptr;   // 128 * K/16 bytes
};

static frt_head_state g_head;

static bool frt_head_load(void) {
    if (g_head.tried) return g_head.ok;
    g_head.tried = true;
    // FRT_HEAD_SWAP serves every output.weight head (full tier). FRT_HEAD_DRAFT
    // alone serves only the spec-draft copy of the head (identified by its Q8_0
    // storage; the target head stays Q6_K/stock) — draft logits only steer
    // acceptance, never the verified output, so this is quality-free.
    const char * sw = getenv("FRT_HEAD_SWAP");
    const bool sw_on = frt_layer_enabled() && sw && sw[0] == '1';   // full tier: opt-in (quality)
    const bool dr_on = frt_window_on("FRT_HEAD_DRAFT", "GGML_FLASHRT_NO_HEAD_DRAFT");
    // opt-in: serving an NVFP4-typed head through the fp4-activation GEMV
    // costs a measured perplexity increment over stock's q8_1-activation mmvq
    // on the same weights, so it is a quality trade like the full-tier swap.
    const char * nat = getenv("FRT_HEAD_NATIVE");
    const bool nat_on = frt_layer_enabled() && nat && nat[0] == '1';
    if (!sw_on && !dr_on && !nat_on) return false;
    g_head.draft_only = !sw_on;
    const char * path = getenv("FRT_HEAD_PACK");
    if (!path && frt_online_on()) {
        g_head.tried = false;   // built by the pre-capture repack hook
        return g_head.ok;
    }
    if (!path) { fprintf(stderr, "frt-head: FRT_HEAD_SWAP/FRT_HEAD_DRAFT set but FRT_HEAD_PACK missing\n"); return false; }
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "frt-head: cannot open %s\n", path); return false; }
    int64_t hdr[4] = {0, 0, 0, 0}; // magic, N, K, alpha bits (f64)
    if (fread(hdr, 8, 4, f) != 4 || hdr[0] != 0x46525448) { fclose(f); fprintf(stderr, "frt-head: bad header\n"); return false; }
    g_head.N = hdr[1]; g_head.K = hdr[2];
    double alpha_d; memcpy(&alpha_d, &hdr[3], 8);
    g_head.alpha = (float) alpha_d;
    const size_t packed_bytes = (size_t) g_head.N * (size_t) g_head.K / 2;
    const size_t nrb = ((size_t) g_head.N + 127) / 128;
    const size_t sf_bytes = nrb * (size_t)((g_head.K + 63) / 64) * 512;
    uint8_t * h = (uint8_t *) malloc(packed_bytes > sf_bytes ? packed_bytes : sf_bytes);
    CUDA_CHECK(cudaMalloc(&g_head.d_packed, packed_bytes));
    if (fread(h, 1, packed_bytes, f) != packed_bytes) { fclose(f); free(h); fprintf(stderr, "frt-head: short packed\n"); return false; }
    CUDA_CHECK(cudaMemcpy(g_head.d_packed, h, packed_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g_head.d_sf, sf_bytes));
    if (fread(h, 1, sf_bytes, f) != sf_bytes) { fclose(f); free(h); fprintf(stderr, "frt-head: short sf\n"); return false; }
    CUDA_CHECK(cudaMemcpy(g_head.d_sf, h, sf_bytes, cudaMemcpyHostToDevice));
    fclose(f); free(h);
    CUDA_CHECK(cudaMalloc(&g_head.d_apack, 4 * (g_head.K / 2)));
    CUDA_CHECK(cudaMalloc(&g_head.d_sfa, 128 * (g_head.K / 16)));
    CUDA_CHECK(cudaMemset(g_head.d_sfa, 0, 128 * (g_head.K / 16)));
    frt_init_ue4m3_lut();   // eager-time upload; never during graph capture
    fprintf(stderr, "frt-head: loaded pack N=%lld K=%lld alpha=%g\n",
            (long long) g_head.N, (long long) g_head.K, (double) g_head.alpha);
    g_head.ok = true;
    return true;
}

// ---- in-process weight repack (FRT_ONLINE_REPACK=1) -----------------------
// Replaces the side-band pack files: region/head FP4 wire buffers are built
// on first sight of the weight tensors in an evaluated graph, before any
// CUDA graph capture (called from the pre-capture hook in ggml-cuda.cu).
// The pipeline reproduces the offline packer bit-for-bit: ggml dequant ->
// bf16 (RNE) -> global amax -> global_scale = amax/2688 -> per-16 ue4m3-ceil
// block scales -> e2m1 nibbles + Sm1xx atom-layout SF bytes.
// FRT_REPACK_CHECK=1 memcmp-validates against the pack files when both are
// given.

__device__ __forceinline__ uint8_t frt_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;
    uint32_t bits = __float_as_uint(v);
    int float_exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;
    int ue_exp = float_exp + 7;
    if (ue_exp <= 0) {
        float scaled = v * 512.0f;
        int m = (int) ceilf(scaled);
        if (m > 7) return (1 << 3) | 0;
        if (m < 1) m = 1;
        return (uint8_t) m;
    }
    if (ue_exp >= 15) return 0xFE;
    int m = (int) (frac >> 20);
    if (frac & 0xFFFFF) m++;
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;
    return (uint8_t) ((ue_exp << 3) | m);
}

__device__ __forceinline__ float frt_ue4m3_f32(uint8_t v) {
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return ldexpf((float) m / 8.0f, -6);
    return ldexpf(1.0f + (float) m / 8.0f, e - 7);
}

// e2m1 with the offline packer's strict-< boundaries (the activation
// quantizer above uses <=; at exact tie values the codes differ, so weight
// repack must use this one to stay byte-identical with the pack files).
__device__ __forceinline__ uint8_t frt_e2m1_weight(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;
    else if (a < 0.75f)  mag = 1;
    else if (a < 1.25f)  mag = 2;
    else if (a < 1.75f)  mag = 3;
    else if (a < 2.5f)   mag = 4;
    else if (a < 3.5f)   mag = 5;
    else if (a < 5.0f)   mag = 6;
    else                 mag = 7;
    return sign | mag;
}

__global__ void frt_w_amax_bf16(const __nv_bfloat16 * __restrict__ w, float * __restrict__ gmax, int N, int K) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const size_t off = (size_t) row * K;
    float tm = 0.f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        const float a = fabsf(__bfloat162float(w[off + c]));
        if (a > tm) tm = a;
    }
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
    if (lane == 0) smem[wid] = tm;
    __syncthreads();
    if (wid == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        tm = (lane < nw) ? smem[lane] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
        if (lane == 0) atomicMax(reinterpret_cast<int *>(gmax), __float_as_int(tm));
    }
}

__global__ void frt_w_gscale(const float * gmax, float * gs) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const float a = *gmax;
        *gs = (a > 0.f) ? (a / 2688.f) : 1.f;
    }
}

// rows [0, n_rows) of w correspond to absolute output rows row_base + r.
__global__ void frt_w_pass2_bf16(const __nv_bfloat16 * __restrict__ w, const float * __restrict__ gs_ptr,
        uint8_t * __restrict__ packed, uint8_t * __restrict__ sf_swz,
        int n_rows, int K, int row_base, int n_col_super) {
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const float gscale = *gs_ptr;
    const float inv_g = (gscale > 0.f) ? (1.f / gscale) : 0.f;
    const int row = row_base + r;
    const size_t in_off  = (size_t) r * K;
    const size_t out_off = (size_t) row * (K / 2);
    const int rb = row / 128, ri = row % 128;
    const int nbr = K / 16;
    for (int b = threadIdx.x; b < nbr; b += blockDim.x) {
        const int col0 = b * 16;
        float v[16];
        float bmax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = __bfloat162float(w[in_off + col0 + i]);
            const float a = fabsf(v[i]);
            if (a > bmax) bmax = a;
        }
        const uint8_t sf_byte = frt_ue4m3_ceil((bmax / 6.f) * inv_g);
        const float bs = frt_ue4m3_f32(sf_byte) * gscale;
        const float inv_bs = (bs > 0.f) ? (1.f / bs) : 0.f;
        uint8_t * prow = packed + out_off;
#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const uint8_t lo = frt_e2m1_weight(v[i]     * inv_bs);
            const uint8_t hi = frt_e2m1_weight(v[i + 1] * inv_bs);
            prow[(col0 + i) >> 1] = (uint8_t) ((hi << 4) | (lo & 0x0F));
        }
        const int cb = b / 4, ci = b % 4;
        sf_swz[(rb * n_col_super + cb) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + ci] = sf_byte;
    }
}

// ggml GGML_TYPE_NVFP4 weight -> GEMV wire format, pure shuffle (no
// requantization): block_nvfp4 is 36 B / 64 elems = d[4] e4m3 sub-scales +
// qs[32] split-nibble codes (sub s at qs[s*8+j]: elem j low nibble, elem
// 8+j high). ggml's doubled e2m1 table and halved ue4m3 decode cancel, so
// the scale bytes pass through unmodified and the GEMV runs with alpha = 1.
// One thread per 16-element sub-block.
__global__ void frt_w_nvfp4_shuffle(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ packed,
        uint8_t * __restrict__ sf_swz, int n_rows, int K, int row_base) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t subs_per_row = K / 16;
    if (idx >= (int64_t) n_rows * subs_per_row) return;
    const int r    = (int) (idx / subs_per_row);
    const int sub  = (int) (idx % subs_per_row);
    const int blk  = sub >> 2, s = sub & 3;
    const uint8_t * b = src + ((size_t) r * (K / 64) + blk) * 36;
    const uint8_t * qs = b + 4 + s * 8;
    const int row = row_base + r;
    uint8_t * out = packed + (size_t) row * (K / 2) + (size_t) sub * 8;
#pragma unroll
    for (int p = 0; p < 4; ++p)
        out[p] = (uint8_t) ((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
#pragma unroll
    for (int p = 0; p < 4; ++p)
        out[4 + p] = (uint8_t) ((qs[2 * p] >> 4) | ((qs[2 * p + 1] >> 4) << 4));
    sf_swz[sfa_offset_128x64(row, sub * 16, K)] = b[s];
}

// build wire buffers for an NVFP4-typed ggml tensor (shuffle only, alpha=1)
static bool frt_repack_shuffle_nvfp4(const ggml_tensor * t, int64_t N, int64_t K,
        uint8_t * d_packed, uint8_t * d_sf, cudaStream_t stream) {
    const int64_t total = N * (K / 16);
    const int64_t blocks = (total + 255) / 256;
    frt_w_nvfp4_shuffle<<<dim3((unsigned) blocks), dim3(256), 0, stream>>>(
        (const uint8_t *) t->data, d_packed, d_sf, (int) N, (int) K, 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

static bool frt_online_on(void) {
    static int on = -1;
    if (on < 0) on = frt_window_on("FRT_ONLINE_REPACK", "GGML_FLASHRT_NO_ONLINE_REPACK") ? 1 : 0;
    return on == 1;
}

// One source tensor contributing `rows` rows to an [N, K] concat target.
struct frt_repack_src { const ggml_tensor * t; int64_t rows; };

// Build packed+SF (+alpha) for a row-concatenation of ggml tensors. Eager
// only (allocates, synchronizes); chunked so even the 248320-row head needs
// a bounded bf16 staging buffer.
static bool frt_repack_build(const frt_repack_src * srcs, int n_src, int64_t N, int64_t K,
        uint8_t * d_packed, uint8_t * d_sf, float * out_alpha, cudaStream_t stream) {
    const int64_t CHUNK = 8192;
    static __nv_bfloat16 * d_stage = nullptr;
    static float * d_scr = nullptr;   // [amax, gscale]
    if (!d_stage) CUDA_CHECK(cudaMalloc(&d_stage, CHUNK * K * sizeof(__nv_bfloat16)));
    if (!d_scr)   CUDA_CHECK(cudaMalloc(&d_scr, 2 * sizeof(float)));
    const int n_col_super = ((int) (K / 16) + 3) / 4;
    CUDA_CHECK(cudaMemsetAsync(d_scr, 0, sizeof(float), stream));
    for (int pass = 0; pass < 2; ++pass) {   // 0 = amax, 1 = quantize
        int64_t row_base = 0;
        for (int s = 0; s < n_src; ++s) {
            const ggml_tensor * t = srcs[s].t;
            const to_bf16_cuda_t conv = ggml_get_to_bf16_cuda(t->type);
            if (conv == nullptr) return false;
            const size_t row_bytes = ggml_row_size(t->type, K);
            for (int64_t r0 = 0; r0 < srcs[s].rows; r0 += CHUNK) {
                const int64_t rows = std::min(CHUNK, srcs[s].rows - r0);
                conv((const char *) t->data + r0 * row_bytes, d_stage, rows * K, stream);
                if (pass == 0) {
                    frt_w_amax_bf16<<<dim3((unsigned) rows), dim3(256), 0, stream>>>(d_stage, d_scr, (int) rows, (int) K);
                } else {
                    frt_w_pass2_bf16<<<dim3((unsigned) rows), dim3(256), 0, stream>>>(d_stage, d_scr + 1,
                        d_packed, d_sf, (int) rows, (int) K, (int) (row_base + r0), n_col_super);
                }
            }
            row_base += srcs[s].rows;
        }
        if (pass == 0) frt_w_gscale<<<1, 1, 0, stream>>>(d_scr, d_scr + 1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(out_alpha, d_scr + 1, sizeof(float), cudaMemcpyDeviceToHost));
    return *out_alpha != 0.0f;
}

// FRT_REPACK_CHECK=1: byte-compare an online-built region against the pack
// file entry it replaces (pack path from FRT_REGIONS_PACK/FRT_HEAD_PACK).
static void frt_repack_check_region(int kind, int layer, const frt_region & r) {
    static int check = -1;
    if (check < 0) { const char * c = getenv("FRT_REPACK_CHECK"); check = (c && c[0] == '1') ? 1 : 0; }
    if (!check) return;
    const char * path = (kind == 5) ? getenv("FRT_HEAD_PACK_REF") : getenv("FRT_REGIONS_PACK_REF");
    if (!path) return;
    const size_t pkb = (size_t) r.N * r.K / 2;
    const size_t sfb = (size_t) ((r.N + 127) / 128) * ((r.K + 63) / 64) * 512;
    std::vector<uint8_t> ref(pkb > sfb ? pkb : sfb), got(pkb > sfb ? pkb : sfb);
    FILE * f = fopen(path, "rb");
    if (!f) return;
    bool found = false;
    double ref_alpha = 0.0;
    if (kind == 5) {   // head pack: single entry
        int64_t hdr[4];
        if (fread(hdr, 8, 4, f) == 4 && hdr[1] == r.N && hdr[2] == r.K) {
            memcpy(&ref_alpha, &hdr[3], 8);
            found = fread(ref.data(), 1, pkb, f) == pkb;
            std::vector<uint8_t> sfref(sfb);
            if (found && fread(sfref.data(), 1, sfb, f) == sfb) {
                CUDA_CHECK(cudaMemcpy(got.data(), r.d_packed, pkb, cudaMemcpyDeviceToHost));
                const bool pk_ok = memcmp(got.data(), ref.data(), pkb) == 0;
                CUDA_CHECK(cudaMemcpy(got.data(), r.d_sf, sfb, cudaMemcpyDeviceToHost));
                const bool sf_ok = memcmp(got.data(), sfref.data(), sfb) == 0;
                fprintf(stderr, "frt-repack-check head: packed=%s sf=%s alpha %.9g vs %.9g\n",
                        pk_ok ? "OK" : "MISMATCH", sf_ok ? "OK" : "MISMATCH", (double) r.alpha, ref_alpha);
            }
        }
        fclose(f);
        return;
    }
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2) { fclose(f); return; }
    for (int64_t e = 0; e < hdr[1]; ++e) {
        int64_t el, ek, en, ekk, epkb, esfb; double ea;
        if (fread(&el, 8, 1, f) != 1) break;
        if (fread(&ek, 8, 1, f) != 1 || fread(&en, 8, 1, f) != 1 || fread(&ekk, 8, 1, f) != 1 ||
            fread(&ea, 8, 1, f) != 1 || fread(&epkb, 8, 1, f) != 1 || fread(&esfb, 8, 1, f) != 1) break;
        if (el == layer && ek == kind) {
            found = (epkb == (int64_t) pkb && esfb == (int64_t) sfb);
            if (found) {
                if (fread(ref.data(), 1, pkb, f) != pkb) break;
                CUDA_CHECK(cudaMemcpy(got.data(), r.d_packed, pkb, cudaMemcpyDeviceToHost));
                const bool pk_ok = memcmp(got.data(), ref.data(), pkb) == 0;
                if (fread(ref.data(), 1, sfb, f) != sfb) break;
                CUDA_CHECK(cudaMemcpy(got.data(), r.d_sf, sfb, cudaMemcpyDeviceToHost));
                const bool sf_ok = memcmp(got.data(), ref.data(), sfb) == 0;
                fprintf(stderr, "frt-repack-check kind%d layer%d: packed=%s sf=%s alpha %.9g vs %.9g\n",
                        kind, layer, pk_ok ? "OK" : "MISMATCH", sf_ok ? "OK" : "MISMATCH", (double) r.alpha, ea);
            }
            break;
        }
        fseek(f, epkb + esfb, SEEK_CUR);
    }
    fclose(f);
    if (!found) fprintf(stderr, "frt-repack-check kind%d layer%d: no reference entry\n", kind, layer);
}

// Pre-capture hook body: scan the graph for region/head weight tensors and
// build any missing online buffers. Eager only — the caller guarantees no
// CUDA graph capture is in flight.
static void frt_online_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (!frt_online_on()) return;
    static bool all_done = false;
    static int evals_seen = 0;
    if (all_done) return;
    ++evals_seen;
    const char * hsw = getenv("FRT_HEAD_SWAP");
    const bool head_full = frt_layer_enabled() && hsw && hsw[0] == '1';
    const bool head_draft = frt_window_on("FRT_HEAD_DRAFT", "GGML_FLASHRT_NO_HEAD_DRAFT");
    if (!frt_regions_load() && !head_full && !head_draft) { all_done = true; return; }

    // once everything structural is built and only a draft-model head sighting
    // is outstanding, drop to a light scan (name check only) — the full member
    // scan on every eval costs measurable host time.
    static bool structural_done = false;
    static bool saw_draft_head = false;   // a loaded spec draft's Q8_0 head copy
    if (structural_done) {
        // the draft model's graphs are small (single MTP layer); skip the walk
        // on full-size target graphs entirely.
        if (!saw_draft_head && cgraph->n_nodes < 512) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                if (n->op == GGML_OP_MUL_MAT && n->src[0] && n->src[0]->type == GGML_TYPE_Q8_0 &&
                    n->src[0]->ne[0] == frt_binding::d_model && strcmp(n->src[0]->name, frt_binding::head_name) == 0) {
                    saw_draft_head = true;
                    break;
                }
            }
        }
        if (!saw_draft_head && evals_seen < 4096) return;  // cheap: small graphs only
        if (!saw_draft_head) { all_done = true; return; }  // no draft model; give up
        // fall through: full scan once, to locate the pack source and build
    }

    using namespace frt_binding;
    const ggml_tensor * mem[n_region_kinds][64][region_max_members] = {};
    const ggml_tensor * head_w = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT || !n->src[0]) continue;
        const ggml_tensor * w = n->src[0];
        if (strcmp(w->name, head_name) == 0 && w->ne[0] == d_model) {
            if (w->type != GGML_TYPE_Q8_0) head_w = w;
            else saw_draft_head = true;
            continue;
        }
        int layer = -1; char rest[64] = {0};
        if (sscanf(w->name, "blk.%d.%63s", &layer, rest) != 2 || layer < 0 || layer >= layer_scan_max) continue;
        if (!frt_layer_in_scope(layer)) continue;
        for (int kind = 0; kind < n_region_kinds; ++kind)
            for (int m = 0; m < region_n_members[kind]; ++m)
                if (strcmp(rest, region_members[kind][m].name) == 0 && w->ne[1] == region_members[kind][m].rows)
                    mem[kind][layer][m] = w;
    }

    cudaStream_t stream = ctx.stream();
    int built[n_region_kinds] = {};
    for (int kind = 0; kind < n_region_kinds; ++kind) {
        if (kind == 0 && !g_reg.inproj_on) continue;
        if (kind == 1 && !g_reg.attn_on) continue;
        const int n_mem = region_n_members[kind];
        const int64_t N = region_n[kind], K = region_k[kind];
        for (int layer = 0; layer < layer_scan_max; ++layer) {
            frt_region & r = g_reg.regions[kind][layer];
            if (r.N != 0) { built[kind]++; continue; }
            bool have = true;
            for (int m = 0; m < n_mem; ++m) have = have && mem[kind][layer][m] != nullptr;
            if (!have) continue;
            frt_repack_src srcs[region_max_members];
            for (int m = 0; m < n_mem; ++m) srcs[m] = { mem[kind][layer][m], region_members[kind][m].rows };
            const size_t pkb = (size_t) N * K / 2;
            const size_t sfb = (size_t) ((N + 127) / 128) * ((K + 63) / 64) * 512;
            CUDA_CHECK(cudaMalloc(&r.d_packed, pkb));
            CUDA_CHECK(cudaMalloc(&r.d_sf, sfb));
            float alpha = 0.f;
            if (!frt_repack_build(srcs, n_mem, N, K, r.d_packed, r.d_sf, &alpha, stream)) {
                fprintf(stderr, "frt-repack: kind%d layer%d FAILED\n", kind, layer);
                cudaFree(r.d_packed); cudaFree(r.d_sf);
                r.d_packed = nullptr; r.d_sf = nullptr;
                continue;
            }
            r.alpha = alpha; r.K = K; r.N = N;   // N last: serve fires only on complete regions
            frt_repack_check_region(kind, layer, r);
            built[kind]++;
        }
    }

    // draft-only serving builds lazily: only once a draft model is actually
    // loaded (its Q8_0 head copy shows up in a graph), so plain runs never
    // spend VRAM on a head pack that would never serve.
    const char * hnat = getenv("FRT_HEAD_NATIVE");
    const bool head_native = head_w && head_w->type == GGML_TYPE_NVFP4 &&
        frt_layer_enabled() && hnat && hnat[0] == '1';
    if (head_w && !g_head.ok && (head_full || head_native || (head_draft && saw_draft_head))) {
        const int64_t N = head_w->ne[1], K = head_w->ne[0];
        const size_t pkb = (size_t) N * K / 2;
        const size_t sfb = (size_t) ((N + 127) / 128) * ((K + 63) / 64) * 512;
        CUDA_CHECK(cudaMalloc(&g_head.d_packed, pkb));
        CUDA_CHECK(cudaMalloc(&g_head.d_sf, sfb));
        float alpha = 0.f;
        bool built;
        if (head_w->type == GGML_TYPE_NVFP4) {
            // FlashRT-edition GGUF: the head is already NVFP4 (quantized from
            // the BF16 checkpoint by llama-quantize) — wire it up by shuffle,
            // no requantization, alpha = 1.
            built = frt_repack_shuffle_nvfp4(head_w, N, K, g_head.d_packed, g_head.d_sf, stream);
            alpha = 1.0f;
        } else {
            frt_repack_src src = { head_w, N };
            built = frt_repack_build(&src, 1, N, K, g_head.d_packed, g_head.d_sf, &alpha, stream);
        }
        if (built) {
            g_head.N = N; g_head.K = K; g_head.alpha = alpha;
            CUDA_CHECK(cudaMalloc(&g_head.d_apack, 4 * (K / 2)));
            CUDA_CHECK(cudaMalloc(&g_head.d_sfa, 128 * (K / 16)));
            CUDA_CHECK(cudaMemset(g_head.d_sfa, 0, 128 * (K / 16)));
            frt_init_ue4m3_lut();
            frt_region hr; hr.N = N; hr.K = K; hr.alpha = alpha; hr.d_packed = g_head.d_packed; hr.d_sf = g_head.d_sf;
            frt_repack_check_region(5, 0, hr);
            fprintf(stderr, "frt-repack: head online N=%lld K=%lld alpha=%g\n", (long long) N, (long long) K, (double) alpha);
            g_head.ok = true;
        } else {
            cudaFree(g_head.d_packed); cudaFree(g_head.d_sf);
            g_head.d_packed = nullptr; g_head.d_sf = nullptr;
        }
    }

    const bool head_pending = (head_full || head_native || (head_draft && saw_draft_head)) && !g_head.ok;
    bool region_pending = false;
    for (int kind = 0; kind < n_region_kinds; ++kind) {
        const bool on = (kind == 0 && g_reg.inproj_on) || (kind == 1 && g_reg.attn_on);
        region_pending = region_pending || (on && built[kind] < region_layers[kind]);
    }
    if (!head_pending && !region_pending) {
        structural_done = true;
        const bool draft_wait = head_draft && !head_full && !saw_draft_head;
        if (!draft_wait) {
            fprintf(stderr, "frt-repack: online repack complete (kind0=%d kind1=%d head=%d)\n",
                    built[0], built[1], (int) g_head.ok);
            all_done = true;
        }
    }
}

} // namespace frt

// ---- GDN cell fusion (FRT_GDN_SWAP) ---------------------------------------
// Replaces the whole per-layer GDN cell span (conv-cache dance + SSM_CONV +
// silu + l2norms + gate prep + GATED_DELTA_NET + state copies + gated norm)
// with: leader GEMV (staging) -> K1 conv -> K2 cell. State/conv caches are
// updated in place: 1R+1W instead of the graph's multi-copy dance.

namespace gdn {

// model dims from the binding (single source; kernels are compile-time
// specialized to these)
constexpr int CONV_ROW = frt_binding::gdn_conv_cache_row;   // conv cache slot stride (floats)
constexpr int STATE_SZ = frt_binding::gdn_state_size;       // recurrent state slot size (floats)

__device__ __forceinline__ float frt_silu(float x)     { return x / (1.0f + expf(-x)); }

// same semantics as quantize_q8_1 (d=amax/127, s=raw sum), one warp = one block
__device__ __forceinline__ void frt_q8_1_block_g(float xi, int lane, block_q8_1 * dst) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, sum);
}
__device__ __forceinline__ float frt_softplus(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

// K1: 4-tap causal conv over [conv_state | x_new(M tokens)] + silu; shifts conv
// state in place (state after = last 3 inputs of the 3+M window). staging is
// token-major (t*12288 + ch); conv_out is token-major (t*8192 + ch).
// MT==1: window shifted in place (the M=1 graph's snapshot CPY targets the same
// slot row). MT>=2 (spec verify): the source slot must stay pristine for
// rollback; instead a per-token window snapshot is written to the cache rows
// the graph's M conv-state CPY nodes target (snap0..snap3, token order).
template <int MT>
__global__ void frt_gdn_conv_silu(
        const float * __restrict__ staging,   // qkv rows, token stride 12288
        const float * __restrict__ conv_w,    // (4, 8192): w(j,ch) = conv_w[ch*4+j]
        float * __restrict__ r_base,          // conv cache base, slot stride 24576 floats
        const int32_t * __restrict__ r_slot,
        float * __restrict__ conv_out,
        float * __restrict__ snap0, float * __restrict__ snap1,
        float * __restrict__ snap2, float * __restrict__ snap3) {
    constexpr int M = MT;
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= 8192) return;
    float * cs = r_base + (size_t) (*r_slot) * CONV_ROW + (size_t) ch * 3;
    const float * w = conv_w + (size_t) ch * 4;
    float s0 = cs[0], s1 = cs[1], s2 = cs[2];
    float * const snaps[4] = { snap0, snap1, snap2, snap3 };
    for (int t = 0; t < M; ++t) {
        const float x = staging[(size_t) t * 12288 + ch];
        const float o = s0 * w[0] + s1 * w[1] + s2 * w[2] + x * w[3];
        conv_out[(size_t) t * 8192 + ch] = frt_silu(o);
        s0 = s1; s1 = s2; s2 = x;
        if (MT >= 2) {
            float * sp = snaps[t] + (size_t) ch * 3;
            sp[0] = s0; sp[1] = s1; sp[2] = s2;
        }
    }
    if (MT == 1) { cs[0] = s0; cs[1] = s1; cs[2] = s2; }
}

// checkpoint save: dst[i] = base[idx[0]*row + i] (the graph's pre-update
// snapshot of the current cache slot into a checkpoint slot).
__global__ void frt_gdn_ckpt_copy(
        const float * __restrict__ base, const int32_t * __restrict__ idx,
        int64_t row, float * __restrict__ dst, int n) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = base[(size_t) idx[0] * row + i];
}

// K0n: norm-fused variant: each block redundantly computes the RMS norm of the
// raw hidden state (8KB read, latency-free) into smem, then proceeds like K0.
__global__ void frt_gdn_norm_quant_ab(
        const float * __restrict__ raw,       // 2048 f32 pre-norm hidden
        const float * __restrict__ normw,     // attn_norm weight
        float eps,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sfa,
        const float * __restrict__ w_alpha,
        const float * __restrict__ w_beta,
        float * __restrict__ staging) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    __shared__ float act[2048];
    __shared__ float red[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float4 * r4 = (const float4 *) raw;
    float s2 = 0.0f;
    for (int k = tid; k < 512; k += 256) {
        const float4 v = r4[k];
        s2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    if (lane == 0) red[warp] = s2;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int q = 0; q < 8; ++q) tot += red[q];
    const float rrms = rsqrtf(tot / 2048.0f + eps);
    for (int k = tid; k < 2048; k += 256) act[k] = raw[k] * rrms * normw[k];
    __syncthreads();
    if (blockIdx.x == 0) {
        frt::quant_act_fp4_f32_body<256>(act, dst_packed, dst_sfa, 2048);
        return;
    }
    const int row = (blockIdx.x - 1) * 8 + warp;
    const float * w = (row < 32 ? w_alpha + (size_t) row * 2048
                                : w_beta + (size_t) (row - 32) * 2048);
    const float4 * w4 = (const float4 *) w;
    float acc = 0.0f;
#pragma unroll 4
    for (int k = lane; k < 512; k += 32) {
        const float4 wv = w4[k];
        acc += wv.x * act[k * 4] + wv.y * act[k * 4 + 1] + wv.z * act[k * 4 + 2] + wv.w * act[k * 4 + 3];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) staging[12288 + row] = acc;
}

// K0: fused act-quant + F32 a/b gate rows. Block 0 quantizes the f32 act for
// the W4A4 GEMV (which is then launched with N=12288 so it never touches the
// gate rows); blocks 1..8 compute the 64 a/b rows in F32 (weights are stored
// F32 in the GGUF; W4A4 staging values for these rows cost ~1% PPL). The ab
// blocks run concurrently with the single quant block, so they are ~free.
// a/b outputs land token-major after the GEMV rows: staging[M*12288 + t*64 + row].
template <int MT>
__global__ void frt_gdn_quant_ab(
        const float * __restrict__ act,       // M x 2048 f32 rows (attn_norm)
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sfa,
        const float * __restrict__ w_alpha,   // (2048, 32) row-major K-contig
        const float * __restrict__ w_beta,
        float * __restrict__ staging) {
    constexpr int M = MT;
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    if (blockIdx.x == 0) {
        for (int t = 0; t < M; ++t)
            frt::quant_act_fp4_f32_body<256>(act + (size_t) t * 2048, dst_packed, dst_sfa, 2048, t);
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x - 1) * 8 + warp;   // 0..63: alpha rows then beta
    const float * w = (row < 32 ? w_alpha + (size_t) row * 2048
                                : w_beta + (size_t) (row - 32) * 2048);
    const float4 * w4 = (const float4 *) w;
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;
#pragma unroll 4
    for (int k = lane; k < 512; k += 32) {
        const float4 wv = w4[k];
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float4 av = ((const float4 *) (act + (size_t) t * 2048))[k];
            acc[t] += wv.x * av.x + wv.y * av.y + wv.z * av.z + wv.w * av.w;
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
        if (lane == 0) staging[(size_t) M * 12288 + t * 64 + row] = acc[t];
    }
}

// K2a: grid (32 heads, 8 col-groups) x 128 thr; each block updates 16 state cols
// in place and writes raw attn cols to attn_buf. l2norm/gates recomputed per block.
// M tokens: sequential recurrence per state column, state kept in registers
// across the token loop (1R + 1W per column regardless of M). conv_out and
// attn_buf are token-major (t*8192 / t*4096); a/b live at staging[M*12288 + t*64 + h].
template <int MT>
__global__ void frt_gdn_cell_part(
        const float * __restrict__ conv_out,
        const float * __restrict__ staging,
        const float * __restrict__ dtb,
        const float * __restrict__ A,
        float * __restrict__ s_base,
        const int32_t * __restrict__ s_slot,
        float * __restrict__ attn_buf,        // (M x 4096)
        float l2eps,
        float * __restrict__ s_snap, int64_t s_snap_stride) {
    constexpr int M = MT;
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int h = blockIdx.x;
    const int cg = blockIdx.y;                 // 0..7 -> cols [cg*16, cg*16+16)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float qh[MT][128], kh[MT][128];
    __shared__ float q2s[4], k2s[4];
    const int qk = h & 15;
    float g_val[MT], beta[MT];

#pragma unroll
    for (int t = 0; t < M; ++t) {
        const float qv = conv_out[(size_t) t * 8192 + qk * 128 + tid];
        const float kv = conv_out[(size_t) t * 8192 + 2048 + qk * 128 + tid];
        float q2 = qv * qv, k2 = kv * kv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { q2 += __shfl_xor_sync(0xffffffffu, q2, o); k2 += __shfl_xor_sync(0xffffffffu, k2, o); }
        if (lane == 0) { q2s[warp] = q2; k2s[warp] = k2; }
        __syncthreads();
        const float q2t = q2s[0] + q2s[1] + q2s[2] + q2s[3];
        const float k2t = k2s[0] + k2s[1] + k2s[2] + k2s[3];
        qh[t][tid] = qv * rsqrtf(fmaxf(q2t, l2eps * l2eps));
        kh[t][tid] = kv * rsqrtf(fmaxf(k2t, l2eps * l2eps));
        __syncthreads();
        const float * ab = staging + (size_t) M * 12288 + (size_t) t * 64;
        g_val[t] = expf(frt_softplus(ab[h] + dtb[h]) * A[h]);
        beta[t]  = 1.0f / (1.0f + expf(-ab[32 + h]));
    }
    const float scale = 0.088388347648318447f;

    float * S = s_base + (size_t) (*s_slot) * STATE_SZ + (size_t) h * 16384;

    for (int cc = 0; cc < 4; ++cc) {
        const int c = cg * 16 + warp * 4 + cc;
        float * Sc = S + (size_t) c * 128;
        float s_sh[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) s_sh[r] = Sc[r * 32 + lane];
#pragma unroll
        for (int t = 0; t < M; ++t) {
            const float vc = conv_out[(size_t) t * 8192 + 4096 + h * 128 + c];
            float kvr = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) kvr += s_sh[r] * kh[t][r * 32 + lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) kvr += __shfl_xor_sync(0xffffffffu, kvr, o);
            const float delta = (vc - g_val[t] * kvr) * beta[t];
            float ap = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = r * 32 + lane;
                s_sh[r] = g_val[t] * s_sh[r] + kh[t][i] * delta;
                ap += s_sh[r] * qh[t][i];
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ap += __shfl_xor_sync(0xffffffffu, ap, o);
            if (lane == 0) attn_buf[(size_t) t * 4096 + h * 128 + c] = ap * scale;
            if (MT >= 2) {   // spec verify: per-token state snapshot; source slot untouched
                float * Dc = s_snap + (size_t) t * s_snap_stride + (size_t) h * 16384 + (size_t) c * 128;
#pragma unroll
                for (int r = 0; r < 4; ++r) Dc[r * 32 + lane] = s_sh[r];
            }
        }
        if (MT == 1) {
#pragma unroll
            for (int r = 0; r < 4; ++r) Sc[r * 32 + lane] = s_sh[r];
        }
    }
}

// K2b: per-head gated RMS norm x silu(z) epilogue. Optionally also emits the
// q8_1 quantization of the output (one 32-elem block per warp) so the
// out-proj span can skip its quant launch.
// grid (32 heads, M tokens); z sits per token at staging[t*12288 + 8192 + ...],
// out/out_q8 are token-major (t*4096 floats / t*128 q8 blocks).
__global__ void frt_gdn_epilogue(
        const float * __restrict__ attn_buf,
        const float * __restrict__ staging,
        const float * __restrict__ normw,
        float * __restrict__ out,
        float rmseps, block_q8_1 * __restrict__ out_q8) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float red[4];
    const float xa = attn_buf[(size_t) t * 4096 + h * 128 + tid];
    float s2 = xa * xa;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    if (lane == 0) red[warp] = s2;
    __syncthreads();
    const float mean2 = (red[0] + red[1] + red[2] + red[3]) / 128.0f;
    const float rrms = rsqrtf(mean2 + rmseps);
    const float z = staging[(size_t) t * 12288 + 8192 + h * 128 + tid];
    const float v = xa * rrms * normw[tid] * frt_silu(z);
    out[(size_t) t * 4096 + h * 128 + tid] = v;
    if (out_q8) frt_q8_1_block_g(v, lane, &out_q8[(size_t) t * 128 + h * 4 + warp]);
}

}  // namespace gdn

// ---- GDN cell surgery: detect span in eval loop and execute our kernels ---

// Returns number of nodes consumed starting at i (0 = not ours).
bool ggml_cuda_frt_gdn_try_impl(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static int mode = -1;
    if (mode < 0) mode = frt_window_on("FRT_GDN_SWAP", "GGML_FLASHRT_NO_GDN") ? 1 : 0;
    if (!mode) return false;
    if (!frt::frt_regions_load() || !frt::g_reg.inproj_on) return false;

    // anchor: preferred = the layer's leading RMS_NORM (lets us fold the attn
    // norm into the quant+ab kernel); fallback = the conv-state GET_ROWS.
    ggml_tensor * n0 = cgraph->nodes[i];
    static int nf_mode = -1;
    if (nf_mode < 0) { const char * m = getenv("FRT_GDN_NORMFOLD"); nf_mode = (m && m[0] == '1') ? 1 : 0; }
    bool norm_anchor = false;
    if (nf_mode && n0->op == GGML_OP_RMS_NORM && n0->ne[0] == frt_binding::d_model && n0->ne[1] == 1) {
        norm_anchor = true;   // must find attn_norm MUL + GDN members below
    } else if (n0->op == GGML_OP_RMS_NORM) { return false;
    } else if (n0->op != GGML_OP_GET_ROWS || n0->ne[0] != gdn::CONV_ROW ||
        strncmp(n0->name, "conv_states", 11) != 0) return false;

    // scan forward for the span members
    const int LIM = i + 70 < cgraph->n_nodes ? i + 70 : cgraph->n_nodes;
    int layer = -1;
    const ggml_tensor * qkv_mm = nullptr, * ssm_conv = nullptr, * gdn = nullptr;
    const ggml_tensor * l2n = nullptr, * rmsn = nullptr, * normw_mul = nullptr;
    const ggml_tensor * add_dtb = nullptr, * mul_A = nullptr;
    const ggml_tensor * alpha_mm = nullptr, * beta_mm = nullptr;
    const ggml_tensor * norm_mul = nullptr;
    const ggml_tensor * gr_r = nullptr, * gr_s = nullptr;
    const ggml_tensor * conv_cpy[4] = {nullptr, nullptr, nullptr, nullptr};
    const ggml_tensor * state_cpy = nullptr;
    const ggml_tensor * ck_r_gr = nullptr, * ck_r_cpy = nullptr;
    const ggml_tensor * ck_s_gr = nullptr, * ck_s_cpy = nullptr;
    int n_conv_cpy = 0;
    ggml_tensor * final_rs = nullptr;
    int end_idx = -1;
    for (int j = i; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (j == i && norm_anchor) continue;
        switch (n->op) {
            case GGML_OP_MUL_MAT:
                if (n->src[0] && strstr(n->src[0]->name, ".attn_qkv.weight")) {
                    qkv_mm = n;
                    sscanf(n->src[0]->name, "blk.%d.", &layer);
                }
                if (n->src[0] && strstr(n->src[0]->name, ".ssm_alpha.weight")) alpha_mm = n;
                if (n->src[0] && strstr(n->src[0]->name, ".ssm_beta.weight"))  beta_mm = n;
                break;
            case GGML_OP_SSM_CONV:        ssm_conv = n; break;
            case GGML_OP_GATED_DELTA_NET: gdn = n; break;
            case GGML_OP_L2_NORM:         if (!l2n) l2n = n; break;
            case GGML_OP_RMS_NORM:        rmsn = n; break;
            case GGML_OP_ADD:
                if (n->src[1] && strstr(n->src[1]->name, ".ssm_dt.bias")) add_dtb = n;
                break;
            case GGML_OP_MUL:
                if (n->src[1] && strstr(n->src[1]->name, ".ssm_a")) mul_A = n;
                if (n->src[1] && strstr(n->src[1]->name, ".ssm_norm.weight")) normw_mul = n;
                if (n->src[1] && strstr(n->src[1]->name, ".attn_norm.weight")) norm_mul = n;
                break;
            case GGML_OP_GET_ROWS:
                // first hit = the slot read; a second single-row one is the
                // checkpoint save (we replay it ourselves); anything else -> stock.
                if (n->ne[1] == 1 && n->ne[0] == gdn::CONV_ROW)  { if (!gr_r) gr_r = n; else if (!ck_r_gr) ck_r_gr = n; else return false; }
                else if (n->ne[1] == 1 && n->ne[0] == gdn::STATE_SZ) { if (!gr_s) gr_s = n; else if (!ck_s_gr) ck_s_gr = n; else return false; }
                else if (ggml_nelements(n) != 0 && (n->ne[0] == gdn::CONV_ROW || n->ne[0] == gdn::STATE_SZ)) return false;
                break;
            case GGML_OP_SCALE:
                if (ggml_nelements(n) != 0) return false;   // reset machinery active -> fall back
                break;
            case GGML_OP_RESHAPE:
                if (strncmp(n->name, "final_output", 12) == 0) { final_rs = n; end_idx = j; }
                break;
            case GGML_OP_CPY:
                if (ggml_nelements(n) == 0) break;
                if (n->src[0] && n->src[0]->op == GGML_OP_VIEW &&
                    n->src[0]->ne[0] == 3 && n->src[0]->ne[1] == 8192) {
                    if (n_conv_cpy >= 4) return false;
                    conv_cpy[n_conv_cpy++] = n;
                } else if (n->ne[0] == gdn::STATE_SZ && n->src[0] && n->src[0]->op == GGML_OP_VIEW &&
                           gdn && n->src[0]->src[0] == gdn) {
                    if (state_cpy) return false;
                    state_cpy = n;
                } else if (ck_r_gr && n->src[0] == ck_r_gr) {
                    if (ck_r_cpy) return false;
                    ck_r_cpy = n;
                } else if (ck_s_gr && n->src[0] == ck_s_gr) {
                    if (ck_s_cpy) return false;
                    ck_s_cpy = n;
                } else {
                    return false;   // unknown non-empty copy -> stock runs this eval
                }
                break;
            case GGML_OP_VIEW: case GGML_OP_TRANSPOSE:
            case GGML_OP_CONCAT: case GGML_OP_UNARY:
                break;
            default:
                return false;   // unexpected op inside span -> not ours
        }
        if (end_idx >= 0) break;
    }
    if (layer < 0 || layer >= 64 || !qkv_mm || !ssm_conv || !gdn || !l2n || !rmsn ||
        !normw_mul || !add_dtb || !mul_A || !gr_r || !gr_s || !final_rs) return false;
    if (norm_anchor && (!norm_mul || qkv_mm->src[1] != norm_mul)) return false;
    frt::frt_region & reg = frt::g_reg.regions[0][layer];
    if (reg.N != frt_binding::region_n[0]) return false;
    if (gdn->src[0]->ne[0] != 128 || gdn->src[0]->ne[1] != 16 || gdn->src[2]->ne[1] != 32) return false;
    const int M = (int) qkv_mm->src[1]->ne[1];   // decode M=1; spec verify M = 1 + n_draft
    if (M < 1 || M > 4 || !ggml_is_contiguous(qkv_mm->src[1])) return false;
    if (norm_anchor && M != 1) return false;
    // spec verify (M>1): the graph stores per-token conv/state snapshots for
    // rollback; we must reproduce them (and leave the source slots pristine).
    float * conv_snap[4] = {nullptr, nullptr, nullptr, nullptr};
    float * state_snap = nullptr;
    int64_t state_snap_stride = 0;   // floats between token snapshots
    if (M > 1) {
        if (n_conv_cpy != M || !state_cpy || state_cpy->ne[2] != M) return false;
        if (!state_cpy->src[1] || state_cpy->src[1]->nb[2] % sizeof(float) != 0) return false;
        // conv snapshot CPYs in token order = ascending src view offset into conv_input
        const ggml_tensor * cc_sorted[4];
        for (int t = 0; t < M; ++t) cc_sorted[t] = conv_cpy[t];
        for (int x = 0; x < M; ++x)
            for (int y = x + 1; y < M; ++y)
                if ((const char *) cc_sorted[y]->src[0]->data < (const char *) cc_sorted[x]->src[0]->data) {
                    const ggml_tensor * tmp = cc_sorted[x]; cc_sorted[x] = cc_sorted[y]; cc_sorted[y] = tmp;
                }
        for (int t = 0; t < M; ++t) conv_snap[t] = (float *) cc_sorted[t]->src[1]->data;
        state_snap        = (float *) state_cpy->src[1]->data;
        state_snap_stride = (int64_t) (state_cpy->src[1]->nb[2] / sizeof(float));
    }

    const float l2eps  = *(const float *) l2n->op_params;
    const float rmseps = *(const float *) rmsn->op_params;

    if ((ck_r_gr != nullptr) != (ck_r_cpy != nullptr)) return false;
    if ((ck_s_gr != nullptr) != (ck_s_cpy != nullptr)) return false;

    cudaStream_t stream = ctx.stream();
    // 0) checkpoint saves (pre-update snapshot of the current slot), if due
    if (ck_r_cpy)
        ggml_cuda_kernel_launch(gdn::frt_gdn_ckpt_copy, ggml_cuda_kernel_launch_params(dim3(gdn::CONV_ROW / 256), dim3(256), 0, stream),
            (const float *) ck_r_gr->src[0]->data, (const int32_t *) ck_r_gr->src[1]->data,
            (int64_t) gdn::CONV_ROW, (float *) ck_r_cpy->src[1]->data, gdn::CONV_ROW);
    if (ck_s_cpy)
        ggml_cuda_kernel_launch(gdn::frt_gdn_ckpt_copy, ggml_cuda_kernel_launch_params(dim3(gdn::STATE_SZ / 256), dim3(256), 0, stream),
            (const float *) ck_s_gr->src[0]->data, (const int32_t *) ck_s_gr->src[1]->data,
            (int64_t) gdn::STATE_SZ, (float *) ck_s_cpy->src[1]->data, gdn::STATE_SZ);
    // 1) act quant (+ F32 a/b gate rows; FRT_GDN_AB=0 falls back to W4A4 rows),
    //    then fused in_proj GEMV into staging
    static int ab_f32 = -1;
    if (ab_f32 < 0) { const char * e = getenv("FRT_GDN_AB"); ab_f32 = (e && e[0] == '0') ? 0 : 1; }
    const bool ab_ok = ab_f32 && alpha_mm && beta_mm &&
        alpha_mm->src[0]->type == GGML_TYPE_F32 && beta_mm->src[0]->type == GGML_TYPE_F32 &&
        alpha_mm->src[1]->data == qkv_mm->src[1]->data &&
        beta_mm->src[1]->data == qkv_mm->src[1]->data;
    if (norm_anchor && !ab_ok) return false;   // fallback: plain anchor triggers later
    if (norm_anchor) {
        const float eps = ((const float *) n0->op_params)[0];
        ggml_cuda_kernel_launch(gdn::frt_gdn_norm_quant_ab, ggml_cuda_kernel_launch_params(dim3(9), dim3(256), 0, stream),
            (const float *) n0->src[0]->data, (const float *) norm_mul->src[1]->data, eps,
            (uint2 *) frt::g_reg.d_apack, frt::g_reg.d_sfa,
            (const float *) alpha_mm->src[0]->data, (const float *) beta_mm->src[0]->data,
            frt::g_reg.d_staging);
    } else if (ab_ok) {
        auto qab = M == 1 ? gdn::frt_gdn_quant_ab<1> : M == 2 ? gdn::frt_gdn_quant_ab<2> :
                   M == 3 ? gdn::frt_gdn_quant_ab<3> : gdn::frt_gdn_quant_ab<4>;
        ggml_cuda_kernel_launch(qab, ggml_cuda_kernel_launch_params(dim3(9), dim3(256), 0, stream),
            (const float *) qkv_mm->src[1]->data, (uint2 *) frt::g_reg.d_apack, frt::g_reg.d_sfa,
            (const float *) alpha_mm->src[0]->data, (const float *) beta_mm->src[0]->data,
            frt::g_reg.d_staging);
    } else {
        if (M != 1) return false;   // W4A4 a/b staging fallback stays M=1
        frt::frt_quant_act_launch((const float *) qkv_mm->src[1]->data, frt::g_reg.d_apack, frt::g_reg.d_sfa, (int) reg.K, 1, 0, stream);
    }
    const int gemv_n = (ab_ok || norm_anchor) ? 12288 : (int) reg.N;   // ab rows owned by K0 when on
    frt::frt_ws_launch(frt::g_reg.d_apack, reg.d_packed, frt::g_reg.d_sfa, reg.d_sf,
        frt::g_reg.d_staging, reg.alpha, gemv_n, (int) reg.K, M, stream);
    // 2) conv + shift + silu
    auto conv = M == 1 ? gdn::frt_gdn_conv_silu<1> : M == 2 ? gdn::frt_gdn_conv_silu<2> :
                M == 3 ? gdn::frt_gdn_conv_silu<3> : gdn::frt_gdn_conv_silu<4>;
    ggml_cuda_kernel_launch(conv, ggml_cuda_kernel_launch_params(dim3(32), dim3(256), 0, stream),
        frt::g_reg.d_staging, (const float *) ssm_conv->src[1]->data,
        (float *) gr_r->src[0]->data, (const int32_t *) gr_r->src[1]->data, frt::g_reg.d_conv_out,
        conv_snap[0], conv_snap[1], conv_snap[2], conv_snap[3]);
    // 3) cell: state update in place (256 blocks) + gated-norm epilogue
    {
        dim3 cg(32, 8);
        auto cell = M == 1 ? gdn::frt_gdn_cell_part<1> : M == 2 ? gdn::frt_gdn_cell_part<2> :
                    M == 3 ? gdn::frt_gdn_cell_part<3> : gdn::frt_gdn_cell_part<4>;
        ggml_cuda_kernel_launch(cell, ggml_cuda_kernel_launch_params(cg, dim3(128), 0, stream),
            frt::g_reg.d_conv_out, frt::g_reg.d_staging,
            (const float *) add_dtb->src[1]->data, (const float *) mul_A->src[1]->data,
            (float *) gr_s->src[0]->data, (const int32_t *) gr_s->src[1]->data,
            frt::g_reg.d_attn_buf, l2eps, state_snap, state_snap_stride);
        ggml_cuda_kernel_launch(gdn::frt_gdn_epilogue, ggml_cuda_kernel_launch_params(dim3(32, (unsigned) M), dim3(128), 0, stream),
            frt::g_reg.d_attn_buf, frt::g_reg.d_staging,
            (const float *) normw_mul->src[1]->data,
            (float *) final_rs->data, rmseps, frt::g_reg.d_outq8);
        frt::g_reg.outq8_node = (const void *) final_rs;
    }
    *skip_count = end_idx - i + 1;
    return true;
}

// ---- MoE glue span takeover (FRT_MOEGLUE_SWAP) ----------------------------
// combine span: MUL(weights) + chained ADDs over 8 expert outputs
// (stock fuses the ADDs only partially: MUL + ~2 bcast-adds) -> 1 kernel.
// (router span not taken: stock already fuses it into one topk_moe_cuda.)

namespace moeglue {

// out[i] = sum_e down[e*hidden + i] * w[e]
__global__ void frt_moe_combine(
        const float * __restrict__ down, const float * __restrict__ w,
        float * __restrict__ out, int hidden, int nexp) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < nexp; ++e) acc += down[(size_t) e * hidden + i] * w[e];
    out[i] = acc;
}

// ---- shared-expert span kernels (FRT_SHEXP_SWAP) ----
// span: gate/up GEMV + swiglu + down GEMV + gate_inp dot + sigmoid + mul + add
// replaced by: pre (quant act + sigmoid gate) -> gate|up GEMV (kind2) ->
// glu+quant -> down GEMV (kind3) -> finish.

// block 0: FP4-quantize the f32 act (K=2048); block 1: sigmoid(ginp . act).
__global__ void frt_shexp_pre(
        const float * __restrict__ act,
        uint2 * __restrict__ dst_packed, uint8_t * __restrict__ dst_sfa,
        const float * __restrict__ ginp, float * __restrict__ s_out) {
    if (blockIdx.x == 0) {
        frt::quant_act_fp4_f32_body<256>(act, dst_packed, dst_sfa, 2048);
        return;
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float red[8];
    const float4 * g4 = (const float4 *) ginp;
    const float4 * a4 = (const float4 *) act;
    float acc = 0.0f;
#pragma unroll 2
    for (int k = tid; k < 512; k += 256) {
        const float4 gv = g4[k], av = a4[k];
        acc += gv.x * av.x + gv.y * av.y + gv.z * av.z + gv.w * av.w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        s_out[0] = 1.0f / (1.0f + expf(-s));
    }
}

// one block: swiglu over staging [gate 512 | up 512] then FP4-quantize (K=512).
__global__ void frt_shexp_glu_quant(
        const float * __restrict__ staging,
        uint2 * __restrict__ dst_packed, uint8_t * __restrict__ dst_sfa) {
    __shared__ float tmp[512];
    for (int i = threadIdx.x; i < 512; i += 256) {
        const float g = staging[i];
        tmp[i] = (g / (1.0f + expf(-g))) * staging[512 + i];
    }
    __syncthreads();
    frt::quant_act_fp4_f32_body<256>(tmp, dst_packed, dst_sfa, 512);
}

// out[i] = moe_out[i] + s * down_out[i]
__global__ void frt_shexp_finish(
        const float * __restrict__ moe_out, const float * __restrict__ down_out,
        const float * __restrict__ s, float * __restrict__ out) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < 2048) out[i] = moe_out[i] + s[0] * down_out[i];
}

// ---- fused MoE expert segment (FRT_MOEFUSE_SWAP) --------------------------
// Consumes the whole expert sub-span (gate/up MUL_MAT_ID + GLU + down
// MUL_MAT_ID + weighted combine, ~7 launches) with 2 kernels that read the
// GGUF-native quant blocks via llama.cpp's own vec_dot device functions and
// replicate its q8_1 activation quantization: same math, no repacking, the
// win is launch-count and intermediate-tensor elimination.

// per-32-elem q8_1 quantization identical to quantize_q8_1 (d=amax/127, s=raw sum)
__device__ __forceinline__ void frt_q8_1_block(float xi, int lane, block_q8_1 * dst) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, sum);
}

// one warp computes one full row dot against q8_1 blocks, mmvq iteration order.
template <ggml_type T>
__device__ __forceinline__ float frt_row_dot(const void * row_base, const block_q8_1 * y,
        int blocks_per_row, int lane) {
    constexpr int qi  = ggml_cuda_type_traits<T>::qi;
    constexpr int qk  = ggml_cuda_type_traits<T>::qk;
    constexpr int vdr = T == GGML_TYPE_Q8_0 ? VDR_Q8_0_Q8_1_MMVQ :
                        T == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ :
                        T == GGML_TYPE_Q6_K ? VDR_Q6_K_Q8_1_MMVQ : VDR_Q5_K_Q8_1_MMVQ;
    float acc = 0.0f;
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row; kbx += vdr * 32 / qi) {
        const int kqs = vdr * (lane % (qi / vdr));
        if constexpr (T == GGML_TYPE_Q8_0) acc += vec_dot_q8_0_q8_1(row_base, &y[kbx * (qk / QK8_1)], kbx, kqs);
        if constexpr (T == GGML_TYPE_Q4_K) acc += vec_dot_q4_K_q8_1(row_base, &y[kbx * (qk / QK8_1)], kbx, kqs);
        if constexpr (T == GGML_TYPE_Q5_K) acc += vec_dot_q5_K_q8_1(row_base, &y[kbx * (qk / QK8_1)], kbx, kqs);
        if constexpr (T == GGML_TYPE_Q6_K) acc += vec_dot_q6_K_q8_1(row_base, &y[kbx * (qk / QK8_1)], kbx, kqs);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// K1: grid (n_used, N/32) x 256. Block = 32 rows of one selected expert:
// quantize act (redundant per block, latency-hidden), gate+up row dots,
// swiglu, requantize the 32 glu outputs into one q8_1 block.
// meta = private snapshot of ids + weights: the graph allocator may alias the
// out tensor onto the (dead-after-us) ids/weights buffers, so K2 must not read
// them while it writes out. K0 snapshots them first (stream order protects K0).
struct frt_moe_meta { int32_t ids[4][8]; float w[4][8]; float sig[4]; };

// K0: grid 8 (+1 when ginp): q8_1-quantize the shared act (M token rows) +
// snapshot per-token ids/weights (+ shexp gate: sigmoid(ginp . act_t) -> meta->sig[t]).
// ids rows are strided (topk is a view of the argsort output), hence ids_srow.
template <int MT>
__global__ void frt_moe_quant_meta(
        const float * __restrict__ act, int K, int64_t act_srow,
        const int32_t * __restrict__ ids, int64_t ids_srow,
        const float * __restrict__ wnorm, int64_t w_srow, int n_used,
        const float * __restrict__ ginp,
        block_q8_1 * __restrict__ act_q8, frt_moe_meta * __restrict__ meta) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = K / 32;
    if (blockIdx.x == 8) {   // shexp gate dots (launched only when ginp != null)
        __shared__ float red[8];
        const float4 * g4 = (const float4 *) ginp;
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const float4 * a4 = (const float4 *) (act + (size_t) t * act_srow);
            float acc = 0.0f;
            for (int k = threadIdx.x; k < K / 4; k += 256) {
                const float4 gv = g4[k], av = a4[k];
                acc += gv.x * av.x + gv.y * av.y + gv.z * av.z + gv.w * av.w;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) red[warp] = acc;
            __syncthreads();
            if (threadIdx.x == 0) {
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < 8; ++q) s += red[q];
                meta->sig[t] = 1.0f / (1.0f + expf(-s));
            }
            __syncthreads();
        }
        return;
    }
    if (blockIdx.x == 0 && threadIdx.x < (unsigned) (n_used * MT)) {
        const int t = MT == 1 ? 0 : (int) threadIdx.x / n_used;
        const int j = MT == 1 ? (int) threadIdx.x : (int) threadIdx.x % n_used;
        meta->ids[t][j] = ids[t * ids_srow + j];
        meta->w[t][j]   = wnorm[t * w_srow + j];
    }
    const int per = (nb + 7) / 8;
#pragma unroll
    for (int t = 0; t < MT; ++t)
        for (int b = blockIdx.x * per + warp; b < (blockIdx.x + 1) * per && b < nb; b += 8)
            frt_q8_1_block(act[(size_t) t * act_srow + b * 32 + lane], lane, &act_q8[t * nb + b]);
}

// K1: grid (n_used, N/32) x 512 (16 warps, 2 rows each): gate+up row dots
// against the pre-quantized act, swiglu, requantize 32 outputs per block.
template <ggml_type TW, int MT>
__global__ void frt_moe_k1(
        const char * __restrict__ gate_w, const char * __restrict__ up_w,
        size_t expert_stride, size_t row_stride,
        const block_q8_1 * __restrict__ act_q8, int K,
        const frt_moe_meta * __restrict__ meta,
        block_q8_1 * __restrict__ glu_q8, int N,
        int n_used, const char * __restrict__ shg_w, const char * __restrict__ shu_w,
        size_t sh_row_stride) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float glu_f32[32];
    const int row0 = blockIdx.y * 32;
    // grid.x covers M x slots pairs: slot in [0,n_used) = routed expert, slot n_used = shexp
    const int slots = shg_w ? n_used + 1 : n_used;
    const int t     = MT == 1 ? 0 : (int) blockIdx.x / slots;
    const int slot  = MT == 1 ? (int) blockIdx.x : (int) blockIdx.x % slots;
    const block_q8_1 * aq   = act_q8 + (size_t) t * (K / 32);
    block_q8_1       * gout = glu_q8 + ((size_t) t * slots + slot) * (N / 32);
    if (shg_w && slot == n_used) {   // shared-expert gate|up rows, Q8_0
        const int bpr8 = K / 32;
        const int ra = 2 * warp, rb = ra + 1;
        const char * ga = shg_w + (size_t)(row0 + ra) * sh_row_stride;
        const char * gb = shg_w + (size_t)(row0 + rb) * sh_row_stride;
        const char * ua = shu_w + (size_t)(row0 + ra) * sh_row_stride;
        const char * ub = shu_w + (size_t)(row0 + rb) * sh_row_stride;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int kbx = lane / 4; kbx < bpr8; kbx += 8) {
            const int kqs = 2 * (lane % 4);
            const block_q8_1 * y = &aq[kbx];
            a0 += vec_dot_q8_0_q8_1(ga, y, kbx, kqs);
            a1 += vec_dot_q8_0_q8_1(ua, y, kbx, kqs);
            a2 += vec_dot_q8_0_q8_1(gb, y, kbx, kqs);
            a3 += vec_dot_q8_0_q8_1(ub, y, kbx, kqs);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a0 += __shfl_xor_sync(0xffffffffu, a0, o);
            a1 += __shfl_xor_sync(0xffffffffu, a1, o);
            a2 += __shfl_xor_sync(0xffffffffu, a2, o);
            a3 += __shfl_xor_sync(0xffffffffu, a3, o);
        }
        if (lane == 0) {
            glu_f32[ra] = (a0 / (1.0f + expf(-a0))) * a1;
            glu_f32[rb] = (a2 / (1.0f + expf(-a2))) * a3;
        }
        __syncthreads();
        if (warp == 0)
            frt_q8_1_block(glu_f32[lane], lane, &gout[blockIdx.y]);
        return;
    }
    const int e = meta->ids[t][slot];
    const int bpr = K / ggml_cuda_type_traits<TW>::qk;
    constexpr int qi  = ggml_cuda_type_traits<TW>::qi;
    constexpr int qk  = ggml_cuda_type_traits<TW>::qk;
    constexpr int vdr = TW == GGML_TYPE_Q8_0 ? VDR_Q8_0_Q8_1_MMVQ :
                        TW == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_Q5_K_Q8_1_MMVQ;
    // warp owns rows {2*warp, 2*warp+1}; 4 independent dot accumulators per
    // kbx iteration (2 rows x gate/up) so the weight loads overlap 4-wide.
    {
        const int ra = 2 * warp, rb = ra + 1;
        const size_t ebase = (size_t) e * expert_stride;
        const char * ga = gate_w + ebase + (size_t)(row0 + ra) * row_stride;
        const char * gb = gate_w + ebase + (size_t)(row0 + rb) * row_stride;
        const char * ua = up_w   + ebase + (size_t)(row0 + ra) * row_stride;
        const char * ub = up_w   + ebase + (size_t)(row0 + rb) * row_stride;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int kbx = lane / (qi / vdr); kbx < bpr; kbx += vdr * 32 / qi) {
            const int kqs = vdr * (lane % (qi / vdr));
            const block_q8_1 * y = &aq[kbx * (qk / QK8_1)];
            if constexpr (TW == GGML_TYPE_Q8_0) {
                a0 += vec_dot_q8_0_q8_1(ga, y, kbx, kqs);
                a1 += vec_dot_q8_0_q8_1(ua, y, kbx, kqs);
                a2 += vec_dot_q8_0_q8_1(gb, y, kbx, kqs);
                a3 += vec_dot_q8_0_q8_1(ub, y, kbx, kqs);
            } else if constexpr (TW == GGML_TYPE_Q4_K) {
                a0 += vec_dot_q4_K_q8_1(ga, y, kbx, kqs);
                a1 += vec_dot_q4_K_q8_1(ua, y, kbx, kqs);
                a2 += vec_dot_q4_K_q8_1(gb, y, kbx, kqs);
                a3 += vec_dot_q4_K_q8_1(ub, y, kbx, kqs);
            } else {
                a0 += vec_dot_q5_K_q8_1(ga, y, kbx, kqs);
                a1 += vec_dot_q5_K_q8_1(ua, y, kbx, kqs);
                a2 += vec_dot_q5_K_q8_1(gb, y, kbx, kqs);
                a3 += vec_dot_q5_K_q8_1(ub, y, kbx, kqs);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a0 += __shfl_xor_sync(0xffffffffu, a0, o);
            a1 += __shfl_xor_sync(0xffffffffu, a1, o);
            a2 += __shfl_xor_sync(0xffffffffu, a2, o);
            a3 += __shfl_xor_sync(0xffffffffu, a3, o);
        }
        if (lane == 0) {
            glu_f32[ra] = (a0 / (1.0f + expf(-a0))) * a1;
            glu_f32[rb] = (a2 / (1.0f + expf(-a2))) * a3;
        }
    }
    __syncthreads();
    if (warp == 0)
        frt_q8_1_block(glu_f32[lane], lane, &gout[blockIdx.y]);
}

// K2: warp per output column: 8 expert row-dots over the staged glu q8_1
// vectors, weighted sum, single write. grid hidden/8 x 256.
template <ggml_type TD, int MT>
__global__ void frt_moe_k2(
        const char * __restrict__ down_w, size_t expert_stride, size_t row_stride,
        const block_q8_1 * __restrict__ glu_q8,
        const frt_moe_meta * __restrict__ meta,
        float * __restrict__ out, int Kd, int n_used, int hidden,
        const char * __restrict__ shd_w, size_t shd_row_stride,
        const float * __restrict__ resid) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = blockIdx.x * 8 + warp;
    if (col >= hidden) return;
    const int t     = MT == 1 ? 0 : (int) blockIdx.y;   // token row (grid.y = M)
    const int slots = shd_w ? n_used + 1 : n_used;      // must mirror K1's glu layout
    const block_q8_1 * gq = glu_q8 + (size_t) t * slots * (Kd / 32);
    const int bpr = Kd / ggml_cuda_type_traits<TD>::qk;
    const int kdb = Kd / 32;
    float res = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const char * row = down_w + (size_t) meta->ids[t][e] * expert_stride + (size_t) col * row_stride;
        res += frt_row_dot<TD>(row, &gq[e * kdb], bpr, lane) * meta->w[t][e];
    }
    if (shd_w) {   // shared expert: sigmoid-gated Q8_0 down
        const char * row = shd_w + (size_t) col * shd_row_stride;
        res += frt_row_dot<GGML_TYPE_Q8_0>(row, &gq[n_used * kdb], Kd / 32, lane) * meta->sig[t];
    }
    const size_t oi = (size_t) t * hidden + col;
    if (lane == 0) out[oi] = resid ? res + resid[oi] : res;
}

// attn output gate: out[i] = fa[i] * sigmoid(gate[i]) (replaces CONT+UNARY+MUL)
__global__ void frt_attn_gate(
        const float * __restrict__ fa, const float * __restrict__ gate,
        float * __restrict__ out, int n) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = fa[i] * (1.0f / (1.0f + expf(-gate[i])));
}

// fused router: gate_inp logits GEMV + exact replication of their topk_moe
// (softmax over n_exp, iterative top-k with lower-index tie-break, clamp-norm).
// grid 8 x 256; last finishing block runs the warp top-k (self-resetting counter).
__global__ void frt_router_fused(
        const float * __restrict__ gate_w,     // [256, 2048] f32 K-contig
        const float * __restrict__ act, int K,
        int n_used, float clamp_val,
        float * __restrict__ logits_buf, unsigned int * __restrict__ counter,
        int32_t * __restrict__ ids_out, float * __restrict__ w_out,
        int64_t ids_srow, int64_t w_srow) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = blockIdx.y;                  // token row (grid.y = M)
    act        += (size_t) t * K;
    logits_buf += (size_t) t * 256;
    counter    += t;
    ids_out    += (size_t) t * ids_srow;
    w_out      += (size_t) t * w_srow;
    {   // phase 1: warp per row, grid 32 x 8 warps = 256 rows
        const int row = blockIdx.x * 8 + warp;
        const float4 * A = (const float4 *) act;
        const float4 * W = (const float4 *) (gate_w + (size_t) row * K);
        float a0 = 0, a1 = 0;
        for (int k = lane; k < K / 4; k += 64) {
            const float4 av0 = A[k], w0 = W[k];
            const float4 av1 = A[k + 32], w1 = W[k + 32];
            a0 += w0.x * av0.x + w0.y * av0.y + w0.z * av0.z + w0.w * av0.w;
            a1 += w1.x * av1.x + w1.y * av1.y + w1.z * av1.z + w1.w * av1.w;
        }
        float acc = a0 + a1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) logits_buf[row] = acc;
    }
    __shared__ bool amlast;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) amlast = (atomicInc(counter, gridDim.x - 1) == gridDim.x - 1);
    __syncthreads();
    if (!amlast || warp != 0) return;

    // phase 2: single warp, exact topk_moe semantics (no bias, softmax, with_norm)
    float wt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wt[i] = logits_buf[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) mx = fmaxf(mx, wt[i]);
    mx = warp_reduce_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float v = expf(wt[i] - mx); wt[i] = v; sum += v; }
    sum = warp_reduce_sum(sum);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        wt[i] *= inv;
        if (__isnanf(wt[i])) wt[i] = -FLT_MAX;
    }
    float wt_sum = 0.0f, outw = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        float max_val = wt[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            const int e = lane + i * 32;
            if (wt[i] > max_val) { max_val = wt[i]; max_expert = e; }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val = __shfl_xor_sync(0xffffffffu, max_val, mask, 32);
            const int   e   = __shfl_xor_sync(0xffffffffu, max_expert, mask, 32);
            if (val > max_val || (val == max_val && e < max_expert)) { max_val = val; max_expert = e; }
        }
        if ((max_expert & 31) == lane) {
            wt[max_expert / 32] = -INFINITY;
            ids_out[k] = max_expert;
            wt_sum += max_val;
        }
        if (k == lane) outw = max_val;
    }
    wt_sum = warp_reduce_sum(wt_sum);
    wt_sum = fmaxf(wt_sum, clamp_val);
    const float invs = 1.0f / wt_sum;
    if (lane < n_used) w_out[lane] = outw * invs;
}

// out-proj (ssm_out / attn_output) native-format GEMV with fused residual add.
// warp per output column, grid.y = token row; reads the q8_1-staged act
// (K/32 blocks per token), writes out = dot + residual.
template <ggml_type TW>
__global__ void frt_outproj_gemv(
        const char * __restrict__ w, size_t row_stride,
        const block_q8_1 * __restrict__ y,
        const float * __restrict__ residual,
        float * __restrict__ out, int N, int K) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = blockIdx.x * 8 + warp;
    if (col >= N) return;
    const int t = blockIdx.y;
    const float acc = frt_row_dot<TW>(w + (size_t) col * row_stride, y + (size_t) t * (K / 32),
                                      K / ggml_cuda_type_traits<TW>::qk, lane);
    const size_t oi = (size_t) t * N + col;
    if (lane == 0) out[oi] = acc + residual[oi];
}

}  // namespace moeglue

// attn-gate glue span: CONT(gate view) -> UNARY sigmoid -> MUL -> 1 kernel.
static bool frt_attn_gate_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static int mode = -1;
    if (mode < 0) { const char * m = getenv("FRT_ATTNGATE_SWAP"); mode = (m && m[0] == '1') ? 1 : 0; }
    if (!mode) return false;
    ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_CONT || strncmp(n0->name, "gate_reshaped", 13) != 0) return false;
    if (!ggml_is_contiguous(n0->src[0])) return false;   // gate view must be flat
    const ggml_tensor * sig = nullptr;
    ggml_tensor * gmul = nullptr;
    int end_idx = -1;
    const int LIM = i + 4 < cgraph->n_nodes ? i + 4 : cgraph->n_nodes;
    for (int j = i + 1; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_UNARY) {
            if (ggml_get_unary_op(n) != GGML_UNARY_OP_SIGMOID || n->src[0] != n0) return false;
            sig = n;
        } else if (n->op == GGML_OP_MUL && sig) {
            if (n->src[1] == sig)      { gmul = n; end_idx = j; }
            else if (n->src[0] == sig) { gmul = n; end_idx = j; }
            break;
        } else if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) {
            continue;
        } else return false;
    }
    if (!sig || !gmul || end_idx < 0) return false;
    const ggml_tensor * fa = gmul->src[0] == sig ? gmul->src[1] : gmul->src[0];
    const int64_t n = gmul->ne[0];
    if (n % 256 != 0 || gmul->ne[1] != 1) return false;
    if (fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa)) return false;
    // out may alias the (dead-after-us) gate source at a different index: check overlap
    const char * gsrc = (const char *) n0->src[0]->data;
    const char * outp = (const char *) gmul->data;
    if (outp < gsrc + n * 4 && gsrc < outp + n * 4 && outp != (const char *) fa->data) {
        if (outp != gsrc) return false;   // partial overlap -> unsafe, fall back
    }
    ggml_cuda_kernel_launch(moeglue::frt_attn_gate, ggml_cuda_kernel_launch_params(dim3((unsigned) (n / 256)), dim3(256), 0, ctx.stream()),
        (const float *) fa->data, (const float *) n0->src[0]->data,
        (float *) gmul->data, (int) n);
    *skip_count = end_idx - i + 1;
    return true;
}

// fused router span: [MUL_MAT gate_inp logits] -> SOFT_MAX -> ARGSORT -> ... -> DIV
// (their path: mmv_f + fused topk_moe = 2 launches) -> 1 kernel.
static bool frt_router_span_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    // mode: unset -> M=1 off (their fused topk_moe wins there, judged -3), M>=2 on
    // (their fusion is M=1-only; the M=2 fallback is the unfused argsort chain).
    // "1" -> on for all M; "0" -> off entirely.
    static int mode = -1;
    if (mode < 0) { const char * m = getenv("FRT_ROUTER_SWAP"); mode = !frt_layer_enabled() ? 0 : (m ? ((m[0] == '1') ? 2 : 0) : 1); }
    if (!mode) return false;
    ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_MUL_MAT || !n0->src[0] ||
        !strstr(n0->src[0]->name, ".ffn_gate_inp.weight")) return false;
    const ggml_tensor * w = n0->src[0];
    const ggml_tensor * act = n0->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 2048 || w->ne[1] != 256) return false;
    if (act->type != GGML_TYPE_F32 || !ggml_is_contiguous(act)) return false;
    const int M = (int) act->ne[1];
    if (M < 1 || M > 4 || act->ne[2] != 1) return false;

    const ggml_tensor * sm = nullptr, * argsort = nullptr, * clampn = nullptr;
    ggml_tensor * divn = nullptr;
    int end_idx = -1;
    const int LIM = i + 12 < cgraph->n_nodes ? i + 12 : cgraph->n_nodes;
    for (int j = i + 1; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_SOFT_MAX:
                if (n->src[0] != n0 || n->src[1] != nullptr) return false;
                if (((const float *) n->op_params)[0] != 1.0f ||
                    ((const float *) n->op_params)[1] != 0.0f) return false;
                sm = n;
                break;
            case GGML_OP_ARGSORT:  argsort = n; break;
            case GGML_OP_CLAMP:    clampn = n; break;
            case GGML_OP_DIV:
                if (strstr(n->name, "ffn_moe_weights_norm")) { divn = n; end_idx = j; }
                break;
            case GGML_OP_RESHAPE: case GGML_OP_VIEW:
            case GGML_OP_GET_ROWS: case GGML_OP_SUM_ROWS:
                break;
            default: return false;
        }
        if (end_idx >= 0) break;
    }
    if (!sm || !argsort || !clampn || !divn || argsort->ne[0] != 256 || divn->ne[0] > 8) return false;
    if (mode == 1 && M == 1) return false;   // default policy: M=1 stays on their fused path
    if (argsort->ne[1] != M || divn->ne[1] != M) return false;
    const int n_used = (int) divn->ne[0];
    const float cmin = ((const float *) clampn->op_params)[0];

    static float * d_logits = nullptr;
    static unsigned int * d_counter = nullptr;
    if (!d_logits) {
        CUDA_CHECK(cudaMalloc(&d_logits, 4 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counter, 4 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_counter, 0, 4 * sizeof(unsigned int)));
    }
    ggml_cuda_kernel_launch(moeglue::frt_router_fused, ggml_cuda_kernel_launch_params(dim3(32, (unsigned) M), dim3(256), 0, ctx.stream()),
        (const float *) w->data, (const float *) act->data, 2048,
        n_used, cmin, d_logits, d_counter,
        (int32_t *) argsort->data, (float *) divn->data,
        (int64_t) (argsort->nb[1] / sizeof(int32_t)), (int64_t) (divn->nb[1] / sizeof(float)));
    *skip_count = end_idx - i + 1;
    return true;
}

// fused MoE expert segment: anchor = MUL_MAT_ID on ffn_gate_exps.weight.
static bool frt_moefuse_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static int mode = -1;
    if (mode < 0) mode = frt_window_on("FRT_MOEFUSE_SWAP", "GGML_FLASHRT_NO_MOEFUSE") ? 1 : 0;
    if (!mode) return false;
    ggml_tensor * gate_id = cgraph->nodes[i];
    if (gate_id->op != GGML_OP_MUL_MAT_ID || !gate_id->src[0] ||
        !strstr(gate_id->src[0]->name, ".ffn_gate_exps.weight")) return false;
    static int stats_on = -1;
    if (stats_on < 0) { const char * e = getenv("FRT_STATS"); stats_on = (e && e[0] == '1') ? 1 : 0; }
    static long anchors = 0; static long fires_ok = 0;
    if (stats_on && (++anchors & 0x3FF) == 1) fprintf(stderr, "frt-stats moefuse anchors=%ld fires=%ld\n", anchors, fires_ok);
    static int dbg = -1;
    if (dbg < 0) { const char * d = getenv("FRT_MOEFUSE_DBG"); dbg = d ? atoi(d) : 0; }
    struct frt_dbg_guard {
        bool armed; ggml_cgraph * g; int i;
        ~frt_dbg_guard() {
            if (!armed) return;
            fprintf(stderr, "frt-moefuse-dbg REJECT at anchor %d:\n", i);
            for (int j = i; j < i + 30 && j < g->n_nodes; ++j) {
                const ggml_tensor * n = g->nodes[j];
                fprintf(stderr, "  %4d %-14s %-26s ne=[%lld,%lld,%lld]\n", j, ggml_op_name(n->op), n->name,
                        (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2]);
            }
        }
    } dbg_guard{false, cgraph, i};
    if (dbg > 0) { dbg_guard.armed = true; --dbg; }

    const ggml_tensor * gw = gate_id->src[0];
    const ggml_tensor * act = gate_id->src[1];
    const ggml_tensor * ids = gate_id->src[2];
    const int64_t K = gw->ne[0], N = gw->ne[1];
    const int n_used = (int) gate_id->ne[1];
    if (gw->type != GGML_TYPE_Q4_K && gw->type != GGML_TYPE_Q8_0) return false;
    if (K != 2048 || N % 32 != 0 || N > 2048) return false;
    if (act->type != GGML_TYPE_F32 || act->ne[1] != 1 || !ggml_is_contiguous(act)) return false;
    const int M = (int) act->ne[2];   // token-batch width (spec verify runs M = 1 + n_draft, default 4)
    if (M < 1 || M > 4) return false;
    if (n_used < 1 || n_used > 8 || gate_id->ne[2] != M) return false;
    if (ids->ne[1] != M) return false;

    static int sh_mode = -1;
    if (sh_mode < 0) sh_mode = frt_window_on("FRT_MOEFUSE_SHEXP", "GGML_FLASHRT_NO_SHEXP_FOLD") ? 1 : 0;
    const ggml_tensor * up_id = nullptr, * glu = nullptr, * down_id = nullptr;
    const ggml_tensor * wmul = nullptr;
    const ggml_tensor * shg = nullptr, * shu = nullptr, * shd = nullptr, * ginp = nullptr;
    const ggml_tensor * sh_glu = nullptr, * sig = nullptr, * gmul = nullptr;
    ggml_tensor * out_add = nullptr, * moe_add = nullptr;
    int end_idx = -1;
    const int LIM = i + 2 * n_used + 22 < cgraph->n_nodes ? i + 2 * n_used + 22 : cgraph->n_nodes;
    for (int j = i + 1; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_MUL_MAT_ID:
                if (n->src[0] && strstr(n->src[0]->name, ".ffn_up_exps.weight"))        up_id = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_down_exps.weight")) down_id = n;
                else return false;
                break;
            case GGML_OP_MUL_MAT:
                if (!sh_mode || !moe_add) return false;   // shexp mul_mats only after moe_out
                if (n->src[0] && strstr(n->src[0]->name, ".ffn_gate_shexp.weight"))          shg = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_up_shexp.weight"))       shu = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_down_shexp.weight"))     shd = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_gate_inp_shexp.weight")) ginp = n;
                else return false;
                break;
            case GGML_OP_GLU:
                if (ggml_get_glu_op(n) != GGML_GLU_OP_SWIGLU) return false;
                if (n->src[0] == gate_id) glu = n;
                else if (shg && n->src[0] == shg) sh_glu = n;
                else return false;
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(n) != GGML_UNARY_OP_SIGMOID || !ginp || n->src[0] != ginp) return false;
                sig = n;
                break;
            case GGML_OP_MUL:
                if (!moe_add) wmul = n;
                else gmul = n;
                break;
            case GGML_OP_ADD:
                if (strstr(n->name, "ffn_moe_out")) {           // also matches mtp_ffn_moe_out (draft)
                    moe_add = n;
                    if (!sh_mode) { out_add = n; end_idx = j; }
                } else if (sh_mode && strstr(n->name, "ffn_out")) {   // also mtp_ffn_out
                    out_add = n; end_idx = j;
                }
                break;
            case GGML_OP_VIEW: case GGML_OP_RESHAPE:
                break;
            default:
                return false;
        }
        if (end_idx >= 0) break;
    }
    if (!up_id || !glu || !down_id || !wmul || !out_add || !moe_add) return false;
    // shexp wiring (only in sh_mode)
    const bool sh_ok = sh_mode && shg && shu && shd && ginp && sh_glu && sig && gmul &&
        shg->src[0]->type == GGML_TYPE_Q8_0 && shu->src[0]->type == GGML_TYPE_Q8_0 &&
        shd->src[0]->type == GGML_TYPE_Q8_0 && ginp->src[0]->type == GGML_TYPE_F32 &&
        shg->src[0]->ne[0] == K && shg->src[0]->ne[1] == N &&
        shu->src[0]->ne[0] == K && shu->src[0]->ne[1] == N &&
        shd->src[0]->ne[0] == N && shd->src[0]->ne[1] == K &&
        shg->src[1]->data == act->data && shu->src[1]->data == act->data &&
        ginp->src[1]->data == act->data &&
        sh_glu->src[1] == shu && shd->src[1] == sh_glu &&
        ((gmul->src[0] == shd && gmul->src[1] == sig) || (gmul->src[0] == sig && gmul->src[1] == shd)) &&
        ((out_add->src[0] == moe_add && out_add->src[1] == gmul) ||
         (out_add->src[0] == gmul && out_add->src[1] == moe_add));
    if (sh_mode && !sh_ok) return false;
    // fold the layer residual add too when the very next real node is l_out = ffn_out + resid
    const ggml_tensor * lresid = nullptr;
    if (sh_ok && end_idx + 1 < cgraph->n_nodes) {
        ggml_tensor * nl = cgraph->nodes[end_idx + 1];
        if (nl->op == GGML_OP_ADD &&
            (strncmp(nl->name, "l_out", 5) == 0 || strncmp(nl->name, "mtp_post_ffn", 12) == 0)) {
            if (nl->src[0] == out_add && nl->src[1] != out_add) { lresid = nl->src[1]; out_add = nl; end_idx += 1; }
            else if (nl->src[1] == out_add && nl->src[0] != out_add) { lresid = nl->src[0]; out_add = nl; end_idx += 1; }
        }
    }
    // wiring
    if (up_id->src[0]->type != gw->type || up_id->src[0]->ne[0] != K || up_id->src[0]->ne[1] != N) return false;
    if (up_id->src[1]->data != act->data || up_id->src[2]->data != ids->data) return false;
    if (glu->src[0] != gate_id || glu->src[1] != up_id) return false;
    if (down_id->src[1] != glu || down_id->src[2]->data != ids->data) return false;
    const ggml_tensor * dw = down_id->src[0];
    if (dw->ne[0] != N || dw->ne[1] != K) return false;
    if (dw->type != GGML_TYPE_Q5_K && dw->type != GGML_TYPE_Q4_K && dw->type != GGML_TYPE_Q8_0 &&
        dw->type != GGML_TYPE_Q6_K) return false;
    if (wmul->src[0] != down_id || wmul->src[1]->type != GGML_TYPE_F32) return false;
    const ggml_tensor * wnorm = wmul->src[1];

    static block_q8_1 * d_gluq8 = nullptr;
    static block_q8_1 * d_actq8 = nullptr;
    static moeglue::frt_moe_meta * d_meta = nullptr;
    if (!d_gluq8) {
        CUDA_CHECK(cudaMalloc(&d_gluq8, 4 * 9 * 64 * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&d_actq8, 4 * 64 * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&d_meta, sizeof(moeglue::frt_moe_meta)));
    }


    cudaStream_t stream = ctx.stream();

    // FRT_MOEFUSE_SELFTEST=1: before the real launches (inputs still pristine —
    // the out write may alias them), replay this span's input as both an M=1 run
    // and a duplicated-token M=2 run on scratch buffers; all three out rows must
    // be bit-identical (kills any (t, stride) indexing bug in the M=2 path).
    static int selftest = -1;
    if (selftest < 0) { const char * s = getenv("FRT_MOEFUSE_SELFTEST"); selftest = (s && s[0]=='1') ? 1 : 0; }
    if (selftest == 1 && M == 1 && dw->type == GGML_TYPE_Q5_K) {
        selftest = 2;   // once
        float * s_act; int32_t * s_ids; float * s_w; float * s_out1; float * s_out2; float * s_resid;
        block_q8_1 * s_actq8; block_q8_1 * s_gluq8; moeglue::frt_moe_meta * s_meta1; moeglue::frt_moe_meta * s_meta2;
        CUDA_CHECK(cudaMalloc(&s_act,   2 * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_ids,   2 * 8 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&s_w,     2 * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_out1,  K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_out2,  2 * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_resid, 2 * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s_actq8, 2 * 64 * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&s_gluq8, 2 * 9 * 64 * sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&s_meta1, sizeof(moeglue::frt_moe_meta)));
        CUDA_CHECK(cudaMalloc(&s_meta2, sizeof(moeglue::frt_moe_meta)));
        for (int t = 0; t < 2; ++t) {
            cudaMemcpyAsync(s_act + t * K, act->data, K * 4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_ids + t * 8, ids->data, n_used * 4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(s_w   + t * 8, wnorm->data, n_used * 4, cudaMemcpyDeviceToDevice, stream);
            if (lresid) cudaMemcpyAsync(s_resid + t * K, lresid->data, K * 4, cudaMemcpyDeviceToDevice, stream);
        }
        const float * s_ginp = sh_ok ? (const float *) ginp->src[0]->data : (const float *) nullptr;
        const char  * s_shg  = sh_ok ? (const char *) shg->src[0]->data : (const char *) nullptr;
        const char  * s_shu  = sh_ok ? (const char *) shu->src[0]->data : (const char *) nullptr;
        const char  * s_shd  = sh_ok ? (const char *) shd->src[0]->data : (const char *) nullptr;
        const size_t s_shs   = sh_ok ? shg->src[0]->nb[1] : (size_t) 0;
        const size_t s_shds  = sh_ok ? shd->src[0]->nb[1] : (size_t) 0;
        for (int m = 1; m <= 2; ++m) {
            block_q8_1 * aq = s_actq8;  moeglue::frt_moe_meta * mt = (m == 1) ? s_meta1 : s_meta2;
            float * so = (m == 1) ? s_out1 : s_out2;
            dim3 tg0(sh_ok ? 9 : 8), tg1((unsigned) (m * (sh_ok ? n_used + 1 : n_used)), (unsigned) (N / 32)), tg2((unsigned) ((K + 7) / 8), (unsigned) m);
            if (m == 1) {
                moeglue::frt_moe_quant_meta<1><<<tg0, dim3(256), 0, stream>>>(
                    s_act, (int) K, (int64_t) K, s_ids, (int64_t) 8, s_w, (int64_t) 8, n_used,
                    s_ginp, aq, mt);
                moeglue::frt_moe_k1<GGML_TYPE_Q4_K, 1><<<tg1, dim3(512), 0, stream>>>(
                    (const char *) gw->data, (const char *) up_id->src[0]->data, gw->nb[2], gw->nb[1],
                    aq, (int) K, mt, s_gluq8, (int) N, n_used, s_shg, s_shu, s_shs);
                moeglue::frt_moe_k2<GGML_TYPE_Q5_K, 1><<<tg2, dim3(256), 0, stream>>>(
                    (const char *) dw->data, dw->nb[2], dw->nb[1], s_gluq8, mt,
                    so, (int) N, n_used, (int) K, s_shd, s_shds,
                    lresid ? s_resid : (const float *) nullptr);
            } else {
                moeglue::frt_moe_quant_meta<2><<<tg0, dim3(256), 0, stream>>>(
                    s_act, (int) K, (int64_t) K, s_ids, (int64_t) 8, s_w, (int64_t) 8, n_used,
                    s_ginp, aq, mt);
                moeglue::frt_moe_k1<GGML_TYPE_Q4_K, 2><<<tg1, dim3(512), 0, stream>>>(
                    (const char *) gw->data, (const char *) up_id->src[0]->data, gw->nb[2], gw->nb[1],
                    aq, (int) K, mt, s_gluq8, (int) N, n_used, s_shg, s_shu, s_shs);
                moeglue::frt_moe_k2<GGML_TYPE_Q5_K, 2><<<tg2, dim3(256), 0, stream>>>(
                    (const char *) dw->data, dw->nb[2], dw->nb[1], s_gluq8, mt,
                    so, (int) N, n_used, (int) K, s_shd, s_shds,
                    lresid ? s_resid : (const float *) nullptr);
            }
        }
        cudaStreamSynchronize(stream);
        std::vector<float> h_ref(K), h0(K), h1(K);
        cudaMemcpy(h_ref.data(), s_out1,    K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h0.data(), s_out2,       K * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h1.data(), s_out2 + K,   K * 4, cudaMemcpyDeviceToHost);
        int bad0 = 0, bad1 = 0;
        for (int c = 0; c < (int) K; ++c) {
            if (h0[c] != h_ref[c]) ++bad0;
            if (h1[c] != h_ref[c]) ++bad1;
        }
        fprintf(stderr, "frt-moefuse-selftest (%s, sh=%d, lresid=%d): m2row0 vs m1 mismatch %d/%d, m2row1 vs m1 mismatch %d/%d %s\n",
                dw->name, (int) sh_ok, (int) (lresid != nullptr), bad0, (int) K, bad1, (int) K,
                (bad0 == 0 && bad1 == 0) ? "PASS" : "FAIL");
        cudaFree(s_act); cudaFree(s_ids); cudaFree(s_w); cudaFree(s_out1); cudaFree(s_out2); cudaFree(s_resid);
        cudaFree(s_actq8); cudaFree(s_gluq8); cudaFree(s_meta1); cudaFree(s_meta2);
    }
    {
        FRT_M_DISPATCH(M, ggml_cuda_kernel_launch(moeglue::frt_moe_quant_meta<MT>, ggml_cuda_kernel_launch_params(dim3(sh_ok ? 9 : 8), dim3(256), 0, stream),
            (const float *) act->data, (int) K, (int64_t) (act->nb[2] / sizeof(float)),
            (const int32_t *) ids->data, (int64_t) (ids->nb[1] / sizeof(int32_t)),
            (const float *) wnorm->data, (int64_t) (wnorm->nb[2] / sizeof(float)), n_used,
            sh_ok ? (const float *) ginp->src[0]->data : (const float *) nullptr,
            d_actq8, d_meta));
        dim3 g1((unsigned) (M * (sh_ok ? n_used + 1 : n_used)), (unsigned) (N / 32));
        if (gw->type == GGML_TYPE_Q4_K) {
            FRT_M_DISPATCH(M, ggml_cuda_kernel_launch((moeglue::frt_moe_k1<GGML_TYPE_Q4_K, MT>), ggml_cuda_kernel_launch_params(g1, dim3(512), 0, stream),
                (const char *) gw->data, (const char *) up_id->src[0]->data,
                gw->nb[2], gw->nb[1],
                d_actq8, (int) K, d_meta, d_gluq8, (int) N,
                n_used,
                sh_ok ? (const char *) shg->src[0]->data : (const char *) nullptr,
                sh_ok ? (const char *) shu->src[0]->data : (const char *) nullptr,
                sh_ok ? shg->src[0]->nb[1] : (size_t) 0));
        } else {   // Q8_0 experts (draft MTP layer)
            FRT_M_DISPATCH(M, ggml_cuda_kernel_launch((moeglue::frt_moe_k1<GGML_TYPE_Q8_0, MT>), ggml_cuda_kernel_launch_params(g1, dim3(512), 0, stream),
                (const char *) gw->data, (const char *) up_id->src[0]->data,
                gw->nb[2], gw->nb[1],
                d_actq8, (int) K, d_meta, d_gluq8, (int) N,
                n_used,
                sh_ok ? (const char *) shg->src[0]->data : (const char *) nullptr,
                sh_ok ? (const char *) shu->src[0]->data : (const char *) nullptr,
                sh_ok ? shg->src[0]->nb[1] : (size_t) 0));
        }
        dim3 g2((unsigned) ((K + 7) / 8), (unsigned) M);
#define FRT_K2_LAUNCH(TT) FRT_M_DISPATCH(M, ggml_cuda_kernel_launch((moeglue::frt_moe_k2<TT, MT>), ggml_cuda_kernel_launch_params(g2, dim3(256), 0, stream), \
                    (const char *) dw->data, dw->nb[2], dw->nb[1], d_gluq8, \
                    d_meta, (float *) out_add->data, (int) N, n_used, (int) K, \
                    sh_ok ? (const char *) shd->src[0]->data : (const char *) nullptr, \
                    sh_ok ? shd->src[0]->nb[1] : (size_t) 0, \
                    lresid ? (const float *) lresid->data : (const float *) nullptr))
        switch (dw->type) {
            case GGML_TYPE_Q6_K: FRT_K2_LAUNCH(GGML_TYPE_Q6_K); break;
            case GGML_TYPE_Q5_K: FRT_K2_LAUNCH(GGML_TYPE_Q5_K); break;
            case GGML_TYPE_Q4_K: FRT_K2_LAUNCH(GGML_TYPE_Q4_K); break;
            default:             FRT_K2_LAUNCH(GGML_TYPE_Q8_0); break;
        }
#undef FRT_K2_LAUNCH
    }

    ++fires_ok;
    dbg_guard.armed = false;
    *skip_count = end_idx - i + 1;
    return true;
}

// out-proj native span: [MUL_MAT ssm_out|attn_output] -> RESHAPE -> ADD residual
// (their path: quantize_q8_1 + mmvq + add = 3 launches) -> quant + gemv/add = 2.
static bool frt_outproj_native_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    static int mode = -1;
    if (mode < 0) mode = frt_window_on("FRT_OUTNATIVE_SWAP", "GGML_FLASHRT_NO_OUTNATIVE") ? 1 : 0;
    if (!mode) return false;
    ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_MUL_MAT || !n0->src[0]) return false;
    const char * wn = n0->src[0]->name;
    if (!strstr(wn, ".ssm_out.weight") && !strstr(wn, ".attn_output.weight")) return false;
    const ggml_tensor * w = n0->src[0];
    const ggml_tensor * act = n0->src[1];
    if (w->type != GGML_TYPE_Q8_0) return false;
    const int64_t K = w->ne[0], N = w->ne[1];
    if (K % 32 != 0 || K > 8192 || N % 8 != 0) return false;
    if (act->type != GGML_TYPE_F32 || !ggml_is_contiguous(act)) return false;
    const int M = (int) act->ne[1];   // token-batch width (spec verify runs M = 1 + n_draft)
    if (M < 1 || M > 4 || act->ne[2] != 1) return false;

    ggml_tensor * out_add = nullptr;
    const ggml_tensor * residual = nullptr;
    int end_idx = -1;
    const int LIM = i + 4 < cgraph->n_nodes ? i + 4 : cgraph->n_nodes;
    for (int j = i + 1; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) continue;
        if (n->op == GGML_OP_ADD) {
            const ggml_tensor * a = n->src[0], * b = n->src[1];
            auto is_mm = [&](const ggml_tensor * t) {
                return t == n0 || ((t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0] == n0);
            };
            if (is_mm(a) && !is_mm(b))      { residual = b; out_add = n; end_idx = j; }
            else if (is_mm(b) && !is_mm(a)) { residual = a; out_add = n; end_idx = j; }
        }
        break;
    }
    if (!out_add || end_idx < 0) return false;
    if (out_add->ne[0] != N || out_add->ne[1] != M || residual->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(out_add) || !ggml_is_contiguous(residual)) return false;

    static block_q8_1 * d_actq8b = nullptr;
    if (!d_actq8b) CUDA_CHECK(cudaMalloc(&d_actq8b, 4 * 256 * sizeof(block_q8_1)));
    static moeglue::frt_moe_meta * d_meta_dummy = nullptr;
    if (!d_meta_dummy) CUDA_CHECK(cudaMalloc(&d_meta_dummy, sizeof(moeglue::frt_moe_meta)));

    cudaStream_t stream = ctx.stream();
    const block_q8_1 * y_q8 = d_actq8b;
    if (frt::g_reg.ok && frt::g_reg.outq8_node == (const void *) act && K == frt_binding::out_proj_k) {
        y_q8 = frt::g_reg.d_outq8;         // GDN epilogue already staged the q8 act
        frt::g_reg.outq8_node = nullptr;
    } else {
        FRT_M_DISPATCH(M, ggml_cuda_kernel_launch(moeglue::frt_moe_quant_meta<MT>, ggml_cuda_kernel_launch_params(dim3(8), dim3(256), 0, stream),
            (const float *) act->data, (int) K, (int64_t) K,
            (const int32_t *) nullptr, (int64_t) 0,
            (const float *) nullptr, (int64_t) 0, 0,
            (const float *) nullptr,
            d_actq8b, d_meta_dummy));
    }
    ggml_cuda_kernel_launch(moeglue::frt_outproj_gemv<GGML_TYPE_Q8_0>, ggml_cuda_kernel_launch_params(dim3((unsigned) (N / 8), (unsigned) M), dim3(256), 0, stream),
        (const char *) w->data, w->nb[1], y_q8,
        (const float *) residual->data, (float *) out_add->data, (int) N, (int) K);
    *skip_count = end_idx - i + 1;
    return true;
}

// shared-expert span: anchor = MUL_MAT on ffn_gate_shexp.weight.
static bool frt_shexp_span_try(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    ggml_tensor * n0 = cgraph->nodes[i];
    if (n0->op != GGML_OP_MUL_MAT || !n0->src[0] ||
        !strstr(n0->src[0]->name, ".ffn_gate_shexp.weight")) return false;
    if (!frt::frt_regions_load() || !frt::g_reg.shexp_on) return false;
    int layer = -1;
    sscanf(n0->src[0]->name, "blk.%d.", &layer);
    if (layer < 0 || layer >= 64) return false;
    frt::frt_region & r2 = frt::g_reg.regions[2][layer];
    frt::frt_region & r3 = frt::g_reg.regions[3][layer];
    if (r2.N != 1024 || r2.K != 2048 || r3.N != 2048 || r3.K != 512) return false;
    if (n0->src[1]->ne[1] != 1 || !ggml_is_contiguous(n0->src[1])) return false;

    const ggml_tensor * up_mm = nullptr, * glu = nullptr, * down_mm = nullptr;
    const ggml_tensor * ginp_mm = nullptr, * sig = nullptr, * gmul = nullptr;
    ggml_tensor * out_add = nullptr;
    int end_idx = -1;
    const int LIM = i + 10 < cgraph->n_nodes ? i + 10 : cgraph->n_nodes;
    for (int j = i + 1; j < LIM; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_MUL_MAT:
                if (n->src[0] && strstr(n->src[0]->name, ".ffn_up_shexp.weight"))       up_mm = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_down_shexp.weight")) down_mm = n;
                else if (n->src[0] && strstr(n->src[0]->name, ".ffn_gate_inp_shexp.weight")) ginp_mm = n;
                else return false;
                break;
            case GGML_OP_GLU:
                if (ggml_get_glu_op(n) != GGML_GLU_OP_SWIGLU) return false;
                glu = n;
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(n) != GGML_UNARY_OP_SIGMOID) return false;
                sig = n;
                break;
            case GGML_OP_MUL:
                gmul = n;
                break;
            case GGML_OP_ADD:
                if (gmul && (n->src[1] == gmul || n->src[0] == gmul)) { out_add = n; end_idx = j; }
                else return false;
                break;
            case GGML_OP_RESHAPE: case GGML_OP_VIEW:
                break;
            default:
                return false;
        }
        if (end_idx >= 0) break;
    }
    if (!up_mm || !glu || !down_mm || !ginp_mm || !sig || !gmul || !out_add) return false;
    // wiring guards
    if (glu->src[0] != n0 || glu->src[1] != up_mm) return false;       // silu(gate)*up
    if (down_mm->src[1] != glu) return false;
    if (sig->src[0] != ginp_mm) return false;
    if (!(gmul->src[0] == down_mm && gmul->src[1] == sig) &&
        !(gmul->src[0] == sig && gmul->src[1] == down_mm)) return false;
    if (up_mm->src[1]->data != n0->src[1]->data ||
        ginp_mm->src[1]->data != n0->src[1]->data) return false;       // same activation
    if (ginp_mm->src[0]->type != GGML_TYPE_F32) return false;
    const ggml_tensor * moe_out = out_add->src[0] == gmul ? out_add->src[1] : out_add->src[0];

    cudaStream_t stream = ctx.stream();
    moeglue::frt_shexp_pre<<<2, 256, 0, stream>>>(
        (const float *) n0->src[1]->data,
        (uint2 *) frt::g_reg.d_apack, frt::g_reg.d_sfa,
        (const float *) ginp_mm->src[0]->data, frt::g_reg.d_scalar);
    frt::frt_ws_launch(frt::g_reg.d_apack, r2.d_packed, frt::g_reg.d_sfa, r2.d_sf,
        frt::g_reg.d_staging, r2.alpha, 1024, 2048, 1, stream);
    moeglue::frt_shexp_glu_quant<<<1, 256, 0, stream>>>(
        frt::g_reg.d_staging, (uint2 *) frt::g_reg.d_apack, frt::g_reg.d_sfa);
    frt::frt_ws_launch(frt::g_reg.d_apack, r3.d_packed, frt::g_reg.d_sfa, r3.d_sf,
        frt::g_reg.d_staging + 4096, r3.alpha, 2048, 512, 1, stream);
    moeglue::frt_shexp_finish<<<8, 256, 0, stream>>>(
        (const float *) moe_out->data, frt::g_reg.d_staging + 4096,
        frt::g_reg.d_scalar, (float *) out_add->data);
    *skip_count = end_idx - i + 1;
    return true;
}

// Returns true and sets *skip_count when either span matched at node i.
bool ggml_cuda_frt_moeglue_try_impl(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * skip_count) {
    if (frt_moefuse_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_router_span_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_attn_gate_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_outproj_native_try(ctx, cgraph, i, skip_count)) return true;
    if (frt_shexp_span_try(ctx, cgraph, i, skip_count)) return true;
    static int mode = -1;
    if (mode < 0) mode = frt_window_on("FRT_MOEGLUE_SWAP", "GGML_FLASHRT_NO_MOEGLUE") ? 1 : 0;
    if (!mode) return false;
    ggml_tensor * n0 = cgraph->nodes[i];

    // ---- combine span ----
    if (n0->op == GGML_OP_MUL && strncmp(n0->name, "ffn_moe_weighted", 16) == 0 &&
        n0->ne[1] > 1 && n0->ne[1] <= 32 && n0->ne[2] == 1 && n0->ne[3] == 1 &&
        n0->src[1]->ne[0] == 1 && ggml_is_contiguous(n0->src[0])) {
        const int hidden = (int) n0->ne[0];
        const int nexp   = (int) n0->ne[1];
        // expect nexp VIEWs of n0 then nexp-1 chained ADDs ending at ffn_moe_out
        ggml_tensor * out_add = nullptr;
        int end_idx = -1;
        uint32_t seen = 0;   // bitmask of expert slices consumed by the ADD chain
        const ggml_tensor * chain = nullptr;
        const int LIM = i + 2 * nexp + 2 < cgraph->n_nodes ? i + 2 * nexp + 2 : cgraph->n_nodes;
        for (int j = i + 1; j < LIM; ++j) {
            ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_VIEW) continue;
            if (n->op != GGML_OP_ADD) return false;
            auto slice_of = [&](const ggml_tensor * t) -> int {
                if (t->op != GGML_OP_VIEW || t->src[0] != n0 || t->ne[0] != hidden) return -1;
                const ptrdiff_t off = (const char *) t->data - (const char *) n0->data;
                if (off < 0 || off % ((ptrdiff_t) hidden * 4) != 0) return -1;
                const ptrdiff_t e = off / ((ptrdiff_t) hidden * 4);
                return e < nexp ? (int) e : -1;
            };
            int e0 = slice_of(n->src[0]);
            int e1 = slice_of(n->src[1]);
            if (chain == nullptr) {
                if (e0 < 0 || e1 < 0) return false;
                seen |= 1u << e0;
            } else {
                if (n->src[0] != chain || e1 < 0) return false;
            }
            if (seen & (1u << e1)) return false;
            seen |= 1u << e1;
            chain = n;
            if (strncmp(n->name, "ffn_moe_out", 11) == 0) { out_add = n; end_idx = j; break; }
        }
        if (!out_add || seen != (nexp >= 32 ? 0xffffffffu : ((1u << nexp) - 1))) return false;
        // write via private staging: the graph allocator may alias out_add onto
        // the (dead-after-us) weights/down buffers this kernel still reads.
        if (hidden > 4096) return false;
        static float * d_comb = nullptr;
        if (!d_comb) CUDA_CHECK(cudaMalloc(&d_comb, 4096 * sizeof(float)));
        ggml_cuda_kernel_launch(moeglue::frt_moe_combine, ggml_cuda_kernel_launch_params(dim3((hidden + 255) / 256), dim3(256), 0, ctx.stream()),
            (const float *) n0->src[0]->data, (const float *) n0->src[1]->data,
            d_comb, hidden, nexp);
        CUDA_CHECK(cudaMemcpyAsync(out_add->data, d_comb,
            (size_t) hidden * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
        *skip_count = end_idx - i + 1;
        return true;
    }

    return false;
}

// ---- MoE expert takeover (ggml-native NVFP4 blocks, FRT_MOE_SWAP) --------

static int frt_moe_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char * m = getenv("FRT_MOE_SWAP");
        mode = (m && m[0] == '1') ? 1 : 0;
    }
    return mode;
}

// blocks llama.cpp's own mmvq/mmf GLU fusion for expert tensors we take over
bool ggml_cuda_frt_moe_blocks_fusion(const ggml_tensor * mm) {
    if (!frt_moe_mode()) return false;
    if (!mm || mm->op != GGML_OP_MUL_MAT_ID) return false;
    const ggml_tensor * w = mm->src[0];
    return w && w->type == GGML_TYPE_NVFP4 && strstr(w->name, "_exps.weight") != nullptr;
}

bool ggml_cuda_frt_moe_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!frt_moe_mode()) return false;
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    if (!w || !x || !ids) return false;
    if (w->type != GGML_TYPE_NVFP4 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (strstr(w->name, "_exps.weight") == nullptr) return false;
    if (ids->type != GGML_TYPE_I32) return false;
    {   // FRT_MOE_KIND=gate|up|down|gu|all (isolation aid)
        static const char * kind = getenv("FRT_MOE_KIND");
        if (kind && strcmp(kind, "all") != 0) {
            const bool g = strstr(w->name, "ffn_gate_exps") != nullptr;
            const bool u = strstr(w->name, "ffn_up_exps") != nullptr;
            const bool d = strstr(w->name, "ffn_down_exps") != nullptr;
            if (strcmp(kind, "gate") == 0 && !g) return false;
            if (strcmp(kind, "up") == 0 && !u) return false;
            if (strcmp(kind, "down") == 0 && !d) return false;
            if (strcmp(kind, "gu") == 0 && d) return false;
        }
    }

    const int64_t K     = w->ne[0];
    const int64_t n_per = w->ne[1];
    if ((K % 64) != 0) return false;
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens != 1 || n_used <= 0 || n_used > 64) return false;
    if (!ggml_is_contiguous(dst) || !ggml_is_contiguous(ids)) return false;
    if (dst->ne[0] != n_per || dst->ne[1] * dst->ne[2] != n_used) return false;

    // activation layout: broadcast (one row for all experts) or per-expert-slot
    bool broadcast;
    int64_t x_stride_f = 0;
    const int64_t x_rows = x->ne[1] * x->ne[2];
    if (x->ne[0] == K && x_rows == 1) {
        broadcast = true;
    } else if (x->ne[0] == K && x_rows == n_used) {
        broadcast = false;
        x_stride_f = (x->ne[1] == n_used ? x->nb[1] : x->nb[2]) / sizeof(float);
    } else {
        return false;
    }

    // one-time UE4M3 LUT upload; never during graph capture
    static bool lut_done = false;
    if (!lut_done) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(ctx.stream(), &cap);
        if (cap != cudaStreamCaptureStatusNone) return false;
        frt::frt_init_ue4m3_lut();
        lut_done = true;
    }

    const int64_t expert_stride = w->nb[2];
    const int64_t rows_total = n_used * n_per;
    dim3 grid((unsigned)((rows_total + 7) / 8));
    cudaStream_t stream = ctx.stream();

    static int check = -1;
    if (check < 0) { const char * c = getenv("FRT_MOE_CHECK"); check = (c && c[0]=='1') ? 1 : 0; }
    if (check == 1) {
        check = 2;  // once
        cudaStreamSynchronize(stream);
        int32_t h_ids[64]; cudaMemcpy(h_ids, ids->data, n_used * 4, cudaMemcpyDeviceToHost);
        std::vector<float> h_x(K);
        const float * xsrc = (const float *) x->data;   // slot 0 row
        cudaMemcpy(h_x.data(), xsrc, K * 4, cudaMemcpyDeviceToHost);
        std::vector<uint8_t> h_row((K / 64) * 36);
        float lut[256];
        for (int i = 0; i < 256; ++i) {
            const int lo = i & 0x7F; const int e = (lo >> 3) & 0xF; const int m = lo & 7;
            float v = (lo == 0x7F) ? 0.f : (e == 0 ? (float) m / 8.f * ldexpf(1.f, -6)
                                                   : (1.f + (float) m / 8.f) * ldexpf(1.f, e - 7));
            lut[i] = (i & 0x80) ? -v : v;
        }
        const float e2m1v[16] = {0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
        fprintf(stderr, "frt-moe-check %s: n_used=%lld n_per=%lld K=%lld estride=%lld ids0=%d bcast=%d xne=[%lld,%lld,%lld]\n",
                w->name, (long long)n_used, (long long)n_per, (long long)K, (long long)expert_stride,
                h_ids[0], (int)broadcast, (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2]);
        for (int n = 0; n < 3; ++n) {
            cudaMemcpy(h_row.data(), (const uint8_t *) w->data + (size_t) h_ids[0] * expert_stride
                       + (size_t) n * ((K / 64) * 36), h_row.size(), cudaMemcpyDeviceToHost);
            double ref = 0;
            for (int kb = 0; kb < K / 64; ++kb) {
                const uint8_t * blk = h_row.data() + (size_t) kb * 36;
                for (int sub = 0; sub < 4; ++sub) {
                    const float d = lut[blk[sub]];
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t q = blk[4 + sub * 8 + j];
                        ref += (double) d * e2m1v[q & 0xF] * h_x[kb * 64 + sub * 16 + j];
                        ref += (double) d * e2m1v[q >> 4]  * h_x[kb * 64 + sub * 16 + j + 8];
                    }
                }
            }
            fprintf(stderr, "frt-moe-check ref out[0][%d] = %g\n", n, ref);
        }
    }
    if (broadcast) {
        frt::frt_moe_mmid_f32<true><<<grid, 256, 0, stream>>>(
            (const float *) x->data, (const uint8_t *) w->data, (const int32_t *) ids->data,
            (float *) dst->data, (int) K, (int) n_per, (int) n_used, expert_stride, 0);
    } else {
        frt::frt_moe_mmid_f32<false><<<grid, 256, 0, stream>>>(
            (const float *) x->data, (const uint8_t *) w->data, (const int32_t *) ids->data,
            (float *) dst->data, (int) K, (int) n_per, (int) n_used, expert_stride, x_stride_f);
    }
    if (check == 2) {
        check = 3;
        const cudaError_t le = cudaGetLastError();
        const cudaError_t se = cudaStreamSynchronize(stream);
        float h_out[3]; cudaMemcpy(h_out, dst->data, 3 * 4, cudaMemcpyDeviceToHost);
        fprintf(stderr, "frt-moe-check OUR out[0][0..2] = %g %g %g  (launch=%s sync=%s grid=%u)\n",
                h_out[0], h_out[1], h_out[2], cudaGetErrorString(le), cudaGetErrorString(se), grid.x);
    }
    return true;
}

// Returns true if it handled the mul_mat.
bool ggml_cuda_frt_head_mul_mat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!frt::g_reg.ok) {
        cudaStreamCaptureStatus rcap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(ctx.stream(), &rcap);
        if (rcap == cudaStreamCaptureStatusNone && frt::frt_regions_load()) { /* loaded */ }
    }
    if (frt::g_reg.ok && frt::frt_regions_mul_mat(ctx, src0, src1, dst)) {
        return true;
    }
    if (!frt::g_head.ok) {
        // never allocate while a CUDA graph capture is in flight
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(ctx.stream(), &cap);
        if (cap != cudaStreamCaptureStatusNone) return false;
    }
    if (!frt::frt_head_load()) return false;
    if (strcmp(src0->name, frt_binding::head_name) != 0) return false;
    // without the full-tier swap, serve only the spec draft's Q8_0 head copy
    // and an NVFP4-typed main head (FlashRT-edition GGUF: same values as
    // stock would dequantize, so the takeover is quality-neutral)
    if (frt::g_head.draft_only && src0->type != GGML_TYPE_Q8_0) {
        static int nat = -1;
        if (nat < 0) { const char * e = getenv("FRT_HEAD_NATIVE"); nat = (e && e[0] == '1') ? 1 : 0; }
        if (!(nat == 1 && src0->type == GGML_TYPE_NVFP4)) return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    const int M = (int) src1->ne[1];   // spec verify asks for logits at M = 1 + n_draft rows
    if (M < 1 || M > 4 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != frt::g_head.K || src0->ne[1] != frt::g_head.N) return false;
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;

    cudaStream_t stream = ctx.stream();
    const float * x = (const float *) src1->data;
    float * out = (float *) dst->data;
    const int N = (int) frt::g_head.N, K = (int) frt::g_head.K;

    static int head_mode = -1;   // 0 = w4a4 (default), 1 = w4a16
    if (head_mode < 0) {
        const char * m = getenv("FRT_HEAD_MODE");
        head_mode = (m && strcmp(m, "w4a16") == 0) ? 1 : 0;
        if (head_mode == 1) frt::frt_init_ue4m3_lut();
    }
    if (head_mode == 1) {
        if (M != 1) return false;   // w4a16 path stays M=1
        const int n_col_super = ((K >> 4) + 3) / 4;
        dim3 grid((N + 7) / 8);
        frt::w4a16_matvec_f32<<<grid, 256, (size_t) K * sizeof(float), stream>>>(
            x, frt::g_head.d_packed, frt::g_head.d_sf, out, frt::g_head.alpha, N, K, n_col_super);
        return true;
    }

    frt::frt_quant_act_launch(x, frt::g_head.d_apack, frt::g_head.d_sfa, (int) frt::g_head.K, M, (int64_t) K, stream);

    frt::frt_ws_launch(frt::g_head.d_apack, frt::g_head.d_packed, frt::g_head.d_sfa, frt::g_head.d_sf,
        out, frt::g_head.alpha, (int) N, (int) K, M, stream, 44);   // head: s4w4 wins (+6 t/s)
    return true;
}

// Pre-capture hook: called at the start of every backend graph evaluation,
// before any CUDA graph capture can begin. Builds online-repacked weight
// buffers (FRT_ONLINE_REPACK=1) so no allocation ever happens mid-capture.
void ggml_cuda_frt_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    frt::frt_online_prepare(ctx, cgraph);
}
