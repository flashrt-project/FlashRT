#pragma once
#include "common/source.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// Exactly-ROWS-row GEMM (no row padding) with weight-only quantized B, BF16 activations and scalar
// FP32 FMAs, partitioned along K for split-K consumers. Weight layout, interleaved for coalesced
// loads across output channels n:
//   bits 4: uint32 words [K/8][N], word (o, n) holds the 4-bit codes of inputs 8o..8o+7 of channel n
//           (input 8o+j in bits 4j..4j+3); BF16 scale and bias [K/64][N]; w = scale * q + bias.
//   bits 8: uint32 words [K/4][N] of four int8 codes (input 4o+j in byte j); FP32 scale [N]; w = scale * q.
// Threadgroup (x, z): channels [64x, 64x + 64) over inputs [512z, 512z + 512); four SIMD groups take
// 128 inputs each, every lane two adjacent channels; activations are staged once in threadgroup memory as
// FP32; partials are reduced in SIMD-group order and stored to partial[z][row][n] (FP32). Weights are
// dequantized in FP32 (float(uint) conversions); no BF16 rounding happens in this kernel.
inline std::string gemm_rows_fma_wq_source(int bits, int rows) {
    if ((bits != 4 && bits != 8) || rows <= 0 || rows > 16)
        throw std::invalid_argument("FMA row GEMM: bits 4/8, 1..16 rows");
    return std::string(metal_preamble) + "#define WQ_BITS " + std::to_string(bits) +
        "\n#define ROWS " + std::to_string(rows) + "\n" + R"MSL(
constant constexpr int CHUNK = 512, SG = 4, PER_SG = CHUNK / SG;
struct RowsFma { int rows, columns, inner, chunk; };
#if WQ_BITS == 8
#define WQ_SCALES float
#else
#define WQ_SCALES bfloat
#endif
kernel void gemm_rows_fma_wq(device const bfloat* a [[buffer(0)]], device const uint* codes [[buffer(1)]],
    device const WQ_SCALES* scales [[buffer(2)]], device const bfloat* biases [[buffer(3)]],
    device float* partial [[buffer(4)]], constant RowsFma& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[ROWS * CHUNK];
    const int k0 = int(group.z) * CHUNK;
    const uint n0 = group.x * 64, n = n0 + 2 * lane, N = uint(p.columns);
    if (k0 >= p.inner || n0 >= N || p.rows != ROWS) return;
    for (uint i = tid; i < ROWS * CHUNK; i += SG * 32) {
        const uint r = i / CHUNK, k = i % CHUNK;
        xs[i] = int(k0 + k) < p.inner ? float(a[r * uint(p.inner) + uint(k0) + k]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc0[ROWS], acc1[ROWS];
    for (int r = 0; r < ROWS; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
    const bool active = n + 1 < N;
    const int kb = k0 + int(sg) * PER_SG, ke = min(p.inner, kb + PER_SG);
#if WQ_BITS == 8
    const float s0 = active ? scales[n] : 0.0f, s1 = active ? scales[n + 1] : 0.0f;
#endif
    for (int k = kb; active && k < ke; k += 8) {
        float w0[8], w1[8];
#if WQ_BITS == 4
        const uint2 q = *(const device uint2*)(codes + size_t(k / 8) * N + n);
        const uint g = uint(k) / 64;
        const float s0 = float(scales[g * N + n]), s1 = float(scales[g * N + n + 1]);
        const float b0 = float(biases[g * N + n]), b1 = float(biases[g * N + n + 1]);
        for (int j = 0; j < 8; ++j) {
            w0[j] = fma(s0, float((q.x >> (4 * j)) & 15u), b0);
            w1[j] = fma(s1, float((q.y >> (4 * j)) & 15u), b1);
        }
#else
        const uint2 qa = *(const device uint2*)(codes + size_t(k / 4) * N + n);
        const uint2 qb = *(const device uint2*)(codes + size_t(k / 4 + 1) * N + n);
        const char4 a0 = as_type<char4>(qa.x), a1 = as_type<char4>(qb.x);
        const char4 c0 = as_type<char4>(qa.y), c1 = as_type<char4>(qb.y);
        w0[0] = s0 * float(a0.x); w0[1] = s0 * float(a0.y); w0[2] = s0 * float(a0.z); w0[3] = s0 * float(a0.w);
        w0[4] = s0 * float(a1.x); w0[5] = s0 * float(a1.y); w0[6] = s0 * float(a1.z); w0[7] = s0 * float(a1.w);
        w1[0] = s1 * float(c0.x); w1[1] = s1 * float(c0.y); w1[2] = s1 * float(c0.z); w1[3] = s1 * float(c0.w);
        w1[4] = s1 * float(c1.x); w1[5] = s1 * float(c1.y); w1[6] = s1 * float(c1.z); w1[7] = s1 * float(c1.w);
#endif
        const int kl = k - k0;
        for (int r = 0; r < ROWS; ++r) {
            const float4 xa = *(const threadgroup float4*)(xs + r * CHUNK + kl);
            const float4 xb = *(const threadgroup float4*)(xs + r * CHUNK + kl + 4);
            float t0 = acc0[r], t1 = acc1[r];
            t0 = fma(xa.x, w0[0], t0); t0 = fma(xa.y, w0[1], t0); t0 = fma(xa.z, w0[2], t0); t0 = fma(xa.w, w0[3], t0);
            t0 = fma(xb.x, w0[4], t0); t0 = fma(xb.y, w0[5], t0); t0 = fma(xb.z, w0[6], t0); t0 = fma(xb.w, w0[7], t0);
            t1 = fma(xa.x, w1[0], t1); t1 = fma(xa.y, w1[1], t1); t1 = fma(xa.z, w1[2], t1); t1 = fma(xa.w, w1[3], t1);
            t1 = fma(xb.x, w1[4], t1); t1 = fma(xb.y, w1[5], t1); t1 = fma(xb.z, w1[6], t1); t1 = fma(xb.w, w1[7], t1);
            acc0[r] = t0; acc1[r] = t1;
        }
    }
    // Reduce the four SIMD groups in index order through the (no longer needed) activation tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* red = xs;
    for (int r = 0; r < ROWS; ++r) {
        red[(sg * ROWS + r) * 64 + 2 * lane] = acc0[r];
        red[(sg * ROWS + r) * 64 + 2 * lane + 1] = acc1[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ROWS * 64; i += SG * 32) {
        const uint r = i / 64, c = i % 64;
        if (n0 + c >= N) continue;
        float sum = red[r * 64 + c];
        for (int s = 1; s < SG; ++s) sum += red[(s * ROWS + r) * 64 + c];
        partial[(group.z * ROWS + r) * N + n0 + c] = sum;
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
