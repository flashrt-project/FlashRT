#pragma once
#include "common/source.h"
#include "kernels/gelu_mul_bf16.h"

namespace flashrt::apple::shaders {
// Consumers of gemm_splitk_f32_bf16 partials [splits][rows][columns]. Each sums the
// partials in split order, rounds the GEMM output to BF16 and then applies exactly the
// arithmetic of the dispatches it replaces:
//   splitk_rope_bf16:           tr_rope_offsets over the packed Q|K|V row (Q/K rotated, V copied).
//   splitk_residual_adarms_bf16: tr_residual (optionally gated) in place, then optionally tr_adarms
//                                of the updated row with a precomputed (scale, shift) modulation.
//   splitk_gelu_mul_bf16:        tr_gelu_mul of a merged [gate | up] product.
inline std::string splitk_reduce_bf16_source() {
    return std::string(metal_preamble) + gelu_mul_bf16_value_source + R"MSL(
inline float splitk_sum(device const float* partial, uint splits, uint stride, uint index) {
    float acc = 0.0f;
    for (uint s = 0; s < splits; ++s) acc += partial[s * stride + index];
    return acc;
}
struct SplitRope { uint rows, heads, dim, position, key_offset, value_offset, splits; };
kernel void splitk_rope_bf16(device const float* partial [[buffer(0)]], device bfloat* q [[buffer(1)]],
    device bfloat* k [[buffer(2)]], device bfloat* v [[buffer(3)]],
    device const bfloat* cs [[buffer(4)]], device const bfloat* sn [[buffer(5)]],
    constant SplitRope& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    k += p.key_offset; v += p.value_offset;
    uint pairs = p.dim / 2;
    if (i >= p.rows * (p.heads + 1) * pairs) return;
    uint pair = i % pairs, h = (i / pairs) % (p.heads + 1), r = i / (pairs * (p.heads + 1));
    uint stride = p.rows * (p.heads + 2) * p.dim;
    uint base = r * (p.heads + 2) * p.dim + h * p.dim + pair * 2;
    float a = float(bfloat(splitk_sum(partial, p.splits, stride, base)));
    float b = float(bfloat(splitk_sum(partial, p.splits, stride, base + 1)));
    float c = float(cs[(p.position + r) * pairs + pair]), s = float(sn[(p.position + r) * pairs + pair]);
    bfloat real = bfloat(float(bfloat(a * c)) - float(bfloat(b * s)));
    bfloat imag = bfloat(float(bfloat(b * c)) + float(bfloat(a * s)));
    if (h < p.heads) { uint out = (r * p.heads + h) * p.dim + pair * 2; q[out] = real; q[out + 1] = imag; }
    else {
        uint out = r * p.dim + pair * 2; k[out] = real; k[out + 1] = imag;
        v[out] = bfloat(splitk_sum(partial, p.splits, stride, base + p.dim));
        v[out + 1] = bfloat(splitk_sum(partial, p.splits, stride, base + p.dim + 1));
    }
}
struct SplitGelu { uint rows, hidden, splits; };
kernel void splitk_gelu_mul_bf16(device const float* partial [[buffer(0)]], device bfloat* y [[buffer(1)]],
    constant SplitGelu& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.hidden) return;
    uint r = i / p.hidden, j = i % p.hidden, stride = p.rows * 2 * p.hidden;
    float g = float(bfloat(splitk_sum(partial, p.splits, stride, r * 2 * p.hidden + j)));
    float u = float(bfloat(splitk_sum(partial, p.splits, stride, r * 2 * p.hidden + p.hidden + j)));
    y[i] = gelu_mul_bf16_value(g, u);
}
struct SplitResidual { uint rows, dim, splits, gated, norm; };
// One threadgroup of 256 threads per row; dim <= 2048.
kernel void splitk_residual_adarms_bf16(device const float* partial [[buffer(0)]], device bfloat* x [[buffer(1)]],
    device const bfloat* gate [[buffer(2)]], device const bfloat* mod [[buffer(3)]], device bfloat* y [[buffer(4)]],
    constant SplitResidual& p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float values[2048];
    threadgroup float sums[8];
    if (row >= p.rows) return;
    uint stride = p.rows * p.dim;
    float sq = 0.0f;
    for (uint j = tid; j < p.dim; j += 256) {
        float u = float(bfloat(splitk_sum(partial, p.splits, stride, row * p.dim + j)));
        float value = float(bfloat(float(x[row * p.dim + j]) + u * (p.gated ? float(gate[j]) : 1.0f)));
        x[row * p.dim + j] = bfloat(value);
        values[j] = value;
        sq += value * value;
    }
    if (!p.norm) return;
    sq = simd_sum(sq);
    if (lane == 0) sums[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < 8; ++s) total += sums[s];
    float inv = rsqrt(total / float(p.dim) + 1e-6f);
    for (uint j = tid; j < p.dim; j += 256)
        y[row * p.dim + j] = bfloat(values[j] * inv * (1.0f + float(mod[j])) + float(mod[p.dim + j]));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
