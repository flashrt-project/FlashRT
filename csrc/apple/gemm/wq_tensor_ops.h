#pragma once
#include "common/source.h"
#include "kernels/gelu_mul_bf16.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// Few-row (M <= 16) GEMM with weight-only quantized B and BF16 activations. Weights are stored
// column-major per output channel n:
//   bits 8: int8 codes [N][K], one FP32 scale per column, w = scale[n] * q.
//   bits 4: unsigned 4-bit codes [N][K/2] (element 2i in the low nibble), one BF16 scale and bias per
//           64 inputs of a column (affine), w = scale * q + bias.
// Each threadgroup (64 threads, two SIMD groups) dequantizes a 64-input x 64-column tile to BF16 in
// threadgroup memory per step and accumulates TILE_M x 64 FP32 products with TensorOps.
//   epilogue 0: FP32 partial of K range [z*chunk, (z+1)*chunk) to partial[z][row][col] (split-K; a
//               consumer sums partials in split order and applies the BF16 rounding).
//   epilogue 1: GELU multiply of a merged gate/up product whose columns are packed in blocks of 32
//               (gate j | up j inside each 64-column tile); writes BF16(gelu_tanh(BF16 g) * BF16 u)
//               to hidden [rows][N/2]. Uses the full K (no split).
inline std::string gemm_rows_wq_source(int bits, int epilogue) {
    if ((bits != 4 && bits != 8) || (epilogue != 0 && epilogue != 1))
        throw std::invalid_argument("quantized row GEMM: bits 4/8, epilogue 0/1");
    return std::string(tensor_ops_preamble) + gelu_mul_bf16_value_source +
        "#define WQ_BITS " + std::to_string(bits) + "\n#define WQ_EPILOGUE " + std::to_string(epilogue) + "\n" + R"MSL(
constant constexpr int TM = 16, TN = 64, KS = 64;
struct RowsWq { int rows, columns, inner, chunk; };
#if WQ_EPILOGUE == 0
#define WQ_OUT float
#else
#define WQ_OUT bfloat
#endif
#if WQ_BITS == 8
#define WQ_SCALES float
#else
#define WQ_SCALES bfloat
#endif
// Integer round-to-nearest-even: in these unrolled stores the hardware float->bfloat conversion's
// 16-bit temporaries can be packed across rows into one register by the compiler.
inline bfloat bf16_rne(float v) {
    uint bits = as_type<uint>(v);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return as_type<bfloat>(ushort(bits >> 16));
}
kernel void gemm_rows_wq_bf16(device bfloat* a [[buffer(0)]], device const uchar* codes [[buffer(1)]],
    device const WQ_SCALES* scales [[buffer(2)]], device const bfloat* biases [[buffer(3)]],
    device WQ_OUT* out [[buffer(4)]], constant RowsWq& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[KS * TN];
    const int k_begin = WQ_EPILOGUE == 0 ? int(group.z) * p.chunk : 0;
    const int k_end = WQ_EPILOGUE == 0 ? min(p.inner, k_begin + p.chunk) : p.inner;
    const uint n0 = group.x * TN;
    if (group.y * TM >= uint(p.rows) || n0 >= uint(p.columns) || k_begin >= p.inner) return;
    using E = dextents<int, 2>;
    tensor<threadgroup bfloat, E, tensor_inline> right(tile, E(TN, KS), array<int, 2>{1, TN});
    constexpr auto desc = matmul2d_descriptor(TM, TN, KS, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<2>> op;
    tensor<device bfloat, E, tensor_inline> left0(a, E(p.inner, p.rows), array<int, 2>{1, p.inner});
    auto l0 = left0.slice(0, int(group.y) * TM);
    auto accum = op.get_destination_cooperative_tensor<decltype(l0), decltype(right), float>();
    for (uint i = 0; i < accum.get_capacity(); ++i) accum[i] = 0.0f;
    const uint n = n0 + tid;                 // this thread dequantizes column n of every step
    const bool column = n < uint(p.columns);
    for (int k = k_begin; k < k_end; k += KS) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column) {
#if WQ_BITS == 8
            const float s = scales[n];
            const device char4* q = (const device char4*)(codes + size_t(n) * uint(p.inner) + uint(k));
            for (int i = 0; i < KS / 4; ++i) {
                char4 v = q[i];
                tile[(4 * i + 0) * TN + tid] = bf16_rne(s * float(v.x));
                tile[(4 * i + 1) * TN + tid] = bf16_rne(s * float(v.y));
                tile[(4 * i + 2) * TN + tid] = bf16_rne(s * float(v.z));
                tile[(4 * i + 3) * TN + tid] = bf16_rne(s * float(v.w));
            }
#else
            const uint g = uint(k) / 64;
            const float s = float(scales[size_t(n) * uint(p.inner / 64) + g]);
            const float b = float(biases[size_t(n) * uint(p.inner / 64) + g]);
            const device uchar4* q = (const device uchar4*)(codes + size_t(n) * uint(p.inner / 2) + uint(k / 2));
            for (int i = 0; i < KS / 8; ++i) {
                uchar4 v = q[i];
                // float(uint) conversions: no exponent tricks, so fast-math cannot refactor them.
                tile[(8 * i + 0) * TN + tid] = bf16_rne(fma(s, float(uint(v.x & 15)), b));
                tile[(8 * i + 1) * TN + tid] = bf16_rne(fma(s, float(uint(v.x >> 4)), b));
                tile[(8 * i + 2) * TN + tid] = bf16_rne(fma(s, float(uint(v.y & 15)), b));
                tile[(8 * i + 3) * TN + tid] = bf16_rne(fma(s, float(uint(v.y >> 4)), b));
                tile[(8 * i + 4) * TN + tid] = bf16_rne(fma(s, float(uint(v.z & 15)), b));
                tile[(8 * i + 5) * TN + tid] = bf16_rne(fma(s, float(uint(v.z >> 4)), b));
                tile[(8 * i + 6) * TN + tid] = bf16_rne(fma(s, float(uint(v.w & 15)), b));
                tile[(8 * i + 7) * TN + tid] = bf16_rne(fma(s, float(uint(v.w >> 4)), b));
            }
#endif
        } else {
            for (int i = 0; i < KS; ++i) tile[i * TN + tid] = bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device bfloat, E, tensor_inline> left(a + k, E(KS, p.rows), array<int, 2>{1, p.inner});
        auto l = left.slice(0, int(group.y) * TM);
        op.run(l, right, accum);
    }
#if WQ_EPILOGUE == 0
    device float* partial = out + uint(group.z) * uint(p.rows) * uint(p.columns);
    for (uint i = 0; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) {
            auto c = accum.get_multidimensional_index(i);
            uint col = n0 + c[0], row = group.y * TM + c[1];
            if (row < uint(p.rows) && col < uint(p.columns)) partial[row * uint(p.columns) + col] = accum[i];
        }
    }
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < accum.get_capacity(); ++i)
        if (accum.is_valid_element(i)) {
            auto c = accum.get_multidimensional_index(i);
            tile[c[1] * TN + c[0]] = bfloat(accum[i]);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hidden = uint(p.columns) / 2;
    for (uint i = tid; i < TM * 32; i += 64) {
        uint r = i / 32, j = i % 32, row = group.y * TM + r, col = group.x * 32 + j;
        if (row < uint(p.rows) && col < hidden)
            out[row * hidden + col] = gelu_mul_bf16_value(float(tile[r * TN + j]), float(tile[r * TN + 32 + j]));
    }
#endif
}
)MSL";
}
}  // namespace flashrt::apple::shaders
