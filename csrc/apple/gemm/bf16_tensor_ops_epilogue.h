#pragma once
#include "common/source.h"
#include "kernels/gelu_mul_bf16.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// BF16 TensorOps GEMMs with fused epilogues. Each epilogue reproduces the rounding
// boundaries of the separate GEMM -> BF16 -> elementwise dispatches it replaces:
//   gelu_mul:      B packs gate/up column blocks of TILE_N/2 side by side; the tile rounds
//                  both products to BF16, then writes BF16(gelu_tanh(gate) * up).
//   residual:      C = BF16(C + BF16(A*B)), in place on the residual stream.
//   bias_gelu:     C = BF16(gelu_tanh(BF16(BF16(A*B) + bias))).
//   bias_residual: C = BF16(C + BF16(BF16(A*B) + bias)), in place.
//   bias:          C = BF16(BF16(A*B) + bias), the gemm_bias_bf16 rule with uniforms bound last.
//   f32:           C = A*B kept in FP32 (a partial product over a K range, rounded by its consumer).
// FP32 accumulation; tile geometry is a build-time specialization.
enum class GemmEpilogue { gelu_mul, residual, bias_gelu, bias_residual, bias, f32 };

inline std::string gemm_bf16_epilogue_source(GemmEpilogue epilogue, int tile_m = 64, int tile_n = 64,
                                             int simdgroups = 2) {
    if (tile_m <= 0 || tile_n <= 0 || simdgroups <= 0 || tile_n % 2)
        throw std::invalid_argument("GEMM epilogue tiles must be positive with an even column tile");
    const char* names[] = {"gemm_gelu_mul_bf16", "gemm_residual_bf16", "gemm_bias_gelu_bf16",
                           "gemm_bias_residual_bf16", "gemm_bias_bf16_last", "gemm_f32_bf16"};
    const int e = static_cast<int>(epilogue);
    return std::string(tensor_ops_preamble) + gelu_mul_bf16_value_source +
        "struct Gemm { int rows, columns, inner; };\n"
        "constant constexpr int GEMM_TILE_M=" + std::to_string(tile_m) + ";\n"
        "constant constexpr int GEMM_TILE_N=" + std::to_string(tile_n) + ";\n"
        "constant constexpr int GEMM_SIMDGROUPS=" + std::to_string(simdgroups) + ";\n"
        "#define GEMM_EPILOGUE " + std::to_string(e) + "\n"
        "#define GEMM_ENTRY " + names[e] + "\n" + R"MSL(
inline float gelu_tanh(float v) { return 0.5f*v*(1.0f+tanh(0.7978845608028654f*(v+0.044715f*v*v*v))); }
// columns = N of B. For gelu_mul the output has columns/2 entries per row.
// Arguments: A, B, C[, bias], uniforms last.
#if GEMM_EPILOGUE == 5
#define GEMM_OUT float
#else
#define GEMM_OUT bfloat
#endif
kernel void GEMM_ENTRY(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                       device GEMM_OUT* c [[buffer(2)]],
#if GEMM_EPILOGUE == 5
                       constant Gemm& p [[buffer(3)]],
#elif GEMM_EPILOGUE >= 2
                       device const bfloat* bias [[buffer(3)]], constant Gemm& p [[buffer(4)]],
#else
                       constant Gemm& p [[buffer(3)]],
#endif
                       uint2 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    if(group.y*GEMM_TILE_M>=uint(p.rows) || group.x*GEMM_TILE_N>=uint(p.columns))return;
    using E = dextents<int, 2>;
    tensor<device bfloat, E, tensor_inline> left(a, E(p.inner, p.rows), array<int,2>{1,p.inner});
    tensor<device bfloat, E, tensor_inline> right(b, E(p.columns,p.inner), array<int,2>{1,p.columns});
    auto l = left.slice(0, int(group.y)*GEMM_TILE_M);
    auto r = right.slice(int(group.x)*GEMM_TILE_N, 0);
    constexpr auto desc = matmul2d_descriptor(GEMM_TILE_M, GEMM_TILE_N, int(dynamic_extent), false, false, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<GEMM_SIMDGROUPS>> op;
    auto accum = op.get_destination_cooperative_tensor<decltype(l), decltype(r), float>();
    op.run(l, r, accum);
#if GEMM_EPILOGUE == 0
    {
        // Stage the BF16-rounded gate/up tile, then pair column j with j + TILE_N/2.
        threadgroup bfloat tile[GEMM_TILE_M*GEMM_TILE_N];
        #pragma clang loop unroll(full)
        for (uint i=0;i<accum.get_capacity();++i) {
            if (accum.is_valid_element(i)) {
                auto coord=accum.get_multidimensional_index(i);
                tile[coord[1]*GEMM_TILE_N+coord[0]]=bfloat(accum[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint HALF=GEMM_TILE_N/2;
        const uint hidden=uint(p.columns)/2;
        for (uint i=tid;i<GEMM_TILE_M*HALF;i+=GEMM_SIMDGROUPS*32) {
            uint rr=i/HALF, j=i%HALF;
            uint row=group.y*GEMM_TILE_M+rr, col=group.x*HALF+j;
            if (row<uint(p.rows) && col<hidden)
                c[row*hidden+col]=gelu_mul_bf16_value(float(tile[rr*GEMM_TILE_N+j]),
                                                      float(tile[rr*GEMM_TILE_N+HALF+j]));
        }
    }
#else
    {
        #pragma clang loop unroll(full)
        for (uint i=0;i<accum.get_capacity();++i) {
            if (accum.is_valid_element(i)) {
                auto coord=accum.get_multidimensional_index(i);
                uint col=group.x*GEMM_TILE_N+coord[0], row=group.y*GEMM_TILE_M+coord[1];
                if (row<uint(p.rows) && col<uint(p.columns)) {
#if GEMM_EPILOGUE == 5
                    c[row*uint(p.columns)+col]=accum[i];
#else
                    device bfloat* out=c+row*uint(p.columns)+col;
                    bfloat value=bfloat(accum[i]);
#if GEMM_EPILOGUE >= 2
                    value=bfloat(float(value)+float(bias[col]));
#endif
#if GEMM_EPILOGUE == 2
                    value=bfloat(gelu_tanh(float(value)));
#elif GEMM_EPILOGUE != 4
                    value=bfloat(float(*out)+float(value));
#endif
                    *out=value;
#endif
                }
            }
        }
    }
#endif
}
)MSL";
}
}  // namespace flashrt::apple::shaders
