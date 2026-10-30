#pragma once
#include "common/source.h"
#include <stdexcept>

namespace flashrt::apple::shaders {
// Pointer-backed tensors bind directly to ICB buffers. FP32 accumulation,
// BF16 output, with separate round-before-bias and FP32-bias epilogues.
// Tile selection is a build-time specialization, shared by runtime and tuning.
inline std::string gemm_bf16_source(int tile_m=64, int tile_n=64, int simdgroups=2) {
    if (tile_m<=0 || tile_n<=0 || simdgroups<=0)
        throw std::invalid_argument("GEMM tile dimensions and SIMD groups must be positive");
    return std::string(tensor_ops_preamble) +
        "struct Gemm { int rows, columns, inner; };\n" +
        "constant constexpr int GEMM_TILE_M=" + std::to_string(tile_m) + ";\n" +
        "constant constexpr int GEMM_TILE_N=" + std::to_string(tile_n) + ";\n" +
        "constant constexpr int GEMM_SIMDGROUPS=" + std::to_string(simdgroups) + ";\n" + R"MSL(
// Accumulation is FP32; the first BF16 rounding is before bias addition.
template<bool AddBias, bool RoundBeforeBias=true>
void gemm_impl(device bfloat* a, device bfloat* b, device bfloat* c,
               constant Gemm& p, uint2 group, device const bfloat* bias, uint output_stride=0) {
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
    // cooperative_tensor::store requires the destination's element type to
    // match its FP32 accumulator. Convert explicitly when writing BF16, using
    // the cooperative tensor's coordinates (never assume a lane layout).
    // Use is_valid_element; some MPP SDK examples incorrectly use get_mask.
    #pragma clang loop unroll(full)
    for (uint i=0;i<accum.get_capacity();++i) {
        if (accum.is_valid_element(i)) {
            auto coord=accum.get_multidimensional_index(i);
            uint col=group.x*GEMM_TILE_N+coord[0], row=group.y*GEMM_TILE_M+coord[1];
            if (row<uint(p.rows) && col<uint(p.columns)) {
                // Preserve GEMM -> BF16 -> bias -> BF16 rounding boundaries.
                bfloat value=bfloat(accum[i]);
                if constexpr (AddBias) {
                    if constexpr (RoundBeforeBias) value=bfloat(float(value)+float(bias[col]));
                    else value=bfloat(accum[i]+float(bias[col]));
                }
                c[row*(output_stride ? output_stride : uint(p.columns))+col]=value;
            }
        }
    }
}

kernel void gemm_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                      device bfloat* c [[buffer(2)]], constant Gemm& p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]]) {
    gemm_impl<false>(a,b,c,p,group,nullptr);
}
// Write a column slice without a temporary or a separate copy dispatch.
struct StridedGemm { Gemm gemm; uint output_stride; };
kernel void gemm_strided_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                      device bfloat* c [[buffer(2)]], constant StridedGemm& p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]]) {
    gemm_impl<false>(a,b,c,p.gemm,group,nullptr,p.output_stride);
}
kernel void gemm_bias_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                           device bfloat* c [[buffer(2)]], constant Gemm& p [[buffer(3)]],
                           device const bfloat* bias [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    gemm_impl<true>(a,b,c,p,group,bias);
}
// Patch convolution folds bias into FP32 accumulation before BF16 rounding.
kernel void gemm_patch_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                           device bfloat* c [[buffer(2)]], constant Gemm& p [[buffer(3)]],
                           device const bfloat* bias [[buffer(4)]],
                           uint2 group [[threadgroup_position_in_grid]]) {
    gemm_impl<true,false>(a,b,c,p,group,bias);
}
)MSL";
}
}  // namespace flashrt::apple::shaders
