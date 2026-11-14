#pragma once
#include "common/source.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// Few-row BF16 TensorOps GEMM partitioned along K. Threadgroup (x, y, z) computes the
// TILE_M x TILE_N block (y, x) over K range [z*chunk, (z+1)*chunk) and stores its FP32
// partial product to partial[z][row][col]. A consumer sums the partials in split order
// (deterministic) and applies the original BF16 rounding of the GEMM output. This keeps
// enough threadgroups in flight when N alone gives too few column tiles for bandwidth.
inline std::string gemm_splitk_bf16_source(int tile_m = 16, int tile_n = 64, int simdgroups = 2) {
    if (tile_m <= 0 || tile_n <= 0 || simdgroups <= 0)
        throw std::invalid_argument("Split-K GEMM tiles must be positive");
    return std::string(tensor_ops_preamble) +
        "constant constexpr int GEMM_TILE_M=" + std::to_string(tile_m) + ";\n"
        "constant constexpr int GEMM_TILE_N=" + std::to_string(tile_n) + ";\n"
        "constant constexpr int GEMM_SIMDGROUPS=" + std::to_string(simdgroups) + ";\n" + R"MSL(
struct SplitKGemm { int rows, columns, inner, chunk; };
kernel void gemm_splitk_f32_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]],
                                 device float* partial [[buffer(2)]], constant SplitKGemm& p [[buffer(3)]],
                                 uint3 group [[threadgroup_position_in_grid]]) {
    const int k0 = int(group.z) * p.chunk;
    if (group.y*GEMM_TILE_M>=uint(p.rows) || group.x*GEMM_TILE_N>=uint(p.columns) || k0>=p.inner) return;
    const int kn = min(p.chunk, p.inner - k0);
    using E = dextents<int, 2>;
    tensor<device bfloat, E, tensor_inline> left(a + k0, E(kn, p.rows), array<int,2>{1, p.inner});
    tensor<device bfloat, E, tensor_inline> right(b + k0 * p.columns, E(p.columns, kn), array<int,2>{1, p.columns});
    auto l = left.slice(0, int(group.y)*GEMM_TILE_M);
    auto r = right.slice(int(group.x)*GEMM_TILE_N, 0);
    constexpr auto desc = matmul2d_descriptor(GEMM_TILE_M, GEMM_TILE_N, int(dynamic_extent), false, false, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<GEMM_SIMDGROUPS>> op;
    auto accum = op.get_destination_cooperative_tensor<decltype(l), decltype(r), float>();
    op.run(l, r, accum);
    device float* out = partial + uint(group.z) * uint(p.rows) * uint(p.columns);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < accum.get_capacity(); ++i) {
        if (accum.is_valid_element(i)) {
            auto coord = accum.get_multidimensional_index(i);
            uint col = group.x*GEMM_TILE_N + coord[0], row = group.y*GEMM_TILE_M + coord[1];
            if (row < uint(p.rows) && col < uint(p.columns)) out[row*uint(p.columns) + col] = accum[i];
        }
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
