#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string position_add_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Rows { uint rows,dim; };
kernel void tr_patch_position(device const bfloat* x [[buffer(0)]],device const bfloat* pos [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant Rows& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p.rows*p.dim)y[i]=bfloat(float(x[i])+float(pos[i%(256*p.dim)]));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
