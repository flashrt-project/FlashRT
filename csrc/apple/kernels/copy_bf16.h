#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string copy_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
kernel void tr_copy(device const bfloat* x [[buffer(0)]],device bfloat* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)y[i]=x[i];}
)MSL";
}
}  // namespace flashrt::apple::shaders
