#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string bias_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Linear { uint count, channels; };
kernel void bias_bf16(device bfloat* x [[buffer(0)]], device const bfloat* b [[buffer(1)]],
                      constant Linear& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<p.count) x[i]=bfloat(float(x[i])+float(b[i%p.channels]));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
