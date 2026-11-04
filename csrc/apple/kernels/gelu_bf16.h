#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string gelu_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Linear { uint count, channels; };
kernel void gelu_bf16(device bfloat* x [[buffer(0)]], constant Linear& p [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i<p.count) {
        float v=float(x[i]);
        x[i]=bfloat(0.5f*v*(1.0f+tanh(0.7978845608028654f*(v+0.044715f*v*v*v))));
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
