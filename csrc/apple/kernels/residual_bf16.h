#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string residual_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Shape { uint views, seq, dim, hidden, heads, layers; };
kernel void residual_bf16(device const bfloat* x [[buffer(0)]], device const bfloat* update [[buffer(1)]],
                          device bfloat* y [[buffer(2)]], constant Shape& s [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i<s.views*s.seq*s.dim) y[i]=bfloat(float(x[i])+float(update[i]));
}
kernel void tr_residual(device const bfloat* x [[buffer(0)]],device const bfloat* u [[buffer(1)]],
    device bfloat* y [[buffer(2)]],device const bfloat* gate [[buffer(3)]],
    constant uint3& p [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i<p.x) y[i]=bfloat(float(x[i])+float(u[i])*(p.z ? float(gate[i%p.y]):1.0f));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
