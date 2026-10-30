#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string gelu_mul_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Rows { uint rows,dim; };
kernel void tr_gelu_mul(device const bfloat* x [[buffer(0)]],device bfloat* y [[buffer(1)]],
    constant Rows& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.dim)return;
    uint r=i/p.dim,j=i%p.dim;float v=float(x[r*2*p.dim+j]);
    y[i]=bfloat(0.5f*v*(1.0f+tanh(0.7978845608028654f*(v+0.044715f*v*v*v)))*float(x[r*2*p.dim+p.dim+j]));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
