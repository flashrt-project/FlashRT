#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string rms_norm_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Rows { uint rows,dim; };
kernel void tr_rms(device const bfloat* x [[buffer(0)]],device bfloat* y [[buffer(1)]],
    constant Rows& p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float sum=0;for(uint j=lane;j<p.dim;j+=32){float v=float(x[row*p.dim+j]);sum+=v*v;}
    float inv=rsqrt(simd_sum(sum)/float(p.dim)+1e-6f);
    for(uint j=lane;j<p.dim;j+=32)y[row*p.dim+j]=bfloat(float(x[row*p.dim+j])*inv);
}
)MSL";
}
}  // namespace flashrt::apple::shaders
