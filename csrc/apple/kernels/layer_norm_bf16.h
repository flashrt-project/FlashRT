#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string layer_norm_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Shape { uint views, seq, dim, hidden, heads, layers; };
kernel void norm_bf16(device const bfloat* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
                      device const bfloat* w [[buffer(2)]], device const bfloat* b [[buffer(3)]],
                      constant Shape& s [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row=group*4+sg;
    if (row>=s.views*s.seq) return;
    uint start=row*s.dim;
    float sum=0;
    for (uint j=lane;j<s.dim;j+=32) sum+=float(x[start+j]);
    float mean=simd_sum(sum)/float(s.dim);
    float var=0;
    for (uint j=lane;j<s.dim;j+=32) { float d=float(x[start+j])-mean; var+=d*d; }
    float inv=rsqrt(simd_sum(var)/float(s.dim)+1e-6f);
    for (uint j=lane;j<s.dim;j+=32)
        y[start+j]=bfloat((float(x[start+j])-mean)*inv*float(w[j])+float(b[j]));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
