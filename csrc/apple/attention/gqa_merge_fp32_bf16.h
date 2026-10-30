#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string gqa_merge_fp32_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Attention {uint rows,keys,heads,dim;};
kernel void tr_gqa_merge(device const float* partial [[buffer(0)]],device const float* stats [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant Attention& p [[buffer(3)]],
    uint row_head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint splits=(p.keys+127)/128,stride=p.rows*p.heads;
    float mx=-INFINITY;
    for(uint s=0;s<splits;++s)mx=max(mx,stats[(s*stride+row_head)*2]);
    float denom=0;
    for(uint s=0;s<splits;++s)denom+=stats[(s*stride+row_head)*2+1]*exp(stats[(s*stride+row_head)*2]-mx);
    for(uint d=lane;d<256;d+=32) {
        float value=0;
        for(uint s=0;s<splits;++s)value+=partial[(s*stride+row_head)*256+d]*exp(stats[(s*stride+row_head)*2]-mx);
        y[row_head*256+d]=bfloat(value/denom);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
