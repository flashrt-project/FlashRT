#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string attention_simd_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Shape { uint views, seq, dim, hidden, heads, layers; };
// One SIMD group per query/head. Channels are distributed over lanes; scores
// live in threadgroup memory instead of a 256-float array in every thread.
// This is parallel exact attention, not a FlashAttention/tensor-core kernel.
kernel void attention_bf16(device const bfloat* qkv [[buffer(0)]], device bfloat* y [[buffer(1)]],
                           constant Shape& s [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[4][256];
    uint i=group*4+sg;
    if (i>=s.views*s.seq*s.heads) return;
    uint head=i%s.heads, row=(i/s.heads)%s.seq, view=i/(s.heads*s.seq);
    uint hd=s.dim/s.heads, base=view*s.seq*3*s.dim;
    uint qbase=base+row*3*s.dim+head*hd;
    for (uint k=0;k<s.seq;++k) {
        uint kbase=base+k*3*s.dim+s.dim+head*hd;
        float dot=0;
        for (uint c=lane;c<hd;c+=32) dot+=float(qkv[qbase+c])*float(qkv[kbase+c]);
        float score=simd_sum(dot)*rsqrt(float(hd));
        if (lane==0) scores[sg][k]=score;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;
    for (uint k=lane;k<s.seq;k+=32) maximum=max(maximum,scores[sg][k]);
    maximum=simd_max(maximum);
    float denom=0;
    for (uint k=lane;k<s.seq;k+=32) {
        float value=exp(scores[sg][k]-maximum);
        scores[sg][k]=value;
        denom+=value;
    }
    denom=simd_sum(denom);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c=lane;c<hd;c+=32) {
        float value=0;
        for (uint k=0;k<s.seq;++k)
            value+=(scores[sg][k]/denom)*float(qkv[base+k*3*s.dim+2*s.dim+head*hd+c]);
        y[(view*s.seq+row)*s.dim+head*hd+c]=bfloat(value);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
