#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
// Symmetric W8A8: one runtime scale per activation row. Finite inputs use
// nearest-even rounding to [-127,127]. Padded columns are always zero.
// Nonfinite input marks the row scale NaN so it cannot silently yield actions.
inline std::string quantize_bf16_int8_rowwise_source(bool small=false,bool smooth=false) {
    return std::string(metal_preamble)+
        "constant constexpr bool SMALL="+std::string(small?"true":"false")+";\n"+R"MSL(
struct Int8Quant { uint rows,inner,stride; };
kernel void quantize_bf16_int8_rowwise(device const bfloat* src [[buffer(0)]],
                                      device int8_t* dst [[buffer(1)]],
                                      device float* scales [[buffer(2)]],
)MSL"+(smooth?std::string("device const float* inverse_channels [[buffer(3)]],\n"):std::string())+
        "constant Int8Quant& p [[buffer("+std::to_string(smooth?4:3)+")]],\n"+R"MSL(
                                      uint group [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    uint row=SMALL?group:group*4+sg;
    if(row>=p.rows) return;
    uint start=SMALL?sg*32+lane:lane,step=SMALL?128:32;
    float maximum=0.0f;
    bool bad=false;
    for(uint k=start;k<p.inner;k+=step) {
        float v=float(src[row*p.inner+k]);
)MSL"+(smooth?std::string("v*=inverse_channels[k];\n"):std::string())+R"MSL(
        bad=bad || !isfinite(v);
        maximum=max(maximum,abs(v));
    }
    maximum=simd_max(bad?INFINITY:maximum);
    threadgroup float partials[4];
    if(SMALL) {
        if(lane==0) partials[sg]=maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum=simd_max(lane<4?partials[lane]:0.0f);
    }
    float scale=max(maximum/127.0f,1e-12f);
    bad=!isfinite(maximum);
    if(start==0) scales[row]=bad?NAN:scale;
    float inverse=1.0f/scale;
    for(uint k=start;k<p.stride;k+=step) {
        float v=0.0f;
        if(k<p.inner && !bad) {
            v=float(src[row*p.inner+k]);
)MSL"+(smooth?std::string("v*=inverse_channels[k];\n"):std::string())+R"MSL(
            v*=inverse;
        }
        dst[row*p.stride+k]=int8_t(rint(clamp(v,-127.0f,127.0f)));
    }
}
)MSL";
}
} // namespace flashrt::apple::shaders
