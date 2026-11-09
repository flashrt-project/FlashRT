#pragma once
#include "common/source.h"
#include "kernels/gelu_mul_bf16.h"

namespace flashrt::apple::shaders {
// Preserve the materialized BF16 activation and its rounding boundary. Combine
// its producer with SmoothQuant amax, avoiding one full hidden-tensor read.
inline std::string gelu_mul_int8_rowwise_source(int simds=16) {
    return std::string(metal_preamble)+gelu_mul_bf16_value_source+
        "constant constexpr uint SIMDS="+std::to_string(simds)+";\n"+R"MSL(
struct Int8Quant { uint rows,inner,stride; };
kernel void gelu_mul_int8_rowwise(device const bfloat* src [[buffer(0)]],
                                device bfloat* hidden [[buffer(1)]],
                                device int8_t* dst [[buffer(2)]],
                                device float* scales [[buffer(3)]],
                                device const float* inverse_channels [[buffer(4)]],
                                constant Int8Quant& p [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    if(row>=p.rows)return;
    uint start=sg*32+lane;
    float maximum=0.0f;
    bool bad=false;
    for(uint k=start;k<p.inner;k+=SIMDS*32) {
        bfloat h=gelu_mul_bf16_value(float(src[row*2*p.inner+k]),
                                   float(src[row*2*p.inner+p.inner+k]));
        hidden[row*p.inner+k]=h;
        float v=float(h)*inverse_channels[k];
        bad=bad || !isfinite(v);
        maximum=max(maximum,abs(v));
    }
    maximum=simd_max(bad?INFINITY:maximum);
    threadgroup float partials[SIMDS];
    if(lane==0)partials[sg]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    maximum=simd_max(lane<SIMDS?partials[lane]:0.0f);
    float scale=max(maximum/127.0f,1e-12f);
    bad=!isfinite(maximum);
    if(start==0)scales[row]=bad?NAN:scale;
    float inverse=1.0f/scale;
    for(uint k=start;k<p.stride;k+=SIMDS*32) {
        float v=0.0f;
        if(k<p.inner && !bad)
            v=(float(hidden[row*p.inner+k])*inverse_channels[k])*inverse;
        dst[row*p.stride+k]=int8_t(rint(clamp(v,-127.0f,127.0f)));
    }
}
)MSL";
}
} // namespace flashrt::apple::shaders
