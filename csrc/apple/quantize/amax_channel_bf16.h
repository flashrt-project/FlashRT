#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
// Setup-only observer. One thread owns each channel; ordered dispatches merge
// maxima across calls without atomics. Initialize destination to zero.
inline std::string amax_channel_bf16_source() {
    return std::string(metal_preamble)+R"MSL(
struct ChannelShape { uint rows,columns; };
kernel void amax_channel_bf16(device const bfloat* x [[buffer(0)]],
                               device float* maxima [[buffer(1)]],
                               constant ChannelShape& p [[buffer(2)]],
                               uint k [[thread_position_in_grid]]) {
    if(k>=p.columns) return;
    float value=maxima[k];
    for(uint row=0;row<p.rows;++row) {
        float v=float(x[row*p.columns+k]);
        value=isfinite(v)?max(value,abs(v)):INFINITY;
    }
    maxima[k]=value;
}
)MSL";
}
} // namespace flashrt::apple::shaders
