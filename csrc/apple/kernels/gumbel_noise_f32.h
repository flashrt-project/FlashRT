#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Gumbel noise rows for token sampling, generated on the GPU from a counter-based hash: element i of row
// s is -log(-log(u)) with u in (0, 1) from 24 bits of hash(seed, position + s, i), so a (seed, position)
// pair always yields the same row. Rows [0, steps) of a [8][vocab] FP32 buffer. The parameter block is
// read in the layout of the Flint decode step's uint32[16] parameters (position = [1], steps = [2],
// seed = [12], [13], enable = [14]; enable 0 leaves host-written noise in place), so one buffer drives
// both; another caller fills those words. Grid: ceil(8 * vocab / 256) threadgroups of 256 threads.
inline std::string gumbel_noise_f32_source() {
    return std::string(metal_preamble) + R"MSL(
inline uint frt_mix(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
kernel void gumbel_noise_f32(device float* noise [[buffer(0)]],const device uint* p [[buffer(1)]],
    constant uint& vocab [[buffer(2)]],uint g [[thread_position_in_grid]]) {
    if(p[14]==0u)return;
    const uint step=g/vocab,i=g%vocab;
    if(step>=min(p[2],8u))return;
    const uint key=frt_mix(p[12]^frt_mix(p[13]^frt_mix(p[1]+step)));
    const uint h=frt_mix(key^frt_mix(i+0x9e3779b9u));
    const float u=(float(h>>8)+0.5f)*(1.0f/16777216.0f);
    noise[g]=-metal::precise::log(-metal::precise::log(u));
}
)MSL";
}
}  // namespace flashrt::apple::shaders
