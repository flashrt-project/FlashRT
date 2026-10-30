#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string rope_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct Rope {uint rows,heads,dim,position;};
kernel void tr_rope(device const bfloat* x [[buffer(0)]],device bfloat* q [[buffer(1)]],
    device bfloat* k [[buffer(2)]],device bfloat* v [[buffer(3)]],
    device const bfloat* cs [[buffer(4)]],device const bfloat* sn [[buffer(5)]],
    constant Rope& p [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    uint pairs=p.dim/2;
    if(i>=p.rows*(p.heads+1)*pairs)return;
    uint pair=i%pairs,h=(i/pairs)%(p.heads+1),r=i/(pairs*(p.heads+1));
    uint base=r*(p.heads+2)*p.dim+h*p.dim+pair*2;
    float a=float(x[base]),b=float(x[base+1]);
    float c=float(cs[(p.position+r)*pairs+pair]),s=float(sn[(p.position+r)*pairs+pair]);
    // The reference evaluates separate BF16 multiplies and add/subtracts.
    bfloat real=bfloat(float(bfloat(a*c))-float(bfloat(b*s)));
    bfloat imag=bfloat(float(bfloat(b*c))+float(bfloat(a*s)));
    if(h<p.heads){uint out=(r*p.heads+h)*p.dim+pair*2;q[out]=real;q[out+1]=imag;}
    else {uint out=r*p.dim+pair*2;k[out]=real;k[out+1]=imag;
        v[out]=x[base+p.dim];v[out+1]=x[base+p.dim+1];}
}
)MSL";
}
}  // namespace flashrt::apple::shaders
