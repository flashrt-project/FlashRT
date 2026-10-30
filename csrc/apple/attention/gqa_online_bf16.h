#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string gqa_online_bf16_source() {
    return std::string(tensor_ops_preamble) + R"MSL(
struct Attention {uint rows,keys,heads,dim;};
// Online softmax, 16 queries x 128 keys, shared KV head. Only tile-local
// scores are materialized. FP32 softmax and PV accumulation, BF16 output.
kernel void tr_gqa(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],device bfloat* y [[buffer(3)]],
    constant Attention& p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[16*128];
    threadgroup float output[16*256];
    threadgroup float maxima[16],sums[16],alpha[16];
    uint row=group.x*16,head=group.y,nr=min(16u,p.rows-row);
    for(uint i=sg*32+lane;i<16*256;i+=128)output[i]=0;
    for(uint r=sg*32+lane;r<16;r+=128){maxima[r]=-INFINITY;sums[r]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> qt(q+(row*p.heads+head)*256,E(256,int(nr)),array<int,2>{1,int(p.heads*256)});
    constexpr auto qkd=matmul2d_descriptor(16,128,int(dynamic_extent),false,true,false);
    constexpr auto pvd=matmul2d_descriptor(16,256,int(dynamic_extent),false,false,false);
    matmul2d<qkd,execution_simdgroups<4>> qkop;
    matmul2d<pvd,execution_simdgroups<4>> pvop;
    for(uint begin=0;begin<p.keys;begin+=128) {
        uint nk=min(128u,p.keys-begin);
        tensor<device bfloat,E,tensor_inline> kt(k+begin*256,E(256,int(nk)),array<int,2>{1,256});
        tensor<threadgroup float,E,tensor_inline> pt(scores,E(int(nk),int(nr)),array<int,2>{1,128});
        qkop.run(qt,kt,pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint r=sg;r<nr;r+=4) {
            float mx=-INFINITY;
            for(uint j=lane;j<nk;j+=32)mx=max(mx,scores[r*128+j]*(1.0f/16));
            mx=max(maxima[r],simd_max(mx));
            float a=exp(maxima[r]-mx),sum=0;
            for(uint j=lane;j<nk;j+=32){float w=exp(scores[r*128+j]*(1.0f/16)-mx);scores[r*128+j]=w;sum+=w;}
            sum=simd_sum(sum);
            if(lane==0){alpha[r]=a;sums[r]=sums[r]*a+sum;maxima[r]=mx;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device bfloat,E,tensor_inline> vt(v+begin*256,E(256,int(nk)),array<int,2>{1,256});
        auto acc=pvop.get_destination_cooperative_tensor<decltype(pt),decltype(vt),float>();
        pvop.run(pt,vt,acc);
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
            auto c=acc.get_multidimensional_index(i);
            if(c[0]<256 && c[1]<nr)output[c[1]*256+c[0]]=output[c[1]*256+c[0]]*alpha[c[1]]+acc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=sg*32+lane;i<nr*256;i+=128)
        y[((row+i/256)*p.heads+head)*256+i%256]=bfloat(output[i]/sums[i/256]);
}
)MSL";
}
}  // namespace flashrt::apple::shaders
