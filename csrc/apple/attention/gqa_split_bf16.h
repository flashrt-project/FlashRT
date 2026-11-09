#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string gqa_split_bf16_source(bool masked = false) {
    return std::string(tensor_ops_preamble) +
        (masked ? "\n#define FRT_MASKED_GQA 1\n" : "\n#define FRT_MASKED_GQA 0\n") + R"MSL(
struct Attention {uint rows,keys,heads,dim;
#if FRT_MASKED_GQA
    uint valid_prefix,prefix_capacity;
#endif
};
// Small-M GQA: distribute KV tiles across the GPU, preserving FP32 partial
// numerators and (maximum, denominator) for a stable softmax merge.
#if FRT_MASKED_GQA
#define GQA_ENTRY tr_gqa_split_masked
#else
#define GQA_ENTRY tr_gqa_split
#endif
kernel void GQA_ENTRY(device bfloat* q [[buffer(0)]],device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],device float* partial [[buffer(3)]],
    device float* stats [[buffer(4)]],constant Attention& p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[16*128];
    uint row=group.x*16,head=group.y,split=group.z,begin=split*128;
    if(row>=p.rows || begin>=p.keys)return;
    uint nr=min(16u,p.rows-row),nk=min(128u,p.keys-begin);
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> qt(q+(row*p.heads+head)*256,
        E(256,int(nr)),array<int,2>{1,int(p.heads*256)});
    tensor<device bfloat,E,tensor_inline> kt(k+begin*256,E(256,int(nk)),array<int,2>{1,256});
    tensor<threadgroup float,E,tensor_inline> pt(scores,E(int(nk),int(nr)),array<int,2>{1,128});
    constexpr auto qkd=matmul2d_descriptor(16,128,int(dynamic_extent),false,true,false);
    constexpr auto pvd=matmul2d_descriptor(16,256,int(dynamic_extent),false,false,false);
    matmul2d<qkd,execution_simdgroups<4>> qkop;
    matmul2d<pvd,execution_simdgroups<4>> pvop;
    qkop.run(qt,kt,pt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if FRT_MASKED_GQA
    for(uint i=sg*32+lane;i<nr*128;i+=128) {
        uint j=i%128,key=begin+j;
        if(j<nk && key>=p.valid_prefix && key<p.prefix_capacity)scores[i]=-INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    for(uint r=sg;r<nr;r+=4) {
        float mx=-INFINITY;
        for(uint j=lane;j<nk;j+=32)mx=max(mx,scores[r*128+j]*(1.0f/16));
        mx=simd_max(mx);
        float sum=0;
        for(uint j=lane;j<nk;j+=32){float w=mx==-INFINITY ? 0.0f : exp(scores[r*128+j]*(1.0f/16)-mx);scores[r*128+j]=w;sum+=w;}
        sum=simd_sum(sum);
        if(lane==0){uint off=((split*p.rows+row+r)*p.heads+head)*2;stats[off]=mx;stats[off+1]=sum;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device bfloat,E,tensor_inline> vt(v+begin*256,E(256,int(nk)),array<int,2>{1,256});
    auto acc=pvop.get_destination_cooperative_tensor<decltype(pt),decltype(vt),float>();
    pvop.run(pt,vt,acc);
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)) {
        auto c=acc.get_multidimensional_index(i);
        if(c[0]<256 && c[1]<nr)partial[((split*p.rows+row+c[1])*p.heads+head)*256+c[0]]=acc[i];
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
