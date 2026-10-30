#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
inline std::string attention_d72_bf16_source() {
    return std::string(tensor_ops_preamble) + R"MSL(
struct Shape { uint views, seq, dim, hidden, heads, layers; };
// Fused short-sequence D72 attention. Sixteen queries share a QK tensor
// operation and a PV tensor operation. Only this query tile's scores live in
// threadgroup memory (16 KiB); no global score matrix or padded Q/K/V copy.
// Softmax and PV inputs remain FP32, including the mixed float/BF16 matmul.
// This full-key tile is limited to the API's sequence <= 256 contract.
kernel void attention_d72_bf16(device bfloat* qkv [[buffer(0)]],
                               device bfloat* y [[buffer(1)]],
                               constant Shape& s [[buffer(2)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[16*256];
    using E = dextents<int, 2>;
    uint row=group.x*16, head=group.y, view=group.z;
    uint base=view*s.seq*3*s.dim+head*72;
    int valid_rows=int(min(16u,s.seq-row));
    tensor<device bfloat,E,tensor_inline> q(qkv+base+row*3*s.dim,
        E(72,valid_rows),array<int,2>{1,int(3*s.dim)});
    tensor<device bfloat,E,tensor_inline> k(qkv+base+s.dim,
        E(72,int(s.seq)),array<int,2>{1,int(3*s.dim)});
    tensor<threadgroup float,E,tensor_inline> p(scores,
        E(int(s.seq),valid_rows),array<int,2>{1,256});
    constexpr auto qk_desc=matmul2d_descriptor(16,256,int(dynamic_extent),false,true,false);
    matmul2d<qk_desc,execution_simdgroups<4>> qk;
    qk.run(q,k,p);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r=sg;r<uint(valid_rows);r+=4) {
        float maximum=-INFINITY;
        for (uint j=lane;j<s.seq;j+=32)
            maximum=max(maximum,scores[r*256+j]);
        maximum=simd_max(maximum);
        float denom=0;
        for (uint j=lane;j<s.seq;j+=32) {
            float value=exp((scores[r*256+j]-maximum)*rsqrt(72.0f));
            scores[r*256+j]=value;
            denom+=value;
        }
        denom=simd_sum(denom);
        for (uint j=lane;j<s.seq;j+=32) scores[r*256+j]/=denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device bfloat,E,tensor_inline> v(qkv+base+2*s.dim,
        E(72,int(s.seq)),array<int,2>{1,int(3*s.dim)});
    constexpr auto pv_desc=matmul2d_descriptor(16,80,int(dynamic_extent),false,false,false);
    matmul2d<pv_desc,execution_simdgroups<4>> pv;
    auto out=pv.get_destination_cooperative_tensor<decltype(p),decltype(v),float>();
    pv.run(p,v,out);
    #pragma clang loop unroll(full)
    for (uint i=0;i<out.get_capacity();++i) {
        if (out.is_valid_element(i)) {
            auto coord=out.get_multidimensional_index(i);
            if (coord[0]<72 && coord[1]<valid_rows)
                y[((view*s.seq+row+coord[1])*s.dim)+head*72+coord[0]]=bfloat(out[i]);
        }
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
