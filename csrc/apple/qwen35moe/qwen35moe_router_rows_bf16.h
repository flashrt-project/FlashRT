#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Router rows of the qwen3_5_moe chunked prefill: logits = x . R^T for 16 rows per threadgroup with TensorOps
// (FP32 accumulation over 64-input slices) from the router dequantized to BF16 (w = bf16(scale * q + bias), the
// tile MLX's quantized matmul forms), each logit rounded to BF16; then per row the FP32 softmax, the eight
// largest probabilities in ascending order (ties: ascending index; equal probabilities at the cut keep the larger
// index) with weights probability / sum, and sigmoid of the shared-expert gate logit (8-bit affine row product
// with the decode step's accumulation). ids, gates: [rows][8]; shared_gate: [rows] FP32.
inline std::string qwen35moe_router_rows_bf16_source() {
    return std::string(tensor_ops_preamble) + R"MSL(
inline float bits_f(uint b){return as_type<float>(b<<16);}
inline float bf_f(bfloat x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
constant constexpr uint EXPERTS=256u,TOPK=8u,HIDDEN=2048u;
struct RouterDense { ulong dense, gw, gs, gb; };   // BF16 router [256][2048]; shared-expert gate codes/scales/biases
kernel void qwen35moe_router_top8_dense_rows_bf16(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* gates [[buffer(3)]],device float* shared_gate [[buffer(4)]],
    constant RouterDense& p [[buffer(5)]],const device uint* state [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint rows=state[1],row0=group*16u;
    if(row0>=rows)return;
    const uint nr=min(16u,rows-row0);
    threadgroup float logits[16*EXPERTS];
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> xt((device bfloat*)x+size_t(row0)*HIDDEN,E(int(HIDDEN),int(nr)),array<int,2>{1,int(HIDDEN)});
    tensor<device bfloat,E,tensor_inline> rt((device bfloat*)(weights+p.dense),E(int(HIDDEN),int(EXPERTS)),array<int,2>{1,int(HIDDEN)});
    constexpr auto d=matmul2d_descriptor(16,EXPERTS,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(xt),decltype(rt),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
    for(uint k0=0;k0<HIDDEN;k0+=64) {
        auto xs=xt.slice(int(k0),0);
        auto ws=rt.slice(int(k0),0);
        op.run(xs,ws,acc);
    }
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]<int(nr))logits[c[1]*EXPERTS+c[0]]=rb(acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint r=sg;r<nr;r+=4) {
        const uint row=row0+r;
        const device bfloat* in=x+size_t(row)*HIDDEN;
        {   // shared-expert gate: 8-bit one-row product (4 inputs per lane and 128-input block)
            float a=0.0f;
            for(uint block=0;block<HIDDEN;block+=128u) {
                const uint scalar=block+lane*4u;
                float v[4];float sum=0.0f;
                for(uint i=0;i<4;++i){v[i]=bf_f(in[scalar+i]);sum+=v[i];}
                const uint gi=scalar/64u;
                const device uchar* w=weights+p.gw+scalar;
                float accum=0.0f;
                for(uint i=0;i<4;++i)accum+=v[i]*float(w[i]);
                a+=bits_f(((const device ushort*)(weights+p.gs))[gi])*accum+sum*bits_f(((const device ushort*)(weights+p.gb))[gi]);
            }
            a=rb(simd_sum(a));
            float e=rb(metal::precise::exp(metal::abs(a)));
            float tail=rb(1.0f/rb(1.0f+e));
            if(lane==0)shared_gate[row]=a<0?tail:rb(1.0f-tail);
        }
        float xv[8];
        for(uint j=0;j<8;++j)xv[j]=logits[r*EXPERTS+(j/4u)*128u+lane*4u+j%4u];
        float m0=-3.402823466e+38f,m1=-3.402823466e+38f;
        for(uint j=0;j<4;++j){m0=(m0<xv[j])?xv[j]:m0;m1=(m1<xv[4+j])?xv[4+j]:m1;}
        m0=simd_max(m0);m1=simd_max(m1);
        const float top=max(m0,m1);
        float n0=0.0f,n1=0.0f;
        for(uint j=0;j<4;++j){xv[j]=metal::fast::exp(xv[j]-top);n0+=xv[j];}
        for(uint j=0;j<4;++j){xv[4+j]=metal::fast::exp(xv[4+j]-top);n1+=xv[4+j];}
        n0=simd_sum(n0);n1=simd_sum(n1);
        const float normalizer=1.0f/simd_sum(lane==0?n0:(lane==1?n1:0.0f));
        float pr[8];
        for(uint j=0;j<8;++j)pr[j]=rb(xv[j]*normalizer);
        float chosen[8];uint picked[8];
        for(uint rank=0;rank<8;++rank) {
            float best=-1.0f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=(j/4u)*128u+lane*4u+j%4u;if(pr[j]>best||(pr[j]==best&&i>idx)){best=pr[j];idx=i;}}
            const float peak=simd_max(best);
            const uint selected=simd_max(best==peak?idx:0u);
            if(lane==0){picked[7u-rank]=selected;chosen[7u-rank]=peak;}
            for(uint j=0;j<8;++j)if((j/4u)*128u+lane*4u+j%4u==selected)pr[j]=-2.0f;
        }
        if(lane==0) {
            float total=0.0f;
            for(uint q=0;q<8;++q)total=rb(total+chosen[q]);
            for(uint q=0;q<8;++q){ids[row*TOPK+q]=picked[q];gates[row*TOPK+q]=rb(chosen[q]/total);}
        }
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
