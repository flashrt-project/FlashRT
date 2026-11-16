#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Chunked-prefill causal attention for grouped queries (8 query heads per KV head, head dim 256) with a
// per-element output gate, on 8x8 SIMD-group MMAs. One threadgroup (256 threads) owns one KV head and 8
// consecutive positions; SIMD group g holds position g's 8 heads as one 8-row fragment (Q stored KV-head-major
// [kv_head][row][8][256]), so every staged 16-key K/V block serves 64 query rows.
//
// Per row: keys in blocks of 16 from position 0, S = Q.K^T by FP32 MMAs chained over the 32 head-dimension
// fragments, scaled by scale*log2(e), masked scores at the lowest finite value, P = exp2(S - max) with a running
// maximum, O rescaled per block, O += P.V, O / sum at the end, rounded to BF16 and multiplied by
// sigmoid(gate) (BF16, each operation rounded). Blocks a row cannot see are skipped. Output BF16
// [row][q_head*256 + d]; the gates come from the projection output [row][q_head*512 + 256 + d].
// Keys and values: cache slots [0, pos0 + rows) (slot == position), K [kv_heads][cap][256] then V.
inline std::string gqa8_d256_causal_gated_prefill_mma_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct GatedPrefillAttention {
    uint kv_heads, cap, gate_stride, pad;    // cap: slots per KV head; gate_stride: elements per projection row
    ulong kv;                                // KV element offset of the layer
    float scale, pad2;
};
#define FRT_UNROLL _Pragma("clang loop unroll(full)")
constant constexpr uint NP=8u;               // positions (= SIMD groups) per threadgroup
constant constexpr uint THREADS=NP*32u;
constant constexpr uint SPAN=16u;            // keys staged per barrier pair
constant constexpr uint NF=SPAN/8u;          // 8-key fragments per stage
constant constexpr uint D=256u,FR=D/8u;      // head dimension, 8-wide fragments
constant constexpr uint LDK=SPAN+8u;         // K blocks, transposed: [D dims][SPAN keys + pad]
constant constexpr uint LDV=D+8u;            // V blocks: [SPAN keys][D dims + pad]
inline float2 ld2(const threadgroup bfloat* p){return float2(*(const threadgroup bfloat2*)p);}
inline float frt_rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
inline float frt_bf(bfloat x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float frt_sigmoid_bf16(float x) {
    float e=frt_rb(metal::precise::exp(metal::abs(x)));
    float tail=frt_rb(1.0f/frt_rb(1.0f+e));
    return x<0?tail:frt_rb(1.0f-tail);
}
kernel void gqa8_d256_causal_gated_prefill_mma_bf16(const device bfloat* q [[buffer(0)]],const device bfloat* kv [[buffer(1)]],
    const device bfloat* gates [[buffer(2)]],device bfloat* out [[buffer(3)]],
    constant GatedPrefillAttention& p [[buffer(4)]],const device uint* state [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint pos0=state[0],rows=state[1];
    const uint row0=group.x*NP,kh=group.y;
    if(row0>=rows)return;
    const uint nr=min(NP,rows-row0);
    threadgroup bfloat Ks[D*LDK];
    threadgroup bfloat Vs[SPAN*LDV];
    const ushort qid=lane/4u,fy=(qid&4u)+(lane/2u)%4u,fx=(qid&2u)*2u+(lane%2u)*2u;
    const float scale=p.scale*M_LOG2E_F;
    const float lowest=-3.402823466e+38f;
    const uint head=kh*8u+fy;
    const bool live=sg<nr;
    const int qpos=int(pos0+row0+sg);
    simdgroup_float8x8 o[FR];
    FRT_UNROLL for(uint dd=0;dd<FR;++dd)o[dd]=simdgroup_float8x8(0.0f);
    const device bfloat* qr=q+((size_t(kh)*rows+row0+(live?sg:0u))*8u+fy)*D+fx;
    float max_score=lowest,sum_score=0.0f;
    const device bfloat* kbase=kv+p.kv+size_t(kh)*p.cap*D;
    const device bfloat* vbase=kv+p.kv+size_t(p.kv_heads)*p.cap*D+size_t(kh)*p.cap*D;
    const uint key_end=pos0+row0+nr;
    // K/V blocks go straight into threadgroup memory: holding the next block in registers costs more than it saves
    // with 64 accumulator registers per lane.
    constexpr uint NL=SPAN*(D/8u)/THREADS;   // 16-byte K and V pieces per thread and stage
    for(uint begin=0;begin<key_end;begin+=SPAN) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FRT_UNROLL for(uint l=0;l<NL;++l) {
            uint i=t+l*THREADS,key=i/(D/8u),c=(i%(D/8u))*8u,src=begin+key;
            bool ok=src<key_end;
            const uint4 kreg=ok?*(const device uint4*)(kbase+size_t(src)*D+c):uint4(0u);
            *(threadgroup uint4*)(Vs+key*LDV+c)=ok?*(const device uint4*)(vbase+size_t(src)*D+c):uint4(0u);
            thread const bfloat* ke=(thread const bfloat*)&kreg;
            FRT_UNROLL for(uint e=0;e<8;++e)Ks[(c+e)*LDK+key]=ke[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 s[NF];
        FRT_UNROLL for(uint j=0;j<NF;++j)s[j]=simdgroup_float8x8(0.0f);
        for(uint dd=0;dd<FR;++dd) {
            simdgroup_float8x8 qd;
            const float2 a=float2(*(const device bfloat2*)(qr+8u*dd));
            qd.thread_elements()[0]=a.x;qd.thread_elements()[1]=a.y;
            const threadgroup bfloat* kr=Ks+(8u*dd+fy)*LDK+fx;
            FRT_UNROLL for(uint j=0;j<NF;++j) {
                simdgroup_float8x8 kf;
                float2 b0=ld2(kr+8u*j);
                kf.thread_elements()[0]=b0.x;kf.thread_elements()[1]=b0.y;
                simdgroup_multiply_accumulate(s[j],qd,kf,s[j]);
            }
        }
        float v[NF][2];
        const bool open=live && begin+SPAN<=key_end && int(begin)+int(SPAN)-1<=qpos;
        FRT_UNROLL for(uint j=0;j<NF;++j)FRT_UNROLL for(uint e=0;e<2;++e) {
            float x=s[j].thread_elements()[e]*scale;
            uint src=begin+8u*j+fx+e;
            bool ok=open || (live && src<key_end && int(src)<=qpos);
            v[j][e]=ok?x:lowest;
        }
        float new_max=max_score;
        FRT_UNROLL for(uint j=0;j<NF;++j) {
            float thr=max(v[j][0],v[j][1]);
            float qgr=max(thr,simd_shuffle_xor(thr,ushort(1)));
            float sgr=max(qgr,simd_shuffle_xor(qgr,ushort(8)));
            new_max=max(new_max,sgr);
        }
        float sum_tmp=0.0f;
        FRT_UNROLL for(uint j=0;j<NF;++j) {
            float a=v[j][0]==lowest?0.0f:metal::fast::exp2(v[j][0]-new_max);
            float b=v[j][1]==lowest?0.0f:metal::fast::exp2(v[j][1]-new_max);
            s[j].thread_elements()[0]=a;s[j].thread_elements()[1]=b;
            float thr=a+b;
            float qgr=thr+simd_shuffle_xor(thr,ushort(1));
            float sgr=qgr+simd_shuffle_xor(qgr,ushort(8));
            sum_tmp=sum_tmp+sgr;
        }
        const float factor=max_score==lowest?0.0f:metal::fast::exp2(max_score-new_max);
        max_score=new_max;
        sum_score=sum_score*factor+sum_tmp;
        const threadgroup bfloat* vr=Vs+fy*LDV+fx;
        FRT_UNROLL for(uint dd=0;dd<FR;++dd) {
            o[dd].thread_elements()[0]*=factor;
            o[dd].thread_elements()[1]*=factor;
            FRT_UNROLL for(uint j=0;j<NF;++j) {
                simdgroup_float8x8 vf;
                float2 c0=ld2(vr+8u*j*LDV+8u*dd);
                vf.thread_elements()[0]=c0.x;vf.thread_elements()[1]=c0.y;
                simdgroup_multiply_accumulate(o[dd],s[j],vf,o[dd]);
            }
        }
    }
    if(!live)return;
    device bfloat* orow=out+(size_t(row0+sg)*(p.kv_heads*8u)+head)*D;
    const device bfloat* grow=gates+size_t(row0+sg)*p.gate_stride+head*(2u*D)+D;
    FRT_UNROLL for(uint dd=0;dd<FR;++dd)for(uint e=0;e<2;++e) {
        const uint d=8u*dd+fx+e;
        const float value=frt_rb(o[dd].thread_elements()[e]/sum_score);
        *((device ushort*)(orow+d))=ushort(as_type<uint>(frt_rb(value*frt_sigmoid_bf16(frt_bf(grow[d]))))>>16);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
