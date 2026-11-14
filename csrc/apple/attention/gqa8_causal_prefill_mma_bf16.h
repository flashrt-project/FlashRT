#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Chunked-prefill attention for grouped queries (8 query heads per KV head, head dim 128) with causal or
// causal sliding-window masking and a learned per-head sink logit, on 8x8 SIMD-group MMAs. One threadgroup
// (256 threads) owns one KV head and 8 consecutive positions; SIMD group g holds position g's 8 heads as
// one 8-row fragment (Q stored KV-head-major [kv_head][row][8][128], kept in registers), so every staged
// 16-key K/V block serves 64 query rows.
//
// The per-row arithmetic is that of MLX's fused attention (steel attention, BK 16, head dim 128; MLX,
// Copyright (c) 2024 Apple Inc., MIT License): keys in blocks of 16 from the first key MLX's cache
// presents, S = Q.K^T by FP32 MMAs chained over the 16 head-dimension fragments, scaled by scale*log2(e),
// running maximum started at sink*log2(e) (row sum 1), masked scores at the lowest finite value,
// P = exp2(S - max), row maximum and sum reduced within each 8x8 fragment (xor 1, xor 8) and over the
// block's two fragments in order, O rescaled per block, O += P.V chained over the two 8-key fragments,
// O / sum at the end. Blocks a row cannot see are skipped (they change nothing in that arithmetic).
// Output is BF16 [row][q_head*128 + d].
//
// Keys: global layers read cache slots [0, pos0+rows) (slot == position); sliding layers read a linear
// scratch whose row s holds position pos0 - window + s.
inline std::string gqa8_causal_prefill_mma_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
struct PrefillAttention {
    uint kv_heads, cap, sliding, window;     // cap: rows per KV head in the key source
    ulong kv, sinks;                          // KV element offset of the source; sinks byte offset (BF16)
    float scale, pad;
};
#define FRT_UNROLL _Pragma("clang loop unroll(full)")
constant constexpr uint NP=8u;               // positions (= SIMD groups) per threadgroup
constant constexpr uint THREADS=NP*32u;
constant constexpr uint SPAN=16u;            // keys staged per barrier pair
constant constexpr uint LDK=SPAN+8u;         // K blocks, transposed: [128 dims][SPAN keys + pad]
constant constexpr uint LDV=128u+8u;         // V blocks: [SPAN keys][128 dims + pad]
inline float2 ld2(const threadgroup bfloat* p){return float2(*(const threadgroup bfloat2*)p);}
kernel void gqa8_causal_prefill_mma_bf16(const device bfloat* q [[buffer(0)]],const device bfloat* kv [[buffer(1)]],
    const device uchar* weights [[buffer(2)]],device bfloat* out [[buffer(3)]],
    constant PrefillAttention& p [[buffer(4)]],const device uint* state [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint pos0=state[0],rows=state[1];
    const uint row0=group.x*NP,kh=group.y;
    if(row0>=rows)return;
    const uint nr=min(NP,rows-row0);
    threadgroup bfloat Ks[128*LDK];
    threadgroup bfloat Vs[SPAN*LDV];
    // Fragment coordinates (MLX BaseMMAFrag): this lane holds row fy, columns fx and fx + 1 of each 8x8 tile.
    const ushort qid=lane/4u,fy=(qid&4u)+(lane/2u)%4u,fx=(qid&2u)*2u+(lane%2u)*2u;
    const float scale=p.scale*M_LOG2E_F;
    const float lowest=-3.402823466e+38f;
    const device bfloat* sink=(const device bfloat*)(p.sinks+weights);
    const uint head=kh*8u+fy;
    const bool live=sg<nr;
    const int qpos=int(pos0+row0+sg);
    simdgroup_float8x8 o[16];
    FRT_UNROLL for(uint dd=0;dd<16;++dd)o[dd]=simdgroup_float8x8(0.0f);
    simdgroup_float8x8 qf[16];
    {
        const device bfloat* qr=q+((size_t(kh)*rows+row0+(live?sg:0u))*8u+fy)*128u+fx;
        FRT_UNROLL for(uint dd=0;dd<16;++dd) {
            float2 a=float2(*(const device bfloat2*)(qr+8u*dd));
            qf[dd].thread_elements()[0]=a.x;qf[dd].thread_elements()[1]=a.y;
        }
    }
    float max_score=M_LOG2E_F*float(sink[head]),sum_score=1.0f;
    const device bfloat* kbase=kv+p.kv+size_t(kh)*p.cap*128u;
    const device bfloat* vbase=kv+p.kv+size_t(p.kv_heads)*p.cap*128u+size_t(kh)*p.cap*128u;
    // Key range in source rows: global [0, last position]; sliding [first visible, last].
    const uint pos_first=pos0+row0,pos_last=pos0+row0+nr-1;
    uint key_begin,key_end;
    const int key_shift=p.sliding?int(pos0)-int(p.window):0;   // position = source row + key_shift
    if(p.sliding) {
        int lo=max(0,int(pos_first)-int(p.window-1));
        uint origin=p.window-min(pos0,p.window);
        key_begin=uint(lo-key_shift);
        key_end=uint(int(pos_last)-key_shift+1);
        key_begin=origin+(key_begin-origin)/16u*16u;
    } else {
        key_begin=0;key_end=pos_last+1;
    }
    const int lo_pos=p.sliding?qpos-int(p.window-1):0;          // lowest visible position of this row
    // K/V blocks are software-pipelined: the next block's rows are read into registers before the current
    // block is computed and stored to threadgroup memory at the top of the next iteration.
    constexpr uint NL=SPAN*16u/THREADS;      // 16-byte K and V pieces per thread and stage
    uint4 kreg[NL],vreg[NL];
#define FRT_FETCH(from) \
    FRT_UNROLL for(uint l=0;l<NL;++l) { \
        uint i=t+l*THREADS,key=i/16u,c=(i%16u)*8u,src=(from)+key; \
        bool ok=src<key_end; \
        kreg[l]=ok?*(const device uint4*)(kbase+size_t(src)*128u+c):uint4(0u); \
        vreg[l]=ok?*(const device uint4*)(vbase+size_t(src)*128u+c):uint4(0u); \
    }
    if(key_begin<key_end){FRT_FETCH(key_begin)}
    for(uint begin=key_begin;begin<key_end;begin+=SPAN) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FRT_UNROLL for(uint l=0;l<NL;++l) {
            uint i=t+l*THREADS,key=i/16u,c=(i%16u)*8u;
            *(threadgroup uint4*)(Vs+key*LDV+c)=vreg[l];
            thread const bfloat* ke=(thread const bfloat*)&kreg[l];
            FRT_UNROLL for(uint e=0;e<8;++e)Ks[(c+e)*LDK+key]=ke[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(begin+SPAN<key_end){FRT_FETCH(begin+SPAN)}
        // rows of this SIMD group that see nothing of the block still run the (neutral) update
        simdgroup_float8x8 s0=simdgroup_float8x8(0.0f),s1=simdgroup_float8x8(0.0f);
        FRT_UNROLL for(uint dd=0;dd<16;++dd) {
            simdgroup_float8x8 k0,k1;
            const simdgroup_float8x8 qd=qf[dd];
            const threadgroup bfloat* kr=Ks+(8u*dd+fy)*LDK+fx;
            float2 b0=ld2(kr),b1=ld2(kr+8);
            k0.thread_elements()[0]=b0.x;k0.thread_elements()[1]=b0.y;
            k1.thread_elements()[0]=b1.x;k1.thread_elements()[1]=b1.y;
            simdgroup_multiply_accumulate(s0,qd,k0,s0);
            simdgroup_multiply_accumulate(s1,qd,k1,s1);
        }
        float v[2][2];
        // the whole block is visible to this SIMD group's row: no masking
        const bool open=live && begin+16u<=key_end && int(begin)+15+key_shift<=qpos && int(begin)+key_shift>=lo_pos;
        FRT_UNROLL for(uint j=0;j<2;++j)FRT_UNROLL for(uint e=0;e<2;++e) {
            float x=(j?s1:s0).thread_elements()[e]*scale;
            uint src=begin+8u*j+fx+e;
            int kpos=int(src)+key_shift;
            bool ok=open || (live && src<key_end && kpos<=qpos && kpos>=lo_pos);
            v[j][e]=ok?x:lowest;
        }
        float new_max=max_score;
        FRT_UNROLL for(uint j=0;j<2;++j) {
            float thr=max(v[j][0],v[j][1]);
            float qgr=max(thr,simd_shuffle_xor(thr,ushort(1)));
            float sgr=max(qgr,simd_shuffle_xor(qgr,ushort(8)));
            new_max=max(new_max,sgr);
        }
        float sum_tmp=0.0f;
        FRT_UNROLL for(uint j=0;j<2;++j) {
            float a=metal::fast::exp2(v[j][0]-new_max),b=metal::fast::exp2(v[j][1]-new_max);
            v[j][0]=a;v[j][1]=b;
            float thr=a+b;
            float qgr=thr+simd_shuffle_xor(thr,ushort(1));
            float sgr=qgr+simd_shuffle_xor(qgr,ushort(8));
            sum_tmp=sum_tmp+sgr;
        }
        s0.thread_elements()[0]=v[0][0];s0.thread_elements()[1]=v[0][1];
        s1.thread_elements()[0]=v[1][0];s1.thread_elements()[1]=v[1][1];
        const float factor=metal::fast::exp2(max_score-new_max);
        max_score=new_max;
        sum_score=sum_score*factor+sum_tmp;
        FRT_UNROLL for(uint dd=0;dd<16;++dd) {
            o[dd].thread_elements()[0]*=factor;
            o[dd].thread_elements()[1]*=factor;
        }
        const threadgroup bfloat* vr=Vs+fy*LDV+fx;
        FRT_UNROLL for(uint dd=0;dd<16;++dd) {
            simdgroup_float8x8 v0,v1;
            float2 c0=ld2(vr+8u*dd),c1=ld2(vr+8u*LDV+8u*dd);
            v0.thread_elements()[0]=c0.x;v0.thread_elements()[1]=c0.y;
            v1.thread_elements()[0]=c1.x;v1.thread_elements()[1]=c1.y;
            simdgroup_multiply_accumulate(o[dd],s0,v0,o[dd]);
            simdgroup_multiply_accumulate(o[dd],s1,v1,o[dd]);
        }
    }
    if(!live)return;
    device bfloat* orow=out+(size_t(row0+sg)*(p.kv_heads*8u)+head)*128u;
    FRT_UNROLL for(uint dd=0;dd<16;++dd) {
        orow[8u*dd+fx]=bfloat(o[dd].thread_elements()[0]/sum_score);
        orow[8u*dd+fx+1u]=bfloat(o[dd].thread_elements()[1]/sum_score);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
