#pragma once
#include "common/source.h"

namespace flashrt::apple::shaders {
// Chunked-prefill attention for grouped queries (8 query heads per KV head, head dim 128) with causal or
// causal sliding-window masking and a learned per-head sink logit. One threadgroup owns one KV head and
// 8 consecutive positions (64 query rows, Q stored KV-head-major [kv_head][row][8][128]); keys come in
// tiles of 64: S = Q.K^T (TensorOps), FP32 online softmax in log2 units in threadgroup memory (maximum
// advanced per 16 keys, as in MLX's fused attention), O += P.V into FP32 cooperative accumulators
// rescaled per row. Output is BF16 [row][q_head*128 + d].
// Keys: global layers read cache slots [0, pos0+rows) (slot == position); sliding layers read a linear
// scratch whose row s holds position pos0 - window + s. Key tiles start on 64-row boundaries counted from
// the first key MLX's cache would present (position 0, or pos0 - window for a continued prefill), so the
// 16-key maximum updates fall on the same keys.
inline std::string gqa8_causal_prefill_bf16_source() {
    return std::string(tensor_ops_preamble) + R"MSL(
constant constexpr uint SC=68u;            // score row stride (floats)
constant constexpr uint TK=64u;            // keys per tile (multiple of 16)
constant constexpr uint NB=TK/16u;         // 16-key blocks per tile
struct PrefillAttention {
    uint kv_heads, cap, sliding, window;     // cap: rows per KV head in the key source
    ulong kv, sinks;                          // KV element offset of the source; sinks byte offset (BF16)
    float scale, pad;
};
kernel void gqa8_causal_prefill_bf16(const device bfloat* q [[buffer(0)]],const device bfloat* kv [[buffer(1)]],
    const device uchar* weights [[buffer(2)]],device bfloat* out [[buffer(3)]],
    constant PrefillAttention& p [[buffer(4)]],const device uint* state [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint pos0=state[0],rows=state[1];
    const uint row0=group.x*8u,kh=group.y;
    if(row0>=rows)return;
    const uint nr=min(8u,rows-row0);
    const float scale=p.scale*M_LOG2E_F;   // scores in log2 units, exp2 softmax (sink logit scaled alike)
    threadgroup float scores[64*SC];
    threadgroup float maxima[64],sums[64],alpha[64*NB];
    threadgroup uint rescale[1];
    const device bfloat* sink=(const device bfloat*)(p.sinks+weights);
    if(t<64) {
        uint head=kh*8u+t%8u;
        maxima[t]=M_LOG2E_F*float(sink[head]);sums[t]=1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> qt((device bfloat*)q+(size_t(kh)*rows+row0)*8u*128u,E(128,int(nr*8u)),array<int,2>{1,128});
    const device bfloat* kbase=kv+p.kv+size_t(kh)*p.cap*128u;
    const device bfloat* vbase=kv+p.kv+size_t(p.kv_heads)*p.cap*128u+size_t(kh)*p.cap*128u;
    constexpr auto qkd=matmul2d_descriptor(64,int(TK),128,false,true,false);
    constexpr auto pvd=matmul2d_descriptor(64,128,int(dynamic_extent),false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qkd,execution_simdgroups<4>> qkop;
    matmul2d<pvd,execution_simdgroups<4>> pvop;
    tensor<threadgroup float,E,tensor_inline> pt(scores,E(int(TK),64),array<int,2>{1,int(SC)});
    // Key range: global [0, pos_last]; sliding scratch rows [first visible, last]
    const uint pos_first=pos0+row0,pos_last=pos0+row0+nr-1;
    uint key_begin,key_end;   // in source rows
    if(p.sliding) {
        long lo=max(0L,long(pos_first)-long(p.window-1));
        uint origin=p.window-min(pos0,p.window);
        key_begin=uint(lo-long(pos0)+long(p.window));
        key_end=uint(long(pos_last)-long(pos0)+long(p.window)+1);
        key_begin=origin+(key_begin-origin)/TK*TK;
    } else {
        key_begin=0;key_end=pos_last+1;
    }
    tensor<device bfloat,E,tensor_inline> vfull((device bfloat*)vbase,E(128,int(key_end)),array<int,2>{1,128});
    auto acc=pvop.get_destination_cooperative_tensor<decltype(pt),decltype(vfull),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
    for(uint begin=key_begin;begin<key_end;begin+=TK) {
        uint nk=min(TK,key_end-begin);
        tensor<device bfloat,E,tensor_inline> kt((device bfloat*)kbase+size_t(begin)*128u,E(128,int(nk)),array<int,2>{1,128});
        qkop.run(qt,kt,pt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Mask + online softmax in log2 units with the running maximum advanced every 16 keys (blocks
        // b = 0..3 of the tile): P_b = exp2(S - m_b), O and the row sum rescaled by exp2(m_{b-1} - m_b) before
        // each block, block sums reduced as two pairwise 8-key trees. Lane l of SIMD group s handles row
        // 32*pass + 8*s + l/4, block l%4; the four lanes of a row exchange block maxima and sums.
        if(t==0)rescale[0]=0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool split=false;
        // Lane l of SIMD group s handles block l % NB of row (pass * 128 + s * 32 + l) / NB.
        for(uint pass=0;pass<NB/2u;++pass) {
            const uint item=pass*128u+sg*32u+lane,r=item/NB,b=item%NB,base=lane-b;
            const uint pos=pos0+row0+r/8u;
            const bool live=r<nr*8u;
            threadgroup float* srow=scores+r*SC+16u*b;
            float v[16];float mb=-INFINITY;
            for(uint j=0;j<16;++j) {
                uint key=16u*b+j;
                long kpos=p.sliding?long(pos0)-long(p.window)+long(begin+key):long(begin+key);
                bool ok=live && key<nk && kpos>=0 && kpos<=long(pos) && (!p.sliding || kpos>=long(pos)-long(p.window-1));
                v[j]=ok?srow[j]*scale:-INFINITY;
                mb=max(mb,v[j]);
            }
            const float m_prev=maxima[r];
            float m[NB],f[NB],rsum[NB];
            for(uint k=0;k<NB;++k){m[k]=max(k?m[k-1]:m_prev,simd_shuffle(mb,ushort(base+k)));f[k]=metal::fast::exp2((k?m[k-1]:m_prev)-m[k]);}
            float mine=m[0];for(uint k=1;k<NB;++k)if(b==k)mine=m[k];
            float w[16];
            for(uint j=0;j<16;++j){w[j]=v[j]==-INFINITY?0.0f:metal::fast::exp2(v[j]-mine);srow[j]=w[j];}
            float s8a=((w[0]+w[1])+(w[2]+w[3]))+((w[4]+w[5])+(w[6]+w[7]));
            float s8b=((w[8]+w[9])+(w[10]+w[11]))+((w[12]+w[13])+(w[14]+w[15]));
            float rs=(0.0f+s8a)+s8b;
            for(uint k=0;k<NB;++k)rsum[k]=simd_shuffle(rs,ushort(base+k));
            if(b==0u) {
                float total=sums[r];
                for(uint k=0;k<NB;++k)total=total*f[k]+rsum[k];
                sums[r]=total;maxima[r]=m[NB-1];
                for(uint k=0;k<NB;++k)alpha[r*NB+k]=f[k];
            }
            for(uint k=1;k<NB;++k)split=split||f[k]!=1.0f;
        }
        if(split)rescale[0]=1u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(rescale[0]==0u) {
            for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)){auto c=acc.get_multidimensional_index(i);acc[i]*=alpha[c[1]*NB];}
            tensor<device bfloat,E,tensor_inline> vt((device bfloat*)vbase+size_t(begin)*128u,E(128,int(nk)),array<int,2>{1,128});
            tensor<threadgroup float,E,tensor_inline> pk(scores,E(int(nk),64),array<int,2>{1,int(SC)});
            pvop.run(pk,vt,acc);
        } else {
            for(uint b=0;b<NB && 16u*b<nk;++b) {
                uint n16=min(16u,nk-16u*b);
                for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i)){auto c=acc.get_multidimensional_index(i);acc[i]*=alpha[c[1]*NB+b];}
                tensor<device bfloat,E,tensor_inline> vt((device bfloat*)vbase+size_t(begin+16u*b)*128u,E(128,int(n16)),array<int,2>{1,128});
                tensor<threadgroup float,E,tensor_inline> pk(scores+16u*b,E(int(n16),64),array<int,2>{1,int(SC)});
                pvop.run(pk,vt,acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        uint r=c[1],d=c[0];
        if(r>=nr*8u || d>=128)continue;
        uint row=row0+r/8u,head=kh*8u+r%8u;
        out[(size_t(row)*(p.kv_heads*8u)+head)*128u+d]=bfloat(acc[i]/sums[r]);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
