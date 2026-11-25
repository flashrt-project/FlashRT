#pragma once
#include "common/source.h"
#include "inkling2/inkling2_common.h"
#include <string>
// DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2) for Inkling 2 Mini, following SGLang's DFlash2DraftModel and
// its greedy draft step with the Inkling integration (features: the hidden state entering visits 5, 23, 41 and the
// final normalized state; candidates over the 200058 real vocabulary rows).
//
// Context of a committed position: fc(concat of the 4 features) -> hidden_norm -> per layer K / V projections,
// k_norm, NeoX RoPE (theta 1e4) at the position, into the drafter cache [layer][k|v][8 heads][cap][128] (BF16).
// Block of 8 rows (the anchor and 7 mask tokens: the target's embedding and embed_norm) at positions a .. a + 7,
// 5 layers: residual RMSNorm, two-tap dynamic grouped convolutions around attention and MLP (per channel c of
// group c / 16: (base[tap][c] + delta[row][tap][group]) times the input of the row and of the previous block row,
// deltas a projection of the sublayer's normalized input), 32 query / 8 KV heads with q / k norms and RoPE,
// attention without causal mask over the context window and the block (the block's own K / V written at its
// positions in the same cache), SwiGLU MLP; final RMSNorm. Drafts only propose (the target verifies), so the
// drafter computes in FP32 with BF16 activations, without matching any reference rounding.
//
// Row metadata as the DSpark kernels: dinfo[r] = (sequence, position, block start, index in block); cinfo[r] =
// (sequence, position, keep, last committed).
namespace flashrt::apple::shaders {
inline std::string inkling2_dflash2_bf16_source() {
    return std::string(metal_preamble) + inkling2_common_msl() + R"MSL(
constant constexpr uint DH=2304u;           // hidden
constant constexpr uint DQ=4096u;           // 32 query heads x 128
constant constexpr uint DKV=1024u;          // 8 KV heads x 128
constant constexpr uint DQKV=DQ+2u*DKV;     // stacked q | k | v rows
constant constexpr uint DI=6912u;           // MLP intermediate
constant constexpr uint DG=144u;            // conv groups of 16 channels
constant constexpr uint DC=2u*2u*DG;        // conv deltas of a sublayer: [in | out][tap][group]
constant constexpr uint DLAYERS=5u;
constant constexpr uint DWIN=2047u;         // attention reach either side
inline uint4 dinfo4(const device uint* ri,uint r){return *((const device uint4*)(ri+r*4u));}
inline float df_sum(float v,threadgroup float* red,uint lane,uint simd) {
    v=simd_sum(v);
    if(lane==0u)red[simd]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f;
    for(uint i=0;i<8u;++i)t+=red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}
// RMSNorm over 128 values held 4 per lane (one SIMD group per head), then NeoX RoPE at `position`
inline void df_head_norm_rope(thread float* v,thread const float* w4,uint lane,uint position,bool rope) {
    float ss=v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
    ss=simd_sum(ss);
    const float inv=metal::precise::rsqrt(ss/128.0f+1e-6f);
    for(uint t=0;t<4u;++t)v[t]=v[t]*inv*w4[t];
    if(!rope)return;
    float other[4];
    for(uint t=0;t<4u;++t)other[t]=simd_shuffle_xor(v[t],ushort(16));
    for(uint t=0;t<4u;++t) {
        const uint d=(4u*lane+t)&63u;
        const float freq=metal::precise::pow(10000.0f,-float(2u*d)/128.0f);
        const float ang=float(position)*freq;
        const float c=metal::precise::cos(ang),s=metal::precise::sin(ang);
        v[t]=lane<16u?v[t]*c-other[t]*s:v[t]*c+other[t]*s;
    }
}

// ---- context: x[r] = [feat0[r] | feat1[r] | feat2[r] | feat3[r]] (feat BF16 [4][R][DH]) -> out BF16 [R][4 DH]
struct DfRows { uint rows, pad0, pad1, pad2; };
kernel void df_ctx_gather(const device ushort* feat [[buffer(0)]],device ushort* out [[buffer(1)]],
    constant DfRows& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*4u*DH)return;
    const uint r=i/(4u*DH),t=(i/DH)%4u,c=i%DH;
    out[i]=feat[(size_t(t)*p.rows+r)*DH+c];
}
// ---- RMSNorm of BF16 rows (eps 1e-6): out[r] = x[r] / rms(x[r]) * w (BF16), one threadgroup per row
struct DfNorm { ulong w; uint rows, x_stride, out_stride, pad; };
kernel void df_norm_rows(const device uchar* weights [[buffer(0)]],const device ushort* x [[buffer(1)]],
    device ushort* out [[buffer(2)]],constant DfNorm& p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float red[8];
    const device ushort* xr=x+size_t(row)*p.x_stride;
    float ss=0.0f;
    for(uint i=tid;i<DH;i+=256u){const float v=bfu(xr[i]);ss+=v*v;}
    const float inv=metal::precise::rsqrt(df_sum(ss,red,lane,simd)/float(DH)+1e-6f);
    const device ushort* w=(const device ushort*)(weights+p.w);
    for(uint i=tid;i<DH;i+=256u)out[size_t(row)*p.out_stride+i]=ubf(bfu(xr[i])*inv*bfu(w[i]));
}
// ---- committed rows' context K / V: kv BF16 [R][DLAYERS][2 DKV] (k | v per layer) -> k_norm + RoPE -> cache
struct DfCtx { ulong knorm[DLAYERS]; ulong layer_elems; uint rows, cap, pad0, pad1; };
kernel void df_ctx_write(const device uchar* weights [[buffer(0)]],const device ushort* kv [[buffer(1)]],
    device ushort* cache [[buffer(2)]],const device uint* cinfo [[buffer(3)]],constant DfCtx& p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row=g.x,l=g.y;
    if(row>=p.rows || l>=DLAYERS)return;
    const uint4 ci=dinfo4(cinfo,row);
    if(ci.z==0u)return;
    const device ushort* src=kv+(size_t(row)*DLAYERS+l)*2u*DKV;
    device ushort* layer=cache+size_t(l)*p.layer_elems;
    const device ushort* wn=(const device ushort*)(weights+p.knorm[l]);
    for(uint hd=simd;hd<16u;hd+=8u) {           // 8 k heads, 8 v heads
        const bool isk=hd<8u;
        const uint h=hd%8u;
        device ushort* dst=layer+((size_t(isk?0u:8u)+h)*p.cap+ci.y)*128u+4u*lane;
        float v[4],w4[4];
        for(uint t=0;t<4u;++t){v[t]=bfu(src[(isk?0u:DKV)+h*128u+4u*lane+t]);w4[t]=bfu(wn[4u*lane+t]);}
        if(isk)df_head_norm_rope(v,w4,lane,ci.y,true);
        for(uint t=0;t<4u;++t)dst[t]=ubf(v[t]);
    }
}

// ---- block: residual (FP32 [8][DH]) += add (BF16, flags & 1), or = add (flags & 2: the first layer);
// out = RMSNorm(residual) * w (BF16). One threadgroup per row.
struct DfAddNorm { ulong w; uint rows, flags, pad0, pad1; };
kernel void df_add_norm(const device uchar* weights [[buffer(0)]],device float* resid [[buffer(1)]],
    const device ushort* add [[buffer(2)]],device ushort* out [[buffer(3)]],constant DfAddNorm& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float red[8];
    device float* r=resid+size_t(row)*DH;
    const device ushort* a=add+size_t(row)*DH;
    float ss=0.0f;
    for(uint i=tid;i<DH;i+=256u) {
        float v=(p.flags&2u)?bfu(a[i]):((p.flags&1u)?r[i]+bfu(a[i]):r[i]);
        v=rb(v);                                  // the residual stream in BF16
        r[i]=v;ss+=v*v;
    }
    const float inv=metal::precise::rsqrt(df_sum(ss,red,lane,simd)/float(DH)+1e-6f);
    const device ushort* w=(const device ushort*)(weights+p.w);
    for(uint i=tid;i<DH;i+=256u)out[size_t(row)*DH+i]=ubf(r[i]*inv*bfu(w[i]));
}
// ---- two-tap dynamic grouped convolution of the block rows: out[b][c] = (base[0][c] + d[b][0][g]) x[b][c]
// + (base[1][c] + d[b][1][g]) x[b - 1][c] (no previous row at b = 0), g = c / 16; d = deltas[b][io] [2][DG]
struct DfConv { ulong base; uint rows, io, pad0, pad1; };
kernel void df_conv(const device uchar* weights [[buffer(0)]],const device ushort* x [[buffer(1)]],
    const device ushort* deltas [[buffer(2)]],device ushort* out [[buffer(3)]],constant DfConv& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*DH)return;
    const uint b=i/DH,c=i%DH,g=c/16u;
    const device ushort* base=(const device ushort*)(weights+p.base)+size_t(p.io)*2u*DH;
    const device ushort* d=deltas+size_t(b)*DC+p.io*2u*DG;
    const float c0=bfu(base[c])+bfu(d[g]),c1=bfu(base[DH+c])+bfu(d[DG+g]);
    const float cur=bfu(x[size_t(b)*DH+c]);
    const float prev=(b%8u)?bfu(x[size_t(b-1u)*DH+c]):0.0f;
    out[i]=ubf(c0*cur+c1*prev);
}
// ---- q / k norms and RoPE of the block rows; k / v into the drafter cache at the rows' positions; q BF16 [8][DQ]
struct DfQK { ulong qn, kn, layer_elems; uint rows, cap, layer, pad; };
kernel void df_qk_rows(const device uchar* weights [[buffer(0)]],const device ushort* qkv [[buffer(1)]],
    device ushort* q [[buffer(2)]],device ushort* cache [[buffer(3)]],const device uint* dinfo [[buffer(4)]],
    constant DfQK& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const uint4 di=dinfo4(dinfo,row);
    const device ushort* src=qkv+size_t(row)*DQKV;
    device ushort* layer=cache+size_t(p.layer)*p.layer_elems;
    for(uint hd=simd;hd<48u;hd+=8u) {            // 32 q heads, 8 k heads, 8 v heads
        if(hd>=40u) {
            const uint h=hd-40u;
            for(uint t=0;t<4u;++t)layer[(size_t(8u+h)*p.cap+di.y)*128u+4u*lane+t]=src[DQ+DKV+h*128u+4u*lane+t];
            continue;
        }
        const bool isq=hd<32u;
        const uint off=isq?hd*128u:DQ+(hd-32u)*128u;
        const device ushort* wn=(const device ushort*)(weights+(isq?p.qn:p.kn));
        float v[4],w4[4];
        for(uint t=0;t<4u;++t){v[t]=bfu(src[off+4u*lane+t]);w4[t]=bfu(wn[4u*lane+t]);}
        df_head_norm_rope(v,w4,lane,di.y,true);
        if(isq)for(uint t=0;t<4u;++t)q[size_t(row)*DQ+hd*128u+4u*lane+t]=ubf(v[t]);
        else for(uint t=0;t<4u;++t)layer[(size_t(hd-32u)*p.cap+di.y)*128u+4u*lane+t]=ubf(v[t]);
    }
}
// ---- attention of a block row over keys [max(0, position - DWIN), block start + 8) (no causal mask); grid (8 KV
// heads, rows), 4 query heads per KV head; 8 SIMD groups split the keys, merged at the end
struct DfAttn { ulong layer_elems; uint rows, cap, layer, pad; };
kernel void df_attn_rows(const device ushort* q [[buffer(0)]],const device ushort* cache [[buffer(1)]],
    device ushort* out [[buffer(2)]],const device uint* dinfo [[buffer(3)]],constant DfAttn& p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
    const uint h=g.x,row=g.y;
    if(row>=p.rows)return;
    const uint4 di=dinfo4(dinfo,row);
    const uint hi=di.z+8u;
    const uint lo=di.y>DWIN?di.y-DWIN:0u;
    const device ushort* layer=cache+size_t(p.layer)*p.layer_elems;
    const device ushort* K=layer+size_t(h)*p.cap*128u;
    const device ushort* V=layer+size_t(8u+h)*p.cap*128u;
    const float scale=0.08838834764831845f;   // 128^-0.5
    float qv[4][4],m[4],l[4],acc[4][4];
    for(uint a=0;a<4u;++a) {
        for(uint t=0;t<4u;++t){qv[a][t]=bfu(q[size_t(row)*DQ+(4u*h+a)*128u+4u*lane+t])*scale;acc[a][t]=0.0f;}
        m[a]=-INFINITY;l[a]=0.0f;
    }
    for(uint k=lo+simd;k<hi;k+=8u) {
        float kv4[4],vv4[4];
        for(uint t=0;t<4u;++t){kv4[t]=bfu(K[size_t(k)*128u+4u*lane+t]);vv4[t]=bfu(V[size_t(k)*128u+4u*lane+t]);}
        for(uint a=0;a<4u;++a) {
            float s=qv[a][0]*kv4[0]+qv[a][1]*kv4[1]+qv[a][2]*kv4[2]+qv[a][3]*kv4[3];
            s=simd_sum(s);
            const float mn=max(m[a],s);
            const float c=metal::fast::exp(m[a]-mn),e=metal::fast::exp(s-mn);
            l[a]=l[a]*c+e;
            for(uint t=0;t<4u;++t)acc[a][t]=acc[a][t]*c+e*vv4[t];
            m[a]=mn;
        }
    }
    threadgroup float tm[8][4],tl[8][4];
    threadgroup float ta[8][4][128];
    for(uint a=0;a<4u;++a) {
        if(lane==0u){tm[simd][a]=m[a];tl[simd][a]=l[a];}
        for(uint t=0;t<4u;++t)ta[simd][a][4u*lane+t]=acc[a][t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<4u*128u;i+=256u) {
        const uint a=i/128u,d=i%128u;
        float M=-INFINITY;
        for(uint s=0;s<8u;++s)M=max(M,tm[s][a]);
        float L=0.0f,A=0.0f;
        for(uint s=0;s<8u;++s) {
            if(tm[s][a]==-INFINITY)continue;
            const float c=metal::fast::exp(tm[s][a]-M);
            L+=tl[s][a]*c;A+=ta[s][a][d]*c;
        }
        out[size_t(row)*DQ+(4u*h+a)*128u+d]=ubf(A/L);
    }
}
// ---- act = silu(gate) * up; gu BF16 [R][2 DI] (gate | up)
kernel void df_silu_rows(const device ushort* gu [[buffer(0)]],device ushort* act [[buffer(1)]],
    constant DfRows& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*DI)return;
    const uint r=i/DI,c=i%DI;
    const float gv=bfu(gu[size_t(r)*2u*DI+c]),uv=bfu(gu[size_t(r)*2u*DI+DI+c]);
    act[i]=ubf(gv/(1.0f+metal::precise::exp(-gv))*uv);
}
// ---- drafted tokens: per row the largest logit over the real vocabulary (ties: lowest id) and the log-sum-exp;
// logits FP32 [rows][stride]; drafts[r], top[r][16]: [0] the id (as float bits), [4] its logit, [8] the lse (the
// layout of the DSpark candidate kernel's statistics). grid (rows), 1024 threads.
struct DfPick { uint rows, vocab, stride, pad; };
kernel void df_pick(const device float* logits [[buffer(0)]],device uint* drafts [[buffer(1)]],device float* top [[buffer(2)]],
    constant DfPick& p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const device float* lg=logits+size_t(row)*p.stride;
    float best=-INFINITY;uint bi=0xffffffffu;
    for(uint v=tid;v<p.vocab;v+=1024u){const float x=lg[v];if(x>best){best=x;bi=v;}}
    for(ushort o=16;o>=1;o>>=1) {
        const float ob=simd_shuffle_down(best,o);const uint oi=simd_shuffle_down(bi,o);
        if(ob>best||(ob==best&&oi<bi)){best=ob;bi=oi;}
    }
    threadgroup float vb[32];threadgroup uint vi[32];threadgroup float vs[32];
    if(lane==0u){vb[simd]=best;vi[simd]=bi;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best=vb[0];bi=vi[0];
    for(uint j=1;j<32u;++j)if(vb[j]>best||(vb[j]==best&&vi[j]<bi)){best=vb[j];bi=vi[j];}
    float e=0.0f;
    for(uint v=tid;v<p.vocab;v+=1024u)e+=metal::fast::exp(lg[v]-best);
    e=simd_sum(e);
    if(lane==0u)vs[simd]=e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u) {
        float z=0.0f;for(uint j=0;j<32u;++j)z+=vs[j];
        drafts[row]=bi;
        device float* t=top+size_t(row)*16u;
        t[0]=as_type<float>(bi);t[4]=best;t[8]=best+metal::log(z);
    }
}
// ---- candidates: per row the 16 largest logits over the real vocabulary (ids and values, unordered), the
// log-sum-exp and the largest logit. Pass 1: the maximum and a histogram of the top 11 bits of the order-preserving
// key; pass 2: the exponential sum, the values above the 16th value's bin and that bin's values (at most 1024 kept)
// collected, the bin resolved by repeated maxima. grid (rows), 1024 threads.
constant constexpr uint DK=16u;
struct DfTop { uint rows, vocab, stride, pad; };
inline uint df_key(float x){const uint u=as_type<uint>(x);return (u&0x80000000u)?~u:(u|0x80000000u);}
kernel void df_top16(const device float* logits [[buffer(0)]],device uint* cand [[buffer(1)]],device float* cval [[buffer(2)]],
    device float* stat [[buffer(3)]],constant DfTop& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const device float* lg=logits+size_t(row)*p.stride;
    threadgroup atomic_uint hist[2048];
    threadgroup float bv[1024];threadgroup uint bi[1024];
    threadgroup float red[32];threadgroup uint sh[4];
    for(uint i=tid;i<2048u;i+=1024u)atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
    if(tid<4u)sh[tid]=0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY;
    for(uint v=tid;v<p.vocab;v+=1024u){const float x=lg[v];m=max(m,x);
        atomic_fetch_add_explicit(&hist[df_key(x)>>21],1u,memory_order_relaxed);}
    m=simd_max(m);
    if(lane==0u)red[simd]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=red[0];for(uint j=1;j<32u;++j)m=max(m,red[j]);
    if(tid==0u) {
        uint above=0u,bin=0u;
        for(int b=2047;b>=0;--b){const uint c=atomic_load_explicit(&hist[b],memory_order_relaxed);
            if(above+c>=DK){bin=uint(b);break;} above+=c;}
        sh[0]=bin;sh[1]=DK-above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint b1=sh[0],need=sh[1];
    threadgroup atomic_uint* cnt=(threadgroup atomic_uint*)(sh+2);
    if(tid==0u){atomic_store_explicit(&cnt[0],0u,memory_order_relaxed);atomic_store_explicit(&cnt[1],0u,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint* ci=cand+size_t(row)*DK;
    device float* cv=cval+size_t(row)*DK;
    float e=0.0f;
    for(uint v=tid;v<p.vocab;v+=1024u) {
        const float x=lg[v];
        e+=metal::fast::exp(x-m);
        const uint b=df_key(x)>>21;
        if(b>b1){const uint at=atomic_fetch_add_explicit(&cnt[0],1u,memory_order_relaxed);ci[at]=v;cv[at]=x;}
        else if(b==b1){const uint at=atomic_fetch_add_explicit(&cnt[1],1u,memory_order_relaxed);
            if(at<1024u){bv[at]=x;bi[at]=v;}}
    }
    e=simd_sum(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0u)red[simd]=e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nabove=DK-need,nb=min(atomic_load_explicit(&cnt[1],memory_order_relaxed),1024u);
    // the bin: need times the largest remaining (ties: lowest id)
    for(uint k=0;k<need;++k) {
        float x=tid<nb?bv[tid]:-INFINITY;uint id=tid<nb?bi[tid]:0xffffffffu;
        for(ushort o=16;o>=1;o>>=1){const float ox=simd_shuffle_down(x,o);const uint oi=simd_shuffle_down(id,o);
            if(ox>x||(ox==x&&oi<id)){x=ox;id=oi;}}
        threadgroup float* wv=(threadgroup float*)(hist);          // reuse: 32 values + 32 ids
        threadgroup uint* wi=(threadgroup uint*)(hist)+32u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane==0u){wv[simd]=x;wi[simd]=id;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x=wv[0];id=wi[0];
        for(uint j=1;j<32u;++j)if(wv[j]>x||(wv[j]==x&&wi[j]<id)){x=wv[j];id=wi[j];}
        if(tid==0u){ci[nabove+k]=id;cv[nabove+k]=x;}
        if(tid<nb && bi[tid]==id)bv[tid]=-INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u){float z=0.0f;for(uint j=0;j<32u;++j)z+=red[j];stat[row*2u]=m;stat[row*2u+1u]=m+metal::log(z);}
}
// ---- selector walk (greedy): position e's predecessor is the anchor (e = 0) or the token chosen at e - 1; per
// candidate c the correction sum_r BF16(pred_cb[pred][r] * hp[e][r]) * succ_cb[cand][r] (BF16), the score
// cap * tanh((logit + correction) / cap), the largest score chosen (ties: lowest id). One threadgroup of 256:
// thread t holds rank t of the predecessor term, candidates 16 threads each over 16 ranks.
struct DfSelect { ulong pred_cb, succ_cb; uint rows, pad0; float cap, pad1; };
kernel void df_select(const device uchar* weights [[buffer(0)]],const device uint* cand [[buffer(1)]],
    const device float* cval [[buffer(2)]],const device float* stat [[buffer(3)]],const device ushort* hp [[buffer(4)]],
    const device uint* tok [[buffer(5)]],device uint* drafts [[buffer(6)]],device float* top [[buffer(7)]],
    constant DfSelect& p [[buffer(8)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a[256];
    threadgroup float sc[DK];
    threadgroup uint chosen[1];
    const device ushort* pc=(const device ushort*)(weights+p.pred_cb);
    const device ushort* scb=(const device ushort*)(weights+p.succ_cb);
    if(tid==0u)chosen[0]=tok[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=0;e<p.rows;++e) {
        const uint pred=chosen[0];
        a[tid]=rb(bfu(pc[size_t(pred)*256u+tid])*bfu(hp[e*256u+tid]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c=tid/16u,j=tid%16u;
        const uint id=cand[e*DK+c];
        const device ushort* sr=scb+size_t(id)*256u+j*16u;
        float s=0.0f;
        for(uint r=0;r<16u;++r)s+=a[j*16u+r]*bfu(sr[r]);
        for(ushort o=8;o>=1;o>>=1)s+=simd_shuffle_xor(s,o);
        if(j==0u){const float v=cval[e*DK+c]+rb(s);sc[c]=p.cap*metal::precise::tanh(v/p.cap);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0u) {
            uint bc=0u;
            for(uint k=1;k<DK;++k)if(sc[k]>sc[bc]||(sc[k]==sc[bc]&&cand[e*DK+k]<cand[e*DK+bc]))bc=k;
            const uint t=cand[e*DK+bc];
            chosen[0]=t;drafts[e]=t;
            device float* o=top+size_t(e)*16u;
            o[0]=as_type<float>(t);o[4]=cval[e*DK+bc];o[8]=stat[e*2u+1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
