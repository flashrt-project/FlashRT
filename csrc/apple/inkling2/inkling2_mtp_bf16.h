#pragma once
#include "common/source.h"
#include "inkling2/inkling2_common.h"
#include <string>
// Inkling 2 Mini speculative decoding (DSpark drafter of mtp.safetensors) on the row programs.
//
// Target side. A verify step runs the row program on 8 rows per sequence (the anchor and 7 drafted tokens) with
// the conv-ring frames and sliding-window K/V of every row staged per visit (ink_ring_stage_rows) instead of
// written: after the accepted prefix is known, ink_commit_rows writes the accepted rows exactly as ink_ring_rows
// would have (global-layer K/V is written by position and never read past the query, so rejected rows need no
// undo there). Draft features of a row: the hidden state entering visits 5, 23 and 41 and the final normalized
// state (feat BF16 [4][R][2304]).
//
// Drafter (DFlash block of 8: the anchor's embedding and 7 mask embeddings, 3 bidirectional layers; layers 0 and 2
// see the 1025 positions before the block, layer 1 all of them). Context K/V of a committed position: per layer a
// softmax-weighted mix of the 4 features, RMSNorm, k / v context projections, k RMSNorm, RoPE. Each drafted
// position: base logits (LM head of the block's final hidden state) plus a Markov RNN head bias (state, embedding
// of the previous token, the position's hidden state and the mix of the last committed features through
// rnn_proj; latent through the folded hidden_out x LM head), greedy over the checkpoint vocabulary. Drafts only
// propose (the target's verification decides the output), so the drafter computes in FP32 without matching any
// reference rounding.
//
// Row metadata: dinfo[r] = (sequence, position, block start, index in block); cinfo[r] = (sequence, position,
// keep, last committed of its sequence).
namespace flashrt::apple::shaders {
inline std::string inkling2_mtp_bf16_source() {
    return std::string(metal_preamble) + inkling2_common_msl() + R"MSL(
struct RowsVisit {
    ulong kv_off, seq_kv, ring_off, seq_ring;
    uint layer, pass, cap, rows;
    uint nblocks, bslot, slot0, slot1, slot2, slot3;
    uint flags, pad;
};
constant constexpr uint FPROJ=8672u;
constant constexpr uint STAGE=FRING+2u*8u*128u;   // staged row of a visit: conv frame, then sliding K / V
inline uint4 rinfo4(const device uint* ri,uint r){return *((const device uint4*)(ri+r*4u));}
inline float tg_sum(float v,threadgroup float* red,uint lane,uint simd) {
    v=simd_sum(v);
    if(lane==0u)red[simd]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=0.0f;
    for(uint i=0;i<8u;++i)t+=red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// ------------------------------------------------------------------------------------------------ target side
// ink_ring_rows with every row's frame and sliding K / V staged at stage[visit][row]
kernel void ink_ring_stage_rows(const device ushort* proj [[buffer(0)]],const device ushort* o [[buffer(1)]],
    const device ushort* moe [[buffer(2)]],const device ushort* chunk [[buffer(3)]],device ushort* stage [[buffer(4)]],
    constant RowsVisit& p [[buffer(5)]],constant uint& visit [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    device ushort* st=stage+(size_t(visit)*R+row)*STAGE;
    for(uint i=tid;i<FKV;i+=256u){st[i]=proj[size_t(row)*FPROJ+P_K+i];st[FKV+i]=proj[size_t(row)*FPROJ+P_V+i];}
    for(uint i=tid;i<FH;i+=256u){st[2048u+i]=o[size_t(row)*FH+i];st[4352u+i]=moe[size_t(row)*FH+i];}
    if((p.flags&1u)==0u)
        for(uint i=tid;i<2u*8u*128u;i+=256u) {
            const uint v=i/(8u*128u),hh=(i/128u)%8u,dd=i%128u;
            st[FRING+i]=chunk[((v?8u:0u)+hh)*size_t(R)*128u+size_t(row)*128u+dd];
        }
}
// Accepted rows into the rings / sliding caches: commit[r] = (keep, last accepted position of the row's sequence).
// grid (rows, visits)
struct CommitVisits { ulong seq_kv, seq_ring; uint rows, visits, pad0, pad1; ulong kv_off[48]; ulong ring_off[48];
                      uint global[48]; };
kernel void ink_commit_rows(const device ushort* stage [[buffer(0)]],device ushort* rings [[buffer(1)]],
    device ushort* kv [[buffer(2)]],const device uint* rows_info [[buffer(3)]],const device uint* commit [[buffer(4)]],
    constant CommitVisits& cv [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint row=g.x,v=g.y,R=cv.rows;
    if(row>=R || v>=cv.visits || commit[row*2u]==0u)return;
    const uint4 inf=rinfo4(rows_info,row);
    const uint position=inf.y,last=commit[row*2u+1u];
    const device ushort* st=stage+(size_t(v)*R+row)*STAGE;
    if(position+3u>=last) {
        device ushort* ring=rings+inf.x*cv.seq_ring+cv.ring_off[v]+(position&3u)*FRING;
        for(uint i=tid;i<FRING;i+=256u)ring[i]=st[i];
    }
    if(cv.global[v]==0u && position+FWIN>last) {
        device ushort* base=kv+inf.x*cv.seq_kv+cv.kv_off[v];
        for(uint i=tid;i<2u*8u*128u;i+=256u) {
            const uint w=i/(8u*128u),hh=(i/128u)%8u,dd=i%128u;
            base[(w?ulong(8u)*FWIN*128u:0ul)+(size_t(hh)*FWIN+(position&(FWIN-1u)))*128u+dd]=st[FRING+i];
        }
    }
}

// ------------------------------------------------------------------------------------------------ drafter
struct MtpMix { ulong norm[4]; float alpha[16]; uint rows, first_mix, feat_stride, pad1; };   // feat_stride 0: rows
// out[m][r] = RMSNorm(sum_t feat[t][r] * alpha[m][t]) * norm_m, m = first_mix..3 (3 drafter layers, the Markov head);
// the Markov mix only for a sequence's last committed row, into frnn[sequence]
kernel void ink_mtp_mix_rows(const device uchar* weights [[buffer(0)]],const device ushort* feat [[buffer(1)]],
    device ushort* out [[buffer(2)]],device ushort* frnn [[buffer(3)]],const device uint* cinfo [[buffer(4)]],
    constant MtpMix& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows,FS=p.feat_stride?p.feat_stride:R;
    if(row>=R)return;
    const uint4 ci=rinfo4(cinfo,row);
    if(ci.z==0u)return;
    threadgroup float red[8];
    threadgroup float mixed[FH];
    for(uint m=p.first_mix;m<4u;++m) {
        if(m==3u && ci.w==0u)break;
        float ss=0.0f;
        for(uint i=tid;i<FH;i+=256u) {
            float v=0.0f;
            for(uint t=0;t<4u;++t)v+=bfu(feat[(size_t(t)*FS+row)*FH+i])*p.alpha[m*4u+t];
            mixed[i]=v;ss+=v*v;
        }
        const float inv=metal::precise::rsqrt(tg_sum(ss,red,lane,simd)/float(FH)+1e-6f);
        const device ushort* w=(const device ushort*)(weights+p.norm[m]);
        device ushort* dst=m==3u?frnn+size_t(ci.x)*FH:out+(size_t(m)*R+row)*FH;
        for(uint i=tid;i<FH;i+=256u)dst[i]=ubf(mixed[i]*inv*bfu(w[i]));
    }
}
// RMSNorm over 128 values held 4 per lane (one SIMD group per head), then NeoX RoPE at `position`
inline void head_norm_rope(thread float* v,thread const float* w4,uint lane,uint position,float theta) {
    float ss=v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
    ss=simd_sum(ss);
    const float inv=metal::precise::rsqrt(ss/128.0f+1e-6f);
    for(uint t=0;t<4u;++t)v[t]=v[t]*inv*w4[t];
    // pair (d, d + 64): lanes l and l + 16 hold dims 4l..4l+3 and 4l+64..4l+67
    float other[4];
    for(uint t=0;t<4u;++t)other[t]=simd_shuffle_xor(v[t],ushort(16));
    for(uint t=0;t<4u;++t) {
        const uint d=(4u*lane+t)&63u;
        const float freq=metal::precise::pow(theta,-float(2u*d)/128.0f);
        const float ang=float(position)*freq;
        const float c=metal::precise::cos(ang),s=metal::precise::sin(ang);
        v[t]=lane<16u?v[t]*c-other[t]*s:v[t]*c+other[t]*s;
    }
}
struct MtpCtx { ulong knorm[3]; ulong seq_kv; uint rows, cap; float theta; uint pad; };
// committed rows' context K / V into the drafter cache dkv[seq][layer][k|v][8][cap][128]; kvc BF16 [R][3][2048]
kernel void ink_mtp_ctx_write_rows(const device uchar* weights [[buffer(0)]],const device ushort* kvc [[buffer(1)]],
    device ushort* dkv [[buffer(2)]],const device uint* cinfo [[buffer(3)]],constant MtpCtx& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const uint4 ci=rinfo4(cinfo,row);
    if(ci.z==0u)return;
    const uint seq=ci.x,pos=ci.y,h=simd;
    for(uint l=0;l<3u;++l) {
        const device ushort* src=kvc+(size_t(row)*3u+l)*2048u;
        device ushort* layer=dkv+size_t(seq)*p.seq_kv+size_t(l)*2u*8u*p.cap*128u;
        const device ushort* wk=(const device ushort*)(weights+p.knorm[l]);
        float k[4],w4[4];
        for(uint t=0;t<4u;++t){k[t]=bfu(src[h*128u+4u*lane+t]);w4[t]=bfu(wk[4u*lane+t]);}
        head_norm_rope(k,w4,lane,pos,p.theta);
        for(uint t=0;t<4u;++t) {
            layer[(size_t(h)*p.cap+pos)*128u+4u*lane+t]=ubf(k[t]);
            layer[(size_t(8u+h)*p.cap+pos)*128u+4u*lane+t]=src[1024u+h*128u+4u*lane+t];
        }
    }
}
struct MtpEmbed { ulong mask; uint rows, pad; };
// block rows: the anchor's normalized embedding (the target's ink_embed_rows output, h) or the mask embedding
kernel void ink_mtp_embed_rows(const device uchar* weights [[buffer(0)]],const device ushort* h [[buffer(1)]],
    device float* dh [[buffer(2)]],const device uint* dinfo [[buffer(3)]],constant MtpEmbed& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const uint4 di=rinfo4(dinfo,row);
    const device ushort* mask=(const device ushort*)(weights+p.mask);
    for(uint i=tid;i<FH;i+=256u)dh[size_t(row)*FH+i]=di.w==0u?bfu(h[size_t(di.x)*FH+i]):bfu(mask[i]);
}
struct MtpNorm { ulong w; uint rows, add; };
// dh += add (BF16, when add bit 0); x = RMSNorm(dh) * w (BF16). add bit 1: x rows permuted by block index (drafted
// rows j >= 1 at (j - 1) * seqs + sequence, the anchors after them) for the LM head and the Markov walk.
kernel void ink_mtp_norm_rows(const device uchar* weights [[buffer(0)]],device float* dh [[buffer(1)]],
    const device ushort* addv [[buffer(2)]],device ushort* x [[buffer(3)]],constant MtpNorm& p [[buffer(4)]],
    const device uint* dinfo [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    uint orow=row;
    if(p.add&2u) {
        const uint4 di=rinfo4(dinfo,row);
        const uint seqs=p.rows/8u;
        orow=di.w==0u?7u*seqs+di.x:(di.w-1u)*seqs+di.x;
    }
    threadgroup float red[8];
    device float* r=dh+size_t(row)*FH;
    float ss=0.0f;
    for(uint i=tid;i<FH;i+=256u) {
        float v=r[i];
        if(p.add&1u){v+=bfu(addv[size_t(row)*FH+i]);r[i]=v;}
        ss+=v*v;
    }
    const float inv=metal::precise::rsqrt(tg_sum(ss,red,lane,simd)/float(FH)+1e-6f);
    const device ushort* w=(const device ushort*)(weights+p.w);
    for(uint i=tid;i<FH;i+=256u)x[size_t(orow)*FH+i]=ubf(r[i]*inv*bfu(w[i]));
}
struct MtpQK { ulong qn, kn, seq_kv, layer_off; uint rows, cap; float theta; uint pad; };
// q / k head norms and RoPE; the block's K / V into the drafter cache at their positions. qkv BF16 [R][5120]
kernel void ink_mtp_qk_rows(const device uchar* weights [[buffer(0)]],const device ushort* qkv [[buffer(1)]],
    device ushort* q [[buffer(2)]],device ushort* dkv [[buffer(3)]],const device uint* dinfo [[buffer(4)]],
    constant MtpQK& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const uint4 di=rinfo4(dinfo,row);
    const device ushort* src=qkv+size_t(row)*5120u;
    device ushort* layer=dkv+size_t(di.x)*p.seq_kv+p.layer_off;
    for(uint hd=simd;hd<40u;hd+=8u) {
        if(hd>=32u) {             // v heads: copy
            const uint h=hd-32u;
            for(uint t=0;t<4u;++t)layer[(size_t(8u+h)*p.cap+di.y)*128u+4u*lane+t]=src[4096u+h*128u+4u*lane+t];
            continue;
        }
        const bool isq=hd<24u;
        const uint off=isq?hd*128u:3072u+(hd-24u)*128u;
        const device ushort* wn=(const device ushort*)(weights+(isq?p.qn:p.kn));
        float v[4],w4[4];
        for(uint t=0;t<4u;++t){v[t]=bfu(src[off+4u*lane+t]);w4[t]=bfu(wn[4u*lane+t]);}
        head_norm_rope(v,w4,lane,di.y,p.theta);
        if(isq)for(uint t=0;t<4u;++t)q[size_t(row)*3072u+hd*128u+4u*lane+t]=ubf(v[t]);
        else for(uint t=0;t<4u;++t)layer[(size_t(hd-24u)*p.cap+di.y)*128u+4u*lane+t]=ubf(v[t]);
    }
}
struct MtpAttn { ulong seq_kv, layer_off; uint rows, cap, window, pad; };
// bidirectional attention of a block row over [lo, block start + 8): lo = block start - (window + 1) (window != 0)
// or 0. grid (8 kv heads, rows); 3 query heads per kv head; 8 SIMD groups split the keys, merged at the end.
kernel void ink_mtp_attn_rows(const device ushort* q [[buffer(0)]],const device ushort* dkv [[buffer(1)]],
    device ushort* out [[buffer(2)]],const device uint* dinfo [[buffer(3)]],constant MtpAttn& p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
    const uint h=g.x,row=g.y;
    if(row>=p.rows)return;
    const uint4 di=rinfo4(dinfo,row);
    const uint hi=di.z+8u;
    const uint lo=p.window && di.z>p.window+1u?di.z-(p.window+1u):0u;
    const device ushort* layer=dkv+size_t(di.x)*p.seq_kv+p.layer_off;
    const device ushort* K=layer+size_t(h)*p.cap*128u;
    const device ushort* V=layer+size_t(8u+h)*p.cap*128u;
    const float scale=0.08838834764831845f;   // 128^-0.5
    float qv[3][4],m[3],l[3],acc[3][4];
    for(uint a=0;a<3u;++a) {
        for(uint t=0;t<4u;++t){qv[a][t]=bfu(q[size_t(row)*3072u+(3u*h+a)*128u+4u*lane+t])*scale;acc[a][t]=0.0f;}
        m[a]=-INFINITY;l[a]=0.0f;
    }
    for(uint k=lo+simd;k<hi;k+=8u) {
        float kv4[4],vv4[4];
        for(uint t=0;t<4u;++t){kv4[t]=bfu(K[size_t(k)*128u+4u*lane+t]);vv4[t]=bfu(V[size_t(k)*128u+4u*lane+t]);}
        for(uint a=0;a<3u;++a) {
            float s=qv[a][0]*kv4[0]+qv[a][1]*kv4[1]+qv[a][2]*kv4[2]+qv[a][3]*kv4[3];
            s=simd_sum(s);
            const float mn=max(m[a],s);
            const float c=metal::precise::exp(m[a]-mn),e=metal::precise::exp(s-mn);
            l[a]=l[a]*c+e;
            for(uint t=0;t<4u;++t)acc[a][t]=acc[a][t]*c+e*vv4[t];
            m[a]=mn;
        }
    }
    threadgroup float tm[8][3],tl[8][3];
    threadgroup float ta[8][3][128];
    for(uint a=0;a<3u;++a) {
        if(lane==0u){tm[simd][a]=m[a];tl[simd][a]=l[a];}
        for(uint t=0;t<4u;++t)ta[simd][a][4u*lane+t]=acc[a][t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<3u*128u;i+=256u) {
        const uint a=i/128u,d=i%128u;
        float M=-INFINITY;
        for(uint s=0;s<8u;++s)M=max(M,tm[s][a]);
        float L=0.0f,A=0.0f;
        for(uint s=0;s<8u;++s) {
            if(tm[s][a]==-INFINITY)continue;
            const float c=metal::precise::exp(tm[s][a]-M);
            L+=tl[s][a]*c;A+=ta[s][a][d]*c;
        }
        out[size_t(row)*3072u+(3u*h+a)*128u+d]=ubf(A/L);
    }
}
// act = silu(gate) * up; gu BF16 [R][18432] (gate 9216 | up 9216)
kernel void ink_mtp_silu_rows(const device ushort* gu [[buffer(0)]],device ushort* act [[buffer(1)]],
    constant uint& rows [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=rows*9216u)return;
    const uint r=i/9216u,c=i%9216u;
    const float g=bfu(gu[size_t(r)*18432u+c]),u=bfu(gu[size_t(r)*18432u+9216u+c]);
    act[i]=ubf(g/(1.0f+metal::precise::exp(-g))*u);
}
struct MtpRnn { ulong w1; uint seqs, step; };
// rnn input of the Markov step: [state 256 | w1[prev] 256 | drafted position's hidden 2304 | frnn 2304]; hn and the
// base logits hold the drafted rows by step (row step * seqs + sequence)
kernel void ink_mtp_rnn_in(const device uchar* weights [[buffer(0)]],const device ushort* hn [[buffer(1)]],
    const device ushort* frnn [[buffer(2)]],const device ushort* state [[buffer(3)]],const device uint* prev [[buffer(4)]],
    device ushort* out [[buffer(5)]],constant MtpRnn& p [[buffer(6)]],uint s [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(s>=p.seqs)return;
    device ushort* o=out+size_t(s)*5120u;
    const device ushort* w1=(const device ushort*)(weights+p.w1)+size_t(prev[s])*256u;
    for(uint i=tid;i<256u;i+=256u){o[i]=state[size_t(s)*256u+i];o[256u+i]=w1[i];}
    for(uint i=tid;i<FH;i+=256u) {
        o[512u+i]=hn[(size_t(p.step)*p.seqs+s)*FH+i];
        o[512u+FH+i]=frnn[size_t(s)*FH+i];
    }
}
struct MtpGates { ulong bias; uint seqs, pad; };
// rnn_proj output (FP32, + bias) -> state' (BF16 rounding of each op, as the reference), latent = tanh(out)
kernel void ink_mtp_gates(const device uchar* weights [[buffer(0)]],const device float* proj [[buffer(1)]],
    device ushort* state [[buffer(2)]],device ushort* latent [[buffer(3)]],constant MtpGates& p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.seqs*256u)return;
    const uint s=i/256u,c=i%256u;
    const device ushort* b=(const device ushort*)(weights+p.bias);
    const float g=rb(proj[size_t(s)*768u+c]+bfu(b[c]));
    const float cand=rb(proj[size_t(s)*768u+256u+c]+bfu(b[256u+c]));
    const float o=rb(proj[size_t(s)*768u+512u+c]+bfu(b[512u+c]));
    const float st=bfu(state[i]);
    const float gate=rb(1.0f/(1.0f+metal::precise::exp(-g)));
    const float kept=rb(gate*st);
    const float blend=rb(rb(1.0f-gate)*rb(metal::precise::tanh(cand)));
    state[i]=ubf(kept+blend);
    latent[i]=ubf(metal::precise::tanh(o));
}
struct MtpPick { ulong fold_b; uint seqs, step, vocab, stride; };
// drafted token of each sequence at this step: argmax over the vocabulary of base logits + Markov bias
kernel void ink_mtp_pick(const device uchar* weights [[buffer(0)]],const device float* logits [[buffer(1)]],
    const device float* bias [[buffer(2)]],device uint* prev [[buffer(3)]],device uint* drafts [[buffer(4)]],
    constant MtpPick& p [[buffer(5)]],uint s [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(s>=p.seqs)return;
    const device float* lg=logits+(size_t(p.step)*p.seqs+s)*p.stride;
    const device float* bs=bias+size_t(s)*p.stride;
    const device ushort* fb=(const device ushort*)(weights+p.fold_b);
    float best=-INFINITY;uint bi=0u;
    for(uint v=tid;v<p.vocab;v+=1024u) {
        const float x=lg[v]+bs[v]+bfu(fb[v]);
        if(x>best){best=x;bi=v;}
    }
    for(ushort o=16;o>=1;o>>=1) {
        const float ob=simd_shuffle_down(best,o);const uint oi=simd_shuffle_down(bi,o);
        if(ob>best||(ob==best&&oi<bi)){best=ob;bi=oi;}
    }
    threadgroup float vb[32];threadgroup uint vi[32];
    if(lane==0u){vb[simd]=best;vi[simd]=bi;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd==0u) {
        best=vb[lane];bi=vi[lane];
        for(ushort o=16;o>=1;o>>=1) {
            const float ob=simd_shuffle_down(best,o);const uint oi=simd_shuffle_down(bi,o);
            if(ob>best||(ob==best&&oi<bi)){best=ob;bi=oi;}
        }
        if(lane==0u){prev[s]=bi;drafts[s*7u+p.step]=bi;}
    }
}
// Candidate drafting: the K largest base logits + fold bias of each drafted position (radix selection on the order-
// preserving key of the FP32 value). grid (rows), 1024 threads; cand_idx / cand_val [rows][K]. One pass over the row
// histograms the top 11 key bits; a second takes the values above the K-th value's bin and collects that bin
// (at most CAP values) in threadgroup memory, where the remaining 21 bits are selected (11 + 10); a larger bin falls
// back to two more passes over the row. Ties at the K-th value: the first collected.
constant constexpr uint MTP_CAND=256u;
constant constexpr uint TOPK_CAP=2048u;
struct MtpTopk { ulong fold_b; uint rows, vocab, stride, pad; };
inline uint order_key(float x){const uint u=as_type<uint>(x);return (u&0x80000000u)?~u:(u|0x80000000u);}
inline float key_value(uint k){return as_type<float>((k&0x80000000u)?(k&0x7fffffffu):~k);}
kernel void ink_mtp_topk(const device uchar* weights [[buffer(0)]],const device float* logits [[buffer(1)]],
    device uint* cand_idx [[buffer(2)]],device float* cand_val [[buffer(3)]],constant MtpTopk& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    const device float* lg=logits+size_t(row)*p.stride;
    const device ushort* fb=(const device ushort*)(weights+p.fold_b);
    threadgroup atomic_uint hist[2048];
    threadgroup uint sh[8];
    threadgroup uint bk[TOPK_CAP],bi[TOPK_CAP];
    device uint* oi=cand_idx+size_t(row)*MTP_CAND;
    device float* ov=cand_val+size_t(row)*MTP_CAND;
    // the bin (key bits shifted by sft, width nb) holding the need-th largest of the counted keys -> sh[0] bin, sh[1]
    // the need left inside it
    #define FRT_TOPK_FIND(nb) \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if(tid==0){uint above=0u,bin=0u; \
            for(int b=int(nb)-1;b>=0;--b){const uint c=atomic_load_explicit(&hist[b],memory_order_relaxed); \
                if(above+c>=need){bin=uint(b);break;} above+=c;} sh[0]=bin;sh[1]=need-above;} \
        threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<2048u;i+=1024u)atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
    if(tid<8u)sh[tid]=0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint need=MTP_CAND;
    for(uint v=tid;v<p.vocab;v+=1024u)
        atomic_fetch_add_explicit(&hist[order_key(lg[v]+bfu(fb[v]))>>21],1u,memory_order_relaxed);
    FRT_TOPK_FIND(2048u)
    const uint b1=sh[0];
    need=sh[1];
    const uint inbin=atomic_load_explicit(&hist[b1],memory_order_relaxed);
    threadgroup atomic_uint* cnt=(threadgroup atomic_uint*)(sh+4);
    if(inbin<=TOPK_CAP) {
        // above the bin: candidates; the bin: collected
        for(uint v=tid;v<p.vocab;v+=1024u) {
            const float x=lg[v]+bfu(fb[v]);
            const uint k=order_key(x),b=k>>21;
            if(b>b1){const uint at=atomic_fetch_add_explicit(&cnt[0],1u,memory_order_relaxed);oi[at]=v;ov[at]=x;}
            else if(b==b1){const uint at=atomic_fetch_add_explicit(&cnt[1],1u,memory_order_relaxed);bk[at]=k;bi[at]=v;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nabove=MTP_CAND-need,nb=inbin;
        uint prefix=b1<<21,pmask=0xffe00000u;
        const uint shifts[2]={10u,0u},widths[2]={11u,10u};
        for(uint pass=0;pass<2u;++pass) {
            const uint sft=shifts[pass],nbin=1u<<widths[pass];
            for(uint i=tid;i<nbin;i+=1024u)atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=tid;i<nb;i+=1024u){const uint k=bk[i];
                if((k&pmask)==prefix)atomic_fetch_add_explicit(&hist[(k>>sft)&(nbin-1u)],1u,memory_order_relaxed);}
            FRT_TOPK_FIND(nbin)
            prefix|=sh[0]<<sft;pmask|=(nbin-1u)<<sft;need=sh[1];
        }
        if(tid==0){atomic_store_explicit(&cnt[0],0u,memory_order_relaxed);atomic_store_explicit(&cnt[1],0u,memory_order_relaxed);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint ngreater=MTP_CAND-nabove-need;      // collected values above the K-th value
        for(uint i=tid;i<nb;i+=1024u) {
            const uint k=bk[i];
            if(k>prefix){const uint at=atomic_fetch_add_explicit(&cnt[0],1u,memory_order_relaxed);
                oi[nabove+at]=bi[i];ov[nabove+at]=key_value(k);}
            else if(k==prefix){const uint at=atomic_fetch_add_explicit(&cnt[1],1u,memory_order_relaxed);
                if(at<need){oi[nabove+ngreater+at]=bi[i];ov[nabove+ngreater+at]=key_value(k);}}
        }
        return;
    }
    // fallback: the remaining 21 bits by two more passes over the row
    uint prefix=b1<<21,pmask=0xffe00000u;
    const uint shifts[2]={10u,0u},widths[2]={11u,10u};
    for(uint pass=0;pass<2u;++pass) {
        const uint sft=shifts[pass],nbin=1u<<widths[pass];
        for(uint i=tid;i<nbin;i+=1024u)atomic_store_explicit(&hist[i],0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint v=tid;v<p.vocab;v+=1024u) {
            const uint k=order_key(lg[v]+bfu(fb[v]));
            if((k&pmask)==prefix)atomic_fetch_add_explicit(&hist[(k>>sft)&(nbin-1u)],1u,memory_order_relaxed);
        }
        FRT_TOPK_FIND(nbin)
        prefix|=sh[0]<<sft;pmask|=(nbin-1u)<<sft;need=sh[1];
    }
    #undef FRT_TOPK_FIND
    if(tid==0){atomic_store_explicit(&cnt[0],0u,memory_order_relaxed);atomic_store_explicit(&cnt[1],0u,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nabove=MTP_CAND-need;
    for(uint v=tid;v<p.vocab;v+=1024u) {
        const float x=lg[v]+bfu(fb[v]);
        const uint k=order_key(x);
        if(k>prefix){const uint at=atomic_fetch_add_explicit(&cnt[0],1u,memory_order_relaxed);oi[at]=v;ov[at]=x;}
        else if(k==prefix){const uint at=atomic_fetch_add_explicit(&cnt[1],1u,memory_order_relaxed);
            if(at<need){oi[nabove+at]=v;ov[nabove+at]=x;}}
    }
}
// Markov step on the candidates: bias = fold_w[v] . latent for the K candidates of this step, argmax of
// candidate value + bias (ties: lowest id). grid (seqs), 256 threads (one candidate each).
struct MtpPickCand { ulong fold_w; uint seqs, step, pad0, pad1; };
kernel void ink_mtp_pick_cand(const device uchar* weights [[buffer(0)]],const device uint* cand_idx [[buffer(1)]],
    const device float* cand_val [[buffer(2)]],const device ushort* latent [[buffer(3)]],device uint* prev [[buffer(4)]],
    device uint* drafts [[buffer(5)]],constant MtpPickCand& p [[buffer(6)]],device float* top [[buffer(7)]],
    uint s [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(s>=p.seqs)return;
    threadgroup float lt[256];
    lt[tid]=bfu(latent[size_t(s)*256u+tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t c=(size_t(p.step)*p.seqs+s)*MTP_CAND+tid;
    const uint v=cand_idx[c];
    const device uint4* fw=(const device uint4*)((const device ushort*)(weights+p.fold_w)+size_t(v)*256u);
    float a0=0.0f,a1=0.0f;
    for(uint i=0;i<32u;++i) {
        const uint4 q=fw[i];
        a0=metal::fma(lo_f(q.x),lt[8u*i],a0);a1=metal::fma(hi_f(q.x),lt[8u*i+1u],a1);
        a0=metal::fma(lo_f(q.y),lt[8u*i+2u],a0);a1=metal::fma(hi_f(q.y),lt[8u*i+3u],a1);
        a0=metal::fma(lo_f(q.z),lt[8u*i+4u],a0);a1=metal::fma(hi_f(q.z),lt[8u*i+5u],a1);
        a0=metal::fma(lo_f(q.w),lt[8u*i+6u],a0);a1=metal::fma(hi_f(q.w),lt[8u*i+7u],a1);
    }
    const float score=cand_val[c]+(a0+a1);
    threadgroup float vb[8];threadgroup uint vi[8];threadgroup float red[2];
    // the 4 largest scores (ties: lowest id) and the log-sum-exp over the candidates
    float m=0.0f;bool taken=false;
    device float* t=top+(size_t(p.step)*p.seqs+s)*16u;
    for(uint k=0;k<4u;++k) {
        float best=taken?-INFINITY:score;uint bi=taken?0xffffffffu:v;
        for(ushort o=16;o>=1;o>>=1) {
            const float ob=simd_shuffle_down(best,o);const uint oi=simd_shuffle_down(bi,o);
            if(ob>best||(ob==best&&oi<bi)){best=ob;bi=oi;}
        }
        if(lane==0u){vb[simd]=best;vi[simd]=bi;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        best=vb[0];bi=vi[0];
        for(uint j=1;j<8u;++j)if(vb[j]>best||(vb[j]==best&&vi[j]<bi)){best=vb[j];bi=vi[j];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(k==0u){m=best;if(tid==0u){prev[s]=bi;drafts[s*7u+p.step]=bi;}}
        if(tid==0u){t[k]=as_type<float>(bi);t[4u+k]=best;}
        if(bi==v)taken=true;
    }
    float e=metal::exp(score-m);
    e=simd_sum(e);
    if(lane==0u)vb[simd]=e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u){float z=0.0f;for(uint j=0;j<8u;++j)z+=vb[j];t[8]=m+metal::log(z);}
}
)MSL";
}
}  // namespace flashrt::apple::shaders
