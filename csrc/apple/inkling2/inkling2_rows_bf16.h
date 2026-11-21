#pragma once
#include "common/source.h"
#include "inkling2/inkling2_common.h"
#include <string>

// Row kernels of the Inkling 2 Mini programs that process several rows per layer visit: a prefill chunk (rows =
// consecutive positions of one sequence) or a batched decode step (rows = sequences, one position each). The
// per-row work (block attention residuals, norms, clips, convolutions, routing, combination) uses the decode
// step's device functions with one 256-thread threadgroup per row; the products run in the TensorOps kernels of
// inkling2_rows_gemm.h.
//
// Row metadata rows_info[r] = (sequence, position, first, last): first / last are the first and last position of
// the row's sequence in this dispatch (a decode row: both its own position). Conv inputs at positions in
// [first, position) come from earlier rows of the dispatch (row r - lag), older ones from the sequence's ring.
// Caches per sequence and visit: global layers K/V [8][cap][128] (slot = position); sliding layers a 1024-slot
// ring (slot = position mod 1024). The dispatch's own sliding keys and values are read from `chunk` ([2][8][rows]
// [128]) until ink_ring_rows copies the window's last 1024 positions into the ring.
//
// Buffer layouts ([rows] = the dispatch's row count R): h, prelude, y, o, moe BF16 [R][2304]; x BF16 [3][R][2304];
// bs FP32 [6][R][2304]; bank FP32 [2][R][2304]; ain FP32 [4][R][2304] (q, k, v, mlp gate inputs); s BF16
// [4][R][4608]; msg FP32 [R][4][2304]; forget, gain FP32 [4][R][2304]; proj BF16 [R][8672]; qc BF16 [R][3072];
// rclip BF16 [R][384]; bias BF16 [R][24][1024]; attn BF16 [R][3072]; resid FP32 [R][2304]; rlog FP32 [R][128];
// ids [R][8], rw FP32 [R][8], slot_of [R][8].

namespace flashrt::apple::shaders {

inline std::string inkling2_rows_bf16_source() {
    return std::string(metal_preamble) + inkling2_common_msl() + R"MSL(
struct RowsVisit {
    ulong kv_off;         // element offset of the visit's K/V inside one sequence's cache region
    ulong seq_kv;         // cache elements per sequence
    ulong ring_off;       // element offset of the visit's conv ring inside one sequence's ring region
    ulong seq_ring;       // ring elements per sequence
    uint layer, pass, cap, rows;
    uint nblocks, bslot, slot0, slot1, slot2, slot3;
    uint flags;           // bit 0 global, bit 1 messages, bit 2 MXFP8, bit 3 block boundary
    uint pad;
};
constant constexpr uint FPROJ=8672u;
inline uint4 rinfo(const device uint* ri,uint r){return *((const device uint4*)(ri+r*4u));}
// The input of a conv at position pos - lag (lag 1..3): an earlier row of this dispatch, the ring, or zero.
inline float conv_hist(const device ushort* rowsrc,uint row_stride,uint row,uint4 inf,uint lag,
                       const device ushort* ring,uint channel,uint ring_channel) {
    const int q=int(inf.y)-int(lag);
    if(q<0)return 0.0f;
    if(uint(q)>=inf.z)return bfu(rowsrc[size_t(row-lag)*row_stride+channel]);
    return bfu(ring[(uint(q)&3u)*FRING+ring_channel]);
}

// ---- embedding + embed_norm
kernel void ink_embed_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device uint* tokens [[buffer(2)]],device ushort* h [[buffer(3)]],constant RowsVisit& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float red[256];
    threadgroup ushort vh[FH];
    const device ulong* f=table+24u*64u;
    const device ushort* e=(const device ushort*)(weights+f[D_EMBED])+size_t(tokens[row])*FH;
    for(uint i=tid;i<FH;i+=256u)vh[i]=e[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
    const device ushort* w=(const device ushort*)(weights+f[D_EMBED_NORM]);
    for(uint i=tid;i<FH;i+=256u)h[size_t(row)*FH+i]=ubf(bfu(w[i])*rb(bfu(vh[i])*inv));
}

// ---- attention inputs: unit-RMS state (a block state at a boundary), depth attention, input norms -> x; with
// messages the gate inputs ain, s and the four messages of the looped layer.
kernel void ink_attn_inputs_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    device ushort* h [[buffer(2)]],device float* bs [[buffer(3)]],device ushort* x [[buffer(4)]],
    device float* ain [[buffer(5)]],device ushort* s [[buffer(6)]],device float* msg [[buffer(7)]],
    const device float* bank [[buffer(8)]],device ushort* prelude [[buffer(9)]],
    constant RowsVisit& p [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    const uint zero=0u;
    threadgroup float red[256];
    threadgroup ushort vh[FH];
    threadgroup float vr[FH];
    const device ulong* d=table+p.layer*64u;
    const bool full=(p.flags&1u)!=0u,messages=(p.flags&2u)!=0u,boundary=(p.flags&8u)!=0u;
    device ushort* hr=h+size_t(row)*FH;
    if(p.layer==6u && p.pass>0u)for(uint i=tid;i<FH;i+=256u)vh[i]=prelude[size_t(row)*FH+i];
    else for(uint i=tid;i<FH;i+=256u)vh[i]=hr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(p.layer==6u)for(uint i=tid;i<FH;i+=256u){if(p.pass==0u)prelude[size_t(row)*FH+i]=vh[i];else hr[i]=vh[i];}
    const float rinv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
    device float* bsr=bs+size_t(row)*FH;
    if(boundary)for(uint i=tid;i<FH;i+=256u)bsr[size_t(p.bslot)*R*FH+i]=bfu(vh[i])*rinv;
    const uint vslots[4]={p.slot0*R,p.slot1*R,p.slot2*R,p.slot3*R};
    float dw[15];
    const device float* aq=(const device float*)(weights+d[D_ATTN_Q]);
    if(p.layer>0u) {
        if(full)depth_weights_k<3>(bsr,vslots,p.nblocks,(const threadgroup ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
        else depth_weights_k<1>(bsr,vslots,p.nblocks,(const threadgroup ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
    }
    device float* mr=msg+size_t(row)*4u*FH;
    if(messages) {
        const uint bk[1]={(2u+p.pass-1u)*R};
        depth_attention<4>(bsr,bk,1u,(const device float*)(bank+(size_t(p.pass-1u)*R+row)*FH),1.0f,
                           (const device float*)(weights+table[24u*64u+D_MSG_Q])+(p.layer-6u)*4u*FH,
                           mr,red,tid,lane,zero);
    }
    const uint nq=(full || messages)?3u:1u;
    for(uint j=0;j<nq;++j) {
        const device ushort* w=(const device ushort*)(weights+d[D_IN_NORM+(full?j:0u)]);
        device ushort* xj=x+(size_t(j)*R+row)*FH;
        if(p.layer==0u) {
            for(uint i=tid;i<FH;i+=256u)xj[i]=ubf(bfu(w[i])*rb(bfu(vh[i])*rinv));
            continue;
        }
        if(full || j==0u) {
            if(full)depth_output_k<3>(bsr,vslots,p.nblocks,(const threadgroup ushort*)vh,rinv,dw,j,vr,tid,zero);
            else depth_output_k<1>(bsr,vslots,p.nblocks,(const threadgroup ushort*)vh,rinv,dw,0u,vr,tid,zero);
        }
        if(!messages) {
            const float inv=rms_inv((const threadgroup float*)vr,red,lane,simd);
            for(uint i=tid;i<FH;i+=256u)xj[i]=ubf(bfu(w[i])*(vr[i]*inv));
        } else {
            const device float* m=mr+j*FH;
            device float* a=ain+(size_t(j)*R+row)*FH;
            for(uint i=tid;i<FH;i+=256u)a[i]=vr[i];
            const float ia=rms_inv((const threadgroup float*)vr,red,lane,simd);
            const float im=rms_inv(m,red,lane,simd);
            device ushort* sr=s+(size_t(j)*R+row)*(2u*FH);
            for(uint i=tid;i<FH;i+=256u){sr[i]=ubf(vr[i]*ia);sr[FH+i]=ubf(m[i]*im);}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!full && !messages) {
        threadgroup_barrier(mem_flags::mem_device);
        for(uint i=tid;i<FH;i+=256u) {
            const ushort v=x[size_t(row)*FH+i];
            x[(size_t(R)+row)*FH+i]=v;x[(2u*size_t(R)+row)*FH+i]=v;
        }
    }
}

// ---- message gate combination: (1 - tanh(f)) * input + tanh(g) * message, then the norms (attention: the q / k /
// v input norms -> x; MLP: post_attention_layernorm -> y). mode 0 attention (3 gates), 1 MLP (gate 3).
kernel void ink_gate_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device float* ain [[buffer(2)]],const device float* msg [[buffer(3)]],const device float* forget [[buffer(4)]],
    const device float* gain [[buffer(5)]],device float* gated [[buffer(6)]],device ushort* out [[buffer(7)]],
    constant RowsVisit& p [[buffer(8)]],constant uint& mode [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    threadgroup float red[256];
    threadgroup float vr[FH];
    const device ulong* d=table+p.layer*64u;
    const bool full=(p.flags&1u)!=0u;
    const uint g0=mode==0u?0u:3u,g1=mode==0u?3u:4u;
    for(uint g=g0;g<g1;++g) {
        const size_t o=(size_t(g)*R+row)*FH;
        const device float* m=msg+size_t(row)*4u*FH+g*FH;
        for(uint i=tid;i<FH;i+=256u)vr[i]=gate_combine(forget[o+i],gain[o+i],ain[o+i],m[i],0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv=rms_inv((const threadgroup float*)vr,red,lane,simd);
        const device ushort* w=(const device ushort*)(weights+d[mode==0u?D_IN_NORM+(full?g:0u):D_POST_NORM]);
        device ushort* dst=out+(size_t(mode==0u?g:0u)*R+row)*FH;
        for(uint i=tid;i<FH;i+=256u)dst[i]=ubf(bfu(w[i])*(vr[i]*inv));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- per head: clipped queries / relative states (x tau in global layers); keys and values through their conv and
// clip / norm into the cache (global) or the chunk buffer (sliding).
kernel void ink_heads_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device ushort* proj [[buffer(2)]],device ushort* qc [[buffer(3)]],device ushort* rclip [[buffer(4)]],
    device ushort* kv [[buffer(5)]],device ushort* chunk [[buffer(6)]],const device ushort* rings [[buffer(7)]],
    constant RowsVisit& p [[buffer(8)]],const device uint* rows_info [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    const uint zero=0u;
    const device ulong* d=table+p.layer*64u;
    const bool full=(p.flags&1u)!=0u;
    const uint4 inf=rinfo(rows_info,row);
    const uint position=inf.y;
    const device ushort* pr=proj+size_t(row)*FPROJ;
    const device ushort* ring=rings+inf.x*p.seq_ring+p.ring_off;
    float tau=1.0f;
    if(full)tau=1.0f+sep(0.1f*metal::precise::log(max(fdiv(float(position)+1.0f,8192.0f),1.0f)),zero);
    for(uint task=simd;task<64u;task+=8u) {
        if(task<24u) {
            const uint hh=task;
            const device ushort* sc=(const device ushort*)(weights+d[D_QCLIP])+hh*128u;
            float sv[4],sq=0.0f;
            for(uint t=0;t<4;++t)sv[t]=bfu(pr[P_Q+hh*128u+lane*4u+t])*bfu(sc[lane*4u+t]);
            for(uint t=0;t<4;++t)sq=sep(sv[t]*sv[t],zero)+sq;
            sq=simd_sum(sq);
            const float k=min(fdiv(22.627416997969522f,metal::precise::sqrt(sq)+1e-06f),1.0f);
            for(uint t=0;t<4;++t){float v=rb(sv[t]*k);if(full)v=rb(v*tau);qc[size_t(row)*FQ+hh*128u+lane*4u+t]=ubf(v);}
        } else if(task<48u) {
            const uint hh=task-24u;
            const float sv=lane<16u?bfu(pr[P_R+hh*16u+lane])*bfu(((const device ushort*)(weights+d[D_RCLIP]))[hh*16u+lane]):0.0f;
            const float sqv=sep(sv*sv,zero);
            float tot=0.0f;
            for(uint t=0;t<16u;++t)tot=simd_shuffle(sqv,ushort(t))+tot;
            const float k=min(fdiv(8.0f,metal::precise::sqrt(tot)+1e-06f),1.0f);
            if(lane<16u){float v=rb(sv*k);if(full)v=rb(v*tau);rclip[size_t(row)*384u+hh*16u+lane]=ubf(v);}
        } else {
            const uint idx=task-48u,hh=idx%8u;
            const bool isk=idx<8u;
            const uint base=(isk?0u:FKV)+hh*128u;
            const uint col=(isk?P_K:P_V)+hh*128u;
            float outv[4];
            for(uint t=0;t<4;++t) {
                const uint c=hh*128u+lane*4u+t;
                const float xt=bfu(pr[col+lane*4u+t]);
                float prev[3];
                // proj columns P_K + channel hold the k / v conv inputs (ring channel = column - P_K)
                for(uint i=0;i<3;++i)prev[i]=conv_hist(proj+P_K,FPROJ,row,inf,3u-i,ring,base+lane*4u+t,base+lane*4u+t);
                if(!full) {
                    const device ushort* cw=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                    outv[t]=short_conv(prev,xt,cw+c*4u);
                } else {
                    const device ushort* st=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                    const device ushort* dyn=pr+(isk?P_KD:P_VD)+hh*4u;
                    float sum=0.0f;
                    for(uint lag=0;lag<4;++lag) {
                        const float wgt=bfu(st[lag*FKV+c])+bfu(dyn[lag]);
                        const float xi=lag==0u?xt:prev[3u-lag];
                        sum=sep(xi*wgt,zero)+sum;
                    }
                    outv[t]=rb(fdiv(sum+xt,3.0f));
                }
            }
            float y[4];
            if(isk) {
                float sq=0.0f;
                for(uint t=0;t<4;++t)sq=sep(outv[t]*outv[t],zero)+sq;
                sq=simd_sum(sq);
                const float k=min(fdiv(22.627416997969522f,metal::precise::sqrt(sq)+1e-06f),1.0f);
                for(uint t=0;t<4;++t)y[t]=rb(outv[t]*k);
            } else {
                float acc=0.0f;
                for(uint t=0;t<4;++t)acc=metal::fma(outv[t],outv[t],acc);
                acc=simd_sum(acc);
                acc=simd_sum(lane==0?acc:0.0f);
                const float inv=metal::precise::rsqrt(fdiv(acc,128.0f)+FEPS);
                for(uint t=0;t<4;++t)y[t]=rb(outv[t]*inv);
            }
            device ushort* dst;
            if(full)dst=kv+inf.x*p.seq_kv+p.kv_off+(isk?0ul:ulong(8u)*p.cap*128u)+(size_t(hh)*p.cap+position)*128u;
            else dst=chunk+((isk?0u:8u)+hh)*size_t(R)*128u+size_t(row)*128u;
            for(uint t=0;t<4;++t)dst[lane*4u+t]=ubf(y[t]);
        }
    }
}

// ---- relative-position bias [R][24][1024] = rclip [R][24][16] @ proj [16][1024] (8x8 MMAs, two k steps)
kernel void ink_bias_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device ushort* rclip [[buffer(2)]],device ushort* bias [[buffer(3)]],constant RowsVisit& p [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row=g.y;
    if(row>=p.rows)return;
    const device ushort* rel=(const device ushort*)(weights+table[p.layer*64u+D_REL]);
    const ushort fm=((lane/4u)&4u)+(lane/2u)%4u;
    const ushort fn=((lane/4u)&2u)*2u+(lane%2u)*2u;
    const device ushort* rc=rclip+size_t(row)*384u;
    device ushort* b=bias+size_t(row)*24u*FWIN;
    for(uint tile=g.x*8u+simd;tile<384u;tile+=8u*16u) {
        const uint ht=tile/128u,ct=tile%128u;
        simdgroup_float8x8 c=simdgroup_float8x8(0.0f);
        for(uint kk=0;kk<16u;kk+=8u) {
            simdgroup_float8x8 a,bb;
            a.thread_elements()[0]=bfu(rc[(ht*8u+fm)*16u+kk+fn]);
            a.thread_elements()[1]=bfu(rc[(ht*8u+fm)*16u+kk+fn+1u]);
            bb.thread_elements()[0]=bfu(rel[(kk+fm)*FWIN+ct*8u+fn]);
            bb.thread_elements()[1]=bfu(rel[(kk+fm)*FWIN+ct*8u+fn+1u]);
            simdgroup_multiply_accumulate(c,a,bb,c);
        }
        b[(ht*8u+fm)*FWIN+ct*8u+fn]=ubf(c.thread_elements()[0]);
        b[(ht*8u+fm)*FWIN+ct*8u+fn+1u]=ubf(c.thread_elements()[1]);
    }
}

// Key position kp of the row's sequence: the dispatch's chunk (kp >= first), the global cache or the sliding ring.
inline const device ushort* key_ptr(const device ushort* kv,const device ushort* chunk,constant RowsVisit& p,
                                    uint4 inf,uint row,uint kvh,uint kp,bool value) {
    const bool full=(p.flags&1u)!=0u;
    if(!full && kp>=inf.z)return chunk+((value?8u:0u)+kvh)*size_t(p.rows)*128u+size_t(row-(inf.y-kp))*128u;
    const device ushort* base=kv+inf.x*p.seq_kv+p.kv_off+(value?ulong(8u)*p.cap*128u:0ul);
    return base+(size_t(kvh)*p.cap+(full?kp:(kp&(FWIN-1u))))*128u;
}

// ---- decode attention: one threadgroup per (row, KV head); the 8 SIMD groups split the visible keys, each keeps
// an online softmax (FP32, sink logit as the initial maximum of partition 0, bias as an additive term) for the 3
// query heads; the partitions merge in threadgroup memory; output BF16 times sigmoid(gate).
kernel void ink_attn_decode_rows(const device ushort* proj [[buffer(0)]],const device ushort* qc [[buffer(1)]],
    const device ushort* bias [[buffer(2)]],const device ushort* kv [[buffer(3)]],const device ushort* chunk [[buffer(4)]],
    device ushort* attn [[buffer(5)]],constant RowsVisit& p [[buffer(6)]],const device uint* rows_info [[buffer(7)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
    const uint row=g.y,kvh=g.x;
    if(row>=p.rows)return;
    const bool full=(p.flags&1u)!=0u;
    const uint4 inf=rinfo(rows_info,row);
    const uint position=inf.y;
    const uint count=full?position+1u:min(position+1u,FWIN);
    const uint k0=position+1u-count;
    const float scale=FHEAD_SCALE;
    threadgroup float om[8][3][128];
    threadgroup float st[8][3][2];
    float q[3][4],o[3][4],mx[3],z[3];
    for(uint hh=0;hh<3u;++hh) {
        const uint h=kvh*3u+hh;
        for(uint t=0;t<4;++t){q[hh][t]=scale*bfu(qc[size_t(row)*FQ+h*128u+lane*4u+t]);o[hh][t]=0.0f;}
        mx[hh]=FRT_MIN;z[hh]=0.0f;
        if(simd==0){mx[hh]=bfu(proj[size_t(row)*FPROJ+P_S+h]);z[hh]=1.0f;}
    }
    for(uint i=simd;i<count;i+=8u) {
        const uint kp=k0+i;
        const uint2 kr=*((const device uint2*)(key_ptr(kv,chunk,p,inf,row,kvh,kp,false)+lane*4u));
        const uint2 vv=*((const device uint2*)(key_ptr(kv,chunk,p,inf,row,kvh,kp,true)+lane*4u));
        const float k[4]={lo_f(kr.x),hi_f(kr.x),lo_f(kr.y),hi_f(kr.y)};
        const float v[4]={lo_f(vv.x),hi_f(vv.x),lo_f(vv.y),hi_f(vv.y)};
        const uint dist=position-kp;
        for(uint hh=0;hh<3u;++hh) {
            const uint h=kvh*3u+hh;
            float score=0.0f;
            for(uint t=0;t<4;++t)score=metal::fma(q[hh][t],k[t],score);
            score=simd_sum(score);
            if(dist<FWIN)score+=bfu(bias[(size_t(row)*24u+h)*FWIN+dist]);
            const float nm=max(mx[hh],score);
            const float factor=metal::fast::exp(mx[hh]-nm),e=metal::fast::exp(score-nm);
            mx[hh]=nm;z[hh]=metal::fma(z[hh],factor,e);
            for(uint t=0;t<4;++t)o[hh][t]=metal::fma(o[hh][t],factor,e*v[t]);
        }
    }
    for(uint hh=0;hh<3u;++hh) {
        for(uint t=0;t<4;++t)om[simd][hh][lane*4u+t]=o[hh][t];
        if(lane==0){st[simd][hh][0]=mx[hh];st[simd][hh][1]=z[hh];}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=tid;e<3u*128u;e+=256u) {
        const uint hh=e/128u,dd=e%128u,h=kvh*3u+hh;
        float top=FRT_MIN;
        for(uint s=0;s<8u;++s)top=max(top,st[s][hh][0]);
        float zs=0.0f,acc=0.0f;
        for(uint s=0;s<8u;++s){const float f=metal::fast::exp(st[s][hh][0]-top);zs=metal::fma(f,st[s][hh][1],zs);acc=metal::fma(f,om[s][hh][dd],acc);}
        const float val=rb(fdiv(acc,zs));
        attn[size_t(row)*FQ+h*128u+dd]=ubf(val*sigmoid_f32(bfu(proj[size_t(row)*FPROJ+P_G+h*128u+dd])));
    }
}

// ---- post-attention: conv of the output projection, residual (restarts at a block boundary), unit RMS, MLP
// depth attention; without messages the post-attention norm -> y; with messages the MLP gate inputs.
kernel void ink_post_attn_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device ushort* o [[buffer(2)]],const device ushort* h [[buffer(3)]],device float* bs [[buffer(4)]],
    device float* resid [[buffer(5)]],device ushort* y [[buffer(6)]],device float* ain [[buffer(7)]],
    device ushort* s [[buffer(8)]],const device float* msg [[buffer(9)]],const device ushort* rings [[buffer(10)]],
    constant RowsVisit& p [[buffer(11)]],const device uint* rows_info [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    const uint zero=0u;
    threadgroup float red[256];
    threadgroup float vr[FH];
    threadgroup float mi[FH];
    const device ulong* d=table+p.layer*64u;
    const bool messages=(p.flags&2u)!=0u,boundary=(p.flags&8u)!=0u;
    const uint4 inf=rinfo(rows_info,row);
    const device ushort* ring=rings+inf.x*p.seq_ring+p.ring_off;
    const device ushort* cw=(const device ushort*)(weights+d[D_ACONV]);
    for(uint i=tid;i<FH;i+=256u) {
        float prev[3];
        for(uint j=0;j<3;++j)prev[j]=conv_hist(o,FH,row,inf,3u-j,ring,i,2048u+i);
        const float c=short_conv(prev,bfu(o[size_t(row)*FH+i]),cw+i*4u);
        vr[i]=boundary?c:bfu(h[size_t(row)*FH+i])+c;
        resid[size_t(row)*FH+i]=vr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nb=p.nblocks+(boundary?1u:0u);
    const uint vslots[4]={p.slot0*R,p.slot1*R,p.slot2*R,p.slot3*R};
    uint sl[4]={vslots[0],vslots[1],vslots[2],vslots[3]};
    if(boundary)sl[p.nblocks]=p.bslot*R;
    device float* bsr=bs+size_t(row)*FH;
    const float inv=rms_inv((const threadgroup float*)vr,red,lane,simd);
    depth_attention<1>(bsr,sl,nb,(const threadgroup float*)vr,inv,(const device float*)(weights+d[D_MLP_Q]),
                       mi,red,tid,lane,zero);
    if(!messages) {
        const float i2=rms_inv((const threadgroup float*)mi,red,lane,simd);
        const device ushort* w=(const device ushort*)(weights+d[D_POST_NORM]);
        for(uint i=tid;i<FH;i+=256u)y[size_t(row)*FH+i]=ubf(bfu(w[i])*(mi[i]*i2));
    } else {
        const device float* m=msg+size_t(row)*4u*FH+3u*FH;
        device float* a=ain+(3u*size_t(R)+row)*FH;
        for(uint i=tid;i<FH;i+=256u)a[i]=mi[i];
        const float ia=rms_inv((const threadgroup float*)mi,red,lane,simd);
        const float im=rms_inv(m,red,lane,simd);
        device ushort* sr=s+(3u*size_t(R)+row)*(2u*FH);
        for(uint i=tid;i<FH;i+=256u){sr[i]=ubf(mi[i]*ia);sr[FH+i]=ubf(m[i]*im);}
    }
}

// ---- routing of every row (one SIMD group per row) and the expert counts
kernel void ink_route_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device float* rlog [[buffer(2)]],device uint* ids [[buffer(3)]],device float* rw [[buffer(4)]],
    device atomic_uint* counts [[buffer(5)]],constant RowsVisit& p [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row=gid*8u+simd;
    if(row>=p.rows)return;
    const device float* rbias=(const device float*)(weights+table[p.layer*64u+D_RBIAS]);
    float score[4],biased[4];
    for(uint i=0;i<4;++i) {
        const uint e=lane+32u*i;
        const float x=rlog[size_t(row)*128u+e];
        const float yv=fdiv(1.0f,1.0f+metal::precise::exp(metal::abs(x)));
        score[i]=x<0.0f?yv:1.0f-yv;
        biased[i]=score[i]+rbias[e];
    }
    float wk[8];uint idk[8];
    for(uint k=0;k<8;++k) {
        float best=-FRT_INF;uint best_e=128u;
        for(uint i=0;i<4;++i){const uint e=lane+32u*i;if(biased[i]>best || (biased[i]==best && e<best_e)){best=biased[i];best_e=e;}}
        const float top=simd_max(best);
        const uint winner=simd_min(best==top?best_e:128u);
        float sv=0.0f;
        for(uint i=0;i<4;++i)if(lane+32u*i==winner){sv=score[i];biased[i]=-FRT_INF;}
        wk[k]=simd_sum(sv);idk[k]=winner;
    }
    float total=0.0f;
    for(uint k=0;k<8;++k)total=wk[k]+total;
    if(lane<8u) {
        ids[row*8u+lane]=idk[lane];
        rw[row*8u+lane]=fdiv(wk[lane],total+1e-20f)*8.0f;
        atomic_fetch_add_explicit(counts+idk[lane],1u,memory_order_relaxed);
    }
}
// expert starts (prefix sums) and the cursors reset; counts cleared for the next visit
// (tile_starts: prefix sums of ceil(count / tm), the expert GEMM's row tiles)
kernel void ink_expert_scan(device uint* counts [[buffer(0)]],device uint* starts [[buffer(1)]],
    device uint* cursors [[buffer(2)]],device uint* tile_starts [[buffer(3)]],constant uint& tm [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint c[4],tl[4];
    for(uint i=0;i<4;++i){c[i]=counts[lane*4u+i];tl[i]=(c[i]+tm-1u)/tm;}
    uint before=simd_prefix_exclusive_sum(c[0]+c[1]+c[2]+c[3]);
    uint tbefore=simd_prefix_exclusive_sum(tl[0]+tl[1]+tl[2]+tl[3]);
    for(uint i=0;i<4;++i) {
        starts[lane*4u+i]=before;tile_starts[lane*4u+i]=tbefore;cursors[lane*4u+i]=0u;counts[lane*4u+i]=0u;
        before+=c[i];tbefore+=tl[i];
    }
    if(lane==31u){starts[128]=before;tile_starts[128]=tbefore;}
}
// place every (row, rank) assignment into its expert's block and copy the row's MLP input there
kernel void ink_expert_place_rows(const device uint* ids [[buffer(0)]],const device uint* starts [[buffer(1)]],
    device atomic_uint* cursors [[buffer(2)]],device uint* slot_of [[buffer(3)]],const device ushort* y [[buffer(4)]],
    device ushort* xp [[buffer(5)]],constant RowsVisit& p [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    const uint a=gid;
    if(a>=p.rows*8u)return;
    threadgroup uint slot;
    if(tid==0){const uint e=ids[a];slot=starts[e]+atomic_fetch_add_explicit(cursors+e,1u,memory_order_relaxed);slot_of[a]=slot;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row=a/8u;
    for(uint i=tid;i<FH/4u;i+=256u)((device uint2*)(xp+size_t(slot)*FH))[i]=((const device uint2*)(y+size_t(row)*FH))[i];
}

// ---- combination (products rounded, summed in rank order) in one dispatch (RowsVisit.pad = 0), then in a second
// dispatch (pad = 1) the post-MLP conv, which reads earlier rows' outputs, the residual and the unit-RMS bank state
// at the end of a pass
kernel void ink_combine_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device ushort* ep [[buffer(2)]],const device uint* slot_of [[buffer(3)]],const device float* rw [[buffer(4)]],
    const device float* resid [[buffer(5)]],device ushort* moe [[buffer(6)]],device ushort* h [[buffer(7)]],
    device float* bank [[buffer(8)]],const device ushort* rings [[buffer(9)]],constant RowsVisit& p [[buffer(10)]],
    const device uint* rows_info [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    const uint zero=0u;
    threadgroup float red[256];
    threadgroup ushort vh[FH];
    const uint4 inf=rinfo(rows_info,row);
    float rwk[8];uint sk[8];
    for(uint k=0;k<8;++k){rwk[k]=rw[row*8u+k];sk[k]=slot_of[row*8u+k];}
    if(p.pad==0u) {
        // pass 1 (its own dispatch): the combined MoE output of every row
        for(uint i=tid;i<FH;i+=256u) {
            float sum=0.0f;
            for(uint k=0;k<8;++k)sum=sep(bfu(ep[size_t(sk[k])*FH+i])*rwk[k],zero)+sum;
            moe[size_t(row)*FH+i]=ubf(sum);
        }
        return;
    }
    // pass 2: post-MLP conv over the rows' MoE outputs, residual
    const device ushort* ring=rings+inf.x*p.seq_ring+p.ring_off;
    const device ushort* cw=(const device ushort*)(weights+table[p.layer*64u+D_MCONV]);
    for(uint i=tid;i<FH;i+=256u) {
        float prev[3];
        for(uint j=0;j<3;++j)prev[j]=conv_hist(moe,FH,row,inf,3u-j,ring,i,4352u+i);
        const float c=short_conv(prev,bfu(moe[size_t(row)*FH+i]),cw+i*4u);
        vh[i]=ubf(resid[size_t(row)*FH+i]+c);
        h[size_t(row)*FH+i]=vh[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(p.layer==17u && p.pass<2u) {
        const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
        for(uint i=tid;i<FH;i+=256u)bank[(size_t(p.pass)*R+row)*FH+i]=bfu(vh[i])*inv;
    }
}

// ---- ring update after a visit: the conv inputs of each sequence's last three positions (k, v pre-conv, the
// output projection, the MoE output) and, in sliding layers, the window's keys / values of the dispatch
kernel void ink_ring_rows(const device ushort* proj [[buffer(0)]],const device ushort* o [[buffer(1)]],
    const device ushort* moe [[buffer(2)]],const device ushort* chunk [[buffer(3)]],device ushort* rings [[buffer(4)]],
    device ushort* kv [[buffer(5)]],constant RowsVisit& p [[buffer(6)]],const device uint* rows_info [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    const uint R=p.rows;
    if(row>=R)return;
    const uint4 inf=rinfo(rows_info,row);
    const uint position=inf.y;
    if(position+3u>=inf.w) {
        device ushort* ring=rings+inf.x*p.seq_ring+p.ring_off+(position&3u)*FRING;
        for(uint i=tid;i<FKV;i+=256u){ring[i]=proj[size_t(row)*FPROJ+P_K+i];ring[FKV+i]=proj[size_t(row)*FPROJ+P_V+i];}
        for(uint i=tid;i<FH;i+=256u){ring[2048u+i]=o[size_t(row)*FH+i];ring[4352u+i]=moe[size_t(row)*FH+i];}
    }
    if((p.flags&1u)==0u && position+FWIN>inf.w) {
        device ushort* base=kv+inf.x*p.seq_kv+p.kv_off;
        for(uint i=tid;i<2u*8u*128u;i+=256u) {
            const uint v=i/(8u*128u),hh=(i/128u)%8u,dd=i%128u;
            base[(v?ulong(8u)*p.cap*128u:0ul)+(size_t(hh)*p.cap+(position&(FWIN-1u)))*128u+dd]=
                chunk[((v?8u:0u)+hh)*size_t(R)*128u+size_t(row)*128u+dd];
        }
    }
}

// ---- final input and norm of the rows that produce logits
kernel void ink_final_rows(const device uchar* weights [[buffer(0)]],const device ulong* table [[buffer(1)]],
    const device ushort* h [[buffer(2)]],const device float* bs [[buffer(3)]],device ushort* xf [[buffer(4)]],
    constant RowsVisit& p [[buffer(5)]],constant uint& first_row [[buffer(6)]],
    uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint R=p.rows,row=first_row+g;
    if(row>=R)return;
    const uint zero=0u;
    threadgroup float red[256];
    threadgroup ushort vh[FH];
    threadgroup float vr[FH];
    for(uint i=tid;i<FH;i+=256u)vh[i]=h[size_t(row)*FH+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sl[4]={0u,1u*R,4u*R,5u*R};
    const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
    depth_attention<1>(bs+size_t(row)*FH,sl,4u,(const threadgroup ushort*)vh,inv,
                       (const device float*)(weights+table[24u*64u+D_FINAL_Q]),vr,red,tid,lane,zero);
    const float i2=rms_inv((const threadgroup float*)vr,red,lane,simd);
    const device ushort* w=(const device ushort*)(weights+table[24u*64u+D_FINAL_NORM]);
    for(uint i=tid;i<FH;i+=256u)xf[size_t(g)*FH+i]=ubf(bfu(w[i])*(vr[i]*i2));
}
// ---- multi-row products for few rows (decode batches): every lane reads its weights once and multiplies them
// with each row's inputs, so each row's arithmetic is the single-row MLX gemv / fp_qmv of the decode step.
// Task table (uint4 per matrix): (weight byte offset lo, hi, rows, column in y) with x / y selectors packed in
// RowsTasks; a SIMD group computes 4 output rows of one matrix for every input row (MAXR rows per pass).
constant constexpr uint MAXR=16u;
struct RowsMatrix { ulong w; uint rows, k, x_sel, y_col, y_stride, mode; };  // mode 0 gemv4, 1 bn8, 3 bn8 FP32 out
struct RowsTasks { uint count, rows, pad0, pad1; RowsMatrix m[10]; };
kernel void ink_gemv_rows(const device uchar* weights [[buffer(0)]],const device ushort* x0 [[buffer(1)]],
    const device ushort* x1 [[buffer(2)]],const device ushort* x2 [[buffer(3)]],device ushort* y [[buffer(4)]],
    device float* yf [[buffer(5)]],constant RowsTasks& p [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint task=gid*8u+simd;
    uint mi=0;
    while(mi<p.count){const uint units=p.m[mi].mode==0u?p.m[mi].rows/4u:p.m[mi].rows;
        if(task<units)break;task-=units;++mi;}
    if(mi>=p.count)return;
    const RowsMatrix m=p.m[mi];
    const uint R=p.rows,K=m.k;
    const device ushort* x=m.x_sel==0u?x0:(m.x_sel==1u?x1:x2);
    const device ushort* w=(const device ushort*)(weights+m.w);
    for(uint r0=0;r0<R;r0+=MAXR) {
        const uint nr=min(MAXR,R-r0);
        if(m.mode==0u) {
            const uint row0=task*4u;
            float acc[4][MAXR];
            for(uint tm=0;tm<4;++tm)for(uint r=0;r<MAXR;++r)acc[tm][r]=0.0f;
            for(uint b=0;b<K;b+=128u) {
                const uint c=b+lane*4u;
                float wv[4][4];
                for(uint tm=0;tm<4;++tm){const uint2 wr=*((const device uint2*)(w+size_t(row0+tm)*K+c));
                    wv[tm][0]=lo_f(wr.x);wv[tm][1]=hi_f(wr.x);wv[tm][2]=lo_f(wr.y);wv[tm][3]=hi_f(wr.y);}
                for(uint r=0;r<nr;++r) {
                    const uint2 xr=*((const device uint2*)(x+size_t(r0+r)*K+c));
                    const float v[4]={lo_f(xr.x),hi_f(xr.x),lo_f(xr.y),hi_f(xr.y)};
                    for(uint tm=0;tm<4;++tm){acc[tm][r]+=wv[tm][0]*v[0];acc[tm][r]+=wv[tm][1]*v[1];acc[tm][r]+=wv[tm][2]*v[2];acc[tm][r]+=wv[tm][3]*v[3];}
                }
            }
            for(uint r=0;r<nr;++r)for(uint tm=0;tm<4;++tm) {
                float v=acc[tm][r];
                for(ushort sn=16;sn>=1;sn>>=1)v+=simd_shuffle_down(v,sn);
                if(lane==0)y[size_t(r0+r)*m.y_stride+m.y_col+row0+tm]=ubf(v);
            }
        } else {
            // bn = 8 configuration, one weight row: BF16 out (mode 1) or FP32 out (mode 3, router logits)
            const uint row=task;
            for(uint r=0;r<nr;++r) {
                const float v=gemv_bn8_row(w+size_t(row)*K,K,x+size_t(r0+r)*K,lane);
                if(lane==0) {
                    if(m.mode==1u)y[size_t(r0+r)*m.y_stride+m.y_col+row]=ubf(v);
                    else yf[size_t(r0+r)*m.y_stride+m.y_col+row]=v;
                }
            }
        }
    }
}
// ---- experts for few rows per expert: task = (expert, 4 weight rows); the expert's assignments (contiguous in xp
// after ink_expert_place_rows) multiply the rows read once. gate/up: the activation of the 2 channels -> act;
// down -> ep. The per-row arithmetic is the decode step's fp_qmv.
kernel void ink_experts_qmv_rows(const device uchar* weights [[buffer(0)]],const device ushort* xin [[buffer(1)]],
    device ushort* out [[buffer(2)]],constant ulong4& wq [[buffer(3)]],const device uint* starts [[buffer(4)]],
    constant uint4& shape [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    // shape: (gateup, mxfp8, weight rows per expert, inputs)
    const uint e=g.y,row0=(g.x*8u+simd)*4u;
    const uint gateup=shape.x,mx=shape.y,wrows=shape.z,K=shape.w;
    if(row0>=wrows)return;
    const uint a0=starts[e],a1=starts[e+1];
    if(a0>=a1)return;
    const device uchar* codes=weights+wq.x;
    const device uchar* scales=weights+wq.y;
    const float inv=mx?1.0f:fdiv(((const device float*)(weights+wq.z))[e],2688.0f);
    const uint nout=gateup?wrows/2u:wrows;
    for(uint b0=a0;b0<a1;b0+=MAXR) {
        const uint nr=min(MAXR,a1-b0);
        float acc[4][MAXR];
        for(uint tm=0;tm<4;++tm)for(uint r=0;r<MAXR;++r)acc[tm][r]=0.0f;
        if(!mx) {
            for(uint b=0;b<K;b+=256u) {
                const uint c=b+lane*8u;
                if(c>=K)break;
                for(uint r=0;r<nr;++r) {
                    float xv[8];
                    const device ushort* xr=xin+size_t(b0+r)*K+c;
                    for(uint i=0;i<8;++i)xv[i]=bfu(xr[i]);
                    for(uint tm=0;tm<4;++tm) {
                        const size_t row=size_t(e)*wrows+row0+tm;
                        acc[tm][r]+=qdot_nv(codes+row*(K/2u)+c/2u,xv,e4m3(scales[row*(K/16u)+c/16u]));
                    }
                }
            }
        } else if(K%256u==0u) {
            for(uint b=0;b<K;b+=256u) {
                const uint c=b+lane*8u;
                for(uint r=0;r<nr;++r) {
                    float xv[8];
                    const device ushort* xr=xin+size_t(b0+r)*K+c;
                    for(uint i=0;i<8;++i)xv[i]=bfu(xr[i]);
                    for(uint tm=0;tm<4;++tm) {
                        const size_t row=size_t(e)*wrows+row0+tm;
                        acc[tm][r]+=qdot_mx<8>(codes+row*K+c,xv,e8m0(scales[row*(K/32u)+c/32u]));
                    }
                }
            }
        } else {
            for(uint b=0;b<K;b+=128u) {
                const uint c=b+lane*4u;
                for(uint r=0;r<nr;++r) {
                    float xv[4];
                    const device ushort* xr=xin+size_t(b0+r)*K+c;
                    for(uint i=0;i<4;++i)xv[i]=bfu(xr[i]);
                    for(uint tm=0;tm<4;++tm) {
                        const size_t row=size_t(e)*wrows+row0+tm;
                        acc[tm][r]+=qdot_mx<4>(codes+row*K+c,xv,e8m0(scales[row*(K/32u)+c/32u]));
                    }
                }
            }
        }
        for(uint r=0;r<nr;++r) {
            float v[4];
            for(uint tm=0;tm<4;++tm){v[tm]=simd_sum(acc[tm][r]);v[tm]=rb(mx?v[tm]:v[tm]*inv);}
            if(lane==0) {
                if(gateup) {
                    out[size_t(b0+r)*nout+row0/2u]=ubf(softcapped_swiglu(v[0],v[1]));
                    out[size_t(b0+r)*nout+row0/2u+1u]=ubf(softcapped_swiglu(v[2],v[3]));
                } else for(uint tm=0;tm<4;++tm)out[size_t(b0+r)*nout+row0+tm]=ubf(v[tm]);
            }
        }
    }
}
// ---- debug: copy the hidden rows after a visit into trace[visit]
kernel void ink_trace_rows(const device ushort* h [[buffer(0)]],device ushort* trace [[buffer(1)]],
    constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)trace[i]=h[i];}
// ---- greedy selection per logit row (ties: lowest id)
kernel void ink_argmax_rows(const device float* logits [[buffer(0)]],device uint* out [[buffer(1)]],
    constant uint& nrows [[buffer(2)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=nrows)return;
    threadgroup float bv[8];threadgroup uint bw[8];
    float best=-FRT_INF;uint sel=0xffffffffu;
    for(uint w=tid;w<FVOCAB_USED;w+=256u){const float v=logits[size_t(row)*FVOCAB+w];if(v>best||(v==best&&w<sel)){best=v;sel=w;}}
    const float pk=simd_max(best);const uint f=simd_min(best==pk?sel:0xffffffffu);
    if(lane==0){bv[simd]=pk;bw[simd]=f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd==0) {
        const float v=lane<8u?bv[lane]:-FRT_INF;const uint w=lane<8u?bw[lane]:0xffffffffu;
        const float m=simd_max(v);const uint c=simd_min(v==m?w:0xffffffffu);
        if(lane==0)out[row]=c;
    }
}
)MSL";
}

}  // namespace flashrt::apple::shaders
