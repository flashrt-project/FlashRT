#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Row kernels of the chunked prefill of the qwen3_5_moe text decoder (BF16 activations, FP32 math, one BF16
// rounding per operation, matching the resident decode step): affine-4 embedding rows; Gated DeltaNet rows
// (depthwise conv + SiLU, unit-RMS q/k, gates), the delta-rule recurrence over the chunk, the conv ring update and
// the gated RMSNorm; per-head Q/K RMSNorm + RoPE with K/V publication for the softmax-attention layers; router
// softmax top-8 rows, expert counting and row placement, and the weighted expert combination.
// Chunk-variable values (first position, valid rows) come from a device `state` buffer [pos0, rows]; grids
// cover the chunk capacity and surplus threadgroups return.
inline std::string qwen35moe_prefill_rows_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
inline float bits_f(uint b){return as_type<float>(b<<16);}
inline float bf_f(bfloat x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float lo_f(uint w){return as_type<float>(w<<16);}
inline float hi_f(uint w){return as_type<float>(w&0xffff0000u);}
inline float rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
inline void put(device bfloat* p,float v){*((device ushort*)p)=ushort(as_type<uint>(v)>>16);}
inline float frt_absf(float x){return as_type<float>(as_type<uint>(x)&0x7fffffffu);}
inline float sigmoid_bf16(float x,bool fast) {
    float e=rb(fast?metal::fast::exp(frt_absf(x)):metal::precise::exp(frt_absf(x)));
    float tail=rb(1.0f/rb(1.0f+e));
    return x<0?tail:rb(1.0f-tail);
}
inline float silu_bf16(float x){return rb(x*sigmoid_bf16(x,true));}
inline float silu_f32(float x) {
    float y=1.0f/(1.0f+metal::fast::exp(frt_absf(x)));
    return x*(x<0?y:1.0f-y);
}
constant constexpr uint HIDDEN=2048u,EXPERTS=256u,TOPK=8u;
constant constexpr float EPS=1e-06f;

struct EmbedRows { uint hidden, pad; ulong w, s, b; };
// h[row, i] = bf16(fma(scale, code, bias)) for token tokens[pos0 + row] (4-bit codes); grid (hidden/256, rows).
kernel void qwen35moe_embed_affine4_rows_bf16(const device uint* tokens [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* h [[buffer(2)]],constant EmbedRows& p [[buffer(3)]],const device uint* state [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    uint row=g.y,i=g.x*256u+t;
    if(row>=state[1] || i>=p.hidden)return;
    uint token=tokens[state[0]+row];
    const device ushort* s=(const device ushort*)(weights+p.s);
    const device ushort* b=(const device ushort*)(weights+p.b);
    uint code=(uint((weights+p.w)[size_t(token)*(p.hidden/2u)+i/2u])>>((i&1u)*4u))&15u;
    uint gi=token*(p.hidden/64u)+i/64u;
    put(h+size_t(row)*p.hidden+i,rb(metal::fma(bits_f(s[gi]),float(code),bits_f(b[gi]))));
}

// ------------------------------------------------------------------------------------- DeltaNet
struct DeltaRows {
    ulong conv, alog, dt, norm;     // weight byte offsets: conv [8192][4], A_log [32], dt_bias [32], norm [128]
    ulong ring, state;              // element offsets of this layer's conv ring ([4][8192] BF16) and state (FP32)
};
// Conv input of channel c at chunk row r, tap i (frame of position pos0 + r - 3 + i): chunk rows from `qkv`
// ([rows][8192]), earlier frames from the ring (slot = position mod 4; zero before position 0 after a reset).
inline float conv_out(const device bfloat* qkv,const device bfloat* ring,const device bfloat* w,uint c,uint r,uint pos0) {
    float acc=0.0f;
    for(uint i=0;i<4;++i) {
        int rr=int(r)-3+int(i);
        float x=rr>=0?bf_f(qkv[size_t(rr)*8192u+c]):bf_f(ring[((pos0+r+1u+i)&3u)*8192u+c]);
        acc+=x*bf_f(w[c*4u+i]);
    }
    return rb(acc);
}
// One threadgroup (256 threads) per row: depthwise conv + SiLU of the 8192 q/k/v channels; q and k (16 heads of
// 128) normalized to unit RMS and scaled (1/128 and BF16(128^-0.5)); gates g = exp(-exp(A_log) softplus(a + dt))
// (FP32) and beta = sigmoid(b) per value head. Outputs: qk [rows][q 2048 | k 2048] and v [rows][4096] BF16,
// gates [rows][g 32 | beta 32] FP32. `ba` holds b [32] then a [32] per row.
kernel void qwen35moe_deltanet_inputs_rows_bf16(const device bfloat* qkv [[buffer(0)]],const device bfloat* rings [[buffer(1)]],
    const device uchar* weights [[buffer(2)]],const device bfloat* ba [[buffer(3)]],device bfloat* qk [[buffer(4)]],
    device bfloat* v [[buffer(5)]],device float* gates [[buffer(6)]],constant DeltaRows& p [[buffer(7)]],
    const device uint* state [[buffer(8)]],uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint pos0=state[0],rows=state[1];
    if(row>=rows)return;
    const device bfloat* w=(const device bfloat*)(weights+p.conv);
    const device bfloat* ring=rings+p.ring;
    for(uint task=sg;task<64u;task+=8u) {   // 16 q heads, 16 k heads, 32 v heads: 128 channels each
        const uint base=task*128u+lane*4u;
        float x[4];
        for(uint i=0;i<4;++i)x[i]=silu_bf16(conv_out(qkv,ring,w,base+i,row,pos0));
        if(task<32u) {
            float acc=0.0f;
            for(uint i=0;i<4;++i)acc+=x[i]*x[i];
            acc=simd_sum(acc);
            float inverse=metal::precise::rsqrt(acc/128.0f+EPS);
            const float scale=task<16u?0.0078125f:rb(0.08838834764831845f);
            for(uint i=0;i<4;++i)put(qk+size_t(row)*4096u+base+i,rb(scale*rb(x[i]*inverse)));
        } else {
            for(uint i=0;i<4;++i)put(v+size_t(row)*4096u+(base-4096u)+i,x[i]);
        }
    }
    if(sg==0) {
        const float b=bf_f(ba[size_t(row)*64u+lane]),a=bf_f(ba[size_t(row)*64u+32u+lane]);
        const float alog=bf_f(((const device bfloat*)(weights+p.alog))[lane]);
        const float dt=bf_f(((const device bfloat*)(weights+p.dt))[lane]);
        const float xs=rb(a+dt);
        const float hi=max(xs,0.0f),lo=min(xs,0.0f);
        const float e=rb(metal::precise::exp(rb(lo-hi)));
        const float xp1=1.0f+e;
        const float l1p=xp1==1.0f?e:rb(e*(metal::precise::log(xp1)/(xp1-1.0f)));
        const float sp=rb(hi+l1p);
        gates[size_t(row)*64u+lane]=metal::precise::exp(-metal::precise::exp(alog)*sp);
        gates[size_t(row)*64u+32u+lane]=sigmoid_bf16(b,false);
    }
}
// The delta rule over the chunk's rows in order. One SIMD group per four state rows of one value head (grid: 128
// threadgroups of 256 threads; group g: head g / 4, rows ((g mod 4) * 8 + simd) * 4 ..+3); a lane owns four of
// the 128 key dimensions. Per row and state row: S *= g; delta = (v - S.k) * beta; S += k delta; y = S.q
// (FP32, the arithmetic of MLX's gated_delta_step kernel). y: [rows][4096] BF16.
kernel void qwen35moe_deltanet_recurrence_bf16(const device bfloat* qk [[buffer(0)]],const device bfloat* v [[buffer(1)]],
    const device float* gates [[buffer(2)]],device float* states [[buffer(3)]],device bfloat* y [[buffer(4)]],
    constant DeltaRows& p [[buffer(5)]],const device uint* state [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint rows=state[1];
    const uint head=group/4u,khead=head/2u,dv=((group%4u)*8u+simd)*4u;
    device float4* cell=(device float4*)(states+p.state+size_t(head)*16384u+dv*128u+lane*4u);
    const float4 c0=cell[0],c1=cell[32],c2=cell[64],c3=cell[96];
    float4 s[4];
    for(uint i=0;i<4;++i)s[i]=float4(c0[i],c1[i],c2[i],c3[i]);
    #pragma clang loop unroll_count(1)
    for(uint t=0;t<rows;++t) {
        const device bfloat* row=qk+size_t(t)*4096u+khead*128u+lane*4u;
        const uint2 qr=*((const device uint2*)row),kr=*((const device uint2*)(row+2048u));
        const float q[4]={lo_f(qr.x),hi_f(qr.x),lo_f(qr.y),hi_f(qr.y)};
        const float k[4]={lo_f(kr.x),hi_f(kr.x),lo_f(kr.y),hi_f(kr.y)};
        const float g=gates[size_t(t)*64u+head],beta=gates[size_t(t)*64u+32u+head];
        const uint2 vr=*((const device uint2*)(v+size_t(t)*4096u+head*128u+dv));
        float4 kv_mem=float4(0.0f);
        for(uint i=0;i<4;++i){s[i]=s[i]*g;kv_mem+=s[i]*k[i];}
        kv_mem=simd_sum(kv_mem);
        const float4 delta=(float4(lo_f(vr.x),hi_f(vr.x),lo_f(vr.y),hi_f(vr.y))-kv_mem)*beta;
        float4 out=float4(0.0f);
        for(uint i=0;i<4;++i){s[i]=s[i]+k[i]*delta;out+=s[i]*q[i];}
        out=simd_sum(out);
        if(lane==0) {
            device uint2* dst=(device uint2*)(y+size_t(t)*4096u+head*128u+dv);
            *dst=uint2((as_type<uint>(rb(out[0]))>>16)|(as_type<uint>(rb(out[1]))&0xffff0000u),
                       (as_type<uint>(rb(out[2]))>>16)|(as_type<uint>(rb(out[3]))&0xffff0000u));
        }
    }
    for(uint r=0;r<4;++r)cell[32u*r]=float4(s[0][r],s[1][r],s[2][r],s[3][r]);
}
// The chunk's last (up to four) conv input frames into the ring (slot = position mod 4); grid (8192, 4) threads.
kernel void qwen35moe_deltanet_ring_bf16(const device bfloat* qkv [[buffer(0)]],device bfloat* rings [[buffer(1)]],
    constant DeltaRows& p [[buffer(2)]],const device uint* state [[buffer(3)]],uint2 g [[thread_position_in_grid]]) {
    const uint pos0=state[0],rows=state[1];
    if(g.y>=rows || g.x>=8192u)return;
    const uint r=rows-1u-g.y;
    rings[p.ring+((pos0+r)&3u)*8192u+g.x]=qkv[size_t(r)*8192u+g.x];
}
// out[row][head] = bf16(silu(z) * bf16(w * bf16(y * rsqrt(mean(y^2) + eps)))) per value head of 128 (SiLU and the
// product in FP32); one threadgroup per row, a SIMD group per head.
kernel void qwen35moe_deltanet_gate_rows_bf16(const device bfloat* y [[buffer(0)]],const device bfloat* z [[buffer(1)]],
    const device uchar* weights [[buffer(2)]],device bfloat* out [[buffer(3)]],constant DeltaRows& p [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    if(row>=state[1])return;
    const device bfloat* nw=(const device bfloat*)(weights+p.norm);
    for(uint head=sg;head<32u;head+=8u) {
        const size_t base=size_t(row)*4096u+head*128u+lane*4u;
        float x[4];float acc=0.0f;
        for(uint i=0;i<4;++i){x[i]=bf_f(y[base+i]);acc+=x[i]*x[i];}
        acc=simd_sum(acc);
        const float inverse=metal::precise::rsqrt(acc/128.0f+EPS);
        for(uint i=0;i<4;++i) {
            const float normed=rb(bf_f(nw[lane*4u+i])*rb(x[i]*inverse));
            put(out+base+i,rb(silu_f32(bf_f(z[base+i]))*normed));
        }
    }
}

// ------------------------------------------------------------------------------- softmax attention
struct QkRows {
    uint cap, pad;                  // KV slots per head
    ulong q_norm, k_norm, kv;       // weight byte offsets; KV element offset of this layer
    float logbase, pad2;
};
// One threadgroup per row over the projection outputs x [rows][q 16 x (query 256 | gate 256) | k 2 x 256 | v 2 x 256]:
// RMSNorm per head (Q and K; MLX's two-leaf reduction over 256), RoPE over the pairs (d, d + 32) of the first 64
// dimensions, Q written KV-head-major ([kv_head][row][8][256]), K/V written to the cache (slot = position;
// layout K [2][cap][256] then V).
kernel void qwen35moe_qk_norm_rope_kv_rows_bf16(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],device bfloat* kv [[buffer(3)]],constant QkRows& p [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint row [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint rows=state[1];
    if(row>=rows)return;
    const uint position=state[0]+row;
    const device bfloat* src=x+size_t(row)*9216u;
    for(uint task=sg;task<20u;task+=8u) {   // 16 query heads, 2 key heads, 2 value heads
        const device bfloat* in=task<16u?src+task*512u:src+8192u+(task-16u)*256u;
        device bfloat* dst;
        if(task<16u)dst=q_out+((size_t(task/8u)*rows+row)*8u+task%8u)*256u;
        else if(task<18u)dst=kv+p.kv+(size_t(task-16u)*p.cap+position)*256u;
        else dst=kv+p.kv+size_t(2u)*p.cap*256u+(size_t(task-18u)*p.cap+position)*256u;
        if(task>=18u){*((device uint4*)(dst+lane*8u))=*((const device uint4*)(in+lane*8u));continue;}
        const device bfloat* w=(const device bfloat*)(weights+(task<16u?p.q_norm:p.k_norm));
        float s0=0.0f,s1=0.0f;
        for(uint j=0;j<4;++j){float e=bf_f(in[lane*4u+j]);s0+=e*e;}
        for(uint j=0;j<4;++j){float e=bf_f(in[128u+lane*4u+j]);s1+=e*e;}
        s0=simd_sum(s0);s1=simd_sum(s1);
        const float total=simd_sum(lane==0?s0:(lane==1?s1:0.0f));
        const float inverse=metal::precise::rsqrt(total/256.0f+EPS);
        const float a=rb(bf_f(w[lane])*rb(bf_f(in[lane])*inverse));
        const float b=rb(bf_f(w[lane+32u])*rb(bf_f(in[lane+32u])*inverse));
        const float theta=float(position)*metal::exp2(-(float(lane)/32.0f)*p.logbase);
        const float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
        put(dst+lane,rb(a*c-b*s));
        put(dst+lane+32u,rb(a*s+b*c));
        for(uint j=0;j<6;++j){const uint d=64u+lane*6u+j;put(dst+d,rb(bf_f(w[d])*rb(bf_f(in[d])*inverse)));}
    }
}

// ------------------------------------------------------------------------------------------- MoE
struct RouterRows { ulong w, s, b, gw, gs, gb; };   // router [256][2048] and shared-expert gate [1][2048], 8-bit
// One threadgroup per row: router logits (8-bit affine products with the accumulation of the decode step),
// FP32 softmax, the eight largest probabilities in ascending order (ties: ascending index; equal probabilities
// at the cut keep the larger index) with weights probability / sum, and sigmoid of the shared-expert gate logit.
// ids, gates: [rows][8]; shared_gate: [rows] FP32.
kernel void qwen35moe_router_top8_rows_bf16(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* gates [[buffer(3)]],device float* shared_gate [[buffer(4)]],
    constant RouterRows& p [[buffer(5)]],const device uint* state [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if(row>=state[1])return;
    threadgroup float logits[EXPERTS];
    const device bfloat* in=x+size_t(row)*HIDDEN;
    const device ushort* scales=(const device ushort*)(weights+p.s);
    const device ushort* biases=(const device ushort*)(weights+p.b);
    for(uint e=sg;e<EXPERTS;e+=8u) {
        float a=0.0f;
        for(uint block=0;block<HIDDEN;block+=256u) {
            const uint scalar=block+lane*8u;
            float v[8];float sum=0.0f;
            for(uint i=0;i<8;++i){v[i]=bf_f(in[scalar+i]);sum+=v[i];}
            const uint gi=e*32u+scalar/64u;
            const device uchar* w=weights+p.w+size_t(e)*HIDDEN+scalar;
            float accum=0.0f;
            for(uint i=0;i<8;++i)accum+=v[i]*float(w[i]);
            a+=bits_f(scales[gi])*accum+sum*bits_f(biases[gi]);
        }
        a=simd_sum(a);
        if(lane==0)logits[e]=rb(a);
    }
    if(sg==7) {
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
        a=simd_sum(a);
        if(lane==0)shared_gate[row]=sigmoid_bf16(rb(a),false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg!=0)return;
    float xv[8];
    for(uint j=0;j<8;++j)xv[j]=logits[(j/4u)*128u+lane*4u+j%4u];
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
        for(uint r=0;r<8;++r)total=rb(total+chosen[r]);
        for(uint r=0;r<8;++r){ids[row*TOPK+r]=picked[r];gates[row*TOPK+r]=rb(chosen[r]/total);}
    }
}
// counts[e] += assignments; grid = rows * 8 threads.
kernel void qwen35moe_expert_count(const device uint* ids [[buffer(0)]],device atomic_uint* counts [[buffer(1)]],
    const device uint* state [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=state[1]*TOPK)return;
    atomic_fetch_add_explicit(counts+ids[i],1u,memory_order_relaxed);
}
// One threadgroup per assignment (row, rank): claim a slot in the expert's block and copy the row.
kernel void qwen35moe_expert_place_bf16(const device bfloat* x [[buffer(0)]],const device uint* ids [[buffer(1)]],
    device atomic_uint* cursors [[buffer(2)]],device uint* slots [[buffer(3)]],device bfloat* xp [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint a [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    if(a>=state[1]*TOPK)return;
    threadgroup uint slot[1];
    if(t==0){slot[0]=atomic_fetch_add_explicit(cursors+ids[a],1u,memory_order_relaxed);slots[a]=slot[0];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint4* src=(const device uint4*)(x+size_t(a/TOPK)*HIDDEN);
    device uint4* dst=(device uint4*)(xp+size_t(slot[0])*HIDDEN);
    dst[t]=src[t];   // 256 threads x 8 BF16 = 2048
}
// h = bf16(h + bf16(sum_r bf16(y_r * w_r) + bf16(sigmoid(gate) * shared))), the sum over the eight experts in
// ascending-probability order with BF16 rounding per add.
kernel void qwen35moe_expert_combine_rows_bf16(device bfloat* h [[buffer(0)]],const device bfloat* yp [[buffer(1)]],
    const device bfloat* shared [[buffer(2)]],const device uint* slots [[buffer(3)]],const device float* gates [[buffer(4)]],
    const device float* shared_gate [[buffer(5)]],const device uint* state [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    if(row>=state[1])return;
    uint slot[TOPK];float g[TOPK];
    for(uint r=0;r<TOPK;++r){slot[r]=slots[row*TOPK+r];g[r]=gates[row*TOPK+r];}
    const float sgate=shared_gate[row];
    for(uint k=0;k<8;++k) {
        const uint i=t*8u+k;
        float s=0.0f;
        for(uint r=0;r<TOPK;++r)s=rb(s+rb(bf_f(yp[size_t(slot[r])*HIDDEN+i])*g[r]));
        const float mlp=rb(s+rb(sgate*bf_f(shared[size_t(row)*HIDDEN+i])));
        put(h+size_t(row)*HIDDEN+i,rb(bf_f(h[size_t(row)*HIDDEN+i])+mlp));
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
