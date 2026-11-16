#pragma once
#include "common/source.h"
#include <string>

// Resident decode step for the qwen3_5_moe text decoder (Qwen3.6-35B-A3B) in MLX affine-quantized form: one
// dispatch of 36 co-resident threadgroups runs up to eight consecutive tokens (embedding, every layer, LM head,
// token selection) separated by device-scope barriers. Layers alternate three Gated DeltaNet (linear attention)
// layers with one gated softmax-attention layer; every layer has a routed MoE (256 experts, top 8) with a shared
// expert. Weights: affine 4-bit codes + BF16 scale/bias per 64 inputs (8-bit for the router and the shared-expert
// gate); activations, residual stream and KV cache in BF16; the DeltaNet recurrent state in FP32.
//
// The arithmetic follows MLX 0.32 / mlx-lm 0.31 operation by operation for a one-token step (quantized
// matrix-vector products, RMSNorm, depthwise conv, the gated delta rule, RoPE, vector attention, softmax and the
// BF16 elementwise operations each round where MLX rounds), so a step reproduces mlx-lm's logits.
//
// Geometry, fixed in this source and checked against the checkpoint by the runtime before compilation: 40 layers
// (every fourth softmax attention), hidden 2048, DeltaNet 16 key / 32 value heads of 128 with a 4-tap depthwise
// conv, attention 16 query / 2 KV heads of 256 with RoPE on the first 64 dimensions (theta 1e7) and an output
// gate, 256 experts top-8 of width 512, shared expert 512, vocabulary 248320, RMSNorm eps 1e-6.

namespace flashrt::apple::shaders {

struct Qwen35MoeDecodeOptions {
    bool sampled = false;   // fused top-k/top-p gumbel selection instead of greedy argmax
    bool trace = false;     // write the residual stream after the embedding and every layer
    int debug_layer = -1;   // >= 0: also dump that layer's intermediates into the trace buffer tail (step 0)
};

inline std::string qwen35moe_decode_resident_bf16_source(const Qwen35MoeDecodeOptions& o = {}) {
    std::string defines = std::string("\n#define FRT_WORKERS 36u") +
        "\n#define FRT_SAMPLED " + std::to_string(o.sampled ? 1 : 0) +
        "\n#define FRT_TRACE " + std::to_string(o.trace ? 1 : 0) +
        "\n#define FRT_DEBUG_LAYER " + std::to_string(o.debug_layer) + "\n";
    return std::string(metal_preamble) + defines + R"MSL(
typedef bfloat bfloat16_t;
constant constexpr uint FRT_GROUPS=FRT_WORKERS*8u;        // SIMD groups in the grid (288)
constant constexpr uint FRT_TEAM_GROUPS=FRT_WORKERS/2u;   // threadgroups per KV team (one KV head)
constant constexpr uint FRT_TEAM_SIMDS=FRT_TEAM_GROUPS*8u;
constant constexpr uint FRT_THREADS=FRT_WORKERS*256u;
constant constexpr uint FRT_VOCAB=248320u;
constant constexpr uint FRT_PER_THREAD=(FRT_VOCAB+FRT_THREADS-1u)/FRT_THREADS;   // logits per thread (27)
constant constexpr uint FRT_LAYERS=40u;
constant constexpr float FRT_EPS=1e-06f;
constant constexpr float FRT_LOGBASE=23.253496664211536f;   // log2(rope_theta = 1e7)
constant constexpr uint FRT_SPIN_LIMIT=10000000u;
constant constexpr float FRT_MIN=-3.402823466e+38f;

// ------------------------------------------------------------------------------------------ layout
// Layer descriptor (ulong[64]): byte offsets into the weight buffer; element offsets into the state buffers.
enum : uint {
    D_FLAGS=0, D_IN_NORM=1, D_POST_NORM=2,
    D_GDN_NORM=3, D_CONV=4, D_ALOG=5, D_DT=6,          // DeltaNet layers
    D_Q_NORM=3, D_K_NORM=4,                            // attention layers
    D_M0=8, D_M1=12, D_M2=16, D_M3=20, D_OUT=24,       // DeltaNet: qkv, z, b, a, out; attention: q, k, v, -, o
    D_ROUTER=28, D_SGATE=32, D_GATE=36, D_UP=40, D_DOWN=44, D_SH_GATE=48, D_SH_UP=52, D_SH_DOWN=56,
    D_STATE=60, D_AUX=61,                              // DeltaNet: state, conv ring; attention: KV, capacity
    // final descriptor (index 40)
    D_FINAL_NORM=1, D_EMBED=8, D_HEAD=12
};
// Workspace (BF16 elements).
enum : uint {
    W_QKV=0,        // attention: q_proj output [16][query 256 | gate 256]; DeltaNet: unused (qkv goes to the ring)
    W_Z=8192,       // DeltaNet z [4096]; attention: k [512], v [512] at W_Z + 512
    W_BA=12288,     // DeltaNet b [32], a [32]
    W_A=12544,      // input of the output projection [4096]
    W_O=16640,      // output projection [2048]
    W_ACT=18688,    // expert activations [9][512]
    W_DOWN=23296,   // expert outputs [9][2048]
    W_ROUTE=41728   // router logits [256], shared-expert gate logit [1]
};
struct Matrix {
    const device uchar* w;
    const device bfloat16_t* s;
    const device bfloat16_t* b;
    uint n, k;
};
inline Matrix weight_matrix(const device uchar* weights, const threadgroup ulong* d) {
    return {weights+d[0],(const device bfloat16_t*)(weights+d[1]),(const device bfloat16_t*)(weights+d[2]),
            uint(d[3]>>32),uint(d[3]&0xffffffffu)};
}
inline Matrix weight_matrix(const device uchar* weights, const device ulong* d) {
    return {weights+d[0],(const device bfloat16_t*)(weights+d[1]),(const device bfloat16_t*)(weights+d[2]),
            uint(d[3]>>32),uint(d[3]&0xffffffffu)};
}
// KV cache of one attention layer: keys [2 heads][cap][256], then values in the same layout.
struct KVView {
    device bfloat16_t* base;
    ulong offset;
    uint cap;
    KVView head(uint index,uint dim) const {KVView v=*this;v.offset+=ulong(index)*cap*256u+dim;return v;}
    KVView value() const {KVView v=*this;v.offset+=ulong(cap)*512u;return v;}
    const device bfloat16_t* address(uint i) const {return base+offset+i;}
    uint4 load8(uint i) const {return *((const device uint4*)(address(i)));}
};
struct DecodeParams {
    uint token, position, steps, eos_count;
    uint eos[4];
    uint top_k; float top_p, inverse_temperature;
    uint debug_layer;      // trace build: layer + 1 whose intermediates go to the trace buffer's second block
};

// ------------------------------------------------------------------------------------------ barriers
// Device-scope barriers over co-resident threadgroups. A barrier that does not complete within the spin
// limit raises a failure flag; every group then exits and reports it through `diagnostic`.
inline bool grid_sync(device atomic_uint* control,uint tid,thread uint& phase,threadgroup uint* ready) {
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*FRT_WORKERS;
        atomic_fetch_add_explicit(control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<FRT_SPIN_LIMIT) ++spins;
        ready[0]=uint(atomic_load_explicit(control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;
}
// Expert units (4 threadgroups per routed rank and for the shared expert) publish gate/up completion per layer.
inline bool expert_handoff(device uint* routing,device atomic_uint* control,uint tid,uint group,uint ticket,
                           uint phase,threadgroup uint* ready) {
    uint first=(group/4)*4;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),ticket+1,memory_order_relaxed);
        bool complete=true;
        for(uint peer=first;peer<first+4;++peer) {
            uint spins=0;
            while(atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)<ticket+1 &&
                  atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<FRT_SPIN_LIMIT) ++spins;
            complete=complete && atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)>=ticket+1;
        }
        ready[0]=uint(complete && atomic_load_explicit(control+2,memory_order_relaxed)==0);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    return ready[0]!=0;
}

// ------------------------------------------------------------------------------ BF16 arithmetic
// MLX evaluates BF16 operations one by one: each result is computed in FP32 and rounded to BF16. Values are
// carried in FP32 registers holding BF16-exact numbers; rb() rounds to nearest even in integer arithmetic (an
// FP32 arithmetic result on this GPU is never subnormal, so this equals the hardware conversion; 16-bit
// temporaries are avoided because the compiler packs and corrupts them in multi-row code).
inline float rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
inline float bf16_f(bfloat16_t x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float lo_f(uint w){return as_type<float>(w<<16);}
inline float hi_f(uint w){return as_type<float>(w&0xffff0000u);}
inline float frt_absf(float x){return as_type<float>(as_type<uint>(x)&0x7fffffffu);}
inline void store_bf16(device bfloat16_t* p,float v){*((device ushort*)p)=ushort(as_type<uint>(v)>>16);}
inline void store_bf16(threadgroup bfloat16_t* p,float v){*((threadgroup ushort*)p)=ushort(as_type<uint>(v)>>16);}
// Sigmoid of a BF16 value: exp, 1+exp, reciprocal and 1-tail each rounded to BF16. MLX's own kernels use the
// precise exponential; the functions mlx-lm compiles at run time (SiLU, SwiGLU, the DeltaNet gates) the fast one.
inline float sigmoid_bf16(float x,bool fast) {
    float e=rb(fast?metal::fast::exp(frt_absf(x)):metal::precise::exp(frt_absf(x)));
    float tail=rb(1.0f/rb(1.0f+e));
    return x<0?tail:rb(1.0f-tail);
}
inline float silu_bf16(float x){return rb(x*sigmoid_bf16(x,true));}
inline float swiglu_bf16(float g,float u){return rb(silu_bf16(g)*u);}
// FP32 SiLU (the gated norm of the DeltaNet output).
inline float silu_f32(float x) {
    float y=1.0f/(1.0f+metal::fast::exp(frt_absf(x)));
    return x*(x<0?y:1.0f-y);
}

// ------------------------------------------------------------------------------ affine products
// MLX qmv_fast: per SIMD group four output rows; each lane owns 16 (4-bit) or 8 (8-bit) consecutive inputs of
// every 512- (256-) input block and accumulates scale*sum(x*q) + bias*sum(x) per block; one simd_sum per row.
// 4-bit activations are pre-divided by 1, 16, 256, 4096 so masked nibbles need no shift.
template<typename P>
inline float load16(P x,thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        float x0=bf16_f(x[i]),x1=bf16_f(x[i+1]),x2=bf16_f(x[i+2]),x3=bf16_f(x[i+3]);
        sum+=rb(rb(rb(x0+x1)+x2)+x3);
        v[i]=x0;v[i+1]=x1/16.0f;v[i+2]=x2/256.0f;v[i+3]=x3/4096.0f;
    }
    return sum;
}
inline float dot16(const device uchar* w,thread float* x,float s,float b,float sum) {
    const device ushort* p=(const device ushort*)w;
    float a=0.0f;
    for(uint j=0;j<4;++j)
        a+=x[4*j]*float(uint(p[j])&15u)+x[4*j+1]*float(uint(p[j])&240u)
           +x[4*j+2]*float(uint(p[j])&3840u)+x[4*j+3]*float(uint(p[j])&61440u);
    return s*a+sum*b;
}
// Four rows starting at `first` of a [n, K] 4-bit matrix times x (K a multiple of 512).
template<uint K,typename P>
inline void q4_block(Matrix m,P x,device bfloat16_t* out,uint lane,uint first) {
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        float v[16];float sum=load16(x+scalar,v);
        for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,bf16_f(m.s[gi]),bf16_f(m.b[gi]),sum);
        }
    }
    for(uint r=0;r<4;++r){float v=simd_sum(a[r]);if(lane==0)store_bf16(out+first+r,rb(v));}
}
template<uint K,typename P>
inline void q4(Matrix m,P x,device bfloat16_t* out,uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4)q4_block<K>(m,x,out,lane,first);
}
// 8-bit qmv_fast (K = 2048): 8 inputs per lane and 256-input block, four scale groups per block. One row.
template<typename P>
inline void q8_fast_row(Matrix m,P x,device bfloat16_t* out,uint lane,uint row) {
    float a=0.0f;
    for(uint block=0;block<2048;block+=256) {
        uint scalar=block+lane*8;
        float v[8];float sum=0.0f;
        for(uint i=0;i<8;++i){v[i]=bf16_f(x[scalar+i]);sum+=v[i];}
        uint gi=row*32+scalar/64;
        const device uchar* w=m.w+size_t(row)*2048+scalar;
        float accum=0.0f;
        for(uint i=0;i<8;++i)accum+=v[i]*float(w[i]);
        a+=bf16_f(m.s[gi])*accum+sum*bf16_f(m.b[gi]);
    }
    float v=simd_sum(a);if(lane==0)store_bf16(out+row,rb(v));
}
// 8-bit general qmv for a one-row matrix (K = 2048): 4 inputs per lane and 128-input block.
template<typename P>
inline void q8_row(Matrix m,P x,device bfloat16_t* out,uint lane) {
    float a=0.0f;
    for(uint block=0;block<2048;block+=128) {
        uint scalar=block+lane*4;
        float v[4];float sum=0.0f;
        for(uint i=0;i<4;++i){v[i]=bf16_f(x[scalar+i]);sum+=v[i];}
        uint gi=scalar/64;
        const device uchar* w=m.w+scalar;
        float accum=0.0f;
        for(uint i=0;i<4;++i)accum+=v[i]*float(w[i]);
        a+=bf16_f(m.s[gi])*accum+sum*bf16_f(m.b[gi]);
    }
    float v=simd_sum(a);if(lane==0)store_bf16(out,rb(v));
}

// ------------------------------------------------------------------------------------------- norms
// MLX rms_single_row over 2048 values: 16 SIMD groups of 128 values (4 per lane), then one reduction of the 16
// leaf sums. Each thread holds 8 of the 2048 residual values (two leaves).
inline void rms_2048(const thread float* x,const device bfloat16_t* w,threadgroup bfloat16_t* out,
                     uint tid,uint lane,uint sg,threadgroup float* temp) {
    float sums[2];
    for(uint leaf=0;leaf<2;++leaf) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=x[leaf*4+j];squared+=v*v;}
        sums[leaf]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+FRT_EPS);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;store_bf16(out+i,rb(bf16_f(w[i])*rb(x[j]*inverse)));}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// RMSNorm of one 256-wide attention head in place (two leaves of 128), then RoPE over the pairs (d, d + 32) of
// the first 64 dimensions. One SIMD group.
inline void head_norm_rope(const device bfloat16_t* p,device bfloat16_t* dst,const device bfloat16_t* w,uint position,uint lane) {
    float s0=0.0f,s1=0.0f;
    for(uint j=0;j<4;++j){float v=bf16_f(p[lane*4+j]);s0+=v*v;}
    for(uint j=0;j<4;++j){float v=bf16_f(p[128+lane*4+j]);s1+=v*v;}
    s0=simd_sum(s0);s1=simd_sum(s1);
    float total=simd_sum(lane==0?s0:(lane==1?s1:0.0f));
    float inverse=metal::precise::rsqrt(total/256.0f+FRT_EPS);
    float a=rb(bf16_f(w[lane])*rb(bf16_f(p[lane])*inverse));
    float b=rb(bf16_f(w[lane+32])*rb(bf16_f(p[lane+32])*inverse));
    float rest[6];
    for(uint j=0;j<6;++j){uint d=64+lane*6+j;rest[j]=rb(bf16_f(w[d])*rb(bf16_f(p[d])*inverse));}
    float dd=float(lane)/32.0f;
    float theta=float(position)*metal::exp2(-dd*FRT_LOGBASE);
    float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
    simdgroup_barrier(mem_flags::mem_device);
    store_bf16(dst+lane,rb(a*c-b*s));
    store_bf16(dst+lane+32,rb(a*s+b*c));
    for(uint j=0;j<6;++j)store_bf16(dst+64+lane*6+j,rest[j]);
}

// --------------------------------------------------------------------------------------- attention
// MLX vector attention. Single pass (fewer than 1024 keys): 32 key-strided partitions per query head with FP32
// partial outputs; two passes: `parts` partitions with BF16 partial outputs (ROUNDED). Each task covers one
// partition for HEADS query heads of one KV head; a lane owns 8 of the 256 dimensions.
template<uint HEADS,bool ROUNDED>
inline void attention_partial(const device bfloat16_t* q,KVView k,KVView v,device float* partial,device float* stats,
                              uint count,uint parts,uint lane,uint sg) {
    const uint per_kv=(8/HEADS)*parts;
    for(uint task=(sg/FRT_TEAM_SIMDS)*per_kv+sg%FRT_TEAM_SIMDS;task<(sg/FRT_TEAM_SIMDS+1)*per_kv;task+=FRT_TEAM_SIMDS) {
        uint kv=task/per_kv,part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*8);
        KVView value_base=v.head(kv,lane*8);
        float query[HEADS][8],acc[HEADS][8];
        float hi=FRT_MIN,z=0.0f;
        for(uint h=0;h<HEADS;++h) {
            const uint4 raw=*((const device uint4*)(q+(first+h)*512+lane*8));
            for(uint j=0;j<4;++j){query[h][2*j]=0.0625f*lo_f(raw[j]);query[h][2*j+1]=0.0625f*hi_f(raw[j]);}
            for(uint j=0;j<8;++j)acc[h][j]=0.0f;
        }
        #pragma clang loop unroll_count(1)
        for(uint token=part;token<count;token+=parts) {
            const uint4 rk=key_base.load8(token*256u),rv=value_base.load8(token*256u);
            float key[8],value[8];
            for(uint j=0;j<4;++j){key[2*j]=lo_f(rk[j]);key[2*j+1]=hi_f(rk[j]);value[2*j]=lo_f(rv[j]);value[2*j+1]=hi_f(rv[j]);}
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h){float score=0.0f;for(uint j=0;j<8;++j)score+=query[h][j]*key[j];scores[h]=simd_sum(score);}
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score);
                old=metal::fast::exp(hi-next);add=metal::fast::exp(score-next);
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<8;++j)acc[h][j]=acc[h][j]*factor+weight*value[j];
            }
        }
        if(lane<HEADS){device float* stat_out=stats+((first+lane)*parts+part)*2;stat_out[0]=hi;stat_out[1]=z;}
        for(uint h=0;h<HEADS;++h) {
            if(ROUNDED) {
                device uint4* out=(device uint4*)(((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*8);
                uint4 r;
                for(uint j=0;j<4;++j)r[j]=(as_type<uint>(rb(acc[h][2*j]))>>16)|(as_type<uint>(rb(acc[h][2*j+1]))&0xffff0000u);
                *out=r;
            } else {
                device float4* out=(device float4*)(partial+((first+h)*parts+part)*256+lane*8);
                out[0]=float4(acc[h][0],acc[h][1],acc[h][2],acc[h][3]);
                out[1]=float4(acc[h][4],acc[h][5],acc[h][6],acc[h][7]);
            }
        }
    }
}
// Merge the partitions of each head (per KV team: 8 heads x 16 slices of 16 dimensions), divide by the
// normalizer, round to BF16 and apply the output gate sigmoid(gate) (BF16) of the same position.
template<bool ROUNDED>
inline void attention_merge_gate(device bfloat16_t* out,const device bfloat16_t* q,device float* partial,
                                 device float* stats,uint parts,uint lane,uint sg) {
    for(uint task=(sg/FRT_TEAM_SIMDS)*128+sg%FRT_TEAM_SIMDS;task<(sg/FRT_TEAM_SIMDS+1)*128;task+=FRT_TEAM_SIMDS) {
        uint h=task/16,dim=(task%16)*16;
        float top=FRT_MIN;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);
        float z=0.0f;
        float a[16];
        for(uint j=0;j<16;++j)a[j]=0.0f;
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;
            float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            if(ROUNDED) {
                for(uint t=0;t<2;++t) {
                    const uint4 raw=*((const device uint4*)(((const device bfloat16_t*)partial)+((h*32+dim/8+t)*parts+p)*8));
                    for(uint j=0;j<4;++j){a[t*8+2*j]+=factor*lo_f(raw[j]);a[t*8+2*j+1]+=factor*hi_f(raw[j]);}
                }
            } else {
                const device float* src=partial+(h*parts+p)*256+dim;
                for(uint j=0;j<16;++j)a[j]+=factor*src[j];
            }
        }
        z=simd_sum(z);
        float owned=0.0f;
        for(uint j=0;j<16;++j){float total=simd_sum(a[j]);if(lane==j)owned=total;}
        if(lane<16) {
            float value=rb(z==0.0f?owned:owned/z);
            float gate=bf16_f(q[h*512+256+dim+lane]);
            store_bf16(out+h*256+dim+lane,rb(value*sigmoid_bf16(gate,false)));
        }
    }
}

// ------------------------------------------------------------------------------------- DeltaNet
// One value head per threadgroup (groups 0..31): depthwise conv + SiLU of its q, k and v channels, unit-RMS q and
// k, the gates, the delta rule over its 128 state rows and the gated RMSNorm of the result.
//   ring: conv input frames [4][8192] of this layer (slot = position mod 4; the current frame is already stored)
//   state: FP32 [32][128][128] (value head, value dim, key dim)
// tg: threadgroup scratch [q 128 | k 128 | v 128 | y 128 | g, beta]. deltanet_head leaves the delta-rule output in
// tg; deltanet_gate (after z is complete) writes the gated norm.
inline float conv_tap(const device bfloat16_t* ring,const device bfloat16_t* w,uint channel,uint position) {
    float acc=0.0f;
    for(uint i=0;i<4;++i)acc+=bf16_f(ring[((position+1u+i)&3u)*8192u+channel])*bf16_f(w[channel*4u+i]);
    return rb(acc);
}
inline void deltanet_head(const device uchar* weights,const threadgroup ulong* d,device bfloat16_t* ring,
                          device float* state,const device bfloat16_t* ws,device bfloat16_t* out,
                          threadgroup float* tg,uint head,uint position,uint tid,uint lane,uint simd,device float* dbg) {
    const device bfloat16_t* cw=(const device bfloat16_t*)(weights+d[D_CONV]);
    const uint khead=head/2u;
    if(simd<3) {
        // conv + SiLU of 128 channels (4 per lane); q and k are then normalized to unit RMS and scaled
        const uint base=simd==0?khead*128u:(simd==1?2048u+khead*128u:4096u+head*128u);
        float x[4];
        for(uint i=0;i<4;++i)x[i]=silu_bf16(conv_tap(ring,cw,base+lane*4u+i,position));
        if(simd<2) {
            float acc=0.0f;
            for(uint i=0;i<4;++i)acc+=x[i]*x[i];
            acc=simd_sum(acc);
            float inverse=metal::precise::rsqrt(acc/128.0f+FRT_EPS);
            // q: (1/128) * norm (exact scale); k: BF16(128^-0.5) * norm
            const float scale=simd==0?0.0078125f:rb(0.08838834764831845f);
            for(uint i=0;i<4;++i)x[i]=rb(scale*rb(x[i]*inverse));
        }
        for(uint i=0;i<4;++i)tg[simd*128u+lane*4u+i]=x[i];
    }
    if(simd==3 && lane==0) {
        const float b=bf16_f(ws[W_BA+head]),a=bf16_f(ws[W_BA+32u+head]);
        const float alog=bf16_f(((const device bfloat16_t*)(weights+d[D_ALOG]))[head]);
        const float dt=bf16_f(((const device bfloat16_t*)(weights+d[D_DT]))[head]);
        // softplus(a + dt) = logaddexp(x, 0) in BF16: max + log1p(exp(min - max))
        const float xs=rb(a+dt);
        const float hi=max(xs,0.0f),lo=min(xs,0.0f);
        const float e=rb(metal::precise::exp(rb(lo-hi)));
        const float xp1=1.0f+e;
        const float l1p=xp1==1.0f?e:rb(e*(metal::precise::log(xp1)/(xp1-1.0f)));
        const float sp=rb(hi+l1p);
        tg[512]=metal::precise::exp(-metal::precise::exp(alog)*sp);
        tg[513]=sigmoid_bf16(b,false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q[4],k[4];
    for(uint i=0;i<4;++i){q[i]=tg[lane*4u+i];k[i]=tg[128u+lane*4u+i];}
    const float g=tg[512],beta=tg[513];
#if FRT_TRACE
    if(dbg) {   // [g 32 | beta 32 | y 32x128 | q 32x128 | k 32x128 | v 32x128]
        if(tid==0){dbg[head]=g;dbg[32u+head]=beta;}
        if(simd<3)for(uint i=0;i<4;++i)dbg[64u+(1u+simd)*4096u+head*128u+lane*4u+i]=tg[simd*128u+lane*4u+i];
    }
#endif
    device float* rows=state+size_t(head)*16384u;
    for(uint dv=simd*16u;dv<simd*16u+16u;dv+=4u) {
        // four rows per iteration (component r = row dv + r); every row's arithmetic is that of one row
        device float4* cell=(device float4*)(rows+dv*128u+lane*4u);
        const float4 c0=cell[0],c1=cell[32],c2=cell[64],c3=cell[96];
        float4 s[4];
        for(uint i=0;i<4;++i)s[i]=float4(c0[i],c1[i],c2[i],c3[i]);
        float4 kv_mem=float4(0.0f);
        for(uint i=0;i<4;++i){s[i]=s[i]*g;kv_mem+=s[i]*k[i];}
        kv_mem=simd_sum(kv_mem);
        const float4 delta=(float4(tg[256u+dv],tg[257u+dv],tg[258u+dv],tg[259u+dv])-kv_mem)*beta;
        float4 y=float4(0.0f);
        for(uint i=0;i<4;++i){s[i]=s[i]+k[i]*delta;y+=s[i]*q[i];}
        y=simd_sum(y);
        for(uint r=0;r<4;++r)cell[32u*r]=float4(s[0][r],s[1][r],s[2][r],s[3][r]);
        if(lane==0)for(uint r=0;r<4;++r)tg[384u+dv+r]=rb(y[r]);
#if FRT_TRACE
        if(dbg && lane==0)for(uint r=0;r<4;++r)dbg[64u+head*128u+dv+r]=y[r];
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Gated RMSNorm of the head's delta-rule output (tg[384..511]) with SiLU(z): SIMD group 0 of the head's group.
inline void deltanet_gate(const device uchar* weights,const threadgroup ulong* d,const device bfloat16_t* ws,
                          device bfloat16_t* out,threadgroup float* tg,uint head,uint lane,uint simd) {
    if(simd==0) {
        const device bfloat16_t* nw=(const device bfloat16_t*)(weights+d[D_GDN_NORM]);
        float y[4];float acc=0.0f;
        for(uint i=0;i<4;++i){y[i]=tg[384u+lane*4u+i];acc+=y[i]*y[i];}
        acc=simd_sum(acc);
        const float inverse=metal::precise::rsqrt(acc/128.0f+FRT_EPS);
        for(uint i=0;i<4;++i) {
            const uint e=lane*4u+i;
            const float normed=rb(bf16_f(nw[e])*rb(y[i]*inverse));
            store_bf16(out+head*128u+e,rb(silu_f32(bf16_f(ws[W_Z+head*128u+e]))*normed));
        }
    }
}

// ------------------------------------------------------------------------------------------- MoE
// Softmax (FP32 accumulation, MLX softmax_single_row over 256 values: two leaves of 128) of the router logits
// and the eight largest probabilities in ascending order (equal probabilities: ascending index), as MLX's
// argpartition leaves them; one SIMD group, results in threadgroup memory: ids[8], weights[8] (score / sum).
inline void route_top8(const device bfloat16_t* logits,threadgroup uint* ids,threadgroup float* weights,uint lane) {
    float x[8];
    for(uint j=0;j<8;++j)x[j]=bf16_f(logits[(j/4u)*128u+lane*4u+j%4u]);
    float m0=FRT_MIN,m1=FRT_MIN;
    for(uint j=0;j<4;++j){m0=(m0<x[j])?x[j]:m0;m1=(m1<x[4+j])?x[4+j]:m1;}
    m0=simd_max(m0);m1=simd_max(m1);
    const float top=max(m0,m1);
    float n0=0.0f,n1=0.0f;
    for(uint j=0;j<4;++j){x[j]=metal::fast::exp(x[j]-top);n0+=x[j];}
    for(uint j=0;j<4;++j){x[4+j]=metal::fast::exp(x[4+j]-top);n1+=x[4+j];}
    n0=simd_sum(n0);n1=simd_sum(n1);
    const float normalizer=1.0f/simd_sum(lane==0?n0:(lane==1?n1:0.0f));
    float p[8];
    for(uint j=0;j<8;++j)p[j]=rb(x[j]*normalizer);
    // descending extraction; among equal probabilities the larger index first
    float chosen[8];
    for(uint rank=0;rank<8;++rank) {
        float best=-1.0f;uint idx=0;
        for(uint j=0;j<8;++j){uint i=(j/4u)*128u+lane*4u+j%4u;if(p[j]>best||(p[j]==best&&i>idx)){best=p[j];idx=i;}}
        const float peak=simd_max(best);
        const uint selected=simd_max(best==peak?idx:0u);
        if(lane==0){ids[7u-rank]=selected;chosen[7u-rank]=peak;}
        for(uint j=0;j<8;++j)if((j/4u)*128u+lane*4u+j%4u==selected)p[j]=-2.0f;
    }
    if(lane==0) {
        float total=0.0f;
        for(uint r=0;r<8;++r)total=rb(total+chosen[r]);
        for(uint r=0;r<8;++r)weights[r]=rb(chosen[r]/total);
    }
}
// Gate/up rows of one expert unit (4 threadgroups = 32 SIMD groups): 256 row pairs, SwiGLU into `out`.
inline void expert_gateup(Matrix g,Matrix u,const threadgroup bfloat16_t* x,uint expert,device bfloat16_t* out,
                          uint lane,uint unit_sg) {
    for(uint local=unit_sg*2;local<512;local+=64) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=expert*512+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,bf16_f(g.s[gi]),bf16_f(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,bf16_f(u.s[gi]),bf16_f(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0)store_bf16(out+local+j,swiglu_bf16(rb(val.x),rb(val.y)));
        }
    }
}
// Down rows of one expert unit: 512 four-row blocks.
inline void expert_down(Matrix m,const device bfloat16_t* x,uint expert,device bfloat16_t* out,uint lane,uint unit_sg) {
    for(uint local=unit_sg*4;local<2048;local+=128) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        uint scalar=lane*16;float v[16];float sum=load16(x+scalar,v);
        for(uint j=0;j<4;++j) {
            uint row=expert*2048+local+j,gi=row*8+scalar/64;
            a[j]+=dot16(m.w+size_t(row)*256+scalar/2,v,bf16_f(m.s[gi]),bf16_f(m.b[gi]),sum);
        }
        for(uint j=0;j<4;++j){float v2=simd_sum(a[j]);if(lane==0)store_bf16(out+local+j,rb(v2));}
    }
}

// ------------------------------------------------------------------------------------------ kernel
#define FRT_FAIL() {if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!grid_sync(control,tid,phase,ready))FRT_FAIL()

kernel void qwen35moe_decode_resident_bf16(
    const device uchar* weights [[buffer(0)]],
    const device ulong* layers [[buffer(1)]],
    device bfloat16_t* kv [[buffer(2)]],
    device float* states [[buffer(3)]],
    device bfloat16_t* rings [[buffer(4)]],
    constant DecodeParams& params [[buffer(5)]],
    const device float* sampling [[buffer(6)]],
    device bfloat16_t* block_logits [[buffer(7)]],
    device bfloat16_t* block_trace [[buffer(8)]],
    device uint* packet [[buffer(9)]],
    device bfloat16_t* workspace [[buffer(10)]],
    device float* partials [[buffer(11)]],
    device float* statistics [[buffer(12)]],
    device uint* routing [[buffer(13)]],
    device uint* private_control [[buffer(14)]],
    device uint* diagnostic [[buffer(15)]],
    uint3 group3 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint group=group3.x,sg=group*8+simd,index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];
    uint phase=0,z_base=0;
    threadgroup float reduction[33];
    threadgroup ulong layer_descriptor[64];
    threadgroup float scratch[520];
    threadgroup uint route_ids[8];
    threadgroup float route_weights[8];
#if FRT_SAMPLED
    threadgroup float frt_cv[256];threadgroup uint frt_cw[256];
#endif
    if(tid==0)atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    float h[8];
    alignas(8) threadgroup bfloat16_t n[2048];
    device bfloat16_t* a=workspace+W_A;
    device bfloat16_t* o=workspace+W_O;
    device bfloat16_t* act=workspace+W_ACT;
    device bfloat16_t* down=workspace+W_DOWN;
    device bfloat16_t* route=workspace+W_ROUTE;
    const device ulong* final=layers+FRT_LAYERS*64;
    const Matrix embedding=weight_matrix(weights,final+D_EMBED);
    uint input_token=params.token;
    #pragma clang loop unroll(disable)
    for(uint step=0;step<params.steps;++step) {
        const uint position=params.position+step;
        device bfloat16_t* logits=block_logits+step*FRT_VOCAB;
#if FRT_TRACE
        device bfloat16_t* trace=block_trace+step*((FRT_LAYERS+1u)*2048u);
#endif
        if(index==0)packet[step*2]=input_token;
        for(uint j=0;j<8;++j) {
            uint i=(tid+(j/4)*256)*4+j%4;
            size_t row=input_token;
            uint code=(uint(embedding.w[row*1024+i/2])>>((i&1u)*4u))&15u;
            uint gi=row*32+i/64;
            h[j]=rb(metal::fma(bf16_f(embedding.s[gi]),float(code),bf16_f(embedding.b[gi])));
#if FRT_TRACE
            if(group==0)store_bf16(trace+i,h[j]);
#endif
        }
        SYNC();
        for(uint layer=0;layer<FRT_LAYERS;++layer) {
            if(tid<64)layer_descriptor[tid]=layers[layer*64+tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const threadgroup ulong* d=layer_descriptor;
            const bool full=(d[D_FLAGS]&1)!=0;
            rms_2048(h,(const device bfloat16_t*)(weights+d[D_IN_NORM]),n,tid,lane,simd,reduction);
            if(!full) {
                device bfloat16_t* ring=rings+d[D_AUX];
                // 2048 qkv + 8 b + 8 a + the first 240 z four-row units: eight per SIMD group
                for(uint unit=sg;unit<2304u;unit+=FRT_GROUPS) {
                    if(unit<2048u)q4_block<2048>(weight_matrix(weights,d+D_M0),n,ring+(position&3u)*8192u,lane,unit*4u);
                    else if(unit<2056u)q4_block<2048>(weight_matrix(weights,d+D_M2),n,workspace+W_BA,lane,(unit-2048u)*4u);
                    else if(unit<2064u)q4_block<2048>(weight_matrix(weights,d+D_M3),n,workspace+W_BA+32u,lane,(unit-2056u)*4u);
                    else q4_block<2048>(weight_matrix(weights,d+D_M1),n,workspace+W_Z,lane,(unit-2064u)*4u);
                }
                SYNC();
#if FRT_TRACE
                device float* dbg=(step==0u && params.debug_layer==layer+1u)?(device float*)(block_trace+(FRT_LAYERS+1u)*2048u):nullptr;
#else
                device float* dbg=nullptr;
#endif
                // The delta rule (groups 0..31) overlaps the rest of the z product: its remaining 784 four-row
                // units are claimed by ticket, by the idle groups at once and by the others after their state rows.
                if(group<32)deltanet_head(weights,d,ring,states+d[D_STATE],workspace,a,scratch,group,position,tid,lane,simd,dbg);
                for(;;) {
                    uint ticket=0;
                    if(lane==0)ticket=atomic_fetch_add_explicit(control+40,1u,memory_order_relaxed);
                    ticket=simd_broadcast(ticket,0);
                    if(ticket>=z_base+784u)break;
                    q4_block<2048>(weight_matrix(weights,d+D_M1),n,workspace+W_Z,lane,(240u+ticket-z_base)*4u);
                }
                z_base+=784u+FRT_GROUPS;
                SYNC();
                if(group<32)deltanet_gate(weights,d,workspace,a,scratch,group,lane,simd);
                SYNC();
            } else {
                const uint cap=uint(d[D_AUX]),count=position+1;
                KVView keys={kv,d[D_STATE],cap};
                KVView vals=keys.value();
                device bfloat16_t* q=workspace+W_QKV;
                device bfloat16_t* kx=workspace+W_Z;
                // 2048 q + 128 k + 128 v four-row units, strided over the SIMD groups
                for(uint unit=sg;unit<2304u;unit+=FRT_GROUPS) {
                    if(unit<2048u)q4_block<2048>(weight_matrix(weights,d+D_M0),n,q,lane,unit*4u);
                    else if(unit<2176u)q4_block<2048>(weight_matrix(weights,d+D_M1),n,kx,lane,(unit-2048u)*4u);
                    else q4_block<2048>(weight_matrix(weights,d+D_M2),n,kx+512u,lane,(unit-2176u)*4u);
                }
                SYNC();
                // q/k RMSNorm + RoPE (SIMD groups 0..15 query heads, 16..17 key heads); values copied to the cache
                if(sg<16)head_norm_rope(q+sg*512u,q+sg*512u,(const device bfloat16_t*)(weights+d[D_Q_NORM]),position,lane);
                else if(sg<18)head_norm_rope(kx+(sg-16u)*256u,(device bfloat16_t*)keys.head(sg-16u,0).address(position*256u),
                                             (const device bfloat16_t*)(weights+d[D_K_NORM]),position,lane);
                else if(sg<20) {
                    const uint hd=sg-18u;
                    device uint4* dst=(device uint4*)(vals.head(hd,0).address(position*256u)+lane*8u);
                    *dst=*((const device uint4*)(kx+512u+hd*256u+lane*8u));
                }
                SYNC();
                const bool rounded=count>=1024;
                const uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
                if(rounded) attention_partial<4,true>(q,keys,vals,partials,statistics,count,parts,lane,sg);
                else attention_partial<2,false>(q,keys,vals,partials,statistics,count,parts,lane,sg);
                SYNC();
                if(rounded) attention_merge_gate<true>(a,q,partials,statistics,parts,lane,sg);
                else attention_merge_gate<false>(a,q,partials,statistics,parts,lane,sg);
                SYNC();
            }
            q4<4096>(weight_matrix(weights,d+D_OUT),(const device bfloat16_t*)a,o,lane,sg,FRT_GROUPS);
            SYNC();
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=rb(h[j]+bf16_f(o[i]));}
            rms_2048(h,(const device bfloat16_t*)(weights+d[D_POST_NORM]),n,tid,lane,simd,reduction);
            // router logits (8-bit) and the shared-expert gate logit
            if(sg<256)q8_fast_row(weight_matrix(weights,d+D_ROUTER),n,route,lane,sg);
            else if(sg==256)q8_row(weight_matrix(weights,d+D_SGATE),n,route+256u,lane);
            SYNC();
            if(simd==0)route_top8(route,route_ids,route_weights,lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                const uint unit=group/4u,unit_sg=(group%4u)*8u+simd;
                const bool shared=unit==8u;
                const uint expert=shared?0u:route_ids[unit];
                expert_gateup(weight_matrix(weights,d+(shared?D_SH_GATE:D_GATE)),weight_matrix(weights,d+(shared?D_SH_UP:D_UP)),
                              n,expert,act+unit*512u,lane,unit_sg);
                if(!expert_handoff(routing,control,tid,group,step*FRT_LAYERS+layer,phase,ready))FRT_FAIL();
                expert_down(weight_matrix(weights,d+(shared?D_SH_DOWN:D_DOWN)),act+unit*512u,expert,down+unit*2048u,lane,unit_sg);
            }
            SYNC();
            // Combine: BF16 products with the renormalized weights summed in ascending order, the shared expert
            // scaled by sigmoid(gate), and the residual.
            {
                const float sgate=sigmoid_bf16(bf16_f(route[256]),false);
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    float sum=0.0f;
                    for(uint r=0;r<8;++r)sum=rb(sum+rb(bf16_f(down[r*2048u+i])*route_weights[r]));
                    const float mlp=rb(sum+rb(sgate*bf16_f(down[8u*2048u+i])));
                    h[j]=rb(h[j]+mlp);
#if FRT_TRACE
                    if(group==0)store_bf16(trace+(layer+1)*2048+i,h[j]);
#endif
                }
            }
        }
        rms_2048(h,(const device bfloat16_t*)(weights+final[D_FINAL_NORM]),n,tid,lane,simd,reduction);
        q4<2048>(weight_matrix(weights,final+D_HEAD),n,logits,lane,sg,FRT_GROUPS);
        SYNC();
#if FRT_SAMPLED
        // Fused top-k/top-p: group-local max, sum of exponentials and sorted top-K, one barrier, then every
        // group merges the per-group lists and selects the same token. Ties rank the larger token id first
        // (mlx-lm top-p: ascending argsort, cumulative > 1-p); gumbel noise comes from `sampling`.
        {
        const uint frt_k=params.top_k;
        const float frt_p=params.top_p,frt_inv_t=params.inverse_temperature;
        float xs[FRT_PER_THREAD];
        float tmax=-as_type<float>(0x7f800000u);
        for(uint j=0;j<FRT_PER_THREAD;++j){uint w=index+j*FRT_THREADS;float x=w<FRT_VOCAB?bf16_f(logits[w]):-as_type<float>(0x7f800000u);xs[j]=x;tmax=max(tmax,x);}
        float smax=simd_max(tmax);
        if(lane==0)reduction[simd]=smax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gmax=reduction[0];for(uint i=1;i<8;++i)gmax=max(gmax,reduction[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float se=0.0f;for(uint j=0;j<FRT_PER_THREAD;++j)se+=metal::precise::exp(xs[j]-gmax);
        se=simd_sum(se);
        if(lane==0)reduction[simd]=se;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gsum=0.0f;for(uint i=0;i<8;++i)gsum+=reduction[i];
        uint removed=0u;
        for(uint r=0;r<frt_k;++r) {
            float best=-as_type<float>(0x7f800000u);uint bw=0u,bj=0u;bool any=false;
            for(uint j=0;j<FRT_PER_THREAD;++j) {
                uint w=index+j*FRT_THREADS;
                if(((removed>>j)&1u)==0u && w<FRT_VOCAB && (!any || xs[j]>best || (xs[j]==best && w>bw))){best=xs[j];bw=w;bj=j;any=true;}
            }
            float peak=simd_max(best);
            uint win=simd_max((any && best==peak)?bw:0u);
            if(any && bw==win && best==peak)removed|=1u<<bj;
            if(lane==0){frt_cv[simd*32u+r]=peak;frt_cw[simd*32u+r]=win;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float* cand=partials;
        if(simd==0) {
            uint ptr=0u;
            for(uint r=0;r<frt_k;++r) {
                float hv=(lane<8u && ptr<frt_k)?frt_cv[lane*32u+ptr]:-as_type<float>(0x7f800000u);
                uint hw=(lane<8u && ptr<frt_k)?frt_cw[lane*32u+ptr]:0u;
                float peak=simd_max(hv);
                uint win=simd_max(hv==peak?hw:0u);
                if(lane<8u && ptr<frt_k && hv==peak && hw==win)++ptr;
                if(lane==0){cand[(group*32u+r)*2u]=peak;((device uint*)cand)[(group*32u+r)*2u+1u]=win;}
            }
            if(lane==0){statistics[group*2]=gmax;statistics[group*2+1]=gsum;}
        }
        SYNC();
        if(simd==0) {
            float M=-as_type<float>(0x7f800000u);
            for(uint g=lane;g<FRT_WORKERS;g+=32u)M=max(M,statistics[g*2]);
            M=simd_max(M);
            float S=0.0f;
            for(uint g=lane;g<FRT_WORKERS;g+=32u)S+=statistics[g*2+1]*metal::precise::exp(statistics[g*2]-M);
            S=simd_sum(S);
            float lse=M+metal::precise::log(S);
            uint p0=0u,p1=0u;
            float myv=-as_type<float>(0x7f800000u);uint myw=0xffffffffu;
            for(uint r=0;r<frt_k;++r) {
                uint g0=lane,g1=lane+32u;
                bool v0=g0<FRT_WORKERS && p0<frt_k,v1=g1<FRT_WORKERS && p1<frt_k;
                float h0=v0?cand[(g0*32u+p0)*2u]:-as_type<float>(0x7f800000u);
                uint w0=v0?((device uint*)cand)[(g0*32u+p0)*2u+1u]:0u;
                float h1=v1?cand[(g1*32u+p1)*2u]:-as_type<float>(0x7f800000u);
                uint w1=v1?((device uint*)cand)[(g1*32u+p1)*2u+1u]:0u;
                bool first=v0 && (!v1 || h0>h1 || (h0==h1 && w0>w1));
                float hv=first?h0:h1;uint hw=first?w0:w1;
                bool live=v0||v1;
                float peak=simd_max(hv);
                uint win=simd_max((live && hv==peak)?hw:0u);
                if(live && hv==peak && hw==win){if(first)++p0;else ++p1;}
                if(lane==r){myv=peak;myw=win;}
            }
            float lp=myv-lse;
            float prob=lane<frt_k?metal::precise::exp(lp):0.0f;
            float before=simd_prefix_exclusive_sum(prob);
            bool kept=lane<frt_k && (frt_p>=1.0f || (1.0f-before)>(1.0f-frt_p));
            float score=kept?lp*frt_inv_t+sampling[step*FRT_VOCAB+myw]:-as_type<float>(0x7f800000u);
            float best=simd_max(score);
            uint pick=simd_min(score==best && kept?myw:0xffffffffu);
            if(lane==0)reduction[32]=as_type<float>(pick);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_token=as_type<uint>(reduction[32]);
        }
#else
        {
        // Greedy: group maxima (ties: lowest id), one barrier, then every group picks the same token.
        float best=-as_type<float>(0x7f800000u);
        uint selected=0xffffffffu;
        for(uint word=index;word<FRT_VOCAB;word+=FRT_THREADS) {
            float value=bf16_f(logits[word]);
            if(value>best || (value==best && word<selected)){best=value;selected=word;}
        }
        float peak=simd_max(best);
        uint first=simd_min(best==peak?selected:0xffffffffu);
        threadgroup uint* words=(threadgroup uint*)scratch;
        if(lane==0){reduction[simd]=peak;words[simd]=first;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simd==0) {
            float value=lane<8?reduction[lane]:-as_type<float>(0x7f800000u);
            uint word=lane<8?words[lane]:0xffffffffu;
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0){statistics[group*2]=maximum;((device uint*)statistics)[group*2+1]=choice;}
        }
        SYNC();
        if(simd==0) {
            float value=FRT_MIN;uint word=0xffffffffu;
            for(uint g=lane;g<FRT_WORKERS;g+=32u) {
                float gv=statistics[g*2];uint gw=((device uint*)statistics)[g*2+1];
                if(gv>value || (gv==value && gw<word)){value=gv;word=gw;}
            }
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)reduction[32]=as_type<float>(choice);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_token=as_type<uint>(reduction[32]);
        }
#endif
        if(index==0){packet[step*2+1]=input_token;packet[16]=step+1u;}
        bool stop=false;
        for(uint e=0;e<params.eos_count;++e)stop=stop||input_token==params.eos[e];
        if(stop)break;
    }
    if(tid==0)diagnostic[group]=0;
}
)MSL";
}

}  // namespace flashrt::apple::shaders
