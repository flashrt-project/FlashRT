#pragma once
#include "common/source.h"
#include <string>

// Resident decode step for an int4-affine (group 64) MoE decoder with GQA sliding/global attention,
// attention sinks and Q/K RMSNorm: one dispatch of 36 co-resident threadgroups runs up to eight
// consecutive tokens (embedding, every layer, LM head, token selection) separated by device-scope
// barriers. Weights: affine 4-bit codes + BF16 scale/bias per 64 inputs (int8 embedding), BF16 router;
// activations, residual stream and KV cache in BF16; FP32 accumulation.
//
// Portions of the schedule and of the arithmetic of the weight products, norms, routing and expert
// combination are adapted from Recursive's Apache-2.0 rec decode megakernel (github.com/recursive-oss/rec),
// so that with exact_attention every step reproduces rec's logits and residual stream bit for bit. The
// default attention path replaces the global-layer partial attention above 1024 keys with an 8x8 FP32
// simdgroup-MMA kernel.
//
// Geometry, fixed in this source and checked against the checkpoint by the Flint Apple runtime
// (flash_rt.models.flint.pipeline_mps.check_geometry) before compilation: 49 layers, hidden 2048, 32 query /
// 4 KV heads of 128, 256 experts top-6 of width 512, shared expert 1024, dense layers 6144, vocabulary
// 150144, window 513, RMSNorm eps 1e-6, RoPE theta 10000.

namespace flashrt::apple::shaders {

struct FlintDecodeOptions {
    bool sampled = false;         // fused top-k/top-p gumbel selection instead of greedy argmax
    bool exact_attention = false; // rec's partial-attention algorithm for every layer and length
    bool trace = false;           // write the residual stream after the embedding and every layer
};

inline std::string flint_decode_resident_bf16_source(const FlintDecodeOptions& o = {}) {
    // 36 co-resident threadgroups of 256 threads (the runtime's grid); global layers above 1024 keys use
    // the simdgroup-MMA attention unless exact_attention is set.
    std::string defines = std::string("\n#define FRT_WORKERS 36u\n#define FRT_MMA_MIN_KEYS 1024u") +
        "\n#define FRT_SAMPLED " + std::to_string(o.sampled ? 1 : 0) +
        "\n#define FRT_EXACT_ATTENTION " + std::to_string(o.exact_attention ? 1 : 0) +
        "\n#define FRT_TRACE " + std::to_string(o.trace ? 1 : 0) + "\n";
    return std::string(metal_preamble) + defines + R"MSL(
typedef bfloat bfloat16_t;
constant constexpr uint FRT_GROUPS=FRT_WORKERS*8u;        // SIMD groups in the grid
constant constexpr uint FRT_TEAM_GROUPS=FRT_WORKERS/4u;   // threadgroups per KV team (one KV head)
constant constexpr uint FRT_TEAM_SIMDS=FRT_TEAM_GROUPS*8u;
constant constexpr uint FRT_THREADS=FRT_WORKERS*256u;
constant constexpr uint FRT_VOCAB=150144u;
constant constexpr uint FRT_LAYERS=49u;
constant constexpr float FRT_EPS=1e-06f;
constant constexpr float FRT_LOGBASE=13.287712379549449f;   // log2(rope_theta=10000)
constant constexpr uint FRT_SPIN_LIMIT=10000000u;

// ------------------------------------------------------------------------------------------ layout
// Layer descriptor (ulong[64]): byte offsets into the weight buffer, element offsets into the KV buffer.
enum : uint {
    D_FLAGS=0, D_IN_NORM=1, D_POST_NORM=2, D_Q_NORM=3, D_K_NORM=4, D_SINKS=5, D_ROUTER=6, D_BIAS=7,
    D_Q=8, D_K=12, D_V=16, D_O=20, D_GATE=24, D_UP=28, D_DOWN=32, D_SH_GATE=36, D_SH_UP=40, D_SH_DOWN=44,
    D_KV=48, D_KV_CAP=49,
    // final descriptor (index 49)
    D_FINAL_NORM=1, D_EMBED=8, D_HEAD=12
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
struct KVView {
    device bfloat16_t* base;
    ulong offset;
    uint cap;
    KVView head(uint index,uint dim) const {KVView v=*this;v.offset+=ulong(index)*cap*128u+dim;return v;}
    KVView value() const {KVView v=*this;v.offset+=ulong(cap)*512u;return v;}
    const device bfloat16_t* address(uint i) const {return base+offset+i;}
    uint2 load4(uint i) const {return *((const device uint2*)(address(i)));}
    void store(uint head,uint position,uint dim,bfloat16_t value) const {
        *((device bfloat16_t*)this->head(head,dim).address(position*128))=value;
    }
};
struct DecodeParams {
    uint token, position, steps, eos_count;
    uint eos[4];
    uint top_k; float top_p, inverse_temperature;
    uint initial;          // ring layout of sliding layers: prompt length of the last prefill (0: none)
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
inline bool team_sync(device atomic_uint* control,uint tid,thread uint& phase,threadgroup uint* ready,uint group) {
    device atomic_uint* team=control+4+(group/FRT_TEAM_GROUPS)*2;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*FRT_TEAM_GROUPS;
        atomic_fetch_add_explicit(team,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(team,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<FRT_SPIN_LIMIT) ++spins;
        ready[0]=uint(atomic_load_explicit(team,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;
}
// Expert teams (4 groups per routed rank, 8 for the shared expert) publish gate/up completion per layer.
inline bool expert_handoff(device uint* routing,device atomic_uint* control,uint tid,uint group,uint layer,
                           uint phase,threadgroup uint* ready) {
    uint first=group>=24?24:(group/4)*4,count=group>=24?8:4;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),layer+1,memory_order_relaxed);
        bool complete=true;
        for(uint peer=first;peer<first+count;++peer) {
            uint spins=0;
            while(atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)<layer+1 &&
                  atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<FRT_SPIN_LIMIT) ++spins;
            complete=complete && atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)>=layer+1;
        }
        ready[0]=uint(complete && atomic_load_explicit(control+2,memory_order_relaxed)==0);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    return ready[0]!=0;
}

// ------------------------------------------------------------------------------ BF16 arithmetic
// The export rounds every BF16 operation to BF16 (each op computed in FP32 and rounded once; s*q+b fuses
// into one FMA). Values are carried in FP32 registers holding BF16-exact numbers: rb() rounds with the
// hardware float->bfloat conversion and widens the result immediately. No 16-bit value stays in a
// register: packing two BF16 values into one register is miscompiled in this kernel by the macOS 26 GPU
// compiler (odd/even element corruption).
inline float rb(float v){return as_type<float>(uint(as_type<ushort>(bfloat16_t(v)))<<16);}
inline float bf16_f(bfloat16_t x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float lo_f(uint w){return as_type<float>(w<<16);}
inline float hi_f(uint w){return as_type<float>(w&0xffff0000u);}
inline float frt_absf(float x){return as_type<float>(as_type<uint>(x)&0x7fffffffu);}
// SiLU(g)*u for BF16 inputs: exp, 1+exp, reciprocal, 1-tail, g*sig and *u each rounded to BF16.
inline float swiglu_bf16(float g,float u,bool fast) {
    float e=rb(fast?metal::fast::exp(frt_absf(g)):metal::exp(frt_absf(g)));
    float tail=rb(1.0f/rb(1.0f+e));
    float sig=g<0?tail:rb(1.0f-tail);
    return rb(rb(g*sig)*u);
}

// ------------------------------------------------------------------------------ int4 affine products
// Activations are pre-divided by 1, 16, 256, 4096 so masked nibbles need no shift; codes become exact
// FP32 integers through the 2^23 exponent trick. scale*sum(x*q) + bias*sum(x) per 64-input group.
inline float load16(const device bfloat16_t* x,thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        float x0=bf16_f(x[i]),x1=bf16_f(x[i+1]),x2=bf16_f(x[i+2]),x3=bf16_f(x[i+3]);
        sum+=rb(rb(rb(x0+x1)+x2)+x3);
        v[i]=x0;v[i+1]=x1/16.0f;v[i+2]=x2/256.0f;v[i+3]=x3/4096.0f;
    }
    return sum;
}
inline float load16_local(const threadgroup bfloat16_t* x,thread float* v) {
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
        a+=x[4*j]*(as_type<float>(0x4b000000u|(uint(p[j])&15u))-8388608.0f)+x[4*j+1]*(as_type<float>(0x4b000000u|(uint(p[j])&240u))-8388608.0f)
           +x[4*j+2]*(as_type<float>(0x4b000000u|(uint(p[j])&3840u))-8388608.0f)+x[4*j+3]*(as_type<float>(0x4b000000u|(uint(p[j])&61440u))-8388608.0f);
    return s*a+sum*b;
}
template<uint K>
inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r){float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);}
    }
}
template<uint K>
inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16_local(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r){float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);}
    }
}
template<uint K>
inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,uint lane,uint first) {
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        float v[16];float sum=load16_local(x+scalar,v);
        for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
        }
    }
    for(uint r=0;r<4;++r){float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);}
}

// ------------------------------------------------------------------------------------------- norms
// Each thread holds 8 of the 2048 residual values; two leaf sums per SIMD group, one FP32 reduction.
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
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=bfloat16_t(rb(bf16_f(w[i])*rb(x[j]*inverse)));}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Per-head RMSNorm; on rotary layers NeoX RoPE over pairs (d, d+64).
inline void head_norm(device bfloat16_t* p,const device bfloat16_t* w,bool rotary,uint position,uint lane) {
    float squared=0.0f;for(uint j=0;j<4;++j){float v=bf16_f(p[lane*4+j]);squared+=v*v;}
    float inverse=metal::precise::rsqrt(simd_sum(squared)/128.0f+FRT_EPS);
    if(rotary) {
        for(uint j=0;j<2;++j) {
            uint pair=lane*2+j;
            float a=rb(bf16_f(w[pair])*rb(bf16_f(p[pair])*inverse));
            float b=rb(bf16_f(w[pair+64])*rb(bf16_f(p[pair+64])*inverse));
            float theta=float(position)*metal::exp2(-float(pair)/64.0f*FRT_LOGBASE);
            float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
            p[pair]=bfloat16_t(rb(a*c-b*s));
            p[pair+64]=bfloat16_t(rb(a*s+b*c));
        }
    } else for(uint j=0;j<4;++j){uint d=lane*4+j;p[d]=bfloat16_t(rb(bf16_f(w[d])*rb(bf16_f(p[d])*inverse)));}
}

// --------------------------------------------------------------------------------------- attention
// Partial attention (one KV team per KV head): each task covers `parts`-strided keys for HEADS query heads.
// ROUNDED stores BF16 partial numerators (global layers from 1024 keys), otherwise FP32.
template<uint HEADS,bool ROUNDED>
inline void attention_partial(const device bfloat16_t* q,KVView k,KVView v,const device bfloat16_t* sinks,
                              device float* partial,device float* stats,uint count,uint parts,uint lane,uint sg) {
    for(uint task=(sg/FRT_TEAM_SIMDS)*((8/HEADS)*parts)+sg%FRT_TEAM_SIMDS;
        task<(sg/FRT_TEAM_SIMDS+1)*((8/HEADS)*parts);task+=FRT_TEAM_SIMDS) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*4);
        KVView value_base=v.head(kv,lane*4);
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?bf16_f(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j){query[h][j]=bf16_f(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;}
        uint token=part;
        if(ROUNDED && part!=0 && token<count) {
            uint at=uint(token)*128;
            uint2 pk=key_base.load4(at),pv=value_base.load4(at);
            float key[4]={lo_f(pk.x),hi_f(pk.x),lo_f(pk.y),hi_f(pk.y)},value[4]={lo_f(pv.x),hi_f(pv.x),lo_f(pv.y),hi_f(pv.y)};
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        uint2 next_key=uint2(0),next_value=uint2(0);
        if(token<count){next_key=key_base.load4(uint(token)*128);next_value=value_base.load4(uint(token)*128);}
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            uint at=uint(token)*128;
            float key[4]={lo_f(next_key.x),hi_f(next_key.x),lo_f(next_key.y),hi_f(next_key.y)};
            float value[4]={lo_f(next_value.x),hi_f(next_value.x),lo_f(next_value.y),hi_f(next_value.y)};
            if(token+parts<count){next_key=key_base.load4(at+parts*128);next_value=value_base.load4(at+parts*128);}
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h){float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];scores[h]=simd_sum(score);}
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*value[j];
            }
        }
        if(lane<HEADS){device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;stat_out[0]=hi;stat_out[1]=z;}
        for(uint h=0;h<HEADS;++h) {
            if(ROUNDED) {
                device uint2* out=(device uint2*)(((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4);
                uint r0=as_type<uint>(rb(acc[h][0]))>>16,r1=as_type<uint>(rb(acc[h][1]))&0xffff0000u;
                uint r2=as_type<uint>(rb(acc[h][2]))>>16,r3=as_type<uint>(rb(acc[h][3]))&0xffff0000u;
                *out=uint2(r0|r1,r2|r3);
            } else {
                device float* out=partial+((first+h)*parts+part)*128+lane*4;
                *((device float4*)out)=float4(acc[h][0],acc[h][1],acc[h][2],acc[h][3]);
            }
        }
    }
}
// Merge the partitions of each head (team-local: 8 heads x 8 dimension slices of 16).
template<bool ROUNDED>
inline void attention_merge(device bfloat16_t* out,device float* partial,device float* stats,uint parts,uint lane,uint sg) {
    for(uint task=(sg/FRT_TEAM_SIMDS)*64+sg%FRT_TEAM_SIMDS;task<(sg/FRT_TEAM_SIMDS+1)*64;task+=FRT_TEAM_SIMDS) {
        uint h=task/8,dim=(task%8)*16;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);
        float z=0.0f;
        float4 a[4]={float4(0),float4(0),float4(0),float4(0)};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;
            float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint tile=0;tile<4;++tile) {
                float4 value;
                if(ROUNDED) {
                    uint at=((h*32+dim/4+tile)*parts+p)*4;
                    uint2 raw=*((const device uint2*)(((const device bfloat16_t*)partial)+at));
                    value=float4(lo_f(raw.x),hi_f(raw.x),lo_f(raw.y),hi_f(raw.y));
                } else {
                    value=*((const device float4*)(partial+(h*parts+p)*128+dim+tile*4));
                }
                a[tile]+=factor*value;
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint tile=0;tile<4;++tile)for(uint component=0;component<4;++component) {
            uint j=tile*4+component;
            float total=simd_sum(a[tile][component]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<8)for(uint t=0;t<2;++t){uint j=lane*2+t;out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);}
    }
}
// Global-layer attention with 8x8 FP32 simdgroup MMA. One contiguous range of 8-key tiles per SIMD group
// of the KV team: S = Q.K^T over 16 k-steps, online softmax per query-head row, O += P.V over 16 n-tiles.
// Dimension permutations make every lane's K/V loads contiguous; Q fragments live in threadgroup memory.
inline float bf16_lo(uint w){return as_type<float>(w<<16);}
inline float bf16_hi(uint w){return as_type<float>(w&0xffff0000u);}
inline void attention_mma(const device bfloat16_t* q,KVView keys,KVView vals,const device bfloat16_t* sinks,
                          device float* partial,device float* stats,uint count,uint lane,uint sg,uint tid,
                          threadgroup float* qs) {
    const uint kv=sg/FRT_TEAM_SIMDS,part=sg%FRT_TEAM_SIMDS,parts=FRT_TEAM_SIMDS;
    const uint qid=lane/4,fm=(qid&4)+((lane/2)%4),fn=(qid&2)*2+(lane%2)*2;
    const uint head=kv*8+fm;
    for(uint idx=tid;idx<1024u;idx+=256u) {
        uint k=idx/64u,r=(idx/8u)%8u,c=idx%8u;
        qs[idx]=bf16_f(q[(kv*8u+r)*128u+16u*c+k])*0.08838834764831845f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    KVView kview=keys.head(kv,0),vview=vals.head(kv,0);
    simdgroup_float8x8 o[16];
    for(uint n=0;n<16;++n)o[n]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    float m=-3.402823466e+38f,l=0.0f;
    if(part==0){m=bf16_f(sinks[head]);l=fn==0?1.0f:0.0f;}
    const uint tiles=(count+7)/8,base=tiles/parts,rem=tiles%parts;
    const uint first=part*base+min(part,rem),mine=base+(part<rem?1u:0u);
    const uint last=count-1;
    for(uint i=0;i<mine;++i) {
        uint tok=(first+i)*8;
        const device uint4* r0=(const device uint4*)(kview.address(min(tok+fn,last)*128u)+16*fm);
        const device uint4* r1=(const device uint4*)(kview.address(min(tok+fn+1,last)*128u)+16*fm);
        uint4 ck0[2]={r0[0],r0[1]},ck1[2]={r1[0],r1[1]};
        simdgroup_float8x8 s=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for(uint k=0;k<16;++k) {
            uint w0=ck0[k/8][(k%8)/2],w1=ck1[k/8][(k%8)/2];
            simdgroup_float8x8 kb,qa;
            simdgroup_load(qa,qs+k*64u,8);
            kb.thread_elements()[0]=(k%2)?bf16_hi(w0):bf16_lo(w0);
            kb.thread_elements()[1]=(k%2)?bf16_hi(w1):bf16_lo(w1);
            simdgroup_multiply_accumulate(s,qa,kb,s);
        }
        float s0=s.thread_elements()[0],s1=s.thread_elements()[1];
        if(tok+fn>=count)s0=-as_type<float>(0x7f800000u);
        if(tok+fn+1>=count)s1=-as_type<float>(0x7f800000u);
        float mt=max(s0,s1);
        mt=max(mt,simd_shuffle_xor(mt,1));
        mt=max(mt,simd_shuffle_xor(mt,8));
        float mn=max(m,mt);
        float alpha=metal::fast::exp(m-mn);
        float p0=metal::fast::exp(s0-mn),p1=metal::fast::exp(s1-mn);
        l=l*alpha+p0+p1;m=mn;
        simdgroup_float8x8 pa;
        pa.thread_elements()[0]=p0;pa.thread_elements()[1]=p1;
        bool vvalid=tok+fm<count;
        const device uint4* rv=(const device uint4*)(vview.address(min(tok+fm,last)*128u)+32*(fn/2));
        for(uint g=0;g<4;++g) {
            uint4 cv=rv[g];
            for(uint h=0;h<4;++h) {
                uint n=g*4+h;
                o[n].thread_elements()[0]*=alpha;o[n].thread_elements()[1]*=alpha;
                simdgroup_float8x8 vb;
                vb.thread_elements()[0]=vvalid?bf16_lo(cv[h]):0.0f;
                vb.thread_elements()[1]=vvalid?bf16_hi(cv[h]):0.0f;
                simdgroup_multiply_accumulate(o[n],pa,vb,o[n]);
            }
        }
    }
    l+=simd_shuffle_xor(l,1);
    l+=simd_shuffle_xor(l,8);
    uint row=head*parts+part;
    if(fn==0){stats[row*2]=m;stats[row*2+1]=l;}
    device float2* out=(device float2*)(partial+row*128+32*(fn/2));
    for(uint n=0;n<16;++n)out[n]=float2(o[n].thread_elements()[0],o[n].thread_elements()[1]);
}

// ------------------------------------------------------------------------------------------- MoE
inline void route_dot(const threadgroup bfloat16_t* x,const device bfloat16_t* w,device float* probs,uint lane,uint sg) {
    for(uint row=sg;row<256;row+=FRT_GROUPS) {
        float a=0.0f;
        for(uint b=0;b<16;++b) {
            uint at=b*128+lane*4;
            const uint2 packed=*((const device uint2*)(w+row*2048+at));
            const float4 weights=float4(lo_f(packed.x),hi_f(packed.x),lo_f(packed.y),hi_f(packed.y));
            for(uint j=0;j<4;++j)a+=weights[j]*bf16_f(x[at+j]);
        }
        for(ushort d=16;d>=1;d>>=1)a+=simd_shuffle_down(a,d);
        a=rb(a);
        if(lane==0){float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(a)));probs[row]=a<0?tail:1.0f-tail;}
    }
}
// Each routed expert team selects only its own rank among the top-6 (ties: lowest expert id first).
inline void route_own_rank(const device float* bias,device float* probs,device float* chosen,device uint* ids,
                           uint lane,uint sg,uint group) {
    if(sg==0 && group<24) {
        uint role=group/4;
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<=role;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            if(rank==role) {
                float value=simd_shuffle(p[selected/32],selected%32);
                if(lane==0){ids[role]=selected;if(group%4==0)chosen[role]=value;}
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}
inline void expert_gateup(const device uchar* weights,const threadgroup ulong* d,const threadgroup bfloat16_t* x,
                          const device uint* ids,device bfloat16_t* out,uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4,team=shared?group-24:group%4;
    uint teams=shared?8:4,width=shared?1024:512;
    Matrix g=weight_matrix(weights,d+(shared?D_SH_GATE:D_GATE));
    Matrix u=weight_matrix(weights,d+(shared?D_SH_UP:D_UP));
    uint expert=shared?0:ids[role];
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*2;local<width;local+=teams*8*2) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0)out[role*512+local+j]=bfloat16_t(swiglu_bf16(rb(val.x),rb(val.y),true));
        }
    }
}
inline void expert_down(const device uchar* weights,const threadgroup ulong* d,const device bfloat16_t* x,
                        const device uint* ids,device bfloat16_t* out,uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4,team=shared?group-24:group%4;
    uint teams=shared?8:4;
    Matrix m=weight_matrix(weights,d+(shared?D_SH_DOWN:D_DOWN));uint expert=shared?0:ids[role];
    const device bfloat16_t* input=x+role*512;
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*4;local<2048;local+=teams*8*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(v);}
    }
}

// ------------------------------------------------------------------------------------------ kernel
#define FRT_FAIL() {if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!grid_sync(control,tid,phase,ready))FRT_FAIL()
#define TEAM_SYNC() if(!team_sync(control,tid,team_phase,ready,group))FRT_FAIL()

kernel void flint_decode_resident_bf16(
    const device uchar* weights [[buffer(0)]],
    const device ulong* layers [[buffer(1)]],
    device bfloat16_t* kv [[buffer(2)]],
    constant DecodeParams& params [[buffer(3)]],
    const device float* sampling [[buffer(4)]],
    device bfloat16_t* block_logits [[buffer(5)]],
    device bfloat16_t* block_trace [[buffer(6)]],
    device uint* packet [[buffer(7)]],
    device bfloat16_t* workspace [[buffer(8)]],
    device float* partials [[buffer(9)]],
    device float* statistics [[buffer(10)]],
    device uint* routing [[buffer(11)]],
    device uint* private_control [[buffer(12)]],
    device uint* diagnostic [[buffer(13)]],
    uint3 group3 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint group=group3.x,sg=group*8+simd,index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];
    uint phase=0,team_phase=0,qkv_base=0;
    threadgroup float reduction[33];
    threadgroup ulong layer_descriptor[64];
    threadgroup float frt_qs[1024];
#if FRT_SAMPLED
    threadgroup float frt_cv[256];threadgroup uint frt_cw[256];
#endif
    device uint* ids=routing+group*16;
    if(tid==0)atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    float h[8];
    alignas(8) threadgroup bfloat16_t n[2048];
    device bfloat16_t* q=workspace+4096;
    device bfloat16_t* k=workspace+8192;
    device bfloat16_t* v=workspace+8704;
    device bfloat16_t* a=workspace+9216;
    device bfloat16_t* o=workspace+13312;
    device bfloat16_t* gate=workspace+15360;
    device bfloat16_t* up=workspace+31744;
    device bfloat16_t* act=workspace+48128;
    device bfloat16_t* down=workspace+64512;
    const device ulong* final=layers+49*64;
    const Matrix embedding=weight_matrix(weights,final+D_EMBED);
    uint input_token=params.token;
    #pragma clang loop unroll(disable)
    for(uint step=0;step<params.steps;++step) {
        const uint position=params.position+step;
        device bfloat16_t* logits=block_logits+step*FRT_VOCAB;
#if FRT_TRACE
        device bfloat16_t* trace=block_trace+step*(50u*2048u);
#endif
        if(index==0)packet[step*2]=input_token;
        for(uint j=0;j<8;++j) {
            uint i=(tid+(j/4)*256)*4+j%4;
            size_t row=input_token;uint code=embedding.w[row*2048+i];
            uint gi=row*32+i/64;
            // s*q+b contracts to one fused multiply-add in the export's arithmetic (one rounding).
            h[j]=rb(metal::fma(bf16_f(embedding.s[gi]),float(code),bf16_f(embedding.b[gi])));
#if FRT_TRACE
            if(group==0)trace[i]=bfloat16_t(h[j]);
#endif
        }
        SYNC();
        for(uint layer=0;layer<FRT_LAYERS;++layer) {
            if(tid<64)layer_descriptor[tid]=layers[layer*64+tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const threadgroup ulong* d=layer_descriptor;
            const bool global=(d[D_FLAGS]&1)!=0,sparse=(d[D_FLAGS]&2)!=0;
            rms_2048(h,(const device bfloat16_t*)(weights+d[D_IN_NORM]),n,tid,lane,simd,reduction);
            const uint cap=uint(d[D_KV_CAP]),count=global?position+1:min(position+1,cap);
            // Sliding ring slot: (position - initial) mod cap once a prefill of >= cap tokens laid the ring
            // out linearly, else position mod cap.
            const uint insert=global?position:(params.initial>=cap?(position%cap+cap-params.initial%cap)%cap:position%cap);
            KVView keys={kv,d[D_KV],cap};
            KVView vals=keys.value();
            // Q/K/V: 320 four-row units per KV team (8 query heads, its K and V head), claimed by ticket.
            const uint kv_team=group/FRT_TEAM_GROUPS,kv_member=group%FRT_TEAM_GROUPS;
            for(;;) {
                uint ticket=0;
                if(lane==0)ticket=atomic_fetch_add_explicit(control+40+kv_team,1u,memory_order_relaxed);
                ticket=simd_broadcast(ticket,0);
                if(ticket>=qkv_base+320u)break;
                uint unit=ticket-qkv_base,member=unit/32u;
                uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
                uint local=head<32?head:head<36?head-32:head-36;
                Matrix m=weight_matrix(weights,d+(head<32?D_Q:head<36?D_K:D_V));
                m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
                q4_local_block<2048>(m,n,(head<32?q:head<36?k:v)+local*128,lane,(unit%32u)*4u);
            }
            qkv_base+=320u+FRT_TEAM_GROUPS*8u;
            TEAM_SYNC();
            for(uint member=kv_member;member<10;member+=FRT_TEAM_GROUPS) {
                uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
                uint local=head<32?head:head<36?head-32:head-36;
                device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
                if(simd==0 && head<36)
                    head_norm(out,(const device bfloat16_t*)(weights+d[head<32?D_Q_NORM:D_K_NORM]),!global,position,lane);
                threadgroup_barrier(mem_flags::mem_device);
                if(head>=32) {
                    KVView target=head<36?keys:vals;
                    for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
                }
            }
            TEAM_SYNC();
            bool rounded=global&&count>=1024;
            uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
            const device bfloat16_t* sinks=(const device bfloat16_t*)(weights+d[D_SINKS]);
#if FRT_EXACT_ATTENTION
            const bool use_mma=false;
#else
            const bool use_mma=global&&count>FRT_MMA_MIN_KEYS;
#endif
            if(use_mma){rounded=false;parts=FRT_TEAM_SIMDS;}
            if(use_mma) attention_mma(q,keys,vals,sinks,partials,statistics,count,lane,sg,tid,frt_qs);
            else if(global && count>1024) {
                if(rounded) attention_partial<8,true>(q,keys,vals,sinks,partials,statistics,count,parts,lane,sg);
                else attention_partial<8,false>(q,keys,vals,sinks,partials,statistics,count,parts,lane,sg);
            } else {
                if(rounded) attention_partial<4,true>(q,keys,vals,sinks,partials,statistics,count,parts,lane,sg);
                else attention_partial<4,false>(q,keys,vals,sinks,partials,statistics,count,parts,lane,sg);
            }
            TEAM_SYNC();
            if(rounded) attention_merge<true>(a,partials,statistics,parts,lane,sg);
            else attention_merge<false>(a,partials,statistics,parts,lane,sg);
            SYNC();
            q4<4096>(weight_matrix(weights,d+D_O),a,o,lane,sg,FRT_GROUPS);
            SYNC();
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=rb(h[j]+bf16_f(o[i]));}
            rms_2048(h,(const device bfloat16_t*)(weights+d[D_POST_NORM]),n,tid,lane,simd,reduction);
            if(sparse) {
                route_dot(n,(const device bfloat16_t*)(weights+d[D_ROUTER]),statistics,lane,sg);
                SYNC();
                route_own_rank((const device float*)(weights+d[D_BIAS]),statistics,(device float*)(routing+8),ids,lane,simd,group);
                if(group<32) {
                    expert_gateup(weights,d,n,ids,act,lane,simd,group);
                    if(!expert_handoff(routing,control,tid,group,step*FRT_LAYERS+layer,phase,ready))FRT_FAIL();
                    expert_down(weights,d,act,ids,down,lane,simd,group);
                }
                SYNC();
                // Combine: renormalized top-6 weights (scores of group 0's teams), each weighted expert
                // product and every partial sum rounded to BF16, then the shared expert and the residual.
                float chosen=lane<6?((device float*)(routing+8))[lane]:0.0f;
                float total=0.0f;
                for(uint j=0;j<6;++j)total=simd_shuffle(chosen,j)+total;
                total+=1.0e-20f;
                float weight=lane<6?chosen/total:0.0f;
                float sums[8];
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    sums[j]=rb(bf16_f(down[i])*rb(simd_shuffle(weight,ushort(0))))+0.0f;
                }
                for(uint j=0;j<8;++j)for(uint r=1;r<6;++r) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    sums[j]=rb((rb(bf16_f(down[r*2048+i])*rb(simd_shuffle(weight,ushort(r))))+0.0f)+sums[j]);
                }
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    float mlp=rb(rb(sums[j])+bf16_f(down[6*2048+i]));
                    h[j]=rb(h[j]+mlp);
#if FRT_TRACE
                    if(group==0)trace[(layer+1)*2048+i]=bfloat16_t(h[j]);
#endif
                }
            } else {
                Matrix gm=weight_matrix(weights,d+D_GATE),um=weight_matrix(weights,d+D_UP),dm=weight_matrix(weights,d+D_DOWN);
                q4_local<2048>(gm,n,gate,lane,sg,FRT_GROUPS);q4_local<2048>(um,n,up,lane,sg,FRT_GROUPS);
                SYNC();
                for(uint i=index;i<gm.n;i+=FRT_THREADS) {
                    act[i]=bfloat16_t(swiglu_bf16(bf16_f(gate[i]),bf16_f(up[i]),false));
                }
                SYNC();
                q4<6144>(dm,act,down,lane,sg,FRT_GROUPS);
                SYNC();
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    h[j]=rb(h[j]+bf16_f(down[i]));
#if FRT_TRACE
                    if(group==0)trace[(layer+1)*2048+i]=bfloat16_t(h[j]);
#endif
                }
            }
        }
        rms_2048(h,(const device bfloat16_t*)(weights+final[D_FINAL_NORM]),n,tid,lane,simd,reduction);
        if(group<32) q4_local<2048>(weight_matrix(weights,final+D_HEAD),n,logits,lane,sg,256);
        SYNC();
#if FRT_SAMPLED
        // Fused top-k/top-p: group-local max, sum of exponentials and sorted top-K, one barrier, then every
        // group merges the per-group lists and selects the same token. Ties rank the larger token id first
        // (mlx-lm top-p: ascending argsort, cumulative > 1-p); gumbel noise comes from `sampling`.
        {
        const uint frt_k=params.top_k;
        const float frt_p=params.top_p,frt_inv_t=params.inverse_temperature;
        float xs[17];
        float tmax=-as_type<float>(0x7f800000u);
        for(uint j=0;j<17u;++j){uint w=index+j*FRT_THREADS;float x=w<FRT_VOCAB?float(logits[w]):-as_type<float>(0x7f800000u);xs[j]=x;tmax=max(tmax,x);}
        float smax=simd_max(tmax);
        if(lane==0)reduction[simd]=smax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gmax=reduction[0];for(uint i=1;i<8;++i)gmax=max(gmax,reduction[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float se=0.0f;for(uint j=0;j<17u;++j)se+=metal::precise::exp(xs[j]-gmax);
        se=simd_sum(se);
        if(lane==0)reduction[simd]=se;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gsum=0.0f;for(uint i=0;i<8;++i)gsum+=reduction[i];
        uint removed=0u;
        for(uint r=0;r<frt_k;++r) {
            float best=-as_type<float>(0x7f800000u);uint bw=0u,bj=0u;bool any=false;
            for(uint j=0;j<17u;++j) {
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
            float value=float(logits[word]);
            if(value>best || (value==best && word<selected)){best=value;selected=word;}
        }
        float peak=simd_max(best);
        uint first=simd_min(best==peak?selected:0xffffffffu);
        threadgroup uint* words=(threadgroup uint*)frt_qs;
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
            float value=-as_type<float>(0x7f800000u);
            uint word=0xffffffffu;
            for(uint peer=lane;peer<FRT_WORKERS;peer+=32) {
                float other=statistics[peer*2];
                uint other_word=((device uint*)statistics)[peer*2+1];
                if(other>value || (other==value && other_word<word)){value=other;word=other_word;}
            }
            float maximum=simd_max(value);
            uint choice=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)words[0]=choice;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        input_token=words[0];
        }
#endif
        if(index==0){packet[step*2+1]=input_token;packet[16]=step+1u;}
        bool stop=false;
        for(uint j=0;j<params.eos_count;++j)stop=stop||input_token==params.eos[j];
        if(stop)break;
    }
    if(tid==0)diagnostic[group]=0;
}
)MSL";
}

}  // namespace flashrt::apple::shaders
