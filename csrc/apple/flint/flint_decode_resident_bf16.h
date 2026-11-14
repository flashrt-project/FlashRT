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
// verify_rows > 0 builds the verification step instead (flint_decode_verify_bf16): one dispatch computes 2..4
// consecutive positions (the last produced token and draft tokens), each row with the arithmetic of the greedy
// resident step at its position, reading every weight word once for all rows; see the kernel's comment.
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
    int verify_rows = 0;          // > 0: the verification step over this many consecutive rows instead
};

inline std::string flint_decode_resident_bf16_source(const FlintDecodeOptions& o = {}) {
    // 36 co-resident threadgroups of 256 threads (the runtime's grid); global layers above 1024 keys use
    // the simdgroup-MMA attention unless exact_attention is set.
    std::string defines = std::string("\n#define FRT_WORKERS 36u\n#define FRT_MMA_MIN_KEYS 1024u") +
        "\n#define FRT_SAMPLED " + std::to_string(o.sampled ? 1 : 0) +
        "\n#define FRT_EXACT_ATTENTION " + std::to_string(o.exact_attention ? 1 : 0) +
        "\n#define FRT_TRACE " + std::to_string(o.trace ? 1 : 0) +
        "\n#define FRT_VERIFY_ROWS " + std::to_string(o.verify_rows) + "\n";
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
#if FRT_VERIFY_ROWS
// The verification step rounds in integer arithmetic (nearest even): the hardware conversion's 16-bit temporaries
// of different rows get packed into one register and corrupted. Every rounded value is an FP32 arithmetic result,
// which this GPU flushes to zero below the normal range, so this equals the hardware conversion bit for bit
// (NaN payloads aside).
inline float rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
#else
inline float rb(float v){return as_type<float>(uint(as_type<ushort>(bfloat16_t(v)))<<16);}
#endif
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
// Sliding-ring slots [lo, lo + n) (mod cap) whose keys and values are read from `side` ([n][K, V][4 heads][128])
// instead of the ring: a verification step's later rows, before they are committed.
struct SideKV {
    const device bfloat16_t* side;
    uint lo, n, cap;
};
// Partial attention (one KV team per KV head): each task covers `parts`-strided keys for HEADS query heads.
// ROUNDED stores BF16 partial numerators (global layers from 1024 keys), otherwise FP32. SIDE reads the slots
// of `sk` from its side buffer.
template<uint HEADS,bool ROUNDED,bool SIDE=false>
inline void attention_partial(const device bfloat16_t* q,KVView k,KVView v,const device bfloat16_t* sinks,
                              device float* partial,device float* stats,uint count,uint parts,uint lane,uint sg,
                              SideKV sk={nullptr,0u,0u,1u}) {
    for(uint task=(sg/FRT_TEAM_SIMDS)*((8/HEADS)*parts)+sg%FRT_TEAM_SIMDS;
        task<(sg/FRT_TEAM_SIMDS+1)*((8/HEADS)*parts);task+=FRT_TEAM_SIMDS) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*4);
        KVView value_base=v.head(kv,lane*4);
#define FRT_KV_LOAD(dk,dv,tok) { \
            uint frt_t=(tok),frt_rel=frt_t>=sk.lo?frt_t-sk.lo:frt_t+sk.cap-sk.lo; \
            if(SIDE && frt_rel<sk.n) { \
                const device bfloat16_t* frt_s=sk.side+(frt_rel*8u+kv)*128u+lane*4u; \
                dk=*((const device uint2*)frt_s);dv=*((const device uint2*)(frt_s+512u)); \
            } else {dk=key_base.load4(frt_t*128u);dv=value_base.load4(frt_t*128u);} }
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?bf16_f(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j){query[h][j]=bf16_f(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;}
        uint token=part;
        if(ROUNDED && part!=0 && token<count) {
            uint2 pk,pv;
            FRT_KV_LOAD(pk,pv,token)
            float key[4]={lo_f(pk.x),hi_f(pk.x),lo_f(pk.y),hi_f(pk.y)},value[4]={lo_f(pv.x),hi_f(pv.x),lo_f(pv.y),hi_f(pv.y)};
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        uint2 next_key=uint2(0),next_value=uint2(0);
        if(token<count)FRT_KV_LOAD(next_key,next_value,token)
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            float key[4]={lo_f(next_key.x),hi_f(next_key.x),lo_f(next_key.y),hi_f(next_key.y)};
            float value[4]={lo_f(next_value.x),hi_f(next_value.x),lo_f(next_value.y),hi_f(next_value.y)};
            if(token+parts<count)FRT_KV_LOAD(next_key,next_value,token+parts)
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
#undef FRT_KV_LOAD
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


#if FRT_VERIFY_ROWS
// ------------------------------------------------------------------- multi-row products (verification)
// Each weight word is read once for FRT_ROWS activation rows; every row's sums are those of q4 / q4_local
// (dot16 on the same codes), so a row's result equals the single-row product bit for bit.
#define FRT_UNROLL _Pragma("clang loop unroll(full)")
constant constexpr uint FRT_ROWS=FRT_VERIFY_ROWS;
constant constexpr uint FRT_ROW_WS=78848u;               // workspace elements per row
constant constexpr uint FRT_ROW_PARTIALS=32u*1024u*64u;  // partial floats per row
constant constexpr uint FRT_ROW_STATS=32u*1024u*2u;      // statistic floats per row
// The codes become the same exact FP32 integers as dot16's through integer conversion: with the 2^23 constant,
// fast math factors x * 2^23 out of the rows that share x (wrong sums with three or more rows).
inline float dot16_words(uint2 c,thread float* x,float s,float b,float sum) {
    const uint p[4]={c.x&0xffffu,c.x>>16,c.y&0xffffu,c.y>>16};
    float a=0.0f;
    for(uint j=0;j<4;++j)
        a+=x[4*j]*float(p[j]&15u)+x[4*j+1]*float(p[j]&240u)+x[4*j+2]*float(p[j]&3840u)+x[4*j+3]*float(p[j]&61440u);
    return s*a+sum*b;
}
inline const device uint2* words(const device bfloat16_t* p){return (const device uint2*)p;}
inline const threadgroup uint2* words(const threadgroup bfloat16_t* p){return (const threadgroup uint2*)p;}
// Sum of 16 activations as load16 forms it (four groups of four, each rounded to BF16 per add).
template<typename P>
inline float sum16_words(P x) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        const uint2 w=x[i/4];
        sum+=rb(rb(rb(lo_f(w.x)+hi_f(w.x))+lo_f(w.y))+hi_f(w.y));
    }
    return sum;
}
// The 16 activations, pre-divided like load16's.
template<typename P>
inline void values16_words(P x,thread float* v) {
    for(uint i=0;i<16;i+=4) {
        const uint2 w=x[i/4];
        v[i]=lo_f(w.x);v[i+1]=hi_f(w.x)/16.0f;v[i+2]=lo_f(w.y)/256.0f;v[i+3]=hi_f(w.y)/4096.0f;
    }
}
// Sums table of every row (rows XS elements apart, K inputs): sums[r * K / 16 + i] = sum16 of inputs 16i..16i+15.
template<uint K,typename P>
inline void sums_table(P x,uint xs,threadgroup float* sums,uint tid) {
    for(uint i=tid;i<FRT_ROWS*(K/16u);i+=256u)
        sums[i]=sum16_words(words(x+(i/(K/16u))*xs+(i%(K/16u))*16u));
}
inline void codes16(uint2 c,thread float* q) {
    const uint p[4]={c.x&0xffffu,c.x>>16,c.y&0xffffu,c.y>>16};
    for(uint j=0;j<4;++j){q[4*j]=float(p[j]&15u);q[4*j+1]=float(p[j]&240u);q[4*j+2]=float(p[j]&3840u);q[4*j+3]=float(p[j]&61440u);}
}
// dot16's sums with the codes as FP32 integers (q) and the pre-divided activations (x).
inline float dot16_f(thread const float* q,thread const float* x,float s,float b,float sum) {
    float a=0.0f;
    for(uint j=0;j<4;++j)a+=x[4*j]*q[4*j]+x[4*j+1]*q[4*j+1]+x[4*j+2]*q[4*j+2]+x[4*j+3]*q[4*j+3];
    return s*a+sum*b;
}
// Four output rows starting at `first` for every activation row (x: rows XS apart, `sums` its sums table;
// out: rows OS apart). Each code word is converted once for all rows.
template<uint K,typename P,typename S>
inline void q4_rows_block(Matrix m,P x,uint xs,S sums,uint ss,device bfloat16_t* out,uint os,uint lane,uint first) {
    float a[FRT_ROWS][4];
    FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)FRT_UNROLL for(uint r=0;r<4;++r)a[j][r]=0.0f;
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        uint2 c[4];float sc[4],bi[4];
        FRT_UNROLL for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            c[r]=*((const device uint2*)(m.w+size_t(row)*(K/2)+scalar/2));sc[r]=float(m.s[gi]);bi[r]=float(m.b[gi]);
        }
        float v[FRT_ROWS][16],sum[FRT_ROWS];
        FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j){values16_words(words(x+j*xs+scalar),v[j]);sum[j]=sums[j*ss+scalar/16u];}
        FRT_UNROLL for(uint r=0;r<4;++r) {
            float q[16];codes16(c[r],q);
            FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)a[j][r]+=dot16_f(q,v[j],sc[r],bi[r],sum[j]);
        }
    }
    FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)FRT_UNROLL for(uint r=0;r<4;++r) {
        float v=simd_sum(a[j][r]);if(lane==0)out[j*os+first+r]=bfloat16_t(v);
    }
}
template<uint K,typename P,typename S>
inline void q4_rows(Matrix m,P x,uint xs,S sums,uint ss,device bfloat16_t* out,uint os,uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4)q4_rows_block<K>(m,x,xs,sums,ss,out,os,lane,first);
}
// rms_2048 that also writes the sums table of its output (load16's group sums of every 16 outputs): a group's
// 16 outputs are four consecutive threads' four.
inline void rms_2048_table(const thread float* x,const device bfloat16_t* w,threadgroup bfloat16_t* out,
                           threadgroup float* table,uint tid,uint lane,uint sg,threadgroup float* temp) {
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
    float y[8];
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;y[j]=rb(bf16_f(w[i])*rb(x[j]*inverse));out[i]=bfloat16_t(y[j]);}
    for(uint hf=0;hf<2;++hf) {
        const float part=rb(rb(rb(y[hf*4]+y[hf*4+1])+y[hf*4+2])+y[hf*4+3]);
        const float p1=simd_shuffle_down(part,ushort(1)),p2=simd_shuffle_down(part,ushort(2)),p3=simd_shuffle_down(part,ushort(3));
        if(lane%4u==0u)table[hf*64u+tid/4u]=(((0.0f+part)+p1)+p2)+p3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// attention_merge that also writes the sums table of its output (each task's 16 outputs are one group).
template<bool ROUNDED>
inline void attention_merge_table(device bfloat16_t* out,device float* partial,device float* stats,uint parts,
                                  device float* table,uint lane,uint sg) {
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
        float val[2];
        for(uint t=0;t<2;++t)val[t]=rb(z==0.0f?owned[t]:owned[t]/z);
        if(lane<8)for(uint t=0;t<2;++t){uint j=lane*2+t;out[h*128+dim+j]=bfloat16_t(val[t]);}
        // lanes 0..7 hold outputs 2l, 2l+1: groups of four on lanes 2m, 2m+1
        const float n0=simd_shuffle_down(val[0],ushort(1)),n1=simd_shuffle_down(val[1],ushort(1));
        const float part=rb(rb(rb(val[0]+val[1])+n0)+n1);
        const float g1=simd_shuffle_down(part,ushort(2)),g2=simd_shuffle_down(part,ushort(4)),g3=simd_shuffle_down(part,ushort(6));
        if(lane==0)table[h*8u+dim/16u]=(((0.0f+part)+g1)+g2)+g3;
    }
}
// Router scores for every row (probs: rows FRT_ROW_STATS apart).
inline void route_dot_rows(const threadgroup bfloat16_t* x,const device bfloat16_t* w,device float* probs,uint lane,uint sg) {
    for(uint row=sg;row<256;row+=FRT_GROUPS) {
        float a[FRT_ROWS];
        FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)a[j]=0.0f;
        for(uint b=0;b<16;++b) {
            uint at=b*128+lane*4;
            const uint2 packed=*((const device uint2*)(w+row*2048+at));
            const float4 weights=float4(lo_f(packed.x),hi_f(packed.x),lo_f(packed.y),hi_f(packed.y));
            FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)for(uint i=0;i<4;++i)a[j]+=weights[i]*bf16_f(x[j*2048+at+i]);
        }
        FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j) {
            float v=a[j];
            for(ushort d=16;d>=1;d>>=1)v+=simd_shuffle_down(v,d);
            v=rb(v);
            if(lane==0){float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(v)));probs[j*FRT_ROW_STATS+row]=v<0?tail:1.0f-tail;}
        }
    }
}
// Shared expert for every row (weights read once).
// Top-6 experts of one row (route_own_rank's selection for every rank) into threadgroup memory.
inline void route_all(const device float* bias,const device float* probs,threadgroup uint* ids,threadgroup float* chosen,
                      uint lane) {
    float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
    for(uint rank=0;rank<6;++rank) {
        float best=-3.402823466e+38f;uint idx=0;
        for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
        float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
        float value=simd_shuffle(p[selected/32],selected%32);
        if(lane==0){ids[rank]=selected;chosen[rank]=value;}
        for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
    }
}
// One routed expert's gate/up row pair `local` for one activation row x (expert_gateup's sums; `sums`: x's table).
inline void routed_gateup_pair(const device uchar* weights,const threadgroup ulong* d,const threadgroup bfloat16_t* x,
                               const threadgroup float* sums,uint expert,uint local,device bfloat16_t* out,uint lane) {
    Matrix g=weight_matrix(weights,d+D_GATE),u=weight_matrix(weights,d+D_UP);
    float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
    for(uint b=0;b<2048;b+=512) {
        uint scalar=b+lane*16;
        float v[16];values16_words(words(x+scalar),v);
        const float sum=sums[scalar/16u];
        FRT_UNROLL for(uint j=0;j<2;++j) {
            uint row=expert*g.n+local+j,gi=row*32+scalar/64;
            float q[16];
            codes16(*((const device uint2*)(g.w+size_t(row)*1024+scalar/2)),q);
            ga[j]+=dot16_f(q,v,float(g.s[gi]),float(g.b[gi]),sum);
            codes16(*((const device uint2*)(u.w+size_t(row)*1024+scalar/2)),q);
            ua[j]+=dot16_f(q,v,float(u.s[gi]),float(u.b[gi]),sum);
        }
    }
    for(uint j=0;j<2;++j) {
        float2 val=simd_sum(float2(ga[j],ua[j]));
        if(lane==0)out[local+j]=bfloat16_t(swiglu_bf16(rb(val.x),rb(val.y),true));
    }
}
// One routed expert's down rows local..local+3 for one activation row (expert_down's sums; `sums`: input's table).
inline void routed_down_quad(const device uchar* weights,const threadgroup ulong* d,const device bfloat16_t* input,
                             const threadgroup float* sums,uint expert,uint local,device bfloat16_t* out,uint lane) {
    Matrix m=weight_matrix(weights,d+D_DOWN);
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint b=0;b<512;b+=512) {
        uint scalar=b+lane*16;
        float v[16];values16_words(words(input+scalar),v);
        const float sum=sums[scalar/16u];
        FRT_UNROLL for(uint j=0;j<4;++j) {
            uint row=expert*2048+local+j,gi=row*8+scalar/64;
            float q[16];codes16(*((const device uint2*)(m.w+size_t(row)*256+scalar/2)),q);
            a[j]+=dot16_f(q,v,float(m.s[gi]),float(m.b[gi]),sum);
        }
    }
    for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[local+j]=bfloat16_t(v);}
}
// The shared expert's gate/up row pair `local` for every row (weights read once).
inline void shared_gateup_pair(const device uchar* weights,const threadgroup ulong* d,const threadgroup bfloat16_t* x,
                               const threadgroup float* sums,uint local,device bfloat16_t* out,uint lane) {
    Matrix g=weight_matrix(weights,d+D_SH_GATE),u=weight_matrix(weights,d+D_SH_UP);
    float ga[FRT_ROWS][2],ua[FRT_ROWS][2];
    FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j){ga[j][0]=ga[j][1]=ua[j][0]=ua[j][1]=0.0f;}
    for(uint b=0;b<2048;b+=512) {
        uint scalar=b+lane*16;
        uint2 cg[2],cu[2];float sg_[2],bg[2],su[2],bu[2];
        FRT_UNROLL for(uint i=0;i<2;++i) {
            uint row=local+i,gi=row*32+scalar/64;
            cg[i]=*((const device uint2*)(g.w+size_t(row)*1024+scalar/2));sg_[i]=float(g.s[gi]);bg[i]=float(g.b[gi]);
            cu[i]=*((const device uint2*)(u.w+size_t(row)*1024+scalar/2));su[i]=float(u.s[gi]);bu[i]=float(u.b[gi]);
        }
        float v[FRT_ROWS][16],sum[FRT_ROWS];
        FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j){values16_words(words(x+j*2048+scalar),v[j]);sum[j]=sums[j*128u+scalar/16u];}
        FRT_UNROLL for(uint i=0;i<2;++i) {
            float q[16];
            codes16(cg[i],q);
            FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)ga[j][i]+=dot16_f(q,v[j],sg_[i],bg[i],sum[j]);
            codes16(cu[i],q);
            FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)ua[j][i]+=dot16_f(q,v[j],su[i],bu[i],sum[j]);
        }
    }
    FRT_UNROLL for(uint j=0;j<FRT_ROWS;++j)FRT_UNROLL for(uint i=0;i<2;++i) {
        float2 val=simd_sum(float2(ga[j][i],ua[j][i]));
        if(lane==0)out[j*FRT_ROW_WS+6*512+local+i]=bfloat16_t(swiglu_bf16(rb(val.x),rb(val.y),true));
    }
}
#endif
// ------------------------------------------------------------------------------------------ kernel
#define FRT_FAIL() {if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!grid_sync(control,tid,phase,ready))FRT_FAIL()
#define TEAM_SYNC() if(!team_sync(control,tid,team_phase,ready,group))FRT_FAIL()

#if !FRT_VERIFY_ROWS
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
#endif
#if FRT_VERIFY_ROWS
// Verification step: FRT_ROWS consecutive positions in one pass (row 0 = the last produced token, rows 1.. =
// draft tokens), every row's arithmetic that of the resident step at its position. Global-layer K/V of every
// row go to their cache slots; sliding-layer K/V of rows 1.. go to `side` ([layer][row - 1][K, V][4][128]) and
// attention reads them from there in place of the ring slots they will occupy, so earlier rows still see the
// ring as it was. Greedy choice per row; accepted = number of leading drafts equal to the previous row's choice;
// the sliding K/V of accepted rows are then committed to the ring. packet: [0] accepted, [1 + i] choice of row i.
kernel void flint_decode_verify_bf16(
    const device uchar* weights [[buffer(0)]],
    const device ulong* layers [[buffer(1)]],
    device bfloat16_t* kv [[buffer(2)]],
    constant DecodeParams& params [[buffer(3)]],
    const device uint* tokens [[buffer(4)]],
    device bfloat16_t* row_logits [[buffer(5)]],
    device bfloat16_t* side [[buffer(6)]],
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
    threadgroup uint choice[FRT_ROWS];
    threadgroup uint vids[FRT_ROWS*8];       // every row's six experts
    threadgroup float vchosen[FRT_ROWS*8];   // and their router scores
    float h[FRT_ROWS][8];
    alignas(8) threadgroup bfloat16_t n[FRT_ROWS*2048];
    threadgroup float nsum[FRT_ROWS*128];   // sums table of n
    threadgroup float xsum[FRT_ROWS*384];   // sums tables of device inputs (expert and dense down products)
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
    const uint position=params.position;
    FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r) {
        const uint token=tokens[r];
        for(uint j=0;j<8;++j) {
            uint i=(tid+(j/4)*256)*4+j%4;
            size_t row=token;uint code=embedding.w[row*2048+i];
            uint gi=row*32+i/64;
            h[r][j]=rb(metal::fma(bf16_f(embedding.s[gi]),float(code),bf16_f(embedding.b[gi])));
        }
    }
    SYNC();
    for(uint layer=0;layer<FRT_LAYERS;++layer) {
        if(tid<64)layer_descriptor[tid]=layers[layer*64+tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup ulong* d=layer_descriptor;
        const bool global=(d[D_FLAGS]&1)!=0,sparse=(d[D_FLAGS]&2)!=0;
        FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r)
            rms_2048_table(h[r],(const device bfloat16_t*)(weights+d[D_IN_NORM]),n+r*2048,nsum+r*128,tid,lane,simd,reduction);
        const uint cap=uint(d[D_KV_CAP]);
        KVView keys={kv,d[D_KV],cap};
        KVView vals=keys.value();
        device bfloat16_t* layer_side=side+size_t(layer)*(FRT_ROWS-1)*1024u;
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
            q4_rows_block<2048>(m,(const threadgroup bfloat16_t*)n,2048u,(const threadgroup float*)nsum,128u,
                                (head<32?q:head<36?k:v)+local*128,FRT_ROW_WS,lane,(unit%32u)*4u);
        }
        qkv_base+=320u+FRT_TEAM_GROUPS*8u;
        TEAM_SYNC();
        for(uint member=kv_member;member<10;member+=FRT_TEAM_GROUPS) {
            uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
            if(simd<FRT_ROWS && head<36)
                head_norm(out+simd*FRT_ROW_WS,(const device bfloat16_t*)(weights+d[head<32?D_Q_NORM:D_K_NORM]),
                          !global,position+simd,lane);
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {
                KVView target=head<36?keys:vals;
                for(uint i=tid;i<FRT_ROWS*128u;i+=256) {
                    uint r=i/128u,dim=i%128u,pos=position+r;
                    bfloat16_t value=out[r*FRT_ROW_WS+dim];
                    if(global || r==0) {
                        uint insert=global?pos:(params.initial>=cap?(pos%cap+cap-params.initial%cap)%cap:pos%cap);
                        target.store(local,insert,dim,value);
                    } else layer_side[((r-1)*2u+(head<36?0u:1u))*512u+local*128u+dim]=value;
                }
            }
        }
        TEAM_SYNC();
        const device bfloat16_t* sinks=(const device bfloat16_t*)(weights+d[D_SINKS]);
        const uint slot1=params.initial>=cap?((position+1)%cap+cap-params.initial%cap)%cap:(position+1)%cap;
        bool row_rounded[FRT_ROWS];uint row_parts[FRT_ROWS];
        for(uint r=0;r<FRT_ROWS;++r) {
            const uint pos=position+r,count=global?pos+1:min(pos+1,cap);
            bool rounded=global&&count>=1024;
            uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
            const bool use_mma=global&&count>FRT_MMA_MIN_KEYS;
            if(use_mma){rounded=false;parts=FRT_TEAM_SIMDS;}
            row_rounded[r]=rounded;row_parts[r]=parts;
            const device bfloat16_t* qr=q+r*FRT_ROW_WS;
            device float* pr=partials+r*FRT_ROW_PARTIALS;
            device float* st=statistics+r*FRT_ROW_STATS;
            SideKV sk={layer_side,slot1,r,cap};
            if(use_mma) attention_mma(qr,keys,vals,sinks,pr,st,count,lane,sg,tid,frt_qs);
            else if(global && count>1024) {
                if(rounded) attention_partial<8,true>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg);
                else attention_partial<8,false>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg);
            } else if(global) {
                if(rounded) attention_partial<4,true>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg);
                else attention_partial<4,false>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg);
            } else if(r==0) attention_partial<4,false>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg);
            else attention_partial<4,false,true>(qr,keys,vals,sinks,pr,st,count,parts,lane,sg,sk);
            threadgroup_barrier(mem_flags::mem_threadgroup);   // frt_qs is reused by the next row
        }
        TEAM_SYNC();
        for(uint r=0;r<FRT_ROWS;++r) {
            device float* pr=partials+r*FRT_ROW_PARTIALS;
            device float* st=statistics+r*FRT_ROW_STATS;
            device float* table=(device float*)(workspace+r*FRT_ROW_WS);   // a's sums table (workspace row head)
            if(row_rounded[r]) attention_merge_table<true>(a+r*FRT_ROW_WS,pr,st,row_parts[r],table,lane,sg);
            else attention_merge_table<false>(a+r*FRT_ROW_WS,pr,st,row_parts[r],table,lane,sg);
        }
        SYNC();
        q4_rows<4096>(weight_matrix(weights,d+D_O),(const device bfloat16_t*)a,FRT_ROW_WS,(const device float*)workspace,
                      FRT_ROW_WS/2u,o,FRT_ROW_WS,lane,sg,FRT_GROUPS);
        SYNC();
        FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r) {
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[r][j]=rb(h[r][j]+bf16_f(o[r*FRT_ROW_WS+i]));}
            rms_2048_table(h[r],(const device bfloat16_t*)(weights+d[D_POST_NORM]),n+r*2048,nsum+r*128,tid,lane,simd,reduction);
        }
        if(sparse) {
            route_dot_rows(n,(const device bfloat16_t*)(weights+d[D_ROUTER]),statistics,lane,sg);
            SYNC();
            // Every threadgroup selects every row's six experts (SIMD group r: row r); then all SIMD groups share one
            // list of gate/up row pairs (each row's six experts, then the shared expert for all rows) and, after a
            // barrier, one list of down row quads. Each item's arithmetic is that of the resident step.
            if(simd<FRT_ROWS)
                route_all((const device float*)(weights+d[D_BIAS]),statistics+simd*FRT_ROW_STATS,vids+simd*8,vchosen+simd*8,lane);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                constexpr uint units=6u*FRT_ROWS;
                for(uint item=sg;item<units*256u+512u;item+=FRT_GROUPS) {
                    if(item<units*256u) {
                        const uint unit=item/256u,r=unit/6u,rank=unit%6u;
                        routed_gateup_pair(weights,d,n+r*2048,nsum+r*128,vids[r*8+rank],(item%256u)*2u,
                                           act+r*FRT_ROW_WS+rank*512,lane);
                    } else shared_gateup_pair(weights,d,n,nsum,(item-units*256u)*2u,act,lane);
                }
                SYNC();
                // sums tables of the down inputs: routed (row, rank) at xsum[(r * 6 + rank) * 32], shared after them
                for(uint i=tid;i<units*32u;i+=256u) {
                    const uint unit=i/32u,r=unit/6u,rank=unit%6u;
                    xsum[i]=sum16_words(words((const device bfloat16_t*)(act+r*FRT_ROW_WS+rank*512+(i%32u)*16u)));
                }
                sums_table<1024>((const device bfloat16_t*)(act+6*512),FRT_ROW_WS,xsum+units*32u,tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const Matrix sd=weight_matrix(weights,d+D_SH_DOWN);
                for(uint item=sg;item<units*512u+512u;item+=FRT_GROUPS) {
                    if(item<units*512u) {
                        const uint unit=item/512u,r=unit/6u,rank=unit%6u;
                        routed_down_quad(weights,d,act+r*FRT_ROW_WS+rank*512,xsum+unit*32u,vids[r*8+rank],(item%512u)*4u,
                                         down+r*FRT_ROW_WS+rank*2048,lane);
                    } else q4_rows_block<1024>(sd,(const device bfloat16_t*)(act+6*512),FRT_ROW_WS,
                                               (const threadgroup float*)(xsum+units*32u),64u,down+6*2048,FRT_ROW_WS,lane,
                                               (item-units*512u)*4u);
                }
            }
            SYNC();
            FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r) {
                const device bfloat16_t* dr=down+r*FRT_ROW_WS;
                float chosen=lane<6?vchosen[r*8+lane]:0.0f;
                float total=0.0f;
                for(uint j=0;j<6;++j)total=simd_shuffle(chosen,j)+total;
                total+=1.0e-20f;
                float weight=lane<6?chosen/total:0.0f;
                float sums[8];
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    sums[j]=rb(bf16_f(dr[i])*rb(simd_shuffle(weight,ushort(0))))+0.0f;
                }
                for(uint j=0;j<8;++j)for(uint e=1;e<6;++e) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    sums[j]=rb((rb(bf16_f(dr[e*2048+i])*rb(simd_shuffle(weight,ushort(e))))+0.0f)+sums[j]);
                }
                for(uint j=0;j<8;++j) {
                    uint i=(tid+(j/4)*256)*4+j%4;
                    float mlp=rb(rb(sums[j])+bf16_f(dr[6*2048+i]));
                    h[r][j]=rb(h[r][j]+mlp);
                }
            }
        } else {
            Matrix gm=weight_matrix(weights,d+D_GATE),um=weight_matrix(weights,d+D_UP),dm=weight_matrix(weights,d+D_DOWN);
            q4_rows<2048>(gm,(const threadgroup bfloat16_t*)n,2048u,(const threadgroup float*)nsum,128u,gate,FRT_ROW_WS,lane,sg,FRT_GROUPS);
            q4_rows<2048>(um,(const threadgroup bfloat16_t*)n,2048u,(const threadgroup float*)nsum,128u,up,FRT_ROW_WS,lane,sg,FRT_GROUPS);
            SYNC();
            for(uint r=0;r<FRT_ROWS;++r)for(uint i=index;i<gm.n;i+=FRT_THREADS) {
                act[r*FRT_ROW_WS+i]=bfloat16_t(swiglu_bf16(bf16_f(gate[r*FRT_ROW_WS+i]),bf16_f(up[r*FRT_ROW_WS+i]),false));
            }
            SYNC();
            sums_table<6144>((const device bfloat16_t*)act,FRT_ROW_WS,xsum,tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            q4_rows<6144>(dm,(const device bfloat16_t*)act,FRT_ROW_WS,(const threadgroup float*)xsum,384u,down,FRT_ROW_WS,lane,sg,
                          FRT_GROUPS);
            SYNC();
            FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r)for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                h[r][j]=rb(h[r][j]+bf16_f(down[r*FRT_ROW_WS+i]));
            }
        }
    }
    FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r)
        rms_2048_table(h[r],(const device bfloat16_t*)(weights+final[D_FINAL_NORM]),n+r*2048,nsum+r*128,tid,lane,simd,reduction);
    if(group<32) q4_rows<2048>(weight_matrix(weights,final+D_HEAD),(const threadgroup bfloat16_t*)n,2048u,
                               (const threadgroup float*)nsum,128u,row_logits,FRT_VOCAB,lane,sg,256);
    SYNC();
    // Greedy choice per row: group maxima (ties: lowest id), one barrier, then every group picks the same tokens.
    threadgroup uint* words=(threadgroup uint*)frt_qs;
    for(uint r=0;r<FRT_ROWS;++r) {
        const device bfloat16_t* logits=row_logits+r*FRT_VOCAB;
        float best=-as_type<float>(0x7f800000u);
        uint selected=0xffffffffu;
        for(uint word=index;word<FRT_VOCAB;word+=FRT_THREADS) {
            float value=float(logits[word]);
            if(value>best || (value==best && word<selected)){best=value;selected=word;}
        }
        float peak=simd_max(best);
        uint first=simd_min(best==peak?selected:0xffffffffu);
        if(lane==0){reduction[simd]=peak;words[simd]=first;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simd==0) {
            float value=lane<8?reduction[lane]:-as_type<float>(0x7f800000u);
            uint word=lane<8?words[lane]:0xffffffffu;
            float maximum=simd_max(value);
            uint pick=simd_min(value==maximum?word:0xffffffffu);
            device float* st=statistics+r*FRT_ROW_STATS;
            if(lane==0){st[group*2]=maximum;((device uint*)st)[group*2+1]=pick;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    SYNC();
    if(simd==0) {
        for(uint r=0;r<FRT_ROWS;++r) {
            const device float* st=statistics+r*FRT_ROW_STATS;
            float value=-as_type<float>(0x7f800000u);
            uint word=0xffffffffu;
            for(uint peer=lane;peer<FRT_WORKERS;peer+=32) {
                float other=st[peer*2];
                uint other_word=((const device uint*)st)[peer*2+1];
                if(other>value || (other==value && other_word<word)){value=other;word=other_word;}
            }
            float maximum=simd_max(value);
            uint pick=simd_min(value==maximum?word:0xffffffffu);
            if(lane==0)choice[r]=pick;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint accepted=0;
    while(accepted+1<FRT_ROWS && tokens[accepted+1]==choice[accepted])++accepted;
    if(index==0) {
        packet[0]=accepted;
        for(uint r=0;r<FRT_ROWS;++r)packet[1+r]=choice[r];
        packet[16]=accepted+1;
    }
    // Commit the sliding-layer K/V of accepted draft rows to their ring slots.
    const uint per_layer=accepted*1024u;
    for(uint i=index;i<FRT_LAYERS*per_layer;i+=FRT_THREADS) {
        const uint layer=i/per_layer,rest=i%per_layer,r=1+rest/1024u,e=rest%1024u;
        const device ulong* ld=layers+layer*64;
        if((ld[D_FLAGS]&1)!=0)continue;
        const uint cap=uint(ld[D_KV_CAP]),pos=position+r;
        const uint insert=params.initial>=cap?(pos%cap+cap-params.initial%cap)%cap:pos%cap;
        const uint sel=e/512u,head=(e%512u)/128u,dim=e%128u;
        KVView keys={kv,ld[D_KV],cap};
        KVView target=sel==0?keys:keys.value();
        target.store(head,insert,dim,side[(size_t(layer)*(FRT_ROWS-1)+(r-1))*1024u+e]);
    }
    if(tid==0)diagnostic[group]=0;
}
#endif
)MSL";
}

}  // namespace flashrt::apple::shaders
