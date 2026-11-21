#pragma once
#include "common/source.h"
#include <string>

// Resident decode step for the inkling2_loop text decoder (Inkling 2 Mini) in its released precisions: one
// dispatch of 36 co-resident threadgroups runs up to eight consecutive tokens (embedding, the 48 layer visits of
// the looped stack, LM head, token selection) separated by device-scope barriers.
//
// Model: 24 stored layers in four blocks of six; layers 6..17 run three times ("passes"), each visit with its own
// caches, so a token makes 48 layer visits. A layer reads its inputs through block attention residuals (learned
// queries attending over the unit-RMS block boundaries and the current state), runs sliding-window (1024) or
// global attention with a learned relative-position bias, per-head sink logits and an output gate, then a routed
// MoE (128 experts, top 8, sigmoid scores with a selection bias, softcapped SwiGLU), with short depthwise
// convolutions after attention and MLP. In passes 2 and 3 low-rank gates mix a message from the previous pass into
// the attention and MLP inputs.
// Weights: BF16 attention, gates, embedding and LM head; experts NVFP4 (E2M1 codes, E4M3 scale per 16 inputs, an
// FP32 global scale per expert) in layers 0..19 and MXFP8 (E4M3 codes, E8M0 scale per 32 inputs) in 20..23.
// Activations, KV and the stream between layers in BF16; residual within a layer and block states in FP32.
//
// The arithmetic follows MLX 0.32 / mlx-lm operation by operation for a one-token step: BF16 and FP32 gemv,
// FP4 / FP8 quantized qmv, RMSNorm, the depth-attention and routing kernels, vector attention with sinks and an
// additive BF16 bias, every elementwise operation rounded where MLX rounds; a step reproduces mlx-lm's logits.

namespace flashrt::apple::shaders {

struct Inkling2DecodeOptions {
    bool sampled = false;   // gumbel top-k / top-p selection instead of greedy argmax
    bool trace = false;     // write the hidden state after every visit and a visit's intermediates
};

inline std::string inkling2_decode_resident_bf16_source(const Inkling2DecodeOptions& o = {}) {
    std::string defines = std::string("\n#define FRT_WORKERS 36u") +
        "\n#define FRT_SAMPLED " + std::to_string(o.sampled ? 1 : 0) +
        "\n#define FRT_TRACE " + std::to_string(o.trace ? 1 : 0) + "\n";
    return std::string(metal_preamble) + defines + R"MSL(
// MLX compiles its kernels in the safe math mode; so does this source (whatever the library default is).
#pragma METAL fp math_mode(safe)
constant constexpr uint FRT_GROUPS=FRT_WORKERS*8u;        // SIMD groups in the grid (288)
constant constexpr uint FRT_THREADS=FRT_WORKERS*256u;
constant constexpr uint FH=2304u;                          // hidden
constant constexpr uint FQ=3072u;                          // query / gate width (24 x 128)
constant constexpr uint FKV=1024u;                         // key / value width (8 x 128)
constant constexpr uint FI=1152u;                          // expert width
constant constexpr uint FVOCAB=201024u;
constant constexpr uint FVOCAB_USED=200058u;
constant constexpr uint FLAYERS=24u;
constant constexpr uint FVISITS=48u;
constant constexpr uint FWIN=1024u;                        // sliding window = relative-bias extent
constant constexpr uint FRING=6656u;                       // conv-ring frame: k 1024 | v 1024 | attn 2304 | mlp 2304
constant constexpr uint FDBG=4096u;                        // floats per debug slot
constant constexpr float FEPS=1e-06f;
constant constexpr uint FRT_SPIN_LIMIT=10000000u;
constant constexpr float FRT_MIN=-3.402823466e+38f;
#define FRT_INF as_type<float>(0x7f800000u)
#define FSQRT5 as_type<float>(0x400f1bbdu)
#define FDEPTH_SCALE as_type<float>(0x3caaaaabu)    /* 2304^-0.5 */
#define FHEAD_SCALE as_type<float>(0x3db504f3u)     /* 128^-0.5 */

// ------------------------------------------------------------------------------------------ layout
// Layer descriptor (ulong[64]): byte offsets into the weight buffer.
enum : uint {
    D_FLAGS=0,                    // bit 0 global attention, bit 1 looped (message gates), bit 2 MXFP8 experts
    D_ATTN_Q=1, D_MLP_Q=2,        // FP32 depth-attention queries (attention: 1 or 3 rows; layer 0 none)
    D_IN_NORM=3, D_K_NORM=4, D_V_NORM=5, D_POST_NORM=6,   // BF16 [2304]; global layers: q, k, v input norms
    D_WQ=7, D_WK=8, D_WV=9, D_WR=10, D_WG=11, D_WS=12, D_WO=13,   // BF16 [rows][inputs]
    D_QCLIP=14, D_RCLIP=15, D_REL=16,                       // BF16 [24][128], [24][16], [16][1024]
    D_KCONV=17, D_VCONV=18,       // sliding: BF16 [1024][4]; global: static BF16 [4][1024]
    D_KDYN=19, D_VDYN=20,         // global: BF16 [32][2304]
    D_ACONV=21, D_MCONV=22,       // BF16 [2304][4]
    D_ROUTER=23, D_RBIAS=24,      // BF16 [128][2304] (MLX's FP32 copy holds the same values), FP32 [128]
    D_W13=25, D_S13=26, D_GS13=27, D_W2=28, D_S2=29, D_GS2=30,
    D_MG=32                       // gates q, k, v, mlp: (down [64][4608], forget [2304][64], gain [2304][64]) BF16
};
// Final descriptor (index 24).
enum : uint { D_EMBED=1, D_EMBED_NORM=2, D_FINAL_Q=3, D_FINAL_NORM=4, D_HEAD=5, D_MSG_Q=6 };
// Visit table (one 64-entry row per visit, after the final descriptor).
enum : uint { V_LAYER=0, V_PASS=1, V_KV=2, V_CAP=3, V_RING=4 };
// Workspace (byte offsets).
enum : uint {
    WS_BS=0,              // FP32 block states [6][2304]: b0, b6, b12 of passes 0..2, b18
    WS_BANK=55296,        // FP32 [2][2304]: unit-RMS state after layer 17 of passes 0, 1
    WS_MSG=73728,         // FP32 [12][4][2304]: messages of the current pass (q, k, v, mlp per looped layer)
    WS_AIN=516096,        // FP32 [3][2304]: message gate inputs
    WS_MDOWN=543744,      // FP32 [4][64]: message gate down projections
    WS_FORGET=544768,     // FP32 [3][2304]
    WS_GAIN=572416,       // FP32 [3][2304]
    WS_GATED=600064,      // FP32 [3][2304]: message gate outputs
    WS_PRELUDE=627712,    // BF16 [2304]: output of layer 5
    WS_PROJ=632320,       // BF16: q 3072 | k 1024 | v 1024 | r 384 | gate 3072 | sink 32 | kdyn 32 | vdyn 32
    WS_QC=649664,         // BF16 [3072]: clipped queries
    WS_RCLIP=655808,      // BF16 [384]: clipped relative states
    WS_BIAS=656576,       // BF16 [24][1024]: relative-position bias
    WS_A=705728,          // BF16 [3072]: gated attention output
    WS_O=711872,          // BF16 [2304]: output projection
    WS_RLOG=716480,       // FP32 [128]: router logits
    WS_ACT=716992,        // BF16 [8][1152]: expert activations
    WS_EDOWN=735424,      // BF16 [8][2304]: gate/up rows, then expert outputs
    WS_STATS=772288,      // FP32 [24][64][2]: attention partition max / sum; selection candidates
    WS_PART=784576,       // BF16 [24][64][128]: attention partition outputs
    WS_END=1177792
};
enum : uint { P_Q=0, P_K=3072, P_V=4096, P_R=5120, P_G=5504, P_S=8576, P_KD=8608, P_VD=8640 };

struct DecodeParams {
    uint token, position, steps, eos_count;
    uint eos[4];
    uint top_k; float top_p, inverse_temperature;
    uint rotation;        // sliding caches: slot = (position - rotation) mod 1024
    uint zero;            // always 0 (keeps separately rounded MLX operations from being contracted)
    uint debug_visit;     // reference build: the visit whose intermediates go to `debug`
    uint pad[2];
};

// ------------------------------------------------------------------------------------------ barriers
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

// ------------------------------------------------------------------------------ number formats
// BF16 values are carried in FP32 registers; rb() rounds to nearest even with integer arithmetic.
inline float rb(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
inline float bfu(ushort x){return as_type<float>(uint(x)<<16);}
inline ushort ubf(float v){return ushort(as_type<uint>(rb(v))>>16);}
inline float lo_f(uint w){return as_type<float>(w<<16);}
inline float hi_f(uint w){return as_type<float>(w&0xffff0000u);}
// E2M1, E4M3 and E8M0 decoded as MLX decodes them (exact values).
inline float e2m1(uint b){
    half c=as_type<half>(ushort((b&7u)<<9));c*=16384.0h;
    return float((b&8u)?-c:c);
}
inline float e4m3(uint b){
    ushort v=ushort(b&127u);ushort sign=ushort(((b>>7)&1u)<<15);
    ushort u=ushort((v<<7)|(((v+1u)>>7)<<14)|sign);
    return float(as_type<half>(u)*half(256.0h));
}
inline float e8m0(uint b){return as_type<float>(b==0u?0x400000u:(b<<23));}
// Correctly rounded FP32 division (MLX's kernels divide in the safe math mode).
inline float fdiv(float a,float b){return metal::precise::divide(a,b);}
// x materialized as a rounded FP32 value before the following operation (MLX rounds an elementwise product in
// its own kernel before a separate add or reduction).
inline float sep(float x,uint zero){return as_type<float>(as_type<uint>(x)|zero);}

// ------------------------------------------------------------------------------ vector sources
inline float ld(const threadgroup float* p,uint i){return p[i];}
inline float ld(const threadgroup ushort* p,uint i){return bfu(p[i]);}
inline float ld(const device float* p,uint i){return p[i];}
inline float ld(const device ushort* p,uint i){return bfu(p[i]);}

// MLX rms_single_row inverse over 2304 values: 18 leaves of 128 (4 per lane, fma), then one reduction of the 18.
template<typename P>
inline float rms_inv(P x,threadgroup float* red,uint lane,uint sg) {
    for(uint leaf=sg;leaf<18u;leaf+=8u) {
        float acc=0.0f;
        for(uint j=0;j<4;++j){float v=ld(x,leaf*128u+lane*4u+j);acc=metal::fma(v,v,acc);}
        acc=simd_sum(acc);
        if(lane==0)red[leaf]=acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=simd_sum(lane<18u?red[lane]:0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return metal::precise::rsqrt(fdiv(total,2304.0f)+FEPS);
}

// mlx-lm depth_attention (a threadgroup of 288 threads, each owning every 288th dimension): scores of Q learned
// queries against K block states and the partial state `partial[i] * pinv`, softmax over the K + 1 sources,
// weighted sum. Output FP32 [Q][2304]. The arithmetic is spelled out as MLX's run-time compiled kernel performs
// it: fma accumulation, the scaled score rounded before the max subtraction, metal::exp (which in that compile
// mode is the fast exponential), a correctly rounded division, fma output sum.
template<typename PP,typename OP>
inline void depth_attention(const device float* blocks,thread const uint* slots,uint K,PP partial,float pinv,
                            const device float* queries,uint Q,OP out,threadgroup float* red,
                            uint tid,uint lane,uint sg,uint zero,device float* dbgout=nullptr) {
    const uint N=K+1u;
    for(uint round=0;round<2;++round) {
        const uint vt=tid+round*256u;
        if(vt>=288u)break;
        float acc[24];
        for(uint i=0;i<N*Q;++i)acc[i]=0.0f;
        for(uint d=vt;d<FH;d+=288u) {
            float x[6];
            for(uint k=0;k<K;++k)x[k]=blocks[slots[k]*FH+d];
            x[K]=sep(ld(partial,d)*pinv,zero);
            for(uint j=0;j<Q;++j) {
                float qv=queries[j*FH+d];
                for(uint i=0;i<N;++i)acc[i*Q+j]=metal::fma(x[i],qv,acc[i*Q+j]);
            }
        }
        const uint vsg=vt/32u;
        for(uint i=0;i<N*Q;++i){float v=simd_sum(acc[i]);if(lane==0)red[i*9u+vsg]=v;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[24];
    for(uint i=0;i<N*Q;++i){float t=0.0f;for(uint s=0;s<9u;++s)t+=red[i*9u+s];w[i]=sep(t*FDEPTH_SCALE,zero);}
    if(dbgout && tid<N*Q*9u)dbgout[tid]=red[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<Q;++j) {
        float m=-FRT_INF;
        for(uint i=0;i<N;++i)m=max(m,w[i*Q+j]);
        float z=0.0f;
        for(uint i=0;i<N;++i){w[i*Q+j]=metal::fast::exp(sep(w[i*Q+j]-m,zero));z+=w[i*Q+j];}
        for(uint i=0;i<N;++i)w[i*Q+j]=fdiv(w[i*Q+j],z);
    }
    if(dbgout && tid==0)for(uint i=0;i<N*Q;++i)dbgout[64u+i]=w[i];
    for(uint round=0;round<2;++round) {
        const uint vt=tid+round*256u;
        if(vt>=288u)break;
        for(uint d=vt;d<FH;d+=288u) {
            float x[6];
            for(uint k=0;k<K;++k)x[k]=blocks[slots[k]*FH+d];
            x[K]=sep(ld(partial,d)*pinv,zero);
            for(uint j=0;j<Q;++j) {
                float o=0.0f;
                for(uint i=0;i<N;++i)o=metal::fma(w[i*Q+j],x[i],o);
                out[j*FH+d]=o;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ------------------------------------------------------------------------------ matrix-vector products
// MLX gemv (configurations with bn = 1): a SIMD group owns 4 rows; each lane 4 consecutive inputs of every
// 128-input block; shuffle-down tree. BF16 weights [rows][K]; x BF16 (threadgroup or device).
template<uint K,typename XP>
inline void gemv4(const device ushort* w,uint row0,XP x,device ushort* out,uint lane) {
    float r[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint b=0;b<K;b+=128u) {
        const uint c=b+lane*4u;
        const float v[4]={ld(x,c),ld(x,c+1u),ld(x,c+2u),ld(x,c+3u)};
        for(uint tm=0;tm<4;++tm) {
            const uint2 wr=*((const device uint2*)(w+size_t(row0+tm)*K+c));
            r[tm]+=lo_f(wr.x)*v[0];r[tm]+=hi_f(wr.x)*v[1];r[tm]+=lo_f(wr.y)*v[2];r[tm]+=hi_f(wr.y)*v[3];
        }
    }
    for(uint tm=0;tm<4;++tm)for(ushort sn=16;sn>=1;sn>>=1)r[tm]+=simd_shuffle_down(r[tm],sn);
    if(lane==0)for(uint tm=0;tm<4;++tm)out[row0+tm]=ubf(r[tm]);
}
// MLX gemv with bn = 8 (K >= 16 * rows): 8 SIMD groups split K into 128-input slices of each 1024-input block;
// per group a shuffle-down tree; the group sums are added in order. One row; the sum on every lane.
template<typename WP,typename XP>
inline float gemv_bn8_row(WP w,uint K,XP x,uint lane) {
    const uint n_iter=K/1024u,left=K-n_iter*1024u;
    float total=0.0f;
    for(uint s=0;s<8;++s) {
        float r=0.0f;
        for(uint i=0;i<n_iter;++i) {
            const uint c=i*1024u+s*128u+lane*4u;
            float v[4];for(uint t=0;t<4;++t)v[t]=ld(x,c+t);
            for(uint t=0;t<4;++t)r+=ld(w,c+t)*v[t];
        }
        if(left>0u) {
            const uint c=n_iter*1024u+s*128u+lane*4u;
            float v[4],m[4];
            for(uint t=0;t<4;++t){bool in=c+t<K;v[t]=in?ld(x,c+t):0.0f;m[t]=in?ld(w,c+t):0.0f;}
            for(uint t=0;t<4;++t)r+=m[t]*v[t];
        }
        for(ushort sn=16;sn>=1;sn>>=1)r+=simd_shuffle_down(r,sn);
        if(s==0)total=r;else total+=r;
    }
    return simd_shuffle(total,ushort(0));
}
// MLX gemv for K = 64 (sm = 8, sn = 4): 32 rows per SIMD group, 4 lanes per row (4 inputs of every 16).
// FP32 weights (stored BF16, the same values) times an FP32 vector; FP32 output.
inline void gemv_k64(const device ushort* w,uint row0,const device float* x,device float* out,uint lane) {
    const uint thr_m=lane/4u,thr_n=lane%4u;
    float r[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint i=0;i<4;++i) {
        const uint c=i*16u+thr_n*4u;
        float v[4];for(uint t=0;t<4;++t)v[t]=x[c+t];
        for(uint tm=0;tm<4;++tm) {
            const uint row=row0+thr_m*4u+tm;
            for(uint t=0;t<4;++t)r[tm]+=bfu(w[row*64u+c+t])*v[t];
        }
    }
    for(uint tm=0;tm<4;++tm){r[tm]+=simd_shuffle_down(r[tm],ushort(2));r[tm]+=simd_shuffle_down(r[tm],ushort(1));}
    if(thr_n==0)for(uint tm=0;tm<4;++tm)out[row0+thr_m*4u+tm]=r[tm];
}

// ------------------------------------------------------------------------------ quantized experts
// MLX fp_qmv: 4 rows per SIMD group; NVFP4 8 inputs per lane of every 256 with an E4M3 scale per 16 and the
// result times global_scale / (448 * 6); MXFP8 8 inputs of every 256 (K % 256 == 0, qmv_fast) or 4 of every 128
// with an E8M0 scale per 32.
inline float qdot_nv(const device uchar* w,thread const float* x,float scale) {
    float accum=0.0f;
    const device ushort* ws=(const device ushort*)w;
    for(uint i=0;i<2;++i) {
        const uint q=ws[i];
        accum+=(x[4*i]*e2m1(q)+x[4*i+1]*e2m1(q>>4)+x[4*i+2]*e2m1(q>>8)+x[4*i+3]*e2m1(q>>12));
    }
    return scale*accum;
}
template<uint N>
inline float qdot_mx(const device uchar* w,thread const float* x,float scale) {
    float accum=0.0f;
    for(uint i=0;i<N;++i)accum+=x[i]*e4m3(w[i]);
    return scale*accum;
}
template<uint K,bool MX,typename XP>
inline void expert_rows(const device uchar* w,const device uchar* s,float inv_enc,uint row0,XP x,
                        device ushort* out,uint lane) {
    float r[4]={0.0f,0.0f,0.0f,0.0f};
    if(!MX) {
        for(uint b=0;b<K;b+=256u) {
            const uint c=b+lane*8u;
            if(c>=K)break;
            float xv[8];for(uint i=0;i<8;++i)xv[i]=ld(x,c+i);
            for(uint tm=0;tm<4;++tm) {
                const uint row=row0+tm;
                r[tm]+=qdot_nv(w+size_t(row)*(K/2u)+c/2u,xv,e4m3(s[size_t(row)*(K/16u)+c/16u]));
            }
        }
        for(uint tm=0;tm<4;++tm){float v=simd_sum(r[tm]);if(lane==0)out[row0+tm]=ubf(v*inv_enc);}
    } else if(K%256u==0u) {
        for(uint b=0;b<K;b+=256u) {
            const uint c=b+lane*8u;
            float xv[8];for(uint i=0;i<8;++i)xv[i]=ld(x,c+i);
            for(uint tm=0;tm<4;++tm) {
                const uint row=row0+tm;
                r[tm]+=qdot_mx<8>(w+size_t(row)*K+c,xv,e8m0(s[size_t(row)*(K/32u)+c/32u]));
            }
        }
        for(uint tm=0;tm<4;++tm){float v=simd_sum(r[tm]);if(lane==0)out[row0+tm]=ubf(v);}
    } else {
        for(uint b=0;b<K;b+=128u) {
            const uint c=b+lane*4u;
            float xv[4];for(uint i=0;i<4;++i)xv[i]=ld(x,c+i);
            for(uint tm=0;tm<4;++tm) {
                const uint row=row0+tm;
                r[tm]+=qdot_mx<4>(w+size_t(row)*K+c,xv,e8m0(s[size_t(row)*(K/32u)+c/32u]));
            }
        }
        for(uint tm=0;tm<4;++tm){float v=simd_sum(r[tm]);if(lane==0)out[row0+tm]=ubf(v);}
    }
}

// ------------------------------------------------------------------------------ elementwise
inline float sigmoid_f32(float x){float y=fdiv(1.0f,1.0f+metal::precise::exp(metal::abs(x)));return x<0.0f?y:1.0f-y;}
// mlx-lm softcapped SwiGLU (FP32): tanh(g / 4) * 4 * sigmoid(g) * tanh(u / 25) * 25, rounded to BF16.
inline float softcapped_swiglu(float g,float u) {
    float gc=metal::precise::tanh(fdiv(g,4.0f))*4.0f;
    float uc=metal::precise::tanh(fdiv(u,25.0f))*25.0f;
    return rb(gc*sigmoid_f32(g)*uc);
}
// mlx-lm ShortConvolution of one channel: BF16 conv of the window [x(t-3) .. x(t)] (FP32 sum of exact products),
// then (conv + x(t)) / sqrt(5), BF16. `prev` holds x(t-3), x(t-2), x(t-1) (0 before the sequence start).
inline float short_conv(thread const float* prev,float xt,const device ushort* w) {
    float acc=0.0f;
    for(uint i=0;i<3;++i)acc+=prev[i]*bfu(w[i]);
    acc+=xt*bfu(w[3]);
    return rb(fdiv(rb(acc)+xt,FSQRT5));
}
inline void ring_prev(const device ushort* ring,uint channel,uint position,thread float* prev) {
    for(uint i=0;i<3;++i) {
        const int p=int(position)-3+int(i);
        prev[i]=p>=0?bfu(ring[(uint(p)&3u)*FRING+channel]):0.0f;
    }
}
// Message gate combination: (1 - tanh(f)) * input + tanh(g) * message, each operation rounded.
inline float gate_combine(float f,float g,float input,float message,uint zero) {
    const float a=sep((1.0f-metal::precise::tanh(f))*input,zero);
    const float b=sep(metal::precise::tanh(g)*message,zero);
    return a+b;
}

// ------------------------------------------------------------------------------------------ kernel
#define FRT_FAIL() {if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!grid_sync(control,tid,phase,ready))FRT_FAIL()

kernel void inkling2_decode_resident_bf16(
    const device uchar* weights [[buffer(0)]],
    const device ulong* table [[buffer(1)]],
    device ushort* kv [[buffer(2)]],
    device ushort* rings [[buffer(3)]],
    constant DecodeParams& params [[buffer(4)]],
    const device float* sampling [[buffer(5)]],
    device float* block_logits [[buffer(6)]],
    device ushort* block_trace [[buffer(7)]],
    device uint* packet [[buffer(8)]],
    device uchar* workspace [[buffer(9)]],
    device float* debug [[buffer(10)]],
    device uint* private_control [[buffer(11)]],
    device uint* diagnostic [[buffer(12)]],
    uint3 group3 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint group=group3.x,sg=group*8+simd,index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    const uint zero=params.zero;
    threadgroup uint ready[1];
    uint phase=0;
    threadgroup float red[256];
    threadgroup ulong desc[64];
    threadgroup uint route_ids[8];
    threadgroup float route_w[8];
    threadgroup ushort vh[FH];             // the hidden state between visits (BF16)
    threadgroup float vr[FH];              // FP32: attention input, then the residual
    threadgroup float vxs[3456];           // BF16 [3][2304] projection inputs | FP32 [2304] + BF16 [2304]
    threadgroup ushort* vx=(threadgroup ushort*)vxs;

    device float* bs=(device float*)(workspace+WS_BS);
    device float* bank=(device float*)(workspace+WS_BANK);
    device float* msg=(device float*)(workspace+WS_MSG);
    device float* ain=(device float*)(workspace+WS_AIN);
    device float* mdown=(device float*)(workspace+WS_MDOWN);
    device float* forget=(device float*)(workspace+WS_FORGET);
    device float* gain=(device float*)(workspace+WS_GAIN);
    device float* gated=(device float*)(workspace+WS_GATED);
    device ushort* prelude=(device ushort*)(workspace+WS_PRELUDE);
    device ushort* proj=(device ushort*)(workspace+WS_PROJ);
    device ushort* qc=(device ushort*)(workspace+WS_QC);
    device ushort* rclip=(device ushort*)(workspace+WS_RCLIP);
    device ushort* bias=(device ushort*)(workspace+WS_BIAS);
    device ushort* attn=(device ushort*)(workspace+WS_A);
    device ushort* oproj=(device ushort*)(workspace+WS_O);
    device float* rlog=(device float*)(workspace+WS_RLOG);
    device ushort* act=(device ushort*)(workspace+WS_ACT);
    device ushort* edown=(device ushort*)(workspace+WS_EDOWN);
    device float* stats=(device float*)(workspace+WS_STATS);
    device ushort* part=(device ushort*)(workspace+WS_PART);
    const device ulong* final=table+FLAYERS*64u;
    const device ulong* visits=final+64u;
    const device ushort* embed=(const device ushort*)(weights+final[D_EMBED]);

    uint input_token=params.token;
    #pragma clang loop unroll(disable)
    for(uint step=0;step<params.steps;++step) {
        const uint position=params.position+step;
#if FRT_TRACE
        device ushort* trace=block_trace+step*(FVISITS*FH);
#endif
        if(index==0)packet[step*2]=input_token;
        // embedding row + embed_norm (RMSNorm of BF16: w * bf16(x * inv), both rounded)
        for(uint i=tid;i<FH;i+=256u)vh[i]=embed[size_t(input_token)*FH+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
            const device ushort* w=(const device ushort*)(weights+final[D_EMBED_NORM]);
            for(uint i=tid;i<FH;i+=256u)vh[i]=ubf(bfu(w[i])*rb(bfu(vh[i])*inv));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint nblocks=0;
        uint slots[5]={0u,0u,0u,0u,0u};
        for(uint visit=0;visit<FVISITS;++visit) {
            const device ulong* vt=visits+visit*64u;
            const uint layer=uint(vt[V_LAYER]),pass=uint(vt[V_PASS]);
            if(tid<64)desc[tid]=table[layer*64u+tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const threadgroup ulong* d=desc;
            const bool full=(d[D_FLAGS]&1ul)!=0ul;
            const bool looped=(d[D_FLAGS]&2ul)!=0ul;
            const bool mxfp8=(d[D_FLAGS]&4ul)!=0ul;
            const bool messages=looped && pass>0u;
            const bool boundary=layer%6u==0u;
            const uint li=layer-6u;
            const uint bslot=layer==0u?0u:(layer==6u?1u:(layer==12u?2u+pass:5u));
#if FRT_TRACE
            const bool dbg=visit==params.debug_visit && group==0;
            device float* dbgp=debug;
#else
            const bool dbg=false;
            device float* dbgp=debug;
#endif

            // ---- pass start: the prelude's state and block states; the messages of the pass
            if(layer==6u) {
                nblocks=1;slots[0]=0;
                if(pass==0u) {
                    if(group==0)for(uint i=tid;i<FH;i+=256u)prelude[i]=vh[i];
                } else {
                    for(uint i=tid;i<FH;i+=256u)vh[i]=prelude[i];
                    // bank = (b12 of the previous pass, unit-RMS state after layer 17 of the previous pass)
                    const uint bk[1]={2u+pass-1u};
                    for(uint q=group;q<12u;q+=FRT_WORKERS)
                        depth_attention(bs,bk,1u,(const device float*)(bank+(pass-1u)*FH),1.0f,
                                        (const device float*)(weights+final[D_MSG_Q])+q*4u*FH,4u,
                                        msg+q*4u*FH,red,tid,lane,simd,zero);
                    SYNC();
                }
            }
            // ---- attention inputs: depth attention over the block states (the hidden state for layer 0)
            const float rinv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
            if(boundary && group==0)for(uint i=tid;i<FH;i+=256u)bs[bslot*FH+i]=bfu(vh[i])*rinv;
            if(!messages) {
                const uint nq=full?3u:1u;
                for(uint j=0;j<nq;++j) {
                    const device ushort* w=(const device ushort*)(weights+d[D_IN_NORM+j]);
                    if(layer==0u) {
                        // RMSNorm of the BF16 hidden state: two roundings
                        for(uint i=tid;i<FH;i+=256u)vx[i]=ubf(bfu(w[i])*rb(bfu(vh[i])*rinv));
                    } else {
                        depth_attention(bs,slots,nblocks,(const threadgroup ushort*)vh,rinv,
                                        (const device float*)(weights+d[D_ATTN_Q])+j*FH,1u,vr,red,tid,lane,simd,zero);
                        const float inv=rms_inv((const threadgroup float*)vr,red,lane,simd);
                        for(uint i=tid;i<FH;i+=256u)vx[j*FH+i]=ubf(bfu(w[i])*(vr[i]*inv));
                        if(dbg)for(uint i=tid;i<FH;i+=256u)dbgp[j*FDBG+i]=vr[i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if(!full) {
                    for(uint i=tid;i<FH;i+=256u){vx[FH+i]=vx[i];vx[2u*FH+i]=vx[i];}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                // message gates q, k, v (threadgroups 12j .. 12j + 11 hold gate j):
                //   s = [rms(input) | rms(message)] -> down (64 rows, BF16) -> forget, gain -> combination
                const uint j=group/12u;
                depth_attention(bs,slots,nblocks,(const threadgroup ushort*)vh,rinv,
                                (const device float*)(weights+d[D_ATTN_Q])+(full?j*FH:0u),1u,vr,red,tid,lane,simd,zero);
                if(group%12u==0u)for(uint i=tid;i<FH;i+=256u)ain[j*FH+i]=vr[i];
                if(dbg)for(uint i=tid;i<FH;i+=256u)dbgp[j*FDBG+i]=vr[i];
                const device float* m=msg+(li*4u+j)*FH;
                const float ia=rms_inv((const threadgroup float*)vr,red,lane,simd);
                const float im=rms_inv(m,red,lane,simd);
                for(uint i=tid;i<FH;i+=256u){vx[i]=ubf(vr[i]*ia);vx[FH+i]=ubf(m[i]*im);}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint r=(group%12u)*8u+simd;
                if(r<64u) {
                    const device ushort* wd=(const device ushort*)(weights+d[D_MG+j*3u])+r*4608u;
                    const float v=gemv_bn8_row(wd,4608u,(const threadgroup ushort*)vx,lane);
                    if(lane==0)mdown[j*64u+r]=rb(v);
                }
                SYNC();
                // forget and gain rows: 3 gates x 2 matrices x 72 tasks of 32 rows
                for(uint task=sg;task<432u;task+=FRT_GROUPS) {
                    const uint gj=task/144u,which=(task/72u)%2u,row0=(task%72u)*32u;
                    const device ushort* w=(const device ushort*)(weights+d[D_MG+gj*3u+1u+which]);
                    gemv_k64(w,row0,mdown+gj*64u,(which==0u?forget:gain)+gj*FH,lane);
                }
                SYNC();
                for(uint i=index;i<3u*FH;i+=FRT_THREADS) {
                    const uint gj=i/FH,e=i%FH;
                    gated[gj*FH+e]=gate_combine(forget[gj*FH+e],gain[gj*FH+e],ain[gj*FH+e],
                                                msg[(li*4u+gj)*FH+e],zero);
                }
                SYNC();
                for(uint jj=0;jj<3u;++jj) {
                    const device float* src=gated+jj*FH;
                    const float inv=rms_inv(src,red,lane,simd);
                    const device ushort* w=(const device ushort*)(weights+d[D_IN_NORM+(full?jj:0u)]);
                    for(uint i=tid;i<FH;i+=256u)vx[jj*FH+i]=ubf(bfu(w[i])*(src[i]*inv));
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if(dbg)for(uint i=tid;i<3u*FH;i+=256u)dbgp[3u*FDBG+(i/FH)*FDBG+i%FH]=bfu(vx[i]);
            // ---- projections: q, r, gate, sink from x_q; k (+ dynamic conv weights) from x_k; v from x_v
            {
                const device ushort* wq=(const device ushort*)(weights+d[D_WQ]);
                const device ushort* wk=(const device ushort*)(weights+d[D_WK]);
                const device ushort* wv=(const device ushort*)(weights+d[D_WV]);
                const device ushort* wr=(const device ushort*)(weights+d[D_WR]);
                const device ushort* wg=(const device ushort*)(weights+d[D_WG]);
                const threadgroup ushort* xq=vx;
                const threadgroup ushort* xk=vx+FH;
                const threadgroup ushort* xv=vx+2u*FH;
                const uint units=2144u+24u+(full?64u:0u);
                for(uint u=sg;u<units;u+=FRT_GROUPS) {
                    if(u<768u)gemv4<FH>(wq,u*4u,xq,proj+P_Q,lane);
                    else if(u<864u)gemv4<FH>(wr,(u-768u)*4u,xq,proj+P_R,lane);
                    else if(u<1632u)gemv4<FH>(wg,(u-864u)*4u,xq,proj+P_G,lane);
                    else if(u<1888u)gemv4<FH>(wk,(u-1632u)*4u,xk,proj+P_K,lane);
                    else if(u<2144u)gemv4<FH>(wv,(u-1888u)*4u,xv,proj+P_V,lane);
                    else if(u<2168u) {
                        const uint r=u-2144u;
                        const float v=gemv_bn8_row((const device ushort*)(weights+d[D_WS])+r*FH,FH,xq,lane);
                        if(lane==0)proj[P_S+r]=ubf(v);
                    } else {
                        const uint r=u-2168u;
                        const bool kd=r<32u;
                        const float v=gemv_bn8_row((const device ushort*)(weights+d[kd?D_KDYN:D_VDYN])+(r%32u)*FH,FH,
                                                   kd?xk:xv,lane);
                        if(lane==0)proj[(kd?P_KD:P_VD)+r%32u]=ubf(v);
                    }
                }
            }
            SYNC();
            if(FRT_TRACE && visit==params.debug_visit && group==0)
                for(uint i=tid;i<8672u;i+=256u)dbgp[18u*FDBG+i]=bfu(proj[i]);
            // ---- per head: clipped queries and relative states (x tau in global layers); keys and values through
            // their convolution and clip / norm into the cache
            const uint cap=uint(vt[V_CAP]);
            device ushort* kcache=kv+vt[V_KV];
            device ushort* vcache=kcache+size_t(8u)*cap*128u;
            device ushort* ring=rings+vt[V_RING];
            const uint count=full?position+1u:min(position+1u,FWIN);
            const uint cslot=full?position:((position-params.rotation)&(FWIN-1u));
            const uint rslot=position&3u;
            float tau=1.0f;
            if(full) {
                const float pf=float(position);
                tau=1.0f+sep(0.1f*metal::precise::log(max(fdiv(pf+1.0f,8192.0f),1.0f)),zero);
            }
            for(uint task=sg;task<64u;task+=FRT_GROUPS) {
                if(task<24u) {
                    // q head: q * scale, L2 clip to 22.6274 (FP32), BF16; x tau, BF16
                    const uint h=task;
                    const device ushort* sc=(const device ushort*)(weights+d[D_QCLIP])+h*128u;
                    float s[4],sq=0.0f;
                    for(uint t=0;t<4;++t)s[t]=bfu(proj[P_Q+h*128u+lane*4u+t])*bfu(sc[lane*4u+t]);
                    for(uint t=0;t<4;++t)sq=sep(s[t]*s[t],zero)+sq;
                    sq=simd_sum(sq);
                    const float k=min(fdiv(22.627416997969522f,metal::precise::sqrt(sq)+1e-06f),1.0f);
                    for(uint t=0;t<4;++t) {
                        float v=rb(s[t]*k);
                        if(full)v=rb(v*tau);
                        qc[h*128u+lane*4u+t]=ubf(v);
                        if(FRT_TRACE && visit==params.debug_visit)dbgp[16u*FDBG+h*128u+lane*4u+t]=v;
                    }
                } else if(task<48u) {
                    // relative state of a head (16 values): L2 clip to 8, BF16, x tau
                    const uint h=task-24u;
                    const float s=lane<16u?bfu(proj[P_R+h*16u+lane])*bfu(((const device ushort*)(weights+d[D_RCLIP]))[h*16u+lane]):0.0f;
                    const float sqv=sep(s*s,zero);
                    float tot=0.0f;
                    for(uint t=0;t<16u;++t)tot=simd_shuffle(sqv,ushort(t))+tot;
                    const float k=min(fdiv(8.0f,metal::precise::sqrt(tot)+1e-06f),1.0f);
                    if(lane<16u) {
                        float v=rb(s*k);
                        if(full)v=rb(v*tau);
                        rclip[h*16u+lane]=ubf(v);
                        if(FRT_TRACE && visit==params.debug_visit)dbgp[22u*FDBG+h*16u+lane]=v;
                    }
                } else {
                    const uint idx=task-48u,h=idx%8u;
                    const bool isk=idx<8u;
                    const uint base=(isk?0u:FKV)+h*128u;
                    const device ushort* cur=proj+(isk?P_K:P_V)+h*128u;
                    float x[4],outv[4];
                    for(uint t=0;t<4;++t)x[t]=bfu(cur[lane*4u+t]);
                    if(!full) {
                        const device ushort* cw=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                        for(uint t=0;t<4;++t) {
                            float prev[3];ring_prev(ring,base+lane*4u+t,position,prev);
                            outv[t]=short_conv(prev,x[t],cw+(h*128u+lane*4u+t)*4u);
                        }
                    } else {
                        // dynamic conv: sum over lag of x(t - lag) * (static[lag][c] + dyn[head][lag]), + x, / 3
                        const device ushort* st=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                        const device ushort* dyn=proj+(isk?P_KD:P_VD)+h*4u;
                        for(uint t=0;t<4;++t) {
                            const uint c=h*128u+lane*4u+t;
                            float prev[3];ring_prev(ring,base+lane*4u+t,position,prev);
                            float sum=0.0f;
                            for(uint lag=0;lag<4;++lag) {
                                const float wgt=bfu(st[lag*FKV+c])+bfu(dyn[lag]);
                                const float xi=lag==0u?x[t]:prev[3u-lag];
                                sum=sep(xi*wgt,zero)+sum;
                            }
                            outv[t]=rb(fdiv(sum+x[t],3.0f));
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_device);
                    for(uint t=0;t<4;++t)ring[rslot*FRING+base+lane*4u+t]=cur[lane*4u+t];
                    float y[4];
                    if(isk) {
                        float sq=0.0f;
                        for(uint t=0;t<4;++t)sq=sep(outv[t]*outv[t],zero)+sq;
                        sq=simd_sum(sq);
                        const float k=min(fdiv(22.627416997969522f,metal::precise::sqrt(sq)+1e-06f),1.0f);
                        for(uint t=0;t<4;++t)y[t]=rb(outv[t]*k);
                    } else {
                        float acc=0.0f;
                        for(uint t=0;t<4;++t)acc+=outv[t]*outv[t];
                        acc=simd_sum(acc);
                        acc=simd_sum(lane==0?acc:0.0f);
                        const float inv=metal::precise::rsqrt(fdiv(acc,128.0f)+FEPS);
                        for(uint t=0;t<4;++t)y[t]=rb(outv[t]*inv);
                    }
                    device ushort* dst=(isk?kcache:vcache)+(size_t(h)*cap+cslot)*128u;
                    for(uint t=0;t<4;++t)dst[lane*4u+t]=ubf(y[t]);
                    if(FRT_TRACE && visit==params.debug_visit)for(uint t=0;t<4;++t){
                        dbgp[(isk?12u:13u)*FDBG+h*128u+lane*4u+t]=outv[t];
                        dbgp[(isk?14u:15u)*FDBG+h*128u+lane*4u+t]=y[t];}
                }
            }
            SYNC();
            // ---- relative-position bias [24][1024] = relative states [24][16] @ proj [16][1024] (8x8 MMAs)
            {
                const device ushort* rel=(const device ushort*)(weights+d[D_REL]);
                const ushort fm=((lane/4u)&4u)+(lane/2u)%4u;
                const ushort fn=((lane/4u)&2u)*2u+(lane%2u)*2u;
                for(uint tile=sg;tile<384u;tile+=FRT_GROUPS) {
                    const uint ht=tile/128u,ct=tile%128u;
                    simdgroup_float8x8 c=simdgroup_float8x8(0.0f);
                    for(uint kk=0;kk<16u;kk+=8u) {
                        simdgroup_float8x8 a,b;
                        a.thread_elements()[0]=bfu(rclip[(ht*8u+fm)*16u+kk+fn]);
                        a.thread_elements()[1]=bfu(rclip[(ht*8u+fm)*16u+kk+fn+1u]);
                        b.thread_elements()[0]=bfu(rel[(kk+fm)*FWIN+ct*8u+fn]);
                        b.thread_elements()[1]=bfu(rel[(kk+fm)*FWIN+ct*8u+fn+1u]);
                        simdgroup_multiply_accumulate(c,a,b,c);
                    }
                    bias[(ht*8u+fm)*FWIN+ct*8u+fn]=ubf(c.thread_elements()[0]);
                    bias[(ht*8u+fm)*FWIN+ct*8u+fn+1u]=ubf(c.thread_elements()[1]);
                    if(FRT_TRACE && visit==params.debug_visit && ht==0u && fm<4u){
                        dbgp[17u*FDBG+fm*FWIN+ct*8u+fn]=rb(c.thread_elements()[0]);
                        dbgp[17u*FDBG+fm*FWIN+ct*8u+fn+1u]=rb(c.thread_elements()[1]);}
                }
            }
            SYNC();
            // ---- attention: MLX vector attention with sinks and the bias as an additive BF16 mask; the output
            // gate sigmoid(gate) (FP32) on the BF16 result
            {
                const float scale=FHEAD_SCALE;
                const device ushort* sinks=proj+P_S;
                const device ushort* gate=proj+P_G;
                if(count<1024u) {
                    // single pass: one threadgroup per query head; 32 key-strided partitions, 4 per SIMD group
                    if(group<24u) {
                        const uint h=group,g=h/3u;
                        float q[4];
                        for(uint t=0;t<4;++t)q[t]=scale*bfu(qc[h*128u+lane*4u+t]);
                        float o[4][4],mx[4],z[4];
                        for(uint r=0;r<4;++r) {
                            const uint vs=simd+r*8u;
                            for(uint t=0;t<4;++t)o[r][t]=0.0f;
                            mx[r]=FRT_MIN;z[r]=0.0f;
                            if(vs==0u){mx[r]=bfu(sinks[h]);z[r]=1.0f;}
                            for(uint i=vs;i<count;i+=32u) {
                                const uint2 kr=*((const device uint2*)(kcache+(size_t(g)*cap+i)*128u+lane*4u));
                                const float k[4]={lo_f(kr.x),hi_f(kr.x),lo_f(kr.y),hi_f(kr.y)};
                                float score=0.0f;
                                for(uint t=0;t<4;++t)score+=q[t]*k[t];
                                score=simd_sum(score);
                                const uint dist=position-i;   // below a full ring the buffer is in position order
                                score+=bfu(bias[h*FWIN+dist]);
                                const float nm=max(mx[r],score);
                                const float factor=metal::fast::exp(mx[r]-nm);
                                const float e=metal::fast::exp(score-nm);
                                mx[r]=nm;
                                z[r]=z[r]*factor+e;
                                const uint2 vv=*((const device uint2*)(vcache+(size_t(g)*cap+i)*128u+lane*4u));
                                const float v[4]={lo_f(vv.x),hi_f(vv.x),lo_f(vv.y),hi_f(vv.y)};
                                for(uint t=0;t<4;++t)o[r][t]=metal::fma(o[r][t],factor,e*v[t]);
                            }
                            if(lane==0){red[vs]=mx[r];red[32u+vs]=z[r];}
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        const float my_m=red[lane];
                        const float nm=simd_max(my_m);
                        const float factor=metal::fast::exp(my_m-nm);
                        const float zs=simd_sum(red[32u+lane]*factor);
                        // partition outputs in two halves of 64 dimensions: [32 partitions][64]
                        for(uint half_=0;half_<2u;++half_) {
                            if((lane/16u)==half_)
                                for(uint r=0;r<4;++r)for(uint t=0;t<4;++t)
                                    vxs[(simd+r*8u)*64u+(lane%16u)*4u+t]=o[r][t];
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            for(uint dd=simd;dd<64u;dd+=8u) {
                                float ov=simd_sum(vxs[lane*64u+dd]*factor);
                                ov=zs==0.0f?ov:fdiv(ov,zs);
                                if(lane==0) {
                                    const uint dim=half_*64u+dd;
                                    const float val=rb(ov);
                                    attn[h*128u+dim]=ubf(val*sigmoid_f32(bfu(gate[h*128u+dim])));
                                    if(FRT_TRACE && visit==params.debug_visit)dbgp[6u*FDBG+h*128u+dim]=val;
                                }
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                    }
                    SYNC();
                } else {
                    // two passes: 64 partitions; a task = (KV head, partition) for its 3 query heads
                    for(uint task=sg;task<512u;task+=FRT_GROUPS) {
                        const uint g=task/64u,b=task%64u;
                        float q[3][4],o[3][4],mx[3],z[3];
                        for(uint hh=0;hh<3u;++hh) {
                            const uint h=g*3u+hh;
                            for(uint t=0;t<4;++t){q[hh][t]=scale*bfu(qc[h*128u+lane*4u+t]);o[hh][t]=0.0f;}
                            mx[hh]=FRT_MIN;z[hh]=0.0f;
                            if(b==0u){mx[hh]=bfu(sinks[h]);z[hh]=1.0f;}
                        }
                        for(uint i=b;i<count;i+=64u) {
                            const uint2 kr=*((const device uint2*)(kcache+(size_t(g)*cap+i)*128u+lane*4u));
                            const float k[4]={lo_f(kr.x),hi_f(kr.x),lo_f(kr.y),hi_f(kr.y)};
                            const uint2 vv=*((const device uint2*)(vcache+(size_t(g)*cap+i)*128u+lane*4u));
                            const float v[4]={lo_f(vv.x),hi_f(vv.x),lo_f(vv.y),hi_f(vv.y)};
                            // key position: global caches hold position i; a full sliding ring holds
                            // ((i - cslot - 1) mod 1024) + position + 1 - 1024
                            const uint kp=full?i:(((i+FWIN-cslot-1u)&(FWIN-1u))+position+1u-FWIN);
                            const uint dist=position-kp;
                            const float madd=dist<FWIN?1.0f:0.0f;
                            for(uint hh=0;hh<3u;++hh) {
                                const uint h=g*3u+hh;
                                float score=0.0f;
                                for(uint t=0;t<4;++t)score+=q[hh][t]*k[t];
                                score=simd_sum(score);
                                score+=madd>0.0f?bfu(bias[h*FWIN+dist]):0.0f;
                                const float nm=max(mx[hh],score);
                                const float factor=metal::fast::exp(mx[hh]-nm);
                                const float e=metal::fast::exp(score-nm);
                                mx[hh]=nm;
                                z[hh]=z[hh]*factor+e;
                                for(uint t=0;t<4;++t)o[hh][t]=metal::fma(o[hh][t],factor,e*v[t]);
                            }
                        }
                        for(uint hh=0;hh<3u;++hh) {
                            const uint h=g*3u+hh;
                            if(lane==0){stats[(h*64u+b)*2u]=mx[hh];stats[(h*64u+b)*2u+1u]=z[hh];}
                            for(uint t=0;t<4;++t)part[(h*64u+b)*128u+lane*4u+t]=ubf(o[hh][t]);
                        }
                    }
                    SYNC();
                    // merge (sdpa_vector_2pass_2): lane l holds partitions l and l + 32
                    for(uint h=sg;h<24u;h+=FRT_GROUPS) {
                        const float m0=stats[(h*64u+lane)*2u],m1=stats[(h*64u+lane+32u)*2u];
                        float top=FRT_MIN;top=max(top,m0);top=max(top,m1);
                        top=simd_max(top);
                        float zs=0.0f;
                        {
                            const float f=metal::fast::exp(m0-top);zs+=f*stats[(h*64u+lane)*2u+1u];
                            const float f2=metal::fast::exp(m1-top);zs+=f2*stats[(h*64u+lane+32u)*2u+1u];
                        }
                        zs=simd_sum(zs);
                        const float f0=metal::fast::exp(m0-top),f1=metal::fast::exp(m1-top);
                        for(uint dd=0;dd<128u;dd+=4u) {
                            float ov[4];
                            for(uint t=0;t<4;++t) {
                                float a=0.0f;
                                a+=f0*bfu(part[(h*64u+lane)*128u+dd+t]);
                                a+=f1*bfu(part[(h*64u+lane+32u)*128u+dd+t]);
                                ov[t]=simd_sum(a);
                            }
                            if(lane==0)for(uint t=0;t<4;++t) {
                                const float val=rb(zs==0.0f?ov[t]:fdiv(ov[t],zs));
                                attn[h*128u+dd+t]=ubf(val*sigmoid_f32(bfu(gate[h*128u+dd+t])));
                                if(FRT_TRACE && visit==params.debug_visit)dbgp[6u*FDBG+h*128u+dd+t]=val;
                            }
                        }
                    }
                    SYNC();
                }
            }
            // ---- output projection
            {
                const device ushort* wo=(const device ushort*)(weights+d[D_WO]);
                for(uint u=sg;u<576u;u+=FRT_GROUPS)gemv4<FQ>(wo,u*4u,(const device ushort*)attn,oproj,lane);
            }
            SYNC();
            // ---- post-attention conv; residual (restarts at a block boundary); MLP input
            {
                const device ushort* cw=(const device ushort*)(weights+d[D_ACONV]);
                for(uint i=tid;i<FH;i+=256u) {
                    float prev[3];ring_prev(ring,2048u+i,position,prev);
                    const float c=short_conv(prev,bfu(oproj[i]),cw+i*4u);
                    vr[i]=boundary?c:bfu(vh[i])+c;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(group==0)for(uint i=tid;i<FH;i+=256u)ring[rslot*FRING+2048u+i]=oproj[i];
                if(boundary){slots[nblocks]=bslot;++nblocks;}
                const float inv=rms_inv((const threadgroup float*)vr,red,lane,simd);
                depth_attention(bs,slots,nblocks,(const threadgroup float*)vr,inv,
                                (const device float*)(weights+d[D_MLP_Q]),1u,vxs,red,tid,lane,simd,zero,
                                (FRT_TRACE && visit==params.debug_visit && group==0)?dbgp+19u*FDBG:nullptr);
                if(dbg)for(uint i=tid;i<FH;i+=256u){dbgp[7u*FDBG+i]=vr[i];dbgp[8u*FDBG+i]=vxs[i];dbgp[21u*FDBG+i]=bs[slots[0]*FH+i];}
                if(dbg && tid==0){dbgp[23u*FDBG]=inv;dbgp[23u*FDBG+1]=float(nblocks);dbgp[23u*FDBG+2]=float(slots[0]);}
                if(dbg)for(uint i=tid;i<FH;i+=256u)dbgp[20u*FDBG+i]=((const device float*)(weights+d[D_MLP_Q]))[i];
            }
            if(messages) {
                // MLP message gate: s = [rms(mi) | rms(m)] (BF16, 4608) replaces mi in vxs; mi kept in `ain`
                const device float* m=msg+(li*4u+3u)*FH;
                if(group==0)for(uint i=tid;i<FH;i+=256u)ain[i]=vxs[i];
                const float ia=rms_inv((const threadgroup float*)vxs,red,lane,simd);
                const float im=rms_inv(m,red,lane,simd);
                float keep[9];
                for(uint k=0;k<9u;++k)keep[k]=vxs[tid+k*256u]*ia;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for(uint k=0;k<9u;++k){const uint i=tid+k*256u;vx[i]=ubf(keep[k]);vx[FH+i]=ubf(m[i]*im);}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(group<8u) {
                    const uint r=group*8u+simd;
                    const device ushort* wd=(const device ushort*)(weights+d[D_MG+9u])+r*4608u;
                    const float v=gemv_bn8_row(wd,4608u,(const threadgroup ushort*)vx,lane);
                    if(lane==0)mdown[192u+r]=rb(v);
                }
                SYNC();
                for(uint task=sg;task<144u;task+=FRT_GROUPS) {
                    const uint which=task/72u,row0=(task%72u)*32u;
                    const device ushort* w=(const device ushort*)(weights+d[D_MG+10u+which]);
                    gemv_k64(w,row0,mdown+192u,which==0u?forget:gain,lane);
                }
                SYNC();
                for(uint i=index;i<FH;i+=FRT_THREADS)
                    gated[i]=gate_combine(forget[i],gain[i],ain[i],m[i],zero);
                SYNC();
            }
            // ---- MLP: post-attention norm -> router -> top 8 -> experts
            {
                threadgroup ushort* y=(threadgroup ushort*)(vxs+FH);
                const device ushort* w=(const device ushort*)(weights+d[D_POST_NORM]);
                if(messages) {
                    const float inv=rms_inv((const device float*)gated,red,lane,simd);
                    for(uint i=tid;i<FH;i+=256u)y[i]=ubf(bfu(w[i])*(gated[i]*inv));
                } else {
                    const float inv=rms_inv((const threadgroup float*)vxs,red,lane,simd);
                    for(uint i=tid;i<FH;i+=256u)y[i]=ubf(bfu(w[i])*(vxs[i]*inv));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(dbg)for(uint i=tid;i<FH;i+=256u)dbgp[9u*FDBG+i]=bfu(y[i]);
                // router: FP32 logits = y @ W^T (gemv, bn = 8), one row per SIMD group
                if(sg<128u) {
                    const float v=gemv_bn8_row((const device ushort*)(weights+d[D_ROUTER])+sg*FH,FH,
                                               (const threadgroup ushort*)y,lane);
                    if(lane==0)rlog[sg]=v;
                }
            }
            SYNC();
            // ---- routing (mlx-lm expert_routing): sigmoid scores, + bias for the selection, top 8 (ties: lowest
            // index), weights = score / (sum + 1e-20) * 8
            if(simd==0) {
                const device float* rbias=(const device float*)(weights+d[D_RBIAS]);
                float score[4],biased[4];
                for(uint i=0;i<4;++i) {
                    const uint e=lane+32u*i;
                    const float x=rlog[e]*1.0f;
                    const float yv=fdiv(1.0f,1.0f+metal::precise::exp(metal::abs(x)));
                    score[i]=x<0.0f?yv:1.0f-yv;
                    biased[i]=score[i]+rbias[e];
                }
                float wk[8];
                for(uint k=0;k<8;++k) {
                    float best=-FRT_INF;uint best_e=128u;
                    for(uint i=0;i<4;++i) {
                        const uint e=lane+32u*i;
                        if(biased[i]>best || (biased[i]==best && e<best_e)){best=biased[i];best_e=e;}
                    }
                    const float top=simd_max(best);
                    const uint winner=simd_min(best==top?best_e:128u);
                    float s=0.0f;
                    for(uint i=0;i<4;++i)if(lane+32u*i==winner){s=score[i];biased[i]=-FRT_INF;}
                    wk[k]=simd_sum(s);
                    if(lane==0)route_ids[k]=winner;
                }
                float total=0.0f;
                for(uint k=0;k<8;++k)total=wk[k]+total;
                if(lane==0)for(uint k=0;k<8;++k)route_w[k]=fdiv(wk[k],total+1e-20f)*8.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(dbg && tid<8){dbgp[10u*FDBG+tid]=float(route_ids[tid]);dbgp[10u*FDBG+8u+tid]=route_w[tid];}
            // ---- experts: gate/up rows (interleaved: row 2c gate, 2c + 1 up) with the activation of their two
            // channels, then down rows
            {
                const device uchar* w13=weights+d[D_W13];
                const device uchar* s13=weights+d[D_S13];
                const device float* gs13=(const device float*)(weights+d[D_GS13]);
                const threadgroup ushort* y=(const threadgroup ushort*)(vxs+FH);
                for(uint u=sg;u<4608u;u+=FRT_GROUPS) {
                    const uint k=u/576u,row0=(u%576u)*4u;
                    const uint e=route_ids[k];
                    device ushort* out=edown+k*FH;
                    if(!mxfp8)
                        expert_rows<FH,false>(w13+size_t(e)*FH*(FH/2u),s13+size_t(e)*FH*(FH/16u),
                                              fdiv(gs13[e],2688.0f),row0,y,out,lane);
                    else
                        expert_rows<FH,true>(w13+size_t(e)*FH*FH,s13+size_t(e)*FH*(FH/32u),1.0f,row0,y,out,lane);
                    simdgroup_barrier(mem_flags::mem_device);
                    if(lane<2u) {
                        const float g=bfu(out[row0+lane*2u]),uu=bfu(out[row0+lane*2u+1u]);
                        act[k*FI+row0/2u+lane]=ubf(softcapped_swiglu(g,uu));
                    }
                }
            }
            SYNC();
            {
                const device uchar* w2=weights+d[D_W2];
                const device uchar* s2=weights+d[D_S2];
                const device float* gs2=(const device float*)(weights+d[D_GS2]);
                for(uint u=sg;u<4608u;u+=FRT_GROUPS) {
                    const uint k=u/576u,row0=(u%576u)*4u;
                    const uint e=route_ids[k];
                    const device ushort* x=act+k*FI;
                    if(!mxfp8)
                        expert_rows<FI,false>(w2+size_t(e)*FH*(FI/2u),s2+size_t(e)*FH*(FI/16u),
                                              fdiv(gs2[e],2688.0f),row0,x,edown+k*FH,lane);
                    else
                        expert_rows<FI,true>(w2+size_t(e)*FH*FI,s2+size_t(e)*FH*(FI/32u),1.0f,row0,x,edown+k*FH,lane);
                }
            }
            SYNC();
            // ---- combination (products rounded, summed in rank order), post-MLP conv, residual
            {
                float rwk[8];
                for(uint k=0;k<8;++k)rwk[k]=route_w[k];
                const device ushort* cw=(const device ushort*)(weights+d[D_MCONV]);
                for(uint i=tid;i<FH;i+=256u) {
                    float sum=0.0f;
                    for(uint k=0;k<8;++k)sum=sep(bfu(edown[k*FH+i])*rwk[k],zero)+sum;
                    const float xt=rb(sum);
                    if(dbg)dbgp[11u*FDBG+i]=xt;
                    float prev[3];ring_prev(ring,4352u+i,position,prev);
                    const float c=short_conv(prev,xt,cw+i*4u);
                    vh[i]=ubf(vr[i]+c);
                    if(group==0)ring[rslot*FRING+4352u+i]=ubf(xt);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
#if FRT_TRACE
                if(group==0)for(uint i=tid;i<FH;i+=256u)trace[visit*FH+i]=vh[i];
#endif
                if(layer==17u && pass<2u) {
                    const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
                    if(group==0)for(uint i=tid;i<FH;i+=256u)bank[pass*FH+i]=bfu(vh[i])*inv;
                }
            }
            SYNC();
        }
        // ---- final input, norm, LM head with softcap, selection
        {
            const uint fslots[4]={0u,1u,4u,5u};
            const float inv=rms_inv((const threadgroup ushort*)vh,red,lane,simd);
            depth_attention(bs,fslots,4u,(const threadgroup ushort*)vh,inv,
                            (const device float*)(weights+final[D_FINAL_Q]),1u,vr,red,tid,lane,simd,zero);
            const float inv2=rms_inv((const threadgroup float*)vr,red,lane,simd);
            const device ushort* w=(const device ushort*)(weights+final[D_FINAL_NORM]);
            for(uint i=tid;i<FH;i+=256u)vx[i]=ubf(bfu(w[i])*(vr[i]*inv2));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            const device ushort* head=(const device ushort*)(weights+final[D_HEAD]);
            device float* logits=block_logits+step*FVOCAB;
            for(uint u=sg;u<FVOCAB/4u;u+=FRT_GROUPS) {
                float r[4]={0.0f,0.0f,0.0f,0.0f};
                for(uint b=0;b<FH;b+=128u) {
                    const uint c=b+lane*4u;
                    const uint2 xr=*((const threadgroup uint2*)(vx+c));
                    const float v[4]={lo_f(xr.x),hi_f(xr.x),lo_f(xr.y),hi_f(xr.y)};
                    for(uint tm=0;tm<4;++tm) {
                        const uint2 wr=*((const device uint2*)(head+size_t(u*4u+tm)*FH+c));
                        r[tm]+=lo_f(wr.x)*v[0];r[tm]+=hi_f(wr.x)*v[1];r[tm]+=lo_f(wr.y)*v[2];r[tm]+=hi_f(wr.y)*v[3];
                    }
                }
                for(uint tm=0;tm<4;++tm)for(ushort sn=16;sn>=1;sn>>=1)r[tm]+=simd_shuffle_down(r[tm],sn);
                if(lane==0)for(uint tm=0;tm<4;++tm)
                    logits[u*4u+tm]=metal::precise::tanh(fdiv(rb(r[tm]),30.0f))*30.0f;
            }
        }
        SYNC();
        {
            device float* logits=block_logits+step*FVOCAB;
            float best=-FRT_INF;
            uint selected=0xffffffffu;
            for(uint word=index;word<FVOCAB_USED;word+=FRT_THREADS) {
                const float value=logits[word];
                if(value>best || (value==best && word<selected)){best=value;selected=word;}
            }
            const float peak=simd_max(best);
            const uint first=simd_min(best==peak?selected:0xffffffffu);
            if(lane==0){red[simd]=peak;((threadgroup uint*)red)[8u+simd]=first;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(simd==0) {
                const float value=lane<8u?red[lane]:-FRT_INF;
                const uint word=lane<8u?((threadgroup uint*)red)[8u+lane]:0xffffffffu;
                const float maximum=simd_max(value);
                const uint choice=simd_min(value==maximum?word:0xffffffffu);
                if(lane==0){stats[group*2u]=maximum;((device uint*)stats)[group*2u+1u]=choice;}
            }
            SYNC();
            if(simd==0) {
                float value=-FRT_INF;uint word=0xffffffffu;
                for(uint g=lane;g<FRT_WORKERS;g+=32u) {
                    const float gv=stats[g*2u];const uint gw=((device uint*)stats)[g*2u+1u];
                    if(gv>value || (gv==value && gw<word)){value=gv;word=gw;}
                }
                const float maximum=simd_max(value);
                const uint choice=simd_min(value==maximum?word:0xffffffffu);
                if(lane==0)red[32]=as_type<float>(choice);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            input_token=as_type<uint>(red[32]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(index==0){packet[step*2+1]=input_token;packet[16]=step+1u;}
        bool stop=false;
        for(uint e=0;e<params.eos_count;++e)stop=stop||input_token==params.eos[e];
        if(stop)break;
        SYNC();
    }
    if(tid==0)diagnostic[group]=0;
}
)MSL";
}

}  // namespace flashrt::apple::shaders
