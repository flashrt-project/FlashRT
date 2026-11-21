#pragma once
#include <string>

namespace flashrt::apple::shaders {
// Shared MSL for the Inkling 2 Mini (inkling2_loop) kernels: geometry constants, the weight-descriptor layout, and
// the device functions whose arithmetic reproduces MLX / mlx-lm for one row (number formats, RMSNorm, block
// attention residuals, BF16 / FP4 / FP8 row products, convolutions, elementwise operations). Every kernel that
// uses them runs 256-thread threadgroups.
inline const char* inkling2_common_msl() {
    return R"MSL(
// MLX compiles its kernels in the safe math mode; so do these sources.
#pragma METAL fp math_mode(safe)
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
// weighted sum. The arithmetic is spelled out as MLX's run-time compiled kernel performs it: fma accumulation, the
// scaled score rounded before the max subtraction, metal::exp (the fast exponential in that compile mode), a
// correctly rounded division, fma output sum. Here 256 threads: thread t owns virtual thread t, SIMD group 0 also
// virtual threads 256..287 (virtual SIMD group 8). K and Q are compile-time so the sums stay in registers.
template<uint K,uint Q,typename PP>
inline void depth_weights(const device float* blocks,thread const uint* slots,PP partial,float pinv,
                          const device float* queries,thread float* w,threadgroup float* red,
                          uint tid,uint lane,uint zero) {
    constexpr uint N=K+1u;
    const device float* bp[K>0?K:1];
    for(uint k=0;k<K;++k)bp[k]=blocks+slots[k]*FH;
    for(uint round=0;round<2;++round) {
        const uint vt=tid+round*256u;
        if(vt>=288u)break;
        float acc[N*Q];
        for(uint i=0;i<N*Q;++i)acc[i]=0.0f;
        for(uint d=vt;d<FH;d+=288u) {
            float x[N];
            for(uint k=0;k<K;++k)x[k]=bp[k][d];
            x[K]=sep(ld(partial,d)*pinv,zero);
            for(uint j=0;j<Q;++j) {
                const float qv=queries[j*FH+d];
                for(uint i=0;i<N;++i)acc[i*Q+j]=metal::fma(x[i],qv,acc[i*Q+j]);
            }
        }
        const uint vsg=vt/32u;
        for(uint i=0;i<N*Q;++i){const float v=simd_sum(acc[i]);if(lane==0)red[i*9u+vsg]=v;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=0;i<N*Q;++i){float t=0.0f;for(uint s=0;s<9u;++s)t+=red[i*9u+s];w[i]=sep(t*FDEPTH_SCALE,zero);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<Q;++j) {
        float m=-FRT_INF;
        for(uint i=0;i<N;++i)m=max(m,w[i*Q+j]);
        float z=0.0f;
        for(uint i=0;i<N;++i){w[i*Q+j]=metal::fast::exp(sep(w[i*Q+j]-m,zero));z+=w[i*Q+j];}
        for(uint i=0;i<N;++i)w[i*Q+j]=fdiv(w[i*Q+j],z);
    }
}
// Output of query j: out[d] = sum_i w[i][j] * source_i[d] (fma, sources in order).
template<uint K,uint Q,typename PP,typename OP>
inline void depth_output(const device float* blocks,thread const uint* slots,PP partial,float pinv,
                         thread const float* w,uint j,OP out,uint tid,uint zero) {
    constexpr uint N=K+1u;
    const device float* bp[K>0?K:1];
    for(uint k=0;k<K;++k)bp[k]=blocks+slots[k]*FH;
    float wj[N];
    for(uint i=0;i<N;++i)wj[i]=w[i*Q+j];
    for(uint d=tid;d<FH;d+=256u) {
        float o=0.0f;
        for(uint k=0;k<K;++k)o=metal::fma(wj[k],bp[k][d],o);
        o=metal::fma(wj[K],sep(ld(partial,d)*pinv,zero),o);
        out[d]=o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Q queries, all outputs (out + j * 2304), K at run time (1..4).
template<uint Q,typename PP,typename OP>
inline void depth_attention(const device float* blocks,thread const uint* slots,uint K,PP partial,float pinv,
                            const device float* queries,OP out,threadgroup float* red,uint tid,uint lane,uint zero) {
    float w[5*Q];
    switch(K) {
#define FRT_DEPTH_CASE(KK) case KK: \
        depth_weights<KK,Q>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); \
        for(uint j=0;j<Q;++j)depth_output<KK,Q>(blocks,slots,partial,pinv,w,j,out+j*FH,tid,zero); break;
        FRT_DEPTH_CASE(1) FRT_DEPTH_CASE(2) FRT_DEPTH_CASE(3) FRT_DEPTH_CASE(4)
#undef FRT_DEPTH_CASE
        default: break;
    }
}
// Run-time K (1..4) entry points: the softmax weights, then the output of one query.
template<uint Q,typename PP>
inline void depth_weights_k(const device float* blocks,thread const uint* slots,uint K,PP partial,float pinv,
                            const device float* queries,thread float* w,threadgroup float* red,uint tid,uint lane,
                            uint zero) {
    switch(K) {
        case 1: depth_weights<1,Q>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); break;
        case 2: depth_weights<2,Q>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); break;
        case 3: depth_weights<3,Q>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); break;
        default: depth_weights<4,Q>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); break;
    }
}
template<uint Q,typename PP,typename OP>
inline void depth_output_k(const device float* blocks,thread const uint* slots,uint K,PP partial,float pinv,
                           thread const float* w,uint j,OP out,uint tid,uint zero) {
    switch(K) {
        case 1: depth_output<1,Q>(blocks,slots,partial,pinv,w,j,out,tid,zero); break;
        case 2: depth_output<2,Q>(blocks,slots,partial,pinv,w,j,out,tid,zero); break;
        case 3: depth_output<3,Q>(blocks,slots,partial,pinv,w,j,out,tid,zero); break;
        default: depth_output<4,Q>(blocks,slots,partial,pinv,w,j,out,tid,zero); break;
    }
}
// Q = 3 weights, output of query j only.
template<typename PP,typename OP>
inline void depth_attention3_one(const device float* blocks,thread const uint* slots,uint K,PP partial,float pinv,
                                 const device float* queries,uint j,OP out,threadgroup float* red,uint tid,uint lane,
                                 uint zero) {
    float w[15];
    switch(K) {
#define FRT_DEPTH_CASE(KK) case KK: \
        depth_weights<KK,3>(blocks,slots,partial,pinv,queries,w,red,tid,lane,zero); \
        depth_output<KK,3>(blocks,slots,partial,pinv,w,j,out,tid,zero); break;
        FRT_DEPTH_CASE(1) FRT_DEPTH_CASE(2) FRT_DEPTH_CASE(3) FRT_DEPTH_CASE(4)
#undef FRT_DEPTH_CASE
        default: break;
    }
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

)MSL";
}
}  // namespace flashrt::apple::shaders
