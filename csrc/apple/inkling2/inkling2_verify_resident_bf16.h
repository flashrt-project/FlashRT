#pragma once
#include "common/source.h"
#include "inkling2/inkling2_common.h"
#include <string>

// Resident verification step for the inkling2_loop text decoder (Inkling 2 Mini): one dispatch of 36 co-resident
// threadgroups runs R (1..16) consecutive rows of one sequence (positions p .. p + R - 1: a speculative block, the
// anchor and its drafts) through the 48 layer visits and the LM head, with device-scope barriers between phases.
//
// Exact build: every row repeats the one-token resident step's arithmetic (inkling2_decode_resident_bf16) operation
// for operation, so each row's logits equal those of a decode step at its position, bit for bit, and the greedy
// targets are the tokens R decode steps would produce. Fast build (FRT_FAST): the same phases with the products in
// their own order: experts in half precision on power-of-two scaled inputs (exact half products, 16-term half
// sums, FP32 beyond), attention by blocks of key positions for all rows, beyond 8 rows the BF16 products by 8x8
// MMAs; with FRT_Q8 the attention projections (up to 12 rows) and LM head from Q8_0 copies. Per-row state lives in device memory (the one-token step's workspace layout, once per
// row, plus the row's hidden state, residual and projection inputs); weights are read once per step for all rows;
// the routed experts of the rows are deduplicated and handed out to threadgroups from a counter, heavy experts first.
//
// Rows interact only through attention (row r sees the keys of rows 0 .. r) and the short convolutions (the
// previous three inputs): the inputs and keys of the block's rows are read from where this step wrote them, the
// rest from the caches. Nothing is written to the convolution rings or the sliding-window caches: each row's ring
// frame and sliding K / V go to `stage` ([visit][row], the layout ink_commit_rows reads), committed after the
// accepted prefix is known; global-layer K / V are written by position (never read past a query). The drafter
// features of every row (the hidden state entering visits 5, 23, 41 and the final normalized state) go to `feat`.
// `stage` and `feat` keep the row stride of VMAX rows whatever the step's rows (the commit and the drafter context
// read the layout of a VMAX-row step).

namespace flashrt::apple::shaders {

struct Inkling2VerifyOptions {
    int rows = 8;           // rows of the step (1..16), a compile-time constant (loops over rows unrolled)
    bool trace = false;     // write every row's hidden state after every visit
    int workers = 36;
    bool fast = false;      // fast verification: own summation order in the products (not the decode step's bits)
    bool q8 = false;        // fast only: attention projections and LM head from Q8_0 copies (buffers 14, 15)
};

inline std::string inkling2_verify_resident_bf16_source(const Inkling2VerifyOptions& o = {}) {
    std::string defines = std::string("\n#define FRT_WORKERS ") + std::to_string(o.workers) + "u" +
        "\n#define FRT_ROWS " + std::to_string(o.rows) + "u" +
        "\n#define FRT_TRACE " + std::to_string(o.trace ? 1 : 0) +
        "\n#define FRT_FAST " + std::to_string(o.fast ? 1 : 0) +
        "\n#define FRT_Q8 " + std::to_string(o.q8 ? 1 : 0) + "\n";
    return std::string(metal_preamble) + defines + inkling2_common_msl() + R"MSL(
constant constexpr uint FRT_GROUPS=FRT_WORKERS*8u;
constant constexpr uint FRT_THREADS=FRT_WORKERS*256u;
constant constexpr uint VMAX=16u;                         // rows of a step at most
static_assert(FRT_ROWS>=1u && FRT_ROWS<=VMAX,"1..16 rows");
constant constexpr uint VSTAGE=FRING+2u*8u*128u;          // staged row of a visit: ring frame, then sliding K / V
// per-row extension after the one-token workspace layout (WS_*)
constant constexpr uint E_H=WS_END;                       // BF16 [2304] hidden state between visits
constant constexpr uint E_VR=E_H+4608u;                   // FP32 [2304] attention input, then the residual
constant constexpr uint E_X=E_VR+9216u;                   // BF16 [3][2304] projection inputs | final state
constant constexpr uint E_MI=E_X+13824u;                  // FP32 [2304] MLP input (depth attention)
constant constexpr uint E_Y=E_MI+9216u;                   // BF16 [2304] router / expert input
constant constexpr uint E_S=E_Y+4608u;                    // BF16 [4][4608] message gate inputs (q, k, v, mlp)
constant constexpr uint E_XT=E_S+36864u;                  // BF16 [2304] combined expert output (MLP conv input)
constant constexpr uint E_ROUTE=E_XT+4608u;               // uint [8] expert ids | FP32 [8] weights
constant constexpr uint E_XSC=E_ROUTE+64u;                // FP32 power-of-two input scales (fast): [0] y of the experts,
                                                          // [1..3] projection inputs q, k, v, [4] the final state (Q8)
constant constexpr uint WSR=E_XSC+64u;                    // bytes per row
constant constexpr uint G_ECOUNT=VMAX*WSR;                // uint: distinct experts of the step
constant constexpr uint G_ELIST=G_ECOUNT+16u;             // uint [128]: their ids
constant constexpr uint G_EMASK=G_ELIST+512u;             // ulong [128]: the expert's rank in each row (4 bits per row)
constant constexpr uint G_EROW=G_EMASK+1024u;             // uint [128]: the rows routed to it (a bit per row)
constant constexpr uint G_SEL=G_EROW+512u;                // [VMAX][FRT_WORKERS] (FP32 max, uint index)
constant constexpr uint FRT_DONE=256u;                    // control: per distinct expert, gate/up units done (fast)
#ifndef FRT_EUNITS
#define FRT_EUNITS 6u
#endif
#ifndef FRT_EUNITS_DOWN
#define FRT_EUNITS_DOWN 6u
#endif
constant constexpr uint EUN=FRT_EUNITS;                   // fast: gate/up units per expert (FH / EUN rows each)
constant constexpr uint EUD=FRT_EUNITS_DOWN;              // fast: down units per expert (FH / EUD rows each)
static_assert((FH/64u)%EUN==0u && (FH/128u)%EUD==0u,"whole tiles per unit");
struct VerifyParams {
    uint position;        // of row 0 (rows at position .. position + FRT_ROWS - 1, one sequence)
    uint zero;            // always 0 (keeps separately rounded MLX operations from being contracted)
    uint pad[2];
};

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
#define FRT_FAIL() {if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!grid_sync(control,tid,phase,ready))FRT_FAIL()

// The previous three inputs of a convolution channel at `position`: positions before the step from the ring
// (slot position mod 4), positions of the step's earlier rows from `cur` (row stride `stride`, BF16).
inline void prev_rows(const device ushort* ring,uint channel,uint position,uint p0,const device ushort* cur,
                      uint stride,thread float* prev) {
    for(uint i=0;i<3;++i) {
        const int pp=int(position)-3+int(i);
        if(pp<0)prev[i]=0.0f;
        else if(uint(pp)<p0)prev[i]=bfu(ring[(uint(pp)&3u)*FRING+channel]);
        else prev[i]=bfu(cur[size_t(uint(pp)-p0)*stride]);
    }
}

// ---------------------------------------------------------------------------- products over the step's rows
// The weights of a SIMD group's output rows are loaded once and applied to every row of the step; per row and
// output the arithmetic is that of the one-row helper (lane decomposition, statement order, reduction tree), so the
// result does not depend on the number of rows or on which output rows share a SIMD group. Row n of an operand
// is at base + n * WSR (the per-row workspace).
constant constexpr uint RS=WSR/2u;                        // row stride in BF16 elements
constant constexpr uint RSF=WSR/4u;                       // row stride in FP32 elements

// Threadgroup-staged gemv4: a threadgroup unit of 32 output rows (4 per SIMD group, row0 = this group's first);
// the rows' inputs are staged 768 at a time in threadgroup memory (xs: [FRT_ROWS][768] BF16), shared by the 8 SIMD
// groups. Every thread of the threadgroup calls it (barriers). HEAD: FP32 softcapped logits, row stride FVOCAB.
constant constexpr uint XCH=768u;                         // staged inputs per row (6 blocks of 128, 3 of 256)
constant constexpr uint PTM=4u;                           // output rows per SIMD group (threadgroup unit 32 rows)
template<uint K,bool HEAD>
inline void gemv_rows_tg(const device ushort* w,uint row0,const device ushort* x,device ushort* out,
                         device float* logits,threadgroup ushort* xs,uint tid,uint lane) {
    static_assert(K%XCH==0u,"inputs in whole chunks");
    float r[FRT_ROWS][PTM];
    for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<PTM;++tm)r[n][tm]=0.0f;
    uint2 wn[PTM];                                        // the next block's weights, loaded a block ahead
    for(uint tm=0;tm<PTM;++tm)wn[tm]=*((const device uint2*)(w+size_t(row0+tm)*K+lane*4u));
    for(uint c0=0;c0<K;c0+=XCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<FRT_ROWS*(XCH/8u);i+=256u) {
            const uint n=i/(XCH/8u),j=i%(XCH/8u);
            ((threadgroup uint4*)xs)[n*(XCH/8u)+j]=*((const device uint4*)(x+size_t(n)*RS+c0+j*8u));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint b=c0;b<c0+XCH;b+=128u) {
            const uint c=b+lane*4u;
            uint2 wr[PTM];
            for(uint tm=0;tm<PTM;++tm)wr[tm]=wn[tm];
            if(b+128u<K)for(uint tm=0;tm<PTM;++tm)wn[tm]=*((const device uint2*)(w+size_t(row0+tm)*K+c+128u));
            for(uint n=0;n<FRT_ROWS;++n) {
                const uint2 xr=((threadgroup const uint2*)xs)[n*(XCH/4u)+(c-c0)/4u];
                const float v[4]={lo_f(xr.x),hi_f(xr.x),lo_f(xr.y),hi_f(xr.y)};
                for(uint tm=0;tm<PTM;++tm) {
                    r[n][tm]+=lo_f(wr[tm].x)*v[0];r[n][tm]+=hi_f(wr[tm].x)*v[1];
                    r[n][tm]+=lo_f(wr[tm].y)*v[2];r[n][tm]+=hi_f(wr[tm].y)*v[3];
                }
            }
        }
    }
    for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<PTM;++tm)for(ushort sn=16;sn>=1;sn>>=1)r[n][tm]+=simd_shuffle_down(r[n][tm],sn);
    // lane 0 holds the sums: one output per lane for the epilogue (the softcap is ~100 instructions), 32 at a time
    for(uint o0=0;o0<FRT_ROWS*PTM;o0+=32u) {
        float mine=0.0f;
        for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<PTM;++tm) {
            const uint o=n*PTM+tm;
            if(o<o0 || o>=o0+32u)continue;
            const float v=simd_shuffle(r[n][tm],ushort(0));
            if(lane==o-o0)mine=v;
        }
        if(o0+lane<FRT_ROWS*PTM) {
            const uint n=(o0+lane)/PTM,tm=(o0+lane)%PTM;
            if(HEAD)logits[size_t(n)*FVOCAB+row0+tm]=metal::precise::tanh(fdiv(rb(mine),30.0f))*30.0f;
            else out[size_t(n)*RS+row0+tm]=ubf(mine);
        }
    }
}
// gemv_bn8_row of one weight row for every row (the sum on every lane): x [n][K]
template<uint K>
inline void gemv_bn8_rows(const device ushort* w,const device ushort* x,thread float* res,uint lane) {
    constexpr uint n_iter=K/1024u,left=K-n_iter*1024u;
    float total[FRT_ROWS];
    for(uint s=0;s<8;++s) {
        float r[FRT_ROWS];
        for(uint n=0;n<FRT_ROWS;++n)r[n]=0.0f;
        for(uint i=0;i<n_iter;++i) {
            const uint c=i*1024u+s*128u+lane*4u;
            float wv[4];for(uint t=0;t<4;++t)wv[t]=ld(w,c+t);
            for(uint n=0;n<FRT_ROWS;++n) {
                const device ushort* xn=x+size_t(n)*RS;
                float v[4];for(uint t=0;t<4;++t)v[t]=ld(xn,c+t);
                for(uint t=0;t<4;++t)r[n]+=wv[t]*v[t];
            }
        }
        if(left>0u) {
            const uint c=n_iter*1024u+s*128u+lane*4u;
            float m[4];
            for(uint t=0;t<4;++t){const bool in=c+t<K;m[t]=in?ld(w,c+t):0.0f;}
            for(uint n=0;n<FRT_ROWS;++n) {
                const device ushort* xn=x+size_t(n)*RS;
                float v[4];
                for(uint t=0;t<4;++t){const bool in=c+t<K;v[t]=in?ld(xn,c+t):0.0f;}
                for(uint t=0;t<4;++t)r[n]+=m[t]*v[t];
            }
        }
        for(uint n=0;n<FRT_ROWS;++n) {
            for(ushort sn=16;sn>=1;sn>>=1)r[n]+=simd_shuffle_down(r[n],sn);
            if(s==0)total[n]=r[n];else total[n]+=r[n];
        }
    }
    for(uint n=0;n<FRT_ROWS;++n)res[n]=simd_shuffle(total[n],ushort(0));
}
// gemv_k64 for every row: x [n][64] FP32 (stride RSF), out [n][rows] FP32
inline void gemv_k64_rows(const device ushort* w,uint row0,const device float* x,device float* out,uint lane) {
    const uint thr_m=lane/4u,thr_n=lane%4u;
    float r[FRT_ROWS][4];
    for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<4;++tm)r[n][tm]=0.0f;
    for(uint i=0;i<4;++i) {
        const uint c=i*16u+thr_n*4u;
        float wv[4][4];
        for(uint tm=0;tm<4;++tm)for(uint t=0;t<4;++t)wv[tm][t]=bfu(w[(row0+thr_m*4u+tm)*64u+c+t]);
        for(uint n=0;n<FRT_ROWS;++n) {
            float v[4];for(uint t=0;t<4;++t)v[t]=x[size_t(n)*RSF+c+t];
            for(uint tm=0;tm<4;++tm)for(uint t=0;t<4;++t)r[n][tm]+=wv[tm][t]*v[t];
        }
    }
    for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<4;++tm){
        r[n][tm]+=simd_shuffle_down(r[n][tm],ushort(2));r[n][tm]+=simd_shuffle_down(r[n][tm],ushort(1));}
    if(thr_n==0)for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<4;++tm)out[size_t(n)*RSF+row0+thr_m*4u+tm]=r[n][tm];
}
// qdot_nv / qdot_mx (inkling2_common) on decoded codes: the same sums, the codes decoded once for all slots
inline float qdot_nv_codes(thread const float* x,thread const float* q,float scale) {
    float accum=0.0f;
    for(uint i=0;i<2;++i)accum+=(x[4*i]*q[4*i]+x[4*i+1]*q[4*i+1]+x[4*i+2]*q[4*i+2]+x[4*i+3]*q[4*i+3]);
    return scale*accum;
}
template<uint N>
inline float qdot_mx_codes(thread const float* x,thread const float* q,float scale) {
    float accum=0.0f;
    for(uint i=0;i<N;++i)accum+=x[i]*q[i];
    return scale*accum;
}
#define FRT_UNROLL _Pragma("unroll")
constant constexpr uint XS_CAP=12288u;                    // staged slot inputs (BF16 / half elements, 24 KB)
constant constexpr uint EUT=8u;                           // exact experts: 32-row tiles per threadgroup unit
// Threadgroup unit of one expert, whole-input variant: the slots' complete inputs staged once (callers pass at most
// XS_CAP / K slots), then each SIMD group runs its tiles (4 rows each, tile t at rbase + t * 32) one after another
// over the whole K, accumulators per tile; per slot and output the arithmetic of expert_rows (codes decoded with
// inkling2_common's e2m1 / e4m3).
template<uint K,bool MX>
inline void expert_whole(const device uchar* w,const device uchar* s,float inv_enc,uint rbase,uint ntiles,
                         const device ushort* xb,thread const uint* sx,device ushort* ob,thread const uint* so,uint ns,
                         threadgroup ushort* xs,uint tid,uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<ns*(K/8u);i+=256u) {
        const uint n=i/(K/8u),j=i%(K/8u);
        ((threadgroup uint4*)xs)[n*(K/8u)+j]=*((const device uint4*)(xb+sx[n]+j*8u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=0;t<ntiles;++t) {
        const uint row0=rbase+t*32u;
        float r[FRT_ROWS][4];
        for(uint n=0;n<FRT_ROWS;++n)for(uint tm=0;tm<4;++tm)r[n][tm]=0.0f;
        if(!MX || K%256u==0u) {
            for(uint b=0;b<K;b+=256u) {
                const uint c=b+lane*8u;
                if(c>=K)break;
                float q[4][8],sc[4];
                for(uint tm=0;tm<4;++tm) {
                    const uint row=row0+tm;
                    if(!MX) {
                        const uint word=*((const device uint*)(w+size_t(row)*(K/2u)+c/2u));
                        for(uint i=0;i<8;++i)q[tm][i]=e2m1(word>>(4u*i));
                        sc[tm]=e4m3(s[size_t(row)*(K/16u)+c/16u]);
                    } else {
                        const uint2 word=*((const device uint2*)(w+size_t(row)*K+c));
                        for(uint i=0;i<4;++i){q[tm][i]=e4m3(word.x>>(8u*i));q[tm][4+i]=e4m3(word.y>>(8u*i));}
                        sc[tm]=e8m0(s[size_t(row)*(K/32u)+c/32u]);
                    }
                }
                for(uint n=0;n<FRT_ROWS;++n) {
                    if(n>=ns)break;
                    const uint4 xr=((threadgroup const uint4*)xs)[n*(K/8u)+c/8u];
                    const float xv[8]={lo_f(xr.x),hi_f(xr.x),lo_f(xr.y),hi_f(xr.y),lo_f(xr.z),hi_f(xr.z),lo_f(xr.w),hi_f(xr.w)};
                    for(uint tm=0;tm<4;++tm)
                        r[n][tm]+=(!MX?qdot_nv_codes(xv,q[tm],sc[tm]):qdot_mx_codes<8>(xv,q[tm],sc[tm]));
                }
            }
        } else {
            for(uint b=0;b<K;b+=128u) {
                const uint c=b+lane*4u;
                float q[4][4],sc[4];
                for(uint tm=0;tm<4;++tm) {
                    const uint row=row0+tm;
                    const uint word=*((const device uint*)(w+size_t(row)*K+c));
                    for(uint i=0;i<4;++i)q[tm][i]=e4m3(word>>(8u*i));
                    sc[tm]=e8m0(s[size_t(row)*(K/32u)+c/32u]);
                }
                for(uint n=0;n<FRT_ROWS;++n) {
                    if(n>=ns)break;
                    const uint2 xr=((threadgroup const uint2*)xs)[n*(K/4u)+c/4u];
                    const float xv[4]={lo_f(xr.x),hi_f(xr.x),lo_f(xr.y),hi_f(xr.y)};
                    for(uint tm=0;tm<4;++tm)r[n][tm]+=qdot_mx_codes<4>(xv,q[tm],sc[tm]);
                }
            }
        }
        for(uint n=0;n<FRT_ROWS;++n) {
            if(n>=ns)break;
            for(uint tm=0;tm<4;++tm){float v=simd_sum(r[n][tm]);if(lane==0)ob[so[n]+row0+tm]=ubf(MX?v:v*inv_enc);}
        }
    }
}

// ------------------------------------------------------------------------------------ fast verification (FRT_FAST)
// Not the decode step's summation order (results within rounding of it): an expert output row is computed by
// LPR = K / 288 lanes, each over 288 consecutive inputs (codes loaded 16 bytes at a time, decoded by shifts and masks
// into half2 pairs: E2M1 codes k, k + 4 of a 32-bit word with the magnitude in bits 9..11 times 2^14, E4M3 bytes k,
// k + 2 times 2^8; the block scale folded into one FP32 multiply), then summed over the row's lanes. A tile:
// RPS = 32 / LPR rows per SIMD group, 8 RPS rows per threadgroup; the slots' whole inputs staged in threadgroup memory
// in the pair order of the codes. (A byte -> half2 table in threadgroup memory is slower: its loads compete with the
// staged inputs'.)
template<uint k> inline half2 e2m1_pair(uint word) {
    constexpr int s1=9-4*int(k),s2=12-4*int(k);
    constexpr uint up=s1>=0?uint(s1):0u,down=s1<0?uint(-s1):0u;
    const uint m=((word<<up)>>down)&0x0E000E00u;
    const uint sg=(word<<uint(s2))&0x80008000u;
    return as_type<half2>(m|sg)*half2(16384.0h);
}
template<uint K,bool MX>
inline void expert_fast(const device uchar* w,const device uchar* s,float gsc,uint unit_row0,uint ntiles,
                        const device ushort* xb,thread const uint* sx,device ushort* ob,thread const uint* so,
                        thread const uint* sa,thread const float* xsc,uint ns,bool act,threadgroup half2* xh,
                        threadgroup const half* sct,uint tid,uint simd,uint lane) {
    constexpr uint LPR=K/288u,RPS=2u*32u/LPR,TROWS=8u*RPS;   // a lane: 2 rows
    constexpr uint NSL=FRT_ROWS<XS_CAP/K?FRT_ROWS:XS_CAP/K;  // slots of a staging group
    // the slots' inputs as half pairs: NVFP4 (x[8j + k], x[8j + k + 4]), k < 4; MXFP8 (x[4j + k], x[4j + k + 2]), k < 2
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<ns*(K/8u);i+=256u) {
        const uint n=i/(K/8u),j=i%(K/8u);
        const uint4 v=*((const device uint4*)(xb+sx[n]+j*8u));
        // times the slot's power-of-two scale (max |x| below 256): BF16 inputs and their products with the codes
        // stay exact halves, 16-input sums far from the half range
        const float q=xsc[n];
        const float f[8]={lo_f(v.x)*q,hi_f(v.x)*q,lo_f(v.y)*q,hi_f(v.y)*q,lo_f(v.z)*q,hi_f(v.z)*q,lo_f(v.w)*q,hi_f(v.w)*q};
        threadgroup half2* d=xh+i*4u;
        if(!MX)FRT_UNROLL for(uint k=0;k<4u;++k)d[k]=half2(half(f[k]),half(f[k+4]));
        else {d[0]=half2(half(f[0]),half(f[2]));d[1]=half2(half(f[1]),half(f[3]));
              d[2]=half2(half(f[4]),half(f[6]));d[3]=half2(half(f[5]),half(f[7]));}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // lanes of a row pair: LPR lanes, lane jl takes 9 of the 32-input chunks of both rows, in 3 runs of 3 consecutive
    // chunks (c = 3 LPR (b / 3) + 3 jl + b % 3: the lanes' runs adjoin, 48-byte runs per lane beat 16-byte
    // interleaving); each staged input pair read once for the two rows
    const uint rg=lane/LPR,jl=lane%LPR;
    for(uint t=0;t<ntiles;++t) {
        const uint rt0=unit_row0+t*TROWS+simd*RPS;
        const uint row=rt0+2u*rg;
        float acc[2][NSL];
        FRT_UNROLL for(uint r=0;r<2u;++r)FRT_UNROLL for(uint n=0;n<NSL;++n)acc[r][n]=0.0f;
        if(!MX) {
            const device uint4* cw=(const device uint4*)(w+size_t(row)*(K/2u));
            const device ushort* sp=(const device ushort*)(s+size_t(row)*(K/16u));
            FRT_UNROLL for(uint b=0;b<9u;++b) {
                const uint c=(b/3u)*(3u*LPR)+jl*3u+b%3u;
                half2 wv[2][16];
                float S0[2],S1[2];
                FRT_UNROLL for(uint r=0;r<2u;++r) {
                    const uint4 q=cw[r*(K/32u)+c];
                    const ushort sq=sp[r*(K/32u)+c];
                    S0[r]=float(sct[sq&255u]);S1[r]=float(sct[sq>>8]);
                    FRT_UNROLL for(uint wi=0;wi<4u;++wi) {
                        wv[r][4u*wi]=e2m1_pair<0>(q[wi]);wv[r][4u*wi+1u]=e2m1_pair<1>(q[wi]);
                        wv[r][4u*wi+2u]=e2m1_pair<2>(q[wi]);wv[r][4u*wi+3u]=e2m1_pair<3>(q[wi]);
                    }
                }
                const uint wb=(32u*c)/8u;
                FRT_UNROLL for(uint n=0;n<NSL;++n) {
                    if(n>=ns)break;
                    const threadgroup uint4* xp=(const threadgroup uint4*)(xh+(n*(K/8u)+wb)*4u);
                    const uint4 xa=xp[0],xb2=xp[1],xc=xp[2],xd=xp[3];
                    FRT_UNROLL for(uint r=0;r<2u;++r) {
                        half2 a=as_type<half2>(xa.x)*wv[r][0],bq=as_type<half2>(xa.y)*wv[r][1];
                        a=fma(as_type<half2>(xa.z),wv[r][2],a);bq=fma(as_type<half2>(xa.w),wv[r][3],bq);
                        a=fma(as_type<half2>(xb2.x),wv[r][4],a);bq=fma(as_type<half2>(xb2.y),wv[r][5],bq);
                        a=fma(as_type<half2>(xb2.z),wv[r][6],a);bq=fma(as_type<half2>(xb2.w),wv[r][7],bq);
                        half2 c2=as_type<half2>(xc.x)*wv[r][8],d2=as_type<half2>(xc.y)*wv[r][9];
                        c2=fma(as_type<half2>(xc.z),wv[r][10],c2);d2=fma(as_type<half2>(xc.w),wv[r][11],d2);
                        c2=fma(as_type<half2>(xd.x),wv[r][12],c2);d2=fma(as_type<half2>(xd.y),wv[r][13],d2);
                        c2=fma(as_type<half2>(xd.z),wv[r][14],c2);d2=fma(as_type<half2>(xd.w),wv[r][15],d2);
                        const float2 f2=float2(a+bq)*S0[r]+float2(c2+d2)*S1[r];
                        acc[r][n]+=f2.x+f2.y;
                    }
                }
            }
        } else {
            // MXFP8: chunk c = 32 code bytes (two 16-byte loads per row), one E8M0 scale; bytes k, k + 2 as a half2
            const device uint4* cw=(const device uint4*)(w+size_t(row)*K);
            const device uchar* sp=s+size_t(row)*(K/32u);
            FRT_UNROLL for(uint b=0;b<9u;++b) {
                const uint c=(b/3u)*(3u*LPR)+jl*3u+b%3u;
                FRT_UNROLL for(uint g=0;g<2u;++g) {
                    half2 wv[2][8];
                    float S[2];
                    FRT_UNROLL for(uint r=0;r<2u;++r) {
                        const uint4 q=cw[r*(K/16u)+2u*c+g];
                        S[r]=e8m0(sp[r*(K/32u)+c]);
                        FRT_UNROLL for(uint wi=0;wi<4u;++wi)
                            FRT_UNROLL for(uint k=0;k<2u;++k) {
                                const uint ws=q[wi]>>(8u*k);
                                wv[r][2u*wi+k]=as_type<half2>(((ws&0x007F007Fu)<<7)|((ws&0x00800080u)<<8))*half2(256.0h);
                            }
                    }
                    const uint wb=(32u*c+16u*g)/8u;
                    FRT_UNROLL for(uint n=0;n<NSL;++n) {
                        if(n>=ns)break;
                        const threadgroup uint4* xp=(const threadgroup uint4*)(xh+(n*(K/8u)+wb)*4u);
                        const uint4 xa=xp[0],xb2=xp[1];
                        FRT_UNROLL for(uint r=0;r<2u;++r) {
                            half2 a=as_type<half2>(xa.x)*wv[r][0],bq=as_type<half2>(xa.y)*wv[r][1];
                            a=fma(as_type<half2>(xa.z),wv[r][2],a);bq=fma(as_type<half2>(xa.w),wv[r][3],bq);
                            a=fma(as_type<half2>(xb2.x),wv[r][4],a);bq=fma(as_type<half2>(xb2.y),wv[r][5],bq);
                            a=fma(as_type<half2>(xb2.z),wv[r][6],a);bq=fma(as_type<half2>(xb2.w),wv[r][7],bq);
                            const float2 f2=float2(a+bq)*S[r];
                            acc[r][n]+=f2.x+f2.y;
                        }
                    }
                }
            }
        }
        // the lane's rows are a gate / up pair (rows 2 c, 2 c + 1) when act: the softcapped SwiGLU of the BF16-rounded
        // pair straight from the sums (the values a store and reload of the rounded outputs would give)
        FRT_UNROLL for(uint n=0;n<NSL;++n) {
            if(n>=ns)break;
            float v[2];
            FRT_UNROLL for(uint r=0;r<2u;++r) {
                v[r]=acc[r][n];
                for(ushort off=LPR/2u;off>=1u;off>>=1)v[r]+=simd_shuffle_xor(v[r],off);
                v[r]=rb(v[r]*(gsc/xsc[n]));
            }
            if(jl!=0u)continue;
            if(act)ob[sa[n]+row/2u]=ubf(softcapped_swiglu(v[0],v[1]));
            else{ob[so[n]+row]=ushort(as_type<uint>(v[0])>>16);ob[so[n]+row+1u]=ushort(as_type<uint>(v[1])>>16);}
        }
    }
}

#if FRT_FAST && FRT_ROWS>8u
// Fast build, more than 8 rows: BF16 products by 8x8 MMAs (FP32 sums in their own order). A threadgroup unit of
// 64 NB output rows, NB blocks of 8 per SIMD group, the rows in tiles of 8 (rows >= FRT_ROWS zero); inputs staged
// XCH at a time ([16][XCH] BF16, the 16-byte groups XOR-swizzled by row against bank conflicts). Inside a 64-wide
// chunk the k order is permuted (MMA step j, fragment position m -> k = 8 m + j): a lane's B elements are 8
// consecutive weights of each of its two output rows (one 16-byte load each), its A elements 16 consecutive staged
// inputs of its row. Out BF16 (row stride RS) or (HEAD) FP32 softcapped logits, vocabulary rows >= FVOCAB skipped.
template<uint K,bool HEAD,uint NB>
inline void gemv_rows_mma(const device ushort* w,uint unit_row0,const device ushort* x,device ushort* out,
                          device float* logits,threadgroup ushort* xs,uint tid,uint simd,uint lane) {
    constexpr uint T=(FRT_ROWS+7u)/8u,RP=8u*T,XG=XCH/8u;
    static_assert(RP*XCH<=XS_CAP && K%XCH==0u && XG%8u==0u,"staged inputs");
    const ushort fm=((lane/4u)&4u)+(lane/2u)%4u,fn=((lane/4u)&2u)*2u+(lane%2u)*2u;
    const uint n0=unit_row0+simd*8u*NB;
    bool live[NB];
    FRT_UNROLL for(uint b=0;b<NB;++b)live[b]=!HEAD || n0+b*8u<FVOCAB;
    simdgroup_float8x8 C[T][NB];
    FRT_UNROLL for(uint t=0;t<T;++t)FRT_UNROLL for(uint b=0;b<NB;++b)C[t][b]=simdgroup_float8x8(0.0f);
    threadgroup uint4* xg=(threadgroup uint4*)xs;
    for(uint c0=0;c0<K;c0+=XCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<RP*XG;i+=256u) {
            const uint n=i/XG,j=i%XG;
            xg[n*XG+(j^(n&7u))]=n<FRT_ROWS?*((const device uint4*)(x+size_t(n)*RS+c0+j*8u)):uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kc=0;kc<XCH;kc+=64u) {
            uint4 w0[NB],w1[NB];
            FRT_UNROLL for(uint b=0;b<NB;++b) {
                const uint n=n0+b*8u+fn;
                w0[b]=live[b]?*((const device uint4*)(w+size_t(n)*K+c0+kc+fm*8u)):uint4(0u);
                w1[b]=live[b]?*((const device uint4*)(w+size_t(n+1u)*K+c0+kc+fm*8u)):uint4(0u);
            }
            uint4 a0[T],a1[T];
            FRT_UNROLL for(uint t=0;t<T;++t) {
                const uint row=t*8u+fm,j0=kc/8u+fn;
                a0[t]=xg[row*XG+(j0^(row&7u))];a1[t]=xg[row*XG+((j0+1u)^(row&7u))];
            }
            FRT_UNROLL for(uint j=0;j<8u;++j) {
                simdgroup_float8x8 A[T];
                FRT_UNROLL for(uint t=0;t<T;++t) {
                    A[t].thread_elements()[0]=j%2u?hi_f(a0[t][j/2u]):lo_f(a0[t][j/2u]);
                    A[t].thread_elements()[1]=j%2u?hi_f(a1[t][j/2u]):lo_f(a1[t][j/2u]);
                }
                FRT_UNROLL for(uint b=0;b<NB;++b) {
                    if(!live[b])continue;
                    simdgroup_float8x8 B;
                    B.thread_elements()[0]=j%2u?hi_f(w0[b][j/2u]):lo_f(w0[b][j/2u]);
                    B.thread_elements()[1]=j%2u?hi_f(w1[b][j/2u]):lo_f(w1[b][j/2u]);
                    FRT_UNROLL for(uint t=0;t<T;++t)simdgroup_multiply_accumulate(C[t][b],A[t],B,C[t][b]);
                }
            }
        }
    }
    FRT_UNROLL for(uint t=0;t<T;++t)FRT_UNROLL for(uint b=0;b<NB;++b) {
        const uint r=t*8u+fm,col=n0+b*8u+fn;
        if(r>=FRT_ROWS || !live[b])continue;
        const float v0=C[t][b].thread_elements()[0],v1=C[t][b].thread_elements()[1];
        if(HEAD) {
            logits[size_t(r)*FVOCAB+col]=metal::precise::tanh(fdiv(rb(v0),30.0f))*30.0f;
            logits[size_t(r)*FVOCAB+col+1u]=metal::precise::tanh(fdiv(rb(v1),30.0f))*30.0f;
        } else {
            *((device uint*)(out+size_t(r)*RS+col))=uint(ubf(v0))|(uint(ubf(v1))<<16);
        }
    }
}
#endif
// Q8_0 threadgroup unit (fast, FRT_Q8): 64 output rows, a SIMD group 8 (lanes 8 rg .. 8 rg + 7: rows row0 + 2 rg,
// + 1; lane jl the 32-input chunks c = 8 b + jl of a half of K); x [rows][K] (row stride RS) times the row's
// power-of-two scale staged as half (max |x| 2^e below 16: 16-term half2 sums below 2^15); per chunk the half2 sums
// times the block scale into FP32; the result divided by the row scale. Out BF16 (row stride RS) or (HEAD) FP32
// softcapped logits. q8 table entry qt: (codes, scales) byte offsets in q8.
enum : uint { Q8_WQ=0, Q8_WK=2, Q8_WV=4, Q8_WR=6, Q8_WG=8, Q8_WO=10, Q8_LAYER=12 };
// staged input chunks of a K-wide product: the fewest that fit every row's chunk in XS_CAP halves (32-wide blocks)
constexpr uint q8_chunks(uint K) {
    uint n=(FRT_ROWS*K+XS_CAP-1u)/XS_CAP;
    while(K%(32u*n)!=0u)++n;
    return n;
}
template<uint K,bool HEAD,uint TP=1u>
inline void q8_rows_tg(const device uchar* q8,const device ulong* qt,uint unit_row0,const device ushort* x,
                       const device float* xsc,uint xsc_stride,device ushort* out,device float* logits,
                       threadgroup half* xh,uint tid,uint simd,uint lane) {
    constexpr uint NCH=q8_chunks(K),KH=K/NCH,NC=KH/32u;          // staged inputs: K / NCH per row
    const device char* codes=(const device char*)(q8+qt[0]);
    const device half* scl=(const device half*)(q8+qt[1]);
    const uint rg=lane/8u,jl=lane%8u;
    const uint row0=unit_row0+simd*8u+rg*2u;            // tile t: rows row0 + 64 t, + 1
    float acc[TP][2][FRT_ROWS];
    FRT_UNROLL for(uint t=0;t<TP;++t)FRT_UNROLL for(uint q=0;q<2u;++q)FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r)acc[t][q][r]=0.0f;
    for(uint hk=0;hk<NCH;++hk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<FRT_ROWS*(KH/8u);i+=256u) {
            const uint r=i/(KH/8u),j=i%(KH/8u);
            const float sc=xsc?xsc[size_t(r)*xsc_stride]:1.0f;
            const uint4 v=*((const device uint4*)(x+size_t(r)*RS+hk*KH+j*8u));
            threadgroup half4* d=(threadgroup half4*)(xh+r*KH+j*8u);
            d[0]=half4(float4(lo_f(v.x),hi_f(v.x),lo_f(v.y),hi_f(v.y))*sc);
            d[1]=half4(float4(lo_f(v.z),hi_f(v.z),lo_f(v.w),hi_f(v.w))*sc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint b=0;b*8u<NC;++b) {
            const uint cl=b*8u+jl;
            if(cl>=NC)break;
            const uint c=hk*NC+cl;
            FRT_UNROLL for(uint t=0;t<TP;++t) {
                const uint row=row0+t*64u;
                if(HEAD && row>=FVOCAB)break;
                half2 wv[2][16];
                float S[2];
                FRT_UNROLL for(uint q=0;q<2u;++q) {
                    const device uint4* cw=(const device uint4*)(codes+size_t(row+q)*K)+2u*c;
                    const uint4 a=cw[0],bb=cw[1];
                    S[q]=float(scl[size_t(row+q)*(K/32u)+c]);
                    const uint w8[8]={a.x,a.y,a.z,a.w,bb.x,bb.y,bb.z,bb.w};
                    FRT_UNROLL for(uint i=0;i<8u;++i){const half4 h=half4(as_type<char4>(w8[i]));wv[q][2u*i]=h.xy;wv[q][2u*i+1u]=h.zw;}
                }
                FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r) {
                    const threadgroup half2* xp=(const threadgroup half2*)(xh+r*KH+32u*cl);
                    FRT_UNROLL for(uint q=0;q<2u;++q) {
                        half2 s0=xp[0]*wv[q][0],s1=xp[1]*wv[q][1];
                        FRT_UNROLL for(uint i=2;i<16u;i+=2){s0=fma(xp[i],wv[q][i],s0);s1=fma(xp[i+1],wv[q][i+1],s1);}
                        const float2 f=float2(s0+s1);
                        acc[t][q][r]=metal::fma(S[q],f.x+f.y,acc[t][q][r]);
                    }
                }
            }
        }
    }
    // the row group's 8 lanes all hold the sums after the xor tree; lane jl writes row r0 + jl of the step
    FRT_UNROLL for(uint t=0;t<TP;++t) {
        const uint row=row0+t*64u;
        if(HEAD && row>=FVOCAB)break;
        FRT_UNROLL for(uint q=0;q<2u;++q)FRT_UNROLL for(uint r=0;r<FRT_ROWS;++r) {
            float v=acc[t][q][r];
            v+=simd_shuffle_xor(v,4);v+=simd_shuffle_xor(v,2);v+=simd_shuffle_xor(v,1);
            acc[t][q][r]=v;
        }
        FRT_UNROLL for(uint r0=0;r0<FRT_ROWS;r0+=8u) {
            float mine[2]={0.0f,0.0f};
            FRT_UNROLL for(uint q=0;q<2u;++q)FRT_UNROLL for(uint r=r0;r<r0+8u && r<FRT_ROWS;++r)if(jl==r-r0)mine[q]=acc[t][q][r];
            const uint rr=r0+jl;
            if(rr<FRT_ROWS) {
                const float inv=1.0f/(xsc?xsc[size_t(rr)*xsc_stride]:1.0f);
                FRT_UNROLL for(uint q=0;q<2u;++q) {
                    const float v=mine[q]*inv;
                    if(HEAD)logits[size_t(rr)*FVOCAB+row+q]=metal::precise::tanh(fdiv(rb(v),30.0f))*30.0f;
                    else out[size_t(rr)*RS+row+q]=ubf(v);
                }
            }
        }
    }
}
// max |x| of the n values a threadgroup wrote (any thread's share in m) -> power-of-two scale (max x 2^e in [8, 16))
inline float tg_pow2_scale(float m,threadgroup float* red,uint tid,uint lane,uint simd) {
    m=simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0)red[simd]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t=red[0];for(uint i=1;i<8u;++i)t=max(t,red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ex=t>0.0f?int((as_type<uint>(t)>>23)&255u)-127:0;
    return as_type<float>(uint(127+clamp(3-ex,-100,100))<<23);
}

kernel void inkling2_verify_resident_bf16(
    const device uchar* weights [[buffer(0)]],
    const device ulong* table [[buffer(1)]],
    device ushort* kv [[buffer(2)]],
    const device ushort* rings [[buffer(3)]],
    constant VerifyParams& params [[buffer(4)]],
    const device uint* tokens [[buffer(5)]],
    device ushort* stage [[buffer(6)]],
    device ushort* feat [[buffer(7)]],
    device float* logits [[buffer(8)]],
    device uint* out [[buffer(9)]],
    device uchar* workspace [[buffer(10)]],
    device ushort* trace [[buffer(11)]],
    device uint* private_control [[buffer(12)]],
    device uint* diagnostic [[buffer(13)]],
#if FRT_Q8
    const device uchar* q8 [[buffer(14)]],
    const device ulong* q8t [[buffer(15)]],
#endif
    uint3 group3 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint group=group3.x,sg=group*8+simd,index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    const uint zero=params.zero,p0=params.position;
    const uint R=FRT_ROWS;
    threadgroup uint ready[1];
    uint phase=0;
    threadgroup float red[256];
    threadgroup ulong desc[64];
    threadgroup float tv[XS_CAP/2u>3072u?XS_CAP/2u:3072u];   // FP32 row scratch | staged inputs (XS_CAP BF16)
    threadgroup ushort* xs=(threadgroup ushort*)tv;
    threadgroup uint slot_x[VMAX],slot_o[VMAX],slot_a[VMAX],slot_n[1];
#if FRT_FAST
    threadgroup half sct[256];              // fast: E4M3 bytes as halves
    sct[tid]=half(e4m3(tid));
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device ulong* final=table+FLAYERS*64u;
    const device ulong* visits=final+64u;
    const device ushort* embed=(const device ushort*)(weights+final[D_EMBED]);
    #define RW(r) (workspace+size_t(r)*WSR)
    #define RF(r,off) ((device float*)(RW(r)+(off)))
    #define RU(r,off) ((device ushort*)(RW(r)+(off)))
    device uint* ecount=(device uint*)(workspace+G_ECOUNT);
    device uint* elist=(device uint*)(workspace+G_ELIST);
    device ulong* emask=(device ulong*)(workspace+G_EMASK);
    device uint* erow=(device uint*)(workspace+G_EROW);
    device float* sel=(device float*)(workspace+G_SEL);

    // embedding row + embed_norm, row r on threadgroup r
    if(group<R) {
        const uint r=group;
        device ushort* vh=RU(r,E_H);
        for(uint i=tid;i<FH;i+=256u)vh[i]=embed[size_t(tokens[r])*FH+i];
        threadgroup_barrier(mem_flags::mem_device);
        const float inv=rms_inv((const device ushort*)vh,red,lane,simd);
        const device ushort* w=(const device ushort*)(weights+final[D_EMBED_NORM]);
        for(uint i=tid;i<FH;i+=256u)vh[i]=ubf(bfu(w[i])*rb(bfu(vh[i])*inv));
    }
    SYNC();
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
        const uint cap=uint(vt[V_CAP]);
        device ushort* kcache=kv+vt[V_KV];
        device ushort* vcache=kcache+size_t(8u)*cap*128u;
        const device ushort* ring=rings+vt[V_RING];
        device ushort* vstage=stage+size_t(visit)*VMAX*VSTAGE;

        // ---- pass start: the prelude's state (restored by threadgroup r); the messages of the pass
        if(layer==6u) {
            nblocks=1;slots[0]=0;
            if(pass>0u) {
                if(group<R)for(uint i=tid;i<FH;i+=256u)RU(group,E_H)[i]=RU(group,WS_PRELUDE)[i];
                const uint bk[1]={2u+pass-1u};
                for(uint u=group;u<12u*R;u+=FRT_WORKERS) {
                    const uint r=u/12u,q=u%12u;
                    depth_attention<4>(RF(r,WS_BS),bk,1u,(const device float*)(RF(r,WS_BANK)+(pass-1u)*FH),1.0f,
                                       (const device float*)(weights+final[D_MSG_Q])+q*4u*FH,
                                       RF(r,WS_MSG)+q*4u*FH,red,tid,lane,zero);
                }
                SYNC();
            }
        }
        // ---- attention inputs (row r on threadgroup r): depth attention over the block states
        float dw[15];
        if(group<R) {
            const uint r=group;
            device ushort* vh=RU(r,E_H);
            device float* bs=RF(r,WS_BS);
            if(layer==6u && pass==0u)for(uint i=tid;i<FH;i+=256u)RU(r,WS_PRELUDE)[i]=vh[i];
            const float rinv=rms_inv((const device ushort*)vh,red,lane,simd);
            if(boundary)for(uint i=tid;i<FH;i+=256u)bs[bslot*FH+i]=bfu(vh[i])*rinv;
            threadgroup_barrier(mem_flags::mem_device);
            const device float* aq=(const device float*)(weights+d[D_ATTN_Q]);
            if(layer>0u) {
                if(full)depth_weights_k<3>(bs,slots,nblocks,(const device ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
                else depth_weights_k<1>(bs,slots,nblocks,(const device ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
            }
            if(!messages) {
                device ushort* vx=RU(r,E_X);
                const uint nq=full?3u:1u;
                for(uint j=0;j<nq;++j) {
                    const device ushort* w=(const device ushort*)(weights+d[D_IN_NORM+j]);
                    if(layer==0u) {
                        for(uint i=tid;i<FH;i+=256u)vx[i]=ubf(bfu(w[i])*rb(bfu(vh[i])*rinv));
                    } else {
                        if(full)depth_output_k<3>(bs,slots,nblocks,(const device ushort*)vh,rinv,dw,j,tv,tid,zero);
                        else depth_output_k<1>(bs,slots,nblocks,(const device ushort*)vh,rinv,dw,0u,tv,tid,zero);
                        const float inv=rms_inv((const threadgroup float*)tv,red,lane,simd);
                        for(uint i=tid;i<FH;i+=256u)vx[j*FH+i]=ubf(bfu(w[i])*(tv[i]*inv));
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if(!full)for(uint i=tid;i<FH;i+=256u){vx[FH+i]=vx[i];vx[2u*FH+i]=vx[i];}
#if FRT_Q8
                threadgroup_barrier(mem_flags::mem_device);
                for(uint j=0;j<3u;++j) {
                    float m=0.0f;
                    for(uint i=tid;i<FH;i+=256u)m=max(m,metal::abs(bfu(vx[j*FH+i])));
                    const float sc=tg_pow2_scale(m,red,tid,lane,simd);
                    if(tid==0)((device float*)(RW(r)+E_XSC))[1u+j]=sc;
                }
#endif
            }
        }
        if(messages) {
            // message gates q, k, v of every row: unit (row, gate) on threadgroup 3 r + j
            for(uint u=group;u<3u*R;u+=FRT_WORKERS) {
                const uint r=u/3u,j=u%3u;
                device ushort* vh=RU(r,E_H);
                device float* bs=RF(r,WS_BS);
                const float rinv=rms_inv((const device ushort*)vh,red,lane,simd);
                const device float* aq=(const device float*)(weights+d[D_ATTN_Q]);
                if(full)depth_weights_k<3>(bs,slots,nblocks,(const device ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
                else depth_weights_k<1>(bs,slots,nblocks,(const device ushort*)vh,rinv,aq,dw,red,tid,lane,zero);
                if(full)depth_output_k<3>(bs,slots,nblocks,(const device ushort*)vh,rinv,dw,j,tv,tid,zero);
                else depth_output_k<1>(bs,slots,nblocks,(const device ushort*)vh,rinv,dw,0u,tv,tid,zero);
                for(uint i=tid;i<FH;i+=256u)RF(r,WS_AIN)[j*FH+i]=tv[i];
                const device float* m=RF(r,WS_MSG)+(li*4u+j)*FH;
                const float ia=rms_inv((const threadgroup float*)tv,red,lane,simd);
                const float im=rms_inv(m,red,lane,simd);
                device ushort* s=RU(r,E_S)+j*4608u;
                for(uint i=tid;i<FH;i+=256u){s[i]=ubf(tv[i]*ia);s[FH+i]=ubf(m[i]*im);}
            }
            SYNC();
            for(uint u=sg;u<192u;u+=FRT_GROUPS) {
                const uint j=u/64u,row=u%64u;
                const device ushort* wd=(const device ushort*)(weights+d[D_MG+j*3u])+row*4608u;
                float res[FRT_ROWS];
                gemv_bn8_rows<4608u>(wd,(const device ushort*)(RU(0,E_S)+j*4608u),res,lane);
                if(lane==0)for(uint r=0;r<R;++r)RF(r,WS_MDOWN)[j*64u+row]=rb(res[r]);
            }
            SYNC();
            for(uint task=sg;task<432u;task+=FRT_GROUPS) {
                const uint gj=task/144u,which=(task/72u)%2u,row0=(task%72u)*32u;
                const device ushort* w=(const device ushort*)(weights+d[D_MG+gj*3u+1u+which]);
                gemv_k64_rows(w,row0,RF(0,WS_MDOWN)+gj*64u,(which==0u?RF(0,WS_FORGET):RF(0,WS_GAIN))+gj*FH,lane);
            }
            SYNC();
            for(uint i=index;i<R*3u*FH;i+=FRT_THREADS) {
                const uint r=i/(3u*FH),gj=(i/FH)%3u,e=i%FH;
                RF(r,WS_GATED)[gj*FH+e]=gate_combine(RF(r,WS_FORGET)[gj*FH+e],RF(r,WS_GAIN)[gj*FH+e],
                                                     RF(r,WS_AIN)[gj*FH+e],RF(r,WS_MSG)[(li*4u+gj)*FH+e],zero);
            }
            SYNC();
            for(uint u=group;u<3u*R;u+=FRT_WORKERS) {
                const uint r=u/3u,jj=u%3u;
                const device float* src=RF(r,WS_GATED)+jj*FH;
                const float inv=rms_inv(src,red,lane,simd);
                const device ushort* w=(const device ushort*)(weights+d[D_IN_NORM+(full?jj:0u)]);
                float m=0.0f;
                for(uint i=tid;i<FH;i+=256u){const ushort v=ubf(bfu(w[i])*(src[i]*inv));RU(r,E_X)[jj*FH+i]=v;m=max(m,metal::abs(bfu(v)));}
#if FRT_Q8
                const float sc=tg_pow2_scale(m,red,tid,lane,simd);
                if(tid==0)((device float*)(RW(r)+E_XSC))[1u+jj]=sc;
#endif
            }
        }
        SYNC();
        // ---- projections of every row: threadgroup units of 64 output rows (q 48, r 6, gate 48, k 16, v 16) with
        // staged inputs; sinks and dynamic conv weights one output row per SIMD group (3 + 8 units)
        {
            const device ushort* wq=(const device ushort*)(weights+d[D_WQ]);
            const device ushort* wk=(const device ushort*)(weights+d[D_WK]);
            const device ushort* wv=(const device ushort*)(weights+d[D_WV]);
            const device ushort* wr=(const device ushort*)(weights+d[D_WR]);
            const device ushort* wg=(const device ushort*)(weights+d[D_WG]);
            const device ushort* xq=RU(0,E_X);
            const device ushort* xk=xq+FH;
            const device ushort* xv=xq+2u*FH;
            device ushort* proj=RU(0,WS_PROJ);
// Q8_0 up to 12 rows; beyond, the BF16 weights by MMAs (the Q8_0 multi-row product is ALU-bound there)
#if FRT_Q8 && FRT_ROWS<=12u
            const device ulong* qt=q8t+layer*Q8_LAYER;
            const device float* xsc=(const device float*)(RW(0)+E_XSC);
            // units of QR rows (two 64-row tiles per staged input): q, r, gate, k, v
            constexpr uint QTP=2u,QR=64u*QTP;
            constexpr uint U1=3072u/QR,U2=U1+384u/QR,U3=U2+3072u/QR,U4=U3+1024u/QR,U5=U4+1024u/QR;
            const uint units8=U5+3u+(full?8u:0u);
            for(uint u=group;u<units8;u+=FRT_WORKERS) {
                if(u<U1)q8_rows_tg<FH,false,QTP>(q8,qt+Q8_WQ,u*QR,xq,xsc+1,RSF,proj+P_Q,logits,(threadgroup half*)xs,tid,simd,lane);
                else if(u<U2)q8_rows_tg<FH,false,QTP>(q8,qt+Q8_WR,(u-U1)*QR,xq,xsc+1,RSF,proj+P_R,logits,(threadgroup half*)xs,tid,simd,lane);
                else if(u<U3)q8_rows_tg<FH,false,QTP>(q8,qt+Q8_WG,(u-U2)*QR,xq,xsc+1,RSF,proj+P_G,logits,(threadgroup half*)xs,tid,simd,lane);
                else if(u<U4)q8_rows_tg<FH,false,QTP>(q8,qt+Q8_WK,(u-U3)*QR,xk,xsc+2,RSF,proj+P_K,logits,(threadgroup half*)xs,tid,simd,lane);
                else if(u<U5)q8_rows_tg<FH,false,QTP>(q8,qt+Q8_WV,(u-U4)*QR,xv,xsc+3,RSF,proj+P_V,logits,(threadgroup half*)xs,tid,simd,lane);
                else if(u<U5+3u) {
                    const uint rr=(u-U5)*8u+simd;
                    float res[FRT_ROWS];
                    gemv_bn8_rows<FH>((const device ushort*)(weights+d[D_WS])+rr*FH,xq,res,lane);
                    if(lane==0)for(uint r=0;r<R;++r)proj[size_t(r)*RS+P_S+rr]=ubf(res[r]);
                } else {
                    const uint rr=(u-U5-3u)*8u+simd;
                    const bool kd=rr<32u;
                    float res[FRT_ROWS];
                    gemv_bn8_rows<FH>((const device ushort*)(weights+d[kd?D_KDYN:D_VDYN])+(rr%32u)*FH,kd?xk:xv,res,lane);
                    if(lane==0)for(uint r=0;r<R;++r)proj[size_t(r)*RS+(kd?P_KD:P_VD)+rr%32u]=ubf(res[r]);
                }
            }
            if(false)
#endif
            {
#if FRT_FAST && FRT_ROWS>8u
            constexpr uint TU=128u;                          // rows of a threadgroup unit (MMA products)
#else
            constexpr uint TU=8u*PTM;                        // rows of a threadgroup unit
#endif
            constexpr uint UQ=3072u/TU,UR=UQ+384u/TU,UG=UR+3072u/TU,UK=UG+1024u/TU,UV=UK+1024u/TU;
            const uint units=UV+3u+(full?8u:0u);
            for(uint u=group;u<units;u+=FRT_WORKERS) {
#if FRT_FAST && FRT_ROWS>8u
                if(u<UQ)gemv_rows_mma<FH,false,2u>(wq,u*TU,xq,proj+P_Q,logits,xs,tid,simd,lane);
                else if(u<UR)gemv_rows_mma<FH,false,2u>(wr,(u-UQ)*TU,xq,proj+P_R,logits,xs,tid,simd,lane);
                else if(u<UG)gemv_rows_mma<FH,false,2u>(wg,(u-UR)*TU,xq,proj+P_G,logits,xs,tid,simd,lane);
                else if(u<UK)gemv_rows_mma<FH,false,2u>(wk,(u-UG)*TU,xk,proj+P_K,logits,xs,tid,simd,lane);
                else if(u<UV)gemv_rows_mma<FH,false,2u>(wv,(u-UK)*TU,xv,proj+P_V,logits,xs,tid,simd,lane);
#else
                if(u<UQ)gemv_rows_tg<FH,false>(wq,u*TU+simd*PTM,xq,proj+P_Q,logits,xs,tid,lane);
                else if(u<UR)gemv_rows_tg<FH,false>(wr,(u-UQ)*TU+simd*PTM,xq,proj+P_R,logits,xs,tid,lane);
                else if(u<UG)gemv_rows_tg<FH,false>(wg,(u-UR)*TU+simd*PTM,xq,proj+P_G,logits,xs,tid,lane);
                else if(u<UK)gemv_rows_tg<FH,false>(wk,(u-UG)*TU+simd*PTM,xk,proj+P_K,logits,xs,tid,lane);
                else if(u<UV)gemv_rows_tg<FH,false>(wv,(u-UK)*TU+simd*PTM,xv,proj+P_V,logits,xs,tid,lane);
#endif
                else if(u<UV+3u) {
                    const uint rr=(u-UV)*8u+simd;
                    float res[FRT_ROWS];
                    gemv_bn8_rows<FH>((const device ushort*)(weights+d[D_WS])+rr*FH,xq,res,lane);
                    if(lane==0)for(uint r=0;r<R;++r)proj[size_t(r)*RS+P_S+rr]=ubf(res[r]);
                } else {
                    const uint rr=(u-UV-3u)*8u+simd;
                    const bool kd=rr<32u;
                    float res[FRT_ROWS];
                    gemv_bn8_rows<FH>((const device ushort*)(weights+d[kd?D_KDYN:D_VDYN])+(rr%32u)*FH,kd?xk:xv,res,lane);
                    if(lane==0)for(uint r=0;r<R;++r)proj[size_t(r)*RS+(kd?P_KD:P_VD)+rr%32u]=ubf(res[r]);
                }
            }
            }
        }
        SYNC();
        // ---- per head and row: clipped queries and relative states (x tau in global layers); keys and values
        // through their convolution (inputs of earlier rows from their projections) and clip / norm
        for(uint task=sg;task<64u*R;task+=FRT_GROUPS) {
            const uint r=task/64u,t64=task%64u;
            const uint position=p0+r;
            device ushort* proj=RU(r,WS_PROJ);
            float tau=1.0f;
            if(full) {
                const float pf=float(position);
                tau=1.0f+sep(0.1f*metal::precise::log(max(fdiv(pf+1.0f,8192.0f),1.0f)),zero);
            }
            if(t64<24u) {
                const uint h=t64;
                const device ushort* sc=(const device ushort*)(weights+d[D_QCLIP])+h*128u;
                float s[4],sq=0.0f;
                for(uint t=0;t<4;++t)s[t]=bfu(proj[P_Q+h*128u+lane*4u+t])*bfu(sc[lane*4u+t]);
                for(uint t=0;t<4;++t)sq=sep(s[t]*s[t],zero)+sq;
                sq=simd_sum(sq);
                const float k=min(fdiv(22.627416997969522f,metal::precise::sqrt(sq)+1e-06f),1.0f);
                for(uint t=0;t<4;++t) {
                    float v=rb(s[t]*k);
                    if(full)v=rb(v*tau);
                    RU(r,WS_QC)[h*128u+lane*4u+t]=ubf(v);
                }
            } else if(t64<48u) {
                const uint h=t64-24u;
                const float s=lane<16u?bfu(proj[P_R+h*16u+lane])*bfu(((const device ushort*)(weights+d[D_RCLIP]))[h*16u+lane]):0.0f;
                const float sqv=sep(s*s,zero);
                float tot=0.0f;
                for(uint t=0;t<16u;++t)tot=simd_shuffle(sqv,ushort(t))+tot;
                const float k=min(fdiv(8.0f,metal::precise::sqrt(tot)+1e-06f),1.0f);
                if(lane<16u) {
                    float v=rb(s*k);
                    if(full)v=rb(v*tau);
                    RU(r,WS_RCLIP)[h*16u+lane]=ubf(v);
                }
            } else {
                const uint idx=t64-48u,h=idx%8u;
                const bool isk=idx<8u;
                const uint base=(isk?0u:FKV)+h*128u;
                const uint off=(isk?P_K:P_V)+h*128u;
                const device ushort* cur=proj+off;
                float x[4],outv[4];
                for(uint t=0;t<4;++t)x[t]=bfu(cur[lane*4u+t]);
                const device ushort* rows0=RU(0,WS_PROJ)+off;      // the step's rows: stride WSR / 2 ushorts
                if(!full) {
                    const device ushort* cw=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                    for(uint t=0;t<4;++t) {
                        float prev[3];prev_rows(ring,base+lane*4u+t,position,p0,rows0+lane*4u+t,WSR/2u,prev);
                        outv[t]=short_conv(prev,x[t],cw+(h*128u+lane*4u+t)*4u);
                    }
                } else {
                    const device ushort* st=(const device ushort*)(weights+d[isk?D_KCONV:D_VCONV]);
                    const device ushort* dyn=proj+(isk?P_KD:P_VD)+h*4u;
                    for(uint t=0;t<4;++t) {
                        const uint c=h*128u+lane*4u+t;
                        float prev[3];prev_rows(ring,base+lane*4u+t,position,p0,rows0+lane*4u+t,WSR/2u,prev);
                        float sum=0.0f;
                        for(uint lag=0;lag<4;++lag) {
                            const float wgt=bfu(st[lag*FKV+c])+bfu(dyn[lag]);
                            const float xi=lag==0u?x[t]:prev[3u-lag];
                            sum=sep(xi*wgt,zero)+sum;
                        }
                        outv[t]=rb(fdiv(sum+x[t],3.0f));
                    }
                }
                device ushort* frame=vstage+size_t(r)*VSTAGE;
                for(uint t=0;t<4;++t)frame[base+lane*4u+t]=cur[lane*4u+t];
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
                if(full) {
                    device ushort* dst=(isk?kcache:vcache)+(size_t(h)*cap+position)*128u;
                    for(uint t=0;t<4;++t)dst[lane*4u+t]=ubf(y[t]);
                } else {
                    device ushort* dst=frame+FRING+(isk?0u:1024u)+h*128u;
                    for(uint t=0;t<4;++t)dst[lane*4u+t]=ubf(y[t]);
                }
            }
        }
        SYNC();
        // ---- relative-position bias per row [24][1024] (8x8 MMAs)
        {
            const device ushort* rel=(const device ushort*)(weights+d[D_REL]);
            const ushort fm=((lane/4u)&4u)+(lane/2u)%4u;
            const ushort fn=((lane/4u)&2u)*2u+(lane%2u)*2u;
            for(uint tile=sg;tile<384u*R;tile+=FRT_GROUPS) {
                const uint r=tile/384u,ht=(tile%384u)/128u,ct=tile%128u;
                const device ushort* rclip=RU(r,WS_RCLIP);
                device ushort* bias=RU(r,WS_BIAS);
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
            }
        }
        SYNC();
#if FRT_FAST
        // ---- attention, fast: a key depends only on its position (before the step: the cache; of the step: the
        // cache in global layers, the stage in sliding ones), so a task takes (query head, block of positions, half
        // of the rows): queries staged in threadgroup memory, QK with a key per lane (FP32, no cross-lane sums), the
        // block's softmax per row, PV with 4 output dimensions per lane; then a merge per (row, head) with the sink
        // and the output gate
        {
            const float scale=FHEAD_SCALE;
            const uint pl=p0+R-1u;
            const uint lo0=full?0u:(p0>=FWIN-1u?p0-(FWIN-1u):0u);
            const uint range=pl-lo0+1u;
            const uint nsub=(range+128u*64u-1u)/(128u*64u);          // 128-position sub-blocks per task
            const uint KB=128u*nsub;
            const uint nblk=(range+KB-1u)/KB;                         // <= 64 blocks (partials in WS_STATS / WS_PART)
            // the rows' queries of the task's head in threadgroup memory (BF16, [row][128] per SIMD group, after the
            // PV probabilities' 8 KB)
            threadgroup float4* qs=(threadgroup float4*)((threadgroup uchar*)xs+8192u+simd*2048u);
            constexpr uint NRH=(FRT_ROWS+3u)/4u;                    // tasks per (head, block): at most 4 rows each
            constexpr uint AR=(FRT_ROWS+NRH-1u)/NRH;
            for(uint task=sg;task<24u*nblk*NRH;task+=FRT_GROUPS) {
                const uint rh=task%NRH,ht=task/NRH;
                const uint h=ht/nblk,blk=ht%nblk,g=h/3u;
                const uint n0=rh*AR;
                float mx[AR],ls[AR],o[AR][4];
                FRT_UNROLL for(uint n=0;n<AR;++n){mx[n]=FRT_MIN;ls[n]=0.0f;for(uint t=0;t<4;++t)o[n][t]=0.0f;}
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for(uint i=lane;i<AR*32u;i+=32u){const uint2 q2=*((const device uint2*)(RU(min(n0+i/32u,R-1u),WS_QC)+h*128u)+i%32u);
                    qs[i]=float4(lo_f(q2.x),hi_f(q2.x),lo_f(q2.y),hi_f(q2.y));}
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for(uint sb=0;sb<nsub;++sb) {
                    const uint P0=lo0+blk*KB+sb*128u;
                    if(P0>pl)break;
                    const device ushort* kp[4];
                    FRT_UNROLL for(uint j=0;j<4u;++j) {
                        const uint P=min(P0+lane+32u*j,pl);
                        kp[j]=(!full && P>=p0)?vstage+size_t(P-p0)*VSTAGE+FRING+g*128u
                                              :kcache+(size_t(g)*cap+(full?P:(P&(FWIN-1u))))*128u;
                    }
                    float sc[AR][4];
                    FRT_UNROLL for(uint n=0;n<AR;++n)for(uint j=0;j<4u;++j)sc[n][j]=0.0f;
                    for(uint d0=0;d0<128u;d0+=8u) {
                        float kf[4][8];
                        FRT_UNROLL for(uint j=0;j<4u;++j) {
                            const uint4 kr=*((const device uint4*)(kp[j]+d0));
                            kf[j][0]=lo_f(kr.x);kf[j][1]=hi_f(kr.x);kf[j][2]=lo_f(kr.y);kf[j][3]=hi_f(kr.y);
                            kf[j][4]=lo_f(kr.z);kf[j][5]=hi_f(kr.z);kf[j][6]=lo_f(kr.w);kf[j][7]=hi_f(kr.w);
                        }
                        FRT_UNROLL for(uint n=0;n<AR;++n) {
                            const float4 qa=qs[n*32u+d0/4u],qb=qs[n*32u+d0/4u+1u];
                            const float qf[8]={qa.x,qa.y,qa.z,qa.w,qb.x,qb.y,qb.z,qb.w};
                            FRT_UNROLL for(uint j=0;j<4u;++j)
                                FRT_UNROLL for(uint t=0;t<8u;++t)sc[n][j]=metal::fma(qf[t],kf[j][t],sc[n][j]);
                        }
                    }
                    // scale, bias, window; online softmax over the sub-block
                    FRT_UNROLL for(uint n=0;n<AR;++n) {
                        const uint pos=p0+min(n0+n,R-1u);
                        const uint lo=full?0u:(pos>=FWIN-1u?pos-(FWIN-1u):0u);
                        const device ushort* bias=RU(min(n0+n,R-1u),WS_BIAS)+h*FWIN;
                        float bm=FRT_MIN;
                        bool any=false;
                        FRT_UNROLL for(uint j=0;j<4u;++j) {
                            const uint P=P0+lane+32u*j;
                            const bool ok=P<=pos && P>=lo;
                            const uint dist=pos-P;
                            const float sv=sc[n][j]*scale+((ok && dist<FWIN)?bfu(bias[dist]):0.0f);
                            sc[n][j]=ok?sv:FRT_MIN;
                            if(ok){bm=max(bm,sv);any=true;}
                        }
                        bm=simd_max(bm);
                        const bool rany=simd_any(any);
                        const float nm=max(mx[n],bm);
                        const float f=rany?metal::fast::exp(mx[n]-nm):1.0f;
                        float lsum=0.0f;
                        FRT_UNROLL for(uint j=0;j<4u;++j) {
                            const uint P=P0+lane+32u*j;
                            const bool ok=P<=pos && P>=lo;
                            const float pv=ok?metal::fast::exp(sc[n][j]-nm):0.0f;
                            sc[n][j]=pv;lsum+=pv;
                        }
                        lsum=simd_sum(lsum);
                        if(rany){ls[n]=ls[n]*f+lsum;for(uint t=0;t<4;++t)o[n][t]*=f;mx[n]=nm;}
                    }
                    // PV: dimensions 4 lane .. 4 lane + 3; the probabilities of 32 keys through threadgroup memory
                    // (broadcast reads of 4 keys: cheaper than a shuffle per key and row)
                    threadgroup float* pt=(threadgroup float*)xs+simd*(8u*32u);
                    FRT_UNROLL for(uint j=0;j<4u;++j) {
                        if(P0+32u*j>pl)break;
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        FRT_UNROLL for(uint n=0;n<AR;++n)pt[n*32u+lane]=sc[n][j];
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        for(uint k4=0;k4<32u;k4+=4u) {
                            if(P0+32u*j+k4>pl)break;
                            uint2 vv[4];
                            FRT_UNROLL for(uint i=0;i<4u;++i) {
                                const uint P=min(P0+32u*j+k4+i,pl);
                                const device ushort* vp=(!full && P>=p0)?vstage+size_t(P-p0)*VSTAGE+FRING+1024u+g*128u
                                                                        :vcache+(size_t(g)*cap+(full?P:(P&(FWIN-1u))))*128u;
                                vv[i]=*((const device uint2*)(vp+lane*4u));
                            }
                            FRT_UNROLL for(uint n=0;n<AR;++n) {
                                const float4 pv=*((const threadgroup float4*)(pt+n*32u+k4));
                                FRT_UNROLL for(uint i=0;i<4u;++i) {
                                    const float pi=pv[i];
                                    o[n][0]=metal::fma(pi,lo_f(vv[i].x),o[n][0]);o[n][1]=metal::fma(pi,hi_f(vv[i].x),o[n][1]);
                                    o[n][2]=metal::fma(pi,lo_f(vv[i].y),o[n][2]);o[n][3]=metal::fma(pi,hi_f(vv[i].y),o[n][3]);
                                }
                            }
                        }
                    }
                }
                FRT_UNROLL for(uint n=0;n<AR;++n) {
                    if(n0+n>=R)break;
                    device float* stats=RF(n0+n,WS_STATS);
                    device ushort* part=RU(n0+n,WS_PART);
                    if(lane==0){stats[(h*64u+blk)*2u]=mx[n];stats[(h*64u+blk)*2u+1u]=ls[n];}
                    for(uint t=0;t<4;++t)part[(h*64u+blk)*128u+lane*4u+t]=ubf(o[n][t]);
                }
            }
            SYNC();
            // merge per (row, head): the sink (value 0) and the blocks; output gate
            for(uint mt=sg;mt<24u*R;mt+=FRT_GROUPS) {
                const uint n=mt/24u,h=mt%24u;
                const device float* stats=RF(n,WS_STATS);
                const device ushort* part=RU(n,WS_PART);
                const float sink=bfu((RU(n,WS_PROJ)+P_S)[h]);
                float M=sink;
                for(uint b=lane;b<nblk;b+=32u)if(stats[(h*64u+b)*2u+1u]>0.0f)M=max(M,stats[(h*64u+b)*2u]);
                M=simd_max(M);
                float lt=0.0f;
                for(uint b=lane;b<nblk;b+=32u){const float lb=stats[(h*64u+b)*2u+1u];
                    if(lb>0.0f)lt+=metal::fast::exp(stats[(h*64u+b)*2u]-M)*lb;}
                lt=simd_sum(lt)+metal::fast::exp(sink-M);
                float ov[4]={0.0f,0.0f,0.0f,0.0f};
                for(uint b=0;b<nblk;++b) {
                    const float lb=stats[(h*64u+b)*2u+1u];
                    if(!(lb>0.0f))continue;
                    const float fb=metal::fast::exp(stats[(h*64u+b)*2u]-M);
                    const uint2 pr=*((const device uint2*)(part+(h*64u+b)*128u+lane*4u));
                    ov[0]=metal::fma(fb,lo_f(pr.x),ov[0]);ov[1]=metal::fma(fb,hi_f(pr.x),ov[1]);
                    ov[2]=metal::fma(fb,lo_f(pr.y),ov[2]);ov[3]=metal::fma(fb,hi_f(pr.y),ov[3]);
                }
                const device ushort* gate=RU(n,WS_PROJ)+P_G;
                const float inv=1.0f/lt;
                for(uint t=0;t<4;++t) {
                    const uint dim=lane*4u+t;
                    RU(n,WS_A)[h*128u+dim]=ubf(rb(ov[t]*inv)*sigmoid_f32(bfu(gate[h*128u+dim])));
                }
            }
        }
#else
        // ---- attention of every row: the one-token step's partitions at the row's position. A key of the step
        // (position >= p) comes from this step's rows (global: the cache, written above; sliding: the stage).
        {
            const float scale=FHEAD_SCALE;
            bool two=false;
            for(uint r=0;r<R;++r){const uint pos=p0+r;const uint count=full?pos+1u:min(pos+1u,FWIN);two=two||count>=1024u;}
            // single pass rows: unit (row, head) on a threadgroup; 32 key-strided partitions, 4 per SIMD group
            for(uint u=group;u<24u*R;u+=FRT_WORKERS) {
                const uint r=u/24u,h=u%24u,g=h/3u;
                const uint position=p0+r;
                const uint count=full?position+1u:min(position+1u,FWIN);
                if(count>=1024u)continue;
                const device ushort* qc=RU(r,WS_QC);
                const device ushort* bias=RU(r,WS_BIAS);
                const device ushort* sinks=RU(r,WS_PROJ)+P_S;
                const device ushort* gate=RU(r,WS_PROJ)+P_G;
                float q[4];
                for(uint t=0;t<4;++t)q[t]=scale*bfu(qc[h*128u+lane*4u+t]);
                // the SIMD group's 4 partitions (keys simd + 8 rr + 32 k) advance together: 4 independent chains,
                // each over its keys in order
                float o[4][4],mx[4],z[4];
                for(uint rr=0;rr<4;++rr) {
                    const uint vs=simd+rr*8u;
                    for(uint t=0;t<4;++t)o[rr][t]=0.0f;
                    mx[rr]=FRT_MIN;z[rr]=0.0f;
                    if(vs==0u){mx[rr]=bfu(sinks[h]);z[rr]=1.0f;}
                }
                for(uint i0=simd;i0<count;i0+=32u) {
                    for(uint rr=0;rr<4;++rr) {
                        const uint i=i0+rr*8u;
                        if(i>=count)break;
                        // below a full ring the slot of a position is the position
                        const device ushort* kp;const device ushort* vp;
                        if(!full && i>=p0){const device ushort* f=vstage+size_t(i-p0)*VSTAGE+FRING;
                            kp=f+g*128u;vp=f+1024u+g*128u;}
                        else {kp=kcache+(size_t(g)*cap+i)*128u;vp=vcache+(size_t(g)*cap+i)*128u;}
                        const uint2 kr=*((const device uint2*)(kp+lane*4u));
                        const float k[4]={lo_f(kr.x),hi_f(kr.x),lo_f(kr.y),hi_f(kr.y)};
                        float score=0.0f;
                        for(uint t=0;t<4;++t)score+=q[t]*k[t];
                        score=simd_sum(score);
                        const uint dist=position-i;
                        score+=bfu(bias[h*FWIN+dist]);
                        const float nm=max(mx[rr],score);
                        const float factor=metal::fast::exp(mx[rr]-nm);
                        const float e=metal::fast::exp(score-nm);
                        mx[rr]=nm;
                        z[rr]=z[rr]*factor+e;
                        const uint2 vv=*((const device uint2*)(vp+lane*4u));
                        const float v[4]={lo_f(vv.x),hi_f(vv.x),lo_f(vv.y),hi_f(vv.y)};
                        for(uint t=0;t<4;++t)o[rr][t]=metal::fma(o[rr][t],factor,e*v[t]);
                    }
                }
                for(uint rr=0;rr<4;++rr)if(lane==0){red[simd+rr*8u]=mx[rr];red[32u+simd+rr*8u]=z[rr];}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float my_m=red[lane];
                const float nm=simd_max(my_m);
                const float factor=metal::fast::exp(my_m-nm);
                const float zs=simd_sum(red[32u+lane]*factor);
                for(uint half_=0;half_<2u;++half_) {
                    if((lane/16u)==half_)
                        for(uint rr=0;rr<4;++rr)for(uint t=0;t<4;++t)
                            tv[(simd+rr*8u)*64u+(lane%16u)*4u+t]=o[rr][t];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for(uint dd=simd;dd<64u;dd+=8u) {
                        float ov=simd_sum(tv[lane*64u+dd]*factor);
                        ov=zs==0.0f?ov:fdiv(ov,zs);
                        if(lane==0) {
                            const uint dim=half_*64u+dd;
                            const float val=rb(ov);
                            RU(r,WS_A)[h*128u+dim]=ubf(val*sigmoid_f32(bfu(gate[h*128u+dim])));
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if(two) {
                // two-pass rows: task (row pair, KV head, partition b of the 64 slot-strided ones) for its 3 query
                // heads (6 chains, each over its keys in slot order); a key slot's K / V is loaded once for both rows
                // when they read the same key (a row reads the cache, or this step's staged key in a full ring)
                constexpr uint RP=(FRT_ROWS+1u)/2u;
                for(uint task=sg;task<512u*RP;task+=FRT_GROUPS) {
                    const uint rp=task/512u,g=(task%512u)/64u,b=task%64u;
                    uint pos[2],cnt[2],csl[2];
                    bool on[2];
                    for(uint j=0;j<2u;++j) {
                        const uint r=rp*2u+j;
                        pos[j]=p0+r;
                        cnt[j]=full?pos[j]+1u:min(pos[j]+1u,FWIN);
                        on[j]=r<R && cnt[j]>=1024u;
                        csl[j]=pos[j]&(FWIN-1u);
                    }
                    if(!on[0] && !on[1])continue;
                    float q[2][3][4],o[2][3][4],mx[2][3],z[2][3];
                    for(uint j=0;j<2u;++j) {
                        const uint r=min(rp*2u+j,R-1u);
                        const device ushort* qc=RU(r,WS_QC);
                        const device ushort* sinks=RU(r,WS_PROJ)+P_S;
                        for(uint hh=0;hh<3u;++hh) {
                            const uint h=g*3u+hh;
                            for(uint t=0;t<4;++t){q[j][hh][t]=scale*bfu(qc[h*128u+lane*4u+t]);o[j][hh][t]=0.0f;}
                            mx[j][hh]=FRT_MIN;z[j][hh]=0.0f;
                            if(b==0u){mx[j][hh]=bfu(sinks[h]);z[j][hh]=1.0f;}
                        }
                    }
                    const uint top=max(on[0]?cnt[0]:0u,on[1]?cnt[1]:0u);
                    // the bias of a row's keys within the window (dist < 1024: at most 16, the task's last keys): lane l
                    // holds key kb[j] + l of the task (kb: all but the last 32 keys)
                    float bb[2][3];
                    uint kb[2];
                    for(uint j=0;j<2u;++j) {
                        const uint r=min(rp*2u+j,R-1u);
                        const device ushort* bias=RU(r,WS_BIAS);
                        const uint nk=cnt[j]>b?(cnt[j]-b+63u)/64u:0u;
                        kb[j]=nk>32u?nk-32u:0u;
                        const uint i=b+64u*(kb[j]+lane);
                        const uint kpl=full?i:(((i+FWIN-csl[j]-1u)&(FWIN-1u))+pos[j]+1u-FWIN);
                        const uint dist=pos[j]-kpl;
                        const bool in=on[j] && i<cnt[j] && dist<FWIN;
                        for(uint hh=0;hh<3u;++hh)bb[j][hh]=in?bfu(bias[(g*3u+hh)*FWIN+dist]):0.0f;
                    }
                    uint kk=0;
                    for(uint i=b;i<top;i+=64u,++kk) {
                        uint kp_[2];
                        const device ushort* kp[2];
                        for(uint j=0;j<2u;++j) {
                            kp_[j]=full?i:(((i+FWIN-csl[j]-1u)&(FWIN-1u))+pos[j]+1u-FWIN);
                            if(!full && kp_[j]>=p0)kp[j]=vstage+size_t(kp_[j]-p0)*VSTAGE+FRING+g*128u;
                            else kp[j]=kcache+(size_t(g)*cap+i)*128u;
                        }
                        const bool same=kp[0]==kp[1];
                        uint2 kr[2],vr2[2];
                        for(uint j=0;j<2u;++j) {
                            if(j==1u && same){kr[1]=kr[0];vr2[1]=vr2[0];break;}
                            const device ushort* vp=(!full && kp_[j]>=p0)?kp[j]+1024u:kp[j]+size_t(8u)*cap*128u;
                            kr[j]=*((const device uint2*)(kp[j]+lane*4u));
                            vr2[j]=*((const device uint2*)(vp+lane*4u));
                        }
                        for(uint j=0;j<2u;++j) {
                            if(!on[j] || i>=cnt[j])continue;
                            const float k[4]={lo_f(kr[j].x),hi_f(kr[j].x),lo_f(kr[j].y),hi_f(kr[j].y)};
                            const float v[4]={lo_f(vr2[j].x),hi_f(vr2[j].x),lo_f(vr2[j].y),hi_f(vr2[j].y)};
                            const uint dist=pos[j]-kp_[j];
                            const float madd=dist<FWIN?1.0f:0.0f;
                            for(uint hh=0;hh<3u;++hh) {
                                const uint h=g*3u+hh;
                                float score=0.0f;
                                for(uint t=0;t<4;++t)score+=q[j][hh][t]*k[t];
                                score=simd_sum(score);
                                const float bv=simd_shuffle(bb[j][hh],ushort(min(kk-min(kk,kb[j]),31u)));
                                score+=madd>0.0f?bv:0.0f;
                                const float nm=max(mx[j][hh],score);
                                const float factor=metal::fast::exp(mx[j][hh]-nm);
                                const float e=metal::fast::exp(score-nm);
                                mx[j][hh]=nm;
                                z[j][hh]=z[j][hh]*factor+e;
                                for(uint t=0;t<4;++t)o[j][hh][t]=metal::fma(o[j][hh][t],factor,e*v[t]);
                            }
                        }
                    }
                    for(uint j=0;j<2u;++j) {
                        if(!on[j])continue;
                        const uint r=rp*2u+j;
                        device float* stats=RF(r,WS_STATS);
                        device ushort* part=RU(r,WS_PART);
                        for(uint hh=0;hh<3u;++hh) {
                            const uint h=g*3u+hh;
                            if(lane==0){stats[(h*64u+b)*2u]=mx[j][hh];stats[(h*64u+b)*2u+1u]=z[j][hh];}
                            for(uint t=0;t<4;++t)part[(h*64u+b)*128u+lane*4u+t]=ubf(o[j][hh][t]);
                        }
                    }
                }
                SYNC();
                for(uint mt=sg;mt<768u*R;mt+=FRT_GROUPS) {
                    const uint r=mt/768u,h=(mt%768u)/32u;
                    const uint position=p0+r;
                    const uint count=full?position+1u:min(position+1u,FWIN);
                    if(count<1024u)continue;
                    const device float* stats=RF(r,WS_STATS);
                    const device ushort* part=RU(r,WS_PART);
                    const device ushort* gate=RU(r,WS_PROJ)+P_G;
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
                    const uint dd=(mt%32u)*4u;
                    float ov[4];
                    for(uint t=0;t<4;++t) {
                        float a=0.0f;
                        a+=f0*bfu(part[(h*64u+lane)*128u+dd+t]);
                        a+=f1*bfu(part[(h*64u+lane+32u)*128u+dd+t]);
                        ov[t]=simd_sum(a);
                    }
                    if(lane==0)for(uint t=0;t<4;++t) {
                        const float val=rb(zs==0.0f?ov[t]:fdiv(ov[t],zs));
                        RU(r,WS_A)[h*128u+dd+t]=ubf(val*sigmoid_f32(bfu(gate[h*128u+dd+t])));
                    }
                }
            }
        }
#endif
        SYNC();
        // ---- output projection
        {
            const device ushort* wo=(const device ushort*)(weights+d[D_WO]);
#if FRT_ROWS>8u && (FRT_FAST || FRT_Q8)
            // more than 8 rows: BF16 weights by MMAs (36 units of 64 rows; Q8_0 too: its multi-row product is ALU-bound)
            for(uint u=group;u<2304u/64u;u+=FRT_WORKERS)
                gemv_rows_mma<FQ,false,1u>(wo,u*64u,(const device ushort*)RU(0,WS_A),RU(0,WS_O),logits,xs,tid,simd,lane);
#elif FRT_Q8
            // attention outputs |a| <= sqrt(128) (values RMS-normalized per head, the gate a sigmoid): scale 1
            for(uint u=group;u<36u;u+=FRT_WORKERS)
                q8_rows_tg<FQ,false>(q8,q8t+layer*Q8_LAYER+Q8_WO,u*64u,(const device ushort*)RU(0,WS_A),(const device float*)nullptr,0u,
                                     RU(0,WS_O),logits,(threadgroup half*)xs,tid,simd,lane);
#else
            for(uint u=group;u<2304u/(8u*PTM);u+=FRT_WORKERS)
                gemv_rows_tg<FQ,false>(wo,u*8u*PTM+simd*PTM,(const device ushort*)RU(0,WS_A),RU(0,WS_O),logits,xs,tid,lane);
#endif
        }
        SYNC();
        // ---- post-attention conv (inputs of earlier rows from their output projections); residual; MLP input
        if(boundary){slots[nblocks]=bslot;++nblocks;}
        if(group<R) {
            const uint r=group,position=p0+r;
            const device ushort* vh=RU(r,E_H);
            device float* vr=RF(r,E_VR);
            const device ushort* cw=(const device ushort*)(weights+d[D_ACONV]);
            const device ushort* oproj=RU(r,WS_O);
            device ushort* frame=vstage+size_t(r)*VSTAGE;
            for(uint i=tid;i<FH;i+=256u) {
                float prev[3];prev_rows(ring,2048u+i,position,p0,RU(0,WS_O)+i,WSR/2u,prev);
                const float c=short_conv(prev,bfu(oproj[i]),cw+i*4u);
                vr[i]=boundary?c:bfu(vh[i])+c;
                frame[2048u+i]=oproj[i];
            }
            threadgroup_barrier(mem_flags::mem_device);
            const float inv=rms_inv((const device float*)vr,red,lane,simd);
            depth_attention<1>(RF(r,WS_BS),slots,nblocks,(const device float*)vr,inv,
                               (const device float*)(weights+d[D_MLP_Q]),RF(r,E_MI),red,tid,lane,zero);
            if(messages) {
                const device float* mi=RF(r,E_MI);
                const device float* m=RF(r,WS_MSG)+(li*4u+3u)*FH;
                threadgroup_barrier(mem_flags::mem_device);
                for(uint i=tid;i<FH;i+=256u)RF(r,WS_AIN)[i]=mi[i];
                const float ia=rms_inv(mi,red,lane,simd);
                const float im=rms_inv(m,red,lane,simd);
                device ushort* s=RU(r,E_S)+3u*4608u;
                for(uint i=tid;i<FH;i+=256u){s[i]=ubf(mi[i]*ia);s[FH+i]=ubf(m[i]*im);}
            }
        }
        if(messages) {
            SYNC();
            for(uint u=sg;u<64u;u+=FRT_GROUPS) {
                const device ushort* wd=(const device ushort*)(weights+d[D_MG+9u])+u*4608u;
                float res[FRT_ROWS];
                gemv_bn8_rows<4608u>(wd,(const device ushort*)(RU(0,E_S)+3u*4608u),res,lane);
                if(lane==0)for(uint r=0;r<R;++r)RF(r,WS_MDOWN)[192u+u]=rb(res[r]);
            }
            SYNC();
            for(uint task=sg;task<144u;task+=FRT_GROUPS) {
                const uint which=task/72u,row0=(task%72u)*32u;
                const device ushort* w=(const device ushort*)(weights+d[D_MG+10u+which]);
                gemv_k64_rows(w,row0,RF(0,WS_MDOWN)+192u,which==0u?RF(0,WS_FORGET):RF(0,WS_GAIN),lane);
            }
            SYNC();
            for(uint i=index;i<R*FH;i+=FRT_THREADS) {
                const uint r=i/FH,e=i%FH;
                RF(r,WS_GATED)[e]=gate_combine(RF(r,WS_FORGET)[e],RF(r,WS_GAIN)[e],RF(r,WS_AIN)[e],
                                               RF(r,WS_MSG)[(li*4u+3u)*FH+e],zero);
            }
            SYNC();
        }
        // (without message gates the MLP input of row r was written by threadgroup r itself)
        // ---- MLP: post-attention norm -> router (FP32 logits, one expert row per SIMD group, rows inside)
        if(group<R) {
            const uint r=group;
            device ushort* y=RU(r,E_Y);
            const device ushort* w=(const device ushort*)(weights+d[D_POST_NORM]);
            const device float* src=messages?RF(r,WS_GATED):RF(r,E_MI);
            const float inv=rms_inv(src,red,lane,simd);
            float ym=0.0f;
            for(uint i=tid;i<FH;i+=256u){const ushort yv=ubf(bfu(w[i])*(src[i]*inv));y[i]=yv;ym=max(ym,metal::abs(bfu(yv)));}
#if FRT_FAST
            ym=simd_max(ym);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane==0)red[simd]=ym;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0) {
                float m=red[0];for(uint i=1;i<8u;++i)m=max(m,red[i]);
                const int ex=m>0.0f?int((as_type<uint>(m)>>23)&255u)-127:0;
                // NVFP4 codes |q| <= 6: max * 2^k in [128, 256); MXFP8 codes |q| <= 448: in [4, 8)
                const int k=clamp((mxfp8?2:7)-ex,-100,100);
                *((device float*)(RW(r)+E_XSC))=as_type<float>(uint(127+k)<<23);
            }
#endif
        }
        SYNC();
        if(sg<128u) {
            const device ushort* wrow=(const device ushort*)(weights+d[D_ROUTER])+sg*FH;
            float res[FRT_ROWS];
            gemv_bn8_rows<FH>(wrow,(const device ushort*)RU(0,E_Y),res,lane);
            if(lane==0)for(uint r=0;r<R;++r)RF(r,WS_RLOG)[sg]=res[r];
        }
        SYNC();
        // ---- routing of every row (simd 0 of threadgroup r), then the distinct experts of the step
        if(group==0)for(uint r=simd;r<R;r+=8u) {
            const device float* rlog=RF(r,WS_RLOG);
            const device float* rbias=(const device float*)(weights+d[D_RBIAS]);
            device uint* rid=(device uint*)(RW(r)+E_ROUTE);
            device float* rw=(device float*)(RW(r)+E_ROUTE+32u);
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
                if(lane==0)rid[k]=winner;
            }
            float total=0.0f;
            for(uint k=0;k<8;++k)total=wk[k]+total;
            if(lane==0)for(uint k=0;k<8;++k)rw[k]=fdiv(wk[k],total+1e-20f)*8.0f;
        }
        if(group==0)threadgroup_barrier(mem_flags::mem_device);
        if(group==0 && simd==0) {
            // lane l: experts l, l + 32, l + 64, l + 96; the rows routed to each (a bit per row) and its rank in
            // them (4 bits per row); compacted in expert order
            uint m[4]={0u,0u,0u,0u};
            ulong rk[4]={0ul,0ul,0ul,0ul};
            for(uint r=0;r<R;++r) {
                const device uint* rid=(const device uint*)(RW(r)+E_ROUTE);
                for(uint k=0;k<8;++k){const uint e=rid[k];if(e%32u==lane){m[e/32u]|=1u<<r;rk[e/32u]|=ulong(k)<<(4u*r);}}
            }
            // most slots first (the expert passes take units dynamically: heavy units start early)
            uint base=0;
            for(uint c=FRT_ROWS;c>=1u;--c)for(uint i=0;i<4;++i) {
                const uint has=popcount(m[i])==c?1u:0u;
                const uint before=simd_prefix_exclusive_sum(has);
                if(has){elist[base+before]=lane+32u*i;emask[base+before]=rk[i];erow[base+before]=m[i];}
                base+=simd_sum(has);
            }
            if(lane==0)ecount[0]=base;
#if FRT_FAST
            for(uint i=lane;i<128u;i+=32u)atomic_store_explicit(control+FRT_DONE+i,0u,memory_order_relaxed);
            // heavy and light experts alternately (ALU-bound and bandwidth-bound units side by side)
            simdgroup_barrier(mem_flags::mem_device);
            uint ze[4],zr[4];ulong zm[4];
            for(uint t=0;t<4u;++t){const uint i=lane+32u*t;if(i<base){const uint src=(i%2u==0u)?i/2u:base-1u-i/2u;
                ze[t]=elist[src];zm[t]=emask[src];zr[t]=erow[src];}}
            simdgroup_barrier(mem_flags::mem_device);
            for(uint t=0;t<4u;++t){const uint i=lane+32u*t;if(i<base){elist[i]=ze[t];emask[i]=zm[t];erow[i]=zr[t];}}
#endif
        }
        SYNC();
        // ---- experts: threadgroup units (distinct expert, 32 rows) for all of the expert's (row, rank) slots (one per
        // routed row), inputs staged; gate/up rows with the activation of their two channels, then down rows
#if FRT_FAST
        // fast: units (distinct expert, FH / EUN rows) of the gate/up rows (EUN per expert, heavy experts first), then
        // of the down rows (EUD per expert, FH / EUD rows) from the same counter; a down unit waits for its expert's EUN gate/up units
        // (a counter per distinct expert), so the passes overlap without a grid barrier. The expert's slots in groups
        // whose whole inputs fit XS_CAP.
        {
            const uint n=ecount[0];
            const device uchar* w13=weights+d[D_W13];
            const device uchar* s13=weights+d[D_S13];
            const device float* gs13=(const device float*)(weights+d[D_GS13]);
            const device uchar* w2=weights+d[D_W2];
            const device uchar* s2=weights+d[D_S2];
            const device float* gs2=(const device float*)(weights+d[D_GS2]);
            device ushort* wsu=(device ushort*)workspace;
            device atomic_uint* wctr=control+64u+visit*2u;
            device atomic_uint* done=control+FRT_DONE;
            for(;;) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(tid==0)slot_n[0]=atomic_fetch_add_explicit(wctr,1u,memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint u=slot_n[0];
                if(u>=n*(EUN+EUD))break;
                const bool down=u>=n*EUN;
                const uint ud=down?u-n*EUN:u,eu=down?EUD:EUN;
                const uint ei=ud/eu,rbase=(ud%eu)*(FH/eu);
                const uint e=elist[ei];
                uint sx[FRT_ROWS],so[FRT_ROWS],sa[FRT_ROWS],ns=0;
                float xq[FRT_ROWS];
                uint rows=erow[ei];
                const ulong rk=emask[ei];
                for(uint i=0;i<FRT_ROWS;++i) {
                    if(rows==0u)break;
                    const uint r=ctz(rows);rows&=rows-1u;
                    const uint k=uint(rk>>(4u*r))&15u;
                    so[i]=(r*WSR+WS_EDOWN)/2u+k*FH;sa[i]=(r*WSR+WS_ACT)/2u+k*FI;ns=i+1u;
                    // down: softcapped SwiGLU outputs |act| <= 100: x 2 (NVFP4), x 2^-4 (MXFP8)
                    sx[i]=down?sa[i]:(r*WSR+E_Y)/2u;
                    xq[i]=down?(mxfp8?0.0625f:2.0f):*((const device float*)(RW(r)+E_XSC));
                }
                if(!down) {
                    constexpr uint G=XS_CAP/FH;
                    for(uint g0=0;g0<ns;g0+=G) {
                        const uint gn=min(G,ns-g0);
                        if(!mxfp8)
                            expert_fast<FH,false>(w13+size_t(e)*FH*(FH/2u),s13+size_t(e)*FH*(FH/16u),fdiv(gs13[e],2688.0f),
                                                  rbase,FH/EUN/64u,wsu,sx+g0,wsu,so+g0,sa+g0,xq+g0,gn,true,(threadgroup half2*)xs,sct,tid,simd,lane);
                        else
                            expert_fast<FH,true>(w13+size_t(e)*FH*FH,s13+size_t(e)*FH*(FH/32u),1.0f,rbase,FH/EUN/64u,wsu,sx+g0,wsu,
                                                 so+g0,sa+g0,xq+g0,gn,true,(threadgroup half2*)xs,sct,tid,simd,lane);
                    }
                    // publish the unit's activations, then count it
                    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
                    threadgroup_barrier(mem_flags::mem_device);
                    if(tid==0)atomic_fetch_add_explicit(done+ei,1u,memory_order_relaxed);
                } else {
                    if(tid==0) {
                        uint spins=0;
                        while(atomic_load_explicit(done+ei,memory_order_relaxed)<EUN &&
                              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<FRT_SPIN_LIMIT)++spins;
                        if(atomic_load_explicit(done+ei,memory_order_relaxed)<EUN)
                            atomic_store_explicit(control+2,phase+1u,memory_order_relaxed);
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
                    constexpr uint G=XS_CAP/FI;
                    for(uint g0=0;g0<ns;g0+=G) {
                        const uint gn=min(G,ns-g0);
                        if(!mxfp8)
                            expert_fast<FI,false>(w2+size_t(e)*FH*(FI/2u),s2+size_t(e)*FH*(FI/16u),fdiv(gs2[e],2688.0f),
                                                  rbase,FH/EUD/128u,wsu,sx+g0,wsu,so+g0,so+g0,xq+g0,gn,false,(threadgroup half2*)xs,sct,tid,simd,lane);
                        else
                            expert_fast<FI,true>(w2+size_t(e)*FH*FI,s2+size_t(e)*FH*(FI/32u),1.0f,rbase,FH/EUD/128u,wsu,sx+g0,wsu,so+g0,
                                                 so+g0,xq+g0,gn,false,(threadgroup half2*)xs,sct,tid,simd,lane);
                    }
                }
            }
        }
        SYNC();
#else
        // unit = (distinct expert, EUT tiles of 32 rows); the expert's slots in groups whose whole inputs fit XS_CAP
        {
            const uint n=ecount[0];
            const device uchar* w13=weights+d[D_W13];
            const device uchar* s13=weights+d[D_S13];
            const device float* gs13=(const device float*)(weights+d[D_GS13]);
            device ushort* wsu=(device ushort*)workspace;
            constexpr uint UPE=FH/(32u*EUT);                 // units per expert
            device atomic_uint* wctr=control+64u+visit*2u+0u;   // units taken dynamically
            for(;;) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(tid==0)slot_n[0]=atomic_fetch_add_explicit(wctr,1u,memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint u=slot_n[0];
                if(u>=n*UPE)break;
                const uint ei=u/UPE,rbase=(u%UPE)*32u*EUT+simd*4u;
                const uint e=elist[ei];
                uint sx[FRT_ROWS],so[FRT_ROWS],sa[FRT_ROWS],ns=0;
                uint rows=erow[ei];
                const ulong rk=emask[ei];
                for(uint i=0;i<FRT_ROWS;++i) {
                    if(rows==0u)break;
                    const uint r=ctz(rows);rows&=rows-1u;
                    const uint k=uint(rk>>(4u*r))&15u;
                    sx[i]=(r*WSR+E_Y)/2u;so[i]=(r*WSR+WS_EDOWN)/2u+k*FH;sa[i]=(r*WSR+WS_ACT)/2u+k*FI;ns=i+1u;
                }
                constexpr uint G=XS_CAP/FH;                  // slots per staging group
                for(uint g0=0;g0<ns;g0+=G) {
                    const uint gn=min(G,ns-g0);
                    if(!mxfp8)
                        expert_whole<FH,false>(w13+size_t(e)*FH*(FH/2u),s13+size_t(e)*FH*(FH/16u),fdiv(gs13[e],2688.0f),
                                               rbase,EUT,wsu,sx+g0,wsu,so+g0,gn,xs,tid,lane);
                    else
                        expert_whole<FH,true>(w13+size_t(e)*FH*FH,s13+size_t(e)*FH*(FH/32u),1.0f,rbase,EUT,wsu,sx+g0,wsu,
                                              so+g0,gn,xs,tid,lane);
                }
                simdgroup_barrier(mem_flags::mem_device);
                if(lane<2u*EUT)for(uint i=0;i<FRT_ROWS;++i) {
                    if(i>=ns)break;
                    const uint rr=rbase+(lane/2u)*32u,l2=lane%2u;
                    const float gg=bfu(wsu[so[i]+rr+l2*2u]),uu=bfu(wsu[so[i]+rr+l2*2u+1u]);
                    wsu[sa[i]+rr/2u+l2]=ubf(softcapped_swiglu(gg,uu));
                }
            }
        }
        SYNC();
        {
            const uint n=ecount[0];
            const device uchar* w2=weights+d[D_W2];
            const device uchar* s2=weights+d[D_S2];
            const device float* gs2=(const device float*)(weights+d[D_GS2]);
            device ushort* wsu=(device ushort*)workspace;
            constexpr uint UPE=FH/(32u*EUT);
            device atomic_uint* wctr=control+64u+visit*2u+1u;   // units taken dynamically
            for(;;) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(tid==0)slot_n[0]=atomic_fetch_add_explicit(wctr,1u,memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint u=slot_n[0];
                if(u>=n*UPE)break;
                const uint ei=u/UPE,rbase=(u%UPE)*32u*EUT+simd*4u;
                const uint e=elist[ei];
                uint sx[FRT_ROWS],so[FRT_ROWS],ns=0;
                uint rows=erow[ei];
                const ulong rk=emask[ei];
                for(uint i=0;i<FRT_ROWS;++i) {
                    if(rows==0u)break;
                    const uint r=ctz(rows);rows&=rows-1u;
                    const uint k=uint(rk>>(4u*r))&15u;
                    sx[i]=(r*WSR+WS_ACT)/2u+k*FI;so[i]=(r*WSR+WS_EDOWN)/2u+k*FH;ns=i+1u;
                }
                constexpr uint G=XS_CAP/FI;
                for(uint g0=0;g0<ns;g0+=G) {
                    const uint gn=min(G,ns-g0);
                    if(!mxfp8)
                        expert_whole<FI,false>(w2+size_t(e)*FH*(FI/2u),s2+size_t(e)*FH*(FI/16u),fdiv(gs2[e],2688.0f),
                                               rbase,EUT,wsu,sx+g0,wsu,so+g0,gn,xs,tid,lane);
                    else
                        expert_whole<FI,true>(w2+size_t(e)*FH*FI,s2+size_t(e)*FH*(FI/32u),1.0f,rbase,EUT,wsu,sx+g0,wsu,
                                              so+g0,gn,xs,tid,lane);
                }
            }
        }
        SYNC();
#endif
        // ---- combination of every row (products rounded, summed in rank order) -> MLP conv input
        for(uint i=index;i<R*FH;i+=FRT_THREADS) {
            const uint r=i/FH,e=i%FH;
            const device float* rw=(const device float*)(RW(r)+E_ROUTE+32u);
            float sum=0.0f;
            for(uint k=0;k<8;++k)sum=sep(bfu(RU(r,WS_EDOWN)[k*FH+e])*rw[k],zero)+sum;
            const ushort xt=ubf(sum);
            RU(r,E_XT)[e]=xt;
            vstage[size_t(r)*VSTAGE+4352u+e]=xt;
        }
        SYNC();
        // ---- post-MLP conv (inputs of earlier rows from their combinations), residual
        if(group<R) {
            const uint r=group,position=p0+r;
            device ushort* vh=RU(r,E_H);
            const device float* vr=RF(r,E_VR);
            const device ushort* cw=(const device ushort*)(weights+d[D_MCONV]);
            for(uint i=tid;i<FH;i+=256u) {
                const float xt=bfu(RU(r,E_XT)[i]);
                float prev[3];prev_rows(ring,4352u+i,position,p0,RU(0,E_XT)+i,WSR/2u,prev);
                const float c=short_conv(prev,xt,cw+i*4u);
                vh[i]=ubf(vr[i]+c);
            }
            threadgroup_barrier(mem_flags::mem_device);
#if FRT_TRACE
            for(uint i=tid;i<FH;i+=256u)trace[(size_t(visit)*R+r)*FH+i]=vh[i];
#endif
            if(visit==4u || visit==22u || visit==40u) {
                const uint f=visit==4u?0u:(visit==22u?1u:2u);
                for(uint i=tid;i<FH;i+=256u)feat[(size_t(f)*VMAX+r)*FH+i]=vh[i];
            }
            if(layer==17u && pass<2u) {
                const float inv=rms_inv((const device ushort*)vh,red,lane,simd);
                for(uint i=tid;i<FH;i+=256u)RF(r,WS_BANK)[pass*FH+i]=bfu(vh[i])*inv;
            }
        }
        // the next visit's attention inputs of row r are threadgroup r's own (no grid barrier) unless it opens with
        // message gates (rows on threadgroups 3 r + j, the pass start's messages on all threadgroups)
        if(visit+1u<FVISITS) {
            const device ulong* nv=visits+(visit+1u)*64u;
            const bool next_messages=(table[uint(nv[V_LAYER])*64u+D_FLAGS]&2ul)!=0ul && uint(nv[V_PASS])>0u;
            if(next_messages){SYNC();}
            else threadgroup_barrier(mem_flags::mem_device);
        }
    }
    // ---- final input and norm of every row (also the drafter's last feature)
    if(group<R) {
        const uint r=group;
        const uint fslots[4]={0u,1u,4u,5u};
        const device ushort* vh=RU(r,E_H);
        const float inv=rms_inv((const device ushort*)vh,red,lane,simd);
        depth_attention<1>(RF(r,WS_BS),fslots,4u,(const device ushort*)vh,inv,
                           (const device float*)(weights+final[D_FINAL_Q]),tv,red,tid,lane,zero);
        const float inv2=rms_inv((const threadgroup float*)tv,red,lane,simd);
        const device ushort* w=(const device ushort*)(weights+final[D_FINAL_NORM]);
        device ushort* xf=RU(r,E_X);
        float m=0.0f;
        for(uint i=tid;i<FH;i+=256u){const ushort v=ubf(bfu(w[i])*(tv[i]*inv2));xf[i]=v;feat[(size_t(3u)*VMAX+r)*FH+i]=v;
            m=max(m,metal::abs(bfu(v)));}
#if FRT_Q8
        const float sc=tg_pow2_scale(m,red,tid,lane,simd);
        if(tid==0)((device float*)(RW(r)+E_XSC))[4]=sc;
#endif
    }
    SYNC();
    // ---- LM head with softcap for every row (threadgroup units of 64 vocabulary rows, staged inputs)
    {
        const device ushort* head=(const device ushort*)(weights+final[D_HEAD]);
#if FRT_Q8
        constexpr uint HTP=2u;
        for(uint u=group;u<(FVOCAB+64u*HTP-1u)/(64u*HTP);u+=FRT_WORKERS)
            q8_rows_tg<FH,true,HTP>(q8,q8t+FLAYERS*Q8_LAYER,u*64u*HTP,(const device ushort*)RU(0,E_X),
                                (const device float*)(RW(0)+E_XSC)+4,RSF,RU(0,E_X),logits,(threadgroup half*)xs,tid,simd,lane);
#elif FRT_FAST && FRT_ROWS>8u
        for(uint u=group;u<(FVOCAB+127u)/128u;u+=FRT_WORKERS)
            gemv_rows_mma<FH,true,2u>(head,u*128u,(const device ushort*)RU(0,E_X),RU(0,E_X),logits,xs,tid,simd,lane);
#else
        for(uint u=group;u<FVOCAB/(8u*PTM);u+=FRT_WORKERS)
            gemv_rows_tg<FH,true>(head,u*8u*PTM+simd*PTM,(const device ushort*)RU(0,E_X),RU(0,E_X),logits,xs,tid,lane);
#endif
    }
    SYNC();
    // ---- greedy targets: the maximum over the checkpoint vocabulary, ties the lowest id
    {
        const uint span=(FVOCAB_USED+FRT_WORKERS-1u)/FRT_WORKERS;
        const uint lo=group*span,hi=min(FVOCAB_USED,lo+span);
        for(uint r=0;r<R;++r) {
            float best=-FRT_INF;uint best_i=0xffffffffu;
            for(uint i=lo+tid;i<hi;i+=256u){const float v=logits[size_t(r)*FVOCAB+i];if(v>best){best=v;best_i=i;}}
            const float top=simd_max(best);
            const uint pick=simd_min(best==top?best_i:0xffffffffu);
            if(lane==0){red[simd]=top;((threadgroup uint*)red)[8u+simd]=pick;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(simd==0) {
                const float value=lane<8u?red[lane]:-FRT_INF;
                const uint word=lane<8u?((threadgroup uint*)red)[8u+lane]:0xffffffffu;
                const float maximum=simd_max(value);
                const uint choice=simd_min(value==maximum?word:0xffffffffu);
                if(lane==0){sel[(r*FRT_WORKERS+group)*2u]=maximum;((device uint*)sel)[(r*FRT_WORKERS+group)*2u+1u]=choice;}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    SYNC();
    if(group==0)for(uint r=simd;r<R;r+=8u) {
        float value=-FRT_INF;uint word=0xffffffffu;
        for(uint g=lane;g<FRT_WORKERS;g+=32u) {
            const float gv=sel[(r*FRT_WORKERS+g)*2u];const uint gw=((const device uint*)sel)[(r*FRT_WORKERS+g)*2u+1u];
            if(gv>value || (gv==value && gw<word)){value=gv;word=gw;}
        }
        const float maximum=simd_max(value);
        const uint choice=simd_min(value==maximum?word:0xffffffffu);
        if(lane==0)out[r]=choice;
    }
    if(tid==0)diagnostic[group]=0;
    #undef RW
    #undef RF
    #undef RU
}
)MSL";
}

}  // namespace flashrt::apple::shaders
