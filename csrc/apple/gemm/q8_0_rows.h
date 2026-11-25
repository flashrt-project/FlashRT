#pragma once
#include "common/source.h"
#include <string>

// Q8_0 weights (blocks of 32 inputs: int8 codes and one FP16 scale = max |w| / 127) for products with a few rows.
//   q8_0_quantize_bf16: BF16 [rows][K] (in the weight buffer) -> codes int8 [rows][K], scales half [rows][K / 32] (one thread per block).
//   q8_0_rows_f32: y[r][n] = sum_k x[r][k] w[n][k] for r < rows (<= 8), FP32 out (K % 64 == 0, K <= 3072).
//   q8_0_rows_f32_staged: the same for rows * K <= 16128, a fixed grid of threadgroups each staging the input once.
//   q8_0_prep_rows + q8_0_rows8: up to 8 rows, any K % 256 == 0, FP32 or BF16 out.
namespace flashrt::apple::shaders {
inline std::string q8_0_rows_source() {
    return std::string(metal_preamble) + R"MSL(
struct Q8Quant { ulong src; uint rows, k, pad0, pad1; };
// src: byte offset of the BF16 matrix in `weights` (bound at 0: binding offsets above 4 GiB are not reliable)
kernel void q8_0_quantize_bf16(const device uchar* weights [[buffer(0)]],device char* codes [[buffer(1)]],
    device half* scales [[buffer(2)]],constant Q8Quant& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    const uint nb=p.k/32u;
    if(i>=p.rows*nb)return;
    const uint row=i/nb,b=i%nb;
    const device ushort* s=(const device ushort*)(weights+p.src)+size_t(row)*p.k+b*32u;
    float v[32],amax=0.0f;
    for(uint j=0;j<32u;++j){v[j]=as_type<float>(uint(s[j])<<16);amax=max(amax,metal::abs(v[j]));}
    const half hs=half(amax/127.0f);
    const float sc=float(hs);
    scales[size_t(row)*nb+b]=hs;
    device char* c=codes+size_t(row)*p.k+b*32u;
    for(uint j=0;j<32u;++j)c[j]=sc>0.0f?char(clamp(metal::rint(v[j]/sc),-127.0f,127.0f)):char(0);
}
struct Q8Rows { uint rows, n, k, x_stride, y_stride, pad0, pad1, pad2; };
// a threadgroup: 64 output rows (a SIMD group 8: LPR = 8 lanes per row pair, 2 rows per lane); x of half of K staged as
// half times a power of two per row (max |x| 2^e below 4: the 16-term half2 sums stay below 2^13); per 32 inputs the
// half2 sums times the block scale into FP32 (K % 64 == 0, rows <= 8)
kernel void q8_0_rows_f32(const device ushort* x [[buffer(0)]],const device char* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],device float* y [[buffer(3)]],constant Q8Rows& p [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half xh[8*1536];
    threadgroup float red[8][8];
    threadgroup float xsc[8];
    const uint R=p.rows,K=p.k,KH=K/2u,NC=KH/32u;
    // per-row scale: max |x| over the row
    for(uint r=0;r<R;++r) {
        float m=0.0f;
        for(uint i=tid;i<K;i+=256u)m=max(m,metal::abs(as_type<float>(uint(x[size_t(r)*p.x_stride+i])<<16)));
        m=simd_max(m);
        if(lane==0)red[r][simd]=m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<R){float m=0.0f;for(uint i=0;i<8u;++i)m=max(m,red[tid][i]);
        const int ex=m>0.0f?int((as_type<uint>(m)>>23)&255u)-127:0;
        xsc[tid]=as_type<float>(uint(127+clamp(1-ex,-100,100))<<23);}
    const uint rg=lane/8u,jl=lane%8u;
    const uint row=g*64u+simd*8u+rg*2u;
    float acc[2][8];
    for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r)acc[q][r]=0.0f;
    for(uint hk=0;hk<2u;++hk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<R*KH;i+=256u){const uint r=i/KH,j=i%KH;
            xh[r*KH+j]=half(as_type<float>(uint(x[size_t(r)*p.x_stride+hk*KH+j])<<16)*xsc[r]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint b=0;b*8u<NC;++b) {
            const uint cl=b*8u+jl;
            if(cl>=NC)break;
            const uint c=hk*NC+cl;
            half2 wv[2][16]; float S[2];
            for(uint q=0;q<2u;++q) {
                const uint n=min(row+q,p.n-1u);
                const device uint4* cw=(const device uint4*)(codes+size_t(n)*K)+2u*c;
                const uint4 a=cw[0],bb=cw[1];
                S[q]=float(scales[size_t(n)*(K/32u)+c]);
                const uint w8[8]={a.x,a.y,a.z,a.w,bb.x,bb.y,bb.z,bb.w};
                for(uint i=0;i<8u;++i){const half4 h=half4(as_type<char4>(w8[i]));wv[q][2u*i]=h.xy;wv[q][2u*i+1u]=h.zw;}
            }
            for(uint r=0;r<8u;++r) {
                if(r>=R)break;
                const threadgroup half2* xp=(const threadgroup half2*)(xh+r*KH+32u*cl);
                for(uint q=0;q<2u;++q) {
                    half2 s0=xp[0]*wv[q][0],s1=xp[1]*wv[q][1];
                    for(uint i=2;i<16u;i+=2){s0=fma(xp[i],wv[q][i],s0);s1=fma(xp[i+1],wv[q][i+1],s1);}
                    const float2 f=float2(s0+s1);
                    acc[q][r]=metal::fma(S[q],f.x+f.y,acc[q][r]);
                }
            }
        }
    }
    for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r) {
        if(r>=R)break;
        float v=acc[q][r];
        v+=simd_shuffle_xor(v,4);v+=simd_shuffle_xor(v,2);v+=simd_shuffle_xor(v,1);
        if(jl==0u && row+q<p.n)y[size_t(r)*p.y_stride+row+q]=v/xsc[r];
    }
}
// Persistent variant for rows * K <= 16128 (e.g. 7 rows of 2304): the whole input staged once per threadgroup, then
// 64-row tiles g, g + groups, ... (same lane layout, block scales and power-of-two input scale as q8_0_rows_f32; the
// input's scale from all of K at once).
kernel void q8_0_rows_f32_staged(const device ushort* x [[buffer(0)]],const device char* codes [[buffer(1)]],
    const device half* scales [[buffer(2)]],device float* y [[buffer(3)]],constant Q8Rows& p [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]],uint groups [[threadgroups_per_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half xh[16128];
    threadgroup float red[8][8];
    threadgroup float xsc[8];
    const uint R=p.rows,K=p.k,NC=K/32u;
    for(uint r=0;r<R;++r) {
        float m=0.0f;
        for(uint i=tid;i<K;i+=256u)m=max(m,metal::abs(as_type<float>(uint(x[size_t(r)*p.x_stride+i])<<16)));
        m=simd_max(m);
        if(lane==0)red[r][simd]=m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<R){float m=0.0f;for(uint i=0;i<8u;++i)m=max(m,red[tid][i]);
        const int ex=m>0.0f?int((as_type<uint>(m)>>23)&255u)-127:0;
        xsc[tid]=as_type<float>(uint(127+clamp(1-ex,-100,100))<<23);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<R*K;i+=256u){const uint r=i/K,j=i%K;
        xh[i]=half(as_type<float>(uint(x[size_t(r)*p.x_stride+j])<<16)*xsc[r]);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rg=lane/8u,jl=lane%8u;
    const uint ntiles=(p.n+63u)/64u;
    for(uint t=g;t<ntiles;t+=groups) {
        const uint row=t*64u+simd*8u+rg*2u;
        float acc[2][8];
        for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r)acc[q][r]=0.0f;
        for(uint b=0;b*8u<NC;++b) {
            const uint c=b*8u+jl;
            if(c>=NC)break;
            half2 wv[2][16]; float S[2];
            for(uint q=0;q<2u;++q) {
                const uint n=min(row+q,p.n-1u);
                const device uint4* cw=(const device uint4*)(codes+size_t(n)*K)+2u*c;
                const uint4 a=cw[0],bb=cw[1];
                S[q]=float(scales[size_t(n)*(K/32u)+c]);
                const uint w8[8]={a.x,a.y,a.z,a.w,bb.x,bb.y,bb.z,bb.w};
                for(uint i=0;i<8u;++i){const half4 h=half4(as_type<char4>(w8[i]));wv[q][2u*i]=h.xy;wv[q][2u*i+1u]=h.zw;}
            }
            for(uint r=0;r<8u;++r) {
                if(r>=R)break;
                const threadgroup half2* xp=(const threadgroup half2*)(xh+r*K+32u*c);
                for(uint q=0;q<2u;++q) {
                    half2 s0=xp[0]*wv[q][0],s1=xp[1]*wv[q][1];
                    for(uint i=2;i<16u;i+=2){s0=fma(xp[i],wv[q][i],s0);s1=fma(xp[i+1],wv[q][i+1],s1);}
                    const float2 f=float2(s0+s1);
                    acc[q][r]=metal::fma(S[q],f.x+f.y,acc[q][r]);
                }
            }
        }
        for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r) {
            if(r>=R)break;
            float v=acc[q][r];
            v+=simd_shuffle_xor(v,4);v+=simd_shuffle_xor(v,2);v+=simd_shuffle_xor(v,1);
            if(jl==0u && row+q<p.n)y[size_t(r)*p.y_stride+row+q]=v/xsc[r];
        }
    }
}
// Products of up to 8 rows with any K (K % 256 == 0): q8_0_prep_rows scales each input row by a power of two (max |x|
// 2^e below 4) into half (xh [rows][K], xsc [rows]); q8_0_rows8 (a fixed grid) takes pairs of 64-row tiles (one tile:
// flags & 2), the input staged KC = p.kc halves at a time (KC % 256 == 0, KC <= 1536), the same lanes / block scales
// as q8_0_rows_f32; out FP32 or BF16 (flags & 1) at y[r * y_stride + y_col + n].
struct Q8Rows8 { uint rows, n, k, x_stride, y_stride, y_col, kc, flags; };
kernel void q8_0_prep_rows(const device ushort* x [[buffer(0)]],device half* xh [[buffer(1)]],device float* xsc [[buffer(2)]],
    constant Q8Rows8& p [[buffer(3)]],uint r [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(r>=p.rows)return;
    threadgroup float red[8];
    const device ushort* xr=x+size_t(r)*p.x_stride;
    float m=0.0f;
    for(uint i=tid;i<p.k;i+=256u)m=max(m,metal::abs(as_type<float>(uint(xr[i])<<16)));
    m=simd_max(m);
    if(lane==0)red[simd]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=red[0];for(uint i=1;i<8u;++i)m=max(m,red[i]);
    const int ex=m>0.0f?int((as_type<uint>(m)>>23)&255u)-127:0;
    const float sc=as_type<float>(uint(127+clamp(1-ex,-100,100))<<23);
    if(tid==0)xsc[r]=sc;
    for(uint i=tid;i<p.k;i+=256u)xh[size_t(r)*p.k+i]=half(as_type<float>(uint(xr[i])<<16)*sc);
}
kernel void q8_0_rows8(const device half* xh [[buffer(0)]],const device float* xsc [[buffer(1)]],
    const device char* codes [[buffer(2)]],const device half* scales [[buffer(3)]],device uchar* y [[buffer(4)]],
    constant Q8Rows8& p [[buffer(5)]],uint g [[threadgroup_position_in_grid]],uint groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half xs[8*1536];
    const uint R=p.rows,K=p.k,KC=p.kc,NCC=KC/32u;
    const uint rg=lane/8u,jl=lane%8u;
    const uint TP=(p.flags&2u)?1u:2u;
    const uint npairs=(p.n+64u*TP-1u)/(64u*TP);
    float inv[8];
    for(uint r=0;r<8u;++r)inv[r]=r<R?1.0f/xsc[r]:0.0f;
    for(uint pr=g;pr<npairs;pr+=groups) {
        float acc[2][2][8];
        for(uint t=0;t<2u;++t)for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r)acc[t][q][r]=0.0f;
        for(uint k0=0;k0<K;k0+=KC) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=tid;i<R*(KC/8u);i+=256u){const uint r=i/(KC/8u),j=i%(KC/8u);
                ((threadgroup uint4*)xs)[i]=*((const device uint4*)(xh+size_t(r)*K+k0)+j);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint t=0;t<TP;++t) {
                const uint row=(pr*TP+t)*64u+simd*8u+rg*2u;
                if(row>=p.n)break;
                for(uint b=0;b*8u<NCC;++b) {
                    const uint cl=b*8u+jl;
                    const uint c=k0/32u+cl;
                    half2 wv[2][16]; float S[2];
                    for(uint q=0;q<2u;++q) {
                        const uint n=min(row+q,p.n-1u);
                        const device uint4* cw=(const device uint4*)(codes+size_t(n)*K)+2u*c;
                        const uint4 a=cw[0],bb=cw[1];
                        S[q]=float(scales[size_t(n)*(K/32u)+c]);
                        const uint w8[8]={a.x,a.y,a.z,a.w,bb.x,bb.y,bb.z,bb.w};
                        for(uint i=0;i<8u;++i){const half4 h=half4(as_type<char4>(w8[i]));wv[q][2u*i]=h.xy;wv[q][2u*i+1u]=h.zw;}
                    }
                    for(uint r=0;r<8u;++r) {
                        if(r>=R)break;
                        const threadgroup half2* xp=(const threadgroup half2*)(xs+r*KC+32u*cl);
                        for(uint q=0;q<2u;++q) {
                            half2 s0=xp[0]*wv[q][0],s1=xp[1]*wv[q][1];
                            for(uint i=2;i<16u;i+=2){s0=fma(xp[i],wv[q][i],s0);s1=fma(xp[i+1],wv[q][i+1],s1);}
                            const float2 f=float2(s0+s1);
                            acc[t][q][r]=metal::fma(S[q],f.x+f.y,acc[t][q][r]);
                        }
                    }
                }
            }
        }
        for(uint t=0;t<TP;++t) {
            const uint row=(pr*TP+t)*64u+simd*8u+rg*2u;
            for(uint q=0;q<2u;++q)for(uint r=0;r<8u;++r) {
                if(r>=R)break;
                float v=acc[t][q][r];
                v+=simd_shuffle_xor(v,4);v+=simd_shuffle_xor(v,2);v+=simd_shuffle_xor(v,1);
                if(jl==0u && row+q<p.n) {
                    const size_t o=size_t(r)*p.y_stride+p.y_col+row+q;
                    const float vv=v*inv[r];
                    const uint u=as_type<uint>(vv);
                    if(p.flags&1u)((device ushort*)y)[o]=ushort((u+0x7fffu+((u>>16)&1u))>>16);   // BF16, nearest even
                    else ((device float*)y)[o]=vv;
                }
            }
        }
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
