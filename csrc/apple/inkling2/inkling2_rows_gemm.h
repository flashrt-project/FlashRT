#pragma once
#include "common/source.h"
#include <string>

// TensorOps products of the Inkling 2 Mini row programs (FP32 accumulation over 64-input slices).
//   ink_gemm_bf16_*: y[r][col0 + n] = x[r] . W[n] for a BF16 weight [N][K] read in place; outputs BF16 (rounded),
//     FP32, or FP32 softcapped logits tanh(bf16(acc) / 30) * 30 (LM head).
//   ink_gemm_experts_*: rows grouped by expert (`starts`, `tile_starts`: prefix sums from ink_expert_scan); weight
//     tiles dequantized to BF16 in threadgroup memory (NVFP4: E2M1 x E4M3 scale, exact in BF16, the expert's global
//     scale / (448 * 6) applied to the accumulator; MXFP8: E4M3 x E8M0); gate/up epilogue = the softcapped SwiGLU of
//     the interleaved gate (2c) and up (2c + 1) rows.

namespace flashrt::apple::shaders {

struct Inkling2GemmOptions {
    int tile_m = 64;
    int tile_n = 64;
    int simdgroups = 4;
    int kslice = 64;          // expert tiles: inputs per dequantized slice
};

inline std::string inkling2_rows_gemm_source(const Inkling2GemmOptions& o = {}) {
    return std::string(tensor_ops_preamble) +
        "constant constexpr int TM=" + std::to_string(o.tile_m) + ";\n" +
        "constant constexpr int TN=" + std::to_string(o.tile_n) + ";\n" +
        "constant constexpr int SG=" + std::to_string(o.simdgroups) + ";\n" +
        "constant constexpr int KS=" + std::to_string(o.kslice) + ";\n" + R"MSL(
#pragma METAL fp math_mode(safe)
inline float rbf(float v){uint u=as_type<uint>(v);return as_type<float>((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}
struct RowsGemm {
    uint rows, n, k, x_stride;     // rows of x, output columns, inputs, x row stride (elements)
    uint y_stride, y_col, pad0, pad1;
    ulong w;                       // byte offset of W [n][k] (BF16) in the weight buffer
};
template<int MODE>   // 0 BF16 out, 1 FP32 out, 2 FP32 softcapped logits
inline void gemm_rows(const device bfloat* x,const device uchar* weights,device void* y,constant RowsGemm& p,uint2 g) {
    const uint n0=g.x*uint(TN),row0=g.y*uint(TM);
    if(n0>=p.n || row0>=p.rows)return;
    const uint rows=min(uint(TM),p.rows-row0),cols=min(uint(TN),p.n-n0);
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> left((device bfloat*)x+size_t(row0)*p.x_stride,E(int(p.k),int(rows)),
                                             array<int,2>{1,int(p.x_stride)});
    tensor<device bfloat,E,tensor_inline> right((device bfloat*)(weights+p.w)+size_t(n0)*p.k,E(int(p.k),int(cols)),
                                              array<int,2>{1,int(p.k)});
    constexpr auto desc=matmul2d_descriptor(TM,TN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
    for(uint k0=0;k0<p.k;k0+=64) {
        auto l=left.slice(int(k0),0);
        auto r=right.slice(int(k0),0);
        op.run(l,r,acc);
    }
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]>=int(rows) || c[0]>=int(cols))continue;
        const size_t at=size_t(row0+c[1])*p.y_stride+p.y_col+n0+c[0];
        if(MODE==0)((device bfloat*)y)[at]=bfloat(rbf(acc[i]));
        else if(MODE==1)((device float*)y)[at]=acc[i];
        else ((device float*)y)[at]=metal::precise::tanh(metal::precise::divide(rbf(acc[i]),30.0f))*30.0f;
    }
}
kernel void ink_gemm_bf16_rows(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device void* y [[buffer(2)]],constant RowsGemm& p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]])
    {gemm_rows<0>(x,weights,y,p,g);}
kernel void ink_gemm_f32_rows(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device void* y [[buffer(2)]],constant RowsGemm& p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]])
    {gemm_rows<1>(x,weights,y,p,g);}
kernel void ink_gemm_logits_rows(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device void* y [[buffer(2)]],constant RowsGemm& p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]])
    {gemm_rows<2>(x,weights,y,p,g);}

// Several independent products in one dispatch: column tiles of matrix i are g.x in [tile0_i, tile0_{i+1}); inputs
// x0..x2 and outputs y0 / y1 chosen per matrix.
struct MultiMat { ulong w; uint n, k, x_sel, x_stride, y_sel, y_col, y_stride, tile0, pad; };
struct MultiGemm { uint count, rows, pad0, pad1; MultiMat m[8]; };
template<int MODE>
inline void gemm_multi(const device bfloat* x0,const device bfloat* x1,const device bfloat* x2,
                       const device uchar* weights,device void* y0,device void* y1,constant MultiGemm& p,uint2 g) {
    uint mi=0;
    while(mi+1u<p.count && g.x>=p.m[mi+1u].tile0)++mi;
    const uint n=p.m[mi].n,K=p.m[mi].k,xs=p.m[mi].x_stride,ys=p.m[mi].y_stride,yc=p.m[mi].y_col;
    const uint n0=(g.x-p.m[mi].tile0)*uint(TN),row0=g.y*uint(TM);
    if(n0>=n || row0>=p.rows)return;
    const device bfloat* x=p.m[mi].x_sel==0u?x0:(p.m[mi].x_sel==1u?x1:x2);
    device void* y=p.m[mi].y_sel==0u?y0:y1;
    const uint rows=min(uint(TM),p.rows-row0),cols=min(uint(TN),n-n0);
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> left((device bfloat*)x+size_t(row0)*xs,E(int(K),int(rows)),array<int,2>{1,int(xs)});
    tensor<device bfloat,E,tensor_inline> right((device bfloat*)(weights+p.m[mi].w)+size_t(n0)*K,E(int(K),int(cols)),
                                              array<int,2>{1,int(K)});
    constexpr auto desc=matmul2d_descriptor(TM,TN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
    for(uint k0=0;k0<K;k0+=64) {
        auto l=left.slice(int(k0),0);
        auto r=right.slice(int(k0),0);
        op.run(l,r,acc);
    }
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]>=int(rows) || c[0]>=int(cols))continue;
        const size_t at=size_t(row0+c[1])*ys+yc+n0+c[0];
        if(MODE==0)((device bfloat*)y)[at]=bfloat(rbf(acc[i]));
        else ((device float*)y)[at]=acc[i];
    }
}
kernel void ink_gemm_bf16_multi(const device bfloat* x0 [[buffer(0)]],const device bfloat* x1 [[buffer(1)]],
    const device bfloat* x2 [[buffer(2)]],const device uchar* weights [[buffer(3)]],device void* y0 [[buffer(4)]],
    device void* y1 [[buffer(5)]],constant MultiGemm& p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]])
    {gemm_multi<0>(x0,x1,x2,weights,y0,y1,p,g);}
kernel void ink_gemm_f32_multi(const device bfloat* x0 [[buffer(0)]],const device bfloat* x1 [[buffer(1)]],
    const device bfloat* x2 [[buffer(2)]],const device uchar* weights [[buffer(3)]],device void* y0 [[buffer(4)]],
    device void* y1 [[buffer(5)]],constant MultiGemm& p [[buffer(6)]],uint2 g [[threadgroup_position_in_grid]])
    {gemm_multi<1>(x0,x1,x2,weights,y0,y1,p,g);}

// ------------------------------------------------------------------------------------------ experts
struct ExpertGemm {
    uint assignments, k, n_out, mx;  // assignment rows, inputs, output columns, 1 = MXFP8
    ulong w, s, gs;                  // codes, scales, NVFP4 global scales (byte offsets)
};
inline float e2m1(uint b){half c=as_type<half>(ushort((b&7u)<<9));c*=16384.0h;return float((b&8u)?-c:c);}
constant float E2M1_LUT[16]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,-0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
inline float e4m3(uint b){
    ushort v=ushort(b&127u);ushort sign=ushort(((b>>7)&1u)<<15);
    ushort u=ushort((v<<7)|(((v+1u)>>7)<<14)|sign);
    return float(as_type<half>(u)*half(256.0h));
}
inline float e8m0(uint b){return as_type<float>(b==0u?0x400000u:(b<<23));}
// rows (weight rows) r0 + stride * j for j < TN, inputs [k0, k0 + 64) of expert e, into tile[TN][64] BF16
inline void expert_tile(const device uchar* codes,const device uchar* scales,uint K,uint e,uint mx,uint rows_total,
                        uint r0,uint stride,uint k0,threadgroup bfloat* tile,uint t) {
    const uint threads=uint(SG)*32u;
    for(uint i=t;i<uint(TN)*uint(KS/8);i+=threads) {     // 8 elements per item
        const uint j=i/uint(KS/8),part=i%uint(KS/8);
        const uint wr=r0+stride*j;
        threadgroup bfloat* out=tile+j*uint(KS)+part*8u;
        if(wr>=rows_total){for(uint q=0;q<8u;++q)out[q]=bfloat(0.0f);continue;}
        const size_t row=size_t(e)*rows_total+wr;
        const uint c=k0+part*8u;
        if(mx==0u) {
            const uint word=*((const device uint*)(codes+row*(K/2u)+c/2u));
            const float sc=e4m3(scales[row*(K/16u)+c/16u]);
            for(uint q=0;q<8u;++q)out[q]=bfloat(E2M1_LUT[(word>>(4u*q))&15u]*sc);
        } else {
            const uint2 word=*((const device uint2*)(codes+row*K+c));
            const float sc=e8m0(scales[row*(K/32u)+c/32u]);
            for(uint q=0;q<4u;++q){out[q]=bfloat(e4m3((word.x>>(8u*q))&255u)*sc);out[4u+q]=bfloat(e4m3((word.y>>(8u*q))&255u)*sc);}
        }
    }
}
inline float sig_f(float x){float y=metal::precise::divide(1.0f,1.0f+metal::precise::exp(metal::abs(x)));return x<0.0f?y:1.0f-y;}
inline float swiglu_cap(float g,float u) {
    float gc=metal::precise::tanh(metal::precise::divide(g,4.0f))*4.0f;
    float uc=metal::precise::tanh(metal::precise::divide(u,25.0f))*25.0f;
    return rbf(gc*sig_f(g)*uc);
}
template<bool GATEUP>
inline void experts(const device bfloat* x,const device uchar* weights,device bfloat* y,constant ExpertGemm& p,
                    const device uint* starts,const device uint* tile_starts,uint2 g,uint t,
                    threadgroup bfloat* tile,threadgroup bfloat* tile2) {
    // which expert / row tile: tile_starts[e] <= g.y < tile_starts[e + 1]
    uint lo=0u,hi=128u;
    if(g.y>=tile_starts[128])return;
    while(hi-lo>1u){const uint mid=(lo+hi)/2u;if(tile_starts[mid]<=g.y)lo=mid;else hi=mid;}
    const uint e=lo;
    const uint row0=starts[e]+(g.y-tile_starts[e])*uint(TM);
    const uint row_end=starts[e+1];
    if(row0>=row_end)return;
    const uint rows=min(uint(TM),row_end-row0);
    const uint n0=g.x*uint(TN);
    if(n0>=p.n_out)return;
    const uint cols=min(uint(TN),p.n_out-n0);
    const device uchar* codes=weights+p.w;
    const device uchar* scales=weights+p.s;
    const uint wrows=GATEUP?2u*p.n_out:p.n_out;
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> left((device bfloat*)x+size_t(row0)*p.k,E(int(p.k),int(rows)),array<int,2>{1,int(p.k)});
    tensor<threadgroup bfloat,E,tensor_inline> right(tile,E(KS,TN),array<int,2>{1,KS});
    constexpr auto desc=matmul2d_descriptor(TM,TN,KS,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
    tensor<threadgroup bfloat,E,tensor_inline> right2(GATEUP?tile2:tile,E(KS,TN),array<int,2>{1,KS});
    auto acc2=op.get_destination_cooperative_tensor<decltype(left),decltype(right2),float>();
    if(GATEUP)for(uint i=0;i<acc2.get_capacity();++i)if(acc2.is_valid_element(i))acc2[i]=0.0f;
    for(uint k0=0;k0<p.k;k0+=uint(KS)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(GATEUP) {
            expert_tile(codes,scales,p.k,e,p.mx,wrows,2u*n0,2u,k0,tile,t);
            expert_tile(codes,scales,p.k,e,p.mx,wrows,2u*n0+1u,2u,k0,tile2,t);
        } else {
            expert_tile(codes,scales,p.k,e,p.mx,wrows,n0,1u,k0,tile,t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto l=left.slice(int(k0),0);
        op.run(l,right,acc);
        if(GATEUP)op.run(l,right2,acc2);
    }
    const float gsc=p.mx?1.0f:metal::precise::divide(((const device float*)(weights+p.gs))[e],2688.0f);
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]>=int(rows) || c[0]>=int(cols))continue;
        const size_t at=size_t(row0+c[1])*p.n_out+n0+c[0];
        if(GATEUP)y[at]=bfloat(swiglu_cap(rbf(acc[i]*gsc),rbf(acc2[i]*gsc)));
        else y[at]=bfloat(rbf(acc[i]*gsc));
    }
}
kernel void ink_experts_gateup(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant ExpertGemm& p [[buffer(3)]],const device uint* starts [[buffer(4)]],
    const device uint* tile_starts [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[TN*KS],tile2[TN*KS];
    experts<true>(x,weights,y,p,starts,tile_starts,g,t,tile,tile2);
}
kernel void ink_experts_down(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant ExpertGemm& p [[buffer(3)]],const device uint* starts [[buffer(4)]],
    const device uint* tile_starts [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]]) {
    threadgroup bfloat tile[TN*KS];
    experts<false>(x,weights,y,p,starts,tile_starts,g,t,tile,tile);
}
)MSL";
}

}  // namespace flashrt::apple::shaders
