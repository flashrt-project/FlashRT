#pragma once
#include "common/source.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// y[N] = W . x for BF16 row vectors x[K] and affine 4-bit weights W[N,K] (codes packed two per byte,
// element 2i in the low nibble; one BF16 scale s and bias b per 64 inputs: w = s*q + b), FP32
// accumulation, BF16 output. The arithmetic is that of MLX's qmv_fast kernel for 4-bit affine weights
// (MLX, Copyright (c) 2023-2024 Apple Inc., MIT License): each lane takes 16 consecutive inputs of every
// 512-input block, x pre-divided by the nibble place values, per-group partial dot
// s * sum(x' * (q << 4j)) + sum(x) * b with the 4-input sums of x formed in BF16, a SIMD sum per output.
// Two SIMD groups of four outputs per threadgroup, 64 threads; requires K % 512 == 0 and N % 8 == 0.
//
// Single vector (rows = false): x and y at byte offsets p.x, p.y; grid (1, N/8).
// Rows (rows = true): grid (R, N/8), row r = threadgroup x. Dense: active while the chunk's M = state[1]
// is below p.limit (MLX's vector-path bound), rows r < M; M == 1 uses the arithmetic above, M >= 2 the
// WIDE variant (MLX's qmv_wide for affine weights, MLX architecture generation 15+): per output row 8 lanes take
// every 8th group of 64 inputs, each group dequantized in 8-input pieces to FP32 (w = s*q + b) and dotted
// sequentially, lane partials combined by a shuffle-down ladder (4, 2, 1). GROUPED: active while the chunk
// has fewer than 1024 expert assignments (p.per_row * state[1]); row r is an expert-sorted assignment, its
// found in `starts`, weights stacked [experts, N, K]. Epilogue: 0 store, 1 add to the BF16 residual in y,
// 2 SwiGLU of two matrices (gate in w, up in w2) as bf16(silu(bf16(g)) * bf16(u)) with BF16 rounding.
struct QmvOptions {
    bool rows = false;
    bool grouped = false;
    bool wide = false;
    int epilogue = 0;
};

inline std::string qmv_w4a16_affine_source(const QmvOptions& o = {}) {
    if (o.grouped && !o.rows) throw std::invalid_argument("grouped row-vector products need rows mode");
    if (!o.rows && o.epilogue) throw std::invalid_argument("single-vector product has no epilogue");
    if (o.wide && (!o.rows || o.grouped)) throw std::invalid_argument("wide row-vector products are dense rows");
    return std::string(metal_preamble) +
        "#define QMV_ROWS " + std::to_string(o.rows ? 1 : 0) + "\n" +
        "#define QMV_GROUPED " + std::to_string(o.grouped ? 1 : 0) + "\n" +
        "#define QMV_WIDE " + std::to_string(o.wide ? 1 : 0) + "\n" +
        "#define QMV_EPILOGUE " + std::to_string(o.epilogue) + "\n" + R"MSL(
struct AffineQmv {
    uint inner, columns, x_stride, y_stride;    // K, N, row strides (elements, rows mode)
    uint limit, experts, per_row, pad;          // GROUPED: assignments per chunk row
    ulong w, s, b, w2, s2, b2;                  // byte offsets (second set: up for SwiGLU)
    ulong x, y;                                 // byte offsets of x and y (single vector)
};
inline float rbf(float v){return as_type<float>(uint(as_type<ushort>(bfloat(v)))<<16);}
inline float swiglu_f(float g,float u) {
    float a=as_type<float>(as_type<uint>(g)&0x7fffffffu);
    float e=rbf(metal::fast::exp(a));
    float tail=rbf(1.0f/rbf(1.0f+e));
    float sig=g<0?tail:rbf(1.0f-tail);
    return rbf(rbf(g*sig)*u);
}
// Partial dot of 16 inputs against one weight row (kept as a separate expression: its rounding, including
// the contraction of scale * accum + sum * bias, is part of the reproduced arithmetic).
inline float qdot16(const device ushort* w,const thread float* x_thread,float scale,float bias,float sum) {
    float accum=0.0f;
    for(int i=0;i<4;++i) {
        accum+=(x_thread[4*i]*(w[i]&0x000f)+x_thread[4*i+1]*(w[i]&0x00f0)+
                x_thread[4*i+2]*(w[i]&0x0f00)+x_thread[4*i+3]*(w[i]&0xf000));
    }
    return scale*accum+sum*bias;
}
// Four outputs [out_row, out_row + 4) of one SIMD group; results valid in every lane.
inline void qmv4(const device uchar* codes,const device bfloat* scales,const device bfloat* biases,
                 const device bfloat* x,int K,int out_row,uint simd_lid,thread float* result) {
    constexpr int values_per_thread=16,block_size=512,group_size=64,scale_step=group_size/values_per_thread;
    const int row_bytes=K/2,row_groups=K/group_size;
    const device uchar* ws=codes+size_t(out_row)*row_bytes+simd_lid*8;
    scales+=size_t(out_row)*row_groups+simd_lid/scale_step;
    biases+=size_t(out_row)*row_groups+simd_lid/scale_step;
    x+=simd_lid*values_per_thread;
    float x_thread[values_per_thread];
    for(int row=0;row<4;++row)result[row]=0.0f;
    for(int k=0;k<K;k+=block_size) {
        float sum=0.0f;
        for(int i=0;i<values_per_thread;i+=4) {
            sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
            x_thread[i]=x[i];
            x_thread[i+1]=x[i+1]/16.0f;
            x_thread[i+2]=x[i+2]/256.0f;
            x_thread[i+3]=x[i+3]/4096.0f;
        }
        for(int row=0;row<4;++row) {
            const device ushort* wl=(const device ushort*)(ws+size_t(row)*row_bytes);
            float s=scales[row*row_groups],b=biases[row*row_groups];
            result[row]+=qdot16(wl,x_thread,s,b,sum);
        }
        ws+=block_size/2;
        scales+=block_size/group_size;
        biases+=block_size/group_size;
        x+=block_size;
    }
    for(int row=0;row<4;++row)result[row]=simd_sum(result[row]);
}
// One output row (out_row) by 8 lanes k_lane = lane % 8; the sum is valid in k_lane 0.
inline float qmv_wide1(const device uchar* codes,const device bfloat* scales,const device bfloat* biases,
                       const device bfloat* x,int K,int out_row,uint k_lane) {
    const int row_bytes=K/2,groups=K/64;
    const device uchar* wrow=codes+size_t(out_row)*row_bytes;
    const device bfloat* srow=scales+size_t(out_row)*groups;
    const device bfloat* brow=biases+size_t(out_row)*groups;
    float result=0.0f;
    for(int g=int(k_lane);g<groups;g+=8) {
        float scale=srow[g],bias=brow[g];
        for(int sc=0;sc<8;++sc) {
            const int k0=g*64+sc*8;
            const device uchar* wc=wrow+k0/2;
            float w_dq[8];
            float s[2]={scale,scale/16.0f};
            for(int i=0;i<4;++i) {
                w_dq[2*i]=s[0]*(wc[i]&0x0f)+bias;
                w_dq[2*i+1]=s[1]*(wc[i]&0xf0)+bias;
            }
            const device bfloat* xc=x+k0;
            float acc=0.0f;
            for(int i=0;i<8;++i)acc+=static_cast<float>(xc[i])*w_dq[i];
            result+=acc;
        }
    }
    result+=simd_shuffle_down(result,ushort(4));
    result+=simd_shuffle_down(result,ushort(2));
    result+=simd_shuffle_down(result,ushort(1));
    return result;
}
kernel void qmv_w4a16_affine(const device uchar* weights [[buffer(0)]],const device bfloat* xbase [[buffer(1)]],
    device bfloat* ybase [[buffer(2)]],constant AffineQmv& p [[buffer(3)]],
#if QMV_ROWS
    const device uint* state [[buffer(4)]],const device uint* starts [[buffer(5)]],
#endif
    uint3 tid [[threadgroup_position_in_grid]],uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    const int K=int(p.inner);
    const int out_row=int(tid.y)*8+int(simd_gid)*4;
    if(out_row>=int(p.columns))return;
    ulong expert_codes=0,expert_scales=0;
#if QMV_ROWS
    const uint r=tid.x;
#if QMV_GROUPED
    const uint assignments=state[1]*p.per_row;
    if(assignments>=1024u || r>=assignments)return;
    uint e_lo=0,e_hi=p.experts;
    while(e_lo<e_hi){uint mid=(e_lo+e_hi)/2;if(starts[mid+1]<=r)e_lo=mid+1;else e_hi=mid;}
    expert_codes=ulong(e_lo)*p.columns*(p.inner/2u);
    expert_scales=ulong(e_lo)*p.columns*(p.inner/64u)*2u;
#else
    if(state[1]>=p.limit || r>=state[1])return;
#if QMV_WIDE
    if(state[1]<2u)return;
#else
    if(state[1]>=2u)return;
#endif
#endif
    const device bfloat* x=xbase+size_t(r)*p.x_stride;
    device bfloat* y=ybase+size_t(r)*p.y_stride+out_row;
#else
    const device bfloat* x=(const device bfloat*)((const device uchar*)xbase+p.x);
    device bfloat* y=(device bfloat*)((device uchar*)ybase+p.y)+out_row;
#endif
#if QMV_WIDE
    // SIMD group: 4 output rows of 8 lanes each.
    const uint k_lane=simd_lid%8u,sg_row=simd_lid/8u;
    const int wrow=out_row+int(sg_row);
    float result1=qmv_wide1(weights+p.w,(const device bfloat*)(weights+p.s),(const device bfloat*)(weights+p.b),
                            x,K,wrow,k_lane);
#if QMV_EPILOGUE==2
    float up1=qmv_wide1(weights+p.w2,(const device bfloat*)(weights+p.s2),(const device bfloat*)(weights+p.b2),
                        x,K,wrow,k_lane);
#endif
    if(k_lane==0u) {
        float v=rbf(result1);
#if QMV_EPILOGUE==1
        v=rbf(v+float(y[sg_row]));
#elif QMV_EPILOGUE==2
        v=swiglu_f(v,rbf(up1));
#endif
        y[sg_row]=static_cast<bfloat>(v);
    }
#else
    float result[4];
    qmv4(weights+p.w+expert_codes,(const device bfloat*)(weights+p.s+expert_scales),
         (const device bfloat*)(weights+p.b+expert_scales),x,K,out_row,simd_lid,result);
#if QMV_EPILOGUE==2
    float up[4];
    qmv4(weights+p.w2+expert_codes,(const device bfloat*)(weights+p.s2+expert_scales),
         (const device bfloat*)(weights+p.b2+expert_scales),x,K,out_row,simd_lid,up);
#endif
    if(simd_lid==0) {
        for(int row=0;row<4;++row) {
            float v=rbf(result[row]);
#if QMV_EPILOGUE==1
            v=rbf(v+float(y[row]));
#elif QMV_EPILOGUE==2
            v=swiglu_f(v,rbf(up[row]));
#endif
            y[row]=static_cast<bfloat>(v);
        }
    }
#endif
}
)MSL";
}
}  // namespace flashrt::apple::shaders
