#pragma once
#include "common/source.h"
#include <stdexcept>
#include <string>

namespace flashrt::apple::shaders {
// Y[M,N] = X[M,K] . W^T for affine 4-bit weights W[N,K] (codes packed two per byte, element 2i in the
// low nibble; one BF16 scale and bias per 64 inputs: w = scale*q + bias). Each 64-input slice of a
// BN-row weight tile is dequantized to BF16 in threadgroup memory (w = bf16(fma(scale, q, bias))) and
// multiplied with TensorOps matmul2d into FP32 accumulators; the output is rounded to BF16 once.
//
// Variants (EPILOGUE): 0 store, 1 add to a BF16 residual (rounded after the add), 2 SwiGLU of two
// matrices (gate, up) written as bf16(silu(bf16(g)) * bf16(u)) with the export's BF16 rounding.
// GROUPED: rows are grouped by expert; `starts[e]` gives each expert's first row and the weight
// matrices are stacked [experts, N, K]. Grid y enumerates (expert, row-tile) pairs by `tiles[e]`.
//
// Row count M comes from `state[1]` (chunk rows; GROUPED: per_row * chunk rows assignments). Each product
// reproduces the accumulation MLX's quantized matmul uses for that M (Metal, M4 Pro/Max): a single K chain
// when its split-K rule gives one partition, otherwise (SPLITK variant) k partitions of K/k inputs whose
// BF16 results are summed as MLX's reduction does (k <= 8: in order with BF16 rounding per add; 16: eight
// interleaved BF16 accumulators, then in order; 32: FP32). M below `limit` (MLX's vector-path bound) or a
// GROUPED product below 1024 assignments is left to the row-vector kernel; each variant returns otherwise.
struct W4A16Options {
    int tile_m = 64, tile_n = 64, simdgroups = 2;
    int epilogue = 0;      // 0 store, 1 residual add, 2 SwiGLU (gate/up)
    bool grouped = false;
    bool splitk = false;   // the split-K partition path (non-grouped)
};

inline std::string gemm_w4a16_affine_source(const W4A16Options& o = {}) {
    if (o.tile_m <= 0 || o.tile_n <= 0 || o.simdgroups <= 0 || o.tile_n % 32)
        throw std::invalid_argument("w4a16 GEMM tile dimensions must be positive, tile_n a multiple of 32");
    if (o.splitk && o.grouped)
        throw std::invalid_argument("w4a16 GEMM split-K applies to non-grouped products");
    return std::string(tensor_ops_preamble) +
        "constant constexpr int TM=" + std::to_string(o.tile_m) + ";\n" +
        "constant constexpr int TN=" + std::to_string(o.tile_n) + ";\n" +
        "constant constexpr int SG=" + std::to_string(o.simdgroups) + ";\n" +
        "#define EPILOGUE " + std::to_string(o.epilogue) + "\n" +
        "#define GROUPED " + std::to_string(o.grouped ? 1 : 0) + "\n" +
        "#define SPLITK " + std::to_string(o.splitk ? 1 : 0) + "\n" + R"MSL(
struct AffineGemm {
    uint rows, columns, inner;          // capacity rows (grid), N, K
    uint x_stride, y_stride, experts;   // row strides (elements); experts for GROUPED
    uint limit, per_row;                // M below `limit` belongs to the row-vector kernel; GROUPED: rows
                                        // (expert assignments) per chunk row
    ulong w, s, b, w2, s2, b2;          // byte offsets of codes/scales/biases (second set: up for SwiGLU)
};
// MLX's split-K partition count for an M x N x K product (32 x 32 tiles, ~512 threadgroups, divisor of K/64).
inline uint splitk_parts(uint m,uint n,uint k) {
    uint tiles=((n+31u)/32u)*((m+31u)/32u);
    uint parts=max(1u,512u/tiles);
    parts=min(parts,k/64u);
    while(parts>1u && k%(parts*64u)!=0u)--parts;
    return parts;
}
inline float bits_f(uint b){return as_type<float>(b<<16);}
inline float rbf(float v){return as_type<float>(uint(as_type<ushort>(bfloat(v)))<<16);}
// Dequantize rows [n0, n0+TN) x inputs [k0, k0+64) of one matrix into tile[TN][64] (BF16).
inline void dequant_tile(const device uchar* codes,const device ushort* scales,const device ushort* biases,
                         uint k,uint n0,uint rows_valid,uint k0,threadgroup bfloat* tile,uint t,uint threads) {
    for(uint task=t;task<uint(TN)*4u;task+=threads) {
        uint r=task/4u,part=task%4u;               // 16 inputs per task
        threadgroup bfloat* out=tile+r*64u+part*16u;
        if(r>=rows_valid){for(uint j=0;j<16;++j)out[j]=bfloat(0.0f);continue;}
        uint row=n0+r;
        float s=bits_f(scales[size_t(row)*(k/64u)+k0/64u]),b=bits_f(biases[size_t(row)*(k/64u)+k0/64u]);
        uint2 q=*((const device uint2*)(codes+size_t(row)*(k/2u)+k0/2u+part*8u));
        for(uint j=0;j<8;++j) {
            uint word=j<4?q.x:q.y,shift=(j%4)*8;
            uint byte=(word>>shift)&0xffu;
            out[j*2]=bfloat(metal::fma(s,float(byte&15u),b));
            out[j*2+1]=bfloat(metal::fma(s,float(byte>>4),b));
        }
    }
}
inline float swiglu_f(float g,float u) {
    float a=as_type<float>(as_type<uint>(g)&0x7fffffffu);
    float e=rbf(metal::fast::exp(a));
    float tail=rbf(1.0f/rbf(1.0f+e));
    float sig=g<0?tail:rbf(1.0f-tail);
    return rbf(rbf(g*sig)*u);
}
#if GROUPED
// Expert-grouped product: one threadgroup per (64-column tile, block of up to 64 rows of one expert), four
// SIMD groups. Each 64-input weight slice is dequantized once by all threads; SIMD group g multiplies it
// into the block's 16-row sub-tile g when that sub-tile holds rows (TensorOps 16 x 64 x 64 per SIMD
// group), so sparsely filled expert blocks do not pay for empty rows.
kernel void gemm_w4a16_affine(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant AffineGemm& p [[buffer(3)]],
    const device uint* starts [[buffer(4)]],const device uint* tiles [[buffer(5)]],
    const device uint* state [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint threads=uint(SG)*32u;
    uint n0=group.x*uint(TN);
    if(n0>=p.columns)return;
    if(state[1]*p.per_row<1024u)return;
    uint e_lo=0,e_hi=p.experts;
    if(group.y>=tiles[p.experts])return;
    while(e_lo<e_hi){uint mid=(e_lo+e_hi)/2;if(tiles[mid+1]<=group.y)e_lo=mid+1;else e_hi=mid;}
    const uint expert=e_lo;
    const uint row0=starts[expert]+(group.y-tiles[expert])*uint(TM);
    const uint rows=min(uint(TM),starts[expert+1]-row0);
    const bool mine=16u*sg<rows;
    const uint my_rows=mine?min(16u,rows-16u*sg):1u;
    const uint my_row0=row0+(mine?16u*sg:0u);
    const uint cols=min(uint(TN),p.columns-n0);
    const ulong expert_codes=ulong(expert)*p.columns*(p.inner/2u);
    const ulong expert_scales=ulong(expert)*p.columns*(p.inner/64u)*2u;
    const device uchar* codes=weights+p.w+expert_codes;
    const device ushort* scales=(const device ushort*)(weights+p.s+expert_scales);
    const device ushort* biases=(const device ushort*)(weights+p.b+expert_scales);
    threadgroup bfloat tile[TN*64];
#if EPILOGUE==2
    const device uchar* codes2=weights+p.w2+expert_codes;
    const device ushort* scales2=(const device ushort*)(weights+p.s2+expert_scales);
    const device ushort* biases2=(const device ushort*)(weights+p.b2+expert_scales);
    threadgroup bfloat tile2[TN*64];
#endif
    using E=dextents<int,2>;
    tensor<threadgroup bfloat,E,tensor_inline> right(tile,E(64,TN),array<int,2>{1,64});
#if EPILOGUE==2
    tensor<threadgroup bfloat,E,tensor_inline> right2(tile2,E(64,TN),array<int,2>{1,64});
#endif
    tensor<device bfloat,E,tensor_inline> left((device bfloat*)x+size_t(my_row0)*p.x_stride,E(int(p.inner),int(my_rows)),
                                             array<int,2>{1,int(p.x_stride)});
    constexpr auto desc=matmul2d_descriptor(16,TN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
#if EPILOGUE==2
    auto up=op.get_destination_cooperative_tensor<decltype(left),decltype(right2),float>();
    for(uint i=0;i<up.get_capacity();++i)if(up.is_valid_element(i))up[i]=0.0f;
#endif
    for(uint k0=0;k0<p.inner;k0+=64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dequant_tile(codes,scales,biases,p.inner,n0,cols,k0,tile,t,threads);
#if EPILOGUE==2
        dequant_tile(codes2,scales2,biases2,p.inner,n0,cols,k0,tile2,t,threads);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(mine) {
            auto l=left.slice(int(k0),0);
            op.run(l,right,acc);
#if EPILOGUE==2
            op.run(l,right2,up);
#endif
        }
    }
    if(!mine)return;
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]>=int(my_rows) || c[0]>=int(cols))continue;
#if EPILOGUE==2
        float value=swiglu_f(rbf(acc[i]),rbf(up[i]));
#else
        float value=rbf(acc[i]);
#endif
        y[size_t(my_row0+c[1])*p.y_stride+n0+c[0]]=bfloat(value);
    }
}
#else
kernel void gemm_w4a16_affine(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* y [[buffer(2)]],constant AffineGemm& p [[buffer(3)]],
    const device uint* starts [[buffer(4)]],const device uint* tiles [[buffer(5)]],
    const device uint* state [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    const uint threads=uint(SG)*32u;
    uint n0=group.x*uint(TN);
    if(n0>=p.columns)return;
#if GROUPED
    if(state[1]*p.per_row<1024u)return;
#else
    const uint m_rows=state[1];
    if(m_rows<p.limit)return;
    const uint parts=splitk_parts(m_rows,p.columns,p.inner);
#if SPLITK
    if(parts<=1u)return;
#else
    if(parts>1u)return;
#endif
#endif
    uint row0,rows_end;
    ulong expert_codes=0,expert_scales=0;
#if GROUPED
    uint e_lo=0,e_hi=p.experts;
    if(group.y>=tiles[p.experts])return;
    while(e_lo<e_hi){uint mid=(e_lo+e_hi)/2;if(tiles[mid+1]<=group.y)e_lo=mid+1;else e_hi=mid;}
    uint expert=e_lo;
    row0=starts[expert]+(group.y-tiles[expert])*uint(TM);
    rows_end=starts[expert+1];
    expert_codes=ulong(expert)*p.columns*(p.inner/2u);
    expert_scales=ulong(expert)*p.columns*(p.inner/64u)*2u;
#else
    row0=group.y*uint(TM);
    rows_end=m_rows;
    if(row0>=rows_end)return;
#endif
    uint rows=min(uint(TM),rows_end-row0);
    uint cols=min(uint(TN),p.columns-n0);
    const device uchar* codes=weights+p.w+expert_codes;
    const device ushort* scales=(const device ushort*)(weights+p.s+expert_scales);
    const device ushort* biases=(const device ushort*)(weights+p.b+expert_scales);
    threadgroup bfloat tile[TN*64];
#if EPILOGUE==2
    const device uchar* codes2=weights+p.w2+expert_codes;
    const device ushort* scales2=(const device ushort*)(weights+p.s2+expert_scales);
    const device ushort* biases2=(const device ushort*)(weights+p.b2+expert_scales);
    threadgroup bfloat tile2[TN*64];
#endif
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> left((device bfloat*)x+size_t(row0)*p.x_stride,E(int(p.inner),int(rows)),
                                             array<int,2>{1,int(p.x_stride)});
    tensor<threadgroup bfloat,E,tensor_inline> right(tile,E(64,TN),array<int,2>{1,64});
    constexpr auto desc=matmul2d_descriptor(TM,TN,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
#if EPILOGUE==2
    tensor<threadgroup bfloat,E,tensor_inline> right2(tile2,E(64,TN),array<int,2>{1,64});
    auto acc2=op.get_destination_cooperative_tensor<decltype(left),decltype(right2),float>();
    for(uint i=0;i<acc2.get_capacity();++i)if(acc2.is_valid_element(i))acc2[i]=0.0f;
#endif
#if SPLITK
    // Partitions in the order 0, 8, 1, 9, ... for k = 16 (pairs of one interleaved accumulator).
    constexpr uint CAP=uint(TM*TN)/(uint(SG)*32u);
    float total[CAP],pending[CAP];
#if EPILOGUE==2
    float total2[CAP],pending2[CAP];
#endif
    const uint span=p.inner/parts;
    for(uint q=0;q<parts;++q) {
        const uint part=parts==16u?((q&1u)?(q>>1)+8u:(q>>1)):q;
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
#if EPILOGUE==2
        for(uint i=0;i<acc2.get_capacity();++i)if(acc2.is_valid_element(i))acc2[i]=0.0f;
#endif
        for(uint k0=part*span;k0<(part+1u)*span;k0+=64) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dequant_tile(codes,scales,biases,p.inner,n0,cols,k0,tile,t,threads);
#if EPILOGUE==2
            dequant_tile(codes2,scales2,biases2,p.inner,n0,cols,k0,tile2,t,threads);
#endif
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto l=left.slice(int(k0),0);
            op.run(l,right,acc);
#if EPILOGUE==2
            op.run(l,right2,acc2);
#endif
        }
        for(uint i=0;i<acc.get_capacity() && i<CAP;++i) {
            if(!acc.is_valid_element(i))continue;
            float v=rbf(acc[i]);
            if(parts==32u)total[i]=(q?total[i]:0.0f)+v;
            else if(parts==16u){if((q&1u)==0u)pending[i]=v;else{float a=rbf(pending[i]+v);total[i]=q==1u?a:rbf(total[i]+a);}}
            else total[i]=q?rbf(total[i]+v):v;
#if EPILOGUE==2
            float v2=rbf(acc2[i]);
            if(parts==32u)total2[i]=(q?total2[i]:0.0f)+v2;
            else if(parts==16u){if((q&1u)==0u)pending2[i]=v2;else{float a=rbf(pending2[i]+v2);total2[i]=q==1u?a:rbf(total2[i]+a);}}
            else total2[i]=q?rbf(total2[i]+v2):v2;
#endif
        }
    }
    for(uint i=0;i<acc.get_capacity() && i<CAP;++i)if(acc.is_valid_element(i)) {
        acc[i]=parts==32u?rbf(total[i]):total[i];
#if EPILOGUE==2
        acc2[i]=parts==32u?rbf(total2[i]):total2[i];
#endif
    }
#else
    for(uint k0=0;k0<p.inner;k0+=64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dequant_tile(codes,scales,biases,p.inner,n0,cols,k0,tile,t,threads);
#if EPILOGUE==2
        dequant_tile(codes2,scales2,biases2,p.inner,n0,cols,k0,tile2,t,threads);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto l=left.slice(int(k0),0);
        op.run(l,right,acc);
#if EPILOGUE==2
        op.run(l,right2,acc2);
#endif
    }
#endif
    for(uint i=0;i<acc.get_capacity();++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        uint col=n0+c[0],row=row0+c[1];
        if(c[1]>=int(rows) || c[0]>=int(cols))continue;
        float value=rbf(acc[i]);
#if EPILOGUE==1
        value=rbf(value+float(y[size_t(row)*p.y_stride+col]));
#elif EPILOGUE==2
        value=swiglu_f(value,rbf(acc2[i]));
#endif
        y[size_t(row)*p.y_stride+col]=bfloat(value);
    }
}
#endif
)MSL";
}
}  // namespace flashrt::apple::shaders
