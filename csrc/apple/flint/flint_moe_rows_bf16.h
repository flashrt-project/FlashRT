#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Mixture-of-experts row kernels for chunked prefill (256 experts, top-6 sigmoid routing with
// renormalization, BF16 activations): router logits + top-k, expert counting, scan and row placement
// (rows copied into contiguous per-expert blocks for grouped GEMMs), and the weighted combination with
// the shared expert and the residual stream. Rounding follows the resident decode step.
inline std::string flint_moe_rows_bf16_source() {
    return std::string(tensor_ops_preamble) + R"MSL(
inline float bf_f(bfloat x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float rbf(float v){return as_type<float>(uint(as_type<ushort>(bfloat(v)))<<16);}
constant constexpr uint EXPERTS=256u,TOPK=6u,HIDDEN=2048u;

struct Router { ulong weight, bias; };
// Router for 16 rows per threadgroup: logits = x . R^T in FP32 with the accumulation of MLX's split-K GEMM
// for a chunk of `rows` rows (M <= 2048, N = 256, K = 2048): P = min(max(2, pow2ceil(16 / ceil(M/32))), 32)
// partitions of K, each accumulated by TensorOps over 64-input slices (the 8-input MMA chain), partition
// sums added in order ((0 + C_0) + C_1) + ...; logit rounded to BF16, sigmoid, selection by sigmoid + bias
// (ties: lowest expert), weights = chosen sigmoid / sum (rank order).
kernel void flint_router_topk_rows_bf16(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device uint* ids [[buffer(2)]],device float* gates [[buffer(3)]],constant Router& p [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint rows=state[1],row0=group*16u;
    if(row0>=rows)return;
    const uint nr=min(16u,rows-row0);
    threadgroup float logits[16*EXPERTS];
    const uint ratio=16u/((rows+31u)/32u);
    uint parts=2u;while(parts<ratio)parts*=2u;parts=min(parts,32u);
    const uint span=HIDDEN/parts;
    using E=dextents<int,2>;
    tensor<device bfloat,E,tensor_inline> xt((device bfloat*)x+size_t(row0)*HIDDEN,E(int(HIDDEN),int(nr)),array<int,2>{1,int(HIDDEN)});
    tensor<device bfloat,E,tensor_inline> rt((device bfloat*)(weights+p.weight),E(int(HIDDEN),int(EXPERTS)),array<int,2>{1,int(HIDDEN)});
    constexpr auto d=matmul2d_descriptor(16,EXPERTS,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d,execution_simdgroups<4>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(xt),decltype(rt),float>();
    constexpr uint CAP=16u*EXPERTS/128u;
    float total[CAP];
    for(uint i=0;i<CAP;++i)total[i]=0.0f;
    for(uint part=0;part<parts;++part) {
        for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
        for(uint k0=part*span;k0<(part+1u)*span;k0+=64) {
            auto xs=xt.slice(int(k0),0);
            auto ws=rt.slice(int(k0),0);
            op.run(xs,ws,acc);
        }
        for(uint i=0;i<acc.get_capacity() && i<CAP;++i)if(acc.is_valid_element(i))total[i]=total[i]+acc[i];
    }
    for(uint i=0;i<acc.get_capacity() && i<CAP;++i) {
        if(!acc.is_valid_element(i))continue;
        auto c=acc.get_multidimensional_index(i);
        if(c[1]<int(nr))logits[c[1]*EXPERTS+c[0]]=total[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device float* bias=(const device float*)(weights+p.bias);
    for(uint r=sg;r<nr;r+=4) {
        float prob[8],choice[8];
        for(uint j=0;j<8;++j) {
            uint e=lane+32*j;
            float a=rbf(logits[r*EXPERTS+e]);
            float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(a)));
            prob[j]=a<0?tail:1.0f-tail;
            choice[j]=prob[j]+bias[e];
        }
        float chosen=0.0f;uint my_id=0;
        for(uint rank=0;rank<TOPK;++rank) {
            float best=-3.402823466e+38f;uint idx=0xffffffffu;
            for(uint j=0;j<8;++j){uint e=lane+32*j;if(choice[j]>best||(choice[j]==best&&e<idx)){best=choice[j];idx=e;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            float value=simd_shuffle(prob[selected/32],ushort(selected%32));
            if(lane==rank){chosen=value;my_id=selected;}
            for(uint j=0;j<8;++j)if(lane+32*j==selected)choice[j]=-3.402823466e+38f;
        }
        float total=0.0f;
        for(uint j=0;j<TOPK;++j)total=simd_shuffle(chosen,ushort(j))+total;
        total+=1.0e-20f;
        if(lane<TOPK){uint row=row0+r;ids[row*TOPK+lane]=my_id;gates[row*TOPK+lane]=chosen/total;}
    }
}

// counts[e] += assignments; grid = rows*6 threads.
kernel void flint_expert_count(const device uint* ids [[buffer(0)]],device atomic_uint* counts [[buffer(1)]],
    const device uint* state [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=state[1]*TOPK)return;
    atomic_fetch_add_explicit(counts+ids[i],1u,memory_order_relaxed);
}
struct ExpertScan { uint tile_rows, pad; };
// One threadgroup of 256 threads: starts (exclusive scan of counts, starts[256] = total), tile table
// (exclusive scan of ceil(count/tile_rows)), cursors = starts; counts reset to zero for the next layer.
kernel void flint_expert_scan(device uint* counts [[buffer(0)]],device uint* starts [[buffer(1)]],
    device uint* tiles [[buffer(2)]],device atomic_uint* cursors [[buffer(3)]],constant ExpertScan& p [[buffer(4)]],
    uint t [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[8],tsum[8];
    uint c=counts[t],tc=(c+p.tile_rows-1)/p.tile_rows;
    uint ex=simd_prefix_exclusive_sum(c),tex=simd_prefix_exclusive_sum(tc);
    if(lane==31){sums[sg]=ex+c;tsum[sg]=tex+tc;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base=0,tbase=0;for(uint i=0;i<sg;++i){base+=sums[i];tbase+=tsum[i];}
    starts[t]=base+ex;tiles[t]=tbase+tex;
    atomic_store_explicit(cursors+t,base+ex,memory_order_relaxed);
    if(t==255){starts[256]=base+ex+c;tiles[256]=tbase+tex+tc;}
    counts[t]=0;
}
// One threadgroup per assignment (row, rank): claim a slot in the expert's block and copy the row.
kernel void flint_expert_place_bf16(const device bfloat* x [[buffer(0)]],const device uint* ids [[buffer(1)]],
    device atomic_uint* cursors [[buffer(2)]],device uint* slots [[buffer(3)]],device bfloat* xp [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint a [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    if(a>=state[1]*TOPK)return;
    threadgroup uint slot[1];
    if(t==0){slot[0]=atomic_fetch_add_explicit(cursors+ids[a],1u,memory_order_relaxed);slots[a]=slot[0];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint4* src=(const device uint4*)(x+size_t(a/TOPK)*HIDDEN);
    device uint4* dst=(device uint4*)(xp+size_t(slot[0])*HIDDEN);
    dst[t]=src[t];   // 256 threads x 8 BF16 = 2048
}
// h = bf16(h + bf16(bf16(sum_r bf16(y_r * bf16(g_r))) + shared)), sum in rank order with per-add rounding.
kernel void flint_expert_combine_rows_bf16(device bfloat* h [[buffer(0)]],const device bfloat* yp [[buffer(1)]],
    const device bfloat* shared [[buffer(2)]],const device uint* slots [[buffer(3)]],const device float* gates [[buffer(4)]],
    const device uint* state [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    if(row>=state[1])return;
    uint slot[TOPK];float g[TOPK];
    for(uint r=0;r<TOPK;++r){slot[r]=slots[row*TOPK+r];g[r]=rbf(gates[row*TOPK+r]);}
    for(uint k=0;k<8;++k) {
        uint i=t*8u+k;
        float s=rbf(bf_f(yp[size_t(slot[0])*HIDDEN+i])*g[0])+0.0f;
        for(uint r=1;r<TOPK;++r)s=rbf((rbf(bf_f(yp[size_t(slot[r])*HIDDEN+i])*g[r])+0.0f)+s);
        float mlp=rbf(rbf(s)+bf_f(shared[size_t(row)*HIDDEN+i]));
        h[size_t(row)*HIDDEN+i]=bfloat(rbf(bf_f(h[size_t(row)*HIDDEN+i])+mlp));
    }
}
)MSL";
}
}  // namespace flashrt::apple::shaders
