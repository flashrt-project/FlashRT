#pragma once
#include "common/source.h"
#include <string>

namespace flashrt::apple::shaders {
// Row kernels of a chunked decoder prefill (BF16 activations, FP32 math, one BF16 rounding per
// operation, matching the resident decode step): affine-8 embedding rows, weighted RMSNorm rows,
// per-head Q/K RMSNorm + NeoX RoPE with K/V publication, and sliding-ring <-> linear KV copies.
// Chunk-variable values (first position, valid rows) come from a device `state` buffer [pos0, rows];
// grids cover the chunk capacity and surplus threadgroups return.
inline std::string flint_prefill_rows_bf16_source() {
    return std::string(metal_preamble) + R"MSL(
inline float bits_f(uint b){return as_type<float>(b<<16);}
inline float bf_f(bfloat x){return as_type<float>(uint(as_type<ushort>(x))<<16);}
inline float rbf(float v){return as_type<float>(uint(as_type<ushort>(bfloat(v)))<<16);}

struct EmbedRows { uint hidden, pad; ulong w, s, b; };
// h[row, i] = bf16(fma(scale, code, bias)) for token tokens[pos0 + row]; grid (hidden/256, rows), 256 threads.
kernel void flint_embed_affine8_rows_bf16(const device uint* tokens [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* h [[buffer(2)]],constant EmbedRows& p [[buffer(3)]],const device uint* state [[buffer(4)]],
    uint2 g [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
    uint row=g.y,i=g.x*256u+t;
    if(row>=state[1] || i>=p.hidden)return;
    uint token=tokens[state[0]+row];
    const device ushort* s=(const device ushort*)(weights+p.s);
    const device ushort* b=(const device ushort*)(weights+p.b);
    uint code=(weights+p.w)[size_t(token)*p.hidden+i];
    uint gi=token*(p.hidden/64u)+i/64u;
    h[size_t(row)*p.hidden+i]=bfloat(metal::fma(bits_f(s[gi]),float(code),bits_f(b[gi])));
}

struct RmsRows { uint hidden, last; ulong w; float eps, pad2; };
// out = bf16(w * bf16(x * rsqrt(mean(x^2) + eps))); one threadgroup of 256 threads per row (hidden 2048).
// `last`: only the chunk's last row (state[1] - 1), written to output row 0 (one threadgroup).
// The sum of squares follows MLX's single-row RMSNorm reduction: virtual threads of 4 contiguous inputs
// (sequential FMA accumulation), a SIMD sum per 32 virtual threads, then a SIMD sum of the 16 partials.
// Thread (s, l) hosts virtual threads 32s + l and 256 + 32s + l.
kernel void flint_rms_rows_bf16(const device bfloat* x [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* out [[buffer(2)]],constant RmsRows& p [[buffer(3)]],const device uint* state [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    if(row>=state[1] || (p.last && row>0))return;
    const uint src=p.last?state[1]-1u:row,dst=p.last?0u:row;
    threadgroup float partial[16];
    const device bfloat* r=x+size_t(src)*p.hidden;
    const device bfloat* w=(const device bfloat*)(weights+p.w);
    float acc[2];
    for(uint h=0;h<2;++h) {
        uint base=(h*256u+t)*4u;
        float a=0.0f;
        for(uint i=0;i<4;++i){float v=bf_f(r[base+i]);a=metal::fma(v,v,a);}
        acc[h]=simd_sum(a);
    }
    if(lane==0){partial[sg]=acc[0];partial[8+sg]=acc[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=simd_sum(lane<16?partial[lane]:0.0f);
    float inverse=metal::precise::rsqrt(total/float(p.hidden)+p.eps);
    for(uint h=0;h<2;++h) {
        uint base=(h*256u+t)*4u;
        for(uint i=0;i<4;++i)out[size_t(dst)*p.hidden+base+i]=bfloat(rbf(bf_f(w[base+i])*rbf(bf_f(r[base+i])*inverse)));
    }
}

struct QkRows {
    uint q_heads, kv_heads, sliding, cap;   // cap: KV slots per head (global: max_seq; sliding: linear scratch rows)
    uint scratch, pad0, pad1, pad2;         // sliding: scratch row of position pos0 (== window)
    ulong q_norm, k_norm, kv;               // weight byte offsets; KV element offset of this layer
    float eps, logbase, pad3, pad4;
};
// One threadgroup per row: RMSNorm per head (Q and K), NeoX RoPE on sliding layers, Q written KV-head-major
// [kv_head][row][group][128]; K/V written to the layer's cache (global, slot = position) or to the linear
// scratch (sliding, slot = scratch + row). Layout of a KV region: K [kv_heads][cap][128] then V.
kernel void flint_qk_norm_rope_kv_rows_bf16(const device bfloat* qkv [[buffer(0)]],const device uchar* weights [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],device bfloat* kv [[buffer(3)]],constant QkRows& p [[buffer(4)]],
    const device uint* state [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint rows=state[1];
    if(row>=rows)return;
    uint position=state[0]+row;
    uint width=(p.q_heads+2*p.kv_heads)*128u;
    uint group_size=p.q_heads/p.kv_heads;
    const device bfloat* src=qkv+size_t(row)*width;
    uint slot=p.sliding?p.scratch+row:position;
    for(uint head=sg;head<p.q_heads+2*p.kv_heads;head+=8) {
        const device bfloat* x=src+head*128u;
        bool is_v=head>=p.q_heads+p.kv_heads;
        float v[4];for(uint j=0;j<4;++j)v[j]=bf_f(x[lane*4+j]);
        float out[4];
        if(is_v){for(uint j=0;j<4;++j)out[j]=v[j];}
        else {
            const device bfloat* w=(const device bfloat*)(weights+(head<p.q_heads?p.q_norm:p.k_norm));
            float sq=0.0f;for(uint j=0;j<4;++j)sq+=v[j]*v[j];
            float inverse=metal::precise::rsqrt(simd_sum(sq)/128.0f+p.eps);
            if(p.sliding) {
                // lane owns dims lane*4..+3 for the norm but RoPE pairs (d, d+64): exchange through shuffles
                float n[4];for(uint j=0;j<4;++j)n[j]=rbf(bf_f(w[lane*4+j])*rbf(v[j]*inverse));
                // gather partner values: dim d<64 pairs with d+64 held by lane+16
                for(uint j=0;j<4;++j) {
                    uint d=lane*4+j;
                    float partner=simd_shuffle_xor(n[j],16);
                    uint pair=d<64?d:d-64;
                    float theta=float(position)*metal::exp2(-float(pair)/64.0f*p.logbase);
                    float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
                    out[j]=d<64?rbf(n[j]*c-partner*s):rbf(partner*s+n[j]*c);
                }
            } else {
                for(uint j=0;j<4;++j)out[j]=rbf(bf_f(w[lane*4+j])*rbf(v[j]*inverse));
            }
        }
        device bfloat* dst;
        if(head<p.q_heads) {
            uint g=head/group_size,j=head%group_size;
            dst=q_out+((size_t(g)*rows+row)*group_size+j)*128u;
        } else if(!is_v) {
            uint kh=head-p.q_heads;
            dst=kv+p.kv+(size_t(kh)*p.cap+slot)*128u;
        } else {
            uint kh=head-p.q_heads-p.kv_heads;
            dst=kv+p.kv+size_t(p.kv_heads)*p.cap*128u+(size_t(kh)*p.cap+slot)*128u;
        }
        for(uint j=0;j<4;++j)dst[lane*4+j]=bfloat(out[j]);
    }
}

struct RingCopy { uint kv_heads, ring, scratch_cap, window; ulong ring_kv, scratch_kv; };
// Linear scratch of a sliding layer: row s holds position pos0 - window + s (the `window` positions before
// the chunk, then the chunk), the key layout of MLX's sliding cache for a continued prefill. Ring slot of
// position p under layout `initial` (state[2] before the chunk, state[3] after it):
// initial >= ring ? (p - initial) mod ring : p mod ring, the slot rule of the resident decode step.
// to_linear: positions [pos0-window, pos0) from the ring into scratch row p - pos0 + window.
// to_ring:   positions [max(0, end-ring), end) from scratch back into the ring (end = pos0 + rows).
// grid: x = 32 lanes of 4 dims, y = row, z = 2*kv_heads (K heads then V heads).
inline uint ring_slot(long position,uint initial,uint ring) {
    uint p=uint(position)%ring;
    return initial>=ring?(p+ring-initial%ring)%ring:p;
}
kernel void flint_kv_ring_copy_bf16(device bfloat* kv [[buffer(0)]],device bfloat* scratch [[buffer(1)]],
    constant RingCopy& p [[buffer(2)]],const device uint* state [[buffer(3)]],constant uint& direction [[buffer(4)]],
    uint3 g [[thread_position_in_grid]]) {
    uint pos0=state[0],rows=state[1];
    uint lane=g.x,r=g.y,which=g.z;           // which: [0,kv_heads) K, [kv_heads, 2kv_heads) V
    uint kh=which%p.kv_heads;bool value=which>=p.kv_heads;
    long position;
    if(direction==0) {
        if(r>=p.window)return;
        position=long(pos0)-long(p.window)+long(r);
        if(position<0)return;
    } else {
        uint end=pos0+rows,first=end>p.ring?end-p.ring:0u;
        if(r>=end-first)return;
        position=long(first+r);
    }
    uint srow=uint(position-long(pos0)+long(p.window));
    uint slot=ring_slot(position,direction==0?state[2]:state[3],p.ring);
    device bfloat* ring=kv+p.ring_kv+(value?size_t(p.kv_heads)*p.ring*128u:0)+(size_t(kh)*p.ring+slot)*128u;
    device bfloat* lin=scratch+p.scratch_kv+(value?size_t(p.kv_heads)*p.scratch_cap*128u:0)+(size_t(kh)*p.scratch_cap+srow)*128u;
    for(uint j=0;j<4;++j){uint d=lane*4+j;if(direction==0)lin[d]=ring[d];else ring[d]=lin[d];}
}
)MSL";
}
}  // namespace flashrt::apple::shaders
