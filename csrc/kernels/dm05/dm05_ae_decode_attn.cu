// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 action-expert decode attention (grouped-query, head_dim 256).
//
// One CTA per (key/value head, key split). The G query heads sharing a
// key/value head and all T query rows form R = G * T <= 32 MMA rows, so each
// key/value byte is read once per key/value head. Keys stream through a
// two-stage cp.async pipeline in 32-key chunks; S = Q K^T and O += P V run on
// m16n8k16 with the V B-fragments via ldmatrix.trans. Each CTA writes a
// partial (running max, running sum, O); the last CTA of a key/value head —
// picked by an atomic counter it resets after use, so graph replays need no
// memset — combines the splits and writes the final output.
//
// See dm05_ae_decode_attn.cuh for the interface and the workspace contract.

#include "dm05_ae_decode_attn.cuh"

#include "dm05_common.cuh"

#include <stdexcept>

namespace flash_rt {
namespace dm05 {
namespace {

constexpr int AHD = 256;   // head_dim
constexpr int KC = 32;     // keys per pipeline chunk
constexpr int QS = AHD + 8;
constexpr int PS = KC + 8;
constexpr int MAXR = 32;   // rows of the packed query tile

template <int R>
__global__ void __launch_bounds__(256)
ae_decode_attn_kernel(const __nv_bfloat16* __restrict__ q, long long q_ld,
                      const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                      long long kv_ld, __nv_bfloat16* __restrict__ out, long long o_ld,
                      float* __restrict__ ws_o, float* __restrict__ ws_ml, int* __restrict__ cnt,
                      int T, int G, int sk, int ns, float scale_log2) {
    extern __shared__ __align__(16) uint8_t smem[];
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem);            // [MAXR][QS]
    __nv_bfloat16* skk = sq + MAXR * QS;                                   // [2][KC][QS]
    __nv_bfloat16* svv = skk + 2 * KC * QS;                                // [2][KC][QS]
    float* ss = reinterpret_cast<float*>(svv + 2 * KC * QS);               // [MAXR][KC]
    __nv_bfloat16* sp = reinterpret_cast<__nv_bfloat16*>(ss + MAXR * KC);  // [MAXR][PS]
    float* sm = reinterpret_cast<float*>(sp + MAXR * PS);
    float* sl = sm + MAXR;
    float* sa = sl + MAXR;
    __shared__ int last;
    const int hk = blockIdx.x, split = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int kps = (((sk + ns - 1) / ns) + KC - 1) / KC * KC;
    const int jb = split * kps, je = min(sk, jb + kps);
    const int nch = je > jb ? (je - jb + KC - 1) / KC : 0;
    auto stage = [&](int ch, int buf) {
        const int j0 = jb + ch * KC;
        for (int i = tid; i < KC * (AHD / 8); i += 256) {
            const int c = i / (AHD / 8), c8 = i % (AHD / 8), j = j0 + c;
            const bool ok = j < je;
            const long long off = (long long)(ok ? j : jb) * kv_ld + (long long)hk * AHD + c8 * 8;
            dm05_cp16(skk + (buf * KC + c) * QS + c8 * 8, k + off, ok);
            dm05_cp16(svv + (buf * KC + c) * QS + c8 * 8, v + off, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (nch > 0) stage(0, 0);
    for (int i = tid; i < MAXR * (AHD / 8); i += 256) {
        const int r = i / (AHD / 8), c8 = i % (AHD / 8);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < R) {
            const int gg = r / T, tt = r % T;
            val = *reinterpret_cast<const uint4*>(q + (long long)tt * q_ld + (long long)(hk * G + gg) * AHD + c8 * 8);
        }
        *reinterpret_cast<uint4*>(sq + r * QS + c8 * 8) = val;
    }
    if (tid < MAXR) { sm[tid] = -INFINITY; sl[tid] = 0.f; }
    float oc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) oc[a][b][c] = 0.f;
    for (int ch = 0; ch < nch; ++ch) {
        const int buf = ch & 1, j0 = jb + ch * KC;
        if (ch + 1 < nch) {
            stage(ch + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        {  // S tile (mt, nt) per warp: 2 x 4 tiles of 16 x 8
            const int mt = warp >> 2, nt = warp & 3;
            float c4[4] = {0.f, 0.f, 0.f, 0.f};
            const __nv_bfloat16* kr = skk + (buf * KC + nt * 8 + g) * QS + 2 * t;
            const __nv_bfloat16* qa = sq + (mt * 16 + g) * QS + 2 * t;
#pragma unroll 4
            for (int k0 = 0; k0 < AHD; k0 += 16) {
                const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(qa + k0),
                                       *reinterpret_cast<const uint32_t*>(qa + 8 * QS + k0),
                                       *reinterpret_cast<const uint32_t*>(qa + k0 + 8),
                                       *reinterpret_cast<const uint32_t*>(qa + 8 * QS + k0 + 8)};
                dm05_mma_bf16_16816(c4, a, *reinterpret_cast<const uint32_t*>(kr + k0),
                                    *reinterpret_cast<const uint32_t*>(kr + k0 + 8));
            }
            float* row = ss + (mt * 16 + g) * KC + nt * 8 + 2 * t;
            row[0] = c4[0]; row[1] = c4[1]; row[8 * KC] = c4[2]; row[8 * KC + 1] = c4[3];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {   // online softmax, one warp per row, one key per lane
            const int r = warp + 8 * i;
            if (r >= R) {
                sp[r * PS + lane] = __float2bfloat16(0.f);
                if (lane == 0) sa[r] = 1.f;
                continue;
            }
            const float x = (j0 + lane < je) ? ss[r * KC + lane] * scale_log2 : -INFINITY;
            float mx = x;
#pragma unroll
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mo = sm[r], mn = fmaxf(mo, mx);
            const float pv = exp2f(x - mn);
            sp[r * PS + lane] = __float2bfloat16(pv);
            float sum = pv;
#pragma unroll
            for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) {
                const float a = exp2f(mo - mn);
                sl[r] = sl[r] * a + sum; sm[r] = mn; sa[r] = a;
            }
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const float a0 = sa[mt * 16 + g], a1 = sa[mt * 16 + g + 8];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                oc[mt][nt][0] *= a0; oc[mt][nt][1] *= a0; oc[mt][nt][2] *= a1; oc[mt][nt][3] *= a1;
            }
        }
#pragma unroll
        for (int k0 = 0; k0 < KC; k0 += 16) {
            uint32_t a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const __nv_bfloat16* pa = sp + (mt * 16 + g) * PS + k0 + 2 * t;
                a[mt][0] = *reinterpret_cast<const uint32_t*>(pa);
                a[mt][1] = *reinterpret_cast<const uint32_t*>(pa + 8 * PS);
                a[mt][2] = *reinterpret_cast<const uint32_t*>(pa + 8);
                a[mt][3] = *reinterpret_cast<const uint32_t*>(pa + 8 * PS + 8);
            }
#pragma unroll
            for (int np = 0; np < 2; ++np) {   // two n-tiles (16 dims) per ldmatrix.x4.trans
                const int d0 = warp * 32 + np * 16;
                const __nv_bfloat16* vp = svv + (buf * KC + k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * QS + d0 + (lane >> 4) * 8;
                uint32_t b[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"((unsigned)__cvta_generic_to_shared(vp)));
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    dm05_mma_bf16_16816(oc[mt][2 * np], a[mt], b[0], b[1]);
                    dm05_mma_bf16_16816(oc[mt][2 * np + 1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    float* po = ws_o + ((long long)(hk * ns + split) * MAXR) * AHD;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = mt * 16 + g + ((e & 2) ? 8 : 0), d = warp * 32 + nt * 8 + 2 * t + (e & 1);
                if (r >= R) continue;
                if (ns == 1) {
                    const int gg = r / T, tt = r % T;
                    out[(long long)tt * o_ld + (long long)(hk * G + gg) * AHD + d] = __float2bfloat16(oc[mt][nt][e] / sl[r]);
                } else {
                    po[r * AHD + d] = oc[mt][nt][e];
                }
            }
    if (ns == 1) return;
    if (tid < R) {
        ws_ml[((hk * ns + split) * MAXR + tid) * 2] = sm[tid];
        ws_ml[((hk * ns + split) * MAXR + tid) * 2 + 1] = sl[tid];
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(cnt + hk, 1) == ns - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    float* cm = ss;                        // [ns <= 32][MAXR] combine weights
    for (int i = tid; i < ns * R; i += 256) {
        const int sidx = i / R, r = i % R;
        cm[sidx * MAXR + r] = __ldcg(ws_ml + ((hk * ns + sidx) * MAXR + r) * 2);
    }
    __syncthreads();
    if (tid < R) {
        float M = -INFINITY;
        for (int sidx = 0; sidx < ns; ++sidx) M = fmaxf(M, cm[sidx * MAXR + tid]);
        float den = 0.f;
        for (int sidx = 0; sidx < ns; ++sidx) {
            const float w = exp2f(cm[sidx * MAXR + tid] - M);
            cm[sidx * MAXR + tid] = w;
            den += w * __ldcg(ws_ml + ((hk * ns + sidx) * MAXR + tid) * 2 + 1);
        }
        sl[tid] = 1.0f / den;
    }
    __syncthreads();
    for (int i = tid; i < R * (AHD / 4); i += 256) {
        const int r = i / (AHD / 4), d = (i % (AHD / 4)) * 4;
        float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int sidx = 0; sidx < ns; ++sidx) {
            const float w = cm[sidx * MAXR + r];
            const float4 vv = __ldcg(reinterpret_cast<const float4*>(ws_o + ((long long)(hk * ns + sidx) * MAXR + r) * AHD + d));
            num.x = fmaf(w, vv.x, num.x); num.y = fmaf(w, vv.y, num.y);
            num.z = fmaf(w, vv.z, num.z); num.w = fmaf(w, vv.w, num.w);
        }
        const float inv = sl[r];
        const int gg = r / T, tt = r % T;
        __nv_bfloat16* op = out + (long long)tt * o_ld + (long long)(hk * G + gg) * AHD + d;
        *reinterpret_cast<__nv_bfloat162*>(op) = __floats2bfloat162_rn(num.x * inv, num.y * inv);
        *reinterpret_cast<__nv_bfloat162*>(op + 2) = __floats2bfloat162_rn(num.z * inv, num.w * inv);
    }
    if (tid == 0) cnt[hk] = 0;
}

template <int R>
void launch_r(const __nv_bfloat16* q, long long q_ld,
              const __nv_bfloat16* k, const __nv_bfloat16* v, long long kv_ld,
              __nv_bfloat16* out, long long o_ld,
              float* ws_o, float* ws_ml, int* cnt,
              int T, int G, int NKV, int sk, int ns, float scale, cudaStream_t stream) {
    constexpr size_t sm = (size_t)MAXR * QS * 2 + 4 * (size_t)KC * QS * 2 + (size_t)MAXR * KC * 4 +
                          (size_t)MAXR * PS * 2 + 3 * MAXR * 4;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ae_decode_attn_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        attr = true;
    }
    ae_decode_attn_kernel<R><<<dim3(NKV, ns), 256, sm, stream>>>(
        q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, sk, ns,
        scale * 1.4426950408889634f);
}

}  // namespace

bool ae_decode_attn_supports_r(int r) {
    switch (r) {
        case 8: case 16: case 20: case 24: case 32:
            return true;
        default:
            return false;
    }
}

void ae_decode_attn(const __nv_bfloat16* q, long long q_ld,
                    const __nv_bfloat16* k, const __nv_bfloat16* v, long long kv_ld,
                    __nv_bfloat16* out, long long o_ld,
                    float* ws_o, float* ws_ml, int* cnt,
                    int T, int G, int NKV, int sk, int ns, float scale,
                    cudaStream_t stream) {
    if (T <= 0 || G <= 0 || NKV <= 0 || sk <= 0)
        throw std::runtime_error("dm05_ae_decode_attn: T, G, NKV and sk must be positive");
    if (ns < 1 || ns > 32)
        throw std::runtime_error("dm05_ae_decode_attn: ns must be in [1, 32]");
    const int R = G * T;
    if (R > MAXR)
        throw std::runtime_error("dm05_ae_decode_attn: G * T must not exceed 32");
    if (!ae_decode_attn_supports_r(R))
        throw std::runtime_error("dm05_ae_decode_attn: G * T is not an instantiated row count "
                                 "(supported: 8, 16, 20, 24, 32)");
    if (ns > 1 && (ws_o == nullptr || ws_ml == nullptr || cnt == nullptr))
        throw std::runtime_error("dm05_ae_decode_attn: ns > 1 needs the ws_o, ws_ml and cnt workspaces");
    if (q_ld < G * NKV * AHD || o_ld < G * NKV * AHD)
        throw std::runtime_error("dm05_ae_decode_attn: q/o row stride is smaller than G * NKV * 256");
    if (kv_ld < NKV * AHD)
        throw std::runtime_error("dm05_ae_decode_attn: k/v row stride is smaller than NKV * 256");

    switch (R) {
        case 8:  launch_r<8>(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, NKV, sk, ns, scale, stream); break;
        case 16: launch_r<16>(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, NKV, sk, ns, scale, stream); break;
        case 20: launch_r<20>(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, NKV, sk, ns, scale, stream); break;
        case 24: launch_r<24>(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, NKV, sk, ns, scale, stream); break;
        case 32: launch_r<32>(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt, T, G, NKV, sk, ns, scale, stream); break;
        default: break;   // unreachable: supports_r checked above
    }
}

}  // namespace dm05
}  // namespace flash_rt
