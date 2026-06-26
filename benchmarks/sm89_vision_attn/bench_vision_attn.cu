// Standalone SM89 vision-attention FA2 tile micro-bench.
//
// Qwen3-VL vision tower runs one non-causal full-attention call over all
// S=6256 patches per block, bf16, 16 heads. PR111 dispatches head_dim 64
// (2B) and 72 (8B) into the FA2 hdim buckets (64 if built, else 96). This
// harness compiles a small candidate table of (BlockM,BlockN,NWarps) tiles
// for ONE head-dim bucket so we can pick the best tile for the large-S
// vision regime in seconds, instead of a 9-minute full FA2 rebuild.
//
// It links ONLY the FA2 forward template (no split-KV, no causal), so the
// expensive ptxas hogs never compile. Reuses the vendored kernel sources
// verbatim via the launch template.
//
// Build:   ./build.sh <HDIM>          (HDIM = 64 or 96)
// Run:     ./build/bench_vision_attn_hdim<HDIM> [--s 6256] [--heads 16] [--d <real>]
#include "namespace_config.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef BENCH_HDIM
#define BENCH_HDIM 64
#endif

using FLASH_NAMESPACE::Flash_fwd_params;
using FLASH_NAMESPACE::run_flash_fwd;
// Flash_fwd_kernel_traits is declared in the GLOBAL namespace (kernel_traits.h
// has no namespace wrap), so it is referenced unqualified below.

namespace {

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} } while(0)

static inline int round_up(int a, int b) { return (a + b - 1) / b * b; }

// One launch candidate: a specific (BlockM, BlockN, NWarps) tile.
struct Cand { const char* name; void (*run)(Flash_fwd_params&, cudaStream_t); };

template <int BM, int BN, int W>
void run_tile(Flash_fwd_params& p, cudaStream_t s) {
    // Non-causal, no dropout — the vision regime.
    run_flash_fwd<Flash_fwd_kernel_traits<BENCH_HDIM, BM, BN, W, false, false,
                  cutlass::bfloat16_t>, /*Is_dropout=*/false, /*Is_causal=*/false>(p, s);
}

// Candidate tile table (per head-dim bucket). Kept small: each candidate
// expands the full FA2 switch matrix (~64 kernel instantiations), so more
// than ~3 makes the single-TU compile explode. These three answer the open
// question for the large-S non-causal vision regime: is a bigger N tile
// (128x128) better than 128x64, and confirm 128x32 is the regression.
std::vector<Cand> candidates() {
    return {
        {"128x128x4", &run_tile<128,128,4>},
        {"128x64x4",  &run_tile<128, 64,4>},
        {"128x32x4",  &run_tile<128, 32,4>},
    };
}

}  // namespace

int main(int argc, char** argv) {
    int S = 6256, heads = 16, d = BENCH_HDIM, iters = 50, warmup = 10;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--s") && i+1<argc) S = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1<argc) heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d") && i+1<argc) d = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1<argc) iters = atoi(argv[++i]);
    }
    // d is the *real* head_dim (e.g. 72 for 8B); the kernel head-dim bucket
    // is BENCH_HDIM (compile-time). FA2 pads d<=BENCH_HDIM internally.
    if (d > BENCH_HDIM) { fprintf(stderr, "d=%d > bucket %d\n", d, BENCH_HDIM); return 1; }

    printf("vision-attn micro-bench: hdim_bucket=%d real_d=%d S=%d heads=%d iters=%d\n",
           BENCH_HDIM, d, S, heads, iters);

    const int H = heads * BENCH_HDIM;  // packed Q/K/V width uses bucket dim
    // NOTE: faithful ONLY when real_d == BENCH_HDIM (e.g. 2B vision: d=64,
    // bucket=64). For 8B vision (d=72, bucket=96) the real model packs with
    // head_stride=72 (hidden=16*72), not 96, so this harness's hd96 numbers
    // are NOT layout-faithful — trust end-to-end nsys for the hd96 tile.
    const size_t nelem = (size_t)S * H;
    __nv_bfloat16 *q,*k,*v,*o; float* lse;
    CUDA_CHECK(cudaMalloc(&q, nelem*sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&k, nelem*sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&v, nelem*sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&o, nelem*sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&lse, (size_t)heads*round_up(S,128)*sizeof(float)));
    // Deterministic fill.
    {
        std::vector<__nv_bfloat16> h(nelem);
        for (size_t i=0;i<nelem;++i) h[i] = __float2bfloat16(((i*1103515245u+12345u)>>16 & 0xff)/255.0f - 0.5f);
        CUDA_CHECK(cudaMemcpy(q,h.data(),nelem*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k,h.data(),nelem*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(v,h.data(),nelem*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    }

    Flash_fwd_params p;
    memset(&p, 0, sizeof(p));
    p.is_bf16 = true;
    p.q_ptr=q; p.k_ptr=k; p.v_ptr=v; p.o_ptr=o; p.softmax_lse_ptr=lse;
    p.q_row_stride=H; p.k_row_stride=H; p.v_row_stride=H; p.o_row_stride=H;
    p.q_head_stride=BENCH_HDIM; p.k_head_stride=BENCH_HDIM; p.v_head_stride=BENCH_HDIM; p.o_head_stride=BENCH_HDIM;
    p.q_batch_stride=nelem; p.k_batch_stride=nelem; p.v_batch_stride=nelem; p.o_batch_stride=nelem;
    p.b=1; p.h=heads; p.h_k=heads; p.h_h_k_ratio=1;
    p.seqlen_q=S; p.seqlen_k=S; p.seqlen_q_rounded=round_up(S,128); p.seqlen_k_rounded=round_up(S,128);
    p.d=d; p.d_rounded=(d+31)&~31;
    float scale = 1.0f/sqrtf((float)d);
    p.scale_softmax=scale; p.scale_softmax_log2=scale*float(M_LOG2E);
    p.softcap=0.0f;
    p.p_dropout=1.0f; p.p_dropout_in_uint8_t=255; p.rp_dropout=1.0f; p.scale_softmax_rp_dropout=scale;
    p.is_causal=false; p.window_size_left=-1; p.window_size_right=-1;
    p.cu_seqlens_q=nullptr; p.cu_seqlens_k=nullptr; p.seqused_k=nullptr; p.p_ptr=nullptr;
    p.alibi_slopes_ptr=nullptr; p.is_seqlens_k_cumulative=true; p.rotary_dim=0;
    p.num_splits=1;

    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    for (auto& c : candidates()) {
        // correctness/runnability: one launch + sync, catch errors
        c.run(p, s); cudaError_t err = cudaStreamSynchronize(s);
        if (err != cudaSuccess) { printf("  %-12s FAILED: %s\n", c.name, cudaGetErrorString(err)); continue; }
        for (int i=0;i<warmup;++i) c.run(p,s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaEventRecord(e0,s));
        for (int i=0;i<iters;++i) c.run(p,s);
        CUDA_CHECK(cudaEventRecord(e1,s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
        printf("  %-12s  %.4f ms/iter\n", c.name, ms/iters);
    }
    return 0;
}
