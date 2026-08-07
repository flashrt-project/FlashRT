# Hy-Embodied-0.5-VLA — Thor SM110 (Authoritative Document)

> **Production configuration: `flash_rt.load_model(ckpt, config="hyvla", framework="torch")` →
> `HyVLATorchFrontendThor(ckpt, use_fp8=True, use_fused=True[, use_autotune=True])`**
> **E2E 159.6 ms (`+use_autotune` 158.3 ms; HF/transformers eager ~930 ms, 5.8–5.9x),
> action cosine 0.999706 vs. HF/transformers eager (same fixed noise), CUDA-graph bitwise reproducible.**
>
> Framework-native: the runtime **does not import any upstream training code** and **does not use torch.compile/Inductor**.
> Composition: CUDA Graph (prefill + 10 denoise steps in a single graph) + dynamic per-tensor FP8 (calibration-free, graph-safe)
> + fused megakernel (rope+qknorm+kvwrite, KV pre-expanded for GQA) + memory-efficient SDPA
> + ViT trailing-stage history-frame drop + per-shape FP8 GEMM autotune.
>
> ```bash
> cmake -B build -S . -DGPU_ARCH=110 -DFLASHRT_ENABLE_HYVLA=ON
> cmake --build build --target flash_rt_kernels            # fused kernels (first build)
> python -m pytest tests/test_hyvla_thor_dispatch.py tests/test_hyvla_arch_gate.py \
>     tests/test_hyvla_fp4_routing.py tests/test_hyvla_kernel_contracts.py -q
> FLASHRT_HYVLA_CHECKPOINT=/path/to/Hy-Embodied-0.5-VLA \
>     python -m pytest tests/test_hyvla_thor_graphsafe.py -q   # graph==eager + replay stability
> ```

## Key Takeaway

Under single-request latency, all three major stages (ViT / prefill / denoise) on Thor (20 SMs, 243 GB/s)
are **memory/latency-bound at batch=1, achieving only ~35% of the ideal roofline — this is already the
practical floor for a per-operator hand-written path**. Further gains come not from low-bit quantization
or single-block fusion (both measured ineffective/harmful), but from **whole-layer/whole-model fusion**
(an early Inductor prototype `--v4` in this repo reached 98.8 ms via automatic fusion — the automated
version of this approach — but suffered 30–60 s recompilation per prompt, cache bloat degradation, and
silent dynamo fallback; it is not framework-grade. The production equivalent = hand-written persistent
megakernels or TRT/MLIR-TRT), **reducing flow steps (distillation)**, or **async chunking to hide
latency** — all of which require model or engineering-form changes.
Industry anchor: NVIDIA achieves 44 ms / 23 Hz on Pi0.5 with hand-written kernels + MLIR-TRT
(with a lighter visual workload than this model).

## Components

| File | Role |
|---|---|
| `flash_rt/frontends/torch/hyvla_thor.py` | `HyVLATorchFrontendThor`: tokenization, image preprocessing, prefix assembly, segmented prefix mask + prefix-LM suffix mask, bf16-round RoPE table, time embedding, CUDA Graph cache (keyed by `(S_p,n_vis)`), FP8/FP4/FFN-mega weight quantization, `set_prompt`/`infer`/`predict_actions`. Flags: `use_fp8/use_fused/use_autotune` (production) + `use_fp8_vit/use_fp4/use_fused_quant/use_ffn_mega` (opt-in diagnostics, off by default) |
| `flash_rt/models/hyvla/pipeline_thor.py` | `HyVLAThorBF16Pipeline`: `vit_forward` (27 layers including 6 spacetime layers), `merger_forward`, `prefill` (32-layer MoT), `denoise` (32-layer expert, 10-step Euler), `_fp8_gemm` (+per-shape autotune), `_ffn_mega_bf16`, efficient-SDPA `_attn` |
| `flash_rt/frontends/torch/_hyvla_thor_spec.py` | Declarative `ModelWeightSpec` (ViT/merger/VLM dual-branch/expert/action head/tied lm_head) |
| `csrc/kernels/hyvla_fused_thor.cu`(+`.cuh`) | Production fused kernel `hyvla_rope_qknorm_kvwrite_bf16` (split+rope+qknorm+kv-write = 1 launch, `kv_rep` pre-expands GQA) |
| `csrc/kernels/hyvla_quant_fp8_thor.cu` | Single-CTA dynamic FP8 quantization (`use_fused_quant` diagnostic; measured net loss, see dead ends) |
| `csrc/kernels/hyvla_ffn_fp8_thor.cu` | FFN megakernel (gu+silu_mul / dn+residual, `use_ffn_mega` diagnostic; measured neutral, see dead ends); reusable Thor plain-FP8-MMA GEMM reference |
| `flash_rt/{executors/torch_weights.py,hardware/__init__.py,api.py,configs/hyvla.yaml}` | `ToBf16` transform / `_PIPELINE_MAP` registration / config allowlist / metadata |
| `tests/test_hyvla_thor_dispatch.py`, `test_hyvla_arch_gate.py`, `test_hyvla_fp4_routing.py` | Registration, SM110 fail-fast, and FP4/fused routing gates (no GPU required) |
| `tests/test_hyvla_kernel_contracts.py` | Host-side contract validation of the fused-kernel pybind APIs |
| `tests/test_hyvla_thor_graphsafe.py` | graph==eager equivalence + replay stability gate (needs Thor + checkpoint) |

## Precision Gates (recorded values, all vs. HF/transformers eager, same fixed noise)

| Checkpoint | cosine |
|---|---|
| ViT + merger (147 visual tokens) | 0.999844 |
| Dual-tower prefill + denoise + action head (fed the original model's prefix) | 0.999978 |
| Full-chain native BF16 (load_model path) | 0.999910 |
| **Full-chain production (fp8 + fused + efficient-SDPA + autotune)** | **0.999706** |

Graph-safety gate (`tests/test_hyvla_thor_graphsafe.py`): `use_graph=True vs False`
equivalence + stable replay, covering the ViT graph and the main graph.

## Key Mechanisms / Correctness Pitfalls (Highest Reuse Value)

1. **Architecture = Pi0 action head + SigLIP-so400m-isomorphic ViT + HunYuan MoT dual tower**. ViT 1152h/hd72/patch16, learned pos_embed (128x128 bilinear rescale) + 6 spacetime causal temporal attention layers (block {3,7,11,15,19,23}, 6 history frames). VLM 2048h/6144i/32L/GQA 16Q-4KV/hd128; expert 1024h/2048i/32L.
2. **RMSNorm is pure-weight (no `1+w`), fp32 upcast, eps=1e-5** (ViT LayerNorm has bias, eps=1e-6). Using `1+w` incorrectly → cos drops to ~0.5.
3. **QK-Norm is applied after RoPE** (RMSNorm over hd=128), opposite to off-the-shelf norm-then-rope kernels → requires a custom fused kernel.
4. **RoPE `inv_freq` must round-trip through bf16**: the original model stores `rotary_emb.inv_freq` in bf16 via `.to(bf16)`, and rounding accumulates with position; using full-precision inv_freq → action cos only reaches ~0.96. This was the most time-consuming correctness issue to locate.
5. **MoT `_v` static routing**: prefix is reordered by modality into contiguous `[vision|text]` slices → two pointer-offset GEMMs per operator, no gather; the expert tower routes entirely through `_v`, reusing the VLM's QK-norm weights.
6. **Determinism / reference pitfall (biggest trap)**: monkeypatching `sample_noise` to inject fixed noise **did not take effect** (run-to-run max|delta|=2.6e-2 was the signal) — the reference became random noise → E2E falsely measured 0.958. Switching to **explicitly passing noise** to `sample_actions` immediately yielded 0.9999. **Always verify run-to-run delta is approximately 0 before trusting fixed-noise results.**

## Performance Optimization Ladder (all measured, same prompt/image, warmup + median)

| Stage | E2E | Notes |
|---|---|---|
| HF/transformers eager (anchor) | ~930 ms | |
| Native BF16 eager | 303 ms | 3.07x, no upstream overhead |
| + CUDA Graph (prefill+denoise in one graph) | 248.6 ms | 3.74x, graph==eager bitwise |
| + Dynamic per-tensor FP8 (denoise, in-graph) | 225.2 ms | 4.13x, cos 0.999833 (FP8 only pays off in-graph) |
| + FP8 prefill (MoT dual-branch) | 207.0 ms | 4.49x |
| + Fused megakernel (rope+qknorm+kvwrite) | 195.5 ms | 4.76x, ~11 torch ops → 1 launch |
| + efficient-SDPA (GQA expansion + memory-efficient backend) | 172.9 ms | 5.38x, attention 41 → ~12 ms, **cos actually improved to 0.999706** (bool-mask + enable_gqa forces the slow math backend) |
| + ViT trailing-stage history-frame drop | 166.1 ms | 5.60x, per-frame independence after last spacetime block → 18 → 3 frames, mathematically equivalent |
| + KV pre-expansion for GQA (megakernel `kv_rep`) | 159.6 ms | 5.83x, zero repeat_interleave in attention |
| **+ Per-shape FP8 GEMM autotune** | **158.3 ms** | **5.9x**, per-shape `autotune_fp8_nn_dev` before capture (following the motus pattern), bitwise identical |

## Roofline / Hardware Utilization (all measured anchors)

Thor SM110: 20 SMs; HBM **243 GB/s**; bf16 GEMM achievable **~110 TFLOPS** (cuBLAS large square matrices;
raw-mma silicon >=378 → cuBLAS bf16 only utilizes ~30% of silicon); fp8 `fp8_nn_dev` **~270 TFLOPS**.

| Component | GEMM FLOP | Bound by | Ideal ceiling | Measured | Utilization |
|---|---|---|---|---|---|
| ViT+merger (bf16, 3 cameras x 6 frames) | 2718 G | compute | 24.7 ms | 71.5 | **35%** |
| prefill (fp8, M=240, x1) | 756 G | memory/weight | 12.7 ms | 33 | **38%** |
| denoise (fp8, M=41, x10 steps) | 333 G | memory/weight+KV | 18.2 ms | 52 | **35%** |

All three stages reach only ~35% of the ideal ceiling, but **all sit at the practical floor for a small
GPU + small batch** (corroborated by *Demystifying VLA Inference*, arXiv 2602.18397: at batch=1 the action
head is strictly memory-bound, and the backbone becomes memory-bound at the low-bandwidth edge; high-end
GPUs achieve 73–82% of theoretical peak in measured kernels). The unified root cause: **small M cannot
saturate** — a single M=41 fp8 GEMM only achieves 34% of BW (41 rows too thin, poor L2 reuse, per-SM
ramp/tail dominates). This explains all dead ends below.

## Dead Ends (all measured; check before writing code)

| Approach | Result | Mechanism |
|---|---|---|
| **FP8-ViT** | Net loss (ViT graph 71.5 → 85.4 ms) | ViT utilization only 35%, not GEMM-compute-bound; FP8 only adds quantization kernel overhead |
| **Single-CTA fused quantization** (4 kernels → 1, `use_fused_quant`) | +2.6 ms regression | Single CTA occupies 1 SM; original 4 kernels spread blocks across all SMs, in-graph launch cost is ~0; fewer nodes gives no benefit, lower occupancy is harmful |
| **FFN megakernel** (`use_ffn_mega`, gu+silu / dn+res two kernels) | E2E neutral (+0.5 ms) | M=41 FFN is latency-bound, cuBLASLt is already good enough; fused silu/mul/residual is negligible |
| **FFN single-kernel fused** (grid-barrier, see two-case comparison) | 14% slower than split | Barrier serializes globally + persistent grid reduces occupancy, exceeding saved launch/HBM round-trips |
| **Elementwise fusion** (residual+norm, silu_mul) | ~0 (reverted) | At M=41, ~42K elements are too cheap, already amortized by graph |
| **FP4 (W4A16 on Thor)** | denoise 1.14x (=FP8), prefill large GEMM 2.67% but only 21% of total | Thor has no native FP4 MMA, see below |
| **grid-barrier 192 CTA launch** | Deadlock | Exceeds co-residency capacity (20 SMs ~160) → late-arriving CTAs never launch. Must use persistent grid launched at capacity |
| bool-mask + `enable_gqa` SDPA | ~41 ms (slow math fallback) | SM110 silently falls back to math backend; fixed with GQA-expansion + memory-efficient |

**CUDA Graph safety rules (learned from bugs):** `_fp8_gemm` transients must use `torch.empty` each time
(entering the graph's private pool) — shared scratch aliases the two outputs of MoT `torch.cat`; during
capture all allocations come from the private pool and are overwritten between replays; pointer kernels
must pass `current_stream().cuda_stream` (passing 0 = default stream = not captured); FP8 weights require
(K,N) layout, `cutlass_fp4` SF requires swizzled 128x4 (wrong layout produces no error but cos collapses).

## Why FP4 Does Not Work (confirmed via DGX Spark SM121 reports + local ptxas verification)

- **Thor has no native FP4 tensor core MMA**: native NVFP4 requires `tcgen05.mma` + TMEM, **only available on datacenter SM100 (B200)**.
  Measured: `mma.sync.kind::f8f6f4` is rejected by ptxas on sm_110; standard `mma.sync.m16n8k32 e4m3` (FP8) works.
  → On edge Blackwell (Thor/Spark), FP4 can only be **W4A16**: decompress back to bf16/fp8 before computing — **zero compute speedup, only saves weight bandwidth**.
- Bandwidth savings only materialize when **weight-bandwidth-bound and saturating BW**: denoise M=41 is latency-bound (34% BW) → FP4=FP8=1.14x.
  On Spark, NVFP4 measured 65 tok/s is actually **slower** than FP8 at 91 (decompression + small smem overhead).
- The only meaningful case = prefill large GEMM (M=240, gu 2.67x), but prefill is only 21% of E2E and ViT (bf16) cannot benefit.
  **Therefore FP4 provides approximately no overall benefit for this model on Thor, consistent with industry edge-Blackwell findings.** Code is correct, gated behind `use_fp4` for archival.

## Two Megakernel Implementation Comparison

Isolated denoise FFN @ M=41 latency benchmark, three approaches:

| Implementation | Latency | Correctness |
|---|---|---|
| per-op (cuBLASLt `fp8_nn_dev` autotuned + torch silu_mul + quant) | **61.9 us** | baseline |
| Case A (2-kernel split, boundary quantization) | **62.5 us (1.01x, tied)** | cos 0.99987 vs per-op |
| Case B (single-kernel fused + grid-barrier) | **69.8 us (1.13x, slower)** | cos 0.9994 |

**Two independent implementations reach the same conclusion: at M=41, fusion cannot improve utilization and is even harmful.** True megakernel gains come from whole-model fusion
(Mirage/MPK, Hazy "no-bubbles" — eliminating launches + bubbles + HBM round-trips across multiple layers) or raising M (batching), not from single-block fusion.
The building blocks are in place and have high reuse value: plain-FP8-MMA layout verified on Thor at cos 0.999998, grid-barrier graph is capturable
(state pre-zeroed + self-resetting), 2-stage cp.async loop tail requires `__syncthreads` (otherwise WAR race).

## Paths Toward <100 ms (all require architectural changes, not incremental)

1. **Reduce flow steps** (10 → 2-4, Consistency Policy / distillation) — denoise 52 ms is the only stage that can be truly cut; requires retraining.
2. **Whole-model megakernel** (Mirage-style) or **TRT/MLIR-TRT + hand-written kernels + Q/DQ removal** (NVIDIA's actual stack for Pi0.5 → 44 ms) — the Inductor prototype `--v4` in this repo already demonstrated **98.8 ms** is achievable via automatic fusion (ViT 51.5 / denoise 29), but it recompiles per prompt, suffers 15% cache-bloat degradation, silent dynamo fallback, and is not registered in `_PIPELINE_MAP` — not framework-grade.
3. **Async action chunking** (Pi real-time chunking) — overlap inference with execution, hide latency at the system level; 158 ms does not block control.
4. Reduce history frames (ViT's largest cost) / smaller backbone / layer skipping.

## Early Inductor Prototype (`--v4`, Historical Reference, Not Production)

An early `torch.compile` whole-block fusion + static FP8 + grouped-bmm/FA2 +
full-graph capture prototype reached E2E **98.8 ms (8.6–9.1x)**, cos 0.9985–0.9998.
It proved the benefit of the "whole-layer fusion" approach (ViT 51.5, denoise 29
are both below native), but due to **30–60 s recompilation per prompt, 15%
Inductor cache-bloat degradation, silent dynamo fallback, and no integration
with `load_model`**, it is not used in production. The six measurement
methodology lessons it exposed (L2 residency illusion / no-fusion baseline
illusion / nsys sum != wall clock / synchronization floor / Inductor cache
bloat / dynamo silent downgrade) informed the production path design.

## Hardware and Model Profile (Measured)

NVIDIA Thor cc11.0 (SM110), 20 SMs, 125.7 GB unified memory, L2 32 MB, CUDA 13.0 / driver 580,
L4T R38.2, torch 2.9.0a0, transformers 5.10.2, GPU full clock 1575 MHz.
Model: VLM tower 2048h/32L/GQA 16Q-4KV/hd128/Dff6144 (dual text+vision projections per layer); expert tower
1024h/inter2048/32L; ViT HYViT2-400M 27L/1152h/patch16/6-frame spacetime (stride 4); chunk 40; 10-step Euler.
